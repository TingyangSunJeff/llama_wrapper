"""Streaming measurement client for the profile-suite harness.

This module generalizes the streaming chat client from
``experiments/smoke/common.py`` (``chat()``) into a reusable measurement
:class:`Client` that talks to a ``llama-server`` over its OpenAI-compatible
``/v1/chat/completions`` endpoint with ``stream=True``.

All timing uses a single monotonic clock (``time.monotonic()``), captured from
the instant the request is submitted:

- **TTFT** (time-to-first-token): submit -> first streamed delta carrying
  ``content`` (the first generated token). This also defines the Warmup phase
  boundary used by the Switch_Cost_Profiler (R1.1, R1.10).
- **latency**: submit -> stream completion (the ``[DONE]`` sentinel or end of
  the response body).
- **n_tokens**: the number of streamed content deltas observed.
- **TPOT** is not stored directly; it is derivable by callers as
  ``(latency_ms - ttft_ms) / (n_tokens - 1)`` when ``n_tokens > 1`` (R2.1).

The three public methods map to the design "Shared harness" ``client.py`` block:

- :meth:`Client.first_token` — warmup first-token timing with ``warmup_timeout_s``
  (R1.1, R1.10).
- :meth:`Client.measure_stream` — steady-state TTFT/TPOT/latency over
  ``max_tokens`` (R2.1).
- :meth:`Client.batched` — ``concurrency`` simultaneous streams sharing one
  session, returning one :class:`RequestTiming` per request (R2.x decode-batch
  driving).

The measurement logic deliberately does not retain generated text; the suite
cares about timing and token counts, not content. Any transport, HTTP, or
decode error is captured into ``RequestTiming.error`` with ``ok=False`` rather
than raised, so a single failed request never aborts a campaign measurement
point (callers exclude failed repeats from aggregates).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

# Default per-request ceiling for steady-state measurement streams. A measurement
# stream is bounded by ``max_tokens`` server-side; this guards against a hung
# connection without capping legitimate decode time on large outputs.
DEFAULT_STREAM_TIMEOUT_S = 600.0

# Temperature is pinned to 0.0 for measurement determinism (matches the smoke
# harness): timing should not vary with sampling randomness.
_TEMPERATURE = 0.0


@dataclass
class RequestTiming:
    """Timing result for one streamed chat-completion request.

    Mirrors the design "Shared harness" ``client.py`` block:

    - ``ok``: ``True`` iff the request completed without error.
    - ``ttft_ms``: submit -> first content delta, in milliseconds (``None`` if no
      token was ever received or the request failed before first token).
    - ``latency_ms``: submit -> stream completion, in milliseconds (``None`` only
      when the request could not be submitted at all).
    - ``n_tokens``: number of streamed content deltas observed.
    - ``error``: a string describing the failure, or ``None`` on success.
    """

    ok: bool
    ttft_ms: Optional[float]
    latency_ms: Optional[float]
    n_tokens: int
    error: Optional[str]


def _chat_payload(prompt: str, max_tokens: int) -> dict:
    """Build the OpenAI-compatible streaming chat-completions payload."""

    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": _TEMPERATURE,
        "stream": True,
        # Disable prompt caching so prefill cost is measured, not amortized.
        "cache_prompt": False,
    }


async def _stream_once(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    max_tokens: int,
    *,
    stop_after_first_token: bool = False,
) -> RequestTiming:
    """Issue one streaming request and measure its timing.

    Args:
        session: An open :class:`aiohttp.ClientSession`. Its configured timeout
            bounds the request.
        base_url: Server base URL, e.g. ``http://127.0.0.1:8090``.
        prompt: The user prompt to send.
        max_tokens: Server-side output-token ceiling for this request.
        stop_after_first_token: When ``True``, stop reading as soon as the first
            content delta arrives (warmup first-token timing); ``latency_ms`` then
            equals ``ttft_ms``.

    Returns:
        A :class:`RequestTiming`. Any error is captured (``ok=False``) rather than
        raised. TTFT and latency are derived from a single ``time.monotonic()``
        origin taken at submit.
    """

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = _chat_payload(prompt, max_tokens)

    submit = time.monotonic()
    ttft_ms: Optional[float] = None
    n_tokens = 0

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestTiming(
                    ok=False,
                    ttft_ms=None,
                    latency_ms=(time.monotonic() - submit) * 1000.0,
                    n_tokens=0,
                    error=f"HTTP {resp.status}: {body[:200]}",
                )

            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                chunk = delta.get("content")
                if chunk:
                    if ttft_ms is None:
                        ttft_ms = (time.monotonic() - submit) * 1000.0
                    n_tokens += 1
                    if stop_after_first_token:
                        break
    except asyncio.TimeoutError:
        return RequestTiming(
            ok=False,
            ttft_ms=ttft_ms,
            latency_ms=(time.monotonic() - submit) * 1000.0,
            n_tokens=n_tokens,
            error="timeout",
        )
    except Exception as e:  # noqa: BLE001 - surface any transport/decode error
        return RequestTiming(
            ok=False,
            ttft_ms=ttft_ms,
            latency_ms=(time.monotonic() - submit) * 1000.0,
            n_tokens=n_tokens,
            error=repr(e),
        )

    latency_ms = (time.monotonic() - submit) * 1000.0

    # A stream that completed but produced no content delta is a failure: there is
    # no first token to time, so the measurement is not usable.
    if n_tokens == 0:
        return RequestTiming(
            ok=False,
            ttft_ms=None,
            latency_ms=latency_ms,
            n_tokens=0,
            error="no tokens streamed",
        )

    return RequestTiming(
        ok=True,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        n_tokens=n_tokens,
        error=None,
    )


class Client:
    """Async measurement client for a ``llama-server`` OpenAI-compatible endpoint.

    The client is stateless across calls; each method opens its own
    :class:`aiohttp.ClientSession` so it can be used directly without lifecycle
    management. :meth:`batched` shares one session across its concurrent requests.
    """

    async def first_token(
        self,
        base_url: str,
        prompt: str,
        warmup_timeout_s: float = 60.0,
    ) -> RequestTiming:
        """Measure time to the first generated token (Warmup phase).

        Sends a single-token streaming request and stops as soon as the first
        content delta arrives. The returned ``ttft_ms`` (== ``latency_ms`` here)
        is the Warmup interval used by the Switch_Cost_Profiler (R1.1). If the
        first token does not complete within ``warmup_timeout_s``, the result is
        a timeout failure (``ok=False``) so the switch can be recorded as failed
        in the Warmup phase and excluded from aggregates (R1.10).

        Args:
            base_url: Server base URL.
            prompt: The first post-switch request prompt.
            warmup_timeout_s: Maximum wait for the first token, in seconds.

        Returns:
            A :class:`RequestTiming` for the warmup request.
        """

        timeout = aiohttp.ClientTimeout(total=warmup_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await _stream_once(
                session,
                base_url,
                prompt,
                max_tokens=1,
                stop_after_first_token=True,
            )

    async def measure_stream(
        self,
        base_url: str,
        prompt: str,
        max_tokens: int,
    ) -> RequestTiming:
        """Measure a steady-state stream's TTFT, latency, and token count.

        Drives one full streaming request bounded by ``max_tokens`` and records
        TTFT (submit -> first content delta), total latency (submit -> stream
        completion), and the streamed token count. Callers derive TPOT as
        ``(latency_ms - ttft_ms) / (n_tokens - 1)`` when ``n_tokens > 1`` (R2.1).

        Args:
            base_url: Server base URL.
            prompt: The campaign-fixed measurement prompt.
            max_tokens: Campaign-fixed generated output token length.

        Returns:
            A :class:`RequestTiming` for the measurement request.
        """

        timeout = aiohttp.ClientTimeout(total=DEFAULT_STREAM_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await _stream_once(
                session,
                base_url,
                prompt,
                max_tokens=max_tokens,
            )

    async def batched(
        self,
        base_url: str,
        prompt: str,
        max_tokens: int,
        concurrency: int,
    ) -> list[RequestTiming]:
        """Issue ``concurrency`` simultaneous measurement streams.

        Launches ``concurrency`` identical streaming requests concurrently over a
        shared session and returns one :class:`RequestTiming` per request, in
        launch order. Used to drive decode-batch behavior; failed requests are
        returned as ``ok=False`` entries rather than aborting the batch (R2.x).

        Args:
            base_url: Server base URL.
            prompt: The measurement prompt sent on every request.
            max_tokens: Generated output token length per request.
            concurrency: Number of simultaneous requests (>= 1).

        Returns:
            A list of :class:`RequestTiming`, one per request.
        """

        if concurrency < 1:
            return []

        timeout = aiohttp.ClientTimeout(total=DEFAULT_STREAM_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                asyncio.create_task(
                    _stream_once(session, base_url, prompt, max_tokens=max_tokens)
                )
                for _ in range(concurrency)
            ]
            return await asyncio.gather(*tasks)


__all__ = ["RequestTiming", "Client"]
