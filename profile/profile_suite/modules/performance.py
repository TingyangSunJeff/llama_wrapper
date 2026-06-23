"""Performance_Profiler: configuration-to-performance surface measurement.

This module measures Prefill_Throughput, Decode_Throughput, TTFT, and TPOT
across the Config_Grid and across Decode_Batch_Size.

This file contains the *pure* per-batch winner/tie computation
(:func:`batch_winner`) and the :class:`Performance_Profiler` module that wires
the **server path** of the configuration-to-performance surface (TTFT, TPOT,
Prefill_Throughput, Decode_Throughput at the campaign-fixed prompt/output token
lengths — task 12.1, R2.1/R2.3/R2.7/R2.8).

The ``llama-bench`` (single-stream batch-1) and ``llama-batched-bench``
(decode-batch sweep) driver paths are implemented in this file as well (task
12.2, R2.2/R2.5/R2.6), generalized from
``experiments/smoke/scenario_a_batchsweep.py``. They are exposed as additional
:class:`~profile_suite.modules.base.PointSpec`\\ s
(``axis={"path": "batched-bench"}`` and ``axis={"path": "bench"}``) and are
dispatched by :meth:`Performance_Profiler.measure_once` to the
``llama-batched-bench`` / ``llama-bench`` subprocess drivers. The binaries are
resolved via ``platform.resolve_binary("batched-bench")`` and
``platform.resolve_binary("bench")``.

Batched-decode path (design "Performance_Profiler", R2.2/R2.6)
--------------------------------------------------------------
:meth:`Performance_Profiler._measure_batched_bench` runs one full
``--output-format jsonl`` ``llama-batched-bench`` sweep over the campaign's
Decode_Batch_Size set (the ``-npl`` axis) at the campaign-fixed prompt/output
token lengths and the Config's context length, then parses the emitted jsonl
(:func:`parse_batched_bench_jsonl`) into one per-batch decode-throughput value
(``decode_throughput_b{pl}``, tok/s) plus the per-batch prefill rate
(``prefill_throughput_b{pl}``). A sweep that produces no parseable rows (binary
crash, no GPU, bad model) is returned as ``ok=False``.

Single-stream path (design "Performance_Profiler", R2.5)
--------------------------------------------------------
:meth:`Performance_Profiler._measure_bench` runs ``llama-bench`` in its default
single-sequence (batch-1) mode with ``-o jsonl``, ``-p prompt_tokens``, and
``-n output_tokens``, then parses the prefill (``n_gen == 0``) and decode
(``n_prompt == 0``) rows (:func:`parse_bench_jsonl`) into batch-1
``prefill_throughput`` / ``decode_throughput`` (tok/s).

Server-path measurement (design "Performance_Profiler", R2.1)
-------------------------------------------------------------
For one Config, :meth:`Performance_Profiler.measure_once` boots a
``llama-server`` (via :class:`~profile_suite.harness.server.ServerHandle`), drives
a single streaming measurement request at the campaign-fixed prompt/output token
lengths (via :meth:`~profile_suite.harness.client.Client.measure_stream`), and
derives the four metrics from the request's TTFT / latency / token count:

- ``ttft_ms``             = request TTFT (submit -> first token), in ms.
- ``tpot_ms``             = decode interval / inter-token gaps, in ms/token
  (``(latency_ms - ttft_ms) / (output_tokens - 1)``; the glossary's "mean
  inter-token interval during decode, excluding TTFT").
- ``prefill_throughput``  = ``prompt_tokens / ttft_s`` (prompt-processing rate,
  tok/s).
- ``decode_throughput``   = ``output_tokens / decode_time_s`` where
  ``decode_time_s = (latency_ms - ttft_ms) / 1000`` (output-token generation
  rate, tok/s).

A run whose server fails to boot, whose request errors/times out, or whose timing
is unusable (no first token, non-positive decode interval) is returned as
``ok=False`` with an error reason; the Reproducibility_Harness run loop excludes
failed runs from the aggregates (R2.7, R2.8) and the campaign continues.

Harness-interface assumptions (documented contract)
---------------------------------------------------
``measure_once`` is handed the shared harness object by the run loop. Because the
concrete harness aggregator is assembled in a later task, this module depends only
on a small, duck-typed surface and falls back to safe defaults so it stays
importable and unit-testable with a lightweight fake. It reads (all optional):

- ``harness.run_index``       -> int, current run index (0 == discarded warmup).
- ``harness.host``            -> str, server bind host (default ``"127.0.0.1"``).
- ``harness.port``            -> int, server bind port (default ``8090``).
- ``harness.gpu_index``       -> int, GPU to pin (default ``0``).
- ``harness.boot_timeout_s``  -> float, boot timeout (default ``300.0``, R1.9).
- ``harness.prompt``          -> str, the campaign-fixed measurement prompt; when
  absent a deterministic prompt of approximately ``prompt_tokens`` words is
  synthesized.
- ``harness.server_log_path(point_id, run_index)`` -> str, the per-run server log
  path under the campaign-scoped ``profile/runs/<...>/`` dir (R5.5); when absent a
  temp-file path is used so the module is still runnable in isolation.

The campaign-fixed ``prompt_tokens`` / ``output_tokens`` are carried on the
:class:`~profile_suite.modules.base.PointSpec` ``params`` (populated by
:meth:`Performance_Profiler.points` from the ``CampaignConfig``). The server binary
is resolved via ``platform.resolve_binary("server")``.

Per-batch quant winner and tie (design "Property 6" / R2.4)
-----------------------------------------------------------
At a single Decode_Batch_Size value, a campaign that swept at least two quant
formats yields a mapping ``format -> (mean, std)`` of aggregated
Decode_Throughput. From that mapping:

- The **winner** is a format achieving the **maximum mean** Decode_Throughput.
- The result is a **tie** iff the gap between the top-two means is **within one
  standard deviation** — i.e. ``(top_mean - second_mean) <= sigma`` where
  ``sigma`` is the standard deviation of the winning (max-mean) format, the
  error bar around the leader. If the runner-up's mean falls inside the
  leader's error bar, the lead is not statistically distinguishable and the
  point is marked a tie.

A single-format mapping has no "top two" to compare, so it is never a tie.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Tuple

from ..config import Config
from ..harness.client import Client, RequestTiming
from ..harness.server import ServerHandle
from ..results import RunRepeatResult
from .base import PointSpec, make_point_id

if TYPE_CHECKING:  # pragma: no cover - typing-only import to avoid a hard dependency
    from ..campaign import CampaignConfig


@dataclass(frozen=True)
class BatchWinner:
    """The winning quant format at one Decode_Batch_Size value (R2.4).

    Attributes:
        winner: A quant format achieving the maximum mean Decode_Throughput.
            When several formats share the maximum mean, the lexicographically
            smallest format name is chosen so the result is deterministic.
        tie: ``True`` iff at least two formats were compared and the gap between
            the top-two means is within one standard deviation of the winner.
        margin: ``top_mean - second_mean`` (``>= 0``); ``inf`` when only one
            format was supplied (no runner-up to compare against).
        sigma: The standard deviation of the winning format, used as the tie
            threshold; ``0.0`` when only one format was supplied.
    """

    winner: str
    tie: bool
    margin: float
    sigma: float


def batch_winner(aggregates: Mapping[str, Tuple[float, float]]) -> BatchWinner:
    """Compute the winning quant format and tie flag at one batch value.

    Args:
        aggregates: Mapping of quant format name to its
            ``(mean, std)`` Decode_Throughput aggregate at a single
            Decode_Batch_Size value. Both ``mean`` and ``std`` are in tok/s,
            and ``std`` is expected to be non-negative.

    Returns:
        A :class:`BatchWinner` whose ``winner`` achieves the maximum mean and
        whose ``tie`` is ``True`` iff the top-two means are within one standard
        deviation of the winner (R2.4).

    Raises:
        ValueError: If ``aggregates`` is empty (no format to win).
    """
    if not aggregates:
        raise ValueError("aggregates must contain at least one quant format")

    # Sort by descending mean, breaking ties on the format name so the winner
    # is deterministic when several formats share the maximum mean.
    ordered = sorted(
        aggregates.items(),
        key=lambda item: (-item[1][0], item[0]),
    )

    winner_format, (winner_mean, winner_std) = ordered[0]
    sigma = winner_std if winner_std > 0.0 else 0.0

    if len(ordered) == 1:
        # Single format: no top-two comparison, so never a tie.
        return BatchWinner(
            winner=winner_format,
            tie=False,
            margin=float("inf"),
            sigma=sigma,
        )

    second_mean = ordered[1][1][0]
    margin = winner_mean - second_mean

    # Tie iff the runner-up's mean falls within one standard deviation of the
    # winner (inclusive) — the lead is within the leader's error bar.
    tie = margin <= sigma

    return BatchWinner(
        winner=winner_format,
        tie=tie,
        margin=margin,
        sigma=sigma,
    )


__all__ = [
    "BatchWinner",
    "batch_winner",
    "BatchedBenchRow",
    "parse_batched_bench_jsonl",
    "batched_decode_throughput",
    "parse_bench_jsonl",
    "Performance_Profiler",
]


# --------------------------------------------------------------------------- #
# Bench-output parsers (task 12.2, R2.6) — pure functions, fixture-testable
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BatchedBenchRow:
    """One parsed ``llama-batched-bench`` jsonl row.

    Mirrors the fields emitted by ``llama-batched-bench --output-format jsonl``
    (see ``tools/batched-bench``). ``pl`` is the decode batch size (parallel
    sequences generated together), ``speed_tg`` the batched decode throughput in
    tok/s, and ``speed_pp`` the prompt-processing (prefill) rate in tok/s. The
    full source object is retained on ``raw`` for transparency.
    """

    pl: int
    pp: int
    tg: int
    n_kv: int
    t_pp: float
    speed_pp: float
    t_tg: float
    speed_tg: float
    t: float
    speed: float
    raw: dict = field(default_factory=dict)


def parse_batched_bench_jsonl(text: str) -> list[BatchedBenchRow]:
    """Parse ``llama-batched-bench --output-format jsonl`` stdout into rows.

    Generalized from ``experiments/smoke/scenario_a_batchsweep.py``: scans the
    text line by line, parses each ``{...}`` line as JSON, and keeps the rows that
    carry both a decode batch (``pl``) and a decode throughput (``speed_tg``).
    Non-json lines (banners, progress, the markdown table) and json objects
    missing those keys are skipped, so a stdout stream mixing logging and jsonl is
    handled gracefully.

    Args:
        text: The captured stdout of a ``llama-batched-bench`` jsonl run.

    Returns:
        The parsed :class:`BatchedBenchRow` list in first-seen order (may be empty
        when no valid rows are present — the caller treats an empty result as a
        failed run).
    """
    rows: list[BatchedBenchRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "pl" not in obj or "speed_tg" not in obj:
            continue
        rows.append(
            BatchedBenchRow(
                pl=int(obj["pl"]),
                pp=int(obj.get("pp", 0)),
                tg=int(obj.get("tg", 0)),
                n_kv=int(obj.get("n_kv", 0)),
                t_pp=float(obj.get("t_pp", float("nan"))),
                speed_pp=float(obj.get("speed_pp", float("nan"))),
                t_tg=float(obj.get("t_tg", float("nan"))),
                speed_tg=float(obj["speed_tg"]),
                t=float(obj.get("t", float("nan"))),
                speed=float(obj.get("speed", float("nan"))),
                raw=obj,
            )
        )
    return rows


def batched_decode_throughput(rows: Iterable[BatchedBenchRow]) -> dict[int, float]:
    """Reduce parsed batched-bench rows to a ``decode-batch -> speed_tg`` map.

    When several rows share the same ``pl`` (e.g. the same batch appears more than
    once in a sweep), the last value wins — matching the dict-overwrite behavior of
    the original ``scenario_a_batchsweep.py`` parser.
    """
    return {row.pl: row.speed_tg for row in rows}


def parse_bench_jsonl(text: str) -> dict[str, float]:
    """Parse ``llama-bench -o jsonl`` stdout into batch-1 prefill/decode throughput.

    ``llama-bench`` emits one json object per test row. A *prefill* row has
    ``n_prompt > 0`` and ``n_gen == 0``; its ``avg_ts`` is the prompt-processing
    rate (tok/s). A *decode* row has ``n_prompt == 0`` and ``n_gen > 0``; its
    ``avg_ts`` is the generation rate (tok/s). Non-json lines and rows missing
    ``avg_ts`` are skipped.

    Args:
        text: The captured stdout of a ``llama-bench`` jsonl run.

    Returns:
        A dict with any of ``prefill_throughput`` / ``decode_throughput`` that
        were found (each in tok/s). When the stddev is present it is also returned
        as ``prefill_throughput_std`` / ``decode_throughput_std``. An empty dict
        means no usable rows were parsed (the caller treats that as a failed run).
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "avg_ts" not in obj:
            continue
        n_prompt = int(obj.get("n_prompt", 0))
        n_gen = int(obj.get("n_gen", 0))
        avg_ts = float(obj["avg_ts"])
        std_ts = float(obj.get("stddev_ts", 0.0))
        if n_prompt > 0 and n_gen == 0:
            out["prefill_throughput"] = avg_ts
            out["prefill_throughput_std"] = std_ts
        elif n_gen > 0 and n_prompt == 0:
            out["decode_throughput"] = avg_ts
            out["decode_throughput_std"] = std_ts
    return out




# Defaults applied when the harness does not supply the corresponding runtime
# context (keeps the module importable and unit-testable with a light fake).
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8090
_DEFAULT_GPU_INDEX = 0
_DEFAULT_BOOT_TIMEOUT_S = 300.0

# Campaign-fixed token-length defaults (mirrors the design's sample campaign,
# R2.1) used only when the CampaignConfig / PointSpec omits them.
_DEFAULT_PROMPT_TOKENS = 512
_DEFAULT_OUTPUT_TOKENS = 128

# Bench-driver defaults (task 12.2). GPU offload mirrors the smoke scripts and
# ServerHandle (env-overridable via NGL). The canonical Decode_Batch_Size set is
# the campaign default (R8.3) used when the campaign omits the set. The subprocess
# timeout bounds a full sweep / bench run (generalized from scenario_a_batchsweep).
_DEFAULT_NGL = int(os.environ.get("NGL", "99"))
_DEFAULT_DECODE_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
_DEFAULT_BENCH_TIMEOUT_S = 1200.0

_MS_PER_S = 1000.0


def _synth_prompt(prompt_tokens: int) -> str:
    """Synthesize a deterministic prompt of approximately ``prompt_tokens`` tokens.

    Used only when the harness does not supply an explicit campaign-fixed prompt.
    The word count is a whitespace-token approximation of the requested length —
    a tokenizer-exact prompt is a refinement left to the harness, which may pass
    ``harness.prompt`` directly. The text is deterministic so prefill timing does
    not vary run-to-run with prompt content.
    """
    n = max(1, int(prompt_tokens))
    # A fixed, content-light filler word keeps the prompt deterministic.
    return " ".join(["token"] * n)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine to completion from this synchronous context.

    ``measure_once`` is invoked synchronously by the Reproducibility_Harness run
    loop, so :func:`asyncio.run` is the normal path. A fresh event loop fallback
    keeps the call working even if it is ever invoked while another loop exists.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _failed_run(run_index: int, log_path: str, error: str) -> RunRepeatResult:
    """Build an ``ok=False`` :class:`RunRepeatResult` carrying an error reason.

    ``discarded_warmup`` reflects the warmup index (0); the run loop overrides it
    authoritatively, so the value here is only a sensible default.
    """
    return RunRepeatResult(
        run_index=run_index,
        discarded_warmup=(run_index == 0),
        ok=False,
        raw_log_path=log_path,
        metrics={},
        error=error,
    )


class Performance_Profiler:
    """Configuration-to-performance surface profiler (server path).

    Implements the :class:`~profile_suite.modules.base.ProfilerModule` Protocol.
    For each Config in the grid this module measures TTFT, TPOT,
    Prefill_Throughput, and Decode_Throughput at the campaign-fixed prompt/output
    token lengths by booting a ``llama-server`` and driving one streaming
    measurement request (R2.1). Per-run failures are returned as ``ok=False`` so
    the harness run loop excludes only the failed runs from the aggregates
    (R2.3, R2.7, R2.8).

    The single-stream ``llama-bench`` and batched ``llama-batched-bench`` driver
    paths (R2.2/R2.5/R2.6) are implemented separately in task 12.2.
    """

    name: str = "Performance_Profiler"

    def points(
        self, cfg: "CampaignConfig", grid: list[Config]
    ) -> Iterable[PointSpec]:
        """Enumerate the performance points for each Config.

        Emits three points per Config, distinguished by their ``axis["path"]`` so
        they never collide:

        - ``{"path": "server"}``        — TTFT/TPOT/prefill/decode from one
          streaming ``llama-server`` request (task 12.1, R2.1).
        - ``{"path": "batched-bench"}`` — one ``llama-batched-bench`` sweep over
          the campaign Decode_Batch_Size set -> per-batch decode throughput
          (R2.2, R2.6). The set is carried on ``params["decode_batch_sizes"]``.
        - ``{"path": "bench"}``         — ``llama-bench`` single-stream batch-1
          prefill/decode throughput (R2.5).

        Each point carries the campaign-fixed ``prompt_tokens`` / ``output_tokens``
        on its ``params``.
        """
        prompt_tokens = int(getattr(cfg, "prompt_tokens", _DEFAULT_PROMPT_TOKENS))
        output_tokens = int(getattr(cfg, "output_tokens", _DEFAULT_OUTPUT_TOKENS))
        decode_batch_sizes = tuple(
            int(b)
            for b in getattr(cfg, "decode_batch_sizes", _DEFAULT_DECODE_BATCH_SIZES)
        )

        specs: list[PointSpec] = []
        for config in grid:
            # Server path: TTFT/TPOT/prefill/decode from one streaming request.
            server_axis = {"path": "server"}
            specs.append(
                PointSpec(
                    module=self.name,
                    config=config,
                    axis=server_axis,
                    point_id=make_point_id(self.name, config, server_axis),
                    params={
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                    },
                )
            )
            # Batched-decode path: one llama-batched-bench sweep over the
            # Decode_Batch_Size set -> per-batch decode throughput (R2.2, R2.6).
            bb_axis = {"path": "batched-bench"}
            specs.append(
                PointSpec(
                    module=self.name,
                    config=config,
                    axis=bb_axis,
                    point_id=make_point_id(self.name, config, bb_axis),
                    params={
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "decode_batch_sizes": decode_batch_sizes,
                    },
                )
            )
            # Single-stream path: llama-bench batch-1 prefill/decode (R2.5).
            bench_axis = {"path": "bench"}
            specs.append(
                PointSpec(
                    module=self.name,
                    config=config,
                    axis=bench_axis,
                    point_id=make_point_id(self.name, config, bench_axis),
                    params={
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                    },
                )
            )
        return specs

    def measure_once(
        self, spec: PointSpec, harness: Any, platform: Any
    ) -> RunRepeatResult:
        """Perform one measurement run for ``spec``, dispatched by ``axis['path']``.

        - ``server``        -> :meth:`_measure_server` (one streaming request).
        - ``batched-bench`` -> :meth:`_measure_batched_bench` (decode-batch sweep).
        - ``bench``         -> :meth:`_measure_bench` (single-stream batch-1).

        Any boot/request/subprocess failure (or unusable output) is returned as an
        ``ok=False`` :class:`RunRepeatResult` with an error reason so the run loop
        excludes only that run from the aggregates (R2.7, R2.8).
        """
        path = spec.axis.get("path", "server")
        if path == "batched-bench":
            return self._measure_batched_bench(spec, harness, platform)
        if path == "bench":
            return self._measure_bench(spec, harness, platform)
        return self._measure_server(spec, harness, platform)

    def _measure_server(
        self, spec: PointSpec, harness: Any, platform: Any
    ) -> RunRepeatResult:
        """Perform one server-path performance measurement run for ``spec``.

        Boots the Config's ``llama-server``, drives a single streaming request at
        the campaign-fixed prompt/output token lengths, and derives ``ttft_ms``,
        ``tpot_ms``, ``prefill_throughput``, and ``decode_throughput``. Any boot or
        request failure (or unusable timing) is returned as an ``ok=False`` result
        with an error reason; the run loop excludes it from the aggregates
        (R2.7, R2.8). Always tears the server down before returning.
        """
        run_index = int(getattr(harness, "run_index", 0))
        host = getattr(harness, "host", _DEFAULT_HOST)
        port = int(getattr(harness, "port", _DEFAULT_PORT))
        gpu_index = int(getattr(harness, "gpu_index", _DEFAULT_GPU_INDEX))
        boot_timeout_s = float(
            getattr(harness, "boot_timeout_s", _DEFAULT_BOOT_TIMEOUT_S)
        )

        prompt_tokens = int(spec.params.get("prompt_tokens", _DEFAULT_PROMPT_TOKENS))
        output_tokens = int(spec.params.get("output_tokens", _DEFAULT_OUTPUT_TOKENS))

        log_path = self._resolve_log_path(harness, spec.point_id, run_index)
        binary = platform.resolve_binary("server")
        prompt = getattr(harness, "prompt", None) or _synth_prompt(prompt_tokens)

        server = ServerHandle(
            binary=binary,
            config=spec.config,
            host=host,
            port=port,
            gpu_index=gpu_index,
            log_path=log_path,
            boot_timeout_s=boot_timeout_s,
        )

        boot = server.boot()
        if not boot.ready:
            # Boot failure: tear down (best effort) and record the failed run.
            server.teardown()
            return _failed_run(
                run_index, log_path, boot.error or "server failed to boot"
            )

        try:
            client = Client()
            timing: RequestTiming = _run_async(
                client.measure_stream(
                    server.base_url, prompt, max_tokens=output_tokens
                )
            )
        finally:
            # Always release the server, even if measurement raised.
            server.teardown()

        return self._result_from_timing(
            run_index=run_index,
            log_path=log_path,
            timing=timing,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    # ------------------------------------------------------------------ #
    # bench / batched-bench driver paths (task 12.2, R2.2/R2.5/R2.6)
    # ------------------------------------------------------------------ #
    def _measure_batched_bench(
        self, spec: PointSpec, harness: Any, platform: Any
    ) -> RunRepeatResult:
        """Run one ``llama-batched-bench`` sweep over the Decode_Batch_Size set.

        Drives the ``llama-batched-bench`` binary resolved via
        ``platform.resolve_binary("batched-bench")`` once with
        ``--output-format jsonl`` over the campaign's Decode_Batch_Size set (the
        ``-npl`` axis) at the campaign-fixed prompt/output token lengths and the
        Config's context length (generalized from
        ``experiments/smoke/scenario_a_batchsweep.py``). The raw stdout is written
        to the per-run log; the jsonl is parsed into one decode-throughput value
        per batch (``decode_throughput_b{pl}``) plus the per-batch prefill rate
        (``prefill_throughput_b{pl}``). Returns ``ok=False`` when the binary errors
        or no parseable rows are produced (R2.2, R2.6, R2.8).
        """
        run_index = int(getattr(harness, "run_index", 0))
        gpu_index = int(getattr(harness, "gpu_index", _DEFAULT_GPU_INDEX))
        ngl = int(getattr(harness, "ngl", _DEFAULT_NGL))
        timeout_s = float(getattr(harness, "bench_timeout_s", _DEFAULT_BENCH_TIMEOUT_S))

        prompt_tokens = int(spec.params.get("prompt_tokens", _DEFAULT_PROMPT_TOKENS))
        output_tokens = int(spec.params.get("output_tokens", _DEFAULT_OUTPUT_TOKENS))
        decode_batch_sizes = tuple(
            int(b)
            for b in spec.params.get("decode_batch_sizes", _DEFAULT_DECODE_BATCH_SIZES)
        )
        log_path = self._resolve_log_path(harness, spec.point_id, run_index)
        binary = platform.resolve_binary("batched-bench")

        npl = ",".join(str(b) for b in decode_batch_sizes)
        cmd = [
            binary,
            "-m", spec.config.quant_file,
            "-c", str(spec.config.ctx_length),
            "-ngl", str(ngl),
            "-npp", str(prompt_tokens),
            "-ntg", str(output_tokens),
            "-npl", npl,
            "--output-format", "jsonl",
        ]

        stdout, error = self._run_subprocess(cmd, gpu_index, timeout_s, log_path)
        if error is not None:
            return _failed_run(run_index, log_path, error)

        rows = parse_batched_bench_jsonl(stdout or "")
        if not rows:
            return _failed_run(
                run_index, log_path, "no batched-bench jsonl rows parsed"
            )

        metrics: dict[str, float] = {}
        for row in rows:
            metrics[f"decode_throughput_b{row.pl}"] = float(row.speed_tg)
            metrics[f"prefill_throughput_b{row.pl}"] = float(row.speed_pp)
        return RunRepeatResult(
            run_index=run_index,
            discarded_warmup=(run_index == 0),
            ok=True,
            raw_log_path=log_path,
            metrics=metrics,
            error=None,
        )

    def _measure_bench(
        self, spec: PointSpec, harness: Any, platform: Any
    ) -> RunRepeatResult:
        """Run one ``llama-bench`` single-stream batch-1 prefill/decode benchmark.

        Drives the ``llama-bench`` binary resolved via
        ``platform.resolve_binary("bench")`` in its default single-sequence
        (batch-1) mode with ``-o jsonl``, ``-p prompt_tokens`` and
        ``-n output_tokens``. The raw stdout is written to the per-run log; the
        jsonl prefill (``n_gen == 0``) and decode (``n_prompt == 0``) rows are
        parsed into batch-1 ``prefill_throughput`` / ``decode_throughput``.
        Returns ``ok=False`` when the binary errors or neither row is parseable
        (R2.5, R2.8).
        """
        run_index = int(getattr(harness, "run_index", 0))
        gpu_index = int(getattr(harness, "gpu_index", _DEFAULT_GPU_INDEX))
        ngl = int(getattr(harness, "ngl", _DEFAULT_NGL))
        timeout_s = float(getattr(harness, "bench_timeout_s", _DEFAULT_BENCH_TIMEOUT_S))

        prompt_tokens = int(spec.params.get("prompt_tokens", _DEFAULT_PROMPT_TOKENS))
        output_tokens = int(spec.params.get("output_tokens", _DEFAULT_OUTPUT_TOKENS))
        log_path = self._resolve_log_path(harness, spec.point_id, run_index)
        binary = platform.resolve_binary("bench")

        cmd = [
            binary,
            "-m", spec.config.quant_file,
            "-ngl", str(ngl),
            "-p", str(prompt_tokens),
            "-n", str(output_tokens),
            "-o", "jsonl",
        ]

        stdout, error = self._run_subprocess(cmd, gpu_index, timeout_s, log_path)
        if error is not None:
            return _failed_run(run_index, log_path, error)

        parsed = parse_bench_jsonl(stdout or "")
        if "prefill_throughput" not in parsed and "decode_throughput" not in parsed:
            return _failed_run(run_index, log_path, "no llama-bench jsonl rows parsed")

        metrics = {k: float(v) for k, v in parsed.items()}
        return RunRepeatResult(
            run_index=run_index,
            discarded_warmup=(run_index == 0),
            ok=True,
            raw_log_path=log_path,
            metrics=metrics,
            error=None,
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _run_subprocess(
        cmd: list[str], gpu_index: int, timeout_s: float, log_path: str
    ) -> tuple[str | None, str | None]:
        """Run a bench subprocess pinned to ``gpu_index``; return ``(stdout, error)``.

        Pins execution to the campaign GPU via ``CUDA_VISIBLE_DEVICES`` (mirrors
        the smoke scripts) and writes the combined stdout to ``log_path`` for raw
        retention (R5.5). On success returns ``(stdout, None)``; on any failure
        (missing binary, non-zero exit, timeout) returns ``(None, reason)`` with a
        human-readable error reason the caller turns into an ``ok=False`` run.
        """
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_s,
            )
        except FileNotFoundError:
            return (None, f"bench binary not found: {cmd[0]}")
        except subprocess.TimeoutExpired:
            return (None, f"bench timed out after {timeout_s:.0f}s: {cmd[0]}")
        except OSError as exc:  # noqa: BLE001 - surface launch failures as a run failure
            return (None, f"bench launch failed: {exc}")

        # Retain raw stdout (+stderr tail) regardless of exit status (R5.5).
        try:
            with open(log_path, "w") as fh:
                fh.write(proc.stdout or "")
                if proc.stderr:
                    fh.write("\n# --- stderr ---\n")
                    fh.write(proc.stderr)
        except OSError:
            pass  # logging is best-effort; never fail the run on a log-write error

        if proc.returncode != 0:
            tail = (proc.stderr or "")[-500:]
            return (None, f"bench exited {proc.returncode}: {tail}")
        return (proc.stdout or "", None)

    @staticmethod
    def _resolve_log_path(harness: Any, point_id: str, run_index: int) -> str:
        """Resolve the per-run server log path from the harness, else a temp file.

        Prefers the campaign-scoped path the harness provides (R5.5); falls back to
        a temp-file path so the module remains runnable in isolation/tests.
        """
        provider = getattr(harness, "server_log_path", None)
        if callable(provider):
            return provider(point_id, run_index)
        fd, path = tempfile.mkstemp(
            prefix=f"perf_{point_id}_run{run_index:02d}_", suffix=".log"
        )
        os.close(fd)
        return path

    @staticmethod
    def _result_from_timing(
        *,
        run_index: int,
        log_path: str,
        timing: RequestTiming,
        prompt_tokens: int,
        output_tokens: int,
    ) -> RunRepeatResult:
        """Derive the four metrics from one request timing into a RunRepeatResult.

        Returns ``ok=False`` when the request failed or its timing cannot yield the
        metrics (no first token, or a non-positive decode interval). On success the
        metrics map carries ``ttft_ms``, ``tpot_ms``, ``prefill_throughput``,
        ``decode_throughput`` plus ``n_tokens`` for transparency.
        """
        if not timing.ok:
            return _failed_run(
                run_index, log_path, timing.error or "measurement request failed"
            )

        ttft_ms = timing.ttft_ms
        latency_ms = timing.latency_ms
        if ttft_ms is None or latency_ms is None or ttft_ms <= 0.0:
            return _failed_run(
                run_index, log_path, "missing or non-positive TTFT in timing"
            )

        decode_ms = latency_ms - ttft_ms
        decode_time_s = decode_ms / _MS_PER_S
        if decode_time_s <= 0.0:
            return _failed_run(
                run_index, log_path, "non-positive decode interval (no decode phase)"
            )

        ttft_s = ttft_ms / _MS_PER_S
        # Prefill rate: prompt tokens processed over the time to the first token.
        prefill_throughput = prompt_tokens / ttft_s
        # Decode rate: campaign-fixed output tokens over the decode interval (R2.1).
        decode_throughput = output_tokens / decode_time_s
        # TPOT: mean inter-token interval during decode (excludes TTFT). With
        # ``output_tokens`` generated tokens there are ``output_tokens - 1`` gaps.
        gaps = output_tokens - 1
        tpot_ms = decode_ms / gaps if gaps > 0 else decode_ms

        metrics = {
            "ttft_ms": float(ttft_ms),
            "tpot_ms": float(tpot_ms),
            "prefill_throughput": float(prefill_throughput),
            "decode_throughput": float(decode_throughput),
            "n_tokens": float(timing.n_tokens),
        }
        return RunRepeatResult(
            run_index=run_index,
            discarded_warmup=(run_index == 0),
            ok=True,
            raw_log_path=log_path,
            metrics=metrics,
            error=None,
        )
