"""Memory footprint decomposition (pure logic).

Splits a Config's observed peak device-memory footprint into three
non-negative components (R3.2):

- **weights**          - model-weights component, the sum of the
  ``* model buffer size`` lines reported by ``llama-server`` at startup.
- **kv_per_slot**      - KV-cache-per-slot component, the total KV footprint
  (sum of the ``* KV buffer size`` lines) divided by the number of parallel
  KV slots (``-np``).
- **scratch_overhead** - scratch-plus-overhead component, the sum of the
  ``* compute buffer size`` lines plus the residual
  ``observed_peak - weights - kv_total - compute`` (allocator/context
  overhead) clamped to be non-negative.

This module only implements the pure :func:`decompose` arithmetic. The
sum-vs-observed reconciliation flag (R3.3), feasibility filtering (R3.4/R3.5),
and the measurement I/O wiring (boot + sample + parse) live in later tasks.

The ``report`` argument is a ``ServerMemoryReport``-like object exposing the
optional attributes ``weights_mib``, ``kv_mib`` and ``compute_mib`` (each may
be ``None``). It is duck-typed so this module stays importable on its own,
independent of ``profile_suite.harness.logparse``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Union


class MemoryReportLike(Protocol):
    """Structural type for the parsed server memory report (duck-typed).

    Mirrors ``profile_suite.harness.logparse.ServerMemoryReport`` without
    importing it, so ``memory.py`` is importable on its own.
    """

    weights_mib: float | None
    kv_mib: float | None
    compute_mib: float | None


@dataclass(frozen=True)
class MemoryDecomposition:
    """The three non-negative footprint components, in MiB (R3.2).

    ``kv_total`` is retained alongside ``kv_per_slot`` because reconciliation
    against the observed peak (R3.3) uses the total KV footprint, not the
    per-slot value.
    """

    weights: float
    kv_total: float
    kv_per_slot: float
    scratch_overhead: float


def _non_negative(value: float | None) -> float:
    """Coerce an optional, possibly-negative MiB value to a non-negative float.

    ``None`` (component absent from the log) is treated as ``0.0``; any
    negative input is clamped to ``0.0`` so every reported component stays
    non-negative (R3.2).
    """
    if value is None:
        return 0.0
    return value if value > 0.0 else 0.0


def decompose(
    report: MemoryReportLike,
    slot_count: int,
    observed_peak: float,
) -> MemoryDecomposition:
    """Decompose an observed peak footprint into non-negative components.

    Args:
        report: A ``ServerMemoryReport``-like object with optional
            ``weights_mib`` / ``kv_mib`` / ``compute_mib`` attributes (MiB).
        slot_count: The number of parallel KV slots (``-np``); must be >= 1.
        observed_peak: The directly observed peak device-memory footprint (MiB).

    Returns:
        A :class:`MemoryDecomposition` whose ``weights``, ``kv_per_slot`` and
        ``scratch_overhead`` are each non-negative, with
        ``kv_per_slot == kv_total / slot_count`` (R3.2).

    Raises:
        ValueError: If ``slot_count`` is less than 1.
    """
    if slot_count < 1:
        raise ValueError(f"slot_count must be >= 1, got {slot_count!r}")

    weights = _non_negative(getattr(report, "weights_mib", None))
    kv_total = _non_negative(getattr(report, "kv_mib", None))
    compute = _non_negative(getattr(report, "compute_mib", None))
    peak = observed_peak if observed_peak > 0.0 else 0.0

    kv_per_slot = kv_total / slot_count

    # Residual = anything in the observed peak not accounted for by weights,
    # KV total, and the compute buffer (allocator/context overhead). Clamp to
    # >= 0 so scratch+overhead never goes negative when the parsed components
    # already exceed the observed peak.
    residual = peak - weights - kv_total - compute
    scratch_overhead = compute + (residual if residual > 0.0 else 0.0)

    return MemoryDecomposition(
        weights=weights,
        kv_total=kv_total,
        kv_per_slot=kv_per_slot,
        scratch_overhead=scratch_overhead,
    )


# The reconciliation tolerance: a Config is flagged when the absolute
# difference between the summed components and the directly observed peak
# exceeds this fraction of the observed peak (R3.3).
RECONCILE_TOLERANCE = 0.05

# What ``reconciliation_flag`` accepts for its ``components`` argument: either
# the structured :class:`MemoryDecomposition` produced by :func:`decompose`, or
# a bare sequence of the three component values ``(weights, kv_total,
# scratch_overhead)`` in MiB.
Components = Union["MemoryDecomposition", Sequence[float]]


def _component_sum(components: Components) -> float:
    """Sum the three footprint components from either accepted input shape.

    Accepts a :class:`MemoryDecomposition` (summing ``weights + kv_total +
    scratch_overhead`` — the per-slot value is deliberately *not* used, since
    reconciliation is against the total footprint per R3.3) or a sequence of
    exactly three component values ``(weights, kv_total, scratch_overhead)``.

    Raises:
        TypeError: If ``components`` is neither a ``MemoryDecomposition`` nor a
            sequence of three numbers.
    """
    if isinstance(components, MemoryDecomposition):
        return components.weights + components.kv_total + components.scratch_overhead

    if isinstance(components, Sequence) and not isinstance(components, (str, bytes)):
        values = list(components)
        if len(values) != 3:
            raise TypeError(
                "components sequence must have exactly 3 values "
                f"(weights, kv_total, scratch_overhead), got {len(values)}"
            )
        return float(values[0]) + float(values[1]) + float(values[2])

    raise TypeError(
        "components must be a MemoryDecomposition or a sequence of 3 floats, "
        f"got {type(components).__name__}"
    )


def reconciliation_flag(components: Components, observed_peak: float) -> bool:
    """Flag a Config when its component sum disagrees with the observed peak.

    Implements the sum-vs-observed reconciliation check (R3.3 / Property 8):
    the Config is flagged if and only if the absolute difference between the
    component sum and the directly observed peak footprint exceeds 5 percent of
    that observed peak.

    Args:
        components: Either the :class:`MemoryDecomposition` produced by
            :func:`decompose`, or a sequence of the three non-negative
            components ``(weights, kv_total, scratch_overhead)`` in MiB.
        observed_peak: The directly observed peak device-memory footprint (MiB);
            expected to be positive.

    Returns:
        ``True`` if ``abs(component_sum - observed_peak) >
        0.05 * observed_peak``, otherwise ``False``.
    """
    component_sum = _component_sum(components)
    return abs(component_sum - observed_peak) > RECONCILE_TOLERANCE * observed_peak


# ---------------------------------------------------------------------------
# Feasibility filtering and shortfall (R3.4, R3.5 / Property 9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shortfall:
    """The observed memory shortfall of an infeasible Config (R3.5).

    Records *both* operands of the subtraction alongside the derived
    ``shortfall`` so the report carries the full provenance of the value, per
    R3.5 ("reporting both operand values"):

    - ``peak``      - the measured peak device-memory footprint (MiB).
    - ``available`` - the available device memory (MiB).
    - ``shortfall`` - ``peak - available`` (MiB); positive when the Config does
      not fit (infeasible), zero or negative when it fits.
    """

    peak: float
    available: float
    shortfall: float

    @property
    def infeasible(self) -> bool:
        """``True`` iff the peak exceeds the available memory (shortfall > 0)."""
        return self.shortfall > 0.0


def feasible_pairs(
    peaks: "dict[object, float]",
    budget: float,
) -> "list[object]":
    """Return exactly the Configs whose measured peak footprint is <= ``budget``.

    Implements the feasibility frontier under a fixed memory budget (R3.4 /
    Property 9). Given a mapping of Config (or any ``{ctx, slots}`` key) to its
    measured peak footprint in MiB, the result contains exactly those keys whose
    peak is less than or equal to ``budget`` — no more, no fewer.

    The result preserves the iteration order of ``peaks`` (insertion order for a
    plain ``dict``), so the returned collection is stable and deterministic
    across repeated calls with the same input.

    Feasibility is monotonic in the budget: because membership is decided solely
    by the ``peak <= budget`` comparison, any Config infeasible at budget ``B``
    (``peak > B``) is also infeasible at every budget ``B' < B`` (``peak > B' ``).

    Args:
        peaks: Mapping of Config -> measured peak footprint (MiB).
        budget: The supplied memory budget (MiB).

    Returns:
        A list of the keys from ``peaks`` whose peak <= ``budget``, in the
        mapping's iteration order.
    """
    return [config for config, peak in peaks.items() if peak <= budget]


def shortfall(peak: float, available: float) -> Shortfall:
    """Compute the memory shortfall of a Config, recording both operands (R3.5).

    Returns a :class:`Shortfall` record capturing the measured ``peak`` and the
    ``available`` device memory together with the derived
    ``shortfall = peak - available``. Both operands are retained so the report
    carries the full provenance of the value (R3.5 / Property 9).

    A positive ``shortfall`` means the Config does not fit (infeasible); a zero
    or negative value means it fits within the available memory. This is
    consistent with :func:`feasible_pairs`: a Config is feasible under a budget
    equal to ``available`` exactly when its ``shortfall`` is not positive.

    Args:
        peak: The measured peak device-memory footprint (MiB).
        available: The available device memory (MiB).

    Returns:
        A :class:`Shortfall` with ``peak``, ``available`` and
        ``shortfall = peak - available``.
    """
    return Shortfall(peak=peak, available=available, shortfall=peak - available)


# ---------------------------------------------------------------------------
# Memory_Profiler measurement I/O wiring (task 13.1, R3.1/R3.2/R3.3/R3.5/R3.7)
# ---------------------------------------------------------------------------
#
# Everything above this banner is the *pure* memory logic (``decompose``,
# ``reconciliation_flag``, ``feasible_pairs``, ``shortfall``) and stays
# importable on its own. The :class:`Memory_Profiler` below is the measurement
# module that wires the harness I/O around that pure logic; its harness/Client/
# SysProbe/LogParser dependencies are imported lazily inside ``measure_once`` so
# the pure functions remain importable even where the I/O deps (e.g. aiohttp)
# are unavailable.

import os as _os
import tempfile as _tempfile
import threading as _threading
from typing import TYPE_CHECKING, Any, Iterable

from ..results import RunRepeatResult

if TYPE_CHECKING:  # pragma: no cover - typing-only imports (no hard dependency)
    from ..campaign import CampaignConfig
    from .base import PointSpec

# Defaults applied when the harness does not supply the corresponding runtime
# context. They keep the module importable and unit-testable with a light fake
# (mirrors ``modules/performance.py``).
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8090
_DEFAULT_GPU_INDEX = 0
_DEFAULT_BOOT_TIMEOUT_S = 300.0

# Campaign-fixed measurement-workload token lengths used only when the
# CampaignConfig / PointSpec omits them (mirrors the design's sample campaign).
_DEFAULT_PROMPT_TOKENS = 512
_DEFAULT_OUTPUT_TOKENS = 128


def _synth_prompt(prompt_tokens: int) -> str:
    """Synthesize a deterministic prompt of roughly ``prompt_tokens`` tokens.

    Used only when the harness does not supply an explicit campaign-fixed
    prompt. The exact length is a whitespace-token approximation; memory
    measurement cares about driving the server through the workload (so KV and
    compute buffers are touched), not about an exact token count.
    """
    n = max(1, int(prompt_tokens))
    return " ".join(["token"] * n)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine to completion from this synchronous context.

    ``measure_once`` is invoked synchronously by the Reproducibility_Harness run
    loop, so :func:`asyncio.run` is the normal path. A fresh-event-loop fallback
    keeps it working even if invoked while another loop already exists.
    """
    import asyncio

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


class _PeakSampler:
    """Run :meth:`SysProbe.sample_peak` on a background thread over a workload.

    The sampler is started just before the server boots and stopped after the
    measurement workload completes, so the observed peak spans the whole
    boot-through-workload interval (design "Memory decomposition", R3.1). A
    :class:`threading.Event` signals the sampler to stop; the worker stores the
    observed peak which is read after :meth:`stop` joins the thread.
    """

    def __init__(self, sysprobe: Any, gpu_index: int) -> None:
        self._sysprobe = sysprobe
        self._gpu_index = gpu_index
        self._stop = _threading.Event()
        self._peak: float = float("nan")
        self._thread = _threading.Thread(
            target=self._run, name="memory-peak-sampler", daemon=True
        )

    def _run(self) -> None:
        self._peak = self._sysprobe.sample_peak(self._gpu_index, self._stop)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> float:
        """Signal the sampler to stop, join the thread, and return the peak MiB."""
        self._stop.set()
        self._thread.join()
        return self._peak


class Memory_Profiler:
    """Memory footprint + capacity-frontier profiler (R3).

    Implements the :class:`~profile_suite.modules.base.ProfilerModule` Protocol.
    For each Config in the grid this module boots a ``llama-server``, runs the
    campaign-defined measurement workload while a background
    :meth:`SysProbe.sample_peak` thread tracks the observed peak device-memory
    footprint (R3.1), parses the server log for the weights / KV / compute
    buffer sizes via :class:`~profile_suite.harness.logparse.LogParser`, then
    feeds those into the pure :func:`decompose` / :func:`reconciliation_flag` /
    :func:`shortfall` logic.

    Each run's :class:`~profile_suite.results.RunRepeatResult` carries the three
    decomposed components (``weights``, ``kv_per_slot``, ``scratch_overhead``)
    plus ``kv_total`` and the directly ``observed_peak`` (all MiB, R3.2/R3.3),
    the boolean ``reconcile_exceeds`` flag (R3.3), and — when the observed peak
    exceeds the device's total memory — the ``infeasible`` flag with its
    ``shortfall_mib`` / ``available_mib`` / ``peak_mib`` operands (R3.5).

    A Config that fails to boot, or whose peak footprint could not be sampled at
    all, is returned as ``ok=False`` with an error reason so the
    Reproducibility_Harness excludes only that run from the aggregates (R3.7).

    Harness-interface assumptions (duck-typed, all optional with safe defaults;
    mirrors ``modules/performance.py`` and ``modules/switch_cost.py``):

    - ``harness.run_index``      -> int, current run index (0 == discarded warmup).
    - ``harness.host``           -> str, server bind host (default ``"127.0.0.1"``).
    - ``harness.port``           -> int, server bind port (default ``8090``).
    - ``harness.gpu_index``      -> int, GPU to pin/sample (default ``0``).
    - ``harness.boot_timeout_s`` -> float, boot timeout (default ``300.0``, R1.9).
    - ``harness.prompt``         -> str, the campaign-fixed workload prompt; when
      absent a deterministic prompt of approximately ``prompt_tokens`` words is
      synthesized.
    - ``harness.sysprobe``       -> a :class:`SysProbe`-like object exposing
      ``sample_peak(gpu_index, stop)``; defaults to a fresh ``SysProbe``.
    - ``harness.client``         -> a :class:`Client`-like object exposing
      ``measure_stream``; defaults to a fresh ``Client``.
    - ``harness.server_log_path(point_id, run_index)`` -> str, the per-run server
      log path under the campaign-scoped ``profile/runs/<...>/`` dir (R5.5); when
      absent a temp-file path is used so the module is runnable in isolation.

    The server binary is resolved via ``platform.resolve_binary("server")`` and
    the available device memory via ``platform.device_total_mib(gpu_index)``.
    """

    name: str = "Memory_Profiler"

    def points(
        self, cfg: "CampaignConfig", grid: list["Config"]
    ) -> Iterable["PointSpec"]:
        """Enumerate one memory measurement point per Config in the grid.

        Each Config yields a single point (one peak-footprint measurement); the
        axis is empty because there is no sub-axis to sweep for memory. The
        campaign-fixed ``prompt_tokens`` / ``output_tokens`` for the measurement
        workload are carried on the point's ``params``.
        """
        from .base import PointSpec  # local import: avoid a module-load cycle

        prompt_tokens = int(getattr(cfg, "prompt_tokens", _DEFAULT_PROMPT_TOKENS))
        output_tokens = int(getattr(cfg, "output_tokens", _DEFAULT_OUTPUT_TOKENS))

        specs: list[PointSpec] = []
        for config in grid:
            specs.append(
                PointSpec(
                    module=self.name,
                    config=config,
                    axis={},
                    params={
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                    },
                )
            )
        return specs

    def measure_once(
        self, spec: "PointSpec", harness: Any, platform: Any
    ) -> RunRepeatResult:
        """Perform one memory-footprint measurement run for ``spec``.

        Sequence:

        1. Resolve runtime context from the (duck-typed) harness and the server
           binary via ``platform.resolve_binary("server")``.
        2. Start the background :meth:`SysProbe.sample_peak` thread *before*
           booting so the observed peak spans the whole boot-through-workload
           interval (R3.1).
        3. Boot the Config's ``llama-server``. A boot failure stops the sampler
           and returns ``ok=False`` (R3.7).
        4. Drive the measurement workload (one streaming request) so the server
           allocates and exercises its KV/compute buffers.
        5. Stop the sampler to obtain the observed peak (R3.1). A peak that could
           not be sampled at all returns ``ok=False`` (R3.7).
        6. Parse the server log for weights/KV/compute sizes
           (:class:`LogParser`), then call the pure :func:`decompose`,
           :func:`reconciliation_flag`, and (when infeasible) :func:`shortfall`
           logic and populate the metrics.

        The server is always torn down before returning.
        """
        # Lazy imports keep the pure logic above importable without the I/O deps.
        from ..harness.client import Client
        from ..harness.logparse import LogParser
        from ..harness.server import ServerHandle
        from ..harness.sysprobe import SysProbe

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

        sysprobe = getattr(harness, "sysprobe", None) or SysProbe()
        client = getattr(harness, "client", None) or Client()

        server = ServerHandle(
            binary=binary,
            config=spec.config,
            host=host,
            port=port,
            gpu_index=gpu_index,
            log_path=log_path,
            boot_timeout_s=boot_timeout_s,
        )

        # Start sampling just before boot so the observed peak spans the entire
        # boot-through-workload interval (R3.1).
        sampler = _PeakSampler(sysprobe, gpu_index)
        sampler.start()

        boot = server.boot()
        if not boot.ready:
            # Boot failure -> stop sampling, tear down, record the failed run (R3.7).
            sampler.stop()
            server.teardown()
            return _failed_run(
                run_index, log_path, boot.error or "server failed to boot"
            )

        workload_error: str | None = None
        try:
            # Drive the measurement workload so KV/compute buffers are exercised
            # (R3.1). A workload error does not by itself invalidate the peak
            # (boot already allocated weights/KV); it is recorded for context.
            timing = _run_async(
                client.measure_stream(
                    server.base_url, prompt, max_tokens=output_tokens
                )
            )
            if not getattr(timing, "ok", False):
                workload_error = getattr(timing, "error", None) or "workload failed"
        except Exception as exc:  # noqa: BLE001 - surface as a workload note
            workload_error = f"workload raised: {exc!r}"
        finally:
            observed_peak = sampler.stop()
            server.teardown()

        # A peak that could not be sampled at all means the footprint could not be
        # measured -> failed run (R3.7).
        if observed_peak != observed_peak:  # NaN check
            reason = "peak device memory could not be sampled"
            if workload_error is not None:
                reason = f"{reason}; workload: {workload_error}"
            return _failed_run(run_index, log_path, reason)

        # Parse weights / KV / compute buffer sizes from the server log (R3.2).
        report = LogParser().parse_memory(log_path)

        slot_count = spec.config.slot_count
        decomposition = decompose(report, slot_count, observed_peak)

        # Sum-vs-observed reconciliation flag (R3.3).
        flagged = reconciliation_flag(decomposition, observed_peak)

        metrics: dict[str, float] = {
            "weights": decomposition.weights,
            "kv_total": decomposition.kv_total,
            "kv_per_slot": decomposition.kv_per_slot,
            "scratch_overhead": decomposition.scratch_overhead,
            "observed_peak": float(observed_peak),
            "reconcile_exceeds": 1.0 if flagged else 0.0,
        }

        # Feasibility / shortfall against the device's total memory (R3.5).
        available = platform.device_total_mib(gpu_index)
        if available == available and available > 0.0:  # not NaN and positive
            short = shortfall(observed_peak, available)
            metrics["available_mib"] = float(available)
            metrics["peak_mib"] = float(short.peak)
            metrics["shortfall_mib"] = float(short.shortfall)
            metrics["infeasible"] = 1.0 if short.infeasible else 0.0

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
    def _resolve_log_path(harness: Any, point_id: str, run_index: int) -> str:
        """Resolve the per-run server log path from the harness, else a temp file.

        Prefers the campaign-scoped path the harness provides (R5.5); falls back
        to a temp-file path so the module remains runnable in isolation/tests.
        """
        provider = getattr(harness, "server_log_path", None)
        if callable(provider):
            return provider(point_id, run_index)
        fd, path = _tempfile.mkstemp(
            prefix=f"memory_{point_id}_run{run_index:02d}_", suffix=".log"
        )
        _os.close(fd)
        return path


__all__ = [
    "MemoryReportLike",
    "MemoryDecomposition",
    "decompose",
    "RECONCILE_TOLERANCE",
    "Components",
    "reconciliation_flag",
    "Shortfall",
    "feasible_pairs",
    "shortfall",
    "Memory_Profiler",
]
