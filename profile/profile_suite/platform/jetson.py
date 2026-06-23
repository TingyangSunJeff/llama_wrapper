"""Jetson-class edge platform adapter.

:class:`JetsonPlatform` is the concrete :class:`~profile_suite.platform.base.Platform`
implementation for an NVIDIA Jetson-class edge device (e.g. Jetson Orin) running
the Jetson-configured build of llama.cpp (R6.4). It is the structural twin of
:class:`~profile_suite.platform.a100.A100CudaPlatform` and answers the same four
Platform questions for a campaign:

- ``descriptor`` is the stable string ``"jetson-orin"`` used to tag every result
  record and the Run_Manifest (R6.5).
- ``resolve_binary(kind)`` maps the binary roles ``"server"`` / ``"bench"`` /
  ``"batched-bench"`` onto the configured Jetson binary paths (R6.4).
- ``device_total_mib(gpu_index)`` reports the device's total memory using the
  shared :class:`~profile_suite.harness.sysprobe.SysProbe` probe. On Jetson the
  GPU shares the system's unified memory, so this reports the unified pool.
- ``is_feasible(config)`` answers whether a serving :class:`~profile_suite.config.Config`
  can run on this Platform, returning a reason when it cannot so the orchestrator
  can skip only that Config and continue (R6.6).

Feasibility heuristic (documented):
    Without loading a model we cannot know its exact device footprint, so the
    default answer is *feasible*. When a memory budget is available -- either an
    explicit ``memory_budget_mib`` passed by the campaign, or the device's
    queried total (unified) memory -- this adapter applies a conservative
    lower-bound estimate of the KV-cache footprint (which grows with
    ``ctx_length`` and ``slot_count``) and declines a Config only when even that
    lower bound already exceeds the budget. Because the estimate excludes model
    weights and compute scratch, it never *over*-accepts: a Config it rejects is
    genuinely infeasible. The estimate is intentionally simple; the
    Memory_Profiler measures the true peak and records exact shortfalls at run
    time (R3.5). This mirrors the conservative A100 heuristic so the two
    Platforms behave consistently; the tighter memory of edge devices makes the
    lower-bound rejection more likely to fire there, which is the intended
    safety margin.
"""

from __future__ import annotations

import math

from profile_suite.config import Config
from profile_suite.harness.sysprobe import SysProbe

from .base import BinaryKind

#: Stable Platform descriptor for the Jetson-class edge target (R6.5).
JETSON_DESCRIPTOR = "jetson-orin"

#: Conservative KV-cache cost used by the feasibility lower-bound estimate, in
#: MiB per 1024 context tokens per parallel slot. This is deliberately a small,
#: model-agnostic constant: the true per-token KV cost depends on the model's
#: layer count and hidden size, which are unknown until the server boots. Using a
#: modest constant keeps the estimate a genuine lower bound so feasibility is only
#: ever declined when a Config is clearly infeasible. Kept identical to the A100
#: constant so the two Platforms share one documented heuristic.
DEFAULT_KV_MIB_PER_KTOKEN_PER_SLOT = 8.0


class JetsonPlatform:
    """Platform adapter for a Jetson-class edge device running llama.cpp.

    Constructed with the campaign's configured Jetson binary paths and the pinned
    GPU index. Conforms structurally to the
    :class:`~profile_suite.platform.base.Platform` Protocol.
    """

    descriptor: str = JETSON_DESCRIPTOR

    def __init__(
        self,
        server_binary: str,
        bench_binary: str,
        batched_bench_binary: str,
        gpu_index: int = 0,
        *,
        sysprobe: SysProbe | None = None,
        memory_budget_mib: float | None = None,
        kv_mib_per_ktoken_per_slot: float = DEFAULT_KV_MIB_PER_KTOKEN_PER_SLOT,
    ) -> None:
        """Create the adapter.

        Args:
            server_binary: Absolute path to the Jetson ``llama-server`` binary.
            bench_binary: Absolute path to the Jetson ``llama-bench`` binary.
            batched_bench_binary: Absolute path to the Jetson
                ``llama-batched-bench`` binary.
            gpu_index: The pinned GPU index for this campaign (default ``0``).
            sysprobe: Optional :class:`SysProbe` used for device-memory queries.
                Defaults to a fresh :class:`SysProbe` instance.
            memory_budget_mib: Optional explicit device-memory budget in MiB. When
                provided it takes precedence over the queried total for
                feasibility decisions (mirrors the campaign ``memory_budget_mib``).
            kv_mib_per_ktoken_per_slot: Override for the KV-cache cost constant
                used by the conservative feasibility estimate.
        """
        self.server_binary = server_binary
        self.bench_binary = bench_binary
        self.batched_bench_binary = batched_bench_binary
        self.gpu_index = gpu_index
        self._sysprobe = sysprobe if sysprobe is not None else SysProbe()
        self._memory_budget_mib = memory_budget_mib
        self._kv_mib_per_ktoken_per_slot = kv_mib_per_ktoken_per_slot

    # ------------------------------------------------------------------ #
    # Binary resolution (R6.4)
    # ------------------------------------------------------------------ #
    def resolve_binary(self, kind: BinaryKind) -> str:
        """Return the configured Jetson binary path for ``kind``.

        ``kind`` is one of ``"server"``, ``"bench"``, or ``"batched-bench"``.
        Raises :class:`ValueError` for any unrecognized kind so a typo surfaces
        immediately rather than launching the wrong binary.
        """
        if kind == "server":
            return self.server_binary
        if kind == "bench":
            return self.bench_binary
        if kind == "batched-bench":
            return self.batched_bench_binary
        raise ValueError(
            f"unknown binary kind {kind!r}; "
            "expected 'server', 'bench', or 'batched-bench'"
        )

    # ------------------------------------------------------------------ #
    # Device memory (R3.5)
    # ------------------------------------------------------------------ #
    def device_total_mib(self, gpu_index: int) -> float:
        """Return total device memory in MiB for ``gpu_index`` via :class:`SysProbe`.

        On Jetson-class devices the GPU shares the system's unified memory, so the
        reported value is the unified memory pool as seen by the probe.
        """
        return self._sysprobe.device_memory_total_mib(gpu_index)

    # ------------------------------------------------------------------ #
    # Feasibility (R6.6)
    # ------------------------------------------------------------------ #
    def is_feasible(self, config: Config) -> tuple[bool, str | None]:
        """Report whether ``config`` can run on this Platform.

        Returns ``(True, None)`` when the Config is feasible (the default), or
        ``(False, reason)`` when a known device-memory budget is already exceeded
        by the conservative KV-cache lower-bound estimate. See the module
        docstring for the heuristic's rationale.
        """
        budget = self._effective_budget()
        if budget is None or not math.isfinite(budget) or budget <= 0:
            # No usable budget -> default feasible (R6.6 default path).
            return (True, None)

        estimate = self._kv_lower_bound_mib(config)
        if estimate > budget:
            reason = (
                "estimated KV-cache lower bound "
                f"{estimate:.0f} MiB (ctx_length={config.ctx_length}, "
                f"slot_count={config.slot_count}) exceeds device-memory budget "
                f"{budget:.0f} MiB on {self.descriptor}"
            )
            return (False, reason)
        return (True, None)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _effective_budget(self) -> float | None:
        """Return the budget to use for feasibility, or ``None`` if unknown.

        Prefers the explicit ``memory_budget_mib`` when set; otherwise falls back
        to the device's queried total (unified) memory. A failed/unavailable
        query yields ``nan``, which the caller treats as "no usable budget".
        """
        if self._memory_budget_mib is not None:
            return self._memory_budget_mib
        try:
            return self.device_total_mib(self.gpu_index)
        except Exception:  # noqa: BLE001 - probing must never abort feasibility
            return None

    def _kv_lower_bound_mib(self, config: Config) -> float:
        """Conservative lower-bound KV-cache footprint estimate in MiB."""
        ktokens = config.ctx_length / 1024.0
        slots = max(config.slot_count, 0)
        return ktokens * slots * self._kv_mib_per_ktoken_per_slot


__all__ = [
    "JetsonPlatform",
    "JETSON_DESCRIPTOR",
    "DEFAULT_KV_MIB_PER_KTOKEN_PER_SLOT",
]
