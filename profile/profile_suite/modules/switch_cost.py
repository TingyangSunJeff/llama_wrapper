"""Switch_Cost_Profiler: C_switch measurement and phase decomposition.

This module measures the reconfiguration cost (C_switch) of changing one or more
serving Knobs and decomposes it into Teardown, Boot, and Warmup phases.

This file currently contains only the *pure* phase-timing reconciliation logic
(`reconcile_phases`). The measurement I/O wiring (`measure_once`, which boots and
tears down a real ``llama-server``) is added later (tasks 11.x). Keeping the pure
logic here lets it be unit/property tested in isolation.

Phase boundaries (design "C_switch phase timing (R1.1, R1.2)") are captured on a
single monotonic clock (``time.monotonic_ns()``), one clock per repeat, so the
component sum reconciles to the total within 50 ms by construction:

- ``t0`` — shutdown signal (SIGINT) sent to the current ``llama-server``
- ``t1`` — current server process exit (``proc.wait()`` returns)
- ``t2`` — new server's first ``/health`` 200 response
- ``t3`` — first generated token of the first post-switch request completes

with::

    Teardown = t1 - t0
    Boot     = t2 - t1
    Warmup   = t3 - t2
    C_switch = t3 - t0
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Protocol, runtime_checkable

from ..config import Config
from ..results import RunRepeatResult

if TYPE_CHECKING:  # pragma: no cover - typing-only imports (no hard runtime dependency)
    from ..campaign import CampaignConfig
    from ..harness.client import Client, RequestTiming
    from ..harness.server import BootResult, ServerHandle
    from .base import PointSpec

# Nanoseconds per millisecond. Monotonic timestamps are expressed in
# nanoseconds (e.g. ``time.monotonic_ns()``); phases are reported in milliseconds.
_NS_PER_MS: float = 1_000_000.0

# Default warmup prompt used for the first post-switch request when the harness
# does not supply one. The content is irrelevant to the measurement; only the
# timing of the first generated token (Warmup phase) matters.
_DEFAULT_WARMUP_PROMPT: str = "Hello"

# Default Warmup-phase timeout (seconds), per R1.10. Overridden by the harness /
# campaign when present.
_DEFAULT_WARMUP_TIMEOUT_S: float = 60.0


@dataclass(frozen=True)
class PhaseTiming:
    """Decomposed C_switch phase timing, all values in milliseconds.

    ``c_switch`` is the total server-unavailable interval; ``teardown``, ``boot``,
    and ``warmup`` are its components. By construction they reconcile to within
    50 ms (Requirement 1.2).
    """

    teardown: float
    boot: float
    warmup: float
    c_switch: float


def reconcile_phases(t0: int, t1: int, t2: int, t3: int) -> PhaseTiming:
    """Reconcile monotonic phase-boundary timestamps into millisecond phases.

    Args:
        t0: Monotonic timestamp (ns) when the shutdown signal is issued.
        t1: Monotonic timestamp (ns) when the old server process exits.
        t2: Monotonic timestamp (ns) when the new server first reports ready.
        t3: Monotonic timestamp (ns) when the first post-switch token completes.

    All four timestamps must come from the same monotonic clock and be
    non-decreasing (``t0 <= t1 <= t2 <= t3``).

    Returns:
        A :class:`PhaseTiming` with ``teardown``, ``boot``, ``warmup``, and
        ``c_switch`` expressed in milliseconds.

    Raises:
        ValueError: If the timestamps are not non-decreasing.
    """
    if not (t0 <= t1 <= t2 <= t3):
        raise ValueError(
            "phase-boundary timestamps must be non-decreasing "
            f"(t0={t0}, t1={t1}, t2={t2}, t3={t3})"
        )

    return PhaseTiming(
        teardown=(t1 - t0) / _NS_PER_MS,
        boot=(t2 - t1) / _NS_PER_MS,
        warmup=(t3 - t2) / _NS_PER_MS,
        c_switch=(t3 - t0) / _NS_PER_MS,
    )


# --------------------------------------------------------------------------- #
# Harness contract required by ``Switch_Cost_Profiler.measure_once``
# --------------------------------------------------------------------------- #
@runtime_checkable
class SwitchCostHarness(Protocol):
    """The harness surface :meth:`Switch_Cost_Profiler.measure_once` requires.

    The concrete campaign harness (full wiring is task 18.1) is free to be richer;
    this Protocol documents only what the Switch_Cost_Profiler depends on, so the
    module stays decoupled from the harness internals and unit/property testable
    with a fake. Two things are needed (design "Module interface"):

    - :attr:`client` — a measurement :class:`~profile_suite.harness.client.Client`
      used to time the Warmup first token (R1.1, R1.10).
    - :meth:`make_server` — constructs (but does not boot) a phase-timed
      :class:`~profile_suite.harness.server.ServerHandle` for a Config. The harness
      is responsible for resolving the server binary
      (``platform.resolve_binary("server")``), assigning host/port/``gpu_index``,
      allocating the per-run log path under the campaign's
      ``profile/runs/<campaign>/`` directory (R5.5), and threading the campaign's
      ``boot_timeout_s`` (R1.9) into the handle.

    Optional attributes (read defensively via ``getattr``) tune the run:

    - ``warmup_prompt`` (str): prompt for the first post-switch request.
    - ``warmup_timeout_s`` (float): Warmup-phase timeout, default 60 s (R1.10).
    """

    client: "Client"

    def make_server(self, config: Config) -> "ServerHandle":
        """Construct an un-booted :class:`ServerHandle` for ``config``.

        The handle's binary, host, port, ``gpu_index``, per-run log path, and
        ``boot_timeout_s`` are all assigned by the harness; the Switch_Cost_Profiler
        only boots and tears it down.
        """
        ...


class Switch_Cost_Profiler:
    """Measure C_switch and decompose it into Teardown / Boot / Warmup (R1).

    Implements the :class:`~profile_suite.modules.base.ProfilerModule` Protocol.
    For each ordered ``(from -> to)`` Config pair in the Config_Grid, one repeat is
    ``teardown(from) + boot(to) + first-token(to)``, timed on a single monotonic
    clock so the components reconcile to the total within 50 ms by construction
    (see :func:`reconcile_phases`). Each measured point is labeled with its
    ``change_type`` (slot-reshape / model-reload / combined), derived purely from
    the Config field deltas via :meth:`Config.change_type` (R1.3-R1.6).

    Boot timeout (``BootResult.ready is False``) and Warmup timeout
    (``RequestTiming.ok is False``) map to phase-tagged failures
    (``ok=False`` with ``error`` prefixed ``boot:`` / ``warmup:``); the
    Reproducibility_Harness run loop excludes failed repeats from the aggregate
    C_switch statistics (R1.9, R1.10).
    """

    #: The module's stable name (used in point ids and result records).
    name: str = "switch_cost"

    def points(
        self, cfg: "CampaignConfig", grid: list[Config]
    ) -> Iterable["PointSpec"]:
        """Enumerate one point per ordered ``(from -> to)`` Config transition.

        Produces every ordered pair of *distinct* Configs in ``grid`` (a Config is
        never switched to itself). Each point's axis carries the derived
        ``change_type`` label so the report can group by change type (R1.6, R7.3),
        and the source/destination Configs are carried on
        :attr:`PointSpec.from_config` / :attr:`PointSpec.to_config`.

        Args:
            cfg: The resolved campaign configuration (unused here beyond the grid,
                but part of the :class:`ProfilerModule` contract).
            grid: The deterministically expanded Config_Grid.

        Returns:
            An iterable of :class:`PointSpec`, one per ordered distinct pair.
        """
        from .base import PointSpec  # local import: avoid a module-load cycle

        specs: list[PointSpec] = []
        for from_cfg in grid:
            for to_cfg in grid:
                if from_cfg == to_cfg:
                    continue
                change_type = from_cfg.change_type(to_cfg)
                specs.append(
                    PointSpec(
                        module=self.name,
                        config=to_cfg,
                        axis={"change_type": change_type},
                        from_config=from_cfg,
                        to_config=to_cfg,
                    )
                )
        return specs

    def measure_once(
        self, spec: "PointSpec", harness: Any, platform: Any
    ) -> RunRepeatResult:
        """Perform one ``teardown(from) + boot(to) + first-token(to)`` repeat.

        Sequence (all timestamps from one :func:`time.monotonic_ns` clock):

        1. **Setup (excluded from C_switch):** construct and boot the *from* server
           so there is a live server to tear down. A setup boot failure is reported
           as a non-phase failure (``error`` prefixed ``setup:``) — there is no
           valid switch to measure.
        2. ``t0`` — immediately before tearing down the *from* server.
        3. ``t1`` — after the *from* server has exited (Teardown complete).
        4. boot the *to* server; ``t2`` — after it reports ready (Boot complete).
           If it does not become ready within the boot timeout, return a
           ``boot:``-tagged failure (R1.9).
        5. issue the first post-switch request; ``t3`` — after the first generated
           token completes (Warmup complete). A Warmup timeout returns a
           ``warmup:``-tagged failure (R1.10).
        6. :func:`reconcile_phases` over ``t0..t3`` populates
           ``teardown_ms`` / ``boot_ms`` / ``warmup_ms`` / ``c_switch_ms``.

        The *to* server is always torn down before returning so each repeat is
        self-contained (the next repeat re-boots its own *from* server in setup).

        Args:
            spec: The transition to measure; uses :attr:`spec.from_config` and
                :attr:`spec.to_config`.
            harness: An object satisfying :class:`SwitchCostHarness`
                (``client`` + ``make_server``; optional ``warmup_prompt`` /
                ``warmup_timeout_s``).
            platform: The platform adapter. Used indirectly — the harness resolves
                the server binary via ``platform.resolve_binary("server")`` when it
                builds each :class:`ServerHandle`; accepted here to honor the
                :class:`ProfilerModule` signature.

        Returns:
            A :class:`RunRepeatResult`. ``ok=True`` carries all four phase metrics;
            a boot/warmup timeout carries ``ok=False`` with a phase-tagged ``error``
            and whatever partial metrics were measured.
        """
        from_cfg = spec.from_config if spec.from_config is not None else spec.config
        to_cfg = spec.to_config if spec.to_config is not None else spec.config

        warmup_prompt: str = getattr(harness, "warmup_prompt", _DEFAULT_WARMUP_PROMPT)
        warmup_timeout_s: float = float(
            getattr(harness, "warmup_timeout_s", _DEFAULT_WARMUP_TIMEOUT_S)
        )

        change_type = spec.axis.get("change_type")

        # --- 1. Setup: bring the *from* server up (excluded from C_switch) ----- #
        from_handle = harness.make_server(from_cfg)
        from_boot: "BootResult" = from_handle.boot()
        if not from_boot.ready:
            # No live *from* server -> there is no valid switch to time. Report a
            # non-phase setup failure; the run loop excludes it from aggregates.
            return RunRepeatResult(
                run_index=0,
                discarded_warmup=False,
                ok=False,
                raw_log_path=from_boot.log_path,
                metrics={},
                error=f"setup: from-server boot failed: {from_boot.error}",
            )

        to_handle = harness.make_server(to_cfg)

        # --- 2-3. Teardown(from): t0 -> t1 ------------------------------------- #
        t0 = time.monotonic_ns()
        from_handle.teardown()
        t1 = time.monotonic_ns()

        # --- 4. Boot(to): t1 -> t2 --------------------------------------------- #
        to_boot: "BootResult" = to_handle.boot()
        t2 = time.monotonic_ns()

        if not to_boot.ready:
            # Boot timeout / early exit -> Boot-phase failure (R1.9).
            partial = reconcile_phases(t0, t1, t1, t1)
            self._safe_teardown(to_handle)
            return RunRepeatResult(
                run_index=0,
                discarded_warmup=False,
                ok=False,
                raw_log_path=to_boot.log_path,
                metrics={
                    "teardown_ms": partial.teardown,
                    "boot_ms": (t2 - t1) / _NS_PER_MS,
                },
                error=f"boot: {to_boot.error}",
            )

        # --- 5. Warmup first token: t2 -> t3 ----------------------------------- #
        warmup: "RequestTiming" = self._first_token(
            harness.client, to_handle.base_url, warmup_prompt, warmup_timeout_s
        )
        t3 = time.monotonic_ns()

        if not warmup.ok:
            # Warmup timeout / error -> Warmup-phase failure (R1.10).
            self._safe_teardown(to_handle)
            return RunRepeatResult(
                run_index=0,
                discarded_warmup=False,
                ok=False,
                raw_log_path=to_boot.log_path,
                metrics={
                    "teardown_ms": (t1 - t0) / _NS_PER_MS,
                    "boot_ms": (t2 - t1) / _NS_PER_MS,
                },
                error=f"warmup: {warmup.error}",
            )

        # --- 6. Success: reconcile the four phases ----------------------------- #
        timing = reconcile_phases(t0, t1, t2, t3)
        self._safe_teardown(to_handle)

        metrics: dict[str, float] = {
            "teardown_ms": timing.teardown,
            "boot_ms": timing.boot,
            "warmup_ms": timing.warmup,
            "c_switch_ms": timing.c_switch,
        }

        return RunRepeatResult(
            run_index=0,
            discarded_warmup=False,
            ok=True,
            raw_log_path=to_boot.log_path,
            metrics=metrics,
            error=None,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_token(
        client: "Client", base_url: str, prompt: str, warmup_timeout_s: float
    ) -> "RequestTiming":
        """Run the async :meth:`Client.first_token` from this synchronous path.

        The Reproducibility_Harness run loop is synchronous, so the warmup
        coroutine is driven to completion here via :func:`asyncio.run`.
        """
        return asyncio.run(
            client.first_token(base_url, prompt, warmup_timeout_s=warmup_timeout_s)
        )

    @staticmethod
    def _safe_teardown(handle: "ServerHandle") -> None:
        """Best-effort teardown of a server handle, swallowing any error.

        Cleanup must never turn a recorded measurement (success or phase failure)
        into an exception, so teardown errors are ignored here.
        """
        try:
            handle.teardown()
        except Exception:  # noqa: BLE001 - cleanup must not mask the result
            pass


__all__ = [
    "PhaseTiming",
    "reconcile_phases",
    "SwitchCostHarness",
    "Switch_Cost_Profiler",
]
