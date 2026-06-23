"""Platform adapter interface.

A :class:`Platform` isolates per-target hardware differences behind a single
interface so the rest of the suite is platform-agnostic. Concrete adapters
(``A100CudaPlatform`` now, ``JetsonPlatform`` later) implement this Protocol.

Each Platform answers four questions for a campaign (see design "Platform
Adapter"):

- ``descriptor``: the Platform's stable identifier (e.g. ``"a100-cuda"``,
  ``"jetson-orin"``) used to tag every result record and the Run_Manifest (R6.5).
- ``resolve_binary(kind)``: the absolute path to the configured llama.cpp binary
  of the requested kind for this Platform (R6.3, R6.4).
- ``device_total_mib(gpu_index)``: the total device memory in mebibytes (MiB) of
  the target device, used for feasibility and shortfall reporting (R3.5).
- ``is_feasible(config)``: whether a given serving Config can run on this
  Platform, with a reason when it cannot, so infeasible Configs are skipped
  without aborting the campaign (R6.6).

This module is intentionally dependency-light: it imports only :class:`Config`
from :mod:`profile_suite.config` and the standard library, so it stays
importable on its own.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from profile_suite.config import Config

#: The kinds of llama.cpp binaries a Platform can resolve.
BinaryKind = Literal["server", "bench", "batched-bench"]


@runtime_checkable
class Platform(Protocol):
    """Per-target adapter resolving binaries, device memory, and feasibility.

    Implementations select the target hardware for a campaign. Exactly one
    Platform is used per campaign (R6.1).
    """

    #: Stable Platform identifier, e.g. ``"a100-cuda"`` or ``"jetson-orin"``.
    descriptor: str

    def resolve_binary(self, kind: BinaryKind) -> str:
        """Return the absolute path to this Platform's binary of ``kind``.

        ``kind`` is one of ``"server"``, ``"bench"``, or ``"batched-bench"``.
        Implementations resolve the configured binary path for the target
        hardware (e.g. the CUDA build for the A100). The returned path is the
        binary the suite will launch for that role (R6.3, R6.4).
        """
        ...

    def device_total_mib(self, gpu_index: int) -> float:
        """Return total device memory in MiB for device ``gpu_index``."""
        ...

    def is_feasible(self, config: Config) -> tuple[bool, str | None]:
        """Report whether ``config`` can run on this Platform.

        Returns ``(True, None)`` when the Config is feasible, or
        ``(False, reason)`` with a human-readable reason when it is not, so the
        orchestrator can skip only that Config and continue the campaign (R6.6).
        """
        ...
