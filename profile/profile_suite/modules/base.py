"""Module interface: the ``ProfilerModule`` Protocol and ``PointSpec``.

Every measurement module (Switch_Cost_Profiler, Performance_Profiler,
Memory_Profiler, Quality_Module) implements the :class:`ProfilerModule`
Protocol. A module is pure measurement logic: given the resolved
``CampaignConfig`` and the expanded Config_Grid it enumerates the
:class:`PointSpec`\\ s it wants to measure (``points``), and given one such
spec, a ready harness, and the platform adapter it performs a single
measurement run (``measure_once``) returning a
:class:`~profile_suite.results.RunRepeatResult`.

Modules never own the run loop and never persist directly. The
Reproducibility_Harness (design "Reproducibility_Harness run loop") drives
``measure_once`` for the discarded warmup plus each retained repeat, aggregates
over the successful repeats only, and writes the results. Keeping the
warmup/retry/persistence logic in one place is what lets the modules stay small
and unit/property testable with fakes.

``PointSpec`` is the atomic descriptor of *one thing to measure* — one module x
one Config x one axis value. It is deliberately general: a single primary
``config`` covers the Performance/Memory/Quality modules, while the optional
``from_config``/``to_config`` pair covers the Switch_Cost_Profiler's ordered
(from -> to) transitions, and the free-form ``params`` dict carries any other
per-module detail (e.g. ``decode_batch`` for performance). The ``axis`` dict is
the report-facing label for the point (e.g. ``{"decode_batch": 8}`` or
``{"change_type": "model-reload"}``) and feeds the deterministic ``point_id``.

Type references:
    - :class:`~profile_suite.config.Config` and
      :class:`~profile_suite.results.RunRepeatResult` are imported directly
      (both modules are complete).
    - ``CampaignConfig`` lives in ``profile_suite.campaign`` which is built in a
      later task (3.3); it is referenced only under ``TYPE_CHECKING`` so this
      module stays importable on its own now.
    - ``harness`` and ``platform`` are typed loosely (``Any``) here because their
      Protocols are introduced in later tasks; modules only use the harness/
      platform surface they need at call time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Protocol, runtime_checkable

from ..config import Config
from ..results import RunRepeatResult

if TYPE_CHECKING:  # pragma: no cover - typing-only import to avoid a hard dependency
    from ..campaign import CampaignConfig


def make_point_id(module: str, config: Config, axis: dict[str, Any]) -> str:
    """Build a deterministic, filesystem-safe point id from module/config/axis.

    The id is stable across runs for the same inputs (axis keys are sorted) so
    it can name a per-point raw-log subdirectory and join a measurement point to
    its source spec.
    """
    parts: list[str] = [
        module,
        config.quant_file,
        f"c{config.ctx_length}",
        f"np{config.slot_count}",
    ]
    for key in sorted(axis):
        parts.append(f"{key}={axis[key]}")
    raw = "__".join(str(p) for p in parts)
    # Keep only characters that are safe in a path segment.
    return "".join(ch if (ch.isalnum() or ch in "._=-") else "_" for ch in raw)


@dataclass
class PointSpec:
    """Everything a module needs to measure one point (one module x Config x axis).

    Attributes:
        module: The owning module's :attr:`ProfilerModule.name`.
        config: The primary/target Config being measured. For the
            Switch_Cost_Profiler this is the *destination* Config of the
            transition (equal to :attr:`to_config`).
        axis: The report-facing axis label for this point, e.g.
            ``{"decode_batch": 8}`` (performance) or
            ``{"change_type": "model-reload"}`` (switch cost). May be empty for
            a single-point-per-Config module.
        point_id: A deterministic id derived from ``module``/``config``/``axis``;
            filled in by :meth:`__post_init__` when left empty.
        from_config: For the Switch_Cost_Profiler, the source Config of an
            ordered (from -> to) transition; ``None`` for single-Config modules.
        to_config: For the Switch_Cost_Profiler, the destination Config of the
            transition (normally identical to :attr:`config`); ``None`` for
            single-Config modules.
        params: Free-form per-module detail that does not belong on the axis,
            e.g. ``{"prompt_tokens": 512, "output_tokens": 128}`` for performance
            or ``{"local_quality": True}`` for quality. Kept generic so new
            modules need not change this dataclass.
    """

    module: str
    config: Config
    axis: dict[str, Any] = field(default_factory=dict)
    point_id: str = ""
    from_config: Config | None = None
    to_config: Config | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.point_id:
            self.point_id = make_point_id(self.module, self.config, self.axis)


@runtime_checkable
class ProfilerModule(Protocol):
    """The measurement-module interface (design "Module interface").

    Implementations are pure measurement logic: they enumerate the points to
    measure and perform a single measurement run per point. The harness owns the
    warmup-discard, retry, aggregation, and persistence around these calls.
    """

    name: str

    def points(
        self, cfg: "CampaignConfig", grid: list[Config]
    ) -> Iterable[PointSpec]:
        """Enumerate the measurement points this module wants to measure.

        Args:
            cfg: The resolved campaign configuration (defaults already applied).
            grid: The deterministically expanded Config_Grid.

        Returns:
            An iterable of :class:`PointSpec`, one per (module x Config x axis)
            unit this module measures for the campaign.
        """
        ...

    def measure_once(
        self, spec: PointSpec, harness: Any, platform: Any
    ) -> RunRepeatResult:
        """Perform a single measurement run for ``spec``.

        Called once for the discarded warmup and once per retained repeat by the
        Reproducibility_Harness; the module does not loop or persist itself.

        Args:
            spec: The point to measure.
            harness: The shared measurement harness (server/client/sysprobe/...).
            platform: The platform adapter (binary resolution, feasibility, ...).

        Returns:
            A :class:`RunRepeatResult` carrying this run's per-run metrics, the
            raw-log path, and any failure reason.
        """
        ...


__all__ = ["ProfilerModule", "PointSpec", "make_point_id"]
