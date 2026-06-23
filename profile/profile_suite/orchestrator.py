"""Campaign Orchestrator: end-to-end campaign execution wiring.

This module is the thin top of the suite (design "Component responsibilities" ->
"Campaign Orchestrator"). :func:`run` ties every other component together into a
single, isolated campaign run:

1. **Load** the campaign file into a fully-resolved
   :class:`~profile_suite.campaign.CampaignConfig`
   (:func:`profile_suite.loader.load_campaign`).
2. **Validate** it (:func:`profile_suite.validation.validate_campaign`). On any
   validation failure the campaign is **declined with its specific reason** and
   **no campaign output directory is produced** (R8.4, R8.6, R6.2) — the
   orchestrator never calls ``begin_campaign`` for an invalid campaign, and
   ``begin_campaign`` is the only thing that creates the campaign-scoped
   directory, so an invalid campaign leaves the filesystem untouched.
3. **Expand** the Config_Grid (:func:`profile_suite.config.expand_grid`).
4. **Begin** the campaign through the Reproducibility_Harness
   (:meth:`ReproHarness.begin_campaign`): pin the GPU, capture the environment +
   model checksums, record the Pinned_Build, and write ``manifest.json`` /
   ``environment.json`` under the single campaign-scoped
   ``profile/runs/<campaign>_<UTC-stamp>/`` directory.
5. **Iterate** the enabled measurement modules x their measurement points,
   driving each point through :meth:`ReproHarness.run_point` (one discarded
   warmup + retry-to-stopping-rule), **skipping platform-infeasible Configs**
   with their reason recorded as a ``"platform-infeasible"`` MeasurementPoint
   (R6.6).
6. **Persist** the resulting :class:`~profile_suite.results.MeasurementPoint`\\ s
   (with their per-metric aggregates) to ``points.json`` under the campaign
   directory.
7. **Finalize** the manifest (attach the retained raw-log paths) and **invoke the
   Reporting_Module** (:func:`profile_suite.reporting.report.generate_artifacts`)
   to emit the per-Platform artifacts under ``artifacts/<platform>/``.

Isolation invariant (R8.1, R8.5): every path the orchestrator writes lives under
the campaign-scoped directory created by ``begin_campaign``, which itself lives
under ``profile/runs/`` (or a caller-supplied ``runs_root`` for tests). The
orchestrator creates no output at all for a declined campaign.

Harness context (the duck-typed object the modules read)
--------------------------------------------------------
The measurement modules (Switch_Cost_Profiler, Performance_Profiler,
Memory_Profiler) are pure measurement logic that read a small, duck-typed surface
off the harness object they are handed (see each module's docstring). The
orchestrator supplies that surface via :class:`MeasurementContext`, which exposes:

- ``run_index``        - current run index (threaded on by the run loop; 0 ==
  discarded warmup).
- ``host`` / ``port``  - server bind host/port.
- ``gpu_index``        - the pinned GPU index.
- ``boot_timeout_s``   - server boot timeout (R1.9).
- ``warmup_timeout_s`` - first-token warmup timeout (R1.10).
- ``prompt``           - the campaign-fixed measurement prompt (``None`` lets a
  module synthesize one).
- ``client``           - a :class:`~profile_suite.harness.client.Client`.
- ``sysprobe``         - a :class:`~profile_suite.harness.sysprobe.SysProbe`.
- ``server_log_path(point_id, run_index)`` - the per-run server log path under the
  campaign ``raw/`` dir (R5.5).
- ``make_server(config)`` - construct an un-booted
  :class:`~profile_suite.harness.server.ServerHandle` for a Config (used by the
  Switch_Cost_Profiler), with its binary/host/port/gpu_index/log_path/boot_timeout
  all assigned.

_Requirements: 5.4, 5.9, 6.2, 6.6, 8.4, 8.6_
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .config import Config, expand_grid
from .loader import load_campaign
from .results import MeasurementPoint, RunManifest, RunRepeatResult
from .validation import validate_campaign

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from .campaign import CampaignConfig
    from .harness.repro import ReproHarness
    from .platform.base import Platform

# Default server bind host/port for the campaign harness context. A campaign runs
# its measurement points sequentially, so a single fixed port is sufficient; the
# value is overridable on the context for tests / parallel runs.
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8090

# The glossary names (as used in ``enabled_modules``) of the three profiler
# modules that follow the ProfilerModule ``points`` / ``measure_once`` Protocol
# and therefore flow through the run loop. ``Quality_Module`` is handled
# separately (it is static/cited logic, not a server-driving profiler), and
# ``Reproducibility_Harness`` / ``Reporting_Module`` are infrastructure rather
# than point-producing measurement modules.
_PROFILER_MODULE_NAMES: tuple[str, ...] = (
    "Switch_Cost_Profiler",
    "Performance_Profiler",
    "Memory_Profiler",
)
_QUALITY_MODULE_NAME = "Quality_Module"
_REPORTING_MODULE_NAME = "Reporting_Module"

# File names written under the campaign directory.
POINTS_FILENAME = "points.json"
QUALITY_FILENAME = "quality.json"
ARTIFACTS_DIRNAME = "artifacts"
MANIFEST_FILENAME = "manifest.json"


@dataclass
class CampaignRunResult:
    """The structured outcome of :func:`run`.

    Attributes:
        ok: ``True`` iff the campaign was valid and executed; ``False`` iff it was
            declined at validation.
        reason: ``None`` on success; the specific validation-failure reason on a
            decline (R6.2, R8.4, R8.6).
        campaign_dir: Absolute path to the campaign-scoped output directory, or
            ``None`` when the campaign was declined (no output is produced on a
            decline).
        manifest: The campaign :class:`RunManifest` (``None`` on a decline).
        points: The measurement points produced (empty on a decline).
        artifacts: The per-Platform artifact map returned by the Reporting_Module
            (empty when reporting was not run / on a decline).
        missing_paths: Any referenced paths that did not exist (populated only for
            a path-existence decline).
    """

    ok: bool
    reason: Optional[str] = None
    campaign_dir: Optional[str] = None
    manifest: Optional[RunManifest] = None
    points: list[MeasurementPoint] = field(default_factory=list)
    artifacts: dict[str, dict[str, Optional[str]]] = field(default_factory=dict)
    missing_paths: list[str] = field(default_factory=list)


class MeasurementContext:
    """The duck-typed harness surface the measurement modules read.

    One context is built per campaign and reused across every measurement point;
    the orchestrator updates :attr:`run_index` (the run loop threads the current
    run index on before each :meth:`ProfilerModule.measure_once` call) and the
    current point id/module (so per-run server logs land in a stable,
    point-scoped location under the campaign ``raw/`` directory, R5.5).

    The object deliberately exposes only the small surface the modules use
    (documented in :mod:`profile_suite.modules.*`), keeping the modules decoupled
    from the orchestrator and unit-testable with a lightweight fake.
    """

    def __init__(
        self,
        *,
        campaign_dir: Path,
        platform: "Platform",
        gpu_index: int,
        boot_timeout_s: float,
        warmup_timeout_s: float,
        prompt: Optional[str],
        client: Any,
        sysprobe: Any,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
    ) -> None:
        self.campaign_dir = Path(campaign_dir)
        self.platform = platform
        self.gpu_index = int(gpu_index)
        self.boot_timeout_s = float(boot_timeout_s)
        self.warmup_timeout_s = float(warmup_timeout_s)
        self.prompt = prompt
        self.client = client
        self.sysprobe = sysprobe
        self.host = host
        self.port = int(port)

        # Threaded on per run / per point by the orchestrator.
        self.run_index: int = 0
        self._current_module: str = "module"
        self._current_point_id: str = "point"

        self._raw_root = self.campaign_dir / "raw"

    # ------------------------------------------------------------------ #
    # Per-point / per-run threading
    # ------------------------------------------------------------------ #
    def _bind_point(self, module: str, point_id: str) -> None:
        """Record the point currently being measured (for log-path scoping)."""
        self._current_module = module or "module"
        self._current_point_id = point_id or "point"

    # ------------------------------------------------------------------ #
    # Surface read by the modules
    # ------------------------------------------------------------------ #
    def server_log_path(self, point_id: str, run_index: int) -> str:
        """Return the per-run server log path under the campaign ``raw/`` dir (R5.5).

        Logs are organized as ``raw/<module>/<point_id>/run<NN>.log`` so the
        Reproducibility_Harness can group retained logs by module when it
        finalizes the manifest. The parent directory is created eagerly because
        :class:`ServerHandle` opens the log file for writing on boot. The path is
        always under the campaign-scoped directory, never outside ``profile/``
        (R8.1, R8.5).
        """
        module = _sanitize_segment(self._current_module)
        pid = _sanitize_segment(point_id or self._current_point_id)
        point_dir = self._raw_root / module / pid
        point_dir.mkdir(parents=True, exist_ok=True)
        return str(point_dir / f"run{int(run_index):02d}.log")

    def make_server(self, config: Config) -> Any:
        """Construct an un-booted :class:`ServerHandle` for ``config``.

        Used by the Switch_Cost_Profiler (which boots/tears down servers itself).
        The binary is resolved via ``platform.resolve_binary("server")`` and the
        per-run log path is allocated under the campaign ``raw/`` dir for the
        current point/run.
        """
        from .harness.server import ServerHandle

        binary = self.platform.resolve_binary("server")
        log_path = self.server_log_path(self._current_point_id, self.run_index)
        return ServerHandle(
            binary=binary,
            config=config,
            host=self.host,
            port=self.port,
            gpu_index=self.gpu_index,
            log_path=log_path,
            boot_timeout_s=self.boot_timeout_s,
        )


def run(
    campaign_path: str,
    *,
    runs_root: Optional[str] = None,
    platform: "Optional[Platform]" = None,
    harness: "Optional[ReproHarness]" = None,
    module_overrides: Optional[dict[str, Any]] = None,
) -> CampaignRunResult:
    """Load, validate, and run a campaign end-to-end (the orchestrator entry point).

    Args:
        campaign_path: Path to the campaign definition file (``.yaml``/``.yml``/
            ``.json``).
        runs_root: Optional base directory for the campaign-scoped output
            directory. Defaults to ``profile/runs`` (the only production location).
            Mainly used by tests to redirect output to a temp dir while preserving
            the isolation invariant.
        platform: Optional Platform adapter. When omitted one is built from the
            campaign's ``platform`` descriptor and binary paths (the same logic
            the Reproducibility_Harness uses). Supplying a fake here lets a
            campaign run without a real GPU / binaries.
        harness: Optional :class:`ReproHarness`. When omitted one is constructed
            against ``runs_root``.
        module_overrides: Optional mapping of glossary module name (e.g.
            ``"Performance_Profiler"``) to a module instance, replacing the default
            profiler module for that name. Lets a campaign run with fake modules
            that do not boot real servers.

    Returns:
        A :class:`CampaignRunResult`. On a validation decline, ``ok=False`` with
        the specific ``reason`` and ``campaign_dir=None`` (no output is produced).
        On success, ``ok=True`` with the campaign directory, manifest, produced
        measurement points, and the per-Platform artifact map.
    """
    from .harness.repro import ReproHarness

    # --- 1. Load -------------------------------------------------------- #
    cfg = load_campaign(campaign_path)

    # --- 2. Validate (decline => reason, NO output dir) ----------------- #
    verdict = validate_campaign(cfg)
    if not verdict.ok:
        # No campaign output is produced for an invalid campaign (R8.4, R8.6,
        # R6.2): we never reach begin_campaign, which is the only thing that
        # creates the campaign-scoped directory.
        return CampaignRunResult(
            ok=False,
            reason=verdict.reason,
            campaign_dir=None,
            missing_paths=list(verdict.missing_paths),
        )

    # --- 3. Expand the Config_Grid -------------------------------------- #
    grid = expand_grid(cfg.config_grid)

    # --- 4. Begin the campaign (manifest, env, pin, dirs) --------------- #
    if harness is None:
        harness = ReproHarness(runs_root=runs_root)
    if platform is None:
        platform = _build_platform(cfg)

    manifest = harness.begin_campaign(cfg, platform)
    campaign_dir = Path(harness._campaign_dir_for(manifest))

    # --- 5. Build the harness context the modules read ------------------ #
    context = _build_context(cfg, platform, campaign_dir)

    # --- 6. Iterate enabled modules x points through the run loop ------- #
    modules = _resolve_profiler_modules(cfg, module_overrides)
    min_success = int(cfg.run_repeats)  # type: ignore[arg-type]
    max_attempts = int(cfg.max_attempts)  # type: ignore[arg-type]

    points: list[MeasurementPoint] = []
    for module in modules:
        for spec in module.points(cfg, grid):
            point = _run_point_or_skip(
                harness=harness,
                module=module,
                spec=spec,
                platform=platform,
                context=context,
                min_success=min_success,
                max_attempts=max_attempts,
            )
            points.append(point)

    # --- 6b. Quality_Module (static/cited; handled outside the run loop)  #
    quality_records = _run_quality(cfg, grid)
    if quality_records is not None:
        _write_json(campaign_dir / QUALITY_FILENAME, quality_records)

    # --- 7. Persist the measurement points ------------------------------ #
    _write_json(
        campaign_dir / POINTS_FILENAME, [p.to_dict() for p in points]
    )

    # --- 8. Finalize the manifest (attach raw-log paths) ---------------- #
    harness.finalize_campaign(manifest)

    # --- 9. Invoke the Reporting_Module (per-Platform artifacts) -------- #
    artifacts: dict[str, dict[str, Optional[str]]] = {}
    if _reporting_enabled(cfg):
        from .reporting.report import generate_artifacts

        artifacts = generate_artifacts(
            points,
            manifest,
            str(campaign_dir / ARTIFACTS_DIRNAME),
            manifest_ref=str(campaign_dir / MANIFEST_FILENAME),
        )

    return CampaignRunResult(
        ok=True,
        reason=None,
        campaign_dir=str(campaign_dir),
        manifest=manifest,
        points=points,
        artifacts=artifacts,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run_point_or_skip(
    *,
    harness: "ReproHarness",
    module: Any,
    spec: Any,
    platform: "Platform",
    context: MeasurementContext,
    min_success: int,
    max_attempts: int,
) -> MeasurementPoint:
    """Run one measurement point, or record it platform-infeasible and skip (R6.6).

    Before measuring, the point's Config is checked with
    ``platform.is_feasible(config)``. An infeasible Config yields a
    ``"platform-infeasible"`` :class:`MeasurementPoint` carrying the reason and no
    repeats — the Config is skipped without running, and the campaign continues
    (R6.6). A feasible Config is measured through :meth:`ReproHarness.run_point`
    with a ``measure_fn`` that threads the current run index onto the context and
    calls ``module.measure_once(spec, context, platform)``.
    """
    feasible, reason = platform.is_feasible(spec.config)
    if not feasible:
        return MeasurementPoint(
            point_id=spec.point_id,
            module=spec.module,
            config=spec.config,
            axis=dict(spec.axis),
            repeats=[],
            aggregates={},
            status="platform-infeasible",
            reason=reason,
        )

    # Bind the point so per-run server logs land in a stable, point-scoped
    # location under the campaign raw/ dir.
    context._bind_point(spec.module, spec.point_id)

    def measure_fn(run_index: int) -> RunRepeatResult:
        # Thread the run index onto the harness context the module reads, then
        # measure once. The run loop owns warmup-discard / retry / aggregation;
        # the module just performs a single run.
        context.run_index = run_index
        result = module.measure_once(spec, context, platform)
        # Stamp the authoritative run index so it matches the loop's bookkeeping.
        result.run_index = run_index
        return result

    return harness.run_point(
        measure_fn,
        min_success=min_success,
        max_attempts=max_attempts,
        point_id=spec.point_id,
        module=spec.module,
        config=spec.config,
        axis=dict(spec.axis),
    )


def _build_context(
    cfg: "CampaignConfig",
    platform: "Platform",
    campaign_dir: Path,
) -> MeasurementContext:
    """Construct the per-campaign :class:`MeasurementContext` the modules read."""
    from .harness.client import Client
    from .harness.sysprobe import SysProbe

    warmup_timeout_s = _as_float(getattr(cfg, "warmup_timeout_s", 60.0), 60.0)
    boot_timeout_s = _as_float(getattr(cfg, "boot_timeout_s", 300.0), 300.0)

    return MeasurementContext(
        campaign_dir=campaign_dir,
        platform=platform,
        gpu_index=int(cfg.gpu_index),
        boot_timeout_s=boot_timeout_s,
        warmup_timeout_s=warmup_timeout_s,
        prompt=None,  # modules synthesize a campaign-fixed prompt when None
        client=Client(),
        sysprobe=SysProbe(),
    )


def _resolve_profiler_modules(
    cfg: "CampaignConfig",
    module_overrides: Optional[dict[str, Any]],
) -> list[Any]:
    """Instantiate the enabled profiler modules, preserving campaign order.

    Only the three point-producing profiler modules
    (:data:`_PROFILER_MODULE_NAMES`) that the campaign enabled are returned, each
    as a module instance. ``module_overrides`` may replace any default with a
    (fake) instance keyed by its glossary name. Modules that are not enabled, and
    the non-profiler module names (Quality / Reproducibility_Harness / Reporting),
    are not included here.
    """
    overrides = module_overrides or {}
    enabled = list(getattr(cfg, "enabled_modules", ()) or ())

    defaults = _default_module_factories()
    resolved: list[Any] = []
    for name in enabled:
        if name not in _PROFILER_MODULE_NAMES:
            continue
        if name in overrides:
            resolved.append(overrides[name])
        else:
            resolved.append(defaults[name]())
    return resolved


def _default_module_factories() -> dict[str, Any]:
    """Map each profiler glossary name to a zero-arg factory for its module."""
    from .modules.memory import Memory_Profiler
    from .modules.performance import Performance_Profiler
    from .modules.switch_cost import Switch_Cost_Profiler

    return {
        "Switch_Cost_Profiler": Switch_Cost_Profiler,
        "Performance_Profiler": Performance_Profiler,
        "Memory_Profiler": Memory_Profiler,
    }


def _run_quality(
    cfg: "CampaignConfig", grid: list[Config]
) -> Optional[list[dict[str, Any]]]:
    """Compute the quality axis for the campaign's quant formats, if enabled.

    The Quality_Module is static/cited decision logic rather than a
    server-driving profiler, so it is computed directly here (outside the run
    loop): one labeled value per distinct quant format in the grid (R4.1), cited
    by default or locally-measured when ``local_quality`` is enabled. Returns the
    serialized values, or ``None`` when the module is not enabled.
    """
    enabled = list(getattr(cfg, "enabled_modules", ()) or ())
    if _QUALITY_MODULE_NAME not in enabled:
        return None

    from .modules.quality import quality_values

    # Distinct quant formats, derived from the grid's quant files in first-seen
    # order so the output is deterministic.
    formats: list[str] = []
    seen: set[str] = set()
    for config in grid:
        fmt = _quant_format(config.quant_file)
        if fmt not in seen:
            seen.add(fmt)
            formats.append(fmt)

    local_quality = bool(getattr(cfg, "local_quality", False))
    values = quality_values(formats, local_quality=local_quality)
    return [v.to_dict() for v in values]


def _reporting_enabled(cfg: "CampaignConfig") -> bool:
    """Return ``True`` iff the campaign enabled the Reporting_Module."""
    enabled = list(getattr(cfg, "enabled_modules", ()) or ())
    return _REPORTING_MODULE_NAME in enabled


def _build_platform(cfg: "CampaignConfig") -> "Platform":
    """Build a Platform adapter from the campaign's platform descriptor.

    Mirrors :meth:`ReproHarness._build_platform` (the harness builds the same
    adapter for binary resolution and device metadata). Replicated here so the
    orchestrator can build the platform it threads into the harness context and
    the feasibility checks without reaching into harness internals.
    """
    from .platform.a100 import A100_DESCRIPTOR, A100CudaPlatform
    from .platform.jetson import JETSON_DESCRIPTOR, JetsonPlatform

    budget = getattr(cfg, "memory_budget_mib", None)
    budget_val = budget if isinstance(budget, (int, float)) else None
    kwargs: dict[str, Any] = dict(
        server_binary=cfg.server_binary,
        bench_binary=cfg.bench_binary,
        batched_bench_binary=cfg.batched_bench_binary,
        gpu_index=int(cfg.gpu_index),
        memory_budget_mib=budget_val,
    )
    descriptor = str(cfg.platform)
    if descriptor == JETSON_DESCRIPTOR:
        return JetsonPlatform(**kwargs)
    if descriptor == A100_DESCRIPTOR:
        return A100CudaPlatform(**kwargs)
    raise ValueError(
        f"unknown platform descriptor {descriptor!r}; "
        f"expected {A100_DESCRIPTOR!r} or {JETSON_DESCRIPTOR!r}"
    )


def _quant_format(quant_file: str) -> str:
    """Derive a quant-format label (e.g. ``Q4_K_M``) from a quant/model file path.

    Scans the file's base name for any known cited quant-format token (longest
    first so ``Q4_K_M`` wins over ``Q4_0``); falls back to the base name with a
    trailing ``.gguf`` stripped when no known token is present.
    """
    from .modules.quality import CITED_QUALITY_TABLE

    import os

    base = os.path.basename(str(quant_file))
    stem = base[:-len(".gguf")] if base.lower().endswith(".gguf") else base
    upper = stem.upper()
    for token in sorted(CITED_QUALITY_TABLE, key=len, reverse=True):
        if token.upper() in upper:
            return token
    return stem or str(quant_file)


def _sanitize_segment(name: str) -> str:
    """Reduce a string to a filesystem-safe single path segment.

    Keeps alphanumerics, dashes, dots, equals, and underscores; replaces every
    other character (including path separators) with ``_`` so a point id / module
    name cannot introduce extra path components or escape the campaign directory.
    """
    cleaned = "".join(
        ch if (ch.isalnum() or ch in "-._=") else "_" for ch in str(name).strip()
    )
    cleaned = cleaned.strip("._") or "x"
    return cleaned


def _as_float(value: Any, default: float) -> float:
    """Coerce ``value`` to float, returning ``default`` when it cannot be coerced."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as pretty-printed JSON to ``path`` (creating parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")


__all__ = ["run", "CampaignRunResult", "MeasurementContext"]
