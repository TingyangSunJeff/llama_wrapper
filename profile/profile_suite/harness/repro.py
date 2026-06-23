"""Reproducibility_Harness: the single owner of the measurement run loop.

This module implements the run-loop stopping rule that every measurement point in
the suite flows through. :meth:`ReproHarness.run_point` is the single enforcement
site for three reproducibility guarantees (design "Reproducibility_Harness run
loop"):

- **Discard exactly one warmup** (R5.4): the first call to ``measure_fn`` is a
  warmup run that is recorded (with ``discarded_warmup=True``) but excluded from all
  reported statistics.
- **Retry to the stopping rule** (R5.9): after the warmup, retained Run_Repeats are
  executed until either ``min_success`` (default 5) successful repeats are retained
  or ``max_attempts`` (default 10) total retained attempts are reached, whichever
  comes first. Each failed retained repeat is recorded with its error reason and
  retried.
- **Aggregate over successful repeats only** (R2.3, R2.7): per-metric
  :class:`~profile_suite.results.Aggregate`\\ s are computed only from the successful
  retained repeats, reusing :func:`profile_suite.harness.stats.aggregate`.

The resulting :class:`~profile_suite.results.MeasurementPoint` status transitions
are (Property 11):

- ``"complete"``   - at least ``min_success`` successful retained repeats.
- ``"incomplete"`` - at least one but fewer than ``min_success`` successful repeats
  (the point still carries the aggregates of the repeats that did succeed, flagged
  ``insufficient``).
- ``"failed"``     - zero successful retained repeats (no aggregates; the campaign
  continues). This is the design's distinguished sub-case of "fewer than
  ``min_success`` successes" (error-handling table: "All repeats fail").

``begin_campaign`` / ``finalize_campaign`` own the campaign-level manifest and
environment wiring (R5.1-R5.6, R6.5): ``begin_campaign`` resolves the binary, pins
and records the GPU device, captures the environment plus per-model checksums,
records the Pinned_Build, and writes ``manifest.json`` / ``environment.json``;
``finalize_campaign`` attaches the retained raw-log paths and re-persists the
manifest. Every path they touch lives under a single campaign-scoped
``profile/runs/<campaign-name>_<UTC-stamp>/`` directory so the suite never writes
outside ``profile/`` (R8.1, R8.5).
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..config import Config, expand_grid
from ..results import (
    UNAVAILABLE,
    Aggregate,
    EnvironmentCapture,
    MeasurementPoint,
    RunManifest,
    RunRepeatResult,
)
from . import stats
from .sysprobe import SysProbe

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..campaign import CampaignConfig
    from ..platform.base import Platform

# A measurement callable: given a run index, perform exactly one run and return its
# raw :class:`RunRepeatResult`. Index 0 is the discarded warmup; 1..N are retained.
MeasureFn = Callable[[int], RunRepeatResult]

# The repository root and the canonical ``profile/runs`` directory, derived from
# this file's location: repro.py lives at
# ``<repo>/profile/profile_suite/harness/repro.py`` so ``parents[2]`` is
# ``<repo>/profile`` and ``parents[3]`` is the repository root.
_PROFILE_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_RUNS_ROOT = _PROFILE_DIR / "runs"
_REPO_ROOT = Path(__file__).resolve().parents[3]


class ReproHarness:
    """Owner of the per-measurement-point run loop and the campaign manifest.

    Implements both the per-point stopping rule (:meth:`run_point`) and the
    campaign-level wiring (:meth:`begin_campaign` / :meth:`finalize_campaign`).
    """

    def __init__(
        self,
        runs_root: Optional[str] = None,
        *,
        sysprobe: Optional[SysProbe] = None,
    ) -> None:
        # Base directory under which every campaign-scoped subdirectory is created.
        # Defaults to the repo's ``profile/runs`` so all writes stay under
        # ``profile/`` (R8.1, R8.5). ``run_point`` itself is pure with respect to
        # the filesystem (the supplied ``measure_fn`` owns any per-run I/O).
        self.runs_root = Path(runs_root) if runs_root is not None else _DEFAULT_RUNS_ROOT
        self._sysprobe = sysprobe if sysprobe is not None else SysProbe()
        # Maps a campaign name to its resolved campaign-scoped directory so
        # ``finalize_campaign`` can locate the directory created in
        # ``begin_campaign`` without the manifest needing to carry a path.
        self._campaign_dirs: dict[str, Path] = {}
        # Resolved server-binary path per campaign, preserved across the manifest
        # rewrite in ``finalize_campaign`` for traceability.
        self._campaign_server_binaries: dict[str, str] = {}

    def run_point(
        self,
        measure_fn: MeasureFn,
        min_success: int = 5,
        max_attempts: int = 10,
        *,
        point_id: str = "",
        module: str = "",
        config: Optional[Config] = None,
        axis: Optional[dict[str, Any]] = None,
    ) -> MeasurementPoint:
        """Run one measurement point under the warmup-discard + retry stopping rule.

        Executes exactly one discarded warmup (``measure_fn(0)``), then retained
        repeats (``measure_fn(1)``, ``measure_fn(2)``, ...) until ``min_success``
        successes are retained or ``max_attempts`` total retained attempts are made.
        Aggregates are computed over the successful retained repeats only.

        Args:
            measure_fn: Callable taking a run index and returning that run's
                :class:`RunRepeatResult`. Index 0 is the warmup; 1.. are retained.
            min_success: Successful retained repeats required for a ``"complete"``
                point (default 5, per R5.4).
            max_attempts: Maximum number of *retained* attempts before stopping,
                regardless of how many succeeded (default 10, per R5.9). The warmup
                run is not counted against this budget.
            point_id: Deterministic identifier for the point (module+config+axis).
            module: Name of the producing module.
            config: The source :class:`Config` for the point.
            axis: The measurement axis (e.g. ``{"decode_batch": 8}``).

        Returns:
            A :class:`MeasurementPoint` holding every run (warmup + retained), the
            per-metric aggregates over successful repeats, and a status of
            ``"complete"`` / ``"incomplete"`` / ``"failed"``.
        """
        repeats: list[RunRepeatResult] = []

        # --- Exactly one discarded warmup (R5.4) ---------------------------- #
        warmup = measure_fn(0)
        # Enforce the warmup labeling regardless of what ``measure_fn`` returned,
        # so the warmup is unambiguously excluded from the statistics below.
        warmup.discarded_warmup = True
        repeats.append(warmup)

        # --- Retained repeats with the retry stopping rule (R5.9) ----------- #
        successes = 0
        attempts = 0
        run_index = 1
        while successes < min_success and attempts < max_attempts:
            result = measure_fn(run_index)
            # Retained repeats are never warmups.
            result.discarded_warmup = False
            repeats.append(result)
            attempts += 1
            run_index += 1
            if result.ok:
                successes += 1

        # --- Aggregate over the successful retained repeats only ------------ #
        successful = [
            r for r in repeats if r.ok and not r.discarded_warmup
        ]
        aggregates = _aggregate_successful(successful)

        # --- Status transitions (Property 11 / design error-handling table) - #
        status, reason = _classify(successes, min_success, attempts, repeats)

        return MeasurementPoint(
            point_id=point_id,
            module=module,
            config=config,  # type: ignore[arg-type]  # populated by the orchestrator
            axis=dict(axis or {}),
            repeats=repeats,
            aggregates=aggregates,
            status=status,
            reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Campaign-level methods
    # ------------------------------------------------------------------ #
    def begin_campaign(
        self,
        cfg: "CampaignConfig",
        platform: "Optional[Platform]" = None,
        *,
        campaign_name: Optional[str] = None,
    ) -> RunManifest:
        """Open a campaign: pin the GPU, capture the environment, write the manifest.

        Performs the campaign-start reproducibility wiring (R5.1-R5.3, R5.5, R5.6,
        R6.5):

        1. Create the single campaign-scoped directory
           ``runs_root/<campaign-name>_<UTC-stamp>/`` (plus its ``raw/`` and
           ``artifacts/`` subdirectories). Every later write for the campaign
           lives under this directory, so nothing is written outside ``profile/``.
        2. Resolve the ``server`` binary via the Platform adapter (R6.3/R6.4).
        3. Pin the campaign GPU (``cfg.gpu_index``) and record the pinned device
           identifier, preferring the device UUID when the probe can read it and
           falling back to the index (R5.3).
        4. Capture the environment (OS / GPU / driver / CUDA / Python) and the
           per-model sha256 checksums for the resolved Config_Grid model files,
           recording the explicit ``"unavailable"`` sentinel for any field that
           cannot be determined (R5.2, R5.8).
        5. Record the Pinned_Build via ``git describe --tags --always`` (falling
           back to the commit hash, then to ``"unavailable"``) (R5.1).
        6. Write ``manifest.json`` and ``environment.json`` under the campaign
           directory (R5.6).

        Args:
            cfg: The campaign definition. Optional settings carrying the
                :data:`~profile_suite.campaign.MISSING` sentinel are resolved to
                their defaults before use.
            platform: The Platform adapter to use for binary resolution / device
                metadata. When omitted, an adapter is constructed from
                ``cfg.platform`` and the campaign's binary paths.
            campaign_name: Optional explicit campaign name used in the directory
                name and the manifest. Defaults to the Platform descriptor.

        Returns:
            The :class:`~profile_suite.results.RunManifest` for the campaign (also
            persisted to ``manifest.json``).
        """
        cfg = self._resolved_config(cfg)
        if platform is None:
            platform = self._build_platform(cfg)

        # Resolve the server binary so the manifest/launch path is pinned and a
        # mis-resolution surfaces at campaign start rather than mid-run (R6.3).
        server_binary = platform.resolve_binary("server")

        name = campaign_name or getattr(cfg, "campaign_name", None) or platform.descriptor
        campaign_dir = self._make_campaign_dir(name)
        self._campaign_dirs[name] = campaign_dir
        self._campaign_server_binaries[name] = server_binary

        # --- Pin + record the GPU device (R5.3) ----------------------------- #
        pinned_device = self._pin_device(cfg.gpu_index)

        # --- Environment capture + per-model checksums (R5.2, R5.8) --------- #
        environment = self._sysprobe.capture_environment()
        grid = expand_grid(cfg.config_grid)
        environment.models = self._model_checksums(grid, cfg.model_dir)

        # --- Pinned_Build (R5.1) ------------------------------------------- #
        pinned_build = self._pinned_build()

        manifest = RunManifest(
            campaign_name=name,
            pinned_build=pinned_build,
            platform=platform.descriptor,
            pinned_device=pinned_device,
            environment=environment,
            config_grid=grid,
            run_repeats=int(cfg.run_repeats),  # type: ignore[arg-type]
            raw_log_paths={},
            decode_batch_sizes=[int(b) for b in cfg.decode_batch_sizes],  # type: ignore[union-attr]
            enabled_modules=list(cfg.enabled_modules),  # type: ignore[arg-type]
            noise_sensitive_metrics=[],
        )

        # Record the resolved server binary alongside the manifest for traceability
        # without altering the RunManifest schema.
        self._write_json(campaign_dir / "environment.json", environment.to_dict())
        self._write_manifest(campaign_dir, manifest, server_binary=server_binary)
        return manifest

    def finalize_campaign(self, manifest: RunManifest) -> None:
        """Close a campaign: attach the retained raw-log paths and re-persist it.

        Scans the campaign directory's ``raw/`` tree for the retained per-run logs
        written during the campaign, groups them by their top-level module
        subdirectory into ``manifest.raw_log_paths`` (paths relative to the
        campaign directory, R5.5), and rewrites ``manifest.json`` so the persisted
        manifest reflects the final raw-log inventory (R5.6).

        Args:
            manifest: The manifest returned by :meth:`begin_campaign`.
        """
        campaign_dir = self._campaign_dir_for(manifest)
        raw_root = campaign_dir / "raw"

        discovered: dict[str, list[str]] = {}
        if raw_root.is_dir():
            for path in sorted(raw_root.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(raw_root)
                # The first path component under ``raw/`` is the module bucket
                # (e.g. ``switch_cost``); fall back to ``"raw"`` for stray files.
                module = rel.parts[0] if len(rel.parts) > 1 else "raw"
                rel_to_campaign = path.relative_to(campaign_dir).as_posix()
                discovered.setdefault(module, []).append(rel_to_campaign)

        # Merge discovered logs with anything the modules already recorded on the
        # manifest, preserving order and dropping duplicates.
        merged: dict[str, list[str]] = {
            k: list(v) for k, v in manifest.raw_log_paths.items()
        }
        for module, paths in discovered.items():
            existing = merged.setdefault(module, [])
            for p in paths:
                if p not in existing:
                    existing.append(p)
        manifest.raw_log_paths = merged

        self._write_manifest(
            campaign_dir,
            manifest,
            server_binary=self._campaign_server_binaries.get(manifest.campaign_name),
        )

    # ------------------------------------------------------------------ #
    # Campaign helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolved_config(cfg: "CampaignConfig") -> "CampaignConfig":
        """Return a fully-populated config (filling any omitted optional defaults)."""
        from ..campaign import MISSING, apply_defaults

        # Resolve only if an optional field still carries the sentinel, so a
        # caller-completed config is used verbatim.
        needs_defaults = any(
            getattr(cfg, name) is MISSING
            for name in (
                "decode_batch_sizes",
                "run_repeats",
                "enabled_modules",
                "boot_timeout_s",
                "warmup_timeout_s",
                "max_attempts",
                "local_quality",
                "memory_budget_mib",
            )
        )
        if not needs_defaults:
            return cfg
        resolved, _applied = apply_defaults(cfg)
        return resolved

    @staticmethod
    def _build_platform(cfg: "CampaignConfig") -> "Platform":
        """Construct a Platform adapter from the campaign's platform descriptor."""
        from ..platform.a100 import A100_DESCRIPTOR, A100CudaPlatform
        from ..platform.jetson import JETSON_DESCRIPTOR, JetsonPlatform

        budget = cfg.memory_budget_mib
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

    def _make_campaign_dir(self, campaign_name: str) -> Path:
        """Create and return the campaign-scoped directory under ``runs_root``.

        The directory name is ``<campaign-name>_<UTC-stamp>``; the ``raw/`` and
        ``artifacts/`` subdirectories are created eagerly so modules and the
        reporting step have a home. The resolved directory is asserted to live
        under ``runs_root`` so a malformed campaign name cannot escape ``profile/``
        (R8.1, R8.5).
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = _sanitize_name(campaign_name)
        runs_root = self.runs_root.resolve()
        campaign_dir = (runs_root / f"{safe}_{stamp}").resolve()
        if runs_root not in campaign_dir.parents:
            raise ValueError(
                f"refusing to create campaign dir outside runs_root: {campaign_dir}"
            )
        (campaign_dir / "raw").mkdir(parents=True, exist_ok=True)
        (campaign_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        return campaign_dir

    def _pin_device(self, gpu_index: int) -> str:
        """Pin the campaign GPU and return its recorded identifier (R5.3).

        Prefers the device UUID (a stable identity that survives index
        renumbering); falls back to ``index:<n>`` when the probe cannot read a
        UUID. The pin itself is recorded rather than imposed via a global
        environment mutation, so the harness has no side effects outside the
        campaign directory; the launching code applies the device index.
        """
        uuid = self._sysprobe._query_gpu_str("uuid", gpu_index)
        if uuid and uuid != UNAVAILABLE:
            return f"{uuid} (index:{gpu_index})"
        return f"index:{gpu_index}"

    def _model_checksums(self, grid: list[Config], model_dir: str) -> dict[str, Any]:
        """Build the per-model {resolved-path -> ModelRef} checksum map (R5.2).

        Resolves each distinct quant/model file in the Config_Grid against
        ``model_dir`` (absolute paths are used as-is) and computes its sha256 via
        :class:`SysProbe` (which records ``"unavailable"`` when a file cannot be
        read, R5.8). Each distinct file is checksummed once.
        """
        models: dict[str, Any] = {}
        for cfg in grid:
            quant = cfg.quant_file
            path = quant if os.path.isabs(quant) else os.path.join(model_dir, quant)
            abs_path = os.path.abspath(path)
            if abs_path in models:
                continue
            models[abs_path] = self._sysprobe.model_ref(path)
        return models

    @staticmethod
    def _pinned_build() -> str:
        """Return the Pinned_Build via ``git describe``, with graceful fallbacks.

        Tries ``git describe --tags --always`` first (the build tag, e.g. a
        ``bNNNN`` release), then ``git rev-parse HEAD`` for the bare commit, then
        the explicit :data:`UNAVAILABLE` sentinel so the campaign can proceed even
        outside a git checkout (R5.1).
        """
        for args in (
            ["git", "-C", str(_REPO_ROOT), "describe", "--tags", "--always"],
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        ):
            try:
                out = subprocess.check_output(
                    args, text=True, stderr=subprocess.DEVNULL, timeout=10.0
                ).strip()
            except Exception:  # noqa: BLE001 - missing git / not a repo / timeout
                continue
            if out:
                return out
        return UNAVAILABLE

    def _campaign_dir_for(self, manifest: RunManifest) -> Path:
        """Return the campaign directory created for ``manifest`` in this harness."""
        try:
            return self._campaign_dirs[manifest.campaign_name]
        except KeyError as exc:  # pragma: no cover - misuse guard
            raise ValueError(
                "finalize_campaign called for a campaign this harness did not "
                f"begin: {manifest.campaign_name!r}"
            ) from exc

    def _write_manifest(
        self,
        campaign_dir: Path,
        manifest: RunManifest,
        *,
        server_binary: Optional[str] = None,
    ) -> None:
        """Persist the manifest as ``manifest.json`` under the campaign directory."""
        payload = manifest.to_dict()
        if server_binary is not None:
            payload["server_binary"] = server_binary
        self._write_json(campaign_dir / "manifest.json", payload)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        """Write ``payload`` as pretty-printed JSON to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")


def _sanitize_name(name: str) -> str:
    """Reduce a campaign name to a filesystem-safe single path segment.

    Keeps alphanumerics, dashes, dots, and underscores; replaces every other
    character (including path separators and whitespace) with ``_`` so the name
    cannot introduce extra path components or escape the campaign directory. An
    empty or all-stripped name falls back to ``"campaign"``.
    """
    cleaned = "".join(
        ch if (ch.isalnum() or ch in "-._") else "_" for ch in name.strip()
    )
    cleaned = cleaned.strip("._")
    return cleaned or "campaign"


def _aggregate_successful(
    successful: list[RunRepeatResult],
) -> dict[str, Aggregate]:
    """Build a per-metric :class:`Aggregate` over the successful repeats only.

    The metric key set is the union of metric names present across the successful
    repeats. For each metric, the per-repeat values from the successful repeats that
    reported it are aggregated via :func:`stats.aggregate` (mean/std over successes,
    with the ``insufficient`` flag), then mapped onto the report-facing
    :class:`profile_suite.results.Aggregate`.
    """
    metric_keys: list[str] = []
    seen: set[str] = set()
    for r in successful:
        for key in r.metrics:
            if key not in seen:
                seen.add(key)
                metric_keys.append(key)

    aggregates: dict[str, Aggregate] = {}
    for key in metric_keys:
        values = [r.metrics[key] for r in successful if key in r.metrics]
        agg = stats.aggregate(values)
        aggregates[key] = Aggregate(
            mean=agg.mean,
            std=agg.std,
            n_success=agg.n_success,
            insufficient=agg.insufficient,
        )
    return aggregates


def _classify(
    successes: int,
    min_success: int,
    attempts: int,
    repeats: list[RunRepeatResult],
) -> tuple[str, Optional[str]]:
    """Map the success count onto a point status and an optional reason string.

    - ``successes >= min_success`` -> ``"complete"`` (no reason).
    - ``0 < successes < min_success`` -> ``"incomplete"`` (insufficient repeats).
    - ``successes == 0`` -> ``"failed"`` (all retained attempts failed).
    """
    if successes >= min_success:
        return "complete", None

    # Summarize the retained-run error reasons for the point's reason field.
    errors = [
        r.error
        for r in repeats
        if (not r.discarded_warmup) and (not r.ok) and r.error
    ]
    error_summary = "; ".join(errors)

    if successes == 0:
        reason = (
            f"all {attempts} retained attempts failed"
            + (f": {error_summary}" if error_summary else "")
        )
        return "failed", reason

    reason = (
        f"only {successes} of {min_success} successful repeats "
        f"after {attempts} attempts"
        + (f": {error_summary}" if error_summary else "")
    )
    return "incomplete", reason


__all__ = ["ReproHarness", "MeasureFn"]
