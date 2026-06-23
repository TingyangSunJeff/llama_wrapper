"""Safety-critical isolation CI guard (task 18.5).

This is the single most important safety check in the suite. The hard invariant
(R8.1, R8.5) is: a campaign run **must never create, modify, or delete any file
outside the single campaign-scoped subdirectory** — and in particular never any
path outside ``profile/`` at all.

Unlike the property test for the isolation invariant (Property 17, task 18.4)
which inspects intercepted write paths, this guard runs a **full fake-harness
campaign** end-to-end through :func:`profile_suite.orchestrator.run` with its
output redirected to a temp ``runs_root`` and then asserts against the **real
filesystem**:

  (a) the campaign output directory is created ONLY under the temp ``runs_root``
      (and every artifact / persisted file the run reports lives under it);
  (b) the real ``profile/runs/`` directory gained no new entries — it still
      contains only ``.gitkeep`` and is byte-for-byte unchanged; and
  (c) nothing under the repository outside ``profile/`` changed — a representative
      set of sentinel trees (the repo-root listing, ``experiments/``, and the
      spec directory) is snapshotted before/after and must be identical.

No real ``llama-server`` is booted: a fake Platform (no GPU) and a fake
measurement module (returns successful runs, writes only to the per-run log path
the harness hands it) stand in for the I/O surface, exactly as in
``test_orchestrator_wiring.py``.

_Requirements: 8.1, 8.5_
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from profile_suite.config import Config
from profile_suite.modules.base import PointSpec, make_point_id
from profile_suite.orchestrator import run
from profile_suite.platform.a100 import A100_DESCRIPTOR
from profile_suite.results import RunRepeatResult

# --------------------------------------------------------------------------- #
# Real-filesystem anchors (the things that must NOT change)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path("/scratch2/tingyang/llama.cpp")
PROFILE_ROOT = REPO_ROOT / "profile"
PROFILE_RUNS = PROFILE_ROOT / "runs"
SPEC_DIR = REPO_ROOT / ".kiro" / "specs" / "profile-suite"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


# --------------------------------------------------------------------------- #
# Fakes (mirrors test_orchestrator_wiring.py): no GPU, no real server boot.
# --------------------------------------------------------------------------- #
class FakePlatform:
    """A Platform adapter that needs no GPU and marks every Config feasible.

    Keeping every Config feasible drives the *full* campaign (every grid point is
    measured), which maximizes the number of files the run writes — the strongest
    exercise of the isolation guard.
    """

    descriptor = A100_DESCRIPTOR

    def resolve_binary(self, kind: str) -> str:
        return f"/fake/llama-{kind}"

    def device_total_mib(self, gpu_index: int) -> float:
        return 40960.0

    def is_feasible(self, config: Config):
        return (True, None)


class FakeModule:
    """A profiler module that returns successful runs without booting a server.

    It only ever writes to the per-run log path the harness hands it
    (``harness.server_log_path(...)``), which by construction lives under the
    campaign-scoped ``raw/`` directory — so a correct suite confines every write
    to ``runs_root``.
    """

    name = "Performance_Profiler"

    def points(self, cfg, grid):
        specs = []
        for config in grid:
            axis = {"path": "server"}
            specs.append(
                PointSpec(
                    module=self.name,
                    config=config,
                    axis=axis,
                    point_id=make_point_id(self.name, config, axis),
                )
            )
        return specs

    def measure_once(self, spec, harness, platform) -> RunRepeatResult:
        idx = harness.run_index
        log_path = harness.server_log_path(spec.point_id, idx)
        Path(log_path).write_text(f"fake server log run {idx}\n")
        return RunRepeatResult(
            run_index=idx,
            discarded_warmup=False,
            ok=True,
            raw_log_path=log_path,
            metrics={"decode_throughput": 100.0 + idx},
            error=None,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_campaign(sandbox: Path) -> Path:
    """Materialize a valid campaign whose inputs all live under ``sandbox``.

    The server binary, bench binaries, model dir, and campaign YAML are all
    created inside the temp sandbox (never under ``profile/``), so the only thing
    that could touch the real tree is the suite itself.
    """
    model_dir = sandbox / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    quant_files = ["Meta-Q4_K_M.gguf", "Meta-Q8_0.gguf"]
    for name in quant_files:
        (model_dir / name).write_text("gguf")

    server_binary = sandbox / "llama-server"
    server_binary.write_text("#!/bin/sh\n")

    spec = {
        "platform": A100_DESCRIPTOR,
        "server_binary": str(server_binary),
        "bench_binary": str(sandbox / "llama-bench"),
        "batched_bench_binary": str(sandbox / "llama-batched-bench"),
        "model_dir": str(model_dir),
        "gpu_index": 0,
        "config_grid": {
            "quant_files": quant_files,
            "ctx_lengths": [4096, 32768],
            "slot_counts": [1, 4],
        },
        "run_repeats": 5,
        "prompt_tokens": 8,
        "output_tokens": 4,
        # A full campaign: measurement + the static quality axis + reporting.
        "enabled_modules": [
            "Performance_Profiler",
            "Quality_Module",
            "Reporting_Module",
        ],
    }
    path = sandbox / "campaign.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    """Recursively snapshot ``root``: maps file path -> (mtime_ns, size_bytes).

    Python bytecode caches (``__pycache__`` / ``*.pyc``) are skipped: importing
    the suite legitimately (re)writes them, and they always live *inside*
    ``profile/`` so they are irrelevant to the outside-``profile/`` guard and
    would only introduce flakiness.
    """
    snap: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return snap
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".pyc"):
                continue
            p = Path(dirpath) / fn
            try:
                stat = p.stat()
            except OSError:
                continue
            snap[str(p)] = (stat.st_mtime_ns, stat.st_size)
    return snap


def _is_under(child: Path, parent: Path) -> bool:
    """True iff ``child`` is ``parent`` itself or located beneath it (resolved)."""
    child_r = child.resolve()
    parent_r = parent.resolve()
    return child_r == parent_r or parent_r in child_r.parents


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #
def test_full_fake_campaign_writes_only_under_temp_runs_root(tmp_path):
    """A full fake campaign must write ONLY under the temp ``runs_root``.

    Strictly asserts the isolation invariant against the real filesystem:
    nothing outside the temp sandbox — and in particular nothing under
    ``profile/runs/`` or anywhere in the repo outside ``profile/`` — is created,
    modified, or deleted by the campaign run (R8.1, R8.5).
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    runs_root = sandbox / "runs"
    runs_root.mkdir()

    campaign = _write_campaign(sandbox)

    # --- Pre-run snapshots of everything that must stay untouched ---------- #
    profile_runs_entries_before = sorted(p.name for p in PROFILE_RUNS.iterdir())
    profile_runs_snap_before = _snapshot_tree(PROFILE_RUNS)
    repo_root_children_before = sorted(p.name for p in REPO_ROOT.iterdir())
    experiments_snap_before = _snapshot_tree(EXPERIMENTS_DIR)
    spec_snap_before = _snapshot_tree(SPEC_DIR)

    # --- Run the full fake campaign, redirected to the temp runs_root ------ #
    result = run(
        str(campaign),
        runs_root=str(runs_root),
        platform=FakePlatform(),
        module_overrides={"Performance_Profiler": FakeModule()},
    )

    # =================================================================== #
    # (a) Output is created ONLY under the temp runs_root.
    # =================================================================== #
    assert result.ok is True, f"campaign did not complete: {result.reason}"
    assert result.campaign_dir is not None
    campaign_dir = Path(result.campaign_dir)
    assert campaign_dir.is_dir()
    assert _is_under(campaign_dir, runs_root), (
        f"campaign dir {campaign_dir} escaped the temp runs_root {runs_root}"
    )
    # The campaign dir must NOT be under the real profile/runs.
    assert not _is_under(campaign_dir, PROFILE_RUNS), (
        f"campaign dir {campaign_dir} leaked into the real profile/runs"
    )

    # Every path the run reports (persisted files + per-Platform artifacts)
    # lives under the campaign dir, hence under the temp runs_root.
    reported_paths = [
        campaign_dir / "manifest.json",
        campaign_dir / "environment.json",
        campaign_dir / "points.json",
        campaign_dir / "quality.json",
    ]
    for platform_artifacts in result.artifacts.values():
        for art_path in platform_artifacts.values():
            if art_path is not None:
                reported_paths.append(Path(art_path))
    for p in reported_paths:
        assert _is_under(p, runs_root), f"reported path {p} escaped runs_root"

    # The run actually produced output (guard is meaningful, not vacuous).
    produced = list(runs_root.rglob("*"))
    produced_files = [p for p in produced if p.is_file()]
    assert produced_files, "campaign produced no files under runs_root"
    # And EVERY file physically present under runs_root is, trivially, under it —
    # the meaningful counterpart is that nothing leaked elsewhere (checked below).
    for p in produced_files:
        assert _is_under(p, runs_root)

    # =================================================================== #
    # (b) The real profile/runs/ gained no new entries and is unchanged.
    # =================================================================== #
    profile_runs_entries_after = sorted(p.name for p in PROFILE_RUNS.iterdir())
    assert profile_runs_entries_after == profile_runs_entries_before == [".gitkeep"], (
        "profile/runs/ changed: before="
        f"{profile_runs_entries_before} after={profile_runs_entries_after}"
    )
    profile_runs_snap_after = _snapshot_tree(PROFILE_RUNS)
    assert profile_runs_snap_after == profile_runs_snap_before, (
        "a file under profile/runs/ was created, modified, or deleted"
    )

    # =================================================================== #
    # (c) Nothing under the repo OUTSIDE profile/ changed.
    # =================================================================== #
    repo_root_children_after = sorted(p.name for p in REPO_ROOT.iterdir())
    assert repo_root_children_after == repo_root_children_before, (
        "the repository root gained or lost a top-level entry: "
        f"added={set(repo_root_children_after) - set(repo_root_children_before)} "
        f"removed={set(repo_root_children_before) - set(repo_root_children_after)}"
    )

    experiments_snap_after = _snapshot_tree(EXPERIMENTS_DIR)
    assert experiments_snap_after == experiments_snap_before, (
        "a file under experiments/ (outside profile/) was created, modified, or deleted"
    )

    spec_snap_after = _snapshot_tree(SPEC_DIR)
    assert spec_snap_after == spec_snap_before, (
        "a file under the spec directory (outside profile/) was created, "
        "modified, or deleted"
    )

    # Sanity: the persisted points.json is readable and reflects the full grid
    # (2 quant x 2 ctx x 2 slots = 8 points), confirming a real, full campaign ran.
    points = json.loads((campaign_dir / "points.json").read_text())
    assert len(points) == 8
