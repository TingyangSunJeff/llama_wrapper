"""Property test for the filesystem isolation invariant (task 18.4, Property 17).

Property 17 (design.md "Correctness Properties"):

    *For any* campaign execution, every file the suite creates, modifies, or
    deletes has a path located under the single campaign-scoped subdirectory
    inside ``profile/``, and no write path escapes ``profile/``.

    **Validates: Requirements 8.1, 8.5**

Approach
--------
We drive a **fake-harness campaign** end-to-end through
:func:`profile_suite.orchestrator.run` (the same entry point exercised by the
orchestrator wiring tests, task 18.2) with:

- a **fake Platform** (no GPU, every Config feasible), and
- **fake measurement modules** that return successful runs without booting any
  real ``llama-server`` (they only write their per-run log under the harness-
  supplied, campaign-scoped path).

The campaign's ``runs_root`` is redirected to a throwaway temp directory that
stands in for the ``profile/`` root, so the campaign-scoped output directory the
suite creates lives under it.

A **RecordingFilesystem** wraps every filesystem write the suite performs during
``run`` — it patches :func:`builtins.open` / :func:`io.open` (recording every
path opened in a write/append/create mode) and :func:`os.mkdir` /
:func:`os.makedirs` (recording every directory created). After the campaign, we
assert that **every** recorded write path is located under the single
campaign-scoped directory (which itself lives under ``runs_root``), and that none
escapes it.

Hypothesis varies the campaign shape (grid sizes via the quant/ctx/slot axes, and
the set of enabled modules) so the invariant is checked across many campaign
executions.
"""

from __future__ import annotations

# --- Force all import-time writes to happen at collection, not during the ---- #
# --- RecordingFilesystem window. Matplotlib (pulled in by the Reporting    ---- #
# --- module) builds caches on first import; pinning MPLCONFIGDIR + importing -- #
# --- it here keeps any such write out of the recording window.               -- #
import os as _os
import tempfile as _tempfile

_os.environ.setdefault(
    "MPLCONFIGDIR", _tempfile.mkdtemp(prefix="profilesuite_mpl_")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot  # noqa: E402,F401

# Pre-import every profile_suite submodule the orchestrator touches lazily, so
# their .pyc compilation (a filesystem write) happens at collection time rather
# than inside the recording window.
import profile_suite.campaign  # noqa: E402,F401
import profile_suite.config  # noqa: E402,F401
import profile_suite.harness.client  # noqa: E402,F401
import profile_suite.harness.repro  # noqa: E402,F401
import profile_suite.harness.sysprobe  # noqa: E402,F401
import profile_suite.loader  # noqa: E402,F401
import profile_suite.modules.memory  # noqa: E402,F401
import profile_suite.modules.performance  # noqa: E402,F401
import profile_suite.modules.quality  # noqa: E402,F401
import profile_suite.modules.switch_cost  # noqa: E402,F401
import profile_suite.platform.a100  # noqa: E402,F401
import profile_suite.platform.jetson  # noqa: E402,F401
import profile_suite.reporting.report  # noqa: E402,F401
import profile_suite.results  # noqa: E402,F401
import profile_suite.validation  # noqa: E402,F401

import builtins  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from profile_suite.config import Config  # noqa: E402
from profile_suite.modules.base import PointSpec, make_point_id  # noqa: E402
from profile_suite.orchestrator import run  # noqa: E402
from profile_suite.platform.a100 import A100_DESCRIPTOR  # noqa: E402
from profile_suite.results import RunRepeatResult  # noqa: E402


# --------------------------------------------------------------------------- #
# RecordingFilesystem: capture every write path the suite performs
# --------------------------------------------------------------------------- #
class RecordingFilesystem:
    """A context manager that records every filesystem *write* path.

    While active it patches the two write surfaces the suite uses:

    - :func:`builtins.open` and :func:`io.open`: any call whose ``mode`` requests
      writing/appending/creating (contains ``w``, ``a``, ``x`` or ``+``) records
      the absolute target path before delegating to the real ``open``.
    - :func:`os.mkdir` and :func:`os.makedirs`: every created directory path is
      recorded before delegating (this also captures ``pathlib.Path.mkdir``, which
      calls ``os.mkdir`` under the hood).

    File-descriptor opens (``open(3, ...)``) carry no path and are ignored. The
    recorded paths are available as :attr:`write_paths` after the block exits.
    """

    _WRITE_MODE_CHARS = ("w", "a", "x", "+")

    def __init__(self) -> None:
        self.write_paths: list[str] = []

    def _record(self, target: object) -> None:
        try:
            path = os.fspath(target)  # type: ignore[arg-type]
        except TypeError:
            return  # an integer file descriptor (or other non-path); no path.
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        self.write_paths.append(os.path.abspath(path))

    def __enter__(self) -> "RecordingFilesystem":
        self._orig_builtins_open = builtins.open
        self._orig_io_open = io.open
        self._orig_mkdir = os.mkdir
        self._orig_makedirs = os.makedirs
        rec = self

        def patched_open(file, mode="r", *args, **kwargs):  # noqa: ANN001
            if any(ch in mode for ch in rec._WRITE_MODE_CHARS):
                rec._record(file)
            return rec._orig_builtins_open(file, mode, *args, **kwargs)

        def patched_mkdir(path, *args, **kwargs):  # noqa: ANN001
            rec._record(path)
            return rec._orig_mkdir(path, *args, **kwargs)

        def patched_makedirs(name, *args, **kwargs):  # noqa: ANN001
            rec._record(name)
            return rec._orig_makedirs(name, *args, **kwargs)

        builtins.open = patched_open
        io.open = patched_open
        os.mkdir = patched_mkdir
        os.makedirs = patched_makedirs
        return self

    def __exit__(self, *exc_info) -> bool:  # noqa: ANN002
        builtins.open = self._orig_builtins_open
        io.open = self._orig_io_open
        os.mkdir = self._orig_mkdir
        os.makedirs = self._orig_makedirs
        return False


# --------------------------------------------------------------------------- #
# Fakes: a no-GPU Platform and server-free measurement modules
# --------------------------------------------------------------------------- #
class FakePlatform:
    """A Platform adapter that needs no GPU and marks every Config feasible."""

    descriptor = A100_DESCRIPTOR

    def resolve_binary(self, kind: str) -> str:
        return f"/fake/llama-{kind}"

    def device_total_mib(self, gpu_index: int) -> float:
        return 40960.0

    def is_feasible(self, config: Config):
        return (True, None)


class FakeModule:
    """A profiler module that succeeds without booting a server.

    It writes its per-run log to the harness-supplied, campaign-scoped path (so
    that write is part of what the isolation invariant must contain) and returns a
    successful :class:`RunRepeatResult` carrying a few representative metrics, so
    the Reporting_Module tables have data to render.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def points(self, cfg, grid):  # noqa: ANN001
        specs = []
        for config in grid:
            axis = {"path": "server"}  # no decode-batch axis -> no figure rendered
            specs.append(
                PointSpec(
                    module=self.name,
                    config=config,
                    axis=axis,
                    point_id=make_point_id(self.name, config, axis),
                )
            )
        return specs

    def measure_once(self, spec, harness, platform) -> RunRepeatResult:  # noqa: ANN001
        idx = harness.run_index
        log_path = harness.server_log_path(spec.point_id, idx)
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"fake {self.name} run {idx}\n")
        return RunRepeatResult(
            run_index=idx,
            discarded_warmup=False,
            ok=True,
            raw_log_path=log_path,
            metrics={
                "c_switch_ms": 10.0 + idx,
                "weights": 100.0,
                "kv_per_slot": 5.0,
                "scratch_overhead": 3.0,
                "decode_throughput": 50.0 + idx,
            },
            error=None,
        )


# The three point-producing profiler modules the orchestrator flows through the
# run loop. Every one is overridden with a FakeModule so no real server is booted.
_PROFILER_NAMES = ("Switch_Cost_Profiler", "Performance_Profiler", "Memory_Profiler")

# Small value pools so generated grids stay fast while still varying in size.
_CTX_POOL = [2048, 4096, 32768]
_SLOT_POOL = [1, 2, 4]


@st.composite
def campaign_params(draw):
    """Generate a varied (valid) campaign shape: grid sizes and enabled modules."""
    n_quants = draw(st.integers(min_value=1, max_value=2))
    ctx_lengths = draw(
        st.lists(st.sampled_from(_CTX_POOL), min_size=1, max_size=2, unique=True)
    )
    slot_counts = draw(
        st.lists(st.sampled_from(_SLOT_POOL), min_size=1, max_size=2, unique=True)
    )
    run_repeats = draw(st.integers(min_value=1, max_value=2))

    # At least one profiler module (so the run actually measures + writes logs),
    # plus optionally the static Quality_Module and the Reporting_Module.
    profilers = draw(
        st.lists(st.sampled_from(_PROFILER_NAMES), min_size=1, max_size=3, unique=True)
    )
    enabled = list(profilers)
    if draw(st.booleans()):
        enabled.append("Quality_Module")
    if draw(st.booleans()):
        enabled.append("Reporting_Module")

    return {
        "n_quants": n_quants,
        "ctx_lengths": ctx_lengths,
        "slot_counts": slot_counts,
        "run_repeats": run_repeats,
        "enabled_modules": enabled,
    }


def _write_campaign(sandbox: Path, params: dict) -> Path:
    """Materialize a valid campaign (server binary + model files + YAML) on disk."""
    model_dir = sandbox / "models"
    model_dir.mkdir()
    server_binary = sandbox / "llama-server"
    server_binary.write_text("#!/bin/sh\n")
    (sandbox / "llama-bench").write_text("x")
    (sandbox / "llama-batched-bench").write_text("x")

    quant_files = []
    for i in range(params["n_quants"]):
        name = f"model_{i}.gguf"
        (model_dir / name).write_text("gguf")
        quant_files.append(name)

    spec = {
        "platform": A100_DESCRIPTOR,
        "server_binary": str(server_binary),
        "bench_binary": str(sandbox / "llama-bench"),
        "batched_bench_binary": str(sandbox / "llama-batched-bench"),
        "model_dir": str(model_dir),
        "gpu_index": 0,
        "config_grid": {
            "quant_files": quant_files,
            "ctx_lengths": list(params["ctx_lengths"]),
            "slot_counts": list(params["slot_counts"]),
        },
        "run_repeats": params["run_repeats"],
        "prompt_tokens": 8,
        "output_tokens": 4,
        "enabled_modules": params["enabled_modules"],
    }
    campaign_path = sandbox / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(spec))
    return campaign_path


# Feature: profile-suite, Property 17: The isolation invariant holds for every filesystem write
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(params=campaign_params())
def test_isolation_invariant_every_write_is_under_campaign_dir(params):
    """Every filesystem write of a campaign run stays under its campaign dir (R8.1, R8.5)."""
    with _tempfile.TemporaryDirectory(prefix="profilesuite_iso_") as sandbox_str:
        sandbox = Path(sandbox_str)
        runs_root = sandbox / "runs"
        runs_root.mkdir()

        campaign_path = _write_campaign(sandbox, params)

        # Override every profiler module so no real server is booted.
        overrides = {name: FakeModule(name) for name in _PROFILER_NAMES}

        # Record every write the suite performs *during* the campaign run only.
        with RecordingFilesystem() as rec:
            result = run(
                str(campaign_path),
                runs_root=str(runs_root),
                platform=FakePlatform(),
                module_overrides=overrides,
            )

        # The campaign ran (a valid campaign always produces a campaign dir).
        assert result.ok is True
        assert result.campaign_dir is not None

        campaign_dir = Path(result.campaign_dir).resolve()
        runs_root_resolved = runs_root.resolve()

        # The single campaign-scoped subdir lives under the (stand-in profile/)
        # runs_root.
        assert runs_root_resolved in campaign_dir.parents

        # The suite actually wrote something (manifest/environment at minimum), so
        # the invariant below is non-vacuous.
        assert rec.write_paths

        # Every recorded write path is the campaign dir itself or located under
        # it; none escapes to runs_root, its parent, or anywhere else (R8.1/R8.5).
        camp = str(campaign_dir)
        prefix = camp + os.sep
        for raw in rec.write_paths:
            norm = os.path.realpath(raw)
            assert norm == camp or norm.startswith(prefix), (
                f"write escaped the campaign-scoped directory: {norm!r} is not "
                f"under {camp!r}"
            )
