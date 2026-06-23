"""Property test for platform-infeasible Config skipping (task 18.3).

Property 15: Platform-infeasible Configs are skipped without aborting the campaign
(Validates Requirements 6.6).

The test drives the real :func:`profile_suite.orchestrator.run` end-to-end over a
mixed Config_Grid using a *fake* Platform (no GPU) and a *fake* measurement module
(no real ``llama-server``). The fake Platform reports an arbitrary, randomly chosen
subset of the expanded grid as infeasible. The property asserted, over many
generated grids and infeasible subsets, is:

  * every platform-infeasible Config produces exactly one MeasurementPoint with
    status ``"platform-infeasible"`` carrying a reason and no repeats, and
  * every feasible Config is still measured (status ``"complete"``),

i.e. infeasible Configs are skipped without aborting the campaign.

The FakePlatform / FakeModule / campaign-YAML patterns mirror
``test_orchestrator_wiring.py`` (task 18.2). Each example uses its own temporary
directory (the run boots no server but does exercise the full run loop), so the
test stays self-contained and replay-safe under Hypothesis.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.config import Config, GridSpec, expand_grid
from profile_suite.loader import load_campaign
from profile_suite.modules.base import PointSpec, make_point_id
from profile_suite.orchestrator import run
from profile_suite.platform.a100 import A100_DESCRIPTOR
from profile_suite.results import RunRepeatResult


# --------------------------------------------------------------------------- #
# Fakes (no GPU / no real server)
# --------------------------------------------------------------------------- #
class FakePlatform:
    """A Platform adapter that marks an explicit set of Configs infeasible.

    ``is_feasible`` returns ``(False, reason)`` for any Config in
    ``infeasible_configs`` and ``(True, None)`` otherwise, so the orchestrator's
    skip path (R6.6) is exercised over an arbitrary feasible/infeasible mix.
    """

    descriptor = A100_DESCRIPTOR

    def __init__(self, infeasible_configs: frozenset[Config]) -> None:
        self._infeasible = infeasible_configs

    def resolve_binary(self, kind: str) -> str:
        return f"/fake/llama-{kind}"

    def device_total_mib(self, gpu_index: int) -> float:
        return 40960.0

    def is_feasible(self, config: Config):
        if config in self._infeasible:
            return (False, f"infeasible on fake platform: {config!r}")
        return (True, None)


class FakeModule:
    """A profiler module that returns successful runs without booting a server."""

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
# Strategy: a mixed grid (axes) + an arbitrary infeasible index subset
# --------------------------------------------------------------------------- #
_QUANT_FILES = ["Meta-Q4_K_M.gguf", "Meta-Q8_0.gguf"]
_CTX_LENGTHS = [2048, 4096, 32768]
_SLOT_COUNTS = [1, 2, 4]

# The expanded grid has at most len(_QUANT_FILES)*... but each axis is capped at 2
# distinct values below, so at most 2*2*2 = 8 Configs.
_MAX_GRID = 8


@st.composite
def mixed_grids(draw):
    """Generate ``(axes, infeasible_idx)`` for a mixed Config_Grid.

    ``axes`` is a dict of the three Config_Grid axes drawn over the value pools
    (a mix of context lengths and slot counts). ``infeasible_idx`` is an arbitrary
    set of integer indices into the *deterministically expanded* grid, selecting
    which Configs the fake Platform reports infeasible. Indices beyond the grid
    length are simply ignored by the test, and the empty / full subsets are both
    reachable, so "skip none" and "skip all" (the campaign must still not abort)
    are exercised.
    """
    axes = {
        "quant_files": draw(
            st.lists(st.sampled_from(_QUANT_FILES), min_size=1, max_size=2, unique=True)
        ),
        "ctx_lengths": draw(
            st.lists(st.sampled_from(_CTX_LENGTHS), min_size=1, max_size=2, unique=True)
        ),
        "slot_counts": draw(
            st.lists(st.sampled_from(_SLOT_COUNTS), min_size=1, max_size=2, unique=True)
        ),
    }
    infeasible_idx = draw(
        st.sets(st.integers(min_value=0, max_value=_MAX_GRID - 1), max_size=_MAX_GRID)
    )
    return axes, infeasible_idx


def _write_campaign(tmp_path: Path, *, server_binary: Path, axes: dict) -> Path:
    """Write a valid campaign YAML over ``axes``, materializing the model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    for name in axes["quant_files"]:
        (model_dir / name).write_text("gguf")

    spec = {
        "platform": A100_DESCRIPTOR,
        "server_binary": str(server_binary),
        "bench_binary": str(tmp_path / "llama-bench"),
        "batched_bench_binary": str(tmp_path / "llama-batched-bench"),
        "model_dir": str(model_dir),
        "gpu_index": 0,
        "config_grid": {
            "quant_files": list(axes["quant_files"]),
            "ctx_lengths": list(axes["ctx_lengths"]),
            "slot_counts": list(axes["slot_counts"]),
        },
        # A small repeat count keeps the (server-less) run loop fast while still
        # exercising one discarded warmup + a retained repeat.
        "run_repeats": 1,
        "prompt_tokens": 8,
        "output_tokens": 4,
        "enabled_modules": ["Performance_Profiler"],
    }
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


# --------------------------------------------------------------------------- #
# Property 15
# --------------------------------------------------------------------------- #
# Feature: profile-suite, Property 15: Platform-infeasible Configs are skipped without aborting the campaign
@settings(max_examples=50, deadline=None)
@given(spec=mixed_grids())
def test_platform_infeasible_configs_are_skipped_without_aborting(spec):
    axes, infeasible_idx = spec

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        server_binary = tmp_path / "llama-server"
        server_binary.write_text("#!/bin/sh\n")

        campaign = _write_campaign(tmp_path, server_binary=server_binary, axes=axes)

        # Derive the *resolved* grid (quant files become absolute paths) so the
        # fake Platform's infeasible set shares Config identity with the
        # orchestrator's grid. Select the infeasible subset by index.
        cfg = load_campaign(str(campaign))
        grid = expand_grid(cfg.config_grid)
        infeasible = frozenset(
            grid[i] for i in infeasible_idx if i < len(grid)
        )

        result = run(
            campaign,
            runs_root=str(runs_root),
            platform=FakePlatform(infeasible),
            module_overrides={"Performance_Profiler": FakeModule()},
        )

        # The campaign ran to completion (not aborted by infeasible Configs).
        assert result.ok is True
        assert result.campaign_dir is not None

        # Exactly one MeasurementPoint per grid Config, no duplicates.
        assert len(result.points) == len(grid)
        measured_configs = [p.config for p in result.points]
        assert sorted(measured_configs, key=_config_key) == sorted(grid, key=_config_key)

        feasible_expected = {c for c in grid if c not in infeasible}

        for point in result.points:
            if point.config in infeasible:
                # Infeasible Config: recorded, skipped, reason, no repeats.
                assert point.status == "platform-infeasible"
                assert point.reason is not None and point.reason != ""
                assert point.repeats == []
                assert point.aggregates == {}
            else:
                # Feasible Config: still measured to completion.
                assert point.config in feasible_expected
                assert point.status == "complete"
                assert point.reason is None
                # One discarded warmup + at least one retained successful repeat.
                assert point.repeats[0].discarded_warmup is True
                assert any(r.ok and not r.discarded_warmup for r in point.repeats)

        # Cross-check: skipped == infeasible, completed == feasible, partition.
        skipped = {p.config for p in result.points if p.status == "platform-infeasible"}
        completed = {p.config for p in result.points if p.status == "complete"}
        assert skipped == set(infeasible)
        assert completed == feasible_expected


def _config_key(c: Config):
    return (c.quant_file, c.ctx_length, c.slot_count)
