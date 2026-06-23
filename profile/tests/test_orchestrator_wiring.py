"""End-to-end wiring tests for ``profile_suite.orchestrator.run`` (task 18.2).

Two paths are covered without booting any real ``llama-server``:

1. **Validate-decline path** (R8.4/R8.6/R6.2): a campaign referencing a
   non-existent server binary is declined with a specific reason and produces
   **no** campaign output directory.

2. **Fake-platform / fake-module happy path** (R5.4/R5.9/R6.6): a valid campaign
   is run with a fake Platform (mixed feasibility, no GPU) and fake measurement
   modules (return successful runs without booting servers). Asserts the run loop
   discards one warmup + retains repeats, platform-infeasible Configs are skipped
   with their reason, results are persisted to ``points.json``, the manifest is
   written, the Reporting_Module artifacts are emitted, and every output stays
   under the campaign-scoped directory.
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
# Fakes
# --------------------------------------------------------------------------- #
class FakePlatform:
    """A Platform adapter that needs no GPU and marks one Config infeasible."""

    descriptor = A100_DESCRIPTOR

    def __init__(self, infeasible_ctx: int | None = None) -> None:
        # A Config whose ctx_length equals this value is reported infeasible, so
        # the skip path (R6.6) is exercised deterministically.
        self._infeasible_ctx = infeasible_ctx

    def resolve_binary(self, kind: str) -> str:
        return f"/fake/llama-{kind}"

    def device_total_mib(self, gpu_index: int) -> float:
        return 40960.0

    def is_feasible(self, config: Config):
        if self._infeasible_ctx is not None and config.ctx_length == self._infeasible_ctx:
            return (False, f"ctx_length={config.ctx_length} infeasible on fake")
        return (True, None)


class FakeModule:
    """A profiler module that returns successful runs without booting a server."""

    name = "Performance_Profiler"

    def __init__(self) -> None:
        self.calls: list[int] = []

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
        # Exercise the harness-context surface so the wiring is real.
        idx = harness.run_index
        self.calls.append(idx)
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
# Campaign fixtures
# --------------------------------------------------------------------------- #
def _write_campaign(tmp_path: Path, *, server_binary: str, model_files: list[str],
                    enabled_modules=None, ctx_lengths=(4096, 32768)) -> Path:
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    quant_files = []
    for name in model_files:
        p = model_dir / name
        p.write_text("gguf")
        quant_files.append(name)

    spec = {
        "platform": A100_DESCRIPTOR,
        "server_binary": server_binary,
        "bench_binary": str(tmp_path / "llama-bench"),
        "batched_bench_binary": str(tmp_path / "llama-batched-bench"),
        "model_dir": str(model_dir),
        "gpu_index": 0,
        "config_grid": {
            "quant_files": quant_files,
            "ctx_lengths": list(ctx_lengths),
            "slot_counts": [1],
        },
        "run_repeats": 5,
        "prompt_tokens": 8,
        "output_tokens": 4,
    }
    if enabled_modules is not None:
        spec["enabled_modules"] = enabled_modules

    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


# --------------------------------------------------------------------------- #
# 1. Validate-decline path: reason + NO output directory (R8.4/R8.6/R6.2)
# --------------------------------------------------------------------------- #
def test_run_declines_invalid_campaign_and_produces_no_output(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # server_binary points to a path that does not exist -> validation declines.
    campaign = _write_campaign(
        tmp_path,
        server_binary=str(tmp_path / "does-not-exist-llama-server"),
        model_files=["Meta-Q4_K_M.gguf"],
    )

    result = run(campaign, runs_root=str(runs_root))

    assert result.ok is False
    assert result.reason is not None
    assert "does-not-exist-llama-server" in result.reason
    assert result.campaign_dir is None
    # No campaign output directory was produced (R8.4/R8.6).
    assert list(runs_root.iterdir()) == []


# --------------------------------------------------------------------------- #
# 2. Fake-platform / fake-module happy path end-to-end
# --------------------------------------------------------------------------- #
def test_run_happy_path_with_fakes_end_to_end(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    server_binary = tmp_path / "llama-server"
    server_binary.write_text("#!/bin/sh\n")

    # Two quant formats x two ctx lengths x one slot = 4 Configs. One ctx length
    # (32768) is marked platform-infeasible so the skip path is exercised.
    campaign = _write_campaign(
        tmp_path,
        server_binary=str(server_binary),
        model_files=["Meta-Q4_K_M.gguf", "Meta-Q8_0.gguf"],
        enabled_modules=["Performance_Profiler", "Quality_Module", "Reporting_Module"],
        ctx_lengths=(4096, 32768),
    )

    fake_module = FakeModule()
    result = run(
        campaign,
        runs_root=str(runs_root),
        platform=FakePlatform(infeasible_ctx=32768),
        module_overrides={"Performance_Profiler": fake_module},
    )

    # --- Success + campaign dir under runs_root (isolation) ---------------- #
    assert result.ok is True
    assert result.campaign_dir is not None
    campaign_dir = Path(result.campaign_dir)
    assert campaign_dir.exists()
    assert runs_root.resolve() in campaign_dir.resolve().parents

    # --- 4 points: 2 feasible (ctx 4096) measured, 2 infeasible skipped ---- #
    assert len(result.points) == 4
    feasible = [p for p in result.points if p.status != "platform-infeasible"]
    infeasible = [p for p in result.points if p.status == "platform-infeasible"]
    assert len(feasible) == 2
    assert len(infeasible) == 2
    for p in infeasible:
        assert p.config.ctx_length == 32768
        assert p.reason and "infeasible" in p.reason
        assert p.repeats == []
    for p in feasible:
        assert p.status == "complete"
        # One discarded warmup + 5 retained successful repeats.
        assert p.repeats[0].discarded_warmup is True
        assert sum(1 for r in p.repeats[1:] if r.ok) == 5
        assert "decode_throughput" in p.aggregates

    # --- run index threaded onto the context (0 warmup, 1..5 retained) ----- #
    # Two feasible points x (1 warmup + 5 retained) = 12 measure_once calls.
    assert len(fake_module.calls) == 12

    # --- points.json persisted under the campaign dir ---------------------- #
    points_json = campaign_dir / "points.json"
    assert points_json.is_file()
    persisted = json.loads(points_json.read_text())
    assert len(persisted) == 4

    # --- manifest + environment written ------------------------------------ #
    assert (campaign_dir / "manifest.json").is_file()
    assert (campaign_dir / "environment.json").is_file()

    # --- quality.json written (Quality_Module enabled) --------------------- #
    quality_json = campaign_dir / "quality.json"
    assert quality_json.is_file()
    quality = json.loads(quality_json.read_text())
    formats = {q["quant_format"] for q in quality}
    assert formats == {"Q4_K_M", "Q8_0"}

    # --- Reporting_Module artifacts emitted per Platform ------------------- #
    assert A100_DESCRIPTOR in result.artifacts
    art = result.artifacts[A100_DESCRIPTOR]
    assert art["results"] is not None and Path(art["results"]).is_file()
    assert art["c_switch_table"] is not None and Path(art["c_switch_table"]).is_file()
    assert art["memory_table"] is not None and Path(art["memory_table"]).is_file()

    # --- isolation: every artifact path lives under the campaign dir ------- #
    for kind, p in art.items():
        if p is not None:
            assert campaign_dir.resolve() in Path(p).resolve().parents
