"""Slow A100 integration test for the Switch_Cost_Profiler (Task 19.2).

This exercises the **real** :class:`Switch_Cost_Profiler.measure_once` against a
real ``llama-server`` boot/teardown on the A100, validating the paper's central
C_switch claim end-to-end:

- one real reconfiguration produces a **positive** total ``c_switch_ms`` (R1.1),
- and its Teardown / Boot / Warmup components **reconcile** to that total within
  the design's 50 ms tolerance (R1.2).

It is marked ``slow`` (registered in ``profile/pyproject.toml``) so it is
deselected from the default fast suite (``pytest -m "not slow"``). The test is
always *collectable* — when the real CUDA server binary or the model files are
absent it issues a clear :func:`pytest.skip` rather than failing, so it is safe
to collect in any environment.

Validates: Requirements 1.1, 1.2
"""

from __future__ import annotations

import os

import pytest

from profile_suite.config import Config
from profile_suite.harness.client import Client
from profile_suite.harness.sysprobe import SysProbe
from profile_suite.modules.base import PointSpec
from profile_suite.modules.switch_cost import Switch_Cost_Profiler
from profile_suite.orchestrator import MeasurementContext
from profile_suite.platform.a100 import A100CudaPlatform

# --------------------------------------------------------------------------- #
# Real environment under test (A100 CUDA build of llama.cpp)
# --------------------------------------------------------------------------- #
_BUILD_BIN = "/scratch2/tingyang/llama.cpp/build-cuda/bin"
SERVER_BINARY = os.path.join(_BUILD_BIN, "llama-server")
BENCH_BINARY = os.path.join(_BUILD_BIN, "llama-bench")
BATCHED_BENCH_BINARY = os.path.join(_BUILD_BIN, "llama-batched-bench")

MODEL_DIR = "/scratch2/tingyang/llama.cpp/models"
# A model-reload switch: same ctx/slots, different quant/model file. Both are
# small real gemma models present under the models directory, so a real boot is
# fast enough for a slow integration test.
FROM_MODEL = "gemma-3-1b-it-Q4_K_M.gguf"
TO_MODEL = "gemma-3-1b-it-Q8_0.gguf"

GPU_INDEX = 0
CTX_LENGTH = 4096
SLOT_COUNT = 1

# Design tolerance for component-vs-total reconciliation (R1.2).
RECONCILE_TOLERANCE_MS = 50.0


@pytest.mark.slow
def test_real_switch_cost_positive_and_reconciles_within_50ms(tmp_path):
    """One real switch yields a positive C_switch whose phases reconcile (R1.1, R1.2).

    Performs a single real ``teardown(from) + boot(to) + first-token(to)`` switch
    on the A100 via the production :class:`Switch_Cost_Profiler.measure_once`,
    then asserts the total is positive and its components sum back to the total
    within 50 ms. Skipped (not failed) when the CUDA server binary or the model
    files are unavailable, so the test stays safe to collect everywhere.
    """
    # --- Environment guards: skip cleanly when the real env is absent -------- #
    if not os.path.exists(SERVER_BINARY):
        pytest.skip(f"CUDA llama-server binary not found at {SERVER_BINARY}")

    from_path = os.path.join(MODEL_DIR, FROM_MODEL)
    to_path = os.path.join(MODEL_DIR, TO_MODEL)
    for model_path in (from_path, to_path):
        if not os.path.exists(model_path):
            pytest.skip(f"model file not found at {model_path}")

    # --- Build the real A100 platform + harness context --------------------- #
    platform = A100CudaPlatform(
        server_binary=SERVER_BINARY,
        bench_binary=BENCH_BINARY,
        batched_bench_binary=BATCHED_BENCH_BINARY,
        gpu_index=GPU_INDEX,
    )

    from_cfg = Config(quant_file=from_path, ctx_length=CTX_LENGTH, slot_count=SLOT_COUNT)
    to_cfg = Config(quant_file=to_path, ctx_length=CTX_LENGTH, slot_count=SLOT_COUNT)

    # The MeasurementContext is the duck-typed harness surface the profiler reads
    # (client + make_server, plus warmup_timeout_s). Its campaign_dir is a pytest
    # tmp dir so per-run server logs land in an isolated, throwaway location.
    context = MeasurementContext(
        campaign_dir=tmp_path,
        platform=platform,
        gpu_index=GPU_INDEX,
        boot_timeout_s=300.0,
        warmup_timeout_s=60.0,
        prompt=None,
        client=Client(),
        sysprobe=SysProbe(),
    )

    # --- Build the (from -> to) transition spec ----------------------------- #
    profiler = Switch_Cost_Profiler()
    change_type = from_cfg.change_type(to_cfg)
    assert change_type == "model-reload"  # same ctx/slots, different model file
    spec = PointSpec(
        module=profiler.name,
        config=to_cfg,
        axis={"change_type": change_type},
        from_config=from_cfg,
        to_config=to_cfg,
    )

    # Bind the point + run index so make_server allocates a stable per-run log path.
    context._bind_point(spec.module, spec.point_id)
    context.run_index = 1

    # --- Perform one real switch ------------------------------------------- #
    result = profiler.measure_once(spec, context, platform)

    # --- Assert: success, positive total, reconciliation within 50 ms ------- #
    assert result.ok, f"real switch failed: {result.error} (log: {result.raw_log_path})"

    metrics = result.metrics
    for key in ("teardown_ms", "boot_ms", "warmup_ms", "c_switch_ms"):
        assert key in metrics, f"missing metric {key!r} in {metrics!r}"

    c_switch_ms = metrics["c_switch_ms"]
    assert c_switch_ms > 0, f"expected positive C_switch, got {c_switch_ms}"

    component_sum = metrics["teardown_ms"] + metrics["boot_ms"] + metrics["warmup_ms"]
    residual = abs(c_switch_ms - component_sum)
    assert residual <= RECONCILE_TOLERANCE_MS, (
        f"C_switch components do not reconcile within {RECONCILE_TOLERANCE_MS} ms: "
        f"c_switch={c_switch_ms:.3f} ms, "
        f"teardown+boot+warmup={component_sum:.3f} ms, residual={residual:.3f} ms"
    )
