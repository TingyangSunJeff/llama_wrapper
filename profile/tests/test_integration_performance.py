"""Slow A100 integration test for the Performance_Profiler server path (task 19.3).

This is a **real** integration test (Requirement 2.1): it boots an actual
``llama-server`` (the CUDA build) on the A100 against a real model and exercises
the unmocked :meth:`Performance_Profiler.measure_once` server path end to end,
asserting that one Config yields all four performance metrics — ``prefill_throughput``,
``decode_throughput``, ``ttft_ms``, and ``tpot_ms`` — each present and positive.

It is marked ``@pytest.mark.slow`` so the default fast suite (``-m "not slow"``)
never runs it. It also **skips cleanly** when the CUDA server binary or a real
model is absent, so the file is always safe to collect on any machine.

Run it explicitly with::

    /scratch2/tingyang/anaconda/envs/mynewenv/bin/python -m pytest \\
        profile/tests/test_integration_performance.py -m slow -v
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from profile_suite.config import Config
from profile_suite.harness.client import Client
from profile_suite.harness.sysprobe import SysProbe
from profile_suite.modules.base import PointSpec
from profile_suite.modules.performance import Performance_Profiler
from profile_suite.orchestrator import MeasurementContext
from profile_suite.platform.a100 import A100CudaPlatform

# Real A100 CUDA build + model locations (R2.1, R6.3). The test skips cleanly when
# either is missing so it is safe to collect anywhere.
SERVER_BINARY = "/scratch2/tingyang/llama.cpp/build-cuda/bin/llama-server"
BENCH_BINARY = "/scratch2/tingyang/llama.cpp/build-cuda/bin/llama-bench"
BATCHED_BENCH_BINARY = "/scratch2/tingyang/llama.cpp/build-cuda/bin/llama-batched-bench"
MODEL_DIR = "/scratch2/tingyang/llama.cpp/models"

# The four metrics the server path must produce for one Config (R2.1).
REQUIRED_METRICS = ("prefill_throughput", "decode_throughput", "ttft_ms", "tpot_ms")


def _find_real_model() -> str | None:
    """Return one real (non vocab-only) ``.gguf`` model path, or ``None``.

    The ``models/`` directory is dominated by ``ggml-vocab-*.gguf`` tokenizer-only
    fixtures that cannot serve generation requests; those are excluded so the test
    boots against a genuine generative model (e.g. a small instruct model).
    """
    model_dir = Path(MODEL_DIR)
    if not model_dir.is_dir():
        return None
    candidates = sorted(
        p
        for p in model_dir.glob("*.gguf")
        if not p.name.startswith("ggml-vocab-")
    )
    return str(candidates[0]) if candidates else None


def _free_port() -> int:
    """Pick an ephemeral free TCP port to avoid colliding with other servers."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.slow
def test_performance_profiler_server_path_real_a100(tmp_path):
    """One real Config produces all four performance metrics, positive (R2.1).

    Boots a real ``llama-server`` via the unmocked server path of
    :meth:`Performance_Profiler.measure_once` and asserts the returned
    :class:`RunRepeatResult` is ``ok`` with ``prefill_throughput``,
    ``decode_throughput``, ``ttft_ms``, and ``tpot_ms`` all present and positive.
    """
    if not os.path.exists(SERVER_BINARY):
        pytest.skip(f"CUDA server binary not found: {SERVER_BINARY}")

    model_path = _find_real_model()
    if model_path is None:
        pytest.skip(f"no real (non vocab-only) model found under {MODEL_DIR}")

    gpu_index = int(os.environ.get("PROFILE_SUITE_GPU_INDEX", "0"))

    # Real A100 CUDA platform adapter (resolves the server binary; R6.3).
    platform = A100CudaPlatform(
        server_binary=SERVER_BINARY,
        bench_binary=BENCH_BINARY,
        batched_bench_binary=BATCHED_BENCH_BINARY,
        gpu_index=gpu_index,
    )

    # The real harness surface the module reads: run_index/host/port/gpu_index/
    # boot_timeout_s/server_log_path/prompt + a real Client. We reuse the
    # production MeasurementContext for maximum fidelity; its raw logs land under
    # the pytest tmp_path campaign dir.
    context = MeasurementContext(
        campaign_dir=tmp_path,
        platform=platform,
        gpu_index=gpu_index,
        boot_timeout_s=180.0,
        warmup_timeout_s=60.0,
        prompt="Explain what a large language model is in one short paragraph.",
        client=Client(),
        sysprobe=SysProbe(),
        host="127.0.0.1",
        port=_free_port(),
    )
    context.run_index = 1  # a retained repeat (0 would be the discarded warmup)

    # One real Config: the resolved model, a modest context and a single KV slot.
    # Output kept small to keep the single real boot+decode reasonably quick.
    config = Config(quant_file=model_path, ctx_length=4096, slot_count=1)
    spec = PointSpec(
        module="Performance_Profiler",
        config=config,
        axis={"path": "server"},
        params={"prompt_tokens": 64, "output_tokens": 16},
    )

    profiler = Performance_Profiler()
    result = profiler.measure_once(spec, context, platform)

    assert result.ok, f"measurement run failed: {result.error}"
    for metric in REQUIRED_METRICS:
        assert metric in result.metrics, f"missing metric {metric!r}"
        value = result.metrics[metric]
        assert value > 0.0, f"metric {metric!r} must be positive, got {value}"
