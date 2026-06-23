"""Slow A100 integration test for Memory_Profiler peak-memory sampling (Task 19.4).

This exercises the **real** measurement path end-to-end on an A100:
``Memory_Profiler.measure_once`` boots a real ``llama-server`` (CUDA build) for
one real :class:`~profile_suite.config.Config`, starts the background
:meth:`SysProbe.sample_peak` ``nvidia-smi`` sampler, drives the measurement
workload through the streaming :class:`~profile_suite.harness.client.Client`,
then decomposes the directly observed peak device-memory footprint into the
weights / KV-per-slot / scratch+overhead components (R3.1, R3.2).

It is marked ``@pytest.mark.slow`` and is therefore **excluded** from the
default fast suite (``pytest -m "not slow"``). It also **skips cleanly** when
the real environment is absent:

- the CUDA ``llama-server`` binary does not exist,
- no real (non-vocab) model GGUF is present under the models directory, or
- ``nvidia-smi`` is unavailable.

Validates: Requirements 3.1
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile

import pytest

from profile_suite.config import Config
from profile_suite.modules.base import PointSpec
from profile_suite.modules.memory import Memory_Profiler
from profile_suite.platform.a100 import A100CudaPlatform

# --------------------------------------------------------------------------- #
# Real-environment locations (the A100 box this suite targets).
# --------------------------------------------------------------------------- #
_BUILD_BIN = "/scratch2/tingyang/llama.cpp/build-cuda/bin"
_SERVER_BINARY = os.path.join(_BUILD_BIN, "llama-server")
_BENCH_BINARY = os.path.join(_BUILD_BIN, "llama-bench")
_BATCHED_BENCH_BINARY = os.path.join(_BUILD_BIN, "llama-batched-bench")
_MODEL_DIR = "/scratch2/tingyang/llama.cpp/models"

# A modest context + a single KV slot keeps the boot fast while still allocating
# real KV and compute buffers so the decomposition has something to split.
_CTX_LENGTH = 2048
_SLOT_COUNT = 1
_GPU_INDEX = int(os.environ.get("PROFILE_TEST_GPU_INDEX", "0"))


def _find_real_model() -> str | None:
    """Return the path to a real (non-vocab) instruct GGUF model, or ``None``.

    The ``models`` directory mixes real model weights with tiny tokenizer-only
    ``ggml-vocab-*.gguf`` fixtures (and their ``.inp``/``.out`` companions); only
    a real model can actually serve completions, so those are filtered out. An
    instruct/chat model is preferred (the streaming workload uses the chat
    endpoint); otherwise the largest remaining ``.gguf`` is used as a heuristic
    for "a real model".
    """
    if not os.path.isdir(_MODEL_DIR):
        return None

    candidates: list[str] = []
    for name in os.listdir(_MODEL_DIR):
        if not name.endswith(".gguf"):
            continue
        if name.startswith("ggml-vocab-"):
            continue
        candidates.append(os.path.join(_MODEL_DIR, name))

    if not candidates:
        return None

    # Prefer an instruct/chat model (its name typically contains "it"/"instruct"
    # /"chat"); these reliably answer the OpenAI-compatible chat endpoint.
    def _is_instruct(path: str) -> bool:
        low = os.path.basename(path).lower()
        return any(tag in low for tag in ("-it-", "-it.", "instruct", "chat"))

    instruct = [p for p in candidates if _is_instruct(p)]
    pool = instruct or candidates
    # Smallest instruct model boots fastest; fall back to any model otherwise.
    pool.sort(key=lambda p: os.path.getsize(p))
    return pool[0]


def _free_port() -> int:
    """Pick a currently-free localhost TCP port for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _env_skip_reason() -> str | None:
    """Return a skip reason if the real A100 environment is absent, else ``None``."""
    if not os.path.isfile(_SERVER_BINARY) or not os.access(_SERVER_BINARY, os.X_OK):
        return f"CUDA llama-server binary not found/executable at {_SERVER_BINARY}"
    if _find_real_model() is None:
        return f"no real (non-vocab) model GGUF found under {_MODEL_DIR}"
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi is unavailable"
    return None


class _IntegrationHarness:
    """Minimal real harness object for ``Memory_Profiler.measure_once``.

    The module duck-types its runtime context off the harness (see
    ``Memory_Profiler`` docstring). This supplies the attributes it reads with
    *real* I/O collaborators (a real ``SysProbe`` peak sampler and a real
    streaming ``Client``) so the test exercises the genuine measurement path,
    not fakes. Per-run server logs go under a campaign-scoped temp directory.
    """

    def __init__(self, host: str, port: int, gpu_index: int, prompt: str, log_dir: str):
        from profile_suite.harness.client import Client
        from profile_suite.harness.sysprobe import SysProbe

        self.run_index = 1  # a retained repeat (not the discarded warmup)
        self.host = host
        self.port = port
        self.gpu_index = gpu_index
        self.boot_timeout_s = float(os.environ.get("PROFILE_TEST_BOOT_TIMEOUT_S", "300"))
        self.prompt = prompt
        self.sysprobe = SysProbe()
        self.client = Client()
        self._log_dir = log_dir

    def server_log_path(self, point_id: str, run_index: int) -> str:
        """Return a per-run server log path under the test's temp directory (R5.5)."""
        safe = "".join(ch if (ch.isalnum() or ch in "._=-") else "_" for ch in point_id)
        return os.path.join(self._log_dir, f"{safe}_run{run_index:02d}.log")


@pytest.mark.slow
def test_memory_profiler_samples_and_decomposes_peak_on_a100() -> None:
    """One real Config's peak device memory is sampled and decomposed (R3.1).

    Boots a real CUDA ``llama-server`` for one Config, runs the real workload
    with the background nvidia-smi peak sampler, and asserts the returned
    ``RunRepeatResult`` is ``ok`` with a positive directly-observed peak and the
    three non-negative decomposition components present.
    """
    skip_reason = _env_skip_reason()
    if skip_reason is not None:
        pytest.skip(skip_reason)

    model_path = _find_real_model()
    assert model_path is not None  # guaranteed by _env_skip_reason()

    config = Config(
        quant_file=model_path,
        ctx_length=_CTX_LENGTH,
        slot_count=_SLOT_COUNT,
    )

    platform = A100CudaPlatform(
        server_binary=_SERVER_BINARY,
        bench_binary=_BENCH_BINARY,
        batched_bench_binary=_BATCHED_BENCH_BINARY,
        gpu_index=_GPU_INDEX,
    )

    profiler = Memory_Profiler()
    spec = PointSpec(
        module=profiler.name,
        config=config,
        axis={},
        params={"prompt_tokens": 64, "output_tokens": 16},
    )

    with tempfile.TemporaryDirectory(prefix="profile_mem_it_") as log_dir:
        harness = _IntegrationHarness(
            host="127.0.0.1",
            port=_free_port(),
            gpu_index=_GPU_INDEX,
            prompt="Summarize the theory of relativity in one short paragraph.",
            log_dir=log_dir,
        )

        result = profiler.measure_once(spec, harness, platform)

    # The run must succeed: the server booted, the peak was sampled, and the
    # log parsed (R3.1/R3.7).
    assert result.ok, f"memory measurement failed: {result.error}"

    metrics = result.metrics

    # Directly observed peak device-memory footprint is positive (R3.1): a real
    # server holding model weights on the GPU must occupy memory.
    assert "observed_peak" in metrics, "observed_peak missing from metrics"
    assert metrics["observed_peak"] > 0.0, (
        f"expected a positive observed peak, got {metrics['observed_peak']}"
    )

    # The three decomposition components are present and non-negative (R3.2).
    for component in ("weights", "kv_per_slot", "scratch_overhead"):
        assert component in metrics, f"{component} missing from metrics"
        assert metrics[component] >= 0.0, (
            f"{component} must be non-negative, got {metrics[component]}"
        )
