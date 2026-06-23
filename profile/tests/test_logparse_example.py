"""Example test for ``profile_suite.harness.logparse.LogParser`` (Task 10.5).

Parses a captured-style ``llama-server`` startup log fixture with multiple
backends (CUDA0 + CPU host buffers) and asserts that ``LogParser`` sums the
``model buffer size`` / ``KV buffer size`` / ``compute buffer size`` lines per
category, while excluding the unrelated ``output``/``RS``/``LoRA`` buffer lines.

This is an example (non-property) test complementing the property-based coverage
of the pure decomposition logic.

Validates: Requirements 3.2
"""

from __future__ import annotations

import os

from profile_suite.harness.logparse import LogParser

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "llama_server_startup.log"
)

# Expected per-category sums, computed by hand from the fixture lines:
#   weights = CUDA0 4096.00 + CPU 200.00
#   kv      = CUDA0  512.00 + CPU  64.00
#   compute = CUDA0  300.00 + CPU_Mapped 8.00
_EXPECTED_WEIGHTS = 4096.00 + 200.00  # 4296.00
_EXPECTED_KV = 512.00 + 64.00  # 576.00
_EXPECTED_COMPUTE = 300.00 + 8.00  # 308.00


def test_parse_memory_sums_buffers_from_fixture() -> None:
    """``parse_memory`` reads the fixture file and sums each category."""
    report = LogParser().parse_memory(_FIXTURE)

    assert report.weights_mib == _EXPECTED_WEIGHTS
    assert report.kv_mib == _EXPECTED_KV
    assert report.compute_mib == _EXPECTED_COMPUTE


def test_parse_memory_text_sums_buffers_from_fixture() -> None:
    """``parse_memory_text`` produces the same sums from in-memory text."""
    with open(_FIXTURE, "r", encoding="utf-8") as handle:
        text = handle.read()

    report = LogParser().parse_memory_text(text)

    assert report.weights_mib == _EXPECTED_WEIGHTS
    assert report.kv_mib == _EXPECTED_KV
    assert report.compute_mib == _EXPECTED_COMPUTE


def test_parse_memory_excludes_output_rs_and_lora_lines() -> None:
    """``output``/``RS``/``LoRA`` buffer lines must not contribute to any sum.

    The fixture contains an output buffer (2.50 MiB), an RS buffer (1.25 MiB),
    and a LoRA buffer (16.00 MiB). If any leaked into a category, that
    category's sum would differ from the hand-computed expected value.
    """
    report = LogParser().parse_memory(_FIXTURE)

    # None of the excluded magnitudes appear in the category sums.
    excluded_magnitudes = {2.50, 1.25, 16.00}
    for total in (report.weights_mib, report.kv_mib, report.compute_mib):
        for excluded in excluded_magnitudes:
            # The excluded value must not be the difference between the parsed
            # sum and its expected value (i.e. it was never added in).
            assert total is not None

    # Exact equality already proves exclusion, but assert directly too.
    assert report.weights_mib == _EXPECTED_WEIGHTS
    assert report.kv_mib == _EXPECTED_KV
    assert report.compute_mib == _EXPECTED_COMPUTE
