"""Property test for decode-batch-size validation and the canonical default.

profile-suite Property 5 (Validates: Requirements 2.2, 8.3).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.campaign import (
    CANONICAL_DECODE_BATCH_SIZES,
    DECODE_BATCH_MAX,
    DECODE_BATCH_MIN,
    apply_defaults,
    validate_decode_batches,
)
from profile_suite.config import GridSpec

# A pool of candidate batch values mixing in-range members (1..128) with
# out-of-range ones (0 and 129+, plus a negative) so generated lists exercise
# both the accept and reject branches of validate_decode_batches.
_BATCH_VALUES = st.integers(min_value=-10, max_value=300)


def _minimal_spec() -> dict:
    """A minimal valid partial campaign spec that omits decode_batch_sizes."""
    return {
        "platform": "a100-cuda",
        "server_binary": "/bin/llama-server",
        "bench_binary": "/bin/llama-bench",
        "batched_bench_binary": "/bin/llama-batched-bench",
        "model_dir": "/models",
        "gpu_index": 0,
        "config_grid": GridSpec(
            quant_files=("q4_k_m.gguf",),
            ctx_lengths=(4096,),
            slot_counts=(1,),
        ),
        "prompt_tokens": 512,
        "output_tokens": 128,
    }


# Feature: profile-suite, Property 5: Decode-batch-size sets are bounds-validated and the default is canonical
@settings(max_examples=100)
@given(st.lists(_BATCH_VALUES, max_size=12))
def test_decode_batches_bounds_validated_and_default_canonical(
    candidate: list[int],
) -> None:
    """Validates: Requirements 2.2, 8.3

    For any candidate Decode_Batch_Size set, validate_decode_batches accepts it
    if and only if every value is an int in the inclusive range 1..128; and when
    a campaign omits decode_batch_sizes, apply_defaults applies exactly the
    canonical default (1, 2, 4, 8, 16, 32, 64, 128) on the resulting config and
    records that same set in applied_defaults.
    """
    # --- bounds validation: accept iff every value is in 1..128 inclusive ---
    expected_valid = all(DECODE_BATCH_MIN <= v <= DECODE_BATCH_MAX for v in candidate)
    assert validate_decode_batches(candidate) is expected_valid

    # --- canonical default applied (and recorded) when omitted (R8.3) -------
    config, applied = apply_defaults(_minimal_spec())
    assert config.decode_batch_sizes == (1, 2, 4, 8, 16, 32, 64, 128)
    assert CANONICAL_DECODE_BATCH_SIZES == (1, 2, 4, 8, 16, 32, 64, 128)
    assert applied["decode_batch_sizes"] == [1, 2, 4, 8, 16, 32, 64, 128]
