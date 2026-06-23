"""Property test for ``profile_suite.modules.memory.decompose`` (Property 7).

Validates that the three reported footprint components (model-weights,
KV-cache-per-slot, scratch-plus-overhead) are each non-negative, that
``kv_per_slot`` equals the total KV footprint divided by the slot count, and
that a slot count below 1 raises ``ValueError``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.modules.memory import decompose


@dataclass
class FakeMemoryReport:
    """A ``ServerMemoryReport``-like object (duck-typed by ``decompose``).

    Each component may be ``None`` (category absent from the log) or a float,
    including negative values, so the non-negativity clamping in ``decompose``
    is exercised.
    """

    weights_mib: float | None
    kv_mib: float | None
    compute_mib: float | None


# Component MiB values: ``None`` (absent) or a finite float that may be negative
# (to exercise the clamp-to-zero behaviour). Magnitudes are bounded so the
# arithmetic stays numerically well-behaved.
_component = st.one_of(
    st.none(),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)


@st.composite
def memory_reports(draw):
    """Generate a ``(report, slot_count, observed_peak)`` triple.

    ``slot_count`` is a positive integer and ``observed_peak`` is a
    non-negative finite float (which may itself be smaller than the parsed
    components, exercising the residual clamp).
    """
    report = FakeMemoryReport(
        weights_mib=draw(_component),
        kv_mib=draw(_component),
        compute_mib=draw(_component),
    )
    slot_count = draw(st.integers(min_value=1, max_value=128))
    observed_peak = draw(
        st.floats(min_value=0.0, max_value=1e7, allow_nan=False, allow_infinity=False)
    )
    return report, slot_count, observed_peak


# Feature: profile-suite, Property 7: Memory components are non-negative and KV-per-slot is the divided total
@settings(max_examples=100)
@given(case=memory_reports())
def test_memory_components_nonnegative_and_kv_per_slot(case) -> None:
    report, slot_count, observed_peak = case

    decomp = decompose(report, slot_count, observed_peak)

    # Each reported component is non-negative (R3.2).
    assert decomp.weights >= 0.0
    assert decomp.kv_total >= 0.0
    assert decomp.kv_per_slot >= 0.0
    assert decomp.scratch_overhead >= 0.0

    # kv_per_slot is exactly the total KV footprint divided by the slot count.
    expected_kv_per_slot = decomp.kv_total / slot_count
    scale = max(1.0, abs(expected_kv_per_slot))
    assert math.isclose(
        decomp.kv_per_slot, expected_kv_per_slot, rel_tol=1e-12, abs_tol=1e-9 * scale
    )

    # A slot count below 1 is rejected.
    with pytest.raises(ValueError):
        decompose(report, slot_count=0, observed_peak=observed_peak)
