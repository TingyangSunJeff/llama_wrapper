"""Property test for ``profile_suite.modules.memory.reconciliation_flag`` (Property 8).

Validates the sum-vs-observed memory reconciliation flag (R3.3): a Config is
flagged if and only if the absolute difference between the summed footprint
components and the directly observed peak exceeds 5 percent of that observed
peak. The check uses a *strict* ``>``, so the exact-5% boundary is NOT flagged.

The single property exercises both accepted input shapes of
``reconciliation_flag`` (a :class:`MemoryDecomposition` and a bare 3-tuple
``(weights, kv_total, scratch_overhead)``) and pins the exact-5% boundary using
peaks chosen so that ``0.05 * peak`` is float-exact.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.modules.memory import (
    RECONCILE_TOLERANCE,
    MemoryDecomposition,
    reconciliation_flag,
)

# A non-negative, finite component value (MiB), magnitude bounded so the
# arithmetic stays numerically well-behaved.
_nonneg = st.floats(
    min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
)


@st.composite
def reconcile_cases(draw):
    """Generate ``(components, observed_peak, boundary_base)``.

    - ``components`` is a triple of non-negative component values
      ``(weights, kv_total, scratch_overhead)`` in MiB.
    - ``observed_peak`` is a positive finite footprint (MiB).
    - ``boundary_base`` is a positive integer used to build a peak whose 5%
      tolerance is float-exact, so the exact-boundary (not-flagged) case can be
      asserted without rounding noise.
    """
    components = (draw(_nonneg), draw(_nonneg), draw(_nonneg))
    observed_peak = draw(
        st.floats(min_value=1e-3, max_value=1e7, allow_nan=False, allow_infinity=False)
    )
    boundary_base = draw(st.integers(min_value=1, max_value=100_000))
    return components, observed_peak, boundary_base


# Feature: profile-suite, Property 8: Memory sum-vs-observed reconciliation flag is correct
@settings(max_examples=100)
@given(case=reconcile_cases())
def test_reconciliation_flag_iff_difference_exceeds_5pct(case) -> None:
    components, observed_peak, boundary_base = case
    weights, kv_total, scratch_overhead = components

    component_sum = weights + kv_total + scratch_overhead
    expected = abs(component_sum - observed_peak) > RECONCILE_TOLERANCE * observed_peak

    # Bare 3-tuple input shape.
    assert reconciliation_flag(components, observed_peak) is expected

    # MemoryDecomposition input shape: reconciliation uses weights + kv_total +
    # scratch_overhead (the per-slot value is intentionally excluded), so the
    # flag must agree with the 3-tuple result for the same three components.
    decomp = MemoryDecomposition(
        weights=weights,
        kv_total=kv_total,
        kv_per_slot=kv_total / 4.0,  # arbitrary; must not affect the flag
        scratch_overhead=scratch_overhead,
    )
    assert reconciliation_flag(decomp, observed_peak) is expected

    # Exact-5% boundary is NOT flagged (strict >). Pick a peak that is a
    # multiple of 20 so tol = 0.05 * peak is an exact integer-valued float, and
    # build component sums exactly tol above and below the peak.
    peak = boundary_base * 20.0
    tol = peak * RECONCILE_TOLERANCE  # == boundary_base, float-exact
    assert reconciliation_flag((peak + tol, 0.0, 0.0), peak) is False
    assert reconciliation_flag((peak - tol, 0.0, 0.0), peak) is False

    # Just beyond the boundary IS flagged.
    assert reconciliation_flag((peak + tol * 1.0001, 0.0, 0.0), peak) is True
