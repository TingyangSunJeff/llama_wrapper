"""Property test for C_switch phase-timing reconciliation.

Covers design Property 3 (Requirements 1.1, 1.2): the pure
``reconcile_phases`` logic in ``profile_suite.modules.switch_cost`` must turn a
sequence of monotonic phase-boundary timestamps into Teardown / Boot / Warmup /
C_switch components that reconcile to the total within 50 ms.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings

from profile_suite.modules.switch_cost import reconcile_phases

from tests.conftest import monotonic_timestamps

# Nanoseconds per millisecond (phases are reported in ms; timestamps are ns).
_NS_PER_MS = 1_000_000.0

# Reconciliation tolerance from Requirement 1.2.
_TOLERANCE_MS = 50.0


# Feature: profile-suite, Property 3: C_switch components reconcile to the total within 50 ms
@settings(max_examples=100)
@given(ts=monotonic_timestamps())
def test_c_switch_components_reconcile_within_50ms(
    ts: tuple[int, int, int, int],
) -> None:
    """Validates: Requirements 1.1, 1.2.

    For any monotonic ``t0 <= t1 <= t2 <= t3`` (ns) on one clock, the reported
    Teardown = t1-t0, Boot = t2-t1, Warmup = t3-t2 (in ms) hold exactly, and
    ``|C_switch - (Teardown + Boot + Warmup)| <= 50 ms``. Non-monotonic
    timestamps must raise ``ValueError``.
    """
    t0, t1, t2, t3 = ts

    timing = reconcile_phases(t0, t1, t2, t3)

    # Each component is exactly the corresponding inter-boundary gap, in ms.
    assert timing.teardown == pytest.approx((t1 - t0) / _NS_PER_MS)
    assert timing.boot == pytest.approx((t2 - t1) / _NS_PER_MS)
    assert timing.warmup == pytest.approx((t3 - t2) / _NS_PER_MS)
    assert timing.c_switch == pytest.approx((t3 - t0) / _NS_PER_MS)

    # The components reconcile to the total within the 50 ms tolerance (R1.2).
    component_sum = timing.teardown + timing.boot + timing.warmup
    assert abs(timing.c_switch - component_sum) <= _TOLERANCE_MS

    # Non-monotonic timestamps are rejected: (t1+1) > t1 always violates t0 <= t1.
    with pytest.raises(ValueError):
        reconcile_phases(t1 + 1, t1, t2, t3)
