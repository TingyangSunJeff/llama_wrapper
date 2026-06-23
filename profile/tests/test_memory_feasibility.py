"""Property test for feasibility filtering and shortfall (Property 9).

Validates the memory capacity frontier (R3.4, R3.5):

- ``feasible_pairs(peaks, B)`` returns exactly the Configs whose measured peak
  footprint is ``<= B`` — no more, no fewer.
- ``shortfall(peak, available)`` records *both* operands and the derived
  ``shortfall == peak - available``.
- Feasibility is *monotonic* in the budget: a Config infeasible at budget ``B``
  (``peak > B``) is infeasible at every budget ``B' < B``, so the feasible set
  at a smaller budget is a subset of the feasible set at a larger budget.

The single property uses the shared ``budgets()`` strategy (defined in
``conftest.py``) generating a mapping of :class:`Config` -> peak footprint (MiB)
plus a budget ``B``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.config import Config
from profile_suite.modules.memory import feasible_pairs, shortfall

from .conftest import budgets


# Feature: profile-suite, Property 9: Feasibility filtering is correct and monotonic in the budget
@settings(max_examples=100)
@given(case=budgets(), delta=st.floats(min_value=0.0, max_value=80_000.0,
                                       allow_nan=False, allow_infinity=False))
def test_feasibility_filtering_is_correct_and_monotonic(case, delta) -> None:
    peaks, budget = case

    feasible = feasible_pairs(peaks, budget)

    # 1. Correctness: feasible_pairs returns exactly the Configs with peak <= B.
    expected = {config for config, peak in peaks.items() if peak <= budget}
    assert set(feasible) == expected
    # No spurious or duplicated entries; each feasible key appears once and is a
    # genuine key of the input mapping.
    assert len(feasible) == len(expected)
    assert all(config in peaks for config in feasible)
    assert all(peaks[config] <= budget for config in feasible)
    # Every infeasible Config is correctly excluded.
    assert all(
        peaks[config] > budget
        for config in peaks
        if config not in set(feasible)
    )

    # 2. shortfall records both operands and shortfall == peak - available.
    for config, peak in peaks.items():
        short = shortfall(peak, budget)
        assert short.peak == peak
        assert short.available == budget
        assert short.shortfall == peak - budget
        # The infeasible flag is consistent with the strict peak > budget split:
        # a Config is infeasible (shortfall > 0) exactly when it is excluded
        # from the feasible set under a budget equal to ``available``.
        assert short.infeasible == (peak > budget)
        assert short.infeasible == (config not in set(feasible))

    # 3. Monotonicity: for B' < B, the feasible set at B' is a subset of the
    # feasible set at B (a Config infeasible at B stays infeasible at every
    # smaller B'). Build a strictly smaller budget B' = B - delta - epsilon.
    lower = budget - delta
    feasible_lower = set(feasible_pairs(peaks, lower))
    assert feasible_lower <= set(feasible)
    # Conversely, a Config infeasible at the larger budget B is infeasible at B'.
    infeasible_at_budget = {c for c in peaks if c not in set(feasible)}
    assert all(c not in feasible_lower for c in infeasible_at_budget)
