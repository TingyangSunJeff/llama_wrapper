"""Property test for ``profile_suite.harness.stats.aggregate`` (Property 4).

Validates that aggregation is computed only over the successful repeats, that the
reported ``n_success`` equals the count of successes, and that the
``insufficient`` flag is set iff fewer than 5 successes were observed.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.harness.stats import MIN_SUCCESS, aggregate

# A successful repeat is a finite float; a failed repeat is None or NaN. We bound the
# float magnitude so the independent reference mean stays numerically well-behaved
# (no inf/overflow), which keeps the equality check meaningful.
_successes = st.floats(
    min_value=-1e9,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
)
_failures = st.sampled_from([None, float("nan")])


def repeat_outcomes() -> st.SearchStrategy[list]:
    """Lists interleaving successful float values and failures (None / NaN)."""
    return st.lists(st.one_of(_successes, _failures), min_size=0, max_size=40)


# Feature: profile-suite, Property 4: Aggregates are computed only over successful repeats, with an insufficiency flag
@settings(max_examples=100)
@given(values=repeat_outcomes())
def test_aggregate_over_successful_repeats(values: list) -> None:
    # Independent recomputation of the successful values (the failures are None/NaN).
    successes = [
        float(v) for v in values if v is not None and not math.isnan(float(v))
    ]
    expected_n = len(successes)

    agg = aggregate(values)

    # n_success equals the count of successes.
    assert agg.n_success == expected_n

    # insufficient flag set iff n_success < 5.
    assert agg.insufficient == (expected_n < MIN_SUCCESS)
    assert MIN_SUCCESS == 5

    if expected_n == 0:
        # No successful values: mean and std are NaN by definition.
        assert math.isnan(agg.mean)
        assert math.isnan(agg.std)
        return

    # Mean is computed using exactly the successful values; compare against an
    # independent computation over those successes.
    expected_mean = sum(successes) / expected_n
    scale = max(1.0, abs(expected_mean))
    assert math.isclose(agg.mean, expected_mean, rel_tol=1e-9, abs_tol=1e-6 * scale)

    # Std uses exactly the successful values: sample std (ddof=1), 0.0 for a single
    # success.
    if expected_n < 2:
        assert agg.std == 0.0
    else:
        variance = sum((x - expected_mean) ** 2 for x in successes) / (expected_n - 1)
        expected_std = math.sqrt(variance)
        std_scale = max(1.0, abs(expected_std))
        assert math.isclose(
            agg.std, expected_std, rel_tol=1e-9, abs_tol=1e-6 * std_scale
        )
