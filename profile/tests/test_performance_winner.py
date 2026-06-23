"""Property test for ``profile_suite.modules.performance.batch_winner`` (Property 6).

Validates that, at one Decode_Batch_Size value, the recorded winner is a quant
format achieving the maximum mean Decode_Throughput, and that the result is
marked a tie if and only if at least two formats were compared and the gap
between the top-two means is within one standard deviation of the winner (R2.4).
A single-format mapping is never a tie; an empty mapping raises ``ValueError``.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.modules.performance import batch_winner

# A quant-format aggregate at one batch value: (mean, std) Decode_Throughput in
# tok/s. Means are bounded finite floats; std is non-negative and finite (the
# tie threshold), as the aggregator guarantees.
_means = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)
_stds = st.floats(
    min_value=0.0,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)
_format_names = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_.", min_size=1, max_size=12
)


def aggregate_maps() -> st.SearchStrategy[dict]:
    """Non-empty mappings of quant-format name -> ``(mean, std)`` aggregates."""
    return st.dictionaries(
        keys=_format_names,
        values=st.tuples(_means, _stds),
        min_size=1,
        max_size=6,
    )


# Feature: profile-suite, Property 6: Per-batch quant winner and tie detection are correct
@settings(max_examples=100)
@given(aggregates=aggregate_maps())
def test_batch_winner_and_tie(aggregates: dict) -> None:
    result = batch_winner(aggregates)

    # Independently reproduce the deterministic ordering: descending mean, then
    # ascending format name to break mean ties.
    ordered = sorted(aggregates.items(), key=lambda item: (-item[1][0], item[0]))
    top_format, (top_mean, top_std) = ordered[0]

    # The winner achieves the maximum mean.
    assert result.winner in aggregates
    assert aggregates[result.winner][0] == top_mean
    max_mean = max(mean for mean, _ in aggregates.values())
    assert top_mean == max_mean

    if len(aggregates) == 1:
        # Single format: never a tie.
        assert result.tie is False
        return

    # At least two formats: tie iff (top_mean - second_mean) <= winner sigma.
    second_mean = ordered[1][1][0]
    margin = top_mean - second_mean
    sigma = top_std if top_std > 0.0 else 0.0
    expected_tie = margin <= sigma

    assert result.tie == expected_tie
    assert result.margin == margin
    assert result.sigma == sigma


# Feature: profile-suite, Property 6: Per-batch quant winner and tie detection are correct
def test_batch_winner_empty_raises() -> None:
    with pytest.raises(ValueError):
        batch_winner({})
