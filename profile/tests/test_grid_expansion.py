"""Property test for expand_grid / GridSpec (profile-suite Property 1)."""

from __future__ import annotations

from itertools import product

from hypothesis import given, settings

from profile_suite.config import Config, GridSpec, expand_grid
from tests.conftest import grids


# Feature: profile-suite, Property 1: Campaign-grid expansion is deterministic and complete
@settings(max_examples=100)
@given(grids())
def test_grid_expansion_is_deterministic_and_complete(spec: GridSpec) -> None:
    """Validates: Requirements 1.6, 2.1, 8.2

    For any GridSpec, expand_grid produces exactly the cross-product of the
    distinct values of each axis, with no duplicates, in a stable/deterministic
    order across repeated expansions of the same input. The count equals the
    product of the distinct-axis sizes, and every produced Config's fields come
    from the respective axes.
    """
    grid = expand_grid(spec)

    # Distinct values per axis, preserving first-seen order.
    distinct_quants = list(dict.fromkeys(spec.quant_files))
    distinct_ctxs = list(dict.fromkeys(spec.ctx_lengths))
    distinct_slots = list(dict.fromkeys(spec.slot_counts))

    # No duplicates: every produced Config is unique.
    assert len(grid) == len(set(grid))

    # Count equals the product of distinct-axis sizes.
    expected_count = len(distinct_quants) * len(distinct_ctxs) * len(distinct_slots)
    assert len(grid) == expected_count

    # Completeness: the produced set is exactly the cross-product of distinct axes.
    expected_set = {
        Config(quant_file=q, ctx_length=c, slot_count=s)
        for q, c, s in product(distinct_quants, distinct_ctxs, distinct_slots)
    }
    assert set(grid) == expected_set

    # Every produced Config's fields come from the respective axes.
    for cfg in grid:
        assert cfg.quant_file in distinct_quants
        assert cfg.ctx_length in distinct_ctxs
        assert cfg.slot_count in distinct_slots

    # Stable, deterministic order: the exact list is reproduced (also matching
    # the nested-iteration cross-product order quant -> ctx -> slot).
    assert expand_grid(spec) == grid
    expected_order = [
        Config(quant_file=q, ctx_length=c, slot_count=s)
        for q in distinct_quants
        for c in distinct_ctxs
        for s in distinct_slots
    ]
    assert grid == expected_order
