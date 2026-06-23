"""Property test for Config.change_type (profile-suite Property 2)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings

from profile_suite.config import Config
from tests.conftest import config_pairs


# Feature: profile-suite, Property 2: Change-type labeling is exhaustive and correct
@settings(max_examples=200)
@given(config_pairs())
def test_change_type_is_exhaustive_and_correct(pair: tuple[Config, Config]) -> None:
    """Validates: Requirements 1.3, 1.4, 1.5

    For any ordered pair (from, to):
      - change_type == "slot-reshape" iff only ctx_length and/or slot_count differ
      - change_type == "model-reload" iff only quant_file differs
      - change_type == "combined"     iff quant_file differs AND at least one of
        ctx_length/slot_count differs
      - identical configs raise ValueError
    """
    frm, to = pair

    model_changed = frm.quant_file != to.quant_file
    shape_changed = (
        frm.ctx_length != to.ctx_length or frm.slot_count != to.slot_count
    )

    if not model_changed and not shape_changed:
        # Identical configs: there is no change to classify.
        with pytest.raises(ValueError):
            frm.change_type(to)
        return

    label = frm.change_type(to)

    if model_changed and shape_changed:
        assert label == "combined"
    elif model_changed:
        assert label == "model-reload"
    else:  # only shape changed
        assert label == "slot-reshape"

    # Exhaustiveness: the label is always exactly one of the three valid values.
    assert label in {"slot-reshape", "model-reload", "combined"}

    # Biconditional checks (iff) stated explicitly per the property.
    assert (label == "slot-reshape") == (shape_changed and not model_changed)
    assert (label == "model-reload") == (model_changed and not shape_changed)
    assert (label == "combined") == (model_changed and shape_changed)
