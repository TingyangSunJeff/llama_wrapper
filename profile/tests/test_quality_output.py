"""Property test for ``profile_suite.modules.quality.quality_values`` (Property 10).

Validates that the Quality_Module emits exactly one value per distinct quant
format, that each emitted value is labeled either ``"cited"`` or
``"locally-measured"`` matching the selected source, and that a format absent
from the cited table (when local measurement is disabled) is marked ``missing``
without dropping the remaining formats.

Validates: Requirements 4.1, 4.2, 4.5, 4.6
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.modules.quality import (
    CITED_QUALITY_TABLE,
    LocalSurrogate,
    quality_values,
)

# A pool of quant formats mixing known cited-table formats and unknown ones, so
# generated lists exercise both the present-in-table and absent-from-table
# (missing) branches of the cited path.
_KNOWN_FORMATS = sorted(CITED_QUALITY_TABLE.keys())
_UNKNOWN_FORMATS = ["Q4_K_XL", "IQ2_XXS", "BF16", "MX4", "custom-quant"]
_ALL_FORMATS = _KNOWN_FORMATS + _UNKNOWN_FORMATS


def formats_lists() -> st.SearchStrategy[list[str]]:
    """Lists of quant formats (with possible duplicates) drawn from the pool."""
    return st.lists(st.sampled_from(_ALL_FORMATS), min_size=0, max_size=12)


def surrogates() -> st.SearchStrategy[LocalSurrogate]:
    """Locally measured surrogates with a mix of sufficient/insufficient repeats."""
    return st.builds(
        LocalSurrogate,
        mean=st.floats(
            min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        n_success=st.integers(min_value=0, max_value=10),
        std=st.one_of(
            st.none(),
            st.floats(
                min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
        ),
    )


# Feature: profile-suite, Property 10: Quality output is one labeled value per format with missing-handling
@settings(max_examples=100)
@given(
    formats=formats_lists(),
    local_quality=st.booleans(),
    data=st.data(),
)
def test_quality_output_one_labeled_value_per_format(
    formats: list[str], local_quality: bool, data: st.DataObject
) -> None:
    distinct = list(dict.fromkeys(formats))  # first-seen order, de-duplicated

    # When local measurement is enabled, build a surrogate map covering a random
    # subset of the formats (some may be missing a surrogate -> marked missing).
    local_surrogates: dict[str, LocalSurrogate] | None = None
    if local_quality:
        local_surrogates = {}
        for fmt in distinct:
            if data.draw(st.booleans()):
                local_surrogates[fmt] = data.draw(surrogates())

    results = quality_values(formats, local_quality, local_surrogates)

    # Exactly one emitted value per distinct format, in first-seen order (R4.1).
    assert [qv.quant_format for qv in results] == distinct
    assert len(results) == len(distinct)

    expected_label = "locally-measured" if local_quality else "cited"
    for qv in results:
        # Each value is labeled to match the selected source (R4.5).
        assert qv.label == expected_label

        if not local_quality:
            # Cited path: a format in the table carries its numeric value and is
            # not missing; a format absent from the table is marked missing
            # without aborting the rest (R4.2, R4.6).
            if qv.quant_format in CITED_QUALITY_TABLE:
                assert qv.missing is False
                assert qv.value == CITED_QUALITY_TABLE[qv.quant_format]
            else:
                assert qv.missing is True
                assert qv.value is None
        else:
            # Local path: a format with a surrogate carries that surrogate's mean
            # and is not missing; a format without a surrogate is marked missing.
            surrogate = (local_surrogates or {}).get(qv.quant_format)
            if surrogate is None:
                assert qv.missing is True
                assert qv.value is None
            else:
                assert qv.missing is False
                assert qv.value == surrogate.mean

    # Missing formats never cause other formats to be dropped: every distinct
    # format is still present regardless of how many were marked missing (R4.6).
    assert {qv.quant_format for qv in results} == set(distinct)
