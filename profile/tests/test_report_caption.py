"""Property test for build_caption completeness (profile-suite Property 18)."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.reporting.report import build_caption
from profile_suite.results import EnvironmentCapture, RunManifest


# Identifier-ish text that always carries at least one non-whitespace char so the
# "is the value present in the caption?" check is meaningful (empty/blank values
# are rendered as the literal "unknown" by build_caption and are not asserted on).
_NONEMPTY_TEXT = st.text(
    alphabet=st.characters(
        min_codepoint=33, max_codepoint=126  # printable, no whitespace/control
    ),
    min_size=1,
    max_size=24,
)

_PINNED_BUILDS = st.one_of(st.just("b9418"), _NONEMPTY_TEXT)
_PLATFORMS = st.one_of(
    st.sampled_from(["a100-cuda", "jetson-orin", "A100 GPU"]), _NONEMPTY_TEXT
)
_RUN_REPEATS = st.integers(min_value=1, max_value=64)


def _make_manifest(pinned_build: str, platform: str, run_repeats: int) -> RunManifest:
    """Build a minimal RunManifest carrying the three caption-relevant fields."""
    env = EnvironmentCapture(
        os=None,
        gpu_model=None,
        driver_version=None,
        cuda_version=None,
        python_version=None,
    )
    return RunManifest(
        campaign_name="cap-test",
        pinned_build=pinned_build,
        platform=platform,
        pinned_device="cuda:0",
        environment=env,
        run_repeats=run_repeats,
    )


# Feature: profile-suite, Property 18: Artifact captions are complete
@settings(max_examples=100)
@given(
    pinned_build=_PINNED_BUILDS,
    platform=_PLATFORMS,
    run_repeats=_RUN_REPEATS,
)
def test_caption_is_complete(
    pinned_build: str, platform: str, run_repeats: int
) -> None:
    """Validates: Requirements 7.5

    Every caption build_caption produces must be a single line stating all three
    of the Pinned_Build identifier, the Platform descriptor, and the Run_Repeat
    count. This is checked for both supported input forms:

      - the RunManifest form (fields read from the manifest), and
      - the keyword-override form (fields supplied explicitly), which
        generate_artifacts uses to stamp each per-Platform artifact set.
    """
    repeats_str = str(run_repeats)

    # --- RunManifest input form ---------------------------------------------
    manifest = _make_manifest(pinned_build, platform, run_repeats)
    caption_manifest = build_caption(manifest)

    # Single line: the caption is one line of text (no embedded newlines).
    assert "\n" not in caption_manifest
    # All three required pieces of information are present.
    assert pinned_build in caption_manifest
    assert platform in caption_manifest
    assert repeats_str in caption_manifest
    # The labels naming each field are present too.
    assert "Pinned_Build" in caption_manifest
    assert "Platform" in caption_manifest
    assert "Run_Repeat" in caption_manifest

    # --- Keyword-override form ----------------------------------------------
    # Build a manifest carrying *different* values, then override all three via
    # keywords; the overrides must win and the caption must reflect them.
    base = _make_manifest("OTHER_BUILD", "other-platform", run_repeats + 1)
    caption_override = build_caption(
        base,
        pinned_build=pinned_build,
        platform=platform,
        run_repeats=run_repeats,
    )

    assert "\n" not in caption_override
    assert pinned_build in caption_override
    assert platform in caption_override
    assert repeats_str in caption_override

    # The mapping input form is also supported and equally complete.
    caption_mapping = build_caption(
        {
            "pinned_build": pinned_build,
            "platform": platform,
            "run_repeats": run_repeats,
        }
    )
    assert "\n" not in caption_mapping
    assert pinned_build in caption_mapping
    assert platform in caption_mapping
    assert repeats_str in caption_mapping
