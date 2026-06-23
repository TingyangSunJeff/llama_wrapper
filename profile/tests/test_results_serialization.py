"""Property test for result/manifest serialization round-trip (Property 12).

Validates that every data model in :mod:`profile_suite.results` (and the
:class:`~profile_suite.config.Config` they reference) survives a full JSON
round-trip with value equality, and that every :class:`MeasuredValue` carries the
report-facing fields required for record completeness (R7.1):

    Model.from_dict(json.loads(json.dumps(m.to_dict()))) == m

The strategies below build arbitrary instances of every model, with extra care
for the two records the property singles out -- :class:`RunManifest` and
:class:`MeasuredValue` -- and they exercise the explicit ``"unavailable"``
sentinel inside :class:`EnvironmentCapture` (R5.8).
"""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.config import Config
from profile_suite.results import (
    UNAVAILABLE,
    Aggregate,
    EnvironmentCapture,
    MeasuredValue,
    MeasurementPoint,
    ModelRef,
    RunManifest,
    RunRepeatResult,
    config_to_dict,
)

# --------------------------------------------------------------------------- #
# Primitive strategies
# --------------------------------------------------------------------------- #
# Finite floats only: NaN/inf would break value-equality after a JSON round-trip
# (NaN != NaN), which is an artifact of IEEE comparison, not of serialization.
_finite_floats = st.floats(
    min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False
)
# Short text kept JSON-safe (Hypothesis's default text alphabet excludes lone
# surrogates, so it is always utf-8 encodable).
_text = st.text(max_size=24)
_pos_int = st.integers(min_value=1, max_value=1_000_000)


# JSON-safe scalar values for free-form dicts (axis / config payloads). Each value
# round-trips through json.dumps/json.loads to an equal Python value.
_json_scalar = st.one_of(
    st.none(), st.booleans(), st.integers(min_value=-10_000, max_value=10_000),
    _finite_floats, _text,
)
_json_dict = st.dictionaries(_text, _json_scalar, max_size=4)


# An environment field: a normal string, ``None``, or the explicit "unavailable"
# sentinel that the harness records when a field cannot be determined (R5.8).
_env_field = st.one_of(_text, st.none(), st.just(UNAVAILABLE))


# --------------------------------------------------------------------------- #
# Model strategies
# --------------------------------------------------------------------------- #
def configs() -> st.SearchStrategy[Config]:
    return st.builds(
        Config, quant_file=_text, ctx_length=_pos_int, slot_count=_pos_int
    )


def aggregates() -> st.SearchStrategy[Aggregate]:
    return st.builds(
        Aggregate,
        mean=_finite_floats,
        std=_finite_floats,
        n_success=st.integers(min_value=0, max_value=20),
        insufficient=st.booleans(),
    )


def model_refs() -> st.SearchStrategy[ModelRef]:
    return st.builds(ModelRef, abs_path=_text, sha256=_text)


def environment_captures() -> st.SearchStrategy[EnvironmentCapture]:
    return st.builds(
        EnvironmentCapture,
        os=_env_field,
        gpu_model=_env_field,
        driver_version=_env_field,
        cuda_version=_env_field,
        python_version=_env_field,
        models=st.dictionaries(_text, model_refs(), max_size=3),
    )


def run_repeat_results() -> st.SearchStrategy[RunRepeatResult]:
    return st.builds(
        RunRepeatResult,
        run_index=st.integers(min_value=0, max_value=20),
        discarded_warmup=st.booleans(),
        ok=st.booleans(),
        raw_log_path=_text,
        metrics=st.dictionaries(_text, _finite_floats, max_size=4),
        error=st.one_of(st.none(), _text),
    )


def measurement_points() -> st.SearchStrategy[MeasurementPoint]:
    return st.builds(
        MeasurementPoint,
        point_id=_text,
        module=_text,
        config=configs(),
        axis=_json_dict,
        repeats=st.lists(run_repeat_results(), max_size=4),
        aggregates=st.dictionaries(_text, aggregates(), max_size=3),
        status=st.sampled_from(
            ["complete", "incomplete", "failed", "platform-infeasible"]
        ),
        reason=st.one_of(st.none(), _text),
    )


def measured_values() -> st.SearchStrategy[MeasuredValue]:
    # Source Config is serialized into the dict-shaped ``config`` field, mirroring
    # how the Reporting_Module records the source Config (R7.1).
    return st.builds(
        MeasuredValue,
        metric=_text,
        unit=_text,
        mean=_finite_floats,
        std=_finite_floats,
        n_success=st.integers(min_value=0, max_value=20),
        config=configs().map(config_to_dict),
        platform=_text,
        manifest_ref=_text,
        flags=st.lists(_text, max_size=4),
        axis=_json_dict,
    )


def run_manifests() -> st.SearchStrategy[RunManifest]:
    return st.builds(
        RunManifest,
        campaign_name=_text,
        pinned_build=_text,
        platform=_text,
        pinned_device=_text,
        environment=environment_captures(),
        config_grid=st.lists(configs(), max_size=4),
        run_repeats=st.integers(min_value=1, max_value=10),
        raw_log_paths=st.dictionaries(_text, st.lists(_text, max_size=3), max_size=3),
        decode_batch_sizes=st.lists(st.integers(min_value=1, max_value=128), max_size=8),
        enabled_modules=st.lists(_text, max_size=6),
        noise_sensitive_metrics=st.lists(_text, max_size=4),
    )


@st.composite
def model_bundle(draw):
    """One arbitrary instance of every serializable model, generated together.

    Bundling them keeps Property 12 a single test while still round-tripping each
    model (the property singles out RunManifest and MeasuredValue, which are the
    last two)."""
    return (
        draw(configs()),
        draw(aggregates()),
        draw(model_refs()),
        draw(environment_captures()),
        draw(run_repeat_results()),
        draw(measurement_points()),
        draw(measured_values()),
        draw(run_manifests()),
    )


# Maps each model class to the module-level (to_dict, from_dict) used to round-trip
# it. ``Config`` uses the module-level helper pair rather than instance methods.
def _roundtrip_config(c: Config) -> Config:
    from profile_suite.results import config_from_dict

    return config_from_dict(json.loads(json.dumps(config_to_dict(c))))


def _roundtrip(model) -> object:
    """Full JSON round-trip via the model's own to_dict/from_dict."""
    restored_dict = json.loads(json.dumps(model.to_dict()))
    return type(model).from_dict(restored_dict)


# Feature: profile-suite, Property 12: Manifest and results records are serialization-complete and round-trip
@settings(max_examples=100)
@given(bundle=model_bundle())
def test_records_round_trip_and_measured_value_is_complete(bundle) -> None:
    """Validates: Requirements 5.5, 5.6, 7.1

    For any campaign inputs:
      - every record (Config, Aggregate, ModelRef, EnvironmentCapture,
        RunRepeatResult, MeasurementPoint, MeasuredValue, RunManifest) satisfies
        ``from_dict(json.loads(json.dumps(m.to_dict()))) == m`` (full JSON
        round-trip equality);
      - the RunManifest preserves Pinned_Build, Platform, pinned device, the
        environment capture, the swept Config_Grid, the Run_Repeat count, and the
        raw-log paths (R5.5, R5.6);
      - every MeasuredValue carries metric, unit, mean, std, source Config,
        Platform, and a Run_Manifest reference (record completeness, R7.1).
    """
    config, agg, ref, env, repeat, point, mv, manifest = bundle

    # --- Full JSON round-trip equality for every model (R5.5, R5.6, 7.1) ------
    assert _roundtrip_config(config) == config
    assert _roundtrip(agg) == agg
    assert _roundtrip(ref) == ref
    assert _roundtrip(env) == env
    assert _roundtrip(repeat) == repeat
    assert _roundtrip(point) == point
    assert _roundtrip(mv) == mv
    assert _roundtrip(manifest) == manifest

    # --- RunManifest preserves the linked reproducibility fields (R5.6) -------
    restored_manifest = _roundtrip(manifest)
    assert restored_manifest.pinned_build == manifest.pinned_build
    assert restored_manifest.platform == manifest.platform
    assert restored_manifest.pinned_device == manifest.pinned_device
    assert restored_manifest.environment == manifest.environment
    assert restored_manifest.config_grid == manifest.config_grid
    assert restored_manifest.run_repeats == manifest.run_repeats
    assert restored_manifest.raw_log_paths == manifest.raw_log_paths

    # --- MeasuredValue record completeness (R7.1) -----------------------------
    payload = json.loads(json.dumps(mv.to_dict()))
    for required_field in (
        "metric", "unit", "mean", "std", "config", "platform", "manifest_ref"
    ):
        assert required_field in payload, f"MeasuredValue missing {required_field!r}"

    # The completeness fields carry their declared types/sources.
    assert isinstance(payload["metric"], str)
    assert isinstance(payload["unit"], str)
    assert isinstance(payload["mean"], float)
    assert isinstance(payload["std"], float)
    assert isinstance(payload["platform"], str)
    assert isinstance(payload["manifest_ref"], str)
    # Source Config is recorded as its serialized {quant_file, ctx_length,
    # slot_count} mapping.
    assert set(payload["config"]) == {"quant_file", "ctx_length", "slot_count"}
