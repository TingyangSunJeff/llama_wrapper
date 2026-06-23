"""Property test for per-Platform tagging and no-mixing (Property 13).

Validates that the Reporting_Module keeps Platforms strictly separated:

    For any set of results spanning one or more Platforms, every result record
    and the Run_Manifest carry the campaign's Platform descriptor, and each
    emitted artifact (figure, table, results file) draws its values from exactly
    one Platform.

The suite runs one Platform per campaign (R6.1), but the Reporting_Module exposes
a ``"platform"`` measurement-axis override (``PLATFORM_AXIS_KEY``) so a results
store that spans more than one Platform can be partitioned and separated. This
test drives that override directly: it generates measurement points that mix the
``a100-cuda`` and ``jetson-orin`` Platforms (some points carry no override and
fall back to the manifest's default Platform) and asserts:

  1. ``group_points_by_platform`` partitions the points so each group contains
     only that Platform's points and no point is dropped or duplicated.
  2. ``build_results`` over a single Platform's points tags every emitted value
     (and the artifact header) with exactly that one Platform.
  3. ``generate_artifacts`` writes a separate per-Platform results.json under a
     tmp dir whose every value carries exactly that one Platform descriptor --
     i.e. no emitted artifact mixes Platforms.

Validates: Requirements 6.5, 7.6
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.config import Config
from profile_suite.reporting.report import (
    build_results,
    generate_artifacts,
    group_points_by_platform,
    point_platform,
)
from profile_suite.results import (
    Aggregate,
    EnvironmentCapture,
    MeasurementPoint,
    RunManifest,
)

# The two Platforms the suite targets (design: A100 CUDA now, Jetson later).
_PLATFORMS = ["a100-cuda", "jetson-orin"]

# Canonical decode-batch sweep values (powers of two) for performance points.
_DECODE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]

# Finite, non-negative measurement values so aggregates are well-formed.
_metric_mean = st.floats(
    min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False
)
_metric_std = st.floats(
    min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False
)


def _aggregate(draw) -> Aggregate:
    """Build one successful Aggregate (n_success >= 1 so it yields a value)."""
    n = draw(st.integers(min_value=1, max_value=10))
    return Aggregate(
        mean=draw(_metric_mean),
        std=draw(_metric_std),
        n_success=n,
        insufficient=n < 5,
    )


@st.composite
def configs(draw) -> Config:
    """A small-pool Config so distinct points frequently share Configs."""
    return Config(
        quant_file=draw(st.sampled_from(["q4_k_m.gguf", "q8_0.gguf", "f16.gguf"])),
        ctx_length=draw(st.sampled_from([2048, 4096, 32768])),
        slot_count=draw(st.sampled_from([1, 2, 4])),
    )


@st.composite
def measurement_points(draw) -> MeasurementPoint:
    """A MeasurementPoint that may carry a ``platform`` axis override.

    The point is one of the three artifact-feeding shapes (performance /
    switch_cost / memory) so ``generate_artifacts`` exercises the figure and both
    tables. A drawn ``platform`` of ``None`` means "no override" -- the point
    falls back to the manifest's default Platform; otherwise the override assigns
    the point to that Platform (mixing a100-cuda / jetson-orin in one store).
    """
    override = draw(st.sampled_from([None] + _PLATFORMS))
    config = draw(configs())
    kind = draw(st.sampled_from(["performance", "switch_cost", "memory"]))

    axis: dict = {}
    if override is not None:
        axis["platform"] = override

    aggregates: dict[str, Aggregate] = {}
    if kind == "performance":
        axis["decode_batch"] = draw(st.sampled_from(_DECODE_BATCHES))
        aggregates["decode_throughput"] = _aggregate(draw)
        module = "performance"
    elif kind == "switch_cost":
        axis["change_type"] = draw(
            st.sampled_from(["slot-reshape", "model-reload", "combined"])
        )
        for metric in ("teardown_ms", "boot_ms", "warmup_ms", "c_switch_ms"):
            aggregates[metric] = _aggregate(draw)
        module = "switch_cost"
    else:
        for metric in ("weights", "kv_per_slot", "scratch_overhead"):
            aggregates[metric] = _aggregate(draw)
        module = "memory"

    point_id = f"p{draw(st.integers(min_value=0, max_value=10_000))}"
    return MeasurementPoint(
        point_id=point_id,
        module=module,
        config=config,
        axis=axis,
        repeats=[],
        aggregates=aggregates,
        status="complete",
        reason=None,
    )


def _manifest(default_platform: str, pinned_build: str, run_repeats: int) -> RunManifest:
    """A minimal RunManifest whose single Platform is ``default_platform``."""
    return RunManifest(
        campaign_name="platform-separation-campaign",
        pinned_build=pinned_build,
        platform=default_platform,
        pinned_device="gpu0",
        environment=EnvironmentCapture(
            os="linux",
            gpu_model="A100",
            driver_version="550.0",
            cuda_version="12.4",
            python_version="3.11",
            models={},
        ),
        config_grid=[],
        run_repeats=run_repeats,
        raw_log_paths={},
        decode_batch_sizes=_DECODE_BATCHES,
        enabled_modules=[],
        noise_sensitive_metrics=[],
    )


# Feature: profile-suite, Property 13: Every persisted record is tagged with exactly one Platform and artifacts never mix Platforms
@settings(max_examples=100, deadline=None)
@given(
    points=st.lists(measurement_points(), max_size=12),
    default_platform=st.sampled_from(_PLATFORMS),
    pinned_build=st.sampled_from(["b9418", "b9999"]),
    run_repeats=st.integers(min_value=1, max_value=10),
)
def test_per_platform_tagging_and_no_mixing(
    points, default_platform, pinned_build, run_repeats
) -> None:
    """Validates: Requirements 6.5, 7.6

    For any set of results spanning one or more Platforms:
      - grouping partitions the points so each group holds only that Platform's
        points (no point lost or duplicated);
      - build_results over one Platform's points tags every value and the header
        with exactly that Platform;
      - generate_artifacts writes a per-Platform results.json whose every value
        carries exactly that one Platform descriptor (no artifact mixes Platforms).
    """
    manifest = _manifest(default_platform, pinned_build, run_repeats)

    # --- 1. group_points_by_platform partitions cleanly (R6.5) ---------------
    groups = group_points_by_platform(points, default_platform)

    # Every group holds only points belonging to that Platform.
    grouped_total = 0
    for platform, group in groups.items():
        grouped_total += len(group)
        for point in group:
            assert point_platform(point, default_platform) == platform

    # The partition is exhaustive and lossless (no point dropped or duplicated).
    assert grouped_total == len(points)
    expected_platforms = {point_platform(p, default_platform) for p in points}
    assert set(groups.keys()) == expected_platforms

    # --- 2. build_results tags one Platform per artifact (R6.5, R7.6) --------
    for platform, group in groups.items():
        plat_manifest = replace(manifest, platform=platform)
        artifact = build_results(group, plat_manifest)
        assert artifact["platform"] == platform
        for value in artifact["values"]:
            assert value["platform"] == platform

    # --- 3. generate_artifacts: per-Platform results never mix (R7.6) --------
    with tempfile.TemporaryDirectory() as tmp_dir:
        written = generate_artifacts(points, manifest, tmp_dir)

        # One artifact set per distinct Platform, and only those Platforms.
        assert set(written.keys()) == expected_platforms

        for platform, artifacts in written.items():
            results_path = artifacts["results"]
            with open(results_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # The results file is tagged with its single source Platform...
            assert data["platform"] == platform
            # ...and every value it carries belongs to exactly that Platform.
            for value in data["values"]:
                assert value["platform"] == platform
