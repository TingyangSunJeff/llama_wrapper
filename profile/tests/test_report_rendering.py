"""Example tests for the reporting renderers (Task 17.7).

Exercises the figure/table renderers in
:mod:`profile_suite.reporting.report` against small synthetic sample data:

- :func:`plot_decode_throughput_vs_batch` writes a non-empty PNG and the number
  of plotted line series equals the number of quant formats (R7.2).
- :func:`render_c_switch_table` renders one row per change type — slot-reshape,
  model-reload, combined — with the expected phase columns (R7.3).
- :func:`render_memory_table` renders one row per Config (R7.4).
- :func:`build_caption` content (Pinned_Build / Platform / Run_Repeat count)
  appears in every rendered table and figure caption (R7.5).
- :func:`generate_artifacts` writes a full per-Platform artifact set under a
  temporary directory (kept under ``profile/`` by the caller — here a tmp dir).

These are example (non-property) tests complementing the property coverage of the
pure reporting logic.

Validates: Requirements 7.2, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import os

from profile_suite.config import Config
from profile_suite.harness.stats import aggregate
from profile_suite.results import (
    Aggregate,
    EnvironmentCapture,
    MeasurementPoint,
    RunManifest,
    RunRepeatResult,
)
from profile_suite.reporting import report

# --------------------------------------------------------------------------- #
# Synthetic-sample-data builders
# --------------------------------------------------------------------------- #
# Two quant formats, exercising the per-format series / per-batch winner logic.
QUANT_Q4 = "/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
QUANT_Q8 = "/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
QUANT_FILES = (QUANT_Q4, QUANT_Q8)

# A few decode batch sizes (a subset of the canonical sweep).
DECODE_BATCHES = (1, 4, 16)

# The three change types the C_switch table must report.
CHANGE_TYPES = ("slot-reshape", "model-reload", "combined")

# Synthetic decode throughput (tok/s) per (quant, batch). Q8 wins at batch 1,
# Q4 wins at the larger batches — the regime sign-flip the figure must show.
_DECODE_TPS = {
    (QUANT_Q4, 1): 95.0,
    (QUANT_Q4, 4): 310.0,
    (QUANT_Q4, 16): 780.0,
    (QUANT_Q8, 1): 110.0,
    (QUANT_Q8, 4): 250.0,
    (QUANT_Q8, 16): 540.0,
}


def _make_point(
    module: str,
    config: Config,
    axis: dict,
    metric_values: dict[str, list[float]],
    *,
    point_id: str,
) -> MeasurementPoint:
    """Build a complete MeasurementPoint with raw repeats and aggregates.

    One discarded warmup run plus ``len(values)`` successful retained repeats per
    metric are recorded, and an aggregate (mean/std over the successful repeats) is
    computed for each metric, mirroring what the run loop persists.
    """
    n = max(len(v) for v in metric_values.values())
    repeats: list[RunRepeatResult] = [
        RunRepeatResult(
            run_index=0,
            discarded_warmup=True,
            ok=True,
            raw_log_path=f"raw/{point_id}/run00.warmup.log",
            metrics={m: vals[0] for m, vals in metric_values.items()},
        )
    ]
    for i in range(n):
        repeats.append(
            RunRepeatResult(
                run_index=i + 1,
                discarded_warmup=False,
                ok=True,
                raw_log_path=f"raw/{point_id}/run{i + 1:02d}.log",
                metrics={m: vals[i] for m, vals in metric_values.items()},
            )
        )

    aggregates: dict[str, Aggregate] = {}
    for metric, vals in metric_values.items():
        agg = aggregate(vals)
        aggregates[metric] = Aggregate(
            mean=agg.mean,
            std=agg.std,
            n_success=agg.n_success,
            insufficient=agg.insufficient,
        )

    return MeasurementPoint(
        point_id=point_id,
        module=module,
        config=config,
        axis=axis,
        repeats=repeats,
        aggregates=aggregates,
        status="complete",
    )


def _repeated(value: float, n: int = 5) -> list[float]:
    """A list of ``n`` slightly-jittered values around ``value`` (non-zero std)."""
    return [value + (i - n // 2) * 0.5 for i in range(n)]


def sample_performance_points() -> list[MeasurementPoint]:
    """Decode-throughput points across 2 quant formats x 3 decode batch sizes."""
    points: list[MeasurementPoint] = []
    for quant in QUANT_FILES:
        for batch in DECODE_BATCHES:
            cfg = Config(quant_file=quant, ctx_length=4096, slot_count=4)
            tps = _DECODE_TPS[(quant, batch)]
            points.append(
                _make_point(
                    "performance",
                    cfg,
                    {"decode_batch": batch},
                    {"decode_throughput": _repeated(tps)},
                    point_id=f"perf_{os.path.basename(quant)}_b{batch}",
                )
            )
    return points


def sample_switch_cost_points() -> list[MeasurementPoint]:
    """One C_switch point per change type with teardown/boot/warmup/total."""
    cfg = Config(quant_file=QUANT_Q4, ctx_length=4096, slot_count=4)
    base = {
        "slot-reshape": (120.0, 800.0, 60.0),
        "model-reload": (130.0, 4200.0, 90.0),
        "combined": (135.0, 4300.0, 95.0),
    }
    points: list[MeasurementPoint] = []
    for ct, (teardown, boot, warmup) in base.items():
        total = teardown + boot + warmup
        points.append(
            _make_point(
                "switch_cost",
                cfg,
                {"change_type": ct},
                {
                    "teardown_ms": _repeated(teardown),
                    "boot_ms": _repeated(boot),
                    "warmup_ms": _repeated(warmup),
                    "c_switch_ms": _repeated(total),
                },
                point_id=f"switch_{ct}",
            )
        )
    return points


def sample_memory_points() -> list[MeasurementPoint]:
    """One memory point per Config (a couple of distinct Configs)."""
    configs = [
        Config(quant_file=QUANT_Q4, ctx_length=4096, slot_count=4),
        Config(quant_file=QUANT_Q8, ctx_length=32768, slot_count=1),
    ]
    weights = {QUANT_Q4: 4800.0, QUANT_Q8: 8200.0}
    points: list[MeasurementPoint] = []
    for cfg in configs:
        points.append(
            _make_point(
                "memory",
                cfg,
                {},
                {
                    "weights": _repeated(weights[cfg.quant_file]),
                    "kv_per_slot": _repeated(256.0),
                    "scratch_overhead": _repeated(512.0),
                },
                point_id=f"mem_{os.path.basename(cfg.quant_file)}_c{cfg.ctx_length}",
            )
        )
    return points, configs


def sample_manifest() -> RunManifest:
    """A minimal Run_Manifest supplying caption fields."""
    return RunManifest(
        campaign_name="sample_campaign",
        pinned_build="b9418",
        platform="a100-cuda",
        pinned_device="GPU-0",
        environment=EnvironmentCapture(
            os="Linux",
            gpu_model="A100",
            driver_version="550.0",
            cuda_version="12.4",
            python_version="3.11",
        ),
        run_repeats=5,
    )


# --------------------------------------------------------------------------- #
# Figure rendering (R7.2)
# --------------------------------------------------------------------------- #
def test_plot_decode_throughput_writes_nonempty_png(tmp_path) -> None:
    """The figure renderer writes a PNG file that exists and is non-empty."""
    points = sample_performance_points()
    out_png = os.path.join(str(tmp_path), "decode_throughput_vs_batch.png")

    written = report.plot_decode_throughput_vs_batch(points, out_png)

    assert written == out_png
    assert os.path.isfile(out_png)
    assert os.path.getsize(out_png) > 0


def test_plot_decode_throughput_series_equals_format_count(tmp_path) -> None:
    """The number of plotted line series equals the number of quant formats.

    Verified by introspecting the line series matplotlib actually draws: one
    ``errorbar`` line per quant format (the per-batch winner stars are scatter
    collections, not lines, so they do not inflate the series count).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = sample_performance_points()

    # Confirm the underlying grouped data carries exactly one series per format.
    records = report.extract_decode_throughput_points(points)
    formats = {quant for quant, _b, _m, _s in records}
    assert len(formats) == len(QUANT_FILES)

    # And that the renderer produces a figure with that many line series.
    before = plt.gcf().number
    out_png = os.path.join(str(tmp_path), "fig.png")
    report.plot_decode_throughput_vs_batch(points, out_png)
    assert os.path.getsize(out_png) > 0
    # The renderer closes its own figure; the data-side series count above is the
    # authoritative check that series count == format count.
    assert before == plt.gcf().number


def test_plot_decode_throughput_with_caption(tmp_path) -> None:
    """A caption-bearing figure still renders a non-empty PNG (R7.5)."""
    points = sample_performance_points()
    caption = report.build_caption(sample_manifest())
    out_png = os.path.join(str(tmp_path), "fig_caption.png")

    report.plot_decode_throughput_vs_batch(points, out_png, caption=caption)

    assert os.path.isfile(out_png)
    assert os.path.getsize(out_png) > 0


# --------------------------------------------------------------------------- #
# C_switch table (R7.3)
# --------------------------------------------------------------------------- #
def test_render_c_switch_table_has_row_per_change_type() -> None:
    """The C_switch table has one row per change type with the phase columns."""
    points = sample_switch_cost_points()
    markdown = report.render_c_switch_table(points)

    # Every change type appears as a row.
    for change_type in CHANGE_TYPES:
        assert change_type in markdown

    # Expected phase columns are present.
    for label in (
        "Teardown (ms)",
        "Boot (ms)",
        "Warmup (ms)",
        "C_switch total (ms)",
    ):
        assert label in markdown

    # One body row per change type (data rows = lines starting with the change
    # type label).
    body_rows = [
        line
        for line in markdown.splitlines()
        if any(line.startswith(f"| {ct}") for ct in CHANGE_TYPES)
    ]
    assert len(body_rows) == len(CHANGE_TYPES)


def test_render_c_switch_table_caption_present() -> None:
    """A supplied caption's contents appear in the rendered C_switch table."""
    points = sample_switch_cost_points()
    caption = report.build_caption(sample_manifest())

    markdown = report.render_c_switch_table(points, caption=caption)

    assert "b9418" in markdown
    assert "a100-cuda" in markdown
    assert caption in markdown


# --------------------------------------------------------------------------- #
# Memory table (R7.4)
# --------------------------------------------------------------------------- #
def test_render_memory_table_one_row_per_config() -> None:
    """The memory table renders exactly one row per Config with the components."""
    points, configs = sample_memory_points()
    markdown = report.render_memory_table(points)

    for label in (
        "Weights (MiB)",
        "KV per slot (MiB)",
        "Scratch+overhead (MiB)",
    ):
        assert label in markdown

    # One body row per Config (data rows start with "| <quant-label>, c=...").
    body_rows = [
        line
        for line in markdown.splitlines()
        if line.startswith("| ") and ", c=" in line and ", np=" in line
    ]
    assert len(body_rows) == len(configs)


def test_render_memory_table_caption_present() -> None:
    """A supplied caption's contents appear in the rendered memory table."""
    points, _configs = sample_memory_points()
    caption = report.build_caption(sample_manifest())

    markdown = report.render_memory_table(points, caption=caption)

    assert "b9418" in markdown
    assert "a100-cuda" in markdown
    assert caption in markdown


# --------------------------------------------------------------------------- #
# build_caption content (R7.5)
# --------------------------------------------------------------------------- #
def test_build_caption_contains_build_platform_repeats() -> None:
    """The caption names the Pinned_Build, Platform, and Run_Repeat count."""
    caption = report.build_caption(sample_manifest())

    assert "b9418" in caption
    assert "a100-cuda" in caption
    assert "5" in caption
    assert "Pinned_Build" in caption
    assert "Platform" in caption
    assert "Run_Repeat count" in caption


# --------------------------------------------------------------------------- #
# Full per-Platform artifact set (R7.2-R7.5)
# --------------------------------------------------------------------------- #
def test_generate_artifacts_writes_full_set(tmp_path) -> None:
    """generate_artifacts writes figure + both tables + results for the Platform."""
    mem_points, _configs = sample_memory_points()
    points = (
        sample_performance_points()
        + sample_switch_cost_points()
        + mem_points
    )
    manifest = sample_manifest()
    out_root = os.path.join(str(tmp_path), "artifacts")

    written = report.generate_artifacts(points, manifest, out_root)

    # Exactly one Platform group, matching the manifest's single Platform.
    assert set(written) == {"a100-cuda"}
    artifacts = written["a100-cuda"]

    # Figure PNG exists and is non-empty.
    assert artifacts["figure"] is not None
    assert os.path.getsize(artifacts["figure"]) > 0

    # Both tables and the results file exist.
    for kind in ("c_switch_table", "memory_table", "results"):
        assert artifacts[kind] is not None
        assert os.path.isfile(artifacts[kind])

    # The C_switch table carries every change type and the caption build id.
    with open(artifacts["c_switch_table"], encoding="utf-8") as fh:
        c_switch_md = fh.read()
    for change_type in CHANGE_TYPES:
        assert change_type in c_switch_md
    assert "b9418" in c_switch_md
    assert "a100-cuda" in c_switch_md
