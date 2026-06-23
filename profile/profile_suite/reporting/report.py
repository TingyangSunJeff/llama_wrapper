"""Reporting_Module: machine-readable results emission (R7.1, R7.7).

This module is the **pure read-side** of the suite. It consumes the persisted
result store of a campaign — a collection of
:class:`~profile_suite.results.MeasurementPoint` records produced by the
Reproducibility_Harness run loop, plus the campaign
:class:`~profile_suite.results.RunManifest` — and emits one machine-readable
``results.json`` per campaign.

For every measured aggregate it produces a
:class:`~profile_suite.results.MeasuredValue` carrying the metric name, its unit,
its mean and standard deviation, the source Config and Platform, the measurement
axis, and a reference to the source Run_Manifest (R7.1).

Zero-success handling (R7.7): a measurement point with **zero successful repeats**
— a ``"failed"`` or ``"platform-infeasible"`` point, or any point that produced no
aggregates — is recorded as *missing* in the artifact (under a separate
``"missing"`` list) and is **excluded from every computed aggregate** (it produces
no :class:`MeasuredValue`).

Unit mapping (``metric_unit``) follows the design's metric naming:

- ``ms``    for phase/latency metrics: anything ending in ``_ms`` and the
  C_switch components (teardown / boot / warmup / c_switch).
- ``tok/s`` for throughput metrics: anything containing ``throughput``.
- ``MiB``   for memory components: weights / KV / scratch / compute / peak /
  shortfall and any ``*_mib`` metric.
- ``count`` for plain counts (``n_tokens`` and other ``n_*`` metrics).
- ``""``    (dimensionless) as the conservative fallback.

The figure/table emission (tasks 17.2-17.4) builds on these same records and is
implemented separately; this module intentionally has no plotting dependency so it
stays cheaply importable.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import replace
from typing import Any, Iterable, Mapping, Optional, Union

from ..harness.stats import Aggregate, aggregate
from ..results import (
    MeasuredValue,
    MeasurementPoint,
    RunManifest,
    config_to_dict,
)

# Default file name for the per-campaign machine-readable results artifact.
RESULTS_FILENAME = "results.json"

# Default file name for the decode-throughput-vs-batch figure (R7.2).
DECODE_FIGURE_FILENAME = "decode_throughput_vs_batch.png"

# Default file names for the C_switch and memory tables (R7.3, R7.4).
C_SWITCH_TABLE_FILENAME = "c_switch_table.md"
MEMORY_TABLE_FILENAME = "memory_table.md"

# Module names (as emitted by the producing modules) used to recognize the
# records that feed each table. The grouping filters below are deliberately
# tolerant — a record is also accepted on the strength of its metric names — so
# the tables render whether they are handed raw MeasurementPoints or the
# report-facing MeasuredValues built from them.
SWITCH_COST_MODULE = "switch_cost"
MEMORY_MODULE = "memory"

# The three change types the C_switch table reports, in fixed presentation order
# (R1.3-R1.5, R7.3).
CHANGE_TYPES: tuple[str, ...] = ("slot-reshape", "model-reload", "combined")

# Axis key carrying the change type on a switch_cost point/value.
CHANGE_TYPE_AXIS_KEY = "change_type"

# The C_switch phase metrics, in table-column order: the three components plus
# the total (R7.3). These are the metric names emitted by Switch_Cost_Profiler.
C_SWITCH_METRICS: tuple[str, ...] = (
    "teardown_ms",
    "boot_ms",
    "warmup_ms",
    "c_switch_ms",
)

# Human-readable column headers for the C_switch metrics, same order.
C_SWITCH_COLUMN_LABELS: dict[str, str] = {
    "teardown_ms": "Teardown (ms)",
    "boot_ms": "Boot (ms)",
    "warmup_ms": "Warmup (ms)",
    "c_switch_ms": "C_switch total (ms)",
}

# The memory footprint components the memory table reports, in column order
# (R3.2, R7.4). These are the metric names emitted by Memory_Profiler.
MEMORY_METRICS: tuple[str, ...] = ("weights", "kv_per_slot", "scratch_overhead")

# Human-readable column headers for the memory metrics, same order.
MEMORY_COLUMN_LABELS: dict[str, str] = {
    "weights": "Weights (MiB)",
    "kv_per_slot": "KV per slot (MiB)",
    "scratch_overhead": "Scratch+overhead (MiB)",
}

# Cell rendered for a metric with no successful repeats (missing/excluded; R7.7).
MISSING_CELL = "—"

# Marker appended to a cell whose aggregate had fewer than 5 successful repeats
# (mirrors Aggregate.insufficient; R1.8). A footnote explains it.
INSUFFICIENT_MARKER = "†"

# The metric name (as emitted by Performance_Profiler) the figure plots.
DECODE_THROUGHPUT_METRIC = "decode_throughput"

# Axis keys, tried in order, that carry the Decode_Batch_Size of a point. The
# batched-decode path tags points with the decode batch on their measurement
# axis; several spellings are accepted so the figure is robust to the exact key
# the producing module used.
_DECODE_BATCH_AXIS_KEYS = (
    "decode_batch",
    "decode_batch_size",
    "batch_size",
    "batch",
)

# Flag recorded on a MeasuredValue whose aggregate had fewer than 5 successful
# repeats (mirrors the Aggregate.insufficient flag; R1.8/R4.7/R7.7).
FLAG_INSUFFICIENT = "insufficient-repeats"


# --------------------------------------------------------------------------- #
# Metric -> unit mapping
# --------------------------------------------------------------------------- #
def metric_unit(metric: str) -> str:
    """Map a metric name to its unit string (design "Data Models").

    The mapping is pattern-based so it covers the metrics emitted by every module
    without an exhaustive enumeration:

    - ``ms``    : any ``*_ms`` metric and the C_switch components
      (``teardown`` / ``boot`` / ``warmup`` / ``c_switch``).
    - ``tok/s`` : any metric containing ``throughput``.
    - ``MiB``   : memory components (``weights``, ``kv`` totals/per-slot,
      ``scratch``/``overhead``, ``compute``, ``peak``, ``shortfall``) and any
      ``*_mib`` metric.
    - ``count`` : plain counts (``n_tokens`` and other ``n_*`` metrics).
    - ``""``    : dimensionless fallback for anything unrecognized.
    """
    name = metric.strip().lower()

    # Latency / phase-duration metrics (milliseconds).
    if name.endswith("_ms"):
        return "ms"
    if name in {"teardown", "boot", "warmup", "c_switch", "cswitch"}:
        return "ms"

    # Throughput metrics (tokens per second).
    if "throughput" in name:
        return "tok/s"

    # Memory metrics (mebibytes).
    if name.endswith("_mib") or name.endswith("_mb"):
        return "MiB"
    memory_tokens = (
        "weights",
        "kv_per_slot",
        "kv_total",
        "kv_mib",
        "scratch_overhead",
        "scratch",
        "overhead",
        "compute",
        "observed_peak",
        "peak",
        "shortfall",
        "memory",
        "footprint",
    )
    if any(tok in name for tok in memory_tokens):
        return "MiB"

    # Plain counts.
    if name == "n_tokens" or name.startswith("n_"):
        return "count"

    # Conservative dimensionless fallback for unknown metrics.
    return ""


# --------------------------------------------------------------------------- #
# Missing-point detection (zero successful repeats -> excluded; R7.7)
# --------------------------------------------------------------------------- #
def point_is_missing(point: MeasurementPoint) -> bool:
    """Return ``True`` iff ``point`` has zero successful repeats (R7.7).

    A point is *missing* when it is a ``"failed"`` or ``"platform-infeasible"``
    point, when it produced no aggregates at all, or when none of its aggregates
    recorded a successful repeat (``n_success == 0``). Missing points are recorded
    in the artifact's ``"missing"`` list and excluded from every computed
    aggregate.
    """
    if point.status in ("failed", "platform-infeasible"):
        return True
    if not point.aggregates:
        return True
    # All aggregates with no successful repeats also count as missing.
    return all(agg.n_success <= 0 for agg in point.aggregates.values())


def missing_record(point: MeasurementPoint) -> dict[str, Any]:
    """Build the ``"missing"``-list record for a zero-success point (R7.7)."""
    return {
        "point_id": point.point_id,
        "module": point.module,
        "config": config_to_dict(point.config),
        "axis": dict(point.axis),
        "status": point.status,
        "reason": point.reason,
    }


# --------------------------------------------------------------------------- #
# Point -> MeasuredValue conversion
# --------------------------------------------------------------------------- #
def measured_values_from_point(
    point: MeasurementPoint,
    platform: str,
    manifest_ref: str,
) -> list[MeasuredValue]:
    """Convert one measurement point's aggregates into :class:`MeasuredValue`\\ s.

    Emits one :class:`MeasuredValue` per metric aggregate on the point, each
    carrying the metric name, its mapped unit, the aggregate mean/std, the success
    count, the source Config (serialized), the Platform, the measurement axis, and
    the supplied Run_Manifest reference (R7.1). Aggregates with no successful
    repeats are skipped (they contribute nothing to the artifact).

    Returns an empty list for a missing point (caller should record it as missing).
    """
    if point_is_missing(point):
        return []

    config_dict = config_to_dict(point.config)
    axis = dict(point.axis)
    values: list[MeasuredValue] = []

    # Stable, deterministic metric order for reproducible artifacts.
    for metric in sorted(point.aggregates):
        agg = point.aggregates[metric]
        if agg.n_success <= 0:
            # Defensive: a per-metric aggregate without successes carries no data.
            continue
        flags: list[str] = []
        if agg.insufficient:
            flags.append(FLAG_INSUFFICIENT)
        values.append(
            MeasuredValue(
                metric=metric,
                unit=metric_unit(metric),
                mean=agg.mean,
                std=agg.std,
                n_success=agg.n_success,
                config=config_dict,
                platform=platform,
                manifest_ref=manifest_ref,
                flags=flags,
                axis=axis,
            )
        )
    return values


# --------------------------------------------------------------------------- #
# Building the results artifact
# --------------------------------------------------------------------------- #
def build_results(
    points: Iterable[MeasurementPoint],
    manifest: RunManifest,
    *,
    manifest_ref: Optional[str] = None,
) -> dict[str, Any]:
    """Build the machine-readable results artifact dict for a campaign (R7.1, R7.7).

    Args:
        points: The campaign's persisted measurement points.
        manifest: The campaign Run_Manifest (supplies the Platform tag and the
            default manifest reference).
        manifest_ref: Explicit reference to the source Run_Manifest (e.g. its
            on-disk path). Defaults to the manifest's campaign name.

    Returns:
        A JSON-serializable dict with:

        - ``campaign`` / ``platform`` / ``manifest_ref`` headers,
        - ``values``  : list of serialized :class:`MeasuredValue` records, one per
          successful metric aggregate (the computed aggregates),
        - ``missing`` : list of zero-success points, excluded from ``values``
          (R7.7).
    """
    ref = manifest_ref if manifest_ref is not None else manifest.campaign_name
    platform = manifest.platform

    values: list[MeasuredValue] = []
    missing: list[dict[str, Any]] = []

    for point in points:
        if point_is_missing(point):
            missing.append(missing_record(point))
            continue
        values.extend(measured_values_from_point(point, platform, ref))

    return {
        "campaign": manifest.campaign_name,
        "platform": platform,
        "manifest_ref": ref,
        "values": [v.to_dict() for v in values],
        "missing": missing,
    }


def emit_results(
    points: Iterable[MeasurementPoint],
    manifest: RunManifest,
    out_path: str,
    *,
    manifest_ref: Optional[str] = None,
) -> dict[str, Any]:
    """Emit one machine-readable ``results.json`` for a campaign (R7.1, R7.7).

    Builds the results artifact from ``points`` + ``manifest`` (see
    :func:`build_results`) and writes it as pretty-printed JSON to ``out_path``.
    If ``out_path`` names an existing directory, the artifact is written to
    ``<out_path>/results.json``. The parent directory is created if needed (kept
    within the campaign-scoped tree by the caller — this function never writes
    outside the path it is given).

    Returns the artifact dict that was written.
    """
    artifact = build_results(points, manifest, manifest_ref=manifest_ref)

    target = out_path
    if os.path.isdir(out_path):
        target = os.path.join(out_path, RESULTS_FILENAME)

    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(target, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=False)
        fh.write("\n")

    return artifact


# --------------------------------------------------------------------------- #
# Reading the persisted store
# --------------------------------------------------------------------------- #
MANIFEST_FILENAME = "manifest.json"
POINTS_FILENAME = "points.json"
POINTS_DIRNAME = "points"


def load_manifest(run_dir: str) -> RunManifest:
    """Load the :class:`RunManifest` from ``<run_dir>/manifest.json``."""
    path = os.path.join(run_dir, MANIFEST_FILENAME)
    with open(path, "r", encoding="utf-8") as fh:
        return RunManifest.from_dict(json.load(fh))


def load_points(run_dir: str) -> list[MeasurementPoint]:
    """Load the persisted measurement points from a campaign run directory.

    Supports two on-disk conventions, tried in order:

    1. A single ``points.json`` file holding a JSON list of serialized
       :class:`MeasurementPoint` dicts.
    2. A ``points/`` subdirectory of ``*.json`` files, one serialized point each
       (loaded in sorted filename order for determinism).

    Returns an empty list if neither convention is present.
    """
    points_file = os.path.join(run_dir, POINTS_FILENAME)
    if os.path.isfile(points_file):
        with open(points_file, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return [MeasurementPoint.from_dict(d) for d in raw]

    points_dir = os.path.join(run_dir, POINTS_DIRNAME)
    if os.path.isdir(points_dir):
        points: list[MeasurementPoint] = []
        for name in sorted(os.listdir(points_dir)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(points_dir, name), "r", encoding="utf-8") as fh:
                points.append(MeasurementPoint.from_dict(json.load(fh)))
        return points

    return []


def emit_results_for_run(
    run_dir: str,
    *,
    out_name: str = RESULTS_FILENAME,
) -> dict[str, Any]:
    """Read a campaign run directory's store and emit its ``results.json``.

    Loads the Run_Manifest and the persisted measurement points from ``run_dir``,
    converts every successful aggregate into a :class:`MeasuredValue`, marks
    zero-success points missing, and writes the artifact to
    ``<run_dir>/<out_name>``. The manifest reference recorded on each value is the
    on-disk path of the source ``manifest.json``.

    Returns the artifact dict that was written.
    """
    manifest = load_manifest(run_dir)
    points = load_points(run_dir)
    manifest_ref = os.path.join(run_dir, MANIFEST_FILENAME)
    out_path = os.path.join(run_dir, out_name)
    return emit_results(points, manifest, out_path, manifest_ref=manifest_ref)


# --------------------------------------------------------------------------- #
# Decode-throughput-vs-batch figure (R7.2)
# --------------------------------------------------------------------------- #
def _quant_label(quant_file: str) -> str:
    """Derive a compact, stable series label from a quant/model file path.

    Uses the file's base name with a trailing ``.gguf`` stripped so the legend is
    readable; falls back to the raw value when it is empty.
    """
    if not quant_file:
        return "(unknown)"
    base = os.path.basename(quant_file)
    if base.lower().endswith(".gguf"):
        base = base[: -len(".gguf")]
    return base or quant_file


def _axis_decode_batch(axis: dict[str, Any]) -> Optional[int]:
    """Extract the Decode_Batch_Size from a point/value measurement axis.

    Tries the accepted axis keys in order and returns the first integer-coercible
    value found, or ``None`` when the axis carries no decode-batch dimension (e.g.
    the server-path performance points tagged ``{"path": "server"}``).
    """
    for key in _DECODE_BATCH_AXIS_KEYS:
        if key in axis and axis[key] is not None:
            try:
                return int(axis[key])
            except (TypeError, ValueError):
                return None
    return None


# One decode-throughput observation: (quant_file, batch, mean, std).
_DecodePoint = tuple[str, int, float, float]


def extract_decode_throughput_points(
    values_or_points: Iterable[Union[MeasuredValue, "MeasurementPoint"]],
) -> list[_DecodePoint]:
    """Collect decode-throughput observations grouped-ready by format and batch.

    Accepts a heterogeneous iterable of either :class:`MeasuredValue` records (the
    report-facing model emitted by :func:`build_results`) or raw
    :class:`MeasurementPoint` records (the persisted store). For each item that
    carries a ``decode_throughput`` measurement with a decode-batch axis and at
    least one successful repeat, yields a ``(quant_file, batch, mean, std)`` tuple.

    Items without a decode-batch axis (e.g. server-path points), without a
    decode-throughput metric, or with zero successful repeats are skipped (R7.7).
    """
    records: list[_DecodePoint] = []

    for item in values_or_points:
        if isinstance(item, MeasuredValue):
            if item.metric != DECODE_THROUGHPUT_METRIC:
                continue
            if item.n_success <= 0:
                continue
            batch = _axis_decode_batch(item.axis)
            if batch is None:
                continue
            quant = str(item.config.get("quant_file", "")) if item.config else ""
            records.append((quant, batch, float(item.mean), float(item.std)))
        elif isinstance(item, MeasurementPoint):
            agg = item.aggregates.get(DECODE_THROUGHPUT_METRIC)
            if agg is None or agg.n_success <= 0:
                continue
            batch = _axis_decode_batch(item.axis)
            if batch is None:
                continue
            quant = item.config.quant_file
            records.append((quant, batch, float(agg.mean), float(agg.std)))
        # Unknown item types are ignored so callers may pass mixed stores.

    return records


def plot_decode_throughput_vs_batch(
    values_or_points: Iterable[Union[MeasuredValue, "MeasurementPoint"]],
    out_png: str,
    *,
    caption: Optional[str] = None,
    title: str = "Decode throughput vs decode batch size",
) -> str:
    """Plot Decode_Throughput vs Decode_Batch_Size, one series per quant format (R7.2).

    Groups the supplied decode-throughput measurements by quant format and plots,
    on a single figure, one line series per format with ``Decode_Batch_Size`` on
    the horizontal axis (log-2 scaled, the canonical sweep is powers of two) and
    ``Decode_Throughput`` (tok/s) on the vertical axis, drawing the per-format
    standard deviation as a y error bar.

    At each swept Decode_Batch_Size value the winning quant format — computed by
    reusing :func:`profile_suite.modules.performance.batch_winner` over the
    per-format ``(mean, std)`` aggregates at that batch — is marked with a star at
    its data point so the per-batch winner is visible directly on the figure
    (R2.4 / R7.2). Batches whose winner is a statistical tie are annotated as such.

    Args:
        values_or_points: Iterable of :class:`MeasuredValue` and/or
            :class:`MeasurementPoint` records (mixed allowed). Only
            ``decode_throughput`` observations carrying a decode-batch axis with a
            successful aggregate are used.
        out_png: Destination PNG path. Its parent directory is created if needed;
            this function never writes outside the path it is given.
        caption: Optional caption text drawn beneath the plot (e.g. Pinned_Build /
            Platform / Run_Repeat count — supplied by the per-Platform caption
            task). ``None`` omits the caption.
        title: Figure title.

    Returns:
        The ``out_png`` path that was written.

    Raises:
        ValueError: If no usable decode-throughput-vs-batch observations are found
            in ``values_or_points`` (nothing to plot).
    """
    # Lazy imports keep this module cheaply importable (no plotting/measurement
    # dependency is pulled in unless a figure is actually rendered).
    import matplotlib

    matplotlib.use("Agg")  # headless backend: no display required.
    import matplotlib.pyplot as plt

    from ..modules.performance import batch_winner

    records = extract_decode_throughput_points(values_or_points)
    if not records:
        raise ValueError(
            "no decode_throughput-vs-batch observations to plot "
            "(need decode_throughput values with a decode-batch axis)"
        )

    # series[quant_file][batch] = (mean, std). A later observation for the same
    # (format, batch) overwrites an earlier one (deterministic last-wins).
    series: dict[str, dict[int, tuple[float, float]]] = {}
    for quant, batch, mean, std in records:
        series.setdefault(quant, {})[batch] = (mean, std)

    # All swept batch values across every format, ascending.
    all_batches = sorted({b for per_batch in series.values() for b in per_batch})

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    # One line series per quant format, in a stable (sorted) order so the legend
    # and colors are deterministic across runs.
    for quant in sorted(series):
        per_batch = series[quant]
        xs = sorted(per_batch)
        ys = [per_batch[b][0] for b in xs]
        errs = [per_batch[b][1] for b in xs]
        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            marker="o",
            capsize=3,
            label=_quant_label(quant),
        )

    # Mark the per-batch winner (reusing batch_winner) with a star at its point.
    winner_label_used = False
    for batch in all_batches:
        aggregates = {
            quant: per_batch[batch]
            for quant, per_batch in series.items()
            if batch in per_batch
        }
        if not aggregates:
            continue
        result = batch_winner(aggregates)
        win_mean = aggregates[result.winner][0]
        ax.scatter(
            [batch],
            [win_mean],
            marker="*",
            s=240,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
            label=None if winner_label_used else "per-batch winner",
        )
        winner_label_used = True
        if result.tie:
            ax.annotate(
                "tie",
                xy=(batch, win_mean),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="gray",
            )

    ax.set_xlabel("Decode batch size")
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title(title)

    # Powers-of-two sweep reads best on a log-2 x axis with explicit ticks.
    if all_batches and min(all_batches) > 0:
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_batches)
        ax.set_xticklabels([str(b) for b in all_batches])

    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    if caption:
        fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8)
        fig.subplots_adjust(bottom=0.18)
    else:
        fig.tight_layout()

    parent = os.path.dirname(out_png)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return out_png


# --------------------------------------------------------------------------- #
# C_switch and memory tables (R7.3, R7.4)
# --------------------------------------------------------------------------- #
def _point_metric_values(point: MeasurementPoint, metric: str) -> list[float]:
    """Return the successful per-repeat values of ``metric`` on a point.

    Prefers the raw retained-repeat values (the most faithful source of the
    within-point variation, so a pooled standard deviation is meaningful):
    the discarded warmup run and any failed repeats are excluded (R5.4, R7.7).
    When the point carries no raw repeat values for the metric (e.g. records
    reconstructed from a results store that kept only aggregates), it falls back
    to the point's aggregate mean as a single observation. Returns ``[]`` when the
    metric has no successful data on the point.
    """
    raw: list[float] = []
    for repeat in point.repeats:
        if repeat.discarded_warmup or not repeat.ok:
            continue
        value = repeat.metrics.get(metric)
        if value is None:
            continue
        fval = float(value)
        if math.isnan(fval):
            continue
        raw.append(fval)
    if raw:
        return raw

    agg = point.aggregates.get(metric)
    if agg is not None and agg.n_success > 0 and not math.isnan(agg.mean):
        return [float(agg.mean)]
    return []


def _fmt_cell(agg: Aggregate) -> str:
    """Format one ``mean ± std`` table cell, or the missing marker (R7.7).

    A metric with no successful repeats (``n_success <= 0`` or a NaN mean) renders
    as :data:`MISSING_CELL`; an aggregate flagged ``insufficient`` (fewer than 5
    successful repeats) is marked with :data:`INSUFFICIENT_MARKER`.
    """
    if agg.n_success <= 0 or math.isnan(agg.mean):
        return MISSING_CELL
    marker = INSUFFICIENT_MARKER if agg.insufficient else ""
    return f"{agg.mean:.2f} ± {agg.std:.2f}{marker}"


def _escape_cell(text: str) -> str:
    """Escape a markdown table cell so embedded pipes do not split columns."""
    return str(text).replace("|", "\\|")


def _render_markdown_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    caption: Optional[str],
    footnotes: list[str],
) -> str:
    """Render a GitHub-flavored markdown table with an optional caption/footnotes.

    The caption (when supplied) is emitted as an italic line above the table; each
    footnote is emitted as its own italic line beneath it. Returns the table as a
    single newline-terminated string.
    """
    lines: list[str] = []
    if caption:
        lines.append(f"_{caption}_")
        lines.append("")
    lines.append("| " + " | ".join(_escape_cell(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(c) for c in row) + " |")
    for note in footnotes:
        lines.append("")
        lines.append(f"_{note}_")
    return "\n".join(lines) + "\n"


def _write_table(markdown: str, out_path: str, default_filename: str) -> str:
    """Write ``markdown`` to ``out_path`` (a file or directory) and return the path.

    If ``out_path`` names an existing directory the table is written to
    ``<out_path>/<default_filename>``. The parent directory is created if needed;
    this helper never writes outside the path it is given.
    """
    target = out_path
    if os.path.isdir(out_path):
        target = os.path.join(out_path, default_filename)
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    return target


def build_c_switch_rows(
    values_or_points: Iterable[Union[MeasuredValue, "MeasurementPoint"]],
) -> tuple[dict[str, dict[str, Aggregate]], list[dict[str, Any]]]:
    """Pool C_switch phase measurements by change type (R7.3, R7.7).

    Accepts a heterogeneous iterable of :class:`MeasuredValue` and/or
    :class:`MeasurementPoint` records (mixed allowed, like the figure path).
    A record is recognized as a C_switch observation when it carries a recognized
    ``change_type`` axis and one of the :data:`C_SWITCH_METRICS`.

    For every change type and every phase metric, the successful observations are
    pooled and reduced via :func:`profile_suite.harness.stats.aggregate`:

    - :class:`MeasurementPoint` contributes its successful per-repeat values (or its
      aggregate mean when only aggregates survive); zero-success points are recorded
      as *missing* and excluded (R7.7).
    - :class:`MeasuredValue` contributes its mean (one observation); zero-success
      values are skipped.

    Returns:
        ``(rows, missing)`` where ``rows[change_type][metric]`` is the pooled
        :class:`Aggregate` for every change type in :data:`CHANGE_TYPES` (a change
        type with no data yields zero-success aggregates), and ``missing`` is the
        list of excluded zero-success point records.
    """
    pooled: dict[str, dict[str, list[float]]] = {
        ct: {m: [] for m in C_SWITCH_METRICS} for ct in CHANGE_TYPES
    }
    missing: list[dict[str, Any]] = []

    for item in values_or_points:
        if isinstance(item, MeasuredValue):
            if item.metric not in C_SWITCH_METRICS:
                continue
            change_type = item.axis.get(CHANGE_TYPE_AXIS_KEY)
            if change_type not in pooled:
                continue
            if item.n_success <= 0 or math.isnan(item.mean):
                continue
            pooled[change_type][item.metric].append(float(item.mean))
        elif isinstance(item, MeasurementPoint):
            change_type = item.axis.get(CHANGE_TYPE_AXIS_KEY)
            if change_type not in pooled:
                continue
            if point_is_missing(item):
                missing.append(missing_record(item))
                continue
            for metric in C_SWITCH_METRICS:
                pooled[change_type][metric].extend(
                    _point_metric_values(item, metric)
                )
        # Unknown item types are ignored so callers may pass mixed stores.

    rows: dict[str, dict[str, Aggregate]] = {
        ct: {m: aggregate(pooled[ct][m]) for m in C_SWITCH_METRICS}
        for ct in CHANGE_TYPES
    }
    return rows, missing


def render_c_switch_table(
    values_or_points: Iterable[Union[MeasuredValue, "MeasurementPoint"]],
    *,
    caption: Optional[str] = None,
    out_path: Optional[str] = None,
    title: str = "C_switch decomposition by change type",
) -> str:
    """Render the C_switch table as markdown (R7.3).

    Produces a table with one row per change type (slot-reshape, model-reload,
    combined) and columns for the mean ± std of Teardown, Boot, Warmup, and total
    C_switch, pooled across every measured transition of that change type (see
    :func:`build_c_switch_rows`). Change types with no successful repeats render as
    missing rows and the excluded points are summarized in a footnote (R7.7).

    Args:
        values_or_points: Mixed iterable of :class:`MeasuredValue` and/or
            :class:`MeasurementPoint` switch_cost records.
        caption: Optional caption (e.g. Pinned_Build / Platform / Run_Repeat count)
            emitted above the table.
        out_path: Optional destination. When given, the markdown is also written
            there (a directory receives ``c_switch_table.md``).
        title: Heading rendered above the table as part of the caption line.

    Returns:
        The markdown table as a string.
    """
    rows, missing = build_c_switch_rows(values_or_points)

    headers = ["Change type"] + [C_SWITCH_COLUMN_LABELS[m] for m in C_SWITCH_METRICS]
    body: list[list[str]] = []
    any_insufficient = False
    for change_type in CHANGE_TYPES:
        cells = [change_type]
        for metric in C_SWITCH_METRICS:
            agg = rows[change_type][metric]
            if agg.n_success > 0 and not math.isnan(agg.mean) and agg.insufficient:
                any_insufficient = True
            cells.append(_fmt_cell(agg))
        body.append(cells)

    footnotes: list[str] = []
    if any_insufficient:
        footnotes.append(
            f"{INSUFFICIENT_MARKER} fewer than 5 successful repeats — "
            "insufficient for error-bar reporting."
        )
    if missing:
        ids = ", ".join(sorted(rec["point_id"] for rec in missing))
        footnotes.append(
            f"Excluded {len(missing)} measurement point(s) with no successful "
            f"repeats (R7.7): {ids}."
        )

    full_caption = title if not caption else f"{title}. {caption}"
    markdown = _render_markdown_table(
        headers, body, caption=full_caption, footnotes=footnotes
    )

    if out_path is not None:
        _write_table(markdown, out_path, C_SWITCH_TABLE_FILENAME)
    return markdown


def _config_key(config: dict[str, Any]) -> tuple[str, int, int]:
    """Deterministic sort/identity key for a serialized Config."""
    return (
        str(config.get("quant_file", "")),
        int(config.get("ctx_length", 0) or 0),
        int(config.get("slot_count", 0) or 0),
    )


def _config_label(config: dict[str, Any]) -> str:
    """Compact, readable row label for a serialized Config in the memory table."""
    quant = _quant_label(str(config.get("quant_file", "")))
    ctx = config.get("ctx_length")
    slots = config.get("slot_count")
    return f"{quant}, c={ctx}, np={slots}"


def _is_memory_record(item: MeasurementPoint) -> bool:
    """Return ``True`` iff a point is a memory-footprint record (R7.4).

    Recognized by its module name or, defensively, by carrying any of the memory
    footprint metrics — so missing memory points (which have no aggregates yet)
    are still recognized and reported as missing.
    """
    if item.module == MEMORY_MODULE:
        return True
    if any(m in item.aggregates for m in MEMORY_METRICS):
        return True
    for repeat in item.repeats:
        if any(m in repeat.metrics for m in MEMORY_METRICS):
            return True
    return False


def build_memory_rows(
    values_or_points: Iterable[Union[MeasuredValue, "MeasurementPoint"]],
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Aggregate]]],
    list[dict[str, Any]],
]:
    """Pool memory footprint measurements by Config (R7.4, R7.7).

    Accepts a heterogeneous iterable of :class:`MeasuredValue` and/or
    :class:`MeasurementPoint` records (mixed allowed). A record is recognized as a
    memory observation by its module name or by carrying any of
    :data:`MEMORY_METRICS` (``weights`` / ``kv_per_slot`` / ``scratch_overhead``).

    For every Config the successful observations of each component are pooled and
    reduced via :func:`profile_suite.harness.stats.aggregate`. A Config whose memory
    point has no successful repeats is still listed (so its row is visible) with
    zero-success aggregates, and is recorded in the returned ``missing`` list and
    excluded from the aggregates (R7.7).

    Returns:
        ``(rows, missing)`` where ``rows`` is a list of
        ``(config_dict, {metric: Aggregate})`` ordered deterministically by Config,
        and ``missing`` is the list of excluded zero-success point records.
    """
    pooled: dict[tuple[str, int, int], dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []

    def _slot(config: dict[str, Any]) -> dict[str, Any]:
        key = _config_key(config)
        if key not in pooled:
            pooled[key] = {
                "config": dict(config),
                "metrics": {m: [] for m in MEMORY_METRICS},
            }
        return pooled[key]

    for item in values_or_points:
        if isinstance(item, MeasuredValue):
            if item.metric not in MEMORY_METRICS:
                continue
            slot = _slot(item.config or {})
            if item.n_success <= 0 or math.isnan(item.mean):
                continue
            slot["metrics"][item.metric].append(float(item.mean))
        elif isinstance(item, MeasurementPoint):
            if not _is_memory_record(item):
                continue
            config_dict = config_to_dict(item.config)
            _slot(config_dict)  # ensure the row exists even if missing
            if point_is_missing(item):
                missing.append(missing_record(item))
                continue
            slot = _slot(config_dict)
            for metric in MEMORY_METRICS:
                slot["metrics"][metric].extend(_point_metric_values(item, metric))
        # Unknown item types are ignored so callers may pass mixed stores.

    rows: list[tuple[dict[str, Any], dict[str, Aggregate]]] = []
    for key in sorted(pooled):
        entry = pooled[key]
        aggs = {m: aggregate(entry["metrics"][m]) for m in MEMORY_METRICS}
        rows.append((entry["config"], aggs))
    return rows, missing


def render_memory_table(
    values_or_points: Iterable[Union[MeasuredValue, "MeasurementPoint"]],
    *,
    caption: Optional[str] = None,
    out_path: Optional[str] = None,
    title: str = "Memory footprint decomposition by Config",
) -> str:
    """Render the memory footprint table as markdown (R7.4).

    Produces a table with one row per Config and columns for the mean ± std of the
    model-weights, KV-cache-per-slot, and scratch-plus-overhead components in MiB,
    pooled per Config (see :func:`build_memory_rows`). Configs whose memory point
    had no successful repeats render as missing rows and are summarized in a
    footnote (R7.7).

    Args:
        values_or_points: Mixed iterable of :class:`MeasuredValue` and/or
            :class:`MeasurementPoint` memory records.
        caption: Optional caption (e.g. Pinned_Build / Platform / Run_Repeat count)
            emitted above the table.
        out_path: Optional destination. When given, the markdown is also written
            there (a directory receives ``memory_table.md``).
        title: Heading rendered above the table as part of the caption line.

    Returns:
        The markdown table as a string.
    """
    rows, missing = build_memory_rows(values_or_points)

    headers = ["Config"] + [MEMORY_COLUMN_LABELS[m] for m in MEMORY_METRICS]
    body: list[list[str]] = []
    any_insufficient = False
    for config_dict, aggs in rows:
        cells = [_config_label(config_dict)]
        for metric in MEMORY_METRICS:
            agg = aggs[metric]
            if agg.n_success > 0 and not math.isnan(agg.mean) and agg.insufficient:
                any_insufficient = True
            cells.append(_fmt_cell(agg))
        body.append(cells)

    if not body:
        body.append([MISSING_CELL] * len(headers))

    footnotes: list[str] = []
    if any_insufficient:
        footnotes.append(
            f"{INSUFFICIENT_MARKER} fewer than 5 successful repeats — "
            "insufficient for error-bar reporting."
        )
    if missing:
        ids = ", ".join(sorted(rec["point_id"] for rec in missing))
        footnotes.append(
            f"Excluded {len(missing)} measurement point(s) with no successful "
            f"repeats (R7.7): {ids}."
        )

    full_caption = title if not caption else f"{title}. {caption}"
    markdown = _render_markdown_table(
        headers, body, caption=full_caption, footnotes=footnotes
    )

    if out_path is not None:
        _write_table(markdown, out_path, MEMORY_TABLE_FILENAME)
    return markdown


# --------------------------------------------------------------------------- #
# Per-Platform separation and captions (R6.5, R7.5, R7.6)
# --------------------------------------------------------------------------- #
# Axis key that, when present on a measurement point, overrides the manifest's
# Platform descriptor for that point. The suite runs one Platform per campaign
# (R6.1), so in normal operation every point belongs to ``manifest.platform``;
# this override lets the Reporting_Module group a results store that spans more
# than one Platform (R7.6 / Property 13) — e.g. a store aggregated across
# campaigns — and is the hook the multi-Platform separation is verified against.
PLATFORM_AXIS_KEY = "platform"


def build_caption(
    manifest_or_fields: Union[RunManifest, Mapping[str, Any]],
    *,
    pinned_build: Optional[str] = None,
    platform: Optional[str] = None,
    run_repeats: Optional[int] = None,
) -> str:
    """Build the artifact caption stating build / platform / repeat count (R7.5).

    Every figure and table the suite emits must carry a caption naming the
    Pinned_Build identifier, the Platform descriptor, and the Run_Repeat count used
    to generate the underlying data (R7.5, Property 18).

    Args:
        manifest_or_fields: Either a :class:`~profile_suite.results.RunManifest` or
            any mapping carrying ``pinned_build`` / ``platform`` / ``run_repeats``
            keys. The mapping form lets callers build a caption without a full
            manifest.
        pinned_build / platform / run_repeats: Optional explicit overrides. When
            given they take precedence over the value read from
            ``manifest_or_fields`` — used by :func:`generate_artifacts` to stamp
            each per-Platform artifact set with its single source Platform.

    Returns:
        A single-line caption of the form
        ``"Pinned_Build: <build> | Platform: <platform> | Run_Repeat count: <n>"``.
    """
    if isinstance(manifest_or_fields, RunManifest):
        src_build = manifest_or_fields.pinned_build
        src_platform = manifest_or_fields.platform
        src_repeats: Any = manifest_or_fields.run_repeats
    else:
        src_build = manifest_or_fields.get("pinned_build")
        src_platform = manifest_or_fields.get("platform")
        src_repeats = manifest_or_fields.get("run_repeats")

    build = pinned_build if pinned_build is not None else src_build
    plat = platform if platform is not None else src_platform
    repeats = run_repeats if run_repeats is not None else src_repeats

    build_str = "unknown" if build in (None, "") else str(build)
    plat_str = "unknown" if plat in (None, "") else str(plat)
    repeats_str = "unknown" if repeats in (None, "") else str(repeats)

    return (
        f"Pinned_Build: {build_str} | "
        f"Platform: {plat_str} | "
        f"Run_Repeat count: {repeats_str}"
    )


def point_platform(point: MeasurementPoint, default: str) -> str:
    """Return the Platform descriptor a point belongs to (R6.5).

    Reads an explicit ``platform`` override from the point's measurement axis when
    present (so a multi-Platform store can be separated), otherwise falls back to
    ``default`` — the campaign's single Platform from its Run_Manifest.
    """
    value = point.axis.get(PLATFORM_AXIS_KEY)
    if value is not None and str(value) != "":
        return str(value)
    return default


def group_points_by_platform(
    points: Iterable[MeasurementPoint],
    default_platform: str,
) -> dict[str, list[MeasurementPoint]]:
    """Group measurement points by their Platform descriptor (R6.5, R7.6).

    Points without an explicit ``platform`` axis override are assigned to
    ``default_platform`` (the campaign's single Platform). The returned mapping
    partitions the points so that no group mixes Platforms — the precondition for
    emitting artifacts that never combine values from more than one Platform.
    """
    groups: dict[str, list[MeasurementPoint]] = {}
    for point in points:
        plat = point_platform(point, default_platform)
        groups.setdefault(plat, []).append(point)
    return groups


# Characters not allowed in a Platform sub-directory name are replaced with '_'
# so descriptors like "a100-cuda" map to a safe folder while pathological values
# (path separators, etc.) cannot escape ``out_root``.
_UNSAFE_DIR_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _platform_dirname(platform: str) -> str:
    """Sanitize a Platform descriptor into a single safe sub-directory name.

    Replaces every run of unsafe characters (including path separators) with an
    underscore and strips leading dots/separators so the result is always a single
    path component under the per-campaign artifacts root (isolation, R8.5).
    """
    name = _UNSAFE_DIR_CHARS.sub("_", str(platform)).strip("._")
    return name or "unknown"


def generate_artifacts(
    points: Iterable[MeasurementPoint],
    manifest: RunManifest,
    out_root: str,
    *,
    manifest_ref: Optional[str] = None,
) -> dict[str, dict[str, Optional[str]]]:
    """Emit a separate, single-Platform artifact set per Platform (R6.5, R7.5, R7.6).

    Groups the campaign's measurement points by Platform descriptor (see
    :func:`group_points_by_platform`) and, **for each Platform**, writes a complete
    artifact set under ``out_root/<platform>/``:

    - ``results.json`` — the machine-readable results for that Platform only,
    - ``decode_throughput_vs_batch.png`` — the decode-throughput figure (omitted
      when the Platform has no decode-batch observations to plot),
    - ``c_switch_table.md`` — the C_switch decomposition table,
    - ``memory_table.md`` — the memory footprint table.

    Every artifact is built from exactly one Platform's points and tagged with that
    single Platform, so **no emitted artifact combines values from more than one
    Platform** (R7.6, Property 13). Each figure and table carries the caption built
    by :func:`build_caption`, stamped with that Platform's descriptor, the campaign
    Pinned_Build, and the Run_Repeat count (R7.5, Property 18).

    Args:
        points: The campaign's persisted measurement points (may span more than
            one Platform via the ``platform`` axis override).
        manifest: The campaign Run_Manifest (supplies the default Platform, the
            Pinned_Build, the Run_Repeat count, and the manifest reference).
        out_root: The artifacts root directory. One ``<platform>/`` sub-directory
            is created beneath it per Platform; all writes stay under ``out_root``.
        manifest_ref: Explicit reference to the source Run_Manifest recorded on
            each emitted value (defaults to the manifest's campaign name).

    Returns:
        A mapping ``{platform: {artifact_kind: written_path_or_None}}`` describing
        every artifact written, where ``artifact_kind`` is one of ``"results"``,
        ``"figure"`` (``None`` when no decode data), ``"c_switch_table"``, and
        ``"memory_table"``.
    """
    groups = group_points_by_platform(points, manifest.platform)

    written: dict[str, dict[str, Optional[str]]] = {}
    for platform in sorted(groups):
        plat_points = groups[platform]

        # Per-Platform manifest so every emitted record carries this single
        # Platform descriptor (no value is tagged with another Platform).
        plat_manifest = replace(manifest, platform=platform)
        caption = build_caption(plat_manifest)

        plat_dir = os.path.join(out_root, _platform_dirname(platform))
        os.makedirs(plat_dir, exist_ok=True)

        artifacts: dict[str, Optional[str]] = {}

        # 1. Machine-readable results for this Platform only (R7.1, R7.6).
        emit_results(
            plat_points, plat_manifest, plat_dir, manifest_ref=manifest_ref
        )
        artifacts["results"] = os.path.join(plat_dir, RESULTS_FILENAME)

        # 2. Decode-throughput-vs-batch figure (R7.2). Skipped cleanly when this
        #    Platform has no decode-batch observations to plot.
        figure_path = os.path.join(plat_dir, DECODE_FIGURE_FILENAME)
        try:
            plot_decode_throughput_vs_batch(
                plat_points, figure_path, caption=caption
            )
            artifacts["figure"] = figure_path
        except ValueError:
            artifacts["figure"] = None

        # 3. C_switch decomposition table (R7.3).
        artifacts["c_switch_table"] = _write_table(
            render_c_switch_table(plat_points, caption=caption),
            plat_dir,
            C_SWITCH_TABLE_FILENAME,
        )

        # 4. Memory footprint table (R7.4).
        artifacts["memory_table"] = _write_table(
            render_memory_table(plat_points, caption=caption),
            plat_dir,
            MEMORY_TABLE_FILENAME,
        )

        written[platform] = artifacts

    return written


__all__ = [
    "RESULTS_FILENAME",
    "DECODE_FIGURE_FILENAME",
    "C_SWITCH_TABLE_FILENAME",
    "MEMORY_TABLE_FILENAME",
    "DECODE_THROUGHPUT_METRIC",
    "FLAG_INSUFFICIENT",
    "PLATFORM_AXIS_KEY",
    "CHANGE_TYPES",
    "C_SWITCH_METRICS",
    "MEMORY_METRICS",
    "metric_unit",
    "point_is_missing",
    "missing_record",
    "measured_values_from_point",
    "build_results",
    "emit_results",
    "load_manifest",
    "load_points",
    "emit_results_for_run",
    "extract_decode_throughput_points",
    "plot_decode_throughput_vs_batch",
    "build_c_switch_rows",
    "render_c_switch_table",
    "build_memory_rows",
    "render_memory_table",
    "build_caption",
    "point_platform",
    "group_points_by_platform",
    "generate_artifacts",
]
