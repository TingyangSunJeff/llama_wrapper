"""Statistics aggregation over successful Run_Repeats.

This module holds the suite's pure aggregation logic. The central entry point is
:func:`aggregate`, which reduces a list of per-repeat metric values to a mean and
standard deviation computed **only over the successful repeats**, together with the
success count and an ``insufficient`` flag (``n_success < 5``). This is the single
shared implementation behind the "report mean and std over successful repeats only"
requirement that recurs across the C_switch, performance, memory, and quality
modules (R1.7, R1.8, R2.3, R2.7, R3.6, R3.7, R4.4, R4.7, R7.7).

``pct`` and ``summarize`` are generalized from ``experiments/smoke/common.py``:
``pct`` keeps the same nearest-rank percentile semantics, and ``summarize`` is
broadened from the smoke-test result-dict shape into a generic numeric summary.

The ``Aggregate`` dataclass is defined here. ``profile_suite/results.py`` (task 2.3,
built in parallel) may also define an ``Aggregate``; the two are intended to be
field-compatible (``mean``, ``std``, ``n_success``, ``insufficient``). If/when
``results.py`` provides one, prefer importing from there and re-exporting to keep a
single source of truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

# Minimum number of successful repeats required for error-bar-grade reporting.
# Below this threshold the aggregate is flagged ``insufficient`` (R1.8, R4.7, ...).
MIN_SUCCESS = 5


@dataclass
class Aggregate:
    """Mean/std summary over the successful repeats of a measurement point.

    Fields mirror the ``Aggregate`` described in the design "Shared harness"
    ``stats.py`` block:

    - ``mean``: arithmetic mean over the successful values (NaN if none).
    - ``std``: sample standard deviation (ddof=1) over the successful values;
      ``0.0`` when fewer than two successes exist (std undefined for n<2),
      NaN when there are no successes.
    - ``n_success``: number of successful repeats that fed the statistics.
    - ``insufficient``: ``True`` iff ``n_success < MIN_SUCCESS`` (i.e. < 5).
    """

    mean: float
    std: float
    n_success: int
    insufficient: bool


def _successful_values(values: Iterable[Optional[float]]) -> list[float]:
    """Return the successful numeric values from ``values``.

    A repeat is considered *failed* (and excluded) when its value is ``None`` or a
    NaN float. Everything else is coerced to ``float`` and retained. This lets
    callers pass a per-repeat list that interleaves successes and failures and still
    get aggregation over the successful repeats only.
    """

    successful: list[float] = []
    for v in values:
        if v is None:
            continue
        f = float(v)
        if math.isnan(f):
            continue
        successful.append(f)
    return successful


def aggregate(values: Iterable[Optional[float]]) -> Aggregate:
    """Aggregate per-repeat metric values over successful repeats only.

    Args:
        values: Per-repeat metric values. Failed repeats may be represented as
            ``None`` or NaN and are excluded from the statistics.

    Returns:
        An :class:`Aggregate` whose ``mean``/``std`` are computed exclusively from
        the successful values, whose ``n_success`` is the count of those values, and
        whose ``insufficient`` flag is set iff ``n_success < 5``.
    """

    successful = _successful_values(values)
    n = len(successful)

    if n == 0:
        return Aggregate(mean=float("nan"), std=float("nan"), n_success=0,
                         insufficient=True)

    mean = math.fsum(successful) / n

    if n < 2:
        # Standard deviation is undefined for a single sample; report 0.0.
        std = 0.0
    else:
        # Sample standard deviation (ddof=1) for error-bar reporting.
        variance = math.fsum((x - mean) ** 2 for x in successful) / (n - 1)
        std = math.sqrt(variance)

    return Aggregate(mean=mean, std=std, n_success=n,
                     insufficient=n < MIN_SUCCESS)


def pct(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile ``p`` (0..100) of ``values``.

    Generalized from ``experiments/smoke/common.py``: same nearest-rank semantics,
    returning NaN for an empty input. ``p`` is clamped via the rank index, so values
    outside 0..100 map to the first/last element.
    """

    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def summarize(values: Iterable[Optional[float]],
              percentiles: Sequence[float] = (50, 95)) -> dict:
    """Summarize a sequence of per-repeat values into a stats dict.

    Generalized from ``experiments/smoke/common.py`` ``summarize`` (which consumed
    smoke-test result dicts) into a generic numeric summary. Failed repeats (``None``
    or NaN) are counted but excluded from the computed statistics.

    Returns a dict with:
        - ``n``: total number of repeats supplied.
        - ``n_success`` / ``failed``: successful and failed counts.
        - ``mean`` / ``std``: from :func:`aggregate` (successful repeats only).
        - ``min`` / ``max``: extrema over successful repeats (NaN if none).
        - ``p{q}``: nearest-rank percentile for each requested percentile ``q``.
    """

    all_values = list(values)
    successful = _successful_values(all_values)
    agg = aggregate(all_values)

    summary: dict = {
        "n": len(all_values),
        "n_success": agg.n_success,
        "failed": len(all_values) - agg.n_success,
        "mean": agg.mean,
        "std": agg.std,
        "min": min(successful) if successful else float("nan"),
        "max": max(successful) if successful else float("nan"),
        "insufficient": agg.insufficient,
    }
    for q in percentiles:
        # Render an integer label when the percentile is whole (e.g. "p95").
        label = f"p{int(q)}" if float(q).is_integer() else f"p{q}"
        summary[label] = pct(successful, q)
    return summary


__all__ = ["Aggregate", "MIN_SUCCESS", "aggregate", "pct", "summarize"]
