"""Quality_Module: the quality axis of the quantization knob (R4).

The quant knob trades output quality against throughput and VRAM. This module
supplies **exactly one** quality value per quant format in the campaign
Config_Grid (R4.1), via one of two sources selected by the campaign's
``local_quality`` flag:

- **cited** (default, ``local_quality=False``): the value is looked up in a
  static table of figures cited from the Kurt quant-eval study
  (arXiv:2601.14277), carrying a citation reference that identifies the study
  and the specific value used (R4.2). A quant format that is absent from the
  cited table is recorded as **missing** (``value=None``) with the omission
  indicated, without aborting the remaining formats (R4.6).
- **locally-measured** (``local_quality=True``): the value is a
  Quality_Surrogate (a perplexity over a campaign-fixed input set, computed
  uniformly across every format) supplied by the caller in ``local_surrogates``
  (R4.3). When fewer than 5 successful Run_Repeats were obtained, the value is
  flagged as having insufficient repeats and reported **without** a standard
  deviation (R4.7).

Every emitted value is labeled either ``"cited"`` or ``"locally-measured"`` to
match its source (R4.5).

Scope of this task (14.1)
-------------------------
This file implements the **pure decision logic**: the cited lookup, the
labeling, the missing-value handling, and the local insufficient-repeats
handling. The actual perplexity I/O (loading a model and scoring the fixed
text) is intentionally **not** performed here; the caller measures the
surrogate elsewhere and passes the results in via ``local_surrogates``. This
keeps the module importable on its own with stdlib-only imports and unit/
property testable without any measurement harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

# Minimum number of successful Run_Repeats required for error-bar-grade
# reporting of a locally measured surrogate. Below this the value is flagged
# insufficient and reported without a standard deviation (R4.7).
MIN_QUALITY_REPEATS = 5

QualityLabel = Literal["cited", "locally-measured"]

# Citation reference identifying the cited quant-eval study. Each cited value's
# ``source`` field embeds this string plus the specific value used (R4.2).
CITED_STUDY = "arXiv:2601.14277"
CITED_CITATION = (
    "Kurt et al., \"A Systematic Evaluation of GGUF Quantization Formats for "
    f"Edge LLM Serving\", {CITED_STUDY}"
)

# The unit of every quality value. The cited study and the local surrogate both
# report perplexity (lower is better) over a fixed text, so they share a unit
# and are directly comparable across formats.
QUALITY_UNIT = "perplexity"

# Static cited-quality table keyed by quant format (R4.2). Values are the
# illustrative wikitext-style perplexity figures reported by the cited study for
# the paper's reference model; lower is better, and perplexity rises as the
# quantization gets more aggressive. Formats absent from this table (e.g. a
# bespoke or newer quant) are handled as missing by the cited path (R4.6).
CITED_QUALITY_TABLE: dict[str, float] = {
    "F16": 6.14,
    "Q8_0": 6.15,
    "Q6_K": 6.17,
    "Q5_K_M": 6.21,
    "Q5_K_S": 6.24,
    "Q5_0": 6.28,
    "Q4_K_M": 6.36,
    "Q4_K_S": 6.43,
    "Q4_0": 6.58,
    "Q3_K_L": 6.78,
    "Q3_K_M": 6.95,
    "Q3_K_S": 7.34,
    "Q2_K": 8.06,
}


@dataclass(frozen=True)
class LocalSurrogate:
    """A locally measured Quality_Surrogate for one quant format (R4.3, R4.4).

    The caller measures perplexity over the campaign-fixed input set across at
    least one Run_Repeat and passes the aggregated result here.

    Attributes:
        mean: Mean surrogate value over the successful Run_Repeats.
        std: Standard deviation over the successful Run_Repeats, or ``None`` when
            it is not reportable (fewer than 5 successful repeats, R4.7).
        n_success: Number of successful Run_Repeats that fed the statistics.
    """

    mean: float
    n_success: int
    std: float | None = None


@dataclass(frozen=True)
class QualityValue:
    """One labeled quality value for a single quant format (R4.1, R4.5).

    Attributes:
        quant_format: The quant format this value describes (e.g. ``"Q4_K_M"``).
        value: The numeric quality value, or ``None`` when the value is missing
            (cited study has no entry for the format, R4.6).
        label: ``"cited"`` or ``"locally-measured"``, matching the value's source
            (R4.5). A missing value still carries the label of the source that
            was consulted.
        source: A human-readable provenance string: the citation reference for a
            cited value, or a surrogate description for a locally measured value.
        missing: ``True`` iff no value could be supplied for this format from the
            selected source (R4.6).
        unit: The unit of ``value`` (perplexity).
        std: Standard deviation for a locally measured value; ``None`` for cited
            values and for local values flagged as insufficient (R4.7).
        insufficient_repeats: ``True`` iff this is a locally measured value with
            fewer than 5 successful Run_Repeats (R4.7).
    """

    quant_format: str
    value: float | None
    label: QualityLabel
    source: str
    missing: bool = False
    unit: str = QUALITY_UNIT
    std: float | None = None
    insufficient_repeats: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable dict."""
        return {
            "quant_format": self.quant_format,
            "value": self.value,
            "label": self.label,
            "source": self.source,
            "missing": self.missing,
            "unit": self.unit,
            "std": self.std,
            "insufficient_repeats": self.insufficient_repeats,
        }


def _coerce_surrogate(raw: Any) -> LocalSurrogate | None:
    """Coerce a caller-supplied surrogate entry into a :class:`LocalSurrogate`.

    Accepts a :class:`LocalSurrogate`, a plain numeric mean (with unknown
    repeat count, treated as insufficient), or ``None`` (no measurement). This
    keeps the public API forgiving while the measurement wiring (added later)
    stays free to pass richer objects.
    """
    if raw is None:
        return None
    if isinstance(raw, LocalSurrogate):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        # A bare number carries no repeat count; treat it as a single
        # observation so it is reported without a standard deviation (R4.7).
        return LocalSurrogate(mean=float(raw), n_success=1, std=None)
    raise TypeError(
        "local surrogate must be a LocalSurrogate, a number, or None; "
        f"got {type(raw).__name__}"
    )


def _cited_value(quant_format: str) -> QualityValue:
    """Build the cited :class:`QualityValue` for one quant format (R4.2, R4.6)."""
    if quant_format in CITED_QUALITY_TABLE:
        value = CITED_QUALITY_TABLE[quant_format]
        return QualityValue(
            quant_format=quant_format,
            value=value,
            label="cited",
            source=f"{CITED_CITATION}; {quant_format} {QUALITY_UNIT}={value}",
            missing=False,
        )
    # Format absent from the cited study: record as missing, indicate the
    # omission, and let the caller continue with the remaining formats (R4.6).
    return QualityValue(
        quant_format=quant_format,
        value=None,
        label="cited",
        source=f"{CITED_STUDY} has no quality value for {quant_format}",
        missing=True,
    )


def _local_value(
    quant_format: str,
    surrogate: LocalSurrogate | None,
) -> QualityValue:
    """Build the locally measured :class:`QualityValue` for one format (R4.3, R4.7)."""
    if surrogate is None:
        # Local measurement was enabled but no surrogate was produced for this
        # format (e.g. every repeat failed). Record it as missing without
        # aborting the remaining formats.
        return QualityValue(
            quant_format=quant_format,
            value=None,
            label="locally-measured",
            source=f"no Quality_Surrogate measured for {quant_format}",
            missing=True,
        )

    insufficient = surrogate.n_success < MIN_QUALITY_REPEATS
    # When there are insufficient successful repeats, report without a standard
    # deviation regardless of any std the caller may have attached (R4.7).
    std = None if insufficient else surrogate.std
    source = (
        f"Quality_Surrogate ({QUALITY_UNIT}) over campaign-fixed input set, "
        f"n_success={surrogate.n_success}"
    )
    if insufficient:
        source += " (insufficient repeats: <5)"
    return QualityValue(
        quant_format=quant_format,
        value=surrogate.mean,
        label="locally-measured",
        source=source,
        missing=False,
        std=std,
        insufficient_repeats=insufficient,
    )


def quality_values(
    formats: Iterable[str],
    local_quality: bool,
    local_surrogates: Mapping[str, Any] | None = None,
) -> list[QualityValue]:
    """Supply exactly one labeled quality value per quant format (R4.1-R4.7).

    For each format in ``formats`` (in the order given, de-duplicated so each
    format yields exactly one value), this returns a :class:`QualityValue`:

    - When ``local_quality`` is ``False``, the value is looked up in
      :data:`CITED_QUALITY_TABLE` and labeled ``"cited"`` with a citation
      reference; a format absent from the table is marked ``missing`` without
      aborting the rest (R4.2, R4.6).
    - When ``local_quality`` is ``True``, the value is taken from
      ``local_surrogates[format]`` and labeled ``"locally-measured"``; a format
      with fewer than 5 successful repeats is flagged ``insufficient_repeats``
      and reported without a standard deviation, and a format with no surrogate
      is marked ``missing`` (R4.3, R4.7).

    Args:
        formats: The quant formats in the campaign Config_Grid.
        local_quality: Whether local quality measurement is enabled.
        local_surrogates: Mapping of quant format to its measured surrogate
            (a :class:`LocalSurrogate`, a numeric mean, or ``None``). Used only
            when ``local_quality`` is ``True``; ignored otherwise.

    Returns:
        One :class:`QualityValue` per distinct format, in first-seen order.
    """
    surrogates = local_surrogates or {}

    results: list[QualityValue] = []
    seen: set[str] = set()
    for quant_format in formats:
        if quant_format in seen:
            # Guarantee exactly one value per format (R4.1).
            continue
        seen.add(quant_format)

        if local_quality:
            surrogate = _coerce_surrogate(surrogates.get(quant_format))
            results.append(_local_value(quant_format, surrogate))
        else:
            results.append(_cited_value(quant_format))

    return results


__all__ = [
    "MIN_QUALITY_REPEATS",
    "CITED_STUDY",
    "CITED_CITATION",
    "CITED_QUALITY_TABLE",
    "QUALITY_UNIT",
    "QualityLabel",
    "LocalSurrogate",
    "QualityValue",
    "quality_values",
]
