"""Campaign configuration model and default application.

A :class:`CampaignConfig` is the fully-resolved, typed description of one
profiling campaign (see design.md "Data Models" -> "Campaign definition").
It is loaded from a hand-authored YAML/JSON campaign file (handled elsewhere,
task 3.7) and frozen so it serves as a single deterministic validation and
serialization boundary.

This module also implements :func:`apply_defaults`, which fills the optional
settings a campaign definition omits (R8.3) and returns, alongside the
fully-populated config, a record of exactly which defaults were applied so the
Reproducibility_Harness can write them into the Run_Manifest.

Defaults applied for omitted optional settings (R8.3):

- ``run_repeats``        -> 5
- ``decode_batch_sizes`` -> the canonical set ``(1, 2, 4, 8, 16, 32, 64, 128)``
- ``enabled_modules``    -> all six modules
- ``boot_timeout_s``     -> 300.0 (R1.9)
- ``warmup_timeout_s``   -> 60.0  (R1.10)
- ``max_attempts``       -> 10    (R5.9)
- ``local_quality``      -> False (R4.2, cited-only by default)

``memory_budget_mib`` is genuinely optional (its absence simply disables the
feasibility frontier, R3.4); when omitted it resolves to ``None`` and is not
recorded as an applied default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import GridSpec

# --------------------------------------------------------------------------- #
# Canonical defaults (R8.3) and module set
# --------------------------------------------------------------------------- #

#: The six modules of the suite, in canonical order (see the Glossary / R8.2).
MODULE_NAMES: tuple[str, ...] = (
    "Switch_Cost_Profiler",
    "Performance_Profiler",
    "Memory_Profiler",
    "Quality_Module",
    "Reproducibility_Harness",
    "Reporting_Module",
)

#: Canonical default Decode_Batch_Size set applied when the campaign omits it
#: (R2.2, R8.3).
CANONICAL_DECODE_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

#: Inclusive bounds for a valid Decode_Batch_Size value (R2.2).
DECODE_BATCH_MIN: int = 1
DECODE_BATCH_MAX: int = 128

DEFAULT_RUN_REPEATS: int = 5
DEFAULT_BOOT_TIMEOUT_S: float = 300.0
DEFAULT_WARMUP_TIMEOUT_S: float = 60.0
DEFAULT_MAX_ATTEMPTS: int = 10
DEFAULT_LOCAL_QUALITY: bool = False


# --------------------------------------------------------------------------- #
# Decode_Batch_Size validation (R2.2, Property 5)
# --------------------------------------------------------------------------- #
def validate_decode_batches(values: Any) -> bool:
    """Return ``True`` iff every value is an int in the inclusive range 1..128.

    A candidate Decode_Batch_Size set is bounds-valid when each of its members
    is an integer lying within ``[DECODE_BATCH_MIN, DECODE_BATCH_MAX]`` (i.e.
    1..128 inclusive). The campaign's canonical default,
    :data:`CANONICAL_DECODE_BATCH_SIZES`, satisfies this by construction.

    Notes
    -----
    - Booleans are rejected: although ``bool`` is a subtype of ``int`` in
      Python, ``True``/``False`` are not meaningful batch sizes.
    - Non-iterable inputs are rejected (return ``False``) rather than raising,
      so the validator is total over arbitrary candidate inputs (Property 5).
    - The bound is purely a per-value range check; this validator does not
      assert the R2.2 "includes 1, 2, 4, ... 128" membership requirement, which
      is a separate concern handled where the set is assembled.

    Parameters
    ----------
    values:
        A candidate Decode_Batch_Size set (any iterable of values).

    Returns
    -------
    bool
        ``True`` if and only if every value is an integer in 1..128 inclusive.

    Validates: Requirements 2.2, 8.3 (Property 5).
    """
    try:
        iterator = iter(values)
    except TypeError:
        return False
    for value in iterator:
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        if value < DECODE_BATCH_MIN or value > DECODE_BATCH_MAX:
            return False
    return True


class _Missing:
    """Sentinel marking an optional setting that the partial spec omitted.

    A single shared instance, :data:`MISSING`, is used so a partial
    :class:`CampaignConfig` (or a dict spec) can leave optional fields explicitly
    unset and have :func:`apply_defaults` fill them. ``MISSING`` is falsy and
    reprs clearly for debugging.
    """

    _instance: "_Missing | None" = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "MISSING"

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return False


#: Shared sentinel for an omitted optional setting.
MISSING: _Missing = _Missing()

# The required fields that have no default; a fully-populated CampaignConfig
# cannot be built without them.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "platform",
    "server_binary",
    "bench_binary",
    "batched_bench_binary",
    "model_dir",
    "gpu_index",
    "config_grid",
    "prompt_tokens",
    "output_tokens",
)


@dataclass(frozen=True)
class CampaignConfig:
    """A fully-resolved campaign definition.

    Field semantics follow design.md "Data Models". Optional settings that a
    campaign file may omit are filled by :func:`apply_defaults`; the optional
    fields below carry the :data:`MISSING` sentinel as their default so a
    partial ``CampaignConfig`` can be constructed and then completed.
    """

    # Required (no default) -------------------------------------------------- #
    platform: str
    server_binary: str
    bench_binary: str
    batched_bench_binary: str
    model_dir: str
    gpu_index: int
    config_grid: GridSpec
    prompt_tokens: int
    output_tokens: int

    # Optional (defaulted by apply_defaults; sentinel until then) ------------ #
    decode_batch_sizes: tuple[int, ...] | _Missing = MISSING
    run_repeats: int | _Missing = MISSING
    enabled_modules: tuple[str, ...] | _Missing = MISSING
    memory_budget_mib: float | None | _Missing = MISSING
    local_quality: bool | _Missing = MISSING
    boot_timeout_s: float | _Missing = MISSING
    warmup_timeout_s: float | _Missing = MISSING
    max_attempts: int | _Missing = MISSING


# --------------------------------------------------------------------------- #
# Default application (R8.3)
# --------------------------------------------------------------------------- #
def _as_grid_spec(value: Any) -> GridSpec:
    """Coerce a config_grid value (GridSpec or mapping) into a GridSpec."""
    if isinstance(value, GridSpec):
        return value
    if isinstance(value, Mapping):
        return GridSpec(
            quant_files=tuple(value.get("quant_files", ())),
            ctx_lengths=tuple(value.get("ctx_lengths", ())),
            slot_counts=tuple(value.get("slot_counts", ())),
        )
    raise TypeError(
        "config_grid must be a GridSpec or a mapping with quant_files/"
        f"ctx_lengths/slot_counts; got {type(value).__name__}"
    )


def _provided(spec: "CampaignConfig | Mapping[str, Any]", key: str) -> Any:
    """Return the provided value for ``key`` or :data:`MISSING` if omitted.

    Accepts either a mapping (a partial spec dict, where an absent key or a
    ``MISSING`` value counts as omitted) or a :class:`CampaignConfig` carrying
    sentinel values for its omitted optional fields.
    """
    if isinstance(spec, CampaignConfig):
        value = getattr(spec, key, MISSING)
    elif isinstance(spec, Mapping):
        value = spec.get(key, MISSING)
    else:  # pragma: no cover - defensive
        raise TypeError(
            "spec must be a CampaignConfig or a mapping; "
            f"got {type(spec).__name__}"
        )
    return value


def apply_defaults(
    spec: "CampaignConfig | Mapping[str, Any]",
) -> tuple[CampaignConfig, dict[str, Any]]:
    """Fill omitted optional settings and record which defaults were applied.

    Parameters
    ----------
    spec:
        A *partial* campaign specification. Either a mapping (e.g. the dict a
        YAML/JSON loader produced) whose omitted optional keys are absent or set
        to :data:`MISSING`, or a :class:`CampaignConfig` whose optional fields
        carry the :data:`MISSING` sentinel. The required fields
        (:data:`_REQUIRED_FIELDS`) must be present.

    Returns
    -------
    tuple(CampaignConfig, dict)
        ``(config, applied_defaults)`` where ``config`` is a fully-populated,
        frozen :class:`CampaignConfig` (no remaining sentinels) and
        ``applied_defaults`` maps each setting name that fell back to its
        default to the JSON-friendly value that was applied (R8.3). Settings the
        caller supplied explicitly do not appear in ``applied_defaults``.

    Raises
    ------
    ValueError
        If any required field is missing from ``spec``.
    """
    missing_required = [
        key for key in _REQUIRED_FIELDS if _provided(spec, key) is MISSING
    ]
    if missing_required:
        raise ValueError(
            "campaign spec is missing required field(s): "
            + ", ".join(missing_required)
        )

    applied: dict[str, Any] = {}

    # --- decode_batch_sizes ------------------------------------------------- #
    raw_batches = _provided(spec, "decode_batch_sizes")
    if raw_batches is MISSING:
        decode_batch_sizes = CANONICAL_DECODE_BATCH_SIZES
        applied["decode_batch_sizes"] = list(CANONICAL_DECODE_BATCH_SIZES)
    else:
        decode_batch_sizes = tuple(int(b) for b in raw_batches)

    # --- run_repeats -------------------------------------------------------- #
    raw_repeats = _provided(spec, "run_repeats")
    if raw_repeats is MISSING:
        run_repeats = DEFAULT_RUN_REPEATS
        applied["run_repeats"] = DEFAULT_RUN_REPEATS
    else:
        run_repeats = int(raw_repeats)

    # --- enabled_modules ---------------------------------------------------- #
    raw_modules = _provided(spec, "enabled_modules")
    if raw_modules is MISSING:
        enabled_modules = MODULE_NAMES
        applied["enabled_modules"] = list(MODULE_NAMES)
    else:
        enabled_modules = tuple(str(m) for m in raw_modules)

    # --- memory_budget_mib (optional; absence => None, not a recorded default) #
    raw_budget = _provided(spec, "memory_budget_mib")
    memory_budget_mib: float | None
    if raw_budget is MISSING or raw_budget is None:
        memory_budget_mib = None
    else:
        memory_budget_mib = float(raw_budget)

    # --- local_quality ------------------------------------------------------ #
    raw_local_quality = _provided(spec, "local_quality")
    if raw_local_quality is MISSING:
        local_quality = DEFAULT_LOCAL_QUALITY
        applied["local_quality"] = DEFAULT_LOCAL_QUALITY
    else:
        local_quality = bool(raw_local_quality)

    # --- boot_timeout_s ----------------------------------------------------- #
    raw_boot = _provided(spec, "boot_timeout_s")
    if raw_boot is MISSING:
        boot_timeout_s = DEFAULT_BOOT_TIMEOUT_S
        applied["boot_timeout_s"] = DEFAULT_BOOT_TIMEOUT_S
    else:
        boot_timeout_s = float(raw_boot)

    # --- warmup_timeout_s --------------------------------------------------- #
    raw_warmup = _provided(spec, "warmup_timeout_s")
    if raw_warmup is MISSING:
        warmup_timeout_s = DEFAULT_WARMUP_TIMEOUT_S
        applied["warmup_timeout_s"] = DEFAULT_WARMUP_TIMEOUT_S
    else:
        warmup_timeout_s = float(raw_warmup)

    # --- max_attempts ------------------------------------------------------- #
    raw_attempts = _provided(spec, "max_attempts")
    if raw_attempts is MISSING:
        max_attempts = DEFAULT_MAX_ATTEMPTS
        applied["max_attempts"] = DEFAULT_MAX_ATTEMPTS
    else:
        max_attempts = int(raw_attempts)

    config = CampaignConfig(
        platform=str(_provided(spec, "platform")),
        server_binary=str(_provided(spec, "server_binary")),
        bench_binary=str(_provided(spec, "bench_binary")),
        batched_bench_binary=str(_provided(spec, "batched_bench_binary")),
        model_dir=str(_provided(spec, "model_dir")),
        gpu_index=int(_provided(spec, "gpu_index")),
        config_grid=_as_grid_spec(_provided(spec, "config_grid")),
        prompt_tokens=int(_provided(spec, "prompt_tokens")),
        output_tokens=int(_provided(spec, "output_tokens")),
        decode_batch_sizes=decode_batch_sizes,
        run_repeats=run_repeats,
        enabled_modules=enabled_modules,
        memory_budget_mib=memory_budget_mib,
        local_quality=local_quality,
        boot_timeout_s=boot_timeout_s,
        warmup_timeout_s=warmup_timeout_s,
        max_attempts=max_attempts,
    )
    return config, applied
