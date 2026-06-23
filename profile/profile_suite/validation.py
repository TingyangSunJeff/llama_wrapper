"""Campaign validation: accept-or-decline with a specific reason.

Before any measurement runs, the orchestrator validates the loaded
:class:`~profile_suite.campaign.CampaignConfig`. Validation is **fail-fast**: a
campaign either passes every check and is accepted, or it is declined with a
single specific reason and **no campaign output is produced** (producing output
is the orchestrator's job and only happens for accepted campaigns).

:func:`validate_campaign` enforces the validity criteria of Property 14
(design.md "Correctness Properties"):

- a **recognized single Platform** descriptor (``a100-cuda`` or ``jetson-orin``)
  (R6.1, R6.2),
- an **existing server binary** on disk (naming it if missing) (R6.2, R8.4),
- **existing model files** for every Config in the expanded Config_Grid (naming
  any missing path) (R8.4),
- a **non-empty Config_Grid** (R8.2, R8.6),
- a **Run_Repeat count >= 1** (R8.6),
- **only known module names** in ``enabled_modules`` (R8.6),
- a **bounds-valid Decode_Batch_Size set** (every value in 1..128) (R2.2, R8.6).

The result is a structured :class:`ValidationResult`. On decline, ``ok`` is
``False``, ``reason`` is a specific human-readable message, and ``missing_paths``
lists every path that did not exist (empty for non-path failures). This module
has **no side effects**: it does not create, read, or write any campaign output;
it only stats the referenced paths to check existence.

_Requirements: 6.1, 6.2, 8.2, 8.4, 8.6_
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .campaign import CampaignConfig, MISSING, MODULE_NAMES, validate_decode_batches
from .config import expand_grid
from .platform.a100 import A100_DESCRIPTOR
from .platform.jetson import JETSON_DESCRIPTOR

#: The recognized Platform descriptors a campaign may target (R6.1). Drawn from
#: the concrete Platform adapters so this set stays in lock-step with them.
RECOGNIZED_PLATFORMS: tuple[str, ...] = (A100_DESCRIPTOR, JETSON_DESCRIPTOR)


@dataclass
class ValidationResult:
    """Structured accept/decline outcome of campaign validation.

    Attributes
    ----------
    ok:
        ``True`` iff the campaign passed every validity check and may start.
    reason:
        ``None`` when ``ok`` is ``True``; otherwise a specific, human-readable
        explanation of the first/aggregated validation failure.
    missing_paths:
        Every referenced path (server binary and/or model files) that did not
        exist on disk. Empty unless the decline was a path-existence failure.
    """

    ok: bool
    reason: str | None = None
    missing_paths: list[str] = field(default_factory=list)


def _accept() -> ValidationResult:
    return ValidationResult(ok=True, reason=None, missing_paths=[])


def _decline(reason: str, missing_paths: list[str] | None = None) -> ValidationResult:
    return ValidationResult(
        ok=False, reason=reason, missing_paths=list(missing_paths or [])
    )


def validate_campaign(
    cfg: CampaignConfig, platform: str | None = None
) -> ValidationResult:
    """Validate ``cfg`` and return a structured accept/decline result.

    Parameters
    ----------
    cfg:
        A fully-resolved :class:`CampaignConfig` (as produced by
        :func:`profile_suite.loader.load_campaign`), so model files in the
        Config_Grid are expected to be absolute paths.
    platform:
        Optional Platform descriptor to validate against. When provided it is the
        descriptor checked for recognition (and used as the campaign's effective
        Platform); when ``None`` the campaign's own ``cfg.platform`` is used.

    Returns
    -------
    ValidationResult
        ``ok=True`` with no reason when every check passes; otherwise
        ``ok=False`` with a specific ``reason`` (and ``missing_paths`` populated
        for path-existence failures). No campaign output is produced regardless
        of the outcome — that is the orchestrator's responsibility.

    Validates: Requirements 6.1, 6.2, 8.2, 8.4, 8.6 (Property 14).
    """
    # --- Platform recognized (R6.1, R6.2) ---------------------------------- #
    effective_platform = platform if platform is not None else cfg.platform
    if effective_platform is None or effective_platform is MISSING:
        return _decline(
            "campaign Platform descriptor is absent; expected one of "
            f"{', '.join(RECOGNIZED_PLATFORMS)}"
        )
    if effective_platform not in RECOGNIZED_PLATFORMS:
        return _decline(
            f"unrecognized Platform descriptor {effective_platform!r}; "
            f"expected one of {', '.join(RECOGNIZED_PLATFORMS)}"
        )

    # --- Non-empty Config_Grid (R8.2, R8.6) -------------------------------- #
    grid = expand_grid(cfg.config_grid)
    if not grid:
        return _decline(
            "Config_Grid is empty; a campaign must sweep at least one Config "
            "(check config_grid.quant_files / ctx_lengths / slot_counts)"
        )

    # --- Run_Repeat count >= 1 (R8.6) -------------------------------------- #
    run_repeats = cfg.run_repeats
    if run_repeats is MISSING:
        return _decline("run_repeats is not set; expected an integer >= 1")
    if int(run_repeats) < 1:
        return _decline(
            f"run_repeats must be >= 1; got {int(run_repeats)}"
        )

    # --- Only known module names (R8.6) ------------------------------------ #
    enabled_modules = cfg.enabled_modules
    if enabled_modules is MISSING:
        return _decline("enabled_modules is not set")
    unknown_modules = [m for m in enabled_modules if m not in MODULE_NAMES]
    if unknown_modules:
        return _decline(
            "unknown module name(s) in enabled_modules: "
            + ", ".join(repr(m) for m in unknown_modules)
            + "; known modules are "
            + ", ".join(MODULE_NAMES)
        )

    # --- Decode_Batch_Size bounds (R2.2, R8.6) ----------------------------- #
    decode_batch_sizes = cfg.decode_batch_sizes
    if decode_batch_sizes is MISSING or not validate_decode_batches(
        decode_batch_sizes
    ):
        return _decline(
            "decode_batch_sizes must contain only integers in the inclusive "
            f"range 1..128; got {decode_batch_sizes!r}"
        )

    # --- Existing server binary and model files (R6.2, R8.4) --------------- #
    # Aggregate every missing path so the report names all of them at once.
    missing_paths: list[str] = []

    server_binary = cfg.server_binary
    if not server_binary or not os.path.isfile(str(server_binary)):
        missing_paths.append(str(server_binary))

    seen_models: set[str] = set()
    for config in grid:
        model_path = str(config.quant_file)
        if model_path in seen_models:
            continue
        seen_models.add(model_path)
        if not os.path.isfile(model_path):
            missing_paths.append(model_path)

    if missing_paths:
        return _decline(
            "referenced path(s) do not exist: " + ", ".join(missing_paths),
            missing_paths=missing_paths,
        )

    return _accept()


__all__ = ["ValidationResult", "validate_campaign", "RECOGNIZED_PLATFORMS"]
