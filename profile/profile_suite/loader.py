"""Campaign file loader.

This module turns a hand-authored campaign definition file into a fully-resolved
:class:`~profile_suite.campaign.CampaignConfig`. It is the single entry point the
orchestrator uses to read a campaign off disk (see design.md "Data Models" ->
"Campaign definition").

The chosen input format is a **YAML file** (human-authored and diff-friendly);
**JSON is accepted too** (same loader) for machine-generated campaigns. The file
extension selects the parser: ``.yaml`` / ``.yml`` -> YAML, ``.json`` -> JSON.

Loading proceeds in three steps:

1. **Parse** the file into a partial spec ``dict`` (the optional settings a
   campaign omits are simply absent).
2. **Resolve model files**: each entry of ``config_grid.quant_files`` is resolved
   against ``model_dir`` into an absolute ``quant_file`` path, so the rest of the
   suite (and validation) deals only in absolute paths. Entries that are already
   absolute are normalized but left as-is.
3. **Apply defaults** via :func:`profile_suite.campaign.apply_defaults`, filling
   omitted optional settings (R8.3) and producing the frozen ``CampaignConfig``.

This module performs **no validation** of path existence, platform recognition,
grid non-emptiness, etc. — that is the job of
:func:`profile_suite.validation.validate_campaign`, which the orchestrator runs on
the loaded config before starting a campaign (R6.2, R8.4, R8.6). Keeping load and
validate separate means a campaign can be loaded, inspected, and validated without
side effects.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import yaml

from .campaign import CampaignConfig, apply_defaults

#: File extensions parsed as YAML.
_YAML_EXTENSIONS = (".yaml", ".yml")
#: File extensions parsed as JSON.
_JSON_EXTENSIONS = (".json",)


def _parse_file(path: str) -> dict[str, Any]:
    """Parse ``path`` into a spec dict, selecting the parser by extension.

    Parameters
    ----------
    path:
        Path to a campaign definition file ending in ``.yaml``/``.yml``
        (parsed as YAML) or ``.json`` (parsed as JSON). YAML is a superset of
        JSON, but the extension is honored so the intended format is explicit.

    Returns
    -------
    dict
        The parsed top-level mapping.

    Raises
    ------
    ValueError
        If the extension is unsupported, or the parsed document is not a mapping.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    if ext in _YAML_EXTENSIONS:
        data = yaml.safe_load(text)
    elif ext in _JSON_EXTENSIONS:
        data = json.loads(text)
    else:
        raise ValueError(
            f"unsupported campaign file extension {ext!r} for {path!r}; "
            "expected one of .yaml, .yml, .json"
        )

    if data is None:
        raise ValueError(f"campaign file {path!r} is empty")
    if not isinstance(data, Mapping):
        raise ValueError(
            f"campaign file {path!r} must contain a mapping at the top level; "
            f"got {type(data).__name__}"
        )
    return dict(data)


def _resolve_quant_file(quant_file: str, model_dir: str | None) -> str:
    """Resolve a single quant/model file reference to an absolute path.

    An absolute reference is normalized and returned unchanged in meaning. A
    relative reference is joined against ``model_dir`` (when available) and made
    absolute, so the Config_Grid carries only absolute ``quant_file`` paths.
    """
    quant_file = str(quant_file)
    if os.path.isabs(quant_file):
        return os.path.normpath(quant_file)
    if model_dir:
        return os.path.abspath(os.path.join(str(model_dir), quant_file))
    # No model_dir to resolve against: normalize relative to the cwd so the
    # result is still absolute and validation can report a concrete path.
    return os.path.abspath(quant_file)


def _resolve_model_files(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``spec`` with ``config_grid.quant_files`` made absolute.

    Each quant file in ``config_grid.quant_files`` is resolved against
    ``model_dir`` into an absolute path (see :func:`_resolve_quant_file`). The
    input mapping is not mutated; nested ``config_grid`` mapping is copied. When
    ``config_grid`` is absent or not a mapping, or has no ``quant_files``, the
    spec is returned as a shallow copy unchanged.
    """
    resolved = dict(spec)
    grid = resolved.get("config_grid")
    if not isinstance(grid, Mapping):
        return resolved

    quant_files = grid.get("quant_files")
    if quant_files is None:
        return resolved

    model_dir = resolved.get("model_dir")
    resolved_grid = dict(grid)
    resolved_grid["quant_files"] = [
        _resolve_quant_file(qf, model_dir) for qf in quant_files
    ]
    resolved["config_grid"] = resolved_grid
    return resolved


def load_campaign(path: str) -> CampaignConfig:
    """Load a campaign definition file into a fully-resolved ``CampaignConfig``.

    Parameters
    ----------
    path:
        Path to a campaign file. ``.yaml``/``.yml`` is parsed as YAML and
        ``.json`` as JSON.

    Returns
    -------
    CampaignConfig
        A frozen, fully-populated campaign config: model files in the
        Config_Grid are resolved to absolute paths against ``model_dir`` and all
        omitted optional settings are filled with their canonical defaults
        (R8.3). The config is **not** validated here; call
        :func:`profile_suite.validation.validate_campaign` before starting a
        campaign.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file extension is unsupported, the document is empty, the top
        level is not a mapping, or a required field is missing.
    """
    spec = _parse_file(path)
    spec = _resolve_model_files(spec)
    config, _applied_defaults = apply_defaults(spec)
    return config


__all__ = ["load_campaign"]
