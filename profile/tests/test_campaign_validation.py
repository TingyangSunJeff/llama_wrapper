"""Property test for validate_campaign (profile-suite Property 14)."""

from __future__ import annotations

import os
import tempfile

from hypothesis import given, settings

from profile_suite.campaign import CampaignConfig, MODULE_NAMES
from profile_suite.config import GridSpec
from profile_suite.validation import validate_campaign
from tests.conftest import campaign_defs


def _snapshot(root: str) -> set[str]:
    """Return the set of every file and directory path under ``root``.

    Used to prove the validator has no filesystem side effects: the snapshot
    before and after validation must be identical (no campaign output created).
    """
    entries: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            entries.add(os.path.join(dirpath, name))
        for name in filenames:
            entries.add(os.path.join(dirpath, name))
    return entries


# Feature: profile-suite, Property 14: Campaign validation accepts iff valid and otherwise declines with a reason and no output
@settings(max_examples=100, deadline=None)
@given(campaign_defs())
def test_campaign_validation_accepts_iff_valid(desc: dict) -> None:
    """Validates: Requirements 6.1, 6.2, 8.2, 8.4, 8.6

    For any campaign definition, validate_campaign accepts (ok=True) if and only
    if the campaign is valid — a recognized single Platform, an existing server
    binary and model files, a non-empty Config_Grid, a Run_Repeat count >= 1,
    only known module names, and a bounds-valid Decode_Batch_Size set. Otherwise
    it declines (ok=False) with a specific non-empty reason and, for path
    failures, names the missing path in ``missing_paths``. In all cases the
    validator is side-effect-free: it creates no campaign output.
    """
    defect = desc["defect"]

    with tempfile.TemporaryDirectory() as tmp:
        # Real server binary + model files so existence checks pass for the
        # valid case and for every non-path defect.
        server_binary = os.path.join(tmp, "llama-server")
        with open(server_binary, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\n")

        model_paths: list[str] = []
        for i in range(desc["n_quants"]):
            model_path = os.path.join(tmp, f"model_{i}.gguf")
            with open(model_path, "wb") as fh:
                fh.write(b"GGUF")
            model_paths.append(model_path)

        # Default-valid field values; a single defect overrides exactly one.
        platform = desc["platform"]
        run_repeats: int = desc["run_repeats"]
        modules: tuple[str, ...] = tuple(MODULE_NAMES)
        decode_batches: tuple[int, ...] = (1, 2, 4, 8)
        missing_path: str | None = None
        empty_grid = False

        if defect == "unknown_platform":
            platform = desc["bad_platform"]
        elif defect == "missing_server_binary":
            server_binary = os.path.join(tmp, "nonexistent-llama-server")
            missing_path = server_binary
        elif defect == "missing_model_file":
            missing_path = os.path.join(tmp, "missing_model.gguf")
            model_paths = model_paths + [missing_path]
        elif defect == "empty_grid":
            empty_grid = True
        elif defect == "run_repeats_too_low":
            run_repeats = desc["bad_run_repeats"]
        elif defect == "unknown_module":
            modules = tuple(MODULE_NAMES) + (desc["bad_module"],)
        elif defect == "decode_batch_out_of_range":
            decode_batches = (1, 2, desc["out_of_range_batch"])

        if empty_grid:
            grid = GridSpec(quant_files=(), ctx_lengths=(), slot_counts=())
        else:
            grid = GridSpec(
                quant_files=tuple(model_paths),
                ctx_lengths=desc["ctx_lengths"],
                slot_counts=desc["slot_counts"],
            )

        cfg = CampaignConfig(
            platform=platform,
            server_binary=server_binary,
            bench_binary=os.path.join(tmp, "llama-bench"),
            batched_bench_binary=os.path.join(tmp, "llama-batched-bench"),
            model_dir=tmp,
            gpu_index=0,
            config_grid=grid,
            prompt_tokens=512,
            output_tokens=128,
            decode_batch_sizes=decode_batches,
            run_repeats=run_repeats,
            enabled_modules=modules,
        )

        before = _snapshot(tmp)
        result = validate_campaign(cfg)
        after = _snapshot(tmp)

        # No side effects / no output: the validator only stats referenced
        # paths; it must create, modify, or delete nothing.
        assert before == after

        is_valid = defect is None

        # Accepts iff valid.
        assert result.ok is is_valid

        if is_valid:
            assert result.reason is None
            assert result.missing_paths == []
        else:
            # Declines with a specific, non-empty reason.
            assert isinstance(result.reason, str) and result.reason.strip()
            if missing_path is not None:
                # Path failures name the missing path.
                assert missing_path in result.missing_paths
            else:
                # Non-path failures report no missing paths (the path-existence
                # checks run last and are short-circuited).
                assert result.missing_paths == []
