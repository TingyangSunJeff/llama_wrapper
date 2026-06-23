"""Property test for apply_defaults / CampaignConfig (profile-suite Property 16)."""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.campaign import (
    CANONICAL_DECODE_BATCH_SIZES,
    DEFAULT_BOOT_TIMEOUT_S,
    DEFAULT_LOCAL_QUALITY,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_RUN_REPEATS,
    DEFAULT_WARMUP_TIMEOUT_S,
    MODULE_NAMES,
    apply_defaults,
)
from profile_suite.config import GridSpec

# --------------------------------------------------------------------------- #
# The optional settings that carry a canonical default and are recorded in the
# Run_Manifest when omitted (R8.3). Each entry maps the setting name to:
#   - the canonical default recorded in applied_defaults (JSON-friendly form)
#   - the canonical default as it appears on the resulting CampaignConfig
#   - a coercion turning a supplied raw value into its CampaignConfig form
# memory_budget_mib is intentionally excluded: it has no canonical default and
# is never recorded as an applied default (its absence simply resolves to None).
# --------------------------------------------------------------------------- #
_DEFAULTED_OPTIONALS: dict[str, dict[str, Any]] = {
    "run_repeats": {
        "applied": DEFAULT_RUN_REPEATS,
        "config": DEFAULT_RUN_REPEATS,
        "coerce": int,
    },
    "decode_batch_sizes": {
        "applied": list(CANONICAL_DECODE_BATCH_SIZES),
        "config": CANONICAL_DECODE_BATCH_SIZES,
        "coerce": lambda v: tuple(int(b) for b in v),
    },
    "enabled_modules": {
        "applied": list(MODULE_NAMES),
        "config": MODULE_NAMES,
        "coerce": lambda v: tuple(str(m) for m in v),
    },
    "local_quality": {
        "applied": DEFAULT_LOCAL_QUALITY,
        "config": DEFAULT_LOCAL_QUALITY,
        "coerce": bool,
    },
    "boot_timeout_s": {
        "applied": DEFAULT_BOOT_TIMEOUT_S,
        "config": DEFAULT_BOOT_TIMEOUT_S,
        "coerce": float,
    },
    "warmup_timeout_s": {
        "applied": DEFAULT_WARMUP_TIMEOUT_S,
        "config": DEFAULT_WARMUP_TIMEOUT_S,
        "coerce": float,
    },
    "max_attempts": {
        "applied": DEFAULT_MAX_ATTEMPTS,
        "config": DEFAULT_MAX_ATTEMPTS,
        "coerce": int,
    },
}

# A raw-value strategy for each defaulted optional setting (the value a campaign
# author might explicitly supply). Values are chosen to be valid but otherwise
# arbitrary so the "supplied is preserved" branch is exercised.
_SUPPLIED_VALUE_STRATEGIES: dict[str, st.SearchStrategy[Any]] = {
    "run_repeats": st.integers(min_value=1, max_value=20),
    "decode_batch_sizes": st.lists(
        st.integers(min_value=1, max_value=128), min_size=1, max_size=8
    ),
    "enabled_modules": st.lists(
        st.sampled_from(MODULE_NAMES), min_size=1, max_size=6, unique=True
    ),
    "local_quality": st.booleans(),
    "boot_timeout_s": st.floats(
        min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
    ),
    "warmup_timeout_s": st.floats(
        min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
    ),
    "max_attempts": st.integers(min_value=1, max_value=50),
}


@st.composite
def partial_campaign_specs(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a partial campaign spec dict.

    The dict always carries the nine required fields and a random subset of the
    optional settings (each independently present or omitted). When present, an
    optional setting carries an explicitly-supplied raw value.
    """
    spec: dict[str, Any] = {
        "platform": draw(st.sampled_from(["a100-cuda", "jetson-orin"])),
        "server_binary": "/bin/llama-server",
        "bench_binary": "/bin/llama-bench",
        "batched_bench_binary": "/bin/llama-batched-bench",
        "model_dir": "/models",
        "gpu_index": draw(st.integers(min_value=0, max_value=7)),
        "config_grid": GridSpec(
            quant_files=("q4_k_m.gguf",),
            ctx_lengths=(4096,),
            slot_counts=(1,),
        ),
        "prompt_tokens": draw(st.integers(min_value=1, max_value=4096)),
        "output_tokens": draw(st.integers(min_value=1, max_value=4096)),
    }

    # Independently decide present/omitted for each defaulted optional setting.
    for name, value_strategy in _SUPPLIED_VALUE_STRATEGIES.items():
        if draw(st.booleans()):
            spec[name] = draw(value_strategy)

    # memory_budget_mib: optional with no canonical default; sometimes supplied.
    budget = draw(
        st.one_of(
            st.none(),
            st.floats(
                min_value=1.0,
                max_value=1.0e6,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    if draw(st.booleans()):
        spec["memory_budget_mib"] = budget

    return spec


# Feature: profile-suite, Property 16: Defaults are applied for omitted optional settings and recorded in the manifest
@settings(max_examples=100)
@given(partial_campaign_specs())
def test_defaults_applied_for_omitted_and_recorded(spec: dict[str, Any]) -> None:
    """Validates: Requirements 8.3

    For any partial campaign spec, every omitted defaulted optional setting is
    filled with its canonical default on the resulting CampaignConfig AND recorded
    in applied_defaults with that value; every explicitly-supplied optional setting
    is absent from applied_defaults and preserved (coerced) on the CampaignConfig.
    The canonical decode-batch default is exactly (1,2,4,8,16,32,64,128) and all
    six modules are enabled when omitted.
    """
    config, applied = apply_defaults(spec)

    for name, info in _DEFAULTED_OPTIONALS.items():
        config_value = getattr(config, name)
        if name in spec:
            # Explicitly supplied: not recorded as a default, preserved on config.
            assert name not in applied
            expected = info["coerce"](spec[name])
            assert config_value == expected
        else:
            # Omitted: filled with the canonical default AND recorded.
            assert applied[name] == info["applied"]
            assert config_value == info["config"]

    # The canonical decode-batch default is exactly this set, and all six modules
    # are enabled, whenever each is omitted (R2.2, R8.3).
    if "decode_batch_sizes" not in spec:
        assert config.decode_batch_sizes == (1, 2, 4, 8, 16, 32, 64, 128)
        assert applied["decode_batch_sizes"] == [1, 2, 4, 8, 16, 32, 64, 128]
    if "enabled_modules" not in spec:
        assert len(config.enabled_modules) == 6
        assert config.enabled_modules == MODULE_NAMES

    # memory_budget_mib has no canonical default: it is never recorded as applied,
    # and resolves to the supplied float or None.
    assert "memory_budget_mib" not in applied
    raw_budget = spec.get("memory_budget_mib", None)
    if raw_budget is None:
        assert config.memory_budget_mib is None
    else:
        assert config.memory_budget_mib == float(raw_budget)
