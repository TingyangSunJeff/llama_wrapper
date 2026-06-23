"""Property test for ``ReproHarness.run_point`` (Property 11).

Validates that the run loop discards exactly one warmup run, obeys the retry
stopping rule (stop at the first point with ``min_success`` successful retained
repeats *or* ``max_attempts`` total retained attempts), classifies the point status
from the success count, and aggregates over the successful retained repeats only.

**Validates: Requirements 5.4, 5.9**
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from profile_suite.config import Config
from profile_suite.harness.repro import ReproHarness
from profile_suite.results import RunRepeatResult

# The run loop, with the spec's stopping rule, can call ``measure_fn`` for the
# warmup (index 0) plus at most ``MAX_ATTEMPTS`` retained attempts (indices
# 1..MAX_ATTEMPTS). Generating exactly that many outcomes guarantees a defined
# outcome for every call the loop can make.
MIN_SUCCESS = 5
MAX_ATTEMPTS = 10

_CONFIG = Config(quant_file="q4_k_m.gguf", ctx_length=4096, slot_count=1)


def outcome_sequences() -> st.SearchStrategy[list[bool]]:
    """Per-attempt success/failure outcomes: index 0 is the warmup, 1.. retained."""
    return st.lists(
        st.booleans(),
        min_size=MAX_ATTEMPTS + 1,
        max_size=MAX_ATTEMPTS + 1,
    )


def _make_measure_fn(outcomes: list[bool]):
    """Build a fake ``measure_fn`` returning a RunRepeatResult per outcome.

    The run at ``run_index`` succeeds iff ``outcomes[run_index]`` is True, and
    carries a metric ``value`` equal to ``run_index`` so aggregation over the
    successful retained repeats is independently checkable.
    """

    def measure_fn(run_index: int) -> RunRepeatResult:
        ok = outcomes[run_index]
        return RunRepeatResult(
            run_index=run_index,
            discarded_warmup=False,
            ok=ok,
            raw_log_path=f"run{run_index:02d}.log",
            metrics={"value": float(run_index)},
            error=None if ok else f"failure at run {run_index}",
        )

    return measure_fn


def _simulate(outcomes: list[bool]) -> tuple[int, int]:
    """Independently replay the stopping rule over the retained outcomes.

    Returns ``(attempts, successes)`` where ``attempts`` is the number of retained
    repeats executed and ``successes`` the number of those that succeeded.
    """
    successes = 0
    attempts = 0
    run_index = 1
    while successes < MIN_SUCCESS and attempts < MAX_ATTEMPTS:
        if outcomes[run_index]:
            successes += 1
        attempts += 1
        run_index += 1
    return attempts, successes


# Feature: profile-suite, Property 11: The run loop discards one warmup and obeys the retry stopping rule
@settings(max_examples=100)
@given(outcomes=outcome_sequences())
def test_run_loop_discards_warmup_and_obeys_stopping_rule(outcomes: list[bool]) -> None:
    harness = ReproHarness()
    point = harness.run_point(
        _make_measure_fn(outcomes),
        min_success=MIN_SUCCESS,
        max_attempts=MAX_ATTEMPTS,
        config=_CONFIG,
    )

    expected_attempts, expected_successes = _simulate(outcomes)

    # --- Exactly one discarded warmup, excluded from aggregates (R5.4) ----- #
    discarded = [r for r in point.repeats if r.discarded_warmup]
    assert len(discarded) == 1
    assert point.repeats[0].discarded_warmup is True
    # The warmup is run index 0 and every other repeat is a retained repeat.
    assert point.repeats[0].run_index == 0
    assert all(not r.discarded_warmup for r in point.repeats[1:])

    # --- Stopping rule: retained attempts match the simulated stopping point  #
    retained = point.repeats[1:]
    assert len(retained) == expected_attempts
    # The loop stops at the first point with >=5 successes OR 10 attempts.
    assert expected_successes <= MIN_SUCCESS
    assert expected_attempts <= MAX_ATTEMPTS
    if expected_successes < MIN_SUCCESS:
        # Stopping early was not possible: it must have exhausted the budget.
        assert expected_attempts == MAX_ATTEMPTS
    if expected_attempts < MAX_ATTEMPTS:
        # It stopped before the budget: must have hit the success threshold.
        assert expected_successes == MIN_SUCCESS

    actual_successes = sum(1 for r in retained if r.ok)
    assert actual_successes == expected_successes

    # --- Status classification (R5.9) -------------------------------------- #
    if expected_successes >= MIN_SUCCESS:
        assert point.status == "complete"
    elif expected_successes == 0:
        assert point.status == "failed"
    else:
        assert point.status == "incomplete"

    # --- Aggregates over successful retained repeats only ------------------ #
    successful_values = [
        r.metrics["value"] for r in retained if r.ok
    ]
    if successful_values:
        agg = point.aggregates["value"]
        assert agg.n_success == len(successful_values)
        expected_mean = sum(successful_values) / len(successful_values)
        assert math.isclose(agg.mean, expected_mean, rel_tol=1e-9, abs_tol=1e-9)
        # The discarded warmup (run index 0, value 0.0) never enters the aggregate.
        assert 0.0 not in successful_values
    else:
        # No successful retained repeats: no aggregate is reported for the metric.
        assert "value" not in point.aggregates
