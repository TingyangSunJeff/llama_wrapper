"""Example test for the ``llama-batched-bench`` jsonl parser (Task 12.3).

Parses a captured-style ``llama-batched-bench --output-format jsonl`` stdout
fixture (a non-json banner line followed by three jsonl batch rows) and asserts
that :func:`profile_suite.modules.performance.parse_batched_bench_jsonl` returns
one :class:`~profile_suite.modules.performance.BatchedBenchRow` per jsonl record
(banner skipped), parses the per-batch fields correctly, and that
:func:`~profile_suite.modules.performance.batched_decode_throughput` reduces the
rows to the expected ``pl -> speed_tg`` decode-throughput map.

This is an example (non-property) test complementing the property-based coverage
of the pure per-batch winner logic.

Validates: Requirements 2.6
"""

from __future__ import annotations

import os

from profile_suite.modules.performance import (
    BatchedBenchRow,
    batched_decode_throughput,
    parse_batched_bench_jsonl,
)

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "llama_batched_bench.jsonl"
)


def _read_fixture() -> str:
    with open(_FIXTURE, "r", encoding="utf-8") as handle:
        return handle.read()


# Expected per-batch (pl -> speed_tg) decode throughput, read straight off the
# three jsonl rows in the fixture (the leading banner line is not json).
_EXPECTED_DECODE_THROUGHPUT = {
    1: 36.532974,
    2: 23.050371,
    4: 35.310345,
}


def test_parse_returns_one_row_per_jsonl_record_banner_skipped() -> None:
    """One :class:`BatchedBenchRow` per jsonl record; the banner line is skipped."""
    rows = parse_batched_bench_jsonl(_read_fixture())

    # The fixture has one non-json banner + three jsonl rows -> exactly 3 rows.
    assert len(rows) == 3
    assert all(isinstance(row, BatchedBenchRow) for row in rows)
    # Rows preserve first-seen order: pl = 1, 2, 4.
    assert [row.pl for row in rows] == [1, 2, 4]


def test_parse_per_batch_fields_parsed_correctly() -> None:
    """Each per-batch field is parsed with the correct type and value."""
    rows = parse_batched_bench_jsonl(_read_fixture())
    by_pl = {row.pl: row for row in rows}

    # pl == 1 row (first jsonl record).
    first = by_pl[1]
    assert first.pp == 128
    assert first.tg == 128
    assert first.n_kv == 256
    assert first.t_pp == 0.233810
    assert first.speed_pp == 547.453064
    assert first.t_tg == 3.503684
    assert first.speed_tg == 36.532974
    assert first.t == 3.737494
    assert first.speed == 68.495094

    # pl == 2 row carries its own distinct timings/speeds.
    second = by_pl[2]
    assert second.n_kv == 512
    assert second.speed_pp == 605.770935
    assert second.speed_tg == 23.050371

    # pl == 4 row.
    fourth = by_pl[4]
    assert fourth.n_kv == 1024
    assert fourth.speed_pp == 630.250000
    assert fourth.speed_tg == 35.310345

    # Integer fields are ints; float fields are floats.
    assert isinstance(first.pl, int)
    assert isinstance(first.n_kv, int)
    assert isinstance(first.speed_tg, float)


def test_batched_decode_throughput_map_matches_expected() -> None:
    """The ``pl -> speed_tg`` reduction matches the hand-read expected map."""
    rows = parse_batched_bench_jsonl(_read_fixture())
    throughput = batched_decode_throughput(rows)

    assert throughput == _EXPECTED_DECODE_THROUGHPUT


def test_non_json_and_unrelated_lines_are_skipped() -> None:
    """Banner / progress / markdown lines and key-less json objects are skipped."""
    noisy = (
        "main: loading model ...\n"
        "| PP | TG | B | N_KV | T_PP s |\n"  # markdown header (no leading '{')
        '{"note": "no pl or speed_tg here"}\n'  # json missing required keys
        + _read_fixture()
    )

    rows = parse_batched_bench_jsonl(noisy)

    # Still exactly the three real batch rows from the fixture.
    assert [row.pl for row in rows] == [1, 2, 4]
    assert batched_decode_throughput(rows) == _EXPECTED_DECODE_THROUGHPUT
