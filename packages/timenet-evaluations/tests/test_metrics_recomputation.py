"""The recomputation that does not call the code it checks.

This module imports nothing from ``timenet_evaluations.metrics``. It is given the serialized record
and does the arithmetic itself: the median of the samples of each rate-task cell, the count that
cell's dataset declared, and one division. A check that obtained its expected values by calling the
run's own derivation would report that a function equals itself, and would pass for every wrong
definition equally.

The two facts it needs are written here rather than imported. The two rate tasks are named as the
strings the record holds, and the nanoseconds of one second are a number. A constant taken from the
module under test would be the same coupling by another route.

``write_json`` does not exist yet, so the record arrives through a fixture as the bytes
``model_dump_json`` produced. When the writer lands, this check points at the file it wrote.
"""

import ast
import json
from pathlib import Path


RATE_TASKS = ("sequential", "block_shuffled")

NANOSECONDS_PER_SECOND = 1_000_000_000

THIS_MODULE = Path(__file__)


def key_of(at: dict[str, str]) -> tuple[str, str, str, str]:
    return (at["dataset"], at["representation"], at["reader"], at["task"])


def median_of(samples: list[int]) -> int:
    return sorted(samples)[len(samples) // 2]


def test_every_stored_rate_recomputes_from_the_record_alone(run_record: str) -> None:
    record = json.loads(run_record)

    counts = {dataset["dataset"]: dataset["item_count"] for dataset in record["metadata"]["datasets"]}
    stored = {key_of(rate["at"]): rate["items_per_second"] for rate in record["metrics"]["rates"]}

    expected: dict[tuple[str, str, str, str], float] = {}
    for entry in record["measurements"]["timed"]:
        if entry["at"]["task"] not in RATE_TASKS:
            continue

        seconds = median_of(entry["samples"]) / NANOSECONDS_PER_SECOND
        expected[key_of(entry["at"])] = counts[entry["at"]["dataset"]] / seconds

    assert expected
    assert stored == expected


def test_the_record_holds_one_rate_for_each_rate_task_cell_and_no_other(run_record: str) -> None:
    record = json.loads(run_record)

    rate_task_cells = [entry["at"] for entry in record["measurements"]["timed"] if entry["at"]["task"] in RATE_TASKS]
    stored = [rate["at"] for rate in record["metrics"]["rates"]]

    assert len(stored) == len(rate_task_cells)
    assert sorted(map(key_of, stored)) == sorted(map(key_of, rate_task_cells))


def test_a_duration_that_the_record_reports_is_the_median_of_the_samples_it_carries(run_record: str) -> None:
    # The rate above divides the median this test computed, so the record has to report that one.
    for entry in json.loads(run_record)["measurements"]["timed"]:
        assert entry["elapsed_ns"] == median_of(entry["samples"])
        assert entry["repetitions"] == len(entry["samples"])


def test_this_check_imports_nothing_from_the_module_it_checks() -> None:
    tree = ast.parse(THIS_MODULE.read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert imported == {"ast", "json", "pathlib"}
