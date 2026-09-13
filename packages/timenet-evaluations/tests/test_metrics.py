"""The one derivation, what it refuses, and the check that catches a metric its evidence denies.

Nothing here is timed, nothing is written, and no artifact exists. The derivation takes a
measurement set and a list of declared counts, and that is the whole of its world.

The independent recomputation is not in this file. It reads the serialized record and imports
nothing from the module under test, so it lives in ``test_metrics_recomputation.py``.
"""

import ast
from pathlib import Path

import pytest

from timenet_evaluations import metrics as metrics_module
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, Task
from timenet_evaluations.harness.repeat import Measurement
from timenet_evaluations.metrics import RATE_TASKS, check_metrics, derive_metrics
from timenet_evaluations.result import DatasetRecord, EvaluationResult, Measurements, Metrics, Rate


PATH = Path("/runs/4f2a91c0e3bd/result.json")

METRICS_MODULE = Path(metrics_module.__file__)
PACKAGE_ROOT = METRICS_MODULE.parent
PACKAGE_MODULES = sorted(PACKAGE_ROOT.rglob("*.py"))

CLOCKS_AND_FILESYSTEMS = {"time", "datetime", "os", "subprocess", "json", "shutil", "glob", "importlib"}


def rates_of(result: EvaluationResult) -> dict[Cell, float]:
    return {rate.at: rate.items_per_second for rate in result.metrics.rates}


def with_metrics(result: EvaluationResult, rates: tuple[Rate, ...]) -> EvaluationResult:
    return result.model_copy(update={"metrics": Metrics(rates=rates)})


def called_names(tree: ast.AST) -> set[str]:
    """Name every object a call in this tree is made on, so ``Rate(...)`` and ``Rate.x()`` both count."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                func = func.value
            if isinstance(func, ast.Name):
                names.add(func.id)

    return names


# Which tasks make a rate, and what one is


def test_only_the_sequential_and_the_block_shuffled_cells_carry_a_rate(
    run_measurements: Measurements, run_datasets: tuple[DatasetRecord, ...]
) -> None:
    derived = derive_metrics(run_measurements, run_datasets, path=PATH)

    assert {rate.at.task for rate in derived.rates} == {Task.SEQUENTIAL, Task.BLOCK_SHUFFLED}
    assert len(derived.rates) == len([entry for entry in run_measurements.timed if entry.at.task in RATE_TASKS])


def test_a_first_item_read_and_a_full_read_are_divided_by_nothing(
    run_measurements: Measurements, run_datasets: tuple[DatasetRecord, ...]
) -> None:
    derived = derive_metrics(run_measurements, run_datasets, path=PATH)

    named = {rate.at for rate in derived.rates}
    for entry in run_measurements.timed:
        if entry.at.task in {Task.FIRST_ITEM, Task.FULL_READ}:
            assert entry.at not in named


def test_a_rate_is_the_declared_count_divided_by_the_seconds_the_cell_reported(
    run_measurements: Measurements, run_datasets: tuple[DatasetRecord, ...]
) -> None:
    counts = {record.dataset: record.item_count for record in run_datasets}
    reported = {entry.at: entry.elapsed_ns for entry in run_measurements.timed}

    # The equality is exact on purpose: a reader who divides again must not find a number that
    # disagrees in the last digit.
    for rate in derive_metrics(run_measurements, run_datasets, path=PATH).rates:
        recomputed = counts[rate.at.dataset] / (reported[rate.at] / 1_000_000_000)
        assert rate.items_per_second == recomputed


def test_a_rate_divides_by_the_count_its_own_dataset_declared(
    run_measurements: Measurements, run_datasets: tuple[DatasetRecord, ...]
) -> None:
    # The two datasets declare different counts, so a rate that divided by the other one differs.
    counts = {record.dataset: record.item_count for record in run_datasets}
    assert len(set(counts.values())) == len(counts)

    derived = derive_metrics(run_measurements, run_datasets, path=PATH)
    reported = {entry.at: entry.elapsed_ns for entry in run_measurements.timed}

    for rate in derived.rates:
        other = next(count for dataset, count in counts.items() if dataset != rate.at.dataset)
        by_the_other_count = other / (reported[rate.at] / 1_000_000_000)
        assert rate.items_per_second != by_the_other_count


def test_the_rate_comes_from_the_reduced_figure_and_not_from_one_sample(
    run_measurements: Measurements, run_datasets: tuple[DatasetRecord, ...]
) -> None:
    # The printed rate has to agree with the printed duration of the same cell.
    derived = derive_metrics(run_measurements, run_datasets, path=PATH)
    counts = {record.dataset: record.item_count for record in run_datasets}
    entries = {entry.at: entry for entry in run_measurements.timed}

    for rate in derived.rates:
        entry = entries[rate.at]
        assert entry.elapsed_ns in entry.samples
        assert entry.elapsed_ns not in {min(entry.samples), max(entry.samples)}

        from_the_reduced_figure = counts[rate.at.dataset] / (entry.elapsed_ns / 1_000_000_000)
        from_the_fastest_sample = counts[rate.at.dataset] / (min(entry.samples) / 1_000_000_000)

        assert rate.items_per_second == from_the_reduced_figure
        assert rate.items_per_second != from_the_fastest_sample


# The derivation is pure, and it happens once


def test_two_calls_over_one_measurement_set_give_one_answer(
    run_measurements: Measurements, run_datasets: tuple[DatasetRecord, ...]
) -> None:
    assert derive_metrics(run_measurements, run_datasets, path=PATH) == derive_metrics(
        run_measurements, run_datasets, path=PATH
    )


def test_the_derivation_runs_with_no_fixture_no_clock_and_no_filesystem() -> None:
    # No fixture: the call above takes two values and a path it never opens.
    tree = ast.parse(METRICS_MODULE.read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported & CLOCKS_AND_FILESYSTEMS == set()
    assert "open" not in called_names(tree)


def test_no_module_outside_the_derivation_makes_a_metric() -> None:
    # A number reaches the metrics group only as a Rate, so the one module that builds a Rate is
    # the one place a metric is made.
    for module in PACKAGE_MODULES:
        if module == METRICS_MODULE:
            continue

        names = called_names(ast.parse(module.read_text(encoding="utf-8")))
        assert "Rate" not in names, f"{module.name} makes a metric"
        assert "Metrics" not in names, f"{module.name} writes into the metrics group"


def test_the_check_does_not_call_the_derivation() -> None:
    # A check that called the derivation would report that a function equals itself.
    tree = ast.parse(METRICS_MODULE.read_text(encoding="utf-8"))
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    assert {"derive_metrics", "check_metrics"} <= defined
    assert "derive_metrics" not in called_names(tree)


# What the derivation refuses


def test_a_dataset_that_declared_no_items_is_refused_by_name(
    run_measurements: Measurements, run_datasets: tuple[DatasetRecord, ...]
) -> None:
    silent = run_datasets[0].model_copy(update={"item_count": 0})

    with pytest.raises(EvaluationError, match="a count below one gives no rate") as raised:
        derive_metrics(run_measurements, (silent, *run_datasets[1:]), path=PATH)

    message = str(raised.value)
    assert f"dataset '{silent.dataset}'" in message
    assert "declared count 0" in message


def test_a_timed_cell_whose_dataset_is_not_declared_is_refused(
    run_measurements: Measurements, run_datasets: tuple[DatasetRecord, ...]
) -> None:
    with pytest.raises(EvaluationError, match="names a dataset that this run does not declare") as raised:
        derive_metrics(run_measurements, run_datasets[:1], path=PATH)

    assert f"dataset '{run_datasets[1].dataset}'" in str(raised.value)
    assert str(PATH) in str(raised.value)


def test_a_cell_that_reported_no_time_is_refused_rather_than_divided_by(
    run_measurements: Measurements, run_datasets: tuple[DatasetRecord, ...]
) -> None:
    stopped = tuple(
        entry.model_copy(update={"samples": (0,) * 5}) if entry.at.task in RATE_TASKS else entry
        for entry in run_measurements.timed
    )

    with pytest.raises(EvaluationError, match="a reported duration below one nanosecond"):
        derive_metrics(run_measurements.model_copy(update={"timed": stopped}), run_datasets, path=PATH)


# What the check refuses


def test_a_result_whose_metrics_came_from_the_derivation_passes(run_result: EvaluationResult) -> None:
    check_metrics(run_result, path=PATH)


def test_a_tampered_rate_raises_naming_the_cell_the_stored_value_and_the_recomputed_one(
    run_result: EvaluationResult,
) -> None:
    altered = run_result.metrics.rates[0]
    stored = altered.items_per_second * 2
    tampered = with_metrics(
        run_result, (altered.model_copy(update={"items_per_second": stored}), *run_result.metrics.rates[1:])
    )

    with pytest.raises(EvaluationError, match="does not recompute from the measurements beside it") as raised:
        check_metrics(tampered, path=PATH)

    message = str(raised.value)
    assert f"dataset '{altered.at.dataset}'" in message
    assert f"representation '{altered.at.representation}'" in message
    assert f"reader '{altered.at.reader}'" in message
    assert f"task '{altered.at.task.value}'" in message
    assert f"stored rate {stored} items/s" in message
    assert f"recomputed rate {altered.items_per_second} items/s" in message


def test_a_rate_that_went_missing_is_a_disagreement_as_much_as_one_that_was_altered(
    run_result: EvaluationResult,
) -> None:
    with pytest.raises(EvaluationError, match="carries exactly one rate, and this cell does not") as raised:
        check_metrics(with_metrics(run_result, run_result.metrics.rates[1:]), path=PATH)

    assert "rates naming this cell 0" in str(raised.value)


def test_a_second_rate_for_one_cell_gives_a_reader_two_numbers_and_is_refused(
    run_result: EvaluationResult,
) -> None:
    doubled = (*run_result.metrics.rates, run_result.metrics.rates[0])

    with pytest.raises(EvaluationError, match="carries exactly one rate, and this cell does not") as raised:
        check_metrics(with_metrics(run_result, doubled), path=PATH)

    assert "rates naming this cell 2" in str(raised.value)


def test_a_rate_on_a_task_that_makes_no_rate_is_refused(run_result: EvaluationResult) -> None:
    first_item = run_result.metrics.rates[0].at.model_copy(update={"task": Task.FIRST_ITEM})
    intruder = (*run_result.metrics.rates, Rate(at=first_item, items_per_second=1.0))

    with pytest.raises(EvaluationError, match="no rate-task measurement in this result reports"):
        check_metrics(with_metrics(run_result, intruder), path=PATH)


def test_a_metric_whose_measurement_left_the_result_is_refused(run_result: EvaluationResult) -> None:
    kept = tuple(entry for entry in run_result.measurements.timed if entry.at != run_result.metrics.rates[0].at)
    without = run_result.model_copy(update={"measurements": run_result.measurements.model_copy(update={"timed": kept})})

    with pytest.raises(EvaluationError, match="no rate-task measurement in this result reports"):
        check_metrics(without, path=PATH)


def test_the_check_reads_no_clock_no_filesystem_and_no_earlier_result(run_result: EvaluationResult) -> None:
    # The only inputs are the result and a path that is named in a message and never opened.
    check_metrics(run_result, path=Path("/no/such/directory/result.json"))


def test_a_measurement_that_reports_no_time_is_refused_by_the_check(run_result: EvaluationResult) -> None:
    stopped = tuple(
        entry.model_copy(update={"samples": (0,) * 5}) if entry.at.task in RATE_TASKS else entry
        for entry in run_result.measurements.timed
    )
    halted = run_result.model_copy(
        update={"measurements": run_result.measurements.model_copy(update={"timed": stopped})}
    )

    with pytest.raises(EvaluationError, match="both are above zero"):
        check_metrics(halted, path=PATH)


def test_a_measurement_is_left_as_the_timing_primitive_wrote_it(run_measurements: Measurements) -> None:
    # The timed entry is the measurement itself, and its count and figure stay computed fields.
    entry = run_measurements.timed[0]

    assert isinstance(entry, Measurement)
    assert entry.repetitions == len(entry.samples)
    assert entry.elapsed_ns == sorted(entry.samples)[len(entry.samples) // 2]
