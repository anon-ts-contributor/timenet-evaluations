"""The one derivation, and the recomputation that refuses a metric its own evidence contradicts.

A metric is a number that nobody observed. ``metrics`` is the one group of a result that can hold
such a number. It can hold one only because a reader who has the file can do the arithmetic again.
This module is where that arithmetic is written down, and it is written down once.

Two of the four tasks make a rate. The sequential walk and the block-shuffled walk read many items,
so a count of items per second is a number about them. The first-item read and the full read are
durations in seconds, and this module divides them by nothing.

A rate is computed once, from the reduced figure the measurement reports. It is not computed for
each sample and reduced afterwards. Both ways give the same number, because the count of
repetitions is odd and division by a duration keeps the order of the samples. Only the first way
makes the printed rate agree with the printed duration of the same cell. A reader who checks one
against the other must not find two numbers that disagree in the last digit.

``derive_metrics`` is pure. It reads no clock, no filesystem, and no earlier run. Every input is in
the result it is building: the measurements of that run, and the count of items each dataset
declared before the run started. The path it takes is text for a refusal message and is never
opened.

``check_metrics`` does the same arithmetic a second time and compares the two numbers. It does not
call ``derive_metrics``. A check that called the derivation would report only that the derivation
is deterministic, and it would pass for every wrong definition equally. The duplicated division is
therefore deliberate, and it is what makes a stored rate worth storing.

That check has a limit, and the limit is stated rather than hidden. It catches a rate that was
altered after the derivation, a rate that a second write path produced, and a rate that names the
wrong cell. It cannot catch a definition that is wrong in both places at once. The test beside this
module is what catches that one: the test reads the serialized record, does the division itself,
and imports nothing from here.

This module names no dataset and no dataset library. A rate divides by the count that a dataset
declared, and that count arrives on the result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, Task
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.result import DatasetRecord, EvaluationResult, Measurements, Metrics, Rate


RATE_TASKS: Final = (Task.SEQUENTIAL, Task.BLOCK_SHUFFLED)
"""The two tasks whose cells carry a rate.

The sequential walk and the block-shuffled walk read every canonical item, so the count of items
per second describes what they did. The first-item read reads one item, and a rate over one item is
a duration with a different name. The full read is one bulk pass, and the seconds it took are the
figure a reader wants.

The names are here rather than left to be inferred, because every number in the metrics group
follows from this tuple.
"""

NANOSECONDS_PER_SECOND: Final = 1_000_000_000
"""How many nanoseconds one second holds.

A measurement reports whole nanoseconds and a rate is per second, so one division stands between
them. The number is named here because the recomputation must produce the same float, and a
constant written twice can be written differently twice.
"""


def derive_metrics(measurements: Measurements, datasets: Sequence[DatasetRecord], *, path: Path) -> Metrics:
    """Compute every rate of one run, one time, from that run's own observations.

    This is the single place a metric is made. The record and the printed summary are both
    downstream of one call of this function, so the two cannot quote different numbers for one
    cell.

    Every input is in the result this function helps to build. The function reads no clock, no
    filesystem, and no earlier run, so a reader who has only the record can do the same arithmetic
    and get the same numbers.

    A rate is the count that the dataset declared, divided by the seconds the cell reported. The
    seconds come from the reduced figure of the measurement, not from each sample.

    Args:
        measurements: What a clock and a filesystem produced for this run. Only the timed entries
            of a rate task are read.
        datasets: The dataset records of the same run. Only the name and the declared count of
            items are read, and a count that a representation reported about itself is never one of
            them.
        path: The record these numbers belong to. A refusal names this path so that a reader knows
            which file the numbers are in. This function does not open it.

    Returns:
        One rate for each rate-task cell, in the order the timed entries hold.

    Raises:
        EvaluationError: If a dataset declared a count below one, if a timed cell names a dataset
            that these records do not declare, or if a cell reported a duration below one
            nanosecond. Each one makes a rate that is not a number about the cell it names.
    """
    counts = {record.dataset: record.item_count for record in datasets}

    for dataset, count in counts.items():
        if count < 1:
            raise EvaluationError(
                "every rate of a dataset divides by the count of items that dataset declared, so a count below "
                f"one gives no rate: dataset {dataset!r}, declared count {count}"
            )

    rates: list[Rate] = []
    for entry in measurements.timed:
        if entry.at.task not in RATE_TASKS:
            continue

        if entry.at.dataset not in counts:
            raise EvaluationError(
                failure_message(
                    "a timed cell names a dataset that this run does not declare, so its rate has no denominator",
                    at=entry.at,
                    path=path,
                    detail=f"declared datasets {sorted(counts)}",
                )
            )

        if entry.elapsed_ns < 1:
            raise EvaluationError(
                failure_message(
                    "a rate divides by the seconds a cell reported, so a reported duration below one nanosecond "
                    "gives no rate",
                    at=entry.at,
                    path=path,
                    detail=f"reported duration {entry.elapsed_ns} ns, samples {list(entry.samples)}",
                )
            )

        seconds = entry.elapsed_ns / NANOSECONDS_PER_SECOND
        rates.append(Rate(at=entry.at, items_per_second=counts[entry.at.dataset] / seconds))

    return Metrics(rates=tuple(rates))


def check_metrics(result: EvaluationResult, *, path: Path) -> None:
    """Refuse a result whose stored rates disagree with the measurements beside them.

    The measurements win. A rate that does not recompute is a defect and not a difference to
    report, so this function raises. It prints no warning, it corrects no number, and the run stops
    before a record makes it to the disk.

    The arithmetic here is written a second time, on purpose. This function does not call
    ``derive_metrics``, because a check that called the derivation would report only that the
    derivation is deterministic.

    Every input is in the result. The function reads no clock, no filesystem, and no earlier run,
    so a reader can run the same check on a stored record a month later.

    Args:
        result: The result to examine. Its metrics are compared against its own measurements and
            its own declared counts.
        path: The record this result was read from, or the record it is about to be written to. A
            refusal names this path. This function does not open it.

    Raises:
        EvaluationError: If a rate-task cell does not carry exactly one rate, if a rate names a
            cell that no rate-task measurement reports, if a rate cannot be recomputed at all, or
            if a stored rate differs from the recomputed one. The message names the cell, the
            stored value and the recomputed value.
    """
    counts = {record.dataset: record.item_count for record in result.metadata.datasets}
    durations = {entry.at: entry.elapsed_ns for entry in result.measurements.timed if entry.at.task in RATE_TASKS}

    _check_the_rates_match_the_rate_task_cells(result.metrics.rates, durations, counts, path=path)

    for rate in result.metrics.rates:
        count = counts[rate.at.dataset]
        recomputed = count / (durations[rate.at] / NANOSECONDS_PER_SECOND)

        if rate.items_per_second != recomputed:
            raise EvaluationError(
                failure_message(
                    "a stored rate does not recompute from the measurements beside it",
                    at=rate.at,
                    path=path,
                    detail=(
                        f"stored rate {rate.items_per_second} items/s, recomputed rate {recomputed} items/s, "
                        f"declared count {count}, reported duration {durations[rate.at]} ns"
                    ),
                )
            )


def _check_the_rates_match_the_rate_task_cells(
    rates: Sequence[Rate],
    durations: Mapping[Cell, int],
    counts: Mapping[str, int],
    *,
    path: Path,
) -> None:
    """Make sure that the rates and the rate-task measurements name the same cells, one for one.

    A rate that went missing is as much a disagreement as a rate that was altered, and a second
    rate for one cell gives a reader two numbers for one question. Both are found here, before the
    division that follows.

    Args:
        rates: The stored rates of the result.
        durations: The reported duration of each rate-task cell the result measured, in
            nanoseconds.
        counts: The count of items each dataset of the result declared.
        path: The record these numbers belong to, for the refusal message.

    Raises:
        EvaluationError: If a rate-task cell carries no rate or more than one, if a rate names a
            dataset the metadata does not declare, if a count or a duration is below one, or if a
            rate names a cell that no rate-task measurement reports.
    """
    named = [rate.at for rate in rates]

    for cell, elapsed_ns in durations.items():
        if named.count(cell) != 1:
            raise EvaluationError(
                failure_message(
                    "a rate-task cell carries exactly one rate, and this cell does not",
                    at=cell,
                    path=path,
                    detail=f"rates naming this cell {named.count(cell)}",
                )
            )

        if cell.dataset not in counts:
            raise EvaluationError(
                failure_message(
                    "a rate divides by a count that this result's metadata does not declare",
                    at=cell,
                    path=path,
                    detail=f"declared datasets {sorted(counts)}",
                )
            )

        if counts[cell.dataset] < 1 or elapsed_ns < 1:
            raise EvaluationError(
                failure_message(
                    "a rate is recomputed from a declared count and a reported duration, and both are above zero",
                    at=cell,
                    path=path,
                    detail=f"declared count {counts[cell.dataset]}, reported duration {elapsed_ns} ns",
                )
            )

    for cell in named:
        if cell not in durations:
            raise EvaluationError(
                failure_message(
                    "a rate names a cell that no rate-task measurement in this result reports",
                    at=cell,
                    path=path,
                    detail=f"rate-task cells measured {len(durations)}",
                )
            )
