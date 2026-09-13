"""The one timing primitive: the drop, the interval, the samples, the median, and the count.

Every timed task calls ``repeat``. No task holds a loop of its own, no task holds a sample list of
its own, and no task reduces its own samples. One implementation is the point. A change to the
reduction, to the drop rule or to the counting rule takes effect on all sixteen cells of a dataset
at once, so the protocol cannot hold in one place and drift in another.

One repetition is a drop and then one timed interval. The drop is privileged, machine-wide, and
outside the clock: it costs real time, it is paid before every repetition, and none of it is
reading. The clock starts after the drop returns and after its exit status has been examined, so a
repetition whose drop failed contributes no sample and the run stops.

``warmup`` is a boolean and it is off. Off, a repetition is *drop, then record*. On, it is *drop,
then one untimed read, then record* — a warm steady-state figure that is not the table's number.
The cache is dropped in both modes, so the starting state is known either way: a warm figure is one
read away from a cold one, and not one read away from what the machine happened to hold.

The samples stay. The median is the figure a cell reports and the samples are what a later reader
examines it against, so a measurement carries both and never a median alone. Both the count and the
figure are computed from the sample list, so a measurement cannot declare a count the loop did not
produce.

Everything here runs in the calling process. The drop is the one subprocess of this capability and
``harness.drop`` owns it. Nothing is given to a second process, nothing crosses a process boundary,
and no peak memory is measured — to measure that correctly needs the process isolation this
capability refuses.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time
from typing import Final

from pydantic import BaseModel, ConfigDict, computed_field

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.harness.drop import DropCaches, DropRecord


REPEATS: Final = 5
"""The number of cold repetitions one cell is measured over.

The count is five and it is odd. Five is the protocol the paper states, so a run that takes four is
not producing the table's figures. An odd count is what makes the median one of the samples instead
of the mean of the two in the middle. The value is named here because it is the default of a
command-line argument and it travels onto the result, so a bare number would be written in more
than one place and could disagree with itself.
"""


def median_ns(samples: tuple[int, ...]) -> int:
    """Reduce the durations one cell recorded to the one figure it reports.

    The figure is the median. It is not the arithmetic mean: one thermal excursion or one slow page
    fault moves a mean and does not move a median, and cold reads give more of those excursions
    than warm reads, not fewer. It is not the minimum either: the fastest pass is the machine at its
    least representative.

    The count of samples is odd, so the value comes from the sorted list at the middle position and
    is one of the samples. No figure is ever the mean of two samples given as an observed one.

    Args:
        samples: The recorded durations, in the order they were taken.

    Returns:
        The median duration, in whole nanoseconds.

    Raises:
        EvaluationError: If the count of samples is zero or even. There is then nothing to reduce,
            or the reduction would report an interpolation as an observed sample.
    """
    if len(samples) < 1 or len(samples) % 2 == 0:
        raise EvaluationError(
            "a cell reports the median of an odd number of samples, so this list has no figure: "
            f"samples {list(samples)}, count {len(samples)}"
        )

    return sorted(samples)[len(samples) // 2]


class Timed(BaseModel):
    """The operation one cell is timed on, and the coordinates that name that cell.

    The three values travel together, so ``repeat`` takes five parameters instead of seven. The
    model is frozen, and it is never stored: a callable is not a record.
    """

    model_config = ConfigDict(frozen=True)

    operation: Callable[[], object]
    """The operation to time. One call is one repetition of one task, and its result is discarded."""
    at: Cell
    """The dataset, the representation, the reader and the task this operation is timed under."""
    path: Path
    """The file or directory the operation reads. The drop is told about it, so a failed drop names
    it."""


class Measurement(BaseModel):
    """What one cell measured: every sample, the state it was taken in, and every drop it needed.

    The figure and the repetition count are computed from ``samples``. Neither is a field, so a
    measurement cannot carry a median without the samples behind it, and it cannot claim a
    repetition count the timing loop did not produce.
    """

    model_config = ConfigDict(frozen=True)

    at: Cell
    """The four coordinates of the cell. A figure that named fewer could be read as an answer to a
    question it did not ask."""
    samples: tuple[int, ...]
    """Every recorded duration, in whole nanoseconds and in the order it was taken. The spread of an
    appendix and the trend check over a run both come from this list, so nothing reduces it away."""
    warmup: bool
    """Whether each repetition performed one untimed read after the drop. A warm figure and a cold
    one are both well formed and differ by several times, so the setting travels with the figure."""
    drops: tuple[DropRecord, ...]
    """The record of every drop this measurement needed, one before each recorded repetition. A drop
    that reported failure stops the run, so every record here is a drop that reported success."""

    @computed_field
    @property
    def repetitions(self) -> int:
        """Count the repetitions the timing loop recorded.

        The count comes from the samples and never from the number a caller asked for. A run that
        asked for five and appended four reports four.

        Returns:
            How many samples were taken.
        """
        return len(self.samples)

    @computed_field
    @property
    def elapsed_ns(self) -> int:
        """Give the figure this cell reports.

        The samples are what the figure is computed from, so a measurement cannot hold one without
        the other. ``median_ns`` refuses a count of samples that has no median, and ``repeat``
        refuses such a count before it drops anything.

        Returns:
            The median of the samples, in whole nanoseconds.
        """
        return median_ns(self.samples)


def repeat(
    timed: Timed,
    *,
    drop_caches: DropCaches,
    repeats: int,
    warmup: bool = False,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> Measurement:
    """Measure one cell over repeated cold reads, and reduce them to one figure.

    One repetition is a drop, then an optional untimed read, then one timed interval. The drop
    comes before every repetition — a repetition that follows a repetition has its own drop — and
    it stays outside the interval, because its cost is real and none of it is reading. The clock
    starts after the drop returns, so a drop that reported failure has already stopped the run.

    The interval covers exactly one call of the operation. Nothing before it and nothing after it is
    inside it: not the drop, not the warm-up read, not the bookkeeping of a sample, and not the
    block plan, which is built once before this function is called.

    Everything happens in the calling process. The drop is the only subprocess, and ``harness.drop``
    starts it.

    Args:
        timed: The operation, the cell it belongs to, and the path it reads.
        drop_caches: The drop each repetition begins with. It is the same mechanism the calibration
            probe measured, and never a second one.
        repeats: How many repetitions to record. It is odd and at least one, and ``REPEATS`` is what
            a run uses.
        warmup: Whether to read once, untimed, after each drop. It is off, so the figure is a cold
            figure. On, the figure is a warm steady state and is not the table's number.
        clock: The monotonic nanosecond clock. A test scripts durations through it.

    Returns:
        The samples, the drops, the ``warmup`` setting, and the cell they belong to.

    Raises:
        EvaluationError: If ``repeats`` is below one or even. No drop and no invocation is performed
            first, because a count that has no median describes no measurement.
    """
    if repeats < 1 or repeats % 2 == 0:
        raise EvaluationError(
            failure_message(
                "a cell is measured over an odd number of repetitions, at least one, so this count has no median",
                at=timed.at,
                path=timed.path,
                detail=f"repeats {repeats}",
            )
        )

    samples: list[int] = []
    drops: list[DropRecord] = []
    for _ in range(repeats):
        drops.append(drop_caches(at=timed.at, path=timed.path))

        if warmup:
            timed.operation()

        started = clock()
        timed.operation()
        samples.append(clock() - started)

    return Measurement(at=timed.at, samples=tuple(samples), warmup=warmup, drops=tuple(drops))
