"""The calibration probe: the one thing in this package that can observe whether the drop works.

The probe reads one artifact twice. The first read follows a drop. The second read follows the
first with no drop between them. The ratio of the two elapsed durations is the evidence. If the
drop evicts pages, the second read is several times faster. If the ratio is near one, the cache was
never dropped, and every figure the run is about to produce would be warm.

A structural test cannot reach this. A test double has no page cache, so it cannot tell a cold read
from a warm one, and ``harness.drop`` says so in its own docstring. Only a measurement can, and
this is the measurement. It runs once, before any cell of the grid, because a refusal there is the
only refusal that costs nothing.

The threshold is a parameter with no default. Nothing in this repository establishes a value for
it: it stands for "several times faster on this machine", and a number invented here would sit in
the one check that decides whether every figure in the run is real. The caller chooses it, the run
records it, and a later reader re-judges the recorded ratio against the recorded threshold rather
than trusting the run.

The probe carries its own coordinates. It reads one (dataset, representation) artifact, so a
``StorageKey`` names it exactly and no coordinate is filled with a placeholder.

The clock is injected for the same reason the runner of the drop is. A test scripts two durations
and gets a ratio, with no sleep and no artifact.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

from pydantic import BaseModel, ConfigDict

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import StorageKey
from timenet_evaluations.harness.drop import DropCaches, DropRecord


class Probe(BaseModel):
    """The one read the probe times twice, and the artifact it reads.

    The three values travel together, so ``calibrate`` takes four parameters instead of six. The
    model is frozen, and it is never stored: a callable is not a record.
    """

    model_config = ConfigDict(frozen=True)

    read: Callable[[], object]
    """The read to time. It reads the artifact at ``path`` and its result is discarded."""
    at: StorageKey
    """The dataset and the representation the artifact belongs to."""
    path: Path
    """The artifact the read is given. The drop is told about it, so a failed drop names it."""


class Calibration(BaseModel):
    """What the probe measured, and what it was judged against.

    Both numbers are kept. A ratio alone cannot be re-judged, because a later reader would not know
    what the run demanded of it.
    """

    model_config = ConfigDict(frozen=True)

    ratio: float
    """The cold duration divided by the warm duration."""
    threshold: float
    """The ratio the run demanded. It is greater than one."""
    cold_ns: int
    """The elapsed nanoseconds of the read that followed the drop."""
    warm_ns: int
    """The elapsed nanoseconds of the read that followed the cold read, with no drop between."""
    drop: DropRecord
    """The command the probe dropped with, and the status it returned."""


def calibrate(
    probe: Probe,
    *,
    drop_caches: DropCaches,
    threshold: float,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> Calibration:
    """Measure whether the drop works, and refuse the run if it does not.

    The order is one drop and two reads. The drop comes first, then the cold read, then the warm
    read with nothing between them. Both reads are timed with a monotonic nanosecond clock.

    Args:
        probe: The read, the coordinates and the path.
        drop_caches: The drop the run uses for every timed repetition. The probe checks the same
            mechanism the grid depends on, and never a second one.
        threshold: The ratio the run demands. It has no default, because no value for it is
            established anywhere.
        clock: The monotonic nanosecond clock. A test scripts durations through it.

    Returns:
        The measured ratio, the threshold it passed, both durations, and the drop's record.

    Raises:
        EvaluationError: If the threshold is not greater than one, if the warm read measured zero
            nanoseconds, or if the ratio is at or below the threshold. The last case is the drop
            that reported success and evicted nothing, so the run stops before it measures a cell.
    """
    if threshold <= 1:
        raise EvaluationError(
            "the calibration threshold must be greater than one, because a working drop makes the "
            f"second read faster than the first: threshold {threshold}"
        )

    record = drop_caches(at=probe.at, path=probe.path)

    cold_start = clock()
    probe.read()
    cold_ns = clock() - cold_start

    warm_start = clock()
    probe.read()
    warm_ns = clock() - warm_start

    if warm_ns == 0:
        raise EvaluationError(
            "the calibration probe measured a warm read of no elapsed time, so no ratio exists: "
            f"command {record.command!r}, cold_ns {cold_ns}, warm_ns {warm_ns}, "
            f"dataset {probe.at.dataset!r}, representation {probe.at.representation!r}"
        )

    ratio = cold_ns / warm_ns
    if ratio <= threshold:
        raise EvaluationError(
            "the page cache drop is not working, so every figure this run would produce is warm: "
            f"ratio {ratio}, threshold {threshold}, command {record.command!r}, "
            f"exit status {record.exit_status}, platform {record.platform!r}, "
            f"cold_ns {cold_ns}, warm_ns {warm_ns}"
        )

    return Calibration(ratio=ratio, threshold=threshold, cold_ns=cold_ns, warm_ns=warm_ns, drop=record)
