"""The trend check: the step after the first sample that a failed drop leaves in every cell.

Under a working cold protocol a cell holds no downward trend inside it. Repetition five begins from
a dropped page cache, the same as repetition one, so the five samples are five draws from one
population. Under a drop that stopped working, repetition one is cold and every repetition after it
is warm. The samples then hold a large step down after the first one.

The drop is machine-wide, so that shape appears in every cell at once. One cell that steps is one
thermal excursion or one slow page fault, which the median already handles. This check counts the
cells that step and condemns the run only when enough of them do. How large a step counts, and how
many cells are enough, are parameters of the check. No specification in this repository states
either value, so this module takes both and invents neither.

**What a condemnation establishes, and what it does not.** A condemned run is evidence about the
instrument and about nothing else. The samples hold the shape that a failed drop leaves. They say
nothing about which representation is faster, and this check grades no format. It never adjusts a
figure, never discounts one, and never marks one. It condemns the whole run, or it says nothing at
all.

**What a clean result does not establish.** A run that was warm from its first repetition holds no
step, because the drop never worked and every sample is warm. That run passes this check and looks
perfect. The calibration probe covers that failure. This check cannot, and no report may quote a
clean result here as evidence that the drop worked.

The check applies to a run taken with no warm-up. Under a warm-up every repetition is warm before it
is timed, so no step is expected and the absence of one is evidence of nothing. This module gives no
verdict for such a run, rather than a clean one.

Nothing here reads a file, a clock or a process. The samples arrive as the values that ``repeat``
recorded. Reading a stored result back from disk belongs to the capability that writes one.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell
from timenet_evaluations.harness.repeat import Measurement


MINIMUM_SAMPLES: Final = 2
"""How many samples one cell needs before a step after the first sample can exist.

A cell of one sample has a first sample and nothing after it, so it can hold no step and this check
reads nothing in it. Such a cell is left out of the count the share is taken over. A cell that
counted as clean would make a run of one-sample cells report a clean check that it never performed.
"""

LOWEST_RATIO: Final = 1.0
"""The largest step ratio that this check refuses.

A ratio of one condemns every cell whose first sample is the largest of its samples, which one cell
in five is by chance alone. A step is a step because the first sample is much larger, so the ratio
is more than one.
"""


class Verdict(StrEnum):
    """What the trend check found in one run's stored samples.

    There are three outcomes and not two. A run can be condemned, a run can be examined and not
    condemned, and a run can be one that this check has nothing to say about.
    """

    CONDEMNED = "condemned"
    """Enough cells step after their first sample. The run is not a cold-read result, and it must be
    discarded and taken again."""

    NOT_CONDEMNED = "not_condemned"
    """The check examined the samples and did not find the shape in enough cells. This is not
    evidence that the drop worked: a run that was warm throughout holds no step either."""

    NO_VERDICT = "no_verdict"
    """The check does not apply. The run was taken with a warm-up, or no cell holds enough samples
    for a step after the first one to exist."""


class StepCriteria(BaseModel):
    """How large a step counts, and how many cells must hold one before a run is condemned.

    Both values travel together, because a verdict means nothing without the two numbers behind it.
    Neither has a default. SPEC-0018 describes the shape and pins no threshold for it, so a default
    here would be a number this repository invented and later defended.
    """

    model_config = ConfigDict(frozen=True)

    step_ratio: float
    """How many times larger than every later sample the first sample must be. The comparison is
    against the largest of the later samples, so a cell steps only when the whole tail is faster."""

    condemning_share: float
    """What share of the examined cells must step before the run is condemned. One cell is noise,
    and the failure this check finds is machine-wide and appears everywhere at once."""


class TrendReport(BaseModel):
    """The verdict over one run's stored samples, and the evidence behind it.

    The report carries the criteria it was judged under. A verdict outlives the run that produced
    it, so a later reader can judge the same samples again under better numbers.
    """

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    """What the check found."""

    criteria: StepCriteria
    """The step ratio and the share the verdict was taken under."""

    stepped: tuple[Cell, ...]
    """Every cell whose first sample steps, in the order the measurements arrived."""

    examined: int
    """How many cells held enough samples to be read. Cells with fewer are not counted."""


def steps_after_the_first(samples: tuple[int, ...], step_ratio: float) -> bool:
    """Find out whether one cell's samples hold a step after the first sample.

    The first sample is compared against the largest of the samples after it. A cell steps when the
    first sample is more than ``step_ratio`` times that largest value, so every later repetition is
    faster by that factor. A comparison against the median of the later samples would call a cell
    stepped while one of its later samples was as slow as the first.

    Args:
        samples: The recorded durations of one cell, in the order they were taken.
        step_ratio: How many times larger than every later sample the first sample must be.

    Returns:
        True when the first sample steps above every sample after it. False when it does not, and
        false when the cell holds fewer than two samples.
    """
    if len(samples) < MINIMUM_SAMPLES:
        return False

    return samples[0] > max(samples[1:]) * step_ratio


def check_trend(measurements: Sequence[Measurement], criteria: StepCriteria) -> TrendReport:
    """Read a run's stored samples for the step that a drop which stopped working leaves.

    Call this over every measurement of one run. The check is machine-wide, so a share taken over
    one dataset or one representation would look for a machine-wide shape in one part of a machine.

    A run in which one measurement carries a warm-up gets no verdict. Every repetition of a warm run
    is warm before it is timed, so no step is expected and its absence proves nothing.

    The result condemns the run or it says nothing. It marks no figure, discounts no figure, and
    names no representation as slow.

    Args:
        measurements: Every timed measurement of the run, each carrying its samples and its
            warm-up setting.
        criteria: How large a step counts, and what share of the cells must hold one.

    Returns:
        The verdict, the criteria it was taken under, and the cells that step.

    Raises:
        EvaluationError: If the step ratio is one or less, or if the share is not more than zero and
            at most one. A verdict under such numbers describes the criteria and not the run.
    """
    if criteria.step_ratio <= LOWEST_RATIO:
        raise EvaluationError(
            f"a step is a first sample much larger than the rest, so the step ratio is more than "
            f"one: step ratio {criteria.step_ratio}, lowest refused ratio {LOWEST_RATIO}"
        )
    if not 0 < criteria.condemning_share <= 1:
        raise EvaluationError(
            f"a run is condemned by a share of its cells, so the share is more than zero and at "
            f"most one: condemning share {criteria.condemning_share}"
        )

    if any(measurement.warmup for measurement in measurements):
        return TrendReport(verdict=Verdict.NO_VERDICT, criteria=criteria, stepped=(), examined=0)

    readable = [measurement for measurement in measurements if len(measurement.samples) >= MINIMUM_SAMPLES]
    if not readable:
        return TrendReport(verdict=Verdict.NO_VERDICT, criteria=criteria, stepped=(), examined=0)

    stepped = tuple(
        measurement.at for measurement in readable if steps_after_the_first(measurement.samples, criteria.step_ratio)
    )
    condemned = len(stepped) / len(readable) >= criteria.condemning_share

    return TrendReport(
        verdict=Verdict.CONDEMNED if condemned else Verdict.NOT_CONDEMNED,
        criteria=criteria,
        stepped=stepped,
        examined=len(readable),
    )


def refuse_a_condemned_run(report: TrendReport) -> None:
    """Stop a condemned run from being published or reported as a cold-read result.

    A condemned run holds warm figures under a cold label. The figures render, the table prints, and
    every number in it is wrong in the same direction, so a caveat beside them would not be read.
    The run is discarded and taken again.

    The message names no single cell, because no single cell is at fault. It names how many cells
    step, how many were read, and the criteria behind the verdict.

    Args:
        report: What :func:`check_trend` found over the run's samples.

    Raises:
        EvaluationError: If the report condemns the run.
    """
    if report.verdict is not Verdict.CONDEMNED:
        return

    cells = ", ".join(
        f"{cell.dataset}/{cell.representation}/{cell.reader}/{cell.task.value}" for cell in report.stepped
    )

    raise EvaluationError(
        f"the stored samples step after the first repetition in {len(report.stepped)} of "
        f"{report.examined} examined cells, which is the shape a drop that stopped working leaves, "
        f"so this run is not a cold-read result and must be taken again: "
        f"step ratio {report.criteria.step_ratio}, "
        f"condemning share {report.criteria.condemning_share}, stepped cells {cells}"
    )
