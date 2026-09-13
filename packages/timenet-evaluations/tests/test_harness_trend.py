"""The trend check over the stored samples: what it condemns, and what it refuses to say.

Every sample here is a number written by hand. Nothing in this file reads an artifact, starts a
clock or drops a cache, so nothing here is evidence that the cold protocol works. What these tests
pin is the shape of the verdict: that a run whose every cell steps after its first sample is
condemned, that one cell stepping is not enough, and that a warm run gets no verdict at all.
"""

import ast
import inspect
from pathlib import Path

import pytest

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.assembly import expected_cells
from timenet_evaluations.grid.cell import Cell, Task
from timenet_evaluations.grid.registry import CELLS_PER_DATASET
from timenet_evaluations.harness import trend as trend_module
from timenet_evaluations.harness.drop import DARWIN, DropRecord
from timenet_evaluations.harness.repeat import REPEATS, Measurement
from timenet_evaluations.harness.trend import (
    LOWEST_RATIO,
    MINIMUM_SAMPLES,
    StepCriteria,
    Verdict,
    check_trend,
    refuse_a_condemned_run,
    steps_after_the_first,
)


DATASET = "a-dataset"

TREND_MODULE = Path(inspect.getfile(trend_module))

RECORD = DropRecord(command="/usr/sbin/purge", exit_status=0, platform=DARWIN)

CRITERIA = StepCriteria(step_ratio=3.0, condemning_share=1.0)
"""What these tests judge a run under. Neither number is a default of the check, and neither is
established anywhere: the spec pins the shape and no threshold for it."""

COLD = (10_000, 9_800, 10_200, 9_900, 10_100)
"""Five cold samples. No one of them steps above the rest, which is the shape of a working drop."""

STEPPED = (100_000, 10_000, 9_800, 10_200, 9_900)
"""One cold sample and four warm ones, which is the shape a drop that stopped working leaves."""


def measurement(at: Cell, samples: tuple[int, ...], *, warmup: bool = False) -> Measurement:
    """One cell's measurement, with a drop record per sample."""
    return Measurement(at=at, samples=samples, warmup=warmup, drops=tuple(RECORD for _ in samples))


def grid(*, stepping: int, warmup: bool = False) -> list[Measurement]:
    """A whole dataset's sixteen measurements, the first `stepping` of which step."""
    return [
        measurement(at, STEPPED if index < stepping else COLD, warmup=warmup)
        for index, at in enumerate(expected_cells(DATASET))
    ]


# One cell steps when its first sample is far above every sample after it


def test_a_first_sample_far_above_every_later_one_is_a_step() -> None:
    assert steps_after_the_first(STEPPED, CRITERIA.step_ratio)


def test_five_alike_samples_are_not_a_step() -> None:
    assert not steps_after_the_first(COLD, CRITERIA.step_ratio)


def test_the_comparison_is_against_the_largest_later_sample_and_not_their_middle() -> None:
    # One slow later repetition means the tail was not all warm. A comparison against the median of
    # the tail would call this a step and condemn a run on one thermal excursion.
    one_slow_tail = (100_000, 10_000, 90_000, 9_800, 10_200)

    assert not steps_after_the_first(one_slow_tail, CRITERIA.step_ratio)


def test_a_first_sample_that_only_just_clears_the_ratio_steps() -> None:
    assert steps_after_the_first((30_001, 10_000, 9_000, 8_000, 7_000), 3.0)
    assert not steps_after_the_first((30_000, 10_000, 9_000, 8_000, 7_000), 3.0)


def test_a_cell_of_one_sample_holds_no_step_because_nothing_follows_its_first() -> None:
    assert MINIMUM_SAMPLES == 2
    assert not steps_after_the_first((10_000,), CRITERIA.step_ratio)
    assert not steps_after_the_first((), CRITERIA.step_ratio)


def test_samples_that_are_all_zero_are_not_a_step() -> None:
    # A scripted duration of zero must not read as a first sample infinitely above the rest.
    assert not steps_after_the_first((0, 0, 0, 0, 0), CRITERIA.step_ratio)


# A drop that stops working shows in every cell at once


def test_a_run_whose_every_cell_steps_is_condemned() -> None:
    report = check_trend(grid(stepping=CELLS_PER_DATASET), CRITERIA)

    assert report.verdict is Verdict.CONDEMNED
    assert len(report.stepped) == CELLS_PER_DATASET
    assert report.examined == CELLS_PER_DATASET


def test_a_run_where_one_cell_alone_steps_is_not_condemned() -> None:
    # One cell stepping is one thermal excursion, which the median already handles. The failure this
    # check finds is machine-wide and appears everywhere at once.
    report = check_trend(grid(stepping=1), CRITERIA)

    assert report.verdict is Verdict.NOT_CONDEMNED
    assert len(report.stepped) == 1
    assert report.examined == CELLS_PER_DATASET


def test_a_run_where_no_cell_steps_is_not_condemned() -> None:
    report = check_trend(grid(stepping=0), CRITERIA)

    assert report.verdict is Verdict.NOT_CONDEMNED
    assert report.stepped == ()


def test_the_share_decides_how_many_stepping_cells_condemn_a_run() -> None:
    half = StepCriteria(step_ratio=3.0, condemning_share=0.5)

    assert check_trend(grid(stepping=8), half).verdict is Verdict.CONDEMNED
    assert check_trend(grid(stepping=7), half).verdict is Verdict.NOT_CONDEMNED


def test_the_stepped_cells_are_named_so_a_reader_can_look_at_them() -> None:
    report = check_trend(grid(stepping=2), CRITERIA)

    assert report.stepped == expected_cells(DATASET)[:2]


def test_the_report_carries_the_criteria_the_verdict_was_taken_under() -> None:
    # A verdict outlives the run, so a later reader can judge the same samples under better numbers.
    report = check_trend(grid(stepping=0), CRITERIA)

    assert report.criteria == CRITERIA


# The check says nothing about a warm run, and nothing about a run it cannot read


def test_a_run_taken_with_a_warm_up_gets_no_verdict_either_way() -> None:
    # Every repetition of a warm run is warm before it is timed, so no step is expected and the
    # absence of one proves nothing.
    warm = grid(stepping=CELLS_PER_DATASET, warmup=True)

    assert check_trend(warm, CRITERIA).verdict is Verdict.NO_VERDICT


def test_a_warm_run_that_holds_no_step_is_not_reported_as_clean_either() -> None:
    assert check_trend(grid(stepping=0, warmup=True), CRITERIA).verdict is Verdict.NO_VERDICT


def test_one_warm_measurement_takes_the_verdict_away_from_the_whole_run() -> None:
    mixed = grid(stepping=CELLS_PER_DATASET)
    mixed[3] = measurement(mixed[3].at, STEPPED, warmup=True)

    assert check_trend(mixed, CRITERIA).verdict is Verdict.NO_VERDICT


def test_a_run_of_one_sample_cells_gets_no_verdict_rather_than_a_clean_one() -> None:
    single = [measurement(at, (10_000,)) for at in expected_cells(DATASET)]

    assert check_trend(single, CRITERIA).verdict is Verdict.NO_VERDICT


def test_no_measurement_at_all_gets_no_verdict() -> None:
    assert check_trend([], CRITERIA).verdict is Verdict.NO_VERDICT


def test_a_cell_too_short_to_read_is_left_out_of_the_share() -> None:
    measurements = [measurement(at, COLD) for at in expected_cells(DATASET)[:3]]
    measurements.extend(measurement(at, (10_000,)) for at in expected_cells(DATASET)[3:])
    measurements[0] = measurement(measurements[0].at, STEPPED)

    # One cell of three is above this share; one cell of sixteen would be far below it.
    report = check_trend(measurements, StepCriteria(step_ratio=3.0, condemning_share=0.33))

    assert report.examined == 3
    assert report.verdict is Verdict.CONDEMNED


# The criteria are refused rather than defaulted


def test_the_check_supplies_no_threshold_of_its_own() -> None:
    # SPEC-0018 pins the shape and no number for it, so a default here would be invented.
    for name in ("step_ratio", "condemning_share"):
        assert StepCriteria.model_fields[name].is_required(), f"{name} carries a default"


@pytest.mark.parametrize("ratio", [1.0, 0.5, 0.0, -2.0])
def test_a_step_ratio_of_one_or_less_is_refused_by_name(ratio: float) -> None:
    # A ratio of one condemns every cell whose first sample is merely the largest of its five.
    with pytest.raises(EvaluationError) as refusal:
        check_trend(grid(stepping=0), StepCriteria(step_ratio=ratio, condemning_share=1.0))

    assert str(ratio) in str(refusal.value)
    assert str(LOWEST_RATIO) in str(refusal.value)


@pytest.mark.parametrize("share", [0.0, -0.5, 1.5])
def test_a_share_outside_the_cells_of_a_run_is_refused_by_name(share: float) -> None:
    with pytest.raises(EvaluationError) as refusal:
        check_trend(grid(stepping=0), StepCriteria(step_ratio=3.0, condemning_share=share))

    assert str(share) in str(refusal.value)


# A condemned run is not published


def test_a_condemned_run_is_refused_naming_the_counts_the_criteria_and_the_cells() -> None:
    report = check_trend(grid(stepping=CELLS_PER_DATASET), CRITERIA)

    with pytest.raises(EvaluationError) as refusal:
        refuse_a_condemned_run(report)

    message = str(refusal.value)
    assert str(CELLS_PER_DATASET) in message
    assert str(CRITERIA.step_ratio) in message
    assert str(CRITERIA.condemning_share) in message
    assert f"{DATASET}/original/pandas/{Task.FIRST_ITEM.value}" in message


@pytest.mark.parametrize("verdict", [Verdict.NOT_CONDEMNED, Verdict.NO_VERDICT])
def test_a_run_that_is_not_condemned_passes_the_refusal_quietly(verdict: Verdict) -> None:
    measurements = grid(stepping=0) if verdict is Verdict.NOT_CONDEMNED else grid(stepping=0, warmup=True)

    assert refuse_a_condemned_run(check_trend(measurements, CRITERIA)) is None


# The check is evidence about the instrument, and it reads nothing


def test_the_check_returns_a_verdict_and_never_a_figure() -> None:
    # It condemns the run or it says nothing. A field that adjusted, discounted or annotated a
    # figure would make a warm number publishable with a caveat beside it.
    report = check_trend(grid(stepping=0), CRITERIA)

    assert set(type(report).model_fields) == {"verdict", "criteria", "stepped", "examined"}
    assert len(Verdict) == 3


@pytest.mark.parametrize(
    "forbidden",
    ["Path", "open", "glob", "rglob", "stat", "read_text", "perf_counter", "monotonic", "subprocess", "sleep"],
)
def test_the_trend_module_reads_no_file_no_clock_and_starts_no_process(forbidden: str) -> None:
    # It is a pure function over sample lists, which is what makes it a check continuous integration
    # can run.
    tree = ast.parse(TREND_MODULE.read_text(encoding="utf-8"))
    named = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert forbidden not in named


def test_the_check_reads_what_the_timing_primitive_already_kept() -> None:
    # The samples are stored for the appendix, so this detector costs nothing further.
    samples = check_trend(grid(stepping=0), CRITERIA)

    assert samples.examined == CELLS_PER_DATASET
    assert len(COLD) == REPEATS
