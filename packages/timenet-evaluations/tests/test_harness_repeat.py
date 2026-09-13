"""The one timing primitive: what it refuses, what it drops, what it times, and how it reduces.

Every test here scripts a clock, a drop and a read. A scripted read has no page cache, so it cannot
tell a cold read from a warm one, and nothing here is evidence that the cold protocol works. What
these tests pin is the shape of the protocol: that a drop precedes every timed interval, that the
interval holds one read and nothing else, that every sample stays, and that the figure is the
median and not the mean or the minimum.

``Machine`` is the double that makes the ordering observable rather than assumed. One counter is
the clock, the drop and the read both advance it, and every call appends to one log. The log says
what happened in what order, and the samples say which of it was inside the interval.
"""

import ast
import inspect
from pathlib import Path
import sys
import time

import pytest

import timenet_evaluations
from timenet_evaluations.cli import build_parser, main, run_evaluation
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, StorageKey, Task
from timenet_evaluations.harness import repeat as repeat_module
from timenet_evaluations.harness.drop import DARWIN, DropRecord
from timenet_evaluations.harness.repeat import REPEATS, Measurement, Timed, median_ns, repeat


DATASET = "sleep-edfx"

CELL = Cell(dataset=DATASET, representation="timef", reader="pandas", task=Task.BLOCK_SHUFFLED)

ARTIFACT = Path("/data/timef/sleep-edfx")

RECORD = DropRecord(command="/usr/sbin/purge", exit_status=0, platform=DARWIN)

PACKAGE = Path(inspect.getfile(timenet_evaluations)).parent

REPEAT_MODULE = Path(inspect.getfile(repeat_module))

DROP_NS = 1_000_000
"""What the scripted drop costs. It is far larger than any scripted read, so a sample that held it
would be unmistakable."""


class Machine:
    """A clock, a drop and a read over one counter, with one log of what happened in what order.

    The drop and the read both move the counter forward, so a sample says what was inside the timed
    interval and what was not. Nothing here sleeps and nothing starts a process.
    """

    def __init__(self, *read_costs: int) -> None:
        self.now = 0
        self.read_costs = list(read_costs)
        self.log: list[str] = []
        self.drops: list[tuple[Cell | StorageKey, Path]] = []
        self.reads = 0

    def clock(self) -> int:
        self.log.append("clock")
        return self.now

    def drop(self, *, at: Cell | StorageKey, path: Path) -> DropRecord:
        self.log.append("drop")
        self.drops.append((at, path))
        self.now += DROP_NS
        return RECORD

    def read(self) -> object:
        self.log.append("read")
        self.now += self.read_costs[self.reads] if self.reads < len(self.read_costs) else 1
        self.reads += 1
        return None


class RefusingDrop:
    """A drop that reports failure, as a missing executable or a non-zero exit does."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, at: Cell | StorageKey, path: Path) -> DropRecord:
        self.calls += 1
        raise EvaluationError(f"the page cache drop failed: dataset {at.dataset!r}, path {path}")


def timed_of(machine: Machine, at: Cell = CELL) -> Timed:
    return Timed(operation=machine.read, at=at, path=ARTIFACT)


def measure(machine: Machine, *, repeats: int = REPEATS, warmup: bool = False) -> Measurement:
    return repeat(timed_of(machine), drop_caches=machine.drop, repeats=repeats, warmup=warmup, clock=machine.clock)


# What the primitive refuses, before it drops or reads anything


@pytest.mark.parametrize("repeats", [0, -1, -5])
def test_a_repetition_count_below_one_is_refused_before_any_drop(repeats: int) -> None:
    machine = Machine()

    with pytest.raises(EvaluationError) as refusal:
        measure(machine, repeats=repeats)

    assert f"repeats {repeats}" in str(refusal.value)
    assert (machine.log, machine.drops, machine.reads) == ([], [], 0)


@pytest.mark.parametrize("repeats", [2, 4, 6])
def test_an_even_repetition_count_is_refused_rather_than_interpolated(repeats: int) -> None:
    # The requirement leaves a choice: refuse an even count, or record the interpolation as one.
    # This refuses, so every figure a cell reports is a sample that was observed.
    machine = Machine()

    with pytest.raises(EvaluationError) as refusal:
        measure(machine, repeats=repeats)

    assert f"repeats {repeats}" in str(refusal.value)
    assert (machine.log, machine.reads) == ([], 0)


def test_the_refusal_names_all_four_coordinates_of_the_cell() -> None:
    with pytest.raises(EvaluationError) as refusal:
        measure(Machine(), repeats=0)

    message = str(refusal.value)

    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert "reader 'pandas'" in message
    assert f"task {Task.BLOCK_SHUFFLED.value!r}" in message
    assert str(ARTIFACT) in message


def test_an_odd_count_of_one_is_accepted() -> None:
    machine = Machine(7)

    measurement = measure(machine, repeats=1)

    assert (measurement.repetitions, measurement.samples) == (1, (7,))


# The drop: one per repetition, immediately before the interval and outside it


def test_five_repetitions_request_five_drops() -> None:
    machine = Machine()

    measurement = measure(machine, repeats=5)

    assert len(machine.drops) == 5
    assert len(machine.drops) != 1
    assert len(measurement.drops) == 5


def test_each_drop_immediately_precedes_its_own_timed_interval() -> None:
    # The ordering is observed and not assumed: one log, in the order the calls were made.
    machine = Machine()

    measure(machine, repeats=3)

    assert machine.log == ["drop", "clock", "read", "clock"] * 3


def test_a_repetition_that_follows_a_repetition_still_drops() -> None:
    machine = Machine()

    measure(machine, repeats=5)

    assert machine.log.count("drop") == 5
    assert all(machine.log[position] == "drop" for position in range(0, len(machine.log), 4))


def test_the_drop_falls_outside_the_timed_interval() -> None:
    # The drop moves the counter by DROP_NS. No sample holds it, so no figure prices a privileged
    # machine-wide operation as though it were part of reading.
    machine = Machine(11, 13, 17)

    measurement = measure(machine, repeats=3)

    assert measurement.samples == (11, 13, 17)
    assert machine.now == 3 * DROP_NS + 11 + 13 + 17


def test_the_drop_is_told_the_cell_and_the_path_it_precedes() -> None:
    machine = Machine()

    measure(machine, repeats=3)

    assert machine.drops == [(CELL, ARTIFACT)] * 3


def test_a_failing_drop_stops_the_run_before_the_clock_starts() -> None:
    machine = Machine()
    drop = RefusingDrop()

    with pytest.raises(EvaluationError):
        repeat(timed_of(machine), drop_caches=drop, repeats=5, clock=machine.clock)

    assert drop.calls == 1
    assert (machine.log, machine.reads) == ([], 0)


def test_a_drop_that_fails_partway_emits_no_measurement_at_all() -> None:
    machine = Machine()
    failing_after = 2

    def drop(*, at: Cell | StorageKey, path: Path) -> DropRecord:
        if len(machine.drops) == failing_after:
            raise EvaluationError("the page cache drop failed")
        return machine.drop(at=at, path=path)

    with pytest.raises(EvaluationError):
        repeat(timed_of(machine), drop_caches=drop, repeats=5, clock=machine.clock)

    # Two repetitions were recorded and no Measurement carries them, so nothing can report five.
    assert machine.reads == failing_after


# The warm-up: absent by default, and the drop happens either way


def test_a_cold_repetition_performs_exactly_one_read() -> None:
    machine = Machine()

    measurement = measure(machine, repeats=5)

    assert machine.reads == 5
    assert measurement.repetitions == 5
    assert measurement.warmup is False


def test_the_warmup_setting_is_off_by_default() -> None:
    assert inspect.signature(repeat).parameters["warmup"].default is False


def test_zero_warm_ups_is_accepted_because_that_is_the_protocol() -> None:
    # The superseded guard raised on a request for no warm-up. That is now the reported protocol.
    machine = Machine(4, 5, 6)

    measurement = measure(machine, repeats=3, warmup=False)

    assert measurement.samples == (4, 5, 6)


def test_the_warm_mode_still_drops_and_then_reads_twice() -> None:
    machine = Machine()

    measurement = measure(machine, repeats=3, warmup=True)

    assert machine.log == ["drop", "read", "clock", "read", "clock"] * 3
    assert (len(machine.drops), machine.reads, measurement.repetitions) == (3, 6, 3)


def test_the_warm_up_read_is_untimed() -> None:
    machine = Machine(900, 3, 900, 5, 900, 7)

    measurement = measure(machine, repeats=3, warmup=True)

    assert measurement.samples == (3, 5, 7)


def test_the_measurement_records_which_setting_it_was_taken_under() -> None:
    # A cold figure and a warm one are both well formed and differ by several times.
    assert measure(Machine(), warmup=True).warmup is True
    assert measure(Machine(), warmup=False).warmup is False


# The reduction: a median, and neither a mean nor a minimum


def test_the_reduction_is_the_median_and_not_the_mean_or_the_minimum() -> None:
    # All three differ: median 3, mean 20, minimum 1. One primitive feeds sixteen cells per
    # dataset, so the same one-word substitution would move every figure at once.
    samples = (1, 2, 3, 4, 90)

    assert median_ns(samples) == 3
    assert median_ns(samples) != sum(samples) / len(samples)
    assert median_ns(samples) != min(samples)


def test_one_slow_pass_among_cold_ones_does_not_move_the_figure() -> None:
    machine = Machine(10, 10, 1_000, 10, 10)

    measurement = measure(machine, repeats=5)

    assert measurement.elapsed_ns == 10


def test_the_figure_is_a_sample_that_was_observed() -> None:
    machine = Machine(31, 17, 53, 11, 41)

    measurement = measure(machine, repeats=5)

    assert measurement.elapsed_ns in measurement.samples


def test_the_reduction_does_not_depend_on_the_order_the_samples_were_taken_in() -> None:
    assert median_ns((90, 1, 4, 2, 3)) == median_ns((1, 2, 3, 4, 90))


@pytest.mark.parametrize("samples", [(), (1, 2), (1, 2, 3, 4)])
def test_a_sample_count_with_no_median_is_refused(samples: tuple[int, ...]) -> None:
    with pytest.raises(EvaluationError) as refusal:
        median_ns(samples)

    assert f"count {len(samples)}" in str(refusal.value)


def test_the_protocol_takes_five_repetitions() -> None:
    assert REPEATS == 5
    assert REPEATS % 2 == 1


# Every sample is kept, and the count is taken from them


def test_every_sample_is_kept_in_the_order_it_was_taken() -> None:
    machine = Machine(19, 23, 29, 31, 37)

    measurement = measure(machine, repeats=5)

    assert measurement.samples == (19, 23, 29, 31, 37)


def test_a_measurement_cannot_carry_a_median_without_its_samples() -> None:
    # elapsed_ns is computed from the samples, so it is not a field anything can set.
    assert "elapsed_ns" not in Measurement.model_fields
    assert "samples" in Measurement.model_fields


def test_the_stored_result_carries_the_samples_beside_the_figure() -> None:
    machine = Machine(19, 23, 29, 31, 37)

    stored = measure(machine, repeats=5).model_dump()

    assert stored["samples"] == (19, 23, 29, 31, 37)
    assert stored["elapsed_ns"] == 29
    assert stored["repetitions"] == 5


def test_the_recorded_count_is_counted_from_the_samples_and_never_declared() -> None:
    assert "repetitions" not in Measurement.model_fields

    four = Measurement(at=CELL, samples=(1, 2, 3, 4), warmup=False, drops=(RECORD,) * 4)

    assert four.repetitions == 4


def test_the_recorded_count_follows_the_loop_and_not_the_request() -> None:
    machine = Machine()

    measurement = measure(machine, repeats=3)

    assert measurement.repetitions == len(measurement.samples) == 3


def test_a_measurement_names_all_four_coordinates_of_its_cell() -> None:
    assert measure(Machine()).at == CELL


def test_the_measurement_is_frozen() -> None:
    # ty reads a frozen field as a read-only property, so an assignment is a type error rather than
    # a runtime one and cannot be asserted on here.
    assert Measurement.model_config["frozen"] is True
    assert Timed.model_config["frozen"] is True


# One primitive, in the calling process, on a monotonic nanosecond clock


def test_the_primitive_takes_the_operation_the_count_the_setting_and_the_cell() -> None:
    parameters = inspect.signature(repeat).parameters

    assert list(parameters) == ["timed", "drop_caches", "repeats", "warmup", "clock"]
    assert set(Timed.model_fields) == {"operation", "at", "path"}


def test_the_clock_is_monotonic_and_counts_nanoseconds() -> None:
    assert inspect.signature(repeat).parameters["clock"].default is time.perf_counter_ns


def test_the_module_starts_no_process_and_measures_no_memory() -> None:
    tree = ast.parse(REPEAT_MODULE.read_text(encoding="utf-8"))
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom) for alias in node.names
    }
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}

    assert not {"subprocess", "multiprocessing", "resource", "tracemalloc"} & (imported | modules)
    assert "concurrent.futures" not in (imported | modules)


def test_no_task_can_obtain_a_figure_without_this_primitive() -> None:
    # One implementation of the drop, the sampling and the reduction. A private loop elsewhere in
    # the package would be a second one.
    loops = [
        path
        for path in PACKAGE.rglob("*.py")
        if path != REPEAT_MODULE and "perf_counter" in path.read_text(encoding="utf-8")
    ]

    assert [path.name for path in loops] == ["calibration.py"]


# The flag: a boolean, off by default, and the count it replaced is gone


def test_the_run_takes_the_setting_as_a_boolean_that_is_off() -> None:
    parameter = inspect.signature(run_evaluation).parameters["warmup"]

    assert (parameter.annotation, parameter.default) == ("bool", False)
    assert "warmups" not in inspect.signature(run_evaluation).parameters


def test_the_parser_leaves_the_setting_off_when_the_flag_is_absent(tmp_path: Path) -> None:
    # Read off the parser rather than out of a refusal message. These two used to run `main` and
    # assert on the text of the refusal that stood where the run now stands; SPEC-0021 Story 3
    # removed that refusal, and the property they were really checking survives it.
    parsed = build_parser().parse_args([f"{DATASET}={tmp_path}"])

    assert parsed.warmup is False


def test_the_parser_turns_the_setting_on_when_the_flag_is_present(tmp_path: Path) -> None:
    parsed = build_parser().parse_args([f"{DATASET}={tmp_path}", "--warmup"])

    assert parsed.warmup is True


def test_the_flag_takes_no_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["timenet-evaluations", f"{DATASET}=/data/sleep-edfx", "--warmup", "2"])

    with pytest.raises(SystemExit):
        main()


def test_the_superseded_count_is_rejected_rather_than_accepted_as_an_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["timenet-evaluations", f"{DATASET}=/data/sleep-edfx", "--warmups", "1"])

    with pytest.raises(SystemExit):
        main()


def test_the_argument_for_a_warm_up_survives_nowhere_in_the_package() -> None:
    # "the first read pays for a cold page cache" argues against the protocol now in force.
    text = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py"))

    assert "warmups" not in text
    assert "cold page cache" not in text
