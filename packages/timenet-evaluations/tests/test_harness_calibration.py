"""The calibration probe: what it measures, what it refuses, and what it records.

Every test here scripts a clock and a drop. A scripted clock cannot tell a cold read from a warm
one either, so these tests pin the arithmetic and the refusal, not the mechanism. The probe is the
only thing in the harness that can observe whether the drop works, and it can only do so on a
machine, at run time.
"""

import ast
import inspect
from pathlib import Path
import time

import pytest

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, StorageKey
from timenet_evaluations.harness import calibration as calibration_module
from timenet_evaluations.harness.calibration import Calibration, Probe, calibrate
from timenet_evaluations.harness.drop import DARWIN, DropRecord


DATASET = "sleep-edfx"

CALIBRATION_MODULE = Path(inspect.getfile(calibration_module))

STORAGE = StorageKey(dataset=DATASET, representation="timef")

ARTIFACT = Path("/data/timef/sleep-edfx")

RECORD = DropRecord(command="/usr/sbin/purge", exit_status=0, platform=DARWIN)


class ScriptedClock:
    """A monotonic clock that reads the nanosecond counts it was handed, in order."""

    def __init__(self, *readings: int) -> None:
        self.readings = list(readings)
        self.taken = 0

    def __call__(self) -> int:
        reading = self.readings[self.taken]
        self.taken += 1
        return reading


class RecordingDrop:
    """A drop that records the coordinates it was handed and returns a fixed record."""

    def __init__(self, record: DropRecord = RECORD) -> None:
        self.record = record
        self.calls: list[tuple[Cell | StorageKey, Path]] = []

    def __call__(self, *, at: Cell | StorageKey, path: Path) -> DropRecord:
        self.calls.append((at, path))
        return self.record


class CountingRead:
    """A read whose result is discarded and whose call count is the assertion."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        return None


def probe_of(read: CountingRead | None = None) -> Probe:
    return Probe(read=read or CountingRead(), at=STORAGE, path=ARTIFACT)


# One drop and two reads, in that order


def test_the_probe_drops_once_and_reads_twice() -> None:
    read = CountingRead()
    drop = RecordingDrop()

    calibrate(
        probe_of(read),
        drop_caches=drop,
        threshold=3,
        clock=ScriptedClock(0, 10_000, 10_000, 11_000),
    )

    assert (len(drop.calls), read.calls) == (1, 2)


def test_the_drop_is_told_the_artifact_the_probe_reads() -> None:
    drop = RecordingDrop()

    calibrate(probe_of(), drop_caches=drop, threshold=3, clock=ScriptedClock(0, 9, 9, 10))

    assert drop.calls == [(STORAGE, ARTIFACT)]


def test_the_probe_carries_two_coordinates_and_never_a_reader() -> None:
    # It reads one (dataset, representation) artifact, so a StorageKey names it exactly.
    assert probe_of().at == STORAGE
    assert not hasattr(probe_of().at, "reader")


def test_the_ratio_is_the_cold_duration_over_the_warm_one() -> None:
    calibration = calibrate(
        probe_of(),
        drop_caches=RecordingDrop(),
        threshold=3,
        clock=ScriptedClock(1_000, 41_000, 41_000, 46_000),
    )

    assert (calibration.cold_ns, calibration.warm_ns) == (40_000, 5_000)
    assert calibration.ratio == 8


def test_the_probe_uses_the_same_drop_the_grid_will_use() -> None:
    # A probe that checked a second mechanism would say nothing about the one the figures rest on.
    parameters = inspect.signature(calibrate).parameters

    assert list(parameters) == ["probe", "drop_caches", "threshold", "clock"]


def test_the_probe_cannot_be_skipped_by_a_flag() -> None:
    parameters = inspect.signature(calibrate).parameters

    assert not [name for name in parameters if "skip" in name or "enable" in name]


# What the run refuses


@pytest.mark.parametrize("cold_ns", [1_000, 2_000, 3_000])
def test_a_ratio_at_or_below_the_threshold_refuses_the_run(cold_ns: int) -> None:
    with pytest.raises(EvaluationError) as refusal:
        calibrate(
            probe_of(),
            drop_caches=RecordingDrop(),
            threshold=3,
            clock=ScriptedClock(0, cold_ns, cold_ns, cold_ns + 1_000),
        )

    message = str(refusal.value)

    assert f"ratio {cold_ns / 1_000}" in message
    assert "threshold 3" in message
    assert "/usr/sbin/purge" in message
    assert "warm" in message


def test_a_ratio_just_above_the_threshold_is_accepted() -> None:
    calibration = calibrate(
        probe_of(),
        drop_caches=RecordingDrop(),
        threshold=3,
        clock=ScriptedClock(0, 3_001, 3_001, 4_001),
    )

    assert (calibration.cold_ns, calibration.warm_ns) == (3_001, 1_000)
    assert calibration.ratio > calibration.threshold


def test_the_refusal_names_the_status_and_the_platform_of_the_drop_that_ran() -> None:
    with pytest.raises(EvaluationError) as refusal:
        calibrate(
            probe_of(),
            drop_caches=RecordingDrop(),
            threshold=2,
            clock=ScriptedClock(0, 1_000, 1_000, 2_000),
        )

    message = str(refusal.value)

    assert "exit status 0" in message
    assert f"platform {DARWIN!r}" in message


@pytest.mark.parametrize("threshold", [1, 0.5, 0, -2])
def test_a_threshold_that_is_not_above_one_is_refused_before_any_drop(threshold: float) -> None:
    drop = RecordingDrop()
    read = CountingRead()

    with pytest.raises(EvaluationError) as refusal:
        calibrate(probe_of(read), drop_caches=drop, threshold=threshold, clock=ScriptedClock())

    assert f"threshold {threshold}" in str(refusal.value)
    assert (drop.calls, read.calls) == ([], 0)


def test_a_warm_read_of_no_elapsed_time_is_refused_rather_than_divided_by() -> None:
    with pytest.raises(EvaluationError) as refusal:
        calibrate(
            probe_of(),
            drop_caches=RecordingDrop(),
            threshold=3,
            clock=ScriptedClock(0, 5_000, 5_000, 5_000),
        )

    message = str(refusal.value)

    assert "warm_ns 0" in message
    assert f"dataset {DATASET!r}" in message


# What the record carries


def test_the_record_carries_the_threshold_beside_the_ratio() -> None:
    # A ratio alone cannot be re-judged, because a later reader would not know what was demanded.
    calibration = calibrate(
        probe_of(),
        drop_caches=RecordingDrop(),
        threshold=4,
        clock=ScriptedClock(0, 10_000, 10_000, 11_000),
    )

    assert calibration == Calibration(ratio=10, threshold=4, cold_ns=10_000, warm_ns=1_000, drop=RECORD)


def test_the_record_carries_the_drop_that_the_probe_issued() -> None:
    record = DropRecord(command="/bin/sh -c 'sync'", exit_status=0, platform="linux")

    calibration = calibrate(
        probe_of(),
        drop_caches=RecordingDrop(record),
        threshold=3,
        clock=ScriptedClock(0, 8_000, 8_000, 1_000 + 8_000),
    )

    assert calibration.drop == record


def test_the_record_is_frozen() -> None:
    # ty reads a frozen field as a read-only property, so an assignment is a type error rather than
    # a runtime one and cannot be asserted on here.
    assert Calibration.model_config["frozen"] is True
    assert Probe.model_config["frozen"] is True


def test_a_failing_drop_stops_the_probe_before_it_reads() -> None:
    read = CountingRead()

    def refusing_drop(*, at: Cell | StorageKey, path: Path) -> DropRecord:
        raise EvaluationError(f"the page cache drop failed: {at.dataset!r}, {path}")

    with pytest.raises(EvaluationError):
        calibrate(probe_of(read), drop_caches=refusing_drop, threshold=3, clock=ScriptedClock())

    assert read.calls == 0


# What this module must never do


def test_the_clock_is_monotonic_and_counts_nanoseconds() -> None:
    assert inspect.signature(calibrate).parameters["clock"].default is time.perf_counter_ns


def test_the_threshold_has_no_default() -> None:
    # A default here would be a number nobody chose, sitting in the one check that decides whether
    # every figure in the run is real.
    parameter = inspect.signature(calibrate).parameters["threshold"]

    assert parameter.default is inspect.Parameter.empty


def test_the_module_starts_no_process_and_sleeps_for_nothing() -> None:
    tree = ast.parse(CALIBRATION_MODULE.read_text(encoding="utf-8"))
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}

    assert "subprocess" not in imported
    assert "sleep" not in names
    assert "run" not in names
