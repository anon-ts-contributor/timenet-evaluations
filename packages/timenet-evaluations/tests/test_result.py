"""The three groups, the keys their entries carry, and what construction refuses.

Every test here builds a result out of the values the harness already produces. Nothing is timed,
nothing is written, and no artifact exists: the shape of the object is the whole subject.

"Before anything is written" is what construction means in this file. The writer is downstream and
is not built yet, so a refusal that happens while the result is being built is a refusal no writer
can be reached past.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import pytest

from timenet_evaluations import result as result_module
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, OpenKey, StorageKey, Task
from timenet_evaluations.harness.drop import DARWIN, DropRecord
from timenet_evaluations.harness.repeat import Measurement
from timenet_evaluations.result import (
    CacheProtocol,
    DatasetRecord,
    Environment,
    EvaluationResult,
    Measurements,
    Metadata,
    Metrics,
    Rate,
    RepresentationRecord,
    StorageMeasurement,
    collect_environment,
)


DATASET = "sleep-edfx"

SOURCE = Path("/data/sleep-edfx")

REPRESENTATIONS = ("original", "timef")

READERS = ("pandas", "pytorch")

RATE_TASKS = (Task.SEQUENTIAL, Task.BLOCK_SHUFFLED)

RECORD = DropRecord(command="/usr/sbin/purge", exit_status=0, platform=DARWIN)

ENVIRONMENT = Environment(
    python="3.13.1",
    platform="darwin-arm64",
    cpu_count=10,
    packages={"timenet-evaluations": "0.1.0"},
)

STARTED_AT = datetime(2026, 9, 6, 11, 30, tzinfo=UTC)


def protocol(*, warmup: bool = False) -> CacheProtocol:
    return CacheProtocol(
        warmup=warmup,
        drop_command="/usr/sbin/purge",
        drop_exit_status=0,
        drop_platform=DARWIN,
        calibration_ratio=6.25,
        calibration_threshold=3.0,
    )


def dataset_record(*, disclosure: str | None = None) -> DatasetRecord:
    return DatasetRecord(
        dataset=DATASET,
        source=SOURCE,
        unit="epoch",
        item_count=1000,
        representations=tuple(
            RepresentationRecord(
                at=StorageKey(dataset=DATASET, representation=representation),
                block_bytes=64 * 1024 * 1024,
                seed=7,
                disclosure=disclosure if representation == "original" else None,
            )
            for representation in REPRESENTATIONS
        ),
    )


def metadata(*, warmup: bool = False, disclosure: str | None = None) -> Metadata:
    return Metadata(
        run_id="4f2a91c0e3bd",
        started_at=STARTED_AT,
        datasets=(dataset_record(disclosure=disclosure),),
        environment=ENVIRONMENT,
        protocol=protocol(warmup=warmup),
    )


def cells() -> tuple[Cell, ...]:
    return tuple(
        Cell(dataset=DATASET, representation=representation, reader=reader, task=task)
        for representation in REPRESENTATIONS
        for reader in READERS
        for task in Task
    )


def measurement(at: Cell, *, warmup: bool = False) -> Measurement:
    return Measurement(
        at=at,
        samples=(30, 10, 20, 40, 50),
        warmup=warmup,
        drops=(RECORD,) * 5,
    )


def measurements(*, warmup: bool = False, odd_cell: Cell | None = None) -> Measurements:
    return Measurements(
        timed=tuple(measurement(at, warmup=warmup if at != odd_cell else not warmup) for at in cells()),
        storage=tuple(
            StorageMeasurement(
                at=StorageKey(dataset=DATASET, representation=representation),
                size_bytes=size,
            )
            for representation, size in zip(REPRESENTATIONS, (4_000_000_000, 900_000_000), strict=True)
        ),
    )


def metrics() -> Metrics:
    return Metrics(
        rates=tuple(Rate(at=at, items_per_second=1000 / (30 / 1e9)) for at in cells() if at.task in RATE_TASKS)
    )


def result(*, warmup: bool = False, disclosure: str | None = None) -> EvaluationResult:
    return EvaluationResult(
        metadata=metadata(warmup=warmup, disclosure=disclosure),
        measurements=measurements(warmup=warmup),
        metrics=metrics(),
    )


def test_environment_records_the_interpreter_and_packages() -> None:
    environment = collect_environment()

    assert environment.python
    assert environment.cpu_count > 0
    assert environment.packages


def test_collect_environment_refuses_a_run_that_would_carry_no_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(result_module, "installed_packages", dict)

    with pytest.raises(EvaluationError, match="no installed distribution reported a name"):
        collect_environment()


def test_a_result_is_composed_of_exactly_three_groups() -> None:
    assert tuple(EvaluationResult.model_fields) == ("metadata", "measurements", "metrics")


def test_one_dataset_gives_sixteen_timed_entries_and_two_storage_entries() -> None:
    built = result()

    assert len(built.measurements.timed) == 16
    assert len(built.measurements.storage) == 2
    assert {(entry.at.representation, entry.at.reader) for entry in built.measurements.timed} == {
        (representation, reader) for representation in REPRESENTATIONS for reader in READERS
    }


@pytest.mark.parametrize(
    ("group", "payload"),
    [
        ("metadata", {"run_id": "4f2a91c0e3bd", "started_at": STARTED_AT}),
        ("measurements", {"timed": ()}),
        ("metrics", {}),
    ],
)
def test_a_missing_field_in_any_group_raises_at_construction(group: str, payload: dict[str, Any]) -> None:
    groups: dict[str, Any] = {
        "metadata": metadata(),
        "measurements": measurements(),
        "metrics": metrics(),
    }
    groups[group] = payload

    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(groups)


@pytest.mark.parametrize(
    ("group", "field", "value"),
    [
        ("metadata", "started_at", "the middle of the afternoon"),
        ("metadata", "environment", "a laptop"),
        ("measurements", "timed", ("a fast one",)),
        ("measurements", "storage", (1_000,)),
        ("metrics", "rates", (12.5,)),
    ],
)
def test_a_field_of_the_wrong_type_in_any_group_raises_at_construction(group: str, field: str, value: object) -> None:
    groups: dict[str, Any] = {
        "metadata": metadata().model_dump(),
        "measurements": measurements().model_dump(),
        "metrics": metrics().model_dump(),
    }
    groups[group][field] = value

    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(groups)


def test_a_group_of_the_wrong_type_raises_at_construction() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(
            {"metadata": metadata(), "measurements": measurements(), "metrics": "fast enough"}
        )


def test_a_record_parses_back_into_the_same_result() -> None:
    built = result(disclosure="the timed cells read PyHealth's materialized epoch cache")

    parsed = EvaluationResult.model_validate_json(built.model_dump_json())

    assert parsed == built
    assert parsed.metadata.started_at == STARTED_AT
    assert parsed.metadata.started_at.tzinfo is not None
    assert parsed.metadata.datasets[0].source == SOURCE
    assert (
        parsed.metadata.datasets[0].representations[0].disclosure
        == built.metadata.datasets[0].representations[0].disclosure
    )
    assert parsed.measurements.timed == built.measurements.timed
    assert parsed.metrics == built.metrics


def test_the_record_carries_the_cache_protocol_the_timings_were_taken_under() -> None:
    written = json.loads(result().model_dump_json())["metadata"]["protocol"]

    assert written == {
        "warmup": False,
        "drop_command": "/usr/sbin/purge",
        "drop_exit_status": 0,
        "drop_platform": DARWIN,
        "calibration_ratio": 6.25,
        "calibration_threshold": 3.0,
    }


def test_a_timed_entry_is_keyed_by_all_four_coordinates() -> None:
    assert Measurements.model_fields["timed"].annotation == tuple[Measurement, ...]
    assert Measurement.model_fields["at"].annotation is Cell
    assert tuple(Cell.model_fields) == ("dataset", "representation", "reader", "task")


def test_a_timed_entry_cannot_be_filed_under_a_key_of_lower_arity() -> None:
    open_key = OpenKey(dataset=DATASET, representation="timef", reader="pandas")

    with pytest.raises(ValidationError, match="valid dictionary or instance of Cell"):
        Measurement.model_validate({"at": open_key, "samples": (1,), "warmup": False, "drops": (RECORD,)})


def test_a_storage_entry_is_keyed_by_two_coordinates_and_names_no_reader() -> None:
    assert StorageMeasurement.model_fields["at"].annotation is StorageKey
    assert tuple(StorageKey.model_fields) == ("dataset", "representation")


def test_a_storage_entry_cannot_be_filed_under_a_key_that_names_a_reader() -> None:
    cell = Cell(dataset=DATASET, representation="timef", reader="pandas", task=Task.FULL_READ)

    with pytest.raises(ValidationError, match="valid dictionary or instance of StorageKey"):
        StorageMeasurement.model_validate({"at": cell, "size_bytes": 900_000_000})


def test_the_block_plan_and_the_disclosure_carry_no_reader_coordinate() -> None:
    assert RepresentationRecord.model_fields["at"].annotation is StorageKey

    plans = {record.at.representation: (record.block_bytes, record.seed) for record in dataset_record().representations}

    assert len(plans) == len(REPRESENTATIONS)


def test_a_representation_with_nothing_to_disclose_carries_no_statement() -> None:
    records = dataset_record(disclosure="the timed cells read a materialized cache").representations

    assert records[0].disclosure == "the timed cells read a materialized cache"
    assert records[1].disclosure is None


def test_a_result_whose_measurements_contradict_its_metadata_about_warmup_refuses() -> None:
    contradicting = Cell(dataset=DATASET, representation="timef", reader="pytorch", task=Task.SEQUENTIAL)

    with pytest.raises(EvaluationError, match="disagree with the metadata about the warm-up setting") as raised:
        EvaluationResult(
            metadata=metadata(),
            measurements=measurements(odd_cell=contradicting),
            metrics=metrics(),
        )

    message = str(raised.value)
    assert "metadata warmup False" in message
    assert "1 of 16 timed entries carry True" in message
    assert "'timef', 'pytorch', 'sequential'" in message


def test_a_warm_run_whose_measurements_agree_is_well_formed() -> None:
    built = result(warmup=True)

    assert built.metadata.protocol.warmup is True
    assert all(entry.warmup for entry in built.measurements.timed)


def test_a_metric_names_the_measurement_key_it_was_computed_from() -> None:
    assert tuple(Rate.model_fields) == ("at", "items_per_second")
    assert Rate.model_fields["at"].annotation is Cell
    assert {rate.at.task for rate in metrics().rates} == set(RATE_TASKS)
