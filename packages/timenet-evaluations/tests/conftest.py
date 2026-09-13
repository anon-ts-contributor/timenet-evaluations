"""The run a derivation test and a recomputation test both need, built once.

The builders sit here rather than in a test module for one reason. The recomputation test must
import nothing from ``timenet_evaluations.metrics``, so it cannot build a derived record for
itself. It takes the serialized record through the ``run_record`` fixture and does its own
arithmetic over the parsed JSON.

Nothing here names a dataset. The names are letters, the unit is the word the frame contract uses,
and the counts are two different numbers, because a rate divides by the count of its own dataset.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from timenet_evaluations.grid.cell import Cell, StorageKey, Task
from timenet_evaluations.harness.drop import DARWIN, DropRecord
from timenet_evaluations.harness.repeat import Measurement
from timenet_evaluations.metrics import derive_metrics
from timenet_evaluations.result import (
    CacheProtocol,
    DatasetRecord,
    Environment,
    EvaluationResult,
    Measurements,
    Metadata,
    Metrics,
    RepresentationRecord,
    StorageMeasurement,
)


DATASETS = ("alpha", "beta")
"""The two datasets the built run read. A run over two is what shows a rate dividing by its own."""

DECLARED_COUNTS = {"alpha": 1000, "beta": 250}
"""What each dataset declared before the run. The two numbers differ, so a rate that divided by the
wrong dataset's count would give a different number."""

REPRESENTATIONS = ("original", "timef")

READERS = ("pandas", "pytorch")

RATE_TASK_VALUES = ("sequential", "block_shuffled")
"""The two rate tasks, written as the strings the record holds. The recomputation test needs them
and may not import the tuple the derivation uses."""

RECORD_PATH = Path("/runs/4f2a91c0e3bd/result.json")
"""The record a refusal names. Nothing opens it."""

DROP = DropRecord(command="/usr/sbin/purge", exit_status=0, platform=DARWIN)

STARTED_AT = datetime(2026, 9, 6, 11, 30, tzinfo=UTC)


def cells() -> tuple[Cell, ...]:
    """Name every timed cell of the built run: sixteen per dataset."""
    return tuple(
        Cell(dataset=dataset, representation=representation, reader=reader, task=task)
        for dataset in DATASETS
        for representation in REPRESENTATIONS
        for reader in READERS
        for task in Task
    )


def samples_of(position: int) -> tuple[int, ...]:
    """Give one cell five distinct durations whose median is the first of them.

    The durations differ from cell to cell, so a rate filed under the wrong cell is a different
    number and not the same one twice.
    """
    base = 100 * (position + 1)
    return (3 * base, base, 2 * base, 4 * base, 5 * base)


def build_measurements() -> Measurements:
    """Build sixteen timed entries and two storage entries for each dataset."""
    return Measurements(
        timed=tuple(
            Measurement(at=at, samples=samples_of(position), warmup=False, drops=(DROP,) * 5)
            for position, at in enumerate(cells())
        ),
        storage=tuple(
            StorageMeasurement(at=StorageKey(dataset=dataset, representation=representation), size_bytes=size)
            for dataset in DATASETS
            for representation, size in zip(REPRESENTATIONS, (4_000_000_000, 900_000_000), strict=True)
        ),
    )


def build_datasets() -> tuple[DatasetRecord, ...]:
    """Build one metadata record per dataset, each carrying its own declared count."""
    return tuple(
        DatasetRecord(
            dataset=dataset,
            source=Path("/data") / dataset,
            unit="item",
            item_count=DECLARED_COUNTS[dataset],
            representations=tuple(
                RepresentationRecord(
                    at=StorageKey(dataset=dataset, representation=representation),
                    block_bytes=64 * 1024 * 1024,
                    seed=7,
                    disclosure=None,
                )
                for representation in REPRESENTATIONS
            ),
        )
        for dataset in DATASETS
    )


def build_metadata() -> Metadata:
    """Build the circumstance group of the built run."""
    return Metadata(
        run_id="4f2a91c0e3bd",
        started_at=STARTED_AT,
        datasets=build_datasets(),
        environment=Environment(
            python="3.13.1",
            platform="darwin-arm64",
            cpu_count=10,
            packages={"timenet-evaluations": "0.1.0"},
        ),
        protocol=CacheProtocol(
            warmup=False,
            drop_command="/usr/sbin/purge",
            drop_exit_status=0,
            drop_platform=DARWIN,
            calibration_ratio=6.25,
            calibration_threshold=3.0,
        ),
    )


def build_result(metrics: Metrics | None = None) -> EvaluationResult:
    """Build one whole result, with the derived metrics unless a caller supplies its own."""
    measurements = build_measurements()
    datasets = build_datasets()

    return EvaluationResult(
        metadata=build_metadata(),
        measurements=measurements,
        metrics=metrics if metrics is not None else derive_metrics(measurements, datasets, path=RECORD_PATH),
    )


@pytest.fixture
def run_measurements() -> Measurements:
    return build_measurements()


@pytest.fixture
def run_datasets() -> tuple[DatasetRecord, ...]:
    return build_datasets()


@pytest.fixture
def run_result() -> EvaluationResult:
    return build_result()


@pytest.fixture
def run_record() -> str:
    """Serialize the built result, which is the only thing the recomputation test is given."""
    return build_result().model_dump_json()
