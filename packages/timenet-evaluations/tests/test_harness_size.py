"""Size on disk: what it counts, what it refuses, and what it never does."""

import ast
import inspect
from pathlib import Path

import pytest

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.harness import measure_size, size as size_module


DATASET = "sleep-edfx"

SIZE_MODULE = Path(inspect.getfile(size_module))


def test_a_file_measures_its_own_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "epochs.parquet"
    artifact.write_bytes(b"\x00" * 512)

    assert measure_size(artifact, dataset=DATASET, representation="timef") == 512


def test_a_directory_measures_the_sum_of_the_files_beneath_it(tmp_path: Path) -> None:
    release = tmp_path / "sleep-edf"
    (release / "subject-00").mkdir(parents=True)
    (release / "subject-00" / "signal.edf").write_bytes(b"\x00" * 300)
    (release / "subject-00" / "hypnogram.edf").write_bytes(b"\x00" * 40)
    (release / "index.csv").write_bytes(b"\x00" * 60)

    assert measure_size(release, dataset=DATASET, representation="original") == 400


def test_a_directory_does_not_measure_its_own_entry(tmp_path: Path) -> None:
    release = tmp_path / "sleep-edf"
    (release / "empty-subject").mkdir(parents=True)
    (release / "index.csv").write_bytes(b"\x00" * 7)

    # The nested directory holds no file, so it contributes nothing at all.
    assert measure_size(release, dataset=DATASET, representation="original") == 7


def test_an_empty_directory_measures_zero_without_raising(tmp_path: Path) -> None:
    artifact = tmp_path / "timef"
    artifact.mkdir()

    assert measure_size(artifact, dataset=DATASET, representation="timef") == 0


def test_an_empty_file_measures_zero_without_raising(tmp_path: Path) -> None:
    artifact = tmp_path / "epochs.parquet"
    artifact.touch()

    assert measure_size(artifact, dataset=DATASET, representation="timef") == 0


def test_a_missing_source_is_refused_and_never_reported_as_zero(tmp_path: Path) -> None:
    absent = tmp_path / "sleep-edf"

    with pytest.raises(EvaluationError) as refusal:
        measure_size(absent, dataset=DATASET, representation="original")

    message = str(refusal.value)

    assert f"dataset {DATASET!r}" in message
    assert "representation 'original'" in message
    assert str(absent) in message
    # A size has no reader and no task, and the message leaves both out rather than naming one
    # that was not there.
    assert "reader '" not in message
    assert "task '" not in message


def test_a_converted_artifact_that_was_never_written_is_refused(tmp_path: Path) -> None:
    # Conversion reported success and left nothing at the path it named.
    named = tmp_path / "timef" / "sleep-edfx" / "1"

    with pytest.raises(EvaluationError) as refusal:
        measure_size(named, dataset=DATASET, representation="timef")

    message = str(refusal.value)
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert str(named) in message


def test_one_function_answers_for_both_representations(tmp_path: Path) -> None:
    release = tmp_path / "sleep-edf"
    release.mkdir()
    (release / "signal.edf").write_bytes(b"\x00" * 900)
    converted = tmp_path / "timef"
    converted.mkdir()
    (converted / "epochs.parquet").write_bytes(b"\x00" * 300)

    original = measure_size(release, dataset=DATASET, representation="original")
    timef = measure_size(converted, dataset=DATASET, representation="timef")

    assert (original, timef) == (900, 300)


def test_counting_the_same_bytes_again_gives_the_same_integer(tmp_path: Path) -> None:
    artifact = tmp_path / "timef"
    artifact.mkdir()
    (artifact / "epochs.parquet").write_bytes(b"\x00" * 128)

    figures = [measure_size(artifact, dataset=DATASET, representation="timef") for _ in range(5)]

    assert figures == [128] * 5


@pytest.mark.parametrize(
    "forbidden", ["median", "perf_counter", "monotonic", "subprocess", "purge", "repeat", "drop_caches"]
)
def test_size_is_counted_and_never_timed(forbidden: str) -> None:
    # The names the module uses, not the prose it carries: the docstring says what size is not.
    tree = ast.parse(SIZE_MODULE.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert forbidden not in names


def test_a_storage_figure_carries_no_reader_coordinate() -> None:
    parameters = inspect.signature(measure_size).parameters

    assert list(parameters) == ["path", "dataset", "representation"]
    assert "reader" not in parameters
