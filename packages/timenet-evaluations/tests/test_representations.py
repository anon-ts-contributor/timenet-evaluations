"""The two representations, the untimed conversion, and the rule that no module here names a dataset."""

import ast
from importlib.util import find_spec
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from timenet_evaluations import grid
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid import (
    Artifact,
    Cell,
    ItemDeclaration,
    ParsedItems,
    ParsingPath,
    Representation,
    Task,
    representations as representations_package,
    to_comparison_form,
)
from timenet_evaluations.grid.representations import DatasetFacts, Original, TimeF, sample_id
from timenet_evaluations.source import LABEL, PATIENT, SIGNAL


TIMEF_INSTALLED = find_spec("timenet") is not None
"""The timef extra is optional, so the conversion and its parsing path are only importable where it
was installed."""

if TIMEF_INSTALLED:
    from timenet_evaluations.grid.representations import conversion
    from timenet_evaluations.grid.representations.timef_parsing import TimeFParsingPath

needs_timef = pytest.mark.skipif(not TIMEF_INSTALLED, reason="the timef extra is not installed")

DATASET = "stub"


def cell_of(reader: str, task: Task = Task.FULL_READ) -> Cell:
    return Cell(dataset=DATASET, representation=TimeF.name, reader=reader, task=task)


N_ITEMS = 6
N_CHANNELS = 2
N_SAMPLES = 300
ITEM_SHAPE = (N_CHANNELS, N_SAMPLES)
CHANNELS = ("first", "second")

GRID_MODULES = sorted(Path(grid.__file__).parent.rglob("*.py"))

# The words a dataset brings with it. None of them may appear in any module of this package: an
# identifier, a modality, a unit, a channel name, a target schema, a licence, or a label
# vocabulary all arrive as arguments, so a grep for a dataset comes back empty.
DATASET_WORDS = ["sleep", "edf", "eeg", "ecg", "microvolt", "pyhealth", "odbl", "epoch", "sleep_stage"]


class StubTorchItems(Dataset[object]):
    def __init__(self, values: np.ndarray) -> None:
        self.values = torch.from_numpy(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.values[index]


class StubParsingPath:
    """A stand-in loader. It declares the two members and inherits nothing."""

    def __init__(self) -> None:
        self.opened: list[tuple[str, Path]] = []

    def open_pandas(self, path: Path) -> ParsedItems:
        self.opened.append(("pandas", path))
        return np.load(path)

    def open_torch(self, path: Path) -> Dataset[object]:
        self.opened.append(("pytorch", path))
        return StubTorchItems(np.load(path))


PARSING_PATHS: tuple[ParsingPath, ...] = (StubParsingPath(),)


def values_of(n_items: int = N_ITEMS, *, n_channels: int = N_CHANNELS) -> np.ndarray:
    generator = np.random.default_rng(0)
    return generator.standard_normal((n_items, n_channels, N_SAMPLES)).astype(np.float32)


def frame_of(n_items: int = N_ITEMS, *, n_channels: int = N_CHANNELS) -> pd.DataFrame:
    return pd.DataFrame(
        {
            SIGNAL: list(values_of(n_items, n_channels=n_channels)),
            LABEL: [f"class-{index % 3}" for index in range(n_items)],
            PATIENT: [f"subject-{index % 2}" for index in range(n_items)],
        }
    )


@pytest.fixture
def declaration() -> ItemDeclaration:
    return ItemDeclaration(count=N_ITEMS, shape=ITEM_SHAPE)


@pytest.fixture
def parsing_path() -> StubParsingPath:
    return StubParsingPath()


@pytest.fixture
def release(tmp_path: Path) -> Path:
    shipped = tmp_path / "release.npy"
    np.save(shipped, values_of())
    return shipped


@pytest.fixture
def facts() -> DatasetFacts:
    return DatasetFacts(
        card_id="timenet/hello_world",
        modality="stub",
        modality_name="Stub",
        unit="volt",
        rate_hz=100,
        channels=CHANNELS,
        target_schema="stub_class",
    )


@pytest.fixture
def converted(tmp_path: Path, facts: DatasetFacts) -> Artifact:
    return conversion.write(frame_of(), tmp_path / "converted", dataset="stub", facts=facts)


# --- The `Original` Representation -----------------------------------------------------------


def test_original_describes_the_release_files_and_writes_nothing(
    release: Path, declaration: ItemDeclaration, parsing_path: StubParsingPath
) -> None:
    before = release.read_bytes()

    original = Original(dataset="stub", source=release, item=declaration, parsing_path=parsing_path)

    assert original.name == "original"
    assert original.artifact == Artifact(representation="original", path=release)
    assert release.read_bytes() == before


def test_original_refuses_a_path_that_is_not_there(
    tmp_path: Path, declaration: ItemDeclaration, parsing_path: StubParsingPath
) -> None:
    # The path was handed to the run rather than produced by it, so it is the one most likely to be
    # wrong. A missing release is refused here and never reported as a representation of zero bytes.
    missing = tmp_path / "not-shipped"

    with pytest.raises(EvaluationError) as failure:
        Original(dataset="stub", source=missing, item=declaration, parsing_path=parsing_path)

    message = str(failure.value)
    assert "representation 'original'" in message
    assert str(missing) in message
    assert "dataset 'stub'" in message


def test_both_readers_of_original_parse_through_one_loader(
    release: Path, declaration: ItemDeclaration, parsing_path: StubParsingPath
) -> None:
    # The parsing path is a property of the representation. Two readers that each built their own
    # parser would put two different loaders in the two cells of this row, and every number would
    # still render.
    original = Original(dataset="stub", source=release, item=declaration, parsing_path=parsing_path)

    original.parsing_path.open_pandas(original.artifact.path)
    original.parsing_path.open_torch(original.artifact.path)

    assert original.parsing_path is parsing_path
    assert parsing_path.opened == [("pandas", release), ("pytorch", release)]


def test_original_takes_its_item_declaration_from_the_dataset(
    release: Path, declaration: ItemDeclaration, parsing_path: StubParsingPath
) -> None:
    original = Original(dataset="stub", source=release, item=declaration, parsing_path=parsing_path)

    assert original.item is declaration


def test_original_satisfies_the_representation_protocol(
    release: Path, declaration: ItemDeclaration, parsing_path: StubParsingPath
) -> None:
    # `ty` checks the assignment below. Nothing checks it at import time, and no protocol here is
    # runtime checkable.
    representation: Representation = Original(
        dataset="stub", source=release, item=declaration, parsing_path=parsing_path
    )

    assert representation.name == "original"


# --- The `TimeF` Representation --------------------------------------------------------------


def test_timef_describes_the_artifact_the_conversion_produced(
    tmp_path: Path, declaration: ItemDeclaration, parsing_path: StubParsingPath
) -> None:
    produced = tmp_path / "evaluations" / "stub-items" / "1.0.0"
    produced.mkdir(parents=True)

    representation: Representation = TimeF(
        dataset="stub",
        artifact=Artifact(representation="timef", path=produced),
        item=declaration,
        parsing_path=parsing_path,
    )

    assert representation.name == "timef"
    assert representation.artifact.path == produced
    assert representation.item is declaration
    assert representation.parsing_path is parsing_path


def test_timef_refuses_an_artifact_that_names_another_representation(
    release: Path, declaration: ItemDeclaration, parsing_path: StubParsingPath
) -> None:
    with pytest.raises(EvaluationError) as failure:
        TimeF(
            dataset="stub",
            artifact=Artifact(representation="original", path=release),
            item=declaration,
            parsing_path=parsing_path,
        )

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert "artifact representation 'original'" in message


def test_timef_refuses_an_artifact_whose_path_is_not_there(
    tmp_path: Path, declaration: ItemDeclaration, parsing_path: StubParsingPath
) -> None:
    # A path composed from constants is a guess about another library's layout. This is what a
    # stale guess looks like when it arrives, and it stops the run rather than being measured.
    guessed = tmp_path / "evaluations" / "stub-items" / "1.0.0"

    with pytest.raises(EvaluationError) as failure:
        TimeF(
            dataset="stub",
            artifact=Artifact(representation="timef", path=guessed),
            item=declaration,
            parsing_path=parsing_path,
        )

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert str(guessed) in message


def test_the_sample_id_sorts_as_its_position_does() -> None:
    # A narrow read asks the storage library for ids and gets the samples back in stored id order.
    # Only a padded id makes that order ascending position order.
    ids = [sample_id(position) for position in range(11)]

    assert ids == sorted(ids)
    assert sample_id(2) < sample_id(10)


# --- Conversion Is Untimed and Produces the TimeF Artifact -----------------------------------


def test_the_representations_import_without_the_conversion() -> None:
    # The conversion imports the storage library, which is an optional extra. Keeping it out of the
    # package's exports is what lets the two representations import where the extra is absent, and
    # it is also what stops a timed region reaching a conversion through the package it opens a
    # representation from.
    package = ast.parse(Path(representations_package.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(package) if isinstance(node, ast.ImportFrom) and node.module}

    assert set(representations_package.__all__) == {"DatasetFacts", "Original", "TimeF", "sample_id"}
    assert all("conversion" not in module for module in imported)
    assert all("timef_parsing" not in module for module in imported)


@needs_timef
def test_write_creates_a_missing_output_directory_and_returns_what_the_writer_produced(
    tmp_path: Path, facts: DatasetFacts
) -> None:
    out = tmp_path / "absent" / "parents"

    artifact = conversion.write(frame_of(), out, dataset="stub", facts=facts)

    assert artifact.representation == "timef"
    assert artifact.path.is_dir()
    assert (artifact.path / "manifest.json").is_file()
    assert out in artifact.path.parents


@needs_timef
def test_the_same_source_converted_twice_lands_in_the_same_version_directory(
    tmp_path: Path, facts: DatasetFacts
) -> None:
    out = tmp_path / "converted"

    first = conversion.write(frame_of(), out, dataset="stub", facts=facts)
    second = conversion.write(frame_of(), out, dataset="stub", facts=facts)

    assert first.path == second.path
    # The version is the card's, not a field of `facts` any more: ADR-0028 moved identity there.
    assert [path.name for path in first.path.parent.iterdir()] == ["1.0.0"]


@needs_timef
def test_the_conversion_produces_a_representation_that_can_be_constructed(
    converted: Artifact, declaration: ItemDeclaration, parsing_path: StubParsingPath
) -> None:
    representation = TimeF(dataset="stub", artifact=converted, item=declaration, parsing_path=parsing_path)

    assert representation.artifact is converted


@needs_timef
def test_the_conversion_refuses_a_frame_the_layout_cannot_represent(tmp_path: Path, facts: DatasetFacts) -> None:
    out = tmp_path / "converted"

    with pytest.raises(EvaluationError) as failure:
        conversion.write(frame_of(n_channels=N_CHANNELS + 1), out, dataset="stub", facts=facts)

    message = str(failure.value)
    assert "representation 'timef'" in message
    assert "dataset 'stub'" in message
    assert f"declared {N_CHANNELS} channels" in message
    assert not out.exists()


@needs_timef
def test_the_conversion_refuses_a_frame_that_holds_nothing(tmp_path: Path, facts: DatasetFacts) -> None:
    with pytest.raises(EvaluationError) as failure:
        conversion.write(frame_of(0), tmp_path / "converted", dataset="stub", facts=facts)

    assert "representation 'timef'" in str(failure.value)


@needs_timef
def test_the_conversion_translates_a_failure_of_the_storage_library(tmp_path: Path, facts: DatasetFacts) -> None:
    # A library's own exception class must not surface untranslated. The licence used to be the
    # trigger; ADR-0028 moved it to the card, so the trigger is now a unit the storage library's
    # unit registry does not know. It is neither a TypeError nor a ValueError this code anticipated.
    unknown = facts.model_copy(update={"unit": "not-a-unit"})

    with pytest.raises(EvaluationError) as failure:
        conversion.write(frame_of(), tmp_path / "converted", dataset="stub", facts=unknown)

    assert "representation 'timef'" in str(failure.value)


@needs_timef
def test_the_conversion_stores_solely_parquet(converted: Artifact) -> None:
    written = {path.suffix for path in converted.path.rglob("*") if path.is_file()}

    assert conversion.VALUES_BACKEND == "parquet"
    assert ".parquet" in written
    assert written.isdisjoint({".pt", ".pth", ".h5", ".hdf5", ".npy", ".zarr", ".nc"})


# --- The converted form's parsing path -------------------------------------------------------


@needs_timef
def test_the_converted_form_reads_back_the_values_it_was_given(converted: Artifact) -> None:
    path = TimeFParsingPath(dataset="stub", channels=CHANNELS)
    items = path.open_pandas(converted.path)

    assert len(items) == N_ITEMS
    assert np.array_equal(np.asarray(items[0:N_ITEMS]), values_of())


@needs_timef
def test_the_converted_form_serves_a_block_from_the_middle(converted: Artifact) -> None:
    items = TimeFParsingPath(dataset="stub", channels=CHANNELS).open_pandas(converted.path)

    assert np.array_equal(np.asarray(items[2:5]), values_of()[2:5])
    assert np.array_equal(np.asarray(items[3]), values_of()[3])


@needs_timef
def test_the_converted_form_refuses_a_read_past_its_end(converted: Artifact) -> None:
    # A slice that clamped would return fewer items than a block names, which is the short result
    # the failure policy forbids.
    items = TimeFParsingPath(dataset="stub", channels=CHANNELS).open_pandas(converted.path)

    with pytest.raises(EvaluationError) as failure:
        items[N_ITEMS - 1 : N_ITEMS + 2]

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert "reader 'pandas'" in message
    assert f"holds {N_ITEMS}" in message


@needs_timef
def test_the_converted_form_refuses_a_path_that_is_not_a_version(tmp_path: Path) -> None:
    path = TimeFParsingPath(dataset="stub", channels=CHANNELS)

    with pytest.raises(EvaluationError) as failure:
        path.open_pandas(tmp_path / "nothing-here")

    assert "representation 'timef'" in str(failure.value)


@needs_timef
def test_the_converted_forms_torch_item_is_the_signal_and_nothing_else(
    converted: Artifact, declaration: ItemDeclaration
) -> None:
    # A canonical item carries the item's values. A label or an identifier beside it makes the
    # comparison form refuse the item, or reduce content that is not the signal.
    dataset = TimeFParsingPath(dataset="stub", channels=CHANNELS).open_torch(converted.path)

    item = dataset[3]

    assert isinstance(item, torch.Tensor)
    assert item.dtype is torch.float32
    assert tuple(item.shape) == ITEM_SHAPE
    reduced = to_comparison_form(item, declaration, at=cell_of("pytorch"), path=converted.path)
    assert np.array_equal(reduced, values_of()[3])


@needs_timef
def test_both_target_forms_of_the_converted_form_hold_the_same_item(
    converted: Artifact, declaration: ItemDeclaration
) -> None:
    # The two cells of one row must reduce to equal values, with zero tolerance.
    path = TimeFParsingPath(dataset="stub", channels=CHANNELS)

    from_pandas = to_comparison_form(
        path.open_pandas(converted.path)[4],
        declaration,
        at=cell_of("pandas"),
        path=converted.path,
    )
    from_pytorch = to_comparison_form(
        path.open_torch(converted.path)[4],
        declaration,
        at=cell_of("pytorch"),
        path=converted.path,
    )

    assert np.array_equal(from_pandas, from_pytorch)


@needs_timef
def test_the_converted_forms_parsing_path_satisfies_the_seam(converted: Artifact) -> None:
    # `ty` checks the assignment. Both readers of this representation are handed this one object.
    path: ParsingPath = TimeFParsingPath(dataset="stub", channels=CHANNELS)

    assert len(path.open_pandas(converted.path)) == N_ITEMS


# --- Datasets Are an Input to a Run ----------------------------------------------------------


@pytest.mark.parametrize("word", DATASET_WORDS)
def test_no_module_of_the_grid_names_a_dataset(word: str) -> None:
    # Adding a dataset means supplying a dataset. A second dataset with a different modality, a
    # different unit and a different task must need no edit here, so a grep for a dataset's own
    # words comes back empty.
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    for module in GRID_MODULES:
        found = pattern.search(module.read_text(encoding="utf-8"))
        assert found is None, f"{module.name} names {word!r}"


def test_every_dataset_fact_is_required() -> None:
    # A default would be this package naming a dataset. Every fact arrives with the dataset.
    assert all(field.is_required() for field in DatasetFacts.model_fields.values())


def test_the_dataset_facts_carry_what_a_frame_does_not() -> None:
    # The five identity fields are gone: ADR-0028 made the dataset card the single source for the
    # id, version, name, description and licence, and `card_id` names the card instead of copying
    # it. What is left is the half a card does not describe.
    assert set(DatasetFacts.model_fields) == {
        "card_id",
        "channels",
        "modality",
        "modality_name",
        "rate_hz",
        "target_schema",
        "unit",
    }
