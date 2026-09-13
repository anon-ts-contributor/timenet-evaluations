"""The grid under test: the hand-written subjects, the four-coordinate key, and completeness."""

import ast
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import Dataset

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid import (
    CELLS_PER_DATASET,
    READERS,
    REPRESENTATIONS,
    STORAGE_FIGURES_PER_DATASET,
    TASKS,
    Artifact,
    Cell,
    ItemDeclaration,
    ParsedItems,
    Representation,
    StorageKey,
    Task,
    assemble,
    check_artifact_agrees,
    check_grid_complete,
    expected_cells,
    expected_storage_keys,
    registry as registry_module,
)
from timenet_evaluations.grid.readers import PANDAS, PYTORCH


DATASET = "a-dataset"
OTHER = "another-dataset"
ITEM_SHAPE = (2, 4)

REGISTRY = Path(registry_module.__file__)


class StubParsingPath:
    """A stand-in loader. It declares the two members and inherits nothing."""

    def open_pandas(self, path: Path) -> ParsedItems:
        return np.load(path)

    def open_torch(self, path: Path) -> Dataset[object]:
        raise AssertionError("assembling the grid never opens a representation")


class StubRepresentation:
    """A representation that declares the four members. Nothing here opens anything."""

    def __init__(self, name: str, *, artifact_name: str | None = None) -> None:
        self.dataset = DATASET
        self.name = name
        self.artifact = Artifact(representation=artifact_name or name, path=Path(f"/nowhere/{name}"))
        self.item = ItemDeclaration(count=4, shape=ITEM_SHAPE)
        self.parsing_path = StubParsingPath()


def representations_of(*names: str) -> tuple[Representation, ...]:
    return tuple(StubRepresentation(name) for name in names)


@pytest.fixture
def declared() -> tuple[Representation, ...]:
    return representations_of(*REPRESENTATIONS)


def cells_of(dataset: str) -> set[Cell]:
    return set(expected_cells(dataset))


def storage_of(dataset: str) -> set[StorageKey]:
    return set(expected_storage_keys(dataset))


def test_the_grid_is_two_representations_two_readers_and_four_tasks() -> None:
    assert REPRESENTATIONS == ("original", "timef")
    assert tuple(reader.name for reader in READERS) == (PANDAS, PYTORCH)
    assert TASKS == (Task.FIRST_ITEM, Task.FULL_READ, Task.SEQUENTIAL, Task.BLOCK_SHUFFLED)


def test_one_dataset_produces_sixteen_cells_and_two_storage_figures() -> None:
    assert CELLS_PER_DATASET == 16
    assert STORAGE_FIGURES_PER_DATASET == 2
    assert len(expected_cells(DATASET)) == CELLS_PER_DATASET
    assert len(expected_storage_keys(DATASET)) == STORAGE_FIGURES_PER_DATASET


def test_the_reader_registry_carries_the_annotation_that_is_the_whole_gate() -> None:
    # Widened to tuple[Any, ...] or dropped, two unrelated classes in a tuple type-check as their
    # own union and `ty` verifies nothing. The annotation is the conformance mechanism.
    assert registry_module.__annotations__["READERS"] == "tuple[Reader, ...]"


@pytest.mark.parametrize(
    "discovery",
    ["entry_points", "iter_modules", "pkgutil", "find_spec", "import_module", "glob", "rglob"],
)
def test_no_subject_is_discovered(discovery: str) -> None:
    # A subject that arrived because it happened to be installed would make two runs on one
    # machine measure different grids, and a result has no field to record that.
    assert discovery not in REGISTRY.read_text(encoding="utf-8")


def test_the_subjects_are_written_out_as_literal_tuples() -> None:
    tree = ast.parse(REGISTRY.read_text(encoding="utf-8"))
    assigned = {
        node.target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert isinstance(assigned["REPRESENTATIONS"], ast.Tuple)
    assert isinstance(assigned["READERS"], ast.Tuple)
    assert len(assigned["READERS"].elts) == 2


def test_a_timed_cell_is_keyed_by_four_coordinates() -> None:
    # A key of fewer coordinates cannot name a cell, and a two-coordinate key means a reader has
    # been pinned to a representation somewhere upstream.
    assert set(Cell.model_fields) == {"dataset", "representation", "reader", "task"}


def test_a_storage_figure_is_keyed_by_two_and_never_by_a_reader() -> None:
    assert set(StorageKey.model_fields) == {"dataset", "representation"}


def test_a_cell_is_hashable_so_a_run_can_key_its_numbers_by_one() -> None:
    cell = Cell(dataset=DATASET, representation="timef", reader=PANDAS, task=Task.FULL_READ)

    assert cell in {cell}
    # `ty` refuses an assignment to a member of either key, so the freeze is checked statically as
    # well. What this asserts is the part a run depends on: a cell is usable as a mapping key.
    assert Cell.model_config["frozen"] is True
    assert StorageKey.model_config["frozen"] is True


def test_every_reader_reads_every_representation(declared: tuple[Representation, ...]) -> None:
    pairings = assemble(DATASET, declared)

    pairs = {(pairing.representation.name, pairing.reader.name) for pairing in pairings}

    assert pairs == {(name, reader.name) for name in REPRESENTATIONS for reader in READERS}
    for reader in READERS:
        read = {pairing.representation.name for pairing in pairings if pairing.reader is reader}
        assert read == set(REPRESENTATIONS)


def test_the_assembled_grid_names_the_cells_the_axes_declare(declared: tuple[Representation, ...]) -> None:
    pairings = assemble(DATASET, declared)

    measured = [cell for pairing in pairings for cell in pairing.cells]

    assert len(measured) == CELLS_PER_DATASET
    assert measured == list(expected_cells(DATASET))
    assert {pairing.storage_key for pairing in pairings} == storage_of(DATASET)


def test_a_representation_pinned_to_one_reader_is_refused() -> None:
    # The grid collapsed to a diagonal leaves no cell in which the representation moves while the
    # reader is held still.
    with pytest.raises(EvaluationError, match="whole declared axis"):
        assemble(DATASET, representations_of("timef"))


def test_a_representation_the_axis_does_not_declare_is_refused() -> None:
    with pytest.raises(EvaluationError, match="whole declared axis"):
        assemble(DATASET, representations_of("original", "torch"))


def test_the_representations_must_arrive_in_the_declared_order() -> None:
    with pytest.raises(EvaluationError, match="whole declared axis"):
        assemble(DATASET, representations_of("timef", "original"))


def test_a_representation_that_disagrees_with_its_artifact_is_refused() -> None:
    # Two independent members with nothing keeping them equal. If they drift, a storage figure is
    # filed under the wrong row and every number still renders.
    drifted = StubRepresentation("timef", artifact_name="original")

    with pytest.raises(EvaluationError) as refusal:
        check_artifact_agrees(drifted, dataset=DATASET)

    message = str(refusal.value)
    assert "do not name the same representation" in message
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert str(drifted.artifact.path) in message
    assert "artifact representation 'original'" in message

    with pytest.raises(EvaluationError, match="do not name the same representation"):
        assemble(DATASET, (StubRepresentation("original"), drifted))


def test_a_representation_that_agrees_with_its_artifact_passes() -> None:
    check_artifact_agrees(StubRepresentation("original"), dataset=DATASET)


def test_a_complete_run_over_two_datasets() -> None:
    cells = cells_of(DATASET) | cells_of(OTHER)
    storage = storage_of(DATASET) | storage_of(OTHER)

    assert len(cells) == 2 * CELLS_PER_DATASET
    assert len(storage) == 2 * STORAGE_FIGURES_PER_DATASET

    check_grid_complete([DATASET, OTHER], cells, storage)


def test_a_result_holding_three_of_the_four_pairs_is_incomplete() -> None:
    cells = {cell for cell in expected_cells(DATASET) if (cell.representation, cell.reader) != ("timef", PYTORCH)}

    with pytest.raises(EvaluationError, match="incomplete"):
        check_grid_complete([DATASET], cells, storage_of(DATASET))


def test_one_cell_measured_on_fewer_tasks_is_incomplete() -> None:
    # Four tasks for three cells, and only the full read for the fourth.
    cells = {
        cell
        for cell in expected_cells(DATASET)
        if (cell.representation, cell.reader) != ("timef", PYTORCH) or cell.task is Task.FULL_READ
    }

    with pytest.raises(EvaluationError, match="incomplete"):
        check_grid_complete([DATASET], cells, storage_of(DATASET))


def test_a_missing_storage_figure_is_refused() -> None:
    storage = {StorageKey(dataset=DATASET, representation="original")}

    with pytest.raises(EvaluationError, match="one storage figure per representation"):
        check_grid_complete([DATASET], cells_of(DATASET), storage)


def test_a_reader_indexed_storage_figure_is_refused() -> None:
    # Four storage numbers for one dataset assert that a reader changes bytes on disk.
    storage = storage_of(DATASET) | {
        StorageKey(dataset=DATASET, representation=f"{name}-{reader.name}")
        for name in REPRESENTATIONS
        for reader in READERS
    }

    with pytest.raises(EvaluationError, match="indexed by a reader"):
        check_grid_complete([DATASET], cells_of(DATASET), storage)


def test_a_cell_naming_a_subject_the_grid_does_not_declare_is_refused() -> None:
    cells = cells_of(DATASET) | {Cell(dataset=DATASET, representation="torch", reader=PANDAS, task=Task.FULL_READ)}

    with pytest.raises(EvaluationError, match="does not declare"):
        check_grid_complete([DATASET], cells, storage_of(DATASET))


def test_a_cell_naming_a_dataset_the_run_was_not_given_is_refused() -> None:
    with pytest.raises(EvaluationError, match="was not given"):
        check_grid_complete([DATASET], cells_of(DATASET) | cells_of(OTHER), storage_of(DATASET))


def test_a_storage_figure_naming_a_dataset_the_run_was_not_given_is_refused() -> None:
    with pytest.raises(EvaluationError, match="was not given"):
        check_grid_complete([DATASET], cells_of(DATASET), storage_of(DATASET) | storage_of(OTHER))


def test_a_dataset_named_twice_is_refused() -> None:
    with pytest.raises(EvaluationError, match="each dataset once"):
        check_grid_complete([DATASET, DATASET], cells_of(DATASET), storage_of(DATASET))
