"""The two readers, their four read operations, and the four cells they make of one dataset."""

import ast
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset, default_collate

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid import (
    Artifact,
    Block,
    Cell,
    ItemDeclaration,
    ParsedItems,
    Reader,
    Task,
    readers as readers_package,
)
from timenet_evaluations.grid.item import to_comparison_form
from timenet_evaluations.grid.readers import (
    BATCH_SIZE,
    PANDAS,
    PYTORCH,
    WORKER_COUNT,
    PandasOpened,
    PandasReader,
    PyTorchOpened,
    PyTorchReader,
)


N_ITEMS = 12
N_CHANNELS = 2
N_SAMPLES = 5
ITEM_SHAPE = (N_CHANNELS, N_SAMPLES)
INTERIOR = 7

# A plan whose final block is short, so the short-block case is in every walk rather than in one
# test of its own. Both readers of one representation walk this same plan.
PLAN = ((0, 5), (5, 5), (10, 2))

DATASET = "a-dataset"

READERS: tuple[Reader, ...] = (PandasReader(), PyTorchReader())
REPRESENTATIONS = ("original", "timef")

READER_MODULES = sorted(Path(readers_package.__file__).parent.glob("*.py"))


class FakeBlock(NamedTuple):
    start: int
    count: int


class FakeItems:
    # Positional access to items held in one file on disk. Every read goes to the file, and every
    # position a read touched is recorded, so a test can tell a narrow read from a full one.
    def __init__(self, path: Path, touched: list[int]) -> None:
        self._values = np.load(path, mmap_mode="r")
        self._touched = touched

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int | slice) -> object:
        if isinstance(index, slice):
            self._touched.extend(range(*index.indices(len(self._values))))
        else:
            self._touched.append(index)

        return np.asarray(self._values[index], dtype=np.float32)


class FakeTorchItems(Dataset[object]):
    def __init__(self, path: Path, touched: list[int]) -> None:
        self._values = np.load(path, mmap_mode="r")
        self._touched = touched

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> torch.Tensor:
        self._touched.append(index)

        return torch.from_numpy(np.array(self._values[index], dtype=np.float32))


class FakeParsingPath:
    # One loader, opened by both readers of one representation. It counts its opens, so a test can
    # tell one open and many reads from one open per read.
    def __init__(self) -> None:
        self.opens = 0
        self.touched: list[int] = []

    def open_pandas(self, path: Path) -> ParsedItems:
        self.opens += 1
        return FakeItems(path, self.touched)

    def open_torch(self, path: Path) -> Dataset[object]:
        self.opens += 1
        return FakeTorchItems(path, self.touched)


class FakeRepresentation:
    def __init__(self, name: str, path: Path, item: ItemDeclaration) -> None:
        self.dataset = DATASET
        self.name = name
        self.artifact = Artifact(representation=name, path=path)
        self.item = item
        self.parsing_path = FakeParsingPath()


def cell_of(representation: str, reader: str, task: Task = Task.FULL_READ) -> Cell:
    return Cell(dataset=DATASET, representation=representation, reader=reader, task=task)


@pytest.fixture
def declaration() -> ItemDeclaration:
    return ItemDeclaration(count=N_ITEMS, shape=ITEM_SHAPE)


@pytest.fixture
def values() -> np.ndarray:
    generator = np.random.default_rng(0)
    return generator.standard_normal((N_ITEMS, *ITEM_SHAPE)).astype(np.float32)


@pytest.fixture
def representations(tmp_path: Path, values: np.ndarray, declaration: ItemDeclaration) -> dict[str, FakeRepresentation]:
    # Two representations holding the same content, which is what the parity requirement of
    # SPEC-0016 establishes before any cell of a real run is timed.
    built = {}
    for name in REPRESENTATIONS:
        path = tmp_path / f"{name}.npy"
        np.save(path, values)
        built[name] = FakeRepresentation(name, path, declaration)

    return built


def item_at(result: object, position: int, reader: str, declaration: ItemDeclaration) -> np.ndarray:
    """Take one canonical item out of a bulk or block result, in either target form."""
    if reader == PANDAS:
        assert isinstance(result, pd.DataFrame)
        rows = declaration.shape[0]
        item = result.iloc[position * rows : (position + 1) * rows]
    else:
        assert isinstance(result, list)
        item = result[position]

    return to_comparison_form(item, declaration, at=cell_of("fake", reader), path=Path("fake"))


def item_count(result: object, reader: str, declaration: ItemDeclaration) -> int:
    """Report how many canonical items a bulk or block result holds, in either target form."""
    if reader == PANDAS:
        assert isinstance(result, pd.DataFrame)
        return len(result) // declaration.shape[0]

    assert isinstance(result, list)
    return len(result)


def open_cell(reader: Reader, representation: FakeRepresentation) -> PandasOpened | PyTorchOpened:
    """Open one cell of the grid, which is one (representation, reader) pair."""
    opened = reader.open(representation)
    assert isinstance(opened, PandasOpened | PyTorchOpened)

    return opened


@pytest.fixture(params=[(name, index) for name in REPRESENTATIONS for index in range(len(READERS))])
def cell(
    request: pytest.FixtureRequest, representations: dict[str, FakeRepresentation]
) -> tuple[FakeRepresentation, Reader]:
    # Every reader opens every representation. The four parameters of this fixture are the four
    # cells of one dataset, and they are exactly the four the superseded model forbade.
    name, index = request.param

    return representations[name], READERS[index]


def test_the_grid_holds_every_representation_crossed_with_every_reader(
    representations: dict[str, FakeRepresentation],
) -> None:
    pairs = {(representation.name, reader.name) for representation in representations.values() for reader in READERS}

    assert pairs == {("original", "pandas"), ("original", "pytorch"), ("timef", "pandas"), ("timef", "pytorch")}
    assert {reader.name for reader in READERS} == {PANDAS, PYTORCH}


def test_a_reader_needs_no_import_no_base_class_and_no_registration() -> None:
    # `ty` checks both readers against the `tuple[Reader, ...]` annotation on READERS above, and
    # nothing checks them at import time. Neither reader inherits from a protocol or from the other.
    assert PandasReader.__bases__ == (object,)
    assert PyTorchReader.__bases__ == (object,)
    assert PandasOpened.__bases__ == (object,)
    assert PyTorchOpened.__bases__ == (object,)


# REQ "The First-Item Read"


def test_read_first_returns_the_canonical_item_at_position_zero(
    cell: tuple[FakeRepresentation, Reader], values: np.ndarray, declaration: ItemDeclaration
) -> None:
    representation, reader = cell
    opened = open_cell(reader, representation)

    first = to_comparison_form(
        opened.read_first(),
        declaration,
        at=cell_of(representation.name, reader.name, Task.FIRST_ITEM),
        path=representation.artifact.path,
    )

    assert np.array_equal(first, values[0])


def test_read_first_does_not_materialize_the_remaining_items(cell: tuple[FakeRepresentation, Reader]) -> None:
    representation, reader = cell
    opened = open_cell(reader, representation)

    opened.read_first()

    assert representation.parsing_path.touched == [0]


def test_read_first_equals_the_first_item_of_the_full_read(
    cell: tuple[FakeRepresentation, Reader], declaration: ItemDeclaration
) -> None:
    representation, reader = cell

    first = to_comparison_form(
        open_cell(reader, representation).read_first(),
        declaration,
        at=cell_of(representation.name, reader.name, Task.FIRST_ITEM),
        path=representation.artifact.path,
    )
    bulk = open_cell(reader, representation).read_all()

    assert np.array_equal(first, item_at(bulk, 0, reader.name, declaration))


# REQ "The Full Read"


def test_read_all_delivers_every_declared_item_in_ascending_position_order(
    cell: tuple[FakeRepresentation, Reader], values: np.ndarray, declaration: ItemDeclaration
) -> None:
    representation, reader = cell

    bulk = open_cell(reader, representation).read_all()

    assert item_count(bulk, reader.name, declaration) == N_ITEMS
    for position in range(N_ITEMS):
        assert np.array_equal(item_at(bulk, position, reader.name, declaration), values[position])


def test_read_all_returns_a_materialized_result_and_touches_every_item(
    cell: tuple[FakeRepresentation, Reader],
) -> None:
    representation, reader = cell

    bulk = open_cell(reader, representation).read_all()

    assert isinstance(bulk, pd.DataFrame | list)
    assert representation.parsing_path.touched == list(range(N_ITEMS))


def test_the_four_cells_of_one_dataset_reduce_to_equal_items(
    representations: dict[str, FakeRepresentation], values: np.ndarray, declaration: ItemDeclaration
) -> None:
    # Item identity across the row, at position zero and at a sampled interior position, with zero
    # tolerance. The two readers count the same object or the row must not be printed.
    for position in (0, INTERIOR):
        reduced = [
            item_at(open_cell(reader, representation).read_all(), position, reader.name, declaration)
            for representation in representations.values()
            for reader in READERS
        ]

        assert len(reduced) == 4
        for item in reduced:
            assert item.dtype == np.float32
            assert item.shape == ITEM_SHAPE
            assert np.array_equal(item, values[position])


@pytest.mark.parametrize("operation", ["read_sequential", "read_block"])
def test_no_read_is_derived_from_the_full_read(operation: str) -> None:
    # The walk and the block read are separate tasks. Either one served from `read_all` would time
    # a bulk pass and report it under a column that exists to measure something else.
    for module in READER_MODULES:
        assert "read_all" not in _names_in(module, operation)


# REQ "The Sequential Walk"


def test_read_sequential_yields_every_item_once_in_ascending_position_order(
    cell: tuple[FakeRepresentation, Reader], values: np.ndarray, declaration: ItemDeclaration
) -> None:
    representation, reader = cell
    opened = open_cell(reader, representation)

    walked = [
        to_comparison_form(
            item,
            declaration,
            at=cell_of(representation.name, reader.name, Task.SEQUENTIAL),
            path=representation.artifact.path,
        )
        for item in opened.read_sequential()
    ]

    assert len(walked) == N_ITEMS
    for position, item in enumerate(walked):
        assert np.array_equal(item, values[position])


def test_read_sequential_materializes_one_item_at_a_time(cell: tuple[FakeRepresentation, Reader]) -> None:
    representation, reader = cell
    opened = open_cell(reader, representation)

    walk = opened.read_sequential()
    next(walk)

    assert representation.parsing_path.touched == [0]


def test_read_sequential_reopens_nothing(cell: tuple[FakeRepresentation, Reader]) -> None:
    representation, reader = cell
    opened = open_cell(reader, representation)

    list(opened.read_sequential())

    assert representation.parsing_path.opens == 1


# REQ "The Block Walk"


def test_read_block_returns_exactly_the_items_the_block_names(
    cell: tuple[FakeRepresentation, Reader], values: np.ndarray, declaration: ItemDeclaration
) -> None:
    representation, reader = cell
    opened = open_cell(reader, representation)

    block = opened.read_block(FakeBlock(8, 4))

    assert item_count(block, reader.name, declaration) == 4
    for offset in range(4):
        assert np.array_equal(item_at(block, offset, reader.name, declaration), values[8 + offset])


def test_read_block_reads_nothing_outside_the_block(cell: tuple[FakeRepresentation, Reader]) -> None:
    representation, reader = cell
    opened = open_cell(reader, representation)

    opened.read_block(FakeBlock(8, 4))

    assert representation.parsing_path.touched == [8, 9, 10, 11]


def test_a_short_final_block_is_served_like_every_other_block(
    cell: tuple[FakeRepresentation, Reader], values: np.ndarray, declaration: ItemDeclaration
) -> None:
    representation, reader = cell
    start, count = PLAN[-1]
    opened = open_cell(reader, representation)

    block = opened.read_block(FakeBlock(start, count))

    assert item_count(block, reader.name, declaration) == count
    assert np.array_equal(item_at(block, 0, reader.name, declaration), values[start])


def test_one_repetition_is_one_open_and_many_block_reads(cell: tuple[FakeRepresentation, Reader]) -> None:
    representation, reader = cell
    opened = open_cell(reader, representation)

    for start, count in PLAN:
        opened.read_block(FakeBlock(start, count))

    assert representation.parsing_path.opens == 1
    assert sorted(representation.parsing_path.touched) == list(range(N_ITEMS))


def test_a_block_read_is_not_served_from_a_retained_result(
    cell: tuple[FakeRepresentation, Reader], declaration: ItemDeclaration
) -> None:
    # The locator may be held for the whole task. A result may not: a representation materialized
    # once and sliced N times reports one read and N slices under a heading that says shuffled rate.
    representation, reader = cell
    opened = open_cell(reader, representation)

    opened.read_all()
    representation.parsing_path.touched.clear()
    opened.read_block(FakeBlock(5, 5))

    assert representation.parsing_path.touched == [5, 6, 7, 8, 9]
    assert representation.parsing_path.opens == 1


def test_both_readers_of_one_representation_walk_the_same_plan(
    representations: dict[str, FakeRepresentation], values: np.ndarray, declaration: ItemDeclaration
) -> None:
    for representation in representations.values():
        for reader in READERS:
            opened = open_cell(reader, representation)
            walked = []
            for start, count in PLAN:
                block = opened.read_block(FakeBlock(start, count))
                walked.extend(item_at(block, offset, reader.name, declaration) for offset in range(count))

            assert len(walked) == N_ITEMS
            assert np.array_equal(np.stack(walked), values)


def test_read_block_refuses_a_block_the_dataset_did_not_declare(
    cell: tuple[FakeRepresentation, Reader],
) -> None:
    representation, reader = cell
    opened = open_cell(reader, representation)

    with pytest.raises(EvaluationError) as failure:
        opened.read_block(FakeBlock(N_ITEMS - 1, 4))

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert f"representation {representation.name!r}" in message
    assert f"reader {reader.name!r}" in message
    assert "task 'block_shuffled'" in message
    assert str(representation.artifact.path) in message
    assert "declared count 12" in message
    assert "block start 11" in message
    assert "block count 4" in message


@pytest.mark.parametrize("block", [FakeBlock(-1, 2), FakeBlock(0, 0)])
def test_read_block_refuses_a_block_that_names_no_item(cell: tuple[FakeRepresentation, Reader], block: Block) -> None:
    representation, reader = cell
    opened = open_cell(reader, representation)

    with pytest.raises(EvaluationError):
        opened.read_block(block)


# REQ "Both Readers Are Configured Identically On Both Representations"


def test_the_dataloader_is_fixed_and_identical_on_both_representations(
    representations: dict[str, FakeRepresentation],
) -> None:
    reader = PyTorchReader()
    loaders = [reader.open(representation)._loader for representation in representations.values()]

    for loader in loaders:
        assert loader.batch_size == BATCH_SIZE == 1
        assert loader.num_workers == WORKER_COUNT == 0
        assert loader.collate_fn is default_collate

    assert {type(loader.sampler) for loader in loaders} == {type(loaders[0].sampler)}


def test_no_custom_collate_function_is_introduced() -> None:
    # A collate function would sit inside all eight timed PyTorch cells and would have to be proved
    # equivalent across representations before either column meant anything.
    source = _source_of("torch_.py")

    assert "collate_fn" not in source


def test_the_dataloader_is_constructed_in_exactly_one_place() -> None:
    # One wrapper, used for the whole dataset and for a block's subset alike, so no read and no
    # representation can be tuned against another.
    source = _source_of("torch_.py")

    assert source.count("DataLoader(") == 1
    assert "def build_loader(" in source


@pytest.mark.parametrize("module", READER_MODULES, ids=lambda module: module.name)
def test_no_reader_names_a_representation(module: Path) -> None:
    # A reader that branched on which representation it was given would be two readers wearing one
    # name, and the difference between the two rows would stop being the representation.
    source = module.read_text(encoding="utf-8")

    for name in REPRESENTATIONS:
        assert f'"{name}"' not in source
        assert f"'{name}'" not in source


# The invariants the seam holds, extended over the readers built on it


def test_nothing_under_readers_absorbs_a_failure() -> None:
    for module in READER_MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            assert _raises_evaluation_error(node), f"{module.name} catches an error and does not raise it again"


@pytest.mark.parametrize("forbidden", ["directory_size", "measure_size", "size_bytes", "st_size", "pyhealth"])
def test_nothing_under_readers_measures_a_size_or_names_a_dataset_library(forbidden: str) -> None:
    for module in READER_MODULES:
        assert forbidden not in module.read_text(encoding="utf-8"), f"{module.name} names {forbidden}"


def test_every_import_sits_at_module_level() -> None:
    for module in READER_MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    assert not isinstance(child, ast.Import | ast.ImportFrom), f"{module.name} defers an import"


def _source_of(name: str) -> str:
    return (Path(readers_package.__file__).parent / name).read_text(encoding="utf-8")


def _names_in(module: Path, function: str) -> set[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != function:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
            if isinstance(child, ast.Attribute):
                names.add(child.attr)

    return names


def _raises_evaluation_error(handler: ast.ExceptHandler) -> bool:
    for child in ast.walk(handler):
        if not isinstance(child, ast.Raise) or not isinstance(child.exc, ast.Call):
            continue
        raised = child.exc.func
        if isinstance(raised, ast.Name) and raised.id == "EvaluationError":
            return True

    return False
