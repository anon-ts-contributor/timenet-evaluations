"""The reader seam, the artifact, the canonical item, and the error contract."""

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from timenet_evaluations import grid
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid import (
    READERS,
    REPRESENTATIONS,
    TASKS,
    Artifact,
    Block,
    Cell,
    ItemDeclaration,
    Opened,
    OpenKey,
    ParsedItems,
    ParsingPath,
    Reader,
    Representation,
    StorageKey,
    Task,
    check_item_count,
    failure_message,
    to_comparison_form,
)


N_ITEMS = 4
ITEM_SHAPE = (2, 4)
DATASET = "a-dataset"

GRID_PACKAGE = Path(grid.__file__).parent
# rglob, not glob: `readers/` and `representations/` are where the filesystem boundary lives, so a
# rule that skipped them would be checking the modules least able to break it.
GRID_MODULES = sorted(GRID_PACKAGE.rglob("*.py"))

COORDINATE_LABELS = ("dataset", "representation", "reader", "task")
# Read off the registry rather than written out here. A third reader extends the rule by being
# added to the axis, and a rule that had to be edited beside it is one nobody edits.
AXIS_VALUES = (*REPRESENTATIONS, *(reader.name for reader in READERS), *(task.value for task in TASKS))


@pytest.fixture
def declaration() -> ItemDeclaration:
    return ItemDeclaration(count=N_ITEMS, shape=ITEM_SHAPE)


@pytest.fixture
def values() -> np.ndarray:
    generator = np.random.default_rng(0)
    return generator.standard_normal((N_ITEMS, *ITEM_SHAPE)).astype(np.float32)


@pytest.fixture
def representation(tmp_path: Path, values: np.ndarray, declaration: ItemDeclaration) -> Representation:
    path = tmp_path / "items.npy"
    np.save(path, values)
    return StubRepresentation(Artifact(representation="original", path=path), declaration)


# A stand-in reader that drives the seam without pandas or torch. It declares no base class,
# subclasses nothing, and registers itself nowhere: it satisfies the seam by shape alone. The
# protocols this module imports are used to annotate the registry below and to assert that no
# isinstance guard exists, never as a base. `ty` checks the stand-in against that annotation, and
# nothing checks it at import time.
class StubBlock(NamedTuple):
    start: int
    count: int


class StubTorchItems(Dataset[object]):
    def __init__(self, path: Path) -> None:
        self.values = torch.from_numpy(np.load(path))

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.values[index]


class StubParsingPath:
    # The one loader both readers of a representation parse through. This stand-in serves the
    # seam's own tests, whose reader opens the file itself. The readers' own tests drive a parsing
    # path that really parses.
    def open_pandas(self, path: Path) -> ParsedItems:
        return np.load(path)

    def open_torch(self, path: Path) -> Dataset[object]:
        return StubTorchItems(path)


class StubRepresentation:
    def __init__(self, artifact: Artifact, item: ItemDeclaration) -> None:
        self.dataset = DATASET
        self.name = "original"
        self.artifact = artifact
        self.item = item
        self.parsing_path = StubParsingPath()


def cell_of(representation: str, reader: str, task: Task = Task.FULL_READ) -> Cell:
    return Cell(dataset=DATASET, representation=representation, reader=reader, task=task)


class StubOpened:
    def __init__(self, path: Path, item: ItemDeclaration) -> None:
        # A binding, and nothing derived from the bytes. Every read opens the file itself.
        self.path = path
        self.item = item

    def read_first(self) -> np.ndarray:
        return np.load(self.path)[0]

    def read_all(self) -> np.ndarray:
        return np.load(self.path)

    def read_sequential(self) -> Iterator[np.ndarray]:
        return iter(np.load(self.path))

    def read_block(self, block: Block) -> np.ndarray:
        return np.load(self.path)[block.start : block.start + block.count]


class StubReader:
    name = "stub"

    def open(self, representation: Representation) -> StubOpened:
        return StubOpened(representation.artifact.path, representation.item)


READERS: tuple[Reader, ...] = (StubReader(),)
BLOCKS: tuple[Block, ...] = (StubBlock(1, 2),)


def test_a_third_reader_needs_no_import_no_base_class_and_no_registration(
    representation: Representation, values: np.ndarray, declaration: ItemDeclaration
) -> None:
    # `ty` checks StubReader against the `tuple[Reader, ...]` annotation on READERS above. The seam
    # hands back an item in the reader's own target form, which the comparison form reduces.
    assert StubReader.__bases__ == (object,)
    assert StubOpened.__bases__ == (object,)

    reader = READERS[0]
    opened = reader.open(representation)

    assert reader.name == "stub"
    first = to_comparison_form(
        opened.read_first(),
        declaration,
        at=cell_of(representation.name, reader.name, Task.FIRST_ITEM),
        path=representation.artifact.path,
    )
    assert np.array_equal(first, values[0])


def test_the_four_read_operations_deliver_the_declared_items(
    representation: Representation, values: np.ndarray
) -> None:
    opened = StubReader().open(representation)

    assert np.array_equal(opened.read_all(), values)
    assert np.array_equal(np.stack(list(opened.read_sequential())), values)
    assert np.array_equal(opened.read_block(BLOCKS[0]), values[1:3])


@pytest.mark.parametrize("protocol", [Reader, Opened, Representation, Block, ParsingPath, ParsedItems])
def test_no_protocol_is_runtime_checkable(protocol: type) -> None:
    # No isinstance guard exists, so `ty` is the whole conformance gate.
    with pytest.raises(TypeError):
        isinstance(StubReader(), protocol)


def test_no_protocol_member_carries_a_body() -> None:
    # A body would give an implementation something to inherit and turn a structural seam into a
    # nominal one.
    members = 0
    for module in GRID_MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not _is_protocol(node):
                continue
            for member in node.body:
                if not isinstance(member, ast.FunctionDef):
                    continue
                members += 1
                assert _body_is_ellipsis(member), (
                    f"{module.relative_to(GRID_PACKAGE)}:{node.name}.{member.name} has a body"
                )

    assert members == 16


def test_nothing_under_grid_absorbs_a_failure() -> None:
    # A failure stops the run, and it stops it as the harness's own error type. A handler translates
    # an error and raises EvaluationError from it. None may swallow one, substitute a default,
    # continue, or let a third-party exception through untranslated.
    handlers = 0
    for module in GRID_MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            handlers += 1
            assert _raises_evaluation_error(node), (
                f"{module.relative_to(GRID_PACKAGE)} catches an error and does not raise EvaluationError"
            )

    assert handlers >= 1


@pytest.mark.parametrize("forbidden", ["directory_size", "measure_size", "size_bytes", "st_size"])
def test_nothing_under_grid_measures_a_size(forbidden: str) -> None:
    # "Nothing here measures itself" is meant to be checkable by grep rather than by argument.
    for module in GRID_MODULES:
        assert forbidden not in module.read_text(encoding="utf-8"), (
            f"{module.relative_to(GRID_PACKAGE)} names {forbidden}"
        )


def test_artifact_carries_exactly_a_representation_and_a_path() -> None:
    assert set(Artifact.model_fields) == {"representation", "path"}


def test_artifact_describes_a_representation_this_harness_did_not_write(tmp_path: Path) -> None:
    release = tmp_path / "sleep-edf"
    release.mkdir()

    artifact = Artifact(representation="original", path=release)

    assert artifact.representation == "original"
    assert artifact.path == release


def test_both_target_forms_reduce_to_one_comparison_form(declaration: ItemDeclaration, tmp_path: Path) -> None:
    item = np.arange(8, dtype=np.float32).reshape(ITEM_SHAPE)
    frame_form = pd.DataFrame(item)
    tensor_form = torch.from_numpy(item)

    from_pandas = to_comparison_form(frame_form, declaration, at=cell_of("timef", "pandas"), path=tmp_path)
    from_pytorch = to_comparison_form(tensor_form, declaration, at=cell_of("timef", "pytorch"), path=tmp_path)

    assert from_pandas.dtype == np.float32
    assert from_pandas.shape == ITEM_SHAPE
    assert np.array_equal(from_pandas, from_pytorch)
    assert np.array_equal(from_pandas, item)


def test_comparison_form_refuses_an_item_of_the_wrong_shape(declaration: ItemDeclaration, tmp_path: Path) -> None:
    coarser = np.zeros((3, *ITEM_SHAPE), dtype=np.float32)

    with pytest.raises(EvaluationError) as failure:
        to_comparison_form(coarser, declaration, at=cell_of("original", "pytorch"), path=tmp_path)

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert "representation 'original'" in message
    assert "reader 'pytorch'" in message
    assert "task 'full_read'" in message
    assert str(tmp_path) in message
    assert "(3, 2, 4)" in message


def test_comparison_form_refuses_values_it_cannot_reduce(declaration: ItemDeclaration, tmp_path: Path) -> None:
    with pytest.raises(EvaluationError) as failure:
        to_comparison_form(object(), declaration, at=cell_of("timef", "pandas"), path=tmp_path)

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert "reader 'pandas'" in message
    assert "task 'full_read'" in message
    assert str(tmp_path) in message


def test_comparison_form_translates_a_target_form_that_refuses_to_become_an_array(
    declaration: ItemDeclaration, tmp_path: Path
) -> None:
    # A tensor that holds a gradient raises RuntimeError, which is neither TypeError nor ValueError.
    # No third-party exception leaves this seam untranslated.
    holds_a_gradient = torch.zeros(ITEM_SHAPE, requires_grad=True)

    with pytest.raises(EvaluationError) as failure:
        to_comparison_form(holds_a_gradient, declaration, at=cell_of("timef", "pytorch"), path=tmp_path)

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert "reader 'pytorch'" in message
    assert "task 'full_read'" in message
    assert str(tmp_path) in message


def test_check_item_count_accepts_the_declared_count(declaration: ItemDeclaration, tmp_path: Path) -> None:
    check_item_count(N_ITEMS, declaration, at=cell_of("timef", "pandas"), path=tmp_path)


def test_check_item_count_refuses_a_count_the_dataset_did_not_declare(
    declaration: ItemDeclaration, tmp_path: Path
) -> None:
    with pytest.raises(EvaluationError) as failure:
        check_item_count(N_ITEMS - 1, declaration, at=cell_of("timef", "pytorch", Task.SEQUENTIAL), path=tmp_path)

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert "reader 'pytorch'" in message
    assert "task 'sequential'" in message
    assert str(tmp_path) in message
    assert "declared count 4" in message
    assert "yielded count 3" in message


def test_a_timed_read_names_every_one_of_the_cells_four_coordinates(tmp_path: Path) -> None:
    # Four coordinates key a cell, so a message naming fewer cannot say which cell failed.
    cell = Cell(dataset=DATASET, representation="original", reader="pandas", task=Task.BLOCK_SHUFFLED)

    message = failure_message("the representation is not there", at=cell, path=tmp_path)

    assert "the representation is not there" in message
    assert f"dataset {DATASET!r}" in message
    assert "representation 'original'" in message
    assert "reader 'pandas'" in message
    assert "task 'block_shuffled'" in message
    assert f"path {tmp_path}" in message


def test_two_failures_in_different_datasets_do_not_read_the_same(tmp_path: Path) -> None:
    # A run measures more than one dataset. Without the dataset coordinate both grids write one
    # message and the reader of a traceback cannot tell which grid failed.
    coordinates = {"representation": "timef", "reader": "pytorch", "task": Task.FULL_READ}

    one = failure_message("the read failed", at=Cell(dataset="first", **coordinates), path=tmp_path)
    other = failure_message("the read failed", at=Cell(dataset="second", **coordinates), path=tmp_path)

    assert one != other
    assert "dataset 'first'" in one
    assert "dataset 'second'" in other


def test_the_open_scope_carries_exactly_three_coordinates() -> None:
    assert set(OpenKey.model_fields) == {"dataset", "representation", "reader"}
    assert OpenKey.model_config["frozen"] is True


def test_a_failure_in_an_open_names_the_three_coordinates_it_has(tmp_path: Path) -> None:
    # One open serves all four tasks, so an open has a reader and no task. The reader is a
    # coordinate and it sits where the other coordinates sit.
    key = OpenKey(dataset=DATASET, representation="timef", reader="pandas")

    message = failure_message("the converted form cannot be read", at=key, path=tmp_path)

    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert "reader 'pandas'" in message
    assert "task '" not in message
    assert message.index("representation 'timef'") < message.index("reader 'pandas'")
    assert message.index("reader 'pandas'") < message.index(f"path {tmp_path}")


def test_a_failure_with_no_reader_and_no_task_names_the_two_coordinates_it_has(tmp_path: Path) -> None:
    # A conversion has no reader and no task, and the message says so by leaving both out rather
    # than by naming coordinates that were not there.
    key = StorageKey(dataset=DATASET, representation="timef")

    message = failure_message("the output directory cannot be created", at=key, path=tmp_path)

    assert "reader '" not in message
    assert "task '" not in message
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert f"path {tmp_path}" in message


def test_a_detail_follows_the_coordinates_and_never_replaces_one(tmp_path: Path) -> None:
    # The detail carries the values behind a failure. A coordinate has its own place in the
    # message, so nothing has to smuggle one through here.
    key = StorageKey(dataset=DATASET, representation="timef")

    message = failure_message("the conversion failed", at=key, path=tmp_path, detail="position 3")

    assert message.endswith("position 3")
    assert message.index(f"dataset {DATASET!r}") < message.index("position 3")


def test_no_detail_under_grid_carries_a_coordinate() -> None:
    # Every scope that occurs has a type, so no raise site needs `detail` to carry a coordinate.
    # The check reads each call's format template, not the rendered message: a literal segment is
    # what the author wrote, and an interpolation collapses to `{}` because no value is known here.
    # A word blocklist would be wrong. "artifact representation {}" names the representation an
    # artifact claims, and "dataset id {}" names a TimeF dataset id, and both are the point of
    # their message.
    details = 0
    for module in GRID_MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                template = _format_template(keyword)
                if template is None:
                    continue
                details += 1
                where = f"{module.relative_to(GRID_PACKAGE)}: detail {template!r}"
                for value in AXIS_VALUES:
                    assert value not in template, f"{where} names {value!r}, which is a value of a grid axis"
                for field in template.split(", "):
                    assert not _labels_a_coordinate(field), f"{where} labels a coordinate in {field!r}"

    assert details >= 1


def _format_template(keyword: ast.keyword) -> str | None:
    # The format template of a `detail` argument, or None where the argument is not one.
    if keyword.arg != "detail":
        return None

    node = keyword.value
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.JoinedStr):
        return None

    segments = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            segments.append(value.value)
        else:
            segments.append("{}")

    return "".join(segments)


def _labels_a_coordinate(field: str) -> bool:
    # One field of a detail labels a coordinate where the label is followed by its own value: an
    # interpolation or a quoted literal, and not another word.
    for label in COORDINATE_LABELS:
        if not field.startswith(label):
            continue
        if field[len(label) :].lstrip(" ")[:1] in {"{", "'", '"'}:
            return True

    return False


def _raises_evaluation_error(handler: ast.ExceptHandler) -> bool:
    for child in ast.walk(handler):
        if not isinstance(child, ast.Raise) or not isinstance(child.exc, ast.Call):
            continue
        raised = child.exc.func
        if isinstance(raised, ast.Name) and raised.id == "EvaluationError":
            return True
        if isinstance(raised, ast.Attribute) and raised.attr == "EvaluationError":
            return True

    return False


def _is_protocol(node: ast.ClassDef) -> bool:
    return any(isinstance(base, ast.Name) and base.id == "Protocol" for base in node.bases)


def _body_is_ellipsis(member: ast.FunctionDef) -> bool:
    body = member.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if len(body) != 1 or not isinstance(body[0], ast.Expr):
        return False

    return isinstance(body[0].value, ast.Constant) and body[0].value.value is Ellipsis
