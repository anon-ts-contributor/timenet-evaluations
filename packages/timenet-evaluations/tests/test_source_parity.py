"""The parity check: the declared count on both sides, two sampled positions, and what it misses."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import timenet_evaluations
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid import Artifact, ItemDeclaration, Opened, Reader, Representation
from timenet_evaluations.grid.readers import PANDAS, PandasReader, PyTorchReader
from timenet_evaluations.source import parity as parity_module
from timenet_evaluations.source.parity import (
    MINIMUM_SAMPLED_COUNT,
    ParityReference,
    _unwrapped,
    check_parity,
    sampled_positions,
)


N_ITEMS = 12
N_CHANNELS = 2
N_SAMPLES = 5
ITEM_SHAPE = (N_CHANNELS, N_SAMPLES)
MIDPOINT = N_ITEMS // 2

DATASET = "a-dataset"
REPRESENTATIONS = ("original", "timef")

READERS: tuple[Reader, ...] = (PandasReader(), PyTorchReader())
DECLARATION = ItemDeclaration(count=N_ITEMS, shape=ITEM_SHAPE)

PARITY_MODULE = Path(parity_module.__file__)
PACKAGE_ROOT = Path(timenet_evaluations.__file__).parent

# Anything that could start a clock. The check runs before the first timed cell of a dataset and
# produces no figure, so a timing call in this module is a figure nothing in the report can hold.
TIMING_NAMES = ("time", "timeit", "datetime", "perf_counter", "perf_counter_ns", "monotonic", "process_time")


class FakeItems:
    # Positional access to one representation's items, as the Pandas parsing path hands it out.
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int | slice) -> object:
        return np.asarray(self._values[index], dtype=np.float32)


class FakeTorchItems(Dataset[object]):
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self._values[index], dtype=np.float32))


class FakeParsingPath:
    # One loader, opened by both readers of one representation.
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def open_pandas(self, path: Path) -> FakeItems:
        return FakeItems(self._values)

    def open_torch(self, path: Path) -> Dataset[object]:
        return FakeTorchItems(self._values)


class FakeRepresentation:
    # The declaration is the dataset's, not this representation's: a representation that carried a
    # declaration of its own would be the thing under test choosing what it is measured against.
    def __init__(self, name: str, values: np.ndarray, declaration: ItemDeclaration = DECLARATION) -> None:
        self.dataset = DATASET
        self.name = name
        self.artifact = Artifact(representation=name, path=Path("/dev/null"))
        self.item = declaration
        self.parsing_path = FakeParsingPath(values)


@pytest.fixture
def values() -> np.ndarray:
    generator = np.random.default_rng(0)
    return generator.standard_normal((N_ITEMS, *ITEM_SHAPE)).astype(np.float32)


@pytest.fixture
def reference(values: np.ndarray) -> ParityReference:
    # signal_stack(frame) is what a run passes here. The array is the frame's, so both
    # representations are measured against one object whose provenance this package knows.
    return ParityReference(dataset=DATASET, declaration=DECLARATION, reference=values)


def representations_of(*arrays: np.ndarray) -> list[FakeRepresentation]:
    return [FakeRepresentation(name, array) for name, array in zip(REPRESENTATIONS, arrays, strict=True)]


# Both representations agree


def test_both_representations_agreeing_lets_the_run_proceed(reference: ParityReference, values: np.ndarray) -> None:
    check_parity(reference, representations_of(values, values.copy()), READERS)


def test_every_pair_of_a_representation_and_a_reader_is_checked(reference: ParityReference, values: np.ndarray) -> None:
    # ADR-0024 asks for item identity across all four pairs, so a check that opened one reader would
    # leave half the grid unverified while reporting that parity held.
    opened: list[tuple[str, str]] = []

    class RecordingReader:
        def __init__(self, inner: Reader) -> None:
            self.name = inner.name
            self._inner = inner

        def open(self, representation: Representation) -> Opened:
            opened.append((representation.name, self.name))
            return self._inner.open(representation)

    readers = (RecordingReader(PandasReader()), RecordingReader(PyTorchReader()))
    check_parity(reference, representations_of(values, values.copy()), readers)

    assert sorted(opened) == sorted((name, reader.name) for name in REPRESENTATIONS for reader in readers)


# A representation yields a coarser unit


def test_a_coarser_unit_raises_and_names_the_declared_count_and_the_yielded_count(
    reference: ParityReference, values: np.ndarray
) -> None:
    # One item per whole recording where the declared item is smaller: the count is wrong, and the
    # count is the one part of this check that covers every item.
    coarse = values[: N_ITEMS // 3]

    with pytest.raises(EvaluationError) as failure:
        check_parity(reference, representations_of(values, coarse), READERS)

    message = str(failure.value)
    assert f"declared count {N_ITEMS}" in message
    assert f"yielded count {N_ITEMS // 3}" in message


def test_a_coarser_unit_is_refused_before_any_value_is_read(reference: ParityReference, values: np.ndarray) -> None:
    # The count check runs first, so a representation cut into the wrong pieces stops the run
    # without a value comparison that would be meaningless anyway.
    reads: list[str] = []

    class CountingParsingPath(FakeParsingPath):
        def open_pandas(self, path: Path) -> FakeItems:
            reads.append("open_pandas")
            return super().open_pandas(path)

        def open_torch(self, path: Path) -> Dataset[object]:
            reads.append("open_torch")
            return super().open_torch(path)

    coarse = FakeRepresentation(REPRESENTATIONS[1], values[: N_ITEMS // 3])
    coarse.parsing_path = CountingParsingPath(values[: N_ITEMS // 3])

    with pytest.raises(EvaluationError):
        check_parity(reference, [coarse], READERS)

    assert reads == ["open_pandas"]


# The right count of the wrong items


def test_the_right_count_of_wrong_items_fails_at_the_sampled_interior_position(
    reference: ParityReference, values: np.ndarray
) -> None:
    # The count is right, position zero is right, and one interior item is not. Only a value check
    # catches this, which is why the check compares values and not counts alone.
    wrong = values.copy()
    wrong[MIDPOINT] += 1.0

    with pytest.raises(EvaluationError) as failure:
        check_parity(reference, representations_of(values, wrong), READERS)

    message = str(failure.value)
    assert f"sampled position {MIDPOINT}" in message
    assert "does not hold at a sampled position" in message


def test_wrong_values_at_position_zero_are_caught(reference: ParityReference, values: np.ndarray) -> None:
    wrong = values.copy()
    wrong[0] += 1.0

    with pytest.raises(EvaluationError) as failure:
        check_parity(reference, representations_of(values, wrong), READERS)

    assert "sampled position 0" in str(failure.value)


def test_a_difference_in_the_last_bit_is_a_difference(reference: ParityReference, values: np.ndarray) -> None:
    # The comparison has zero tolerance, so a representation that lost precision is refused rather
    # than accepted as close enough.
    wrong = values.copy()
    wrong[MIDPOINT, 0, 0] = np.nextafter(wrong[MIDPOINT, 0, 0], np.float32(np.inf))

    with pytest.raises(EvaluationError):
        check_parity(reference, representations_of(values, wrong), READERS)


def test_an_item_moved_between_unsampled_positions_is_not_caught(
    reference: ParityReference, values: np.ndarray
) -> None:
    # The honest weakness, written as a test rather than only as prose. Two items are swapped, the
    # count is right, and neither sampled position is one of them, so the check passes over a
    # representation that does not hold the dataset's items in the dataset's order.
    reordered = values.copy()
    reordered[[2, 3]] = reordered[[3, 2]]

    check_parity(reference, representations_of(values, reordered), READERS)

    assert not np.array_equal(reordered, values)


# The reference is the frame


def test_two_representations_that_agree_with_each_other_and_not_the_frame_are_refused(
    reference: ParityReference, values: np.ndarray
) -> None:
    # Neither representation is checked against the other. Two readers wrong in the same way would
    # otherwise agree and pass, and the frame is the one array whose provenance this package knows.
    wrong = values + 1.0

    with pytest.raises(EvaluationError):
        check_parity(reference, representations_of(wrong, wrong.copy()), READERS)


def test_a_reference_that_is_not_the_declared_dataset_is_refused(values: np.ndarray) -> None:
    reference = ParityReference(dataset=DATASET, declaration=DECLARATION, reference=values[:-1])

    with pytest.raises(EvaluationError) as failure:
        check_parity(reference, representations_of(values, values.copy()), READERS)

    message = str(failure.value)
    assert DATASET in message
    assert f"declared shape {(N_ITEMS, *ITEM_SHAPE)}" in message
    assert "reference shape" in message


def test_the_reference_failure_names_no_representation(values: np.ndarray) -> None:
    # No representation is at fault: the caller stacked a frame that is not this dataset's.
    reference = ParityReference(dataset=DATASET, declaration=DECLARATION, reference=values[:-1])

    with pytest.raises(EvaluationError) as failure:
        check_parity(reference, representations_of(values, values.copy()), READERS)

    for name in REPRESENTATIONS:
        assert name not in str(failure.value)


# The scope a failure carries


def test_a_failure_names_the_three_coordinates_it_has_and_no_task(
    reference: ParityReference, values: np.ndarray
) -> None:
    # The check runs before any task starts, so a message that named a task would name one that had
    # not begun.
    wrong = values.copy()
    wrong[0] += 1.0

    with pytest.raises(EvaluationError) as failure:
        check_parity(reference, representations_of(values, wrong), READERS)

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert f"representation {REPRESENTATIONS[1]!r}" in message
    assert "reader " in message
    assert "task " not in message


def test_the_count_failure_names_the_reader_whose_parsing_path_reported_the_count(
    reference: ParityReference, values: np.ndarray
) -> None:
    with pytest.raises(EvaluationError) as failure:
        check_parity(reference, representations_of(values, values[:-1]), READERS)

    assert f"reader {PANDAS!r}" in str(failure.value)


def test_both_readers_are_named_by_the_failures_they_raise(reference: ParityReference, values: np.ndarray) -> None:
    wrong = values.copy()
    wrong[0] += 1.0
    representations = representations_of(values, wrong)

    for reader in READERS:
        with pytest.raises(EvaluationError) as failure:
            check_parity(reference, representations, [reader])

        assert f"reader {reader.name!r}" in str(failure.value)


# The positions the check samples


def test_the_sampled_positions_are_position_zero_and_the_midpoint() -> None:
    assert sampled_positions(N_ITEMS) == (0, MIDPOINT)


def test_a_dataset_of_one_item_samples_the_one_position_it_has() -> None:
    assert sampled_positions(1) == (0,)
    assert sampled_positions(MINIMUM_SAMPLED_COUNT) == (0, 1)


def test_the_sampled_positions_are_the_same_on_every_call() -> None:
    # A position drawn at random makes a failed run impossible to repeat, and lets two runs of one
    # dataset check different things while both report that parity held.
    assert sampled_positions(N_ITEMS) == sampled_positions(N_ITEMS)


def test_the_check_reads_only_the_sampled_positions(reference: ParityReference, values: np.ndarray) -> None:
    touched: list[int] = []

    class TouchedItems(FakeItems):
        def __getitem__(self, index: int | slice) -> object:
            if isinstance(index, slice):
                touched.extend(range(*index.indices(N_ITEMS)))
            else:
                touched.append(index)

            return super().__getitem__(index)

    class TouchedParsingPath(FakeParsingPath):
        def open_pandas(self, path: Path) -> FakeItems:
            return TouchedItems(values)

    representation = FakeRepresentation(REPRESENTATIONS[0], values)
    representation.parsing_path = TouchedParsingPath(values)

    check_parity(reference, [representation], [PandasReader()])

    assert sorted(set(touched)) == [0, MIDPOINT]


# One item out of a one-item read


def test_a_container_of_one_item_gives_up_its_item() -> None:
    # One target form returns a list of one item and the other returns the item, so the values
    # arrive with or without a first axis of length one.
    item = np.zeros(ITEM_SHAPE, dtype=np.float32)

    assert _unwrapped([item]) is item


def test_values_that_are_not_a_container_of_one_item_are_passed_on_unchanged() -> None:
    # No column is dropped and no pair is unpacked here. An item that carries its label is refused
    # by the comparison form, which is where that refusal belongs.
    pair = (np.zeros(ITEM_SHAPE, dtype=np.float32), "a-label")

    assert _unwrapped(pair) is pair


def test_an_item_of_the_declared_shape_is_passed_on_unchanged() -> None:
    item = np.zeros(ITEM_SHAPE, dtype=np.float32)

    assert _unwrapped(item) is item


# The check is untimed, and it is not a representation's to make


def test_nothing_in_the_parity_module_starts_a_timer() -> None:
    tree = ast.parse(PARITY_MODULE.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in TIMING_NAMES, f"the parity check imports {alias.name}"
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".")[0] not in TIMING_NAMES, f"the parity check imports {node.module}"
        if isinstance(node, ast.Attribute):
            assert node.attr not in TIMING_NAMES, f"the parity check calls {node.attr}"


def test_the_parity_module_imports_no_measurement_module() -> None:
    # The harness owns every timer. A parity check that reached it would be one import away from
    # becoming a figure.
    text = PARITY_MODULE.read_text(encoding="utf-8")

    assert "timenet_evaluations.harness" not in text


def test_the_parity_module_imports_neither_the_subject_under_test_nor_a_dataset_library() -> None:
    # Both are absent from the environment CI builds, and a module that imported one would make
    # every module importing it fail to collect rather than skip.
    tree = ast.parse(PARITY_MODULE.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            roots.add(node.module.split(".")[0])

    assert "timenet" not in roots
    assert "pyhealth" not in roots


def test_no_representation_and_no_reader_module_holds_the_parity_check() -> None:
    # A representation verifying its own parity is the same defect as a representation declaring its
    # own item count: the party being checked would be doing the checking.
    elsewhere = sorted((PACKAGE_ROOT / "grid").rglob("*.py"))

    assert elsewhere
    for module in elsewhere:
        assert "check_parity" not in module.read_text(encoding="utf-8"), f"{module.name} holds the parity check"


def test_the_module_docstring_states_what_the_check_does_not_establish() -> None:
    # The strength of the guarantee is part of the deliverable. A check that overstated what it
    # establishes would be worse than one that admits its limit, because a mismatch renders.
    docstring = ast.get_docstring(ast.parse(PARITY_MODULE.read_text(encoding="utf-8")))

    assert docstring is not None
    assert "not proof" in docstring
    assert "midpoint" in docstring
