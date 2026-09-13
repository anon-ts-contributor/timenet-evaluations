"""The four timed tasks, the shared plan, and the failure boundary.

Every reader here is a double that records what was asked of it and reads nothing. A double has no
page cache and no artifact, so nothing in this file is evidence that a figure is cold or that a read
is fast. What these tests pin is the shape of the call graph: which reader call each task times,
that one repetition of the shuffled task is one whole pass over the plan, that both readers of one
representation walk the very same blocks, and that a failure anywhere stops the run rather than
leaving a hole in the grid.

The artifacts on disk are byte files of a chosen size. They exist because ``measure_size`` refuses a
path that is not there and because the size decides how large a block is. Nothing reads them.
"""

import ast
from collections.abc import Iterable, Iterator, Sequence
import inspect
from pathlib import Path

import pytest
from torch.utils.data import Dataset

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid import (
    CELLS_PER_DATASET,
    REPRESENTATIONS,
    STORAGE_FIGURES_PER_DATASET,
    TASKS,
    Artifact,
    Cell,
    ItemDeclaration,
    Pairing,
    ParsedItems,
    Representation,
    StorageKey,
    Task,
    assembly as assembly_module,
    expected_cells,
    expected_storage_keys,
)
from timenet_evaluations.grid.reader import Block as BlockShape
from timenet_evaluations.grid.readers import PANDAS, PYTORCH
from timenet_evaluations.harness import tasks as tasks_module
from timenet_evaluations.harness.drop import DARWIN, DropCaches, DropRecord
from timenet_evaluations.harness.plan import BlockPlan, record_plan
from timenet_evaluations.harness.repeat import REPEATS
from timenet_evaluations.harness.tasks import (
    DatasetGrid,
    RunParameters,
    Timing,
    measure_dataset,
    measure_pairing,
    operation_for,
)


DATASET = "a-dataset"

ITEMS = 40
"""How many canonical items the stub dataset declares. It is also the block count of the plan the
single-pairing tests walk, because those plan one item per block."""

ITEM_SHAPE = (2, 4)

SEED = 11

SIZES = {"original": 800, "timef": 400}
"""The bytes each representation holds. They differ, so the two plans of one dataset differ."""

BLOCK_BYTES = 40
"""The byte budget the dataset tests plan under: two items per block for ``original`` and four for
``timef``, which is the per-representation variation the shuffled task turns on."""

ONE_ITEM_BUDGET = 10
"""The byte budget that buys exactly one item of a 400-byte, 40-item representation, so the plan
holds forty blocks."""

RECORD = DropRecord(command="/usr/sbin/purge", exit_status=0, platform=DARWIN)

TASKS_MODULE = Path(inspect.getfile(tasks_module))

OPERATIONS = ("first_item", "full_read", "sequential_walk", "block_walk")
"""The four functions a timed interval covers. Nothing else runs inside a clock."""

READ_CALLS = {"open", "read_first", "read_all", "read_sequential", "read_block"}
"""Every call a timed operation is allowed to make. A conversion, a size or a plan is not one."""


class StubParsingPath:
    """A loader nothing opens. The readers here are doubles, so no parsing path is ever used."""

    def open_pandas(self, path: Path) -> ParsedItems:
        raise AssertionError("a timed task opens through its reader, never through the harness")

    def open_torch(self, path: Path) -> Dataset[object]:
        raise AssertionError("a timed task opens through its reader, never through the harness")


class StubRepresentation:
    """A representation that declares the five members and holds a real path of a chosen size."""

    def __init__(self, name: str, path: Path, *, items: int = ITEMS) -> None:
        self.dataset = DATASET
        self.name = name
        self.artifact = Artifact(representation=name, path=path)
        self.item = ItemDeclaration(count=items, shape=ITEM_SHAPE)
        self.parsing_path = StubParsingPath()


class SpyOpened:
    """One representation opened by a double. It records the call and hands back a placeholder."""

    def __init__(self, reader: "SpyReader") -> None:
        self._reader = reader

    def _record(self, call: str) -> None:
        self._reader.log.append(call)
        if call in self._reader.failing:
            raise EvaluationError(f"the read failed: call {call!r}, reader {self._reader.name!r}")

    def read_first(self) -> object:
        self._record("read_first")
        return 0

    def read_all(self) -> object:
        self._record("read_all")
        return list(range(self._reader.items))

    def read_sequential(self) -> Iterator[object]:
        self._record("read_sequential")
        for position in range(self._reader.items):
            self._reader.log.append("item")
            yield position

    def read_block(self, block: BlockShape) -> object:
        self._record("read_block")
        self._reader.blocks.append(block)
        return block


class SpyReader:
    """A reader that records every call and reads nothing.

    ``blocks`` holds the block objects the shuffled walk handed it, in the order it walked them.
    Two spies of one run therefore make "both readers walked the same plan" an identity check and
    not an equality one.
    """

    def __init__(self, name: str, *, items: int = ITEMS, failing: tuple[str, ...] = ()) -> None:
        self.name = name
        self.items = items
        self.failing = failing
        self.log: list[str] = []
        self.blocks: list[BlockShape] = []

    def open(self, representation: Representation) -> SpyOpened:
        self.log.append("open")
        return SpyOpened(self)


type Spies = tuple[SpyReader, SpyReader]
"""The two reader doubles of one run, in the order the registry declares the readers."""


class SpyDrop:
    """A drop that records what it was asked to precede. It starts no process."""

    def __init__(self) -> None:
        self.calls: list[tuple[Cell | StorageKey, Path]] = []

    def __call__(self, *, at: Cell | StorageKey, path: Path) -> DropRecord:
        self.calls.append((at, path))
        return RECORD


class RefusingDrop:
    """A drop that reports failure, as a missing executable or a non-zero exit does."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, at: Cell | StorageKey, path: Path) -> DropRecord:
        self.calls += 1
        raise EvaluationError(f"the page cache drop failed: path {path}")


def artifacts(root: Path, sizes: dict[str, int] | None = None) -> tuple[Representation, ...]:
    stated = sizes if sizes is not None else SIZES
    built = []
    for name in REPRESENTATIONS:
        path = root / name
        path.write_bytes(b"x" * stated[name])
        built.append(StubRepresentation(name, path))

    return tuple(built)


def one_pairing(root: Path, reader: SpyReader, *, size_bytes: int = 400) -> Pairing:
    path = root / "timef"
    path.write_bytes(b"x" * size_bytes)

    return Pairing(dataset=DATASET, representation=StubRepresentation("timef", path), reader=reader)


def cell_of(task: Task, reader: str = PANDAS, representation: str = "timef") -> Cell:
    return Cell(dataset=DATASET, representation=representation, reader=reader, task=task)


def parameters(drop: DropCaches, *, repeats: int = REPEATS, block_bytes: int = BLOCK_BYTES) -> RunParameters:
    return RunParameters(timing=Timing(drop_caches=drop, repeats=repeats), seed=SEED, block_bytes=block_bytes)


@pytest.fixture
def readers(monkeypatch: pytest.MonkeyPatch) -> Spies:
    spies = (SpyReader(PANDAS), SpyReader(PYTORCH))
    monkeypatch.setattr(assembly_module, "READERS", spies)

    return spies


@pytest.fixture
def grid(tmp_path: Path, readers: Spies) -> DatasetGrid:
    return measure_dataset(DATASET, artifacts(tmp_path), parameters(SpyDrop()))


# Four tasks per cell, and nothing else is timed


def test_one_dataset_produces_sixteen_timed_cells_and_two_storage_figures(grid: DatasetGrid) -> None:
    assert len(grid.measurements) == CELLS_PER_DATASET
    assert len(grid.prepared) == STORAGE_FIGURES_PER_DATASET
    assert tuple(measurement.at for measurement in grid.measurements) == expected_cells(DATASET)


def test_every_measurement_names_all_four_coordinates(grid: DatasetGrid) -> None:
    for measurement in grid.measurements:
        assert measurement.at.dataset == DATASET
        assert measurement.at.representation in REPRESENTATIONS
        assert measurement.at.reader in {PANDAS, PYTORCH}
        assert measurement.at.task in TASKS


def test_every_cell_is_measured_on_all_four_tasks(grid: DatasetGrid) -> None:
    per_pair: dict[tuple[str, str], list[Task]] = {}
    for measurement in grid.measurements:
        per_pair.setdefault((measurement.at.representation, measurement.at.reader), []).append(measurement.at.task)

    assert len(per_pair) == len(REPRESENTATIONS) * 2
    assert all(tuple(tasks) == TASKS for tasks in per_pair.values())


def test_a_storage_figure_is_indexed_by_representation_and_never_by_a_reader(grid: DatasetGrid) -> None:
    assert tuple(one.at for one in grid.prepared) == expected_storage_keys(DATASET)
    assert [one.size_bytes for one in grid.prepared] == [SIZES[name] for name in REPRESENTATIONS]
    assert not any(hasattr(one.at, "reader") for one in grid.prepared)


def test_each_task_times_its_own_read_and_nothing_else(tmp_path: Path) -> None:
    reader = SpyReader(PANDAS)
    pairing = one_pairing(tmp_path, reader)
    blocks = record_plan(400, ITEMS, ONE_ITEM_BUDGET, SEED).blocks

    logs: dict[Task, list[str]] = {}
    for task in TASKS:
        reader.log.clear()
        operation_for(pairing, cell_of(task), blocks)()
        logs[task] = list(reader.log)

    assert logs[Task.FIRST_ITEM] == ["open", "read_first"]
    assert logs[Task.FULL_READ] == ["open", "read_all"]
    assert logs[Task.SEQUENTIAL] == ["open", "read_sequential", *["item"] * ITEMS]
    assert logs[Task.BLOCK_SHUFFLED] == ["open", *["read_block"] * len(blocks)]


def test_the_sequential_task_hands_over_one_item_at_a_time_and_is_not_derived_from_the_full_read(
    tmp_path: Path,
) -> None:
    # A rate computed as the item count divided by the full-read figure would have measured three
    # tasks and printed four. This walk asks for the items and consumes them one at a time.
    reader = SpyReader(PANDAS)
    pairing = one_pairing(tmp_path, reader)

    operation_for(pairing, cell_of(Task.SEQUENTIAL), ())()

    assert "read_all" not in reader.log
    assert reader.log.count("read_sequential") == 1
    assert reader.log.count("item") == ITEMS


def test_every_declared_task_has_a_timed_operation(tmp_path: Path) -> None:
    pairing = one_pairing(tmp_path, SpyReader(PANDAS))

    assert all(callable(operation_for(pairing, cell_of(task), ())) for task in TASKS)


def test_a_timed_operation_calls_the_reader_and_nothing_else() -> None:
    # A conversion, a size, a plan or a reduction inside one of these bodies would put it inside a
    # clock. The check reads the source, so it catches the call that was added rather than the one
    # a test happened to exercise.
    tree = ast.parse(TASKS_MODULE.read_text(encoding="utf-8"))
    called: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in OPERATIONS:
            continue
        called[node.name] = {
            child.func.attr if isinstance(child.func, ast.Attribute) else getattr(child.func, "id", "?")
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
        }

    assert set(called) == set(OPERATIONS)
    for name, names in called.items():
        assert names <= READ_CALLS, f"{name} calls {sorted(names - READ_CALLS)} inside a timed interval"


def test_every_figure_comes_from_the_one_shared_primitive() -> None:
    source = TASKS_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "repeat"
    ]

    assert len(calls) == 1
    assert "perf_counter" not in source
    assert "median" not in source


def test_the_drop_precedes_every_timed_repetition_of_every_task(tmp_path: Path, readers: Spies) -> None:
    drop = SpyDrop()

    measure_dataset(DATASET, artifacts(tmp_path), parameters(drop, repeats=5))

    assert len(drop.calls) == CELLS_PER_DATASET * 5


# One repetition of the shuffled task is one whole pass over the plan


def test_one_repetition_of_the_shuffled_task_is_one_whole_pass_over_the_plan(tmp_path: Path) -> None:
    reader = SpyReader(PANDAS)
    pairing = one_pairing(tmp_path, reader)
    plan = record_plan(400, ITEMS, ONE_ITEM_BUDGET, SEED)
    repeats = 5

    assert plan.block_count == 40

    measured = measure_pairing(pairing, blocks=plan.blocks, timing=Timing(drop_caches=SpyDrop(), repeats=repeats))
    shuffled = measured[TASKS.index(Task.BLOCK_SHUFFLED)]

    assert shuffled.repetitions == repeats
    assert len(shuffled.samples) == repeats
    assert len(shuffled.samples) != plan.block_count
    assert len(shuffled.samples) != plan.block_count * repeats
    assert len(reader.blocks) == plan.block_count * repeats


def test_each_repetition_visits_every_block_in_plan_order_and_never_sorts_or_coalesces(tmp_path: Path) -> None:
    # A walk that re-sorted the blocks would measure a full scan and report it as a shuffled rate.
    reader = SpyReader(PANDAS)
    pairing = one_pairing(tmp_path, reader)
    plan = record_plan(400, ITEMS, ONE_ITEM_BUDGET, SEED)
    repeats = 3

    assert [block.start for block in plan.blocks] != sorted(block.start for block in plan.blocks)

    measure_pairing(pairing, blocks=plan.blocks, timing=Timing(drop_caches=SpyDrop(), repeats=repeats))

    passes = [
        reader.blocks[start : start + plan.block_count] for start in range(0, len(reader.blocks), plan.block_count)
    ]

    assert len(passes) == repeats
    assert all(tuple(walked) == plan.blocks for walked in passes)


# Both readers of one representation walk the same plan


def test_both_readers_of_one_representation_walk_the_very_same_blocks(tmp_path: Path, readers: Spies) -> None:
    # Identity and not equality. Two plans built from identical arguments would pass an equality
    # check and would still be two things that can drift apart later.
    pandas_spy, torch_spy = readers
    repeats = 5

    grid = measure_dataset(DATASET, artifacts(tmp_path), parameters(SpyDrop(), repeats=repeats))

    walked = [block for one in grid.prepared for _ in range(repeats) for block in one.plan.blocks]

    assert len(pandas_spy.blocks) == len(walked)
    assert all(seen is planned for seen, planned in zip(pandas_spy.blocks, walked, strict=True))
    assert all(seen is planned for seen, planned in zip(torch_spy.blocks, walked, strict=True))


@pytest.mark.parametrize("repeats", [1, 5])
def test_the_plan_is_built_once_per_representation_and_never_per_cell_or_per_repetition(
    tmp_path: Path,
    readers: Spies,
    monkeypatch: pytest.MonkeyPatch,
    repeats: int,
) -> None:
    planned: list[tuple[int, int, int, int]] = []
    building = tasks_module.record_plan

    def counting(size_bytes: int, n_items: int, block_bytes: int, seed: int) -> BlockPlan:
        planned.append((size_bytes, n_items, block_bytes, seed))
        return building(size_bytes, n_items, block_bytes, seed)

    monkeypatch.setattr(tasks_module, "record_plan", counting)

    measure_dataset(DATASET, artifacts(tmp_path), parameters(SpyDrop(), repeats=repeats))

    assert len(planned) == len(REPRESENTATIONS)
    assert len(planned) != CELLS_PER_DATASET
    assert planned == [(SIZES[name], ITEMS, BLOCK_BYTES, SEED) for name in REPRESENTATIONS]


def test_the_seed_is_a_run_parameter_shared_by_every_pair(tmp_path: Path, readers: Spies) -> None:
    grid = measure_dataset(DATASET, artifacts(tmp_path), parameters(SpyDrop()))

    assert {one.plan.seed for one in grid.prepared} == {SEED}
    assert {one.plan.block_bytes for one in grid.prepared} == {BLOCK_BYTES}
    assert {one.plan.n_items for one in grid.prepared} == {ITEMS}


def test_the_two_representations_of_one_dataset_get_two_plans(grid: DatasetGrid) -> None:
    # The plan belongs to a (dataset, representation) pair. The byte budget is shared and the size
    # is not, so the denser representation buys more items per block.
    original, timef = grid.prepared

    assert (original.plan.items_per_block, original.plan.block_count) == (2, 20)
    assert (timef.plan.items_per_block, timef.plan.block_count) == (4, 10)


# A failure at the measurement boundary stops the run


def test_the_module_catches_nothing() -> None:
    tree = ast.parse(TASKS_MODULE.read_text(encoding="utf-8"))

    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Try)]


@pytest.mark.parametrize("failing", ["read_first", "read_all", "read_sequential", "read_block"])
def test_a_read_that_fails_stops_the_run_and_produces_no_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    spies = (SpyReader(PANDAS, failing=(failing,)), SpyReader(PYTORCH))
    monkeypatch.setattr(assembly_module, "READERS", spies)

    with pytest.raises(EvaluationError) as refusal:
        measure_dataset(DATASET, artifacts(tmp_path), parameters(SpyDrop()))

    assert failing in str(refusal.value)
    assert spies[1].log == []


def test_one_cell_failing_after_the_other_three_tasks_succeeded_still_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The shuffled task is the last of the four, so the first three tasks of that pairing were
    # measured before this failure. A grid with a hole in it cannot be read, so nothing is kept.
    spies = (SpyReader(PANDAS, failing=("read_block",)), SpyReader(PYTORCH))
    monkeypatch.setattr(assembly_module, "READERS", spies)

    with pytest.raises(EvaluationError):
        measure_dataset(DATASET, artifacts(tmp_path), parameters(SpyDrop()))

    assert spies[0].log.count("read_first") >= 1
    assert spies[1].log == []


def test_a_failing_drop_stops_the_run_before_anything_is_read(tmp_path: Path, readers: Spies) -> None:
    drop = RefusingDrop()

    with pytest.raises(EvaluationError):
        measure_dataset(DATASET, artifacts(tmp_path), parameters(drop))

    assert drop.calls == 1
    assert [spy.log for spy in readers] == [[], []]


def test_an_absent_artifact_is_refused_before_any_cell_is_measured(tmp_path: Path, readers: Spies) -> None:
    missing = tmp_path / "gone"
    present = tmp_path / "original"
    present.write_bytes(b"x" * SIZES["original"])
    representations = (StubRepresentation("original", present), StubRepresentation("timef", missing))
    drop = SpyDrop()

    with pytest.raises(EvaluationError) as refusal:
        measure_dataset(DATASET, representations, parameters(drop))

    assert str(missing) in str(refusal.value)
    assert (drop.calls, [spy.log for spy in readers]) == ([], [[], []])


def test_two_canonical_item_counts_in_one_dataset_are_refused(tmp_path: Path, readers: Spies) -> None:
    # n_items is the plan's second argument, so two counts give one row two plans over different
    # numbers of items and every rate still renders.
    built = artifacts(tmp_path)
    representations = (built[0], StubRepresentation("timef", built[1].artifact.path, items=ITEMS + 1))
    drop = SpyDrop()

    with pytest.raises(EvaluationError) as refusal:
        measure_dataset(DATASET, representations, parameters(drop))

    message = str(refusal.value)

    assert f"dataset {DATASET!r}" in message
    assert str(ITEMS) in message and str(ITEMS + 1) in message
    assert drop.calls == []


def test_the_completeness_check_sees_every_cell_and_both_storage_figures(
    tmp_path: Path, readers: Spies, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, list[object]] = {}

    def spy(datasets: Sequence[str], cells: Iterable[Cell], storage_keys: Iterable[StorageKey]) -> None:
        seen["datasets"] = list(datasets)
        seen["cells"] = list(cells)
        seen["storage"] = list(storage_keys)

    monkeypatch.setattr(tasks_module, "check_grid_complete", spy)

    measure_dataset(DATASET, artifacts(tmp_path), parameters(SpyDrop()))

    assert seen["datasets"] == [DATASET]
    assert set(seen["cells"]) == set(expected_cells(DATASET))
    assert set(seen["storage"]) == set(expected_storage_keys(DATASET))
