"""The two timed opens of the release's own files, and the narrow reads they make available."""

import ast
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from pyhealth.datasets import SampleDataset
import pytest
import torch
from torch.utils.data import IterableDataset

from timenet_evaluations import source as source_package
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid import Artifact, ItemDeclaration
from timenet_evaluations.grid.cell import OpenKey
from timenet_evaluations.grid.item import to_comparison_form
from timenet_evaluations.grid.parsing import ParsedItems, ParsingPath
from timenet_evaluations.grid.readers import PANDAS, PYTORCH, PandasReader, PyTorchReader, build_loader
from timenet_evaluations.source import (
    LABEL,
    PATIENT,
    SIGNAL,
    sleep_edfx_access as access,
    sleep_edfx_preparation as preparation,
)
from timenet_evaluations.source.sleep_edfx import NAME, RECORDINGS_DIRECTORY, SUBJECT_SPREADSHEET, SleepEdfxConnector
from timenet_evaluations.source.sleep_edfx_access import SleepEdfxItems, SleepEdfxSignals
from timenet_evaluations.source.sleep_edfx_preparation import ORIGINAL, task_cache_index
from timenet_evaluations.source.sleep_edfx_task import SleepStaging


SOURCE_PACKAGE = Path(source_package.__file__).parent
ACCESS_MODULE = SOURCE_PACKAGE / "sleep_edfx_access.py"
SLEEP_EDFX_MODULE = SOURCE_PACKAGE / "sleep_edfx.py"

N_ITEMS = 12
N_CHANNELS = 2
N_SAMPLES = 5
ITEM_SHAPE = (N_CHANNELS, N_SAMPLES)


class FakeBlock(NamedTuple):
    start: int
    count: int


# A block from the middle and a short final block, which is the case a plan always ends with.
INTERIOR = FakeBlock(start=8, count=4)
FINAL = FakeBlock(start=10, count=2)


def _code_of(module: Path, name: str) -> str:
    """Unparse one function's body, without its docstring, so prose is not read as behaviour."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            statements = node.body[1:] if ast.get_docstring(node) is not None else node.body
            return "\n".join(ast.unparse(statement) for statement in statements)

    raise AssertionError(f"{module.name} declares no {name}")


def _code_of_module(module: Path) -> str:
    """Unparse a whole module, without its docstrings, so prose is not read as behaviour."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef) and ast.get_docstring(node) is not None:
            node.body = node.body[1:]

    return ast.unparse(tree)


class FakeSamples:
    """The sample object the reference loader serves, as much of it as the adapters use.

    Every position a read touched is recorded, and an integer read is recorded apart from a slice
    read, so a test can tell a narrow read from a full one and one pass from another.
    """

    def __init__(self, values: np.ndarray, *, carries_signal: bool = True) -> None:
        self._values = values
        self._carries_signal = carries_signal
        self.touched: list[int] = []
        self.indices: list[int] = []
        self.slices: list[tuple[int, int]] = []
        self.shuffle: list[bool] = []
        self.lengths = 0

    def set_shuffle(self, shuffle: bool) -> None:
        self.shuffle.append(shuffle)

    def __len__(self) -> int:
        self.lengths += 1
        return len(self._values)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            positions = range(*index.indices(len(self._values)))
            self.slices.append((index.start, index.stop))
            self.touched.extend(positions)
            return [self._sample(position) for position in positions]

        self.indices.append(index)
        self.touched.append(index)

        return self._sample(index)

    def _sample(self, position: int) -> dict[str, Any]:
        # The keys the schema passes through arrive on every item beside the three of the contract,
        # and the label is an integer class index rather than a stage name.
        sample = {
            LABEL: position % 6,
            PATIENT: f"{position:04d}",
            "night": 1,
            "patient_age": 58,
            "patient_sex": "M",
        }
        if self._carries_signal:
            sample[SIGNAL] = torch.from_numpy(self._values[position])

        return sample


class FakeLoader:
    """The constructed reference loader, as much of it as the gate and the open use."""

    def __init__(self, cache_dir: Path, samples: FakeSamples) -> None:
        self.cache_dir = cache_dir
        self.tasks = 0
        self._samples = samples

    def set_task(self, task: SleepStaging) -> FakeSamples:
        self.tasks += 1
        return self._samples


class FakeRepresentation:
    """The release's own files as one representation, parsed through the connector itself."""

    def __init__(self, path: Path) -> None:
        self.dataset = NAME
        self.name = ORIGINAL
        self.artifact = Artifact(representation=ORIGINAL, path=path)
        self.item = ItemDeclaration(count=N_ITEMS, shape=ITEM_SHAPE)
        self.parsing_path = SleepEdfxConnector()


@pytest.fixture
def values() -> np.ndarray:
    generator = np.random.default_rng(0)

    return generator.standard_normal((N_ITEMS, *ITEM_SHAPE)).astype(np.float32)


@pytest.fixture
def release_root(tmp_path: Path) -> Path:
    """A directory shaped like the root of the release: the spreadsheet and the recordings beside it."""
    root = tmp_path / "a-release-root"
    (root / RECORDINGS_DIRECTORY).mkdir(parents=True)
    (root / SUBJECT_SPREADSHEET).write_bytes(b"")

    return root


@pytest.fixture
def samples(monkeypatch: pytest.MonkeyPatch, values: np.ndarray) -> FakeSamples:
    """One sample object, served to every open, so a test can read what the opens asked it for.

    The route into the library is replaced and nothing else is: the root precondition, both opens
    and both adapters are the shipped ones. No corpus exists to open, and building one would test
    the reference loader rather than this connector.
    """
    double = FakeSamples(values)
    monkeypatch.setattr(access, "open_samples", lambda source, *, name: double)

    return double


@pytest.fixture
def representation(release_root: Path, samples: FakeSamples) -> FakeRepresentation:
    return FakeRepresentation(release_root)


def _at(reader: str) -> OpenKey:
    return OpenKey(dataset=NAME, representation=ORIGINAL, reader=reader)


def _reached(monkeypatch: pytest.MonkeyPatch, cache_dir: Path, samples: FakeSamples) -> FakeLoader:
    """Put a loader the gate can answer about in front of the library, and return it.

    The gate reads the cache directory the constructed loader reports, and constructing the real one
    needs a corpus. Replacing the one call that constructs it is what lets the gate be driven both
    ways without one.
    """
    loader = FakeLoader(cache_dir, samples)
    monkeypatch.setattr(preparation, "_reach", lambda source: (loader, SleepStaging()))

    return loader


# Both adapters satisfy the protocols the grid declares, by shape


def test_both_adapters_satisfy_the_protocols_the_grid_declares_by_shape(
    release_root: Path, samples: FakeSamples
) -> None:
    # ty is the gate: these functions are annotated against the grid's own protocols, and the
    # connector imports neither of them. A renamed member is a type-check diagnostic.
    def needs_items(items: ParsedItems) -> ParsedItems:
        return items

    def needs_a_parsing_path(parsing_path: ParsingPath) -> ParsingPath:
        return parsing_path

    connector = SleepEdfxConnector()
    items = connector.open_pandas(release_root)

    assert needs_items(items) is items
    assert needs_a_parsing_path(connector) is connector


def test_the_access_module_imports_neither_reader_protocol_module() -> None:
    # The dependency runs one way, source to grid, and the shape is the whole of what they share.
    tree = ast.parse(ACCESS_MODULE.read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}

    for protocol in ("timenet_evaluations.grid.parsing", "timenet_evaluations.grid.reader"):
        assert protocol not in imported, f"the adapters import {protocol}"


def test_neither_adapter_inherits_anything_of_this_projects() -> None:
    # The one base class is the framework's, which the harness's loader is written against.
    assert SleepEdfxItems.__mro__ == (SleepEdfxItems, object)
    assert [base.__module__.split(".")[0] for base in SleepEdfxSignals.__bases__] == ["torch"]


# Each open is a timed cell of its own, and is not the call the conversion makes


def test_each_open_reaches_the_library_again_and_serves_nothing_a_previous_call_produced(
    monkeypatch: pytest.MonkeyPatch, release_root: Path, values: np.ndarray
) -> None:
    # A cell served from memory reports an in-memory attribute access under a column that says read
    # from disk, and it is several times faster with no structural tell.
    opened: list[FakeSamples] = []

    def open_samples(source: Path, *, name: str) -> FakeSamples:
        opened.append(FakeSamples(values))
        return opened[-1]

    monkeypatch.setattr(access, "open_samples", open_samples)
    connector = SleepEdfxConnector()

    first = connector.open_pandas(release_root)
    second = connector.open_pandas(release_root)
    third = connector.open_torch(release_root)

    assert first is not second
    assert second is not third
    assert len(opened) == 3
    assert len(set(map(id, opened))) == 3


def test_no_open_is_served_from_anything_the_untimed_preparation_produced() -> None:
    code = _code_of_module(ACCESS_MODULE)

    for produced in ("prepare", "drain", "Preparation", "load_for_conversion"):
        assert produced not in code, f"a timed open reaches {produced}"


@pytest.mark.parametrize("member", ["open_pandas", "open_torch"])
def test_the_connector_members_reach_the_library_only_through_the_gated_route(member: str) -> None:
    # open_samples is the one route to the call that could build, and the gate sits inside it.
    body = _code_of(SLEEP_EDFX_MODULE, member)

    assert "check_source_root" in body
    for reach in ("SleepEDFDataset", "set_task", "get_dataloader"):
        assert reach not in body, f"{member} reaches {reach} itself"


def test_the_access_module_reaches_the_library_only_through_the_gated_route() -> None:
    code = _code_of_module(ACCESS_MODULE)

    assert "open_samples" in code
    for reach in ("SleepEDFDataset", "set_task", "get_dataloader"):
        assert reach not in code, f"the adapters reach {reach} themselves"


# No timed cell may build the task cache


@pytest.mark.parametrize("member", ["open_pandas", "open_torch"])
def test_a_timed_open_against_an_absent_cache_refuses_before_the_library_is_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, release_root: Path, values: np.ndarray, member: str
) -> None:
    # Otherwise the cell would time a parse of every recording and a multi-gigabyte write, under a
    # column heading that reports a read.
    loader = _reached(monkeypatch, tmp_path / "a-user-cache", FakeSamples(values))

    with pytest.raises(EvaluationError) as failure:
        getattr(SleepEdfxConnector(), member)(release_root)

    assert loader.tasks == 0
    message = str(failure.value)
    assert NAME in message
    assert str(release_root) in message


@pytest.mark.parametrize("member", ["open_pandas", "open_torch"])
def test_a_timed_open_against_a_built_cache_reaches_the_library_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, release_root: Path, values: np.ndarray, member: str
) -> None:
    cache_dir = tmp_path / "a-user-cache"
    index = task_cache_index(cache_dir, SleepStaging())
    index.parent.mkdir(parents=True)
    index.write_text("{}", encoding="utf-8")
    loader = _reached(monkeypatch, cache_dir, FakeSamples(values))

    opened = getattr(SleepEdfxConnector(), member)(release_root)

    assert loader.tasks == 1
    assert len(opened) == N_ITEMS


# The torch open goes through the reference loader's own torch path


def test_the_torch_path_builds_no_frame_and_no_dataloader() -> None:
    # Routing the torch cell through a frame would put every cost of the other target form inside
    # the torch column; a second dataloader could not be checked against the harness's own.
    code = _code_of_module(ACCESS_MODULE)

    for built in ("DataFrame", "pandas", "DataLoader", "collate", "batch_size", "num_workers"):
        assert built not in code, f"the access module builds or names {built}"


def test_neither_open_is_built_on_the_other() -> None:
    # Routing the torch cell through a frame would put every cost of the other target form inside
    # the torch column, and the two reader columns of that row would stop differing by the form.
    assert "open_signals" not in _code_of(ACCESS_MODULE, "open_items")
    assert "open_items" not in _code_of(ACCESS_MODULE, "open_signals")
    assert "open_torch" not in _code_of(SLEEP_EDFX_MODULE, "open_pandas")
    assert "open_pandas" not in _code_of(SLEEP_EDFX_MODULE, "open_torch")


def test_the_torch_adapter_is_map_style_and_the_sample_object_it_wraps_is_not(
    release_root: Path, samples: FakeSamples
) -> None:
    # A loader given an iterable-style dataset ignores every index-based sampler and walks __iter__,
    # so the first-item read would go through the streaming path and would not be narrow.
    assert issubclass(SampleDataset, IterableDataset)

    dataset = SleepEdfxConnector().open_torch(release_root)

    assert isinstance(dataset, SleepEdfxSignals)
    assert not isinstance(dataset, IterableDataset)
    assert len(dataset) == N_ITEMS
    assert isinstance(dataset[0], torch.Tensor)


def test_the_harness_loader_walks_the_adapter_by_index_in_ascending_order(
    release_root: Path, samples: FakeSamples, values: np.ndarray
) -> None:
    dataset = SleepEdfxConnector().open_torch(release_root)

    batches = list(build_loader(dataset))

    assert samples.touched == list(range(N_ITEMS))
    assert samples.slices == []
    assert [tuple(batch.shape) for batch in batches] == [(1, *ITEM_SHAPE)] * N_ITEMS
    assert np.array_equal(np.stack([batch[0].numpy() for batch in batches]), values)


# The first-item read is a narrow read on this connector


def test_an_integer_index_reads_one_item_and_touches_nothing_else(
    release_root: Path, samples: FakeSamples, values: np.ndarray
) -> None:
    items = SleepEdfxConnector().open_pandas(release_root)
    assert isinstance(items, SleepEdfxItems)

    first = items[0]

    assert samples.touched == [0]
    assert samples.slices == []
    assert np.array_equal(first, values[0])


@pytest.mark.parametrize("reader", [PandasReader(), PyTorchReader()])
def test_the_first_item_read_of_either_cell_resolves_position_zero_only(
    representation: FakeRepresentation, samples: FakeSamples, values: np.ndarray, reader: Any
) -> None:
    opened = reader.open(representation)

    item = opened.read_first()

    assert samples.touched == [0]
    assert np.array_equal(
        to_comparison_form(item, representation.item, at=_at(reader.name), path=representation.artifact.path),
        values[0],
    )


def test_the_two_cells_of_the_row_hold_the_same_item_at_position_zero(
    representation: FakeRepresentation, samples: FakeSamples
) -> None:
    path = representation.artifact.path
    frame = PandasReader().open(representation).read_first()
    tensor = PyTorchReader().open(representation).read_first()

    assert np.array_equal(
        to_comparison_form(frame, representation.item, at=_at(PANDAS), path=path),
        to_comparison_form(tensor, representation.item, at=_at(PYTORCH), path=path),
    )


# The block read uses the sample object's indexed access


def test_a_slice_reads_exactly_the_items_it_names_in_ascending_order(
    release_root: Path, samples: FakeSamples, values: np.ndarray
) -> None:
    items = SleepEdfxConnector().open_pandas(release_root)
    assert isinstance(items, SleepEdfxItems)

    run = items[INTERIOR.start : INTERIOR.start + INTERIOR.count]

    assert samples.touched == [8, 9, 10, 11]
    assert samples.indices == []
    assert np.array_equal(run, values[8:12])


@pytest.mark.parametrize("block", [INTERIOR, FINAL])
@pytest.mark.parametrize("reader", [PandasReader(), PyTorchReader()])
def test_a_block_read_touches_the_blocks_positions_and_nothing_outside_them(
    representation: FakeRepresentation, samples: FakeSamples, block: FakeBlock, reader: Any
) -> None:
    # A short final block is served like any other, without padding and without a raise.
    opened = reader.open(representation)

    read = opened.read_block(block)

    assert samples.touched == list(range(block.start, block.start + block.count))
    assert len(read) == block.count * (N_CHANNELS if reader.name == PANDAS else 1)


def test_a_read_past_the_end_is_refused_naming_the_scope_and_the_path(release_root: Path, samples: FakeSamples) -> None:
    items = SleepEdfxConnector().open_pandas(release_root)

    with pytest.raises(EvaluationError) as failure:
        items[N_ITEMS - 2 : N_ITEMS + 2]

    message = str(failure.value)
    for coordinate in (NAME, ORIGINAL, PANDAS, str(release_root)):
        assert coordinate in message
    assert str(N_ITEMS) in message


def test_a_block_is_never_served_by_slicing_a_full_read(
    representation: FakeRepresentation, samples: FakeSamples
) -> None:
    # Indexed access exists on this connector, so a block that cut a full read would measure
    # repeated full passes under the column that reports the shuffled walk.
    PandasReader().open(representation).read_block(INTERIOR)

    assert samples.slices == [(INTERIOR.start, INTERIOR.start + INTERIOR.count)]
    assert samples.touched == list(range(INTERIOR.start, INTERIOR.start + INTERIOR.count))


# The full read and the sequential walk are distinct passes


def test_the_full_read_is_one_slice_over_the_declared_count(
    representation: FakeRepresentation, samples: FakeSamples, values: np.ndarray
) -> None:
    frame = PandasReader().open(representation).read_all()

    assert samples.slices == [(0, N_ITEMS)]
    assert samples.indices == []
    assert np.array_equal(frame.to_numpy().reshape(N_ITEMS, *ITEM_SHAPE), values)


def test_the_full_read_is_materialised_and_is_not_a_lazy_handle(release_root: Path, samples: FakeSamples) -> None:
    items = SleepEdfxConnector().open_pandas(release_root)

    run = items[0:N_ITEMS]

    assert isinstance(run, np.ndarray)
    assert run.shape == (N_ITEMS, *ITEM_SHAPE)


def test_the_sequential_walk_asks_for_one_position_at_a_time(
    representation: FakeRepresentation, samples: FakeSamples
) -> None:
    # Not a bulk pass the walk then iterates: that would measure the bulk pass and report it in
    # items per second under a column that exists to measure something else.
    walked = list(PandasReader().open(representation).read_sequential())

    assert len(walked) == N_ITEMS
    assert samples.indices == list(range(N_ITEMS))
    assert samples.slices == []


def test_the_torch_sequential_walk_is_the_item_at_a_time_pass(
    representation: FakeRepresentation, samples: FakeSamples
) -> None:
    walked = list(PyTorchReader().open(representation).read_sequential())

    assert len(walked) == N_ITEMS
    assert samples.indices == list(range(N_ITEMS))
    assert samples.slices == []


@pytest.mark.parametrize("reader", [PandasReader(), PyTorchReader()])
def test_the_full_read_and_the_sequential_walk_are_two_passes_and_not_one(
    representation: FakeRepresentation, samples: FakeSamples, reader: Any
) -> None:
    opened = reader.open(representation)

    opened.read_all()
    touched_by_the_bulk_pass = list(samples.touched)
    list(opened.read_sequential())

    assert samples.touched == touched_by_the_bulk_pass + list(range(N_ITEMS))


# The ordering, the length, and the retention flag


def test_each_open_turns_shuffling_off_rather_than_relies_on_a_default(
    release_root: Path, samples: FakeSamples
) -> None:
    # The harness's loader calls neither the library's shuffle setter nor its dataloader helper, so
    # the adapters establish the declared ordering themselves.
    connector = SleepEdfxConnector()

    connector.open_pandas(release_root)
    connector.open_torch(release_root)

    assert samples.shuffle == [False, False]


def test_the_length_is_the_one_the_sample_object_reports_and_is_read_once_at_the_open(
    release_root: Path, samples: FakeSamples
) -> None:
    # No rate divides by it. The library computes it from the loader configuration in force when it
    # is asked, so a length read again during a task could move the range every read is checked
    # against.
    items = SleepEdfxConnector().open_pandas(release_root)

    assert samples.lengths == 1
    assert len(items) == N_ITEMS
    assert len(items) == N_ITEMS
    assert samples.lengths == 1


def test_the_adapters_neither_read_nor_write_the_retention_flag() -> None:
    # The slice path of the storage library sets it. Within one repetition that retention is a real
    # property of the loader, and across repetitions the opened object is discarded.
    assert "on_demand_bytes" not in _code_of_module(ACCESS_MODULE)


# One canonical item is that item's signal values and nothing else


@pytest.mark.parametrize("member", ["open_pandas", "open_torch"])
def test_an_item_carries_the_signal_and_none_of_the_keys_beside_it(
    release_root: Path, samples: FakeSamples, values: np.ndarray, member: str
) -> None:
    opened = getattr(SleepEdfxConnector(), member)(release_root)

    item = opened[0]

    assert not isinstance(item, dict | tuple)
    assert tuple(np.asarray(item).shape) == ITEM_SHAPE
    assert np.array_equal(np.asarray(item), values[0])


@pytest.mark.parametrize("member", ["open_pandas", "open_torch"])
def test_a_sample_that_carries_no_signal_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, release_root: Path, values: np.ndarray, member: str
) -> None:
    # A value guessed at here would reach a report as a rate over content nobody checked.
    monkeypatch.setattr(access, "open_samples", lambda source, *, name: FakeSamples(values, carries_signal=False))
    opened = getattr(SleepEdfxConnector(), member)(release_root)

    with pytest.raises(EvaluationError) as failure:
        opened[3]

    message = str(failure.value)
    assert "position 3" in message
    assert str(release_root) in message


def test_the_item_is_float32_in_both_target_forms(release_root: Path, samples: FakeSamples) -> None:
    connector = SleepEdfxConnector()
    items = connector.open_pandas(release_root)
    signals = connector.open_torch(release_root)

    assert isinstance(items, SleepEdfxItems)
    assert isinstance(signals, SleepEdfxSignals)
    assert items[0].dtype == np.float32
    assert signals[0].dtype == torch.float32
