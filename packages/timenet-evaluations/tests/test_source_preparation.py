"""The untimed preparation, the cache gate that keeps a build out of every timer, and the two
statements the preparation obliges every report to make."""

import ast
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.datasets.utils import collate_fn_dict_with_padding
import pytest
import torch

from timenet_evaluations import source as source_package
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.representations.original import Original
from timenet_evaluations.source import COLUMNS, LABEL, PATIENT, SIGNAL
from timenet_evaluations.source.sleep_edfx import NAME
from timenet_evaluations.source.sleep_edfx_preparation import (
    CACHE_DISCLOSURE,
    DISCLOSURES,
    ORIGINAL,
    SAMPLES_CACHE_INDEX,
    SAMPLES_CACHE_PREFIX,
    SAMPLES_CACHE_SUFFIX,
    STORAGE_DISCLOSURE,
    TASK_CACHE_DIRECTORY,
    Preparation,
    check_task_cache,
    drain,
    task_cache_index,
)
from timenet_evaluations.source.sleep_edfx_task import SleepStaging


PREPARATION_MODULE = Path(source_package.__file__).parent / "sleep_edfx_preparation.py"

N_ITEMS = 3
N_CHANNELS = 7
N_SAMPLES = 16
SOURCE = Path("/data/a-release-root")

# The library writes these below the source directory and below its own cache directory, and the
# audits below say the connector touches neither. The names live here rather than in the connector,
# which needs only the second one and reaches it through the constructed loader.
WRITING_CALLS = frozenset(
    {
        "mkdir",
        "touch",
        "unlink",
        "rmdir",
        "rmtree",
        "remove",
        "removedirs",
        "rename",
        "replace",
        "write_text",
        "write_bytes",
        "to_csv",
        "to_parquet",
        "copy",
        "copy2",
        "copytree",
        "move",
    }
)


def _task(chunk_duration: float | None = None) -> SleepStaging:
    """The task the connector asks for, built here the way the connector builds it."""
    return SleepStaging() if chunk_duration is None else SleepStaging(chunk_duration=chunk_duration)


def _items(count: int, *, first_label: int = 0, channels: int = N_CHANNELS) -> list[dict[str, Any]]:
    """The items the sample object yields one at a time, extra keys and all.

    The three keys beyond the contract are what the task emits and the schema passes through, and
    the label is an integer class index rather than a stage name. Both are the readings the
    requirement says are wrong when guessed.
    """
    generator = np.random.default_rng(count)

    return [
        {
            SIGNAL: generator.standard_normal((channels, N_SAMPLES)),
            LABEL: first_label + position,
            PATIENT: f"{position:04d}",
            "night": 1,
            "patient_age": 58,
            "patient_sex": "M",
        }
        for position in range(count)
    ]


def _tree_below(root: Path) -> dict[str, tuple[int, int]]:
    """Every path below a directory, with the size and the modification time of each."""
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns) for path in sorted(root.rglob("*"))
    }


def _code_of(module: Path, name: str) -> str:
    """Unparse one function's body, without its docstring, so prose is not read as behaviour."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            statements = node.body[1:] if ast.get_docstring(node) is not None else node.body
            return "\n".join(ast.unparse(statement) for statement in statements)

    raise AssertionError(f"{module.name} declares no {name}")


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """A cache directory the way a constructed loader reports one: outside the source path."""
    directory = tmp_path / "a-user-cache" / "a-dataset-uuid"
    directory.mkdir(parents=True)

    return directory


def _build_the_cache(cache_dir: Path, task: SleepStaging) -> Path:
    """Put the index a completed build leaves behind exactly where the connector looks for it."""
    index = task_cache_index(cache_dir, task)
    index.parent.mkdir(parents=True)
    index.write_text("{}", encoding="utf-8")

    return index


# The gate: an unbuilt cache is detected without building one


def test_the_gate_refuses_when_the_cache_index_is_absent(cache_dir: Path) -> None:
    with pytest.raises(EvaluationError) as failure:
        check_task_cache(cache_dir, _task(), name=NAME, source=SOURCE)

    message = str(failure.value)
    assert NAME in message
    assert str(SOURCE) in message
    assert str(task_cache_index(cache_dir, _task())) in message


def test_the_refusal_says_what_a_timed_cell_would_otherwise_have_rebuilt(cache_dir: Path) -> None:
    # Not "cache missing". The reader has to learn that the alternative is a full corpus parse and a
    # multi-gigabyte write inside a cell whose column heading says read.
    with pytest.raises(EvaluationError) as failure:
        check_task_cache(cache_dir, _task(), name=NAME, source=SOURCE)

    message = str(failure.value)
    for phrase in ("parse every", "write", "timer", "preparation"):
        assert phrase in message


def test_the_gate_passes_when_the_cache_index_is_present(cache_dir: Path) -> None:
    _build_the_cache(cache_dir, _task())

    assert check_task_cache(cache_dir, _task(), name=NAME, source=SOURCE) is None


def test_a_directory_of_chunks_without_the_index_is_still_an_unbuilt_cache(cache_dir: Path) -> None:
    # The index is written last, after every chunk is flushed, so a directory that holds chunks and
    # no index is an interrupted build the library would redo.
    index = task_cache_index(cache_dir, _task())
    index.parent.mkdir(parents=True)
    (index.parent / "chunk-0.bin").write_bytes(b"")

    with pytest.raises(EvaluationError):
        check_task_cache(cache_dir, _task(), name=NAME, source=SOURCE)


@pytest.mark.parametrize("built", [True, False])
def test_the_gate_creates_and_modifies_nothing_whichever_way_it_answers(cache_dir: Path, built: bool) -> None:
    # The whole point of the gate is that it costs nothing: a run that reaches it with no cache
    # stops having spent nothing, and one that passes has not touched the cache it is about to read.
    if built:
        _build_the_cache(cache_dir, _task())
    before = _tree_below(cache_dir)

    if built:
        check_task_cache(cache_dir, _task(), name=NAME, source=SOURCE)
    else:
        with pytest.raises(EvaluationError):
            check_task_cache(cache_dir, _task(), name=NAME, source=SOURCE)

    assert _tree_below(cache_dir) == before


def test_the_gate_reaches_neither_the_loader_nor_the_call_that_builds() -> None:
    for member in ("check_task_cache", "task_cache_index"):
        body = _code_of(PREPARATION_MODULE, member)
        for reach in ("SleepEDFDataset", "set_task", "get_dataloader", "mkdir"):
            assert reach not in body, f"{member} reaches {reach}"


def test_the_computed_index_names_the_task_and_sits_below_the_cache_directory(cache_dir: Path) -> None:
    index = task_cache_index(cache_dir, _task())

    assert index.name == SAMPLES_CACHE_INDEX
    assert index.parent.name.startswith(SAMPLES_CACHE_PREFIX)
    assert index.parent.name.endswith(SAMPLES_CACHE_SUFFIX)
    assert index.parent.parent.parent.name == TASK_CACHE_DIRECTORY
    assert index.is_relative_to(cache_dir)


def test_the_layout_the_gate_computes_is_still_the_layout_the_library_writes() -> None:
    # The gate names a path the library builds and never tells anyone about, so the one thing that
    # can silently break it is the library moving that path. This reads the library's own source and
    # fails when it does, in the suite, rather than as a refusal nobody can explain on a real corpus.
    built_by = inspect.getsource(BaseDataset.set_task)

    for part in (TASK_CACHE_DIRECTORY, SAMPLES_CACHE_PREFIX, SAMPLES_CACHE_SUFFIX, SAMPLES_CACHE_INDEX):
        assert part in built_by, f"the library no longer names {part!r} where the cache is built"
    for keyed_on in ("input_schema", "output_schema", "input_processors", "output_processors"):
        assert keyed_on in built_by, f"the cache is no longer keyed on {keyed_on!r}"
    assert "uuid.uuid5(uuid.NAMESPACE_DNS" in built_by
    assert "sort_keys=True" in built_by


def test_a_cache_built_for_another_unit_does_not_answer_for_this_one(cache_dir: Path) -> None:
    # The path is a function of the task's own arguments. A gate that accepted any cache under the
    # dataset would pass a build the next call would redo inside a timer.
    _build_the_cache(cache_dir, _task(chunk_duration=15.0))

    assert task_cache_index(cache_dir, _task()) != task_cache_index(cache_dir, _task(chunk_duration=15.0))
    with pytest.raises(EvaluationError):
        check_task_cache(cache_dir, _task(), name=NAME, source=SOURCE)


def test_the_cache_the_gate_looks_at_is_never_below_the_source_path(cache_dir: Path) -> None:
    # A cache below the recordings would be counted by the storage measurement as part of the
    # release, so the connector leaves the location at the library's own default.
    assert not task_cache_index(cache_dir, _task()).is_relative_to(SOURCE)
    assert "cache_dir" not in _code_of(PREPARATION_MODULE, "_reach")


# A timed open reaches the gate before the call that would build


def test_every_route_to_the_building_call_passes_the_gate_first() -> None:
    body = _code_of(PREPARATION_MODULE, "open_samples")

    assert body.index("check_task_cache") < body.index("set_task")


def test_the_preparation_is_the_one_caller_that_builds_and_does_not_gate_itself() -> None:
    # A gate in front of the preparation would refuse the only call allowed to do the building.
    body = _code_of(PREPARATION_MODULE, "prepare")

    assert "set_task" in body
    assert "check_task_cache" not in body


def test_the_preparation_confirms_the_index_the_gate_will_look_for(cache_dir: Path) -> None:
    # If the library's layout moves, the run stops here, in untimed work, rather than as a refusal
    # nobody can explain inside a timed open later.
    body = _code_of(PREPARATION_MODULE, "prepare")

    assert body.index("set_task") < body.index("task_cache_index")
    assert "EvaluationError" in body


def test_a_timed_open_is_not_served_from_anything_the_preparation_produced() -> None:
    body = _code_of(PREPARATION_MODULE, "open_samples")

    for produced in ("prepare", "drain", "Preparation", "DataFrame"):
        assert produced not in body, f"the timed open reaches {produced}"


# The preparation writes nothing below the source path, and deletes nothing anywhere


def test_the_connector_creates_moves_and_deletes_nothing() -> None:
    # The one write below the source path is the library's own, at construction. Deleting it would
    # make the next run rebuild it inside whichever step ran first, so nothing here removes it.
    tree = ast.parse(PREPARATION_MODULE.read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert not called & WRITING_CALLS, f"the preparation writes or deletes: {sorted(called & WRITING_CALLS)}"


def test_the_connector_imports_no_module_whose_job_is_removing_files() -> None:
    tree = ast.parse(PREPARATION_MODULE.read_text(encoding="utf-8"))
    roots = {
        alias.name.split(".")[0] if isinstance(node, ast.Import) else (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }

    for remover in ("shutil", "os", "tempfile"):
        assert remover not in roots, f"the preparation imports {remover}"


def test_nothing_in_the_preparation_absorbs_a_failure() -> None:
    # A parse failure, a filesystem failure and a library failure all reach the caller with their
    # own type. The gate and the drain raise before the failing call, never around it.
    tree = ast.parse(PREPARATION_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.ExceptHandler), "the preparation catches an error"
        assert not isinstance(node, ast.Try), "the preparation wraps a boundary in a try"


# The frame the drain produces


def test_the_frame_carries_exactly_the_three_contract_columns() -> None:
    prepared = drain(_items(N_ITEMS), name=NAME, source=SOURCE)

    assert tuple(prepared.frame.columns) == COLUMNS
    for passed_through in ("night", "patient_age", "patient_sex"):
        assert passed_through not in prepared.frame.columns


def test_an_integer_class_index_reaches_the_frame_as_its_string_form() -> None:
    # The task emits 0 to 5 through its multiclass output processor. It is not a stage name, and a
    # test or a report that called it one would be describing a different task.
    prepared = drain(_items(N_ITEMS, first_label=2), name=NAME, source=SOURCE)

    assert list(prepared.frame[LABEL]) == ["2", "3", "4"]
    assert prepared.frame[LABEL].map(type).eq(str).all()


def test_the_frame_is_materialized_and_not_a_view_over_the_loader() -> None:
    prepared = drain(iter(_items(N_ITEMS)), name=NAME, source=SOURCE)

    assert isinstance(prepared.frame, pd.DataFrame)
    assert len(prepared.frame) == N_ITEMS


def test_the_count_is_the_number_of_rows_this_pass_produced() -> None:
    # Not a length the sample object reports about itself, which is a function of the loader
    # configuration in force when it is read.
    prepared = drain([*_items(2), *_items(N_ITEMS)], name=NAME, source=SOURCE)

    assert prepared.item_count == 2 + N_ITEMS
    assert prepared.item_count == len(prepared.frame)


def test_the_channel_count_is_read_from_the_items_this_pass_produced() -> None:
    prepared = drain(_items(N_ITEMS), name=NAME, source=SOURCE)

    assert prepared.channel_count == N_CHANNELS


def test_a_source_that_drains_no_items_is_refused_rather_than_measured() -> None:
    # An empty frame is a value a caller can mistake for a good read of an empty dataset, and every
    # rate would divide by nothing.
    with pytest.raises(EvaluationError) as failure:
        drain([], name=NAME, source=SOURCE)

    message = str(failure.value)
    assert NAME in message
    assert str(SOURCE) in message


def test_an_item_short_of_a_contract_column_is_refused_by_the_contract_and_not_defaulted() -> None:
    # patient_id is read directly. A fallback would satisfy the type of the contract while carrying
    # no subject at all.
    items = _items(N_ITEMS)
    del items[0][PATIENT]

    with pytest.raises(EvaluationError) as failure:
        drain(items, name=NAME, source=SOURCE)

    assert PATIENT in str(failure.value)


# The drain reads the items itself, so nothing pads them


def test_two_items_that_disagree_on_their_channel_count_stop_the_run() -> None:
    # PyHealth's collate pads the first axis, and the first axis of an item is the channel axis, so
    # a two-channel item beside a three-channel one used to come back as three channels for both,
    # the shorter one carrying a channel of zeros nothing downstream could see.
    items = [*_items(1), *_items(1, channels=N_CHANNELS + 1)]

    with pytest.raises(EvaluationError) as failure:
        drain(items, name=NAME, source=SOURCE)

    message = str(failure.value)
    assert NAME in message
    assert str(SOURCE) in message
    assert str(N_CHANNELS) in message
    assert str(N_CHANNELS + 1) in message
    assert "item 1" in message
    assert items[1][PATIENT] in message


def test_a_channel_count_that_disagrees_is_refused_rather_than_padded_into_agreement() -> None:
    # The frame that the old path produced was uniform and wrong. Nothing about it showed the
    # fabricated channel, so the refusal is the only thing that can.
    with pytest.raises(EvaluationError):
        drain([*_items(1, channels=N_CHANNELS + 1), *_items(1)], name=NAME, source=SOURCE)


def test_the_collate_the_drain_avoids_still_fabricates_a_channel() -> None:
    # The reason the drain reads the items itself, read from the library rather than remembered.
    # get_dataloader installs this function, it pads the first axis of the values it stacks, and the
    # first axis of one item is the channel axis. The item with fewer channels comes back with a
    # channel of zeros nothing in the frame, the conversion or the parity check could see.
    collated = collate_fn_dict_with_padding(
        [
            {SIGNAL: torch.ones(N_CHANNELS, N_SAMPLES), LABEL: torch.tensor(0), PATIENT: "0000"},
            {SIGNAL: torch.ones(N_CHANNELS + 1, N_SAMPLES), LABEL: torch.tensor(1), PATIENT: "0001"},
        ]
    )

    assert collated[SIGNAL].shape == (2, N_CHANNELS + 1, N_SAMPLES)
    assert torch.equal(collated[SIGNAL][0][N_CHANNELS], torch.zeros(N_SAMPLES))


def test_the_drain_reaches_no_dataloader_and_no_collate_function() -> None:
    # A dataloader is a reader, not a preparation step, and any collate is a chance to reshape the
    # values behind this connector's back.
    tree = ast.parse(PREPARATION_MODULE.read_text(encoding="utf-8"))
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom) for alias in node.names
    }

    assert "get_dataloader" not in imported
    for member in ("prepare", "drain"):
        assert "get_dataloader" not in _code_of(PREPARATION_MODULE, member)
        assert "DataLoader" not in _code_of(PREPARATION_MODULE, member)


def test_what_the_pass_established_cannot_be_moved_afterwards() -> None:
    prepared = drain(_items(N_ITEMS), name=NAME, source=SOURCE)

    with pytest.raises(AttributeError):
        prepared.item_count = 0  # ty: ignore[invalid-assignment]
    assert isinstance(prepared, Preparation)


# The two statements the preparation obliges every report to make


def test_the_statements_are_keyed_by_the_representation_they_describe() -> None:
    # One statement per representation, which is the shape the declaration carries. The converted
    # representation has no entry, because absence is the signal rather than an empty note.
    assert set(DISCLOSURES) == {Original.name}
    assert Original.name == ORIGINAL


def test_the_statement_says_what_the_measured_cells_read() -> None:
    assert CACHE_DISCLOSURE in DISCLOSURES[ORIGINAL]
    for phrase in ("cache", "untimed preparation", "not a parse"):
        assert phrase in CACHE_DISCLOSURE


def test_the_statement_says_what_the_storage_figure_includes() -> None:
    assert STORAGE_DISCLOSURE in DISCLOSURES[ORIGINAL]
    for phrase in ("storage figure includes", "source directory", "left in "):
        assert phrase in STORAGE_DISCLOSURE


def test_the_statements_come_from_the_connector_and_not_from_anything_downstream() -> None:
    # The wording is composed here, by the code that caused the two facts. Nothing outside the seam
    # may hold it, because nothing outside the seam may name this dataset or its library.
    package_root = Path(source_package.__file__).parent.parent
    for module in sorted(package_root.rglob("*.py")):
        if module == PREPARATION_MODULE:
            continue

        text = module.read_text(encoding="utf-8")
        assert CACHE_DISCLOSURE not in text
        assert STORAGE_DISCLOSURE not in text
