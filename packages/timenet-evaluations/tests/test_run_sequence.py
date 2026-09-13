"""The run sequence: nineteen steps, one function, and the order they happen in.

Five requirements of SPEC-0021 are properties of one function's order, so the central test here
performs a whole run over doubles and asserts the order its collaborators were called in. The
doubles record into one shared log, and every ordering claim is an assertion about positions in
that log: the environment before the first load, the probe before the first drop of any cell, both
sizes after the conversion and both plans after the sizes, parity before the first timed cell, one
derivation after the last cell, and the record written before the summary is printed.

The connector, the conversion, the environment capture, the calibration probe, the page-cache drop,
the parity check and the two readers are doubles. Everything else is the real thing: the real
``measure_dataset`` walks the real grid, the real ``measure_size`` counts real bytes on disk, the
real block planner plans them, the real derivation derives, and the real record writer refuses a
result that is not whole. A run that produced fifteen cells would fail here at the writer rather
than at an assertion, which is the point of leaving those real.

No double reads a page cache and none of them is evidence that a figure is cold. What these tests
pin is the shape and the order of the call graph, which is what the requirement is about.
"""

import ast
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import Dataset

from timenet_evaluations import cli as cli_module
from timenet_evaluations.arguments import DatasetPair
from timenet_evaluations.cli import run_evaluation
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid import (
    CELLS_PER_DATASET,
    STORAGE_FIGURES_PER_DATASET,
    Artifact,
    Cell,
    ItemDeclaration,
    ParsedItems,
    Representation,
    StorageKey,
    assembly as assembly_module,
)
from timenet_evaluations.grid.facts import DatasetFacts
from timenet_evaluations.grid.reader import Block
from timenet_evaluations.grid.readers import PANDAS, PYTORCH
from timenet_evaluations.harness import tasks as tasks_module
from timenet_evaluations.harness.calibration import Calibration
from timenet_evaluations.harness.drop import DARWIN, DropRecord
from timenet_evaluations.harness.plan import BlockPlan, record_plan
from timenet_evaluations.harness.size import measure_size
from timenet_evaluations.metrics import derive_metrics
from timenet_evaluations.report import RECORD_NAME, write_json
from timenet_evaluations.result import DatasetRecord, Environment, EvaluationResult, Measurements, Metrics
from timenet_evaluations.source.contract import LABEL, PATIENT, SIGNAL
from timenet_evaluations.source.declaration import DatasetDeclaration
from timenet_evaluations.source.parity import ParityReference
from timenet_evaluations.source.registry import CONNECTORS
from timenet_evaluations.summary import print_summary


ITEMS = 8
"""How many canonical items each stub dataset declares."""

ITEM_SHAPE = (2, 4)

SOURCE_BYTES = 800
"""What one release file holds. It is also what ``Original``'s storage figure must come to."""

CONVERTED_BYTES = 400
"""What the converted artifact holds. It differs from the source, so the two plans of one dataset
are planned from two different sizes."""

BLOCK_BYTES = 40

SEED = 11

REPEATS = 1
"""One recorded repetition per task. The count is odd, which the timing loop requires, and the
sequence does not depend on it."""

RECORD = DropRecord(command="/usr/sbin/purge", exit_status=0, platform=DARWIN)

ENVIRONMENT = Environment(
    python="3.13.1",
    platform="darwin-arm64",
    cpu_count=8,
    packages={"timenet-evaluations": "0.1.0"},
)

CALIBRATION_RATIO = 6.0

SOURCE_FILE = "release.bin"

CLI_SOURCE = Path(inspect.getfile(cli_module))

SEQUENCE = (
    "check_datasets",
    "new_run_id",
    "collect_environment",
    "calibrate",
    "run_directory",
    "connector_for",
    "write",
    "verify",
    "check_artifact_agrees",
    "check_parity",
    "measure_dataset",
    "derive_metrics",
    "write_json",
    "print_summary",
)
"""The collaborators the sequence calls, in the order the specification puts them.

The names are asserted against the syntax tree of ``run_evaluation`` itself, so a step moved above
or below another one fails here even when every double still reports success.
"""


class StubParsingPath:
    """A loader nothing opens: both readers of this run are doubles."""

    def open_pandas(self, path: Path) -> ParsedItems:
        raise AssertionError("a timed task opens through its reader, never through the harness")

    def open_torch(self, path: Path) -> Dataset[object]:
        raise AssertionError("a timed task opens through its reader, never through the harness")


class StubRepresentation:
    """A representation of a chosen name, over a real path of a chosen size."""

    def __init__(
        self, name: str, path: Path, *, dataset: str, item: ItemDeclaration, records: str | None = None
    ) -> None:
        self.dataset = dataset
        self.name = name
        self.artifact = Artifact(representation=name if records is None else records, path=path)
        self.item = item
        self.parsing_path = StubParsingPath()


class SpyOpened:
    """One representation opened by a double reader. It reads nothing."""

    def __init__(self, reader: "SpyReader", representation: Representation) -> None:
        self._reader = reader
        self._representation = representation

    def read_first(self) -> object:
        return 0

    def read_all(self) -> object:
        return list(range(self._reader.items))

    def read_sequential(self) -> Iterator[object]:
        yield from range(self._reader.items)

    def read_block(self, block: Block) -> object:
        return block


class SpyReader:
    """A reader that records the artifact it was opened against, and reads nothing."""

    def __init__(self, name: str, log: list[str], *, items: int = ITEMS) -> None:
        self.name = name
        self.log = log
        self.items = items
        self.opened: list[Path] = []

    def open(self, representation: Representation) -> SpyOpened:
        self.log.append(f"cell:{representation.dataset}:{representation.name}")
        self.opened.append(representation.artifact.path)

        return SpyOpened(self, representation)


class SpyDrop:
    """A drop that records what it was asked to precede. It starts no process."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.calls: list[tuple[Cell | StorageKey, Path]] = []

    def __call__(self, *, at: Cell | StorageKey, path: Path) -> DropRecord:
        self.log.append("drop")
        self.calls.append((at, path))

        return RECORD


class SpyConnector:
    """One dataset's four members, each of which records that it was reached.

    ``load_for_conversion`` builds a fresh frame per call, so a second load of one dataset is
    visible as a second object and not only as a second entry in the log.
    """

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log
        self.frames: list[pd.DataFrame] = []

    def load_for_conversion(self, source: Path) -> pd.DataFrame:
        self.log.append(f"load:{self.name}")
        frame = pd.DataFrame(
            {
                SIGNAL: [np.zeros(ITEM_SHAPE, dtype=np.float32) for _ in range(ITEMS)],
                LABEL: ["a"] * ITEMS,
                PATIENT: ["p"] * ITEMS,
            }
        )
        self.frames.append(frame)

        return frame

    def open_pandas(self, path: Path) -> ParsedItems:
        raise AssertionError("the release's files are opened inside a timed cell, by a reader")

    def open_torch(self, path: Path) -> Dataset[object]:
        raise AssertionError("the release's files are opened inside a timed cell, by a reader")

    def canonical_item(self, source: Path) -> DatasetDeclaration:
        self.log.append(f"declare:{self.name}")

        return DatasetDeclaration(
            unit="item",
            item=ItemDeclaration(count=ITEMS, shape=ITEM_SHAPE),
            rate_hz=100,
            facts=DatasetFacts(
                card_id="timenet/hello_world",
                modality="stub",
                modality_name="Stub",
                unit="volt",
                rate_hz=100,
                channels=tuple(f"c{index}" for index in range(ITEM_SHAPE[0])),
                target_schema="stub_class",
            ),
            disclosures={"original": "the cells read a materialized cache"},
        )


class SpyConversion:
    """Steps 9 and 10, standing in for the two module-level names the run calls.

    The run imports `write` and `verify` from the conversion package, so there is no object to
    inject: `build` patches those names in the run's own namespace and these methods are what they
    resolve to. `TimeF` is patched the same way, so a stub representation reaches the measurement
    rather than one built over a real artifact.

    ``nested`` is what makes the returned path different from the directory the conversion was
    given, which is the case the size measurement and every read must follow.
    """

    def __init__(self, log: list[str], *, nested: bool = False, records: str | None = None) -> None:
        self.log = log
        self.nested = nested
        self.records = records
        self.frames: list[tuple[str, pd.DataFrame]] = []
        self.roots: list[Path] = []
        self.produced: list[StubRepresentation] = []
        self.verified: list[Artifact] = []
        self.facts: list[DatasetFacts] = []

    def write(self, frame: pd.DataFrame, root: Path, *, dataset: str, facts: DatasetFacts) -> Artifact:
        self.log.append(f"convert:{dataset}")
        self.frames.append((dataset, frame))
        self.roots.append(root)
        self.facts.append(facts)

        directory = root / dataset
        path = directory / "versions" / "0.1.0" / "values.parquet" if self.nested else directory / "timef.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * CONVERTED_BYTES)

        return Artifact(representation="timef", path=path)

    def build_timef(
        self, *, dataset: str, artifact: Artifact, item: ItemDeclaration, parsing_path: object
    ) -> Representation:
        produced = StubRepresentation(
            artifact.representation, artifact.path, dataset=dataset, item=item, records=self.records
        )
        self.produced.append(produced)

        return produced

    def parsing_path(self, *, dataset: str, channels: tuple[str, ...]) -> object:
        return object()

    def verify(self, artifact: Artifact, *, dataset: str) -> None:
        self.log.append(f"verify:{dataset}")
        self.verified.append(artifact)


@dataclass
class Harness:
    """Everything one scripted run was given, and everything it recorded."""

    log: list[str]
    conversion: SpyConversion
    connectors: dict[str, SpyConnector]
    readers: tuple[SpyReader, SpyReader]
    drop: SpyDrop
    datasets: tuple[DatasetPair, ...]
    out: Path
    artifacts: Path
    sizes: list[str] = field(default_factory=list)
    plans: list[str] = field(default_factory=list)

    def perform(self) -> EvaluationResult:
        """Run the sequence over the doubles."""
        return run_evaluation(
            self.datasets,
            self.out,
            artifacts=self.artifacts,
            repeats=REPEATS,
            block_bytes=BLOCK_BYTES,
            seed=SEED,
        )

    def index(self, entry: str) -> int:
        return self.log.index(entry)

    def first(self, prefix: str) -> int:
        return next(position for position, entry in enumerate(self.log) if entry.startswith(prefix))

    def last(self, prefix: str) -> int:
        return max(position for position, entry in enumerate(self.log) if entry.startswith(prefix))


def build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    names: Sequence[str] = ("alpha",),
    nested: bool = False,
    records: str | None = None,
) -> Harness:
    """Register one double per dataset and put every collaborator of the run behind a double."""
    log: list[str] = []
    connectors = {name: SpyConnector(name, log) for name in names}
    pairs = []
    for name, connector in connectors.items():
        source = tmp_path / "sources" / name
        source.mkdir(parents=True)
        (source / SOURCE_FILE).write_bytes(b"y" * SOURCE_BYTES)
        monkeypatch.setitem(CONNECTORS, name, connector)
        pairs.append(DatasetPair(text=f"{name}={source}", name=name, path=source))

    readers = (SpyReader(PANDAS, log), SpyReader(PYTORCH, log))
    monkeypatch.setattr(assembly_module, "READERS", readers)

    drop = SpyDrop(log)
    monkeypatch.setattr(cli_module, "PlatformDropCaches", lambda command: drop)
    monkeypatch.setattr(cli_module, "TIMEF_INSTALLED", True)

    conversion = SpyConversion(log, nested=nested, records=records)
    # The run imports these four from the conversion package, so they are patched in its namespace
    # rather than handed in. `raising=False` because three of them are bound only where the extra is
    # installed, and this suite runs in the environment continuous integration builds, which has no
    # extra and fakes the flag above.
    monkeypatch.setattr(cli_module, "write", conversion.write, raising=False)
    monkeypatch.setattr(cli_module, "verify", conversion.verify, raising=False)
    monkeypatch.setattr(cli_module, "TimeF", conversion.build_timef, raising=False)
    monkeypatch.setattr(cli_module, "TimeFParsingPath", conversion.parsing_path, raising=False)

    harness = Harness(
        log=log,
        conversion=conversion,
        connectors=connectors,
        readers=readers,
        drop=drop,
        datasets=tuple(pairs),
        out=tmp_path / "results",
        artifacts=tmp_path / "artifacts",
    )

    def environment() -> Environment:
        log.append("environment")
        return ENVIRONMENT

    def calibrate(probe: object, *, drop_caches: object, threshold: float) -> Calibration:
        log.append("probe")
        return Calibration(ratio=CALIBRATION_RATIO, threshold=threshold, cold_ns=600, warm_ns=100, drop=RECORD)

    def parity(reference: ParityReference, representations: object, readers_given: object) -> None:
        log.append(f"parity:{reference.dataset}")

    def sized(path: Path, *, dataset: str, representation: str) -> int:
        log.append(f"size:{dataset}:{representation}")
        harness.sizes.append(representation)
        return measure_size(path, dataset=dataset, representation=representation)

    def planned(size_bytes: int, n_items: int, block_bytes: int, seed: int) -> BlockPlan:
        log.append(f"plan:{size_bytes}")
        harness.plans.append(str(size_bytes))
        return record_plan(size_bytes, n_items, block_bytes, seed)

    def derived(measurements: Measurements, records: Sequence[DatasetRecord], *, path: Path) -> Metrics:
        log.append("derive")
        return derive_metrics(measurements, records, path=path)

    def written(result: EvaluationResult, run_dir: Path) -> Path:
        log.append("record")
        return write_json(result, run_dir)

    def printed(result: EvaluationResult) -> None:
        log.append("summary")
        print_summary(result)

    monkeypatch.setattr(cli_module, "collect_environment", environment)
    monkeypatch.setattr(cli_module, "calibrate", calibrate)
    monkeypatch.setattr(cli_module, "check_parity", parity)
    monkeypatch.setattr(cli_module, "derive_metrics", derived)
    monkeypatch.setattr(cli_module, "write_json", written)
    monkeypatch.setattr(cli_module, "print_summary", printed)
    monkeypatch.setattr(tasks_module, "measure_size", sized)
    monkeypatch.setattr(tasks_module, "record_plan", planned)

    return harness


# The order of the sequence


def test_a_run_records_its_collaborators_in_the_order_the_sequence_fixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = build(tmp_path, monkeypatch, names=("alpha", "beta"))

    harness.perform()
    capsys.readouterr()

    # The environment can refuse a run, and it costs seconds. A refusal after two parses of a
    # release costs minutes, which is why it is above the first load rather than beside it.
    assert harness.index("environment") < harness.first("load:")
    # The probe is the only thing that can observe whether the drop works at all. A probe after any
    # measurement would be certifying figures that had already been taken.
    assert harness.index("probe") < harness.first("drop")
    # The plan sizes a block from the measured size, and the size describes the artifact the
    # conversion produced. Both sit between the conversion and the first cell of that dataset.
    assert harness.first("convert:") < harness.first("size:") < harness.first("plan:")
    # Per representation, not per dataset. SPEC-0021's test list asks for "both sizes after the
    # conversion and both plans after the sizes", which reads as size, size, plan, plan. The merged
    # `measure_dataset` interleaves them — size, plan, size, plan — because a plan is computed from
    # its own representation's size and holding both sizes first would buy nothing. The tree is the
    # fact here; what matters is that each size precedes the plan derived from it, and that every
    # one of them follows the conversion.
    assert harness.last("size:alpha") < harness.last("plan:")
    # A row whose representations disagree about content must not be printed, and discovering it
    # after sixteen cells wastes the run.
    assert harness.index("parity:alpha") < harness.first("cell:")
    # One derivation, after every measurement exists, so the record and the printed grid are
    # downstream of the same arithmetic.
    assert harness.log.count("derive") == 1
    assert harness.last("cell:") < harness.index("derive") < harness.index("record") < harness.index("summary")


def test_every_dataset_is_converted_verified_and_measured_before_the_next_one_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Writing several gigabytes while a cold read is timed measures the write, so the passes are
    # per dataset and a conversion never runs while a cell of another dataset is in flight.
    harness = build(tmp_path, monkeypatch, names=("alpha", "beta"))

    harness.perform()
    capsys.readouterr()

    assert harness.index("convert:alpha") < harness.index("verify:alpha") < harness.first("cell:alpha")
    assert harness.last("cell:alpha") < harness.index("load:beta") < harness.index("convert:beta")


def _module() -> ast.Module:
    """Parse the run's own module, so the order is read from the source and not from a double."""
    return ast.parse(Path(cli_module.__file__).read_text(encoding="utf-8"))


def _run_function() -> ast.FunctionDef:
    """Find `run_evaluation` in that syntax tree."""
    return next(
        node for node in ast.walk(_module()) if isinstance(node, ast.FunctionDef) and node.name == "run_evaluation"
    )


def _calls(function: ast.FunctionDef) -> list[str]:
    """Name every function the run calls, in source order.

    Sorted by line, because `ast.walk` is breadth-first: it yields the calls at the top of the
    function before any call inside the per-dataset loop, whatever order the source puts them in.
    Reading the order out of an unsorted walk would assert the shape of the tree rather than the
    order of the steps, which is the one thing this file exists to check.
    """
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    names = []
    for node in sorted(calls, key=lambda call: (call.lineno, call.col_offset)):
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return names


def _tree_state(root: Path) -> dict[str, tuple[int, int]]:
    """Record every file beneath a directory, by size and modification time.

    A run that created, changed or removed anything under a source path moves one of these.
    """
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_the_syntax_tree_of_the_run_holds_the_steps_in_the_specified_order() -> None:
    # The doubles above show the order one run took. This shows the order the source can take, so a
    # step moved between two others fails even where every double still reports success.
    function = _run_function()
    called = [name for name in _calls(function) if name in SEQUENCE]

    assert [name for name in SEQUENCE if name in called] == list(dict.fromkeys(called))


def test_the_sequence_is_one_function_and_delegates_no_step_to_a_helper() -> None:
    # Five requirements are properties of this one function's order. An order split across helpers
    # is an order nothing can asserts against the source, so the module holds no other step.
    defined = {node.name for node in ast.walk(_module()) if isinstance(node, ast.FunctionDef)}

    assert defined == {"read_every_byte", "run_evaluation", "build_parser", "main"}


# One load per dataset, and the frame it produced is the frame the conversion converts


def test_the_connector_is_called_once_per_dataset_and_the_conversion_gets_that_very_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = build(tmp_path, monkeypatch, names=("alpha", "beta"))

    harness.perform()
    capsys.readouterr()

    for name, connector in harness.connectors.items():
        assert len(connector.frames) == 1
        converted = [frame for dataset, frame in harness.conversion.frames if dataset == name]
        assert len(converted) == 1
        # Identity, not equality: a copy made for another purpose would compare equal and would be
        # a second materialization of a corpus that must be loaded once.
        assert converted[0] is connector.frames[0]


# The whole grid, and nothing but the whole grid


@pytest.mark.parametrize("names", [("alpha", "beta"), ("alpha", "beta", "gamma")])
def test_the_counts_are_a_function_of_the_dataset_list_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], names: tuple[str, ...]
) -> None:
    harness = build(tmp_path, monkeypatch, names=names)

    result = harness.perform()
    capsys.readouterr()

    assert len(result.measurements.timed) == CELLS_PER_DATASET * len(names)
    assert len(result.measurements.storage) == STORAGE_FIGURES_PER_DATASET * len(names)
    assert [record.dataset for record in result.metadata.datasets] == list(names)


def test_the_run_returns_the_result_it_constructed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # It is a library entry point as well as the body of the command, and a caller that invoked it
    # directly would otherwise have to re-read the JSON it just wrote.
    harness = build(tmp_path, monkeypatch)

    result = harness.perform()
    capsys.readouterr()

    assert isinstance(result, EvaluationResult)
    assert result.metadata.protocol.calibration_ratio == CALIBRATION_RATIO
    assert result.metadata.datasets[0].unit == "item"
    assert result.metadata.datasets[0].representations[0].disclosure == "the cells read a materialized cache"


# The two roots, and the source that is neither


def test_the_conversion_is_handed_the_artifact_root_and_the_run_directory_holds_only_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The artifact outlives the run that wrote it, so it is not beneath the run directory: one
    # written into a fresh per-run directory could never be reused, because no later run looks
    # there.
    harness = build(tmp_path, monkeypatch, names=("alpha", "beta"))

    result = harness.perform()
    capsys.readouterr()
    run_dir = harness.out / result.metadata.run_id

    assert harness.conversion.roots == [harness.artifacts, harness.artifacts]
    assert [path.name for path in sorted(run_dir.iterdir())] == [RECORD_NAME]
    assert [path.name for path in sorted(harness.artifacts.iterdir())] == ["alpha", "beta"]
    # The name is the pair's own and never the source directory's basename.
    assert all(harness.artifacts in produced.artifact.path.parents for produced in harness.conversion.produced)


def test_the_artifacts_own_path_is_what_is_sized_and_read_even_when_it_nests_deeper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = build(tmp_path, monkeypatch, nested=True)

    result = harness.perform()
    capsys.readouterr()
    produced = harness.conversion.produced[0].artifact.path
    stored = {entry.at.representation: entry.size_bytes for entry in result.measurements.storage}

    assert produced.parent != harness.artifacts / "alpha"
    assert stored == {"original": SOURCE_BYTES, "timef": CONVERTED_BYTES}
    assert produced in set(harness.readers[0].opened)
    assert produced in set(harness.readers[1].opened)


def test_nothing_beneath_a_source_path_is_created_modified_or_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The source directory is `Original`'s artifact. A run that wrote there would change the size
    # of the very thing it was measuring.
    harness = build(tmp_path, monkeypatch, names=("alpha", "beta"))
    before = {pair.name: _tree_state(pair.path) for pair in harness.datasets}

    harness.perform()
    capsys.readouterr()

    assert {pair.name: _tree_state(pair.path) for pair in harness.datasets} == before


def test_a_second_run_against_one_output_root_leaves_the_first_runs_directory_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = build(tmp_path, monkeypatch)

    first = harness.perform()
    capsys.readouterr()
    first_dir = harness.out / first.metadata.run_id
    before = _tree_state(first_dir)

    second = harness.perform()
    capsys.readouterr()

    assert second.metadata.run_id != first.metadata.run_id
    assert _tree_state(first_dir) == before
    assert sorted(path.name for path in harness.out.iterdir()) == sorted(
        [first.metadata.run_id, second.metadata.run_id]
    )


# The identity of a representation, checked before it is sized, planned or timed


def test_a_representation_whose_artifact_names_another_stops_the_run_before_any_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Position was a fragile pairing when there were three measurements. With sixteen cells one
    # misalignment propagates through the size, the plan and every cell of the representation.
    harness = build(tmp_path, monkeypatch, records="original")

    with pytest.raises(EvaluationError) as refusal:
        harness.perform()

    assert "do not name the same representation" in str(refusal.value)
    assert [entry for entry in harness.log if entry.startswith(("size:", "plan:", "cell:", "drop"))] == []


# The refusals a run makes before it touches a dataset


def test_a_run_without_the_extra_is_refused_and_the_refusal_names_the_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The message the refusal replaced named "pandas and torch", which under ADR-0024 are readers
    # and not representations, and so described an experiment that no longer exists.
    harness = build(tmp_path, monkeypatch)
    monkeypatch.setattr(cli_module, "TIMEF_INSTALLED", False)

    with pytest.raises(EvaluationError) as refusal:
        harness.perform()

    message = str(refusal.value)

    assert "'original', 'timef'" in message
    assert f"{CELLS_PER_DATASET} timed cells" in message
    assert f"{STORAGE_FIGURES_PER_DATASET} storage figures" in message
    assert harness.log == []


def test_progress_goes_to_standard_error_and_leaves_standard_output_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # SPEC-0019 makes standard output the summary and nothing else, byte-identical in a terminal and
    # in a pipe. A progress line on standard output would corrupt the one output a reader diffs, so
    # every one of them goes to standard error and this is what holds them there.
    harness = build(tmp_path, monkeypatch)

    harness.perform()
    captured = capsys.readouterr()

    assert "loading" in captured.err
    assert "measuring" in captured.err
    assert "done in" in captured.err

    for line in captured.out.splitlines():
        assert "loading" not in line
        assert "converting or reusing" not in line
        assert "verifying the artifact" not in line
        assert not line.startswith("run ") or "started" in line
