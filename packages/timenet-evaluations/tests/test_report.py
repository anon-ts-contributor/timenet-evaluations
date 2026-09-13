"""The record that stays and the summary that does not.

The record is checked as a file: written, read back, and compared field by field against a second
record of the same shape. The summary is checked as bytes: written to two streams that answer
``isatty`` differently, and compared.

Nothing here times anything. The run these tests report is the one the fixtures build, and its
durations are integers that were never on a clock.
"""

import ast
from datetime import datetime
import io
import json
from pathlib import Path
from typing import Any

import pytest

from timenet_evaluations import result as result_module, source as source_package
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.assembly import check_grid_complete
from timenet_evaluations.grid.cell import Cell, Task
from timenet_evaluations.harness.repeat import Measurement
from timenet_evaluations.metrics import derive_metrics
from timenet_evaluations.report import (
    RECORD_NAME,
    check_each_slot_measured_once,
    check_one_protocol,
    new_run_id,
    run_directory,
    write_json,
)
from timenet_evaluations.result import (
    CacheProtocol,
    Environment,
    EvaluationResult,
    Measurements,
    Metadata,
    Metrics,
    Rate,
    RepresentationRecord,
    StorageMeasurement,
)
from timenet_evaluations.summary import print_summary


PACKAGE_ROOT = Path(result_module.__file__).parent

PACKAGE_MODULES = sorted(PACKAGE_ROOT.rglob("*.py"))

SOURCE_PACKAGE = Path(source_package.__file__).parent

AUDITED = [module for module in PACKAGE_MODULES if SOURCE_PACKAGE not in module.parents]

OUTPUT_MODULES = (PACKAGE_ROOT / "report.py", PACKAGE_ROOT / "summary.py")

ALLOWED_IMPORTS = {"__future__", "collections", "pathlib", "sys", "typing", "uuid", "timenet_evaluations"}
"""What the two output modules may reach for: the standard library they need and their own package.

A formatting library is not on the list, and neither is anything that would ask a stream what it is
writing to. `uuid` is here because a run names itself without coordination."""

DISCLOSURE = "These figures measure a cold read of a cache the untimed preparation materialized."

RATE_TASK_VALUES = ("sequential", "block_shuffled")

ESCAPE = "\x1b"

COMPOSED = (
    'note = f"reads: {record.disclosure}"',
    'note = "these cells read " + record.disclosure',
    'note = " ".join(["reads", record.disclosure])',
    'note = record.disclosure.replace("cache", "index")',
    'PREFIX = "what these cells read: "\nnote = f"{PREFIX}{record.disclosure}"',
    'DISCLOSURE_FOR_ORIGINAL = "these cells read a cache"',
)
"""Wording a reporter might grow, each of which the audit must refuse."""

RENDERED = (
    'lines.append(f"{INDENT * 2}{record.disclosure}")',
    "lines.append(record.disclosure)",
    'INDENT = "  "\nlines.append(f"{INDENT}{record.disclosure}")',
    'if record.disclosure is not None:\n    lines.append(f"{INDENT}{record.disclosure}")',
    'lines.append(f"{record.at.representation}  {size} bytes on disk")',
)
"""Lines the tree holds, or would hold, that hand a disclosure on unchanged and must clear the
audit. The last one adds wording to a representation name, which is a column label and not a
disclosure."""


# What a second run of the same source looks like


def another_run(result: EvaluationResult) -> EvaluationResult:
    """Build a second run of the same source: a new identity, and every duration doubled."""
    measurements = Measurements(
        timed=tuple(
            entry.model_copy(update={"samples": tuple(2 * sample for sample in entry.samples)})
            for entry in result.measurements.timed
        ),
        storage=result.measurements.storage,
    )
    metadata = result.metadata.model_copy(update={"run_id": "9c07be14a2f5"})

    return EvaluationResult(
        metadata=metadata,
        measurements=measurements,
        metrics=derive_metrics(measurements, metadata.datasets, path=Path("/runs/9c07be14a2f5") / RECORD_NAME),
    )


def disclosed(result: EvaluationResult) -> EvaluationResult:
    """Give the first representation of the first dataset a statement, and leave the rest alone."""
    first, *rest = result.metadata.datasets
    representations = (
        first.representations[0].model_copy(update={"disclosure": DISCLOSURE}),
        *first.representations[1:],
    )
    datasets = (first.model_copy(update={"representations": representations}), *rest)

    return result.model_copy(update={"metadata": result.metadata.model_copy(update={"datasets": datasets})})


def shape(value: Any) -> Any:
    """Reduce a parsed record to its field order alone, at every level of nesting."""
    if isinstance(value, dict):
        return [(key, shape(item)) for key, item in value.items()]
    if isinstance(value, list):
        return [shape(item) for item in value]

    return None


class Recorder(io.StringIO):
    """A stream that answers ``isatty`` however a test asks it to."""

    def __init__(self, *, tty: bool) -> None:
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


def summary_of(result: EvaluationResult) -> str:
    stream = Recorder(tty=False)
    print_summary(result, stream=stream)

    return stream.getvalue()


def sections(summary: str) -> dict[str, list[str]]:
    """Split a summary into the lines of each dataset group, keyed by the dataset."""
    grouped: dict[str, list[str]] = {}
    current = ""
    for line in summary.splitlines():
        if line.startswith("dataset "):
            current = line.split()[1]
            grouped[current] = []
        elif current:
            grouped[current].append(line)

    return grouped


# The record keeps every field of all three groups


def test_the_record_is_written_into_the_run_directory_under_one_fixed_name(
    run_result: EvaluationResult, tmp_path: Path
) -> None:
    path = write_json(run_result, tmp_path)

    assert path == tmp_path / RECORD_NAME
    assert path.read_text(encoding="utf-8")


def test_the_record_holds_the_three_groups_and_every_declared_field_of_each(
    run_result: EvaluationResult, tmp_path: Path
) -> None:
    record = json.loads(write_json(run_result, tmp_path).read_text(encoding="utf-8"))

    assert list(record) == ["metadata", "measurements", "metrics"]
    assert set(record["metadata"]) == set(Metadata.model_fields)
    assert set(record["metadata"]["environment"]) == set(Environment.model_fields)
    assert set(record["metadata"]["protocol"]) == set(CacheProtocol.model_fields)
    assert set(record["measurements"]) == set(Measurements.model_fields)
    assert set(record["metrics"]) == set(Metrics.model_fields)


def test_the_record_keeps_the_samples_and_the_key_each_metric_names(
    run_result: EvaluationResult, tmp_path: Path
) -> None:
    # The spread of an appendix and the failed-drop trend check both read the samples, and a metric
    # a reader cannot trace back to a cell is a claim they can only trust.
    record = json.loads(write_json(run_result, tmp_path).read_text(encoding="utf-8"))

    entry = record["measurements"]["timed"][0]
    assert set(entry) == set(Measurement.model_fields) | {"repetitions", "elapsed_ns"}
    assert len(entry["samples"]) == entry["repetitions"] > 1

    rate = record["metrics"]["rates"][0]
    assert set(rate) == set(Rate.model_fields)
    assert set(rate["at"]) == set(Cell.model_fields)


def test_the_record_keeps_the_canonical_item_the_block_plan_and_the_disclosure(
    run_result: EvaluationResult, tmp_path: Path
) -> None:
    record = json.loads(write_json(disclosed(run_result), tmp_path).read_text(encoding="utf-8"))

    dataset = record["metadata"]["datasets"][0]
    assert dataset["unit"]
    assert dataset["item_count"] > 0
    assert set(dataset["representations"][0]) == set(RepresentationRecord.model_fields)
    assert dataset["representations"][0]["disclosure"] == DISCLOSURE
    assert dataset["representations"][1]["disclosure"] is None


def test_a_written_record_parses_back_into_the_same_result(run_result: EvaluationResult, tmp_path: Path) -> None:
    path = write_json(run_result, tmp_path)

    assert EvaluationResult.model_validate_json(path.read_text(encoding="utf-8")) == run_result


# Two runs of one source diff to only what changed


def test_two_records_of_one_source_share_a_field_order_at_every_level_of_nesting(
    run_result: EvaluationResult, tmp_path: Path
) -> None:
    # The top level is three keys now, so an assertion over it alone would cover almost nothing.
    first = json.loads(write_json(run_result, tmp_path).read_text(encoding="utf-8"))
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = json.loads(write_json(another_run(run_result), second_dir).read_text(encoding="utf-8"))

    assert shape(first) == shape(second)
    assert first != second


def test_the_two_records_differ_in_the_identity_the_timings_and_the_rates(
    run_result: EvaluationResult, tmp_path: Path
) -> None:
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    first = json.loads(write_json(run_result, tmp_path).read_text(encoding="utf-8"))
    second = json.loads(write_json(another_run(run_result), second_dir).read_text(encoding="utf-8"))

    assert first["metadata"]["run_id"] != second["metadata"]["run_id"]
    assert first["measurements"]["timed"] != second["measurements"]["timed"]
    assert first["metrics"] != second["metrics"]
    assert first["metadata"]["environment"] == second["metadata"]["environment"]


# The summary is the same bytes wherever it goes


def test_the_summary_is_byte_identical_in_a_terminal_and_in_a_pipe(run_result: EvaluationResult) -> None:
    terminal = Recorder(tty=True)
    pipe = Recorder(tty=False)

    print_summary(run_result, stream=terminal)
    print_summary(run_result, stream=pipe)

    assert terminal.getvalue() == pipe.getvalue()
    assert terminal.getvalue()


def test_the_summary_carries_no_escape_sequence_and_no_colour(run_result: EvaluationResult) -> None:
    summary = summary_of(run_result)

    assert ESCAPE not in summary
    assert "\r" not in summary
    assert "\\033" not in summary


def test_the_default_stream_receives_the_same_bytes(
    run_result: EvaluationResult, capsys: pytest.CaptureFixture[str]
) -> None:
    print_summary(run_result)

    assert capsys.readouterr().out == summary_of(run_result)


@pytest.mark.parametrize("module", OUTPUT_MODULES, ids=lambda path: path.name)
def test_neither_output_reaches_for_a_formatting_library(module: Path) -> None:
    tree = ast.parse(module.read_text(encoding="utf-8"))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= ALLOWED_IMPORTS, f"{module.name} reaches {sorted(imported - ALLOWED_IMPORTS)}"


# One header line, then the run's numbers, and no slot left out


def test_the_summary_opens_with_one_header_line_naming_the_run_and_its_protocol(
    run_result: EvaluationResult,
) -> None:
    header, *rest = summary_of(run_result).splitlines()

    assert header.startswith(f"run {run_result.metadata.run_id}")
    assert "warmup off" in header
    assert str(run_result.metadata.protocol.calibration_ratio) in header
    assert not [line for line in rest if line.startswith("run ")]


def test_the_summary_prints_every_timed_cell_and_every_storage_figure(run_result: EvaluationResult) -> None:
    # The table has one column per (task, reader) and one row per representation, so every one of
    # the sixteen cells has exactly one place to be. A cell missing from the table is a number the
    # record holds and the reader never sees.
    summary = summary_of(run_result)

    for entry in run_result.measurements.timed:
        if entry.at.task.value in RATE_TASK_VALUES:
            continue
        assert f"{entry.elapsed_ns / 1_000_000_000:.6f}" in summary

    for entry in run_result.measurements.storage:
        assert f"{entry.size_bytes / 1_000_000_000:.3f}" in summary


def test_the_summary_quotes_the_rate_the_record_holds_and_computes_none(run_result: EvaluationResult) -> None:
    # The printed rate is read out of the metrics group. Recomputing it here would let the printed
    # figure and the stored one drift apart with nothing to notice.
    summary = summary_of(run_result)

    for rate in run_result.metrics.rates:
        assert f"{rate.items_per_second:.1f}" in summary


def test_only_a_rate_task_cell_carries_a_rate(run_result: EvaluationResult) -> None:
    # Two tasks make a rate and two do not, and the table's two column groups follow that split.
    # `metrics` decides it: a rate over the one item of a first-item read is a duration with another
    # name, and the full read is a bulk pass whose seconds are the figure a reader wants.
    lines = summary_of(run_result).splitlines()
    headings = next(line for line in lines if "read time [s]" in line)

    assert headings.index("read time [s]") < headings.index("read rate [items/s]")

    tasks = next(line for line in lines if "first item" in line)

    assert tasks.index("first item") < tasks.index("full read") < tasks.index("sequential")


def test_a_result_holding_a_cell_the_metadata_does_not_name_refuses_rather_than_leaving_it_out(
    run_result: EvaluationResult, capsys: pytest.CaptureFixture[str]
) -> None:
    stray = Measurement(
        at=Cell(dataset="alpha", representation="ghost", reader="pandas", task=Task.FULL_READ),
        samples=(1, 2, 3),
        warmup=False,
        drops=run_result.measurements.timed[0].drops,
    )
    measurements = run_result.measurements.model_copy(update={"timed": (*run_result.measurements.timed, stray)})

    with pytest.raises(EvaluationError) as failure:
        print_summary(run_result.model_copy(update={"measurements": measurements}))

    assert "ghost" in str(failure.value)
    assert capsys.readouterr().out == ""


# Rates are grouped by dataset and never ranked across them


def test_each_dataset_group_names_the_item_its_rates_count(run_result: EvaluationResult) -> None:
    headings = [line for line in summary_of(run_result).splitlines() if line.startswith("dataset ")]

    assert len(headings) == len(run_result.metadata.datasets)
    for heading, dataset in zip(headings, run_result.metadata.datasets, strict=True):
        assert dataset.dataset in heading
        assert dataset.unit in heading
        assert str(dataset.item_count) in heading


def test_every_dataset_group_says_its_rates_count_that_dataset_own_items(run_result: EvaluationResult) -> None:
    # The shape of the output invites the comparison the canonical item forbids, so the statement
    # goes where the numbers are rather than being left for a reader to infer.
    grouped = sections(summary_of(run_result))

    assert set(grouped) == {dataset.dataset for dataset in run_result.metadata.datasets}
    for lines in grouped.values():
        assert [line for line in lines if "two datasets count different things" in line]


def test_a_dataset_group_holds_its_own_rates_in_the_stored_order_and_no_others(
    run_result: EvaluationResult,
) -> None:
    # A rate counts the canonical item of its own dataset, so one column holding two datasets' rates
    # would read as a ranking. Each dataset gets its own block, headed by the item its rates count.
    lines = summary_of(run_result).splitlines()

    for dataset in run_result.metadata.datasets:
        heading = next(index for index, line in enumerate(lines) if line.startswith(f"dataset {dataset.dataset} "))
        following = [line for line in lines[heading:] if line.startswith("dataset ")]

        assert following[0].startswith(f"dataset {dataset.dataset} ")
        assert f"the item is {dataset.unit!r}" in lines[heading]


def test_a_disclosure_is_printed_beneath_the_table_and_names_the_row_it_describes(
    run_result: EvaluationResult,
) -> None:
    # In a table there is no row to hang a sentence on, so the caveat is a footnote — and it names
    # its representation, because a caveat that does not say which row it covers is decoration.
    lines = summary_of(disclosed(run_result)).splitlines()
    footnote = next(line for line in lines if DISCLOSURE in line)
    table_row = next(index for index, line in enumerate(lines) if line.strip().startswith("original"))

    assert lines.index(footnote) > table_row
    assert footnote.strip().startswith("original:")


def test_the_statement_is_printed_word_for_word_as_the_loader_wrote_it(run_result: EvaluationResult) -> None:
    # The wording is the loader's. Nothing here composes one, shortens one, or wraps one: the row it
    # belongs to is named in front of it and the sentence itself is untouched.
    summary = summary_of(disclosed(run_result))

    assert DISCLOSURE in summary


def test_a_representation_with_nothing_to_disclose_prints_no_placeholder(run_result: EvaluationResult) -> None:
    # Absence is the signal, so an empty note would be a statement that nothing was prepared.
    summary = summary_of(run_result)

    assert DISCLOSURE not in summary
    assert not [line for line in summary.splitlines() if line.strip() in {"", "-", "none", "None"} and line]
    assert "None" not in summary


@pytest.mark.parametrize("composed", COMPOSED)
def test_a_module_that_builds_disclosure_wording_fails_the_audit(composed: str) -> None:
    # This direction first. An audit that only runs over a clean tree passes whether it detects
    # anything or not, and the tree is clean today.
    assert _composed_disclosures(composed), f"the audit reads no composed wording in: {composed}"


@pytest.mark.parametrize("rendered", RENDERED)
def test_handing_a_disclosure_on_unchanged_clears_the_audit(rendered: str) -> None:
    assert not _composed_disclosures(rendered), f"the audit reads composed wording in: {rendered}"


@pytest.mark.parametrize("module", AUDITED, ids=lambda path: path.name)
def test_no_module_outside_the_source_package_composes_disclosure_wording(module: Path) -> None:
    # The loader that prepared the dataset is the only component that knows what it prepared, and
    # this capability may not hold a dataset-specific fact at all.
    found = _composed_disclosures(module.read_text(encoding="utf-8"))

    assert not found, f"{module.relative_to(PACKAGE_ROOT)} composes a disclosure: {found}"


# Two protocols are never combined


def test_two_results_of_different_warm_up_settings_refuse_naming_both(
    run_result: EvaluationResult, capsys: pytest.CaptureFixture[str]
) -> None:
    warm = _warmed(run_result)

    with pytest.raises(EvaluationError) as failure:
        check_one_protocol([run_result, warm])

    message = str(failure.value)
    assert "[False, True]" in message
    assert f"run {run_result.metadata.run_id!r} warmup False" in message
    assert f"run {warm.metadata.run_id!r} warmup True" in message
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("warmup", [False, True])
def test_results_taken_under_one_setting_combine(run_result: EvaluationResult, warmup: bool) -> None:
    one = _warmed(run_result) if warmup else run_result

    check_one_protocol([one, another_run(one)])


def test_one_result_alone_is_always_one_protocol(run_result: EvaluationResult) -> None:
    check_one_protocol([run_result])
    check_one_protocol([])


def _warmed(result: EvaluationResult) -> EvaluationResult:
    """Rebuild a result as the warm run it would have been, evidence and claim together."""
    protocol = result.metadata.protocol.model_copy(update={"warmup": True})
    measurements = result.measurements.model_copy(
        update={"timed": tuple(entry.model_copy(update={"warmup": True}) for entry in result.measurements.timed)}
    )

    return EvaluationResult(
        metadata=result.metadata.model_copy(update={"run_id": "1a2b3c4d5e6f", "protocol": protocol}),
        measurements=measurements,
        metrics=result.metrics,
    )


def _letters(text: str) -> bool:
    return any(character.isalpha() for character in text)


def _composed_disclosures(source: str) -> list[str]:
    """Every place one module adds wording of its own to a disclosure, or holds a canned one.

    A module may put whitespace around the text it was handed and nothing else. Anything that adds
    a letter is the module writing part of the statement, which only the loader may do.
    """
    tree = ast.parse(source)
    bound = {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    found = [f"{name} holds wording" for name, text in bound.items() if "disclos" in name.lower() and _letters(text)]

    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr | ast.BinOp | ast.Call):
            continue
        if not any(isinstance(child, ast.Attribute) and child.attr == "disclosure" for child in ast.walk(node)):
            continue

        added = [
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str) and _letters(child.value)
        ]
        added += [
            bound[child.id]
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and _letters(bound.get(child.id, ""))
        ]

        if added:
            found.append(f"{ast.unparse(node)} adds {added}")

    return found


# Each run owns a fresh directory


def test_a_run_identity_is_twelve_hexadecimal_characters_and_no_two_agree() -> None:
    identities = {new_run_id() for _ in range(100)}

    assert len(identities) == 100
    for identity in identities:
        assert len(identity) == 12
        assert set(identity) <= set("0123456789abcdef")


def test_two_runs_against_one_output_root_leave_two_directories(run_result: EvaluationResult, tmp_path: Path) -> None:
    first, second = new_run_id(), new_run_id()

    write_json(named(run_result, first), run_directory(tmp_path, run_id=first))
    write_json(named(another_run(run_result), second), run_directory(tmp_path, run_id=second))

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([first, second])
    assert (tmp_path / first / RECORD_NAME).exists()
    assert (tmp_path / second / RECORD_NAME).exists()


def test_neither_run_touches_the_other_run_files(run_result: EvaluationResult, tmp_path: Path) -> None:
    first, second = new_run_id(), new_run_id()

    written = write_json(named(run_result, first), run_directory(tmp_path, run_id=first))
    before = written.read_text(encoding="utf-8")
    write_json(named(another_run(run_result), second), run_directory(tmp_path, run_id=second))

    assert written.read_text(encoding="utf-8") == before
    assert (tmp_path / second / RECORD_NAME).read_text(encoding="utf-8") != before


def test_creating_the_run_directory_twice_within_one_run_succeeds(tmp_path: Path) -> None:
    # The artifacts and the record are written at different points, so each asks when it needs it.
    run_id = new_run_id()

    first = run_directory(tmp_path, run_id=run_id)
    (first / "an-artifact").write_text("bytes", encoding="utf-8")
    second = run_directory(tmp_path, run_id=run_id)

    assert first == second == tmp_path / run_id
    assert (second / "an-artifact").read_text(encoding="utf-8") == "bytes"


def test_a_record_moved_out_of_its_directory_still_names_its_run(run_result: EvaluationResult, tmp_path: Path) -> None:
    run_id = new_run_id()
    written = write_json(named(run_result, run_id), run_directory(tmp_path, run_id=run_id))
    moved = tmp_path / "somewhere-else.json"
    moved.write_text(written.read_text(encoding="utf-8"), encoding="utf-8")

    assert EvaluationResult.model_validate_json(moved.read_text(encoding="utf-8")).metadata.run_id == run_id


def test_ordering_two_runs_by_time_reads_the_instant_from_inside_each_record(
    run_result: EvaluationResult, tmp_path: Path
) -> None:
    # A listing of run directories carries no chronology, so the instant is a field.
    run_id = new_run_id()
    written = write_json(named(run_result, run_id), run_directory(tmp_path, run_id=run_id))
    record = json.loads(written.read_text(encoding="utf-8"))

    assert datetime.fromisoformat(record["metadata"]["started_at"]) == run_result.metadata.started_at


def test_a_run_directory_that_cannot_be_created_raises_naming_the_run_and_the_path(tmp_path: Path) -> None:
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory", encoding="utf-8")
    run_id = new_run_id()

    with pytest.raises(EvaluationError) as failure:
        run_directory(blocked, run_id=run_id)

    assert run_id in str(failure.value)
    assert str(blocked / run_id) in str(failure.value)


# Every refusal reaches the caller before the record reaches the disk


def test_a_result_short_of_one_timed_slot_refuses_and_leaves_nothing_on_disk(
    run_result: EvaluationResult, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    short = with_measurements(run_result, timed=run_result.measurements.timed[:-1])

    with pytest.raises(EvaluationError) as failure:
        write_json(short, tmp_path)

    message = str(failure.value)
    assert "16 cells declared" in message
    assert "15 measured" in message
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().out == ""


def test_a_result_short_of_a_storage_figure_refuses_and_leaves_nothing_on_disk(
    run_result: EvaluationResult, tmp_path: Path
) -> None:
    short = with_measurements(run_result, storage=run_result.measurements.storage[:-1])

    with pytest.raises(EvaluationError) as failure:
        write_json(short, tmp_path)

    message = str(failure.value)
    assert "2 declared" in message
    assert "1 measured" in message
    assert list(tmp_path.iterdir()) == []


def test_a_seventeenth_entry_duplicating_a_cell_is_what_the_product_check_alone_lets_through(
    run_result: EvaluationResult,
) -> None:
    # The scenario the requirement describes -- sixteen entries, two naming one cell -- is caught by
    # the product check, because the cell that went missing fails the difference. This is the case
    # that actually escapes it: nothing missing, and one cell named twice.
    doubled = duplicated(run_result)

    check_grid_complete(
        [record.dataset for record in doubled.metadata.datasets],
        [entry.at for entry in doubled.measurements.timed],
        [entry.at for entry in doubled.measurements.storage],
    )

    with pytest.raises(EvaluationError):
        check_each_slot_measured_once(doubled.measurements)


def test_a_duplicated_cell_refuses_naming_the_cell_and_how_many_entries_name_it(
    run_result: EvaluationResult, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doubled = duplicated(run_result)
    at = run_result.measurements.timed[0].at

    with pytest.raises(EvaluationError) as failure:
        write_json(doubled, tmp_path)

    message = str(failure.value)
    assert f"({at.dataset}, {at.representation}, {at.reader}, {at.task.value}) x 2" in message
    assert "33 timed entries over 32 cells" in message
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().out == ""


def test_a_second_storage_figure_for_one_representation_refuses(run_result: EvaluationResult, tmp_path: Path) -> None:
    first = run_result.measurements.storage[0]
    priced_twice = with_measurements(
        run_result, storage=(*run_result.measurements.storage, first.model_copy(update={"size_bytes": 1}))
    )

    with pytest.raises(EvaluationError) as failure:
        write_json(priced_twice, tmp_path)

    assert f"({first.at.dataset}, {first.at.representation}) x 2" in str(failure.value)
    assert list(tmp_path.iterdir()) == []


def test_a_metric_that_disagrees_with_its_measurements_refuses_before_the_write(
    run_result: EvaluationResult, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, *rest = run_result.metrics.rates
    tampered = run_result.model_copy(
        update={"metrics": Metrics(rates=(first.model_copy(update={"items_per_second": 1.0}), *rest))}
    )

    with pytest.raises(EvaluationError) as failure:
        write_json(tampered, tmp_path)

    assert "does not recompute" in str(failure.value)
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().out == ""


def test_a_write_that_fails_raises_naming_the_run_and_the_path(
    run_result: EvaluationResult, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The directory was never created, which is the "the parent may not exist" failure.
    absent = tmp_path / "never-created"

    with pytest.raises(EvaluationError) as failure:
        write_json(run_result, absent)

    message = str(failure.value)
    assert run_result.metadata.run_id in message
    assert str(absent / RECORD_NAME) in message
    assert not absent.exists()
    assert capsys.readouterr().out == ""


def test_a_failed_write_keeps_the_error_it_translated(run_result: EvaluationResult, tmp_path: Path) -> None:
    # A raise site may translate a failure from another package and must not absorb one.
    with pytest.raises(EvaluationError) as failure:
        write_json(run_result, tmp_path / "never-created")

    assert isinstance(failure.value.__cause__, OSError)


def test_a_result_that_passes_every_guard_is_written(run_result: EvaluationResult, tmp_path: Path) -> None:
    # The guards must not refuse a good result, which is the half an audit of refusals never shows.
    written = write_json(run_result, tmp_path)

    assert EvaluationResult.model_validate_json(written.read_text(encoding="utf-8")) == run_result


def named(result: EvaluationResult, run_id: str) -> EvaluationResult:
    """Give a result the identity of the directory it is about to be written into."""
    return result.model_copy(update={"metadata": result.metadata.model_copy(update={"run_id": run_id})})


def with_measurements(
    result: EvaluationResult,
    *,
    timed: tuple[Measurement, ...] | None = None,
    storage: tuple[StorageMeasurement, ...] | None = None,
) -> EvaluationResult:
    """Rebuild a result over a different measurement set, leaving the other two groups alone."""
    measurements = result.measurements.model_copy(
        update={
            "timed": result.measurements.timed if timed is None else timed,
            "storage": result.measurements.storage if storage is None else storage,
        }
    )

    return result.model_copy(update={"measurements": measurements})


def duplicated(result: EvaluationResult) -> EvaluationResult:
    """Add one entry naming a cell that is already measured, with nothing missing."""
    first = result.measurements.timed[0]

    return with_measurements(
        result, timed=(*result.measurements.timed, first.model_copy(update={"samples": (7, 8, 9)}))
    )
