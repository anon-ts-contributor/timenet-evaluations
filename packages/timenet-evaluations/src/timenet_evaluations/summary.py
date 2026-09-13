"""The glance at the end of a run: plain lines, and the same bytes wherever they go.

The summary is read once, at the end of a run, and it is gone with the scrollback. It is not the
record. ``report.write_json`` writes that, it keeps every field, and it is what a reader opens a
month later. So the record is where authority lives: this module prints the
grid, and ``result.json`` is what carries the provenance behind it.

It therefore uses no formatting library, and emits no colour and no escape sequence. The bytes a
terminal receives are the bytes a pipe, a log and a captured stream receive, because no part of this
rendering asks what it is writing to.

It computes no number of its own. Turning nanoseconds into seconds for display is presentation and
belongs here; dividing a declared count by a duration is a metric, it belongs to
:mod:`timenet_evaluations.metrics`, and it already ran. The printed rate is read out of the metrics
group, so the printed rate and the stored rate cannot disagree.

It is a module of its own for the reason ``metrics.py`` is: the record and the glance are two
outputs with two jobs and two lifetimes, and one file holding both passes the line cap this
repository keeps.

How the layout was chosen
-------------------------
The grid is printed as a table: one run header line, then one block per dataset, then one row per
representation carrying a storage figure and the eight numbers of that representation's cells.

That is the shape the published table takes, and it is why the layout changed. The specification
used to forbid a table, on the grounds that a formatted table becomes the thing people quote. The
risk is real, and refusing to draw one does not remove it: a reader who wants a table builds one by
hand from a listing, and a hand-built table carries no provenance and no way to check the
transcription. Printing the grid in the shape it will be read in is the safer of the two.

The dataset block is required rather than chosen. A rate counts the canonical item of its own
dataset, that item means something different for each dataset, and one column holding two datasets'
rates would look exactly like a ranking. So each block names the item its rates count, and nothing
sorts or aggregates across blocks.

Two of the four tasks carry a rate and two do not, which is not a presentation choice: ``metrics``
derives a rate for the sequential and block-shuffled walks alone, because a rate over the one item
of the first-item read is a duration with a different name, and the full read is one bulk pass whose
seconds are the figure a reader wants. The table's two column groups follow that split exactly.

The disclosure is a footnote under the block rather than a line among the rows. In a table there is
no row to put a sentence on, and a caveat that travels with the table is what stops the table being
quoted without it. The wording is the loader's; nothing here composes one, shortens one, or supplies
one where the loader gave none.

Alignment is padding written here. There is no width detection and no truncation, so the bytes do
not depend on the terminal: a row too wide for a window wraps, and a summary whose content changed
with the window would not be the same in the four places it is read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import sys
from typing import Final, TextIO

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, StorageKey, Task
from timenet_evaluations.grid.registry import READERS
from timenet_evaluations.harness.repeat import Measurement
from timenet_evaluations.metrics import NANOSECONDS_PER_SECOND
from timenet_evaluations.result import DatasetRecord, EvaluationResult, Metadata, RepresentationRecord


INDENT: Final = "  "
"""One step of indentation in the summary.

Two spaces, repeated. Indentation is the whole of the summary's structure: a dataset group is one
step in, a representation group two, and a timed cell three. Nothing else marks a group, because
anything else would be a drawn table.
"""

SECONDS_PLACES: Final = 6
"""How many decimal places a duration is printed to. The record holds the whole nanoseconds."""

RATE_PLACES: Final = 1
"""How many decimal places a rate is shown to. A rate near a thousand items a second says nothing
in its third decimal, and a narrower column fits the table."""

STORAGE_PLACES: Final = 3
"""How many decimal places a storage figure is shown to. Three places of a gigabyte is a megabyte,
which is the resolution a size difference is read at."""

"""How many decimal places a rate is printed to. The record holds the number the derivation made."""


def print_summary(result: EvaluationResult, *, stream: TextIO | None = None) -> None:
    """Write one run's numbers, as plain lines, for reading once.

    Every timed cell and every storage figure the result holds is printed. A summary that showed the
    slots it could and left out the rest would be a partial report presented as a complete one.

    The lines are built first and written afterwards, so a result the summary cannot show whole
    produces no output at all rather than half of one. :func:`_check_every_slot_is_shown` is what
    decides that, and it stops the run with ``EvaluationError`` before a byte is written.

    Args:
        result: What the run produced.
        stream: Where to write. Standard output by default. Any text stream receives the same bytes,
            because no part of this rendering asks what it is writing to.
    """
    lines = _summary_lines(result)
    out = sys.stdout if stream is None else stream

    for line in lines:
        out.write(f"{line}\n")


def _summary_lines(result: EvaluationResult) -> tuple[str, ...]:
    """Render the whole summary as lines, with no trailing line separator on any of them.

    The guard runs first, so a result this layout cannot show whole gives no lines at all.

    Args:
        result: What the run produced.

    Returns:
        The header line, then one group per dataset.
    """
    _check_every_slot_is_shown(result)

    sizes = {entry.at: entry.size_bytes for entry in result.measurements.storage}
    rates = {rate.at: rate.items_per_second for rate in result.metrics.rates}

    lines = [_header_line(result.metadata)]
    for dataset in result.metadata.datasets:
        lines.append("")
        lines.extend(_dataset_lines(dataset, result.measurements.timed, sizes, rates))

    return tuple(lines)


def _header_line(metadata: Metadata) -> str:
    """Say what run this is and under what protocol its numbers were taken.

    The protocol is on the header because a result that does not say so is not a cold-read result,
    it is an unlabelled one. The measured calibration ratio and the bar it had to clear are both
    here: a ratio near one means the page cache was never dropped and every figure below is warm.

    Args:
        metadata: The circumstance group of the run.

    Returns:
        The one header line.
    """
    protocol = metadata.protocol
    warmup = "on" if protocol.warmup else "off"

    return (
        f"run {metadata.run_id}  started {metadata.started_at.isoformat()}  "
        f"{metadata.environment.platform}  warmup {warmup}  "
        f"drop {protocol.drop_command} exit {protocol.drop_exit_status}  "
        f"calibration {protocol.calibration_ratio} against {protocol.calibration_threshold}  "
        f"datasets {len(metadata.datasets)}"
    )


BYTES_PER_GB = 1_000_000_000
"""Bytes in the gigabyte a storage figure is shown in. Decimal, because a disk is sold in decimal
gigabytes and the number is read against one."""

REPRESENTATION_WIDTH = 12
"""How wide the representation name column is."""

STORAGE_WIDTH = 9
"""How wide the storage column is."""

TIME_WIDTH = 11
"""How wide one read-time column is, including the space that separates it from the one before. A
number that filled its field would touch its neighbour and the two would read as one."""

RATE_WIDTH = 13
"""How wide one read-rate column is, including its separating space. Wider than a time column
because a rate has digits before the point and a duration has almost none."""

TIME_TASKS: Final = (Task.FIRST_ITEM, Task.FULL_READ)
"""The two tasks shown as seconds. They are the two `metrics` derives no rate for."""

RATE_TASK_ORDER: Final = (Task.SEQUENTIAL, Task.BLOCK_SHUFFLED)
"""The two tasks shown as items per second, in the order the columns take."""

TASK_HEADINGS: Final = {
    Task.FIRST_ITEM: "first item",
    Task.FULL_READ: "full read",
    Task.SEQUENTIAL: "sequential",
    Task.BLOCK_SHUFFLED: "block-shuf",
}
"""What each task is called in the column heading."""


def _dataset_lines(
    dataset: DatasetRecord,
    timed: Sequence[Measurement],
    sizes: Mapping[StorageKey, int],
    rates: Mapping[Cell, float],
) -> list[str]:
    """Render one dataset's block: its heading, the table, and any disclosures under it.

    The heading names the canonical item, so the unit of every rate below travels with the numbers
    instead of being looked up. The line under it says that these rates count this dataset's own
    item, because the shape of a table invites the comparison the item forbids.

    Args:
        dataset: What the metadata records about this dataset.
        timed: Every timed entry of the run.
        sizes: The bytes on disk of each representation of the run.
        rates: The rate of each rate-task cell of the run.

    Returns:
        The lines of this dataset's block.
    """
    readers = tuple(reader.name for reader in READERS)
    lines = [
        f"dataset {dataset.dataset}  {dataset.item_count} items declared  the item is {dataset.unit!r}",
        f"{INDENT}every rate below counts this dataset's own items, and two datasets count different things",
        "",
    ]
    lines.extend(_heading_lines(readers))

    durations = {entry.at: entry.elapsed_ns / NANOSECONDS_PER_SECOND for entry in timed}
    for record in dataset.representations:
        lines.append(_representation_row(record, readers, durations, sizes, rates))

    disclosures = [
        f"{INDENT}{record.at.representation}: {record.disclosure}"
        for record in dataset.representations
        if record.disclosure is not None
    ]
    if disclosures:
        lines.append("")
        lines.extend(disclosures)

    return lines


def _heading_lines(readers: tuple[str, ...]) -> list[str]:
    """Build the three heading lines: the column groups, the tasks, and the readers.

    Each group heading is centred over the columns it spans and each task heading over its pair of
    readers, so a column's meaning is read straight up from the number.

    Args:
        readers: The readers of the run, in the order the registry declares them.

    Returns:
        The three lines that head the table.
    """
    span = len(readers)
    stub = f"{INDENT}{'':<{REPRESENTATION_WIDTH}}"
    time_span = TIME_WIDTH * span * len(TIME_TASKS)
    rate_span = RATE_WIDTH * span * len(RATE_TASK_ORDER)

    groups = f"{stub}{'':>{STORAGE_WIDTH}}{'read time [s]'.center(time_span)}{'read rate [items/s]'.center(rate_span)}"

    tasks = f"{stub}{'storage':>{STORAGE_WIDTH}}"
    for task in TIME_TASKS:
        tasks += TASK_HEADINGS[task].center(TIME_WIDTH * span)
    for task in RATE_TASK_ORDER:
        tasks += TASK_HEADINGS[task].center(RATE_WIDTH * span)

    names = f"{stub}{'[GB]':>{STORAGE_WIDTH}}"
    for _ in TIME_TASKS:
        for reader in readers:
            names += f"{reader:>{TIME_WIDTH}}"
    for _ in RATE_TASK_ORDER:
        for reader in readers:
            names += f"{reader:>{RATE_WIDTH}}"

    return [groups.rstrip(), tasks.rstrip(), names.rstrip()]


def _representation_row(
    record: RepresentationRecord,
    readers: tuple[str, ...],
    durations: Mapping[Cell, float],
    sizes: Mapping[StorageKey, int],
    rates: Mapping[Cell, float],
) -> str:
    """Render one representation as one row: its size, then its eight numbers.

    Every number is read out of the run rather than computed here. A duration is nanoseconds shown
    as seconds, which is presentation; a rate comes from the metrics group, so the printed rate and
    the stored rate cannot disagree.

    Args:
        record: What the metadata records about this (dataset, representation) pair.
        readers: The readers of the run, in the order the registry declares them.
        durations: The seconds each timed cell reported.
        sizes: The bytes on disk of each representation of the run.
        rates: The rate of each rate-task cell of the run.

    Returns:
        The one row of this representation.
    """
    dataset = record.at.dataset
    representation = record.at.representation
    gigabytes = sizes[record.at] / BYTES_PER_GB
    row = f"{INDENT}{representation:<{REPRESENTATION_WIDTH}}{gigabytes:>{STORAGE_WIDTH}.{STORAGE_PLACES}f}"

    for task in TIME_TASKS:
        for reader in readers:
            at = Cell(dataset=dataset, representation=representation, reader=reader, task=task)
            row += f"{durations[at]:>{TIME_WIDTH}.{SECONDS_PLACES}f}"
    for task in RATE_TASK_ORDER:
        for reader in readers:
            at = Cell(dataset=dataset, representation=representation, reader=reader, task=task)
            row += f"{rates[at]:>{RATE_WIDTH}.{RATE_PLACES}f}"

    return row


def _check_every_slot_is_shown(result: EvaluationResult) -> None:
    """Make sure that this layout reaches every entry the result holds.

    The summary walks the (dataset, representation) pairs the metadata names. An entry filed under a
    pair the metadata does not name would be skipped in silence, and a partial report presented as
    a complete one is worse than no report. So the pairs are compared here, before a line is built.

    Args:
        result: What the run produced.

    Raises:
        EvaluationError: If a timed entry or a storage figure names a (dataset, representation) pair
            that the metadata does not, or if the storage figures and the named pairs are not one
            for one.
    """
    named = [
        (record.at.dataset, record.at.representation)
        for dataset in result.metadata.datasets
        for record in dataset.representations
    ]
    stored = [(entry.at.dataset, entry.at.representation) for entry in result.measurements.storage]

    if sorted(named) != sorted(stored):
        raise EvaluationError(
            "the summary prints one storage figure under each pair the metadata names, so a figure that no pair "
            f"names would not be printed at all: metadata names {sorted(named)}, measurements hold {sorted(stored)}"
        )

    unreachable = sorted(
        {(entry.at.dataset, entry.at.representation) for entry in result.measurements.timed} - set(named)
    )

    if unreachable:
        raise EvaluationError(
            "a timed entry names a pair the metadata does not, so the summary would leave it out: "
            f"metadata names {sorted(set(named))}, also measured {unreachable}"
        )
