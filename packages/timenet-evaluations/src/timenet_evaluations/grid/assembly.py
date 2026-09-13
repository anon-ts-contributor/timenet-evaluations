"""The grid, assembled: every reader paired with every representation, for one dataset.

Reading across representations is the experiment. A user who is deciding whether to adopt the
format is not choosing a library, because they already have one. They are choosing what sits on
disk under the library they keep, and only a cell in which the reader is held still and the
representation moves answers that. So this module pairs each of the two readers with each of the
two representations, and it refuses a run in which a reader reaches only one of them.

The superseded rule that a subject reads back only what it wrote is reversed. It was correct for a
benchmark of three self-contained formats. It is backwards for this one: under that rule the Pandas
reader can open only a file that a Pandas writer produced, the release's own files are not one, and
the question this repository exists to answer has no expressible form.

The cost of the reversal is real and is stated rather than argued away. A connector written for
this benchmark now sits inside every timed cell of the representation it parses. The mitigation is
partial. Where a release ships its own reference loader, the adapter inside the timer is the
release's. Where no reference loader exists, the adapter is this project's, and every report of
those numbers says so.

This module holds one silent-failure guard. A representation declares its own ``name`` and carries
an ``artifact`` that declares a representation of its own. The two are independent members, and
nothing else keeps them equal. If they drift, a storage figure is filed under the wrong row and
every number still renders: no type error, no failed read, and a report that looks correct. This is
the one place where a representation and its artifact are both in scope, so the agreement is
checked here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, StorageKey
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.reader import Reader
from timenet_evaluations.grid.registry import (
    CELLS_PER_DATASET,
    READERS,
    REPRESENTATIONS,
    STORAGE_FIGURES_PER_DATASET,
    TASKS,
)
from timenet_evaluations.grid.representation import Representation


@dataclass(frozen=True)
class Pairing:
    """One representation and one reader of one dataset. The four timed cells of a cell share it.

    A pairing is what a run opens. It is not a number and it holds none: the reader opens the
    representation once per timed repetition, and the opened reader is discarded when that
    repetition ends.
    """

    dataset: str
    """The dataset both members belong to."""
    representation: Representation
    """The representation to read, which this reader did not produce."""
    reader: Reader
    """The reader that reads it. It reads the other representation as well."""

    @property
    def cells(self) -> tuple[Cell, ...]:
        """Name the four timed cells this pairing produces, one per task.

        Returns:
            The four cells, in the order the tasks are declared in.
        """
        return tuple(
            Cell(
                dataset=self.dataset,
                representation=self.representation.name,
                reader=self.reader.name,
                task=task,
            )
            for task in TASKS
        )

    @property
    def storage_key(self) -> StorageKey:
        """Name the storage figure of this pairing's representation.

        Two pairings of one representation give the same key, because a reader does not change
        bytes on disk. That is why a dataset has two storage figures and not four.

        Returns:
            The storage figure's two coordinates.
        """
        return StorageKey(dataset=self.dataset, representation=self.representation.name)


def check_artifact_agrees(representation: Representation, *, dataset: str) -> None:
    """Make sure that a representation and its artifact name the same representation.

    The two are independent members. If they drift, the storage figure of one representation is
    filed under the row of the other, and every number in the report still renders. Nothing later
    finds the mistake, so it is found here.

    Args:
        representation: The representation to examine.
        dataset: The name of the dataset it holds, as the run was given it. The run's own name is
            what the message carries, because that is the name every other coordinate of this run
            is filed under.

    Raises:
        EvaluationError: If the artifact names a representation other than this one.
    """
    if representation.artifact.representation != representation.name:
        raise EvaluationError(
            failure_message(
                "a representation and its artifact do not name the same representation",
                at=StorageKey(dataset=dataset, representation=representation.name),
                path=representation.artifact.path,
                detail=f"artifact representation {representation.artifact.representation!r}",
            )
        )


def assemble(dataset: str, representations: Sequence[Representation]) -> tuple[Pairing, ...]:
    """Pair every reader with every representation of one dataset.

    The result holds four pairings and therefore sixteen timed cells. A run that paired each reader
    with one representation would collapse the grid to a diagonal, and no cell would remain in
    which the representation moves while the reader is held still.

    Args:
        dataset: The name of the dataset these representations hold.
        representations: The dataset's representations, in the order the registry declares them.
            The run builds them, because each one needs a path, an item declaration, and a parsing
            path that arrive with the dataset.

    Returns:
        Every (representation, reader) pairing, representation by representation.

    Raises:
        EvaluationError: If the representations are not exactly the declared axis in the declared
            order, or if one of them disagrees with its own artifact.
    """
    supplied = tuple(representation.name for representation in representations)
    if supplied != REPRESENTATIONS:
        raise EvaluationError(
            f"the representations of a dataset must be the whole declared axis, in order, "
            f"because a run reduced to one representation compares nothing: dataset "
            f"{dataset!r}, declared {list(REPRESENTATIONS)}, supplied {list(supplied)}"
        )

    for representation in representations:
        check_artifact_agrees(representation, dataset=dataset)

    return tuple(
        Pairing(dataset=dataset, representation=representation, reader=reader)
        for representation in representations
        for reader in READERS
    )


def expected_cells(dataset: str) -> tuple[Cell, ...]:
    """Name every timed cell one dataset must produce.

    The names come from the three declared axes, so a cell that no axis names cannot appear here
    and a cell that every axis names cannot go missing.

    Args:
        dataset: The name of the dataset.

    Returns:
        The sixteen cells, representation by representation, then reader, then task.
    """
    return tuple(
        Cell(dataset=dataset, representation=representation, reader=reader.name, task=task)
        for representation in REPRESENTATIONS
        for reader in READERS
        for task in TASKS
    )


def expected_storage_keys(dataset: str) -> tuple[StorageKey, ...]:
    """Name every storage figure one dataset must produce.

    Args:
        dataset: The name of the dataset.

    Returns:
        The two figures, one per representation, in the order the registry declares them.
    """
    return tuple(StorageKey(dataset=dataset, representation=representation) for representation in REPRESENTATIONS)


def check_grid_complete(
    datasets: Sequence[str],
    cells: Iterable[Cell],
    storage_keys: Iterable[StorageKey],
) -> None:
    """Make sure that a run measured the whole grid of every dataset it was given.

    A complete run over one dataset carries sixteen timed cells and two storage figures. Over N
    datasets it carries 16N and 2N. The count is unconditional, and a result that holds fewer cells
    than the whole grid is an incomplete run and not a partial result.

    Run this before a report is printed or written. A report that looks complete and is not is
    worse than no report.

    Args:
        datasets: The datasets the run measured, each named once.
        cells: The timed cells the run produced.
        storage_keys: The storage figures the run produced.

    Raises:
        EvaluationError: If a dataset is named twice, or if the cells or the storage figures of any
            dataset are not exactly the ones the grid declares.
    """
    if len(set(datasets)) != len(datasets):
        raise EvaluationError(
            f"a run names each dataset once, because two grids under one name cannot be told "
            f"apart: datasets {list(datasets)}"
        )

    measured = set(cells)
    stored = set(storage_keys)
    named = set(datasets)

    for cell in sorted(measured, key=_cell_order):
        if cell.dataset not in named:
            raise EvaluationError(
                f"a timed cell names a dataset the run was not given: cell {cell.dataset!r}, "
                f"{cell.representation!r}, {cell.reader!r}, {cell.task.value!r}, datasets "
                f"{sorted(named)}"
            )

    for key in sorted(stored, key=_storage_order):
        if key.dataset not in named:
            raise EvaluationError(
                f"a storage figure names a dataset the run was not given: figure "
                f"{key.dataset!r}, {key.representation!r}, datasets {sorted(named)}"
            )

    for dataset in datasets:
        _check_dataset_complete(dataset, measured, stored)


def _check_dataset_complete(dataset: str, measured: set[Cell], stored: set[StorageKey]) -> None:
    """Make sure that one dataset carries its whole grid.

    Args:
        dataset: The name of the dataset.
        measured: Every timed cell the run produced, over every dataset.
        stored: Every storage figure the run produced, over every dataset.

    Raises:
        EvaluationError: If the dataset's cells or storage figures are not the declared ones.
    """
    required = set(expected_cells(dataset))
    missing = sorted(required - measured, key=_cell_order)
    if missing:
        absent = ", ".join(f"({cell.representation}, {cell.reader}, {cell.task.value})" for cell in missing)
        raise EvaluationError(
            f"a run measures the whole grid of every dataset, and this one is incomplete: "
            f"dataset {dataset!r}, {CELLS_PER_DATASET} cells declared, "
            f"{len(required & measured)} measured, missing {absent}"
        )

    off_axis = sorted({cell for cell in measured if cell.dataset == dataset} - required, key=_cell_order)
    if off_axis:
        extra = ", ".join(f"({cell.representation}, {cell.reader}, {cell.task.value})" for cell in off_axis)
        raise EvaluationError(
            f"a timed cell names a subject the grid does not declare: dataset {dataset!r}, "
            f"representations {list(REPRESENTATIONS)}, readers "
            f"{[reader.name for reader in READERS]}, tasks {[task.value for task in TASKS]}, "
            f"also measured {extra}"
        )

    figures = set(expected_storage_keys(dataset))
    absent_figures = sorted(figures - stored, key=_storage_order)
    if absent_figures:
        names = ", ".join(key.representation for key in absent_figures)
        raise EvaluationError(
            f"a dataset carries one storage figure per representation: dataset {dataset!r}, "
            f"{STORAGE_FIGURES_PER_DATASET} declared, {len(figures & stored)} measured, "
            f"missing {names}"
        )

    reader_indexed = {key for key in stored if key.dataset == dataset} - figures
    if reader_indexed:
        extra = ", ".join(sorted(key.representation for key in reader_indexed))
        raise EvaluationError(
            f"a storage figure names a representation the grid does not declare, and a figure "
            f"indexed by a reader is the usual cause: dataset {dataset!r}, declared "
            f"{list(REPRESENTATIONS)}, also measured {extra}"
        )


def _cell_order(cell: Cell) -> tuple[str, int, int, int]:
    """Order cells the way the registry declares the three axes.

    Args:
        cell: The cell to place.

    Returns:
        The dataset, then the position of the representation, the reader, and the task. An axis
        member the registry does not declare sorts last.
    """
    readers = tuple(reader.name for reader in READERS)
    tasks = tuple(task.value for task in TASKS)

    return (
        cell.dataset,
        _position(cell.representation, REPRESENTATIONS),
        _position(cell.reader, readers),
        _position(cell.task.value, tasks),
    )


def _storage_order(key: StorageKey) -> tuple[str, int]:
    """Order storage figures the way the registry declares the representations.

    Args:
        key: The storage figure to place.

    Returns:
        The dataset, then the position of the representation.
    """
    return (key.dataset, _position(key.representation, REPRESENTATIONS))


def _position(name: str, axis: tuple[str, ...]) -> int:
    """Find where one name sits on a declared axis.

    Args:
        name: The name to place.
        axis: The declared axis, in report order.

    Returns:
        The position of the name, or the length of the axis when the axis does not declare it.
    """
    return axis.index(name) if name in axis else len(axis)
