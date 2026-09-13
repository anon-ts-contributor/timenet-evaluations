"""The Pandas reader: a ``DataFrame``, built from every representation this run holds.

This reader is a target in-memory form together with the loader that produces it. The form is a
``DataFrame``. The loader is not this module's: it arrives with the representation, as the parsing
path both readers of that representation share, and this module never names a dataset library and
never opens a container itself.

One canonical item in this target form is the contiguous group of rows at that item's position. A
``DataFrame`` that holds several items therefore holds their row groups end to end, in ascending
position order, and the item at position ``i`` is the rows from ``i * n_channels``. The values of
one item are that item's signal and nothing else, so one item reduces to the comparison form and
equality against the PyTorch reader's tensor is a test rather than an argument.

That convention binds here, on the item this reader hands out, and not one step later.
``to_comparison_form`` does ``np.asarray`` and then an exact shape check. It unpacks no tuple and
drops no column. An item that carried its label or its subject identifier gets a refusal, and not a
silent strip.

``open`` constructs the locator, and ``open`` is inside the timer. The locator is held for the
whole task, because one task opens once: a walk that reopened per item or per block would measure
this harness rather than the representation. No result is held. Every read materializes its own
``DataFrame`` and hands it back, so a later call in the same task cannot be served from an earlier
one.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, Task
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.reader import Block
from timenet_evaluations.grid.readers.block import block_failure_message, names_declared_items
from timenet_evaluations.grid.representation import Representation


PANDAS = "pandas"
"""This reader's own name. Every measurement this reader produces carries it."""


class PandasOpened:
    """One representation, opened by the Pandas reader. This is one cell of the grid.

    The object holds a locator and the coordinates a failure message needs. It holds no
    ``DataFrame``, no decoded item and no list of items, so the block walk is one open and many
    reads rather than one read and many slices.

    Three of the four coordinates are fixed when the cell opens. The fourth is the task, which each
    read names for itself, so a message says which of the four reads failed and not merely that
    this cell did.
    """

    def __init__(self, representation: Representation, reader: str) -> None:
        self._dataset = representation.dataset
        self._representation = representation.name
        self._reader = reader
        self._path = representation.artifact.path
        self._declaration = representation.item
        self._items = representation.parsing_path.open_pandas(self._path)

    def _at(self, task: Task) -> Cell:
        """Name the cell one read of this opened representation belongs to.

        Args:
            task: Which of the four read operations is running.

        Returns:
            The four coordinates of that cell.
        """
        return Cell(dataset=self._dataset, representation=self._representation, reader=self._reader, task=task)

    def read_first(self) -> pd.DataFrame:
        """Read the canonical item at position zero of the declared item ordering.

        The read asks the locator for position zero and for nothing else. Where the representation
        and its loader offer a narrow read, this is the narrow one. Where they do not, the loader
        reads what it must and this cell reports that number.

        Returns:
            The item at position zero, as a ``DataFrame`` of the declared item shape.
        """
        return self._one(self._items[0], Task.FIRST_ITEM)

    def read_all(self) -> pd.DataFrame:
        """Read every canonical item of the dataset in one bulk pass.

        The count comes from the dataset's declaration and never from the locator, so the bulk pass
        covers what the dataset says it holds. The result is fully materialized before it is
        returned: no lazy handle and no deferred view leaves this call.

        Returns:
            Every item's rows, in ascending position order, in one ``DataFrame``.
        """
        return self._many(self._items[0 : self._declaration.count], Task.FULL_READ)

    def read_sequential(self) -> Iterator[pd.DataFrame]:
        """Walk the canonical items in ascending position order, one at a time.

        The walk asks the locator for one position at a time and materializes each item before it
        yields it. It does not call ``read_all``, because a walk served from a bulk pass would
        measure the bulk pass and report it in items per second.

        Yields:
            Each canonical item, in ascending position order, as a ``DataFrame`` of the declared
            item shape.
        """
        for position in range(self._declaration.count):
            yield self._one(self._items[position], Task.SEQUENTIAL)

    def read_block(self, block: Block) -> pd.DataFrame:
        """Read exactly the canonical items that one block names.

        The block is served by one slice of the locator, which is the narrow read where the loader
        has one. A short final block is served the same way as every other block. Nothing outside
        the block is read, and nothing is padded to a uniform count.

        Args:
            block: The contiguous run of canonical items to read.

        Returns:
            The rows of exactly the items the block names, in ascending position order.

        Raises:
            EvaluationError: If the block names items the dataset did not declare.
        """
        if not names_declared_items(block, self._declaration):
            raise EvaluationError(
                block_failure_message(
                    block,
                    self._declaration,
                    at=self._at(Task.BLOCK_SHUFFLED),
                    path=self._path,
                )
            )

        return self._many(self._items[block.start : block.start + block.count], Task.BLOCK_SHUFFLED)

    def _one(self, values: object, task: Task) -> pd.DataFrame:
        """Build the target form of one canonical item.

        Args:
            values: One item's values, as the parsing path handed them back.
            task: Which of the four read operations asked for it, so a failure names its own cell.

        Returns:
            The item as a ``DataFrame`` whose values are the declared item shape.

        Raises:
            EvaluationError: If the values do not become a ``DataFrame`` of the declared shape.
        """
        try:
            return pd.DataFrame(np.asarray(values, dtype=np.float32))
        except Exception as error:
            raise EvaluationError(
                failure_message(
                    "a canonical item did not become a DataFrame",
                    at=self._at(task),
                    path=self._path,
                    detail=f"declared shape {self._declaration.shape}: {error}",
                )
            ) from error

    def _many(self, values: object, task: Task) -> pd.DataFrame:
        """Build the target form of a run of canonical items.

        The items keep the order the parsing path gave them, and one item's rows stay together.
        The row at index ``i * n_channels`` is therefore the first row of the item at position
        ``i`` of the run.

        Args:
            values: The values of a run of items, as the parsing path handed them back.
            task: Which of the four read operations asked for them, so a failure names its own
                cell.

        Returns:
            The items' rows end to end, in ascending position order, in one ``DataFrame``.

        Raises:
            EvaluationError: If the values do not become a ``DataFrame`` of stacked item shapes.
        """
        try:
            run = np.asarray(values, dtype=np.float32)
            return pd.DataFrame(run.reshape(-1, run.shape[-1]))
        except Exception as error:
            raise EvaluationError(
                failure_message(
                    "a run of canonical items did not become a DataFrame",
                    at=self._at(task),
                    path=self._path,
                    detail=f"declared shape {self._declaration.shape}: {error}",
                )
            ) from error


class PandasReader:
    """The target form ``DataFrame``, together with the loader each representation supplies.

    One reader per run, and it opens both representations. It is not pinned to one: a reader that
    read only the representation it was written for would collapse the grid to a diagonal, and no
    cell would remain in which the representation moves while the reader is held still.
    """

    name = PANDAS

    def open(self, representation: Representation) -> PandasOpened:
        """Open one representation for reading.

        The call is inside the timed interval and it constructs the locator, which is what a real
        adopter's loader does and what makes the narrow reads narrow. The opened reader is
        discarded when the repetition ends.

        Args:
            representation: The representation to read, which this reader did not produce.

        Returns:
            The opened reader for this cell.
        """
        return PandasOpened(representation, self.name)
