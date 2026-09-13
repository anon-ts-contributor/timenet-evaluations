"""The PyTorch reader: tensors, delivered through a dataloader, from every representation.

This reader is a target in-memory form together with the loader that produces it. The form is
tensors. The dataset under the dataloader is not this module's: it arrives with the representation,
as the parsing path both readers of that representation share, and this module never names a
dataset library and never opens a container itself.

One canonical item in this target form is the tensor the dataloader yields at that position. The
values of one item are that item's signal and nothing else, so one item reduces to the comparison
form and equality against the Pandas reader's row group is a test rather than an argument.

That convention binds here, on the item this reader hands out, and not one step later.
``to_comparison_form`` does ``np.asarray`` and then an exact shape check. It unpacks no tuple and
drops no column. An item that arrived as a ``(signal, label)`` pair gets a refusal, and not a silent
strip, so this reader takes the batch axis off a tensor and hands out nothing else.

The dataloader configuration is fixed here and is identical on both representations:
``batch_size=1``, ``num_workers=0``, and no custom collate function. It is fixed because a
dataloader's configuration moves a rate in items per second, and a row that compared a tuned loader
against an untuned one would attribute the difference to the representation. ``batch_size=1`` is
what makes the collate function unnecessary: samples vary in shape, so a loader that batched them
would need a collate function, and that function would sit inside all eight timed PyTorch cells and
would have to be proved equivalent across representations before either column meant anything.

**This is not what a training loader does.** A tuned loader batches and prefetches across worker
processes, and both raise throughput. This reader measures per-item delivery through one process,
which is a floor and not an achievable rate. Every report that prints these numbers must say so.
The block walk's "what a training loader sees" framing holds for the ordering it imposes and must
not be claimed for the throughput it reports.

``open`` constructs the dataset and the dataloader, and ``open`` is inside the timer. Both are held
for the whole task, because one task opens once. No result is held: each read drives its own pass
over the dataloader, so a later call in the same task cannot be served from an earlier one.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, Task
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.reader import Block
from timenet_evaluations.grid.readers.block import block_failure_message, names_declared_items
from timenet_evaluations.grid.representation import Representation


PYTORCH = "pytorch"
"""This reader's own name. Every measurement this reader produces carries it."""

BATCH_SIZE = 1
"""One item per batch, on both representations. A larger batch needs a collate function, and that
function would sit inside every timed PyTorch cell."""

WORKER_COUNT = 0
"""No worker process, on both representations. The count belongs to this wrapper and not to the
dataset a representation supplies."""


def build_loader(dataset: Dataset[object]) -> DataLoader[object]:
    """Wrap a dataset in the one dataloader configuration this benchmark measures.

    This is the same wrapper on both representations and for every read. It takes no configuration
    of its own, so no caller can tune one cell against another.

    Args:
        dataset: The map-style dataset the representation's parsing path supplied, or a contiguous
            subset of it.

    Returns:
        The dataloader, at ``batch_size=1``, ``num_workers=0``, with the default collate function
        and in ascending position order.
    """
    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=WORKER_COUNT, shuffle=False)


class PyTorchOpened:
    """One representation, opened by the PyTorch reader. This is one cell of the grid.

    The object holds the dataset, the dataloader over it, and the coordinates a failure message
    needs. It holds no tensor and no list of tensors, so the block walk is one open and many reads
    rather than one read and many slices.

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
        self._items = representation.parsing_path.open_torch(self._path)
        self._loader = build_loader(self._items)

    def _at(self, task: Task) -> Cell:
        """Name the cell one read of this opened representation belongs to.

        Args:
            task: Which of the four read operations is running.

        Returns:
            The four coordinates of that cell.
        """
        return Cell(dataset=self._dataset, representation=self._representation, reader=self._reader, task=task)

    def read_first(self) -> torch.Tensor:
        """Read the canonical item at position zero of the declared item ordering.

        The pass over the dataloader stops after the first batch, which at ``batch_size=1`` is the
        item at position zero. The reader does not return whichever item the loader surfaces
        soonest: this task measures the latency to a known item.

        Returns:
            The item at position zero, as a tensor of the declared item shape.

        Raises:
            EvaluationError: If the dataloader yields no batch at all.
        """
        for batch in self._loader:
            return self._one(batch, Task.FIRST_ITEM)

        raise EvaluationError(
            failure_message(
                "the representation yielded no canonical item at position zero",
                at=self._at(Task.FIRST_ITEM),
                path=self._path,
                detail=f"declared count {self._declaration.count}",
            )
        )

    def read_all(self) -> list[torch.Tensor]:
        """Read every canonical item of the dataset in one bulk pass.

        Every item is materialized before this call returns. The result is a list of tensors and
        not a lazy handle, because a lazy return would make the timer measure an open instead of a
        read.

        Returns:
            Every canonical item, in ascending position order, as tensors of the declared item
            shape.
        """
        return [self._one(batch, Task.FULL_READ) for batch in self._loader]

    def read_sequential(self) -> Iterator[torch.Tensor]:
        """Walk the canonical items in ascending position order, one at a time.

        The walk drives its own pass over the dataloader and materializes each item before it
        yields it. It does not call ``read_all``, because a walk served from a bulk pass would
        measure the bulk pass and report it in items per second. This walk is the item-at-a-time
        read an adopter of this loader would write.

        Yields:
            Each canonical item, in ascending position order, as a tensor of the declared item
            shape.
        """
        for batch in self._loader:
            yield self._one(batch, Task.SEQUENTIAL)

    def read_block(self, block: Block) -> list[torch.Tensor]:
        """Read exactly the canonical items that one block names.

        The block's positions are driven through the same dataloader configuration, over a subset
        of the dataset this cell already holds. The dataset is not opened again: the subset and the
        dataloader over it are a sampler around a locator that is already there. A short final
        block is served the same way as every other block.

        Args:
            block: The contiguous run of canonical items to read.

        Returns:
            Exactly the items the block names, in ascending position order, as tensors.

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

        positions = range(block.start, block.start + block.count)
        loader = build_loader(Subset(self._items, positions))

        return [self._one(batch, Task.BLOCK_SHUFFLED) for batch in loader]

    def _one(self, batch: object, task: Task) -> torch.Tensor:
        """Take the one canonical item out of one batch.

        At ``batch_size=1`` a batch is one item that carries a batch axis of length one. The item
        is the batch without that axis, so what this reader hands back has the declared item shape
        and reduces to the comparison form the Pandas reader's row group reduces to.

        Args:
            batch: One batch, as the dataloader yielded it.
            task: Which of the four read operations yielded it, so a failure names its own cell.

        Returns:
            The batch's one canonical item.

        Raises:
            EvaluationError: If the batch is not a tensor, or if it does not carry exactly one
                canonical item.
        """
        if not isinstance(batch, torch.Tensor) or batch.shape[0] != BATCH_SIZE:
            raise EvaluationError(
                failure_message(
                    "a dataloader batch did not carry exactly one canonical item",
                    at=self._at(task),
                    path=self._path,
                    detail=f"batch size {BATCH_SIZE}, batch form {type(batch).__name__}",
                )
            )

        return batch[0]


class PyTorchReader:
    """The target form tensors, together with the loader each representation supplies.

    One reader per run, and it opens both representations. It is not pinned to one: a reader that
    read only the representation it was written for would collapse the grid to a diagonal, and no
    cell would remain in which the representation moves while the reader is held still.
    """

    name = PYTORCH

    def open(self, representation: Representation) -> PyTorchOpened:
        """Open one representation for reading.

        The call is inside the timed interval and it constructs the dataset and the dataloader,
        which is what a real adopter's loader does. The opened reader is discarded when the
        repetition ends.

        Args:
            representation: The representation to read, which this reader did not produce.

        Returns:
            The opened reader for this cell.
        """
        return PyTorchOpened(representation, self.name)
