"""The reader seam: two structural protocols with two different lifetimes.

A reader is a target in-memory form together with the loader that produces it. ``Reader`` is one
per run. ``Opened`` is one per cell, where a cell is one (representation, reader) pair of one
dataset. Every reader opens every representation.

Both protocols are structural. An implementation declares the members with matching signatures. It
does not import a protocol, subclass one, or register itself anywhere. Neither protocol carries a
concrete method body, because a body gives an implementation something to inherit and turns a
structural seam into a nominal one. The two readers share a shape and no behaviour.

``ty`` is the whole gate. Neither protocol is runtime checkable, so no ``isinstance`` guard exists.
The mechanism is the annotation on the subject registry, ``READERS: tuple[Reader, ...]``. Do not
widen that annotation to ``tuple[Any, ...]`` and do not drop it: two unrelated classes in a tuple
type-check as their own union, and then nothing is verified. A reader whose signature drifted is a
type-check diagnostic and not a registration error, so a run started without ``ty`` reaches an
``AttributeError`` in the middle of the benchmark.

``open`` is inside the timer, and one task opens exactly once. A timed repetition begins before the
representation is opened and ends when the task's result stands in the reader's target form.
Opening is part of what a cell measures.

``open`` can do what the reader's real loader does. A parse at construction is permitted, because
the timed interval starts before the open. Such a parse is paid in full, in every repetition, in the
cell it belongs to. An earlier rule here forbade it, on the grounds that it moved the parse out of
every timed cell. That reasoning depended on ``open`` being untimed, and ``open`` is not untimed.

Nothing survives the repetition. The opened reader is discarded when the repetition ends, and the
next repetition opens again and pays for it again. No decoded item, built ``DataFrame``,
materialized tensor, file handle, or memory map passes from one repetition to the next, or from one
task to another.

Within one repetition the line falls between a locator and a result. A locator is a file handle, a
Parquet footer, a constructed reference loader, or a row-group index. An opened reader can hold a
locator across the calls of one task, and that is what makes a narrow read narrow. It is also what a
real adopter's loader holds, so it is part of what the task measures.

A result is a materialized ``DataFrame``, a stacked tensor, or a decoded list of items. An opened
reader must not hold a result, and must not serve a later call of the same task from one. This is
the rule the block walk turns on: one repetition is one ``open`` and many ``read_block`` calls. A
representation materialized once and sliced N times reports one read and N slices, under a heading
that says shuffled rate. The repetition boundary does not catch that, because it happens inside one
repetition.

The anti-cheat therefore sits at two places and not at one: the repetition boundary for everything,
and the locator/result line within one task.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from timenet_evaluations.grid.item import Bulk, Item, Items
from timenet_evaluations.grid.representation import Representation


class Block(Protocol):
    """A contiguous run of canonical items, named by a start position and a count.

    The read-measurement capability owns the value and the plan that partitions the items. This
    seam states only the shape a block read is handed, so that the value satisfies it structurally
    and this package plans nothing.
    """

    @property
    def start(self) -> int:
        """The position of the block's first canonical item.

        Returns:
            The start position, counted from zero in the dataset's declared item ordering.
        """
        ...

    @property
    def count(self) -> int:
        """How many canonical items the block names.

        Returns:
            The count, which is at least one. The final block of a plan can be shorter than the
            others.
        """
        ...


class Opened(Protocol):
    """One representation, opened by one reader. This is one cell of the grid.

    An opened reader holds a locator: a path, a name, an item declaration, and what the loader
    built when it opened. It must not hold a result. An opened reader that materializes the
    representation once, and then serves later calls from what it kept, reports a rate that
    describes no representation.
    """

    def read_first(self) -> Item:
        """Read the canonical item at position zero of the declared item ordering.

        The values equal the first item of ``read_all`` exactly. A reader does not define its own
        notion of first, because this task measures the latency to a known item.

        This read does not have to read the whole representation. Where the representation and the
        loader together offer a narrower read, use it. Where they do not, read everything and
        return the first item, and report that number without apology.

        Returns:
            The canonical item at position zero, in this reader's target form.
        """
        ...

    def read_all(self) -> Bulk:
        """Read every canonical item of the dataset in one bulk pass.

        The result is fully materialized in this reader's target form, holds the declared item
        count in ascending position order, and equals the representation's content with zero
        tolerance. A partial read, a lazy handle, a deferred memory-mapped view, or a subset of the
        items makes the timer measure an open instead of a read.

        This task and the sequential walk are separate tasks. Neither is derived from the other.

        Returns:
            Every canonical item, in this reader's target form.
        """
        ...

    def read_sequential(self) -> Iterator[Item]:
        """Walk the canonical items in ascending position order, one at a time.

        The walk yields exactly the declared item count and materializes each item before it
        yields it. It does not reorder, coalesce, skip, or repeat items, and it is not served by
        materializing everything and then iterating over the result.

        The sequential rate bounds the block-shuffled rate from above.

        Returns:
            An iterator over the canonical items, in this reader's target form.
        """
        ...

    def read_block(self, block: Block) -> Items:
        """Read exactly the canonical items that one block names.

        The items come back in ascending position order and equal the items at those positions of
        this cell's full read. This read does not read outside the block, return more items than
        the count, or reorder, coalesce, or skip items inside the block. A short final block is
        served like any other.

        Where the representation and the loader together permit a narrower read, use it. Where they
        do not, read everything and cut, and report that cost as this cell's.

        Both readers of one representation walk the same plan, because the block size follows from
        bytes per item and bytes per item is a property of what is on disk.

        Args:
            block: The contiguous run of canonical items to read.

        Returns:
            Exactly the items the block names, in this reader's target form.
        """
        ...


class Reader(Protocol):
    """A target in-memory form together with the loader that produces it.

    There are two readers, and each one opens both representations. A reader pinned to one
    representation collapses the grid to a diagonal, and no cell then remains in which the
    representation moves while the reader is held still.
    """

    name: str
    """The reader's own name, ``pandas`` or ``pytorch``. Every measurement carries it."""

    def open(self, representation: Representation) -> Opened:
        """Open one representation for reading.

        This call binds a path, a name, and an item declaration. It decodes nothing, opens no
        container, and constructs no reference loader.

        Args:
            representation: The representation to read, which this reader did not produce.

        Returns:
            The opened reader for this cell, which is discarded when the repetition ends.
        """
        ...
