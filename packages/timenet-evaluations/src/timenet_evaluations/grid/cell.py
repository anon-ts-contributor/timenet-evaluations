"""The coordinates of one timed cell, and the coordinates of the two smaller scopes.

A timed cell is one number a run measures, and four coordinates name it: the dataset, the
representation, the reader, and the task. A key of fewer coordinates cannot name a cell of this
grid, so the arity is the check. A two-coordinate key, of the kind the superseded model used, means
that a reader has been pinned to a representation somewhere upstream.

An opened reader carries three coordinates and never a task coordinate. One open serves all four
tasks, so an open happens before a task is chosen, and a message that named a task would name one
that had not started.

A storage figure carries two coordinates and never a reader coordinate. A reader does not change
bytes on disk, and a result that held one storage figure per (representation, reader) pair would
say that it does.

Each of the three scopes has a type of its own, and a failure passes the type of the scope it is
in. A scope with no type must write its coordinates into free text, where nothing holds them in one
place and nothing can find them again.

This module names the four tasks and holds nothing else about them. It states no unit, times
nothing, and carries no number. The read-measurement capability owns the timing protocol and the
value of every number a cell reports.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Task(StrEnum):
    """One of the four read operations that every cell of the grid is measured on.

    Every cell is measured on all four. A run that measured fewer tasks for one cell than for
    another produces a row whose columns a reader cannot compare, and the missing tasks are exactly
    the ones a reader would want.

    A conversion is not a task. It runs once before the first timer starts, so it has no member
    here and no column in any report.
    """

    FIRST_ITEM = "first_item"
    """``read_first``: the canonical item at position zero, from the open to that item."""

    FULL_READ = "full_read"
    """``read_all``: every canonical item, in one bulk pass."""

    SEQUENTIAL = "sequential"
    """``read_sequential``: the canonical items in ascending position order, one at a time."""

    BLOCK_SHUFFLED = "block_shuffled"
    """``read_block``: one complete pass over the block plan, in plan order."""


class Cell(BaseModel):
    """The four coordinates of one timed cell.

    The four names travel together, so no number can be read as an answer to a question it did not
    ask. The model is frozen, which makes it usable as the key of the mapping a run collects its
    numbers into.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str
    """The dataset this cell measures. A run takes more than one, so a message that omits the
    dataset cannot say which grid the cell belongs to."""
    representation: str
    """The representation this cell reads, ``original`` or ``timef``."""
    reader: str
    """The reader that reads it, ``pandas`` or ``pytorch``. Every reader reads every
    representation, so this coordinate moves independently of the one above it."""
    task: Task
    """Which of the four read operations this cell was measured on."""


class OpenKey(BaseModel):
    """The three coordinates of one opened reader.

    An open builds a parsing path, and that path serves all four tasks. A failure in it therefore
    names a reader and no task. The task coordinate is left out and not filled with a placeholder,
    because no task had started.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str
    """The dataset the opened reader reads."""
    representation: str
    """The representation it opened, ``original`` or ``timef``."""
    reader: str
    """The reader that opened it, ``pandas`` or ``pytorch``."""


class StorageKey(BaseModel):
    """The two coordinates of one storage figure.

    There are exactly two storage figures per dataset, one per representation. Neither is indexed
    by a reader, and the arity of this key is what makes that checkable rather than argued.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str
    """The dataset whose bytes on disk this figure describes."""
    representation: str
    """The representation those bytes are, ``original`` or ``timef``."""
