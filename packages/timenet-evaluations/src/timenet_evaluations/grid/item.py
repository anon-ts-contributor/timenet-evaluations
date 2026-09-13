"""The canonical item: the unit a rate counts, and the form both readers reduce it to.

The canonical item is the smallest unit the dataset's task consumes. A dataset declares it in
advance of any measurement, and it is a property of the dataset. It is not a property of a
representation, of a reader, or of a stored layout. No code path here lets a representation or a
reader announce what it can conveniently yield and accepts that as the unit.

The two readers hold one item in two different target forms. Pandas holds the contiguous group of
rows at that item's position in a ``DataFrame``. PyTorch holds the tensor, or the tuple of tensors,
that the dataloader yields at that position. Both forms reduce to one comparison form, so that
"the same item" is a test and not an argument.

The reduction is benchmark scaffolding. It runs outside every timer, because it must not compete
with the thing under test.

A rate in ``items/s`` is computed from the declared count and never from a count a representation
or a reader reports about itself. Such a rate does not compare across datasets, because the
canonical item differs per dataset. Every report that prints one says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, OpenKey
from timenet_evaluations.grid.failure import failure_message


type Item = object
"""One canonical item in a reader's target form. The form differs per reader, so the seam names
the unit and not the type: a row group of a ``DataFrame`` for one reader, a tensor for the other."""

type Bulk = object
"""Every canonical item of a dataset in a reader's target form, materialized in one bulk pass."""

type Items = object
"""The canonical items that one block names, in a reader's target form, in ascending position
order."""


class ItemDeclaration(BaseModel):
    """What one canonical item of a dataset is, and how many of them there are.

    A dataset declares this before any measurement, and it travels with the dataset. This package
    holds no default for it: a dataset arrives with its files, its parsing path, and this
    declaration.
    """

    count: int
    """How many canonical items the dataset holds. Every one of the four cells must yield exactly
    this many, and every rate divides by this number."""
    shape: tuple[int, ...]
    """The shape of one item's values in the comparison form."""


def to_comparison_form(values: object, declaration: ItemDeclaration, *, at: Cell | OpenKey, path: Path) -> np.ndarray:
    """Reduce one canonical item to the form the four cells are compared in.

    The comparison form is the item's values as a ``float32`` array of the dataset's declared item
    shape. Both target forms reduce to it, so equality across the four cells of one dataset is an
    assertion with zero tolerance.

    Run this outside every timed region.

    Args:
        values: One item's values, in the reader's target form. Anything that numpy can read as an
            array is accepted, which covers a tensor and a group of rows. A target form that refuses
            to become an array raises through this function and not past it: the libraries behind
            the two target forms raise more than ``TypeError`` and ``ValueError``, and a tensor that
            holds a gradient raises ``RuntimeError``.
        declaration: The dataset's canonical item declaration.
        at: The coordinates of the scope the item was read in. A ``Cell`` names all four, which
            is what a timed read has. An ``OpenKey`` names the three an open has, which is what
            the untimed parity check has: it runs before a task starts, so a task coordinate
            would name one that had not begun.
        path: The path the reader was opened against.

    Returns:
        The item's values as a ``float32`` array of the declared item shape.

    Raises:
        EvaluationError: If the values do not reduce to a ``float32`` array, or if the array does
            not have the declared item shape.
    """
    try:
        array = np.asarray(values, dtype=np.float32)
    except Exception as error:
        raise EvaluationError(
            failure_message(
                "a canonical item did not reduce to a float32 array",
                at=at,
                path=path,
                detail=f"target form {type(values).__name__}: {error}",
            )
        ) from error

    if array.shape != declaration.shape:
        raise EvaluationError(
            failure_message(
                "a canonical item does not have the declared item shape",
                at=at,
                path=path,
                detail=f"declared shape {declaration.shape}, read shape {array.shape}",
            )
        )

    return array


def check_item_count(counted: int, declaration: ItemDeclaration, *, at: Cell | OpenKey, path: Path) -> None:
    """Make sure that a read yielded exactly the declared number of canonical items.

    A cell that yields a different number counts a different object from the other three cells of
    that dataset. The four rates then wear one unit over different work, and nothing in the table
    shows it. The run stops instead.

    Args:
        counted: How many canonical items the read yielded.
        declaration: The dataset's canonical item declaration.
        at: The coordinates of the scope that was read. A ``Cell`` names all four, which is what
            a timed read has. An ``OpenKey`` names the three an open has, which is what the
            untimed parity check has.
        path: The path the reader was opened against.

    Raises:
        EvaluationError: If the counted number is not the declared number.
    """
    if counted != declaration.count:
        raise EvaluationError(
            failure_message(
                "a read yielded a number of canonical items that the dataset did not declare",
                at=at,
                path=path,
                detail=f"declared count {declaration.count}, yielded count {counted}",
            )
        )
