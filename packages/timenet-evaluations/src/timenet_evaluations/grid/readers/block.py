"""The one precondition a block read has, shared by both readers so that neither can relax it.

SPEC-0018 owns the plan and the values in it. A reader is handed a block and does what this
capability specifies: it returns exactly the items the block names. A block that names items the
dataset did not declare names nothing a reader can return, so the run stops there.

Both readers of one representation walk the same plan. A plan that one reader refused and the other
clipped would make the two shuffled rates of one row rates over different workloads, so the test
and the message live here and each reader raises them.

This module holds a test and a message. It holds no exception factory and no base class. Each
reader writes ``raise EvaluationError(...)`` at its own call site, which is what keeps the harness's
own error type at the raise site and keeps the traceback pointing at the read that failed.
"""

from __future__ import annotations

from pathlib import Path

from timenet_evaluations.grid.cell import Cell
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.item import ItemDeclaration
from timenet_evaluations.grid.reader import Block


def names_declared_items(block: Block, declaration: ItemDeclaration) -> bool:
    """Report whether a block names only canonical items the dataset declared.

    A block that runs past the last declared item is refused by its reader. It is not clipped to
    the items that are there: a clipped block returns fewer items than its count, and the shuffled
    walk then covers the dataset with a hole in it that no number in the report shows.

    Args:
        block: The contiguous run of canonical items a read was handed.
        declaration: The dataset's canonical item declaration.

    Returns:
        True if the block starts at position zero or later, names one item or more, and ends at
        the last declared item or before it.
    """
    return block.start >= 0 and block.count >= 1 and block.start + block.count <= declaration.count


def block_failure_message(block: Block, declaration: ItemDeclaration, *, at: Cell, path: Path) -> str:
    """Build the message a reader raises when it refuses a block.

    Args:
        block: The block the reader refused.
        declaration: The dataset's canonical item declaration.
        at: The coordinates of the cell that refused the block.
        path: The path the reader was opened against.

    Returns:
        The message, which names the cell's four coordinates, the path, the declared count and the
        block.
    """
    return failure_message(
        "a block names canonical items the dataset did not declare",
        at=at,
        path=path,
        detail=f"declared count {declaration.count}, block start {block.start}, block count {block.count}",
    )
