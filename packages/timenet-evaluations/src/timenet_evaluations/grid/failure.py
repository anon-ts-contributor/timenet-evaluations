"""How the grid reports a failure at the filesystem boundary.

Every read and every conversion touches the filesystem, and every filesystem operation can fail.
A failure stops the run. Nothing in this package catches a failure and continues, substitutes a
default value or an empty result for a failed read, or lets a run write a report that looks
complete when one cell did not complete. A measurement that cannot be trusted is worse than no
measurement.

This module owns the message contract, not the raise sites. Every raise site raises
``EvaluationError`` and builds its message here, so that every message names the same values: the
cell's own coordinates and the path. A reader of the traceback then knows which cell was at fault
and does not have to run the benchmark again to find out.

The coordinates arrive as one value and not as one keyword argument each. A cell is its
coordinates, so the thing itself is what a raise site passes: a timed read passes a ``Cell``, which
names all four; an open passes an ``OpenKey``, which names the three it has; and a conversion or a
size passes a ``StorageKey``, which names the two it has. A coordinate that does not apply is left
out, never filled with a placeholder. Every scope that occurs has a type, so no raise site has to
write a coordinate into ``detail``.

A raise site can translate a failure from another package. It must not absorb one. Where the code
catches an error, it raises ``EvaluationError`` from that error and loses nothing.
"""

from __future__ import annotations

from pathlib import Path

from timenet_evaluations.grid.cell import Cell, OpenKey, StorageKey


def failure_message(problem: str, *, at: Cell | OpenKey | StorageKey, path: Path, detail: str | None = None) -> str:
    """Build the message of a failed read or a failed conversion.

    The caller writes ``raise EvaluationError(failure_message(...))``, so that every raise site
    names the harness's own error type and the traceback points at the operation that failed.

    Args:
        problem: What went wrong, in one phrase and with no trailing punctuation.
        at: The coordinates of the scope the failure belongs to. A ``Cell`` names the dataset, the
            representation, the reader and the task. An ``OpenKey`` names the first three, which is
            what an open has. A ``StorageKey`` names the first two, which is what a conversion and a
            size have.
        path: The file or directory the failed operation was given.
        detail: The values behind the failure, where the problem alone does not name them.

    Returns:
        The message, which names every coordinate the operation has and then the path.
    """
    parts = [f"dataset {at.dataset!r}", f"representation {at.representation!r}"]
    if isinstance(at, Cell | OpenKey):
        parts.append(f"reader {at.reader!r}")
    if isinstance(at, Cell):
        parts.append(f"task {at.task.value!r}")
    parts.append(f"path {path}")
    if detail is not None:
        parts.append(detail)

    return f"{problem}: {', '.join(parts)}"
