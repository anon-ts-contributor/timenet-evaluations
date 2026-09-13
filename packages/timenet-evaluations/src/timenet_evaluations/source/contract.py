"""The frame contract: the three columns a connector produces, and the helpers that build them.

A connector reads one dataset into memory and returns one ``pandas.DataFrame``. This module holds
what that frame is, apart from what any dataset is. Nothing here names a dataset or imports a
dataset library, so the second connector inherits this file rather than copies it.

The frame carries exactly three columns and nothing else. A fourth column would be visible to the
conversion, which was not written to expect one, and it would travel into the converted
representation while having no counterpart in the release's own files.

Every caller reaches a column through ``SIGNAL``, ``LABEL`` or ``PATIENT``. A caller that writes the
string literal makes a second copy of the contract, and no rename finds that copy.

Uniform item shape is assumed here and is not enforced. Every item is assumed to carry the same
``(n_channels, n_samples)`` shape. Neither :func:`as_row` nor :func:`unbatch` compares one item's
shape against another's, so a dataset whose items disagree on the sample count passes the load. It
surfaces later, and the places are worth naming because nothing in the frame shows the problem:

- The load returns normally. The frame looks like any other frame.
- :func:`signal_stack` fails, because one dense array cannot be built from rows that disagree on
  shape. The error comes from numpy and names neither the position nor the dataset.
- The conversion fails for the same reason. It validates row by row, so its message does name the
  offending position and the shape it found, and the directories it already made stay on disk.

All of that is untimed work, so a ragged dataset costs a run and never a number. A connector for a
dataset whose items are not uniform must widen this assumption first. A pad or a reshape inside the
conversion is not the answer: the converted representation would then hold content that the
release's own files do not, and the parity check would be the thing that failed rather than the
thing that explained why.

The channel count is the one part of that shape a connector must not leave to the path above. A
dataset library that collates a batch pads the first axis of the values it stacks, the first axis of
one item is the channel axis, and a channel of zeros added there makes items agree on a shape they
never had. Nothing downstream then shows a problem: the frame is uniform, the conversion succeeds,
both representations carry the fabricated channel, and the parity check compares it against itself.
So a connector reads its items one at a time and refuses a channel count that changes between them,
and the connector that ships does exactly that.

Nothing in this package absorbs a failure. Loading crosses two fallible boundaries, the filesystem
and a third-party dataset reader, and neither is wrapped. There is no ``except`` here: a parse
error, a malformed recording or a filesystem error reaches the caller with its own type and its own
traceback, and there is no retry, no fallback and no partial frame. A connector must not skip an
unreadable recording and continue, because a frame that is silently missing a subject's items
produces a converted artifact that disagrees with the release's files at positions the parity check
does not sample. The checks below are this capability's own preconditions, so they raise
``EvaluationError``. A reader of the traceback then sees the suite refusing rather than a dependency
failing.
"""

# ancestor of three artifacts)

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from timenet_evaluations.errors import EvaluationError


SIGNAL = "signal"
"""Column holding one canonical item's values, shaped ``(n_channels, n_samples)`` as ``float32``."""
LABEL = "label"
"""Column holding the scored class of that item, as ``str``."""
PATIENT = "patient_id"
"""Column holding the subject the item came from, as ``str``."""

COLUMNS = (SIGNAL, LABEL, PATIENT)
"""The three columns the frame carries. A frame has these and no others."""

SIGNAL_DIMENSIONS = 3
"""How many dimensions the stacked signal column has: items, channels, and samples."""


def check_source_directory(source: Path) -> None:
    """Make sure that a source path is a directory that exists.

    Call this before a dataset library is touched. The test is a directory test and not an existence
    test. A path that exists as a regular file is refused with the same failure as a path that is
    absent, because a connector needs a directory of release files and one file cannot supply one.

    A run stops here rather than continues with an empty frame. An empty frame, or a frame with no
    columns, is a value a caller can mistake for a good load of an empty dataset.

    The source directory is also the artifact of the release's own files, and this package never
    writes below it.

    Args:
        source: The directory a connector was given.

    Raises:
        EvaluationError: If ``source`` is not a directory that exists.
    """
    if not source.is_dir():
        raise EvaluationError(f"source directory does not exist: {source}")


def as_row(item: Mapping[str, Any]) -> dict[str, Any]:
    """Build one frame row from one item of a dataset.

    A connector that reads its items one at a time calls this for each item it reads. The row
    carries the three contract columns and nothing else, and this function is where the ``float32``
    cast and the two ``str`` coercions happen. A caller that assembled the dict itself would make a
    second copy of the contract, and no rename finds that copy.

    A value arrives as a framework tensor or as a plain value, and this function converts it without
    knowledge of which. A dataset library can change the type between versions, and that variation
    must not reach the frame.

    A column that is absent is refused by name. No column gets a default value: a fabricated subject
    identifier satisfies the type of the contract while it carries no subject at all.

    Args:
        item: One item as the dataset yields it. An item can carry more keys than the contract
            names, and the keys the contract does not name are dropped.

    Returns:
        One row, which holds ``signal`` as ``float32``, ``label`` as ``str`` and ``patient_id`` as
        ``str``.

    Raises:
        EvaluationError: If the item does not have one of the three columns.
    """
    missing = [column for column in COLUMNS if column not in item]
    if missing:
        raise EvaluationError(
            f"item is missing a column the frame contract requires: missing {missing}, present {sorted(item)}"
        )

    return {
        SIGNAL: np.asarray(_to_numpy(item[SIGNAL]), dtype=np.float32),
        LABEL: str(_as_scalar(item[LABEL])),
        PATIENT: str(_as_scalar(item[PATIENT])),
    }


def unbatch(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one collated batch into one row per canonical item.

    A dataloader collates a batch into one dict of stacked columns. The frame holds one row per
    canonical item, so the columns are zipped apart again. Each row is built by :func:`as_row`, so
    the row carries the three contract columns and nothing else and one function holds what a row
    is.

    A column arrives as a framework tensor or as a plain sequence, and this function converts it
    without knowledge of which. A dataset library can collate differently between versions, and that
    variation must not reach the frame.

    A column that is absent is refused by name. No column gets a default value: a fabricated subject
    identifier satisfies the type of the contract while it carries no subject at all, and a default
    built at the length of another column disarms the arm of the item-count check that it feeds.

    Once the counts agree, the rows are built with a strict zip. A defect in the count check then
    raises rather than drops items.

    Args:
        batch: One batch as the dataloader yields it.

    Returns:
        One dict per canonical item in the batch, each holding ``signal`` as ``float32``, ``label``
        as ``str`` and ``patient_id`` as ``str``.

    Raises:
        EvaluationError: If the batch omits one of the three columns, or if its columns disagree on
            how many items they hold.
    """
    missing = [column for column in COLUMNS if column not in batch]
    if missing:
        raise EvaluationError(
            f"batch is missing a column the frame contract requires: missing {missing}, present {sorted(batch)}"
        )

    signals = _to_numpy(batch[SIGNAL])
    labels = _as_list(batch[LABEL])
    patients = _as_list(batch[PATIENT])

    if not len(signals) == len(labels) == len(patients):
        raise EvaluationError(
            f"batch columns disagree on item count: "
            f"{SIGNAL}={len(signals)}, {LABEL}={len(labels)}, {PATIENT}={len(patients)}"
        )

    return [
        as_row({SIGNAL: signal, LABEL: label, PATIENT: patient})
        for signal, label, patient in zip(signals, labels, patients, strict=True)
    ]


def signal_stack(frame: pd.DataFrame) -> np.ndarray:
    """Stack the frame's signal column into one array.

    This is the single place the per-row object column becomes a dense array, so the shape and the
    dtype of that array are decided once. A caller that needs the whole dataset as one array calls
    this rather than stacks the column itself, and it adds no cast of its own.

    The result is ``float32`` whatever the dtype of the column is. The cast uses ``copy=False``,
    which matters because this array is the largest single allocation a load makes.

    A caller that must refuse a ragged frame by naming the offending row does not use this function.
    One array cannot be built at all from rows that disagree on shape, so this function raises a
    numpy error that names neither the position nor the shape found. Such a caller validates row by
    row and can stack afterwards. The merged conversion does exactly this.

    Args:
        frame: A frame that satisfies the contract of this module.

    Returns:
        The signals, shaped ``(n_items, n_channels, n_samples)``, as ``float32``.

    Raises:
        EvaluationError: If the stacked signals do not have three dimensions. The shape is promised
            to every caller and is the reference the parity check compares against, and ``np.stack``
            builds a two-dimensional result from one-dimensional rows without complaint.
    """
    stacked = np.stack(frame[SIGNAL].to_numpy()).astype(np.float32, copy=False)

    if stacked.ndim != SIGNAL_DIMENSIONS:
        raise EvaluationError(
            f"the stacked signal column does not have {SIGNAL_DIMENSIONS} dimensions: "
            f"expected (n_items, n_channels, n_samples), got shape {stacked.shape}"
        )

    return stacked


def _to_numpy(values: Any) -> np.ndarray:
    """Convert a batch column to numpy, whether it arrived as a tensor or an array.

    Args:
        values: One collated column.

    Returns:
        The column as a numpy array.
    """
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()

    return np.asarray(values)


def _as_scalar(value: Any) -> Any:
    """Reduce one item's value to a plain Python value, whether or not it arrived as a tensor.

    Args:
        value: One value of one item.

    Returns:
        The value as a plain Python value.
    """
    if hasattr(value, "item"):
        return value.item()

    return value


def _as_list(values: Any) -> list[Any]:
    """Convert a batch column to a plain list.

    Args:
        values: One collated column.

    Returns:
        The column as a list, one entry per canonical item.
    """
    if hasattr(values, "tolist"):
        return list(values.tolist())

    return list(values)
