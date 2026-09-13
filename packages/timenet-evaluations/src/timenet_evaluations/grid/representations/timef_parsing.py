"""The parsing path of the converted representation: its own reader, in both target forms.

Both readers of this representation parse through this one object, and both parse through the
storage library's own reader. A read that opened the stored Parquet directly would measure a path
this project built rather than what a user of the format gets.

Neither open constructs a dataloader and neither exposes a worker count. That wrapper belongs to
the harness, and it is the same wrapper on both representations.

An open is inside the timed interval, so each open does what the library's own loader does. It
builds a locator — a reader over the version, and the control-plane rows that locate a sample's
values — and the values themselves load when a read asks for them. A locator may be held for the
calls of one task. A result may not, so nothing here keeps what a read decoded.

The narrow read is the library's, not this project's. ``TimeFReader.iter_samples`` filters on the
stored id column and prunes the row groups that cannot hold a requested id, so a block reads the
items it names instead of the whole version. Two things this depends on, both stated here because
neither is enforced: the conversion writes zero-padded sample ids, so the stored order the library
returns is ascending position order; and a position maps to an id by that same convention. The
converted form's layout is not chosen here — this uses the pruning the library already has.

A parsing path serves all four tasks and is built once per open, before any of them names itself.
A failure here therefore has a dataset, a representation and a reader, and no task. That is the
scope ``OpenKey`` names, and each message below passes one. The reader is a coordinate and it goes
where the other coordinates go: ``detail`` carries the values behind a failure and never a
coordinate.

``_version`` is the one call here that cannot name its own reader, because both opens share it. It
is given the ``OpenKey`` of the open that called it, which knows.

``timenet`` is the ``timef`` extra and not a dependency, so it is absent from the environment
continuous integration builds. Each import below carries a suppression for that reason. Import this
module only when ``find_spec("timenet")`` has already found the package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from timenet.reader import TimeFReader  # ty: ignore[unresolved-import]
from timenet.registry import DatasetVersion  # ty: ignore[unresolved-import]
from timenet.torch import TimeFTorchDataset  # ty: ignore[unresolved-import]
import torch
from torch.utils.data import Dataset

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import OpenKey
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.parsing import ParsedItems
from timenet_evaluations.grid.representations.timef import TimeF, sample_id


class TimeFParsingPath:
    """The one loader both readers of the converted representation parse through.

    Both target forms stack one item's values in the order the storage library returns the sample's
    series, so the two cells of this row hold the same rows in the same order. The library's torch
    item carries the series without their channel names, so a reorder by channel name is available
    to one target form and not to the other, and one that reordered only the form that can would
    make the two cells disagree. Whether the stored order matches the release's channel order is
    the parity check's question, and that check belongs to the dataset rather than here.

    The channel count arrives here, so an item that does not hold the dataset's channels stops the
    run instead of returning the wrong number of rows.
    """

    def __init__(self, *, dataset: str, channels: tuple[str, ...]) -> None:
        """Describe how one dataset's converted form is parsed.

        Args:
            dataset: The name of the dataset. It names the failing cell in every message this
                parsing path raises.
            channels: The dataset's channel names, as the conversion was given them. One item must
                hold this many series.
        """
        self._dataset = dataset
        self._channels = channels

    def open_pandas(self, path: Path) -> ParsedItems:
        """Open the converted form for the Pandas reader.

        A version that is not there stops the run at this call, through the same refusal the
        PyTorch open uses.

        Args:
            path: The version directory the conversion produced.

        Returns:
            Positional access to the canonical items, narrow where the library's pruning allows.
        """
        at = OpenKey(dataset=self._dataset, representation=TimeF.name, reader="pandas")

        return TimeFItems(_version(path, at=at), path=path, at=at, channels=self._channels)

    def open_torch(self, path: Path) -> Dataset[object]:
        """Open the converted form for the PyTorch reader.

        The library's own torch view is map-style already. Its item is a dictionary holding the
        sample's series, its id, its tasks and its annotations, so this call passes the library's
        own transform hook to reduce that item to one item's signal. The reduction is the
        library's extension point and not a second reader written over its files.

        Args:
            path: The version directory the conversion produced.

        Returns:
            A map-style torch dataset whose item at a position is that item's signal, as a
            ``float32`` tensor of the declared item shape.

        Raises:
            EvaluationError: If the version cannot be opened.
        """
        at = OpenKey(dataset=self._dataset, representation=TimeF.name, reader="pytorch")
        reader = TimeFReader(_version(path, at=at))
        try:
            # `read()` builds the control plane and leaves each series' values behind a loader that
            # holds the reader, so the values load when an item is asked for and the reader stays
            # alive for as long as the dataset does.
            return TimeFTorchDataset(reader.read(), transform=_item_tensor)
        except Exception as error:
            raise EvaluationError(
                failure_message(
                    "the converted form cannot be opened for the PyTorch reader",
                    at=at,
                    path=path,
                    detail=f"{error}",
                )
            ) from error


class TimeFItems:
    """Positional access to the canonical items of one opened converted version.

    This is a locator. It finds items through the library's reader and keeps none of what it read,
    so a later call in the same task pays for its own read.
    """

    def __init__(self, version: DatasetVersion, *, path: Path, at: OpenKey, channels: tuple[str, ...]) -> None:
        """Bind one opened version.

        Building the reader reads nothing. It resolves the control plane and a series' values when
        a read asks for them, which is what makes this a locator rather than a result.

        Args:
            version: The opened version of the converted form.
            path: The version directory, which names a failure.
            at: The coordinates of the open this locator belongs to, which name a failure.
            channels: The channel names, as the conversion was given them.
        """
        self._reader = TimeFReader(version)
        self._count = int(version.manifest.counts.samples)
        self._path = path
        self._at = at
        self._channels = channels

    def __len__(self) -> int:
        """Report how many canonical items the converted form holds.

        The number comes from the manifest the version already parsed, so this costs no scan. No
        rate divides by it: a reader drives every read from the dataset's declared count.

        Returns:
            The number of items the converted form holds.
        """
        return self._count

    def __getitem__(self, index: int | slice) -> np.ndarray:
        """Read one canonical item, or the contiguous run of items a slice names.

        Args:
            index: The position of one item, or the slice naming a contiguous run.

        Returns:
            One item's values for an integer index, shaped as the item is. For a slice, the items'
            values in ascending position order, with position as the leading axis.

        Raises:
            EvaluationError: If the index names items the converted form does not hold, or if the
                values cannot be read.
        """
        positions = self._positions(index)
        wanted = [sample_id(position) for position in positions]

        try:
            # The library filters on the stored id column and prunes row groups, so this reads the
            # items named and not the version. It returns them in stored order, which the
            # conversion's zero-padded ids make ascending position order.
            samples = list(self._reader.iter_samples(wanted))
        except Exception as error:
            raise EvaluationError(
                failure_message(
                    "the converted form cannot be read",
                    at=self._at,
                    path=self._path,
                    detail=f"positions {positions.start}..{positions.stop}: {error}",
                )
            ) from error

        if len(samples) != len(wanted):
            raise EvaluationError(
                failure_message(
                    "the converted form served fewer items than were asked for",
                    at=self._at,
                    path=self._path,
                    detail=f"asked {len(wanted)}, served {len(samples)}",
                )
            )

        values = np.stack([self._item_values(sample) for sample in samples])

        return values if isinstance(index, slice) else values[0]

    def _positions(self, index: int | slice) -> range:
        """Turn an index into the positions it names, refusing one that runs past the end.

        A slice that clamped would return fewer items than a block names, which is the short result
        the failure policy forbids.

        Args:
            index: The position of one item, or the slice naming a contiguous run.

        Returns:
            The positions to read, in ascending order.

        Raises:
            EvaluationError: If the index names a position the converted form does not hold.
        """
        held = len(self)
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = held if index.stop is None else index.stop
            wanted = range(start, stop)
        else:
            position = index + held if index < 0 else index
            wanted = range(position, position + 1)

        if wanted.start < 0 or wanted.stop > held or wanted.stop < wanted.start:
            raise EvaluationError(
                failure_message(
                    "a read names items the converted form does not hold",
                    at=self._at,
                    path=self._path,
                    detail=f"asked {wanted.start}..{wanted.stop}, holds {held}",
                )
            )

        return wanted

    def _item_values(self, sample: Any) -> np.ndarray:
        """Stack one sample's series into that item's values, in stored order.

        Args:
            sample: One sample the reader rebuilt.

        Returns:
            The item's values, one row per channel, as ``float32``.

        Raises:
            EvaluationError: If the sample does not hold as many series as the dataset has
                channels.
        """
        series = sample.time_series
        if len(series) != len(self._channels):
            raise EvaluationError(
                failure_message(
                    "an item does not hold as many channels as the dataset declared",
                    at=self._at,
                    path=self._path,
                    detail=f"sample {sample.sample_id!r}, declared {len(self._channels)}, read {len(series)}",
                )
            )

        return np.stack([one.to_numpy() for one in series]).astype(np.float32, copy=False)


def _version(path: Path, *, at: OpenKey) -> DatasetVersion:
    """Open one committed version of the converted form.

    Both opens share this call, so it cannot know which reader asked. The caller gives it the
    coordinates of its own open, which name that reader.

    Args:
        path: The version directory the conversion produced.
        at: The coordinates of the open that asked.

    Returns:
        The version handle, which carries the parsed manifest and holds no decoded values.

    Raises:
        EvaluationError: If the path is not a committed version of the converted form.
    """
    try:
        return DatasetVersion.open_local(path)
    except Exception as error:
        raise EvaluationError(
            failure_message(
                "the converted form is not there or cannot be opened",
                at=at,
                path=path,
                detail=f"{error}",
            )
        ) from error


def _item_tensor(item: dict[str, Any]) -> torch.Tensor:
    """Reduce one of the library's torch items to that item's signal.

    The library's item is a dictionary holding the sample's series, its id, its tasks and its
    annotations. A canonical item is the signal and nothing else: a label or an identifier beside
    it makes the comparison form refuse the item, or reduce content that is not the signal.

    Args:
        item: The library's item.

    Returns:
        The item's values as a ``float32`` tensor, one row per channel, in stored order.
    """
    return torch.stack(list(item["series"])).to(torch.float32)
