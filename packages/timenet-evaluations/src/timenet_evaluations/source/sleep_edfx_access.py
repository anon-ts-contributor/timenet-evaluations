"""The two objects a timed open returns: positional access to this dataset's canonical items.

A timed cell over the release's own files is an open and then a read, and the open is inside the
timer. What the open returns decides what every read after it costs, so the two adapters below are
where this connector's contribution to the four read operations lives. The readers implement the
operations; these objects supply the positional access those operations drive.

Both adapters wrap the sample object the reference loader serves, and both reach it through
:func:`~timenet_evaluations.source.sleep_edfx_preparation.open_samples`. That function is the only
route this connector takes to the call that could build the task cache, and the gate sits inside it.
A timed open against an unbuilt cache therefore refuses before the build rather than times it.

**The reads are narrow, and that is a property of the cache rather than of the release.** The sample
object extends the storage library's streaming dataset, so an integer index resolves one chunked
index and reads the single chunk that holds it, and a slice resolves each index and reads only the
chunks that cover the range. Nothing here reads everything and cuts. The figure a first-item read
produces is therefore the timed open plus one chunk, and it is reported as measured: if the open
dominates it, that is a property of the cell and the open is not moved out of the timer to improve
it.

**A canonical item is that item's signal values and nothing else**, so both adapters project each
sample to its signal. The sample the loader stores holds the label, the subject, the night, the
subject's age and the subject's sex beside the signal, and an item that carried one of them would
be refused by the comparison form. The projection is one function below, used by both adapters, so
the two cells of this row reduce the same content in the same way.

The projection is not the storage library's own ``transform`` hook, and the reason is a property of
that hook. The library applies it on the integer path and not on the slice path, so a hook would
project one adapter's items and leave the other adapter's slices unprojected. The requirement the
hook satisfies is that the reduction is in the dataset and not in a collate function, and these
adapters keep it there.

**The torch adapter is map-style, and the sample object is not.** ``SampleDataset`` inherits the
iterable-style dataset, so a dataloader given one ignores every index-based sampler and walks
``__iter__``. The first-item read would then go through the streaming path and the narrowness argued
for above would hold on one side of the row only. The adapter is needed in any case, to project each
sample to its signal, so making it map-style costs nothing more and keeps both cells of the row
honest about the same claim.

Neither adapter constructs a dataloader, names a batch size, or names a worker count. That wrapper
is the harness's, it is the same wrapper on both representations, and a second one built here could
not be checked against it. The ordering the wrapper walks is established here instead: the harness's
loader calls neither ``set_shuffle`` nor the library's own dataloader helper, so this module turns
shuffling off itself rather than relies on a default.

Nothing here is held from one open to the next. Each open reaches the library again, which is what
makes the figure describe a read from disk rather than an attribute access, and the repetition
boundary discards what the open returned.

The slice path of the storage library sets a retention flag on the sample object. Nothing here reads
that flag or writes it. Within one repetition that retention is a real property of the loader and
belongs inside the timer; across repetitions the opened object is discarded.
"""

# opens and neither is built on the other)

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from pyhealth.datasets import SampleDataset
import torch
from torch.utils.data import Dataset

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import OpenKey
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.readers import PANDAS, PYTORCH
from timenet_evaluations.source.contract import SIGNAL
from timenet_evaluations.source.sleep_edfx_preparation import ORIGINAL, open_samples


SHUFFLE = False
"""Whether the sample object shuffles the order it serves its items in.

The declared ordering is the one the reference loader produced when it built the cache, and every
position-based check and the block plan index into it. The harness's loader neither turns shuffling
off nor asks the library's own helper to do it, so this module does it at each open.
"""


def open_items(source: Path, *, name: str) -> SleepEdfxItems:
    """Open the release's own files for the Pandas reader, inside a timed cell.

    Call this once per repetition, from the connector member of the same name. It reaches the
    reference loader itself every time. It is served from no frame, no cached object and nothing a
    previous repetition left open, so the figure of the cell that wraps it describes a read from
    disk.

    The cache gate runs inside the call this makes, before the call that could build, so an open
    that would have parsed the whole release refuses instead.

    Args:
        source: The root of the release, checked to be that root by the caller before this call.
        name: The registered name of the dataset, which names the failing cell in every message.

    Returns:
        Positional access to this dataset's canonical items, narrow for one item and for a run.
    """
    return SleepEdfxItems(_ordered(open_samples(source, name=name)), path=source, at=_at(name, PANDAS))


def open_signals(source: Path, *, name: str) -> SleepEdfxSignals:
    """Open the release's own files for the PyTorch reader, inside a timed cell.

    Call this once per repetition, from the connector member of the same name. It goes through the
    reference loader's own torch path: the sample object that loader serves, projected to one item's
    signal, as the map-style dataset the harness wraps. It builds no frame and is built on no other
    open, because every cost of the other target form would otherwise sit inside this column.

    Args:
        source: The root of the release, checked to be that root by the caller before this call.
        name: The registered name of the dataset, which names the failing cell in every message.

    Returns:
        A map-style dataset whose item at a position is that item's signal, as a ``float32`` tensor.
    """
    return SleepEdfxSignals(_ordered(open_samples(source, name=name)), path=source, at=_at(name, PYTORCH))


class SleepEdfxItems:
    """Positional access to this dataset's canonical items, for the Pandas reader.

    This is a locator. It finds items through the reference loader and keeps none of what it read,
    so a later call in the same task pays for its own read.

    The length is the one the sample object reports about itself. No rate divides by it: every rate
    divides by the count the untimed preparation established. It is read once, at the open, because
    the library computes it from the loader configuration in force when it is asked, and a length
    that moved during a task would move the range every read is checked against.
    """

    def __init__(self, samples: SampleDataset, *, path: Path, at: OpenKey) -> None:
        """Bind one opened sample object.

        Args:
            samples: The sample object the reference loader serves the scored windows from.
            path: The root of the release, which names a failure.
            at: The coordinates of the open this locator belongs to, which name a failure.
        """
        self._samples = samples
        self._path = path
        self._at = at
        self._count = len(samples)

    def __len__(self) -> int:
        """Report how many canonical items the sample object says it holds.

        Returns:
            The length the sample object reported at the open.
        """
        return self._count

    def __getitem__(self, index: int | slice) -> np.ndarray:
        """Read one canonical item, or the contiguous run of items a slice names.

        Both are narrow. An integer index resolves one chunked index and reads the single chunk that
        holds it. A slice resolves each index and reads only the chunks that cover the range. Nothing
        here reads the whole cache and cuts a result out of it.

        Args:
            index: The position of one item, or the slice that names a contiguous run of items.

        Returns:
            One item's values for an integer index, shaped as the item is. For a slice, the items'
            values in ascending position order, with position as the leading axis.

        Raises:
            EvaluationError: If the index names items the sample object does not hold, or if the
                sample object serves fewer items than the slice asked for.
        """
        positions = self._positions(index)
        if not isinstance(index, slice):
            return self._values(self._samples[positions.start], positions.start)

        served = self._samples[positions.start : positions.stop]
        if len(served) != len(positions):
            raise EvaluationError(
                failure_message(
                    "the reference loader served fewer canonical items than were asked for",
                    at=self._at,
                    path=self._path,
                    detail=f"asked {len(positions)}, served {len(served)}",
                )
            )

        return np.stack([self._values(item, position) for item, position in zip(served, positions, strict=True)])

    def _positions(self, index: int | slice) -> range:
        """Turn an index into the positions it names, refusing one that runs past the end.

        A slice that clamped would serve fewer items than a block names, which is the short result
        the failure policy forbids.

        Args:
            index: The position of one item, or the slice that names a contiguous run of items.

        Returns:
            The positions to read, in ascending order.

        Raises:
            EvaluationError: If the index names a position the sample object does not hold.
        """
        held = self._count
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
                    "a read names canonical items the release does not hold",
                    at=self._at,
                    path=self._path,
                    detail=f"asked {wanted.start}..{wanted.stop}, holds {held}",
                )
            )

        return wanted

    def _values(self, item: object, position: int) -> np.ndarray:
        """Reduce one sample to that item's values.

        Args:
            item: One sample, as the reference loader served it.
            position: Where that sample sits in the declared ordering, for the message.

        Returns:
            The item's values, one row per channel, as ``float32``. A sample that carries no signal
            this connector can read stops the run inside :func:`_signal`.
        """
        return _signal(item, position, path=self._path, at=self._at).numpy()


class SleepEdfxSignals(Dataset[torch.Tensor]):
    """This dataset's canonical items as tensors, map-style, for the PyTorch reader.

    The harness wraps this object in the one dataloader configuration it measures. A map-style
    dataset is what that wrapper needs: the sample object itself is iterable-style, so a dataloader
    given one walks it from the start and reaches no position by index.

    The object holds a locator and no tensor, so every read of the task drives its own pass and a
    later read cannot be served from an earlier one.

    The base class is the framework's, which the harness's loader is written against. Nothing of
    this project's is inherited: the protocols this connector satisfies are structural.
    """

    def __init__(self, samples: SampleDataset, *, path: Path, at: OpenKey) -> None:
        """Bind one opened sample object.

        Args:
            samples: The sample object the reference loader serves the scored windows from.
            path: The root of the release, which names a failure.
            at: The coordinates of the open this dataset belongs to, which name a failure.
        """
        self._samples = samples
        self._path = path
        self._at = at
        self._count = len(samples)

    def __len__(self) -> int:
        """Report how many canonical items the sample object says it holds.

        The harness's loader walks this many positions in ascending order.

        Returns:
            The length the sample object reported at the open.
        """
        return self._count

    def __getitem__(self, index: int) -> torch.Tensor:
        """Read the canonical item at one position, as that item's signal.

        The read is narrow: one integer index resolves one chunked index and reads the single chunk
        that holds it.

        Args:
            index: Where the item sits in the declared ordering. The name is the framework's, which
                the harness's loader is written against.

        Returns:
            The item's signal as a ``float32`` tensor, one row per channel. A sample that carries no
            signal this connector can read stops the run inside :func:`_signal`.
        """
        return _signal(self._samples[index], index, path=self._path, at=self._at)


def _ordered(samples: SampleDataset) -> SampleDataset:
    """Fix the order one opened sample object serves its items in.

    The harness's loader does not do this. It builds the dataloader and nothing else, and the
    library's own dataloader helper, which turns shuffling off before it wraps anything, is not the
    helper the harness uses. So the adapters establish the declared ordering rather than assume it.

    Args:
        samples: The sample object the reference loader served.

    Returns:
        The same object, serving its items in the order the cache was built in.
    """
    samples.set_shuffle(SHUFFLE)

    return samples


def _at(name: str, reader: str) -> OpenKey:
    """Name the open one adapter belongs to.

    An open serves all four read operations and happens before one of them is chosen, so the scope
    has three coordinates and no task.

    Args:
        name: The registered name of the dataset.
        reader: The name of the reader that opened it.

    Returns:
        The three coordinates of that open.
    """
    return OpenKey(dataset=name, representation=ORIGINAL, reader=reader)


def _signal(item: object, position: int, *, path: Path, at: OpenKey) -> torch.Tensor:
    """Reduce one sample to that item's signal, which is the whole of a canonical item.

    Both adapters reduce through this function, so the two cells of this row hold the same content.
    The sample carries the label, the subject, the night, the subject's age and the subject's sex
    beside the signal, and an item that kept one of them would be refused by the comparison form.

    The signal arrives as a tensor because the task declares a tensor input processor, and that
    processor runs when the cache is built rather than when an item is read. A sample that carries
    something else stops the run here: a value this function guessed at would reach a report as a
    number of items per second over content nobody checked.

    Args:
        item: One sample, as the reference loader served it.
        position: Where that sample sits in the declared ordering, for the message.
        path: The root of the release, which names a failure.
        at: The coordinates of the open the sample was read in.

    Returns:
        The item's signal as a ``float32`` tensor, one row per channel.

    Raises:
        EvaluationError: If the sample is not a mapping, if it carries no signal, or if its signal
            is not a tensor.
    """
    values = item.get(SIGNAL) if isinstance(item, Mapping) else None
    if not isinstance(values, torch.Tensor):
        raise EvaluationError(
            failure_message(
                "a sample of the release does not carry a signal this connector can read",
                at=at,
                path=path,
                detail=f"position {position}, sample form {type(item).__name__}, signal form {type(values).__name__}",
            )
        )

    return values.to(torch.float32)
