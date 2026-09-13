"""The parsing path: the one loader that both readers of a representation parse through.

A representation is a set of bytes together with the way those bytes are parsed. The way to parse
them belongs to the representation and not to a reader. Two readers of one representation that
parsed through two different loaders would post two rates whose difference is the parser, and the
report would attribute that difference to the target form.

SPEC-0016 owns the value. Its connector declares four members, and two of them are the two members
a reader calls. This module declares only those two, as a minimal structural stand-in, so that this
package depends on the shape and not on the capability that supplies it. A SPEC-0016 connector
satisfies ``ParsingPath`` by shape alone: it imports nothing here, subclasses nothing here, and
registers itself nowhere.

The two opens are two members and not one member with a mode argument. Each reader materializes its
own target form, and on ``Original`` the release supplies a different code path for each form. One
member with a mode argument moves that choice inside the parsing path, where a timer cannot see
which path it measures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from torch.utils.data import Dataset


class ParsedItems(Protocol):
    """Positional access to the canonical items of one opened representation.

    The Pandas reader gets this object from ``ParsingPath.open_pandas`` and holds it for one task.
    The object is a locator: it finds items. It is not a result, so a reader must not keep what it
    read through it and serve a later call from that.

    One item's values carry that item's signal and nothing else. A label, a subject identifier, or
    a tuple that holds one of them makes the comparison form refuse the item, because the
    comparison form is the item's values as a ``float32`` array of the declared item shape.
    """

    def __len__(self) -> int:
        """Report how many canonical items this representation can serve.

        Returns:
            The item count the representation holds. No rate divides by this number: a reader
            drives every read from the dataset's declared count.
        """
        ...

    def __getitem__(self, index: int | slice) -> object:
        """Read one canonical item, or the contiguous run of items that a slice names.

        If the representation and the loader together permit a narrow read, this call must be
        narrow. A whole-representation read behind a one-item index makes all four tasks measure
        the full read.

        Args:
            index: The position of one item, or the slice that names a contiguous run of items.

        Returns:
            One item's values for an integer index. For a slice, the values of the items the slice
            names, in ascending position order.
        """
        ...


class ParsingPath(Protocol):
    """The loader that both readers of one representation parse through.

    A representation supplies this object. It is the same object for both readers, so the
    difference between the two reader columns of one row is the target form and the
    materialization into it.
    """

    def open_pandas(self, path: Path) -> ParsedItems:
        """Open this representation for the Pandas reader.

        This call runs inside the timed interval, so it can do what the loader really does,
        including a parse at construction. It must not return a result that a later call is then
        served from.

        Args:
            path: The artifact path of the representation.

        Returns:
            Positional access to this representation's canonical items.
        """
        ...

    def open_torch(self, path: Path) -> Dataset[object]:
        """Open this representation for the PyTorch reader.

        The dataset must be map-style: it declares ``__len__`` and ``__getitem__``, so the reader
        can walk it in ascending position order and can name a contiguous run of positions. The
        reader wraps it in the dataloader. The parsing path constructs no dataloader and exposes no
        worker count.

        Args:
            path: The artifact path of the representation.

        Returns:
            A torch dataset whose item at a position is that item's signal, as a ``float32`` tensor
            of the declared item shape.
        """
        ...
