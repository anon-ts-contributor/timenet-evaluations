"""The connector protocol: the four members every dataset reaches this harness through.

One object stands for one dataset. It reads the release's own files for the untimed conversion, it
opens those files again for each of the two readers inside a timed cell, and it declares what one
canonical item is. Those four jobs are the four members below, and there is no fifth.

The seam does not state this contract and cannot stand in for it. "Only ``source`` imports a dataset
library" says where dataset knowledge lives. It says nothing about what the author of the second
dataset has to write, and a package is not an interface. These four members are the interface, and
they are what makes the connector that ships a worked example of a contract rather than a second
opinion about one.

The two opens are two members, and neither one is built from the other. A reader is a target
in-memory form together with the loader that produces it, and on the release's own files the release
supplies a different code path for each of the two forms. One member with a mode argument moves that
choice inside the connector, where a timer cannot see which path it measures. An open for the torch
form that called the open for the pandas form would put a ``DataFrame`` build inside a cell whose
figure is supposed to describe the release's torch path and nothing else.

A fifth member must not be added. A connector can hold as many private helpers as it needs, and the
one that ships does. What must not grow is the surface every future connector has to implement: a
member added here is a member the next dataset pays for, and whoever writes that dataset is not in
the room. A capability that finds it needs something else from a dataset must first establish that
it is something every dataset has. The rate is the case that comes up: it travels inside the
declaration :meth:`Connector.canonical_item` returns, because a value the conversion reads is not a
call a reader makes.

This protocol is structural, like every protocol beside it in the grid. A connector declares the
four members and inherits nothing. It imports no protocol, subclasses nothing, and registers itself
nowhere but in the one hand-written table in :mod:`timenet_evaluations.source.registry`.

``ty`` is the gate. The protocol is not runtime checkable, so no ``isinstance`` guard exists, and the
mechanism is the annotation ``CONNECTORS: dict[str, Connector]`` on that table. A connector whose
signature drifted is a type-check diagnostic. :func:`timenet_evaluations.source.registry.connector_for`
adds the one refusal a type checker cannot make at run time, because a run that reached the first
cell before it found a missing member has already paid for a load, a conversion and every cache drop
taken before it.

The two opens carry the signatures ``timenet_evaluations.grid.parsing`` declares, and a connector
satisfies that protocol by shape alone. Nothing in this package imports that module. The dependency
runs one way, ``source`` to ``grid``, which is what keeps the build graph acyclic, so
:class:`ParsedItems` below states the shape a second time rather than imports it.
"""

# opens)

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd
from torch.utils.data import Dataset

from timenet_evaluations.source.declaration import DatasetDeclaration


class ParsedItems(Protocol):
    """Positional access to the canonical items of one opened representation.

    The Pandas reader gets this object from :meth:`Connector.open_pandas` and holds it for one task.
    The object is a locator: it finds items. It is not a result, so a reader must not keep what it
    read through it and serve a later call from that.

    This is the shape the grid declares for the same object. The two are matched by shape and by
    nothing else, so that this package depends on no module of the grid that a reader also depends
    on.
    """

    def __len__(self) -> int:
        """Report how many canonical items this representation can serve.

        Returns:
            The item count the representation holds. No rate divides by this number: a reader drives
            every read from the dataset's declared count.
        """
        ...

    def __getitem__(self, index: int | slice) -> object:
        """Read one canonical item, or the contiguous run of items that a slice names.

        Args:
            index: The position of one item, or the slice that names a contiguous run of items.

        Returns:
            One item's values for an integer index. For a slice, the values of the items the slice
            names, in ascending position order.
        """
        ...


class Connector(Protocol):
    """One dataset, as the four members this harness reaches it through.

    A connector holds no fact about the harness. It knows the release's own files and the unit that
    release's task consumes, and everything downstream of it sees a frame, a declaration, and two
    opened objects.

    Adding a dataset means writing one of these and adding one line to the registry. It must not
    mean editing a reader, a representation, the harness, or the result model.
    """

    def load_for_conversion(self, source: Path) -> pd.DataFrame:
        """Read one dataset into memory, untimed, for the conversion to write from.

        Call this exactly once per dataset per run. The frame it returns is the input the conversion
        writes the converted representation from, and it exists for no other purpose. The release's
        own files are read as they ship and are never produced from this frame.

        This call must not sit inside a timed region, and no reported figure includes it. The
        conversion is not measured, so the read that feeds the conversion is not measured for the
        same reason. A run that reported it would price adoption cost in a table that leaves
        adoption cost out on purpose.

        The read is eager and complete. It drains the source fully and returns a materialized frame.
        It must not return a generator, an iterator, a lazy view, or any object whose values are
        fetched when the conversion touches them: every timed repetition starts from a dropped page
        cache, so source input and output that is deferred out of this call lands in a phase whose
        page cache state nothing controls.

        A second load of one dataset, a re-read, or a load per representation must not be
        introduced. One frame per dataset is what makes the declared item count and the converted
        artifact statements about the same object. Nothing in this package can enforce that clause,
        because it is a property of the run sequence, and the run orchestration owns it.

        Args:
            source: The directory of release files this dataset was given, as the run was given it.

        Returns:
            One row per canonical item, with the three columns of the frame contract.
        """
        ...

    def open_pandas(self, path: Path) -> ParsedItems:
        """Open the release's own files for the Pandas reader, inside a timed cell.

        This call runs inside the timed interval, so it does what the release's loader really does,
        including a parse at construction.

        This is not the same activity as :meth:`load_for_conversion`, and the two can use the same
        reference loader. One is untimed and happens once per dataset. This one is a timed cell,
        repeated with the page cache dropped before each repetition. This call must open the
        release's files itself, inside its own timed interval. It must not be served from the frame
        the conversion already loaded, from a cached object, from anything a previous step
        materialized, or from a reader that a previous repetition left open. Such a cell would
        report an in-memory attribute access under a column that says read from disk, and it would
        be several times faster with no structural tell.

        Args:
            path: The artifact path of the representation, which for the release's own files is the
                source directory the run was given.

        Returns:
            Positional access to this representation's canonical items.
        """
        ...

    def open_torch(self, path: Path) -> Dataset[object]:
        """Open the release's own files for the PyTorch reader, inside a timed cell.

        The dataset is map-style: it declares ``__len__`` and ``__getitem__``, so the reader walks it
        in ascending position order and names a contiguous run of positions. The reader wraps it in
        the dataloader. A connector constructs no dataloader and exposes no worker count.

        This member is required even where a release ships no torch path of its own. The adapter
        that runs inside the timer is then this project's rather than the release's, and the report
        must say so. This member must not be implemented on :meth:`open_pandas`.

        Args:
            path: The artifact path of the representation, which for the release's own files is the
                source directory the run was given.

        Returns:
            A torch dataset whose item at a position is that item's signal, as a ``float32`` tensor
            of the declared item shape.
        """
        ...

    def canonical_item(self, source: Path) -> DatasetDeclaration:
        """State what this dataset declares about itself, in advance of any measurement.

        Call this in untimed work, before the first timer of the source it names. The declaration is
        fixed before a run starts and is never negotiated at run time. Nothing derives it from an
        artifact, from a representation, from a reader, or from anything a run produced. A
        representation that announced what it can conveniently yield, and was believed, would be the
        subject under test choosing the denominator its own rates divide by.

        The member takes the source it is asked about, and it must not answer without one. One
        connector is registered per name and one instance serves every run, so a member with no
        argument can only answer from state it stored. Such a member returns the first path's count
        for a second path under the same name, and the registry forbids a connector that holds a
        path. A connector whose count is a literal ignores the argument. A connector whose count
        cannot be known without reading the corpus has somewhere to put the answer, and that case is
        real: where the items are scored intervals, the untimed preparation is what establishes the
        count.

        The rate is one of the facts this value carries. It is here, and not a fifth member, because
        the connector surface is fixed at four and a value the conversion reads is not a call a
        reader makes. What a representation's measured cells really read is here for the same
        reason.

        Args:
            source: The directory of release files this dataset was given, as the run was given it.

        Returns:
            The unit one canonical item is, how many of them this source holds, the rate every
            channel was delivered at, and what each representation's cells really read.
        """
        ...
