"""The two readers, each of which opens both representations.

A reader is a target in-memory form together with the loader that produces it. There are two of
them, Pandas and PyTorch, and each one reads every representation a run holds. That cross-reading
is the experiment: a user who is deciding whether to adopt TimeF is not choosing a library, because
they already have one. They are choosing what sits on disk under the library they keep, and only a
cell in which the reader is held still and the representation moves answers that.

The two readers share a shape and no behavior. Neither imports the other, neither subclasses a
protocol, and there is no base class between them. They share one module, ``block``, which holds
the test and the message for the one precondition a block read has, because both readers of one
representation walk the same plan.

Neither reader knows what dataset it is reading. Each one is handed a representation, and the
representation carries the parsing path, the artifact and the canonical item declaration. Adding a
dataset is writing a connector. It is never editing a reader.

Nothing in this package may import the storage library the ``timef`` extra supplies. This module
imports both readers, so such an import in either one makes the whole package unimportable where the
extra is absent. Every reader test then fails to collect, and none of them skips. A representation
can keep such an import in a module its own package does not import, and one does. A reader cannot,
because this module imports it. Neither reader needs one: a reader reaches every representation
through that representation's parsing path.
"""

from timenet_evaluations.grid.readers.block import block_failure_message, names_declared_items
from timenet_evaluations.grid.readers.pandas_ import PANDAS, PandasOpened, PandasReader
from timenet_evaluations.grid.readers.torch_ import (
    BATCH_SIZE,
    PYTORCH,
    WORKER_COUNT,
    PyTorchOpened,
    PyTorchReader,
    build_loader,
)


__all__ = [
    "BATCH_SIZE",
    "PANDAS",
    "PYTORCH",
    "WORKER_COUNT",
    "PandasOpened",
    "PandasReader",
    "PyTorchOpened",
    "PyTorchReader",
    "block_failure_message",
    "build_loader",
    "names_declared_items",
]
