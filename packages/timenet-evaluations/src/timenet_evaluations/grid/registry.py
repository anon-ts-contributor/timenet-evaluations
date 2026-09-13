"""The subjects a run measures, written out by hand: two representations, two readers, four tasks.

A benchmark's subject list is part of its result. Nothing here is discovered. No entry point is
read, no installed package is probed, and no directory is scanned. A subject that arrived because
it happened to be installed would make two runs on the same machine measure different grids without
saying so, and a result has no field to record that.

The literals are a maintenance chore, and that is the point: the same explicitness makes a run
auditable by reading two lines. The cost is real and this module states it. Adding a reader or a
representation is an edit in more than one place, and nothing finds a module that was written and
never added here. Such a subject is silent, and a report simply does not mention it.

``READERS`` carries the annotation ``tuple[Reader, ...]``, and that annotation is the whole
conformance gate. Neither reader protocol is runtime checkable, so no ``isinstance`` guard exists.
Do not widen the annotation to ``tuple[Any, ...]`` and do not drop it: two unrelated classes in a
tuple type-check as their own union, and then nothing is verified.

``REPRESENTATIONS`` holds names and not objects, because a representation is per dataset. It needs
the release's path, the canonical item declaration, and the parsing path, and a run supplies all
three. The order and the membership of the axis are fixed here. The objects are built per dataset
and are given to :func:`timenet_evaluations.grid.assembly.assemble`.
"""

from __future__ import annotations

from timenet_evaluations.grid.cell import Task
from timenet_evaluations.grid.reader import Reader
from timenet_evaluations.grid.readers import PandasReader, PyTorchReader
from timenet_evaluations.grid.representations import Original, TimeF


REPRESENTATIONS: tuple[str, ...] = (Original.name, TimeF.name)
"""The two representations a run holds every dataset in, in report order.

The names come from the two representation classes, so the axis and the objects on it cannot drift
apart. The tuple is still hand-written: it names both members and their order, and nothing adds a
third by being importable.
"""

READERS: tuple[Reader, ...] = (PandasReader(), PyTorchReader())
"""The two readers, each of which opens both representations.

Both readers are constructed once per run. Neither one holds a representation, a path, or a result,
so one instance serves every cell of its column. What a read holds is the opened reader, and a
repetition discards that.
"""

TASKS: tuple[Task, ...] = (Task.FIRST_ITEM, Task.FULL_READ, Task.SEQUENTIAL, Task.BLOCK_SHUFFLED)
"""The four read operations every cell is measured on, in report order.

Conversion is not among them. It runs once before the first timer starts, so it is not a task and
has no column.
"""

CELLS_PER_DATASET = len(REPRESENTATIONS) * len(READERS) * len(TASKS)
"""How many timed cells one dataset produces: sixteen.

The count is derived from the three axes rather than written as a number, so an axis and the count
cannot disagree. It is unconditional. There is no scoping and no declared region of the grid,
because a run reduced to one representation compares nothing.
"""

STORAGE_FIGURES_PER_DATASET = len(REPRESENTATIONS)
"""How many storage figures one dataset produces: two, one per representation.

The readers do not multiply this number. Nothing a reader does changes bytes on disk.
"""
