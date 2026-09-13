"""The grid: two representations, two readers, and the canonical item both readers count.

A run holds each dataset in two representations and reads both with both readers, over four tasks.
One dataset therefore produces 16 timed cells and 2 storage figures. This package holds the seam
those cells are made of, and nothing here measures itself.

It also holds the grid itself. ``registry`` writes out the subjects a run measures, and ``assembly``
pairs every reader with every representation and refuses a run that measured less than the whole
grid. Nothing here times a read: the read-measurement capability owns every number.
"""

from timenet_evaluations.grid.artifact import Artifact
from timenet_evaluations.grid.assembly import (
    Pairing,
    assemble,
    check_artifact_agrees,
    check_grid_complete,
    expected_cells,
    expected_storage_keys,
)
from timenet_evaluations.grid.cell import Cell, OpenKey, StorageKey, Task
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.item import (
    Bulk,
    Item,
    ItemDeclaration,
    Items,
    check_item_count,
    to_comparison_form,
)
from timenet_evaluations.grid.parsing import ParsedItems, ParsingPath
from timenet_evaluations.grid.reader import Block, Opened, Reader
from timenet_evaluations.grid.registry import (
    CELLS_PER_DATASET,
    READERS,
    REPRESENTATIONS,
    STORAGE_FIGURES_PER_DATASET,
    TASKS,
)
from timenet_evaluations.grid.representation import Representation


__all__ = [
    "CELLS_PER_DATASET",
    "READERS",
    "REPRESENTATIONS",
    "STORAGE_FIGURES_PER_DATASET",
    "TASKS",
    "Artifact",
    "Block",
    "Bulk",
    "Cell",
    "Item",
    "ItemDeclaration",
    "Items",
    "OpenKey",
    "Opened",
    "Pairing",
    "ParsedItems",
    "ParsingPath",
    "Reader",
    "Representation",
    "StorageKey",
    "Task",
    "assemble",
    "check_artifact_agrees",
    "check_grid_complete",
    "check_item_count",
    "expected_cells",
    "expected_storage_keys",
    "failure_message",
    "to_comparison_form",
]
