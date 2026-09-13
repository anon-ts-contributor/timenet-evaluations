"""The block plan: four numbers in, one tuple of blocks out, and nothing read.

The block-shuffled task walks a plan. The plan cuts the canonical items into runs of consecutive
items, and it visits those runs in a seeded order. It never reorders the items inside a run, so the
task models a training loader: contiguous streaming within a block, one seek between blocks.

This module is pure. It opens no file, reads no clock, and draws nothing from the process-global
random generator. That is why the partition, the ordering and the size of a block are all testable
with no artifact, no fixture and no dataset, and why continuous integration can hold them.

The order of the four steps below is a requirement and not a convenience. Within one
(dataset, representation) pair a run does them in this sequence, and a change that moves ``size``
after ``plan`` gives the plan a number that describes no artifact:

1. Convert, where this harness produces the representation. This step is untimed.
2. Measure the size. ``harness.size.measure_size`` counts the bytes of what is now on disk.
3. Plan the blocks. ``size_bytes`` is the first argument, so a plan cannot be built before step 2.
   A run plans once per pair, outside every timed interval, and both readers walk the one plan.
4. Walk the plan. The block-shuffled task is timed and the plan is already complete.

``Block`` is the value the reader seam already states the shape of. ``grid.reader.Block`` declares a
structural protocol of two read-only members, and this class satisfies it by shape alone: it
imports nothing from that module, subclasses nothing, and registers itself nowhere. ``ty`` is the
gate on that agreement.

The number of items in a block follows a byte budget, so it differs per representation. Two
shuffled rates in one row are therefore not a like-for-like count of operations, and a report that
prints them must print each representation's ``items_per_block`` and block count beside the shared
``block_bytes``. The fixed budget slightly favours whichever representation has the larger item.
That direction is recorded. The mechanism is not established, and it is not seek count. Nobody must
defend a figure here by naming a cause.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict

from timenet_evaluations.errors import EvaluationError


BLOCK_BYTES: Final = 64 * 1024 * 1024
"""The byte budget of one block, fixed at 64 MB.

It is identical for both representations of a dataset and for every dataset in a run. It is the one
quantity the comparison holds constant, so a per-representation budget would leave two shuffled
figures with nothing shared at all. Every measurement records it.
"""


class Block(BaseModel):
    """A contiguous run of canonical items, named by a start position and a count.

    This is the value the reader seam states the shape of. The model is frozen, so a plan a result
    recorded is the plan a later run replays.
    """

    model_config = ConfigDict(frozen=True)

    start: int
    """The position of the block's first canonical item, counted from zero in the dataset's
    declared item ordering."""
    count: int
    """How many canonical items the block names. It is at least one. The final block of a plan can
    be shorter than the others."""


class BlockPlan(BaseModel):
    """A block plan together with the four numbers it came from.

    A shuffled figure alone describes a workload nobody can recover. This model is what travels
    with the figure: the four arguments, the derived block size, the block count, and the blocks
    themselves. Re-planning from the four arguments reproduces ``blocks`` exactly.

    ``items_per_block`` differs per representation by construction, so the two representations of
    one dataset cross a different number of block boundaries over the same items. A report must
    print both numbers beside the shared ``block_bytes``, because two shuffled figures are not a
    like-for-like count of operations.
    """

    model_config = ConfigDict(frozen=True)

    size_bytes: int
    """The representation's own size on disk, as ``measure_size`` counted it before the plan was
    built."""
    n_items: int
    """The dataset's canonical item count. It is identical for both representations, so
    ``size_bytes`` is the only input that differs between the two plans of one dataset."""
    block_bytes: int
    """The byte budget one block was given. It is the same for both representations."""
    seed: int
    """The run's seed. It fixes the order of the blocks and nothing else."""
    items_per_block: int
    """How many consecutive canonical items the budget bought this representation."""
    block_count: int
    """How many blocks the items were cut into."""
    blocks: tuple[Block, ...]
    """The blocks, in the order the pass visits them."""


def items_per_block(size_bytes: int, n_items: int, block_bytes: int) -> int:
    """Derive how many consecutive canonical items fit the byte budget for one representation.

    The count is ``block_bytes // (size_bytes / n_items)`` — the budget divided by this
    representation's mean bytes per item. The right side of that division is a float, so floor
    division gives a float back, and the result is made an integer here on purpose.

    A budget smaller than one item's mean size gives a count below one. The count is then one, so
    the plan holds one item per block. No block is ever empty.

    Args:
        size_bytes: This representation's size on disk, in bytes.
        n_items: The dataset's canonical item count.
        block_bytes: The byte budget of one block.

    Returns:
        The number of canonical items in a block, which is one or more.

    Raises:
        EvaluationError: If any of the three numbers is below one. A representation of no bytes,
            of no items, or a budget of no bytes describes no workload.
    """
    if size_bytes < 1 or n_items < 1 or block_bytes < 1:
        raise EvaluationError(
            "a block plan needs a positive size, item count and byte budget: "
            f"size_bytes {size_bytes}, n_items {n_items}, block_bytes {block_bytes}"
        )

    return max(1, int(block_bytes // (size_bytes / n_items)))


def plan_blocks(size_bytes: int, n_items: int, block_bytes: int, seed: int) -> tuple[Block, ...]:
    """Cut the canonical items into blocks and put those blocks in a seeded order.

    This function is pure. It takes four numbers, and it takes no path, no artifact, no
    representation and no reader. It touches no filesystem, reads no clock, and draws the
    permutation from a generator that the ``seed`` argument alone seeds.

    It decides three things, in this order:

    1. How large a block is. See ``items_per_block``. This step is what varies per representation.
    2. Where the boundaries fall. The items are cut into consecutive runs of that size, and the
       last run can be short. Every item is in exactly one run, so one pass reads the dataset
       exactly once. The boundaries do not depend on the seed.
    3. What order the runs are visited in. The seed permutes the runs. It never permutes the items
       inside a run, and a reader must deliver the items of one block in ascending order. The
       permutation comes from a generator that this seed alone seeds, and never from the
       process-global generator.

    The output is data and not behaviour. The whole plan stands before the first block is read, so
    a test can assert on it, a result can store it, and a later run can replay it.

    Call this after ``measure_size`` and before the shuffled walk. See the module docstring for the
    four steps of one (dataset, representation) pair and why their order is fixed.

    Args:
        size_bytes: This representation's size on disk, in bytes, as ``measure_size`` counted it
            after the conversion and before this call.
        n_items: The dataset's canonical item count.
        block_bytes: The byte budget of one block, which is ``BLOCK_BYTES`` in a run.
        seed: The run's seed. The same seed over the same three numbers gives the same tuple.

    Returns:
        The blocks, in the order the pass visits them.
    """
    count = items_per_block(size_bytes, n_items, block_bytes)
    runs = [Block(start=start, count=min(count, n_items - start)) for start in range(0, n_items, count)]
    order = np.random.default_rng(seed).permutation(len(runs))

    return tuple(runs[int(position)] for position in order)


def record_plan(size_bytes: int, n_items: int, block_bytes: int, seed: int) -> BlockPlan:
    """Build a block plan and the record that travels with the measurement taken over it.

    A run calls this once per (dataset, representation) pair, outside every timed interval. Both
    readers of that representation walk the ``blocks`` of the one record. A second call per reader,
    or per repetition, is a second thing that can drift.

    Args:
        size_bytes: This representation's size on disk, in bytes.
        n_items: The dataset's canonical item count.
        block_bytes: The byte budget of one block, which is ``BLOCK_BYTES`` in a run.
        seed: The run's seed, chosen once for the run and shared by every pair.

    Returns:
        The plan together with the four numbers it came from, the derived block size and the block
        count.
    """
    blocks = plan_blocks(size_bytes, n_items, block_bytes, seed)

    return BlockPlan(
        size_bytes=size_bytes,
        n_items=n_items,
        block_bytes=block_bytes,
        seed=seed,
        items_per_block=items_per_block(size_bytes, n_items, block_bytes),
        block_count=len(blocks),
        blocks=blocks,
    )


def replay(plan: BlockPlan) -> tuple[Block, ...]:
    """Build the blocks again from the four numbers a record holds.

    A stored result is replayable exactly as far as this goes: the tuple this gives back equals the
    ``blocks`` of the record, block for block and in the same order.

    Args:
        plan: The recorded plan.

    Returns:
        The blocks, in the order the pass visited them.
    """
    return plan_blocks(plan.size_bytes, plan.n_items, plan.block_bytes, plan.seed)
