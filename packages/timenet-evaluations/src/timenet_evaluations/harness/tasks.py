"""The four timed tasks: where the instrument meets the grid.

Every cell of the grid is measured on four tasks and on nothing else: the first item, the full read,
the sequential walk, and the block-shuffled walk. Each one wraps a reader operation in a callable
and hands that callable to ``harness.repeat.repeat``. No task holds a clock, a sample list or a
reduction of its own, so the protocol has one implementation for all sixteen cells of a dataset.

**Only a read is inside the timer.** The conversion ran before this module was called, the size was
counted before the plan was built, the plan was built before the cell was timed, and the reduction
happens after the last sample. The four operations below call the reader and call nothing else,
which is what makes that checkable rather than argued. ``open`` is inside the interval, because
SPEC-0017 settles that opening is part of what a cell measures.

**One repetition is one whole pass.** The shuffled task walks every block of the plan, in plan
order, inside one interval. A plan of forty blocks walked five times records five samples, each
covering all forty blocks — never forty samples and never two hundred. The sequential task is one
in-order walk over every canonical item, item at a time. It is a task of its own and is never an
item count divided by the full-read figure: a run that derived it would have measured three tasks
and printed four.

**One plan per (dataset, representation), and both readers walk it.** ``record_plan`` is called once
for each representation, outside every timed interval, and the two pairings of that representation
are handed the same tuple of blocks. The two shuffled rates of one row are compared against each
other, so two permutations or two block sizes would make them rates over different workloads. A
second call with identical arguments is a second thing that can drift.

**A failure stops the run.** Nothing here catches anything. A missing artifact, a failed drop, a
reader that raises partway through a task — each one propagates, and no measurement, no cell and no
dataset is skipped so that the rest can carry on. A grid with a hole in it cannot be read, and
sixteen cells per dataset give it sixteen ways to happen. ``check_grid_complete`` runs before the
dataset's result is returned, so a run that produced a subset refuses rather than reports one.

One requirement of § Four Tasks Per Cell is not built here: the gate that verifies a reference
loader's materialisation exists before the dataset's first timed repetition. Neither
``Representation`` nor ``ParsingPath`` has a member that answers the question, both protocols belong
to SPEC-0017, and nothing in this package may widen them. The gap is recorded rather than filled
with a guess.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

from pydantic import BaseModel, ConfigDict

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.assembly import Pairing, assemble, check_grid_complete
from timenet_evaluations.grid.cell import Cell, StorageKey, Task
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.representation import Representation
from timenet_evaluations.harness.drop import DropCaches
from timenet_evaluations.harness.plan import BLOCK_BYTES, Block, BlockPlan, record_plan
from timenet_evaluations.harness.repeat import REPEATS, Measurement, Timed, repeat
from timenet_evaluations.harness.size import measure_size


@dataclass(frozen=True)
class Timing:
    """The protocol every timed cell of a run is measured under.

    The three values travel together, so a task takes one argument instead of three and every cell
    of the run is measured under one setting. A per-cell setting is what would let a cold figure and
    a warm figure meet in one table.
    """

    drop_caches: DropCaches
    """The drop each repetition begins with. It is the mechanism the calibration probe measured."""
    repeats: int = REPEATS
    """How many cold repetitions one cell is measured over."""
    warmup: bool = False
    """Whether each repetition performs one untimed read after the drop. It is off, so the figures
    are cold figures."""


@dataclass(frozen=True)
class RunParameters:
    """What one run measures every dataset under.

    The seed is chosen once for the run and shared by every (dataset, representation) pair, so a
    recorded plan reproduces from its recorded arguments. It is never derived per reader, per
    repetition or per pair.
    """

    timing: Timing
    """The drop, the repetition count and the warm-up setting."""
    seed: int
    """The run's seed. It orders the blocks of every plan and nothing else."""
    block_bytes: int = BLOCK_BYTES
    """The byte budget of one block. It is the one quantity the comparison holds constant, so it is
    identical for both representations of a dataset and for every dataset of the run."""


class Prepared(BaseModel):
    """The two untimed numbers one representation needs before any clock starts.

    A cell needs both of them, and both are produced outside every timed interval: the size on
    disk, which is counted and never timed, and the block plan, which the shuffled walk replays.
    The order is fixed — ``size_bytes`` is the plan's first argument — so a plan built before the
    size describes no artifact.

    There is one of these per (dataset, representation) pair and therefore two per dataset. Neither
    carries a reader coordinate, because nothing a reader does changes bytes on disk.
    """

    model_config = ConfigDict(frozen=True)

    at: StorageKey
    """The dataset and the representation these two numbers belong to."""
    size_bytes: int
    """The representation's own size on disk, as ``measure_size`` counted it."""
    plan: BlockPlan
    """The block plan both readers of this representation walk, and the four numbers it came
    from."""


class DatasetGrid(BaseModel):
    """One dataset's whole grid: sixteen timed measurements and two storage figures.

    A result that looks complete and is not is worse than no result, so this is built only after
    ``check_grid_complete`` has agreed that every declared cell is in it.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str
    """The dataset these numbers describe, as the run was given it."""
    prepared: tuple[Prepared, ...]
    """The untimed inputs of each representation, in the order the registry declares them."""
    measurements: tuple[Measurement, ...]
    """Every timed measurement, representation by representation, then reader, then task."""


def first_item(pairing: Pairing) -> object:
    """Open the representation and deliver the canonical item at position zero.

    The whole task is one call on the opened reader. This measures the latency to a known item, so
    the reader does not define a notion of first of its own.

    Args:
        pairing: The representation to read and the reader that reads it.

    Returns:
        The item at position zero, in the reader's target form. The caller discards it.
    """
    return pairing.reader.open(pairing.representation).read_first()


def full_read(pairing: Pairing) -> object:
    """Open the representation and read every canonical item in one bulk pass.

    The whole task is one call on the opened reader. This task and the sequential walk are two
    tasks, and neither is computed from the other.

    Args:
        pairing: The representation to read and the reader that reads it.

    Returns:
        Every canonical item, in the reader's target form. The caller discards it.
    """
    return pairing.reader.open(pairing.representation).read_all()


def sequential_walk(pairing: Pairing) -> object:
    """Open the representation and walk every canonical item in order, one item at a time.

    The walk asks the reader for the items and consumes them as they arrive. It does not call the
    full read, and the rate it produces is not an item count divided by the full-read figure: a run
    that derived it would have measured three tasks and printed four.

    Args:
        pairing: The representation to read and the reader that reads it.

    Returns:
        Nothing. Each item is materialized by the walk and then discarded.
    """
    for _ in pairing.reader.open(pairing.representation).read_sequential():
        pass

    return None


def block_walk(pairing: Pairing, blocks: tuple[Block, ...]) -> object:
    """Open the representation and walk the whole block plan once, in plan order.

    One call of this function is one repetition of the shuffled task, and one repetition is one
    complete pass over the plan. A plan of forty blocks is forty reads inside one interval, not
    forty intervals.

    The blocks are visited in the order the plan holds them. Nothing here sorts them, merges them,
    or coalesces adjacent ones: the walk would then be a second full scan reported as a shuffled
    rate.

    Args:
        pairing: The representation to read and the reader that reads it.
        blocks: The plan of this representation, which the other reader of it also walks.

    Returns:
        Nothing. Each block's items are materialized by the walk and then discarded.
    """
    opened = pairing.reader.open(pairing.representation)
    for block in blocks:
        opened.read_block(block)

    return None


def operation_for(pairing: Pairing, at: Cell, blocks: tuple[Block, ...]) -> Callable[[], object]:
    """Give the operation one timed interval covers.

    The interval covers exactly what the returned callable does, so this is the whole statement of
    what a task times. Nothing else may be put in it: no conversion, no size, no planning and no
    reduction.

    Args:
        pairing: The representation to read and the reader that reads it.
        at: The four coordinates of the cell, whose task selects the operation.
        blocks: The plan of this representation. Only the shuffled task walks it.

    Returns:
        The operation to time, which takes no argument and whose result is discarded.

    Raises:
        EvaluationError: If the task has no operation here. A task the grid declares and this
            module does not serve would leave a hole where a measurement belongs, so the run stops
            rather than reporting fifteen cells as sixteen.
    """
    if at.task is Task.FIRST_ITEM:
        return partial(first_item, pairing)
    if at.task is Task.FULL_READ:
        return partial(full_read, pairing)
    if at.task is Task.SEQUENTIAL:
        return partial(sequential_walk, pairing)
    if at.task is Task.BLOCK_SHUFFLED:
        return partial(block_walk, pairing, blocks)

    raise EvaluationError(
        failure_message(
            "a declared task has no timed operation, so this cell of the grid cannot be measured",
            at=at,
            path=pairing.representation.artifact.path,
        )
    )


def measure_task(pairing: Pairing, at: Cell, *, blocks: tuple[Block, ...], timing: Timing) -> Measurement:
    """Measure one task of one cell.

    The figure comes from ``repeat`` and from nowhere else. This function decides what is timed; it
    does not decide how the samples are taken, how many there are, or how they are reduced.

    Args:
        pairing: The representation to read and the reader that reads it.
        at: The four coordinates of the cell being measured.
        blocks: The plan of this representation, built before this call.
        timing: The drop, the repetition count and the warm-up setting of the run.

    Returns:
        Every sample the timing loop recorded, and the cell they belong to.
    """
    timed = Timed(
        operation=operation_for(pairing, at, blocks),
        at=at,
        path=pairing.representation.artifact.path,
    )

    return repeat(
        timed,
        drop_caches=timing.drop_caches,
        repeats=timing.repeats,
        warmup=timing.warmup,
    )


def measure_pairing(pairing: Pairing, *, blocks: tuple[Block, ...], timing: Timing) -> tuple[Measurement, ...]:
    """Measure all four tasks of one (representation, reader) pair.

    The four tasks come from the pairing's own cells, which the registry declares. A pairing
    measured on fewer tasks than another produces a row whose columns nobody can compare.

    Args:
        pairing: The representation to read and the reader that reads it.
        blocks: The plan of this representation, which the other reader of it also walks.
        timing: The drop, the repetition count and the warm-up setting of the run.

    Returns:
        The four measurements, in the order the registry declares the tasks.
    """
    return tuple(measure_task(pairing, at, blocks=blocks, timing=timing) for at in pairing.cells)


def prepare(representation: Representation, *, dataset: str, run: RunParameters) -> Prepared:
    """Produce the two untimed numbers of one (dataset, representation) pair.

    The order is a requirement and not a convenience. The size is counted first, because it is the
    plan's first argument. Both happen outside every timed interval, and the plan is built once
    here rather than once per reader or once per repetition.

    Args:
        representation: The representation to measure and plan for. The conversion that produced it
            already ran, untimed.
        dataset: The name of the dataset, as the run was given it.
        run: The seed and the byte budget the run plans every pair under.

    Returns:
        The storage figure and the block plan of this representation.
    """
    size_bytes = measure_size(
        representation.artifact.path,
        dataset=dataset,
        representation=representation.name,
    )
    plan = record_plan(size_bytes, representation.item.count, run.block_bytes, run.seed)

    return Prepared(
        at=StorageKey(dataset=dataset, representation=representation.name),
        size_bytes=size_bytes,
        plan=plan,
    )


def check_one_item_count(dataset: str, representations: Sequence[Representation]) -> None:
    """Make sure that every representation of one dataset declares the same canonical item count.

    The count is a property of the dataset and not of a representation, and it is the second
    argument of every plan. Two counts would give the two representations of one row two plans over
    different numbers of items, and every rate would still render.

    Args:
        dataset: The name of the dataset, as the run was given it.
        representations: The dataset's representations.

    Raises:
        EvaluationError: If two representations declare different counts.
    """
    counts = {representation.name: representation.item.count for representation in representations}
    if len(set(counts.values())) > 1:
        raise EvaluationError(
            f"the representations of a dataset declare one canonical item count, because a rate "
            f"over two counts counts two different things: dataset {dataset!r}, declared {counts}"
        )


def measure_dataset(dataset: str, representations: Sequence[Representation], run: RunParameters) -> DatasetGrid:
    """Measure the whole grid of one dataset.

    The order is fixed and is identical from run to run: the representations in the order the
    registry declares them, then the readers, then the tasks. Whatever ordering effect in-process
    measurement leaves is then at least constant between two runs being compared.

    Everything untimed happens first. The axis and the artifacts are checked, the item count is
    checked, every size is counted and every plan is built — and only then does the first clock
    start. Both readers of one representation are handed that representation's one plan.

    Nothing here is caught. An absent artifact, a failed drop, a reader that raises partway through
    a task, and a grid that is not the whole declared grid all stop the run through
    ``EvaluationError``, and this function returns nothing at all in that case.

    Args:
        dataset: The name of the dataset, as the run was given it.
        representations: The dataset's representations, in the order the registry declares them.
        run: The timing protocol, the seed and the byte budget of the run.

    Returns:
        The dataset's sixteen timed measurements and two storage figures.
    """
    pairings = assemble(dataset, representations)
    check_one_item_count(dataset, representations)
    prepared = {
        representation.name: prepare(representation, dataset=dataset, run=run) for representation in representations
    }

    measurements: list[Measurement] = []
    for pairing in pairings:
        blocks = prepared[pairing.representation.name].plan.blocks
        measurements.extend(measure_pairing(pairing, blocks=blocks, timing=run.timing))

    check_grid_complete(
        (dataset,),
        [measurement.at for measurement in measurements],
        [one.at for one in prepared.values()],
    )

    return DatasetGrid(
        dataset=dataset,
        prepared=tuple(prepared.values()),
        measurements=tuple(measurements),
    )
