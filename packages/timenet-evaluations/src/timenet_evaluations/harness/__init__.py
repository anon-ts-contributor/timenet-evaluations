"""The harness: every number a run measures, and the state the machine was in when it was taken.

The grid states what a cell is. This package produces the numbers that fill it. Two of those
numbers are untimed inputs and land here first, because a cell needs both of them before any clock
starts: the size of a representation on disk, and the block plan the shuffled pass walks.

The state the machine is in also lands here. Every timed repetition begins from a dropped page
cache, so this package owns the drop, and it owns the calibration probe that measures whether the
drop did anything. It produces the record both of them leave; it does not store one.

One primitive holds the timing. ``repeat`` performs the drop, times the interval, keeps every
sample, reduces them to a median, and counts what it recorded. Every timed task calls it, so the
protocol has one implementation and cannot hold in one task and drift in another.

The four tasks are where the instrument meets the grid. One call per task wraps a reader operation
and hands it to ``repeat``, one plan is built per (dataset, representation) and walked by both
readers of it, and a failure at any of those points stops the run rather than leaving a hole in the
grid.

Nothing in this package is imported by ``grid``, and nothing here imports ``grid.reader``. The
concrete ``Block`` satisfies the reader seam's protocol by shape alone, which is what keeps the
dependency in one direction.
"""

from timenet_evaluations.harness.calibration import Calibration, Probe, calibrate
from timenet_evaluations.harness.drop import (
    DARWIN,
    DARWIN_ARGV,
    LINUX,
    LINUX_ARGV,
    DropCaches,
    DropCommand,
    DropRecord,
    PlatformDropCaches,
    platform_drop_command,
)
from timenet_evaluations.harness.plan import (
    BLOCK_BYTES,
    Block,
    BlockPlan,
    items_per_block,
    plan_blocks,
    record_plan,
    replay,
)
from timenet_evaluations.harness.repeat import REPEATS, Measurement, Timed, median_ns, repeat
from timenet_evaluations.harness.size import measure_size
from timenet_evaluations.harness.tasks import (
    DatasetGrid,
    Prepared,
    RunParameters,
    Timing,
    block_walk,
    check_one_item_count,
    first_item,
    full_read,
    measure_dataset,
    measure_pairing,
    measure_task,
    operation_for,
    prepare,
    sequential_walk,
)
from timenet_evaluations.harness.trend import (
    LOWEST_RATIO,
    MINIMUM_SAMPLES,
    StepCriteria,
    TrendReport,
    Verdict,
    check_trend,
    refuse_a_condemned_run,
    steps_after_the_first,
)


__all__ = [
    "BLOCK_BYTES",
    "DARWIN",
    "DARWIN_ARGV",
    "LINUX",
    "LINUX_ARGV",
    "LOWEST_RATIO",
    "MINIMUM_SAMPLES",
    "REPEATS",
    "Block",
    "BlockPlan",
    "Calibration",
    "DatasetGrid",
    "DropCaches",
    "DropCommand",
    "DropRecord",
    "Measurement",
    "PlatformDropCaches",
    "Prepared",
    "Probe",
    "RunParameters",
    "StepCriteria",
    "Timed",
    "Timing",
    "TrendReport",
    "Verdict",
    "block_walk",
    "calibrate",
    "check_one_item_count",
    "check_trend",
    "first_item",
    "full_read",
    "items_per_block",
    "measure_dataset",
    "measure_pairing",
    "measure_size",
    "measure_task",
    "median_ns",
    "operation_for",
    "plan_blocks",
    "platform_drop_command",
    "prepare",
    "record_plan",
    "refuse_a_condemned_run",
    "repeat",
    "replay",
    "sequential_walk",
    "steps_after_the_first",
]
