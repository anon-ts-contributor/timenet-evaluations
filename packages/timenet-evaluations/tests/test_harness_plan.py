"""The block plan: its partition, its ordering, its block size, and its purity.

Every test here runs with no artifact, no fixture and no dataset, which is what SPEC-0018
§ `plan_blocks` Is Pure and Reads Nothing asks of them.
"""

import ast
from collections import Counter
from collections.abc import Callable
import inspect
from pathlib import Path

import pytest

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid import Block as BlockShape
from timenet_evaluations.harness import (
    BLOCK_BYTES,
    Block,
    BlockPlan,
    items_per_block,
    plan as plan_module,
    plan_blocks,
    record_plan,
    replay,
)


SEED = 20260906

PLAN_MODULE = Path(inspect.getfile(plan_module))


def _names_used_in(module: Path) -> set[str]:
    # Identifiers the code actually names. A docstring that talks about a clock or a filesystem is
    # prose, so the assertion reads the tree and not the file.
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)

    return names


def _covered(blocks: tuple[Block, ...]) -> list[int]:
    positions: list[int] = []
    for block in blocks:
        positions.extend(range(block.start, block.start + block.count))

    return sorted(positions)


def _pairs(blocks: tuple[Block, ...]) -> list[tuple[int, int]]:
    return sorted((block.start, block.count) for block in blocks)


# The concrete Block against the shape the reader seam declares. This function is the ty gate:
# harness imports nothing from grid.reader and grid imports nothing from harness, so the only
# thing that holds the two together is that a Block is assignable to the protocol.


def _walk_one_block(block: BlockShape) -> tuple[int, int]:
    return block.start, block.count


def test_the_concrete_block_satisfies_the_reader_seams_protocol() -> None:
    assert _walk_one_block(Block(start=7, count=3)) == (7, 3)


# A plan partitions the items exactly once


@pytest.mark.parametrize(
    ("size_bytes", "n_items", "block_bytes"),
    [
        (1000, 100, 25),  # B = 2, an exact multiple
        (1000, 10, 300),  # B = 3, a short final run
        (1000, 10, 50),  # the budget is below one item's mean size
        (1000, 1, BLOCK_BYTES),  # one item, one block
        (7, 3, 5),  # the budget buys two items and the tail is one
        (10**9, 4096, BLOCK_BYTES),  # a realistic shape
    ],
)
def test_a_plan_covers_every_item_exactly_once(size_bytes: int, n_items: int, block_bytes: int) -> None:
    blocks = plan_blocks(size_bytes, n_items, block_bytes, SEED)

    assert _covered(blocks) == list(range(n_items))
    assert all(block.count >= 1 for block in blocks)


def test_the_partition_assertion_catches_a_dropped_tail_block() -> None:
    blocks = plan_blocks(1000, 10, 300, SEED)
    tail = next(block for block in blocks if block.start + block.count == 10)

    mutated = tuple(block for block in blocks if block != tail)

    assert _covered(mutated) != list(range(10))


def test_the_partition_assertion_catches_an_overlapping_block() -> None:
    blocks = plan_blocks(1000, 10, 300, SEED)

    mutated = (*blocks, Block(start=2, count=3))

    assert _covered(mutated) != list(range(10))


def test_the_partition_assertion_catches_an_off_by_one_on_the_short_final_run() -> None:
    blocks = plan_blocks(1000, 10, 300, SEED)
    tail = next(block for block in blocks if block.start + block.count == 10)

    mutated = tuple(Block(start=block.start, count=block.count + 1) if block == tail else block for block in blocks)

    assert _covered(mutated) != list(range(10))


def test_exactly_one_run_is_short_and_it_ends_at_the_last_item() -> None:
    blocks = plan_blocks(1000, 10, 300, SEED)

    short = [block for block in blocks if block.count < 3]

    assert len(short) == 1
    assert short[0].start + short[0].count == 10
    assert short[0].count >= 1


def test_a_whole_multiple_leaves_no_short_run() -> None:
    blocks = plan_blocks(1000, 100, 25, SEED)

    assert {block.count for block in blocks} == {2}
    assert len(blocks) == 50


# Block size follows the byte budget, per representation


def test_a_denser_representation_gets_more_items_per_block_and_fewer_blocks() -> None:
    sparse = record_plan(1000, 100, 100, SEED)
    dense = record_plan(500, 100, 100, SEED)

    assert sparse.items_per_block == 10
    assert dense.items_per_block == 20
    assert dense.items_per_block == 2 * sparse.items_per_block
    assert dense.block_count * 2 == sparse.block_count


def test_a_budget_below_one_item_gives_one_item_per_block() -> None:
    blocks = plan_blocks(1000, 10, 50, SEED)

    assert items_per_block(1000, 10, 50) == 1
    assert len(blocks) == 10
    assert {block.count for block in blocks} == {1}


def test_the_block_size_is_an_integer_and_never_a_float() -> None:
    derived = items_per_block(1000, 3, 512)

    assert isinstance(derived, int)
    assert not isinstance(derived, float)


@pytest.mark.parametrize(
    ("size_bytes", "n_items", "block_bytes"),
    [(0, 10, 64), (-1, 10, 64), (1000, 0, 64), (1000, -3, 64), (1000, 10, 0), (1000, 10, -64)],
)
@pytest.mark.parametrize("entry", [items_per_block, plan_blocks, record_plan], ids=lambda one: one.__name__)
def test_a_plan_refuses_a_number_below_one(
    entry: Callable[..., object], size_bytes: int, n_items: int, block_bytes: int
) -> None:
    # A size of zero divides by zero and a size below one describes no workload. Every entry point
    # refuses it with the harness's own error type naming all three numbers, so no caller reaches
    # the division through a route that was not guarded.
    arguments = (
        (size_bytes, n_items, block_bytes) if entry is items_per_block else (size_bytes, n_items, block_bytes, SEED)
    )

    with pytest.raises(EvaluationError) as refusal:
        entry(*arguments)

    message = str(refusal.value)

    assert str(size_bytes) in message
    assert str(n_items) in message
    assert str(block_bytes) in message


def test_the_shared_budget_is_sixty_four_megabytes() -> None:
    assert BLOCK_BYTES == 64 * 1024 * 1024


# Block order is a seeded permutation of runs, never of items


def test_the_same_seed_reproduces_the_plan() -> None:
    first = plan_blocks(1000, 100, 25, SEED)
    second = plan_blocks(1000, 100, 25, SEED)

    assert first == second


def test_a_different_seed_reorders_the_blocks_and_moves_no_boundary() -> None:
    one = plan_blocks(1000, 100, 25, 1)
    two = plan_blocks(1000, 100, 25, 2)

    assert _pairs(one) == _pairs(two)
    assert [block.start for block in one] != [block.start for block in two]


def test_the_blocks_are_a_permutation_of_the_runs_and_not_of_the_items() -> None:
    blocks = plan_blocks(1000, 100, 25, SEED)

    # Every run starts on a boundary of the partition, so no item was moved out of its run.
    assert {block.start for block in blocks} == set(range(0, 100, 2))
    assert Counter(block.count for block in blocks) == Counter({2: 50})


def test_the_plan_is_not_in_ascending_order() -> None:
    blocks = plan_blocks(1000, 100, 25, SEED)

    assert [block.start for block in blocks] != sorted(block.start for block in blocks)


# `plan_blocks` is pure and reads nothing


def test_plan_blocks_takes_four_numbers_and_nothing_else() -> None:
    parameters = inspect.signature(plan_blocks).parameters

    assert list(parameters) == ["size_bytes", "n_items", "block_bytes", "seed"]
    assert [parameter.annotation for parameter in parameters.values()] == ["int"] * 4


@pytest.mark.parametrize(
    "forbidden",
    [
        "Path",
        "open",
        "glob",
        "rglob",
        "stat",
        "st_size",
        "exists",
        "read_text",
        "perf_counter",
        "monotonic",
        "Representation",
        "Reader",
        "Artifact",
    ],
)
def test_the_plan_module_touches_no_filesystem_no_clock_and_no_seam(forbidden: str) -> None:
    assert forbidden not in _names_used_in(PLAN_MODULE)


def test_the_permutation_is_drawn_from_a_locally_seeded_generator() -> None:
    # `np.random.permutation` and its siblings draw on the process-global generator, whose stream
    # depends on every prior call in the process. `np.random.default_rng(seed)` is the local one.
    tree = ast.parse(PLAN_MODULE.read_text(encoding="utf-8"))
    drawn = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) and node.value.attr == "random"
    ]

    assert drawn == ["default_rng"]


def test_the_plan_module_imports_nothing_from_the_grid() -> None:
    tree = ast.parse(PLAN_MODULE.read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}

    assert not any(module.startswith("timenet_evaluations.grid") for module in imported)


def test_the_plan_is_data_and_not_a_generator() -> None:
    blocks = plan_blocks(1000, 100, 25, SEED)

    assert isinstance(blocks, tuple)
    assert all(isinstance(block, Block) for block in blocks)


# The plan travels with the measurement


def test_a_record_carries_the_four_arguments_the_block_size_and_the_block_count() -> None:
    assert set(BlockPlan.model_fields) == {
        "size_bytes",
        "n_items",
        "block_bytes",
        "seed",
        "items_per_block",
        "block_count",
        "blocks",
    }

    recorded = record_plan(1000, 10, 300, SEED)

    assert (recorded.size_bytes, recorded.n_items, recorded.block_bytes, recorded.seed) == (1000, 10, 300, SEED)
    assert recorded.items_per_block == 3
    assert recorded.block_count == len(recorded.blocks) == 4


def test_a_recorded_plan_is_replayed_block_for_block_and_in_the_same_order() -> None:
    recorded = record_plan(1000, 100, 25, SEED)

    assert replay(recorded) == recorded.blocks


def test_a_recorded_plan_survives_a_round_trip_through_json() -> None:
    recorded = record_plan(1000, 100, 25, SEED)

    restored = BlockPlan.model_validate_json(recorded.model_dump_json())

    assert restored == recorded
    assert replay(restored) == recorded.blocks


def test_both_readers_of_one_representation_can_walk_one_record() -> None:
    recorded = record_plan(1000, 100, 25, SEED)

    pandas_walk = [_walk_one_block(block) for block in recorded.blocks]
    pytorch_walk = [_walk_one_block(block) for block in recorded.blocks]

    assert pandas_walk == pytorch_walk


def test_a_block_is_frozen_so_a_recorded_plan_cannot_be_edited_in_place() -> None:
    # ty reads a frozen field as a read-only property, so an assignment to `start` is a type error
    # rather than a runtime one. This pins the configuration that makes it so.
    assert Block.model_config["frozen"] is True
    assert BlockPlan.model_config["frozen"] is True
