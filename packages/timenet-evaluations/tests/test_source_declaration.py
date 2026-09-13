"""What this dataset declares about itself, and the five assumptions the declaration turns on."""

import ast
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError
from pyhealth.tasks import SleepStagingSleepEDF
import pytest

from timenet_evaluations import source as source_package
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.source import LABEL, PATIENT, SIGNAL
from timenet_evaluations.source.sleep_edfx import (
    ASSUMPTIONS,
    CHUNK_DURATION_SECONDS,
    NAME,
    RATE_HZ,
    SAMPLES_PER_ITEM,
    UNIT,
    Assumption,
    check_reported_length,
    declare,
)
from timenet_evaluations.source.sleep_edfx_preparation import DISCLOSURES, Preparation, drain


SLEEP_EDFX_MODULE = Path(source_package.__file__).parent / "sleep_edfx.py"

N_ITEMS = 3
N_CHANNELS = 7
N_SAMPLES = 16
SOURCE = Path("/data/a-release-root")

ASSUMPTION_COUNT = 6
DECLARED_RATE_HZ = 100
DECLARED_DURATION_SECONDS = 30
DECLARED_SAMPLES = 3000


def _items(count: int, channels: int) -> list[dict[str, Any]]:
    """The items the sample object yields one at a time."""
    generator = np.random.default_rng(count)

    return [
        {
            SIGNAL: generator.standard_normal((channels, N_SAMPLES)),
            LABEL: position,
            PATIENT: f"{position:04d}",
        }
        for position in range(count)
    ]


def _prepared(*, items: int = N_ITEMS, channels: int = N_CHANNELS) -> Preparation:
    """What one untimed pass establishes, driven from items rather than from a corpus."""
    return drain(_items(items, channels), name=NAME, source=SOURCE)


def _code_of(module: Path, name: str) -> str:
    """Unparse one function's body, without its docstring, so prose is not read as behaviour."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            statements = node.body[1:] if ast.get_docstring(node) is not None else node.body
            return "\n".join(ast.unparse(statement) for statement in statements)

    raise AssertionError(f"{module.name} declares no {name}")


def _assigned(module: Path, name: str) -> ast.expr:
    """The expression one module-level name is bound to."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            assert node.value is not None
            return node.value

    raise AssertionError(f"{module.name} binds no {name}")


# The canonical item is the 30-second scored epoch


def test_the_unit_and_the_rate_are_the_two_facts_this_release_fixes() -> None:
    assert UNIT == "epoch"
    assert RATE_HZ == DECLARED_RATE_HZ
    assert CHUNK_DURATION_SECONDS == DECLARED_DURATION_SECONDS


def test_the_declared_count_is_the_number_the_untimed_pass_established() -> None:
    # The count is a property of the hypnograms of the recordings a run was given, so it is read
    # from what the preparation drained rather than written down here.
    declaration = declare(_prepared(items=11))

    assert declaration.item.count == 11


def test_the_declared_shape_is_the_channel_count_the_pass_observed() -> None:
    declaration = declare(_prepared(channels=2))

    assert declaration.item.shape[0] == 2


def test_the_declared_shape_is_the_rate_times_the_window_duration() -> None:
    # A change to either one that left the shape alone would declare a shape no read produces.
    declaration = declare(_prepared())

    assert SAMPLES_PER_ITEM == RATE_HZ * CHUNK_DURATION_SECONDS
    assert SAMPLES_PER_ITEM == DECLARED_SAMPLES
    assert declaration.item.shape == (N_CHANNELS, SAMPLES_PER_ITEM)


def test_the_sample_count_is_computed_from_the_two_facts_and_is_not_a_third_one() -> None:
    # Bound to a literal it would be a number that stayed still while the rate or the window
    # duration moved, and the shape would then disagree with both.
    bound = _assigned(SLEEP_EDFX_MODULE, "SAMPLES_PER_ITEM")

    assert isinstance(bound, ast.BinOp)
    assert {node.id for node in ast.walk(bound) if isinstance(node, ast.Name)} == {
        "RATE_HZ",
        "CHUNK_DURATION_SECONDS",
    }


def test_the_window_duration_declared_here_is_the_one_the_task_defaults_to() -> None:
    # The declaration tracks the task's own code rather than restates it from memory. The connector
    # passes no duration, so a library that moved its default would move the unit under this number.
    assert SleepStagingSleepEDF().chunk_duration == CHUNK_DURATION_SECONDS


def test_the_declaration_carries_what_the_preparation_disclosed() -> None:
    assert declare(_prepared()).disclosures == DISCLOSURES


def test_the_declaration_cannot_be_moved_after_it_is_made() -> None:
    declaration = declare(_prepared())

    with pytest.raises(ValidationError):
        declaration.rate_hz = 1  # ty: ignore[invalid-assignment]


def test_the_count_is_never_the_length_the_sample_object_reports_about_itself() -> None:
    # litdata's __len__ is get_len(num_workers, batch_size), so it is a function of the loader
    # configuration in force when it is read rather than a property of the dataset.
    for member in ("declare", "canonical_item"):
        body = _code_of(SLEEP_EDFX_MODULE, member)
        assert "len(" not in body, f"{member} takes a length from an object that reports one"
        assert "samples" not in body, f"{member} reaches the sample object"


def test_the_declaration_reads_the_source_it_was_asked_about_and_no_stored_path() -> None:
    body = _code_of(SLEEP_EDFX_MODULE, "canonical_item")

    assert "check_source_root(source)" in body
    assert "prepare(source, name=NAME)" in body


def test_the_declaration_is_not_derived_from_a_representation_or_a_reader() -> None:
    # The untimed preparation is the one pass over this dataset that is neither, which is what makes
    # it the only honest source of the count.
    body = _code_of(SLEEP_EDFX_MODULE, "canonical_item")

    for reached in ("open_pandas", "open_torch", "set_task", "get_dataloader"):
        assert reached not in body, f"the declaration is derived through {reached}"


def test_a_reported_length_equal_to_the_declared_count_is_accepted() -> None:
    declaration = declare(_prepared(items=11))

    assert check_reported_length(11, declaration, source=SOURCE) is None


def test_a_reported_length_that_disagrees_is_refused_naming_both_numbers() -> None:
    declaration = declare(_prepared(items=11))

    with pytest.raises(EvaluationError) as failure:
        check_reported_length(9, declaration, source=SOURCE)

    message = str(failure.value)
    assert NAME in message
    assert str(SOURCE) in message
    assert "11" in message
    assert "9" in message


# The assumptions this connector makes are recorded


def test_all_six_assumptions_are_recorded_beside_the_declaration() -> None:
    assert len(ASSUMPTIONS) == ASSUMPTION_COUNT
    assert all(isinstance(assumption, Assumption) for assumption in ASSUMPTIONS)
    assert _assigned(SLEEP_EDFX_MODULE, "ASSUMPTIONS") is not None


def test_every_assumption_states_what_breaks_when_it_is_wrong() -> None:
    # An assumption recorded without its consequence is a comment. One recorded with it is a check a
    # reviewer can apply to a release nobody here has seen.
    for assumption in ASSUMPTIONS:
        assert assumption.statement.strip()
        assert assumption.consequence.strip()
        assert assumption.consequence != assumption.statement


@pytest.mark.parametrize(
    "phrase",
    ["100 Hz", "30 seconds", "same channels", "Movement time", "cassette subset"],
    ids=["rate", "window", "channels", "dropped", "subset"],
)
def test_the_five_assumptions_are_the_five_the_declaration_turns_on(phrase: str) -> None:
    stating = [assumption for assumption in ASSUMPTIONS if phrase in assumption.statement]

    assert len(stating) == 1, f"{phrase!r} is stated by {len(stating)} assumptions"


def test_the_rate_is_recorded_as_a_property_of_the_read_and_not_as_a_resampling_asked_for() -> None:
    # Nothing here asks PyHealth for a resampling, and nothing here would notice if the release
    # changed. Describing it the other way was the deleted constant's error.
    rate = next(assumption for assumption in ASSUMPTIONS if "100 Hz" in assumption.statement)

    assert "read_raw_edf" in rate.statement
    assert "highest rate" in rate.statement
    assert "Nothing here asks for a resampling" in rate.statement
    assert "resampled to" not in SLEEP_EDFX_MODULE.read_text(encoding="utf-8")


def test_the_channel_assumption_names_the_failure_a_different_channel_set_produces() -> None:
    # A channel count that changes stops the untimed preparation, because the alternative was a
    # padded channel nothing downstream could see. A sample count that changes still surfaces as a
    # numpy shape error inside the untimed conversion.
    channels = next(assumption for assumption in ASSUMPTIONS if "same channels" in assumption.statement)

    assert "compares the channel count" in channels.statement
    assert "stops the run" in channels.consequence
    assert "numpy" in channels.consequence
    assert "conversion" in channels.consequence


def test_the_subset_assumption_says_a_second_subset_is_a_second_registered_name() -> None:
    subset = next(assumption for assumption in ASSUMPTIONS if "cassette subset" in assumption.statement)

    assert "telemetry" in subset.consequence.lower()
    assert "registered name" in subset.consequence


def test_an_assumption_cannot_be_edited_after_it_is_recorded() -> None:
    with pytest.raises(AttributeError):
        ASSUMPTIONS[0].statement = ""  # ty: ignore[invalid-assignment]
