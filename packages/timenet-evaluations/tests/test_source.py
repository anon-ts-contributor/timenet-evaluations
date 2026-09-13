"""The frame contract, what a dataset declares about itself, and the seam that keeps a dataset in
one package."""

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
import torch

from timenet_evaluations import source as source_package
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.facts import DatasetFacts
from timenet_evaluations.grid.item import ItemDeclaration
from timenet_evaluations.source import (
    COLUMNS,
    LABEL,
    PATIENT,
    SIGNAL,
    Connector,
    DatasetDeclaration,
    as_row,
    check_frame_row_count,
    check_source_directory,
    signal_stack,
    unbatch,
)


N_ITEMS = 4
N_CHANNELS = 2
N_SAMPLES = 8
RATE = 100

SOURCE_PACKAGE = Path(source_package.__file__).parent
SOURCE_MODULES = sorted(SOURCE_PACKAGE.glob("*.py"))

PACKAGE_ROOT = SOURCE_PACKAGE.parent
PACKAGE_MODULES = sorted(PACKAGE_ROOT.rglob("*.py"))
CONTRACT_MODULE = SOURCE_PACKAGE / "contract.py"
DECLARATION_MODULE = SOURCE_PACKAGE / "declaration.py"


def _values() -> np.ndarray:
    generator = np.random.default_rng(0)
    return generator.standard_normal((N_ITEMS, N_CHANNELS, N_SAMPLES))


@pytest.fixture
def batch() -> dict[str, Any]:
    return {
        SIGNAL: _values(),
        LABEL: [f"stage-{position}" for position in range(N_ITEMS)],
        PATIENT: [f"subject-{position}" for position in range(N_ITEMS)],
    }


@pytest.fixture
def frame(batch: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(unbatch(batch))


@pytest.fixture
def declaration() -> DatasetDeclaration:
    return DatasetDeclaration(
        unit="window",
        item=ItemDeclaration(count=N_ITEMS, shape=(N_CHANNELS, N_SAMPLES)),
        rate_hz=RATE,
        facts=DatasetFacts(
            card_id="timenet/hello_world",
            modality="stub",
            modality_name="Stub",
            unit="volt",
            rate_hz=RATE,
            channels=tuple(f"c{index}" for index in range(N_CHANNELS)),
            target_schema="stub_class",
        ),
        disclosures={},
    )


# The frame contract: three columns, and the types they carry


def test_a_frame_carries_one_row_per_item_and_exactly_the_three_columns(frame: pd.DataFrame) -> None:
    assert len(frame) == N_ITEMS
    assert tuple(frame.columns) == COLUMNS


def test_the_column_constants_name_the_columns_the_contract_states() -> None:
    assert COLUMNS == (SIGNAL, LABEL, PATIENT) == ("signal", "label", "patient_id")


def test_a_float64_column_lands_in_the_frame_as_float32(batch: dict[str, Any]) -> None:
    # The cast belongs to the connector, so the conversion is entitled to assume it already happened.
    assert batch[SIGNAL].dtype == np.float64

    for row in unbatch(batch):
        assert row[SIGNAL].dtype == np.float32


def test_the_label_and_the_patient_are_coerced_to_str() -> None:
    batch = {
        SIGNAL: _values()[:1],
        LABEL: np.array([2]),
        PATIENT: torch.tensor([7]),
    }

    (row,) = unbatch(batch)

    assert row[LABEL] == "2"
    assert row[PATIENT] == "7"


def test_a_row_carries_the_three_columns_and_nothing_else(batch: dict[str, Any]) -> None:
    batch["recording_id"] = list(range(N_ITEMS))

    for row in unbatch(batch):
        assert set(row) == set(COLUMNS)


# Column names are reached through the constants


@pytest.mark.parametrize("literal", ["signal", "label", "patient_id"])
def test_no_module_outside_the_contract_writes_a_column_name_as_a_literal(literal: str) -> None:
    # A literal is a second copy of the contract that no rename finds.
    for module in PACKAGE_MODULES:
        if module == CONTRACT_MODULE:
            continue

        text = module.read_text(encoding="utf-8")
        assert f'"{literal}"' not in text, f"{module.name} names the column {literal!r}"
        assert f"'{literal}'" not in text, f"{module.name} names the column {literal!r}"


# Building one row from one item


@pytest.fixture
def item() -> dict[str, Any]:
    return {
        SIGNAL: _values()[0],
        LABEL: 2,
        PATIENT: "subject-0",
        "night": 1,
    }


def test_one_item_becomes_one_row_of_the_three_columns(item: dict[str, Any]) -> None:
    row = as_row(item)

    assert set(row) == set(COLUMNS)
    assert row[SIGNAL].shape == (N_CHANNELS, N_SAMPLES)
    assert row[SIGNAL].dtype == np.float32
    assert row[LABEL] == "2"
    assert row[PATIENT] == "subject-0"


@pytest.mark.parametrize("column", COLUMNS)
def test_an_item_missing_any_one_column_is_refused_rather_than_defaulted(item: dict[str, Any], column: str) -> None:
    del item[column]

    with pytest.raises(EvaluationError) as failure:
        as_row(item)

    assert column in str(failure.value)


def test_a_tensor_item_and_a_plain_item_produce_indistinguishable_rows(item: dict[str, Any]) -> None:
    # A dataset library can change what it hands back between versions, and that variation must not
    # reach the frame.
    from_tensor = as_row(
        {SIGNAL: torch.from_numpy(item[SIGNAL]), LABEL: torch.tensor(item[LABEL]), PATIENT: item[PATIENT]}
    )
    from_plain = as_row(item)

    assert from_tensor[LABEL] == from_plain[LABEL]
    assert from_tensor[PATIENT] == from_plain[PATIENT]
    assert from_tensor[SIGNAL].dtype == from_plain[SIGNAL].dtype
    assert np.array_equal(from_tensor[SIGNAL], from_plain[SIGNAL])


# Unbatching a collated batch


def test_a_well_formed_batch_is_split_into_one_row_per_item(batch: dict[str, Any]) -> None:
    rows = unbatch(batch)

    assert len(rows) == N_ITEMS
    for position, row in enumerate(rows):
        assert row[SIGNAL].shape == (N_CHANNELS, N_SAMPLES)
        assert row[LABEL] == f"stage-{position}"
        assert row[PATIENT] == f"subject-{position}"


def test_a_batch_whose_columns_disagree_is_refused_by_name_and_count(batch: dict[str, Any]) -> None:
    batch[LABEL] = batch[LABEL][:-1]

    with pytest.raises(EvaluationError) as failure:
        unbatch(batch)

    message = str(failure.value)
    assert f"{SIGNAL}={N_ITEMS}" in message
    assert f"{LABEL}={N_ITEMS - 1}" in message
    assert f"{PATIENT}={N_ITEMS}" in message


@pytest.mark.parametrize("column", COLUMNS)
def test_a_batch_missing_any_one_column_is_refused_rather_than_defaulted(batch: dict[str, Any], column: str) -> None:
    # The patient column used to be defaulted to empty strings, which fabricated a subject and
    # disarmed its own arm of the item-count check.
    del batch[column]

    with pytest.raises(EvaluationError) as failure:
        unbatch(batch)

    assert column in str(failure.value)


def test_a_tensor_column_and_a_sequence_column_produce_indistinguishable_rows(batch: dict[str, Any]) -> None:
    collated = {
        SIGNAL: torch.from_numpy(batch[SIGNAL]),
        LABEL: batch[LABEL],
        PATIENT: batch[PATIENT],
    }

    from_sequence = unbatch(batch)
    from_tensor = unbatch(collated)

    assert len(from_tensor) == len(from_sequence)
    for tensor_row, sequence_row in zip(from_tensor, from_sequence, strict=True):
        assert tensor_row[LABEL] == sequence_row[LABEL]
        assert tensor_row[PATIENT] == sequence_row[PATIENT]
        assert tensor_row[SIGNAL].dtype == sequence_row[SIGNAL].dtype
        np.testing.assert_array_equal(tensor_row[SIGNAL], sequence_row[SIGNAL])


# Stacking the signal column


def test_the_stack_is_float32_and_carries_the_promised_shape(frame: pd.DataFrame) -> None:
    stacked = signal_stack(frame)

    assert stacked.shape == (N_ITEMS, N_CHANNELS, N_SAMPLES)
    assert stacked.dtype == np.float32


def test_the_stack_casts_a_float64_column_without_a_caller_adding_one() -> None:
    frame = pd.DataFrame(
        {
            SIGNAL: [np.zeros((N_CHANNELS, N_SAMPLES), dtype=np.float64) for _ in range(N_ITEMS)],
            LABEL: ["a"] * N_ITEMS,
            PATIENT: ["b"] * N_ITEMS,
        }
    )

    assert signal_stack(frame).dtype == np.float32


def test_the_stack_refuses_a_frame_whose_items_are_not_two_dimensional() -> None:
    # np.stack builds a two-dimensional result from one-dimensional rows without complaint, and
    # this array is the reference the parity check compares both representations against.
    frame = pd.DataFrame(
        {
            SIGNAL: [np.zeros(N_SAMPLES, dtype=np.float32) for _ in range(N_ITEMS)],
            LABEL: ["a"] * N_ITEMS,
            PATIENT: ["b"] * N_ITEMS,
        }
    )

    with pytest.raises(EvaluationError) as failure:
        signal_stack(frame)

    assert f"({N_ITEMS}, {N_SAMPLES})" in str(failure.value)


# The canonical item is declared per dataset, in advance


def test_a_declaration_carries_the_unit_the_count_and_the_shape_of_one_item(
    declaration: DatasetDeclaration,
) -> None:
    assert declaration.unit == "window"
    assert declaration.item.count == N_ITEMS
    assert declaration.item.shape == (N_CHANNELS, N_SAMPLES)


def test_a_declaration_whose_count_disagrees_with_the_frame_is_refused_naming_both_numbers(
    frame: pd.DataFrame,
) -> None:
    coarser = DatasetDeclaration(
        unit="recording",
        item=ItemDeclaration(count=N_ITEMS - 1, shape=(N_CHANNELS, N_SAMPLES)),
        rate_hz=RATE,
        facts=DatasetFacts(
            card_id="timenet/hello_world",
            modality="stub",
            modality_name="Stub",
            unit="volt",
            rate_hz=RATE,
            channels=tuple(f"c{index}" for index in range(N_CHANNELS)),
            target_schema="stub_class",
        ),
        disclosures={},
    )

    with pytest.raises(EvaluationError) as failure:
        check_frame_row_count(frame, coarser, dataset="a-dataset")

    message = str(failure.value)
    assert "a-dataset" in message
    assert str(N_ITEMS - 1) in message
    assert str(N_ITEMS) in message


def test_a_frame_holding_the_declared_number_of_rows_is_accepted(
    frame: pd.DataFrame, declaration: DatasetDeclaration
) -> None:
    assert check_frame_row_count(frame, declaration, dataset="a-dataset") is None


def test_the_declaration_is_a_value_and_nothing_builds_one_from_what_a_run_produced() -> None:
    # A declaration derived from an artifact, a representation or a reader would let the thing
    # under test choose the denominator its own rates divide by.
    tree = ast.parse(DECLARATION_MODULE.read_text(encoding="utf-8"))

    returned = {
        ast.unparse(node.returns)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.returns is not None
    }
    assert DatasetDeclaration.__name__ not in returned

    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    for produced in ("artifact", "parsing", "reader", "registry", "representation", "representations"):
        assert not any(module.endswith(produced) for module in imported), f"the declaration reads {produced}"


def test_the_declaration_cannot_be_moved_once_it_has_been_made(declaration: DatasetDeclaration) -> None:
    # A declaration a run can move is a denominator a run can move. The field is named through a
    # variable because the type checker refuses the assignment outright, which is the point.
    for field in ("unit", "item", "rate_hz", "disclosures"):
        with pytest.raises(ValidationError):
            setattr(declaration, field, None)


def test_a_dataset_with_nothing_to_disclose_carries_an_empty_mapping(declaration: DatasetDeclaration) -> None:
    # Absence is the signal, so the field is stated rather than omitted: a report shows a statement
    # where there is one and prints nothing at all where there is not.
    assert declaration.disclosures == {}
    assert "disclosures" in DatasetDeclaration.model_fields


def test_a_declaration_carries_one_statement_per_representation_it_was_given() -> None:
    # The wording comes from the connector that did the preparation. Nothing downstream composes it
    # from a representation's name or from a path, so it has to travel on the declaration.
    disclosed = DatasetDeclaration(
        unit="window",
        item=ItemDeclaration(count=N_ITEMS, shape=(N_CHANNELS, N_SAMPLES)),
        rate_hz=RATE,
        facts=DatasetFacts(
            card_id="timenet/hello_world",
            modality="stub",
            modality_name="Stub",
            unit="volt",
            rate_hz=RATE,
            channels=tuple(f"c{index}" for index in range(N_CHANNELS)),
            target_schema="stub_class",
        ),
        disclosures={"a-representation": "these cells read something the run prepared"},
    )

    assert disclosed.disclosures["a-representation"] == "these cells read something the run prepared"
    assert "another-representation" not in disclosed.disclosures


def test_the_disclosures_are_not_a_fifth_member_of_the_connector() -> None:
    # A member added to the connector is a member every future dataset pays for, and its author is
    # not in the room. What a connector states without being called travels in the declaration.
    assert "disclosures" not in {name for name in vars(Connector) if not name.startswith("_")}


# The sample rate is declared with the dataset


def test_the_rate_is_declared_per_dataset_and_the_module_holds_no_constant_to_inherit(
    declaration: DatasetDeclaration,
) -> None:
    # A second dataset at another rate must not inherit the first one's number in silence.
    faster = DatasetDeclaration(
        unit="window",
        item=ItemDeclaration(count=N_ITEMS, shape=(N_CHANNELS, N_SAMPLES)),
        rate_hz=200,
        facts=DatasetFacts(
            card_id="timenet/hello_world",
            modality="stub",
            modality_name="Stub",
            unit="volt",
            rate_hz=RATE,
            channels=tuple(f"c{index}" for index in range(N_CHANNELS)),
            target_schema="stub_class",
        ),
        disclosures={},
    )

    assert declaration.rate_hz == RATE
    assert faster.rate_hz == 200

    tree = ast.parse(DECLARATION_MODULE.read_text(encoding="utf-8"))
    assert not [node for node in tree.body if isinstance(node, ast.Assign | ast.AnnAssign)]


def test_the_declaration_module_names_no_dataset() -> None:
    # It states what a declaration is, the way the contract states what a frame is.
    text = DECLARATION_MODULE.read_text(encoding="utf-8").lower()

    for word in ("pyhealth", "sleep", "edf", "eeg", "epoch"):
        assert word not in text, f"the declaration names {word}"


# The source directory precondition


def test_a_source_path_that_is_absent_is_refused_by_name(tmp_path: Path) -> None:
    missing = tmp_path / "not-there"

    with pytest.raises(EvaluationError) as failure:
        check_source_directory(missing)

    assert str(missing) in str(failure.value)


def test_a_source_path_that_is_a_regular_file_is_refused_the_same_way(tmp_path: Path) -> None:
    # A directory test, not an existence test: a file cannot supply a directory of recordings.
    missing = tmp_path / "not-there"
    regular_file = tmp_path / "recording.edf"
    regular_file.write_bytes(b"")

    with pytest.raises(EvaluationError) as absent:
        check_source_directory(missing)
    with pytest.raises(EvaluationError) as file_given:
        check_source_directory(regular_file)

    assert str(absent.value).replace(str(missing), "") == str(file_given.value).replace(str(regular_file), "")


def test_a_source_directory_that_exists_is_accepted(tmp_path: Path) -> None:
    assert check_source_directory(tmp_path) is None


# Uniform item shape is assumed, not enforced


def test_a_ragged_frame_loads_without_complaint_and_fails_only_when_it_is_stacked() -> None:
    # A dataloader collates each batch on its own, so items disagree across batches and never
    # inside one. unbatch casts each item's dtype and compares no shape against another's.
    rows = [
        *unbatch({SIGNAL: np.zeros((1, N_CHANNELS, N_SAMPLES)), LABEL: ["a"], PATIENT: ["x"]}),
        *unbatch({SIGNAL: np.zeros((1, N_CHANNELS, N_SAMPLES + 1)), LABEL: ["b"], PATIENT: ["y"]}),
    ]

    frame = pd.DataFrame(rows)
    assert len(frame) == 2

    with pytest.raises(ValueError):
        signal_stack(frame)


# Failures at the loading boundary are not absorbed


def test_nothing_in_the_source_package_absorbs_a_failure() -> None:
    for module in SOURCE_MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            assert not isinstance(node, ast.ExceptHandler), f"{module.name} catches an error"
            assert not isinstance(node, ast.Try), f"{module.name} wraps a boundary in a try"


def test_the_packages_own_preconditions_raise_the_harnesss_own_error(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError):
        check_source_directory(tmp_path / "not-there")
    with pytest.raises(EvaluationError):
        unbatch({SIGNAL: _values()})


# The seam


def test_every_import_in_the_source_package_sits_at_module_level() -> None:
    for module in SOURCE_MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    assert not isinstance(child, ast.Import | ast.ImportFrom), f"{module.name} defers an import"

        assert "PLC0415" not in module.read_text(encoding="utf-8"), f"{module.name} excuses a deferred import"


def test_pyhealth_is_named_beneath_the_source_package_and_nowhere_else() -> None:
    named_in = {module for module in PACKAGE_MODULES if "pyhealth" in module.read_text(encoding="utf-8")}

    assert named_in
    for module in named_in:
        assert module.parent == SOURCE_PACKAGE, f"{module} names a dataset library outside the seam"


def test_the_contract_module_names_no_dataset_and_no_dataset_library() -> None:
    # The contract is what a second connector inherits rather than copies.
    text = CONTRACT_MODULE.read_text(encoding="utf-8").lower()

    for word in ("pyhealth", "sleep", "edf", "eeg", "epoch"):
        assert word not in text, f"the frame contract names {word}"
