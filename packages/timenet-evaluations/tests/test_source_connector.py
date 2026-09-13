"""The four members a connector exposes, the registry that dispatches to them, and the seam."""

import ast
import inspect
from pathlib import Path
import re

import pandas as pd
import pytest
from torch.utils.data import Dataset

from timenet_evaluations import source as source_package
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.parsing import ParsingPath
from timenet_evaluations.source import (
    CONNECTOR_MEMBERS,
    CONNECTORS,
    Connector,
    DatasetDeclaration,
    ParsedItems,
    connector_for,
)
from timenet_evaluations.source.sleep_edfx import (
    NAME,
    RECORDINGS_DIRECTORY,
    SUBJECT_SPREADSHEET,
    SleepEdfxConnector,
    check_source_root,
)


SOURCE_PACKAGE = Path(source_package.__file__).parent
SOURCE_MODULES = sorted(SOURCE_PACKAGE.glob("*.py"))
CONNECTOR_MODULE = SOURCE_PACKAGE / "connector.py"
REGISTRY_MODULE = SOURCE_PACKAGE / "registry.py"
SLEEP_EDFX_MODULE = SOURCE_PACKAGE / "sleep_edfx.py"
# Every module the one connector is split across, discovered rather than listed. The seam is that
# the dataset library may be named only beneath source/, in this connector, and not that exactly one
# file may name it: the connector outgrows one module and the audit must still hold over all of it.
SLEEP_EDFX_MODULES = sorted(SOURCE_PACKAGE.glob("sleep_edfx*.py"))

PACKAGE_ROOT = SOURCE_PACKAGE.parent
PACKAGE_MODULES = sorted(PACKAGE_ROOT.rglob("*.py"))

MEMBER_COUNT = 4
SECOND = "second-dataset"


def _public_members(protocol: type) -> set[str]:
    declared = {name for name in vars(protocol) if not name.startswith("_")}
    annotated = {name for name in getattr(protocol, "__annotations__", {}) if not name.startswith("_")}
    return declared | annotated


def _code_of(module: Path, name: str) -> str:
    """Unparse one function's body, without its docstring, so prose is not read as behaviour."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            statements = node.body[1:] if ast.get_docstring(node) is not None else node.body
            return "\n".join(ast.unparse(statement) for statement in statements)

    raise AssertionError(f"{module.name} declares no {name}")


def _imported_modules(module: Path) -> set[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)

    return names


def _imported_roots(module: Path) -> set[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    return roots


class _WholeConnector:
    """A connector as a second dataset's author would write it, with every member in place.

    Every member raises when it is called, so a test that gets past the registry proves that the
    registry called none of them.
    """

    @staticmethod
    def load_for_conversion(source: Path) -> pd.DataFrame:
        raise AssertionError(source)

    @staticmethod
    def open_pandas(path: Path) -> ParsedItems:
        raise AssertionError(path)

    @staticmethod
    def open_torch(path: Path) -> Dataset[object]:
        raise AssertionError(path)

    @staticmethod
    def canonical_item(source: Path) -> DatasetDeclaration:
        raise AssertionError(source)


def _tree_below(root: Path) -> dict[str, tuple[int, int]]:
    """Every path below a directory, with the size and the modification time of each."""
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns) for path in sorted(root.rglob("*"))
    }


@pytest.fixture
def release_root(tmp_path: Path) -> Path:
    """A directory shaped like the root of the release: the spreadsheet and the recordings beside it."""
    root = tmp_path / "a-name-that-is-not-the-registered-one"
    (root / RECORDINGS_DIRECTORY).mkdir(parents=True)
    (root / SUBJECT_SPREADSHEET).write_bytes(b"")

    return root


def _short_of(member: str) -> type:
    """Build a connector that declares every member but the one named."""
    kept = {name: vars(_WholeConnector)[name] for name in CONNECTOR_MEMBERS if name != member}

    return type("ShortConnector", (), kept)


# A connector exposes exactly four members


def test_the_protocol_declares_exactly_the_four_members_and_no_fifth() -> None:
    # A member added here is a member the next dataset pays for, and its author is not in the room.
    assert _public_members(Connector) == set(CONNECTOR_MEMBERS)
    assert len(CONNECTOR_MEMBERS) == MEMBER_COUNT


def test_the_rate_is_not_a_fifth_member_and_travels_inside_the_declaration() -> None:
    # A value the conversion reads is not a call a reader makes.
    assert "rate_hz" not in _public_members(Connector)
    assert "rate_hz" in DatasetDeclaration.model_fields


def test_the_registered_name_is_not_a_member_of_the_connector_object() -> None:
    # The name is the key of the registry entry, so it costs no room on the protocol surface.
    assert "name" not in _public_members(Connector)
    assert not hasattr(SleepEdfxConnector, "name")


def test_the_shipped_connector_supplies_all_four_members() -> None:
    for member in CONNECTOR_MEMBERS:
        assert hasattr(SleepEdfxConnector, member), f"the shipped connector is short of {member}"


def test_the_declaration_is_asked_about_one_source_and_answers_from_no_stored_state() -> None:
    # One connector is registered per name and one instance serves every run, so a member that
    # answered without a source could only answer from a count stored for some earlier path.
    signature = inspect.signature(SleepEdfxConnector.canonical_item)

    assert list(signature.parameters) == ["source"]
    assert not isinstance(vars(SleepEdfxConnector)["canonical_item"], property)
    assert not isinstance(vars(Connector)["canonical_item"], property)


def test_the_shipped_connector_satisfies_the_grid_parsing_path_by_shape() -> None:
    # ty is the gate: this function is annotated against the grid's own protocol, and the connector
    # imports nothing from it. A renamed member is a type-check diagnostic, not a failed assertion.
    def needs_a_parsing_path(parsing_path: ParsingPath) -> ParsingPath:
        return parsing_path

    connector = SleepEdfxConnector()

    assert needs_a_parsing_path(connector) is connector


def test_nothing_in_the_source_package_imports_the_grid_parsing_module() -> None:
    # The dependency runs one way, source to grid, and the shape is the whole of what they share.
    for module in SOURCE_MODULES:
        assert "timenet_evaluations.grid.parsing" not in _imported_modules(module), (
            f"{module.name} imports the grid's parsing module"
        )


@pytest.mark.parametrize(("member", "forbidden"), [("open_torch", "open_pandas"), ("open_pandas", "open_torch")])
def test_neither_open_is_built_on_the_other(member: str, forbidden: str) -> None:
    # An open for the torch form built on the pandas one puts a DataFrame build inside a cell whose
    # figure is supposed to describe the release's torch path and nothing else.
    assert forbidden not in _code_of(SLEEP_EDFX_MODULE, member)


# The registry refuses before the first cell


def test_the_registry_returns_the_connector_a_name_selects() -> None:
    assert connector_for(NAME) is CONNECTORS[NAME]
    assert isinstance(connector_for(NAME), SleepEdfxConnector)


def test_a_name_with_no_registered_connector_is_refused_naming_what_is_registered() -> None:
    with pytest.raises(EvaluationError) as failure:
        connector_for("not-a-dataset")

    message = str(failure.value)
    assert "not-a-dataset" in message
    assert NAME in message


@pytest.mark.parametrize("member", CONNECTOR_MEMBERS)
def test_a_connector_short_of_a_member_is_refused_by_the_registry(monkeypatch: pytest.MonkeyPatch, member: str) -> None:
    # Refused here rather than at the first cell that would have called it: that cell has already
    # cost a load, a conversion and every cache drop taken before it, and the grid it prints is
    # short of cells it claims to cover.
    monkeypatch.setitem(CONNECTORS, SECOND, _short_of(member)())

    with pytest.raises(EvaluationError) as failure:
        connector_for(SECOND)

    message = str(failure.value)
    assert member in message
    assert "ShortConnector" in message


def test_a_whole_connector_is_accepted_without_any_member_being_called(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every member of the fixture raises when it is called, so acceptance proves none of them ran.
    monkeypatch.setitem(CONNECTORS, SECOND, _WholeConnector())

    assert isinstance(connector_for(SECOND), _WholeConnector)


def test_the_registry_is_hand_written_and_discovers_nothing() -> None:
    # A connector that arrived because it happened to be installed would make two runs on one
    # machine measure different datasets, with no field on the result to record that they had.
    roots = _imported_roots(REGISTRY_MODULE)

    for discovery in ("importlib", "pkgutil", "glob", "pkg_resources"):
        assert discovery not in roots, f"the registry discovers connectors through {discovery}"


def test_the_registry_knows_the_connector_and_the_protocol_and_nothing_else() -> None:
    # Adding a dataset is a connector and one line here. It is not an edit to a reader, to a
    # representation, to the harness, or to the result model.
    assert _imported_roots(REGISTRY_MODULE) == {"__future__", "timenet_evaluations"}


# The conversion input is read here, once per dataset, and is never timed


def test_the_untimed_load_takes_a_source_and_nothing_else() -> None:
    # batch_size is gone: how the values are drained is private to the connector and never changes
    # what ends up in the frame.
    signature = inspect.signature(SleepEdfxConnector.load_for_conversion)

    assert list(signature.parameters) == ["source"]


def test_the_retired_names_are_gone_from_the_package() -> None:
    for retired in ("load_frame", "RATE_HZ"):
        assert not hasattr(source_package, retired), f"{retired} still has a home in the package"


def test_nothing_in_the_source_package_starts_a_timer() -> None:
    # The conversion is not measured, so the read that feeds the conversion is not measured either.
    for module in SOURCE_MODULES:
        roots = _imported_roots(module)
        for clock in ("time", "timeit"):
            assert clock not in roots, f"{module.name} imports a clock"


def test_the_load_refuses_a_source_that_is_not_a_directory_before_it_touches_a_library(tmp_path: Path) -> None:
    missing = tmp_path / "not-there"

    with pytest.raises(EvaluationError) as failure:
        SleepEdfxConnector().load_for_conversion(missing)

    assert str(missing) in str(failure.value)


# Reading the release's own files for a measured cell is a different activity


@pytest.mark.parametrize("member", ["open_pandas", "open_torch"])
def test_a_timed_open_refuses_a_path_that_is_not_the_root_before_it_touches_a_library(
    tmp_path: Path, member: str
) -> None:
    # Each timed open runs the same root precondition the untimed load runs, and it runs it first.
    with pytest.raises(EvaluationError) as failure:
        getattr(SleepEdfxConnector(), member)(tmp_path)

    message = str(failure.value)
    assert str(tmp_path) in message
    assert SUBJECT_SPREADSHEET in message


@pytest.mark.parametrize("member", ["open_pandas", "open_torch"])
def test_no_timed_open_reads_what_the_untimed_load_produced(member: str) -> None:
    # A cell served from memory reports an in-memory attribute access under a column that says read
    # from disk, and it is several times faster with no structural tell.
    assert "load_for_conversion" not in _code_of(SLEEP_EDFX_MODULE, member)


# The registered name is sleep-edfx, and the root it is bound to


def test_the_registered_name_is_the_one_the_specification_fixes() -> None:
    # The string a user types before "=", and the dataset coordinate of every measurement key and
    # every storage key the run produces.
    assert NAME == "sleep-edfx"
    assert sorted(CONNECTORS) == [NAME]


def test_the_registered_name_is_a_literal_and_is_inferred_from_nothing() -> None:
    # Not from a directory basename, not from the contents of a path, and not from anything else
    # about the location the recordings happen to sit in.
    tree = ast.parse(SLEEP_EDFX_MODULE.read_text(encoding="utf-8"))
    bound = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "NAME" for target in node.targets)
    ]

    assert len(bound) == 1
    assert isinstance(bound[0], ast.Constant)
    assert bound[0].value == NAME


def test_a_root_holding_the_spreadsheet_and_the_recordings_is_accepted(release_root: Path) -> None:
    assert check_source_root(release_root) is None


def test_the_recordings_subdirectory_given_as_the_root_is_refused_by_layout(release_root: Path) -> None:
    # PyHealth resolves both names relative to the root it is given, so without this refusal the
    # run fails inside its metadata preparation with a missing file and no mention of the layout.
    recordings = release_root / RECORDINGS_DIRECTORY

    with pytest.raises(EvaluationError) as failure:
        check_source_root(recordings)

    message = str(failure.value)
    assert NAME in message
    assert str(recordings) in message
    assert SUBJECT_SPREADSHEET in message
    assert RECORDINGS_DIRECTORY in message


def test_a_directory_that_is_neither_is_refused_the_same_way(tmp_path: Path) -> None:
    recordings = tmp_path / "root" / RECORDINGS_DIRECTORY
    recordings.mkdir(parents=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    with pytest.raises(EvaluationError) as subdirectory:
        check_source_root(recordings)
    with pytest.raises(EvaluationError) as neither:
        check_source_root(unrelated)

    assert str(subdirectory.value).replace(str(recordings), "") == str(neither.value).replace(str(unrelated), "")


@pytest.mark.parametrize("absent", [SUBJECT_SPREADSHEET, RECORDINGS_DIRECTORY])
def test_a_directory_short_of_one_entry_is_refused_naming_the_one_that_is_missing(
    release_root: Path, absent: str
) -> None:
    entry = release_root / absent
    if entry.is_dir():
        entry.rmdir()
    else:
        entry.unlink()

    with pytest.raises(EvaluationError) as failure:
        check_source_root(release_root)

    assert f"missing ['{absent}']" in str(failure.value)


def test_a_path_that_is_not_a_directory_is_refused_by_the_general_check_first(tmp_path: Path) -> None:
    # check_source_directory stays the general check and runs first, so a path that is not a
    # directory never reaches the layout refusal and is not described as a bad layout.
    missing = tmp_path / "not-there"

    with pytest.raises(EvaluationError) as failure:
        check_source_root(missing)

    message = str(failure.value)
    assert str(missing) in message
    assert SUBJECT_SPREADSHEET not in message


@pytest.mark.parametrize("accepted", [True, False])
def test_the_check_creates_and_modifies_nothing_below_the_path_it_was_given(release_root: Path, accepted: bool) -> None:
    # It is a directory listing. The write into the source directory is PyHealth's, it happens at
    # construction, and when it happens belongs to the untimed preparation.
    given = release_root if accepted else release_root / RECORDINGS_DIRECTORY
    before = _tree_below(release_root)

    if accepted:
        check_source_root(given)
    else:
        with pytest.raises(EvaluationError):
            check_source_root(given)

    assert _tree_below(release_root) == before


def test_the_check_does_not_construct_the_reference_loader_to_find_out() -> None:
    # Construction is what derives the metadata table into the source directory, and a precondition
    # that triggered it would decide, for the preparation, when that file appears.
    body = _code_of(SLEEP_EDFX_MODULE, "check_source_root")

    for reach in ("SleepEDFDataset", "set_task", "get_dataloader"):
        assert reach not in body, f"the root precondition reaches {reach}"


def test_the_untimed_load_refuses_a_path_that_is_not_the_root_before_it_touches_a_library(
    release_root: Path,
) -> None:
    with pytest.raises(EvaluationError) as failure:
        SleepEdfxConnector().load_for_conversion(release_root / RECORDINGS_DIRECTORY)

    assert SUBJECT_SPREADSHEET in str(failure.value)


# The dataset seam


def test_only_the_connector_imports_a_dataset_library() -> None:
    # The audit is an import audit and not a text search: an aliased import is what would let a
    # dataset dependency reach a module that is not supposed to have one.
    #
    # The connector is what may import it, and the connector is as many modules as it needs. The set
    # is discovered from the tree, so a module that named the library and was not one of them fails
    # here, and an empty set fails too rather than passing the audit by naming nothing.
    importers = {module for module in PACKAGE_MODULES if "pyhealth" in _imported_roots(module)}

    assert importers
    assert importers <= set(SLEEP_EDFX_MODULES), f"a module outside the connector imports the library: {importers}"


def test_every_line_that_names_the_dataset_library_in_the_connector_is_an_import() -> None:
    lines = [
        line
        for module in SLEEP_EDFX_MODULES
        for line in module.read_text(encoding="utf-8").splitlines()
        if "pyhealth" in line
    ]

    assert lines
    for line in lines:
        assert line.startswith(("import ", "from ")), f"the connector names the library outside an import: {line}"


def test_no_module_outside_the_source_package_names_the_dataset() -> None:
    # A reader that knew it was reading sleep data would be a reader that could not read the next
    # dataset.
    #
    # The terms are matched on word boundaries, and they are the dataset's own names rather than
    # the ordinary English words they contain. "sleep" alone rejects a docstring that says a test
    # needs no sleep, which is time.sleep and says nothing about any dataset; "epoch" alone rejects
    # a unix epoch. A rule that fires on those teaches the next reader to work around it.
    for module in PACKAGE_MODULES:
        if module.parent == SOURCE_PACKAGE:
            continue

        text = module.read_text(encoding="utf-8").lower()
        for term in ("pyhealth", "sleep-edf", "sleep_edf", "sleepedf", "edf"):
            assert not re.search(rf"\b{re.escape(term)}", text), (
                f"{module.relative_to(PACKAGE_ROOT)} names the dataset: {term}"
            )


def test_the_connector_protocol_module_names_no_dataset_and_no_dataset_library() -> None:
    # The protocol is what the second dataset's author reads, so it inherits nothing from the first.
    text = CONNECTOR_MODULE.read_text(encoding="utf-8").lower()

    for word in ("pyhealth", "sleep", "edf", "eeg", "epoch"):
        assert word not in text, f"the connector protocol names {word}"
