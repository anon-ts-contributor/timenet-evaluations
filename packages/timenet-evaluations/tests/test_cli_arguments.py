"""The argument surface: what one run accepts, and every refusal it makes from ``argv`` alone.

Two layers are exercised here and they fail differently. The parser converts, and a value it cannot
convert ends the process through argparse with status 2 and a usage line. The four checks are
preconditions of the run, and each raises the package's own error type with a traceback, which is
status 1. The distinction is the point: a typo in a count and a dataset nothing is registered for
are not the same kind of mistake, and an operator reads them apart by how the process ended.

Nothing here names a dataset. The registered name is read out of the registry, and the tests that
need two or three datasets add names to it for their own duration. A test written against
``sleep-edfx`` would be a test of the one connector that ships rather than of a run over N of them.

No test starts a process, so nothing here compares the two entry points by running them.
``test_cli_entry_points.py`` holds what can be asserted about that pair without one.
"""

import ast
import inspect
from pathlib import Path
import sys

import pytest

from timenet_evaluations import cli as cli_module
from timenet_evaluations.arguments import DATASET_FORM, SEPARATOR, check_datasets, parse_dataset
from timenet_evaluations.cli import build_parser, main, run_evaluation
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.harness.plan import BLOCK_BYTES
from timenet_evaluations.harness.repeat import REPEATS
from timenet_evaluations.source.registry import CONNECTORS


REGISTERED = sorted(CONNECTORS)[0]
"""One name a connector is registered under, read out of the registry rather than written here."""

PARSER_STATUS = 2
"""What argparse exits with when it refuses a value. A failed run raises instead, so it ends with
the interpreter's own status for an uncaught exception, which is 1."""

SUPERSEDED = (["--timef-only"], ["--warmups", "1"], ["--batch-size", "64"])

OPTIONS = {"--out", "--artifacts", "--repeats", "--warmup", "--block-bytes", "--seed"}

DEFAULTS = ("artifacts", "repeats", "warmup", "block_bytes", "seed")

CLI_SOURCE = Path(inspect.getfile(cli_module))


def _refusal(*texts: str) -> str:
    """The message one run's checks raise for the positionals given as text."""
    with pytest.raises(EvaluationError) as refusal:
        check_datasets([parse_dataset(text) for text in texts])

    return str(refusal.value)


# The pair: two halves, split on the first separator


def test_a_pair_carries_the_name_and_the_path_it_was_given(tmp_path: Path) -> None:
    pair = parse_dataset(f"{REGISTERED}{SEPARATOR}{tmp_path}")

    assert (pair.text, pair.name, pair.path) == (f"{REGISTERED}={tmp_path}", REGISTERED, tmp_path)


def test_the_split_is_on_the_first_separator_so_a_path_may_hold_one(tmp_path: Path) -> None:
    directory = tmp_path / "a=b"
    pair = parse_dataset(f"{REGISTERED}{SEPARATOR}{directory}")

    assert (pair.name, pair.path) == (REGISTERED, directory)


def test_the_positional_is_repeatable_and_each_entry_becomes_its_own_pair(tmp_path: Path) -> None:
    parsed = build_parser().parse_args([f"first={tmp_path}", f"second={tmp_path}", f"third={tmp_path}"])

    assert [pair.name for pair in parsed.datasets] == ["first", "second", "third"]


# A name is present, and it is never the directory's basename


def test_a_bare_path_is_refused_and_its_basename_is_taken_as_no_ones_name(tmp_path: Path) -> None:
    # The basename is a registered name, so a run that inferred one would accept this and measure a
    # dataset nobody named.
    directory = tmp_path / REGISTERED
    directory.mkdir()
    pair = parse_dataset(str(directory))

    assert pair.name == ""

    message = _refusal(str(directory))

    assert str(directory) in message
    assert DATASET_FORM in message
    # The basename is inside the argument the message quotes back. Everywhere else in the message
    # it must be absent, because the only way it could appear there is as a name the run invented.
    assert directory.name not in message.replace(str(directory), "")


def test_a_positional_with_a_name_and_no_path_is_refused_rather_than_read_as_the_working_directory() -> None:
    # `name=` splits into a name and an empty half, and an empty half converts to a path that is a
    # directory that exists. Both halves are required, so the pair never reaches that conversion.
    assert parse_dataset(f"{REGISTERED}{SEPARATOR}").name == ""

    assert DATASET_FORM in _refusal(f"{REGISTERED}{SEPARATOR}")


def test_a_positional_with_a_path_and_no_name_is_refused(tmp_path: Path) -> None:
    assert DATASET_FORM in _refusal(f"{SEPARATOR}{tmp_path}")


# The name has a registered connector


def test_an_unregistered_name_is_refused_and_the_registered_names_are_listed(tmp_path: Path) -> None:
    message = _refusal(f"not-a-registered-dataset={tmp_path}")

    assert "not-a-registered-dataset" in message
    assert str(sorted(CONNECTORS)) in message


def test_the_name_is_settled_before_the_path_is_looked_at(tmp_path: Path) -> None:
    # Both halves of this pair are wrong. The refusal names the connector rather than the path,
    # which is what says the checks run in the order the sequence fixes rather than pair by pair.
    absent = tmp_path / "absent"
    message = _refusal(f"not-a-registered-dataset={absent}")

    assert str(sorted(CONNECTORS)) in message
    assert str(absent) not in message


# No name appears twice


def test_one_name_given_twice_with_two_different_paths_is_refused(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    message = _refusal(f"{REGISTERED}={first}", f"{REGISTERED}={second}")

    assert REGISTERED in message
    assert str(first) in message
    assert str(second) in message


# Every path is a directory that exists, and every pair is checked


def test_the_third_pairs_missing_path_is_refused_although_the_first_two_are_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(CONNECTORS, "second", CONNECTORS[REGISTERED])
    monkeypatch.setitem(CONNECTORS, "third", CONNECTORS[REGISTERED])
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    third = tmp_path / "third"

    message = _refusal(f"{REGISTERED}={first}", f"second={second}", f"third={third}")

    assert str(third) in message


def test_a_path_that_is_a_regular_file_is_refused_with_the_failure_a_missing_one_gets(tmp_path: Path) -> None:
    regular = tmp_path / "regular"
    regular.write_text("", encoding="utf-8")
    absent = tmp_path / "absent"

    messages = [_refusal(f"{REGISTERED}={location}").replace(str(location), "<path>") for location in (regular, absent)]

    assert messages[0] == messages[1]


def test_the_run_makes_its_four_checks_before_it_does_anything_else() -> None:
    # Everything the four read is visible in the command line, so none of it may be deferred to the
    # point of use. The check is the first statement of the function, ahead of the refusal.
    tree = ast.parse(CLI_SOURCE.read_text(encoding="utf-8"))
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == run_evaluation.__name__
    )
    statements = [
        node for node in function.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ]

    assert ast.unparse(statements[0]) == "check_datasets(datasets)"


# The parser refuses a superseded flag, and refuses it as unrecognised


@pytest.mark.parametrize("flag", SUPERSEDED, ids=lambda flag: flag[0])
def test_a_flag_from_the_previous_protocol_exits_through_the_parser(flag: list[str], tmp_path: Path) -> None:
    # Not aliased, not ignored, not accepted as a no-op: a script written for the previous protocol
    # stops rather than producing figures under a scope it did not ask for.
    with pytest.raises(SystemExit) as refusal:
        build_parser().parse_args([f"{REGISTERED}={tmp_path}", *flag])

    assert refusal.value.code == PARSER_STATUS


def test_a_non_integer_count_exits_through_the_parser_with_its_own_status(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as refusal:
        build_parser().parse_args([f"{REGISTERED}={tmp_path}", "--repeats", "many"])

    assert refusal.value.code == PARSER_STATUS


def test_every_value_the_parser_converts_arrives_converted(tmp_path: Path) -> None:
    parsed = build_parser().parse_args(
        [
            f"{REGISTERED}={tmp_path}",
            "--out",
            "elsewhere",
            "--artifacts",
            "kept",
            "--repeats",
            "3",
            "--block-bytes",
            "16777216",
            "--seed",
            "7",
        ]
    )

    assert (parsed.out, parsed.artifacts) == (Path("elsewhere"), Path("kept"))
    assert (parsed.repeats, parsed.block_bytes, parsed.seed) == (3, 16777216, 7)
    assert parsed.datasets[0].path == tmp_path


# The help output: the module docstring, help on every option, and nothing else


def test_the_parser_describes_itself_with_the_module_docstring() -> None:
    assert build_parser().description is cli_module.__doc__


def test_the_parser_declares_no_epilogue_and_no_version_flag() -> None:
    parser = build_parser()
    declared = {option for action in parser._actions for option in action.option_strings}

    assert parser.epilog is None
    assert "--version" not in declared


def test_the_parser_declares_exactly_the_five_options_and_one_positional() -> None:
    parser = build_parser()
    declared = {option for action in parser._actions for option in action.option_strings}
    positionals = [action.dest for action in parser._actions if not action.option_strings]

    assert declared - {"-h", "--help"} == OPTIONS
    assert positionals == ["datasets"]


def test_every_argument_carries_help_text() -> None:
    # None of the five explains itself from its name, and `--warmup` in particular has to say that
    # it makes the run report a warm figure rather than the protocol's cold one.
    parser = build_parser()

    helps = {option: action.help or "" for action in parser._actions for option in action.option_strings}

    assert all(action.help for action in parser._actions)
    assert "warm" in helps["--warmup"]


# The two sets of defaults agree, and two of them are written once


def test_the_parsers_defaults_and_the_signatures_defaults_are_the_same(tmp_path: Path) -> None:
    # They are written twice, in the parser and in the signature, so a library caller who omits
    # them gets what the command line would have given. Nothing else checks that they agree.
    parsed = build_parser().parse_args([f"{REGISTERED}={tmp_path}"])
    signature = inspect.signature(run_evaluation)

    assert [getattr(parsed, name) for name in DEFAULTS] == [signature.parameters[name].default for name in DEFAULTS]


def test_the_two_defaults_that_belong_to_another_module_are_imported_rather_than_retyped(tmp_path: Path) -> None:
    parsed = build_parser().parse_args([f"{REGISTERED}={tmp_path}"])
    signature = inspect.signature(run_evaluation)

    assert (parsed.repeats, parsed.block_bytes) == (REPEATS, BLOCK_BYTES)
    assert (signature.parameters["repeats"].default, signature.parameters["block_bytes"].default) == (
        REPEATS,
        BLOCK_BYTES,
    )


def test_the_output_root_defaults_to_a_bare_relative_path(tmp_path: Path) -> None:
    # It is resolved against the working directory the run is started from, so the same command
    # from two directories writes two trees. It names no fixed location in the repository.
    default = build_parser().parse_args([f"{REGISTERED}={tmp_path}"]).out

    assert (default, default.is_absolute()) == (Path("results"), False)


def test_the_artifact_root_defaults_to_a_bare_relative_path_of_its_own(tmp_path: Path) -> None:
    # A run has three roots now: the source paths it reads, `--out` for the record, and
    # `--artifacts` for the converted copies. The second and the third are separate directories,
    # because the converted copy outlives the run that wrote it and the record does not.
    parsed = build_parser().parse_args([f"{REGISTERED}={tmp_path}"])

    assert (parsed.artifacts, parsed.artifacts.is_absolute()) == (Path("artifacts"), False)
    assert parsed.artifacts != parsed.out


def test_the_artifact_root_is_not_a_precondition_and_nothing_creates_it_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The four checks are about the `name=path` pairs. The artifact root is kept across runs, so a
    # previous run's copies are expected to be sitting in it and it is never required to exist, to
    # be empty, or to be a directory at the point the arguments are read.
    absent = tmp_path / "no-such-root"
    monkeypatch.setattr(sys, "argv", ["timenet-evaluations", f"{REGISTERED}={tmp_path}", "--artifacts", str(absent)])

    # The run gets past the argument checks and stops later, on this empty source directory, which
    # is the point: nothing about the artifact root refused it. The assertion used to read the root
    # back out of the grid refusal that stood where the run now stands; what it was really checking
    # is that the root is neither required nor created here, and that survives the refusal.
    with pytest.raises(EvaluationError) as refusal:
        main()

    assert "artifact root" not in str(refusal.value)
    assert not absent.exists()


def test_the_run_takes_the_datasets_and_no_single_source(tmp_path: Path) -> None:
    parameters = inspect.signature(run_evaluation).parameters

    assert "source" not in parameters
    assert "batch_size" not in parameters
    assert next(iter(parameters)) == "datasets"
