"""The cache drop: what it issues, what it records, and what it refuses.

Nothing in this module is evidence that the drop works. Every test drives a runner double, and a
double has no page cache, so no assertion here can tell a cold read from a warm one. What these
tests pin is the shape: that a command is issued, that its status is inspected rather than assumed,
that a failure stops the run, and that the record names what ran.
"""

import ast
from collections.abc import Sequence
import inspect
from pathlib import Path
import subprocess

import pytest

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, StorageKey, Task
from timenet_evaluations.harness import drop as drop_module
from timenet_evaluations.harness.drop import (
    DARWIN,
    DARWIN_ARGV,
    LINUX,
    LINUX_ARGV,
    DropCommand,
    DropRecord,
    PlatformDropCaches,
    platform_drop_command,
)


DATASET = "sleep-edfx"

DROP_MODULE = Path(inspect.getfile(drop_module))

STORAGE = StorageKey(dataset=DATASET, representation="timef")

CELL = Cell(dataset=DATASET, representation="timef", reader="pandas", task=Task.FULL_READ)


class RecordingRunner:
    """A runner that records the argv it was handed and returns a scripted status."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[tuple[str, ...]] = []
        self.flags: tuple[bool, bool] | None = None

    def __call__(
        self, args: Sequence[str], /, *, check: bool, capture_output: bool
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append(tuple(args))
        self.flags = (check, capture_output)
        return subprocess.CompletedProcess(args=list(args), returncode=self.returncode)


class AbsentRunner:
    """A runner standing in for a mechanism that is not on the machine."""

    def __call__(
        self, args: Sequence[str], /, *, check: bool, capture_output: bool
    ) -> subprocess.CompletedProcess[bytes]:
        raise FileNotFoundError(2, "No such file or directory", args[0])


# The command each platform is given


def test_macos_drops_through_purge_named_by_absolute_path() -> None:
    command = platform_drop_command(DARWIN)

    assert command.argv == ("/usr/sbin/purge",)
    assert command.platform == DARWIN


def test_linux_drops_through_a_write_to_the_proc_file() -> None:
    command = platform_drop_command(LINUX)

    assert command.argv == ("/bin/sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches")
    assert command.platform == LINUX


@pytest.mark.parametrize("platform", ["win32", "cygwin", "emscripten", ""])
def test_a_platform_with_neither_mechanism_is_refused(platform: str) -> None:
    with pytest.raises(EvaluationError) as refusal:
        platform_drop_command(platform)

    message = str(refusal.value)

    assert f"sys.platform {platform!r}" in message
    assert "/usr/sbin/purge" in message
    assert "/proc/sys/vm/drop_caches" in message
    # The refusal fires before any dataset, path or artifact exists, so it names no cell.
    assert "dataset '" not in message
    assert "path " not in message


def test_the_refusal_says_what_would_have_to_change() -> None:
    with pytest.raises(EvaluationError) as refusal:
        platform_drop_command("win32")

    assert DARWIN in str(refusal.value)
    assert LINUX in str(refusal.value)


def test_the_command_for_a_platform_is_pure_and_starts_nothing() -> None:
    # Building the command spawns no process, so a machine with neither mechanism can still assert
    # what each platform would be given.
    tree = ast.parse(DROP_MODULE.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "platform_drop_command"
    )
    names = {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}

    assert "run" not in names
    assert "Popen" not in names


# What the drop issues, and what it records


def test_the_drop_issues_the_command_it_was_given() -> None:
    runner = RecordingRunner()
    drop = PlatformDropCaches(DropCommand(argv=DARWIN_ARGV, platform=DARWIN), runner=runner)

    drop(at=CELL, path=Path("/data/timef"))

    assert runner.calls == [DARWIN_ARGV]


def test_the_drop_inspects_the_status_rather_than_letting_the_runner_raise() -> None:
    runner = RecordingRunner()
    drop = PlatformDropCaches(DropCommand(argv=LINUX_ARGV, platform=LINUX), runner=runner)

    drop(at=STORAGE, path=Path("/data/timef"))

    # check=False, so this module reads the status itself; capture_output=True, so the drop writes
    # nothing into a report.
    assert runner.flags == (False, True)


def test_the_record_names_the_command_that_ran_and_the_status_it_returned() -> None:
    drop = PlatformDropCaches(DropCommand(argv=LINUX_ARGV, platform=LINUX), runner=RecordingRunner())

    record = drop(at=STORAGE, path=Path("/data/timef"))

    assert record == DropRecord(
        command="/bin/sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'",
        exit_status=0,
        platform=LINUX,
    )


def test_the_record_is_frozen_so_it_cannot_be_edited_after_the_fact() -> None:
    # ty reads a frozen field as a read-only property, so an assignment is a type error rather than
    # a runtime one and cannot be asserted on here.
    assert DropCommand.model_config["frozen"] is True
    assert DropRecord.model_config["frozen"] is True


def test_a_failed_drop_produces_no_record_at_all() -> None:
    # The one property the whole module exists for: a command named in prose and never run has no
    # record, so a record cannot describe an intention.
    runner = RecordingRunner(returncode=1)
    drop = PlatformDropCaches(DropCommand(argv=DARWIN_ARGV, platform=DARWIN), runner=runner)

    with pytest.raises(EvaluationError):
        drop(at=CELL, path=Path("/data/timef"))


# A failing drop stops the run


def test_a_non_zero_exit_stops_the_run_and_names_the_status() -> None:
    runner = RecordingRunner(returncode=1)
    drop = PlatformDropCaches(DropCommand(argv=DARWIN_ARGV, platform=DARWIN), runner=runner)
    path = Path("/data/timef")

    with pytest.raises(EvaluationError) as refusal:
        drop(at=CELL, path=path)

    message = str(refusal.value)

    assert "/usr/sbin/purge" in message
    assert "exit status 1" in message
    assert f"platform {DARWIN!r}" in message
    assert str(path) in message


def test_a_non_zero_exit_names_all_four_coordinates_of_the_cell() -> None:
    drop = PlatformDropCaches(DropCommand(argv=DARWIN_ARGV, platform=DARWIN), runner=RecordingRunner(returncode=64))

    with pytest.raises(EvaluationError) as refusal:
        drop(at=CELL, path=Path("/data/timef"))

    message = str(refusal.value)

    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert "reader 'pandas'" in message
    assert "task 'full_read'" in message


def test_a_drop_before_a_storage_figure_names_the_two_coordinates_it_has() -> None:
    drop = PlatformDropCaches(DropCommand(argv=DARWIN_ARGV, platform=DARWIN), runner=RecordingRunner(returncode=1))

    with pytest.raises(EvaluationError) as refusal:
        drop(at=STORAGE, path=Path("/data/timef"))

    message = str(refusal.value)

    assert f"dataset {DATASET!r}" in message
    assert "reader '" not in message
    assert "task '" not in message


def test_a_missing_executable_stops_the_run_rather_than_reading_warm() -> None:
    drop = PlatformDropCaches(DropCommand(argv=DARWIN_ARGV, platform=DARWIN), runner=AbsentRunner())

    with pytest.raises(EvaluationError) as refusal:
        drop(at=CELL, path=Path("/data/timef"))

    message = str(refusal.value)

    assert "/usr/sbin/purge" in message
    assert "warm" in message


def test_a_missing_executable_keeps_the_error_it_came_from() -> None:
    drop = PlatformDropCaches(DropCommand(argv=DARWIN_ARGV, platform=DARWIN), runner=AbsentRunner())

    with pytest.raises(EvaluationError) as refusal:
        drop(at=CELL, path=Path("/data/timef"))

    assert isinstance(refusal.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("returncode", [1, 2, 64, 126, 127, -9])
def test_every_status_other_than_zero_stops_the_run(returncode: int) -> None:
    drop = PlatformDropCaches(
        DropCommand(argv=DARWIN_ARGV, platform=DARWIN), runner=RecordingRunner(returncode=returncode)
    )

    with pytest.raises(EvaluationError):
        drop(at=CELL, path=Path("/data/timef"))


# What this module must never do


def test_no_test_here_starts_the_platform_mechanism() -> None:
    # Every drop in this module goes through an injected runner. A test that ran `purge` would need
    # a privilege, would evict the machine's cache, and would still prove nothing.
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    calls = {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "run" not in calls
    assert "Popen" not in calls
    assert "call" not in calls
    assert "system" not in calls


def test_the_drop_module_starts_no_process_outside_the_injected_runner() -> None:
    tree = ast.parse(DROP_MODULE.read_text(encoding="utf-8"))
    calls = {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "Popen" not in calls
    assert "system" not in calls
    assert "check_output" not in calls
