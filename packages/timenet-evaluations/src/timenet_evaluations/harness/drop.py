"""The page cache drop: the command that was issued, and the status the process gave back.

Every timed repetition begins from a dropped page cache. The drop is privileged,
platform-specific and machine-wide, so no pure-Python measurement can perform one. This module
runs the platform's own mechanism and inspects what the process returned. It is the one place in
this package that starts a process, and it starts one outside every timed interval.

A drop that silently does nothing produces a warm figure that is identical in shape to a cold one.
There is no tell in the number. The record therefore carries what was attempted and observed, and
never what was intended. ``command`` is the argv that was issued, so a command named in prose and
never run cannot appear in it. ``exit_status`` is what the process gave back, so it is never zero
because a run assumed success.

**The tests of this module confirm the protocol's shape only. They are not evidence that the drop
works.** A test double has no page cache, so no test here can tell a cold read from a warm one. A
suite that appeared to test the drop would be worse than one that says it cannot, because the whole
hazard is the drop that reports success and evicts nothing. ``harness.calibration`` measures
whether the mechanism functions; this module cannot.

The two mechanisms are named by absolute path, because a name resolved through ``PATH`` is a second
thing that can be wrong on the machine that produced a figure.

The seam is ``DropCaches``, and the coordinates travel through it. A drop knows nothing about cells
and nothing about the grid, and it still names the cell it failed under, because the caller hands
it the coordinates it already holds.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import shlex
import subprocess
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, StorageKey
from timenet_evaluations.grid.failure import failure_message


DARWIN: Final = "darwin"
"""The value ``sys.platform`` holds on macOS."""

LINUX: Final = "linux"
"""The value ``sys.platform`` holds on Linux."""

DARWIN_ARGV: Final = ("/usr/sbin/purge",)
"""macOS drops the page cache through ``purge``, named by its absolute path."""

LINUX_ARGV: Final = ("/bin/sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches")
"""Linux drops the page cache through a write to ``/proc/sys/vm/drop_caches``.

The write follows a ``sync``, so dirty pages are on disk before the cache is emptied. A shell is
needed for the redirection, and the shell is named by its absolute path.
"""


class DropCommand(BaseModel):
    """The command one platform drops its page cache with.

    The model is frozen, so the argv a record names is the argv that was issued.
    """

    model_config = ConfigDict(frozen=True)

    argv: tuple[str, ...]
    """The program and its arguments, with the program named by an absolute path."""
    platform: str
    """The ``sys.platform`` value this command belongs to."""


class DropRecord(BaseModel):
    """What one drop attempted and what it observed.

    Nothing here is an intention. A run that did not issue a command has no record to show, and a
    record that exists names a process that ran and the status it returned.
    """

    model_config = ConfigDict(frozen=True)

    command: str
    """The command that was issued, as one shell-quoted line."""
    exit_status: int
    """The status the process returned. A record exists only for a status of zero, because any
    other status stops the run."""
    platform: str
    """The ``sys.platform`` value the command ran on."""


class Runner(Protocol):
    """What starts the drop process.

    ``subprocess.run`` satisfies this shape. A test hands a callable of the same shape instead, so
    a non-zero status and a missing executable are both reachable with no process at all.
    """

    def __call__(
        self, args: Sequence[str], /, *, check: bool, capture_output: bool
    ) -> subprocess.CompletedProcess[bytes]:
        """Start the command and wait for it.

        Args:
            args: The program and its arguments.
            check: Whether the runner itself raises on a non-zero status. It is always ``False``
                here, because this module inspects the status and reports it.
            capture_output: Whether the runner keeps the process output. It is always ``True``
                here, so the drop writes nothing into a report.

        Returns:
            The completed process, whose ``returncode`` is the status.
        """
        ...


class DropCaches(Protocol):
    """The seam every timed repetition drops the page cache through.

    A caller holds the coordinates of what it is about to measure, so it passes them here. That is
    what lets a drop name the cell it failed under without knowing what a cell is.
    """

    def __call__(self, *, at: Cell | StorageKey, path: Path) -> DropRecord:
        """Drop the page cache before one timed interval.

        Args:
            at: The coordinates of the cell or the storage figure this drop precedes.
            path: The file or directory the following read is given.

        Returns:
            The record of the command that ran and the status it returned.
        """
        ...


def platform_drop_command(platform: str) -> DropCommand:
    """Give the command that drops the page cache on one platform.

    This function is pure. It starts no process and reads no file, so a test can assert what each
    platform is given and what an unsupported platform is refused with.

    Args:
        platform: The ``sys.platform`` value of the machine the run is on.

    Returns:
        The command for that platform.

    Raises:
        EvaluationError: If the platform has neither mechanism. The refusal comes before any
            dataset, path or artifact exists, so it names no cell. It names the platform and both
            supported mechanisms instead, because those are what would have to change.
    """
    if platform == DARWIN:
        return DropCommand(argv=DARWIN_ARGV, platform=platform)

    if platform == LINUX:
        return DropCommand(argv=LINUX_ARGV, platform=platform)

    raise EvaluationError(
        "the page cache cannot be dropped on this platform, so every figure a run produced would "
        f"be warm: sys.platform {platform!r}, supported mechanisms {shlex.join(DARWIN_ARGV)!r} on "
        f"{DARWIN!r} and {shlex.join(LINUX_ARGV)!r} on {LINUX!r}"
    )


class PlatformDropCaches:
    """A drop that runs one platform's own mechanism and inspects what it returned.

    The runner is injected. That is what lets a test reach a non-zero status and a missing
    executable with no process, and it is why no test in this package needs a privilege.
    """

    def __init__(self, command: DropCommand, *, runner: Runner = subprocess.run) -> None:
        self.command = command
        self.runner = runner

    def __call__(self, *, at: Cell | StorageKey, path: Path) -> DropRecord:
        """Drop the page cache, and stop the run if the command reports that it did not.

        The drop falls outside the timed interval. Its cost is real and it is paid on every
        repetition, and none of it belongs in a figure about reading.

        Args:
            at: The coordinates of the cell or the storage figure this drop precedes.
            path: The file or directory the following read is given.

        Returns:
            The record of the command that ran and the status it returned.

        Raises:
            EvaluationError: If the command cannot be started, or if it returns a status other
                than zero. The next read would then be warm, and a warm figure under a cold
                heading is worse than no figure.
        """
        issued = shlex.join(self.command.argv)
        try:
            completed = self.runner(self.command.argv, check=False, capture_output=True)
        except OSError as error:
            raise EvaluationError(
                failure_message(
                    "the page cache drop could not be started, so the next read would be warm",
                    at=at,
                    path=path,
                    detail=f"command {issued!r}, platform {self.command.platform!r}, {error}",
                )
            ) from error

        if completed.returncode != 0:
            raise EvaluationError(
                failure_message(
                    "the page cache drop failed, so the next read would be warm",
                    at=at,
                    path=path,
                    detail=(
                        f"command {issued!r}, exit status {completed.returncode}, platform {self.command.platform!r}"
                    ),
                )
            )

        return DropRecord(command=issued, exit_status=completed.returncode, platform=self.command.platform)
