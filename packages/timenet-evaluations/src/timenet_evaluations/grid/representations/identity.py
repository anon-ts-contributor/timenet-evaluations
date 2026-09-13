"""What makes a persisted artifact the artifact this code would write from this source.

The converted artifact outlives the run that wrote it. A run that finds a version already beneath
the artifact root reuses it and converts nothing. Two answers are wrong here, and they are wrong in
different directions. An answer that is too strict reconverts a corpus that did not change, which is
the cost this design exists to remove. An answer that is too loose reuses an artifact some other
code wrote, or wrote from other content, which is worse: every number still renders, and the run
prices bytes it did not produce and did not read.

This module gives the answer as an address rather than as a comparison. The key is three path
components, and the artifact of one source is the committed version beneath them:

    <artifact root>/<revision>/<writer>/<source digest>/<dataset id>/<version>

Nothing parses the key back, and nothing keeps a record beside the artifact that can disagree with
it. A run computes the three components, looks beneath them, and finds either a committed version or
nothing.

The three answer one question each, and all three have to be asked. The revision says which build of
the storage library wrote the bytes. The writer says which state of the conversion in this repository
asked it to. The digest says what content it was given. Miss the first and a library release changes
the layout under a reused artifact. Miss the second and a change to the conversion here reports
today's numbers against yesterday's bytes, silently, and the mistake gets quieter as the layout
drifts further. Miss the third and a run prices a dataset it never read.

The two code components come first and stand next to each other, so one state of the code is one
subtree. Two revisions of the library over one source then stand side by side under the same digest,
which is the shape a run comparing two revisions needs.

The writer component covers every module in this package directory rather than the conversion's own
file. The conversion does not write sample ids itself: it calls a function beside it, whose padding
constant decides every id in the artifact. A component that covered one file would leave that
constant, and every other constant a neighbour holds, outside the address. The cost of the wider
scope is stated rather than hidden: an edit to any module here, a comment included, converts once
more. That is one untimed pass, against a silent result whose meaning is wrong.

This module imports no part of the storage library. It names the library's distribution to read the
revision that installed it, which needs the metadata and not the package, so this module imports
where the ``timef`` extra is absent.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from importlib.metadata import Distribution, PackageNotFoundError
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import StorageKey
from timenet_evaluations.grid.facts import DatasetFacts
from timenet_evaluations.grid.failure import failure_message


LIBRARY = "timenet"
"""The distribution that writes and reads the converted form. Its revision is half of the key,
because an artifact one revision wrote is not the artifact another revision would write."""

PROVENANCE = "direct_url.json"
"""The file an installer writes to record where a distribution came from. It carries the commit of
a distribution installed from a repository, which is what the pin in this workspace produces."""

GIT = "git"
"""The prefix of a revision read from the commit an installer recorded."""

RELEASE = "release"
"""The prefix of a revision read from the distribution's own version number.

An artifact keyed on this prefix is keyed on a number that moves only when the library releases.
Two different builds of one release therefore share a key, so an installation that is not pinned to
a commit can reuse an artifact another build of the same release wrote. The prefix is in the path
so that this is visible on disk rather than argued about.
"""

WRITER_MODULES = Path(__file__).parent
"""The directory whose modules the writer component covers.

The scope is the directory and not the conversion's own file, because the conversion executes code
from its neighbours: the sample id and its padding constant, the facts the metadata is built from,
and this module's own rules for the address. A scope of one file would need a list of the others,
and a list is only as good as whoever remembers to add to it.
"""

MODULE_SUFFIX = "*.py"
"""What counts as a module of the writer. Nothing else in the directory is read."""


class ArtifactKey(BaseModel):
    """The three components that address the artifact this code writes from one source.

    The components are path names and they are used as path names. A key that was recorded beside
    the artifact instead could disagree with where the artifact is, and nothing would find the
    disagreement.
    """

    model_config = ConfigDict(frozen=True)

    revision: str
    """The revision of the library that writes the artifact, as ``library_revision`` read it."""
    writer: str
    """The digest of the conversion's own modules, as ``writer_digest`` computed it."""
    source: str
    """The digest of the content the conversion is given, as ``source_digest`` computed it."""

    def root(self, out: Path) -> Path:
        """Give the registry root that holds the artifacts of this code and this source.

        Args:
            out: The artifact root the run was given.

        Returns:
            The directory beneath which the storage library writes and finds its versions.
        """
        return out / self.revision / self.writer / self.source


def library_revision(*, at: StorageKey, path: Path) -> str:
    """Read the revision of the library that writes the converted form.

    The revision comes from the record the installer wrote. A distribution installed from a
    repository carries the commit, which is what the pin in this workspace produces and what makes
    two pins two different keys. A distribution installed in another way carries no commit, so the
    revision falls back to the release number and says so in the value.

    Args:
        at: The coordinates of the conversion that asked, which name the failure.
        path: The artifact root, which names the failure.

    Returns:
        The revision, as a path name that says what it was read from.

    Raises:
        EvaluationError: If the library is not installed, or if the record of where it was
            installed from is there and cannot be read.
    """
    try:
        distribution = Distribution.from_name(LIBRARY)
    except PackageNotFoundError as error:
        raise EvaluationError(
            failure_message(
                "the library that writes the converted form is not installed",
                at=at,
                path=path,
                detail=f"distribution {LIBRARY!r}: {error}",
            )
        ) from error

    release = f"{RELEASE}-{distribution.version}"
    recorded = distribution.read_text(PROVENANCE)
    if recorded is None:
        return release

    try:
        parsed = json.loads(recorded)
    except json.JSONDecodeError as error:
        raise EvaluationError(
            failure_message(
                "the record of where the library was installed from cannot be read",
                at=at,
                path=path,
                detail=f"{PROVENANCE} of {LIBRARY!r}: {error}",
            )
        ) from error

    commit = _recorded_commit(parsed)

    return f"{GIT}-{commit}" if commit is not None else release


def writer_digest(*, at: StorageKey, path: Path) -> str:
    """Reduce the conversion's own modules to the digest that addresses what they write.

    The digest covers the source bytes of every module beside the conversion, each with its name.
    An edit to any of them therefore gives a new address, so no run reads bytes that another state
    of this code wrote. The scope is wider than the modules that decide the layout, and that is
    deliberate: it invalidates on an edit that changes nothing, which costs one untimed conversion,
    and it cannot miss an edit that changes everything.

    Args:
        at: The coordinates of the conversion that asked, which name the failure.
        path: The artifact root, which names the failure.

    Returns:
        The digest, as a hexadecimal string.

    Raises:
        EvaluationError: If a module of the conversion cannot be read.
    """
    digest = hashlib.sha256()
    try:
        for module in sorted(WRITER_MODULES.glob(MODULE_SUFFIX)):
            digest.update(_framed(module.name.encode("utf-8")))
            digest.update(_framed(module.read_bytes()))
    except OSError as error:
        raise EvaluationError(
            failure_message(
                "the modules of the conversion cannot be read",
                at=at,
                path=path,
                detail=f"{WRITER_MODULES}: {error}",
            )
        ) from error

    return digest.hexdigest()


def source_digest(
    signals: Sequence[np.ndarray], *, labels: Sequence[str], patients: Sequence[str], facts: DatasetFacts
) -> str:
    """Reduce the content one conversion is given to the digest that addresses its artifact.

    The digest covers every value the conversion writes: each item's values, its scored class, its
    subject, and the facts the frame does not carry. Two frames that differ anywhere the conversion
    reads therefore give two digests and two addresses, so a reuse cannot serve content it was not
    built from.

    Every part is written into the hash with its length in front of it. Without the length two
    different sequences of parts can join into one identical run of bytes, and two sources would
    share an address.

    Args:
        signals: One array per item, in the order the conversion writes them.
        labels: The scored class of each item, in the same order.
        patients: The subject of each item, in the same order.
        facts: Everything about the dataset that the frame does not carry.

    Returns:
        The digest, as a hexadecimal string.
    """
    digest = hashlib.sha256()
    digest.update(_framed(facts.model_dump_json().encode("utf-8")))
    digest.update(_framed(str(len(signals)).encode("utf-8")))
    for signal, label, patient in zip(signals, labels, patients, strict=True):
        digest.update(_framed(json.dumps([signal.shape, signal.dtype.str]).encode("utf-8")))
        digest.update(_framed(np.ascontiguousarray(signal).tobytes()))
        digest.update(_framed(label.encode("utf-8")))
        digest.update(_framed(patient.encode("utf-8")))

    return digest.hexdigest()


def _framed(part: bytes) -> bytes:
    """Put the length of one part in front of it, so that two sequences of parts cannot join alike.

    Args:
        part: The bytes of one part.

    Returns:
        The part, with its length and a separator in front of it.
    """
    return f"{len(part)}:".encode() + part


def _recorded_commit(parsed: object) -> str | None:
    """Take the commit out of the record an installer wrote.

    A record of an installation from a directory or from a file carries no commit, and that is not
    a failure: the caller falls back to the release number and says so in the value it returns.

    Args:
        parsed: The parsed record.

    Returns:
        The commit, or ``None`` where the record names no commit.
    """
    if not isinstance(parsed, dict):
        return None

    vcs = parsed.get("vcs_info")
    commit = vcs.get("commit_id") if isinstance(vcs, dict) else None

    return commit if isinstance(commit, str) and commit else None
