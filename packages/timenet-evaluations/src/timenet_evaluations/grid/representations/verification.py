"""The untimed check that the converted artifact is still the artifact that was written.

The artifact persists between runs, so a run reads bytes it did not write. Everything that can
happen to a file between two runs can therefore happen to this one: a truncated write, a half
copy, a bad block, a partial erase. This module reads the whole artifact and hashes every file the
manifest lists against the value the manifest recorded.

This is the guard the persistence rests on, and it is why it is not a flag and not conditional on
reuse. **A truncated file reads faster than an intact one.** Damage does not make the converted
form look broken in a report of read times. It makes it look quick, in the column where the
converted form is the subject under test, and no number carries a sign of it. A guard against a
failure that flatters the subject is the last thing to leave switchable, because the run that most
needs it is the one nobody thought to switch it on for.

The check belongs to the untimed stretch between the conversion and the first timed cell, beside
the size measurement, the block plan and the parity check. It reads the whole artifact and so it
warms the page cache, which is safe only where it is: it runs before the first drop, and every
timed repetition drops the cache again. A call moved after the first drop would put a warm figure
under a cold label. Nothing here reads a clock and nothing here is reachable from a timed task.

This is not the parity check and neither replaces the other. This check asks whether the artifact
is the one that was written. Parity asks whether what was written is what the source held. A
damaged file passes parity at every position parity did not sample, and a faithful conversion of
the wrong frame passes this check completely.

The hashing is the storage library's own. A second implementation written here would be a second
opinion about the library's own manifest, and the two would drift.

``timenet`` is the ``timef`` extra and not a dependency, so it is absent from the environment
continuous integration builds. Each import below carries a suppression for that reason. Import this
module only when ``find_spec("timenet")`` has already found the package.
"""

from __future__ import annotations

from timenet.reader import TimeFReader  # ty: ignore[unresolved-import]
from timenet.registry import DatasetVersion  # ty: ignore[unresolved-import]

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.artifact import Artifact
from timenet_evaluations.grid.cell import StorageKey
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.representations.timef import TimeF


def verify(artifact: Artifact, *, dataset: str) -> None:
    """Make sure that every file of one converted artifact matches the manifest that lists it.

    Call this after the artifact is on disk and before the first timed cell of its dataset, whether
    the conversion wrote it in this run or found it from an earlier one. The check is the same in
    both cases, and it must not be skipped in either.

    Args:
        artifact: The converted artifact, as the conversion returned it.
        dataset: The name of the dataset it holds, as the run was given it. It names the failure.

    Raises:
        EvaluationError: If the artifact is of another representation, if the version cannot be
            opened, or if a file the manifest lists is missing or does not match what the manifest
            recorded for it.
    """
    at = StorageKey(dataset=dataset, representation=TimeF.name)
    if artifact.representation != TimeF.name:
        raise EvaluationError(
            failure_message(
                "only the converted representation has a manifest to be checked against",
                at=at,
                path=artifact.path,
                detail=f"artifact representation {artifact.representation!r}",
            )
        )

    try:
        version = DatasetVersion.open_local(artifact.path)
    except Exception as error:
        raise EvaluationError(
            failure_message(
                "the converted artifact cannot be opened to be checked",
                at=at,
                path=artifact.path,
                detail=f"{error}",
            )
        ) from error

    try:
        # The library reopens each file the manifest lists, compares its size, and hashes its
        # contents. Its own message names the file that failed, which is the one fact a person
        # needs to see, so the message travels into the detail rather than being summarized.
        with TimeFReader(version) as reader:
            reader.verify()
    except Exception as error:
        raise EvaluationError(
            failure_message(
                "the converted artifact does not match its own manifest",
                at=at,
                path=artifact.path,
                detail=f"{error}",
            )
        ) from error
