"""Size on disk: counted by the harness, from a path, after the fact.

One function owns the definition of a representation's size, and no representation and no reader
computes one for itself. A number a conversion reported could only ever describe what that
conversion produced, and one of the two representations is never written here at all: ``Original``
is the release's own files. Counting from a path after the fact is what lets one definition answer
for both.

The refusal is what the function is for. A path that is not there is not a size of zero, and a
caller that is handed zero has nothing to refuse. A run that recorded ``0`` for a real dataset
would look complete and would not be, so this raises instead and the run stops.

Size is counted and never timed. It has no cache drop, no repetitions and no median, because
counting the same bytes again gives the same integer back. It is not one of the four tasks, because
it is not an access.

There are exactly two storage figures per dataset, one per representation. Neither carries a reader
coordinate: nothing a reader does changes bytes on disk.
"""

from __future__ import annotations

from pathlib import Path

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import StorageKey
from timenet_evaluations.grid.failure import failure_message


def measure_size(path: Path, *, dataset: str, representation: str) -> int:
    """Count the bytes one representation of one dataset holds on disk.

    A file measures its own size. A directory measures the sum of the bytes of every file beneath
    it, and not the size of its own directory entry. A directory that exists and holds no file
    measures ``0``, because zero is a legal size.

    The order is a requirement. Within one (dataset, representation) pair this runs after the
    conversion, which is untimed, and before ``harness.plan.plan_blocks``, which takes the size as
    its first argument. A plan built before this call describes no artifact.

    Args:
        path: The file or directory that holds the representation's bytes.
        dataset: The name of the dataset these bytes belong to.
        representation: The name of the representation these bytes are.

    Returns:
        The number of bytes on disk.

    Raises:
        EvaluationError: If the path does not exist. An absent artifact is not a size, and this
            harness must not report one as ``0``.
    """
    if not path.exists():
        raise EvaluationError(
            failure_message(
                "a representation cannot be measured because its path does not exist",
                at=StorageKey(dataset=dataset, representation=representation),
                path=path,
            )
        )

    if path.is_file():
        return path.stat().st_size

    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
