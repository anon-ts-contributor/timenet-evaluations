"""The datasets a run was given: one ``name=path`` pair each, and the four refusals they earn.

A run measures the datasets it was named and no others, so the positional argument is repeatable
and each entry carries two halves. The name selects the connector that opens one release's own
files, and it is the ``dataset`` coordinate of every measurement key and every storage key the run
produces. The path says where those files are.

The name is an input and is never taken from a directory's basename. A basename says nothing about
which connector opens the files inside a directory, so a dataset moved to another directory would
select another connector or none at all, and every key it produces would move with it.

The split is a conversion the parser makes, and the four checks are preconditions of the run.
:func:`parse_dataset` refuses nothing. It splits one positional and gives the pair an empty name
when the text is not a pair at all. :func:`check_datasets` then makes the refusals, and each of the
four covers every pair before the next one starts. A typo in the third of three pairs is therefore
reported before the first dataset is loaded, and no conversion and no cache drop is spent on a run
that cannot complete.

Two of the four checks are not written here. ``registry.connector_for`` already refuses a name
nothing is registered under, and ``contract.check_source_directory`` already refuses a path that is
not a directory. This module calls both, and the connector calls the second one too, so one
condition cannot be worded differently at the two sites. The two checks this module adds are the
two that nothing else makes: a positional that is not a pair, and a name that appears twice.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.source.contract import check_source_directory
from timenet_evaluations.source.registry import connector_for


SEPARATOR = "="
"""What divides the two halves of one positional.

The split is on the first one, so a path that holds an ``=`` of its own keeps it.
"""

DATASET_FORM = f"name{SEPARATOR}path"
"""The form every positional takes.

It is the metavar of the usage line, and every refusal of a positional that is not a pair shows it.
"""


@dataclass(frozen=True, slots=True)
class DatasetPair:
    """One dataset a run was given: the name that selects its connector, and where its files are."""

    text: str
    """The positional as the command line carried it. A refusal names this, so an operator reads
    back what they typed rather than a value the parser made of it."""
    name: str
    """The half before the first ``=``, and the ``dataset`` coordinate of every key this dataset
    produces. It is empty when the positional is not a pair, and :func:`check_datasets` refuses
    the pair before anything reads it."""
    path: Path
    """The half after the first ``=``. It has no meaning while ``name`` is empty."""


def parse_dataset(text: str) -> DatasetPair:
    """Split one positional into the name it carries and the path it carries.

    This is the parser's own conversion and it refuses nothing. A positional that is not a pair
    gives a pair with an empty name, and :func:`check_datasets` then refuses it with the package's
    own error type. The parser's error path is for a value a conversion rejects. A name that was
    never given is a precondition of the run, and the run reports it in the same way it reports a
    name nothing is registered under.

    Args:
        text: One positional, as the command line carried it.

    Returns:
        The pair. Its name is empty when the text carries no ``=``, or when either half is empty.
    """
    name, separator, location = text.partition(SEPARATOR)

    if not (separator and name and location):
        return DatasetPair(text=text, name="", path=Path())

    return DatasetPair(text=text, name=name, path=Path(location))


def check_pairs_are_named(datasets: Sequence[DatasetPair]) -> None:
    """Refuse a positional that is not a ``name=path`` pair.

    Both halves are required. A path alone cannot identify a dataset, because the name selects the
    connector that opens the release's files and declares the canonical item every rate divides by.

    Args:
        datasets: Every pair the command line carried, in the order it carried them.

    Raises:
        EvaluationError: If a positional carries no ``=``, or if either half of one is empty.
    """
    malformed = [pair.text for pair in datasets if not pair.name]

    if malformed:
        raise EvaluationError(
            f"a dataset argument is not a pair of a name and a path: arguments {malformed}, "
            f"expected form {DATASET_FORM!r}. Both halves are required, and the name is never "
            f"taken from the directory's basename: a basename says nothing about which connector "
            f"opens the files inside that directory"
        )


def check_names_are_unique(datasets: Sequence[DatasetPair]) -> None:
    """Refuse a dataset name that one run was given more than once.

    The paths are not compared, because two different paths under one name is the case this
    refuses. Both datasets collide in every measurement key and in every storage key, and a result
    that holds two sets of figures under one coordinate cannot be read at all.

    Args:
        datasets: Every pair the command line carried, in the order it carried them.

    Raises:
        EvaluationError: If two pairs carry the same name.
    """
    counts = Counter(pair.name for pair in datasets)
    repeated = sorted(name for name, count in counts.items() if count > 1)

    if repeated:
        collisions = [pair.text for pair in datasets if pair.name in repeated]
        raise EvaluationError(
            f"a dataset name was given more than once in one run: names {repeated}, arguments "
            f"{collisions}. Two datasets under one name write into the same measurement and "
            f"storage keys, and the result cannot say which figures belong to which"
        )


def check_datasets(datasets: Sequence[DatasetPair]) -> None:
    """Make the four refusals a run can make from its command line alone.

    Call this at the top of a run, before the environment is captured and before the first dataset
    is loaded. Everything the four read is visible in the command line, so none of it is deferred
    to the point of use.

    The four run in this order, and each one covers every pair before the next one starts:

    1. :func:`check_pairs_are_named` refuses a positional that is not a pair.
    2. ``connector_for`` refuses a name nothing is registered under, and a registered connector
       that is short of one of the four members.
    3. :func:`check_names_are_unique` refuses a name that appears twice.
    4. ``check_source_directory`` refuses a path that is not a directory that exists. A path that
       exists as a regular file is refused with the same failure as one that is absent.

    Each of the four raises ``EvaluationError`` and names the pair behind the failure.

    Args:
        datasets: Every pair the command line carried, in the order it carried them.
    """
    check_pairs_are_named(datasets)

    for pair in datasets:
        connector_for(pair.name)

    check_names_are_unique(datasets)

    for pair in datasets:
        check_source_directory(pair.path)
