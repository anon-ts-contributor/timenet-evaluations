"""The ``timef`` representation: the converted form, stored solely as Parquet.

The conversion produces it once, before any timer starts, so that it is on disk to be read and
measured. Nothing else on disk counts as this representation, and no second container is written
beside it and measured as though it were part of it.

Parquet is not a rival under test. It is the substrate of the subject: there is no cell in which
Parquet stands opposite TimeF, and no document, docstring or report here calls it a baseline or a
competitor.

The layout findings recorded against that substrate bear on this representation's own numbers.
Index columns that existed nowhere in the source, and per-item metadata repeated once per component
rather than once per item, inflate the stored size; dictionary encoding, run-length encoding and
compression reduce that inflation without removing it. A read that materializes columns it discards
pays for them on every timed repetition, and pays proportionally more on a block read than on a full
pass. A positional reshape is correct only while rows stay grouped by item and the storage preserves
that order, and a block read locates its run by position, so it depends on the same unenforced
ordering and depends on it harder.

This module names no dataset. Everything one dataset needs to be converted arrives in
:class:`DatasetFacts`, which the run supplies alongside the dataset.
"""

from __future__ import annotations

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.artifact import Artifact
from timenet_evaluations.grid.cell import StorageKey
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.item import ItemDeclaration
from timenet_evaluations.grid.parsing import ParsingPath


SAMPLE_ID_DIGITS = 8
"""How many digits a sample id's position is padded to. The padding is what makes the stored order
of the ids ascending position order, which is what lets a narrow read of a run of positions come
back in the order it was asked for."""


def sample_id(position: int) -> str:
    """Name the sample that holds the canonical item at one position.

    The conversion writes this id and the parsing path reads it back, so the two agree by using one
    function. The id is padded, because the storage library returns the samples of a filtered read
    in stored id order and only a padded id sorts as its position does.

    Args:
        position: The item's position, counted from zero.

    Returns:
        The sample id for that position.
    """
    return f"item-{position:0{SAMPLE_ID_DIGITS}d}"


class TimeF:
    """The converted form of one dataset, as the conversion left it on disk.

    The artifact this is constructed from comes from the conversion's return value and from
    nowhere else. A path composed from constants is a guess about another library's on-disk
    layout, and a stale guess is exactly the artifact that reads back wrong.
    """

    name = "timef"
    """The representation's own name, which is also what the artifact records."""

    def __init__(self, *, dataset: str, artifact: Artifact, item: ItemDeclaration, parsing_path: ParsingPath) -> None:
        """Describe one dataset's converted form.

        Args:
            dataset: The name of the dataset this form was converted from. It names the failing
                cell in every message this representation raises.
            artifact: What the conversion produced, as the conversion reported it.
            item: The dataset's canonical item declaration, made in advance of any measurement.
            parsing_path: The loader both readers of this representation parse through. It reads
                through the storage library's own reader, so no read of this representation goes
                through a path this project built over its files.

        Raises:
            EvaluationError: If the artifact names another representation, or its path is not
                there.
        """
        if artifact.representation != self.name:
            raise EvaluationError(
                failure_message(
                    "the artifact describes another representation",
                    at=StorageKey(dataset=dataset, representation=self.name),
                    path=artifact.path,
                    detail=f"artifact representation {artifact.representation!r}",
                )
            )

        if not artifact.path.exists():
            raise EvaluationError(
                failure_message(
                    "the converted form is not where the conversion said it was",
                    at=StorageKey(dataset=dataset, representation=self.name),
                    path=artifact.path,
                )
            )

        self._dataset = dataset
        self._artifact = artifact
        self._item = item
        self._parsing_path = parsing_path

    @property
    def dataset(self) -> str:
        """The name of the dataset this representation holds.

        Returns:
            The dataset's name, as the run was given it.
        """
        return self._dataset

    @property
    def artifact(self) -> Artifact:
        """Which representation these bytes are, and where they are.

        Returns:
            The artifact naming this representation and the path the conversion produced.
        """
        return self._artifact

    @property
    def item(self) -> ItemDeclaration:
        """The dataset's canonical item declaration.

        Returns:
            What one canonical item is, and how many of them the dataset holds.
        """
        return self._item

    @property
    def parsing_path(self) -> ParsingPath:
        """The one loader both readers of this representation parse through.

        It reads through the storage library's own reader. A read that opened the stored Parquet
        directly would measure a path this project built rather than what a user of the format
        gets.

        Returns:
            The loader for the converted form.
        """
        return self._parsing_path
