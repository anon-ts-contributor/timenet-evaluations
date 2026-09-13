"""The ``original`` representation: the dataset exactly as the release ships it.

This harness never writes these files. It does not convert them, re-encode them, repack them, or
normalize them. They are the input to a run and not an output of one. A change that rewrote them
would price a representation nobody ships, and the storage figure would then describe this harness
rather than the release.

The path is one the run was handed rather than one it produced, which makes it the path most likely
to be wrong. A path that is not there is refused here, at the earliest point that can see it. The
size measurement refuses it as well, and that refusal belongs to the read-measurement capability;
this one is earlier, not instead.

Both readers of this representation parse through one parsing path, which the representation
carries. Where the release ships its own reference loader, that parsing path reaches the release's
loader and the adapter inside every timed cell is the release's. Where it does not, the adapter is
this project's, and the report says which case a dataset is in.
"""

from __future__ import annotations

from pathlib import Path

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.artifact import Artifact
from timenet_evaluations.grid.cell import StorageKey
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.item import ItemDeclaration
from timenet_evaluations.grid.parsing import ParsingPath


class Original:
    """The release's own files, described but never produced.

    This class holds no fact about any dataset. The identifier, the modality, the unit, the target
    schema, the channel names, the licence, and the label vocabulary all arrive with the dataset:
    the name and the item declaration as arguments, and everything the files themselves need
    through the parsing path.
    """

    name = "original"
    """The representation's own name, which is also what the artifact records."""

    def __init__(self, *, dataset: str, source: Path, item: ItemDeclaration, parsing_path: ParsingPath) -> None:
        """Describe one dataset's release files.

        Args:
            dataset: The name of the dataset these files belong to. It names the failing cell in
                every message this representation raises, because a run measures more than one
                dataset and a message that omits it cannot say which one failed.
            source: The file or directory the release ships. This harness reads it and never
                writes it.
            item: The dataset's canonical item declaration, made in advance of any measurement.
            parsing_path: The dataset's own loader, which both readers of this representation
                parse through.

        Raises:
            EvaluationError: If ``source`` is not there.
        """
        if not source.exists():
            raise EvaluationError(
                failure_message(
                    "the release's files are not there",
                    at=StorageKey(dataset=dataset, representation=self.name),
                    path=source,
                )
            )

        self._dataset = dataset
        self._artifact = Artifact(representation=self.name, path=source)
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

        The path is the release's own, so the size measurement reads what the release ships and
        never something this harness produced.

        Returns:
            The artifact naming this representation and the release's path.
        """
        return self._artifact

    @property
    def item(self) -> ItemDeclaration:
        """The dataset's canonical item declaration.

        The declaration travels with the dataset and arrives here. This module holds no default
        for it.

        Returns:
            What one canonical item is, and how many of them the dataset holds.
        """
        return self._item

    @property
    def parsing_path(self) -> ParsingPath:
        """The one loader both readers of this representation parse through.

        Sharing it is the whole point of the member. Two readers that each built their own parser
        would put two different loaders in the two cells of this row, and the difference between
        the reader columns would stop being the target form.

        Returns:
            The dataset's own loader.
        """
        return self._parsing_path
