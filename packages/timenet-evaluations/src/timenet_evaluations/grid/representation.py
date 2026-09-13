"""What a reader is opened against: one representation of one dataset.

A dataset is held in two representations for a run. ``Original`` is the release's own files, which
this harness never writes. ``TimeF`` is the converted form, stored solely as Parquet. Both readers
open both representations. Reading across representations is the experiment, and no representation
belongs to one reader.

This protocol is structural, like the reader protocols beside it. A representation declares the
three members and inherits nothing.
"""

from __future__ import annotations

from typing import Protocol

from timenet_evaluations.grid.artifact import Artifact
from timenet_evaluations.grid.item import ItemDeclaration
from timenet_evaluations.grid.parsing import ParsingPath


class Representation(Protocol):
    """One representation of one dataset, and everything a reader needs to open it.

    A representation carries no dataset-specific fact of its own. An identifier, a modality, a
    unit, a target schema, a channel name, a licence, and a label vocabulary all arrive with the
    dataset. Adding a dataset means supplying a dataset. It does not mean editing a representation
    or a reader.

    ``dataset`` is the dataset's name and not a fact about it. It is here because a reader must
    name the failing cell's dataset and ``open`` takes exactly one argument, which is the same
    reason ``item`` is here.
    """

    @property
    def dataset(self) -> str:
        """The name of the dataset this representation holds.

        A reader raises with the coordinates of the cell it failed in, and the dataset is one of
        them. ``open`` takes exactly one argument, so this member is the only route by which the
        dataset name reaches an opened reader. Without it a run over two datasets writes the same
        message twice and neither one says which grid failed.

        Returns:
            The dataset's name, as the run was given it.
        """
        ...

    @property
    def name(self) -> str:
        """The representation's own name, ``original`` or ``timef``.

        Returns:
            The name, which is also what the artifact records.
        """
        ...

    @property
    def artifact(self) -> Artifact:
        """Which representation these bytes are, and where they are.

        Returns:
            The artifact, whose ``representation`` is this representation's name and whose ``path``
            is the path a reader is opened against.
        """
        ...

    @property
    def item(self) -> ItemDeclaration:
        """The dataset's canonical item declaration.

        The declaration travels with the dataset and arrives here. This package holds no default
        for it.

        Returns:
            What one canonical item is, and how many of them the dataset holds.
        """
        ...

    @property
    def parsing_path(self) -> ParsingPath:
        """The one loader that both readers of this representation parse through.

        The parsing path is a property of the representation. Without a member that carries it, a
        reader holds only a path and must construct its own parser, and nothing then stops the two
        readers of one representation parsing through two different loaders. That failure is
        silent, because every number still renders.

        Returns:
            The parsing path, which both readers open and neither one owns.
        """
        ...
