"""What a dataset declares about itself, in advance of every measurement.

Five facts travel with a dataset, and a run produces none of them. They are what one canonical item
is, how many of those items the dataset holds, the rate every channel was resampled to before the
items were cut, what the conversion needs about the release that a frame of numbers cannot carry,
and what a measured cell over each representation really reads. A connector states all five
together, in one act of declaration.

The last two are here for the same reason as the third. Everything a connector says about a dataset
without being called travels in this value rather than as another member on the connector, because a
member added there is a member every future dataset has to implement, and its author is not in the
room. A statement about what was prepared is not a call a reader makes.

They are one act because a dataset that declared its item without its rate would still convert. The
converted form would hold the right values on a time axis built from the wrong number of values per
second, and nothing downstream compares an axis. The values agree, so the parity check passes, and
the artifact is wrong in a way no figure in the report shows.

The declaration is fixed before any measurement and is never negotiated at run time. Nothing here
derives it from an artifact, from a representation, from a reader, or from anything a run produced.
A representation that announced what it can conveniently yield, and was believed, would be the
subject under test choosing the denominator its own rates divide by.

The three facts arrive together as one value rather than as three module constants. A constant is
one dataset's fact in a place every other dataset can reach, so a second dataset inherits the first
one's rate in silence and only the time axis of its converted form records that it happened. A run
takes as many datasets as it is given, and each one carries its own declaration.

This module names no dataset. It states what a declaration is, the way
:mod:`timenet_evaluations.source.contract` states what a frame is, so the second connector inherits
this file rather than copies it.

What the declaration does not do is decide anything. The count it carries is a human judgement that
nothing in the package checks, which ADR-0026 records as its own central cost: a wrong item sits
upstream of every rate for that dataset, moves all of them by the ratio of the two units, and leaves
a result that validates and a table that renders. :func:`check_frame_row_count` is the one
disagreement this module can find, and it finds it only for a dataset whose frame rows are its
canonical items.
"""

# artifacts held the same content because one frame wrote them)

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from pydantic import BaseModel, ConfigDict

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.facts import DatasetFacts
from timenet_evaluations.grid.item import ItemDeclaration


class DatasetDeclaration(BaseModel):
    """What one dataset declares about itself before a run measures it.

    Every field is a fact about one release, and no field has a default. A default would be this
    module naming a dataset, and the second dataset would then inherit the first one's number
    without stating that it did.

    The model is frozen, so a caller cannot rebind the unit, the item declaration or the rate it
    was given. A declaration that a run could move is a denominator that a run could move.
    """

    model_config = ConfigDict(frozen=True)

    unit: str
    """The name of one canonical item, as the dataset's own task names it. Every rate the run
    reports is quoted per one of these, and the result records the name beside the number, so a
    stored rate says what it counted. Rates do not compare across datasets, because this name means
    something different for each one."""

    item: ItemDeclaration
    """The smallest unit this dataset's task consumes: how many of them the dataset holds, and the
    shape of one of them in the comparison form. Both representations must yield exactly this count,
    and every rate divides by it. A count that a representation reports about itself is never used
    in its place, even when the two agree."""

    rate_hz: int
    """The rate every channel was resampled to, in values per second, before the items were cut. One
    rate covers every channel of every item of this dataset. A conversion that needs a time axis
    builds it from this number rather than assumes, hard-codes or infers one."""

    facts: DatasetFacts
    """What the conversion needs about this release that neither the frame nor its card carries.

    The dataset's identity — its id, version, name, description and licence — is not in here. It is
    in the card the connectors distribution ships, which ADR-0028 makes the single source for it,
    and this value names that card rather than copying it. What is here is the other half: the
    modality, the unit, the rate, the channel names and the label vocabulary, which describe what
    one item holds rather than which release it came from.

    It travels in the declaration for the same reason the disclosures do. A member added to the
    connector is a member every future dataset has to implement, and its author is not in the
    room."""

    disclosures: Mapping[str, str]
    """What a representation's measured cells really read, as one statement per representation name.

    A representation names where the values came from. It does not always name what a measured cell
    read off the disk: a run whose untimed preparation materializes a cache, an index or a derived
    corpus makes every later cell a read of that instead, while the heading still names the release.
    The connector that did the preparation is the only component that knows, so the wording arrives
    here and nothing downstream composes it from a name or a path.

    A representation with nothing to disclose has no entry, and a dataset with nothing to say
    carries an empty mapping rather than no field at all. The presence of a statement is the signal,
    so an absent one is not rendered as a placeholder."""


def check_frame_row_count(frame: pd.DataFrame, declaration: DatasetDeclaration, *, dataset: str) -> None:
    """Make sure that a loaded frame holds as many rows as the dataset declared items.

    Call this from a connector whose frame rows are its canonical items. That is the single-dataset
    special case, and it is not the definition: a dataset whose item is coarser or finer than a
    frame row is expressible, and such a connector does not call this function. The declaration
    stays the source of the count either way, and the frame is only ever the cross-check.

    A disagreement stops the run. It says that the connector loaded a different number of items from
    the number every rate for this dataset is about to divide by, and neither number can be trusted
    until someone reads both.

    Args:
        frame: The frame the connector loaded for conversion, one row per canonical item.
        declaration: What this dataset declared about itself, in advance of the load.
        dataset: The name of the dataset, as the run was given it. A run takes more than one, so a
            message that omits the name cannot say which load was wrong.

    Raises:
        EvaluationError: If the number of rows is not the declared item count.
    """
    if len(frame) != declaration.item.count:
        raise EvaluationError(
            f"the loaded frame holds a different number of items from the number the dataset declared: "
            f"dataset {dataset!r}, declared count {declaration.item.count}, frame rows {len(frame)}"
        )
