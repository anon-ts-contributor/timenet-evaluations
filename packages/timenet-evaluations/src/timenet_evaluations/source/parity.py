"""The parity check: both representations yield the declared item, and it is the same item.

A dataset is held in two representations for a run, and this harness writes only one of them. The
superseded model did not need this module. One frame wrote every artifact, so the artifacts held the
same content because of the way they were made, and no assertion was necessary. That premise is
gone: the release's own files are read as they ship and are derived from no frame, so nothing about
the way the two representations were made says that they hold the same items.

What replaces the guarantee is this check. Before a dataset's timed cells run, each representation
is opened by each reader, and what comes back is compared against one reference array. Sixteen
figures then describe one dataset, and this is the only thing in the run that says they describe the
same object.

The reference is the loaded frame, through
:func:`~timenet_evaluations.source.contract.signal_stack`. The two representations are not compared
against each other. Two readers that were wrong in the same way would agree with each other and
disagree with the frame, and the frame is the one array whose provenance this package knows.

Nothing here starts a timer and nothing here produces a figure. The check is untimed, it runs once
per dataset, and it is not one of the four tasks. A run that reported it would print a number for
work that no adopter of either representation does.

The check lives here, and it must stay here. Parity is a statement about a dataset, and this package
is the one that knows what a dataset is. A representation that verified its own parity would be the
subject under test grading its own paper, which is the same defect as a representation that
announces its own item count.

**What this check does not establish.** It reads two positions. It compares the item at position
zero and the item at one interior position, which is the midpoint of the declared count, and it
reads nothing between them and nothing after them. Agreement at those two positions is evidence that
the two representations hold the same items. It is not proof, and no report may quote it as proof.

A representation that put its interior in a different order, or that holds one bad item, agrees at
position zero, agrees at the midpoint unless the fault is at the midpoint, and passes. Every rate
for that dataset is then a rate over content that does not agree, and nothing later in the run looks
again. The failure renders correctly: the table prints, the result is valid, and the numbers are
wrong.

Two things make that less likely, and neither one removes it. The count is checked over the whole of
each representation and is not sampled, and a representation that cut the dataset into the wrong
pieces almost always gets the count wrong as well as the values. And a failure stops the run instead
of writing a warning, so a disagreement that this check does find cannot reach a report.

The complete check is a full read of both representations, outside every timer, on every run. It
costs more than the benchmark it guards, and it is not what this module does. A stronger statement
than the one above must be paid for on purpose.
"""

# files are derived from no frame)

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import OpenKey
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.item import ItemDeclaration, check_item_count, to_comparison_form
from timenet_evaluations.grid.reader import Opened, Reader
from timenet_evaluations.grid.readers import PANDAS
from timenet_evaluations.grid.representation import Representation


MINIMUM_SAMPLED_COUNT = 2
"""How many items a dataset must hold before a second position exists to sample.

A dataset of one item has one position, and position zero is it. The check then compares that one
item and says so, instead of reports two positions that it did not have.
"""


class _OneItemBlock(NamedTuple):
    """One canonical item, named as the block that holds only it.

    An opened reader gives no positional access of its own. A block of one item at the position to
    check is how this module asks for one item and for nothing else, and this value satisfies the
    block shape that a reader expects by shape alone.
    """

    start: int
    count: int = 1


@dataclass(frozen=True, eq=False)
class ParityReference:
    """The dataset the check is about, and the array both representations are measured against.

    The three facts travel together because they are about one dataset and must not be taken from
    three places. The declaration is the dataset's own, supplied once, and every representation is
    checked against this one copy of it. A check that read each representation's own declaration
    would let two representations that disagree each pass against its own number, which is the
    failure this module exists to catch.

    Equality is off and the value is frozen. The array is large, an equality test on it does not
    give a boolean, and no caller compares two of these.
    """

    dataset: str
    """The name of the dataset, as the run was given it. A run takes more than one, so a message
    that omits the name cannot say which check failed."""

    declaration: ItemDeclaration
    """What one canonical item of this dataset is, and how many of them it holds. This is the
    dataset's own declaration and never a number that a representation reported about itself."""

    reference: np.ndarray
    """Every item's values, shaped ``(count, *shape)``, from
    :func:`~timenet_evaluations.source.contract.signal_stack` over the loaded frame. Both
    representations are compared against this array, and neither is compared against the other."""


def sampled_positions(count: int) -> tuple[int, ...]:
    """Name the positions that the check compares values at.

    The positions are position zero and the midpoint of the declared count. They are fixed and are
    not taken at random. A random position makes a run that failed impossible to repeat, and it lets
    two runs of one dataset check different things while both report that parity held.

    Everything that the returned positions do not name stays unread. That is the limit of this
    check, and the module docstring says what follows from it.

    Args:
        count: How many canonical items the dataset declared.

    Returns:
        Position zero, and the midpoint where the dataset holds more than one item.
    """
    if count < MINIMUM_SAMPLED_COUNT:
        return (0,)

    return (0, count // 2)


def check_parity(
    reference: ParityReference,
    representations: Sequence[Representation],
    readers: Sequence[Reader],
) -> None:
    """Make sure that every representation gives the declared item, and gives the same item.

    Call this once per dataset, before that dataset's first timed cell. The call is untimed and
    makes no figure. A failure stops the run, so no row is printed for a dataset whose two
    representations do not agree: four rates over content that differs are four numbers about
    different work that wear one unit.

    Every representation is checked against ``reference`` and never against another representation.
    Each one is opened by each reader, so the comparison covers every pair of a representation and a
    reader that the run is about to measure.

    This establishes the declared count over the whole of each representation, and equal values at
    the positions that :func:`sampled_positions` names. It does not establish that the two
    representations agree at a position that it did not read. The module docstring says what that
    costs.

    A disagreement raises ``EvaluationError``: from :func:`_check_reference` if the reference array
    is not the dataset the declaration describes, from :func:`_check_count` if a representation
    holds a number of items that the dataset did not declare, and from :func:`_check_values` if a
    sampled position holds values that the loaded dataset does not hold.

    Args:
        reference: The dataset, its declaration, and the array both representations are measured
            against.
        representations: The representations of that one dataset, already made, each one carrying
            its own parsing path.
        readers: The readers that open them, which are the readers the run measures with.
    """
    _check_reference(reference)

    for representation in representations:
        _check_count(reference, representation)
        for reader in readers:
            _check_values(reference, representation, reader)


def _check_reference(reference: ParityReference) -> None:
    """Make sure that the reference array is the shape that the declaration gives the dataset.

    The array is what every representation is measured against, so an array that does not agree with
    the declaration makes every later comparison a comparison with the wrong thing. The failure
    names no representation, because no representation is at fault: the caller stacked a frame that
    is not this dataset's.

    Args:
        reference: The dataset, its declaration, and the array.

    Raises:
        EvaluationError: If the array does not hold the declared count of items of the declared
            shape.
    """
    expected = (reference.declaration.count, *reference.declaration.shape)
    if reference.reference.shape != expected:
        raise EvaluationError(
            f"the parity reference does not have the shape the dataset declared: "
            f"dataset {reference.dataset!r}, declared shape {expected}, "
            f"reference shape {reference.reference.shape}"
        )


def _check_count(reference: ParityReference, representation: Representation) -> None:
    """Make sure that one representation holds the declared number of canonical items.

    This is the one part of the check that covers every item and does not sample. A representation
    that cut the dataset into whole recordings, where the declared item is smaller than a recording,
    reports a count that the dataset never declared, and it is refused here before a value is read.

    The count is the number that the representation reports about itself, and this is the only use
    that this package makes of such a number. No rate divides by it. It is read through the
    representation's parsing path, because that is where the reader protocols give a length at all.

    Args:
        reference: The dataset, its declaration, and the array.
        representation: The representation to count.
    """
    path = representation.artifact.path
    items = representation.parsing_path.open_pandas(path)

    check_item_count(
        len(items),
        reference.declaration,
        at=_scope(reference, representation, PANDAS),
        path=path,
    )


def _check_values(reference: ParityReference, representation: Representation, reader: Reader) -> None:
    """Make sure that one opened representation holds the reference's values at the sampled positions.

    The comparison has no tolerance. Both target forms become one comparison form, so equality is a
    test and not a judgement, and a value that differs in the last bit is a value that differs.

    Position zero is read as the first-item read, which is the read that names position zero. Each
    other sampled position is read as a block of one item, because an opened reader gives no
    positional access of its own.

    Args:
        reference: The dataset, its declaration, and the array.
        representation: The representation to read.
        reader: The reader that opens it.

    Raises:
        EvaluationError: If a sampled position holds values that the loaded dataset does not hold.
    """
    at = _scope(reference, representation, reader.name)
    path = representation.artifact.path
    declaration = reference.declaration
    opened = reader.open(representation)

    for position in sampled_positions(declaration.count):
        values = opened.read_first() if position == 0 else _one_item_of_block(opened, position)
        item = to_comparison_form(_unwrapped(values), declaration, at=at, path=path)

        if not np.array_equal(item, reference.reference[position]):
            raise EvaluationError(
                failure_message(
                    "a representation holds values the loaded dataset does not hold at a sampled position",
                    at=at,
                    path=path,
                    detail=f"sampled position {position}, declared item shape {declaration.shape}",
                )
            )


def _one_item_of_block(opened: Opened, position: int) -> object:
    """Read the one canonical item at a position, as the block that holds only it.

    Args:
        opened: The opened representation to read from.
        position: The position of the item to read.

    Returns:
        What the reader gave back for a block of one item, in that reader's target form.
    """
    return opened.read_block(_OneItemBlock(start=position))


def _unwrapped(values: object) -> object:
    """Take the one canonical item out of what a read of one item gave back.

    A read of one item gives that item in the reader's target form. One target form gives the item
    itself, and the other gives a container that holds it, so a list or a tuple of exactly one
    element is that one item and is unwrapped here.

    Nothing else is touched. No column is dropped and no pair is unpacked: a value that carries an
    item together with its label holds more than one element, is passed on as it came, and is
    refused by the comparison form, which is where that refusal belongs.

    Nothing here reads the values as an array. A value that numpy refuses must reach the comparison
    form and be refused there with the coordinates of the scope it failed in, and not with a shape
    error from this line that names nothing.

    Args:
        values: What the read gave back.

    Returns:
        The one element of a container that holds exactly one. The values as they came in each other
        case.
    """
    if isinstance(values, list | tuple) and len(values) == 1:
        return values[0]

    return values


def _scope(reference: ParityReference, representation: Representation, reader: str) -> OpenKey:
    """Name the scope that one part of the check belongs to.

    The scope has three coordinates and never a fourth. This check runs before a task starts, so a
    message that named a task would name one that had not begun.

    Args:
        reference: The dataset, its declaration, and the array.
        representation: The representation being checked.
        reader: The name of the reader that opened it.

    Returns:
        The three coordinates of that scope.
    """
    return OpenKey(dataset=reference.dataset, representation=representation.name, reader=reader)
