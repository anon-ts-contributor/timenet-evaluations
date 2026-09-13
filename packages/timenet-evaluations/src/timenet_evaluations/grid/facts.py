"""What one dataset is, beyond the numbers in its frame.

A frame carries values, labels and subjects. It does not carry what the values measure, at what
rate, under which channel names, or which vocabulary the labels belong to. A conversion needs all of
that and can derive none of it, so a dataset states it.

This is not a representation and it does not live with them. A representation is one of the two
forms a dataset is held in for a run; this is a description of the dataset itself, true before
either form exists. Keeping it here is what lets a connector's declaration carry it: the declaration
may not read the representation layer, because a declaration derived from a representation would let
the thing under test choose the denominator its own rates divide by.

Nothing here imports the storage library. The module is pure description, so it resolves in the
environment continuous integration builds, where the extra is absent.
"""

# model names a card rather than copying one)

from __future__ import annotations

from pydantic import BaseModel


class DatasetFacts(BaseModel):
    """What the conversion needs about one dataset that neither the frame nor the card carries.

    Each field is a fact about one release. No field has a default, and no module here holds one:
    a default would be this capability naming a dataset, and a second dataset with a different
    modality, a different unit and a different task would then need a subject under test to be
    edited before it could be measured.

    **The dataset's identity is not here.** Its id, version, display name, description and licence
    live in the card that ``timenet-connectors`` ships beside the connector for that release, which
    that card calls the single source for them, and :func:`card_for` is how they are read. This
    model held copies of all five until ADR-0028, and the copies had drifted: the id was a bare
    name where a card states an ``org/name`` pair, which the storage library rejects outright.

    What remains is the half a card does not describe. A card says who a dataset is; these say what
    one item of it holds, and the storage library derives its own schema from the data rather than
    from the card. A future release that states its channels or its rate in machine-readable form
    can have them read from there, and this model shrinks again.
    """

    card_id: str
    """The dataset id naming the card this release's identity is read from, as ``org/name``.

    It is a reference and not a copy. :func:`card_for` resolves it against the connectors
    distribution, and the id, version, name, description and licence all come back from there.
    """
    modality: str
    """The type tag every channel shares, because one modality covers them all."""
    modality_name: str
    """The modality's display name."""
    unit: str
    """The unit the values are measured in, as a name the unit registry recognizes."""
    rate_hz: int
    """The rate every channel has been resampled to, in values per second. It is a whole number
    because every real sampling rate is, and because a rate that is not whole has no single
    reading: 29.97 is 2997/100 by its spelling and 30000/1001 by its intent."""
    channels: tuple[str, ...]
    """The release's channel names, in the order the signal's rows are in. The conversion refuses
    a signal whose row count does not equal this many, because the layout could not represent it
    and the reads would mis-shape it."""
    target_schema: str
    """The name of the label vocabulary the scored class belongs to."""
