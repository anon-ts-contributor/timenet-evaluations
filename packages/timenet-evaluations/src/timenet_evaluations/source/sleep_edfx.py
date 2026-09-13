"""The one connector that ships: Sleep-EDF, read through PyHealth.

This module and the ``sleep_edfx*`` modules beside it are the only place in the package that names a
dataset or imports a dataset library. The frame this one returns is described by
:mod:`timenet_evaluations.source.contract`, and every module downstream of the connector sees that
frame and nothing else.

The connector is one object with the four members
:class:`~timenet_evaluations.source.connector.Connector` declares. It is registered under one name
in :mod:`timenet_evaluations.source.registry`, and that name is the dataset coordinate of every
measurement key and every storage key a run built from it produces.

The name selects this connector, and the path it is bound to is the root of the expanded database:
the directory that holds the subject spreadsheet and the recordings beside each other.
:func:`check_source_root` refuses a path that is not that root, and it refuses it by name. PyHealth
resolves both of those relative to the root it is given, so a path one level too deep fails inside
PyHealth's metadata preparation with a missing-file error that names neither this dataset nor the
layout it wanted.

The connector is more than one module, because one PyHealth call produces most of what it needs and
that call has a lot to say for itself. :mod:`timenet_evaluations.source.sleep_edfx_preparation`
holds the untimed pass — the reach into PyHealth, the frame it drains, the counts it establishes,
the cache it leaves on disk, the gate that stops a timed cell rebuilding that cache, and the two
statements every report carrying this dataset's rows has to make. The dependency runs one way: that
module knows nothing of this one, so the registered name reaches it as an argument.

The declaration is what this dataset says about itself before a run measures it, and :func:`declare`
assembles it from what the preparation established. The unit and the rate are literals of this
release. The epoch count and the channel count are not: they depend on the hypnograms and the
channels of the recordings a run was given, so both are read rather than written down. That is why
:meth:`SleepEdfxConnector.canonical_item` takes the source it is asked about. One connector is
registered per name and one instance serves every run, so a member that answered without a source
could only answer from a count it had stored for some earlier path.

The five assumptions behind that declaration are recorded in :data:`ASSUMPTIONS`, beside it, each
with what breaks if it is wrong.

The two timed opens return the two adapters in
:mod:`timenet_evaluations.source.sleep_edfx_access`, which is where this connector's contribution to
the four read operations lives. Both reach PyHealth through
:func:`~timenet_evaluations.source.sleep_edfx_preparation.open_samples`, where the cache gate sits,
so a timed open against an unbuilt cache refuses before the build rather than times it.
"""

# opens)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.facts import DatasetFacts
from timenet_evaluations.grid.item import ItemDeclaration
from timenet_evaluations.source.connector import ParsedItems
from timenet_evaluations.source.contract import check_source_directory
from timenet_evaluations.source.declaration import DatasetDeclaration
from timenet_evaluations.source.sleep_edfx_access import open_items, open_signals
from timenet_evaluations.source.sleep_edfx_preparation import DISCLOSURES, Preparation, prepare


NAME = "sleep-edfx"
"""The name a run selects this connector with, and the dataset coordinate of every key it produces.

The name is a fact about this dataset, so it lives beside the connector rather than in the registry
that maps it. Nothing infers it from a directory name, from the contents of a path, or from anything
else about where the recordings sit.
"""

SUBJECT_SPREADSHEET = "SC-subjects.xls"
"""The subject spreadsheet the cassette subset of this release ships in its root.

PyHealth reads this file to build its metadata table, and it resolves the name relative to the root
it is given. The telemetry subset ships a different spreadsheet, and it is a different corpus with
its own item count, so it is a second registered name rather than this one pointed elsewhere.
"""

RECORDINGS_DIRECTORY = "sleep-cassette"
"""The directory of recordings the cassette subset of this release ships beside the spreadsheet.

The root is the directory that holds both entries. A run that names this directory itself is one
level too deep, and that is the mistake :func:`check_source_root` exists to refuse.
"""

UNIT = "epoch"
"""The name of one canonical item, as this dataset's own task names it.

One epoch is one scored 30-second window of one recording, across every channel that recording
carries. It is the smallest unit the sleep-staging task consumes, and every rate this dataset
reports is quoted per one of them.
"""

RATE_HZ = 100
"""The rate every channel of an item is delivered at, in values per second.

This is a property of what the read path yields, and it is assumption one below. It is not a
resampling anything here asks PyHealth to perform.
"""

CHUNK_DURATION_SECONDS = 30
"""How many seconds of one recording one item covers, which is assumption two below.

The task's window duration decides the unit, and this connector leaves that argument at the task's
own default. A change to the argument moves the unit every rate divides by, so it moves this number
in the same commit.
"""

SAMPLES_PER_ITEM = RATE_HZ * CHUNK_DURATION_SECONDS
"""How many values one channel of one item holds.

The number is the product of the two facts above rather than a third fact beside them. A change to
the rate or to the window duration that left this number alone would declare a shape that no read
produces.
"""


CHANNELS = (
    "Fpz-Cz",
    "Pz-Oz",
    "horizontal",
    "oro-nasal",
    "submental",
    "rectal",
    "Event marker",
)
"""The release's channel names, in the order the signal's rows are in.

These are the names ``mne.io.read_raw_edf`` reports for a cassette recording read with
``infer_types=True``, which is how the reference loader reads one. The task picks no subset, so
every channel of the file becomes a row of the item, in this order.

The conversion refuses a signal whose row count is not ``len(CHANNELS)``, so a release carrying a
different set stops the run rather than writing rows under the wrong names. That refusal is the only
thing standing behind this tuple, which is why the names are stated rather than counted: a count
would agree with any seven channels at all.
"""

FACTS = DatasetFacts(
    card_id="physionet/sleep-edfx",
    modality="psg",
    modality_name="Polysomnography",
    unit="microvolt",
    rate_hz=RATE_HZ,
    channels=CHANNELS,
    target_schema="sleep-stage",
)
"""What the conversion needs about this release beyond its card and its frame.

The identity is not here. ``card_id`` names the card ``timenet-connectors`` ships for this dataset,
and the id, version, name, description and licence are read from it. Per ADR-0028 this repository
does not keep a second copy of any of them.

``modality`` and ``unit`` are the two fields this dataset cannot answer honestly, and assumption six
below says so. Both are declared because the card's schema allows one of each, over a corpus that
has several.
"""


@dataclass(frozen=True)
class Assumption:
    """One thing this connector assumes about the corpus, with what breaks if it is wrong.

    The consequence is not decoration. An assumption stated on its own is a comment. An assumption
    stated with what it costs is a check a reviewer can apply to a release nobody here has seen.
    """

    statement: str
    """What this connector assumes, and where in the read path the assumption comes from."""

    consequence: str
    """What happens to the declared item, the count or the figures when the assumption is wrong."""


ASSUMPTIONS: tuple[Assumption, ...] = (
    Assumption(
        statement=(
            "Every channel arrives at 100 Hz. The release samples the EEG and EOG channels at 100 Hz, and the "
            "EMG envelope, the oro-nasal airflow, the rectal temperature and the event marker at 1 Hz. "
            "mne.io.read_raw_edf brings the whole file up to its highest rate, so the reference loader reports "
            "100 Hz and delivers every channel at it. The rate is a property of what this read path yields. "
            "Nothing here asks for a resampling, and nothing here would notice if the release changed."
        ),
        consequence=(
            "The declared item's last axis is the rate times the window duration. A release delivered at another "
            "rate therefore declares a shape no read produces, and the parity check fails on the shape. A "
            "conversion that builds a time axis from this number instead writes the right values against the "
            "wrong axis, and nothing downstream compares an axis."
        ),
    ),
    Assumption(
        statement=(
            "One item covers 30 seconds, which is the window duration the sleep-staging task defaults to. This "
            "connector leaves that argument at the default and passes no duration of its own."
        ),
        consequence=(
            "The window duration is the definition of the denominator. A different one moves every items/s "
            "figure on this dataset's rows by the ratio of the two units, and the result still validates and "
            "the table still renders."
        ),
    ),
    Assumption(
        statement=(
            "Every recording carries the same channels, so one item's shape holds across the dataset. Nothing "
            "in PyHealth checks this, so the untimed preparation does: it reads the items one at a time and "
            "compares the channel count of each one against the first. The channel count is read from the items "
            "that pass and is recorded beside the declared item, so a reader can tell what shape the numbers "
            "describe."
        ),
        consequence=(
            "A recording whose channel set differs from the rest stops the run in the untimed preparation, with "
            "a message naming both counts and the item that disagreed. How many samples one channel holds is "
            "not compared, so a release that varied there loads without complaint and the failure surfaces as a "
            "raw numpy shape error inside the untimed conversion."
        ),
    ),
    Assumption(
        statement=(
            "A window outside the six scored stages is not an item. PyHealth cuts the epochs against an event "
            "map naming Sleep stage W, 1, 2, 3, 4 and R, so 'Sleep stage ?' and 'Movement time' produce no "
            "event and therefore no epoch."
        ),
        consequence=(
            "The item count is a property of the hypnograms rather than of the recording durations, which is "
            "the assumption that most changes the count. It is also why the count is established by reading a "
            "corpus rather than written down here: a copy of the release with other annotations holds a "
            "different number of items and nothing would detect a literal that disagreed."
        ),
    ),
    Assumption(
        statement=(
            "The cassette subset is what is measured, because SleepEDFDataset defaults to it and this "
            "connector passes no subset of its own."
        ),
        consequence=(
            "The telemetry subset is a different corpus with its own item count. A run over it is a second "
            "registered name carrying its own declaration, not this name pointed at another directory, and a "
            "change that switched the subset would have to move the name, the declaration and this record "
            "together."
        ),
    ),
    Assumption(
        statement=(
            "One modality and one unit describe every channel. They do not. A cassette recording carries two "
            "EEG derivations, one EOG, an EMG envelope, oro-nasal airflow, a rectal temperature and an event "
            "marker, so the seven rows of an item are five kinds of signal in at least three units. The card "
            "schema holds one modality and one unit, so this connector declares the composite recording: "
            "polysomnography, in microvolts, which is what the electrode channels that dominate the file are "
            "measured in."
        ),
        consequence=(
            "The temperature and the airflow rows are labelled with a unit that is not theirs, in the "
            "converted artifact's metadata and nowhere else. No figure this benchmark reports is affected: a "
            "run measures how long bytes take to come back and how many of them there are, and neither reads "
            "a unit. What it costs is a reader of the converted artifact who trusts its metadata, and a "
            "future dataset whose channels really are one modality will not reveal that this one's are not."
        ),
    ),
)
"""The six things this connector assumes about the corpus, each with what breaks if it is wrong.

They sit beside the declaration because each one is a statement about what makes the declared item
what it is: the rate that fixes 3000 samples, the window duration that fixes 30 seconds, the channel
count that fixes the first axis, the dropped annotations that fix the count, the subset that fixes
which corpus the count is of, and the single modality the card claims over channels that have
several.

They are recorded and not enforced. Nothing here reads a release to find out whether they hold, and
a corpus that broke one would be measured anyway. The record is what lets someone feeding this
connector a different release read what it was written against.
"""


def check_source_root(source: Path) -> None:
    """Make sure that a source path is the root of this dataset's release.

    Call this before PyHealth is touched, in the untimed preparation and in each timed open.
    :func:`~timenet_evaluations.source.contract.check_source_directory` stays the general check and
    runs first: a path that is not a directory is refused there, and a directory that is not this
    release's root is refused here.

    The check reads the two names beside the path and does nothing else. It writes nothing below the
    source path, and it does not construct the reference loader to find out. That construction is
    what derives the metadata table into the source directory, and when it happens is a decision of
    the untimed preparation rather than of a precondition.

    The refusal exists because of where the failure lands without it. PyHealth resolves the subject
    spreadsheet and the recordings relative to the root it is given, and it defaults to the cassette
    subset. A path one level too deep therefore fails inside PyHealth's metadata preparation, with a
    missing-file error that names neither this dataset nor the layout it wanted.

    The check is written against the cassette subset, which is the only subset this connector
    measures. It does not accept the telemetry subset as well: that corpus has its own item count,
    so it is a different registered name with its own declaration.

    Args:
        source: The directory a run gave this dataset, which must hold the subject spreadsheet and
            the recordings directory beside each other.

    Raises:
        EvaluationError: If ``source`` is not a directory, or if it does not carry both of the
            entries the root of this release carries.
    """
    check_source_directory(source)

    missing = [
        name
        for name, present in (
            (SUBJECT_SPREADSHEET, (source / SUBJECT_SPREADSHEET).is_file()),
            (RECORDINGS_DIRECTORY, (source / RECORDINGS_DIRECTORY).is_dir()),
        )
        if not present
    ]
    if missing:
        raise EvaluationError(
            f"the source path is not the root of this dataset's release: dataset {NAME!r}, path "
            f"{source}, missing {missing}. The root holds the subject spreadsheet "
            f"{SUBJECT_SPREADSHEET!r} and the recordings directory {RECORDINGS_DIRECTORY!r} beside "
            f"each other, so a path that names the recordings directory is one level too deep and "
            f"the root is its parent. Refused here rather than inside the reference loader, which "
            f"resolves both names relative to the root it is given and fails with a missing-file "
            f"error that names neither this dataset nor the layout it wanted"
        )


def declare(prepared: Preparation) -> DatasetDeclaration:
    """Turn what the untimed preparation established into this dataset's declaration.

    Two of the four facts are literals of this release, and two are not. The unit and the rate are
    written above. The item count and the channel count are read from ``prepared``, because neither
    is knowable without parsing the hypnograms and the recordings of the corpus a run was given.

    The count comes from the rows the preparation drained and from nothing else. It is never the
    length the PyHealth sample object reports about itself: ``litdata.StreamingDataset.__len__`` is
    ``get_len(num_workers, batch_size)``, so that length is a function of the loader configuration
    in force at the moment it is read rather than a property of the dataset.
    :func:`check_reported_length` is where that length is compared with this count, and it is the
    only thing that length is ever used for.

    The item shape is the observed channel count against :data:`SAMPLES_PER_ITEM`, which is the rate
    times the window duration. A change to either one moves the shape, so a shape that stayed still
    while one of them moved is a disagreement the parity check finds.

    Args:
        prepared: What the one untimed pass over this dataset established.

    Returns:
        The unit, the item count and shape, the rate, and what this dataset's cells really read.
    """
    return DatasetDeclaration(
        unit=UNIT,
        item=ItemDeclaration(count=prepared.item_count, shape=(prepared.channel_count, SAMPLES_PER_ITEM)),
        rate_hz=RATE_HZ,
        facts=FACTS,
        disclosures=DISCLOSURES,
    )


def check_reported_length(reported: int, declaration: DatasetDeclaration, *, source: Path) -> None:
    """Make sure that the length PyHealth reports agrees with the count this dataset declared.

    Call this from a timed open, after the sample object is in hand. The length is a cross-check and
    is never the source of the count. ``litdata.StreamingDataset.__len__`` is
    ``get_len(num_workers, batch_size)``: it sets both attributes as a side effect and returns a
    value that depends on them, so a divisor taken from it would move with a loader setting nobody
    recorded.

    A disagreement stops the run. It says that the object a timed cell is about to read holds a
    different number of items from the number every rate for this dataset divides by, and neither
    number can be trusted until someone reads both.

    Args:
        reported: The length the sample object reports about itself.
        declaration: What this dataset declared, whose count is the one every rate divides by.
        source: The source path the open was given, for the message. A run measures more than one
            dataset, so a message without it cannot say which open disagreed.

    Raises:
        EvaluationError: If the reported length is not the declared count.
    """
    if reported != declaration.item.count:
        raise EvaluationError(
            f"the reference loader reports a different number of items from the number this dataset "
            f"declared: dataset {NAME!r}, path {source}, declared count {declaration.item.count}, "
            f"reported length {reported}. The declared count is established once by the untimed "
            f"preparation and is what every rate on this dataset's rows divides by. The reported "
            f"length is a function of the loader configuration in force when it is read, so it is "
            f"compared with the declared count here and is never the source of it"
        )


class SleepEdfxConnector:
    """The Sleep-EDF Database Expanded, read through PyHealth.

    PyHealth is the release's own reference loader, which is what the release's files are read
    through where one exists. The object holds no path, no frame and no result: a run gives it the
    source directory on every call.

    The connector holds no state at all today, so its three call members take no instance. The
    protocol is satisfied by shape, so a member that later needs state becomes an instance method
    and no caller changes. The registered name is not a member of this object: it is the key of the
    one entry in the registry, because the protocol surface is four members and a fifth is a member
    every future dataset pays for.
    """

    @staticmethod
    def load_for_conversion(source: Path) -> pd.DataFrame:
        """Read this dataset into memory, untimed, for the conversion to write from.

        This frame is the input the untimed conversion writes the converted representation from. The
        release's own files are read as they ship and are never produced from this frame.

        The read is eager and complete. It must not return a generator, an iterator or a lazy view:
        source input and output that is deferred out of the load lands in a phase whose page cache
        state the cold-read protocol does not control.

        :func:`check_source_root` runs first, so an ``EvaluationError`` names a source path that is
        not this release's root before PyHealth is touched.

        The body of the read is
        :func:`~timenet_evaluations.source.sleep_edfx_preparation.prepare`, and that module says
        what the pass leaves behind. Three steps happen there. ``SleepEDFDataset`` reads the
        metadata table, deriving it into the source directory when it is not already there.
        ``set_task`` parses each EDF, cuts it into 30-second epochs, drops those outside the six
        scored stages, and **writes every epoch to a cache**, which is what makes it a build step
        rather than an accessor. Draining the dataloader then copies those values from that cache
        into memory: the signals are already materialized on disk before the loader is constructed,
        so a timed cell over this dataset reads the cache and never parses an EDF. Every failure
        below those three propagates with its own type: an unreadable recording stops the run rather
        than leaves a subject out of the frame.

        The three columns are what PyHealth hands back, narrowed to the contract. Two of them read
        the wrong way round unless they are said out loud. ``label`` is an **integer class index**
        between 0 and 5, produced by the task's multiclass output processor, and the frame carries
        its string form; it is not a stage name such as ``"W"``. And ``patient_id`` is the subject
        number from the release's own spreadsheet, so it identifies a **subject and not a
        recording**: one subject contributes two nights, and ``night`` is the key that separates
        them and is one of the keys the drain drops.

        This is not the same activity as :meth:`open_pandas`, and both use the same reference
        loader. This one is untimed and happens once per dataset. That one is a timed cell, repeated
        with the page cache dropped before each repetition. The duration of this call is not any
        task's figure, and it is not this dataset's full-read time however similar the work looks.

        Args:
            source: Directory holding the release, which is the root that carries the subject
                spreadsheet and the recordings beside it.

        Returns:
            One row per scored epoch, with the three columns of the frame contract.
        """
        check_source_root(source)

        return prepare(source, name=NAME).frame

    @staticmethod
    def open_pandas(path: Path) -> ParsedItems:
        """Open the release's own files for the Pandas reader, inside a timed cell.

        This call is inside the timer, so everything it does is part of what the cell reports:
        constructing the reference loader, asking it for the task, and reading the cache metadata.
        :func:`~timenet_evaluations.source.sleep_edfx_access.open_items` is the body, and the
        adapter it returns says what each read after the open costs.

        This is not the same activity as :meth:`load_for_conversion`, and both use the same
        reference loader. That one is untimed and happens once per dataset. This one runs once per
        repetition, with the page cache dropped before each, and it is served from no frame, no
        cached object and no reader a previous repetition left open.

        Args:
            path: The artifact path of the release's own files, which is the source directory the
                run was given.

        Returns:
            Positional access to this dataset's canonical items, narrow for one item and for a run.
        """
        check_source_root(path)

        return open_items(path, name=NAME)

    @staticmethod
    def open_torch(path: Path) -> Dataset[object]:
        """Open the release's own files for the PyTorch reader, inside a timed cell.

        The call goes through the reference loader's own torch path and builds no frame. It is not
        built on :meth:`open_pandas`: the figure of this cell is supposed to describe the release's
        torch path, and every cost of the other target form would otherwise sit inside this column.

        The dataset is map-style, because the harness wraps it in the one dataloader configuration
        it measures. This connector constructs no dataloader, names no batch size and names no
        worker count.

        Args:
            path: The artifact path of the release's own files, which is the source directory the
                run was given.

        Returns:
            A map-style dataset whose item at a position is that item's signal, as a ``float32``
            tensor of the declared item shape.
        """
        check_source_root(path)

        return open_signals(path, name=NAME)

    @staticmethod
    def canonical_item(source: Path) -> DatasetDeclaration:
        """State what one epoch of this dataset is, for the source path this call names.

        Call this in untimed work, before the first timer of that source. One epoch is one scored
        30-second window of one recording, across every channel that recording carries, at 100 Hz.
        The five things this declaration assumes about the corpus are recorded in
        :data:`ASSUMPTIONS`.

        The member takes the source because the count is not a literal. It is the number of scored
        windows the hypnograms of the given recordings hold, so a second root under this name holds
        a different number. The connector is registered once and one instance serves every run, so
        it stores neither the path nor the answer:
        :func:`~timenet_evaluations.source.sleep_edfx_preparation.prepare` reads the source this
        call was given, and :func:`declare` turns what it established into the declaration.

        The preparation is the one pass over this dataset that is neither a representation nor a
        reader, which is what makes it the only honest source of the count. It is untimed, and no
        reported figure includes it. It builds PyHealth's task cache when that cache is not already
        there, so a run that reaches this member first pays for the build here rather than inside a
        timer.

        Args:
            source: Directory holding the release, which is the root that carries the subject
                spreadsheet and the recordings beside it.

        Returns:
            The unit, the number of epochs this source holds, the shape of one of them, the rate,
            and what this dataset's measured cells really read.
        """
        check_source_root(source)

        return declare(prepare(source, name=NAME))
