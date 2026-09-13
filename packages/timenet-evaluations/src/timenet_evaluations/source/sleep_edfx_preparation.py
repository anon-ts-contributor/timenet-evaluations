"""The untimed preparation: the one pass that reaches PyHealth, and the state it leaves behind.

Everything this connector needs is a product of one call chain. ``SleepEDFDataset`` reads the
subject spreadsheet and derives its own table beside the recordings. ``set_task`` parses every
recording, cuts the scored windows, applies the schema's processors and writes the result as chunks
under the user cache directory. Draining the sample object copies those values into memory, and the
frame the conversion writes from is what comes out.

That call chain is one story because it is one call. ``set_task`` is a build step and an accessor
behind one signature, and only the filesystem tells the two apart: the first call parses the whole
corpus and writes it, and every later call finds the completed index and returns a view over what is
already there. Same function, same arguments, wildly different cost.

**This is why a timed cell must never be the caller that builds.** Left alone, repetition one of a
timed cell would price a corpus parse and a multi-gigabyte write, repetitions two to five would
price a cache hit, and the median over those describes no workload at all. Worse, which of the two a
run gets depends on whether the machine happened to run the benchmark before. :func:`check_task_cache`
is the refusal, :func:`open_samples` is the only route to ``set_task`` a timed open takes, and the
refusal sits inside it. That is what makes the rule self-enforcing rather than reviewed.

The gate answers one question — would this ``set_task`` build? — and it answers it without building
anything. It computes the path of the index the library writes last, from the cache directory the
constructed dataset reports and from the task's own arguments, and it looks. It creates no
directory, opens no recording and constructs nothing. :func:`prepare` confirms that arithmetic
against reality once, in untimed work, immediately after the build: if the index this module
computes is not where the build left one, the run stops there with a message about the cache layout
rather than later, inside a timed open, with a refusal nobody can explain.

The drain reads that sample object one item at a time and reaches no dataloader. ``get_dataloader``
installs PyHealth's ``collate_fn_dict_with_padding``, which pads the first axis of the values it
stacks whenever two of them disagree on shape. The first axis of one item is the channel axis, so a
batch that held an item with two channels beside an item with three came back as three channels for
both, and the item that had two carried a third channel of zeros this harness invented. Nothing
downstream could see it: the frame was uniform, the conversion succeeded, both representations held
the fabricated channel, and the parity check compared it against itself. So the values are read
without a collate, and :func:`drain` compares the channel count of every item against the first.

Two facts about this dataset have to reach every report that carries its rows, and they are the two
statements in :data:`DISCLOSURES`. Both are consequences of the preparation above, so they are
stated here, by the code that causes them, and never composed by a reporter. They travel to the
result inside the declaration, keyed by the representation they describe.

Nothing here writes or deletes below the source path. The one write into that directory is the
library's own, at construction, and it is disclosed rather than prevented or undone: preventing it
means not using the release's reference loader, and undoing it means the next run pays to build it
again inside whichever step runs first.

The registered name arrives as an argument rather than as an import. The module that binds it
imports this one, so the dependency has to run one way, and a message that names the wrong dataset
is worse than a parameter.
"""

# is not one of the readings that are measured)
# so the file state is fixed here)

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import uuid

import pandas as pd
from pyhealth.datasets import SampleDataset, SleepEDFDataset

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.source.contract import PATIENT, SIGNAL, as_row
from timenet_evaluations.source.sleep_edfx_task import SleepStaging


TASK_CACHE_DIRECTORY = "tasks"
"""The directory the library keeps one subdirectory per task under, below its own cache directory."""

SAMPLES_CACHE_PREFIX = "samples_"
"""The first part of the name of the directory the processed items are written to."""

SAMPLES_CACHE_SUFFIX = ".ld"
"""The last part of that name."""

SAMPLES_CACHE_INDEX = "index.json"
"""The file the chunk writer creates last, after every chunk is flushed.

Its presence is what the library itself tests to decide whether it has anything to build, so it is
the one file that answers the gate's question. A directory that exists and holds chunks but no index
is an interrupted build, and the library rebuilds it.
"""

ORIGINAL = "original"
"""The name of the representation the two statements below describe.

The release's own files are one representation of this dataset, and the timed cells over them are
the cells the statements are about. The name is written here rather than imported: this package
depends on the grid by shape and not by import, and a connector that reached into a representation
to read its name would be the second dataset's problem as well as this one's.
"""

CACHE_DISCLOSURE = (
    "These figures measure a cold read of the task cache PyHealth materialised during untimed "
    "preparation, through PyHealth's own reader. They are not a parse of the release's own "
    "containers. That is the read path an adopter of PyHealth pays on every run after the first, "
    "which is why it is the figure worth measuring, but the heading does not say so on its own."
)
"""What this dataset's cells over the release's own files really read.

The preparation writes the whole corpus to a cache, and every timed repetition afterwards reads that
cache. The heading over those columns names the release, so without this sentence a reader takes the
number for the cost of parsing the containers the release ships.
"""

STORAGE_DISCLOSURE = (
    "The storage figure includes the metadata table PyHealth derives into the source directory the "
    "first time it opens it. It is a few kilobytes against a corpus of gigabytes, and it is left in "
    "place on purpose: deleting it would make the next run rebuild it inside whichever step ran "
    "first."
)
"""What this dataset's storage figure over the release's own files includes.

Constructing the reference loader writes one derived table beside the recordings. This harness must
not write below a source path and does not; the release's own loader does it anyway, and there is no
way to stop it while still reading through that loader. So the figure carries a file this run caused
to exist, and the report says so rather than presenting the number as the release's bytes alone.
"""

DISCLOSURES: Mapping[str, str] = {ORIGINAL: f"{CACHE_DISCLOSURE} {STORAGE_DISCLOSURE}"}
"""What every report carrying this dataset's rows must state, by the representation it is about.

One statement per representation, which is the shape the declaration carries and the result records.
The converted representation has no entry: its cells read an artifact this run wrote on purpose,
under a heading that says so, and there is nothing to disclose. Absence is the signal, so it must
not be printed as an empty note.
"""


@dataclass(frozen=True)
class Preparation:
    """What the one untimed pass over this dataset established.

    The frame is the conversion's input. The two counts are the facts nothing can know without
    parsing the recordings, and they are established here because this pass is the one reading that
    is neither a representation nor a reader. A count a representation reported about itself would
    be the subject under test choosing the divisor its own rates are quoted against.
    """

    frame: pd.DataFrame
    """One row per scored window, with the three columns of the frame contract and no others."""

    item_count: int
    """How many scored windows the drain produced, counted as they were produced.

    This is the number the dataset declares and every rate divides by. It is not the length the
    sample object reports about itself: that length is a function of the loader configuration in
    force when it is read, so it is asserted equal to this number and never consulted for it.
    """

    channel_count: int
    """How many channels one window carries, read from the first item the drain produced.

    The drain compares every later item against it and stops the run when two disagree, because a
    channel count that changed would otherwise reach the declared item shape as one number for a
    dataset that has two. How many samples one channel holds is not compared, which is the frame
    contract's assumption and not a new one.
    """


def task_cache_index(cache_dir: Path, task: SleepStaging) -> Path:
    """Work out where the completed task cache for one task would be.

    This is arithmetic over names. It reads nothing, creates nothing and constructs nothing, which
    is what lets the gate below answer "would this build?" without building.

    The path is a function of the task's own arguments, so a cache built for a different window
    duration lives somewhere else and does not answer for this one. That is deliberate: a gate that
    accepted any cache under the dataset would pass a build that the next ``set_task`` would then
    redo inside a timer.

    The layout is the library's, and this function tracks it rather than restates it from memory. If
    the library moves the layout, this function points at nothing, the gate refuses and
    :func:`prepare` says so during untimed work. That direction is safe; the other one, a gate that
    passed while the build was still to come, is the failure the gate exists to prevent.

    Args:
        cache_dir: The cache directory the constructed dataset reports. It is the library's own user
            cache directory by default, and it is never below the source path, where the storage
            measurement would count it as part of the release.
        task: The task the samples would be built for. Its arguments name the cache.

    Returns:
        The path of the index file a completed build leaves behind.
    """
    task_params = json.dumps(
        {**vars(task), "input_schema": task.input_schema, "output_schema": task.output_schema},
        sort_keys=True,
        default=str,
    )
    processor_params = json.dumps({"input_processors": None, "output_processors": None}, sort_keys=True, default=str)

    task_cache = cache_dir / TASK_CACHE_DIRECTORY / f"{task.task_name}_{uuid.uuid5(uuid.NAMESPACE_DNS, task_params)}"
    samples = f"{SAMPLES_CACHE_PREFIX}{uuid.uuid5(uuid.NAMESPACE_DNS, processor_params)}{SAMPLES_CACHE_SUFFIX}"

    return task_cache / samples / SAMPLES_CACHE_INDEX


def check_task_cache(cache_dir: Path, task: SleepStaging, *, name: str, source: Path) -> None:
    """Refuse a timed open that would build the task cache instead of reading it.

    Call this inside every timed open, before the call that would build. :func:`open_samples` is the
    only route this connector's opens take to that call, and it calls this first, so the refusal is
    a property of the code rather than of the review.

    The check is one look at one path. It builds nothing, so a run that reaches it with no cache
    stops having spent nothing.

    Args:
        cache_dir: The cache directory the constructed dataset reports.
        task: The task the timed open is about to ask for.
        name: The registered name of the dataset, for the message. A run measures more than one
            dataset, and a message that omits the name cannot say which one refused.
        source: The source path the timed open was given, for the message.

    Raises:
        EvaluationError: If the completed cache is not there, because the call that follows would
            then parse the whole corpus and write it while a timer ran.
    """
    index = task_cache_index(cache_dir, task)
    if not index.is_file():
        raise EvaluationError(
            f"a timed open found no task cache and must not build one: dataset {name!r}, path "
            f"{source}, cache index {index}. The call this open is about to make would parse every "
            f"recording in the release, cut the scored windows and write them as chunks, all of it "
            f"inside the timer of a cell that reports a read. The cache is built once, untimed, by "
            f"this connector's preparation, and every repetition afterwards reads it cold. Run the "
            f"preparation for this source path first"
        )


def open_samples(source: Path, *, name: str) -> SampleDataset:
    """Open the release's own files through the reference loader, for a timed cell.

    Call this from a timed open, once per repetition. It reaches the library itself every time: it
    is served from no frame, no cached object and nothing a previous repetition left open, so the
    figure of the cell that wraps it describes the release's loader rather than an attribute access.

    The gate runs before the call that could build, which is the whole reason this function exists
    rather than each open reaching the library itself. Construction is cheap once the preparation
    has run: the derived table is already beside the recordings, so the constructor reads it instead
    of building it, and the task call finds a completed index instead of a corpus to parse.

    :func:`check_task_cache` refuses an unbuilt cache from inside this call, so a timed cell that
    would have built one stops before the build rather than times it.

    Args:
        source: The root of the release, checked to be that root by the caller before this call.
        name: The registered name of the dataset, for the gate's message.

    Returns:
        The sample object the reference loader serves the scored windows from.
    """
    dataset, task = _reach(source)
    check_task_cache(dataset.cache_dir, task, name=name, source=source)

    return dataset.set_task(task)


def prepare(source: Path, *, name: str) -> Preparation:
    """Read this dataset once, untimed, and leave every timed cell the state it assumes.

    Call this exactly once per dataset per run, in untimed work, before the storage figure is
    measured and before the first repetition of any cell. It is not the same activity as a timed
    open, however alike the two look: this one runs once and no reported figure includes it, and
    that one runs inside a timer, once per repetition, against the artifacts this one left behind.

    The pass leaves three things on disk and in memory. The reference loader's constructor derives
    its metadata table into the source directory, which is the one write below that path and is
    disclosed rather than prevented. The task call parses every recording and writes the values as
    chunks under the library's cache directory, which is what every timed cell then reads. And the
    drain returns the frame, the item count and the channel count.

    Nothing is gated here. This is the pass that builds, so a gate in front of it would refuse the
    only call allowed to do the building. What runs instead is the check afterwards: the index this
    module computes must be where the build left one, and a disagreement stops the run here, in
    untimed work, rather than as an unexplainable refusal inside a timed open later.

    Args:
        source: The root of the release, checked to be that root by the caller before this call.
        name: The registered name of the dataset, for the messages.

    Returns:
        The frame the conversion writes from, and the two counts nothing can know without this pass.

    Raises:
        EvaluationError: If the build left no index where this module computes one, or if the drain
            produced no items at all.
    """
    dataset, task = _reach(source)
    samples = dataset.set_task(task)

    index = task_cache_index(dataset.cache_dir, task)
    if not index.is_file():
        raise EvaluationError(
            f"the untimed preparation ran and left no cache index where this connector computes "
            f"one: dataset {name!r}, path {source}, cache index {index}. The reference loader's "
            f"cache layout has moved, so the gate every timed open calls cannot see the cache this "
            f"pass just built and would refuse a cell that is in fact ready. Refused here, in "
            f"untimed work, rather than inside a timer later"
        )

    return drain(samples, name=name, source=source)


def drain(items: Iterable[Mapping[str, Any]], *, name: str, source: Path) -> Preparation:
    """Pull every item out of the sample object and establish what this dataset holds.

    The read is eager and complete. It returns a materialized frame rather than a generator, an
    iterator or a lazy view: every timed repetition starts from a dropped page cache, so input and
    output deferred out of this call would land in a phase whose page cache state nothing controls.

    The items arrive one at a time and no dataloader wraps them, so no collate function stands
    between the sample object and this frame. A collate is the one step that can change the shape of
    what it is given, and PyHealth's pads the channel axis, so the drain reads the values itself and
    the module docstring above says what the padding did.

    Each item becomes one row through the frame contract's own helper, so the frame carries the
    three contract columns and nothing else. The sample object yields more keys than that — the
    night the recording was made, the subject's age and the subject's sex all arrive on every item,
    because the schema passes through what it has no processor for. Naming the three is what drops
    them. They would otherwise reach the frame, travel into the converted representation, and have
    no counterpart in the release's own files.

    The channel count of the first item is the count this dataset declares, and every later item is
    compared against it. Two items that disagree stop the run here. A run that continued would
    declare one shape for a dataset that has two, and every figure it printed would be about a shape
    no item has.

    The count comes from the rows this drain produced, and from nothing else. The sample object
    reports a length of its own, and that length is a function of the loader configuration in force
    when it is read, so it is a cross-check and never the source.

    Args:
        items: The items the sample object yields, each one a dict of one item's values.
        name: The registered name of the dataset, for the message.
        source: The source path this dataset was read from, for the message.

    Returns:
        The frame, the number of items the drain produced, and the channel count of one item.

    Raises:
        EvaluationError: If two items disagree on how many channels they carry, or if the drain
            produced no items, because an empty frame is a value a caller can mistake for a good
            read of an empty dataset.
    """
    rows: list[dict[str, Any]] = []
    channel_count = 0
    for position, item in enumerate(items):
        row = as_row(item)
        channels = int(row[SIGNAL].shape[0])

        if position == 0:
            channel_count = channels
        elif channels != channel_count:
            raise EvaluationError(
                f"two items of this release carry a different number of channels: dataset "
                f"{name!r}, path {source}, item 0 carries {channel_count} channels and item "
                f"{position}, of patient {row[PATIENT]!r}, carries {channels}. The declared "
                f"item, the conversion and the parity check all describe one shape per dataset, "
                f"so a run that continued would report every figure against a shape no item has. "
                f"Refused here, in untimed work, rather than measured"
            )

        rows.append(row)

    if not rows:
        raise EvaluationError(
            f"the untimed preparation drained no items from the release: dataset {name!r}, path "
            f"{source}. Every rate this dataset reports divides by the count this pass establishes, "
            f"and a run that measured zero items would divide by zero or report a grid of empty "
            f"cells. A root that holds no scored recording is refused rather than measured"
        )

    return Preparation(frame=pd.DataFrame(rows), item_count=len(rows), channel_count=channel_count)


def _reach(source: Path) -> tuple[SleepEDFDataset, SleepStaging]:
    """Construct the reference loader and the task, the one way this connector does.

    Both the untimed preparation and every timed open reach the library through this, so the two
    reach the same corpus with the same task arguments and the gate's arithmetic answers for both.
    Constructing the loader is what derives the metadata table into the source directory when it is
    not already there.

    The task keeps its default window duration. That argument decides what one item is, so changing
    it changes the unit every rate on this dataset's rows is quoted against.

    Args:
        source: The root of the release.

    Returns:
        The constructed loader and the task it is asked for.
    """
    return SleepEDFDataset(root=str(source)), SleepStaging()
