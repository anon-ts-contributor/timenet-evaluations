"""What a run produced: three groups, and the rule each one carries.

A run makes three kinds of claim, so the result holds three groups rather than one flat list of
fields. ``metadata`` says who ran, when, where, and under what protocol. ``measurements`` holds
what a clock or a filesystem observed. ``metrics`` holds what arithmetic over those observations
produced. A field name is never the only thing that tells an observed number from a derived one:
the group is.

A well-formed result never describes a run that did not happen. That is why this module validates
at construction rather than at write time. There is no status field, no partial mode, and no way to
say that a run attempted sixteen cells and delivered twelve. A result that exists is a result that
is complete, and the checks a type cannot hold are held by the guards the later stories add before
the record reaches the disk.

The keys come from the grid and are never restated here. A timed entry is keyed by a ``Cell``,
which names the dataset, the representation, the reader and the task. A storage entry is keyed by a
``StorageKey``, which names the dataset and the representation and carries no reader coordinate. A
key of lower arity cannot name a cell of this grid, so the arity is the check.

The timed entry is ``harness.repeat.Measurement`` itself, and this module does not wrap it. That
model already carries the samples, the warm-up setting and the drop records, and it already computes
the repetition count and the reported figure from the samples. A second model that renamed those
fields would be a second place the protocol could drift.

This is a breaking change to ``result.json``. There is no migration and there is no version field. A
record written against the earlier flat shape stops matching the model that reads it, and stays
readable only as the file it is.

This module names no dataset and no dataset library. The statement of what a representation's cells
really read arrives from the loader that prepared it, and travels through here as text.
"""

from __future__ import annotations

from datetime import datetime
from importlib.metadata import distributions
import os
from pathlib import Path
import platform
import sys

from pydantic import BaseModel, ConfigDict, model_validator

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.cell import Cell, StorageKey
from timenet_evaluations.harness.repeat import Measurement


class Environment(BaseModel):
    """What produced a result.

    Timings from two machines are not comparable, so a result without its environment cannot be
    read later.
    """

    python: str
    """Interpreter version."""
    platform: str
    """Operating system and machine, for example ``darwin-arm64``."""
    cpu_count: int
    """Logical processors visible to the run."""
    packages: dict[str, str]
    """Installed distributions and their versions."""


class CacheProtocol(BaseModel):
    """The protocol every timing in this result was taken under.

    A drop that reports success and evicts nothing gives a figure several times faster than a cold
    one, and the two are identical in shape. Nothing in the number tells them apart. This model is
    the evidence beside the numbers, so a reader of a stored record can do a check instead of
    trusting the run.

    Everything here is what was attempted and observed. A command named in a docstring and never
    issued has no place in it, and an exit status is never zero because a run assumed success.
    """

    model_config = ConfigDict(frozen=True)

    warmup: bool
    """Whether each repetition read once, untimed, after the drop. Off gives a cold figure, on gives
    a warm one, and the two differ by several times. Every measurement carries this setting as well,
    and ``EvaluationResult`` refuses a result where the two disagree."""
    drop_command: str
    """The cache-drop command that was issued, verbatim, as one shell-quoted line."""
    drop_exit_status: int
    """The status that command's process returned."""
    drop_platform: str
    """The ``sys.platform`` value the command ran on. The platform decides which drop command is
    available at all, so a record without it cannot be re-judged on another machine."""
    calibration_ratio: float
    """The cold duration divided by the warm duration, as the probe measured it before the grid
    began. A ratio near one means the cache was never dropped and every figure here is warm."""
    calibration_threshold: float
    """The ratio the run demanded of the probe. It is stored beside the measured ratio because a
    ratio without the bar it had to clear cannot be re-judged, and the bar is a choice the run was
    given rather than a constant of the harness."""


class RepresentationRecord(BaseModel):
    """What ``metadata`` records about one (dataset, representation) pair.

    Two facts belong to exactly this pair and to no smaller or larger scope. The first is the block
    plan the shuffled task walked, recorded as the two inputs the run chose. The second is the
    loader's statement of what that representation's cells really read.

    The key carries no reader coordinate, and that is what makes both readers of one representation
    recorded as walking one plan. Two shuffled rates in one row would otherwise be rates over
    different workloads, and the shape refuses that instead of a check finding it later.
    """

    model_config = ConfigDict(frozen=True)

    at: StorageKey
    """The dataset and the representation these facts belong to."""
    block_bytes: int
    """The byte budget one block of the shuffled plan was given. It is identical for both
    representations of a dataset and for every dataset of the run."""
    seed: int
    """The run's seed, which fixed the order of the blocks and nothing else."""
    disclosure: str | None
    """What this representation's timed cells really read, in the words of the loader that prepared
    it.

    A representation names where the values came from. It does not always name what a timed cell
    read off the disk: an untimed preparation can materialize a cache, an index or a derived corpus,
    and every later cell then reads that instead of the release's own containers. The figure is
    still the read path an adopter pays, so nothing discounts it. What this statement fixes is that
    the representation's name does not say so.

    The wording arrives from the loader and is never composed here, inferred from a path, or held as
    a constant. This capability may not name a dataset or a dataset library, so the loader is the
    only component that knows what it prepared.

    A representation whose cells read the files the run was handed carries ``None``. The presence of
    a statement is the signal, so an absent one is never rendered as an empty note."""


class DatasetRecord(BaseModel):
    """What ``metadata`` records about one dataset the run read.

    The datasets are the one thing that legitimately varies between runs, and every count in the
    result is per dataset. A reader who holds this record knows what the expected measurement set
    was without consulting a declaration, because the expected set is a constant times the number of
    these records.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str
    """The dataset, as the run was given it."""
    source: Path
    """The directory the release files were read from. ``Original`` is not written by this harness
    and is not copied into the run directory, so the record names where it was measured."""
    unit: str
    """The name of one canonical item, as this dataset's own task names it. Every rate for this
    dataset counts these, so a stored rate is a number about a stated thing. The name means
    something different for each dataset, which is why rates do not compare across them."""
    item_count: int
    """How many canonical items this dataset declared before the run. It is the denominator of every
    rate in this dataset's cells, and it is never a count that a representation reported about
    itself."""
    representations: tuple[RepresentationRecord, ...]
    """The block plan inputs and the disclosure of each representation, in the order the registry
    declares them."""


class Metadata(BaseModel):
    """The circumstances a run's observations were taken under.

    This group holds identity, instant, environment, protocol, canonical item, block plan and
    disclosure. Every one of them describes the run or the conditions it was taken under, which is
    what this group is for.

    It holds no requested scope and no subject selection. Every run measures the whole grid, so such
    a field could hold only one value, and a field that can hold only one value is an invitation to
    make it hold two. Completeness is therefore read off the shape: a result is complete when it
    holds 16 timed entries and 2 storage entries per dataset named here.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    """The identity of this run. It sits here as well as on the directory path, so a record moved
    out of its directory still names its run."""
    started_at: datetime
    """When the run started, with its time zone. Ordering two runs by time means reading this field
    from inside each record, because a directory listing carries no chronology."""
    datasets: tuple[DatasetRecord, ...]
    """The datasets this run read, in the order it read them."""
    environment: Environment
    """The machine the timings are facts about."""
    protocol: CacheProtocol
    """The cache protocol every timing here was produced under."""


class StorageMeasurement(BaseModel):
    """The bytes one representation of one dataset occupies on disk.

    The harness counted these bytes from a path after the conversion, so this is an observation and
    not a number a representation set on an artifact while writing it. A size that cannot be
    determined raises where it is measured and is never recorded as zero.

    The key names two coordinates and never a reader. Nothing a reader does changes bytes on disk,
    and a third storage figure for a dataset would assert that it does.
    """

    model_config = ConfigDict(frozen=True)

    at: StorageKey
    """The dataset and the representation these bytes are."""
    size_bytes: int
    """The size on disk, as ``harness.size.measure_size`` counted it."""


class Measurements(BaseModel):
    """Everything a clock or a filesystem produced, and nothing else.

    The guarantee of this group is that every number in it was observed. No rate is stored here,
    even where it would be convenient, because a rate is arithmetic over a duration and a declared
    count.

    Both collections are tuples of entries that each carry their own key, so an entry is addressed
    by its key and a metric can name the entry it came from. A tuple is what keeps the entries in a
    declared order in the record, and it is also what keeps a duplicated cell visible: a mapping
    would let one cell be measured twice and another not at all while comparing equal.
    """

    model_config = ConfigDict(frozen=True)

    timed: tuple[Measurement, ...]
    """One entry per grid cell, keyed by all four coordinates. There are 16 per dataset. Each entry
    carries every recorded sample, the state it was taken in, and every drop it needed; the reported
    figure and the repetition count are computed from the samples."""
    storage: tuple[StorageMeasurement, ...]
    """One entry per representation, keyed by two coordinates. There are 2 per dataset."""


class Rate(BaseModel):
    """One derived number: how many canonical items per second one rate-task cell read.

    Two of the four tasks produce a rate. The sequential walk and the block-shuffled walk are rate
    tasks. The first-item read and the full read are durations in seconds and are divided by
    nothing.

    The entry names its inputs, because a number a reader cannot trace is a claim they can only
    trust. ``at`` is the timed entry this rate was computed from, and ``at.dataset`` is the dataset
    whose declared count it divided by. Both are in the same result, so a reader holding only the
    file can do the division again and get this number back.

    Nothing here names an origin or a baseline. The two representations are peers, and the
    comparison between them belongs to whoever reads the table.
    """

    model_config = ConfigDict(frozen=True)

    at: Cell
    """The timed entry this rate was computed from, and the dataset whose declared item count was
    the numerator."""
    items_per_second: float
    """The dataset's declared item count divided by the seconds the cell reported. It is computed
    once from the reduced figure, never per sample and reduced afterwards, so the printed rate
    corresponds to the printed duration of the same cell."""


class Metrics(BaseModel):
    """What arithmetic over the observations produced.

    This is the one group that may hold a number nobody observed, and it holds one kind of entry. It
    is not a name-to-number map: a bare number cannot say where it came from, so recomputability
    would be a convention rather than a property of the shape, and map keys are not
    declaration-ordered, so two runs would stop diffing to only what changed.
    """

    model_config = ConfigDict(frozen=True)

    rates: tuple[Rate, ...]
    """One rate per rate-task cell, each naming the timed entry it came from."""


class EvaluationResult(BaseModel):
    """Everything one run produced, in three groups.

    This object is the deliverable. Every field a later reader needs is on it, rather than implied
    by the file it is written in or the directory it sits under.

    Construction is where the shape is settled. A missing field, or a field of the wrong type in any
    of the three groups, raises here with the run still on the stack and before anything has been
    written. Each group is a model and validates as one, so three levels of nesting do not weaken
    the floor.

    That floor covers field types and not cross-field agreement. Three invariants are checked by the
    guards that stand between this object and the disk: that ``measurements`` holds every slot of
    the grid, that the (representation, reader) pairs present form the full product, and that every
    metric recomputes from the result beside it. One more is checked here, because it compares two
    fields of this object alone.
    """

    model_config = ConfigDict(frozen=True)

    metadata: Metadata
    """Circumstance: who ran, when, where, and under what protocol."""
    measurements: Measurements
    """Observation: what a clock or a filesystem produced."""
    metrics: Metrics
    """Arithmetic: what was derived from those observations, each entry naming its inputs."""

    @model_validator(mode="after")
    def _reconcile_warmup(self) -> EvaluationResult:
        """Refuse a result whose measurements disagree with ``metadata`` about the warm-up setting.

        The warm-up setting is written twice. ``metadata`` carries the run's claim, and every timed
        entry carries what its own repetitions did. Two entries that can disagree are worse than
        one, and this check is what makes carrying both safe: the per-cell records are the evidence,
        the run-level field is the claim, and a claim no evidence supports must not be written.

        Returns:
            This result, when every timed entry agrees with the run-level setting.

        Raises:
            EvaluationError: If one or more timed entries carry a different warm-up setting from the
                one ``metadata`` claims. The message names the claim and the cells that contradict
                it.
        """
        claimed = self.metadata.protocol.warmup
        disagreeing = [entry.at for entry in self.measurements.timed if entry.warmup != claimed]

        if disagreeing:
            cells = [
                f"({cell.dataset!r}, {cell.representation!r}, {cell.reader!r}, {cell.task.value!r})"
                for cell in disagreeing
            ]
            raise EvaluationError(
                "the measurements disagree with the metadata about the warm-up setting, so the run "
                f"claims a protocol its own evidence does not support: metadata warmup {claimed}, "
                f"{len(disagreeing)} of {len(self.measurements.timed)} timed entries carry "
                f"{not claimed}, at {', '.join(cells)}"
            )

        return self


def installed_packages() -> dict[str, str]:
    """Read every installed distribution and its version, from this interpreter's metadata.

    Collecting it here keeps the record independent of the thing being measured: a result must
    still describe its environment when the subject under test is not installed at all, which is
    the case in continuous integration.

    Returns:
        Distribution name to version, sorted by name.
    """
    packages: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if name:
            packages[name] = dist.version

    return dict(sorted(packages.items()))


def collect_environment() -> Environment:
    """Record the interpreter, the machine, and every installed package.

    An empty package list is treated as a failure rather than written out. A result that
    satisfies its own schema while carrying no provenance is worse than no result, because
    nothing downstream can tell it apart from a good one.

    Call this before any dataset is loaded. The failure it can raise has nothing to do with the
    sources or the representations, so it costs seconds at the top of a run and a whole grid of cold
    reads at the end of one.

    Returns:
        The environment this run is executing in.

    Raises:
        EvaluationError: If no installed distribution reports a name.
    """
    packages = installed_packages()

    if not packages:
        raise EvaluationError(
            "no installed distribution reported a name, so the result would carry no provenance; "
            f"importlib.metadata found {len(list(distributions()))} distributions"
        )

    return Environment(
        python=sys.version.split()[0],
        platform=f"{sys.platform}-{platform.machine()}",
        cpu_count=os.cpu_count() or 0,
        packages=packages,
    )
