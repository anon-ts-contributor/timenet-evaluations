"""The untimed conversion that produces the ``timef`` artifact.

Conversion runs once per dataset, before any timer starts. It is not timed, it has no task and no
column in any result, and it is never called from inside a timed region. A cell whose timer covered
the production of the thing it reads would not be the column it is printed under.

It exists here for exactly one reason: the converted representation has to be on disk before it can
be read and before its size can be measured. It never produces the release's own files, which this
harness does not write at all.

The artifact outlives the run that wrote it. ``out`` is an artifact root a run is given, not the
run's own directory, and a run that finds a version already there for its source, its pinned library
revision and the state of the modules in this directory reuses it and converts nothing. Converting a
corpus per run is the largest fixed cost a run has, and it buys nothing while none of those three
has changed. The ``identity`` module beside this one holds the address that answers this question,
and an edit to any module here moves that address.

A reused artifact is trustworthy only because something reads it back. The verification module
beside this one does that, before the first timed cell and outside every timer. A truncated file
reads faster than an intact one, so a damaged artifact does not look damaged in a report of read
times. It looks like the subject under test winning.

The cost this design does not price must be stated wherever the table appears. Adopting the
converted form costs a conversion, the table has no column for it, and a reader who infers from the
table that adoption is free draws a conclusion the experiment does not support.

``timenet`` is the ``timef`` extra and not a dependency, so it is absent from the environment
continuous integration builds. Each import below carries a suppression for that reason. Import this
module only when ``find_spec("timenet")`` has already found the package; the two modules beside it
hold the representations themselves and need nothing from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from timenet.dataset import TimeFDataset, TimeSeries  # ty: ignore[unresolved-import]
from timenet.dataset.axis import RegularAxis  # ty: ignore[unresolved-import]
from timenet.registry import LocalRegistry  # ty: ignore[unresolved-import]
from timenet.types import (  # ty: ignore[unresolved-import]
    ClassificationTask,
    DatasetMetadata,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.discovery import connector_dir  # ty: ignore[unresolved-import]

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.artifact import Artifact
from timenet_evaluations.grid.cell import StorageKey
from timenet_evaluations.grid.facts import DatasetFacts
from timenet_evaluations.grid.failure import failure_message
from timenet_evaluations.grid.representations.identity import (
    ArtifactKey,
    library_revision,
    source_digest,
    writer_digest,
)
from timenet_evaluations.grid.representations.timef import TimeF, sample_id
from timenet_evaluations.source import LABEL, PATIENT, SIGNAL


CARD_NAME = "dataset.yaml"
"""The file a connector's card is written in. `connector_dir` guarantees one is beside it."""

VALUES_BACKEND = "parquet"
"""The converted form is stored solely as Parquet. Nothing else on disk counts as it, and no second
container is written beside it and measured as though it were part of it."""

SIGNAL_DIMENSIONS = 2
"""How many dimensions one row's values have: one channel axis and one time axis. The layout gives
every channel its own series, so it can represent nothing else."""


def write(source: pd.DataFrame, out: Path, *, dataset: str, facts: DatasetFacts) -> Artifact:
    """Give the converted representation of one dataset, converting it only where none is there.

    ``out`` is an artifact root and not one run's directory. The address of an artifact is the
    library revision, the state of the conversion's own modules, and the digest of the content, and
    the conversion runs only where that address holds no committed version. A run that finds one
    there returns it and writes nothing, so a corpus that did not change is converted once and not
    once per run. A run whose code has changed converts again, because a reused artifact this code
    would not have written is a result that says nothing about this code.

    A conversion writes one sample per canonical item, one series per channel, and the scored class
    as a classification task on that sample. It creates ``out`` and any missing parent, and it
    writes into a directory that is already there rather than refusing one.

    The version is fixed by ``facts`` rather than generated, so a conversion at one address always
    lands in the same version directory instead of accumulating a version per run.

    The returned path is the one the storage library resolved, never one composed from constants.
    A composed path is a guess about another library's on-disk layout, and a stale guess is how a
    conversion produces an artifact nothing can read back.

    A returned artifact is not yet known to be intact. The caller must verify it, whether this call
    wrote it or found it, before the first timed cell.

    Args:
        source: The frame the connector loaded for conversion, one row per canonical item.
        out: The artifact root the converted version is written beneath and found beneath.
        dataset: The name of the dataset being converted. It names the failure in every message
            this conversion raises, because a run converts more than one dataset.
        facts: Everything about this dataset that the frame does not carry.

    Returns:
        The artifact naming the converted representation and the path the storage library
        resolved, which is the path the size measurement is given and the path both readers of
        this representation are opened against.

    Raises:
        EvaluationError: If the frame holds nothing, if a signal has a shape the layout cannot
            represent, if the artifact already at the address cannot be read, if the output
            directory cannot be created, or if the storage library refuses the dataset or cannot
            say where it wrote it.
    """
    at = StorageKey(dataset=dataset, representation=TimeF.name)
    # The identity is read once, here, and handed down. Every place below that needs an id or a
    # version takes it from this value rather than from `facts`, which no longer carries either.
    card = card_for(facts, at=at)
    signals = _validated_signals(source, out, dataset=dataset, facts=facts)
    key = ArtifactKey(
        revision=library_revision(at=at, path=out),
        writer=writer_digest(at=at, path=out),
        source=source_digest(
            signals,
            labels=_written_text(source, LABEL),
            patients=_written_text(source, PATIENT),
            facts=facts,
        ),
    )
    home = key.root(out)

    already = _committed(home, at=at, card=card)
    if already is not None:
        return Artifact(representation=TimeF.name, path=already)

    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise EvaluationError(
            failure_message(
                "the output directory cannot be created",
                at=at,
                path=home,
                detail=f"{error}",
            )
        ) from error

    try:
        version = _convert(source, home, signals=signals, facts=facts, card=card)
    except Exception as error:
        raise EvaluationError(
            failure_message(
                "the conversion failed",
                at=at,
                path=home,
                detail=f"dataset id {card.dataset_id!r}: {error}",
            )
        ) from error

    return Artifact(representation=TimeF.name, path=version)


def _committed(home: Path, *, at: StorageKey, card: DatasetMetadata) -> Path | None:
    """Find the version already at one address, where one is there.

    A version that is there is reused. A directory that holds something the storage library cannot
    read is not treated as nothing: a new conversion would erase it, and an artifact that exists
    and cannot be decoded is what this harness refuses rather than replaces.

    Args:
        home: The registry root the address names.
        at: The coordinates of the conversion that asked, which name the failure.
        card: This dataset's identity, as its card states it. Its id and version are the
            address the storage library looks under.

    Returns:
        The version directory, as the storage library resolved it, or ``None`` where the address
        holds no committed version.

    Raises:
        EvaluationError: If the address holds something the storage library cannot read.
    """
    registry = LocalRegistry(home)
    try:
        stored = str(card.dataset_version)
        if not registry.exists(card.dataset_id, stored):
            return None

        return Path(registry.open_version(card.dataset_id, stored).root)
    except Exception as error:
        raise EvaluationError(
            failure_message(
                "the converted form already at this address cannot be read",
                at=at,
                path=home,
                detail=f"dataset id {card.dataset_id!r} version {card.dataset_version!s}: {error}",
            )
        ) from error


def _written_text(source: pd.DataFrame, column: str) -> tuple[str, ...]:
    """Reduce one column of the frame to the text the conversion writes from it.

    The digest that addresses an artifact and the conversion that writes it read the same values
    through this one call. Two calls that spelled the reduction twice could drift, and the digest
    would then stop covering part of what is written.

    Args:
        source: The frame the connector loaded for conversion.
        column: The column to reduce.

    Returns:
        One string per row, in the frame's order.
    """
    return tuple(str(value) for value in source[column].to_numpy())


def _validated_signals(source: pd.DataFrame, out: Path, *, dataset: str, facts: DatasetFacts) -> list[np.ndarray]:
    """Reduce every row's values to an array, and refuse a frame the layout cannot represent.

    This runs before anything is written, so a frame the reads would mis-shape stops the run
    rather than leaving a converted form behind that reads back wrong.

    Args:
        source: The frame the connector loaded for conversion.
        out: The artifact root the converted version would have been written beneath. It names the
            failure.
        dataset: The name of the dataset being converted.
        facts: Everything about this dataset that the frame does not carry.

    Returns:
        One array per row, each shaped ``(n_channels, n_samples)``.

    Raises:
        EvaluationError: If the frame holds no rows, if a row's values do not reduce to an array,
            or if a row holds a number of channels the dataset did not declare.
    """
    if source.empty:
        raise EvaluationError(
            failure_message(
                "the frame to convert holds no rows",
                at=StorageKey(dataset=dataset, representation=TimeF.name),
                path=out,
            )
        )

    signals: list[np.ndarray] = []
    for position, values in enumerate(source[SIGNAL]):
        try:
            signal = np.asarray(values, dtype=np.float32)
        except Exception as error:
            raise EvaluationError(
                failure_message(
                    "a row's values do not reduce to an array of numbers",
                    at=StorageKey(dataset=dataset, representation=TimeF.name),
                    path=out,
                    detail=f"position {position}: {error}",
                )
            ) from error

        if signal.ndim != SIGNAL_DIMENSIONS or signal.shape[0] != len(facts.channels):
            raise EvaluationError(
                failure_message(
                    "a row holds a shape the layout cannot represent",
                    at=StorageKey(dataset=dataset, representation=TimeF.name),
                    path=out,
                    detail=(f"position {position}: declared {len(facts.channels)} channels, read shape {signal.shape}"),
                )
            )

        signals.append(signal)

    return signals


def _convert(
    source: pd.DataFrame,
    home: Path,
    *,
    signals: list[np.ndarray],
    facts: DatasetFacts,
    card: DatasetMetadata,
) -> Path:
    """Build the dataset the storage library writes, store it, and ask where it landed.

    Every call this makes belongs to the storage library, so the caller translates whatever any of
    them raises. Nothing is caught here.

    Args:
        source: The frame the connector loaded for conversion.
        home: The registry root the address names, which the converted version is written beneath.
        signals: One validated array per row, in the frame's order.
        facts: Everything about this dataset that neither the frame nor the card carries.
        card: This dataset's identity, read from the card the connectors distribution ships.

    Returns:
        The version directory, as the storage library resolved it.
    """
    spec = TimeSeriesSpec(spec_type=facts.modality, name=facts.modality_name, unit_value=ureg.Unit(facts.unit))
    axis = RegularAxis.from_rate_hz(facts.rate_hz)
    built = TimeFDataset(metadata=card)

    labels = _written_text(source, LABEL)
    patients = _written_text(source, PATIENT)
    for position, signal in enumerate(signals):
        series = tuple(
            TimeSeries.from_values(signal[index], spec=spec, channel=channel, time_axis=axis)
            for index, channel in enumerate(facts.channels)
        )
        sample = built.add_sample(
            time_series=series,
            subject_ids=(patients[position],),
            sample_id=sample_id(position),
        )
        built.add_task(
            sample,
            ClassificationTask(target=labels[position], target_schema=facts.target_schema),
        )

    built.derive_schema()
    registry = LocalRegistry(home)
    # A conversion runs only where the address holds no committed version, so `force` is not what
    # answers a rebuild any more. What it answers is a directory left by a write that did not
    # commit: without it the storage library would write beside that leftover rather than over it.
    stored = registry.store(built, force=True, values_backend=VALUES_BACKEND)

    # The path comes from the library, which resolves it from the manifest it just committed. It is
    # not composed here, so a release of the library that moves its layout stops the run at this
    # call rather than handing back a path that is not there.
    return Path(registry.open_version(card.dataset_id, stored).root)


def card_for(facts: DatasetFacts, *, at: StorageKey) -> DatasetMetadata:
    """Read one dataset's identity from the card the connectors distribution ships for it.

    The card is the single source for a dataset's id, version, display name, description and
    licence, and this reads it rather than restating it. ADR-0028 is why: those five facts had a
    copy in this repository, the copy drifted, and nothing compared the two.

    Neither the lookup nor the parsing is done here. ``discovery.connector_dir`` finds the
    connector's directory and is import-free by design, so the card is reachable without the
    connector's own download requirements, which belong to fetching and parsing a release rather than
    to describing one. ``DatasetMetadata.from_yaml`` validates the file. What this function adds
    is the harness's error type and the coordinates of the cell that asked, which the storage
    library has no way to know.

    Args:
        facts: This dataset's facts, whose ``card_id`` names the card to read.
        at: The dataset and representation being converted, for the message.

    Returns:
        The identity the converted version records, as the storage library validated it.

    Raises:
        EvaluationError: If the distribution ships no connector for that id, or if its card cannot
            be read.
    """
    try:
        directory = Path(connector_dir(facts.card_id))
    except LookupError as error:
        raise EvaluationError(
            failure_message(
                "the connectors distribution ships no dataset card for this id",
                at=at,
                path=Path(CARD_NAME),
                detail=(
                    f"card id {facts.card_id!r}: {error}. Either the id is wrong or the pinned "
                    f"revision predates that connector"
                ),
            )
        ) from error

    card = directory / CARD_NAME
    try:
        return DatasetMetadata.from_yaml(str(card))
    except Exception as error:
        raise EvaluationError(
            failure_message(
                "the dataset card for this id cannot be read",
                at=at,
                path=card,
                detail=f"card id {facts.card_id!r}: {error}",
            )
        ) from error
