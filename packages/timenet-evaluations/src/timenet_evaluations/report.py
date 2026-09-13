"""The record of a run, the directory it owns, and the refusals that come before the write.

A run produces one object. ``write_json`` turns it into the file that stays. The record keeps every
field of all three groups, in the order the fields are declared at every level of nesting, so two
runs of one source diff to only what changed and a reader a month later has the whole file. Nothing
is dropped for readability, because the thing a summary would drop is the provenance.

The other output of a run is the glance at the end of it, and :mod:`timenet_evaluations.summary`
writes that. The two live in two modules rather than one because they are two outputs with two jobs
and two lifetimes, and one module holding both passes the line cap this repository keeps.

``check_one_protocol`` has no caller yet, and that is deliberate. A warm figure and a cold figure
are both well formed and differ by several times, so a report that combined them would give back
what the cold protocol buys. The rule is written before the first script that combines records,
rather than after it.

Check, and then write
---------------------
Every refusal happens inside ``write_json``, before the first byte reaches the disk. The code this
one replaces wrote the record and checked it afterwards, in the printing path, so a result the run
itself rejected was already on the disk when the refusal arrived, and a caller that wrote without
printing was never checked at all. The ordering was not incidental to that failure; it was the
failure.

The guards are the three cross-field invariants that no type can hold: the grid must be whole, each
slot must be measured once, and every metric must recompute from the measurements beside it. They
sit inside the writer rather than beside its call site, so no caller can reach the disk past them.

Each run owns a directory named by an identity generated for it, and the record goes in beside the
artifacts whose sizes and timings it reports. The identity is also a field of ``metadata``, so a
record moved out of its directory still names its run. It is generated with no lock, no counter and
no shared sequence, so two runs cannot collide by racing. Forty-eight bits is ample for hand-driven
runs and it is not a guarantee: the directory is created with ``exist_ok=True``, which is what makes
creation repeatable within one run and, in the same stroke, what makes a collision silent. Nothing
detects one. That is recorded here rather than fixed, because a lock would buy detection at the cost
of the very coordination the identity exists to avoid.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final
from uuid import uuid4

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.assembly import check_grid_complete
from timenet_evaluations.metrics import check_metrics
from timenet_evaluations.result import EvaluationResult, Measurements


RECORD_NAME: Final = "result.json"
"""What the record is called inside a run's own directory.

The name is fixed and carries no run identity, because the directory already does. A reader who
holds a directory knows the file, and a reader who holds the file finds the run identity inside it.
"""

RUN_ID_LENGTH: Final = 12
"""How many hexadecimal characters of a random UUID name a run: forty-eight bits.

Long enough that hand-driven runs do not collide, and short enough to type and to read in a
directory listing. It is not a guarantee, and nothing here detects a collision.
"""


def new_run_id() -> str:
    """Name one run, without asking anything else for permission.

    No lock, no counter, and no shared sequence: two runs cannot collide by racing, on this machine
    or on another, because neither run consults anything the other could hold.

    Returns:
        Twelve hexadecimal characters of a fresh random UUID.
    """
    return uuid4().hex[:RUN_ID_LENGTH]


def run_directory(out_dir: Path, *, run_id: str) -> Path:
    """Create the directory one run owns, beneath the output root it was given.

    The unit of output is not the record alone. It is the record together with the artifacts whose
    sizes and timings that record reports, so both go in here and a figure can always be checked
    against the file it came from.

    Calling this twice within one run succeeds. The artifacts and the record are written at
    different points of a run, and each one asks for the directory when it needs it.

    This is the one place the path of a run directory is built. A caller that derived it a second
    time would agree with this one because both derive from the identity, and not because either was
    told the other's answer.

    Args:
        out_dir: The output root every run of this machine writes beneath.
        run_id: The identity of this run, which is also a field of its ``metadata``.

    Returns:
        The directory, which now exists.

    Raises:
        EvaluationError: If the directory cannot be created. The message names the run and the path,
            because a run that cannot write has to say where it tried.
    """
    path = out_dir / run_id

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as failure:
        raise EvaluationError(
            f"a run writes its record and its artifacts into a directory of its own, and this one could not be "
            f"created, so the run stops with nothing on disk: run {run_id!r}, path {path}, {failure}"
        ) from failure

    return path


def write_json(result: EvaluationResult, run_dir: Path) -> Path:
    """Check the result whole, and only then write the record of one run.

    Every refusal comes first. The grid must hold every cell of every dataset the metadata names,
    each cell must be measured exactly once, and every metric must recompute from the measurements
    beside it. A result that fails any of the three stops the run here, with nothing on the disk and
    nothing on standard output, because a record that a run itself rejected must never exist.

    The guards are inside this function rather than beside its call site. A check that lives only in
    the printing path protects no caller that writes without printing, which is how the code this
    replaces came to serialize an unchecked result.

    The record keeps every field of all three groups. Nothing is dropped for readability: the
    provenance, every recorded sample, the cache protocol, the canonical item of each dataset, the
    block plan inputs and the measurement key each metric names are all in the file, because the
    reader this file is for is not in the room.

    The field order follows the field declarations at every level, and every collection is a tuple,
    so two runs of one source produce records that differ in content and not in shape.

    Args:
        result: What the run produced.
        run_dir: The directory this run owns, as :func:`run_directory` built it.

    Returns:
        The path the record was written to.

    Raises:
        EvaluationError: If the result does not hold the whole grid, if one cell is measured more
            than once, if a metric disagrees with its measurements, or if the write itself fails.
            The message of a failed write names the run and the path.
    """
    check_grid_complete(
        [record.dataset for record in result.metadata.datasets],
        [entry.at for entry in result.measurements.timed],
        [entry.at for entry in result.measurements.storage],
    )
    check_each_slot_measured_once(result.measurements)

    path = run_dir / RECORD_NAME
    check_metrics(result, path=path)

    try:
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except OSError as failure:
        raise EvaluationError(
            f"the record of a run is the run's one durable output, and writing it failed, so the run stops rather "
            f"than reporting success with nothing on disk: run {result.metadata.run_id!r}, path {path}, {failure}"
        ) from failure

    return path


def check_each_slot_measured_once(measurements: Measurements) -> None:
    """Refuse a result that measured one slot twice.

    This is the half of completeness that the product check cannot do. That check turns its
    arguments into sets on its first line, so it compares membership and never multiplicity. The
    case that walks past it is a seventeenth entry duplicating a cell while nothing is missing: the
    product is whole, so the difference is empty, and the result then holds two figures for one cell
    and gives a metric two entries to name.

    A count that matches while the product does not is caught by the product check instead. Both
    checks are needed and neither one implies the other.

    Args:
        measurements: What a clock and a filesystem produced for this run.

    Raises:
        EvaluationError: If any timed cell or any storage figure is named by more than one entry.
            The message names the slot and how many entries name it.
    """
    timed = Counter(entry.at for entry in measurements.timed)
    repeated = sorted(
        (
            f"({at.dataset}, {at.representation}, {at.reader}, {at.task.value}) x {count}"
            for at, count in timed.items()
            if count > 1
        )
    )

    if repeated:
        raise EvaluationError(
            f"a run measures each cell of the grid once, and a cell measured twice gives a metric two entries to "
            f"name: {len(measurements.timed)} timed entries over {len(timed)} cells, measured more than once "
            f"{repeated}"
        )

    stored = Counter(entry.at for entry in measurements.storage)
    twice = sorted(f"({at.dataset}, {at.representation}) x {count}" for at, count in stored.items() if count > 1)

    if twice:
        raise EvaluationError(
            f"a dataset carries one storage figure per representation, and a second figure for one representation "
            f"prices the same bytes twice: {len(measurements.storage)} storage entries over {len(stored)} "
            f"representations, measured more than once {twice}"
        )


def check_one_protocol(results: Sequence[EvaluationResult]) -> None:
    """Refuse to combine results that were taken under different warm-up settings.

    A cold figure and a warm one are both well formed, both validate, and differ by several times.
    A report that put them side by side with a note would make that difference easy to overlook,
    which is the whole of what the cold protocol buys. So this refuses instead: it annotates
    nothing, it drops nothing, and it produces no table.

    Call this before a report, a table or a figure reads more than one result. One result alone is
    always one protocol.

    Args:
        results: The results a caller wants to combine.

    Raises:
        EvaluationError: If two of the results carry different warm-up settings. The message names
            the setting each result carried, so an operator can see which came from which protocol
            without opening them.
    """
    settings = {result.metadata.protocol.warmup for result in results}

    if len(settings) > 1:
        carried = ", ".join(
            f"run {result.metadata.run_id!r} warmup {result.metadata.protocol.warmup}" for result in results
        )
        raise EvaluationError(
            "a report never combines results taken under two warm-up settings, because a warm figure and a cold "
            f"one are both well formed and differ by several times: {len(results)} results carry {sorted(settings)}, "
            f"at {carried}"
        )
