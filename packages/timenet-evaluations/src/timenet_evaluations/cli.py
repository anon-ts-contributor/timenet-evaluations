"""The command line of a benchmark run, and the one function that performs the run.

The subject of a run is a grid: two representations, read by two readers, on four tasks, for each
dataset a run is given. One dataset produces 16 timed cells and 2 storage figures. The subjects are
written out by hand in ``timenet_evaluations.grid.registry``, and ``grid.assembly`` pairs every
reader with every representation.

``run_evaluation`` performs the whole sequence in one function, top to bottom. There is no phase
object, no dispatch table and no helper that holds a step, because five requirements of SPEC-0021
are properties of the order those steps happen in, and an order split across helpers is an order
nothing can assert. The function reads as the sequence reads: validate, refuse a machine that
cannot measure, take an identity and an instant, capture the environment, probe the drop, derive
the run directory, then per dataset load once, convert or reuse, verify, size, plan, check parity
and measure sixteen cells — and then derive once, construct the result, write the record, print the
summary and return the object.

Progress is written to standard error, never to standard output. SPEC-0019 makes standard
output the summary and nothing else, byte-identical in a terminal and in a pipe, so a line
saying where the run has got to would corrupt the one output a reader diffs. Standard error is
where a long-running command says what it is doing.

Nothing here is timed and nothing here catches. Every raise from a connector, a conversion, a
verification, a size, a plan, a parity check, a probe, a reader, a derivation, a model validator, a
record writer or a summary printer leaves this module untouched and ends the process with a
traceback. Whether the subject under test is importable is settled by ``find_spec``, which is a
question about the environment rather than a failure being swallowed.

**The converted artifact is not this run's.** It is written beneath the artifact root, where it
outlives the run and is reused by the next one, and the run directory holds the record of one run
and nothing else. Where a version already exists for a source under the pinned library revision,
step 9 reuses it and converts nothing, and step 10 hashes it against its own manifest before any
clock starts — a truncated shard reads faster than an intact one, so corruption here does not look
like damage. It looks like the subject under test winning.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from importlib.util import find_spec
from pathlib import Path
import sys
import time

from timenet_evaluations.arguments import DATASET_FORM, DatasetPair, check_datasets, parse_dataset
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid.assembly import check_artifact_agrees
from timenet_evaluations.grid.cell import StorageKey
from timenet_evaluations.grid.registry import (
    CELLS_PER_DATASET,
    READERS,
    REPRESENTATIONS,
    STORAGE_FIGURES_PER_DATASET,
    TASKS,
)
from timenet_evaluations.grid.representations import Original
from timenet_evaluations.harness.calibration import Probe, calibrate
from timenet_evaluations.harness.drop import PlatformDropCaches, platform_drop_command
from timenet_evaluations.harness.plan import BLOCK_BYTES
from timenet_evaluations.harness.repeat import REPEATS, Measurement
from timenet_evaluations.harness.tasks import RunParameters, Timing, measure_dataset
from timenet_evaluations.metrics import derive_metrics
from timenet_evaluations.report import RECORD_NAME, new_run_id, run_directory, write_json
from timenet_evaluations.result import (
    CacheProtocol,
    DatasetRecord,
    EvaluationResult,
    Measurements,
    Metadata,
    RepresentationRecord,
    StorageMeasurement,
    collect_environment,
)
from timenet_evaluations.source.contract import signal_stack
from timenet_evaluations.source.logging_stream import send_library_logging_to_stderr
from timenet_evaluations.source.parity import ParityReference, check_parity
from timenet_evaluations.source.registry import connector_for
from timenet_evaluations.summary import print_summary


TIMEF_INSTALLED = find_spec("timenet") is not None
"""Whether the subject under test can be imported.

``timenet`` is the ``timef`` extra, not a dependency, because it resolves from a private
repository and continuous integration must be able to check this package without one. Probing
with ``find_spec`` rather than catching an ``ImportError`` keeps the package's no-``except`` rule
intact: this is a question about the environment, not a failure being swallowed.
"""

if TIMEF_INSTALLED:
    # A conditional import at module level, not a deferred one inside a function. `conversion` and
    # `verification` import the storage library, which is the extra and is absent from continuous
    # integration, so an unconditional import here would stop this module loading there. Both names
    # are bound before `run_evaluation` can reach them: step 2 refuses a run without the extra, and
    # nothing below that refusal runs without it.
    from timenet_evaluations.grid.representations.conversion import write
    from timenet_evaluations.grid.representations.timef import TimeF
    from timenet_evaluations.grid.representations.timef_parsing import TimeFParsingPath
    from timenet_evaluations.grid.representations.verification import verify


READ_CHUNK_BYTES = 1024 * 1024
"""How much of the probe's file is read at a time. It bounds the memory one probe read costs, and
the size does not change what is measured: every byte is read either way."""


def read_every_byte(path: Path) -> None:
    """Read one file from start to end, discarding what it holds.

    The calibration probe times this against a dropped page cache and then against a warm one. It
    parses nothing and builds nothing, so what it measures is the filesystem and not a loader.

    Args:
        path: The file to read.
    """
    with path.open("rb") as handle:
        while handle.read(READ_CHUNK_BYTES):
            pass


CALIBRATION_THRESHOLD = 3.0
"""How many times faster the probe's second read must be than its first.

``harness.calibration`` takes this as a parameter and holds no default for it, because the number
decides whether every figure of the run is real and no measurement in this repository established
one. The run is the caller, so the run chooses it, and the result records the ratio beside this
threshold so that a later reader re-judges the run rather than trusts it.

Three stands for "several times faster on this machine", which is what a working drop produces and
what a drop that evicted nothing cannot. It is a judgement and not a measurement, and the argument
surface deliberately does not make it settable: a threshold a run could lower is a threshold a run
would lower.
"""

PROBE_CHUNK_BYTES = 1024 * 1024
"""How many bytes the probe reads at a time.

The probe reads a whole file, and a file of the size that makes a cold read distinguishable from a
warm one does not belong in memory at once.
"""


def run_evaluation(
    datasets: Sequence[DatasetPair],
    out_dir: Path,
    *,
    artifacts: Path = Path("artifacts"),
    repeats: int = REPEATS,
    warmup: bool = False,
    block_bytes: int = BLOCK_BYTES,
    seed: int = 0,
) -> EvaluationResult:
    """Perform one run: the whole grid, for every dataset the run was given, in one fixed order.

    The order is the specification and it is why this is one function. Everything a run can refuse
    from its command line alone is refused first, then everything that describes the machine, and
    only then is a dataset touched. Each dataset is loaded exactly once, and the object that load
    returned is the object the conversion is given.

    Nothing between the load and the last cell rebuilds what the load prepared, and nothing in the
    measurement pass converts: both representations of a dataset exist, are verified, sized,
    planned and parity-checked before that dataset's first clock starts.

    Args:
        datasets: One ``name=path`` pair for each dataset the run was given.
        out_dir: Directory the run directory is created beneath.
        artifacts: Directory the converted representations are written beneath. It is kept across
            runs, so it is created on demand and is never required to exist or to be empty.
        repeats: Recorded cold repetitions per cell and task.
        warmup: Whether each repetition reads once, untimed, after the cache drop. It is off, so a
            run reports a cold figure.
        block_bytes: The byte budget one shuffled block must fit in.
        seed: The seed that fixes the order a shuffled walk visits its blocks in.

    Returns:
        Everything the run produced, already written to JSON and printed as a grid.

    Raises:
        EvaluationError: If the ``timef`` extra is not installed, or if the dataset the probe
            reads holds no file at all. Every other failure of the run comes from a collaborator and
            passes through here untouched.
    """
    check_datasets(datasets)
    # After the checks, never before them: the four preconditions are the first thing this run does,
    # and a clock read or a message ahead of them would put something between argv and the refusal.
    started = time.monotonic()
    # The dataset library points its own logger at standard output when it is imported, and standard
    # output is this run's record. A library announcing its cache directory there would break the
    # one output a reader diffs, once per open, so the records go to standard error instead.
    send_library_logging_to_stderr()
    print(f"{len(datasets)} dataset(s) accepted", file=sys.stderr, flush=True)

    if not TIMEF_INSTALLED:
        readers = [reader.name for reader in READERS]
        raise EvaluationError(
            f"the timef extra is not installed, so the converted representation cannot be produced "
            f"and this run would measure less than the whole grid: representations "
            f"{list(REPRESENTATIONS)}, readers {readers}, tasks {[task.value for task in TASKS]}, "
            f"{CELLS_PER_DATASET} timed cells and {STORAGE_FIGURES_PER_DATASET} storage figures "
            f"per dataset. Install it with `uv sync --all-groups --all-extras`, which needs read "
            f"access to the timenet repository; see `make configure-source`. Refused for datasets "
            f"{[pair.text for pair in datasets]}"
        )

    print("calibrating the page cache drop", file=sys.stderr, flush=True)
    run_id = new_run_id()
    started_at = datetime.now(UTC)
    environment = collect_environment()

    drop_caches = PlatformDropCaches(platform_drop_command(sys.platform))
    probed = datasets[0]
    files = [path for path in sorted(probed.path.rglob("*")) if path.is_file()]
    if not files:
        raise EvaluationError(
            f"the calibration probe reads one of the release's own files and this dataset holds "
            f"none, so whether the page cache drop works cannot be established and every figure "
            f"the run would produce might be warm: dataset {probed.name!r}, path {probed.path}"
        )
    # The largest file is the one whose cold read and warm read differ by the most, and a probe
    # that cannot tell them apart is the probe this run depends on being able to.
    probe_path = max(files, key=lambda path: path.stat().st_size)
    calibration = calibrate(
        Probe(
            read=partial(read_every_byte, probe_path),
            at=StorageKey(dataset=probed.name, representation=Original.name),
            path=probe_path,
        ),
        drop_caches=drop_caches,
        threshold=CALIBRATION_THRESHOLD,
    )

    run_dir = run_directory(out_dir, run_id=run_id)
    print(f"run {run_id} -> {run_dir}", file=sys.stderr, flush=True)
    run = RunParameters(
        timing=Timing(drop_caches=drop_caches, repeats=repeats, warmup=warmup),
        seed=seed,
        block_bytes=block_bytes,
    )

    records: list[DatasetRecord] = []
    timed: list[Measurement] = []
    storage: list[StorageMeasurement] = []
    for position, pair in enumerate(datasets, start=1):
        print(f"[{position}/{len(datasets)}] {pair.name}: loading", file=sys.stderr, flush=True)
        connector = connector_for(pair.name)
        declaration = connector.canonical_item(pair.path)
        frame = connector.load_for_conversion(pair.path)
        original = Original(
            dataset=pair.name,
            source=pair.path,
            item=declaration.item,
            parsing_path=connector,
        )
        # The conversion writes beneath the artifact root and never beneath the run directory or
        # a source path, and it reuses what is already there for this source, this library revision
        # and this conversion code. The representation carries the path the library resolved, not
        # the root it was handed.
        print(
            f"[{position}/{len(datasets)}] {pair.name}: converting or reusing beneath {artifacts}",
            file=sys.stderr,
            flush=True,
        )
        artifact = write(frame, artifacts, dataset=pair.name, facts=declaration.facts)
        timef = TimeF(
            dataset=pair.name,
            artifact=artifact,
            item=declaration.item,
            parsing_path=TimeFParsingPath(dataset=pair.name, channels=declaration.facts.channels),
        )
        # Untimed, and before the first cell of this dataset. A reused artifact is checked as
        # closely as a fresh one: a truncated shard reads faster, so damage flatters the subject.
        print(f"[{position}/{len(datasets)}] {pair.name}: verifying the artifact", file=sys.stderr, flush=True)
        verify(timef.artifact, dataset=pair.name)

        representations = (original, timef)
        # Each representation is checked against its own artifact before it is sized, planned or
        # timed. The pairing is by identity: a representation built from another one's artifact
        # raises here rather than producing eight consistent figures about the wrong bytes.
        for representation in representations:
            check_artifact_agrees(representation, dataset=pair.name)
        check_parity(
            ParityReference(dataset=pair.name, declaration=declaration.item, reference=signal_stack(frame)),
            representations,
            READERS,
        )
        print(
            f"[{position}/{len(datasets)}] {pair.name}: measuring {CELLS_PER_DATASET} cells, "
            f"{repeats} cold repetitions each. This is the long one and it reports nothing until it ends",
            file=sys.stderr,
            flush=True,
        )
        grid = measure_dataset(pair.name, representations, run)

        timed.extend(grid.measurements)
        storage.extend(StorageMeasurement(at=one.at, size_bytes=one.size_bytes) for one in grid.prepared)
        records.append(
            DatasetRecord(
                dataset=pair.name,
                source=pair.path,
                unit=declaration.unit,
                item_count=declaration.item.count,
                representations=tuple(
                    RepresentationRecord(
                        at=StorageKey(dataset=pair.name, representation=representation.name),
                        block_bytes=block_bytes,
                        seed=seed,
                        disclosure=declaration.disclosures.get(representation.name),
                    )
                    for representation in representations
                ),
            )
        )

    print(
        f"deriving metrics and writing the record ({time.monotonic() - started:.1f}s so far)",
        file=sys.stderr,
        flush=True,
    )
    measurements = Measurements(timed=tuple(timed), storage=tuple(storage))
    metrics = derive_metrics(measurements, records, path=run_dir / RECORD_NAME)
    result = EvaluationResult(
        metadata=Metadata(
            run_id=run_id,
            started_at=started_at,
            datasets=tuple(records),
            environment=environment,
            protocol=CacheProtocol(
                warmup=warmup,
                drop_command=calibration.drop.command,
                drop_exit_status=calibration.drop.exit_status,
                drop_platform=calibration.drop.platform,
                calibration_ratio=calibration.ratio,
                calibration_threshold=calibration.threshold,
            ),
        ),
        measurements=measurements,
        metrics=metrics,
    )
    write_json(result, run_dir)
    print_summary(result)

    print(f"done in {time.monotonic() - started:.1f}s", file=sys.stderr, flush=True)

    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the parser that both entry points read their arguments with.

    No program name is given, so argparse takes one from ``sys.argv[0]``. The usage line therefore
    names the console script under one entry point and ``__main__.py`` under the other. That is the
    one permitted difference between the two, it is cosmetic, and nothing may be told apart by it.

    Returns:
        The parser, described by this module's docstring.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        type=parse_dataset,
        metavar=DATASET_FORM,
        help="one dataset per pair: the name that selects its connector, and the directory that "
        "holds its release files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results"),
        help="directory the run directory is created beneath, resolved against the working "
        "directory the run is started from",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts"),
        help="directory the converted representations are kept beneath, across runs, so that a run "
        "over unchanged inputs converts nothing",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help="recorded cold repetitions per timed cell and task, an odd count",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="read once after the cache drop and discard it, so the run reports a warm figure "
        "instead of the protocol's cold one",
    )
    parser.add_argument(
        "--block-bytes",
        type=int,
        default=BLOCK_BYTES,
        help="byte budget one shuffled block must fit in, from which each representation derives its own item count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed that fixes the order a shuffled walk visits the blocks of its plan in",
    )

    return parser


def main() -> None:
    """Run the comparison from the command line.

    Both entry points reach this function, so it is the whole of what either of them does. The
    arguments are read and then handed to :func:`run_evaluation`, which performs the run.

    The result is discarded rather than inspected, and no exit status is returned: a completed run
    exits zero by falling off the end. Nothing is caught here or below, so a failure anywhere ends
    the process with the interpreter's own traceback and a non-zero status.
    """
    args = build_parser().parse_args()

    run_evaluation(
        args.datasets,
        args.out,
        artifacts=args.artifacts,
        repeats=args.repeats,
        warmup=args.warmup,
        block_bytes=args.block_bytes,
        seed=args.seed,
    )
