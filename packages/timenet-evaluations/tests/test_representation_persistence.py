"""The persisted artifact: the address it is found at, and the check that it is still intact.

The failure these tests exist for is the one that flatters the subject. A truncated file reads
faster than an intact one, so a damaged artifact produces a fast number under the column where the
converted form is the subject under test, and nothing in the number says so. The test that matters
most here is the one that truncates a file between two conversions and asserts that the second run
stops before any measurement, naming the file.
"""

import ast
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import timenet_evaluations
from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.grid import Artifact, Cell, ItemDeclaration, Reader, StorageKey, Task
from timenet_evaluations.grid.assembly import Pairing
from timenet_evaluations.grid.facts import DatasetFacts
from timenet_evaluations.grid.readers import PandasReader, PyTorchReader
from timenet_evaluations.grid.representations import TimeF, timef as timef_module
from timenet_evaluations.grid.representations.identity import (
    MODULE_SUFFIX,
    WRITER_MODULES,
    ArtifactKey,
    library_revision,
    source_digest,
    writer_digest,
)
from timenet_evaluations.harness.plan import Block
from timenet_evaluations.harness.tasks import operation_for
from timenet_evaluations.source import LABEL, PATIENT, SIGNAL


TIMEF_INSTALLED = find_spec("timenet") is not None
"""The timef extra is optional, so the conversion, the check and the parsing path are only
importable where it was installed."""

if TIMEF_INSTALLED:
    from timenet.reader import TimeFReader  # ty: ignore[unresolved-import]

    from timenet_evaluations.grid.representations import conversion, verification
    from timenet_evaluations.grid.representations.timef_parsing import TimeFParsingPath

needs_timef = pytest.mark.skipif(not TIMEF_INSTALLED, reason="the timef extra is not installed")

DATASET = "stub"

N_ITEMS = 6
N_CHANNELS = 2
N_SAMPLES = 300
ITEM_SHAPE = (N_CHANNELS, N_SAMPLES)
CHANNELS = ("first", "second")

PACKAGE = Path(timenet_evaluations.__file__).parent

TIMED_MODULES = ("readers", "tasks.py", "repeat.py", "parsing.py", "timef_parsing.py")
"""The parts a timed repetition runs through. None of them may reach the check, which reads the
whole artifact once and would put that read inside a figure."""

AT = StorageKey(dataset=DATASET, representation=TimeF.name)


def values_of(n_items: int = N_ITEMS, *, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.standard_normal((n_items, N_CHANNELS, N_SAMPLES)).astype(np.float32)


def frame_of(n_items: int = N_ITEMS, *, seed: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            SIGNAL: list(values_of(n_items, seed=seed)),
            LABEL: [f"class-{index % 3}" for index in range(n_items)],
            PATIENT: [f"subject-{index % 2}" for index in range(n_items)],
        }
    )


def signals_of(frame: pd.DataFrame) -> list[np.ndarray]:
    return [np.asarray(values, dtype=np.float32) for values in frame[SIGNAL]]


def digest_of(frame: pd.DataFrame, facts: DatasetFacts) -> str:
    return source_digest(
        signals_of(frame),
        labels=[str(value) for value in frame[LABEL]],
        patients=[str(value) for value in frame[PATIENT]],
        facts=facts,
    )


@pytest.fixture
def facts() -> DatasetFacts:
    return DatasetFacts(
        card_id="timenet/hello_world",
        modality="stub",
        modality_name="Stub",
        unit="volt",
        rate_hz=100,
        channels=CHANNELS,
        target_schema="stub_class",
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def shard_of(artifact: Artifact) -> Path:
    """Give one file of the artifact that the manifest lists and that holds values."""
    shards = sorted(path for path in artifact.path.rglob("*.parquet") if path.is_file())
    assert shards, f"no stored file beneath {artifact.path}"

    return shards[0]


# --- The address a persisted artifact is found at --------------------------------------------


def test_the_address_puts_the_code_above_the_content(tmp_path: Path) -> None:
    # The two components that say which code wrote the bytes stand together and outermost, so one
    # state of the code is one subtree and two revisions of one source stand side by side.
    key = ArtifactKey(revision="git-abc", writer="4567", source="0123")

    assert key.root(tmp_path) == tmp_path / "git-abc" / "4567" / "0123"


def test_the_same_content_gives_the_same_digest(facts: DatasetFacts) -> None:
    assert digest_of(frame_of(), facts) == digest_of(frame_of(), facts)


def test_a_changed_value_changes_the_digest(facts: DatasetFacts) -> None:
    # The digest is what stops a reuse serving content it was not built from, so a source that
    # differs anywhere the conversion reads must land at another address.
    changed = frame_of()
    changed[SIGNAL].iloc[0][0][0] += np.float32(1)

    assert digest_of(changed, facts) != digest_of(frame_of(), facts)


def test_a_changed_scored_class_or_subject_changes_the_digest(facts: DatasetFacts) -> None:
    labelled = frame_of()
    labelled.loc[0, LABEL] = "another-class"
    subjected = frame_of()
    subjected.loc[0, PATIENT] = "another-subject"

    assert digest_of(labelled, facts) != digest_of(frame_of(), facts)
    assert digest_of(subjected, facts) != digest_of(frame_of(), facts)


def test_a_changed_fact_changes_the_digest(facts: DatasetFacts) -> None:
    # The facts are written into the artifact, so an artifact built with other facts is another
    # artifact even where every value is the same.
    assert digest_of(frame_of(), facts.model_copy(update={"rate_hz": 200})) != digest_of(frame_of(), facts)


def test_two_sources_that_join_alike_do_not_share_an_address(facts: DatasetFacts) -> None:
    # Without a length in front of each part, "ab" then "c" and "a" then "bc" are one run of bytes.
    left = frame_of(2)
    left.loc[0, LABEL] = "ab"
    left.loc[1, LABEL] = "c"
    right = frame_of(2)
    right.loc[0, LABEL] = "a"
    right.loc[1, LABEL] = "bc"

    assert digest_of(left, facts) != digest_of(right, facts)


def test_a_shorter_source_does_not_share_an_address(facts: DatasetFacts) -> None:
    assert digest_of(frame_of(N_ITEMS - 1), facts) != digest_of(frame_of(), facts)


def test_the_writer_digest_is_the_same_from_one_call_to_the_next(tmp_path: Path) -> None:
    assert writer_digest(at=AT, path=tmp_path) == writer_digest(at=AT, path=tmp_path)


def test_an_edit_to_any_module_of_the_conversion_moves_the_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The failure this component exists for. A conversion whose code changed writes other bytes, so
    # an artifact the older code wrote must not be reused: the run would report today's numbers
    # against yesterday's bytes, and nothing in the result would say so.
    modules = tmp_path / "writer"
    modules.mkdir()
    (modules / "conversion.py").write_text("the conversion", encoding="utf-8")
    (modules / "timef.py").write_text("the sample id", encoding="utf-8")
    monkeypatch.setattr("timenet_evaluations.grid.representations.identity.WRITER_MODULES", modules)
    before = writer_digest(at=AT, path=tmp_path)

    (modules / "timef.py").write_text("the sample id, padded differently", encoding="utf-8")

    assert writer_digest(at=AT, path=tmp_path) != before


def test_a_module_added_beside_the_conversion_moves_the_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The scope is the directory, so a module written tomorrow is covered the day it appears and
    # nobody has to add it to a list.
    modules = tmp_path / "writer"
    modules.mkdir()
    (modules / "conversion.py").write_text("the conversion", encoding="utf-8")
    monkeypatch.setattr("timenet_evaluations.grid.representations.identity.WRITER_MODULES", modules)
    before = writer_digest(at=AT, path=tmp_path)

    (modules / "layout.py").write_text("a new neighbour", encoding="utf-8")

    assert writer_digest(at=AT, path=tmp_path) != before


def test_the_writer_covers_the_module_that_names_a_sample() -> None:
    # The one this component would miss if its scope were the conversion's own file. The padding of
    # a sample id decides every id in the artifact, and it is a constant in the module beside it.
    covered = set(WRITER_MODULES.glob(MODULE_SUFFIX))

    assert Path(timef_module.__file__) in covered
    assert {"conversion.py", "identity.py", "timef.py"} <= {module.name for module in covered}


@needs_timef
def test_the_revision_names_what_it_was_read_from(tmp_path: Path) -> None:
    revision = library_revision(at=AT, path=tmp_path)

    assert revision.startswith(("git-", "release-"))
    assert revision == library_revision(at=AT, path=tmp_path)


@pytest.mark.skipif(TIMEF_INSTALLED, reason="the timef extra is installed, so a revision is readable")
def test_the_revision_is_refused_where_the_library_is_absent(tmp_path: Path) -> None:
    # A run refuses a missing extra before it converts anything. This is the belt behind that.
    with pytest.raises(EvaluationError) as failure:
        library_revision(at=AT, path=tmp_path)

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message


# --- The artifact persists and a second run reuses it ----------------------------------------


@needs_timef
def test_the_artifact_lands_beneath_the_revision_and_the_source(root: Path, facts: DatasetFacts) -> None:
    frame = frame_of()

    artifact = conversion.write(frame, root, dataset=DATASET, facts=facts)

    key = ArtifactKey(
        revision=library_revision(at=AT, path=root),
        writer=writer_digest(at=AT, path=root),
        source=digest_of(frame, facts),
    )
    assert key.root(root) in artifact.path.parents


@needs_timef
def test_a_second_conversion_of_one_source_reuses_the_artifact(root: Path, facts: DatasetFacts) -> None:
    # The whole point of the persistence: the second call returns the same artifact and writes
    # nothing. A rewrite would erase the version directory and stage a new one, so the manifest
    # would be a different file.
    first = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)
    manifest = first.path / "manifest.json"
    before = manifest.stat()

    second = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)

    after = manifest.stat()
    assert second.path == first.path
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


@needs_timef
def test_a_changed_source_converts_beside_the_one_already_there(root: Path, facts: DatasetFacts) -> None:
    first = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)

    second = conversion.write(frame_of(seed=1), root, dataset=DATASET, facts=facts)

    assert second.path != first.path
    assert first.path.is_dir() and second.path.is_dir()


@needs_timef
def test_another_library_revision_converts_again(
    root: Path, facts: DatasetFacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A benchmark whose subject moves under it is not reproducible, so an artifact one revision
    # wrote is not the artifact another revision would write, and the two do not share an address.
    first = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)
    monkeypatch.setattr(conversion, "library_revision", lambda **_: "git-another")

    second = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)

    assert second.path != first.path
    assert root / "git-another" in second.path.parents


@needs_timef
def test_another_state_of_the_conversion_converts_again(
    root: Path, facts: DatasetFacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An artifact this code would not have written is not this code's artifact, whatever the source
    # and the library revision say.
    first = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)
    monkeypatch.setattr(conversion, "writer_digest", lambda **_: "another-writer")

    second = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)

    assert second.path != first.path
    assert first.path.is_dir() and second.path.is_dir()


@needs_timef
def test_a_reused_artifact_reads_back_the_values_it_was_built_from(root: Path, facts: DatasetFacts) -> None:
    conversion.write(frame_of(), root, dataset=DATASET, facts=facts)

    reused = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)

    items = TimeFParsingPath(dataset=DATASET, channels=CHANNELS).open_pandas(reused.path)
    assert np.array_equal(np.asarray(items[0:N_ITEMS]), values_of())


@needs_timef
def test_an_artifact_at_the_address_that_cannot_be_read_stops_the_run(root: Path, facts: DatasetFacts) -> None:
    # An artifact that exists and cannot be decoded is refused rather than replaced. A conversion
    # over the top would erase whatever is there and report nothing.
    artifact = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)
    (artifact.path / "manifest.json").write_text("not a manifest", encoding="utf-8")

    with pytest.raises(EvaluationError) as failure:
        conversion.write(frame_of(), root, dataset=DATASET, facts=facts)

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message


# --- Every run checks the artifact before it measures anything -------------------------------


@needs_timef
def test_the_check_passes_the_artifact_the_conversion_wrote(root: Path, facts: DatasetFacts) -> None:
    artifact = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)

    verification.verify(artifact, dataset=DATASET)


@needs_timef
def test_a_file_truncated_between_two_runs_stops_the_second_run(root: Path, facts: DatasetFacts) -> None:
    # The failure this whole design exists for. A truncated file reads faster than an intact one,
    # so the damage does not look like damage in the table: it looks like the subject winning.
    # The second call reuses the damaged artifact, and the check is what stops the run.
    first = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)
    damaged = shard_of(first)
    with damaged.open("r+b") as handle:
        handle.truncate(damaged.stat().st_size // 2)

    reused = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)

    assert reused.path == first.path
    with pytest.raises(EvaluationError) as failure:
        verification.verify(reused, dataset=DATASET)

    message = str(failure.value)
    assert f"dataset {DATASET!r}" in message
    assert "representation 'timef'" in message
    assert damaged.relative_to(first.path).as_posix() in message


@needs_timef
def test_a_file_erased_between_two_runs_stops_the_second_run(root: Path, facts: DatasetFacts) -> None:
    artifact = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)
    missing = shard_of(artifact)
    missing.unlink()

    with pytest.raises(EvaluationError) as failure:
        verification.verify(artifact, dataset=DATASET)

    assert missing.relative_to(artifact.path).as_posix() in str(failure.value)


@needs_timef
def test_a_changed_file_of_the_same_length_stops_the_run(root: Path, facts: DatasetFacts) -> None:
    # A size check alone would pass this one. The manifest records a checksum for each file, and
    # the check is what reads it.
    artifact = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)
    edited = shard_of(artifact)
    held = bytearray(edited.read_bytes())
    held[len(held) // 2] ^= 0xFF
    edited.write_bytes(bytes(held))

    with pytest.raises(EvaluationError) as failure:
        verification.verify(artifact, dataset=DATASET)

    assert edited.relative_to(artifact.path).as_posix() in str(failure.value)


@needs_timef
def test_the_check_refuses_an_artifact_of_the_other_representation(root: Path) -> None:
    # The release's own files carry no manifest, and this harness never writes them.
    with pytest.raises(EvaluationError) as failure:
        verification.verify(Artifact(representation="original", path=root), dataset=DATASET)

    assert "artifact representation 'original'" in str(failure.value)


@needs_timef
def test_the_check_refuses_a_path_that_is_not_a_version(root: Path) -> None:
    with pytest.raises(EvaluationError) as failure:
        verification.verify(Artifact(representation="timef", path=root / "nothing-here"), dataset=DATASET)

    assert "representation 'timef'" in str(failure.value)


# --- The check is outside every timed interval -----------------------------------------------


@needs_timef
@pytest.mark.parametrize("task", list(Task))
@pytest.mark.parametrize("reader", [PandasReader(), PyTorchReader()], ids=lambda one: one.name)
def test_no_timed_operation_reaches_the_check(
    root: Path, facts: DatasetFacts, task: Task, reader: Reader, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The check reads and hashes the whole artifact. A timed cell that reached it would report that
    # read under a column that exists to report the cost of one task.
    artifact = conversion.write(frame_of(), root, dataset=DATASET, facts=facts)
    representation = TimeF(
        dataset=DATASET,
        artifact=artifact,
        item=ItemDeclaration(count=N_ITEMS, shape=ITEM_SHAPE),
        parsing_path=TimeFParsingPath(dataset=DATASET, channels=CHANNELS),
    )
    pairing = Pairing(dataset=DATASET, representation=representation, reader=reader)
    at = Cell(dataset=DATASET, representation=TimeF.name, reader=reader.name, task=task)
    monkeypatch.setattr(TimeFReader, "verify", _refuse)

    operation = operation_for(pairing, at, (Block(start=0, count=N_ITEMS),))

    operation()


def _refuse(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("a timed operation reached the check of the whole artifact")


def test_no_module_of_a_timed_path_imports_the_check() -> None:
    # The behavioural test above covers the four tasks as they are written today. This one covers
    # the shape: nothing a repetition runs through may import the module at all.
    modules = [path for path in PACKAGE.rglob("*.py") if any(part in path.as_posix() for part in TIMED_MODULES)]
    assert modules, f"no timed module found beneath {PACKAGE}"

    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        named = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        assert all("verification" not in name for name in named), f"{module.name} imports the check"
