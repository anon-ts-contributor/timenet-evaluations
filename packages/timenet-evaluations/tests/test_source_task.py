"""The corrected sleep-staging task, and the upstream defect it exists for.

Two of these read PyHealth's own source rather than run it. The recording that triggers the defect
is a whole night of EDF, so reproducing it end to end would need the release; reading the two lines
that cause it needs nothing, and says the same thing.
"""

import ast
import inspect
from pathlib import Path

import mne
import numpy as np
from pyhealth.tasks import SleepStagingSleepEDF, sleep_staging_v2
import pytest

from timenet_evaluations.source.sleep_edfx_task import STAGES, SleepStaging


UPSTREAM = Path(inspect.getfile(sleep_staging_v2))


def _epochs_call(source: str) -> ast.Call:
    """Find the `mne.Epochs(...)` call in a module's source."""
    tree = ast.parse(source)
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Epochs"
    )


def test_the_upstream_defect_this_task_exists_for_is_still_there() -> None:
    # The day this fails, PyHealth has fixed it: delete `source/sleep_edfx_task.py`, pass
    # `SleepStagingSleepEDF` again, and delete this file. The whole point of the subclass is that it
    # is temporary, and nothing else would tell us when it stopped being needed.
    call = _epochs_call(UPSTREAM.read_text(encoding="utf-8"))
    passed = call.args[2]

    # Upstream hands `Epochs` the same name it asked `events_from_annotations` for, rather than the
    # map that call returned. That is the defect in one line.
    assert isinstance(passed, ast.Name)
    assert passed.id == "event_id"
    assert not any(keyword.arg == "on_missing" for keyword in call.keywords)


def test_the_corrected_task_passes_the_map_the_annotations_produced() -> None:
    call = _epochs_call(inspect.getsource(SleepStaging))
    passed = call.args[2]

    # `found` is the second value of `events_from_annotations`: the stages this recording holds.
    assert isinstance(passed, ast.Name)
    assert passed.id == "found"


def test_mne_raises_when_a_stage_the_map_names_is_absent_from_the_events() -> None:
    # The mechanism, on three fabricated events and no recording: a map naming a stage that the
    # events do not contain is what `Epochs` refuses, and it is why a night without stage 4 stops a
    # run that hands it all six.
    info = mne.create_info(["ch"], sfreq=100.0, ch_types="eeg", verbose="error")
    raw = mne.io.RawArray(np.zeros((1, 3000), dtype=float), info, verbose="error")
    events = np.array([[0, 0, 0], [1000, 0, 1], [2000, 0, 2]])

    with pytest.raises(ValueError, match="No matching events found"):
        mne.Epochs(raw, events, {"present": 0, "absent": 9}, tmin=0.0, tmax=0.1, baseline=None, verbose="error")

    kept = mne.Epochs(raw, events, {"present": 0}, tmin=0.0, tmax=0.1, baseline=None, verbose="error")

    assert len(kept.event_id) == 1


def test_the_corrected_task_changes_nothing_else_about_an_epoch() -> None:
    # The window duration, the schemas and the task name decide what an item is and what the cache
    # is keyed on. A subclass that moved any of them would be measuring a different dataset.
    assert SleepStaging.task_name == SleepStagingSleepEDF.task_name
    assert SleepStaging.input_schema == SleepStagingSleepEDF.input_schema
    assert SleepStaging.output_schema == SleepStagingSleepEDF.output_schema
    assert SleepStaging().chunk_duration == SleepStagingSleepEDF().chunk_duration
    assert issubclass(SleepStaging, SleepStagingSleepEDF)


def test_the_stage_codes_are_the_ones_upstream_gives_them() -> None:
    # A label this task produces must be the label PyHealth's produces, or the vocabulary the
    # converted artifact records is this repository's invention.
    tree = ast.parse(UPSTREAM.read_text(encoding="utf-8"))
    upstream = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant) and str(key.value).startswith("Sleep stage") for key in node.keys)
    )

    assert ast.literal_eval(upstream) == STAGES
