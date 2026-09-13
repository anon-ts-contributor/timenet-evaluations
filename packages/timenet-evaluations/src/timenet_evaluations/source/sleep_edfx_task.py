"""The sleep-staging task this connector runs, with the defect that stops it removed.

PyHealth's own ``SleepStagingSleepEDF`` cannot read this release. It asks mne
for six sleep stages and then requires that all six were found:

.. code-block:: python

    ann_events, _ = mne.events_from_annotations(data, event_id=event_id, ...)
    epochs_train = mne.Epochs(data, ann_events, event_id, ...)

``events_from_annotations`` returns only the stages a recording holds, and returns them twice: as
the events array, and as the map of the stages it actually matched. PyHealth keeps the array and
throws the map away into ``_``, then hands the original six-key map to ``mne.Epochs``. ``Epochs``
checks every key of the map it is given against the events array and calls ``_on_missing`` for any
that is absent, which defaults to raising:

.. code-block:: python

    for key, val in self.event_id.items():
        if val not in events[:, 2]:
            msg = f"No matching events found for {key} (event id {val})"
            _on_missing(on_missing, msg)

A night with no stage 4 is ordinary data. Deep sleep is scored 3 or 4 depending on the night and
the scorer, and plenty of subjects never reach the deeper score: four of the first forty hypnograms
of this release carry ``W, 1, 2, 3, R`` and no stage 4 at all. The run dies on the sixth recording
of three hundred and ninety-four, and about a tenth of the corpus would trip it.

The fix is to hand ``Epochs`` the map ``events_from_annotations`` returned rather than the one it
was asked for. That is the intersection, computed by mne, from this recording. On a night that does
hold all six the two maps are equal and nothing changes.

**This class is a stopgap and is meant to be deleted.** It exists because the corrected code is one
line and the alternative was measuring a subset of the corpus without saying so. When PyHealth fixes
this upstream, delete this module and pass their task again;
``test_the_upstream_defect_this_task_exists_for_is_still_there`` fails on the day that happens, so
the deletion is prompted rather than remembered.

Nothing here changes what an epoch is. The window duration, the channels, the rate and the label
vocabulary are PyHealth's, and the samples this yields for a recording holding all six stages are
the samples PyHealth yields for it.
"""

from __future__ import annotations

from typing import Any

import mne
from pyhealth.tasks import SleepStagingSleepEDF

from timenet_evaluations.source.contract import LABEL, PATIENT, SIGNAL


STAGES: dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}
"""The six scored stages, with the codes PyHealth gives them.

The codes are copied from PyHealth so that a label this task produces is the label its task
produces. A window outside these six is not an item: ``Sleep stage ?`` and ``Movement time`` match
nothing here and produce no epoch, which is this connector's fourth recorded assumption.
"""


class SleepStaging(SleepStagingSleepEDF):
    """PyHealth's sleep-staging task, reading the stages a recording holds rather than six.

    Everything except the event map is inherited. The task name, the schemas and the window
    duration are PyHealth's, so the samples are theirs too.
    """

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Cut one subject's recordings into scored epochs.

        Args:
            patient: One subject, as the reference loader yields it.

        Returns:
            One dict per scored epoch, carrying the subject, the night, the age, the sex, the
            signal and the label, exactly as PyHealth's own task returns them.
        """
        samples: list[dict[str, Any]] = []
        for event in patient.get_events():
            data = mne.io.read_raw_edf(
                event.signal_file,
                stim_channel="Event marker",
                infer_types=True,
                preload=True,
                verbose="error",
            )
            data.set_annotations(mne.read_annotations(event.label_file), emit_warning=False)

            # Both halves are used. `found` is the stages this recording holds, which is what the
            # epochs are cut against; PyHealth discards it and passes STAGES, which is the defect.
            annotated, found = mne.events_from_annotations(data, event_id=STAGES, chunk_duration=self.chunk_duration)

            epochs = mne.Epochs(
                data,
                annotated,
                found,
                tmin=0.0,
                tmax=self.chunk_duration - 1.0 / data.info["sfreq"],
                baseline=None,
                preload=True,
                verbose="error",
            )

            signals = epochs.get_data()
            labels = epochs.events[:, 2]
            samples.extend(
                {
                    PATIENT: patient.patient_id,
                    "night": event.night,
                    "patient_age": event.age,
                    "patient_sex": event.sex,
                    SIGNAL: signals[position, ...],
                    LABEL: int(labels[position, ...]),
                }
                for position in range(labels.shape[0])
            )

        return samples
