"""The two representations one dataset is held in for a run.

``Original`` is the release's own files, which this harness never writes. ``TimeF`` is the converted
form, stored solely as Parquet, which an untimed conversion produces once so that it is on disk to
be read and measured. Both readers open both representations: reading across representations is the
experiment, and no representation belongs to one reader.

Neither module here holds a fact about any dataset. An identifier, a modality, a unit, a target
schema, a channel name, a licence, and a label vocabulary all arrive as arguments — the release's
own files through the connector, and the converted form through :class:`DatasetFacts`. Adding a
dataset means supplying a dataset. It does not mean editing a representation.

``write``, the check of a converted artifact, and the converted form's parsing path are
deliberately absent from this package's exports. They live in ``conversion``, in ``verification``
and in ``timef_parsing`` beside these two modules, and all three import ``timenet``, which is the
``timef`` extra and not a dependency. Import them only when ``find_spec("timenet")`` has already
found the package, the way the command line does. The two representations themselves need nothing
from the extra, so this package imports in an environment that does not have it.

``identity`` is the fourth module beside them and it is the one that imports no part of the storage
library. It holds the address a persisted artifact is found at, which a run computes from the
source and the installed revision. It is out of these exports for the same reason as the other
three: nothing that opens a representation needs it.
"""

from timenet_evaluations.grid.facts import DatasetFacts
from timenet_evaluations.grid.representations.original import Original
from timenet_evaluations.grid.representations.timef import TimeF, sample_id


__all__ = ["DatasetFacts", "Original", "TimeF", "sample_id"]
