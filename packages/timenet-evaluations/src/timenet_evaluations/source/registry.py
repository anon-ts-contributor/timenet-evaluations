"""The dispatch point: the name a run is given, and the connector that name selects.

A run is given as many datasets as the caller names, never a fixed set. This table is what makes
that true in the tree rather than only in the specification. One name, one connector, one
hand-written line.

Nothing here is discovered. No entry point is read, no installed package is probed, and no directory
is scanned. A connector that arrived because it happened to be installed would make two runs on the
same machine measure different datasets without saying so, and a result has no field to record that.
A connector that was written and never added here is silent, and a report simply does not mention
it.

``CONNECTORS`` carries the annotation ``dict[str, Connector]``, and that annotation is the whole
conformance gate. The protocol is not runtime checkable, so no ``isinstance`` guard exists. Do not
widen the annotation to ``dict[str, Any]`` and do not drop it: two unrelated classes in one mapping
type-check as their own union, and then nothing is verified.

:func:`connector_for` adds the one refusal the annotation cannot make while a run is going. It
refuses a name nothing is registered under, and it refuses a registered connector that is short of
one of the four members. Both refusals belong before the first load. A missing member found at the
first cell that would have called it has already cost a load, a conversion and every cache drop
taken before it, and it leaves a grid printed short of the cells it claims to cover. Where the run
sequence calls this function is the run orchestration's, not this package's.

Adding a dataset is a connector and one line here. It is not an edit to a reader, to a
representation, to the harness, or to the result model.
"""

from __future__ import annotations

from timenet_evaluations.errors import EvaluationError
from timenet_evaluations.source.connector import Connector
from timenet_evaluations.source.sleep_edfx import NAME, SleepEdfxConnector


CONNECTOR_MEMBERS = ("load_for_conversion", "open_pandas", "open_torch", "canonical_item")
"""The four members every registered connector supplies.

The names are written out because the run has to refuse a connector short of one of them before it
loads anything, and a protocol carries no list a caller can read at run time. A member added to
:class:`~timenet_evaluations.source.connector.Connector` and not added here is a member nothing
refuses the absence of.
"""

CONNECTORS: dict[str, Connector] = {NAME: SleepEdfxConnector()}
"""Every dataset this harness can be given, by the name a run selects it with.

One connector is constructed per registered name, once. A connector holds no path, no frame and no
result, so one instance serves every dataset that name resolves to.
"""


def connector_for(name: str) -> Connector:
    """Find the connector a dataset name selects, and refuse one that is not whole.

    Call this during the validation of a run's arguments, before any dataset is loaded, converted or
    measured.

    Args:
        name: The name the run was given for one dataset.

    Returns:
        The connector registered under that name.

    Raises:
        EvaluationError: If no connector is registered under the name, or if the registered
            connector is short of one of the four members.
    """
    if name not in CONNECTORS:
        raise EvaluationError(
            f"no connector is registered for a dataset name the run was given: "
            f"name {name!r}, registered names {sorted(CONNECTORS)}"
        )

    connector = CONNECTORS[name]
    missing = [member for member in CONNECTOR_MEMBERS if not hasattr(type(connector), member)]
    if missing:
        raise EvaluationError(
            f"a registered connector is short of a member the connector protocol requires: "
            f"name {name!r}, connector {type(connector).__name__}, missing {missing}, "
            f"required {list(CONNECTOR_MEMBERS)}"
        )

    return connector
