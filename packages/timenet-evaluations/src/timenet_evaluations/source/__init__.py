"""The dataset seam: the frame contract, and the one connector that produces it.

This package is the only place in ``timenet_evaluations`` that may name a dataset or import a
dataset library. A grep for ``pyhealth`` across ``packages/timenet-evaluations/src/`` returns
matches below this package and nowhere else. Every module downstream of it — no representation, no
reader, not the result model, not the reporter, not the command line — sees a ``pandas.DataFrame``
and nothing else. Adding a dataset means writing a connector here, and it must not mean editing a
representation or a reader.

The seam holds even when a timer runs. A timed read of the release's own files needs the reference
loader, and it reaches that loader through this package rather than imports it. The position of the
timer does not widen the seam.

Every import in this package sits at module top level. A dataset library imported inside a function
is how a dataset dependency reaches a module that is not supposed to have it without appearing at
the top of the file, so no lint suppression is added to permit one.

A dataset reaches a run as one connector, and a run is given as many of them as the caller names.
The connector declares four members and no more: the untimed read the conversion writes from, the
two timed opens of the release's own files, and what the dataset declares about itself. The name a
run types selects one, through the one hand-written table in
:mod:`timenet_evaluations.source.registry`.

The public surface is re-exported here, so a caller imports from
``timenet_evaluations.source`` and does not name the module the value happens to live in.
:mod:`timenet_evaluations.source.contract` holds what a frame is, apart from any dataset.
:mod:`timenet_evaluations.source.declaration` holds what a dataset declares about itself, apart from
any dataset. :mod:`timenet_evaluations.source.connector` holds the four members, apart from any
dataset. The ``sleep_edfx*`` modules hold the one connector that ships: the registered name and the
four members in :mod:`timenet_evaluations.source.sleep_edfx`, and the untimed pass that reaches the
dataset library in :mod:`timenet_evaluations.source.sleep_edfx_preparation`. A connector is as many
modules as it needs, and the seam is that only they may name the library.
"""

# opens)

from __future__ import annotations

from timenet_evaluations.source.connector import Connector, ParsedItems
from timenet_evaluations.source.contract import (
    COLUMNS,
    LABEL,
    PATIENT,
    SIGNAL,
    as_row,
    check_source_directory,
    signal_stack,
    unbatch,
)
from timenet_evaluations.source.declaration import DatasetDeclaration, check_frame_row_count
from timenet_evaluations.source.logging_stream import send_library_logging_to_stderr
from timenet_evaluations.source.parity import ParityReference, check_parity, sampled_positions
from timenet_evaluations.source.registry import CONNECTOR_MEMBERS, CONNECTORS, connector_for


__all__ = [
    "COLUMNS",
    "CONNECTORS",
    "CONNECTOR_MEMBERS",
    "LABEL",
    "PATIENT",
    "SIGNAL",
    "Connector",
    "DatasetDeclaration",
    "ParityReference",
    "ParsedItems",
    "as_row",
    "check_frame_row_count",
    "check_parity",
    "check_source_directory",
    "connector_for",
    "sampled_positions",
    "send_library_logging_to_stderr",
    "signal_stack",
    "unbatch",
]
