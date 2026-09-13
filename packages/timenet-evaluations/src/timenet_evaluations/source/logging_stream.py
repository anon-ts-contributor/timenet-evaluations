"""Keep the dataset library's logging off standard output.

PyHealth configures its own logger when it is imported, and it attaches a stream handler pointing
at standard output:

.. code-block:: python

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)

Standard output is the run's record. SPEC-0019 requires it to hold the summary and nothing else,
byte-identical in a terminal, a pipe, a log and a captured stream, so that two runs of one source
diff to only what changed. A library announcing its cache directory on that stream breaks the
guarantee, and it breaks it once per open: every timed cell opens its reader, so a run over one
dataset prints the same four lines a dozen times.

The lines themselves are worth keeping. They say which cache directory was used and whether the
task cache was hit, which is exactly what someone reading a slow run wants. So this moves them to
standard error rather than silencing them, which is where this package already writes its own
progress.

Nothing here catches anything. Re-pointing a log handler is configuration, not error handling, and
a failure inside the library still raises and still ends the run.
"""

from __future__ import annotations

import logging
import sys


LIBRARY_LOGGER = "pyhealth"
"""The logger the dataset library configures when it is imported."""


def send_library_logging_to_stderr() -> None:
    """Move the dataset library's log records from standard output to standard error.

    Call this once, when the connector package is imported and before any run begins. It replaces
    the handlers the library installed rather than adding to them, so a record is written once and
    on one stream.

    A handler that already writes somewhere other than standard output is left alone: a caller who
    configured logging deliberately is not overridden by this.
    """
    logger = logging.getLogger(LIBRARY_LOGGER)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            logger.removeHandler(handler)
            replacement = logging.StreamHandler(sys.stderr)
            replacement.setFormatter(handler.formatter)
            replacement.setLevel(handler.level)
            logger.addHandler(replacement)
