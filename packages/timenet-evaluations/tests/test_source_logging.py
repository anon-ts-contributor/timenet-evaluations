"""The dataset library's logging stays off the stream that holds the record."""

import ast
import logging
from pathlib import Path
import sys

from timenet_evaluations import cli
from timenet_evaluations.source.logging_stream import LIBRARY_LOGGER, send_library_logging_to_stderr


def _streams() -> list[object]:
    return [
        handler.stream
        for handler in logging.getLogger(LIBRARY_LOGGER).handlers
        if isinstance(handler, logging.StreamHandler)
    ]


def test_the_run_moves_it_before_it_prints_anything() -> None:
    # The call belongs to the run rather than to an import, so a library caller of `run_evaluation`
    # gets it too and nothing happens as a side effect of importing this package.
    body = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    run = next(n for n in ast.walk(body) if isinstance(n, ast.FunctionDef) and n.name == "run_evaluation")
    called = [node.func.id for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]

    assert "send_library_logging_to_stderr" in called


def test_a_handler_on_standard_output_is_replaced_not_added_to() -> None:
    # Adding a second handler would print every record twice, which is worse than the problem.
    logger = logging.getLogger(LIBRARY_LOGGER)
    before = list(logger.handlers)
    added = logging.StreamHandler(sys.stdout)
    added.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(added)

    try:
        send_library_logging_to_stderr()
        streams = _streams()

        assert sys.stdout not in streams
        assert streams.count(sys.stderr) == len([h for h in logger.handlers if isinstance(h, logging.StreamHandler)])
    finally:
        logger.handlers = before


def test_a_handler_pointed_somewhere_else_is_left_alone() -> None:
    # A caller who configured logging deliberately is not overridden. Only standard output is taken,
    # because only standard output is the record.
    logger = logging.getLogger(LIBRARY_LOGGER)
    before = list(logger.handlers)
    elsewhere = logging.StreamHandler(sys.stderr)
    logger.handlers = [elsewhere]

    try:
        send_library_logging_to_stderr()

        assert logger.handlers == [elsewhere]
    finally:
        logger.handlers = before


def test_the_formatter_and_level_survive_the_move() -> None:
    # The lines are worth keeping and worth keeping readable: this moves them, it does not reformat
    # them or change what is emitted.
    logger = logging.getLogger(LIBRARY_LOGGER)
    before = list(logger.handlers)
    original = logging.StreamHandler(sys.stdout)
    original.setFormatter(logging.Formatter("%(message)s"))
    original.setLevel(logging.WARNING)
    logger.handlers = [original]

    try:
        send_library_logging_to_stderr()
        moved = logger.handlers[0]

        assert isinstance(moved, logging.StreamHandler)
        assert moved.stream is sys.stderr
        assert moved.level == logging.WARNING
        assert moved.formatter is not None
        assert moved.formatter._fmt == "%(message)s"
    finally:
        logger.handlers = before
