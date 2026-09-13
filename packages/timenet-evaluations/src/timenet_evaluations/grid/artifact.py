"""Which representation a set of bytes is, and where those bytes are.

One artifact describes one representation of one dataset. The same two facts describe the
representation this harness converted and the representation it was handed, so one size function
can answer for both.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Artifact(BaseModel):
    """Which representation these bytes are, and where they are.

    An artifact carries no size. Nothing in this package computes, records, or reports how many
    bytes are on disk. The harness measures the size from the path, after the fact, and refuses a
    path that is not there. A size that a conversion reported can only describe something that
    conversion produced, and the release's own files are never produced here.

    The two storage figures of one dataset are indexed by representation and never by reader. A
    reader does not change bytes on disk. The two figures are also not a compression ratio: the
    two representations hold different content by construction, so their quotient prices the
    decision to adopt the format.
    """

    representation: str
    """The representation's own name, ``original`` or ``timef``. It is the representation's name
    and not a copy of it, so the artifact and the measurement cannot disagree about which row of
    the report describes these bytes."""
    path: Path
    """The file or directory that holds these bytes. It is the path the size measurement is given,
    and the path a reader is opened against. For the converted representation it is the path the
    writing library produced, never one composed from constants."""
