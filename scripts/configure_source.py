"""Supply git credentials for the private repository the ``timef`` extra resolves from.

``timenet`` is declared in ``[tool.uv.sources]`` as a pinned git revision of a private repository.
Git fetches it during dependency resolution, before any of this package's code runs, and it does so
inside uv's own cache directory — so configuration local to this checkout never reaches it, and the
only options are a machine-wide setting or the environment. This uses the environment.

Two modes:

    scripts/configure_source.py                 report how authentication will be supplied
    scripts/configure_source.py --exec CMD ...   run CMD with the credential in its environment

The credential comes from ``TIMENET_SOURCE_TOKEN`` when set — the continuous integration path — and
otherwise from the GitHub CLI, which keeps a developer from having to write a token to disk at all.
It is passed as ``GIT_CONFIG_*`` variables scoped to the child process, so nothing is written to any
git configuration file and no token appears in a command line.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

from pydantic import ValidationError


sys.path.insert(0, str(Path(__file__).resolve().parent))

from source_settings import SourceSettings


def _token_from_gh() -> str | None:
    """Ask the GitHub CLI for a token, if it is installed and logged in.

    Returns:
        The token, or None when the CLI cannot supply one.
    """
    if shutil.which("gh") is None:
        return None

    found = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=False)

    return found.stdout.strip() or None if found.returncode == 0 else None


def _resolve(settings: SourceSettings) -> tuple[str, str]:
    """Find a usable token and say where it came from.

    Args:
        settings: The validated source configuration.

    Returns:
        The token and a human-readable description of its origin.

    Raises:
        SystemExit: When no credential is available from either source.
    """
    if settings.source_token is not None:
        return settings.source_token.get_secret_value(), "TIMENET_SOURCE_TOKEN"

    token = _token_from_gh()
    if token is not None:
        return token, "the GitHub CLI"

    message = (
        f"no credential for {settings.source_repo}. Either log in with `gh auth login`, or set "
        f"TIMENET_SOURCE_TOKEN to a token with Contents: read on that repository — see .env.example."
    )
    raise SystemExit(message)


def _git_env(settings: SourceSettings, token: str) -> dict[str, str]:
    """Build the environment that rewrites the source host's URLs to carry the token.

    Args:
        settings: The validated source configuration.
        token: The credential to embed in the rewrite.

    Returns:
        The child process environment, the parent's plus the rewrite.
    """
    return os.environ | {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"url.https://x-access-token:{token}@{settings.source_host}/.insteadOf",
        "GIT_CONFIG_VALUE_0": f"https://{settings.source_host}/",
    }


def main() -> int:
    """Report the credential source, or run a command with it applied.

    Returns:
        The exit status of the command under ``--exec``, or 0 when only reporting.

    Raises:
        SystemExit: When the settings are invalid or no credential is available.
    """
    try:
        settings = SourceSettings()
    except ValidationError as error:
        message = f"the source settings are not valid:\n{error}"
        raise SystemExit(message) from error

    token, origin = _resolve(settings)

    if "--exec" not in sys.argv:
        print(f"source:     {settings.source_url}")
        print(f"credential: {origin}")
        print("nothing was written to disk; the credential is passed per command")
        return 0

    command = sys.argv[sys.argv.index("--exec") + 1 :]
    if not command:
        message = "--exec needs a command to run"
        raise SystemExit(message)

    return subprocess.run(command, env=_git_env(settings, token), check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
