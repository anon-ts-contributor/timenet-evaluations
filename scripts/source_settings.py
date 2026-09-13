"""Typed configuration for reaching the private repository the ``timef`` extra resolves from.

``timenet`` is declared in ``[tool.uv.sources]`` as a pinned git revision of a private repository.
Git, not this package, is what fetches it, and it does so during dependency resolution — before any
of our code runs. So nothing here can install anything. What it can do is validate the settings that
decide *how* git will authenticate, and fail with a message naming the variable rather than leaving
a bare git error to be interpreted.

Values come from the environment, or from a ``.env`` file at the repository root. See ``.env.example``.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SourceSettings(BaseSettings):
    """How this checkout authenticates to the repository behind the ``timef`` extra."""

    model_config = SettingsConfigDict(
        env_prefix="TIMENET_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    source_token: SecretStr | None = Field(
        default=None,
        description=(
            "A GitHub token with read access to the source repository. Set this in continuous "
            "integration, where no interactive credential helper exists. Leave it unset locally "
            "and let the GitHub CLI supply credentials instead."
        ),
    )
    source_host: str = Field(
        default="github.com",
        description="The host git authenticates to.",
        pattern=r"^[A-Za-z0-9.-]+$",
    )
    source_repo: str = Field(
        default="OpenTSLM/TimeNet",
        description="The owner/name of the repository the timef extra resolves from.",
        pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$",
    )

    @property
    def source_url(self) -> str:
        """The HTTPS URL git will clone.

        Returns:
            The full https URL of the source repository.
        """
        return f"https://{self.source_host}/{self.source_repo}.git"
