.PHONY: sync sync-ci lock configure-source test check install-hooks lint-fix build license-check clean

# Every per-component requirements.txt under packages/, turned into `uv run` overlay flags. These
# files are dependency declarations that `uv sync` never reads, so the licence gate below pulls them
# in deliberately instead of hoping something else lists the same packages. Globbed, not listed, so
# a new component is covered the moment its requirements.txt exists.
LICENSE_SCAN_REQUIREMENTS := $(patsubst %,--with-requirements %,$(shell find packages -name requirements.txt 2>/dev/null | sort))

# Distributions exempted from the undeclared-licence half of the gate, as `name` or `name:version`,
# space-separated. Empty on purpose: an exemption is a reviewed decision, not a default. Add an entry
# only together with a comment naming the licence that was confirmed by hand and the pull request
# that confirmed it, so the exemption stays auditable.
LICENSE_EXCEPTIONS :=

# Runs a command with a credential for the private source in its environment. Invoked with
# --no-project so it works before the workspace is synced, which is the whole point of it.
SOURCE_AUTH := uv run --no-project --with pydantic-settings python scripts/configure_source.py

# `timenet` resolves from a private repository, so the full environment needs a credential. The
# helper supplies one per command from the GitHub CLI, or from TIMENET_SOURCE_TOKEN where no CLI
# exists; nothing is written to any git configuration file. `make configure-source` reports which.
sync:
	$(SOURCE_AUTH) --exec uv sync --all-groups --all-extras

# Regenerate the lock. Needs the same credential as `sync`, because locking has to resolve the git
# source in order to record it. CI never runs this; it syncs from the committed lock.
lock:
	$(SOURCE_AUTH) --exec uv lock

# The environment CI builds. It omits the extras, and the `timef` extra is the only one, so this
# resolves without reaching the private repository `timenet` lives in. Everything the gates need -
# ruff, ty, pytest, pre-commit - is in the dev group, so the checks are the same checks; what
# changes is that the format under test is absent, and the tests that need it skip.
sync-ci:
	uv sync --all-groups

# Report which credential the private source will be fetched with, and fail if there is none. There
# is nothing to install: the credential is injected per command by `sync` and `lock`, never written
# to a git configuration file. See scripts/configure_source.py and .env.example.
configure-source:
	$(SOURCE_AUTH)

test:
	uv run pytest

# Fail the build if a copyleft dependency (GPL/LGPL/AGPL/SSPL/EUPL/CDDL/OSL) or a dependency that
# declares no licence at all enters the scanned environment. What is scanned: the environment
# `make sync` builds - the workspace member under packages/ with every extra and every dependency
# group, and everything those pull in transitively - overlaid with every per-component
# requirements.txt found under packages/ (see LICENSE_SCAN_REQUIREMENTS above). The overlay is the
# point: without it those dependencies would be gated only by coincidence.
# Denylist over --allow-only on purpose: --allow-only trips on PEP 639 combined expressions (e.g.
# numpy's "BSD-3-Clause AND 0BSD AND ..."), while --partial-match --fail-on only fires when a denied
# token actually appears. UNKNOWN is one of those tokens because a distribution whose metadata
# declares no licence is an unreviewed licence, not a permissive one; pip-licenses prints
# `fail-on license UNKNOWN was found for package <name>:<version>` and exits 1. Clear a genuine case
# by adding that distribution to LICENSE_EXCEPTIONS, never by shrinking the token list.
license-check:
	uv run $(LICENSE_SCAN_REQUIREMENTS) --with "pip-licenses>=5.0" pip-licenses --partial-match \
		--fail-on="GPL;LGPL;AGPL;SSPL;EUPL;CDDL;OSL;UNKNOWN" \
		$(if $(LICENSE_EXCEPTIONS),--ignore-packages $(LICENSE_EXCEPTIONS)) \
		--format=markdown

# One workspace member today. --all-packages builds every member as its own distribution, so a
# second member needs no edit here.
build:
	uv build --all-packages

check:
	uv run ruff format .
	uv run ruff check .
	uv run ty check .

lint-fix:
	uv run ruff check . --fix

install-hooks:
	uv run pre-commit install

clean:
	rm -rf .venv .pytest_cache .ruff_cache .ty __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
