"""The seam, audited against the finished connector: what may name this dataset, and what fixes the
interpreter range.

Two audits already stand beside this one. `test_source.py` asserts that `pyhealth` is named beneath
the source package and nowhere else, and `test_source_connector.py` asserts that only the connector
imports it and that no module outside the package matches the dataset's own names. Both look for a
name. This file looks for the dataset facts that carry no name at all.

A sleep stage and the number 100 are dataset facts. Neither one spells `sleep-edf`, so both walk
past a search for the dataset's names, and a reader that held either would be a reader that could
not read the next dataset. The rule here is therefore about vocabulary and about bindings, not
about the library.

The rule must fire on a real leak and must stay silent on ordinary English. An earlier version of
the audit beside this one rejected the bare word `sleep`, and it failed on a docstring that said a
test needs no `time.sleep`. A rule that fires on prose teaches the next reader to phrase around it,
and the phrasing they choose still names the dataset. So each term below is matched with the word
boundaries that separate it from the English word it sits inside, and two tests drive the same
detectors in both directions: a fabricated leak must be found, and the legitimate lines the tree
already holds must be cleared.

Three terms are deliberately absent, and each one is absent because the tree already shows the
false positive it would produce:

- `rate`, alone. `result.py` counts items per second and names a class `Rate`. The dataset fact is
  the sampling rate, so the audit reads the sampling-rate vocabulary and the value a name is bound
  to, and it never reads the word `rate` on its own.
- `sampling rate`, as text. `grid/representations/timef.py` explains that its rate field is a whole
  number because every real sampling rate is. That sentence names no dataset.
- `telemetry`. It is the other subset of this release, and it is also an ordinary word for what a
  measurement module reports about itself.

`epoch` is the hard one, because it is the item's name here and the start of unix time everywhere
else. It is matched over the code alone: every prose string is blanked before the search, so a
docstring may say `unix epoch` and no reader may bind the word or write it as a value. A comment is
prose and is not read, which the full-text terms above cover instead.

The interpreter range is the second requirement and it is a different kind of check. Four places
state it — the member's `requires-python`, the comment that says which dependency imposed it, the
classifiers, and `.python-version` — and continuous integration runs both ends of it. The audit
reads all of them and fails when they disagree, rather than restating the range in prose.
"""

import ast
from pathlib import Path
import re
import tomllib

import pytest

from timenet_evaluations import source as source_package


SOURCE_PACKAGE = Path(source_package.__file__).parent

CONNECTOR_MODULES = sorted(SOURCE_PACKAGE.glob("sleep_edfx*.py"))

REGISTRY_MODULE = SOURCE_PACKAGE / "registry.py"

PACKAGE_ROOT = SOURCE_PACKAGE.parent

PACKAGE_MODULES = sorted(PACKAGE_ROOT.rglob("*.py"))

AUDITED = [module for module in PACKAGE_MODULES if module not in CONNECTOR_MODULES]

MEMBER = Path(__file__).resolve().parent.parent

REPOSITORY = MEMBER.parents[1]

MEMBER_PYPROJECT = MEMBER / "pyproject.toml"

PYTHON_VERSION_FILE = REPOSITORY / ".python-version"

WORKFLOW_FILES = sorted(
    path for path in (REPOSITORY / ".github" / "workflows").glob("*") if path.suffix in {".yml", ".yaml"}
)

DATASET_TERMS = (
    r"\bsleep[ _-]stages?\b",
    r"\bmovement[ _-]time\b",
    r"\bhypnogram",
    r"\bpolysomnogra",
    r"\bphysionet\b",
    r"\bcassette\b",
    r"\beeg\b",
    r"\beog\b",
    r"\bemg\b",
    r"\brem\b",
)
"""What this release calls the things it scored, and what it calls the signals it scored them from.

Each one is bounded on both sides, so `rem` does not match `remove` and `sleep stage` does not match
`time.sleep`. `hypnogram` and `polysomnogra` are bounded on the left alone, because their endings
vary and no English word begins with either.
"""

RATE_TERMS = (
    r"\bhz\b",
    r"\bsfreq\b",
)
"""What a sampling rate is called when it is written out.

The unit and the reference loader's own field name for it. Both are absent from the tree today, and
both would arrive with assumption one if that assumption were copied out of the connector.
"""

ITEM_TERMS = (r"\bepochs?\b",)
"""What this dataset calls one canonical item, matched over code and never over prose."""

RATE_NAMES = (
    "hz",
    "sfreq",
    "sample_rate",
    "samplerate",
    "sampling_rate",
    "sample_frequency",
    "sampling_frequency",
    "frequency",
)
"""What a name that holds a sampling rate is built from.

A bare `100` cannot be searched for, so the audit reads the name it is bound to instead. The value
has to be a literal number: `rate_hz: int` is the field the declaration travels in and is how the
rate is supposed to reach a conversion, while `RATE_HZ = 100` is a second copy of one dataset's fact
in a module that serves every dataset.
"""


def _prose_free(source: str) -> str:
    """Unparse one module with every prose string blanked, so a docstring is not read as code.

    Args:
        source: The text of one module.

    Returns:
        The module's code, with each bare string statement replaced by an empty one.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            node.value = ast.Constant(value="")

    return ast.unparse(tree)


def _matched(text: str, terms: tuple[str, ...]) -> list[str]:
    lowered = text.lower()

    return [term for term in terms if re.search(term, lowered)]


def _bindings(source: str) -> list[tuple[str, ast.expr]]:
    """Every name one module binds, with the expression it is bound to.

    Args:
        source: The text of one module.

    Returns:
        One pair per assignment, annotated assignment with a value, and argument default.
    """
    bound: list[tuple[str, ast.expr]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            bound.extend((target.id, node.value) for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
            bound.append((node.target.id, node.value))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            positional = [*node.args.posonlyargs, *node.args.args]
            given = positional[len(positional) - len(node.args.defaults) :]
            bound.extend((argument.arg, default) for argument, default in zip(given, node.args.defaults, strict=True))
            bound.extend(
                (argument.arg, default)
                for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
                if default is not None
            )

    return bound


def _rate_constants(source: str) -> list[str]:
    """Every name one module binds to a number that a sampling rate would be called.

    Args:
        source: The text of one module.

    Returns:
        The names bound to a literal number whose spelling names a sampling rate.
    """
    return [
        name
        for name, value in _bindings(source)
        if any(word in name.lower() for word in RATE_NAMES)
        and isinstance(value, ast.Constant)
        and isinstance(value.value, int | float)
        and not isinstance(value.value, bool)
    ]


def _dataset_facts(source: str) -> list[str]:
    """Every dataset fact one module carries, by the rule that found it.

    Args:
        source: The text of one module.

    Returns:
        One entry per finding, empty when the module names no dataset fact.
    """
    return [
        *(f"names {term}" for term in _matched(source, DATASET_TERMS + RATE_TERMS)),
        *(f"names {term} in code" for term in _matched(_prose_free(source), ITEM_TERMS)),
        *(f"binds the rate constant {name}" for name in _rate_constants(source)),
    ]


def _imported(module: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)

    return names


def _defined(module: Path) -> set[str]:
    return {name for name, _ in _bindings(module.read_text(encoding="utf-8"))}


def _uncommented(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()

    return "\n".join(line for line in lines if not line.strip().startswith("#"))


MEMBER_METADATA = tomllib.loads(MEMBER_PYPROJECT.read_text(encoding="utf-8"))

REQUIRES_PYTHON = MEMBER_METADATA["project"]["requires-python"]

DEPENDENCY = "pyhealth"

DECLARED_BY_THE_DATASET = (
    "NAME",
    "SUBJECT_SPREADSHEET",
    "RECORDINGS_DIRECTORY",
    "UNIT",
    "RATE_HZ",
    "CHUNK_DURATION_SECONDS",
    "SAMPLES_PER_ITEM",
    "ASSUMPTIONS",
)
"""The registered name, the layout it is bound to, the declared item, the rate, and the assumptions.

Every one of them is a fact about one dataset, so every one of them is defined in the connector and
in no other module of the package.
"""

LEAKS = (
    'STAGES = ("Sleep stage W", "Sleep stage R")',
    'UNSCORED = "Movement time"',
    'CHANNELS = ("EEG Fpz-Cz", "EOG horizontal", "EMG submental")',
    '"""The hypnogram says which windows were scored."""',
    "RATE_HZ = 100",
    "SAMPLE_RATE = 100",
    "class Axis:\n    sampling_rate = 100",
    "def axis(sampling_frequency: int = 100) -> int:\n    return sampling_frequency",
    '"""Every channel arrives at 100 Hz."""',
    'UNIT = "epoch"',
    "def read() -> int:\n    epochs = 3\n    return epochs",
)
"""Modules a reader might plausibly grow, each of which the audit must refuse."""

CLEARED = (
    '"""The double needs no time.sleep, because the clock this suite drives is a fake."""',
    '"""One representation would collapse the grid, and no cell would then remain."""',
    '"""A repetition removes nothing and rebuilds nothing beneath the source path."""',
    '"""Every rate for this dataset counts declared items, so a stored rate is about a stated thing."""',
    '"""The values are whole because every real sampling rate is."""',
    '"""The stored instants are seconds since the unix epoch."""',
    "class Rate:\n    at: str\n    items_per_second: float",
    "rate_hz: int",
    "rates = tuple(sorted(measured))",
    "def scale(rate: float, elapsed: float) -> float:\n    return rate / elapsed",
)
"""Lines the tree already holds, or would hold, that name no dataset and must pass the audit."""


# The audit has something to audit


def test_the_connector_is_found_and_the_rest_of_the_package_is_what_is_audited() -> None:
    assert CONNECTOR_MODULES, f"no connector module under {SOURCE_PACKAGE}"
    assert AUDITED
    assert not set(AUDITED) & set(CONNECTOR_MODULES)


# A dataset fact that carries no dataset name


@pytest.mark.parametrize("leak", LEAKS)
def test_a_module_that_carries_a_dataset_fact_fails_the_audit(leak: str) -> None:
    # The rule is checked in this direction first. An audit that only runs over a clean tree passes
    # whether it detects anything or not, and the tree is clean today.
    assert _dataset_facts(leak), f"the audit reads no dataset fact in: {leak}"


@pytest.mark.parametrize("cleared", CLEARED)
def test_ordinary_english_and_the_rates_this_harness_reports_clear_the_audit(cleared: str) -> None:
    # The other direction, and the one that decides whether the rule is usable. Every line here is
    # one the tree holds or would hold, and a rule that refused one of them would be worked around
    # rather than obeyed.
    assert not _dataset_facts(cleared), f"the audit reads a dataset fact in ordinary text: {cleared}"


@pytest.mark.parametrize("module", AUDITED, ids=lambda path: path.name)
def test_no_module_outside_the_connector_carries_a_dataset_fact(module: Path) -> None:
    # A reader that knew a sleep stage from a rate would be a reader that could not read the next
    # dataset, and it would fail by producing a wrong number rather than by refusing.
    found = _dataset_facts(module.read_text(encoding="utf-8"))

    assert not found, f"{module.relative_to(PACKAGE_ROOT)} carries a dataset fact: {found}"


def test_the_connector_is_where_those_facts_are_allowed_to_be() -> None:
    # The audit above passes trivially if the terms match nothing anywhere. The connector is what
    # proves they match something, and it is the one place they may.
    found = [fact for module in CONNECTOR_MODULES for fact in _dataset_facts(module.read_text(encoding="utf-8"))]

    assert found


# What the dataset declares about itself lives with the connector


@pytest.mark.parametrize("declared", DECLARED_BY_THE_DATASET)
def test_what_the_dataset_declares_is_defined_in_the_connector_and_nowhere_else(declared: str) -> None:
    holders = {module for module in PACKAGE_MODULES if declared in _defined(module)}

    assert holders, f"nothing in the package defines {declared}"
    assert holders <= set(CONNECTOR_MODULES), f"{declared} is defined outside the connector: {holders}"


def test_the_registry_is_the_only_module_that_reaches_the_connector() -> None:
    # Adding a dataset is a connector and one line in the registry. A reader, a representation or
    # the result model that imported this one would have to be edited for the second dataset too.
    for module in AUDITED:
        if module == REGISTRY_MODULE:
            continue

        reaching = [name for name in _imported(module) if "sleep_edfx" in name]
        assert not reaching, f"{module.relative_to(PACKAGE_ROOT)} imports the connector: {reaching}"

    assert [name for name in _imported(REGISTRY_MODULE) if "sleep_edfx" in name]


def test_no_module_in_the_package_excuses_a_deferred_import() -> None:
    # A deferred import is how a dataset dependency reaches a module that is not supposed to have
    # one without appearing at the top of the file. The rule holds over the package, not only over
    # the source package, because that is where the excuse would be written.
    for module in PACKAGE_MODULES:
        assert "PLC0415" not in module.read_text(encoding="utf-8"), (
            f"{module.relative_to(PACKAGE_ROOT)} excuses a deferred import"
        )


# The dependency fixes the interpreter range for the whole package


def test_the_dependency_that_imposes_the_interpreter_range_is_the_one_the_connector_imports() -> None:
    requirements = MEMBER_METADATA["project"]["dependencies"]

    assert [requirement for requirement in requirements if requirement.startswith(DEPENDENCY)]
    assert REQUIRES_PYTHON == ">=3.12,<3.14"


def test_the_dependency_declaration_names_what_imposed_the_bound_and_what_it_costs() -> None:
    # The range is narrower than the rest of the workspace would need, and the constraint itself
    # says nothing about which dependency imposed it. The note is also not a detail of one library:
    # every connector brings its own, and the package's range is the intersection of all of them.
    lines = MEMBER_PYPROJECT.read_text(encoding="utf-8").splitlines()
    declared = next(index for index, line in enumerate(lines) if line.strip().startswith(f'"{DEPENDENCY}'))
    note = []
    for line in reversed(lines[:declared]):
        if not line.strip().startswith("#"):
            break
        note.append(line)

    stated = "\n".join(note).lower()
    assert DEPENDENCY in stated
    assert REQUIRES_PYTHON in stated
    assert "connector" in stated


def test_the_pinned_development_interpreter_is_inside_the_range_the_dependency_fixes() -> None:
    floor, ceiling = _bounds()
    pinned = tuple(int(part) for part in PYTHON_VERSION_FILE.read_text(encoding="utf-8").strip().split("."))

    assert floor <= pinned[:2] < ceiling, f"{PYTHON_VERSION_FILE.name} pins {pinned} outside {REQUIRES_PYTHON}"


def test_the_classifiers_name_every_interpreter_the_range_allows_and_no_other() -> None:
    floor, ceiling = _bounds()
    classified = {
        classifier.rsplit(" :: ", 1)[1]
        for classifier in MEMBER_METADATA["project"]["classifiers"]
        if classifier.startswith("Programming Language :: Python :: ") and "." in classifier
    }
    allowed = {f"{floor[0]}.{minor}" for minor in range(floor[1], ceiling[1])}

    assert floor[0] == ceiling[0], "the range spans two major versions and this audit reads one"
    assert classified == allowed, f"the classifiers say {sorted(classified)} and the range allows {sorted(allowed)}"


@pytest.mark.parametrize("workflow", WORKFLOW_FILES, ids=lambda path: path.name)
def test_continuous_integration_runs_the_suite_on_both_ends_of_the_range(workflow: Path) -> None:
    # The floor is run through UV_PYTHON and the pin is run without it. A range that grew a version
    # nothing ran would be a claim about an interpreter no gate ever saw.
    executed = _uncommented(workflow)
    if "make test" not in executed:
        pytest.skip(f"{workflow.name} runs no test suite")

    floor, _ = _bounds()
    assert f"UV_PYTHON={floor[0]}.{floor[1]} make test" in executed, f"{workflow.name} never runs the floor"
    assert re.search(r"^\s*run: make test\s*$", executed, re.MULTILINE), f"{workflow.name} never runs the pin"


def _bounds() -> tuple[tuple[int, int], tuple[int, int]]:
    """Read the lowest interpreter the member supports and the first one it does not.

    Returns:
        The floor and the ceiling, each as a major and minor pair.
    """
    match = re.fullmatch(r">=(\d+)\.(\d+),<(\d+)\.(\d+)", REQUIRES_PYTHON)
    assert match, f"requires-python is not a range this audit can read: {REQUIRES_PYTHON}"
    (
        lowest,
        highest,
    ) = (int(match[1]), int(match[2])), (int(match[3]), int(match[4]))

    return lowest, highest
