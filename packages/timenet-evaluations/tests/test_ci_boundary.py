"""The boundary: continuous integration gates the instrument, and it never takes a measurement.

Timings from two machines are not comparable, and a shared runner cannot drop its host's page cache
at all. A benchmark on a runner would therefore either fail the calibration probe or print warm
figures under a cold label. This file is where that rule is checked over the finished tree rather
than trusted to each story that honoured it.

Two paths lead into continuous integration and both are audited here. The first is a workflow step
that invokes the benchmark, or a `schedule:` or `workflow_dispatch:` trigger added to make one run.
The second is the open one: the unit tests do run on every pull request, so a duration assertion, a
sleep or a real cache drop arriving through the suite puts a measurement in the build.

The audits walk a syntax tree and a workflow's own keys rather than searching for text. A text
search over a whole file reads a comment as behaviour and is defeated by an alias or a rename. The
names the audit forbids are read out of the package itself — the console script from the member's
own metadata, the entry points from `cli`, the measuring functions from `harness.__all__` — so a
rename moves the audit with the code instead of past it.
"""

import ast
import inspect
from pathlib import Path
import re
import tomllib

import pytest

import timenet_evaluations
from timenet_evaluations import cli as cli_module, harness as harness_package
from timenet_evaluations.harness.drop import PlatformDropCaches


TESTS = Path(__file__).resolve().parent

MEMBER = TESTS.parent

REPOSITORY = MEMBER.parents[1]

WORKFLOWS = REPOSITORY / ".github" / "workflows"

MAKEFILE = REPOSITORY / "Makefile"

TEST_MODULES = sorted(TESTS.glob("test_*.py"))

WORKFLOW_FILES = sorted(path for path in WORKFLOWS.glob("*") if path.suffix in {".yml", ".yaml"})

FORBIDDEN_TRIGGERS = ("schedule", "workflow_dispatch")

CLOCK_MODULE = "time"

MEASURING_LIBRARIES = frozenset(
    {"timeit", "cProfile", "profile", "pstats", "resource", "tracemalloc", "pytest_benchmark", "asyncio"}
)
"""Libraries that measure a duration, a profile or a footprint. A test that imported one would put a
measurement in the build, whatever it then asserted."""

PROCESS_STARTERS = {
    "subprocess": frozenset({"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}),
    "os": frozenset({"system", "popen", "execv", "execvp", "execl", "execlp", "spawnv", "spawnl", "fork"}),
    "runpy": frozenset({"run_module", "run_path"}),
}
"""The members that start a process, by the module that holds them. Naming the members rather than
the modules keeps `subprocess.CompletedProcess` legal, which is what a runner double returns."""


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _module_aliases(tree: ast.Module, module: str) -> set[str]:
    """Every local name bound to one imported module, under any alias."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".")[0]
                for alias in node.names
                if alias.name == module or alias.name.startswith(f"{module}.")
            )

    return names


def _member_aliases(tree: ast.Module, module: str, members: frozenset[str] | None = None) -> set[str]:
    """Every local name bound to a member of one module, under any alias."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.asname or alias.name for alias in node.names if members is None or alias.name in members)

    return names


def _called_members(tree: ast.Module, module: str, members: frozenset[str] | None = None) -> list[str]:
    """Every call in one file on a member of one module, whatever the member was imported as.

    A reference that is not a call stays legal, so a test can assert that a default is
    ``time.perf_counter_ns`` without reading the clock.
    """
    modules = _module_aliases(tree, module)
    imported = _member_aliases(tree, module, members)
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in imported:
            calls.append(node.func.id)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in modules
            and (members is None or node.func.attr in members)
        ):
            calls.append(f"{node.func.value.id}.{node.func.attr}")

    return calls


def _imported_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    return roots


KEY = re.compile(r"^(?P<lead>\s*(?:- )?)(?P<key>run|uses):(?P<inline>.*)$")

BLOCK = frozenset({"|", ">", "|-", ">-", "|+", ">+"})


def _scripts(path: Path) -> list[str]:
    """Every ``run`` script and every ``uses`` value of one workflow, block scalars included.

    Only what the runner executes is returned. A comment is left behind, so prose about the
    benchmark cannot fail this audit and a step that invokes one cannot pass it.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    scripts: list[str] = []
    index = 0
    while index < len(lines):
        match = KEY.match(lines[index])
        index += 1
        if match is None:
            continue

        inline = match["inline"].strip()
        if inline not in BLOCK:
            scripts.append(inline)
            continue

        column = match.start("key")
        block: list[str] = []
        while index < len(lines) and (
            not lines[index].strip() or len(lines[index]) - len(lines[index].lstrip()) > column
        ):
            block.append(lines[index].strip())
            index += 1
        scripts.append("\n".join(line for line in block if not line.startswith("#")))

    return scripts


def _recipes() -> dict[str, str]:
    """Every make target of the repository, and the recipe it runs."""
    recipes: dict[str, str] = {}
    target: str | None = None
    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("\t") and target is not None:
            recipes[target] = f"{recipes[target]}\n{line.strip()}"
        elif (match := re.match(r"^(?P<name>[A-Za-z][\w-]*):(?!=)", line)) and match["name"] != ".PHONY":
            target = match["name"]
            recipes[target] = ""
        elif not line.startswith("\t"):
            target = None

    return recipes


def _executed(path: Path) -> str:
    """What one workflow runs, with every make target it invokes expanded into its recipe.

    A benchmark step reaches continuous integration through a make target as easily as through a
    command, and the workflow would then name neither the console script nor the package.
    """
    scripts = _scripts(path)
    recipes = _recipes()
    invoked = {name for script in scripts for name in re.findall(r"\bmake\s+([A-Za-z][\w-]*)", script)}

    return "\n".join([*scripts, *(recipes.get(name, "") for name in sorted(invoked))])


def _public_functions(module: object) -> set[str]:
    return {
        name for name in dir(module) if not name.startswith("_") and inspect.isfunction(getattr(module, name, None))
    }


CONSOLE_SCRIPTS = frozenset(
    tomllib.loads((MEMBER / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
)

IMPORT_PACKAGE = timenet_evaluations.__name__

ENTRY_POINTS = frozenset(_public_functions(cli_module))

MEASURING_FUNCTIONS = frozenset(name for name in harness_package.__all__ if name in _public_functions(harness_package))


# The audit has something to audit


def test_the_repository_holds_workflows_and_a_suite_for_this_audit_to_read() -> None:
    assert WORKFLOW_FILES, f"no workflow under {WORKFLOWS}"
    assert len(TEST_MODULES) > 1, f"no suite under {TESTS}"
    assert CONSOLE_SCRIPTS and ENTRY_POINTS and MEASURING_FUNCTIONS


def test_the_package_still_states_the_decision_this_audit_enforces() -> None:
    # ADR-0016 names the package docstring as the governing text, so a benchmark step in a workflow
    # would make it false.
    assert "never part of\ncontinuous integration" in (timenet_evaluations.__doc__ or "")


# No workflow step invokes the benchmark


@pytest.mark.parametrize("workflow", WORKFLOW_FILES, ids=lambda path: path.name)
def test_no_workflow_step_invokes_the_console_script_or_the_module_entry_point(workflow: Path) -> None:
    executed = _executed(workflow)

    for name in (*CONSOLE_SCRIPTS, IMPORT_PACKAGE):
        assert not re.search(rf"\b{re.escape(name)}\b", executed), f"{workflow.name} runs {name}"


@pytest.mark.parametrize("workflow", WORKFLOW_FILES, ids=lambda path: path.name)
def test_no_workflow_step_invokes_an_entry_point_or_a_measuring_function(workflow: Path) -> None:
    # The names are read out of `cli` and out of `harness.__all__`, so a rename cannot walk past
    # this audit and an alias in a workflow script is not a thing that exists.
    executed = _executed(workflow)

    for name in ENTRY_POINTS | MEASURING_FUNCTIONS:
        assert not re.search(rf"\b{re.escape(name)}\b", executed), f"{workflow.name} runs {name}"


@pytest.mark.parametrize("workflow", WORKFLOW_FILES, ids=lambda path: path.name)
def test_no_workflow_carries_a_schedule_or_a_dispatch_trigger(workflow: Path) -> None:
    lines = [line for line in workflow.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]

    for trigger in FORBIDDEN_TRIGGERS:
        assert not [line for line in lines if re.match(rf"^\s*{trigger}\s*:", line)], (
            f"{workflow.name} carries a {trigger} trigger"
        )


@pytest.mark.parametrize("workflow", WORKFLOW_FILES, ids=lambda path: path.name)
def test_continuous_integration_still_runs_the_suite_it_is_there_to_run(workflow: Path) -> None:
    # The boundary is between the benchmark and the harness. Removing the second half would satisfy
    # every audit above and leave the instrument ungated.
    assert "pytest" in _executed(workflow), f"{workflow.name} runs no test suite"


# No measurement arrives through the suite


@pytest.mark.parametrize("module", TEST_MODULES, ids=lambda path: path.name)
def test_no_test_reads_a_clock_or_sleeps(module: Path) -> None:
    # A duration assertion needs a clock or a sleep, and both live in one module. A reference that
    # is not a call stays legal, so a test can still assert which clock a default names.
    calls = _called_members(_tree(module), CLOCK_MODULE)

    assert calls == [], f"{module.name} calls {sorted(set(calls))}"


@pytest.mark.parametrize("module", TEST_MODULES, ids=lambda path: path.name)
def test_no_test_imports_a_library_that_measures(module: Path) -> None:
    named = _imported_roots(_tree(module)) & MEASURING_LIBRARIES

    assert not named, f"{module.name} imports {sorted(named)}"


@pytest.mark.parametrize("module", TEST_MODULES, ids=lambda path: path.name)
def test_no_test_starts_a_process(module: Path) -> None:
    # The drop is the one subprocess of this package, and it is exercised through a double. A test
    # that started a process could invoke the platform's own drop, which needs a privilege the
    # runner does not have and evicts the cache of every other job on the machine.
    tree = _tree(module)
    started = [call for name, members in PROCESS_STARTERS.items() for call in _called_members(tree, name, members)]

    assert started == [], f"{module.name} starts a process: {sorted(set(started))}"


@pytest.mark.parametrize("module", TEST_MODULES, ids=lambda path: path.name)
def test_no_test_builds_the_real_drop_without_a_runner_double(module: Path) -> None:
    # The runner defaults to the real one, so a construction that omits the keyword shells out to
    # the platform's own mechanism the moment the drop is called.
    tree = _tree(module)
    built = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == PlatformDropCaches.__name__
    ]

    for node in built:
        assert "runner" in {keyword.arg for keyword in node.keywords}, (
            f"{module.name}:{node.lineno} builds the drop with the real runner"
        )


def test_the_real_runner_is_the_default_this_audit_guards_against() -> None:
    # The audit above is about a keyword only because the parameter has a default that starts a
    # process. A default of None would make that audit meaningless rather than satisfied.
    assert inspect.signature(PlatformDropCaches.__init__).parameters["runner"].default is not None
