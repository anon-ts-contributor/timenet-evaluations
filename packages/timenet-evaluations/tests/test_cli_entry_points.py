"""The process boundary: what may start a run, what a failure does to it, and what it leaves behind.

Three rules, and every one of them is enforced by an absence. There are two entry points and no
third. Nothing catches, so a raise leaves ``main`` untouched and the process ends non-zero with a
traceback. And a run writes into two roots that both grow and reclaims neither.

An absence cannot be observed by running the thing. Each audit therefore walks a syntax tree, the
member's own metadata, or a file of the repository, and none of them starts a process. That is not
a preference: ``test_ci_boundary.py`` forbids ``subprocess`` and ``runpy`` in every test module,
because a test that started one could invoke the platform's own cache drop.

**The equivalence of the two entry points is not asserted here, because it cannot be.** Comparing
them needs two processes. What can be asserted is the thing the equivalence rests on: that
``__main__.py`` is an import of ``main`` and a guard that calls it, and nothing else. Review
confirms the rest. The one difference the specification permits is the program name in a usage
line, which follows from the parser being built with no ``prog``, so that absence is what is
asserted rather than two names that differ.
"""

import ast
import inspect
from pathlib import Path
import sys
import tomllib

import pytest

import timenet_evaluations
from timenet_evaluations import cli as cli_module
from timenet_evaluations.cli import build_parser, main


PACKAGE = Path(inspect.getfile(timenet_evaluations)).parent

MEMBER = PACKAGE.parents[1]

REPOSITORY = MEMBER.parents[1]

MODULES = sorted(PACKAGE.rglob("*.py"))

MODULE_ENTRY_POINT = PACKAGE / "__main__.py"

GOVERNED = (PACKAGE / "cli.py", MODULE_ENTRY_POINT)
"""The two modules SPEC-0021 governs in full. The rest of the package translates failures at its
own boundaries and holds handlers on purpose, so the no-catch audit is scoped to these two."""

GUARD = "__name__ == '__main__'"
"""The guard as ``ast.unparse`` writes it. Matching the tree rather than the text means a rename of
the constant, a different quote or added whitespace cannot walk past this audit."""

REMOVERS = frozenset({"rmtree", "unlink", "rmdir", "remove", "removedirs"})
"""The members that delete a file or a directory tree. `shutil.rmtree`, `Path.unlink`,
`Path.rmdir` and `os.remove` are the four the requirement names; the fifth is the one a change
would reach for next."""

CLEAN = "clean"

ROOTS = ("out", "artifacts")
"""The two roots a run writes into and never reclaims. They are read out of the parser by the name
of the option that sets each one, so a change to either default moves these audits with it."""

EXITS = frozenset({"sys.exit", "exit", "quit", "os._exit", "SystemExit"})


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(node: ast.AST) -> list[str]:
    """Every call under one node, written the way the source writes it."""
    return [ast.unparse(child.func) for child in ast.walk(node) if isinstance(child, ast.Call)]


def _guards(tree: ast.Module) -> list[ast.If]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.If) and ast.unparse(node.test) == GUARD]


def _imported(tree: ast.Module) -> set[str]:
    """Every module one file imports, by the name it imports it under."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)

    return names


def _recipe(target: str) -> str:
    """One make target's recipe: the tab-indented lines that follow its rule."""
    lines = REPOSITORY.joinpath("Makefile").read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(f"{target}:"))
    recipe = []
    for line in lines[start + 1 :]:
        if not line.startswith("\t"):
            break
        recipe.append(line)

    return "\n".join(recipe)


def _identifier(path: Path) -> str:
    return str(path.relative_to(PACKAGE))


def _root(option: str) -> str:
    """One of the run's two roots, as the parser's default names it."""
    return str(build_parser().get_default(option))


# The audit has something to audit


def test_the_package_holds_the_modules_this_audit_reads() -> None:
    assert len(MODULES) > 1
    assert MODULE_ENTRY_POINT.is_file()
    assert all(path.is_file() for path in GOVERNED)


# Two entry points, and no third


def test_the_member_declares_one_console_script_and_it_resolves_to_the_command() -> None:
    metadata = tomllib.loads((MEMBER / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = metadata["project"]["scripts"]

    assert list(scripts.values()) == [f"{cli_module.__name__}:{main.__name__}"]


def test_the_module_entry_point_imports_the_command_calls_it_and_holds_nothing_else() -> None:
    # Any behaviour added here would exist under one entry point and not the other, which is the
    # one failure this arrangement exists to prevent. Three statements is the whole file.
    tree = _tree(MODULE_ENTRY_POINT)

    assert len(tree.body) == 3

    docstring, imported, guard = tree.body

    assert isinstance(docstring, ast.Expr) and isinstance(docstring.value, ast.Constant)
    assert isinstance(imported, ast.ImportFrom)
    assert imported.module == cli_module.__name__
    assert [(alias.name, alias.asname) for alias in imported.names] == [(main.__name__, None)]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == GUARD
    assert [ast.unparse(statement) for statement in guard.body] == [f"{main.__name__}()"]
    assert guard.orelse == []


def test_only_the_module_entry_point_carries_a_main_guard() -> None:
    # A guard anywhere else is a third way to start a run that nothing documents. Asserted over
    # every module of the package rather than over `cli.py` alone.
    guarded = sorted(_identifier(module) for module in MODULES if _guards(_tree(module)))

    assert guarded == [MODULE_ENTRY_POINT.name]


def test_the_parser_is_built_with_no_program_name() -> None:
    # The program name is the one divergence the two entry points are permitted, and it exists
    # because argparse falls back to `sys.argv[0]`. Comparing the two names needs two processes;
    # the absence of `prog` is what the suite can hold.
    constructed = [
        node
        for node in ast.walk(_tree(PACKAGE / "cli.py"))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "argparse.ArgumentParser"
    ]

    assert len(constructed) == 1
    assert "prog" not in {keyword.arg for keyword in constructed[0].keywords}
    assert build_parser().prog == Path(sys.argv[0]).name


def test_the_command_discards_the_result_and_returns_no_exit_status() -> None:
    # A completed run exits zero by falling off the end, so nothing here returns a status and
    # nothing calls `sys.exit` on the success path.
    function = _function(_tree(PACKAGE / "cli.py"), main.__name__)
    returned = [node for node in ast.walk(function) if isinstance(node, ast.Return) and node.value is not None]
    discarded = [
        ast.unparse(node.value.func)
        for node in function.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]

    assert inspect.signature(main).return_annotation == "None"
    assert returned == []
    assert not EXITS & set(_calls(function))
    assert "run_evaluation" in discarded


# A failure anywhere ends the process


@pytest.mark.parametrize("module", GOVERNED, ids=lambda path: path.name)
def test_no_governed_module_catches_anything(module: Path) -> None:
    # Every failure guarantee of this capability rests on nothing being caught: a `try` however
    # narrow, an `except` however specific, or a `contextlib.suppress`.
    tree = _tree(module)
    caught = [type(node).__name__ for node in ast.walk(tree) if isinstance(node, ast.Try | ast.ExceptHandler)]
    suppressed = [call for call in _calls(tree) if call.split(".")[-1] == "suppress"]

    assert caught == []
    assert suppressed == []
    assert "contextlib" not in _imported(tree)


def test_the_missing_extra_is_settled_by_a_question_and_never_by_an_import() -> None:
    # Whether the extra is importable is a question about the environment, not a failure being
    # swallowed. Asking it with `find_spec` is what keeps the no-catch rule intact in the one place
    # a handler would be most tempting.
    tree = _tree(PACKAGE / "cli.py")
    imported = _imported(tree)

    assert not [name for name in imported if name == "timenet" or name.startswith("timenet.")]
    assert _calls(tree).count("find_spec") == 1
    assert isinstance(cli_module.TIMEF_INSTALLED, bool)


# The growth a run creates, and the absence of anything that reclaims it


@pytest.mark.parametrize("module", MODULES, ids=_identifier)
def test_no_module_of_the_package_removes_a_file_or_a_directory(module: Path) -> None:
    # The conversion is the only place in the package that produces an artifact, and there is no
    # corresponding step anywhere that removes one. Reclaiming the space is manual on purpose.
    removals = [
        ast.unparse(node.func)
        for node in ast.walk(_tree(module))
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr in REMOVERS)
            or (isinstance(node.func, ast.Name) and node.func.id in REMOVERS)
        )
    ]

    assert removals == []


def test_the_clean_target_removes_caches_and_leaves_both_roots_alone() -> None:
    # An operator who wants the space back deletes the directory by hand. The target removes only
    # build and tool caches, and it must not be expected to touch either root.
    recipe = _recipe(CLEAN)

    assert ".venv" in recipe
    assert [option for option in ROOTS if _root(option) in recipe] == []


@pytest.mark.parametrize("option", ROOTS)
def test_both_roots_are_ignored_so_the_growth_appears_in_no_status_listing(option: str) -> None:
    # The artifact root is the large one: a converted corpus that showed up as untracked would
    # eventually be committed by someone staging the whole tree.
    patterns = [line.strip() for line in REPOSITORY.joinpath(".gitignore").read_text(encoding="utf-8").splitlines()]

    assert f"{_root(option)}/" in patterns
