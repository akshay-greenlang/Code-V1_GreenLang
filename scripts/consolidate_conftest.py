#!/usr/bin/env python3
"""
Conftest.py Fixture Consolidation Script
=========================================

Audits, reports, and cleans up duplicate/redundant conftest.py files
across the GreenLang test suite.

Modes:
    --audit       (default) Full audit report of all conftest.py files
    --clean       Remove empty/trivial conftest.py files (dry-run unless --apply)
    --overrides   Report fixtures that shadow parent conftest definitions

Usage:
    python scripts/consolidate_conftest.py                           # Audit report
    python scripts/consolidate_conftest.py --audit                   # Audit report
    python scripts/consolidate_conftest.py --clean                   # Dry-run clean
    python scripts/consolidate_conftest.py --clean --apply           # Actually delete
    python scripts/consolidate_conftest.py --overrides               # Parent override report
    python scripts/consolidate_conftest.py --dir tests/agents        # Scan specific dir
    python scripts/consolidate_conftest.py --audit --json out.json   # JSON output
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

# Directories to skip entirely (virtual environments, vendored code, etc.)
SKIP_DIRS: Set[str] = {
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "site-packages",
    "__pycache__",
    ".git",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

# Fixture body patterns considered "no-op" (yield nothing / return nothing)
NOOP_BODY_PATTERNS: Set[str] = {
    "yield",
    "yield {}",
    "yield None",
    "yield ()",
    "yield []",
    "return",
    "return None",
    "return {}",
    "return ()",
    "return []",
    "pass",
}

logger = logging.getLogger("consolidate_conftest")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FixtureInfo:
    """Parsed information about a single @pytest.fixture decorated function."""

    name: str
    file_path: str
    line_number: int
    scope: str  # "function", "class", "module", "session"
    autouse: bool
    body_lines: int
    body_source: str  # raw source of the function body
    is_noop: bool  # body matches a no-op pattern
    is_async: bool
    params: Optional[List[str]] = None  # fixture parameter names

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "name": self.name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "scope": self.scope,
            "autouse": self.autouse,
            "body_lines": self.body_lines,
            "is_noop": self.is_noop,
            "is_async": self.is_async,
        }


@dataclass
class ConftestInfo:
    """Parsed information about a single conftest.py file."""

    file_path: str
    relative_path: str
    total_lines: int
    code_lines: int  # non-blank, non-comment lines
    fixtures: List[FixtureInfo] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    has_pytest_configure: bool = False
    has_pytest_plugins: bool = False
    has_pytest_collection_modifyitems: bool = False
    has_other_hooks: bool = False
    parse_error: Optional[str] = None

    @property
    def fixture_count(self) -> int:
        """Number of fixtures defined in this file."""
        return len(self.fixtures)

    @property
    def fixture_names(self) -> List[str]:
        """Names of all fixtures in this file."""
        return [f.name for f in self.fixtures]

    @property
    def is_empty(self) -> bool:
        """File has no meaningful content (only comments/docstrings/blank lines)."""
        return self.code_lines == 0

    @property
    def is_trivial(self) -> bool:
        """File has less than 5 lines of code and no substantive fixtures."""
        if self.code_lines >= 5:
            return False
        if self.has_pytest_configure or self.has_pytest_plugins:
            return False
        if self.has_pytest_collection_modifyitems or self.has_other_hooks:
            return False
        # All fixtures are noop
        return all(f.is_noop for f in self.fixtures)

    @property
    def is_import_only(self) -> bool:
        """File contains only import statements (no fixtures, no hooks)."""
        if self.fixture_count > 0:
            return False
        if self.has_pytest_configure or self.has_pytest_plugins:
            return False
        if self.has_pytest_collection_modifyitems or self.has_other_hooks:
            return False
        return self.code_lines > 0 and len(self.imports) > 0

    @property
    def all_fixtures_noop(self) -> bool:
        """Every fixture in this file is a no-op."""
        return self.fixture_count > 0 and all(f.is_noop for f in self.fixtures)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "total_lines": self.total_lines,
            "code_lines": self.code_lines,
            "fixture_count": self.fixture_count,
            "fixture_names": self.fixture_names,
            "has_pytest_configure": self.has_pytest_configure,
            "has_pytest_plugins": self.has_pytest_plugins,
            "is_empty": self.is_empty,
            "is_trivial": self.is_trivial,
            "is_import_only": self.is_import_only,
            "all_fixtures_noop": self.all_fixtures_noop,
            "parse_error": self.parse_error,
            "fixtures": [f.to_dict() for f in self.fixtures],
        }


@dataclass
class OverrideInfo:
    """A fixture in a child conftest that shadows a parent conftest fixture."""

    fixture_name: str
    child_path: str
    child_line: int
    child_is_noop: bool
    parent_path: str
    parent_line: int
    parent_is_noop: bool


# ---------------------------------------------------------------------------
# AST parsing helpers
# ---------------------------------------------------------------------------


def _count_code_lines(source: str) -> int:
    """Count non-blank, non-comment-only lines in source code."""
    count = 0
    in_docstring = False
    docstring_delim = None

    for raw_line in source.splitlines():
        line = raw_line.strip()

        # Track triple-quoted docstrings
        if not in_docstring:
            if line.startswith('"""') or line.startswith("'''"):
                delim = line[:3]
                # Single-line docstring: """text"""
                if line.count(delim) >= 2 and len(line) > 3:
                    continue  # docstring line, skip
                in_docstring = True
                docstring_delim = delim
                continue
        else:
            if docstring_delim and docstring_delim in line:
                in_docstring = False
                docstring_delim = None
            continue

        if not line:
            continue
        if line.startswith("#"):
            continue

        count += 1

    return count


def _get_fixture_scope(decorator_node: ast.AST) -> Tuple[str, bool]:
    """
    Extract scope and autouse from a @pytest.fixture decorator.

    Returns:
        (scope, autouse) tuple
    """
    scope = "function"
    autouse = False

    if isinstance(decorator_node, ast.Call):
        for kw in decorator_node.keywords:
            if kw.arg == "scope" and isinstance(kw.value, ast.Constant):
                scope = str(kw.value.value)
            elif kw.arg == "autouse" and isinstance(kw.value, ast.Constant):
                autouse = bool(kw.value.value)

    return scope, autouse


def _is_pytest_fixture_decorator(node: ast.AST) -> bool:
    """Check if an AST node is a @pytest.fixture decorator."""
    # @pytest.fixture
    if isinstance(node, ast.Attribute):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "pytest"
            and node.attr == "fixture"
        ):
            return True
    # @pytest.fixture(...)
    if isinstance(node, ast.Call):
        return _is_pytest_fixture_decorator(node.func)
    # @fixture (bare import)
    if isinstance(node, ast.Name) and node.id == "fixture":
        return True
    return False


def _extract_body_source(func_node: ast.FunctionDef, source_lines: List[str]) -> str:
    """Extract the source code of a function body (excluding decorators and def line)."""
    # Function body starts after the def line
    body_start = func_node.body[0].lineno if func_node.body else func_node.lineno + 1
    body_end = func_node.end_lineno if hasattr(func_node, "end_lineno") and func_node.end_lineno else body_start

    if body_start <= len(source_lines) and body_end <= len(source_lines):
        body_lines = source_lines[body_start - 1 : body_end]
        return "\n".join(body_lines)
    return ""


def _is_noop_body(func_node: ast.FunctionDef) -> bool:
    """
    Determine if a function body is a no-op (pass, yield, yield None, etc.).

    Handles:
        - Docstring followed by a single statement
        - Single statement only
        - pass
        - yield / yield None / yield {} / yield ()
        - return / return None / return {} / return ()
    """
    stmts = func_node.body

    # Filter out docstrings (Expr nodes with Constant string values)
    meaningful = []
    for stmt in stmts:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue  # docstring
        meaningful.append(stmt)

    if not meaningful:
        return True  # only docstrings

    if len(meaningful) > 1:
        return False

    stmt = meaningful[0]

    # pass
    if isinstance(stmt, ast.Pass):
        return True

    # yield / yield None / yield {} / yield () / yield []
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
        yield_val = stmt.value.value
        if yield_val is None:
            return True  # bare yield
        if isinstance(yield_val, ast.Constant) and yield_val.value is None:
            return True  # yield None
        if isinstance(yield_val, ast.Dict) and not yield_val.keys:
            return True  # yield {}
        if isinstance(yield_val, ast.Tuple) and not yield_val.elts:
            return True  # yield ()
        if isinstance(yield_val, ast.List) and not yield_val.elts:
            return True  # yield []
        return False

    # return / return None / return {} / return ()
    if isinstance(stmt, ast.Return):
        ret_val = stmt.value
        if ret_val is None:
            return True  # bare return
        if isinstance(ret_val, ast.Constant) and ret_val.value is None:
            return True  # return None
        if isinstance(ret_val, ast.Dict) and not ret_val.keys:
            return True  # return {}
        if isinstance(ret_val, ast.Tuple) and not ret_val.elts:
            return True  # return ()
        if isinstance(ret_val, ast.List) and not ret_val.elts:
            return True  # return []
        return False

    return False


def _extract_imports(tree: ast.Module) -> List[str]:
    """Extract import module names from an AST."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _has_hook(tree: ast.Module, hook_name: str) -> bool:
    """Check if the AST defines a specific pytest hook function."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == hook_name:
                return True
    return False


KNOWN_HOOKS = {
    "pytest_configure",
    "pytest_collection_modifyitems",
    "pytest_addoption",
    "pytest_runtest_setup",
    "pytest_runtest_teardown",
    "pytest_sessionstart",
    "pytest_sessionfinish",
    "pytest_runtest_makereport",
    "pytest_terminal_summary",
}


def parse_conftest(file_path: Path, root: Path) -> ConftestInfo:
    """
    Parse a conftest.py file and extract fixture/hook information via AST.

    Args:
        file_path: Absolute path to the conftest.py file.
        root: Project root for computing relative paths.

    Returns:
        ConftestInfo with all parsed metadata.
    """
    rel_path = str(file_path.relative_to(root)).replace("\\", "/")

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return ConftestInfo(
            file_path=str(file_path),
            relative_path=rel_path,
            total_lines=0,
            code_lines=0,
            parse_error=f"Read error: {exc}",
        )

    source_lines = source.splitlines()
    total_lines = len(source_lines)
    code_lines = _count_code_lines(source)

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        return ConftestInfo(
            file_path=str(file_path),
            relative_path=rel_path,
            total_lines=total_lines,
            code_lines=code_lines,
            parse_error=f"SyntaxError: {exc}",
        )

    # Extract imports
    imports = _extract_imports(tree)

    # Detect pytest hooks
    has_configure = _has_hook(tree, "pytest_configure")
    has_plugins = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "pytest_plugins"
            for t in (node.targets if isinstance(node, ast.Assign) else [])
        )
        for node in ast.iter_child_nodes(tree)
    )
    has_collection_modify = _has_hook(tree, "pytest_collection_modifyitems")
    has_other = any(
        _has_hook(tree, hook)
        for hook in KNOWN_HOOKS - {"pytest_configure", "pytest_collection_modifyitems"}
    )

    # Extract fixtures
    fixtures: List[FixtureInfo] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for deco in node.decorator_list:
            if _is_pytest_fixture_decorator(deco):
                scope, autouse = _get_fixture_scope(deco)
                body_source = _extract_body_source(node, source_lines)
                is_noop = _is_noop_body(node)
                body_line_count = (
                    (node.end_lineno - node.lineno)
                    if hasattr(node, "end_lineno") and node.end_lineno
                    else 1
                )
                # Gather parameter names (excluding self)
                params = [
                    arg.arg
                    for arg in node.args.args
                    if arg.arg != "self"
                ]

                fixtures.append(
                    FixtureInfo(
                        name=node.name,
                        file_path=rel_path,
                        line_number=node.lineno,
                        scope=scope,
                        autouse=autouse,
                        body_lines=body_line_count,
                        body_source=body_source,
                        is_noop=is_noop,
                        is_async=isinstance(node, ast.AsyncFunctionDef),
                        params=params,
                    )
                )
                break  # Only count each function once

    return ConftestInfo(
        file_path=str(file_path),
        relative_path=rel_path,
        total_lines=total_lines,
        code_lines=code_lines,
        fixtures=fixtures,
        imports=imports,
        has_pytest_configure=has_configure,
        has_pytest_plugins=has_plugins,
        has_pytest_collection_modifyitems=has_collection_modify,
        has_other_hooks=has_other,
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_conftest_files(scan_dir: Path) -> List[Path]:
    """
    Walk the directory tree and collect all conftest.py files,
    skipping SKIP_DIRS.

    Args:
        scan_dir: Root directory to scan.

    Returns:
        Sorted list of absolute Paths to conftest.py files.
    """
    results: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(scan_dir):
        # Prune skipped directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS
        ]
        if "conftest.py" in filenames:
            results.append(Path(dirpath) / "conftest.py")
    results.sort()
    return results


# ---------------------------------------------------------------------------
# Override detection
# ---------------------------------------------------------------------------


def _get_conftest_hierarchy(conftest_path: Path, scan_root: Path) -> List[Path]:
    """
    Get the list of parent conftest.py files for a given conftest, ordered
    from closest parent to furthest ancestor.

    Pytest resolves fixtures by walking up the directory tree from a test
    file to the rootdir. We replicate that logic here.

    Args:
        conftest_path: Path to the child conftest.py.
        scan_root: The root directory being scanned.

    Returns:
        List of parent conftest.py paths (closest first).
    """
    parents: List[Path] = []
    current = conftest_path.parent.parent  # Start from the directory above

    while current >= scan_root:
        candidate = current / "conftest.py"
        if candidate.exists() and candidate != conftest_path:
            parents.append(candidate)
        if current == scan_root:
            break
        current = current.parent

    return parents


def find_overrides(
    conftest_map: Dict[str, ConftestInfo],
    scan_root: Path,
) -> List[OverrideInfo]:
    """
    Identify fixtures in child conftest files that shadow fixtures defined
    in parent conftest files.

    Args:
        conftest_map: Mapping from relative path to ConftestInfo.
        scan_root: Root directory for hierarchy resolution.

    Returns:
        List of OverrideInfo instances.
    """
    overrides: List[OverrideInfo] = []

    # Build fixture-name -> file lookup for quick parent resolution
    file_fixtures: Dict[str, Dict[str, FixtureInfo]] = {}
    for rel, info in conftest_map.items():
        file_fixtures[info.file_path] = {f.name: f for f in info.fixtures}

    for rel, info in conftest_map.items():
        if not info.fixtures:
            continue

        child_path = Path(info.file_path)
        parent_paths = _get_conftest_hierarchy(child_path, scan_root)

        for parent_path in parent_paths:
            parent_str = str(parent_path)
            parent_fixtures = file_fixtures.get(parent_str, {})
            if not parent_fixtures:
                continue

            for child_fix in info.fixtures:
                if child_fix.name in parent_fixtures:
                    parent_fix = parent_fixtures[child_fix.name]
                    overrides.append(
                        OverrideInfo(
                            fixture_name=child_fix.name,
                            child_path=info.relative_path,
                            child_line=child_fix.line_number,
                            child_is_noop=child_fix.is_noop,
                            parent_path=parent_fix.file_path,
                            parent_line=parent_fix.line_number,
                            parent_is_noop=parent_fix.is_noop,
                        )
                    )

    overrides.sort(key=lambda o: (o.fixture_name, o.child_path))
    return overrides


# ---------------------------------------------------------------------------
# Mode: Audit
# ---------------------------------------------------------------------------


def run_audit(
    conftest_infos: List[ConftestInfo],
    conftest_map: Dict[str, ConftestInfo],
    scan_root: Path,
    json_output: Optional[str] = None,
) -> int:
    """
    Generate a comprehensive audit report of all conftest.py files.

    Args:
        conftest_infos: Parsed conftest information.
        conftest_map: Mapping for override detection.
        scan_root: Root scan directory.
        json_output: Optional path to write JSON report.

    Returns:
        Exit code (0 = success).
    """
    total_files = len(conftest_infos)
    total_fixtures = sum(ci.fixture_count for ci in conftest_infos)
    parse_errors = [ci for ci in conftest_infos if ci.parse_error]

    # Group fixtures by name across all files
    fixture_groups: Dict[str, List[FixtureInfo]] = defaultdict(list)
    for ci in conftest_infos:
        for fix in ci.fixtures:
            fixture_groups[fix.name].append(fix)

    # Duplicates: fixtures appearing in 2+ files
    duplicates = {
        name: fixes
        for name, fixes in fixture_groups.items()
        if len(fixes) >= 2
    }
    duplicates_sorted = sorted(duplicates.items(), key=lambda x: -len(x[1]))

    # Empty files
    empty_files = [ci for ci in conftest_infos if ci.is_empty]

    # Trivial files
    trivial_files = [ci for ci in conftest_infos if ci.is_trivial and not ci.is_empty]

    # Import-only files
    import_only_files = [ci for ci in conftest_infos if ci.is_import_only]

    # All-noop fixture files
    noop_files = [ci for ci in conftest_infos if ci.all_fixtures_noop]

    # Files with no fixtures and no hooks
    no_fixture_no_hook = [
        ci for ci in conftest_infos
        if ci.fixture_count == 0
        and not ci.has_pytest_configure
        and not ci.has_pytest_plugins
        and not ci.has_pytest_collection_modifyitems
        and not ci.has_other_hooks
        and not ci.is_empty
    ]

    # Candidates for deletion
    deletable = set()
    for ci in empty_files:
        deletable.add(ci.relative_path)
    for ci in trivial_files:
        if ci.code_lines <= 2 and ci.fixture_count == 0:
            deletable.add(ci.relative_path)

    # Noop-only override files (no hooks, all fixtures are noop)
    noop_override_deletable = set()
    for ci in noop_files:
        if not ci.has_pytest_configure and not ci.has_pytest_plugins and not ci.has_other_hooks:
            if not ci.has_pytest_collection_modifyitems:
                noop_override_deletable.add(ci.relative_path)

    # Print report
    sep = "=" * 72
    print()
    print(sep)
    print("  CONFTEST FIXTURE AUDIT REPORT")
    print(sep)
    print()
    print(f"  Scan root:              {scan_root}")
    print(f"  Total conftest.py files: {total_files}")
    print(f"  Total fixture defs:      {total_fixtures}")
    print(f"  Unique fixture names:    {len(fixture_groups)}")
    print(f"  Parse errors:            {len(parse_errors)}")
    print()

    # --- Duplicate fixtures ---
    print(f"{sep}")
    print(f"  DUPLICATE FIXTURES (name appears in 2+ files): {len(duplicates)}")
    print(f"{sep}")
    for name, fixes in duplicates_sorted[:30]:
        noop_count = sum(1 for f in fixes if f.is_noop)
        print(f"\n  {name}: {len(fixes)} files ({noop_count} no-ops)")
        for fix in sorted(fixes, key=lambda f: f.file_path)[:10]:
            noop_tag = " [NOOP]" if fix.is_noop else ""
            scope_tag = f" scope={fix.scope}" if fix.scope != "function" else ""
            auto_tag = " autouse" if fix.autouse else ""
            print(f"    - {fix.file_path}:{fix.line_number}{scope_tag}{auto_tag}{noop_tag}")
        if len(fixes) > 10:
            print(f"    ... and {len(fixes) - 10} more")

    if len(duplicates_sorted) > 30:
        print(f"\n  ... and {len(duplicates_sorted) - 30} more duplicate fixture names")

    # --- Empty files ---
    print(f"\n{sep}")
    print(f"  EMPTY CONFTEST FILES (0 code lines): {len(empty_files)}")
    print(f"{sep}")
    for ci in sorted(empty_files, key=lambda c: c.relative_path):
        print(f"    - {ci.relative_path} ({ci.total_lines} total lines)")

    # --- Trivial files ---
    print(f"\n{sep}")
    print(f"  TRIVIAL CONFTEST FILES (<5 code lines, no real fixtures): {len(trivial_files)}")
    print(f"{sep}")
    for ci in sorted(trivial_files, key=lambda c: c.relative_path):
        print(
            f"    - {ci.relative_path} "
            f"({ci.code_lines} code lines, {ci.fixture_count} fixtures)"
        )

    # --- Import-only files ---
    print(f"\n{sep}")
    print(f"  IMPORT-ONLY CONFTEST FILES (no fixtures/hooks): {len(import_only_files)}")
    print(f"{sep}")
    for ci in sorted(import_only_files, key=lambda c: c.relative_path):
        top_imports = ci.imports[:3]
        import_str = ", ".join(top_imports)
        if len(ci.imports) > 3:
            import_str += f" +{len(ci.imports) - 3} more"
        print(f"    - {ci.relative_path} (imports: {import_str})")

    # --- All-noop fixture files ---
    print(f"\n{sep}")
    print(f"  ALL-NOOP FIXTURE FILES (every fixture is yield/return nothing): {len(noop_files)}")
    print(f"{sep}")
    for ci in sorted(noop_files, key=lambda c: c.relative_path):
        fix_names = ", ".join(ci.fixture_names[:5])
        if len(ci.fixture_names) > 5:
            fix_names += f" +{len(ci.fixture_names) - 5} more"
        print(f"    - {ci.relative_path} ({ci.fixture_count} noop fixtures: {fix_names})")

    # --- No-fixture no-hook files ---
    print(f"\n{sep}")
    print(f"  NO-FIXTURE NO-HOOK FILES (code but no fixtures/hooks): {len(no_fixture_no_hook)}")
    print(f"{sep}")
    for ci in sorted(no_fixture_no_hook, key=lambda c: c.relative_path)[:20]:
        print(f"    - {ci.relative_path} ({ci.code_lines} code lines)")
    if len(no_fixture_no_hook) > 20:
        print(f"    ... and {len(no_fixture_no_hook) - 20} more")

    # --- Parse errors ---
    if parse_errors:
        print(f"\n{sep}")
        print(f"  PARSE ERRORS: {len(parse_errors)}")
        print(f"{sep}")
        for ci in sorted(parse_errors, key=lambda c: c.relative_path):
            print(f"    - {ci.relative_path}: {ci.parse_error}")

    # --- Recommendations ---
    print(f"\n{sep}")
    print("  RECOMMENDATIONS")
    print(f"{sep}")
    print(f"  - {len(empty_files)} conftest.py files are EMPTY and can be deleted")
    print(
        f"  - {len(trivial_files)} conftest.py files are TRIVIAL "
        f"(<5 code lines, no real logic)"
    )
    print(
        f"  - {len(noop_files)} conftest.py files have ONLY no-op fixtures "
        f"(yield {{}}, yield None)"
    )
    print(
        f"  - {len(import_only_files)} conftest.py files contain ONLY imports "
        f"(no fixtures or hooks)"
    )
    print(
        f"  - {len(duplicates)} fixture names are defined in 2+ files "
        f"(potential consolidation)"
    )
    top5 = duplicates_sorted[:5]
    if top5:
        print(f"\n  Top 5 most-duplicated fixtures:")
        for name, fixes in top5:
            print(f"    {name}: {len(fixes)} files")

    # --- Fixture frequency table ---
    print(f"\n{sep}")
    print("  FIXTURE FREQUENCY TABLE (top 40)")
    print(f"{sep}")
    print(f"  {'Fixture Name':<45} {'Files':>6} {'Noop':>6} {'Autouse':>8}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*8}")
    for name, fixes in duplicates_sorted[:40]:
        noop_count = sum(1 for f in fixes if f.is_noop)
        autouse_count = sum(1 for f in fixes if f.autouse)
        print(f"  {name:<45} {len(fixes):>6} {noop_count:>6} {autouse_count:>8}")

    print()

    # JSON output
    if json_output:
        report = {
            "scan_root": str(scan_root),
            "total_files": total_files,
            "total_fixtures": total_fixtures,
            "unique_fixture_names": len(fixture_groups),
            "parse_errors": len(parse_errors),
            "empty_files": [ci.relative_path for ci in empty_files],
            "trivial_files": [ci.relative_path for ci in trivial_files],
            "import_only_files": [ci.relative_path for ci in import_only_files],
            "noop_files": [ci.relative_path for ci in noop_files],
            "duplicates": {
                name: [f.to_dict() for f in fixes]
                for name, fixes in duplicates_sorted
            },
            "all_conftest_files": [ci.to_dict() for ci in conftest_infos],
        }
        out_path = Path(json_output)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  JSON report written to: {out_path}")
        print()

    return 0


# ---------------------------------------------------------------------------
# Mode: Clean
# ---------------------------------------------------------------------------


def run_clean(
    conftest_infos: List[ConftestInfo],
    apply: bool = False,
) -> int:
    """
    Identify and optionally delete empty/trivial conftest.py files.

    Targets:
        1. Completely empty files (0 lines or only whitespace).
        2. Files with only comments and/or docstrings (0 code lines).
        3. Files containing only 'pass'.
        4. Files where ALL fixtures are no-ops AND there are no pytest hooks.

    Args:
        conftest_infos: Parsed conftest information.
        apply: If True, actually delete files. Otherwise dry-run.

    Returns:
        Number of files deleted (or would be deleted).
    """
    candidates: List[Tuple[str, ConftestInfo, str]] = []

    for ci in conftest_infos:
        reason = None

        if ci.is_empty:
            reason = "EMPTY (no code)"
        elif ci.code_lines <= 2 and ci.fixture_count == 0 and not ci.has_pytest_configure and not ci.has_pytest_plugins and not ci.has_other_hooks and not ci.has_pytest_collection_modifyitems:
            reason = f"TRIVIAL ({ci.code_lines} code lines, no fixtures/hooks)"
        elif ci.all_fixtures_noop and not ci.has_pytest_configure and not ci.has_pytest_plugins and not ci.has_other_hooks and not ci.has_pytest_collection_modifyitems and ci.code_lines <= ci.fixture_count * 6 + 5:
            # File content is basically just noop fixtures + imports
            reason = f"ALL-NOOP ({ci.fixture_count} noop fixtures, no hooks)"

        if reason:
            candidates.append((ci.file_path, ci, reason))

    candidates.sort(key=lambda x: x[1].relative_path)

    sep = "=" * 72
    mode_str = "APPLY" if apply else "DRY-RUN"
    print()
    print(sep)
    print(f"  CONFTEST CLEANUP ({mode_str})")
    print(sep)
    print()

    if not candidates:
        print("  No files to clean. All conftest.py files have substantive content.")
        print()
        return 0

    deleted_count = 0
    for file_path, ci, reason in candidates:
        action = "DELETE" if apply else "WOULD DELETE"
        print(f"  {action}: {ci.relative_path}")
        print(f"           Reason: {reason}")
        if ci.fixture_names:
            print(f"           Noop fixtures: {', '.join(ci.fixture_names)}")

        if apply:
            try:
                Path(file_path).unlink()
                deleted_count += 1
                print(f"           Status: DELETED")
            except OSError as exc:
                print(f"           Status: FAILED ({exc})")
        else:
            deleted_count += 1

        print()

    print(f"{sep}")
    if apply:
        print(f"  CLEANED: {deleted_count} files deleted")
    else:
        print(f"  DRY-RUN: {deleted_count} files would be deleted")
        print(f"  Run with --clean --apply to actually delete.")
    print(f"{sep}")
    print()

    return deleted_count


# ---------------------------------------------------------------------------
# Mode: Overrides
# ---------------------------------------------------------------------------


def run_overrides(
    conftest_infos: List[ConftestInfo],
    conftest_map: Dict[str, ConftestInfo],
    scan_root: Path,
) -> int:
    """
    Report all fixtures that shadow parent conftest fixtures.

    Args:
        conftest_infos: Parsed conftest information.
        conftest_map: Mapping for lookup.
        scan_root: Root directory for hierarchy resolution.

    Returns:
        Exit code (0 = success).
    """
    overrides = find_overrides(conftest_map, scan_root)

    # Group by fixture name
    by_name: Dict[str, List[OverrideInfo]] = defaultdict(list)
    for ov in overrides:
        by_name[ov.fixture_name].append(ov)

    by_name_sorted = sorted(by_name.items(), key=lambda x: -len(x[1]))

    # Classify
    noop_overrides = [ov for ov in overrides if ov.child_is_noop]
    intentional_overrides = [ov for ov in overrides if not ov.child_is_noop]

    sep = "=" * 72
    print()
    print(sep)
    print("  CONFTEST PARENT OVERRIDE REPORT")
    print(sep)
    print()
    print(f"  Total overrides found:      {len(overrides)}")
    print(f"  No-op overrides (suspect):  {len(noop_overrides)}")
    print(f"  Substantive overrides:      {len(intentional_overrides)}")
    print(f"  Unique fixture names:       {len(by_name)}")
    print()

    # --- No-op overrides (likely accidental) ---
    print(f"{sep}")
    print(f"  NO-OP OVERRIDES (child yields nothing -- likely accidental): {len(noop_overrides)}")
    print(f"{sep}")
    noop_by_name: Dict[str, List[OverrideInfo]] = defaultdict(list)
    for ov in noop_overrides:
        noop_by_name[ov.fixture_name].append(ov)

    for name, ovs in sorted(noop_by_name.items(), key=lambda x: -len(x[1])):
        print(f"\n  {name}: {len(ovs)} no-op overrides")
        for ov in sorted(ovs, key=lambda o: o.child_path):
            print(f"    child:  {ov.child_path}:{ov.child_line}")
            print(f"    parent: {ov.parent_path}:{ov.parent_line}")

    # --- Substantive overrides (likely intentional) ---
    print(f"\n{sep}")
    print(f"  SUBSTANTIVE OVERRIDES (child has real logic): {len(intentional_overrides)}")
    print(f"{sep}")
    intent_by_name: Dict[str, List[OverrideInfo]] = defaultdict(list)
    for ov in intentional_overrides:
        intent_by_name[ov.fixture_name].append(ov)

    for name, ovs in sorted(intent_by_name.items(), key=lambda x: -len(x[1]))[:20]:
        print(f"\n  {name}: {len(ovs)} overrides")
        for ov in sorted(ovs, key=lambda o: o.child_path)[:5]:
            print(f"    child:  {ov.child_path}:{ov.child_line}")
            print(f"    parent: {ov.parent_path}:{ov.parent_line}")
        if len(ovs) > 5:
            print(f"    ... and {len(ovs) - 5} more")

    if len(intent_by_name) > 20:
        remaining = len(intent_by_name) - 20
        print(f"\n  ... and {remaining} more fixture names with substantive overrides")

    # --- Summary table ---
    print(f"\n{sep}")
    print("  OVERRIDE FREQUENCY TABLE (top 30)")
    print(f"{sep}")
    print(f"  {'Fixture Name':<45} {'Total':>6} {'Noop':>6} {'Real':>6}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6}")
    for name, ovs in by_name_sorted[:30]:
        noop_ct = sum(1 for o in ovs if o.child_is_noop)
        real_ct = len(ovs) - noop_ct
        print(f"  {name:<45} {len(ovs):>6} {noop_ct:>6} {real_ct:>6}")

    print()
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point for the consolidate_conftest script."""
    parser = argparse.ArgumentParser(
        description="Audit, report, and clean up duplicate conftest.py fixtures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/consolidate_conftest.py                           # Audit
              python scripts/consolidate_conftest.py --audit --json report.json
              python scripts/consolidate_conftest.py --clean                   # Dry-run
              python scripts/consolidate_conftest.py --clean --apply           # Delete
              python scripts/consolidate_conftest.py --overrides
              python scripts/consolidate_conftest.py --dir tests/agents
        """),
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--audit",
        action="store_true",
        default=True,
        help="Generate full audit report (default)",
    )
    mode_group.add_argument(
        "--clean",
        action="store_true",
        help="Identify and remove empty/trivial conftest.py files",
    )
    mode_group.add_argument(
        "--overrides",
        action="store_true",
        help="Report fixtures that shadow parent conftest definitions",
    )

    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Scan a specific subdirectory instead of the project root",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files (only used with --clean)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write JSON report to file (only used with --audit)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Determine scan root
    if args.dir:
        scan_root = (ROOT / args.dir).resolve()
    else:
        scan_root = ROOT

    if not scan_root.is_dir():
        print(f"ERROR: Directory does not exist: {scan_root}", file=sys.stderr)
        return 1

    # Discover and parse
    print(f"\nScanning for conftest.py files under: {scan_root}")
    conftest_paths = discover_conftest_files(scan_root)
    print(f"Found {len(conftest_paths)} conftest.py files. Parsing...")

    conftest_infos: List[ConftestInfo] = []
    for path in conftest_paths:
        info = parse_conftest(path, ROOT)
        conftest_infos.append(info)
        if args.verbose and info.parse_error:
            logger.warning("Parse error in %s: %s", info.relative_path, info.parse_error)

    conftest_map = {ci.relative_path: ci for ci in conftest_infos}

    total_fixtures = sum(ci.fixture_count for ci in conftest_infos)
    print(f"Parsed {len(conftest_infos)} files, {total_fixtures} fixture definitions.\n")

    # Dispatch
    if args.clean:
        return run_clean(conftest_infos, apply=args.apply)
    elif args.overrides:
        return run_overrides(conftest_infos, conftest_map, scan_root)
    else:
        return run_audit(conftest_infos, conftest_map, scan_root, json_output=args.json)


if __name__ == "__main__":
    sys.exit(main())
