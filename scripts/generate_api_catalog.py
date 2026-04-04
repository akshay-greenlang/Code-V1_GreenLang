#!/usr/bin/env python3
"""
generate_api_catalog.py - Scan the entire GreenLang codebase for FastAPI
endpoint definitions using AST parsing and generate a comprehensive
Markdown catalog at docs/api/API_CATALOG.md.

Usage:
    python scripts/generate_api_catalog.py
    python scripts/generate_api_catalog.py --output docs/api/API_CATALOG.md
    python scripts/generate_api_catalog.py --json docs/api/api_catalog.json

The script:
    1. Walks all Python files in configured scan directories.
    2. Uses the ``ast`` module (not regex) to parse each file.
    3. Identifies decorators matching @router.<method>(...) and
       @app.<method>(...) for GET/POST/PUT/DELETE/PATCH.
    4. Extracts HTTP method, path, function name, ``summary`` keyword
       argument from the decorator, and the function body docstring.
    5. Classifies every endpoint into a category (Foundation, Data, MRV,
       EUDR, Infrastructure, Applications, CBAM Pack MVP, Other).
    6. Emits a Markdown catalog organized by category with summary
       statistics at the top.

Author: GreenLang Platform Team
Date: April 2026
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

SCAN_DIRS: List[Path] = [
    REPO_ROOT / "greenlang" / "agents" / "foundation",
    REPO_ROOT / "greenlang" / "agents" / "data",
    REPO_ROOT / "greenlang" / "agents" / "mrv",
    REPO_ROOT / "greenlang" / "agents" / "eudr",
    REPO_ROOT / "greenlang" / "agents" / "intelligence",
    REPO_ROOT / "greenlang" / "infrastructure",
    REPO_ROOT / "applications",
    REPO_ROOT / "cbam-pack-mvp" / "src",
]

DEFAULT_OUTPUT = REPO_ROOT / "docs" / "api" / "API_CATALOG.md"

HTTP_METHODS = frozenset({"get", "post", "put", "delete", "patch"})

# Variable names that hold FastAPI router or app instances.
ROUTER_NAMES = frozenset({"router", "app", "api_router", "api"})

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EndpointInfo:
    """Parsed information about a single FastAPI endpoint."""

    http_method: str
    path: str
    function_name: str
    summary: str
    docstring: str
    file_path: Path
    line_number: int
    category: str = ""

    @property
    def docstring_status(self) -> str:
        """Classify the docstring completeness."""
        if not self.docstring:
            return "None"
        stripped = self.docstring.strip()
        if not stripped:
            return "None"
        # A "Full" docstring contains at least one sentence plus an
        # additional section indicator (Args, Returns, Raises, Attributes,
        # Example, or a second paragraph).
        section_markers = (
            "Args:", "Returns:", "Raises:", "Attributes:",
            "Example:", "Parameters:", "Note:", "Notes:",
        )
        has_section = any(marker in stripped for marker in section_markers)
        has_multi_para = "\n\n" in stripped
        if has_section or has_multi_para or len(stripped) > 200:
            return "Full"
        return "Partial"

    @property
    def relative_path(self) -> str:
        """Return the file path relative to the repository root."""
        try:
            return str(self.file_path.relative_to(REPO_ROOT)).replace("\\", "/")
        except ValueError:
            return str(self.file_path).replace("\\", "/")


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

# Order matters -- first match wins.
_CATEGORY_RULES: List[Tuple[str, str]] = [
    ("greenlang/agents/foundation", "Foundation Agents"),
    ("greenlang/agents/data", "Data Agents"),
    ("greenlang/agents/mrv", "MRV Agents"),
    ("greenlang/agents/eudr", "EUDR Agents"),
    ("greenlang/agents/intelligence", "Intelligence Agents"),
    ("greenlang/infrastructure", "Infrastructure Services"),
    ("cbam-pack-mvp", "CBAM Pack MVP"),
    ("applications/GL-Agent-Factory", "Applications - Agent Factory"),
    ("applications/GL-CSRD-APP", "Applications - CSRD"),
    ("applications/GL-CBAM-APP", "Applications - CBAM"),
    ("applications/GL-VCCI-Carbon-APP", "Applications - VCCI Carbon"),
    ("applications/GL-EUDR-APP", "Applications - EUDR"),
    ("applications/GL-GHG-APP", "Applications - GHG"),
    ("applications/GL-ISO14064-APP", "Applications - ISO 14064"),
    ("applications/GL-Taxonomy-APP", "Applications - EU Taxonomy"),
    ("applications/GL-SBTi-APP", "Applications - SBTi"),
    ("applications/GL-TCFD-APP", "Applications - TCFD"),
    ("applications/GL-CDP-APP", "Applications - CDP"),
    ("applications/GL-SB253-APP", "Applications - SB 253"),
    ("applications/GL Agents", "Applications - GL Agents"),
    ("applications", "Applications - Other"),
]


def _classify(file_path: Path) -> str:
    """Return a human-readable category for a file path."""
    rel = str(file_path.relative_to(REPO_ROOT)).replace("\\", "/")
    for prefix, category in _CATEGORY_RULES:
        if rel.startswith(prefix):
            return category
    return "Other"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _extract_string_value(node: ast.expr) -> Optional[str]:
    """Extract a plain string value from an AST expression node.

    Handles ``ast.Constant`` (Python 3.8+) and ``ast.Str`` (legacy).
    Also handles ``ast.JoinedStr`` (f-strings) by returning None since
    f-string path values cannot be statically resolved.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # Legacy Python < 3.8
    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return node.s  # type: ignore[attr-defined]
    return None


def _extract_keyword_string(
    keywords: List[ast.keyword], name: str
) -> Optional[str]:
    """Return the string value of a keyword argument by name."""
    for kw in keywords:
        if kw.arg == name:
            return _extract_string_value(kw.value)
    return None


def _get_first_positional_string(args: List[ast.expr]) -> Optional[str]:
    """Return the string value of the first positional argument."""
    if args:
        return _extract_string_value(args[0])
    return None


def _get_docstring(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract the docstring from a function definition node."""
    return ast.get_docstring(func_node) or ""


def _is_endpoint_decorator(
    decorator: ast.expr,
) -> Optional[Tuple[str, str, str, List[ast.keyword]]]:
    """Check if a decorator is a FastAPI endpoint decorator.

    Returns ``(variable_name, http_method, path, keywords)`` or None.

    Matches patterns:
        @router.get("/path", ...)
        @app.post("/path", ...)
    Where the first positional argument is the path string.
    """
    if not isinstance(decorator, ast.Call):
        return None

    func = decorator.func
    if not isinstance(func, ast.Attribute):
        return None

    method_name = func.attr
    if method_name not in HTTP_METHODS:
        return None

    # The object the method is called on (e.g. ``router``, ``app``).
    if isinstance(func.value, ast.Name):
        var_name = func.value.id
    elif isinstance(func.value, ast.Attribute):
        # Handles ``self.router.get(...)`` -- use the last attribute.
        var_name = func.value.attr
    else:
        return None

    if var_name not in ROUTER_NAMES:
        return None

    # Extract path from first positional arg or ``path`` keyword.
    path = _get_first_positional_string(decorator.args)
    if path is None:
        path = _extract_keyword_string(decorator.keywords, "path")
    if path is None:
        path = "<dynamic>"

    summary = _extract_keyword_string(decorator.keywords, "summary") or ""

    return (var_name, method_name.upper(), path, decorator.keywords)


# ---------------------------------------------------------------------------
# File scanner
# ---------------------------------------------------------------------------


def _scan_file(file_path: Path) -> List[EndpointInfo]:
    """Parse a single Python file and return all endpoint definitions."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError) as exc:
        logger.warning("Cannot read %s: %s", file_path, exc)
        return []

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        logger.debug("Syntax error in %s: %s", file_path, exc)
        return []

    category = _classify(file_path)
    endpoints: List[EndpointInfo] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for decorator in node.decorator_list:
            result = _is_endpoint_decorator(decorator)
            if result is None:
                continue

            _var_name, http_method, path, _keywords = result
            summary = _extract_keyword_string(_keywords, "summary") or ""
            docstring = _get_docstring(node)

            endpoints.append(
                EndpointInfo(
                    http_method=http_method,
                    path=path,
                    function_name=node.name,
                    summary=summary,
                    docstring=docstring,
                    file_path=file_path,
                    line_number=node.lineno,
                    category=category,
                )
            )
            # A function can only define one endpoint per decorator, but
            # may have multiple decorators (e.g. @router.get + @router.post).
            # Continue the loop to capture all of them.

    return endpoints


def scan_directories(directories: Sequence[Path]) -> List[EndpointInfo]:
    """Walk all directories and collect endpoint definitions."""
    all_endpoints: List[EndpointInfo] = []
    seen_files: set[Path] = set()

    for scan_dir in directories:
        if not scan_dir.exists():
            logger.info("Skipping non-existent directory: %s", scan_dir)
            continue

        for py_file in sorted(scan_dir.rglob("*.py")):
            resolved = py_file.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)

            # Skip test files, __pycache__, and hidden directories.
            parts = resolved.parts
            if any(
                p.startswith("__pycache__")
                or p.startswith(".")
                or p == "node_modules"
                for p in parts
            ):
                continue

            file_endpoints = _scan_file(resolved)
            all_endpoints.extend(file_endpoints)

    return all_endpoints


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def _method_badge(method: str) -> str:
    """Return a visually distinct method label."""
    return f"**{method}**"


def _docstring_badge(status: str) -> str:
    """Return a status indicator for docstring completeness."""
    if status == "Full":
        return "Full"
    if status == "Partial":
        return "Partial"
    return "None"


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate a string to max_len, appending ellipsis if needed."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _escape_pipe(text: str) -> str:
    """Escape pipe characters for Markdown table cells."""
    return text.replace("|", "\\|")


def generate_markdown(
    endpoints: List[EndpointInfo],
    output_path: Path,
) -> str:
    """Generate the full Markdown catalog string."""
    # Group by category.
    by_category: Dict[str, List[EndpointInfo]] = {}
    for ep in endpoints:
        by_category.setdefault(ep.category, []).append(ep)

    # Sort categories in a sensible order.
    category_order = [
        "Foundation Agents",
        "Data Agents",
        "MRV Agents",
        "EUDR Agents",
        "Intelligence Agents",
        "Infrastructure Services",
        "CBAM Pack MVP",
    ]
    # Append application categories in alphabetical order.
    app_categories = sorted(
        c for c in by_category if c.startswith("Applications")
    )
    remaining = sorted(
        c
        for c in by_category
        if c not in category_order and c not in app_categories
    )
    ordered_categories = [
        c for c in category_order if c in by_category
    ] + app_categories + remaining

    # Compute statistics.
    total = len(endpoints)
    doc_full = sum(1 for ep in endpoints if ep.docstring_status == "Full")
    doc_partial = sum(1 for ep in endpoints if ep.docstring_status == "Partial")
    doc_none = sum(1 for ep in endpoints if ep.docstring_status == "None")
    documented_pct = (
        round((doc_full + doc_partial) / total * 100, 1) if total else 0.0
    )
    full_pct = round(doc_full / total * 100, 1) if total else 0.0

    method_counts: Dict[str, int] = {}
    for ep in endpoints:
        method_counts[ep.http_method] = method_counts.get(ep.http_method, 0) + 1

    unique_files = len({ep.file_path for ep in endpoints})

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: List[str] = []
    _a = lines.append

    # --- Header ---
    _a("# GreenLang API Catalog")
    _a("")
    _a(f"> Auto-generated by `scripts/generate_api_catalog.py` on {now}")
    _a(">")
    _a("> Scanned using Python AST parsing for accurate endpoint extraction.")
    _a("")

    # --- Summary statistics ---
    _a("## Summary Statistics")
    _a("")
    _a(f"| Metric | Value |")
    _a(f"|--------|-------|")
    _a(f"| Total Endpoints | **{total}** |")
    _a(f"| Source Files | {unique_files} |")
    _a(f"| Documented (Full + Partial) | {doc_full + doc_partial} ({documented_pct}%) |")
    _a(f"| Fully Documented | {doc_full} ({full_pct}%) |")
    _a(f"| Partially Documented | {doc_partial} |")
    _a(f"| Undocumented | {doc_none} |")
    _a("")

    # --- Methods breakdown ---
    _a("### Endpoints by HTTP Method")
    _a("")
    _a("| Method | Count |")
    _a("|--------|-------|")
    for method in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        count = method_counts.get(method, 0)
        if count:
            _a(f"| {method} | {count} |")
    _a("")

    # --- Category breakdown ---
    _a("### Endpoints by Category")
    _a("")
    _a("| Category | Endpoints | Files | Documented % |")
    _a("|----------|-----------|-------|-------------|")
    for cat in ordered_categories:
        cat_eps = by_category[cat]
        cat_files = len({ep.file_path for ep in cat_eps})
        cat_documented = sum(
            1 for ep in cat_eps if ep.docstring_status != "None"
        )
        cat_pct = (
            round(cat_documented / len(cat_eps) * 100, 1)
            if cat_eps
            else 0.0
        )
        _a(f"| {cat} | {len(cat_eps)} | {cat_files} | {cat_pct}% |")
    _a("")

    # --- Table of Contents ---
    _a("## Table of Contents")
    _a("")
    for cat in ordered_categories:
        anchor = cat.lower().replace(" ", "-").replace("---", "--")
        _a(f"- [{cat}](#{anchor})")
    _a("")

    # --- Per-category tables ---
    for cat in ordered_categories:
        _a(f"## {cat}")
        _a("")
        cat_eps = by_category[cat]

        # Sub-group by file for readability.
        by_file: Dict[str, List[EndpointInfo]] = {}
        for ep in cat_eps:
            by_file.setdefault(ep.relative_path, []).append(ep)

        _a("| Method | Path | Function | Summary | Docstring | File |")
        _a("|--------|------|----------|---------|-----------|------|")
        for file_key in sorted(by_file.keys()):
            file_eps = by_file[file_key]
            # Sort by line number to preserve source order.
            file_eps.sort(key=lambda e: e.line_number)
            for ep in file_eps:
                method_cell = _method_badge(ep.http_method)
                path_cell = _escape_pipe(f"`{ep.path}`")
                func_cell = f"`{ep.function_name}`"
                summary_cell = _escape_pipe(_truncate(ep.summary, 60))
                doc_cell = _docstring_badge(ep.docstring_status)
                file_cell = _escape_pipe(ep.relative_path)
                _a(
                    f"| {method_cell} | {path_cell} | {func_cell} "
                    f"| {summary_cell} | {doc_cell} | {file_cell} |"
                )

        _a("")

    # --- Footer ---
    _a("---")
    _a("")
    _a(f"*Catalog contains {total} endpoints across {unique_files} files "
       f"in {len(ordered_categories)} categories.*")
    _a("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def generate_json(endpoints: List[EndpointInfo]) -> str:
    """Serialize endpoints to a JSON string for programmatic use."""
    records = []
    for ep in endpoints:
        records.append({
            "http_method": ep.http_method,
            "path": ep.path,
            "function_name": ep.function_name,
            "summary": ep.summary,
            "docstring_status": ep.docstring_status,
            "docstring_length": len(ep.docstring),
            "file_path": ep.relative_path,
            "line_number": ep.line_number,
            "category": ep.category,
        })
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_endpoints": len(records),
        "endpoints": records,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan GreenLang codebase for FastAPI endpoints and generate an API catalog.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Markdown output path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        type=Path,
        default=None,
        help="Optional JSON output path for machine-readable catalog.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )
    parser.add_argument(
        "--scan-dir",
        dest="extra_dirs",
        type=Path,
        nargs="*",
        default=[],
        help="Additional directories to scan beyond the defaults.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the API catalog generator."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    scan_dirs = SCAN_DIRS + list(args.extra_dirs)
    logger.info(
        "Scanning %d directories for FastAPI endpoints ...", len(scan_dirs)
    )

    endpoints = scan_directories(scan_dirs)
    logger.info("Found %d endpoints total.", len(endpoints))

    if not endpoints:
        logger.warning(
            "No endpoints found. Check that scan directories exist and "
            "contain FastAPI route definitions."
        )

    # Ensure output directory exists.
    args.output.parent.mkdir(parents=True, exist_ok=True)

    markdown = generate_markdown(endpoints, args.output)
    args.output.write_text(markdown, encoding="utf-8")
    logger.info("Markdown catalog written to %s", args.output)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        json_str = generate_json(endpoints)
        args.json_output.write_text(json_str, encoding="utf-8")
        logger.info("JSON catalog written to %s", args.json_output)

    # Print summary to stdout.
    total = len(endpoints)
    documented = sum(1 for ep in endpoints if ep.docstring_status != "None")
    doc_pct = round(documented / total * 100, 1) if total else 0.0
    categories = len({ep.category for ep in endpoints})
    files = len({ep.file_path for ep in endpoints})

    print()
    print("=" * 60)
    print("  GreenLang API Catalog Generation Complete")
    print("=" * 60)
    print(f"  Endpoints:    {total}")
    print(f"  Source files: {files}")
    print(f"  Categories:   {categories}")
    print(f"  Documented:   {documented}/{total} ({doc_pct}%)")
    print(f"  Output:       {args.output}")
    if args.json_output:
        print(f"  JSON:         {args.json_output}")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
