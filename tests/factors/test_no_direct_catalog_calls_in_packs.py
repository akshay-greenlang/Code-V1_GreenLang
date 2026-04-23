# -*- coding: utf-8 -*-
"""Guard test for CTO non-negotiable #6.

> "Policy workflows must call method profiles, not raw factors."

This test walks every ``applications/GL-*-APP/`` directory and fails the
build if any pack source file calls the catalog repository directly,
instead of going through ``ResolutionEngine.resolve(method_profile=...)``.

See ``docs/factors/POLICY_RESOLUTION_PATTERN.md`` for the migration
recipe and the canonical call pattern.

Allowed direct-access call sites (bootstrap, admin, ops tooling) are
listed explicitly in :data:`ALLOWED_DIRECT_CATALOG_CALLS`.  Adding a
new entry requires a code-owner review — the default answer is
"refactor to call ``ResolutionEngine.resolve()``".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import pytest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Repo root, computed from this file's location.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Directory holding all GreenLang policy applications (one per workflow).
APPLICATIONS_ROOT = REPO_ROOT / "applications"

#: Glob used to find each pack root.  Every directory matching this glob
#: is treated as a policy workflow that must obey non-negotiable #6.
PACK_DIR_GLOB = "GL-*-APP"

#: Forbidden patterns — any of these in pack source code is a violation.
#: The patterns intentionally cover the catalog-repository layer
#: (``catalog_repository.get(...)``, ``FileCatalogRepository(...)``,
#: ``PgCatalogRepository(...)``) AND the legacy SDK / per-pack data-module
#: shortcuts (``EmissionFactorClient(...).get_fuel_factor(...)``,
#: ``ef.get_emission_factor_by_cn_code(...)``).  Both bypass the
#: ``ResolutionEngine`` cascade and therefore the method-pack
#: ``SelectionRule``.
FORBIDDEN_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]", str], ...] = (
    (
        "catalog_repository.get(",
        re.compile(r"\bcatalog_repository\s*\.\s*get\s*\("),
        "Use ResolutionEngine.resolve(method_profile=...) instead "
        "of calling the catalog repository directly.",
    ),
    (
        "catalog.get_factor(",
        re.compile(r"\bcatalog\s*\.\s*get_factor\s*\("),
        "Use ResolutionEngine.resolve(method_profile=...) instead "
        "of catalog.get_factor().",
    ),
    (
        "FileCatalogRepository(",
        re.compile(r"\bFileCatalogRepository\s*\("),
        "Pack code must not construct a catalog repository directly. "
        "Receive a pre-built ResolutionEngine via dependency injection.",
    ),
    (
        "PgCatalogRepository(",
        re.compile(r"\bPgCatalogRepository\s*\("),
        "Pack code must not construct a catalog repository directly. "
        "Receive a pre-built ResolutionEngine via dependency injection.",
    ),
    (
        "ef.get_emission_factor_by_cn_code(",
        re.compile(r"\bef\s*\.\s*get_emission_factor_by_cn_code\s*\("),
        "Legacy CBAM helper bypasses the EU_CBAM method profile. "
        "Use ResolutionEngine.resolve(method_profile=MethodProfile.EU_CBAM).",
    ),
    (
        "EmissionFactorClient.get_factor(",
        re.compile(r"\.\s*get_factor\s*\("),
        "EmissionFactorClient.get_factor() bypasses the method-pack "
        "selection rule. Use ResolutionEngine.resolve(method_profile=...).",
    ),
    (
        "EmissionFactorClient.get_fuel_factor(",
        re.compile(r"\.\s*get_fuel_factor\s*\("),
        "EmissionFactorClient.get_fuel_factor() bypasses the method-pack "
        "selection rule. Use ResolutionEngine.resolve(method_profile=...).",
    ),
    (
        "EmissionFactorClient.get_factor_by_name(",
        re.compile(r"\.\s*get_factor_by_name\s*\("),
        "EmissionFactorClient.get_factor_by_name() bypasses the method-pack "
        "selection rule. Use ResolutionEngine.resolve(method_profile=...).",
    ),
)

#: At least one of these tokens MUST appear in any pack file that talks
#: about emission factors at all.  Used by the inverse "every factor
#: lookup goes through ResolutionEngine" assertion.
RESOLUTION_TOKENS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bResolutionEngine\.resolve\s*\("),
    re.compile(r"\bengine\.resolve\s*\("),
    re.compile(r"\bMethodProfile\.[A-Z_]+\b"),
    re.compile(r"\bResolutionRequest\s*\("),
)

#: Files explicitly allowed to call the catalog directly.  Paths are
#: stored relative to the repo root with forward-slash separators
#: regardless of host OS.  Add a new entry only with code-owner review;
#: include a one-line reason in the comment beside it.
ALLOWED_DIRECT_CATALOG_CALLS: Set[str] = {
    # ---- bootstrap / seed scripts (no entries today) ----
    # "applications/GL-CBAM-APP/scripts/seed_catalog.py",  # bootstrap only
}

#: Substrings — if any appears in the file's path, the file is excluded
#: from the scan entirely.  Keeps the test focused on production code.
PATH_EXCLUDES: Tuple[str, ...] = (
    "/tests/",
    "\\tests\\",
    "/test_",
    "\\test_",
    "/__pycache__/",
    "\\__pycache__\\",
    "/_archive/",
    "\\_archive\\",
    "/scripts/",
    "\\scripts\\",
    "/migrations/",
    "\\migrations\\",
    "/v1/",          # explicit legacy snapshots kept for compatibility
    "\\v1\\",
    # Doc / data modules (factor tables themselves are bootstrap data,
    # not policy code) — they may define helper readers but they are
    # not the workflow that must obey non-negotiable #6.
    "/data/emission_factors.py",
    "\\data\\emission_factors.py",
    "/data/emission_factors_v2.py",
    "\\data\\emission_factors_v2.py",
)

#: File-name suffixes the scan considers "policy code".
SOURCE_SUFFIXES: Tuple[str, ...] = (".py",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_pack_dirs() -> Iterable[Path]:
    """Yield every ``applications/GL-*-APP`` directory present on disk."""
    if not APPLICATIONS_ROOT.exists():
        return
    for entry in sorted(APPLICATIONS_ROOT.glob(PACK_DIR_GLOB)):
        if entry.is_dir():
            yield entry


def _iter_source_files(pack_dir: Path) -> Iterable[Path]:
    """Yield every Python source file inside a pack root, applying excludes."""
    for path in pack_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        rel = _rel_posix(path)
        if any(token in rel for token in PATH_EXCLUDES):
            continue
        yield path


def _rel_posix(path: Path) -> str:
    """Return ``path`` relative to the repo root with forward slashes."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


#: Per-line escape marker — when this token appears in the original
#: source line (anywhere, including inside a comment), the line is
#: ignored by the scan.  Use sparingly and ALWAYS pair with a code
#: comment explaining why the call is the engine's adapter, not a
#: direct policy-workflow call.  Format:
#:     return list(ef.get_emission_factor_by_cn_code(cn_code) or [])  # noqa: NN6 — engine adapter
NOQA_MARKER = "noqa: NN6"


def _strip_strings_and_comments(line: str) -> str:
    """Remove '# …' comments and obvious string literals.

    Cheap heuristic — good enough to ignore patterns that appear inside
    docstrings, comments, or example strings.  Multi-line triple-quoted
    docstrings are filtered separately at the file level.
    """
    # Strip trailing comments.
    if "#" in line:
        line = line.split("#", 1)[0]
    # Drop simple single-line string literals.
    line = re.sub(r"'[^']*'", "''", line)
    line = re.sub(r'"[^"]*"', '""', line)
    return line


def _split_non_string_lines(text: str) -> List[Tuple[int, str, bool]]:
    """Yield (line_number, code_only_content, has_noqa) for every code line.

    Removes triple-quoted blocks (docstrings + multi-line strings).  The
    fence detection is intentionally simple: a line that contains an odd
    number of triple-quote tokens toggles the "inside docstring" flag.

    The third tuple element is ``True`` when the original (pre-strip)
    line carries the :data:`NOQA_MARKER`.
    """
    out: List[Tuple[int, str, bool]] = []
    in_docstring = False
    fence: str = ""
    for lineno, raw in enumerate(text.splitlines(), start=1):
        has_noqa = NOQA_MARKER in raw
        scan_line = raw
        emit_after_strip = True
        if in_docstring:
            # Look for the closing fence on this line.
            close_idx = scan_line.find(fence)
            if close_idx != -1:
                in_docstring = False
                scan_line = scan_line[close_idx + len(fence):]
            else:
                continue  # entire line is inside a docstring
        # Now check for new opening fences on the remainder of the line.
        for candidate in ('"""', "'''"):
            opens = scan_line.count(candidate)
            if opens == 0:
                continue
            # Odd count → unbalanced → docstring opened (or closed and
            # reopened).  We only care about the trailing state.
            if opens % 2 == 1:
                in_docstring = True
                fence = candidate
                # Drop content from the last fence onward.
                scan_line = scan_line.rsplit(candidate, 1)[0]
                break
            else:
                # Balanced on the line — collapse all triple-quoted spans.
                scan_line = re.sub(
                    re.escape(candidate) + r".*?" + re.escape(candidate),
                    '""',
                    scan_line,
                    flags=re.DOTALL,
                )
        if emit_after_strip:
            stripped = _strip_strings_and_comments(scan_line)
            out.append((lineno, stripped, has_noqa))
    return out


def _is_allowlisted(rel_path: str) -> bool:
    return rel_path in ALLOWED_DIRECT_CATALOG_CALLS


def _scan_file(path: Path) -> List[Tuple[int, str, str, str]]:
    """Return ``(lineno, label, snippet, hint)`` for every violation."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    violations: List[Tuple[int, str, str, str]] = []
    for lineno, code, has_noqa in _split_non_string_lines(text):
        if not code.strip():
            continue
        if has_noqa:
            continue  # explicit per-line escape — engine adapter wiring
        for label, pattern, hint in FORBIDDEN_PATTERNS:
            if pattern.search(code):
                violations.append((lineno, label, code.strip()[:160], hint))
    return violations


#: Tokens that indicate the file performs an ACTUAL catalog-or-engine
#: factor lookup (not just declares an ``emission_factor: float``
#: Pydantic field, and not just maintains a local in-memory EF
#: registry).  Used by the inverse "every pack performing lookups must
#: bind a method profile" assertion to avoid false-positives on
#: schema-only or registry-only modules.
LOOKUP_TOKENS: Tuple[str, ...] = (
    # Legacy CBAM helper that pulls from the local data module.
    "get_emission_factor_by_cn_code(",
    # Legacy SDK entry points that bypass the cascade.
    ".get_fuel_factor(",
    "ef_client.get_factor(",
    "client.get_factor(",
    "EmissionFactorClient(",
    # Direct catalog-repository constructors (always disallowed in
    # policy code).
    "FileCatalogRepository(",
    "PgCatalogRepository(",
    "FactorCatalogRepository(",
    "catalog_repository.get(",
    "catalog.get_factor(",
    # The compliant path — counted because we want the inverse check
    # to PASS when at least one production file is engine-bound.
    "ResolutionEngine.resolve(",
    "engine.resolve(",
)


def _file_uses_factor_lookup(text: str) -> bool:
    """True if a file actually fetches a factor (not just types it)."""
    return any(n in text for n in LOOKUP_TOKENS)


def _file_uses_resolution_engine(text: str) -> bool:
    return any(p.search(text) for p in RESOLUTION_TOKENS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_applications_root_exists() -> None:
    """Sanity-check that the applications/ tree is reachable from tests."""
    assert APPLICATIONS_ROOT.is_dir(), (
        f"applications root not found at {APPLICATIONS_ROOT!s}; "
        "the guard cannot run."
    )


def test_at_least_one_pack_present() -> None:
    """We must see at least one GL-*-APP, otherwise the scan is vacuous."""
    packs = list(_iter_pack_dirs())
    assert packs, (
        f"No GL-*-APP packs found under {APPLICATIONS_ROOT!s}; "
        "the guard cannot validate non-negotiable #6."
    )


def test_no_direct_catalog_calls_in_packs() -> None:
    """No pack source file may call the catalog repository directly.

    Every offending file:line is included in the failure message,
    grouped by file, with the recommended fix.
    """
    failures: List[str] = []

    for pack_dir in _iter_pack_dirs():
        for source_file in _iter_source_files(pack_dir):
            rel = _rel_posix(source_file)
            if _is_allowlisted(rel):
                continue
            violations = _scan_file(source_file)
            if not violations:
                continue
            lines: List[str] = [f"  - {rel}:"]
            for lineno, label, snippet, hint in violations:
                lines.append(
                    f"      L{lineno}: {label}\n"
                    f"          code:   {snippet}\n"
                    f"          fix:    {hint}"
                )
            failures.append("\n".join(lines))

    if failures:
        msg = (
            "\n\nCTO non-negotiable #6 violation: policy workflows must call\n"
            "ResolutionEngine.resolve(method_profile=...) instead of touching\n"
            "the catalog repository directly.\n\n"
            "Offending file(s):\n\n"
            + "\n\n".join(failures)
            + "\n\nSee docs/factors/POLICY_RESOLUTION_PATTERN.md for the\n"
            "migration recipe.  Bootstrap / admin tooling that genuinely\n"
            "needs raw catalog access must be added to\n"
            "ALLOWED_DIRECT_CATALOG_CALLS in this test, with a one-line\n"
            "reason and code-owner approval.\n"
        )
        pytest.fail(msg, pytrace=False)


def test_packs_route_factor_lookups_through_resolution_engine() -> None:
    """Inverse assertion: every pack file that talks about emission factors
    must mention ``ResolutionEngine`` / ``MethodProfile`` somewhere.

    This catches the failure mode where a pack quietly stops doing
    factor lookups (so it passes the forbidden-pattern scan trivially)
    while still emitting numbers — i.e. it has hard-coded the values.
    """
    offenders: List[str] = []
    for pack_dir in _iter_pack_dirs():
        # Per-pack: at least one production file must mention the engine.
        pack_has_engine_caller = False
        files_with_factor_talk: List[Path] = []
        for source_file in _iter_source_files(pack_dir):
            try:
                text = source_file.read_text(encoding="utf-8")
            except OSError:
                continue
            if _file_uses_factor_lookup(text):
                files_with_factor_talk.append(source_file)
                if _file_uses_resolution_engine(text):
                    pack_has_engine_caller = True
        if files_with_factor_talk and not pack_has_engine_caller:
            offenders.append(
                f"  - {_rel_posix(pack_dir)}: pack mentions emission "
                "factors but no file in it calls ResolutionEngine.resolve("
                ") or names a MethodProfile."
            )

    if offenders:
        pytest.fail(
            "\n\nCTO non-negotiable #6 violation: at least one policy pack\n"
            "performs factor lookups without binding to a method profile.\n\n"
            + "\n".join(offenders)
            + "\n\nSee docs/factors/POLICY_RESOLUTION_PATTERN.md.\n",
            pytrace=False,
        )


def test_allowlist_entries_exist_on_disk() -> None:
    """Every entry in the allow-list must point at a file that exists.

    Prevents the allow-list from drifting out of date silently.
    """
    missing: List[str] = []
    for rel in ALLOWED_DIRECT_CATALOG_CALLS:
        if not (REPO_ROOT / rel).is_file():
            missing.append(rel)
    assert not missing, (
        "Allow-list entries point at files that no longer exist:\n  - "
        + "\n  - ".join(missing)
        + "\n\nDelete the entry from ALLOWED_DIRECT_CATALOG_CALLS."
    )
