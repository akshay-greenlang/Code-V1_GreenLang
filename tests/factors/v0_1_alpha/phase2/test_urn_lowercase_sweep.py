# -*- coding: utf-8 -*-
"""Phase 2 / WS2 — Lowercase URN sweep over catalog seed + source registry.

Per CTO Phase 2 brief Section 2.2 (URN compliance):

    "Uppercase namespace segments in any URN kind | 0 |
     Lowercase regex sweep across factor/source/pack/methodology/
     geography/unit/activity"

This test walks every YAML/JSON data file under the v0.1 alpha catalog
seed roots, the legacy ``catalog_seed/``, the source registry, and any
other URN-bearing data file, then asserts:

  * Every ``urn:gl:...`` literal parses cleanly via
    :func:`greenlang.factors.ontology.urn.parse`.
  * Every URN segment is lowercase, EXCEPT for the factor ``<id>``
    segment, which the parser permits ``T`` and ``Z`` for ISO-8601
    timestamps.

A passing run reports the total number of URNs scanned plus the
file-level breakdown. A failing run lists every offender with the
file path and 1-based line number, then fails with a structured
message so the CI log is grep-able.

Coverage: this is the canonical Phase 2 source-of-truth check that
the catalog, source registry, and any embedded URN literal in
ontology fixtures pass through the parser. Activity URNs are
exercised by a separate WS5 test once that workstream lands.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pytest

from greenlang.factors.ontology.urn import (
    InvalidUrnError,
    parse,
)


_REPO_ROOT = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# Files & directories swept.
#
# The walker visits every JSON/YAML beneath these roots. Symlinks and
# generated artefacts under ``__pycache__`` / ``.git`` are skipped via
# ``Path.rglob`` natural exclusion.
# ---------------------------------------------------------------------------


_DATA_ROOT = _REPO_ROOT / "greenlang" / "factors" / "data"

_SWEEP_DIRS = (
    _DATA_ROOT / "catalog_seed_v0_1",
    _DATA_ROOT / "catalog_seed",
)
_SWEEP_FILES = (
    _DATA_ROOT / "source_registry.yaml",
)
_SWEEP_GLOBS_EXTRA = (
    # Method-pack and ontology YAMLs that may carry URN references.
    _DATA_ROOT / "method_packs",
    _DATA_ROOT / "taxonomies",
    _DATA_ROOT / "residual_mix",
)


# Match every URN literal in either text-style YAML or quoted JSON.
# We allow uppercase here so we can flag them — the parser does the
# strict rejection downstream.
_URN_LINE_RE = re.compile(
    r"urn:gl:[A-Za-z][A-Za-z0-9._/\-:^%]*"
)


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def _iter_data_files() -> Iterable[Path]:
    """Yield every JSON/YAML file we want to sweep."""
    seen: set[Path] = set()

    def _emit(p: Path) -> Iterable[Path]:
        rp = p.resolve()
        if rp in seen:
            return
        seen.add(rp)
        yield rp

    for d in _SWEEP_DIRS:
        if not d.exists():
            continue
        for sub in sorted(d.rglob("*")):
            if sub.is_file() and sub.suffix.lower() in (".json", ".yaml", ".yml"):
                yield from _emit(sub)
    for f in _SWEEP_FILES:
        if f.exists() and f.is_file():
            yield from _emit(f)
    for d in _SWEEP_GLOBS_EXTRA:
        if not d.exists():
            continue
        for sub in sorted(d.rglob("*")):
            if sub.is_file() and sub.suffix.lower() in (".json", ".yaml", ".yml"):
                yield from _emit(sub)


def _is_factor_id_uppercase_legal(urn: str) -> bool:
    """``True`` if the only uppercase chars sit in the factor ``<id>``
    segment and are restricted to ``T`` / ``Z`` (ISO-8601 markers).

    This mirrors the parser's ``_validate_factor_id`` rule and is
    deliberately strict: any other uppercase = fail.
    """
    if not urn.startswith("urn:gl:factor:"):
        # Only factor URNs are allowed any uppercase at all.
        return urn == urn.lower()
    # Strip the "urn:gl:factor:" prefix and the trailing ":v<n>"
    # version segment, then split into source / namespace / id.
    body = urn[len("urn:gl:factor:"):]
    last_colon = body.rfind(":")
    if last_colon < 0:
        return urn == urn.lower()
    head = body[:last_colon]  # "<source>:<namespace>:<id>"
    parts = head.split(":", 2)
    if len(parts) < 3:
        return urn == urn.lower()
    source, namespace, fid = parts[0], parts[1], parts[2]
    # Source + namespace must be strictly lowercase.
    if source != source.lower():
        return False
    if namespace != namespace.lower():
        return False
    # Factor id may contain uppercase ONLY if every uppercase char is
    # T or Z (ISO-8601 timestamp markers).
    for ch in fid:
        if ch.isupper() and ch not in ("T", "Z"):
            return False
    # Version segment is digits + leading 'v' — already lowercase by
    # construction.
    return True


def _scan_file_for_urns(path: Path) -> List[Tuple[int, str]]:
    """Return ``[(lineno, urn), ...]`` for every URN literal in ``path``.

    Lineno is 1-based. Files that fail to decode as UTF-8 are skipped
    (the suite is text-only; binary fixtures don't carry URNs).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    out: List[Tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in _URN_LINE_RE.finditer(line):
            urn = m.group(0)
            # Strip trailing punctuation that JSON / YAML might attach
            # (e.g. quote, comma, brace) — these are NOT part of a URN.
            urn = urn.rstrip("\"',}]:;)")
            if urn:
                out.append((lineno, urn))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lowercase_sweep_zero_offenders() -> None:
    """Every URN in catalog seed + source registry must be canonical.

    On success: prints a summary line ``[lowercase_sweep] scanned N URNs
    across F files; offenders=0``.
    On failure: lists each offender with file path + line number.
    """
    offenders: List[Tuple[Path, int, str, str]] = []
    total_urns = 0
    files_scanned = 0

    for path in _iter_data_files():
        files_scanned += 1
        for lineno, urn in _scan_file_for_urns(path):
            total_urns += 1
            # Rule 1: structural parse. The parser is the canonical
            # truth — it rejects anything not matching the grammar.
            try:
                parse(urn)
            except InvalidUrnError as exc:
                offenders.append(
                    (path, lineno, urn, f"parse() raised: {exc}")
                )
                continue
            # Rule 2: the lowercase invariant. The parser already
            # enforces this for every kind, but we re-check it here
            # so the error message names the offending file:line and
            # we never silently rely on parser-rule drift.
            if not _is_factor_id_uppercase_legal(urn):
                offenders.append(
                    (
                        path,
                        lineno,
                        urn,
                        "uppercase segment outside factor-id T/Z exception",
                    )
                )

    # Always print a summary for CI logs.
    summary = (
        f"[lowercase_sweep] scanned {total_urns} URNs across "
        f"{files_scanned} files; offenders={len(offenders)}"
    )
    print(summary)

    if offenders:
        rendered = "\n".join(
            f"  {p}:{lineno}: {urn}  -- {reason}"
            for (p, lineno, urn, reason) in offenders
        )
        pytest.fail(
            f"Found {len(offenders)} URN(s) violating the Phase 2 "
            f"lowercase invariant:\n{rendered}\n"
            f"({summary})"
        )


def test_sweep_finds_at_least_one_urn() -> None:
    """Sanity check — the sweep MUST find URNs to be meaningful.

    If this fails the seed-data tree was renamed/moved without updating
    ``_SWEEP_DIRS`` / ``_SWEEP_FILES`` and the lowercase sweep would be
    silently no-op. We assert at least 100 URNs to make accidental
    bypass loud.
    """
    total = 0
    for path in _iter_data_files():
        total += len(_scan_file_for_urns(path))
    assert total >= 100, (
        f"lowercase sweep found only {total} URNs; expected >=100. "
        f"Check that _SWEEP_DIRS / _SWEEP_FILES still point at the "
        f"v0.1 alpha catalog seed."
    )
