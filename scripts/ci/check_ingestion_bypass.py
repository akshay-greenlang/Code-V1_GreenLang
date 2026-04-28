#!/usr/bin/env python3
"""check_ingestion_bypass.py — Phase 3 Wave 3.0 CI gate (gate 5).

Authority: CTO Phase 3 brief 2026-04-28, Block 7 box 5.
Owner    : GL-Factors Engineering (Wave 3.0).

Purpose: scan ``greenlang/**/*.py`` and ``scripts/**/*.py`` for SQL strings
that INSERT into the canonical Phase 2 ``factor`` table OUTSIDE the
whitelisted canonical repository file. Such writes would bypass the
Phase 2 7-gate publish orchestrator and let unverified factors land in
production.

The Phase 2 audit (PHASE_2_EXIT_CHECKLIST.md Block 5) confirmed there are
exactly TWO canonical writes:
    * ``INSERT INTO alpha_factors_v0_1 (`` (sqlite alpha repository)
    * ``INSERT INTO factors_v0_1.factor (`` (postgres alpha repository)

Both live inside ``greenlang/factors/repositories/alpha_v0_1_repository.py``
which is the sole whitelisted writer.

Patterns flagged (case-insensitive). All MUST end at a word-boundary so we
do NOT catch sibling tables (factor_aliases, factor_pack, factors_batch_jobs,
factors_catalog, factor_versions, factors_review_votes, factor_lineage,
factors_v0_1.activity / .geography / .unit / .methodology, etc.):

    * ``INSERT INTO factors_v0_1.factor`` followed by space/(/EOL
      (NOT followed by another _identifier_ char like ``_aliases``).
    * ``INSERT INTO alpha_factors_v0_1`` followed by space/(/EOL
      (NOT followed by ``_aliases``, ``_review_votes``, etc.).
    * ``executemany(...alpha_factors_v0_1...)`` and
      ``executemany(...factors_v0_1.factor[^_]...)``.

Whitelist:
    * greenlang/factors/repositories/alpha_v0_1_repository.py
    * Any path passed to --whitelist-extra (test mode).
    * The script itself (it contains the patterns as string literals).

Exit codes:
    0 PASS    — no bypass writes found.
    1 FAIL    — at least one bypass write found.
    2 INVALID — usage / I/O error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


WHITELIST = {
    "greenlang/factors/repositories/alpha_v0_1_repository.py",
    # The gate script itself contains the patterns as string literals.
    "scripts/ci/check_ingestion_bypass.py",
    "tests/ci/test_check_ingestion_bypass.py",
}

# Match INSERT INTO factors_v0_1.factor followed by NON-word-char or "(" .
# The trailing ``(?!\w)`` prevents matches against factor_aliases /
# factor_versions / factors_v0_1.activity / .geography / etc.
INSERT_FACTORS_V01_RE = re.compile(
    r"INSERT\s+INTO\s+[\"'`]?factors_v0_1\.factor(?!\w)",
    re.IGNORECASE,
)
# Same idea for the sqlite alpha table name.
INSERT_ALPHA_RE = re.compile(
    r"INSERT\s+INTO\s+[\"'`]?alpha_factors_v0_1(?!\w)",
    re.IGNORECASE,
)
# executemany on a string mentioning either canonical table (with the
# same word-boundary guard) — typical bulk-insert smell.
EXECUTEMANY_FACTORS_V01_RE = re.compile(
    r"executemany\s*\([^)]*factors_v0_1\.factor(?!\w)[^)]*\)",
    re.IGNORECASE | re.DOTALL,
)
EXECUTEMANY_ALPHA_RE = re.compile(
    r"executemany\s*\([^)]*alpha_factors_v0_1(?!\w)[^)]*\)",
    re.IGNORECASE | re.DOTALL,
)


class GateError(RuntimeError):
    pass


def _resolve_root(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg).resolve()
    here = Path(__file__).resolve().parent
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(here), check=False, capture_output=True, text=True,
    )
    if proc.returncode == 0:
        return Path(proc.stdout.strip()).resolve()
    return Path.cwd().resolve()


def _enumerate_python_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for sub in ("greenlang", "scripts"):
        p = root / sub
        if not p.is_dir():
            continue
        out.extend(sorted(p.rglob("*.py")))
    return out


def _is_whitelisted(rel_path: str, whitelist: Iterable[str]) -> bool:
    norm = rel_path.replace("\\", "/")
    for w in whitelist:
        if norm == w.replace("\\", "/"):
            return True
    return False


def _scan_text(text: str) -> List[Tuple[str, str]]:
    """Return list of (pattern_label, snippet) hits."""
    hits: List[Tuple[str, str]] = []

    def _record(label: str, m: re.Match) -> None:
        snippet = text[max(0, m.start() - 20): min(len(text), m.end() + 30)]
        snippet = snippet.replace("\n", " ").strip()
        hits.append((label, snippet))

    for m in INSERT_FACTORS_V01_RE.finditer(text):
        _record("INSERT INTO factors_v0_1.factor", m)
    for m in INSERT_ALPHA_RE.finditer(text):
        _record("INSERT INTO alpha_factors_v0_1", m)
    for m in EXECUTEMANY_FACTORS_V01_RE.finditer(text):
        _record("executemany(...factors_v0_1.factor...)", m)
    for m in EXECUTEMANY_ALPHA_RE.finditer(text):
        _record("executemany(...alpha_factors_v0_1...)", m)

    return hits


def evaluate(
    *,
    files: List[Path],
    repo_root: Path,
    whitelist: Iterable[str],
) -> Tuple[int, List[str]]:
    msgs: List[str] = []
    failures: List[Tuple[str, str, str]] = []

    for fp in files:
        try:
            rel = str(fp.resolve().relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            rel = str(fp).replace("\\", "/")
        if _is_whitelisted(rel, whitelist):
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for label, snippet in _scan_text(text):
            failures.append((rel, label, snippet))

    if not failures:
        msgs.append(
            "[ingestion-bypass] %d file(s) scanned; PASS (no factor-table writes outside whitelist)."
            % len(files)
        )
        return 0, msgs

    msgs.append(
        "[ingestion-bypass] FAIL: factor-table writes found OUTSIDE the canonical repository:"
    )
    for rel, label, snippet in failures:
        msgs.append("  - %s" % rel)
        msgs.append("      pattern: %s" % label)
        msgs.append("      snippet: %s" % snippet)
    msgs.append(
        "  Move all factor-table writes into "
        "greenlang/factors/repositories/alpha_v0_1_repository.py — the only "
        "module gated by the Phase 2 7-gate publish orchestrator."
    )
    return 1, msgs


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="check_ingestion_bypass")
    p.add_argument("--root", default=None)
    p.add_argument("--whitelist-extra", action="append", default=[])
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)
    try:
        root = _resolve_root(args.root)
        files = _enumerate_python_files(root)
        whitelist = set(WHITELIST) | set(args.whitelist_extra or [])
        exit_code, msgs = evaluate(files=files, repo_root=root, whitelist=whitelist)
    except GateError as exc:
        print("[ingestion-bypass] error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        print("[ingestion-bypass] unexpected error: %s" % exc, file=sys.stderr)
        return 2

    for line in msgs:
        print(line)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
