#!/usr/bin/env python3
"""check_parser_snapshot_drift.py — Phase 3 Wave 3.0 CI gate (gate 1).

Authority: CTO Phase 3 brief 2026-04-28, Block 7 box 1.
Owner    : GL-Factors Engineering (Wave 3.0).

Purpose: every change to a parser file under
``greenlang/factors/ingestion/parsers/**.py`` MUST be paired with a
corresponding snapshot regeneration under
``tests/factors/v0_1_alpha/phase3/parser_snapshots/**.golden.json``.

The gate diffs the working tree against ``--base-ref`` and compares the
two file lists. If a parser file changed and no snapshot json was
added/modified in the same diff, the gate fails.

Override: a parser may carry the line marker
``parser-snapshot-drift: intentional, regenerated via UPDATE_PARSER_SNAPSHOT=1``
inside its source. When present, the gate allows the change.

Args (all optional, with sensible defaults for CI):
    --base-ref       git base ref (default: origin/master)
    --head-ref       git head ref (default: HEAD)
    --base-content   path or '-' for stdin: explicit base file list (test mode)
    --head-content   path or '-' for stdin: explicit head file list (test mode)
    --repo-root      override repo root (defaults to git toplevel)

Exit codes:
    0  PASS    — every parser change has a matching snapshot change.
    1  FAIL    — at least one parser change is missing its snapshot.
    2  INVALID — usage / git / I/O error.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


PARSER_DIR_PREFIX = "greenlang/factors/ingestion/parsers/"
SNAPSHOT_DIR_PREFIX = "tests/factors/v0_1_alpha/phase3/parser_snapshots/"
OVERRIDE_MARKER = (
    "parser-snapshot-drift: intentional, regenerated via UPDATE_PARSER_SNAPSHOT=1"
)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

class GateError(RuntimeError):
    """Raised on any usage / I/O / git failure (-> exit 2)."""


def _run_git(args: List[str], *, cwd: Path) -> str:
    cmd = ["git"] + args
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), check=False, capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        raise GateError("git executable not found: %s" % exc) from exc
    if proc.returncode != 0:
        raise GateError(
            "git failed (exit=%d): %s\nstderr: %s"
            % (proc.returncode, " ".join(shlex.quote(a) for a in cmd), proc.stderr.strip())
        )
    return proc.stdout


def _list_changed_files(base: str, head: str, *, cwd: Path) -> List[str]:
    out = _run_git(["diff", "--name-only", "%s..%s" % (base, head)], cwd=cwd)
    return [line.strip() for line in out.splitlines() if line.strip()]


def _resolve_repo_root(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg).resolve()
    here = Path(__file__).resolve().parent
    try:
        out = _run_git(["rev-parse", "--show-toplevel"], cwd=here)
        return Path(out.strip()).resolve()
    except GateError:
        return Path.cwd().resolve()


# ---------------------------------------------------------------------------
# Test-mode: read explicit file lists from --base-content / --head-content
# ---------------------------------------------------------------------------

def _read_content_arg(arg: str) -> List[str]:
    """Read a newline-separated file list from a path or stdin ('-')."""
    if arg == "-":
        text = sys.stdin.read()
    else:
        text = Path(arg).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Override-marker check
# ---------------------------------------------------------------------------

def _parser_has_override(repo_root: Path, parser_path: str) -> bool:
    """Return True if the parser source carries the override marker line."""
    full = repo_root / parser_path
    if not full.is_file():
        return False
    try:
        text = full.read_text(encoding="utf-8")
    except OSError:
        return False
    return OVERRIDE_MARKER in text


# ---------------------------------------------------------------------------
# Core gate logic
# ---------------------------------------------------------------------------

def _classify(changed_files: Iterable[str]) -> Tuple[List[str], List[str]]:
    parsers: List[str] = []
    snapshots: List[str] = []
    for f in changed_files:
        norm = f.replace("\\", "/")
        if norm.startswith(PARSER_DIR_PREFIX) and norm.endswith(".py"):
            # exclude __init__.py and adapter helpers? Spec says "any .py".
            parsers.append(norm)
        elif norm.startswith(SNAPSHOT_DIR_PREFIX) and norm.endswith(".json"):
            snapshots.append(norm)
    return parsers, snapshots


def evaluate(
    *,
    changed_files: List[str],
    repo_root: Path,
) -> Tuple[int, List[str]]:
    """Pure logic. Returns (exit_code, message_lines)."""
    parsers, snapshots = _classify(changed_files)

    msgs: List[str] = []
    if not parsers:
        msgs.append("[parser-snapshot-drift] no parser changes detected; PASS.")
        return 0, msgs

    if snapshots:
        msgs.append(
            "[parser-snapshot-drift] parser change(s) detected AND snapshot change(s) present; PASS."
        )
        msgs.append("  parsers:   " + ", ".join(parsers))
        msgs.append("  snapshots: " + ", ".join(snapshots))
        return 0, msgs

    # No snapshots changed — check the override marker on every parser
    parsers_without_override = [
        p for p in parsers if not _parser_has_override(repo_root, p)
    ]
    if not parsers_without_override:
        msgs.append(
            "[parser-snapshot-drift] all parser changes carry the override marker; PASS."
        )
        msgs.append("  parsers (with override): " + ", ".join(parsers))
        return 0, msgs

    msgs.append(
        "[parser-snapshot-drift] FAIL: parser file(s) changed without "
        "matching snapshot change AND without the override marker:"
    )
    for p in parsers_without_override:
        msgs.append("  - %s" % p)
    msgs.append(
        "  Either regenerate the affected golden(s) under "
        "%s OR add the line:" % SNAPSHOT_DIR_PREFIX
    )
    msgs.append('  "%s"' % OVERRIDE_MARKER)
    return 1, msgs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="check_parser_snapshot_drift")
    p.add_argument("--base-ref", default="origin/master")
    p.add_argument("--head-ref", default="HEAD")
    p.add_argument("--base-content", default=None,
                   help="path or '-' for stdin: explicit base file list (test mode)")
    p.add_argument("--head-content", default=None,
                   help="path or '-' for stdin: explicit head file list (test mode)")
    p.add_argument("--repo-root", default=None)
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)
    try:
        repo_root = _resolve_repo_root(args.repo_root)

        if args.head_content is not None:
            # Test mode: explicit changed-file list.
            head_files = _read_content_arg(args.head_content)
            base_files: List[str] = (
                _read_content_arg(args.base_content)
                if args.base_content is not None
                else []
            )
            # The "changed" set is symmetric difference + intersection of names
            # whose contents differ. In test mode we just take head_files as
            # the changed set: it's simpler and tests pass an explicit list of
            # changed paths.
            changed = head_files
        else:
            changed = _list_changed_files(args.base_ref, args.head_ref, cwd=repo_root)

        exit_code, msgs = evaluate(changed_files=changed, repo_root=repo_root)
    except GateError as exc:
        print("[parser-snapshot-drift] error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        print("[parser-snapshot-drift] unexpected error: %s" % exc, file=sys.stderr)
        return 2

    for line in msgs:
        print(line)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
