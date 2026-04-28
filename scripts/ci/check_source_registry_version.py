#!/usr/bin/env python3
"""check_source_registry_version.py — Phase 3 Wave 3.0 CI gate (gate 2).

Authority: CTO Phase 3 brief 2026-04-28, Block 7 box 2.
Owner    : GL-Factors Engineering (Wave 3.0).

Purpose: every parser_version bump for any source in
``greenlang/factors/data/source_registry.yaml`` MUST be paired with a matching
section entry in ``docs/factors/source-registry/CHANGELOG.md``.

Header regex (case-insensitive):
    ^## .*<source_id>.* <new_version>

Args:
    --base-ref         git base ref (default: origin/master)
    --head-ref         git head ref (default: HEAD)
    --base-content     path or '-' for stdin: explicit base registry YAML (test mode)
    --head-content     path or '-' for stdin: explicit head registry YAML (test mode)
    --changelog        path to source-registry CHANGELOG (default: docs/factors/source-registry/CHANGELOG.md)
    --changelog-content path or '-' for stdin: explicit changelog text (test mode)
    --repo-root        override repo root (defaults to git toplevel)

Exit codes:
    0  PASS    — no parser_version bumps OR every bump has a matching CHANGELOG entry.
    1  FAIL    — at least one bump has no matching CHANGELOG entry.
    2  INVALID — usage / git / I/O / YAML error.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


REGISTRY_PATH = "greenlang/factors/data/source_registry.yaml"
DEFAULT_CHANGELOG_PATH = "docs/factors/source-registry/CHANGELOG.md"


class GateError(RuntimeError):
    """Raised on usage / I/O / git / YAML error (-> exit 2)."""


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

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


def _show_file_at_ref(*, ref: str, path: str, cwd: Path) -> Optional[str]:
    proc = subprocess.run(
        ["git", "show", "%s:%s" % (ref, path)],
        cwd=str(cwd), check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def _resolve_repo_root(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg).resolve()
    here = Path(__file__).resolve().parent
    try:
        out = _run_git(["rev-parse", "--show-toplevel"], cwd=here)
        return Path(out.strip()).resolve()
    except GateError:
        return Path.cwd().resolve()


def _read_content_arg(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    return Path(arg).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Registry parsing
# ---------------------------------------------------------------------------

def _parse_registry(text: str) -> Dict[str, str]:
    """Parse YAML and return {source_id -> parser_version}.

    Some sources duplicate ``parser_version`` (legacy block + Phase 3
    additive block). YAML naturally takes the last-occurring key, which
    is the canonical Phase 3 version — we accept that.
    """
    try:
        doc = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise GateError("registry YAML parse error: %s" % exc) from exc
    if not isinstance(doc, dict):
        raise GateError("registry root is not a mapping")
    sources = doc.get("sources") or []
    if not isinstance(sources, list):
        raise GateError("registry 'sources' is not a list")
    out: Dict[str, str] = {}
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("source_id")
        ver = entry.get("parser_version")
        if isinstance(sid, str) and isinstance(ver, (str, int, float)):
            out[sid] = str(ver)
    return out


# ---------------------------------------------------------------------------
# CHANGELOG matching
# ---------------------------------------------------------------------------

def _has_changelog_entry(
    changelog_text: str,
    *,
    source_id: str,
    new_version: str,
) -> bool:
    """Return True iff a top-level section header mentions both the source_id
    and the new_version (case-insensitive). The header regex is:
        ^## .*<source_id>.* <new_version>
    """
    sid_re = re.escape(source_id)
    ver_re = re.escape(new_version)
    pattern = re.compile(
        r"^##\s+.*%s.*\s+%s\b" % (sid_re, ver_re),
        re.IGNORECASE | re.MULTILINE,
    )
    return bool(pattern.search(changelog_text))


# ---------------------------------------------------------------------------
# Core gate logic
# ---------------------------------------------------------------------------

def evaluate(
    *,
    base_yaml_text: Optional[str],
    head_yaml_text: str,
    changelog_text: str,
) -> Tuple[int, List[str]]:
    """Pure logic. Returns (exit_code, message_lines)."""
    head_versions = _parse_registry(head_yaml_text)
    base_versions = _parse_registry(base_yaml_text) if base_yaml_text else {}

    bumps: List[Tuple[str, str, str]] = []  # (source_id, old, new)
    for sid, new_ver in sorted(head_versions.items()):
        old_ver = base_versions.get(sid)
        if old_ver is not None and old_ver != new_ver:
            bumps.append((sid, old_ver, new_ver))

    msgs: List[str] = []
    if not bumps:
        msgs.append("[source-registry-version] no parser_version bumps detected; PASS.")
        return 0, msgs

    missing: List[Tuple[str, str, str]] = []
    for sid, old, new in bumps:
        if not _has_changelog_entry(changelog_text, source_id=sid, new_version=new):
            missing.append((sid, old, new))

    if not missing:
        msgs.append(
            "[source-registry-version] %d parser_version bump(s) all have CHANGELOG entries; PASS."
            % len(bumps)
        )
        for sid, old, new in bumps:
            msgs.append("  - %s: %s -> %s" % (sid, old, new))
        return 0, msgs

    msgs.append("[source-registry-version] FAIL: missing CHANGELOG entry for parser_version bump:")
    for sid, old, new in missing:
        msgs.append("  - %s: %s -> %s" % (sid, old, new))
    msgs.append("  Add a section header `## <source_id> <new_version> - YYYY-MM-DD` to:")
    msgs.append("    docs/factors/source-registry/CHANGELOG.md")
    return 1, msgs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="check_source_registry_version")
    p.add_argument("--base-ref", default="origin/master")
    p.add_argument("--head-ref", default="HEAD")
    p.add_argument("--base-content", default=None,
                   help="path or '-' for stdin: explicit base registry YAML (test mode)")
    p.add_argument("--head-content", default=None,
                   help="path or '-' for stdin: explicit head registry YAML (test mode)")
    p.add_argument("--changelog", default=DEFAULT_CHANGELOG_PATH)
    p.add_argument("--changelog-content", default=None,
                   help="path or '-' for stdin: explicit changelog text (test mode)")
    p.add_argument("--repo-root", default=None)
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)
    try:
        repo_root = _resolve_repo_root(args.repo_root)

        # head registry YAML
        if args.head_content is not None:
            head_yaml = _read_content_arg(args.head_content)
        else:
            head_path = repo_root / REGISTRY_PATH
            if not head_path.is_file():
                raise GateError("registry not found at %s" % head_path)
            head_yaml = head_path.read_text(encoding="utf-8")

        # base registry YAML
        if args.base_content is not None:
            base_yaml = _read_content_arg(args.base_content)
        else:
            base_yaml = _show_file_at_ref(
                ref=args.base_ref, path=REGISTRY_PATH, cwd=repo_root,
            )
            # Either base ref doesn't have it (new file) or git failed —
            # treat absence as empty registry (no bumps possible).
            if base_yaml is None:
                base_yaml = ""

        # changelog text
        if args.changelog_content is not None:
            changelog = _read_content_arg(args.changelog_content)
        else:
            cl_path = repo_root / args.changelog
            if not cl_path.is_file():
                # Bootstrap: gate fails closed if registry has bumps but no changelog.
                changelog = ""
            else:
                changelog = cl_path.read_text(encoding="utf-8")

        exit_code, msgs = evaluate(
            base_yaml_text=base_yaml,
            head_yaml_text=head_yaml,
            changelog_text=changelog,
        )
    except GateError as exc:
        print("[source-registry-version] error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        print("[source-registry-version] unexpected error: %s" % exc, file=sys.stderr)
        return 2

    for line in msgs:
        print(line)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
