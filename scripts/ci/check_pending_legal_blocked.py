#!/usr/bin/env python3
"""check_pending_legal_blocked.py — Phase 3 Wave 3.0 CI gate (gate 4).

Authority: CTO Phase 3 brief 2026-04-28, Block 7 box 4.
Owner    : GL-Factors Engineering (Wave 3.0).

Purpose: scan all GitHub Actions workflows under ``.github/workflows/*.yml``
plus shell/batch wrappers under ``scripts/**/*.sh|*.bat`` for ``gl factors
ingest`` invocations. Fail the gate when any invocation:

    - Targets ``--env production`` (or sets ``GL_FACTORS_ENV=production``); AND
    - Names a ``--source <id>`` whose registry entry has either:
        * ``status`` in {pending_legal_review, blocked}; OR
        * ``release_milestone`` later than ``v0.1``.

Args:
    --workflows-dir       override workflows dir (default: .github/workflows)
    --scripts-dir         override scripts dir (default: scripts)
    --registry            override registry path (default: greenlang/factors/data/source_registry.yaml)
    --registry-content    explicit registry YAML for testing
    --workflows-content   explicit text containing fake workflow snippets (test mode)
    --repo-root           override repo root

Exit codes:
    0  PASS    — no offending invocations.
    1  FAIL    — at least one offending invocation found.
    2  INVALID — usage / I/O / YAML error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


REGISTRY_PATH = "greenlang/factors/data/source_registry.yaml"
WORKFLOWS_DIR = ".github/workflows"
SCRIPTS_DIR = "scripts"

BLOCKING_STATUSES = frozenset({"pending_legal_review", "blocked"})

# Match ``gl factors ingest`` and the trailing arg list up to a newline.
INGEST_CMD_RE = re.compile(
    r"gl\s+factors\s+ingest[^\n]*", re.IGNORECASE
)
ENV_FLAG_RE = re.compile(r"--env[\s=]+(\S+)")
ENV_VAR_RE = re.compile(r"GL_FACTORS_ENV\s*[:=]\s*['\"]?production['\"]?", re.IGNORECASE)
SOURCE_FLAG_RE = re.compile(r"--source[\s=]+([A-Za-z0-9_\-]+)")


class GateError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def _parse_release_milestone(value: object) -> Optional[Tuple[int, int]]:
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if not s.startswith("v"):
        return None
    rest = s[1:]
    parts = rest.split(".")
    if len(parts) < 2:
        return None
    try:
        major = int(parts[0])
        # strip any -rc1 suffix from minor
        minor_raw = parts[1].split("-")[0]
        minor = int(minor_raw)
    except (TypeError, ValueError):
        return None
    return (major, minor)


def _load_registry(text: str) -> Dict[str, Dict[str, object]]:
    try:
        doc = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise GateError("registry YAML parse error: %s" % exc) from exc
    sources = doc.get("sources") if isinstance(doc, dict) else None
    if not isinstance(sources, list):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for entry in sources:
        if isinstance(entry, dict):
            sid = entry.get("source_id")
            if isinstance(sid, str):
                out[sid] = entry
    return out


# ---------------------------------------------------------------------------
# Invocation scanning
# ---------------------------------------------------------------------------

def _find_invocations(text: str) -> List[Tuple[str, bool, Optional[str]]]:
    """Return list of (raw_command, is_production_env, source_id_or_None)."""
    out: List[Tuple[str, bool, Optional[str]]] = []
    has_envvar = bool(ENV_VAR_RE.search(text))
    for m in INGEST_CMD_RE.finditer(text):
        cmd = m.group(0)
        env_match = ENV_FLAG_RE.search(cmd)
        is_prod = (
            (env_match is not None and env_match.group(1).strip("'\"") == "production")
            or has_envvar
        )
        src_match = SOURCE_FLAG_RE.search(cmd)
        sid = src_match.group(1) if src_match else None
        out.append((cmd, is_prod, sid))
    return out


def _scan_files(paths: Iterable[Path]) -> List[Tuple[Path, str, bool, Optional[str]]]:
    out: List[Tuple[Path, str, bool, Optional[str]]] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for cmd, is_prod, sid in _find_invocations(text):
            out.append((p, cmd, is_prod, sid))
    return out


def _is_source_blocked_for_prod(
    sid: str,
    registry: Dict[str, Dict[str, object]],
) -> Tuple[bool, str]:
    """Return (blocked?, reason)."""
    entry = registry.get(sid)
    if entry is None:
        return True, "source_id %s not found in registry" % sid
    status_raw = entry.get("status")
    status = str(status_raw or "").strip().lower()
    if status in BLOCKING_STATUSES:
        return True, "status=%s" % status
    milestone = _parse_release_milestone(entry.get("release_milestone"))
    if milestone is None:
        return True, "release_milestone missing/invalid"
    if milestone > (0, 1):
        return True, "release_milestone=%s > v0.1" % entry.get("release_milestone")
    return False, ""


# ---------------------------------------------------------------------------
# Core gate
# ---------------------------------------------------------------------------

def _enumerate_target_files(
    repo_root: Path,
    workflows_dir: Path,
    scripts_dir: Path,
) -> List[Path]:
    out: List[Path] = []
    if workflows_dir.is_dir():
        out.extend(sorted(workflows_dir.glob("*.yml")))
        out.extend(sorted(workflows_dir.glob("*.yaml")))
    if scripts_dir.is_dir():
        out.extend(sorted(scripts_dir.rglob("*.sh")))
        out.extend(sorted(scripts_dir.rglob("*.bat")))
    return out


def evaluate(
    *,
    files: List[Path],
    registry: Dict[str, Dict[str, object]],
    extra_text_blobs: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[int, List[str]]:
    """Pure logic. Returns (exit_code, message_lines).

    extra_text_blobs: list of (label, text) for test mode injection.
    """
    invocations = _scan_files(files)
    if extra_text_blobs:
        for label, text in extra_text_blobs:
            for cmd, is_prod, sid in _find_invocations(text):
                invocations.append((Path(label), cmd, is_prod, sid))

    msgs: List[str] = []
    if not invocations:
        msgs.append("[pending-legal-blocked] no `gl factors ingest` invocations found; PASS.")
        return 0, msgs

    failures: List[Tuple[Path, str, str, str]] = []  # (file, cmd, sid, reason)
    for file_path, cmd, is_prod, sid in invocations:
        if not is_prod:
            continue
        if sid is None:
            failures.append((file_path, cmd, "<missing>", "production invocation without --source"))
            continue
        blocked, reason = _is_source_blocked_for_prod(sid, registry)
        if blocked:
            failures.append((file_path, cmd, sid, reason))

    if not failures:
        msgs.append(
            "[pending-legal-blocked] %d invocation(s) inspected; PASS (no production runs against unapproved sources)."
            % len(invocations)
        )
        return 0, msgs

    msgs.append("[pending-legal-blocked] FAIL: production ingest invocations target unapproved sources:")
    for file_path, cmd, sid, reason in failures:
        msgs.append("  - %s" % file_path)
        msgs.append("      cmd:    %s" % cmd.strip())
        msgs.append("      source: %s" % sid)
        msgs.append("      reason: %s" % reason)
    return 1, msgs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="check_pending_legal_blocked")
    p.add_argument("--workflows-dir", default=None)
    p.add_argument("--scripts-dir", default=None)
    p.add_argument("--registry", default=None)
    p.add_argument("--registry-content", default=None,
                   help="path or '-' for stdin: explicit registry YAML (test mode)")
    p.add_argument("--workflows-content", default=None,
                   help="path or '-' for stdin: explicit workflow text (test mode)")
    p.add_argument("--repo-root", default=None)
    return p


def _resolve_repo_root(arg: Optional[str]) -> Path:
    if arg:
        return Path(arg).resolve()
    here = Path(__file__).resolve().parent
    # Walk up to find a directory containing .git.
    cur = here
    for _ in range(8):
        if (cur / ".git").exists():
            return cur.resolve()
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path.cwd().resolve()


def _read_content_arg(arg: str) -> str:
    if arg == "-":
        return sys.stdin.read()
    return Path(arg).read_text(encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)
    try:
        repo_root = _resolve_repo_root(args.repo_root)

        # Registry
        if args.registry_content is not None:
            reg_text = _read_content_arg(args.registry_content)
        else:
            reg_path = (
                Path(args.registry).resolve()
                if args.registry
                else (repo_root / REGISTRY_PATH)
            )
            if not reg_path.is_file():
                raise GateError("registry not found at %s" % reg_path)
            reg_text = reg_path.read_text(encoding="utf-8")
        registry = _load_registry(reg_text)

        # Files to scan
        wf_dir = Path(args.workflows_dir).resolve() if args.workflows_dir else (repo_root / WORKFLOWS_DIR)
        sc_dir = Path(args.scripts_dir).resolve() if args.scripts_dir else (repo_root / SCRIPTS_DIR)
        files = _enumerate_target_files(repo_root, wf_dir, sc_dir)

        extra: List[Tuple[str, str]] = []
        if args.workflows_content is not None:
            extra.append(("<workflows-content>", _read_content_arg(args.workflows_content)))

        exit_code, msgs = evaluate(
            files=files, registry=registry, extra_text_blobs=extra,
        )
    except GateError as exc:
        print("[pending-legal-blocked] error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        print("[pending-legal-blocked] unexpected error: %s" % exc, file=sys.stderr)
        return 2

    for line in msgs:
        print(line)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
