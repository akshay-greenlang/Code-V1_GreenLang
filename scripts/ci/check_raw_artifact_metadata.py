#!/usr/bin/env python3
"""check_raw_artifact_metadata.py — Phase 3 Wave 3.0 CI gate (gate 3).

Authority: CTO Phase 3 brief 2026-04-28, Block 7 box 3.
Owner    : GL-Factors Engineering (Wave 3.0).

Purpose: every factor record ADDED in a diff to a catalog seed JSON under
``greenlang/factors/data/catalog_seed/**/*.json`` MUST carry:
    - extraction.raw_artifact_uri  (non-empty string)
    - extraction.raw_artifact_sha256  (lowercase 64-hex)

Existing records that lack these fields are NOT failed. Instead, the script
writes a backfill TODO list to:
    docs/factors/source-registry/PHASE_3_BACKFILL_TODO.md

Args:
    --base-ref         git base ref (default: origin/master)
    --head-ref         git head ref (default: HEAD)
    --base-content     path or '-' for stdin: NEWLINE-separated list of catalog JSON paths to treat as base (test mode)
    --head-content     path or '-' for stdin: NEWLINE-separated list of catalog JSON paths to treat as head (test mode)
    --base-dir         override base content root (test mode)
    --head-dir         override head content root (test mode); defaults to repo root
    --backfill-out     path for the backfill TODO output (default: docs/factors/source-registry/PHASE_3_BACKFILL_TODO.md)
    --repo-root        override repo root (defaults to git toplevel)

Exit codes:
    0  PASS    — every NEWLY-added factor carries extraction metadata.
    1  FAIL    — at least one NEWLY-added factor is missing extraction metadata.
    2  INVALID — usage / git / I/O error.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


CATALOG_PREFIX = "greenlang/factors/data/catalog_seed/"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_BACKFILL_OUT = "docs/factors/source-registry/PHASE_3_BACKFILL_TODO.md"


class GateError(RuntimeError):
    pass


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


def _list_changed_files(base: str, head: str, *, cwd: Path) -> List[str]:
    out = _run_git(["diff", "--name-only", "%s..%s" % (base, head)], cwd=cwd)
    return [line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()]


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
# Record helpers
# ---------------------------------------------------------------------------

def _record_key(rec: Dict[str, Any]) -> str:
    """Stable identity for a factor record across base/head."""
    for k in ("urn", "factor_id", "factor_id_alias"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Fallback: serialize the record so an identical-bodied record is never
    # double-counted.
    return json.dumps(rec, sort_keys=True)[:128]


def _factors_from_doc(doc: Any) -> List[Dict[str, Any]]:
    """Catalog seed files store factors under ``factors`` (list)."""
    if not isinstance(doc, dict):
        return []
    facs = doc.get("factors")
    if isinstance(facs, list):
        return [f for f in facs if isinstance(f, dict)]
    return []


def _has_valid_extraction(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (valid, reason). reason is empty when valid."""
    extraction = rec.get("extraction")
    if not isinstance(extraction, dict):
        return False, "missing 'extraction' object"
    uri = extraction.get("raw_artifact_uri")
    if not isinstance(uri, str) or not uri.strip():
        return False, "missing 'extraction.raw_artifact_uri'"
    sha = extraction.get("raw_artifact_sha256")
    if not isinstance(sha, str):
        return False, "missing 'extraction.raw_artifact_sha256'"
    if not SHA256_RE.match(sha):
        return False, "extraction.raw_artifact_sha256 not lowercase 64-hex"
    return True, ""


def _load_catalog_text(text: str) -> List[Dict[str, Any]]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GateError("catalog JSON parse error: %s" % exc) from exc
    return _factors_from_doc(doc)


# ---------------------------------------------------------------------------
# Core gate
# ---------------------------------------------------------------------------

def _classify_paths(changed: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in changed:
        norm = p.replace("\\", "/")
        if norm.startswith(CATALOG_PREFIX) and norm.endswith(".json"):
            out.append(norm)
    return out


def evaluate_files(
    *,
    catalog_paths: List[str],
    repo_root: Path,
    base_ref: Optional[str],
    head_dir: Optional[Path] = None,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """For each changed catalog file, classify each record as either:
       - newly-added missing metadata  -> failures (path, key, reason)
       - pre-existing missing metadata -> backfill (path, key, reason)
    Returns (failures, backfill).
    """
    failures: List[Tuple[str, str, str]] = []
    backfill: List[Tuple[str, str, str]] = []

    head_root = head_dir if head_dir is not None else repo_root

    for path in catalog_paths:
        # base
        base_keys: Set[str] = set()
        if base_ref:
            base_text = _show_file_at_ref(ref=base_ref, path=path, cwd=repo_root)
            if base_text is not None:
                try:
                    base_facs = _load_catalog_text(base_text)
                except GateError:
                    base_facs = []
                base_keys = {_record_key(r) for r in base_facs}

        # head
        head_path = head_root / path
        if not head_path.is_file():
            # File deleted at HEAD -> nothing to scan
            continue
        try:
            head_facs = _load_catalog_text(head_path.read_text(encoding="utf-8"))
        except GateError as exc:
            raise GateError("could not parse %s: %s" % (path, exc)) from exc

        for rec in head_facs:
            key = _record_key(rec)
            valid, reason = _has_valid_extraction(rec)
            if valid:
                continue
            if key in base_keys:
                backfill.append((path, key, reason))
            else:
                failures.append((path, key, reason))

    return failures, backfill


def _write_backfill_todo(
    out_path: Path,
    backfill: List[Tuple[str, str, str]],
) -> None:
    if not backfill:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Phase 3 — raw_artifact metadata backfill TODO")
    lines.append("")
    lines.append("> Authority: CTO Phase 3 brief 2026-04-28, Block 7 box 3.")
    lines.append(">")
    lines.append("> Pre-existing factor records still missing `extraction.raw_artifact_uri`")
    lines.append("> and/or `extraction.raw_artifact_sha256`. New records MUST carry both;")
    lines.append("> these grandfathered ones are tracked here for backfill via the next")
    lines.append("> ingestion run on each source.")
    lines.append("")
    lines.append("| Catalog file | Record key | Missing |")
    lines.append("|---|---|---|")
    for path, key, reason in backfill:
        # escape pipes in key
        safe_key = key.replace("|", "\\|")
        lines.append("| `%s` | `%s` | %s |" % (path, safe_key, reason))
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="check_raw_artifact_metadata")
    p.add_argument("--base-ref", default="origin/master")
    p.add_argument("--head-ref", default="HEAD")
    p.add_argument("--base-content", default=None,
                   help="path or '-' for stdin: NEWLINE-separated list of catalog paths to treat as base (test mode)")
    p.add_argument("--head-content", default=None,
                   help="path or '-' for stdin: NEWLINE-separated list of catalog paths to treat as head (test mode)")
    p.add_argument("--base-dir", default=None,
                   help="override base directory root (test mode)")
    p.add_argument("--head-dir", default=None,
                   help="override head directory root (test mode)")
    p.add_argument("--backfill-out", default=DEFAULT_BACKFILL_OUT)
    p.add_argument("--repo-root", default=None)
    return p


def _read_path_list(arg: str) -> List[str]:
    text = _read_content_arg(arg)
    return [line.strip() for line in text.splitlines() if line.strip()]


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)
    try:
        repo_root = _resolve_repo_root(args.repo_root)

        if args.head_content is not None:
            head_paths = _read_path_list(args.head_content)
            catalog_paths = _classify_paths(head_paths)
            head_dir = Path(args.head_dir).resolve() if args.head_dir else repo_root

            # In test mode we read base from --base-dir/<path> rather than git.
            failures: List[Tuple[str, str, str]] = []
            backfill: List[Tuple[str, str, str]] = []
            base_dir = Path(args.base_dir).resolve() if args.base_dir else None
            for path in catalog_paths:
                base_keys: Set[str] = set()
                if base_dir is not None:
                    base_p = base_dir / path
                    if base_p.is_file():
                        try:
                            base_facs = _load_catalog_text(base_p.read_text(encoding="utf-8"))
                        except GateError:
                            base_facs = []
                        base_keys = {_record_key(r) for r in base_facs}
                head_p = head_dir / path
                if not head_p.is_file():
                    continue
                head_facs = _load_catalog_text(head_p.read_text(encoding="utf-8"))
                for rec in head_facs:
                    key = _record_key(rec)
                    valid, reason = _has_valid_extraction(rec)
                    if valid:
                        continue
                    if key in base_keys:
                        backfill.append((path, key, reason))
                    else:
                        failures.append((path, key, reason))
        else:
            changed = _list_changed_files(args.base_ref, args.head_ref, cwd=repo_root)
            catalog_paths = _classify_paths(changed)
            failures, backfill = evaluate_files(
                catalog_paths=catalog_paths,
                repo_root=repo_root,
                base_ref=args.base_ref,
            )

        # Write backfill TODO if any (idempotent — overwrites each run).
        backfill_out = (
            Path(args.backfill_out)
            if Path(args.backfill_out).is_absolute()
            else (repo_root / args.backfill_out)
        )
        if backfill:
            _write_backfill_todo(backfill_out, backfill)

        if not catalog_paths:
            print("[raw-artifact-metadata] no catalog seed JSON changes detected; PASS.")
            return 0

        if not failures:
            print(
                "[raw-artifact-metadata] all NEWLY-added records carry "
                "extraction.raw_artifact_uri + raw_artifact_sha256; PASS."
            )
            if backfill:
                print(
                    "[raw-artifact-metadata] %d pre-existing record(s) still need "
                    "backfill — see %s" % (len(backfill), backfill_out)
                )
            return 0

        print("[raw-artifact-metadata] FAIL: NEW records missing required extraction metadata:")
        for path, key, reason in failures:
            print("  - %s  key=%s  reason=%s" % (path, key, reason))
        if backfill:
            print(
                "[raw-artifact-metadata] (%d pre-existing record(s) need backfill — see %s)"
                % (len(backfill), backfill_out)
            )
        return 1

    except GateError as exc:
        print("[raw-artifact-metadata] error: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        print("[raw-artifact-metadata] unexpected error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
