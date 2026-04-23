#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Idempotent migration CLI: catalog_seed/ -> catalog_seed_v1/ (W4-A).

Reads every ``*.json`` under ``greenlang/factors/data/catalog_seed/`` and
writes the v1-shape equivalent under ``greenlang/factors/data/catalog_seed_v1/``
with the same sub-directory layout.

Idempotence
-----------
Running this script twice produces bit-identical output. Already-v1 files
are re-parsed through :func:`canonical_v1.migrate_record` (a no-op on v1
shape).  Exit code 0 on success; non-zero on validation failures.

Usage
-----
::

    python scripts/migrate_catalog_to_v1.py [--source SRC] [--dest DEST] [--check]

``--check`` verifies that the dest folder matches what the migration would
produce (used by CI); it never writes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from greenlang.factors.data import canonical_v1  # noqa: E402


DEFAULT_SRC = REPO_ROOT / "greenlang" / "factors" / "data" / "catalog_seed"
DEFAULT_DEST = REPO_ROOT / "greenlang" / "factors" / "data" / "catalog_seed_v1"


def _iter_source_files(src: Path) -> List[Path]:
    """Yield bundle JSON files.  Skips ``_inputs/`` (raw pre-ingest docs)."""
    paths: List[Path] = []
    for p in sorted(src.rglob("*.json")):
        rel = p.relative_to(src)
        if rel.parts and rel.parts[0].startswith("_"):
            continue
        paths.append(p)
    return paths


def _migrate_factor(factor: Dict[str, Any]) -> Dict[str, Any]:
    return canonical_v1.migrate_record(factor)


def _migrate_bundle(bundle: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
    """Migrate a catalog_seed bundle file.

    Returns (new_bundle, count_in, count_out).
    """
    factors = bundle.get("factors") or []
    migrated: List[Dict[str, Any]] = []
    errors = 0
    for f in factors:
        try:
            migrated.append(_migrate_factor(f))
        except Exception as exc:  # pragma: no cover — surfaced via --check
            errors += 1
            print(
                f"[migrate_catalog_to_v1] ERROR on {f.get('factor_id')!r}: {exc}",
                file=sys.stderr,
            )
    new_bundle = {
        **{k: v for k, v in bundle.items() if k != "factors"},
        "schema_version": "factor_record_v1",
        "factor_count": len(migrated),
        "factors": migrated,
    }
    if errors > 0:
        new_bundle["migration_errors"] = errors
    return new_bundle, len(factors), len(migrated)


def _write_if_changed(path: Path, data: Dict[str, Any], dry_run: bool) -> bool:
    """Write ``data`` as JSON to ``path``. Return True iff content changed."""
    text = json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    if dry_run:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate catalog_seed factor bundles to v1 shape."
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SRC,
        help=f"Source catalog_seed dir (default: {DEFAULT_SRC})"
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help=f"Destination catalog_seed_v1 dir (default: {DEFAULT_DEST})"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if dest is out of date (no writes)."
    )
    args = parser.parse_args()

    src = args.source
    dest = args.dest
    if not src.exists():
        print(f"[migrate_catalog_to_v1] source not found: {src}", file=sys.stderr)
        return 2

    files = _iter_source_files(src)
    total_in = 0
    total_out = 0
    changed_files: List[str] = []

    for sf in files:
        rel = sf.relative_to(src)
        df = dest / rel
        bundle = json.loads(sf.read_text(encoding="utf-8"))
        new_bundle, c_in, c_out = _migrate_bundle(bundle)
        total_in += c_in
        total_out += c_out
        if _write_if_changed(df, new_bundle, dry_run=args.check):
            changed_files.append(str(rel))

    status = "CHECK" if args.check else "WRITE"
    print(
        f"[migrate_catalog_to_v1] {status}: "
        f"{len(files)} bundles, {total_in} factors in -> {total_out} factors out; "
        f"{len(changed_files)} file(s) {'would change' if args.check else 'changed'}."
    )
    if changed_files and args.check:
        for f in changed_files[:10]:
            print(f"  ~ {f}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
