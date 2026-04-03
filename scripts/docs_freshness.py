#!/usr/bin/env python3
"""Refresh/check MVP+V1 docs freshness snapshot for V2.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO_ROOT / "docs" / "v2" / "DOC_FRESHNESS_SNAPSHOT.json"
TRACKED_DOCS = [
    "README.md",
    "docs/getting-started.md",
    "cbam-pack-mvp/README.md",
    "docs/v1/DOCS_CONTRACT.md",
    "docs/v1/QUICKSTART.md",
    "docs/v1/STANDARDS.md",
    "docs/v1/apps/GL-CBAM-APP_RUNBOOK.md",
    "docs/v1/apps/GL-CSRD-APP_RUNBOOK.md",
    "docs/v1/apps/GL-VCCI-Carbon-APP_RUNBOOK.md",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_payload() -> dict:
    docs = {}
    for rel in TRACKED_DOCS:
        fp = REPO_ROOT / rel
        docs[rel] = {"exists": fp.exists(), "sha256": _digest(fp) if fp.exists() else None}
    return {"generated_at_utc": _now(), "tracked_docs": docs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    current = _snapshot_payload()
    if args.check:
        if not SNAPSHOT.exists():
            print("docs freshness snapshot missing")
            return 1
        baseline = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        if baseline.get("tracked_docs") != current.get("tracked_docs"):
            print("docs freshness drift detected")
            return 1
        print("docs freshness snapshot matches")
        return 0

    SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"snapshot refreshed: {SNAPSHOT.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
