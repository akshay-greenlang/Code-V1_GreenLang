#!/usr/bin/env python3
"""Insert N duplicate-safe rows into a SQLite catalog for GA load smoke (C/GA)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository
from greenlang.factors.etl.ingest import ingest_builtin_database


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", required=True)
    p.add_argument("--edition-id", required=True)
    p.add_argument("--label", default="load-smoke")
    p.add_argument("--rounds", type=int, default=1, help="Repeat ingest rounds for row growth")
    args = p.parse_args()
    path = Path(args.sqlite)
    for _ in range(max(1, args.rounds)):
        n = ingest_builtin_database(path, args.edition_id, label=args.label)
        print("ingested", n)
    repo = SqliteFactorCatalogRepository(path)
    eid = repo.get_default_edition_id()
    stats = repo.coverage_stats(eid)
    print("coverage", stats.get("total_factors"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
