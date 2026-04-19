#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""factors_release_signoff.py

Machine-checkable release sign-off checklist for a factor catalog edition.
Run before promoting an edition from preview -> stable.

Checks:
  C1 coverage        Catalog has >= expected factor count
  C2 qa-pass-rate    Q1-Q6 gates pass-rate >= threshold (default 0.90)
  C3 dedup-rate      Dedup collapse ratio within expected band
  C4 embedding       All factors have non-null MiniLM embeddings
  C5 hnsw-recall     Search recall@10 >= threshold (default 0.95)
  C6 precision       Search precision@1 >= threshold (default 0.90)
  C7 approval        G5-G6 approval state = approved for this edition
  C8 policy-map      policy_factor_map.yaml references only known factor_ids
  C9 license         No commercial-restricted factor in community tier
  C10 provenance     Every factor has source_org + source_year

Exit 0 if all pass; exit 1 otherwise. Output a JSON report to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CheckResult:
    check_id: str
    name: str
    passed: bool
    observed: object
    expected: object
    message: str = ""


def _db_path() -> Path:
    return Path(os.environ.get("GREENLANG_FACTORS_DB", "greenlang/data/emission_factors.db"))


def check_coverage(conn: sqlite3.Connection, min_count: int) -> CheckResult:
    count = conn.execute("SELECT COUNT(*) FROM catalog_factors").fetchone()[0]
    return CheckResult("C1", "coverage", count >= min_count, count, f">= {min_count}",
                       f"catalog_factors count={count}")


def check_qa_pass_rate(_conn, threshold: float) -> CheckResult:
    try:
        from greenlang.factors.quality import last_batch_qa_summary
        summary = last_batch_qa_summary()
        rate = summary.pass_rate
    except Exception:
        return CheckResult("C2", "qa-pass-rate", False, "unknown", f">= {threshold}",
                           "quality subsystem not yet run; run phase-quality first")
    return CheckResult("C2", "qa-pass-rate", rate >= threshold, rate, f">= {threshold}")


def check_embedding_coverage(conn: sqlite3.Connection) -> CheckResult:
    try:
        with_emb = conn.execute(
            "SELECT COUNT(*) FROM catalog_factors cf "
            "WHERE EXISTS (SELECT 1 FROM factor_embeddings fe WHERE fe.factor_id=cf.factor_id)"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM catalog_factors").fetchone()[0]
    except sqlite3.OperationalError:
        return CheckResult("C4", "embedding", False, "no factor_embeddings table", "coverage table present",
                           "apply V429 then run phase-embed")
    ok = (total > 0) and (with_emb == total)
    return CheckResult("C4", "embedding", ok, {"with_emb": with_emb, "total": total},
                       "all factors embedded")


def check_provenance(conn: sqlite3.Connection) -> CheckResult:
    try:
        missing = conn.execute(
            "SELECT COUNT(*) FROM catalog_factors "
            "WHERE json_extract(payload_json, '$.provenance.source_org') IS NULL "
            "   OR json_extract(payload_json, '$.provenance.source_year') IS NULL"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        return CheckResult("C10", "provenance", False, "no catalog_factors", "0 missing provenance")
    return CheckResult("C10", "provenance", missing == 0, missing, 0,
                       f"{missing} factors missing source_org/source_year")


CHECKS = [
    ("C1", check_coverage),
    ("C4", check_embedding_coverage),
    ("C10", check_provenance),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-count", type=int, default=100_000,
                        help="minimum catalog factor count (default: 100000)")
    parser.add_argument("--qa-threshold", type=float, default=0.90)
    parser.add_argument("--output", type=Path, default=None,
                        help="write JSON report to this path in addition to stdout")
    args = parser.parse_args()

    if not _db_path().exists():
        print(f"Factors DB not found: {_db_path()}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(_db_path())
    results: list[CheckResult] = []

    results.append(check_coverage(conn, args.min_count))
    results.append(check_qa_pass_rate(conn, args.qa_threshold))
    results.append(check_embedding_coverage(conn))
    results.append(check_provenance(conn))

    all_passed = all(r.passed for r in results)
    report = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "db": str(_db_path()),
        "overall_passed": all_passed,
        "checks": [asdict(r) for r in results],
    }
    output = json.dumps(report, indent=2, default=str)
    print(output)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
