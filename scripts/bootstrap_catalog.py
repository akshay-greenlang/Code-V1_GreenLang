#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI for the GreenLang Factors catalog bootstrapper (Wave 2.5).

Usage
-----
    python scripts/bootstrap_catalog.py run
        Runs every Safe-to-Certify / Needs-Legal-Review parser whose seed
        input is present under ``greenlang/factors/data/catalog_seed/_inputs/``.
        Writes to ``greenlang/factors/data/catalog_seed/<source_id>/v*.json``.

    python scripts/bootstrap_catalog.py run --source desnz_ghg_conversion
        Restrict to a single source_id.

    python scripts/bootstrap_catalog.py verify
        Re-loads every seed envelope on disk and validates schema-ish
        invariants per-factor (N5 gate + license_class drift checks).

    python scripts/bootstrap_catalog.py stats
        Prints family / jurisdiction coverage matrix.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import the module directly so we don't rely on repo-root PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from greenlang.factors.ingestion.bootstrap import (  # noqa: E402
    SOURCE_SPECS,
    bootstrap_catalog,
    load_seed_envelopes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bootstrap_catalog")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    only = [args.source] if args.source else None
    report = bootstrap_catalog(only_sources=only)

    print("")
    print("=== catalog bootstrap run ===")
    print(f"started:       {report.run_started_at}")
    print(f"finished:      {report.run_finished_at}")
    print(f"factor count:  {report.total_factor_count}")
    print("")
    print("ingested sources:")
    for r in report.ingested:
        print(
            f"  [OK]   {r.source_id:40s}  {r.factor_count:5d} factors  "
            f"(rejected={r.rejected_count})"
        )
        if r.errors:
            for e in r.errors[:3]:
                print(f"         err: {e}")

    print("")
    print("skipped parsers:")
    for r in report.skipped:
        print(f"  [SKIP] {r.source_id:40s}  {r.reason}")

    if report.errored:
        print("")
        print("errored sources:")
        for r in report.errored:
            print(f"  [ERR]  {r.source_id:40s}  {r.reason}")

    if args.json:
        out_path = _REPO_ROOT / "build" / "bootstrap_catalog_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nreport also written to {out_path}")

    return 0 if not report.errored else 1


def cmd_verify(_args: argparse.Namespace) -> int:
    envelopes = load_seed_envelopes()
    if not envelopes:
        print("no seed envelopes found on disk; run `bootstrap_catalog run` first.")
        return 1

    total = 0
    by_source: Dict[str, int] = Counter()
    invalid: List[str] = []
    for env in envelopes:
        src = env.get("source_id")
        factors = env.get("factors") or []
        for rec in factors:
            total += 1
            by_source[src] += 1
            if rec.get("redistribution_class") in (
                "licensed_embedded", "customer_private", "oem_redistributable"
            ):
                invalid.append(
                    f"license-class drift: {src} factor {rec.get('factor_id')!r} "
                    f"marked {rec.get('redistribution_class')!r} — must NOT ship"
                )

    print("=== catalog verify ===")
    print(f"envelopes inspected: {len(envelopes)}")
    print(f"total factors:       {total}")
    print("")
    print("per-source:")
    for src, count in sorted(by_source.items()):
        print(f"  {count:5d}  {src}")
    if invalid:
        print("")
        print("INTEGRITY ISSUES:")
        for msg in invalid[:20]:
            print(f"  - {msg}")
        return 1
    return 0


def cmd_stats(_args: argparse.Namespace) -> int:
    envelopes = load_seed_envelopes()
    if not envelopes:
        print("no seed envelopes found on disk; run `bootstrap_catalog run` first.")
        return 1

    by_family: Dict[str, int] = Counter()
    by_jurisdiction: Dict[str, int] = Counter()
    family_x_jurisdiction: Dict[str, Dict[str, int]] = defaultdict(Counter)

    for env in envelopes:
        for rec in env.get("factors") or []:
            fam = rec.get("factor_family") or "<unspecified>"
            jur = None
            raw_jur = rec.get("jurisdiction")
            if isinstance(raw_jur, dict):
                jur = raw_jur.get("country")
            if not jur:
                jur = rec.get("geography")
            jur = jur or "<unspecified>"
            by_family[fam] += 1
            by_jurisdiction[jur] += 1
            family_x_jurisdiction[fam][jur] += 1

    print("=== factor family coverage ===")
    for fam, n in sorted(by_family.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5d}  {fam}")

    print("")
    print("=== jurisdiction coverage (top 30) ===")
    for jur, n in sorted(by_jurisdiction.items(), key=lambda kv: -kv[1])[:30]:
        print(f"  {n:5d}  {jur}")

    print("")
    print("=== family x jurisdiction matrix (non-zero cells, top 40) ===")
    flat = []
    for fam, jd in family_x_jurisdiction.items():
        for jur, n in jd.items():
            flat.append((n, fam, jur))
    for n, fam, jur in sorted(flat, key=lambda x: -x[0])[:40]:
        print(f"  {n:5d}  {fam:35s}  {jur}")

    return 0


def cmd_list_sources(_args: argparse.Namespace) -> int:
    print("=== registered sources ===")
    for s in SOURCE_SPECS:
        print(
            f"  {s.source_id:40s}  "
            f"redist={s.redistribution_class:20s}  "
            f"license={s.license_name:25s}  "
            f"seed={s.seed_input or '-'}"
        )
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="GreenLang Factors catalog bootstrapper"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run all (or one) parsers and write seeds")
    p_run.add_argument("--source", help="restrict to a single source_id")
    p_run.add_argument("--json", action="store_true", help="also write JSON report under build/")
    p_run.set_defaults(fn=cmd_run)

    p_verify = sub.add_parser("verify", help="re-load seeds & validate")
    p_verify.set_defaults(fn=cmd_verify)

    p_stats = sub.add_parser("stats", help="print family/jurisdiction coverage")
    p_stats.set_defaults(fn=cmd_stats)

    p_list = sub.add_parser("list-sources", help="list all registered source specs")
    p_list.set_defaults(fn=cmd_list_sources)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
