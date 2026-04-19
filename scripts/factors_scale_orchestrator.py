#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""factors_scale_orchestrator.py

End-to-end orchestrator for FACTORS-SCALE phases #1-#8 (327 -> 100K factors).

Phases:
  1 source-acquire   Download full archives per greenlang/factors/data/source_registry.yaml
  2 parse            Run parsers in parallel across artifacts
  3 quality          Run Q1-Q6 gates on parsed payload
  4 dedupe           Cross-source dedup via dedupe_rules
  5 embed            Generate MiniLM + MPNet embeddings
  6 index            Populate pgvector HNSW index
  7 migrate          Run V437/V438 migrations against prod PG
  8 validate         Regression against gold-eval + release sign-off

Usage:
  python -m scripts.factors_scale_orchestrator --phases all
  python -m scripts.factors_scale_orchestrator --phases 1,2,3 --dry-run
  python -m scripts.factors_scale_orchestrator --phase embed --sources DEFRA,EPA
  python -m scripts.factors_scale_orchestrator --status
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("factors-scale")


PHASE_DEFS = [
    ("source-acquire", "1"),
    ("parse", "2"),
    ("quality", "3"),
    ("dedupe", "4"),
    ("embed", "5"),
    ("index", "6"),
    ("migrate", "7"),
    ("validate", "8"),
]


@dataclass
class PhaseResult:
    name: str
    ok: bool
    duration_seconds: float
    details: dict = field(default_factory=dict)


# ---- phase implementations ----

def phase_source_acquire(sources: Optional[list[str]], dry_run: bool) -> PhaseResult:
    from greenlang.factors.source_registry import load_source_registry

    start = time.perf_counter()
    registry = load_source_registry()
    if sources:
        registry = [s for s in registry if s.source_id in sources]
    details = {"sources": [s.source_id for s in registry], "count": len(registry)}
    if dry_run:
        logger.info("[source-acquire] dry-run — would fetch %d sources", len(registry))
        return PhaseResult("source-acquire", True, time.perf_counter() - start, details)
    # Real fetch via greenlang.factors.ingestion.fetchers
    from greenlang.factors.ingestion.fetchers import HttpFetcher

    fetcher = HttpFetcher()
    fetched = 0
    failed = []
    for s in registry:
        try:
            fetcher.fetch(s.url, s.cache_path or f"/tmp/factors/{s.source_id}.bin")
            fetched += 1
        except Exception as exc:
            logger.error("[source-acquire] %s failed: %s", s.source_id, exc)
            failed.append(s.source_id)
    details.update({"fetched": fetched, "failed": failed})
    return PhaseResult("source-acquire", not failed, time.perf_counter() - start, details)


def phase_parse(sources: Optional[list[str]], dry_run: bool) -> PhaseResult:
    start = time.perf_counter()
    from greenlang.factors.ingestion.parser_harness import ParserHarness

    harness = ParserHarness()
    details = {"parsers": list(harness.available_parsers())}
    if dry_run:
        return PhaseResult("parse", True, time.perf_counter() - start, details)

    total = 0
    for parser_name in harness.available_parsers():
        if sources and parser_name not in sources:
            continue
        try:
            out = harness.run_parser(parser_name)
            total += len(out)
            logger.info("[parse] %s -> %d factors", parser_name, len(out))
        except Exception as exc:
            logger.exception("[parse] %s failed: %s", parser_name, exc)
    details["factors_parsed"] = total
    return PhaseResult("parse", True, time.perf_counter() - start, details)


def phase_quality(sources: Optional[list[str]], dry_run: bool) -> PhaseResult:
    start = time.perf_counter()
    if dry_run:
        return PhaseResult("quality", True, time.perf_counter() - start, {"mode": "dry-run"})

    from greenlang.factors.quality import run_batch_qa  # noqa: WPS433

    result = run_batch_qa(sources=sources)
    return PhaseResult(
        "quality",
        result.passed,
        time.perf_counter() - start,
        {
            "passed_count": result.passed_count,
            "failed_count": result.failed_count,
            "warning_count": result.warning_count,
        },
    )


def phase_dedupe(sources: Optional[list[str]], dry_run: bool) -> PhaseResult:
    start = time.perf_counter()
    if dry_run:
        return PhaseResult("dedupe", True, time.perf_counter() - start, {"mode": "dry-run"})
    from greenlang.factors.dedupe_rules import run_dedup

    result = run_dedup(sources=sources)
    return PhaseResult(
        "dedupe",
        True,
        time.perf_counter() - start,
        {"before": result.before, "after": result.after, "removed": result.removed},
    )


def phase_embed(sources: Optional[list[str]], dry_run: bool) -> PhaseResult:
    start = time.perf_counter()
    if dry_run:
        return PhaseResult("embed", True, time.perf_counter() - start, {"mode": "dry-run"})
    from greenlang.factors.matching import generate_embeddings_for_catalog

    result = generate_embeddings_for_catalog(sources=sources, batch_size=128)
    return PhaseResult(
        "embed",
        True,
        time.perf_counter() - start,
        {
            "items_embedded": result.items_embedded,
            "minilm_seconds": result.minilm_seconds,
            "mpnet_seconds": result.mpnet_seconds,
        },
    )


def phase_index(sources: Optional[list[str]], dry_run: bool) -> PhaseResult:
    start = time.perf_counter()
    if dry_run:
        return PhaseResult("index", True, time.perf_counter() - start, {"mode": "dry-run"})
    from greenlang.factors.index_manager import build_hnsw_indexes

    result = build_hnsw_indexes(m=24, ef_construction=200, ef_search=100)
    return PhaseResult(
        "index",
        True,
        time.perf_counter() - start,
        {"built": result.built, "build_seconds": result.build_seconds},
    )


def phase_migrate(sources: Optional[list[str]], dry_run: bool) -> PhaseResult:
    start = time.perf_counter()
    db_url = os.environ.get("GREENLANG_DATABASE_URL", "")
    if not db_url:
        return PhaseResult(
            "migrate", False, time.perf_counter() - start,
            {"error": "GREENLANG_DATABASE_URL not set"},
        )
    if dry_run:
        return PhaseResult("migrate", True, time.perf_counter() - start, {"url": "<set>"})
    import subprocess  # noqa: S404

    sql_dir = Path("deployment/database/migrations/sql")
    migrations = sorted(sql_dir.glob("V4*.sql"))
    logger.info("[migrate] applying %d migrations", len(migrations))
    for m in migrations:
        subprocess.run(
            ["psql", db_url, "-f", str(m)], check=True, capture_output=True, text=True
        )
    return PhaseResult(
        "migrate", True, time.perf_counter() - start,
        {"applied": [m.name for m in migrations]},
    )


def phase_validate(sources: Optional[list[str]], dry_run: bool) -> PhaseResult:
    start = time.perf_counter()
    import subprocess  # noqa: S404

    if dry_run:
        return PhaseResult("validate", True, time.perf_counter() - start, {"mode": "dry-run"})
    result = subprocess.run(
        [sys.executable, "scripts/factors_match_eval.py"],
        capture_output=True, text=True, check=False,
    )
    ok = result.returncode == 0
    return PhaseResult(
        "validate", ok, time.perf_counter() - start,
        {"stdout": result.stdout[-800:], "stderr": result.stderr[-800:]},
    )


PHASE_FUNCS: dict[str, Callable[[Optional[list[str]], bool], PhaseResult]] = {
    "source-acquire": phase_source_acquire,
    "parse": phase_parse,
    "quality": phase_quality,
    "dedupe": phase_dedupe,
    "embed": phase_embed,
    "index": phase_index,
    "migrate": phase_migrate,
    "validate": phase_validate,
}


def print_status() -> None:
    """Print a short status summary suitable for operators."""
    try:
        import sqlite3
        db_path = Path(
            os.environ.get(
                "GREENLANG_FACTORS_DB",
                "greenlang/data/emission_factors.db",
            )
        )
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            total_legacy = conn.execute("SELECT COUNT(*) FROM emission_factors").fetchone()[0]
            total_catalog = conn.execute("SELECT COUNT(*) FROM catalog_factors").fetchone()[0]
            editions = conn.execute("SELECT COUNT(*) FROM editions").fetchone()[0]
            print(f"Factor DB: {db_path}")
            print(f"  legacy factors  : {total_legacy:>7}")
            print(f"  catalog factors : {total_catalog:>7}   (target: 100000)")
            print(f"  editions        : {editions:>7}")
            coverage = 100.0 * total_catalog / 100_000
            print(f"  coverage       : {coverage:.2f}% of 100K target")
    except Exception as exc:
        logger.warning("status: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phases", default="all",
                        help="comma-separated phase names or numbers, 'all', or empty for --status")
    parser.add_argument("--sources", default="",
                        help="comma-separated source_id filter (e.g. EPA,DEFRA)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true", help="print status and exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if args.status:
        print_status()
        return 0

    if args.phases == "all":
        phases = [p[0] for p in PHASE_DEFS]
    else:
        names_by_num = {num: name for name, num in PHASE_DEFS}
        phases = []
        for token in args.phases.split(","):
            t = token.strip()
            if t in names_by_num:
                phases.append(names_by_num[t])
            elif t in PHASE_FUNCS:
                phases.append(t)
            else:
                parser.error(f"unknown phase: {t}")

    sources = [s.strip() for s in args.sources.split(",") if s.strip()] or None

    results: list[PhaseResult] = []
    for phase in phases:
        logger.info("=== phase %s ===", phase)
        try:
            r = PHASE_FUNCS[phase](sources, args.dry_run)
        except Exception as exc:
            logger.exception("phase %s crashed", phase)
            r = PhaseResult(phase, False, 0.0, {"crash": str(exc)})
        results.append(r)
        logger.info("[%s] ok=%s duration=%.1fs details=%s", r.name, r.ok, r.duration_seconds, r.details)
        if not r.ok:
            logger.error("Stopping orchestrator: phase %s failed", phase)
            break

    passed = sum(1 for r in results if r.ok)
    logger.info("Done: %d/%d phases passed", passed, len(results))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
