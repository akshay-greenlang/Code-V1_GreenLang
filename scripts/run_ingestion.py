#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GreenLang Factors catalog ingestion runner.

Convenience script to populate a SQLite factor catalog with emission factors.
Supports three modes:

- ``builtin``:   Load 327 built-in factors from EmissionFactorDatabase.
- ``synthetic``: Generate 10K-50K realistic synthetic factors for dev/test.
- ``full``:      Load built-in + synthetic (real source ingestion when
                 data files are available).

Usage:
    python scripts/run_ingestion.py --mode synthetic --count 25000 \\
        --sqlite ./factors.db --edition 2026.04.0

    python scripts/run_ingestion.py --mode builtin \\
        --sqlite ./factors.db --edition builtin-v1.0.0

    python scripts/run_ingestion.py --mode full --count 25000 \\
        --sqlite ./factors.db --edition 2026.04.0-full

Options:
    --mode       One of: builtin, synthetic, full (default: synthetic)
    --count      Number of synthetic factors to generate (default: 25000)
    --sqlite     Path to SQLite database file (default: ./factors_catalog.db)
    --edition    Edition identifier (default: auto-generated from mode + date)
    --seed       Random seed for synthetic generation (default: 42)
    --dry-run    Validate without writing to database
    --years      Comma-separated years for synthetic data (default: 2022,2023,2024,2025)
    --verbose    Enable debug logging
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("run_ingestion")


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the ingestion runner."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _auto_edition_id(mode: str) -> str:
    """Generate an edition ID from mode and current date."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y.%m")
    return f"{date_str}.0-{mode}"


# ---------------------------------------------------------------------------
# Mode: builtin
# ---------------------------------------------------------------------------

def run_builtin(
    sqlite_path: Path,
    edition_id: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Load built-in EmissionFactorDatabase factors into the catalog.

    Args:
        sqlite_path: Path to SQLite database.
        edition_id: Edition identifier.
        dry_run: If True, validate but do not write.

    Returns:
        Summary statistics dict.
    """
    from greenlang.data.emission_factor_database import EmissionFactorDatabase

    logger.info("Loading built-in EmissionFactorDatabase...")
    db = EmissionFactorDatabase(enable_cache=False)
    records = list(db.factors.values())
    logger.info("Found %d built-in factors", len(records))

    if dry_run:
        logger.info("[DRY RUN] Would insert %d factors into %s edition=%s",
                     len(records), sqlite_path, edition_id)
        return {
            "mode": "builtin",
            "dry_run": True,
            "total_available": len(records),
            "total_ingested": 0,
            "edition_id": edition_id,
            "sqlite": str(sqlite_path),
        }

    from greenlang.factors.etl.ingest import ingest_builtin_database

    count = ingest_builtin_database(
        sqlite_path,
        edition_id,
        label="Built-in v2 EmissionFactorDatabase",
        status="stable",
    )
    return {
        "mode": "builtin",
        "dry_run": False,
        "total_ingested": count,
        "edition_id": edition_id,
        "sqlite": str(sqlite_path),
    }


# ---------------------------------------------------------------------------
# Mode: synthetic
# ---------------------------------------------------------------------------

def run_synthetic(
    sqlite_path: Path,
    edition_id: str,
    count: int = 25000,
    seed: int = 42,
    years: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Generate and load synthetic emission factors.

    Args:
        sqlite_path: Path to SQLite database.
        edition_id: Edition identifier.
        count: Number of synthetic factors to generate.
        seed: Random seed for deterministic generation.
        years: Reporting years (default: 2022-2025).
        dry_run: If True, generate and validate but do not write.

    Returns:
        Summary statistics dict.
    """
    from greenlang.factors.ingestion.synthetic_data import generate_and_validate
    from greenlang.factors.etl.normalize import dict_to_emission_factor_record
    from greenlang.factors.etl.qa import validate_factor_dict

    logger.info("Generating %d synthetic factors (seed=%d)...", count, seed)
    t0 = time.monotonic()

    valid_dicts, total_generated, total_rejected = generate_and_validate(
        count=count, seed=seed, years=years,
    )

    gen_time = time.monotonic() - t0
    logger.info(
        "Generation complete in %.1fs: generated=%d valid=%d rejected=%d",
        gen_time, total_generated, len(valid_dicts), total_rejected,
    )

    if dry_run:
        # Sample a few for inspection
        sample = valid_dicts[:3]
        logger.info("[DRY RUN] Would insert %d factors. Sample factor_ids:", len(valid_dicts))
        for fd in sample:
            logger.info("  %s (%s, %s, scope=%s)",
                         fd["factor_id"], fd["fuel_type"], fd["geography"], fd["scope"])
        return {
            "mode": "synthetic",
            "dry_run": True,
            "total_generated": total_generated,
            "total_valid": len(valid_dicts),
            "total_rejected": total_rejected,
            "generation_time_s": round(gen_time, 2),
            "edition_id": edition_id,
            "sqlite": str(sqlite_path),
            "sample_ids": [fd["factor_id"] for fd in sample],
        }

    # Convert to EmissionFactorRecord objects
    logger.info("Converting %d factor dicts to EmissionFactorRecord...", len(valid_dicts))
    t1 = time.monotonic()
    records = []
    conversion_errors = []
    for fd in valid_dicts:
        try:
            rec = dict_to_emission_factor_record(fd)
            records.append(rec)
        except Exception as exc:
            conversion_errors.append(f"{fd.get('factor_id')}: {exc}")
            logger.debug("Conversion error for %s: %s", fd.get("factor_id"), exc)

    convert_time = time.monotonic() - t1
    logger.info(
        "Conversion complete in %.1fs: records=%d errors=%d",
        convert_time, len(records), len(conversion_errors),
    )

    if not records:
        logger.error("No records survived conversion. Aborting.")
        return {
            "mode": "synthetic",
            "dry_run": False,
            "error": "No records survived conversion",
            "conversion_errors": conversion_errors[:20],
        }

    # Write to SQLite
    logger.info("Writing %d records to %s edition=%s...", len(records), sqlite_path, edition_id)
    t2 = time.monotonic()

    from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository
    from greenlang.factors.edition_manifest import build_manifest_for_factors

    repo = SqliteFactorCatalogRepository(sqlite_path)
    manifest = build_manifest_for_factors(
        edition_id,
        "stable",
        records,
        changelog=[
            f"Synthetic ingestion: {len(records)} factors (seed={seed})",
            f"Generated {total_generated}, validated {len(valid_dicts)}, rejected {total_rejected}",
        ],
    )
    repo.upsert_edition(
        edition_id,
        "stable",
        f"Synthetic factors (n={len(records)}, seed={seed})",
        manifest.to_dict(),
        manifest.changelog,
    )
    repo.insert_factors(edition_id, records)

    write_time = time.monotonic() - t2
    total_time = time.monotonic() - t0
    logger.info("Write complete in %.1fs. Total pipeline: %.1fs", write_time, total_time)

    # Coverage stats
    stats = repo.coverage_stats(edition_id)

    return {
        "mode": "synthetic",
        "dry_run": False,
        "total_generated": total_generated,
        "total_valid": len(valid_dicts),
        "total_rejected": total_rejected,
        "total_ingested": len(records),
        "conversion_errors": len(conversion_errors),
        "edition_id": edition_id,
        "sqlite": str(sqlite_path),
        "manifest_hash": manifest.manifest_fingerprint(),
        "timing": {
            "generation_s": round(gen_time, 2),
            "conversion_s": round(convert_time, 2),
            "write_s": round(write_time, 2),
            "total_s": round(total_time, 2),
        },
        "coverage": {
            "total_factors": stats.get("total_factors", 0),
            "geographies": stats.get("geographies", 0),
            "fuel_types": stats.get("fuel_types", 0),
            "scopes": stats.get("scopes", {}),
        },
    }


# ---------------------------------------------------------------------------
# Mode: full
# ---------------------------------------------------------------------------

def run_full(
    sqlite_path: Path,
    edition_id: str,
    count: int = 25000,
    seed: int = 42,
    years: Optional[List[int]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run builtin + synthetic ingestion into a single edition.

    The builtin factors are loaded first as certified, then synthetic
    factors are added as preview.

    Args:
        sqlite_path: Path to SQLite database.
        edition_id: Edition identifier.
        count: Number of synthetic factors.
        seed: Random seed.
        years: Reporting years.
        dry_run: If True, validate only.

    Returns:
        Combined summary statistics dict.
    """
    logger.info("Running full ingestion: builtin + synthetic")

    # Phase 1: Built-in
    builtin_result = run_builtin(sqlite_path, edition_id, dry_run=dry_run)
    builtin_count = builtin_result.get("total_ingested", 0)

    # Phase 2: Synthetic
    synthetic_result = run_synthetic(
        sqlite_path, edition_id,
        count=count, seed=seed, years=years, dry_run=dry_run,
    )
    synthetic_count = synthetic_result.get("total_ingested", 0)

    total = builtin_count + synthetic_count
    logger.info("Full ingestion complete: builtin=%d synthetic=%d total=%d",
                 builtin_count, synthetic_count, total)

    return {
        "mode": "full",
        "dry_run": dry_run,
        "builtin": builtin_result,
        "synthetic": synthetic_result,
        "total_ingested": total,
        "edition_id": edition_id,
        "sqlite": str(sqlite_path),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GreenLang Factors catalog ingestion runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["builtin", "synthetic", "full"],
        default="synthetic",
        help="Ingestion mode (default: synthetic)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=25000,
        help="Number of synthetic factors to generate (default: 25000)",
    )
    parser.add_argument(
        "--sqlite",
        type=str,
        default="./factors_catalog.db",
        help="Path to SQLite database (default: ./factors_catalog.db)",
    )
    parser.add_argument(
        "--edition",
        type=str,
        default=None,
        help="Edition identifier (default: auto-generated)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic generation (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without writing to database",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2022,2023,2024,2025",
        help="Comma-separated years for synthetic data (default: 2022,2023,2024,2025)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the ingestion runner."""
    args = parse_args(argv)
    _setup_logging(verbose=args.verbose)

    sqlite_path = Path(args.sqlite).resolve()
    edition_id = args.edition or _auto_edition_id(args.mode)
    years = [int(y.strip()) for y in args.years.split(",")]

    logger.info("=" * 60)
    logger.info("GreenLang Factors Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info("Mode:      %s", args.mode)
    logger.info("SQLite:    %s", sqlite_path)
    logger.info("Edition:   %s", edition_id)
    logger.info("Dry-run:   %s", args.dry_run)
    if args.mode in ("synthetic", "full"):
        logger.info("Count:     %d", args.count)
        logger.info("Seed:      %d", args.seed)
        logger.info("Years:     %s", years)
    logger.info("-" * 60)

    try:
        if args.mode == "builtin":
            result = run_builtin(sqlite_path, edition_id, dry_run=args.dry_run)
        elif args.mode == "synthetic":
            result = run_synthetic(
                sqlite_path, edition_id,
                count=args.count, seed=args.seed, years=years,
                dry_run=args.dry_run,
            )
        elif args.mode == "full":
            result = run_full(
                sqlite_path, edition_id,
                count=args.count, seed=args.seed, years=years,
                dry_run=args.dry_run,
            )
        else:
            logger.error("Unknown mode: %s", args.mode)
            return 1

    except Exception as exc:
        logger.error("Ingestion pipeline failed: %s", exc, exc_info=True)
        return 1

    # Print summary
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
