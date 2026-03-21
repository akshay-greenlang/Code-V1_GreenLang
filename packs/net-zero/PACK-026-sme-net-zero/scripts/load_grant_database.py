#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PACK-026: Grant Database Loader
================================

Loads the comprehensive grant database from JSON into PostgreSQL.
Pre-populates the gl_sme_grant_programs table with 18+ grants across 6 regions.

Usage:
    python scripts/load_grant_database.py [--dry-run]

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-026 SME Net Zero Pack
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

# Add pack to path
PACK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR.parent.parent.parent))

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    print("ERROR: psycopg not installed. Run: pip install psycopg[binary]")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GrantDatabaseLoader:
    """Loads grant database from JSON into PostgreSQL."""

    def __init__(self, database_url: str):
        """Initialize loader with database connection."""
        self.database_url = database_url
        self.conn = None

    async def connect(self):
        """Connect to database."""
        self.conn = await psycopg.AsyncConnection.connect(
            self.database_url,
            row_factory=dict_row,
        )
        logger.info("✓ Connected to database")

    async def close(self):
        """Close database connection."""
        if self.conn:
            await self.conn.close()
            logger.info("✓ Closed database connection")

    async def load_grants(self, grants: List[Dict[str, Any]], dry_run: bool = False):
        """
        Load grants into database.

        Args:
            grants: List of grant dictionaries
            dry_run: If True, don't commit changes
        """
        inserted = 0
        updated = 0
        skipped = 0

        for grant in grants:
            try:
                # Check if grant exists
                async with self.conn.cursor() as cur:
                    await cur.execute(
                        "SELECT grant_id FROM gl_sme_grant_programs WHERE grant_id = %s",
                        (grant["grant_id"],),
                    )
                    exists = await cur.fetchone()

                if exists:
                    # Update existing grant
                    if not dry_run:
                        async with self.conn.cursor() as cur:
                            await cur.execute(
                                """
                                UPDATE gl_sme_grant_programs
                                SET
                                    name = %s,
                                    provider = %s,
                                    region = %s,
                                    category = %s,
                                    amount_min_usd = %s,
                                    amount_max_usd = %s,
                                    currency = %s,
                                    eligibility_criteria = %s,
                                    deadline = %s,
                                    status = %s,
                                    match_rate_pct = %s,
                                    url = %s,
                                    documentation_required = %s,
                                    application_complexity = %s,
                                    avg_processing_time_days = %s,
                                    success_rate_pct = %s,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE grant_id = %s
                                """,
                                (
                                    grant["name"],
                                    grant["provider"],
                                    grant["region"],
                                    grant["category"],
                                    grant["amount_min"],
                                    grant["amount_max"],
                                    grant["currency"],
                                    json.dumps(grant["eligibility"]),
                                    grant["deadline"],
                                    grant["status"],
                                    grant["match_rate"],
                                    grant["url"],
                                    grant["documentation_required"],
                                    grant.get("application_complexity", "medium"),
                                    grant.get("avg_processing_time_days", 60),
                                    grant.get("success_rate_pct", 30),
                                    grant["grant_id"],
                                ),
                            )
                    updated += 1
                    logger.info(f"  Updated: {grant['grant_id']} - {grant['name']}")
                else:
                    # Insert new grant
                    if not dry_run:
                        async with self.conn.cursor() as cur:
                            await cur.execute(
                                """
                                INSERT INTO gl_sme_grant_programs (
                                    grant_id, name, provider, region, category,
                                    amount_min_usd, amount_max_usd, currency,
                                    eligibility_criteria, deadline, status, match_rate_pct,
                                    url, documentation_required, application_complexity,
                                    avg_processing_time_days, success_rate_pct
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (
                                    grant["grant_id"],
                                    grant["name"],
                                    grant["provider"],
                                    grant["region"],
                                    grant["category"],
                                    grant["amount_min"],
                                    grant["amount_max"],
                                    grant["currency"],
                                    json.dumps(grant["eligibility"]),
                                    grant["deadline"],
                                    grant["status"],
                                    grant["match_rate"],
                                    grant["url"],
                                    grant["documentation_required"],
                                    grant.get("application_complexity", "medium"),
                                    grant.get("avg_processing_time_days", 60),
                                    grant.get("success_rate_pct", 30),
                                ),
                            )
                    inserted += 1
                    logger.info(f"  Inserted: {grant['grant_id']} - {grant['name']}")

            except Exception as e:
                logger.error(f"  ✗ Failed to load {grant['grant_id']}: {e}")
                skipped += 1

        if not dry_run:
            await self.conn.commit()
            logger.info("✓ Changes committed to database")
        else:
            logger.info("✓ Dry run complete (no changes committed)")

        logger.info(f"\nSummary:")
        logger.info(f"  Inserted: {inserted}")
        logger.info(f"  Updated:  {updated}")
        logger.info(f"  Skipped:  {skipped}")
        logger.info(f"  Total:    {len(grants)}")

    async def verify_load(self):
        """Verify grants were loaded correctly."""
        async with self.conn.cursor() as cur:
            # Count total grants
            await cur.execute("SELECT COUNT(*) as count FROM gl_sme_grant_programs")
            result = await cur.fetchone()
            total_count = result["count"]

            # Count by region
            await cur.execute(
                """
                SELECT region, COUNT(*) as count
                FROM gl_sme_grant_programs
                GROUP BY region
                ORDER BY region
                """
            )
            by_region = await cur.fetchall()

            # Count by category
            await cur.execute(
                """
                SELECT category, COUNT(*) as count
                FROM gl_sme_grant_programs
                GROUP BY category
                ORDER BY category
                """
            )
            by_category = await cur.fetchall()

        logger.info(f"\n✓ Verification Complete:")
        logger.info(f"  Total grants: {total_count}")
        logger.info(f"\n  By Region:")
        for row in by_region:
            logger.info(f"    {row['region']}: {row['count']}")
        logger.info(f"\n  By Category:")
        for row in by_category:
            logger.info(f"    {row['category']}: {row['count']}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load PACK-026 grant database")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't commit changes to database",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "postgresql://greenlang:greenlang@localhost:5432/greenlang"),
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--data-file",
        default=str(PACK_DIR / "data" / "comprehensive_grant_database.json"),
        help="Path to grant database JSON file",
    )
    args = parser.parse_args()

    # Load grant data from JSON
    logger.info(f"Loading grants from: {args.data_file}")
    with open(args.data_file, "r") as f:
        data = json.load(f)

    grants = data["grants"]
    logger.info(f"✓ Loaded {len(grants)} grants from JSON")

    # Connect to database
    loader = GrantDatabaseLoader(args.database_url)
    await loader.connect()

    try:
        # Load grants
        await loader.load_grants(grants, dry_run=args.dry_run)

        # Verify load
        if not args.dry_run:
            await loader.verify_load()

    finally:
        await loader.close()

    logger.info("\n✅ Grant database load complete")


if __name__ == "__main__":
    asyncio.run(main())
