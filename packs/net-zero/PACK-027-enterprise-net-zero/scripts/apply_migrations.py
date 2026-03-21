#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PACK-027: Migration Application Script
=======================================

Applies PACK-027 database migrations (V166-V180) to PostgreSQL.

Usage:
    python scripts/apply_migrations.py [--database-url URL] [--dry-run]

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-027 Enterprise Net Zero Pack
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

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


# PACK-027 migrations (V166-V180)
MIGRATIONS = [
    ("V166", "PACK027_enterprise_schema_and_profiles"),
    ("V167", "PACK027_multi_entity_hierarchy"),
    ("V168", "PACK027_comprehensive_baselines"),
    ("V169", "PACK027_sbti_targets"),
    ("V170", "PACK027_scenario_models"),
    ("V171", "PACK027_carbon_pricing"),
    ("V172", "PACK027_scope4_projects"),
    ("V173", "PACK027_supply_chain_mapping"),
    ("V174", "PACK027_financial_integration"),
    ("V175", "PACK027_risk_assessments"),
    ("V176", "PACK027_regulatory_compliance"),
    ("V177", "PACK027_assurance_records"),
    ("V178", "PACK027_board_reporting"),
    ("V179", "PACK027_data_quality_tracking"),
    ("V180", "PACK027_views_and_indexes"),
]


class MigrationApplicator:
    """Applies PACK-027 database migrations."""

    def __init__(self, database_url: str, dry_run: bool = False):
        """Initialize migration applicator."""
        self.database_url = database_url
        self.dry_run = dry_run
        self.conn = None
        self.applied_count = 0
        self.skipped_count = 0

    async def connect(self):
        """Connect to database."""
        try:
            self.conn = await psycopg.AsyncConnection.connect(
                self.database_url,
                row_factory=dict_row,
            )
            logger.info("✓ Connected to database")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to database: {e}")
            return False

    async def close(self):
        """Close database connection."""
        if self.conn:
            await self.conn.close()
            logger.info("✓ Closed database connection")

    async def ensure_migrations_table(self):
        """Ensure schema_migrations table exists."""
        async with self.conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    applied_by TEXT DEFAULT CURRENT_USER
                )
            """)
            await self.conn.commit()
        logger.info("✓ schema_migrations table ready")

    async def is_migration_applied(self, version: int) -> bool:
        """Check if migration is already applied."""
        async with self.conn.cursor() as cur:
            await cur.execute(
                "SELECT version FROM schema_migrations WHERE version = %s",
                (version,)
            )
            result = await cur.fetchone()
            return result is not None

    async def apply_migration(self, version: str, description: str):
        """Apply a single migration."""
        version_num = int(version[1:])  # Remove 'V' prefix

        # Check if already applied
        if await self.is_migration_applied(version_num):
            logger.info(f"  ⊙ {version} already applied - skipping")
            self.skipped_count += 1
            return

        # Find migration file
        migration_file = PACK_DIR / "migrations" / f"{version}__{description}.sql"

        if not migration_file.exists():
            logger.error(f"  ✗ Migration file not found: {migration_file}")
            return

        # Read migration SQL
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_sql = f.read()

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would apply {version}: {description}")
            logger.info(f"    SQL length: {len(migration_sql)} bytes")
            return

        # Apply migration
        try:
            async with self.conn.cursor() as cur:
                await cur.execute(migration_sql)

            # Record in schema_migrations
            async with self.conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO schema_migrations (version, description)
                    VALUES (%s, %s)
                    """,
                    (version_num, description)
                )

            await self.conn.commit()
            logger.info(f"  ✓ {version} applied successfully")
            self.applied_count += 1

        except Exception as e:
            await self.conn.rollback()
            logger.error(f"  ✗ Failed to apply {version}: {e}")
            raise

    async def apply_all_migrations(self):
        """Apply all PACK-027 migrations."""
        logger.info("\n" + "=" * 70)
        logger.info("PACK-027 Database Migrations (V166-V180)")
        logger.info("=" * 70 + "\n")

        await self.ensure_migrations_table()

        for version, description in MIGRATIONS:
            logger.info(f"Applying {version}: {description}...")
            await self.apply_migration(version, description)

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"\n  Total migrations: {len(MIGRATIONS)}")
        logger.info(f"  Applied:          {self.applied_count}")
        logger.info(f"  Skipped:          {self.skipped_count}")

        if self.dry_run:
            logger.info("\n  [DRY RUN] No changes made to database")
        elif self.applied_count > 0:
            logger.info(f"\n✅ Successfully applied {self.applied_count} migrations")
        else:
            logger.info("\n✓ All migrations already applied")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Apply PACK-027 database migrations")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "postgresql://greenlang:greenlang@localhost:5432/greenlang"),
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be applied without making changes",
    )
    args = parser.parse_args()

    applicator = MigrationApplicator(args.database_url, args.dry_run)

    if not await applicator.connect():
        sys.exit(1)

    try:
        await applicator.apply_all_migrations()
        exit_code = 0
    except Exception as e:
        logger.error(f"\n❌ Migration failed: {e}")
        exit_code = 1
    finally:
        await applicator.close()

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
