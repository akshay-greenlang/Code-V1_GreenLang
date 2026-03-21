#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PACK-027: Migration Verification Script
========================================

Verifies that all PACK-027 database migrations (V166-V180) have been applied correctly.

Usage:
    python scripts/verify_migrations.py

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-027 Enterprise Net Zero Pack
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

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


# Expected migrations for PACK-027
EXPECTED_MIGRATIONS = {
    166: "Create schema enterprise_net_zero and profiles table",
    167: "Create entities and consolidation tables",
    168: "Create baselines and targets tables",
    169: "Create scenarios and pathways tables",
    170: "Create carbon pricing and Scope 4 tables",
    171: "Create supply chain mapping tables",
    172: "Create supplier engagement tracking tables",
    173: "Create assurance records and evidence tables",
    174: "Create regulatory filings and compliance tables",
    175: "Create board reporting and executive dashboards tables",
    176: "Create data quality scoring tables",
    177: "Create calculation cache and performance tables",
    178: "Create audit trail and lineage tables",
    179: "Create integration status and sync logs tables",
    180: "Create views (dashboard, compliance, supply chain, assurance)",
}

# Expected tables
EXPECTED_TABLES = [
    "gl_enterprise_profiles",
    "gl_enterprise_entities",
    "gl_enterprise_baselines",
    "gl_enterprise_targets",
    "gl_enterprise_scenarios",
    "gl_enterprise_carbon_prices",
    "gl_enterprise_scope4_registry",
    "gl_enterprise_supply_chain_map",
    "gl_enterprise_supplier_engagement",
    "gl_enterprise_assurance_records",
    "gl_enterprise_regulatory_filings",
    "gl_enterprise_board_reports",
    "gl_enterprise_data_quality_scores",
    "gl_enterprise_audit_trail",
    "gl_enterprise_calculation_cache",
]

# Expected views
EXPECTED_VIEWS = [
    "vw_enterprise_dashboard",
    "vw_enterprise_compliance",
    "vw_enterprise_supply_chain",
    "vw_enterprise_assurance",
]


class MigrationVerifier:
    """Verifies PACK-027 database migrations."""

    def __init__(self, database_url: str):
        """Initialize verifier with database connection."""
        self.database_url = database_url
        self.conn = None
        self.errors = []
        self.warnings = []

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

    async def check_migrations(self) -> bool:
        """Check if all expected migrations are applied."""
        logger.info("\n1. Checking migrations...")

        async with self.conn.cursor() as cur:
            # Get applied migrations in V166-V180 range
            await cur.execute(
                """
                SELECT version, description, applied_at
                FROM schema_migrations
                WHERE version >= 166 AND version <= 180
                ORDER BY version
                """
            )
            applied = await cur.fetchall()

        applied_versions = {row["version"]: row for row in applied}

        all_found = True
        for version, expected_desc in EXPECTED_MIGRATIONS.items():
            if version in applied_versions:
                logger.info(f"  ✓ V{version:03d}: {applied_versions[version]['description']}")
            else:
                logger.error(f"  ✗ V{version:03d}: NOT APPLIED (expected: {expected_desc})")
                self.errors.append(f"Migration V{version:03d} not applied")
                all_found = False

        return all_found

    async def check_tables(self) -> bool:
        """Check if all expected tables exist."""
        logger.info("\n2. Checking tables...")

        async with self.conn.cursor() as cur:
            # Get all tables in public schema
            await cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename LIKE 'gl_enterprise_%'
                ORDER BY tablename
                """
            )
            existing_tables = {row["tablename"] for row in await cur.fetchall()}

        all_found = True
        for table in EXPECTED_TABLES:
            if table in existing_tables:
                logger.info(f"  ✓ {table}")
            else:
                logger.error(f"  ✗ {table} - NOT FOUND")
                self.errors.append(f"Table {table} not found")
                all_found = False

        return all_found

    async def check_views(self) -> bool:
        """Check if all expected views exist."""
        logger.info("\n3. Checking views...")

        async with self.conn.cursor() as cur:
            # Get all views in public schema
            await cur.execute(
                """
                SELECT viewname
                FROM pg_views
                WHERE schemaname = 'public'
                AND viewname LIKE 'vw_enterprise_%'
                ORDER BY viewname
                """
            )
            existing_views = {row["viewname"] for row in await cur.fetchall()}

        all_found = True
        for view in EXPECTED_VIEWS:
            if view in existing_views:
                logger.info(f"  ✓ {view}")
            else:
                logger.error(f"  ✗ {view} - NOT FOUND")
                self.errors.append(f"View {view} not found")
                all_found = False

        return all_found

    async def check_indexes(self) -> bool:
        """Check index counts on critical tables."""
        logger.info("\n4. Checking indexes...")

        critical_tables = [
            "gl_enterprise_profiles",
            "gl_enterprise_entities",
            "gl_enterprise_baselines",
            "gl_enterprise_scenarios",
            "gl_enterprise_supply_chain_map",
        ]

        async with self.conn.cursor() as cur:
            for table in critical_tables:
                await cur.execute(
                    """
                    SELECT COUNT(*) as count
                    FROM pg_indexes
                    WHERE tablename = %s
                    """,
                    (table,),
                )
                result = await cur.fetchone()
                count = result["count"]

                if count >= 3:  # Expect at least primary key + 2 other indexes
                    logger.info(f"  ✓ {table}: {count} indexes")
                else:
                    logger.warning(f"  ⚠ {table}: only {count} indexes (expected >=3)")
                    self.warnings.append(f"Table {table} has only {count} indexes")

        return True

    async def verify_all(self) -> bool:
        """Run all verification checks."""
        results = []

        results.append(await self.check_migrations())
        results.append(await self.check_tables())
        results.append(await self.check_views())
        results.append(await self.check_indexes())

        return all(results)


async def main():
    """Main entry point."""
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://greenlang:greenlang@localhost:5432/greenlang",
    )

    logger.info("=" * 70)
    logger.info("PACK-027 Migration Verification")
    logger.info("=" * 70)

    verifier = MigrationVerifier(database_url)

    if not await verifier.connect():
        logger.error("\n✗ Cannot proceed without database connection")
        sys.exit(1)

    try:
        success = await verifier.verify_all()

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 70)

        if verifier.errors:
            logger.error(f"\n✗ {len(verifier.errors)} ERRORS FOUND:")
            for error in verifier.errors:
                logger.error(f"  - {error}")

        if verifier.warnings:
            logger.warning(f"\n⚠ {len(verifier.warnings)} WARNINGS:")
            for warning in verifier.warnings:
                logger.warning(f"  - {warning}")

        if success and not verifier.errors:
            logger.info("\n✅ All verifications passed!")
            logger.info("PACK-027 migrations (V166-V180) are correctly applied.")
            exit_code = 0
        else:
            logger.error("\n❌ Verification failed!")
            logger.error("Please review errors above and re-apply migrations if needed.")
            exit_code = 1

    finally:
        await verifier.close()

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
