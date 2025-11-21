# -*- coding: utf-8 -*-
"""
Generate Golden Fixtures for Connector Testing
===============================================

Creates deterministic golden fixtures for multiple regions and scenarios.

Usage:
    python scripts/generate_golden_fixtures.py

Output:
    - tests/goldens/connectors/grid/mock_{REGION}_{DATE}_{DURATION}.snap.json
    - tests/goldens/connectors/grid/mock_{REGION}_{DATE}_{DURATION}.sha256
"""

import asyncio
import hashlib
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from greenlang.connectors.grid.mock import GridIntensityMockConnector
from greenlang.connectors.models import GridIntensityQuery, TimeWindow, REGION_METADATA
from greenlang.connectors.context import ConnectorContext


async def generate_fixture(region: str, start: datetime, hours: int, output_dir: Path):
    """
    Generate a single golden fixture.

    Args:
        region: Region code (e.g., "US-CAISO")
        start: Start datetime (UTC)
        hours: Duration in hours
        output_dir: Output directory for fixtures

    Returns:
        Path to generated snapshot file
    """
    connector = GridIntensityMockConnector()

    query = GridIntensityQuery(
        region=region,
        window=TimeWindow(
            start=start,
            end=start + timedelta(hours=hours),
            resolution="hour"
        )
    )

    ctx = ConnectorContext.for_record("grid/intensity/mock")
    payload, provenance = await connector.fetch(query, ctx)

    # Generate snapshot bytes
    snapshot_bytes = connector.snapshot(payload, provenance)

    # Save snapshot
    filename = f"mock_{region}_{start.strftime('%Y-%m-%d')}_{hours}h.snap.json"
    filepath = output_dir / filename
    filepath.write_bytes(snapshot_bytes)

    # Save SHA-256
    sha256_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    sha_filepath = output_dir / f"{filename}.sha256"
    sha_filepath.write_text(sha256_hash)

    print(f"[OK] Generated: {filename} (hash: {sha256_hash[:16]}...)")
    return filepath


async def main():
    """Generate all golden fixtures"""
    output_dir = Path("tests/goldens/connectors/grid")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

    # Define scenarios
    scenarios = [
        # US-CAISO
        ("US-CAISO", 24),
        ("US-CAISO", 48),

        # EU-DE
        ("EU-DE", 24),
        ("EU-DE", 48),

        # IN-NO
        ("IN-NO", 24),
        ("IN-NO", 48),

        # UK-GB
        ("UK-GB", 24),
        ("UK-GB", 48),

        # US-PJM
        ("US-PJM", 24),
        ("US-PJM", 48),

        # CA-ON weekly (already have 24h)
        ("CA-ON", 48),
        ("CA-ON", 168),
    ]

    print("=" * 60)
    print("GENERATING GOLDEN FIXTURES")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Start date: {start_date.isoformat()}")
    print()

    # Generate all fixtures
    generated_files = []
    for region, hours in scenarios:
        filepath = await generate_fixture(region, start_date, hours, output_dir)
        generated_files.append(filepath)

    print()
    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"[SUCCESS] Generated {len(scenarios)} new golden fixtures")
    print(f"[LOCATION] {output_dir.absolute()}")

    # List all files in the directory (including existing)
    all_fixtures = sorted(output_dir.glob("*.snap.json"))
    all_sha_files = sorted(output_dir.glob("*.sha256"))

    print()
    print(f"Total fixtures in directory: {len(all_fixtures)}")
    print(f"Total SHA-256 files: {len(all_sha_files)}")

    # Verify regions covered
    regions_covered = set()
    for filepath in all_fixtures:
        # Extract region from filename: mock_{REGION}_{DATE}_{DURATION}.snap.json
        parts = filepath.stem.split("_")
        if len(parts) >= 4:
            region = parts[1]
            regions_covered.add(region)

    print()
    print(f"Regions covered: {', '.join(sorted(regions_covered))}")

    return output_dir


if __name__ == "__main__":
    asyncio.run(main())
