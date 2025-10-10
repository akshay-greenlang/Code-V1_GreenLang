"""
Golden Fixtures Report
======================

Generates a comprehensive report of all golden fixtures.

Usage:
    python scripts/fixture_report.py
"""

import hashlib
from pathlib import Path
from collections import defaultdict


def analyze_fixtures():
    """Analyze all golden fixtures and generate report"""
    fixtures_dir = Path("tests/goldens/connectors/grid")

    if not fixtures_dir.exists():
        print(f"[ERROR] Directory not found: {fixtures_dir}")
        return

    # Find all snapshot files
    snapshot_files = sorted(fixtures_dir.glob("*.snap.json"))
    sha_files = sorted(fixtures_dir.glob("*.sha256"))

    # Organize by region and duration
    by_region = defaultdict(list)
    by_duration = defaultdict(list)

    for snapshot_path in snapshot_files:
        # Parse filename: mock_{REGION}_{DATE}_{DURATION}.snap.json
        # Example: mock_CA-ON_2025-01-01_24h.snap.json
        name_without_ext = snapshot_path.name.replace('.snap.json', '')
        parts = name_without_ext.split("_")
        if len(parts) >= 4:
            region = parts[1]
            duration = parts[3]  # e.g., "24h", "48h", "168h"

            file_size = snapshot_path.stat().st_size

            by_region[region].append({
                "filename": snapshot_path.name,
                "duration": duration,
                "size": file_size
            })

            by_duration[duration].append({
                "filename": snapshot_path.name,
                "region": region,
                "size": file_size
            })

    # Calculate totals
    total_snap_size = sum(f.stat().st_size for f in snapshot_files)
    total_sha_size = sum(f.stat().st_size for f in sha_files)
    total_size = total_snap_size + total_sha_size

    # Print report
    print("=" * 70)
    print("GOLDEN FIXTURES COMPREHENSIVE REPORT")
    print("=" * 70)
    print(f"Directory: {fixtures_dir.absolute()}")
    print()

    print("SUMMARY")
    print("-" * 70)
    print(f"Total fixtures:        {len(snapshot_files)}")
    print(f"Total SHA-256 files:   {len(sha_files)}")
    print(f"Total regions:         {len(by_region)}")
    print(f"Total scenarios:       {len(by_duration)}")
    print()

    print("DISK SPACE")
    print("-" * 70)
    print(f"Snapshot files (.snap.json):  {total_snap_size:,} bytes ({total_snap_size / 1024:.1f} KB)")
    print(f"SHA-256 files (.sha256):      {total_sha_size:,} bytes")
    print(f"Total:                        {total_size:,} bytes ({total_size / 1024:.1f} KB)")
    print()

    print("REGIONS COVERED")
    print("-" * 70)
    for region in sorted(by_region.keys()):
        fixtures = by_region[region]
        region_size = sum(f["size"] for f in fixtures)
        durations = sorted([f["duration"] for f in fixtures])
        print(f"  {region:12} | {len(fixtures)} fixtures | {', '.join(durations):20} | {region_size / 1024:6.1f} KB")
    print()

    print("SCENARIOS BY DURATION")
    print("-" * 70)
    for duration in sorted(by_duration.keys(), key=lambda x: int(x[:-1])):
        fixtures = by_duration[duration]
        scenario_size = sum(f["size"] for f in fixtures)
        regions = sorted([f["region"] for f in fixtures])
        print(f"  {duration:6} | {len(fixtures)} fixtures | {', '.join(regions)}")
        print(f"         | Avg size: {scenario_size / len(fixtures) / 1024:.1f} KB | Total: {scenario_size / 1024:.1f} KB")
    print()

    print("ALL FIXTURES")
    print("-" * 70)
    for snapshot_path in snapshot_files:
        size = snapshot_path.stat().st_size
        sha_path = fixtures_dir / f"{snapshot_path.name}.sha256"

        if sha_path.exists():
            hash_value = sha_path.read_text().strip()[:16]
            status = "[OK]"
        else:
            hash_value = "MISSING"
            status = "[FAIL]"

        print(f"  {status} {snapshot_path.name:45} | {size:6,} bytes | {hash_value}...")

    print()
    print("=" * 70)
    print("[SUCCESS] Report generated successfully")
    print("=" * 70)


if __name__ == "__main__":
    analyze_fixtures()
