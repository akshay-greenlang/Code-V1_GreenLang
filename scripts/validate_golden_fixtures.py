# -*- coding: utf-8 -*-
"""
Validate Golden Fixtures
=========================

Verifies that all golden fixtures have correct SHA-256 hashes.

Usage:
    python scripts/validate_golden_fixtures.py
"""

import hashlib
from pathlib import Path


def validate_fixture(snapshot_path: Path) -> tuple[bool, str]:
    """
    Validate a single fixture's SHA-256 hash.

    Args:
        snapshot_path: Path to .snap.json file

    Returns:
        Tuple of (is_valid, message)
    """
    # Read snapshot bytes
    snapshot_bytes = snapshot_path.read_bytes()

    # Compute actual hash
    actual_hash = hashlib.sha256(snapshot_bytes).hexdigest()

    # Check for corresponding .sha256 file
    sha_path = snapshot_path.parent / f"{snapshot_path.name}.sha256"

    if not sha_path.exists():
        return False, f"Missing SHA-256 file: {sha_path.name}"

    # Read expected hash
    expected_hash = sha_path.read_text().strip()

    # Compare
    if actual_hash == expected_hash:
        return True, f"[OK] {snapshot_path.name} (hash: {actual_hash[:16]}...)"
    else:
        return False, (
            f"[FAIL] {snapshot_path.name}\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}"
        )


def main():
    """Validate all golden fixtures"""
    fixtures_dir = Path("tests/goldens/connectors/grid")

    if not fixtures_dir.exists():
        print(f"[ERROR] Directory not found: {fixtures_dir}")
        return 1

    # Find all snapshot files
    snapshot_files = sorted(fixtures_dir.glob("*.snap.json"))

    if not snapshot_files:
        print(f"[ERROR] No snapshot files found in {fixtures_dir}")
        return 1

    print("=" * 60)
    print("VALIDATING GOLDEN FIXTURES")
    print("=" * 60)
    print(f"Directory: {fixtures_dir.absolute()}")
    print(f"Total fixtures: {len(snapshot_files)}")
    print()

    # Validate each fixture
    all_valid = True
    for snapshot_path in snapshot_files:
        is_valid, message = validate_fixture(snapshot_path)
        print(message)

        if not is_valid:
            all_valid = False

    print()
    print("=" * 60)

    if all_valid:
        print("[SUCCESS] All fixtures validated successfully!")
        print(f"Total validated: {len(snapshot_files)}")
        return 0
    else:
        print("[FAILURE] Some fixtures failed validation")
        return 1


if __name__ == "__main__":
    exit(main())
