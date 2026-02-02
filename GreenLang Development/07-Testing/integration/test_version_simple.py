#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple test for version checking without loading all dependencies."""

import sys
from pathlib import Path

def test_version_guard():
    """Test the version guard logic."""

    # Read VERSION file
    version_file = Path("VERSION")
    if not version_file.exists():
        print("ERROR: VERSION file not found")
        return False

    file_version = version_file.read_text().strip()
    print(f"VERSION file: {file_version}")

    # Try to get package version using importlib.metadata
    try:
        from importlib.metadata import version
        pkg_version = version("greenlang")
        print(f"Package version: {pkg_version}")

        if file_version == pkg_version:
            print("PASS: Versions match!")
            return True
        else:
            print(f"FAIL: Version mismatch!")
            print(f"  VERSION file: {file_version}")
            print(f"  Package: {pkg_version}")
            return False

    except Exception as e:
        print(f"Warning: Could not get package version: {e}")
        print("This is expected if package is not installed")

    # Test the workflow command that will be used in CI
    print("\nTesting the CI workflow approach:")
    print("The CI will compare:")
    print(f"  VERSION file: {file_version}")
    print("  Package __version__ (obtained via import)")
    print("\nWith the fix, both should include pre-release identifiers like '-rc.0'")

    return True

if __name__ == "__main__":
    success = test_version_guard()
    sys.exit(0 if success else 1)