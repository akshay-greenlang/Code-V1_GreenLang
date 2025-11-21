#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Next RC Version Generator
=========================

Automatically derives the next RC version from existing git tags.
Supports both sequential numbering and week-based RC versions.
"""

import subprocess
import sys
import re
from datetime import datetime
from typing import List, Optional, Tuple
from greenlang.determinism import DeterministicClock


def get_git_tags() -> List[str]:
    """Get all git tags from the repository"""
    try:
        result = subprocess.run(
            ["git", "tag", "-l"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n') if result.stdout else []
    except subprocess.CalledProcessError:
        print("Error: Failed to get git tags. Ensure you're in a git repository.")
        sys.exit(1)


def parse_rc_version(tag: str) -> Optional[Tuple[str, str, str, int]]:
    """
    Parse RC version tag

    Returns:
        Tuple of (major, minor, patch, rc_number) or None if not an RC tag
    """
    # Pattern: v0.2.0-rc.1 or v0.2.0-rc.2025w40
    pattern = r'^v?(\d+)\.(\d+)\.(\d+)-rc\.(\d+|202\dw\d+)$'
    match = re.match(pattern, tag)

    if match:
        major, minor, patch, rc = match.groups()
        # If week-based, extract week number
        if 'w' in rc:
            rc_num = int(rc.split('w')[1])
        else:
            rc_num = int(rc)
        return (major, minor, patch, rc_num)
    return None


def get_current_week() -> str:
    """Get current year and week in format: 2025w40"""
    now = DeterministicClock.now()
    year = now.year
    week = now.isocalendar()[1]
    return f"{year}w{week:02d}"


def get_next_rc_version(base_version: str, format_type: str = "sequential") -> str:
    """
    Generate next RC version

    Args:
        base_version: Base version (e.g., "0.3.0")
        format_type: "sequential" or "weekly"

    Returns:
        Next RC version string
    """
    tags = get_git_tags()

    # Parse base version
    version_parts = base_version.lstrip('v').split('.')
    if len(version_parts) != 3:
        print(f"Error: Invalid base version format: {base_version}")
        sys.exit(1)

    major, minor, patch = version_parts

    # Find existing RC tags for this version
    rc_tags = []
    for tag in tags:
        parsed = parse_rc_version(tag)
        if parsed:
            tag_major, tag_minor, tag_patch, rc_num = parsed
            if tag_major == major and tag_minor == minor and tag_patch == patch:
                rc_tags.append(rc_num)

    if format_type == "weekly":
        # Use year-week format
        return f"v{major}.{minor}.{patch}-rc.{get_current_week()}"
    else:
        # Sequential numbering
        if rc_tags:
            next_rc = max(rc_tags) + 1
        else:
            next_rc = 1
        return f"v{major}.{minor}.{patch}-rc.{next_rc}"


def get_latest_version() -> str:
    """Get the latest version tag (non-RC)"""
    tags = get_git_tags()

    # Filter for version tags (not RC)
    version_pattern = r'^v?(\d+)\.(\d+)\.(\d+)$'
    versions = []

    for tag in tags:
        match = re.match(version_pattern, tag)
        if match:
            versions.append(tag)

    if not versions:
        # Default to 0.1.0 if no versions exist
        return "0.1.0"

    # Sort versions and get latest
    versions.sort(key=lambda v: list(map(int, v.lstrip('v').split('.'))))
    return versions[-1].lstrip('v')


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate next RC version for GreenLang"
    )
    parser.add_argument(
        "--base-version",
        help="Base version (e.g., 0.3.0). If not specified, uses latest tag."
    )
    parser.add_argument(
        "--format",
        choices=["sequential", "weekly"],
        default="sequential",
        help="RC version format: sequential (rc.1, rc.2) or weekly (rc.2025w40)"
    )
    parser.add_argument(
        "--create-tag",
        action="store_true",
        help="Create the git tag (requires confirmation)"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the tag to origin (requires --create-tag)"
    )

    args = parser.parse_args()

    # Get base version
    if args.base_version:
        base_version = args.base_version
    else:
        base_version = get_latest_version()
        print(f"Using latest version as base: {base_version}")

    # Generate next RC version
    next_rc = get_next_rc_version(base_version, args.format)
    print(f"Next RC version: {next_rc}")

    # Create tag if requested
    if args.create_tag:
        response = input(f"Create tag '{next_rc}'? [y/N]: ")
        if response.lower() == 'y':
            try:
                # Create annotated tag
                message = f"Release candidate {next_rc}"
                subprocess.run(
                    ["git", "tag", "-a", next_rc, "-m", message],
                    check=True
                )
                print(f"✅ Created tag: {next_rc}")

                # Push if requested
                if args.push:
                    subprocess.run(
                        ["git", "push", "origin", next_rc],
                        check=True
                    )
                    print(f"✅ Pushed tag to origin: {next_rc}")
                else:
                    print(f"ℹ️  Tag created locally. Push with: git push origin {next_rc}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to create/push tag: {e}")
                sys.exit(1)

    # Output just the version for CI consumption
    if not sys.stdout.isatty():  # Running in CI/script
        print(next_rc)


if __name__ == "__main__":
    main()