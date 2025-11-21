#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Synchronize version numbers across the project
"""

import re
import sys
from pathlib import Path


def get_pyproject_version():
    """Extract version from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        sys.exit(1)
    
    content = pyproject_path.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        print("Error: Version not found in pyproject.toml")
        sys.exit(1)
    
    return match.group(1)


def update_init_version(version):
    """Update version in greenlang/__init__.py"""
    init_path = Path("greenlang/__init__.py")
    if not init_path.exists():
        print("Error: greenlang/__init__.py not found")
        return False
    
    content = init_path.read_text()
    new_content = re.sub(
        r'^__version__\s*=\s*"[^"]+"',
        f'__version__ = "{version}"',
        content,
        flags=re.MULTILINE
    )
    
    if new_content != content:
        init_path.write_text(new_content)
        print(f"Updated greenlang/__init__.py to version {version}")
        return True
    return False


def update_cli_version(version):
    """Update version references in CLI"""
    cli_path = Path("greenlang/cli/main.py")
    if cli_path.exists():
        content = cli_path.read_text(encoding='utf-8')
        # Update any hardcoded version strings
        new_content = re.sub(
            r'version_text = "v[\d\.]+ - ',
            f'version_text = "v{version} - ',
            content
        )
        if new_content != content:
            cli_path.write_text(new_content, encoding='utf-8')
            print(f"Updated CLI version references to {version}")


def main():
    """Main version sync function"""
    print("Synchronizing version numbers...")
    
    # Get the authoritative version from pyproject.toml
    version = get_pyproject_version()
    print(f"Found version: {version}")
    
    # Update all other locations
    updated = False
    updated |= update_init_version(version)
    update_cli_version(version)
    
    if updated:
        print("[OK] Version synchronization complete")
    else:
        print("[OK] All versions already in sync")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())