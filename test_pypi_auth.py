#!/usr/bin/env python
"""Test PyPI authentication and diagnose upload issues"""

import subprocess
import sys
import os

def test_auth():
    print("=" * 60)
    print("PyPI Authentication Test & Diagnosis")
    print("=" * 60)
    print()

    # Check if .pypirc exists
    pypirc_path = os.path.expanduser("~/.pypirc")
    if os.path.exists(pypirc_path):
        print("✓ Found .pypirc file at:", pypirc_path)
        print("  Note: If token is saved there, it might be incorrect")
    else:
        print("✗ No .pypirc file found (this is OK)")
    print()

    # Check package name availability
    print("Checking if 'greenlang' name is available on PyPI...")
    import urllib.request
    import json

    try:
        with urllib.request.urlopen("https://pypi.org/pypi/greenlang/json") as response:
            data = json.loads(response.read())
            print("✗ Package 'greenlang' already exists on PyPI!")
            print(f"  Latest version: {data['info']['version']}")
            print(f"  Owner: {data['info']['author']}")
            print()
            print("IMPORTANT: You cannot upload if you're not the owner!")
            print("Solutions:")
            print("1. If you own this package, make sure your token has access")
            print("2. If you don't own it, you need to use a different name")
            return False
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("✓ Package name 'greenlang' is available!")
        else:
            print(f"? Could not check package: {e}")
    print()

    # Test with verbose mode
    print("Let's try upload with verbose mode to see detailed error...")
    print("=" * 60)
    print()
    print("Run this command to see detailed error:")
    print()
    print("python -m twine upload --verbose dist\\greenlang-0.2.0-py3-none-any.whl")
    print()
    print("=" * 60)

    return True

def check_alternative_names():
    """Suggest alternative package names"""
    print()
    print("Alternative Package Names You Could Use:")
    print("-" * 40)

    alternatives = [
        "greenlang-cli",
        "greenlang-framework",
        "greenlang-ai",
        "greenlang-core",
        "greenlang-official",
        "pygreenlang",
        "greenlang-sdk"
    ]

    import urllib.request

    for name in alternatives:
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json") as response:
                print(f"✗ {name} - Already taken")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"✓ {name} - Available!")

    print()
    print("To use a different name:")
    print("1. Update 'name' in pyproject.toml")
    print("2. Rebuild: python -m build")
    print("3. Upload with new name")

if __name__ == "__main__":
    if test_auth():
        print("\nNext steps:")
        print("1. Make sure you're the owner of 'greenlang' on PyPI")
        print("2. Or choose a different package name")
    else:
        check_alternative_names()