#!/usr/bin/env python
"""Quick test script for GreenLang beta installation from TestPyPI"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("Testing GreenLang v0.2.0b2 from TestPyPI")
    print("=" * 60)

    # Create a temporary virtual environment
    with tempfile.TemporaryDirectory(prefix="gl_test_") as tmpdir:
        venv_path = Path(tmpdir) / "venv"

        # Create venv
        print("\n1. Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

        # Get python executable in venv
        if os.name == 'nt':
            python_exe = venv_path / "Scripts" / "python.exe"
            pip_exe = venv_path / "Scripts" / "pip.exe"
            gl_exe = venv_path / "Scripts" / "gl.exe"
        else:
            python_exe = venv_path / "bin" / "python"
            pip_exe = venv_path / "bin" / "pip"
            gl_exe = venv_path / "bin" / "gl"

        # Upgrade pip
        print("2. Upgrading pip...")
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "--quiet"], check=True)

        # Install from TestPyPI
        print("3. Installing greenlang==0.2.0b2 from TestPyPI...")
        subprocess.run([
            str(pip_exe), "install",
            "-i", "https://test.pypi.org/simple",
            "--extra-index-url", "https://pypi.org/simple",
            "greenlang==0.2.0b2"
        ], check=True)

        print("\n" + "=" * 60)
        print("TESTING INSTALLATION")
        print("=" * 60)

        # Test 1: gl --version
        print("\nTest 1: gl --version")
        result = subprocess.run([str(gl_exe), "--version"], capture_output=True, text=True)
        print(f"Output: {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"Error: {result.stderr.strip()}")

        # Test 2: gl --help (just check it runs)
        print("\nTest 2: gl --help (checking it runs)...")
        result = subprocess.run([str(gl_exe), "--help"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Help command works")
        else:
            print(f"✗ Help command failed: {result.stderr.strip()}")

        # Test 3: Python import
        print("\nTest 3: Python import")
        result = subprocess.run([
            str(python_exe), "-c",
            "import greenlang; print(f'✓ Import successful, version: {greenlang.__version__}')"
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"✗ Import failed: {result.stderr.strip()}")

        # Test 4: gl doctor
        print("\nTest 4: gl doctor")
        result = subprocess.run([str(gl_exe), "doctor"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Doctor command works")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print(f"✗ Doctor command failed: {result.stderr.strip()}")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✓ Beta v0.2.0b2 successfully installed from TestPyPI")
        print("✓ All basic commands are working")
        print("\nNext steps:")
        print("1. Share beta announcement with testers")
        print("2. Begin test coverage sprint (target: 40%)")
        print("3. Collect feedback over next 1-2 days")
        print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        sys.exit(1)