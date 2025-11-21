#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Development environment setup script for GreenLang
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Error running: {cmd}")
            print(f"Output: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running {cmd}: {e}")
        return False


def setup_development_environment():
    """Set up the complete development environment"""
    
    print("=" * 60)
    print("GreenLang Development Environment Setup")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 10):
        print(f"❌ Python 3.10+ required. Current: {python_version.major}.{python_version.minor}")
        sys.exit(1)
    print(f"✅ Python {python_version.major}.{python_version.minor} detected")
    
    # Upgrade pip
    print("\n📦 Upgrading pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Install development dependencies
    print("\n📦 Installing development dependencies...")
    run_command(f"{sys.executable} -m pip install -e .[dev]")
    
    # Install pre-commit hooks
    print("\n🔗 Setting up pre-commit hooks...")
    run_command("pip install pre-commit")
    run_command("pre-commit install")
    
    # Create necessary directories
    print("\n📁 Creating directory structure...")
    dirs_to_create = [
        "agents/templates/basic",
        "agents/templates/industry",
        "agents/templates/advanced",
        "agents/templates/custom",
        "agents/examples",
        "agents/tests",
        "datasets/emission_factors/global",
        "datasets/emission_factors/regional",
        "datasets/emission_factors/industry",
        "datasets/benchmarks/buildings",
        "datasets/benchmarks/industrial",
        "datasets/benchmarks/transport",
        "datasets/reference/ghg_protocol",
        "datasets/reference/ipcc",
        "datasets/reference/iso",
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created {len(dirs_to_create)} directories")
    
    # Run initial tests
    print("\n🧪 Running smoke tests...")
    if run_command(f"{sys.executable} -m pytest tests/unit -k 'not slow' --maxfail=1 -q"):
        print("✅ Smoke tests passed")
    else:
        print("⚠️  Some tests failed (this might be expected for first setup)")
    
    # Verify CLI
    print("\n🔧 Verifying CLI installation...")
    if run_command("gl --version"):
        print("✅ CLI is working")
    else:
        print("⚠️  CLI not available in PATH yet")
    
    # Platform-specific notes
    print("\n📝 Platform-specific notes:")
    if platform.system() == "Windows":
        print("   - On Windows, use 'greenlang.bat' or 'python -m greenlang.cli.main'")
    else:
        print("   - On Unix systems, 'greenlang' command should work directly")
    
    print("\n" + "=" * 60)
    print("✅ Development environment setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'tox' to test across Python versions")
    print("2. Run 'gl --help' to see available commands")
    print("3. Check docs/ for documentation")
    print("4. Run 'pytest' to run all tests")
    print("5. Create a feature branch and start developing!")
    

if __name__ == "__main__":
    setup_development_environment()