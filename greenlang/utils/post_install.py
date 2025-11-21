# -*- coding: utf-8 -*-
"""
Post-installation script for GreenLang CLI

This script runs automatically after pip installation to configure
Windows PATH and ensure the 'gl' command is available.
"""

import sys
import os
from pathlib import Path


def run_post_install():
    """Run post-installation setup."""
    print("GreenLang CLI: Running post-installation setup...")

    # Only run on Windows
    if sys.platform != "win32":
        print("Post-install setup only needed on Windows.")
        return

    try:
        # Import Windows PATH utilities
        from .windows_path import setup_windows_path, diagnose_path_issues

        # Try to set up PATH
        success, message = setup_windows_path()

        if success:
            print(f"✓ {message}")
            print("GreenLang CLI should now be available via 'gl' command.")
            print("You may need to restart your command prompt.")
        else:
            print(f"⚠ {message}")
            print("\nTo manually fix this issue:")
            print("1. Find your Python Scripts directory")
            print("2. Add it to your Windows PATH environment variable")
            print("3. Or use: python -m greenlang.cli instead of 'gl'")

    except ImportError as e:
        print(f"Could not import PATH utilities: {e}")
        print("Manual PATH configuration may be required.")

    except Exception as e:
        print(f"Post-install setup failed: {e}")
        print("The 'gl' command may not be available.")
        print("Use 'python -m greenlang.cli' as an alternative.")


def create_batch_wrapper():
    """Create a batch wrapper in a location that's likely to be in PATH."""
    if sys.platform != "win32":
        return

    try:
        # Try to find a good location for the batch file
        python_dir = Path(sys.executable).parent
        scripts_dir = python_dir / "Scripts"

        if scripts_dir.exists():
            batch_file = scripts_dir / "gl.bat"

            # Only create if it doesn't exist
            if not batch_file.exists():
                batch_content = '''@echo off
REM GreenLang CLI Wrapper
REM Auto-generated during installation

REM Try direct execution first
where gl.exe >nul 2>&1
if %errorlevel% equ 0 (
    gl.exe %*
    exit /b %errorlevel%
)

REM Fallback to Python module
python -m greenlang.cli %*
'''
                batch_file.write_text(batch_content)
                print(f"✓ Created batch wrapper: {batch_file}")

    except Exception as e:
        print(f"Could not create batch wrapper: {e}")


if __name__ == "__main__":
    run_post_install()
    create_batch_wrapper()