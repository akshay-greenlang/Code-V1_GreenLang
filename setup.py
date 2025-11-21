#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for GreenLang CLI

This file is kept for legacy compatibility and pip editable installs.
The main package configuration is in pyproject.toml.
"""

import sys
import platform
from setuptools import setup, find_packages
from setuptools.command.install import install
from pathlib import Path

# Version is now set in pyproject.toml
VERSION = "0.3.0"

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "GreenLang Climate Intelligence Platform - Enterprise Infrastructure"

class PostInstallCommand(install):
    """Custom post-installation command for Windows PATH setup."""

    def run(self):
        install.run(self)

        # Run Windows-specific setup only on Windows
        if platform.system() == "Windows":
            try:
                self.setup_windows_path()
            except Exception as e:
                print(f"Warning: Windows PATH setup failed: {e}")
                print("You may need to manually add Python Scripts to PATH")

    def setup_windows_path(self):
        """Set up Windows PATH for gl command."""
        try:
            from greenlang.utils.post_install import run_post_install
            run_post_install()
        except ImportError:
            print("Post-install setup not available")

# Main setup configuration is in pyproject.toml
# This just provides the custom command class
setup(
    version=VERSION,
    cmdclass={
        'install': PostInstallCommand,
    },
    # All other configuration comes from pyproject.toml
)