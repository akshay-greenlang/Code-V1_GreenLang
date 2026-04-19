# -*- coding: utf-8 -*-
"""
Backward-compatible setup.py for greenlang-factors-sdk.

Modern builds use pyproject.toml. This file exists for compatibility
with older pip versions and editable installs.

Package: greenlang-factors-sdk
Version: 1.0.0
Entry point: greenlang.factors.sdk (or greenlang_factors_sdk)
Dependencies: none (stdlib only)
Python: >=3.9
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
