# -*- coding: utf-8 -*-
"""
GreenLang Utilities

This package contains various utility modules for GreenLang CLI,
including Windows-specific PATH configuration tools and unit conversion.
"""

__version__ = "0.3.0"

# Import core utilities
from .unit_converter import UnitConverter, UnitConversionError

# Import Windows utilities conditionally
import sys

_exports = ["UnitConverter", "UnitConversionError"]

if sys.platform == "win32":
    try:
        from .windows_path import (
            setup_windows_path,
            diagnose_path_issues,
            find_gl_executable,
            add_to_user_path,
        )
        _exports.extend([
            "setup_windows_path",
            "diagnose_path_issues",
            "find_gl_executable",
            "add_to_user_path",
        ])
    except ImportError:
        # Windows utilities not available
        pass

__all__ = _exports