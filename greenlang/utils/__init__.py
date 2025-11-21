# -*- coding: utf-8 -*-
"""
GreenLang Utilities

This package contains various utility modules for GreenLang CLI,
including Windows-specific PATH configuration tools.
"""

__version__ = "0.3.0"

# Import Windows utilities conditionally
import sys

if sys.platform == "win32":
    try:
        from .windows_path import (
            setup_windows_path,
            diagnose_path_issues,
            find_gl_executable,
            add_to_user_path,
        )
        __all__ = [
            "setup_windows_path",
            "diagnose_path_issues",
            "find_gl_executable",
            "add_to_user_path",
        ]
    except ImportError:
        # Windows utilities not available
        __all__ = []
else:
    __all__ = []