"""
Root-level conftest.py for test path configuration.

This file is loaded by pytest before any tests are collected,
ensuring the backend module is in sys.path for imports.
"""

import os
import sys

# Add backend to path for imports - must be done before any test collection
_backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if _backend_path not in sys.path:
    sys.path.insert(0, _backend_path)
