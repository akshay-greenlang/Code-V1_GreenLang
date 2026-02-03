# -*- coding: utf-8 -*-
"""
GreenLang Smoke Tests Package
=============================

This package contains smoke tests for post-release validation.
These tests verify that the GreenLang package installs correctly
and core functionality is accessible after publishing to PyPI.

Test Modules:
    - test_release_smoke.py: Main release smoke tests
    - beta_test/: Beta-specific smoke tests

Usage:
    # Run all smoke tests
    pytest tests/smoke/ -v

    # Run release smoke tests only
    pytest tests/smoke/test_release_smoke.py -v

    # Run with specific version
    GL_EXPECTED_VERSION=0.3.0 pytest tests/smoke/ -v

    # Run in strict mode (fail on warnings)
    GL_SMOKE_STRICT=1 pytest tests/smoke/ -v

Environment Variables:
    GL_EXPECTED_VERSION: Expected version to verify
    GL_SMOKE_STRICT: Set to "1" for strict mode
"""

__all__ = ["test_release_smoke"]