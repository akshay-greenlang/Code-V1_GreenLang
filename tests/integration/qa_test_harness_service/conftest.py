# -*- coding: utf-8 -*-
"""
Pytest Fixtures for QA Test Harness Integration Tests (AGENT-FOUND-009)
========================================================================

Provides shared fixtures for end-to-end integration testing of the QA
Test Harness service components working together.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_qa_test_harness_env(monkeypatch):
    """Remove any GL_QA_TEST_HARNESS_ env vars between tests."""
    prefix = "GL_QA_TEST_HARNESS_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents fixture to avoid patching non-existent attributes."""
    yield
