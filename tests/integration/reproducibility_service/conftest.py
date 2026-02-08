# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Reproducibility Service Integration Tests (AGENT-FOUND-008)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_reproducibility_env(monkeypatch):
    """Remove any GL_REPRODUCIBILITY_ env vars between tests."""
    prefix = "GL_REPRODUCIBILITY_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents to avoid registry patching errors.

    The parent integration conftest patches greenlang.agents.registry.get_agent
    which may not exist. This override yields None to keep tests self-contained.
    """
    yield None
