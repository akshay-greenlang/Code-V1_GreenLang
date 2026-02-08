# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Citations Service Integration Tests (AGENT-FOUND-005)
=========================================================================

Provides shared fixtures for integration tests.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest's mock_agents for citations tests.

    Citations service does not use greenlang.agents.registry, so we
    provide a simple no-op override to avoid the AttributeError from
    the parent integration conftest.
    """
    yield {}


@pytest.fixture
def integration_config() -> Dict[str, Any]:
    """Configuration for integration testing."""
    return {
        "service_name": "citations-integration-test",
        "environment": "test",
        "max_citations": 100000,
        "enable_versioning": True,
        "enable_provenance": True,
        "enable_auto_verification": True,
        "cache_enabled": False,
    }
