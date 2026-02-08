# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Agent Registry Service Integration Tests (AGENT-FOUND-007)
==============================================================================

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
    """Override parent conftest's mock_agents for agent registry tests.

    Agent registry service does not use greenlang.agents.registry, so we
    provide a simple no-op override to avoid the AttributeError from
    the parent integration conftest.
    """
    yield {}


@pytest.fixture
def integration_config() -> Dict[str, Any]:
    """Configuration for integration testing."""
    return {
        "service_name": "agent-registry-integration-test",
        "environment": "test",
        "health_check_enabled": True,
        "hot_reload_enabled": True,
        "dependency_resolution_enabled": True,
        "provenance_enabled": True,
        "max_agents": 500,
    }
