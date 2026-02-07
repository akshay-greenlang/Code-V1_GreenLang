# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Assumptions Service Integration Tests (AGENT-FOUND-004)
============================================================================

Provides shared fixtures for integration tests. Overrides the parent
integration conftest's mock_agents fixture since the assumptions service
tests do not use the agent registry.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest's mock_agents for assumptions tests.

    Assumptions service does not use greenlang.agents.registry, so we
    provide a simple no-op override to avoid the AttributeError from
    the parent integration conftest.
    """
    yield {}


@pytest.fixture
def integration_config() -> Dict[str, Any]:
    """Configuration for integration testing."""
    return {
        "service_name": "assumptions-integration-test",
        "environment": "test",
        "max_versions": 50,
        "enable_validation": True,
        "enable_provenance": True,
        "cache_enabled": False,
    }
