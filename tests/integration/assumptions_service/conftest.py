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

import pytest


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
