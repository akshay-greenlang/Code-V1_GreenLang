# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Normalizer Service Integration Tests (AGENT-FOUND-003)
===========================================================================

Provides shared fixtures for integration tests. Overrides the parent
integration conftest's mock_agents fixture since the normalizer service
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
        "service_name": "normalizer-integration-test",
        "environment": "test",
        "precision": 10,
        "gwp_version": "AR6",
        "cache_enabled": False,
    }
