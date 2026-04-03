# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Access Guard Service Integration Tests (AGENT-FOUND-006)
============================================================================

Provides shared fixtures for integration tests.

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
        "service_name": "access-guard-integration-test",
        "environment": "test",
        "strict_mode": True,
        "simulation_mode": False,
        "rate_limiting_enabled": True,
        "audit_enabled": True,
        "strict_tenant_isolation": True,
    }
