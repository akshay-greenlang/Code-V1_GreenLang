# -*- coding: utf-8 -*-
"""
Pytest Fixtures for ERP Connector Service Integration Tests (AGENT-DATA-003)
=============================================================================

Provides shared fixtures for integration tests.
Overrides parent conftest's autouse fixtures that are not relevant.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest's mock_agents for ERP connector tests.

    ERP connector service does not use greenlang.agents.registry, so we
    provide a simple no-op override to avoid the AttributeError from
    the parent integration conftest.
    """
    yield {}


@pytest.fixture
def integration_config() -> Dict[str, Any]:
    """Configuration for integration testing."""
    return {
        "service_name": "erp-connector-integration-test",
        "environment": "test",
        "default_erp_system": "simulated",
        "max_connections": 10,
        "sync_batch_size": 500,
        "enable_currency_conversion": True,
        "enable_scope3_mapping": True,
        "enable_emissions_calculation": True,
        "default_emission_methodology": "eeio",
        "default_currency": "USD",
    }
