# -*- coding: utf-8 -*-
"""
Conftest overrides for Observability Agent Service Integration Tests.

Overrides the parent conftest autouse fixtures (mock_agents, block_network)
that conflict with the observability agent self-contained integration tests.
"""

import pytest


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent mock_agents fixture (no-op for observability tests)."""
    yield


@pytest.fixture(scope="session", autouse=True)
def block_network():
    """Override parent block_network fixture (no-op for observability tests)."""
    yield
