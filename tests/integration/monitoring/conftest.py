# -*- coding: utf-8 -*-
"""
Conftest overrides for monitoring integration tests.

The parent conftest (tests/integration/conftest.py) has autouse fixtures
that patch greenlang.agents.registry.get_agent and block the network.
Monitoring integration tests only validate dashboard JSON files on disk
and do not use the agent registry or network, so we override those
fixtures to be no-ops.
"""
import pytest


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent mock_agents -- monitoring tests do not use agents."""
    yield {}


@pytest.fixture(scope="session", autouse=True)
def block_network():
    """Override parent block_network -- monitoring tests read local JSON only."""
    yield
