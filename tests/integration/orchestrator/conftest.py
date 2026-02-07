# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Orchestrator Integration Tests (AGENT-FOUND-001)
=====================================================================

Provides integration-level fixtures with mock agent registries,
async helpers for Windows compatibility, and socket restoration.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# Capture original socket at import time before NetworkBlocker
_ORIGINAL_SOCKET = socket.socket
_ORIGINAL_CREATE_CONNECTION = socket.create_connection


# ---------------------------------------------------------------------------
# Async helper for Windows compatibility
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run async coroutine synchronously. Windows-compatible."""
    if sys.platform == "win32":
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    else:
        return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Override parent conftest fixtures that interfere
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_sockets():
    """Restore real sockets for async tests."""
    saved = socket.socket
    saved_cc = socket.create_connection
    socket.socket = _ORIGINAL_SOCKET
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION
    yield
    socket.socket = saved
    socket.create_connection = saved_cc


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents (not needed for orchestrator tests)."""
    yield


# ---------------------------------------------------------------------------
# Integration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_async():
    """Provide _run_async helper as a fixture."""
    return _run_async


@pytest.fixture
def mock_agent_registry():
    """Create a mock agent registry that returns deterministic agents."""
    agents: Dict[str, MagicMock] = {}

    def _make_agent(agent_id: str) -> MagicMock:
        mock = MagicMock()
        mock.agent_id = agent_id
        mock.run.return_value = {
            "success": True,
            "data": {"output": f"result_from_{agent_id}"},
        }
        mock.run_async = None
        return mock

    def _get_agent(agent_id: str) -> MagicMock:
        if agent_id not in agents:
            agents[agent_id] = _make_agent(agent_id)
        return agents[agent_id]

    return _get_agent


@pytest.fixture
def failing_agent_registry():
    """Agent registry where specific agents fail N times then succeed."""
    fail_counts: Dict[str, int] = {}
    call_counts: Dict[str, int] = {}

    def _configure(agent_id: str, fail_count: int):
        fail_counts[agent_id] = fail_count
        call_counts[agent_id] = 0

    def _get_agent(agent_id: str) -> MagicMock:
        mock = MagicMock()
        mock.agent_id = agent_id
        mock.run_async = None

        def _run(ctx):
            call_counts[agent_id] = call_counts.get(agent_id, 0) + 1
            if call_counts[agent_id] <= fail_counts.get(agent_id, 0):
                raise RuntimeError(f"Simulated failure #{call_counts[agent_id]}")
            return {"success": True, "data": {"output": f"result_from_{agent_id}"}}

        mock.run.side_effect = _run
        return mock

    return _get_agent, _configure
