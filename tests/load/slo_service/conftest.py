# -*- coding: utf-8 -*-
"""
Load test conftest for SLO Service (OBS-005).

Restores real socket access for FastAPI TestClient which needs asyncio
event loops (and thus socket.socketpair()) to function.
"""

from __future__ import annotations

import socket

import pytest

# Capture original socket class at import time, before the integration
# conftest's NetworkBlocker may replace it during session setup.
_ORIGINAL_SOCKET = socket.socket
_ORIGINAL_CREATE_CONNECTION = socket.create_connection


@pytest.fixture(autouse=True)
def restore_sockets_for_load_tests():
    """Restore real sockets so FastAPI TestClient can create event loops.

    The integration conftest blocks all socket access, but FastAPI's
    TestClient (via anyio) requires socket.socketpair() to create an
    event loop.  We restore the real socket constructor for each test.
    """
    saved = socket.socket
    saved_cc = socket.create_connection
    socket.socket = _ORIGINAL_SOCKET
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION
    yield
    socket.socket = saved
    socket.create_connection = saved_cc


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents (not needed for SLO load tests)."""
    yield
