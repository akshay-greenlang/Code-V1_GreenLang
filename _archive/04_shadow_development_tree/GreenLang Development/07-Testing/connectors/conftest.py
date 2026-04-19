# -*- coding: utf-8 -*-
"""
Connector-specific test fixtures
=================================

Overrides global fixtures to allow asyncio internals for connector tests.
"""

import pytest
import socket


@pytest.fixture(autouse=True)
def _allow_asyncio_sockets(monkeypatch, request):
    """
    Override global socket guard to allow asyncio internals

    The global conftest blocks all socket operations to prevent network access.
    However, asyncio needs internal socket pairs for event loop operations.

    This fixture allows asyncio internals while still blocking real network.
    It runs before other fixtures due to being defined in a child conftest.
    """
    # Skip if this is an integration/e2e test that needs full network
    if any(mark in request.keywords for mark in ("integration", "e2e", "network")):
        return

    # Store original socket functions BEFORE global conftest patches them
    import importlib
    importlib.reload(socket)
    original_socket = socket.socket
    original_socketpair = socket.socketpair

    # Monkeypatch to restore for asyncio internals
    monkeypatch.setattr("socket.socket", original_socket, raising=False)
    monkeypatch.setattr("socket.socketpair", original_socketpair, raising=False)

    # Still block actual network connections
    def guard_create_connection(*args, **kwargs):
        raise RuntimeError("Network access disabled in connector tests")

    monkeypatch.setattr("socket.create_connection", guard_create_connection, raising=False)
