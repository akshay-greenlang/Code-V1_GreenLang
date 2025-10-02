"""
Intelligence test configuration.

Override network guard for provider tests since they use mocked clients.
"""
import pytest
import socket as socket_module


@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch):
    """Override parent conftest to allow sockets for asyncio event loops.

    Provider tests use fully mocked HTTP clients (no real network calls),
    but asyncio requires socket support for internal event loop operation on Windows.

    We still block actual network libraries while allowing socket creation.
    """
    def mock_network_call(*args, **kwargs):
        raise RuntimeError("Network calls are disabled in tests")

    # Allow socket.socket for asyncio but block network libraries
    # Do NOT block socket.socket since asyncio needs it for event loops

    # Disable common network libraries
    monkeypatch.setattr("urllib.request.urlopen", mock_network_call, raising=False)
    monkeypatch.setattr("urllib.request.Request", mock_network_call, raising=False)

    # Disable httpx if it's imported (skip on Python 3.13 compatibility issues)
    try:
        import httpx
        monkeypatch.setattr("httpx.Client.request", mock_network_call, raising=False)
        monkeypatch.setattr("httpx.AsyncClient.request", mock_network_call, raising=False)
    except (ImportError, TypeError):
        # TypeError can occur on Python 3.13 with httpx compatibility issues
        pass
