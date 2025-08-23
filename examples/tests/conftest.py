"""Pytest configuration for example tests."""

import pytest
import socket

def pytest_configure(config):
    """Add custom markers for example tests."""
    config.addinivalue_line("markers", "example: human-friendly example tests")

@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Deterministic examples: no network access."""
    def guard(*args, **kwargs):
        raise RuntimeError("Network access disabled in examples tests")
    for name in ("create_connection", "socketpair"):
        monkeypatch.setattr(socket, name, guard, raising=True)