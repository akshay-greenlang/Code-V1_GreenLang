# -*- coding: utf-8 -*-
"""
Integration test fixtures for GL-005 CombustionControlAgent.

Provides mock servers and integration test helpers.
"""

import pytest
import asyncio
from typing import Generator
from .mock_servers import MockServerManager


@pytest.fixture(scope="session")
async def mock_server_manager() -> Generator[MockServerManager, None, None]:
    """Provide mock server manager for integration tests."""
    manager = MockServerManager()
    await manager.start_all()
    yield manager
    await manager.stop_all()


@pytest.fixture
async def opcua_server(mock_server_manager):
    """Provide mock OPC UA server."""
    return mock_server_manager.opcua_server


@pytest.fixture
async def modbus_server(mock_server_manager):
    """Provide mock Modbus server."""
    return mock_server_manager.modbus_server


@pytest.fixture
async def mqtt_broker(mock_server_manager):
    """Provide mock MQTT broker."""
    return mock_server_manager.mqtt_broker


@pytest.fixture
async def flame_scanner(mock_server_manager):
    """Provide mock flame scanner server."""
    return mock_server_manager.flame_scanner
