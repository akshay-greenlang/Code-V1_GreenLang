# -*- coding: utf-8 -*-
"""
Comprehensive tests for SDK Connector abstraction.

Tests cover:
- Connector initialization
- Connection lifecycle (connect/disconnect)
- Read and write operations
- Context manager usage
- Error handling
"""

import pytest
from typing import Any, Optional
from greenlang.sdk.base import Connector, Result


class MockConnector(Connector):
    """Mock connector for testing."""

    def __init__(self, config: Optional[dict] = None, should_fail: bool = False):
        """Initialize mock connector."""
        super().__init__(config)
        self.should_fail = should_fail
        self.connection_attempts = 0
        self.read_count = 0
        self.write_count = 0
        self.data_store = {}

    def connect(self) -> bool:
        """Simulate connection."""
        self.connection_attempts += 1
        if self.should_fail:
            return False
        self.connected = True
        return True

    def disconnect(self) -> bool:
        """Simulate disconnection."""
        if not self.connected:
            return False
        self.connected = False
        return True

    def read(self, query: Any) -> Result:
        """Simulate reading data."""
        if not self.connected:
            return Result(success=False, error="Not connected")

        self.read_count += 1
        if query in self.data_store:
            return Result(success=True, data=self.data_store[query])
        return Result(success=False, error=f"Query not found: {query}")

    def write(self, data: Any) -> Result:
        """Simulate writing data."""
        if not self.connected:
            return Result(success=False, error="Not connected")

        self.write_count += 1
        if isinstance(data, dict) and "key" in data:
            self.data_store[data["key"]] = data["value"]
            return Result(success=True, metadata={"written": True})
        return Result(success=False, error="Invalid data format")


@pytest.mark.unit
class TestConnectorInitialization:
    """Test Connector initialization."""

    def test_connector_default_init(self):
        """Test creating connector with defaults."""
        connector = MockConnector()

        assert connector.config is None
        assert connector.connected is False
        assert connector.logger is not None

    def test_connector_with_config(self):
        """Test creating connector with configuration."""
        config = {"host": "localhost", "port": 5432}
        connector = MockConnector(config=config)

        assert connector.config == config
        assert connector.connected is False

    def test_connector_initial_state(self):
        """Test connector initial state."""
        connector = MockConnector()

        assert not connector.is_connected()
        assert connector.connection_attempts == 0


@pytest.mark.unit
class TestConnectorConnection:
    """Test connector connection lifecycle."""

    def test_connect_success(self):
        """Test successful connection."""
        connector = MockConnector()
        result = connector.connect()

        assert result is True
        assert connector.is_connected()
        assert connector.connection_attempts == 1

    def test_connect_failure(self):
        """Test failed connection."""
        connector = MockConnector(should_fail=True)
        result = connector.connect()

        assert result is False
        assert not connector.is_connected()

    def test_disconnect_when_connected(self):
        """Test disconnecting when connected."""
        connector = MockConnector()
        connector.connect()

        result = connector.disconnect()

        assert result is True
        assert not connector.is_connected()

    def test_disconnect_when_not_connected(self):
        """Test disconnecting when not connected."""
        connector = MockConnector()

        result = connector.disconnect()

        assert result is False

    def test_multiple_connections(self):
        """Test connecting multiple times."""
        connector = MockConnector()

        connector.connect()
        first_state = connector.is_connected()

        connector.connect()
        second_state = connector.is_connected()

        assert first_state is True
        assert second_state is True
        assert connector.connection_attempts == 2

    def test_connect_disconnect_cycle(self):
        """Test multiple connect/disconnect cycles."""
        connector = MockConnector()

        for _ in range(3):
            assert connector.connect() is True
            assert connector.is_connected()
            assert connector.disconnect() is True
            assert not connector.is_connected()


@pytest.mark.unit
class TestConnectorRead:
    """Test connector read operations."""

    def test_read_when_connected(self):
        """Test reading data when connected."""
        connector = MockConnector()
        connector.connect()
        connector.data_store["test"] = {"data": "value"}

        result = connector.read("test")

        assert result.success is True
        assert result.data == {"data": "value"}
        assert connector.read_count == 1

    def test_read_when_not_connected(self):
        """Test reading data when not connected."""
        connector = MockConnector()

        result = connector.read("test")

        assert result.success is False
        assert "Not connected" in result.error

    def test_read_nonexistent_data(self):
        """Test reading data that doesn't exist."""
        connector = MockConnector()
        connector.connect()

        result = connector.read("missing")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_multiple_reads(self):
        """Test multiple read operations."""
        connector = MockConnector()
        connector.connect()
        connector.data_store["key1"] = "value1"
        connector.data_store["key2"] = "value2"

        result1 = connector.read("key1")
        result2 = connector.read("key2")

        assert result1.success is True
        assert result2.success is True
        assert connector.read_count == 2


@pytest.mark.unit
class TestConnectorWrite:
    """Test connector write operations."""

    def test_write_when_connected(self):
        """Test writing data when connected."""
        connector = MockConnector()
        connector.connect()

        data = {"key": "test", "value": "data"}
        result = connector.write(data)

        assert result.success is True
        assert connector.write_count == 1
        assert "test" in connector.data_store

    def test_write_when_not_connected(self):
        """Test writing data when not connected."""
        connector = MockConnector()

        result = connector.write({"key": "test", "value": "data"})

        assert result.success is False
        assert "Not connected" in result.error

    def test_write_invalid_data(self):
        """Test writing invalid data."""
        connector = MockConnector()
        connector.connect()

        result = connector.write("invalid")

        assert result.success is False
        assert "Invalid data" in result.error

    def test_write_and_read_back(self):
        """Test writing data and reading it back."""
        connector = MockConnector()
        connector.connect()

        # Write data
        write_result = connector.write({"key": "mykey", "value": "myvalue"})
        assert write_result.success is True

        # Read it back
        read_result = connector.read("mykey")
        assert read_result.success is True
        assert read_result.data == "myvalue"

    def test_multiple_writes(self):
        """Test multiple write operations."""
        connector = MockConnector()
        connector.connect()

        for i in range(5):
            result = connector.write({"key": f"key{i}", "value": f"value{i}"})
            assert result.success is True

        assert connector.write_count == 5


@pytest.mark.unit
class TestConnectorContextManager:
    """Test connector as context manager."""

    def test_context_manager_connect(self):
        """Test context manager automatically connects."""
        connector = MockConnector()

        with connector:
            assert connector.is_connected()

    def test_context_manager_disconnect(self):
        """Test context manager automatically disconnects."""
        connector = MockConnector()

        with connector:
            assert connector.is_connected()

        assert not connector.is_connected()

    def test_context_manager_read_write(self):
        """Test read/write within context manager."""
        connector = MockConnector()

        with connector:
            write_result = connector.write({"key": "test", "value": "data"})
            assert write_result.success is True

            read_result = connector.read("test")
            assert read_result.success is True
            assert read_result.data == "data"

    def test_context_manager_exception_handling(self):
        """Test context manager disconnects even on exception."""
        connector = MockConnector()

        try:
            with connector:
                assert connector.is_connected()
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still disconnect
        assert not connector.is_connected()

    def test_nested_context_managers(self):
        """Test nested usage of context managers."""
        connector1 = MockConnector()
        connector2 = MockConnector()

        with connector1:
            assert connector1.is_connected()
            with connector2:
                assert connector2.is_connected()
                assert connector1.is_connected()
            assert not connector2.is_connected()
            assert connector1.is_connected()

        assert not connector1.is_connected()
        assert not connector2.is_connected()


@pytest.mark.unit
class TestConnectorIsConnected:
    """Test is_connected method."""

    def test_is_connected_initial_state(self):
        """Test is_connected returns False initially."""
        connector = MockConnector()
        assert connector.is_connected() is False

    def test_is_connected_after_connect(self):
        """Test is_connected returns True after connecting."""
        connector = MockConnector()
        connector.connect()
        assert connector.is_connected() is True

    def test_is_connected_after_disconnect(self):
        """Test is_connected returns False after disconnecting."""
        connector = MockConnector()
        connector.connect()
        connector.disconnect()
        assert connector.is_connected() is False


@pytest.mark.unit
class TestConnectorConfiguration:
    """Test connector configuration handling."""

    def test_connector_with_simple_config(self):
        """Test connector with simple configuration."""
        config = {"timeout": 30}
        connector = MockConnector(config=config)

        assert connector.config["timeout"] == 30

    def test_connector_with_complex_config(self):
        """Test connector with complex configuration."""
        config = {
            "host": "localhost",
            "port": 5432,
            "credentials": {"user": "admin", "password": "secret"},
            "options": {"ssl": True, "timeout": 60},
        }
        connector = MockConnector(config=config)

        assert connector.config == config
        assert connector.config["credentials"]["user"] == "admin"
        assert connector.config["options"]["ssl"] is True

    def test_connector_config_none(self):
        """Test connector with None configuration."""
        connector = MockConnector(config=None)
        assert connector.config is None
