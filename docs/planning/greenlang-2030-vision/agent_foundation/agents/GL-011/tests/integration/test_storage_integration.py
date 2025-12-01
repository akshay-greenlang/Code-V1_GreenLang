# -*- coding: utf-8 -*-
"""
Integration tests for fuel storage connector.

Tests the FuelStorageConnector integration:
- MODBUS protocol communication
- OPC-UA data retrieval
- REST API fallback
- Data synchronization
- Error handling and retry logic
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integrations.fuel_storage_connector import (
    FuelStorageConnector,
    StorageConnectionConfig,
    TankLevel,
    StorageReading
)


class TestFuelStorageConnector:
    """Test suite for FuelStorageConnector integration."""

    @pytest.fixture
    def connector_config(self):
        """Standard connector configuration."""
        return StorageConnectionConfig(
            protocol='modbus',
            host='192.168.1.100',
            port=502,
            timeout_seconds=30,
            retry_attempts=3,
            retry_delay_seconds=5
        )

    @pytest.fixture
    def connector(self, connector_config):
        """Create connector instance."""
        return FuelStorageConnector(connector_config)

    def test_connector_initialization(self, connector):
        """Test connector initializes correctly."""
        assert connector is not None
        assert connector.protocol == 'modbus'
        assert connector.is_connected is False

    @patch('integrations.fuel_storage_connector.ModbusClient')
    def test_modbus_connection(self, mock_modbus, connector):
        """Test MODBUS connection establishment."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_modbus.return_value = mock_client

        result = connector.connect()

        assert result is True
        mock_client.connect.assert_called_once()

    @patch('integrations.fuel_storage_connector.ModbusClient')
    def test_modbus_connection_failure_retry(self, mock_modbus, connector):
        """Test MODBUS connection retries on failure."""
        mock_client = Mock()
        mock_client.connect.side_effect = [False, False, True]  # Fail twice, succeed
        mock_modbus.return_value = mock_client

        result = connector.connect()

        assert result is True
        assert mock_client.connect.call_count == 3

    @patch('integrations.fuel_storage_connector.ModbusClient')
    def test_read_tank_levels(self, mock_modbus, connector):
        """Test reading tank levels via MODBUS."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        # Simulate register values for tank level (0-65535 maps to 0-100%)
        mock_client.read_holding_registers.return_value.registers = [32768]  # ~50%
        mock_modbus.return_value = mock_client

        connector.connect()
        levels = connector.read_tank_levels(['tank_001'])

        assert len(levels) == 1
        assert levels['tank_001'].level_percent >= 40
        assert levels['tank_001'].level_percent <= 60

    @patch('integrations.fuel_storage_connector.ModbusClient')
    def test_read_multiple_tanks(self, mock_modbus, connector):
        """Test reading multiple tanks."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        # Different levels for different tanks
        mock_client.read_holding_registers.side_effect = [
            Mock(registers=[49152]),  # 75%
            Mock(registers=[16384]),  # 25%
            Mock(registers=[65535]),  # 100%
        ]
        mock_modbus.return_value = mock_client

        connector.connect()
        levels = connector.read_tank_levels(['tank_001', 'tank_002', 'tank_003'])

        assert len(levels) == 3
        assert levels['tank_001'].level_percent > 70
        assert levels['tank_002'].level_percent < 30
        assert levels['tank_003'].level_percent > 95

    def test_data_validation(self, connector):
        """Test data validation for tank readings."""
        # Invalid level (over 100%) should be rejected
        with pytest.raises(ValueError):
            TankLevel(
                tank_id='tank_001',
                level_percent=150.0,  # Invalid
                timestamp=datetime.now()
            )

    def test_timestamp_included(self, connector):
        """Test readings include timestamps."""
        reading = StorageReading(
            tank_id='tank_001',
            fuel_type='natural_gas',
            level_percent=75.0,
            volume_m3=1500.0,
            temperature_c=25.0
        )

        assert reading.timestamp is not None
        assert reading.timestamp <= datetime.now()


class TestOPCUAIntegration:
    """Test suite for OPC-UA protocol integration."""

    @pytest.fixture
    def opcua_config(self):
        """OPC-UA connector configuration."""
        return StorageConnectionConfig(
            protocol='opcua',
            host='opc.tcp://192.168.1.100',
            port=4840,
            timeout_seconds=30,
            security_mode='SignAndEncrypt',
            certificate_path='/certs/client.pem'
        )

    @pytest.fixture
    def connector(self, opcua_config):
        """Create OPC-UA connector."""
        return FuelStorageConnector(opcua_config)

    @patch('integrations.fuel_storage_connector.OPCUAClient')
    def test_opcua_connection(self, mock_opcua, connector):
        """Test OPC-UA connection with security."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_opcua.return_value = mock_client

        result = connector.connect()

        assert result is True

    @patch('integrations.fuel_storage_connector.OPCUAClient')
    def test_opcua_node_browsing(self, mock_opcua, connector):
        """Test OPC-UA node browsing for tank data."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_node = Mock()
        mock_node.get_value.return_value = 75.5
        mock_client.get_node.return_value = mock_node
        mock_opcua.return_value = mock_client

        connector.connect()
        levels = connector.read_tank_levels(['tank_001'])

        assert levels['tank_001'].level_percent == 75.5


class TestRESTAPIIntegration:
    """Test suite for REST API fallback integration."""

    @pytest.fixture
    def rest_config(self):
        """REST API connector configuration."""
        return StorageConnectionConfig(
            protocol='rest',
            host='https://api.storage.example.com',
            port=443,
            api_key='test_api_key',
            timeout_seconds=30
        )

    @pytest.fixture
    def connector(self, rest_config):
        """Create REST API connector."""
        return FuelStorageConnector(rest_config)

    @patch('integrations.fuel_storage_connector.requests')
    def test_rest_api_get_levels(self, mock_requests, connector):
        """Test REST API level retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'tanks': [
                {'tank_id': 'tank_001', 'level_percent': 80.0},
                {'tank_id': 'tank_002', 'level_percent': 45.0}
            ]
        }
        mock_requests.get.return_value = mock_response

        levels = connector.read_tank_levels(['tank_001', 'tank_002'])

        assert len(levels) == 2
        assert levels['tank_001'].level_percent == 80.0
        assert levels['tank_002'].level_percent == 45.0

    @patch('integrations.fuel_storage_connector.requests')
    def test_rest_api_authentication(self, mock_requests, connector):
        """Test REST API includes authentication."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'tanks': []}
        mock_requests.get.return_value = mock_response

        connector.read_tank_levels([])

        # Check API key was included in headers
        call_args = mock_requests.get.call_args
        assert 'headers' in call_args.kwargs or 'headers' in call_args[1]


class TestDataSynchronization:
    """Test suite for data synchronization."""

    @pytest.fixture
    def connector(self):
        """Create connector with mock."""
        config = StorageConnectionConfig(
            protocol='modbus',
            host='192.168.1.100',
            port=502
        )
        return FuelStorageConnector(config)

    def test_cache_invalidation(self, connector):
        """Test cache is invalidated on stale data."""
        connector._cache['tank_001'] = StorageReading(
            tank_id='tank_001',
            fuel_type='natural_gas',
            level_percent=50.0,
            volume_m3=1000.0,
            timestamp=datetime.now() - timedelta(minutes=10)  # Old data
        )

        # Old cache should be invalidated
        assert connector._is_cache_valid('tank_001', max_age_seconds=300) is False

    def test_fresh_cache_used(self, connector):
        """Test fresh cache is used."""
        connector._cache['tank_001'] = StorageReading(
            tank_id='tank_001',
            fuel_type='natural_gas',
            level_percent=50.0,
            volume_m3=1000.0,
            timestamp=datetime.now()  # Fresh data
        )

        # Fresh cache should be valid
        assert connector._is_cache_valid('tank_001', max_age_seconds=300) is True

    def test_sync_interval_respected(self, connector):
        """Test synchronization respects polling interval."""
        connector.sync_interval_seconds = 60
        connector._last_sync = datetime.now()

        # Should not sync again immediately
        assert connector._should_sync() is False

        # After interval passed
        connector._last_sync = datetime.now() - timedelta(seconds=120)
        assert connector._should_sync() is True


class TestErrorHandling:
    """Test suite for error handling and recovery."""

    @pytest.fixture
    def connector(self):
        """Create connector."""
        config = StorageConnectionConfig(
            protocol='modbus',
            host='192.168.1.100',
            port=502,
            retry_attempts=3,
            retry_delay_seconds=1
        )
        return FuelStorageConnector(config)

    @patch('integrations.fuel_storage_connector.ModbusClient')
    def test_connection_timeout_handling(self, mock_modbus, connector):
        """Test connection timeout is handled gracefully."""
        from integrations.fuel_storage_connector import ConnectionTimeoutError

        mock_client = Mock()
        mock_client.connect.side_effect = TimeoutError("Connection timed out")
        mock_modbus.return_value = mock_client

        with pytest.raises(ConnectionTimeoutError):
            connector.connect()

    @patch('integrations.fuel_storage_connector.ModbusClient')
    def test_read_error_recovery(self, mock_modbus, connector):
        """Test read error triggers reconnection."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.read_holding_registers.side_effect = [
            Exception("Read failed"),
            Mock(registers=[32768])  # Success on retry
        ]
        mock_modbus.return_value = mock_client

        connector.connect()
        levels = connector.read_tank_levels(['tank_001'])

        # Should recover and return valid data
        assert 'tank_001' in levels

    def test_invalid_protocol_rejection(self):
        """Test invalid protocol is rejected."""
        with pytest.raises(ValueError):
            StorageConnectionConfig(
                protocol='invalid_protocol',
                host='192.168.1.100',
                port=502
            )

    @patch('integrations.fuel_storage_connector.ModbusClient')
    def test_graceful_degradation(self, mock_modbus, connector):
        """Test graceful degradation when connection lost."""
        mock_client = Mock()
        mock_client.connect.return_value = True
        mock_client.read_holding_registers.side_effect = Exception("Connection lost")
        mock_modbus.return_value = mock_client

        connector.connect()

        # Should return last known values or raise appropriate error
        result = connector.read_tank_levels_safe(['tank_001'])

        # Should indicate stale data or error
        assert result is None or result.get('status') == 'stale'


class TestProvenanceTracking:
    """Test suite for provenance tracking in storage data."""

    @pytest.fixture
    def connector(self):
        """Create connector."""
        config = StorageConnectionConfig(
            protocol='modbus',
            host='192.168.1.100',
            port=502
        )
        return FuelStorageConnector(config)

    def test_reading_includes_provenance(self, connector):
        """Test readings include provenance hash."""
        reading = StorageReading(
            tank_id='tank_001',
            fuel_type='natural_gas',
            level_percent=75.0,
            volume_m3=1500.0,
            temperature_c=25.0
        )

        assert reading.provenance_hash is not None
        assert len(reading.provenance_hash) == 64

    def test_provenance_deterministic(self, connector):
        """Test provenance hash is deterministic."""
        reading1 = StorageReading(
            tank_id='tank_001',
            fuel_type='natural_gas',
            level_percent=75.0,
            volume_m3=1500.0,
            temperature_c=25.0,
            timestamp=datetime(2024, 1, 1, 12, 0, 0)
        )

        reading2 = StorageReading(
            tank_id='tank_001',
            fuel_type='natural_gas',
            level_percent=75.0,
            volume_m3=1500.0,
            temperature_c=25.0,
            timestamp=datetime(2024, 1, 1, 12, 0, 0)
        )

        assert reading1.provenance_hash == reading2.provenance_hash
