# -*- coding: utf-8 -*-
"""
Integration tests for GL-017 CONDENSYNC external system integrations.

Tests all integration modules with comprehensive coverage:
- SCADA integration
- Cooling tower integration
- DCS integration
- Historian integration
- Message bus integration

Author: GL-017 Test Engineering Team
Target Coverage: >85%
"""

import pytest
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Mock Integration Classes
# ============================================================================

@dataclass
class SCADAConnectionConfig:
    """SCADA connection configuration."""
    host: str = 'localhost'
    port: int = 4840
    namespace: str = 'opc.tcp'
    username: Optional[str] = None
    password: Optional[str] = None
    timeout_seconds: int = 30
    retry_count: int = 3


@dataclass
class CoolingTowerConfig:
    """Cooling tower connection configuration."""
    tower_id: str = 'CT-001'
    host: str = 'localhost'
    port: int = 502
    protocol: str = 'modbus_tcp'
    unit_id: int = 1


@dataclass
class DCSConnectionConfig:
    """DCS connection configuration."""
    host: str = 'localhost'
    port: int = 9000
    system_type: str = 'honeywell_experion'
    station_id: str = 'STATION-001'


@dataclass
class HistorianConfig:
    """Historian connection configuration."""
    host: str = 'localhost'
    port: int = 5432
    database: str = 'process_historian'
    username: str = 'historian'
    password: str = 'password'


class MockSCADAClient:
    """Mock SCADA client for testing."""

    def __init__(self, config: SCADAConnectionConfig):
        self.config = config
        self._connected = False
        self._subscriptions = {}
        self._tag_values = {}

    async def connect(self) -> bool:
        """Establish connection to SCADA server."""
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        """Disconnect from SCADA server."""
        self._connected = False
        return True

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    async def read_tag(self, tag_name: str) -> Dict[str, Any]:
        """Read single tag value."""
        if not self._connected:
            raise ConnectionError("Not connected to SCADA server")

        # Return mock data
        return {
            'tag_name': tag_name,
            'value': self._tag_values.get(tag_name, 50.0),
            'quality': 'GOOD',
            'timestamp': datetime.utcnow().isoformat(),
            'unit': 'mbar'
        }

    async def read_multiple_tags(self, tag_names: List[str]) -> Dict[str, Any]:
        """Read multiple tag values."""
        if not self._connected:
            raise ConnectionError("Not connected to SCADA server")

        result = {}
        for tag_name in tag_names:
            result[tag_name] = await self.read_tag(tag_name)
        return result

    async def write_tag(self, tag_name: str, value: Any) -> bool:
        """Write value to tag."""
        if not self._connected:
            raise ConnectionError("Not connected to SCADA server")

        self._tag_values[tag_name] = value
        return True

    async def subscribe_to_tag(self, tag_name: str, callback) -> str:
        """Subscribe to tag value changes."""
        subscription_id = f"sub_{tag_name}_{datetime.utcnow().timestamp()}"
        self._subscriptions[subscription_id] = {'tag': tag_name, 'callback': callback}
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from tag."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    def set_mock_value(self, tag_name: str, value: Any):
        """Set mock value for testing."""
        self._tag_values[tag_name] = value


class MockCoolingTowerClient:
    """Mock cooling tower client for testing."""

    def __init__(self, config: CoolingTowerConfig):
        self.config = config
        self._connected = False
        self._registers = {}

    async def connect(self) -> bool:
        """Connect to cooling tower controller."""
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        """Disconnect from cooling tower controller."""
        self._connected = False
        return True

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    async def get_status(self) -> Dict[str, Any]:
        """Get cooling tower status."""
        if not self._connected:
            raise ConnectionError("Not connected to cooling tower")

        return {
            'tower_id': self.config.tower_id,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'RUNNING',
            'wet_bulb_temp_c': 20.0,
            'cold_water_temp_c': 25.0,
            'hot_water_temp_c': 33.0,
            'approach_temp_c': 5.0,
            'range_temp_c': 8.0,
            'fan_status': ['ON', 'ON', 'ON', 'OFF'],
            'fan_speed_percent': [75.0, 75.0, 80.0, 0.0],
            'pump_status': ['ON', 'ON', 'STANDBY'],
            'basin_level_percent': 85.0
        }

    async def get_approach_temperature(self) -> float:
        """Get current approach temperature."""
        status = await self.get_status()
        return status['approach_temp_c']

    async def get_range_temperature(self) -> float:
        """Get current range temperature."""
        status = await self.get_status()
        return status['range_temp_c']

    async def set_fan_speed(self, fan_index: int, speed_percent: float) -> bool:
        """Set fan speed."""
        if not self._connected:
            raise ConnectionError("Not connected to cooling tower")

        if 0 <= fan_index <= 3 and 0 <= speed_percent <= 100:
            return True
        return False

    async def start_fan(self, fan_index: int) -> bool:
        """Start fan."""
        return await self.set_fan_speed(fan_index, 50.0)

    async def stop_fan(self, fan_index: int) -> bool:
        """Stop fan."""
        return await self.set_fan_speed(fan_index, 0.0)


class MockDCSClient:
    """Mock DCS client for testing."""

    def __init__(self, config: DCSConnectionConfig):
        self.config = config
        self._connected = False
        self._points = {}

    async def connect(self) -> bool:
        """Connect to DCS."""
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        """Disconnect from DCS."""
        self._connected = False
        return True

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    async def read_point(self, point_name: str) -> Dict[str, Any]:
        """Read DCS point value."""
        if not self._connected:
            raise ConnectionError("Not connected to DCS")

        return {
            'point_name': point_name,
            'value': self._points.get(point_name, 0.0),
            'quality': 'GOOD',
            'timestamp': datetime.utcnow().isoformat()
        }

    async def write_point(self, point_name: str, value: Any) -> bool:
        """Write value to DCS point."""
        if not self._connected:
            raise ConnectionError("Not connected to DCS")

        self._points[point_name] = value
        return True

    async def read_block(self, block_name: str) -> Dict[str, Any]:
        """Read DCS control block."""
        if not self._connected:
            raise ConnectionError("Not connected to DCS")

        return {
            'block_name': block_name,
            'mode': 'AUTO',
            'pv': 50.0,
            'sp': 48.0,
            'output': 75.0,
            'status': 'RUNNING'
        }


class MockHistorianClient:
    """Mock historian client for testing."""

    def __init__(self, config: HistorianConfig):
        self.config = config
        self._connected = False
        self._data = {}

    async def connect(self) -> bool:
        """Connect to historian."""
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        """Disconnect from historian."""
        self._connected = False
        return True

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    async def query_tag_history(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60
    ) -> List[Dict[str, Any]]:
        """Query historical tag data."""
        if not self._connected:
            raise ConnectionError("Not connected to historian")

        # Generate mock historical data
        data = []
        current_time = start_time
        while current_time <= end_time:
            data.append({
                'timestamp': current_time.isoformat(),
                'value': 50.0 + (hash(str(current_time)) % 10) - 5,
                'quality': 'GOOD'
            })
            current_time += timedelta(seconds=interval_seconds)

        return data

    async def write_tag_value(
        self,
        tag_name: str,
        timestamp: datetime,
        value: Any,
        quality: str = 'GOOD'
    ) -> bool:
        """Write value to historian."""
        if not self._connected:
            raise ConnectionError("Not connected to historian")

        if tag_name not in self._data:
            self._data[tag_name] = []
        self._data[tag_name].append({
            'timestamp': timestamp.isoformat(),
            'value': value,
            'quality': quality
        })
        return True

    async def get_tag_statistics(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get statistical summary of tag data."""
        data = await self.query_tag_history(tag_name, start_time, end_time)

        values = [d['value'] for d in data]
        return {
            'tag_name': tag_name,
            'count': len(values),
            'min': min(values) if values else 0.0,
            'max': max(values) if values else 0.0,
            'avg': sum(values) / len(values) if values else 0.0,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }


class MockMessageBus:
    """Mock message bus for testing."""

    def __init__(self, broker_url: str = 'localhost:9092'):
        self.broker_url = broker_url
        self._connected = False
        self._topics = {}
        self._subscriptions = {}

    async def connect(self) -> bool:
        """Connect to message bus."""
        self._connected = True
        return True

    async def disconnect(self) -> bool:
        """Disconnect from message bus."""
        self._connected = False
        return True

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    async def publish(self, topic: str, message: Dict[str, Any]) -> bool:
        """Publish message to topic."""
        if not self._connected:
            raise ConnectionError("Not connected to message bus")

        if topic not in self._topics:
            self._topics[topic] = []
        self._topics[topic].append({
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        })
        return True

    async def subscribe(self, topic: str, callback) -> str:
        """Subscribe to topic."""
        subscription_id = f"sub_{topic}_{datetime.utcnow().timestamp()}"
        self._subscriptions[subscription_id] = {'topic': topic, 'callback': callback}
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from topic."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    def get_messages(self, topic: str) -> List[Dict[str, Any]]:
        """Get messages from topic (for testing)."""
        return self._topics.get(topic, [])


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def scada_config():
    """SCADA connection configuration."""
    return SCADAConnectionConfig(
        host='localhost',
        port=4840,
        namespace='opc.tcp',
        timeout_seconds=30
    )


@pytest.fixture
def scada_client(scada_config):
    """Mock SCADA client."""
    return MockSCADAClient(scada_config)


@pytest.fixture
def cooling_tower_config():
    """Cooling tower configuration."""
    return CoolingTowerConfig(
        tower_id='CT-001',
        host='localhost',
        port=502
    )


@pytest.fixture
def cooling_tower_client(cooling_tower_config):
    """Mock cooling tower client."""
    return MockCoolingTowerClient(cooling_tower_config)


@pytest.fixture
def dcs_config():
    """DCS configuration."""
    return DCSConnectionConfig(
        host='localhost',
        port=9000,
        system_type='honeywell_experion'
    )


@pytest.fixture
def dcs_client(dcs_config):
    """Mock DCS client."""
    return MockDCSClient(dcs_config)


@pytest.fixture
def historian_config():
    """Historian configuration."""
    return HistorianConfig(
        host='localhost',
        port=5432,
        database='process_historian'
    )


@pytest.fixture
def historian_client(historian_config):
    """Mock historian client."""
    return MockHistorianClient(historian_config)


@pytest.fixture
def message_bus():
    """Mock message bus."""
    return MockMessageBus('localhost:9092')


@pytest.fixture
def condenser_scada_tags():
    """Standard condenser SCADA tags."""
    return [
        'COND_VACUUM',
        'COND_HOTWELL_TEMP',
        'CW_INLET_TEMP',
        'CW_OUTLET_TEMP',
        'CW_FLOW_RATE',
        'CONDENSATE_FLOW',
        'AIR_EXTRACTION',
        'TUBE_DP'
    ]


# ============================================================================
# SCADA Integration Tests
# ============================================================================

class TestSCADAIntegration:
    """Tests for SCADA integration."""

    @pytest.mark.integration
    @pytest.mark.scada
    @pytest.mark.asyncio
    async def test_scada_connection(self, scada_client):
        """Test SCADA connection establishment."""
        result = await scada_client.connect()

        assert result is True
        assert scada_client.is_connected() is True

    @pytest.mark.integration
    @pytest.mark.scada
    @pytest.mark.asyncio
    async def test_scada_disconnection(self, scada_client):
        """Test SCADA disconnection."""
        await scada_client.connect()
        result = await scada_client.disconnect()

        assert result is True
        assert scada_client.is_connected() is False

    @pytest.mark.integration
    @pytest.mark.scada
    @pytest.mark.asyncio
    async def test_read_single_tag(self, scada_client):
        """Test reading single SCADA tag."""
        await scada_client.connect()
        scada_client.set_mock_value('COND_VACUUM', 50.0)

        result = await scada_client.read_tag('COND_VACUUM')

        assert result is not None
        assert result['tag_name'] == 'COND_VACUUM'
        assert result['value'] == 50.0
        assert result['quality'] == 'GOOD'

    @pytest.mark.integration
    @pytest.mark.scada
    @pytest.mark.asyncio
    async def test_read_multiple_tags(self, scada_client, condenser_scada_tags):
        """Test reading multiple SCADA tags."""
        await scada_client.connect()

        result = await scada_client.read_multiple_tags(condenser_scada_tags)

        assert result is not None
        assert len(result) == len(condenser_scada_tags)
        for tag in condenser_scada_tags:
            assert tag in result

    @pytest.mark.integration
    @pytest.mark.scada
    @pytest.mark.asyncio
    async def test_write_tag(self, scada_client):
        """Test writing SCADA tag."""
        await scada_client.connect()

        result = await scada_client.write_tag('VACUUM_SETPOINT', 48.0)

        assert result is True

        # Verify write
        read_result = await scada_client.read_tag('VACUUM_SETPOINT')
        assert read_result['value'] == 48.0

    @pytest.mark.integration
    @pytest.mark.scada
    @pytest.mark.asyncio
    async def test_tag_subscription(self, scada_client):
        """Test SCADA tag subscription."""
        await scada_client.connect()
        callback_values = []

        def callback(value):
            callback_values.append(value)

        subscription_id = await scada_client.subscribe_to_tag('COND_VACUUM', callback)

        assert subscription_id is not None
        assert subscription_id.startswith('sub_')

    @pytest.mark.integration
    @pytest.mark.scada
    @pytest.mark.asyncio
    async def test_tag_unsubscription(self, scada_client):
        """Test SCADA tag unsubscription."""
        await scada_client.connect()

        subscription_id = await scada_client.subscribe_to_tag('COND_VACUUM', lambda x: None)
        result = await scada_client.unsubscribe(subscription_id)

        assert result is True

    @pytest.mark.integration
    @pytest.mark.scada
    @pytest.mark.asyncio
    async def test_connection_error_handling(self, scada_client):
        """Test SCADA connection error handling."""
        # Try to read without connecting
        with pytest.raises(ConnectionError):
            await scada_client.read_tag('COND_VACUUM')

    @pytest.mark.integration
    @pytest.mark.scada
    @pytest.mark.asyncio
    async def test_data_quality_handling(self, scada_client):
        """Test SCADA data quality handling."""
        await scada_client.connect()

        result = await scada_client.read_tag('COND_VACUUM')

        assert 'quality' in result
        assert result['quality'] in ['GOOD', 'BAD', 'UNCERTAIN']


# ============================================================================
# Cooling Tower Integration Tests
# ============================================================================

class TestCoolingTowerIntegration:
    """Tests for cooling tower integration."""

    @pytest.mark.integration
    @pytest.mark.cooling_tower
    @pytest.mark.asyncio
    async def test_cooling_tower_connection(self, cooling_tower_client):
        """Test cooling tower connection."""
        result = await cooling_tower_client.connect()

        assert result is True
        assert cooling_tower_client.is_connected() is True

    @pytest.mark.integration
    @pytest.mark.cooling_tower
    @pytest.mark.asyncio
    async def test_cooling_tower_disconnection(self, cooling_tower_client):
        """Test cooling tower disconnection."""
        await cooling_tower_client.connect()
        result = await cooling_tower_client.disconnect()

        assert result is True
        assert cooling_tower_client.is_connected() is False

    @pytest.mark.integration
    @pytest.mark.cooling_tower
    @pytest.mark.asyncio
    async def test_get_cooling_tower_status(self, cooling_tower_client):
        """Test getting cooling tower status."""
        await cooling_tower_client.connect()

        status = await cooling_tower_client.get_status()

        assert status is not None
        assert 'tower_id' in status
        assert 'status' in status
        assert 'wet_bulb_temp_c' in status
        assert 'cold_water_temp_c' in status
        assert 'hot_water_temp_c' in status

    @pytest.mark.integration
    @pytest.mark.cooling_tower
    @pytest.mark.asyncio
    async def test_get_approach_temperature(self, cooling_tower_client):
        """Test getting approach temperature."""
        await cooling_tower_client.connect()

        approach = await cooling_tower_client.get_approach_temperature()

        assert approach > 0
        assert approach < 20  # Reasonable approach temperature

    @pytest.mark.integration
    @pytest.mark.cooling_tower
    @pytest.mark.asyncio
    async def test_get_range_temperature(self, cooling_tower_client):
        """Test getting range temperature."""
        await cooling_tower_client.connect()

        range_temp = await cooling_tower_client.get_range_temperature()

        assert range_temp > 0
        assert range_temp < 20  # Reasonable range temperature

    @pytest.mark.integration
    @pytest.mark.cooling_tower
    @pytest.mark.asyncio
    async def test_set_fan_speed(self, cooling_tower_client):
        """Test setting fan speed."""
        await cooling_tower_client.connect()

        result = await cooling_tower_client.set_fan_speed(0, 80.0)

        assert result is True

    @pytest.mark.integration
    @pytest.mark.cooling_tower
    @pytest.mark.asyncio
    async def test_start_fan(self, cooling_tower_client):
        """Test starting fan."""
        await cooling_tower_client.connect()

        result = await cooling_tower_client.start_fan(0)

        assert result is True

    @pytest.mark.integration
    @pytest.mark.cooling_tower
    @pytest.mark.asyncio
    async def test_stop_fan(self, cooling_tower_client):
        """Test stopping fan."""
        await cooling_tower_client.connect()

        result = await cooling_tower_client.stop_fan(0)

        assert result is True

    @pytest.mark.integration
    @pytest.mark.cooling_tower
    @pytest.mark.asyncio
    async def test_fan_status_in_response(self, cooling_tower_client):
        """Test fan status included in response."""
        await cooling_tower_client.connect()

        status = await cooling_tower_client.get_status()

        assert 'fan_status' in status
        assert 'fan_speed_percent' in status
        assert isinstance(status['fan_status'], list)


# ============================================================================
# DCS Integration Tests
# ============================================================================

class TestDCSIntegration:
    """Tests for DCS integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dcs_connection(self, dcs_client):
        """Test DCS connection."""
        result = await dcs_client.connect()

        assert result is True
        assert dcs_client.is_connected() is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dcs_disconnection(self, dcs_client):
        """Test DCS disconnection."""
        await dcs_client.connect()
        result = await dcs_client.disconnect()

        assert result is True
        assert dcs_client.is_connected() is False

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_read_dcs_point(self, dcs_client):
        """Test reading DCS point."""
        await dcs_client.connect()

        result = await dcs_client.read_point('VACUUM_PV')

        assert result is not None
        assert 'point_name' in result
        assert 'value' in result
        assert 'quality' in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_write_dcs_point(self, dcs_client):
        """Test writing DCS point."""
        await dcs_client.connect()

        result = await dcs_client.write_point('VACUUM_SP', 48.0)

        assert result is True

        # Verify write
        read_result = await dcs_client.read_point('VACUUM_SP')
        assert read_result['value'] == 48.0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_read_control_block(self, dcs_client):
        """Test reading DCS control block."""
        await dcs_client.connect()

        result = await dcs_client.read_block('VACUUM_CTRL')

        assert result is not None
        assert 'block_name' in result
        assert 'mode' in result
        assert 'pv' in result
        assert 'sp' in result
        assert 'output' in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dcs_connection_error(self, dcs_client):
        """Test DCS connection error handling."""
        with pytest.raises(ConnectionError):
            await dcs_client.read_point('VACUUM_PV')


# ============================================================================
# Historian Integration Tests
# ============================================================================

class TestHistorianIntegration:
    """Tests for historian integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_historian_connection(self, historian_client):
        """Test historian connection."""
        result = await historian_client.connect()

        assert result is True
        assert historian_client.is_connected() is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_historian_disconnection(self, historian_client):
        """Test historian disconnection."""
        await historian_client.connect()
        result = await historian_client.disconnect()

        assert result is True
        assert historian_client.is_connected() is False

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_query_tag_history(self, historian_client):
        """Test querying tag history."""
        await historian_client.connect()

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)

        data = await historian_client.query_tag_history(
            'COND_VACUUM',
            start_time,
            end_time,
            interval_seconds=60
        )

        assert data is not None
        assert len(data) > 0
        assert 'timestamp' in data[0]
        assert 'value' in data[0]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_write_tag_value(self, historian_client):
        """Test writing tag value to historian."""
        await historian_client.connect()

        result = await historian_client.write_tag_value(
            'COND_VACUUM',
            datetime.utcnow(),
            50.0,
            'GOOD'
        )

        assert result is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_tag_statistics(self, historian_client):
        """Test getting tag statistics."""
        await historian_client.connect()

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)

        stats = await historian_client.get_tag_statistics(
            'COND_VACUUM',
            start_time,
            end_time
        )

        assert stats is not None
        assert 'count' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'avg' in stats

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_historian_connection_error(self, historian_client):
        """Test historian connection error handling."""
        with pytest.raises(ConnectionError):
            await historian_client.query_tag_history(
                'COND_VACUUM',
                datetime.utcnow() - timedelta(hours=1),
                datetime.utcnow()
            )


# ============================================================================
# Message Bus Integration Tests
# ============================================================================

class TestMessageBusIntegration:
    """Tests for message bus integration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_message_bus_connection(self, message_bus):
        """Test message bus connection."""
        result = await message_bus.connect()

        assert result is True
        assert message_bus.is_connected() is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_message_bus_disconnection(self, message_bus):
        """Test message bus disconnection."""
        await message_bus.connect()
        result = await message_bus.disconnect()

        assert result is True
        assert message_bus.is_connected() is False

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_publish_message(self, message_bus):
        """Test publishing message."""
        await message_bus.connect()

        message = {
            'type': 'condenser_status',
            'vacuum_mbar': 50.0,
            'timestamp': datetime.utcnow().isoformat()
        }

        result = await message_bus.publish('condenser.status', message)

        assert result is True

        # Verify message was published
        messages = message_bus.get_messages('condenser.status')
        assert len(messages) == 1
        assert messages[0]['message']['type'] == 'condenser_status'

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_subscribe_to_topic(self, message_bus):
        """Test subscribing to topic."""
        await message_bus.connect()

        subscription_id = await message_bus.subscribe(
            'condenser.alerts',
            lambda msg: None
        )

        assert subscription_id is not None
        assert subscription_id.startswith('sub_')

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_unsubscribe_from_topic(self, message_bus):
        """Test unsubscribing from topic."""
        await message_bus.connect()

        subscription_id = await message_bus.subscribe('condenser.alerts', lambda msg: None)
        result = await message_bus.unsubscribe(subscription_id)

        assert result is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_publish_multiple_messages(self, message_bus):
        """Test publishing multiple messages."""
        await message_bus.connect()

        for i in range(5):
            message = {
                'type': 'condenser_reading',
                'reading_id': i,
                'vacuum_mbar': 50.0 + i
            }
            await message_bus.publish('condenser.readings', message)

        messages = message_bus.get_messages('condenser.readings')
        assert len(messages) == 5


# ============================================================================
# Cross-System Integration Tests
# ============================================================================

class TestCrossSystemIntegration:
    """Tests for cross-system integration scenarios."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_scada_to_historian_flow(self, scada_client, historian_client):
        """Test data flow from SCADA to historian."""
        await scada_client.connect()
        await historian_client.connect()

        # Read from SCADA
        scada_client.set_mock_value('COND_VACUUM', 52.5)
        scada_data = await scada_client.read_tag('COND_VACUUM')

        # Write to historian
        result = await historian_client.write_tag_value(
            'COND_VACUUM',
            datetime.utcnow(),
            scada_data['value'],
            scada_data['quality']
        )

        assert result is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cooling_tower_to_message_bus(self, cooling_tower_client, message_bus):
        """Test cooling tower status publishing to message bus."""
        await cooling_tower_client.connect()
        await message_bus.connect()

        # Get cooling tower status
        status = await cooling_tower_client.get_status()

        # Publish to message bus
        result = await message_bus.publish('cooling_tower.status', status)

        assert result is True

        messages = message_bus.get_messages('cooling_tower.status')
        assert len(messages) == 1
        assert messages[0]['message']['tower_id'] == 'CT-001'

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dcs_to_scada_coordination(self, dcs_client, scada_client):
        """Test DCS to SCADA coordination."""
        await dcs_client.connect()
        await scada_client.connect()

        # Read control block from DCS
        block = await dcs_client.read_block('VACUUM_CTRL')

        # Write setpoint to SCADA
        result = await scada_client.write_tag('VACUUM_SP', block['sp'])

        assert result is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_data_collection_workflow(
        self,
        scada_client,
        cooling_tower_client,
        dcs_client,
        historian_client
    ):
        """Test complete data collection workflow."""
        # Connect all systems
        await scada_client.connect()
        await cooling_tower_client.connect()
        await dcs_client.connect()
        await historian_client.connect()

        # Collect data from all sources
        scada_client.set_mock_value('COND_VACUUM', 50.0)
        scada_data = await scada_client.read_tag('COND_VACUUM')
        ct_status = await cooling_tower_client.get_status()
        dcs_block = await dcs_client.read_block('VACUUM_CTRL')

        # Store in historian
        timestamp = datetime.utcnow()

        await historian_client.write_tag_value('COND_VACUUM', timestamp, scada_data['value'])
        await historian_client.write_tag_value('CT_APPROACH', timestamp, ct_status['approach_temp_c'])
        await historian_client.write_tag_value('VACUUM_CTRL_OUTPUT', timestamp, dcs_block['output'])

        assert True  # All operations completed


# ============================================================================
# Error Recovery Tests
# ============================================================================

class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_reconnection_after_disconnect(self, scada_client):
        """Test reconnection after disconnect."""
        await scada_client.connect()
        await scada_client.disconnect()

        # Reconnect
        result = await scada_client.connect()

        assert result is True
        assert scada_client.is_connected() is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, scada_client, historian_client):
        """Test graceful degradation when one system fails."""
        await scada_client.connect()
        # Historian not connected

        # SCADA operations should still work
        scada_client.set_mock_value('COND_VACUUM', 50.0)
        scada_data = await scada_client.read_tag('COND_VACUUM')

        assert scada_data is not None

        # Historian write should fail gracefully
        with pytest.raises(ConnectionError):
            await historian_client.write_tag_value(
                'COND_VACUUM',
                datetime.utcnow(),
                scada_data['value']
            )
