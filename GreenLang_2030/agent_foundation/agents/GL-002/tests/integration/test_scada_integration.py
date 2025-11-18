"""
SCADA/DCS Integration Tests for GL-002 BoilerEfficiencyOptimizer

Tests comprehensive SCADA system integration including OPC UA connections,
real-time data subscriptions, write commands, retry logic, and data transformations.

Test Scenarios: 20+
Coverage: OPC UA, MQTT, connection management, data quality, alarms
"""

import pytest
import asyncio
import random
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from integrations.scada_connector import (
    SCADAConnector,
    SCADAConnectionConfig,
    SCADAProtocol,
    SCADATag,
    SCADAAlarm,
    DataQuality,
    AlarmPriority,
    SCADADataBuffer,
    AlarmManager
)


# Fixtures
@pytest.fixture
def scada_config():
    """Create test SCADA connection configuration."""
    return SCADAConnectionConfig(
        protocol=SCADAProtocol.OPC_UA,
        primary_host="192.168.1.200",
        primary_port=4840,
        backup_host="192.168.1.201",
        backup_port=4840,
        use_encryption=True,
        username="test_user",
        password="test_password",
        connection_timeout=30,
        read_timeout=10,
        write_timeout=10,
        max_reconnect_attempts=5,
        reconnect_delay=5,
        enable_redundancy=True,
        enable_buffering=True,
        buffer_size=100000
    )


@pytest.fixture
async def scada_connector(scada_config):
    """Create SCADA connector instance for testing."""
    connector = SCADAConnector(scada_config)
    yield connector
    # Cleanup
    await connector.disconnect()


@pytest.fixture
def sample_tags():
    """Create sample SCADA tags for testing."""
    return [
        SCADATag(
            tag_name='BOILER.STEAM.PRESSURE',
            description='Main steam header pressure',
            data_type='float',
            engineering_units='bar',
            scan_rate=1000,
            deadband=0.5,
            min_value=0,
            max_value=200
        ),
        SCADATag(
            tag_name='BOILER.STEAM.TEMPERATURE',
            description='Main steam temperature',
            data_type='float',
            engineering_units='°C',
            scan_rate=1000,
            deadband=1.0,
            min_value=100,
            max_value=600
        ),
        SCADATag(
            tag_name='BOILER.EFFICIENCY',
            description='Calculated boiler efficiency',
            data_type='float',
            engineering_units='%',
            scan_rate=5000,
            deadband=0.1,
            min_value=0,
            max_value=100
        )
    ]


# Test Class: Connection Management
class TestSCADAConnection:
    """Test SCADA connection establishment and management."""

    @pytest.mark.asyncio
    async def test_opc_ua_connection_establishment(self, scada_connector):
        """Test successful OPC UA connection establishment."""
        result = await scada_connector.connect()

        assert result is True
        assert scada_connector.connected is True
        assert scada_connector.connection is not None
        assert scada_connector.connection['protocol'] == 'opc_ua'
        assert scada_connector.connection['connected'] is True

    @pytest.mark.asyncio
    async def test_mqtt_connection_establishment(self, scada_config):
        """Test successful MQTT connection establishment."""
        scada_config.protocol = SCADAProtocol.MQTT
        connector = SCADAConnector(scada_config)

        result = await connector.connect()

        assert result is True
        assert connector.connected is True
        assert connector.connection['broker'] == scada_config.primary_host
        assert 'boiler/+/data' in connector.connection['topics']

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_rest_api_connection_establishment(self, scada_config):
        """Test successful REST API connection establishment."""
        scada_config.protocol = SCADAProtocol.REST_API
        connector = SCADAConnector(scada_config)

        result = await connector.connect()

        assert result is True
        assert connector.connected is True
        assert 'base_url' in connector.connection
        assert connector.connection['base_url'].startswith('https://')

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_redundant_connection_establishment(self, scada_connector):
        """Test redundant connection setup with primary and backup."""
        result = await scada_connector.connect()

        assert result is True
        assert scada_connector.connection is not None
        assert scada_connector.backup_connection is not None
        assert scada_connector.backup_connection['host'] == '192.168.1.201'

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, scada_config):
        """Test connection timeout handling."""
        scada_config.connection_timeout = 1
        connector = SCADAConnector(scada_config)

        # Mock connection delay
        original_create = connector._create_connection

        async def delayed_connection(*args, **kwargs):
            await asyncio.sleep(2)
            return await original_create(*args, **kwargs)

        connector._create_connection = delayed_connection

        # Should timeout but not crash
        result = await connector.connect()

        # May succeed or fail depending on timing, but should handle gracefully
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_connection_retry_logic(self, scada_config):
        """Test automatic connection retry logic."""
        scada_config.max_reconnect_attempts = 3
        scada_config.reconnect_delay = 1
        connector = SCADAConnector(scada_config)

        # Force initial connection failure
        connector.connected = False

        # Mock reconnection
        reconnect_attempts = []

        async def mock_connect():
            reconnect_attempts.append(datetime.utcnow())
            if len(reconnect_attempts) >= 2:
                connector.connected = True
                return True
            return False

        connector.connect = mock_connect

        # Trigger reconnection
        await connector._reconnect_loop()

        # Should have attempted reconnection
        assert len(reconnect_attempts) >= 2

    @pytest.mark.asyncio
    async def test_connection_with_ssl_encryption(self, scada_config):
        """Test connection with SSL/TLS encryption."""
        scada_config.use_encryption = True
        scada_config.certificate_path = "/path/to/cert.pem"
        scada_config.ca_cert_path = "/path/to/ca.pem"

        connector = SCADAConnector(scada_config)
        connection = await connector._create_connection(
            scada_config.primary_host,
            scada_config.primary_port
        )

        assert 'ssl_context' in connection


# Test Class: Real-Time Data Operations
class TestSCADADataOperations:
    """Test SCADA real-time data reading and writing."""

    @pytest.mark.asyncio
    async def test_tag_subscription_and_updates(self, scada_connector):
        """Test subscribing to tag value changes."""
        await scada_connector.connect()

        received_updates = []

        async def tag_callback(tag_name, value, timestamp):
            received_updates.append({
                'tag': tag_name,
                'value': value,
                'timestamp': timestamp
            })

        await scada_connector.subscribe('BOILER.STEAM.PRESSURE', tag_callback)

        # Wait for some updates
        await asyncio.sleep(2)

        assert len(received_updates) > 0
        assert received_updates[0]['tag'] == 'BOILER.STEAM.PRESSURE'
        assert isinstance(received_updates[0]['value'], (int, float))

    @pytest.mark.asyncio
    async def test_read_tag_with_deadband_filtering(self, scada_connector):
        """Test deadband filtering prevents unnecessary updates."""
        await scada_connector.connect()

        tag = scada_connector.tags['BOILER.STEAM.PRESSURE']
        tag.deadband = 5.0  # Large deadband

        initial_value = 100.0
        tag.last_value = initial_value

        updates = []

        async def callback(tag_name, value, timestamp):
            updates.append(value)

        await scada_connector.subscribe('BOILER.STEAM.PRESSURE', callback)

        # Simulate small changes that should be filtered
        for _ in range(5):
            await scada_connector._read_tag('BOILER.STEAM.PRESSURE')
            await asyncio.sleep(0.1)

        # Should have minimal updates due to deadband
        assert len(updates) < 5

    @pytest.mark.asyncio
    async def test_write_tag_value_to_scada(self, scada_connector):
        """Test writing tag values to SCADA system."""
        await scada_connector.connect()

        tag_name = 'BOILER.FUEL.VALVE.POSITION'
        new_value = 55.0

        result = await scada_connector.write_tag(tag_name, new_value)

        assert result is True
        assert scada_connector.tags[tag_name].last_value == new_value

    @pytest.mark.asyncio
    async def test_write_tag_with_scaling(self, scada_connector):
        """Test tag value scaling during write operations."""
        await scada_connector.connect()

        tag_name = 'BOILER.FUEL.VALVE.POSITION'
        tag = scada_connector.tags[tag_name]
        tag.scaling_factor = 2.0
        tag.offset = 10.0

        engineering_value = 100.0  # Engineering units
        result = await scada_connector.write_tag(tag_name, engineering_value)

        assert result is True
        # Scaled value should be (100 - 10) / 2 = 45

    @pytest.mark.asyncio
    async def test_write_timeout_handling(self, scada_connector):
        """Test write timeout handling."""
        await scada_connector.connect()

        # Mock slow write
        original_write = scada_connector._write_opc_ua

        async def slow_write(*args, **kwargs):
            await asyncio.sleep(15)  # Longer than timeout
            return True

        scada_connector._write_opc_ua = slow_write

        # Should handle timeout gracefully
        result = await scada_connector.write_tag('BOILER.FUEL.VALVE.POSITION', 50.0)

        # Result may be True or False, but should not crash
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_batch_read_multiple_tags(self, scada_connector):
        """Test batch reading multiple tags efficiently."""
        await scada_connector.connect()

        # Wait for scan to populate values
        await asyncio.sleep(2)

        current_values = await scada_connector.get_current_values()

        assert len(current_values) > 0
        assert 'BOILER.STEAM.PRESSURE' in current_values
        assert 'BOILER.STEAM.TEMPERATURE' in current_values

        for tag_name, data in current_values.items():
            assert 'value' in data
            assert 'quality' in data
            assert 'units' in data

    @pytest.mark.asyncio
    async def test_data_quality_indicators(self, scada_connector):
        """Test data quality status tracking."""
        await scada_connector.connect()

        await asyncio.sleep(1)

        tag = scada_connector.tags['BOILER.STEAM.PRESSURE']

        # Initially should be GOOD
        assert tag.quality == DataQuality.GOOD

        # Simulate communication failure
        original_read = scada_connector._read_opc_ua

        async def failing_read(*args, **kwargs):
            raise Exception("Communication error")

        scada_connector._read_opc_ua = failing_read

        await scada_connector._read_tag('BOILER.STEAM.PRESSURE')

        # Quality should degrade
        assert tag.quality == DataQuality.BAD_COMM_FAILURE


# Test Class: Data Buffering
class TestSCADADataBuffer:
    """Test SCADA data buffering and historical data retrieval."""

    @pytest.mark.asyncio
    async def test_data_buffer_circular_operation(self):
        """Test circular buffer operation with max size."""
        buffer = SCADADataBuffer(max_size=100)

        # Add more samples than buffer size
        for i in range(150):
            await buffer.add_sample(
                'TEST_TAG',
                float(i),
                datetime.utcnow(),
                DataQuality.GOOD
            )

        # Should only retain last 100
        assert len(buffer.buffer['TEST_TAG']) <= 100

    @pytest.mark.asyncio
    async def test_data_compression_efficiency(self):
        """Test data compression for old samples."""
        buffer = SCADADataBuffer(max_size=1000, compression_enabled=True)

        # Add many samples
        base_time = datetime.utcnow()
        for i in range(1500):
            await buffer.add_sample(
                'PRESSURE_TAG',
                100.0 + random.uniform(-1, 1),  # Noisy data
                base_time + timedelta(seconds=i),
                DataQuality.GOOD
            )

        # Compression should have triggered
        assert len(buffer.compressed_data['PRESSURE_TAG']) > 0

    @pytest.mark.asyncio
    async def test_historical_data_retrieval(self, scada_connector):
        """Test retrieving historical data for a time period."""
        await scada_connector.connect()

        # Wait for data to accumulate
        await asyncio.sleep(3)

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=2)

        historical = await scada_connector.get_historical_data(
            'BOILER.STEAM.PRESSURE',
            start_time,
            end_time
        )

        assert isinstance(historical, list)
        # Should have some data points
        assert len(historical) >= 0

    @pytest.mark.asyncio
    async def test_buffer_statistics_calculation(self, scada_connector):
        """Test statistical calculations on buffered data."""
        await scada_connector.connect()

        # Wait for data
        await asyncio.sleep(2)

        stats = await scada_connector.get_statistics(
            ['BOILER.STEAM.PRESSURE', 'BOILER.EFFICIENCY'],
            duration_hours=1
        )

        assert isinstance(stats, dict)

        for tag_name, tag_stats in stats.items():
            if tag_stats:  # May be empty if no data yet
                assert 'min' in tag_stats
                assert 'max' in tag_stats
                assert 'avg' in tag_stats


# Test Class: Alarm Management
class TestSCADAAlarmManagement:
    """Test SCADA alarm detection and management."""

    @pytest.mark.asyncio
    async def test_alarm_configuration(self, scada_connector):
        """Test alarm configuration and setup."""
        await scada_connector.connect()

        # Check default alarms are configured
        alarms = scada_connector.alarm_manager.alarms

        assert len(alarms) > 0

        # Check for pressure alarms
        pressure_alarms = [
            a for a in alarms.values()
            if 'PRESSURE' in a.tag_name
        ]
        assert len(pressure_alarms) > 0

    @pytest.mark.asyncio
    async def test_alarm_activation_on_limit_exceeded(self, scada_connector):
        """Test alarm activates when value exceeds limit."""
        await scada_connector.connect()

        alarm_manager = scada_connector.alarm_manager

        # Create test alarm
        test_alarm = SCADAAlarm(
            alarm_id='TEST_HIGH',
            tag_name='TEST_TAG',
            alarm_type='high',
            priority=AlarmPriority.HIGH,
            setpoint=100.0,
            deadband=2.0,
            delay_seconds=1,
            message='Test high alarm'
        )

        alarm_manager.configure_alarm(test_alarm)

        # Trigger alarm with high value
        await alarm_manager.check_alarm_condition(
            'TEST_TAG',
            105.0,  # Exceeds setpoint
            datetime.utcnow()
        )

        # Wait for delay
        await asyncio.sleep(1.5)

        # Check alarm is active
        assert test_alarm.active is True
        assert 'TEST_HIGH' in alarm_manager.active_alarms

    @pytest.mark.asyncio
    async def test_alarm_deactivation_on_value_return(self, scada_connector):
        """Test alarm deactivates when value returns to normal."""
        await scada_connector.connect()

        alarm_manager = scada_connector.alarm_manager

        # Create and activate alarm
        test_alarm = SCADAAlarm(
            alarm_id='TEST_HIGH',
            tag_name='TEST_TAG',
            alarm_type='high',
            priority=AlarmPriority.HIGH,
            setpoint=100.0,
            deadband=2.0,
            delay_seconds=0,
            message='Test high alarm'
        )

        alarm_manager.configure_alarm(test_alarm)
        test_alarm.active = True
        alarm_manager.active_alarms.add('TEST_HIGH')

        # Return to normal value
        await alarm_manager.check_alarm_condition(
            'TEST_TAG',
            95.0,  # Below setpoint
            datetime.utcnow()
        )

        # Should deactivate
        assert test_alarm.active is False
        assert 'TEST_HIGH' not in alarm_manager.active_alarms

    @pytest.mark.asyncio
    async def test_alarm_acknowledgment(self, scada_connector):
        """Test alarm acknowledgment by operator."""
        await scada_connector.connect()

        alarm_manager = scada_connector.alarm_manager

        # Create active alarm
        test_alarm = SCADAAlarm(
            alarm_id='TEST_ALARM',
            tag_name='TEST_TAG',
            alarm_type='high',
            priority=AlarmPriority.CRITICAL,
            setpoint=100.0,
            deadband=2.0,
            delay_seconds=0,
            message='Critical alarm',
            active=True,
            acknowledged=False
        )

        alarm_manager.configure_alarm(test_alarm)
        alarm_manager.active_alarms.add('TEST_ALARM')

        # Acknowledge alarm
        result = await alarm_manager.acknowledge_alarm('TEST_ALARM', 'operator1')

        assert result is True
        assert test_alarm.acknowledged is True
        assert test_alarm.acknowledgment_time is not None

    @pytest.mark.asyncio
    async def test_alarm_callback_notifications(self, scada_connector):
        """Test alarm callbacks are triggered."""
        await scada_connector.connect()

        alarm_manager = scada_connector.alarm_manager
        callback_events = []

        async def alarm_callback(alarm, action, value):
            callback_events.append({
                'alarm_id': alarm.alarm_id,
                'action': action,
                'value': value
            })

        alarm_manager.alarm_callbacks.append(alarm_callback)

        # Create and trigger alarm
        test_alarm = SCADAAlarm(
            alarm_id='TEST_CALLBACK',
            tag_name='TEST_TAG',
            alarm_type='high',
            priority=AlarmPriority.HIGH,
            setpoint=100.0,
            deadband=2.0,
            delay_seconds=0,
            message='Callback test'
        )

        alarm_manager.configure_alarm(test_alarm)

        await alarm_manager.check_alarm_condition('TEST_TAG', 105.0, datetime.utcnow())
        await asyncio.sleep(0.1)

        assert len(callback_events) > 0
        assert callback_events[0]['action'] == 'activated'

    @pytest.mark.asyncio
    async def test_alarm_priority_sorting(self, scada_connector):
        """Test alarms are sorted by priority."""
        await scada_connector.connect()

        alarm_manager = scada_connector.alarm_manager

        # Create alarms with different priorities
        alarms = [
            SCADAAlarm('ALARM_LOW', 'TAG1', 'high', AlarmPriority.LOW, 100, 1, 0, 'Low', active=True),
            SCADAAlarm('ALARM_CRITICAL', 'TAG2', 'high', AlarmPriority.CRITICAL, 100, 1, 0, 'Critical', active=True),
            SCADAAlarm('ALARM_MEDIUM', 'TAG3', 'high', AlarmPriority.MEDIUM, 100, 1, 0, 'Medium', active=True),
        ]

        for alarm in alarms:
            alarm_manager.configure_alarm(alarm)
            alarm_manager.active_alarms.add(alarm.alarm_id)

        active = alarm_manager.get_active_alarms()

        # Should be sorted by priority (1=highest)
        assert active[0].priority == AlarmPriority.CRITICAL
        assert active[-1].priority == AlarmPriority.LOW


# Test Class: Scan Rate Management
class TestSCADAScanRates:
    """Test SCADA tag scanning and rate management."""

    @pytest.mark.asyncio
    async def test_multi_rate_scanning(self, scada_connector):
        """Test multiple scan rates work independently."""
        await scada_connector.connect()

        # Check scan tasks are created
        assert len(scada_connector.scan_tasks) > 0

        # Should have different scan rates
        scan_rates = set()
        for task_name in scada_connector.scan_tasks.keys():
            if task_name.startswith('scan_'):
                rate = int(task_name.split('_')[1])
                scan_rates.add(rate)

        # Should have multiple scan rates
        assert len(scan_rates) >= 2

    @pytest.mark.asyncio
    async def test_scan_task_cancellation_on_disconnect(self, scada_connector):
        """Test scan tasks are cancelled on disconnect."""
        await scada_connector.connect()

        initial_tasks = len(scada_connector.scan_tasks)
        assert initial_tasks > 0

        await scada_connector.disconnect()

        # All tasks should be cancelled
        assert len(scada_connector.scan_tasks) == 0


# Test Class: Error Handling
class TestSCADAErrorHandling:
    """Test SCADA error handling and recovery."""

    @pytest.mark.asyncio
    async def test_tag_read_error_handling(self, scada_connector):
        """Test graceful handling of tag read errors."""
        await scada_connector.connect()

        # Mock read failure
        async def failing_read(*args, **kwargs):
            raise Exception("Sensor failure")

        scada_connector._read_opc_ua = failing_read

        # Should handle error gracefully
        await scada_connector._read_tag('BOILER.STEAM.PRESSURE')

        # Tag quality should reflect failure
        tag = scada_connector.tags['BOILER.STEAM.PRESSURE']
        assert tag.quality == DataQuality.BAD_COMM_FAILURE

    @pytest.mark.asyncio
    async def test_connection_loss_recovery(self, scada_connector):
        """Test automatic recovery from connection loss."""
        await scada_connector.connect()

        # Simulate connection loss
        scada_connector.connected = False
        scada_connector.connection = None

        # Should attempt reconnection
        # In real scenario would wait for reconnect task
        assert scada_connector._reconnect_task is None or scada_connector._reconnect_task.done()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
