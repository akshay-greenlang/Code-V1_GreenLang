# -*- coding: utf-8 -*-
"""
Steam Quality Meter Integration Tests for GL-012 SteamQualityController

Comprehensive integration tests for steam quality meter connectivity including:
- Connection to steam quality meters via Modbus/OPC-UA
- Reading quality parameters (dryness, moisture, pressure, temperature)
- Connection failure handling and resilience
- Automatic reconnection logic
- Data validation from meters
- Mock Modbus/OPC-UA server responses

Test Count: 25+ tests
Coverage Target: 90%+

Standards: ASME PTC 4.4 (Steam Quality Measurement)

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import struct

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.steam_meter]


# =============================================================================
# STEAM METER CONNECTION TESTS
# =============================================================================

class TestSteamMeterConnection:
    """Test steam quality meter connection handling."""

    async def test_modbus_meter_connection_success(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test successful Modbus connection to steam quality meter."""
        connection_config = {
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port,
            'unit_id': 1,
            'timeout_seconds': 5.0
        }

        result = await steam_quality_controller.connect_meter(connection_config)

        assert result['status'] == 'connected'
        assert result['protocol'] == 'modbus_tcp'
        assert result['meter_id'] is not None
        assert mock_modbus_steam_meter.connection_count >= 1

    async def test_opcua_meter_connection_success(
        self,
        mock_opcua_steam_meter,
        steam_quality_controller
    ):
        """Test successful OPC-UA connection to steam quality meter."""
        connection_config = {
            'protocol': 'opcua',
            'endpoint': f'opc.tcp://{mock_opcua_steam_meter.host}:{mock_opcua_steam_meter.port}',
            'security_mode': 'None',
            'timeout_seconds': 5.0
        }

        result = await steam_quality_controller.connect_meter(connection_config)

        assert result['status'] == 'connected'
        assert result['protocol'] == 'opcua'
        assert result['session_id'] is not None

    async def test_meter_connection_with_authentication(
        self,
        mock_opcua_steam_meter,
        steam_quality_controller
    ):
        """Test meter connection with username/password authentication."""
        connection_config = {
            'protocol': 'opcua',
            'endpoint': f'opc.tcp://{mock_opcua_steam_meter.host}:{mock_opcua_steam_meter.port}',
            'security_mode': 'SignAndEncrypt',
            'username': 'steam_operator',
            'password': 'secure_password',
            'timeout_seconds': 5.0
        }

        result = await steam_quality_controller.connect_meter(connection_config)

        assert result['status'] == 'connected'
        assert result['authenticated'] is True

    async def test_meter_connection_invalid_host(
        self,
        steam_quality_controller
    ):
        """Test connection failure with invalid host."""
        connection_config = {
            'protocol': 'modbus_tcp',
            'host': 'invalid-host.example.com',
            'port': 502,
            'timeout_seconds': 2.0
        }

        result = await steam_quality_controller.connect_meter(connection_config)

        assert result['status'] in ['error', 'timeout', 'connection_failed']
        assert 'error_message' in result or 'message' in result

    async def test_meter_connection_invalid_port(
        self,
        steam_quality_controller
    ):
        """Test connection failure with invalid port."""
        connection_config = {
            'protocol': 'modbus_tcp',
            'host': 'localhost',
            'port': 59999,  # Invalid port
            'timeout_seconds': 2.0
        }

        result = await steam_quality_controller.connect_meter(connection_config)

        assert result['status'] in ['error', 'timeout', 'connection_failed']

    async def test_meter_connection_timeout(
        self,
        mock_slow_meter,
        steam_quality_controller
    ):
        """Test connection timeout handling."""
        connection_config = {
            'protocol': 'modbus_tcp',
            'host': mock_slow_meter.host,
            'port': mock_slow_meter.port,
            'timeout_seconds': 0.5  # Very short timeout
        }

        result = await steam_quality_controller.connect_meter(connection_config)

        assert result['status'] in ['timeout', 'error']


# =============================================================================
# STEAM QUALITY PARAMETER READING TESTS
# =============================================================================

class TestSteamQualityParameterReading:
    """Test reading quality parameters from steam meters."""

    async def test_read_steam_dryness_fraction(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test reading steam dryness fraction (quality)."""
        # Connect first
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_quality()

        assert 'dryness_fraction' in result
        assert 0.0 <= result['dryness_fraction'] <= 1.0
        assert result['quality'] == 'GOOD'
        assert 'timestamp' in result

    async def test_read_steam_moisture_content(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test reading steam moisture content."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_quality()

        assert 'moisture_percent' in result
        assert 0.0 <= result['moisture_percent'] <= 100.0
        # Moisture + Dryness should equal 100%
        expected_moisture = (1.0 - result['dryness_fraction']) * 100.0
        assert abs(result['moisture_percent'] - expected_moisture) < 0.1

    async def test_read_steam_pressure(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test reading steam pressure."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_parameters()

        assert 'pressure_bar' in result
        assert result['pressure_bar'] > 0
        assert result['pressure_bar'] < 300  # Reasonable upper limit

    async def test_read_steam_temperature(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test reading steam temperature."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_parameters()

        assert 'temperature_c' in result
        assert result['temperature_c'] > 100  # Above boiling
        assert result['temperature_c'] < 700  # Below superheat limit

    async def test_read_steam_flow_rate(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test reading steam flow rate."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_parameters()

        assert 'flow_rate_kg_s' in result
        assert result['flow_rate_kg_s'] >= 0

    async def test_read_superheat_degree(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test reading superheat degree calculation."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_parameters()

        assert 'superheat_c' in result
        # Superheat can be zero (saturated) or positive
        assert result['superheat_c'] >= 0

    async def test_read_enthalpy_calculation(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test steam enthalpy calculation from meter readings."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_parameters()

        assert 'enthalpy_kj_kg' in result
        # Typical steam enthalpy range
        assert 2000 < result['enthalpy_kj_kg'] < 4000

    async def test_batch_parameter_reading(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test batch reading of all parameters."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_all_parameters()

        required_params = [
            'dryness_fraction',
            'moisture_percent',
            'pressure_bar',
            'temperature_c',
            'flow_rate_kg_s',
            'superheat_c',
            'enthalpy_kj_kg'
        ]

        for param in required_params:
            assert param in result, f"Missing parameter: {param}"


# =============================================================================
# CONNECTION FAILURE HANDLING TESTS
# =============================================================================

class TestConnectionFailureHandling:
    """Test connection failure handling and recovery."""

    async def test_connection_loss_detection(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test detection of connection loss during operation."""
        # Establish connection
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        # Simulate connection loss
        await mock_modbus_steam_meter.stop()
        await asyncio.sleep(0.1)

        # Attempt to read - should detect connection loss
        result = await steam_quality_controller.read_steam_quality()

        assert result['status'] in ['disconnected', 'error', 'connection_lost']
        assert steam_quality_controller.connection_state == 'disconnected'

    async def test_graceful_degradation_on_failure(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test graceful degradation when connection fails."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        # Get initial reading
        initial_result = await steam_quality_controller.read_steam_quality()

        # Simulate connection loss
        await mock_modbus_steam_meter.stop()

        # Should return cached data with quality flag
        fallback_result = await steam_quality_controller.read_steam_quality()

        assert fallback_result['quality'] in ['STALE', 'CACHED', 'UNCERTAIN']
        assert 'last_valid_timestamp' in fallback_result

    async def test_connection_failure_event_notification(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller,
        mock_event_handler
    ):
        """Test event notification on connection failure."""
        steam_quality_controller.register_event_handler(mock_event_handler)

        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        # Simulate connection loss
        await mock_modbus_steam_meter.stop()
        await asyncio.sleep(0.1)

        # Trigger read to detect failure
        await steam_quality_controller.read_steam_quality()

        # Verify event was raised
        assert mock_event_handler.connection_lost_called is True
        assert mock_event_handler.last_event['event_type'] == 'CONNECTION_LOST'

    async def test_multiple_consecutive_failures(
        self,
        mock_unstable_meter,
        steam_quality_controller
    ):
        """Test handling of multiple consecutive connection failures."""
        failure_count = 0
        max_failures = 5

        for i in range(max_failures):
            result = await steam_quality_controller.connect_meter({
                'protocol': 'modbus_tcp',
                'host': mock_unstable_meter.host,
                'port': mock_unstable_meter.port,
                'timeout_seconds': 1.0
            })

            if result['status'] in ['error', 'timeout']:
                failure_count += 1

        # System should track failure count
        assert steam_quality_controller.consecutive_failures >= failure_count - 1


# =============================================================================
# RECONNECTION LOGIC TESTS
# =============================================================================

class TestReconnectionLogic:
    """Test automatic reconnection logic."""

    async def test_automatic_reconnection_on_failure(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test automatic reconnection after connection failure."""
        # Initial connection
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port,
            'auto_reconnect': True,
            'reconnect_interval_seconds': 0.5
        })

        # Simulate temporary outage
        await mock_modbus_steam_meter.stop()
        await asyncio.sleep(0.2)
        await mock_modbus_steam_meter.start()

        # Wait for reconnection
        await asyncio.sleep(1.0)

        # Should be reconnected
        result = await steam_quality_controller.read_steam_quality()
        assert result['status'] != 'disconnected'

    async def test_exponential_backoff_reconnection(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test exponential backoff in reconnection attempts."""
        steam_quality_controller.configure_reconnection({
            'initial_delay_seconds': 0.1,
            'max_delay_seconds': 2.0,
            'backoff_multiplier': 2.0
        })

        # Track reconnection attempt times
        attempt_times = []

        async def track_reconnect():
            attempt_times.append(asyncio.get_event_loop().time())

        steam_quality_controller.on_reconnect_attempt = track_reconnect

        # Simulate persistent failure
        await mock_modbus_steam_meter.stop()

        # Trigger multiple reconnection attempts
        for _ in range(4):
            await steam_quality_controller.attempt_reconnect()

        # Verify backoff pattern
        if len(attempt_times) >= 3:
            delay_1 = attempt_times[1] - attempt_times[0]
            delay_2 = attempt_times[2] - attempt_times[1]

            # Each delay should be approximately 2x the previous
            assert delay_2 >= delay_1 * 1.5  # Allow some tolerance

    async def test_reconnection_limit(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test maximum reconnection attempts limit."""
        steam_quality_controller.configure_reconnection({
            'max_attempts': 3,
            'initial_delay_seconds': 0.1
        })

        await mock_modbus_steam_meter.stop()

        # Attempt reconnections
        for _ in range(5):
            await steam_quality_controller.attempt_reconnect()

        # Should stop after max attempts
        assert steam_quality_controller.reconnect_attempts <= 3
        assert steam_quality_controller.connection_state == 'failed'

    async def test_successful_reconnection_resets_counter(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test that successful reconnection resets attempt counter."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        # Simulate failures
        steam_quality_controller.reconnect_attempts = 3

        # Simulate successful reconnection
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        assert steam_quality_controller.reconnect_attempts == 0


# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

class TestMeterDataValidation:
    """Test validation of data received from meters."""

    async def test_validate_dryness_fraction_range(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test validation of dryness fraction within valid range."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_quality()

        # Validate range
        assert result['validation']['dryness_valid'] is True
        assert 0.0 <= result['dryness_fraction'] <= 1.0

    async def test_detect_out_of_range_pressure(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test detection of out-of-range pressure values."""
        # Configure meter to return invalid pressure
        mock_modbus_steam_meter.set_register_value('pressure', -5.0)

        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_parameters()

        assert result['validation']['pressure_valid'] is False
        assert 'pressure_error' in result['validation']

    async def test_detect_sensor_spike(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test detection of sudden sensor value spikes."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        # Get baseline reading
        await steam_quality_controller.read_steam_parameters()

        # Simulate spike
        mock_modbus_steam_meter.set_register_value('temperature', 900.0)  # Spike

        result = await steam_quality_controller.read_steam_parameters()

        assert result['validation']['rate_of_change_valid'] is False
        assert result['quality'] == 'SUSPECT'

    async def test_validate_thermodynamic_consistency(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test validation of thermodynamic consistency between parameters."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_parameters()

        # Pressure and temperature should be thermodynamically consistent
        assert result['validation']['thermodynamic_consistent'] is True

    async def test_data_quality_code_assignment(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test proper data quality code assignment."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_quality()

        assert result['quality'] in ['GOOD', 'UNCERTAIN', 'BAD', 'STALE', 'SUSPECT']
        assert 'quality_code' in result

    async def test_timestamp_validation(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test validation of meter timestamps."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_steam_parameters()

        # Timestamp should be recent
        reading_time = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        age_seconds = (now - reading_time).total_seconds()

        assert age_seconds < 5.0  # Less than 5 seconds old


# =============================================================================
# MOCK MODBUS/OPC-UA RESPONSE TESTS
# =============================================================================

class TestMockProtocolResponses:
    """Test handling of various Modbus/OPC-UA response scenarios."""

    async def test_modbus_holding_register_response(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test parsing of Modbus holding register responses."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        # Read specific registers
        registers = await steam_quality_controller.read_registers(
            start_address=0,
            count=10
        )

        assert len(registers) == 10
        assert all(isinstance(r, (int, float)) for r in registers)

    async def test_modbus_float32_decoding(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test proper decoding of 32-bit float values from Modbus."""
        # Set a known float value
        test_value = 123.456
        mock_modbus_steam_meter.set_float32_register(100, test_value)

        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_float32_register(100)

        assert abs(result - test_value) < 0.001

    async def test_opcua_node_read_response(
        self,
        mock_opcua_steam_meter,
        steam_quality_controller
    ):
        """Test OPC-UA node read response handling."""
        await steam_quality_controller.connect_meter({
            'protocol': 'opcua',
            'endpoint': f'opc.tcp://{mock_opcua_steam_meter.host}:{mock_opcua_steam_meter.port}'
        })

        result = await steam_quality_controller.read_opcua_node(
            'ns=2;s=Steam.Quality.DrynessFraction'
        )

        assert 'value' in result
        assert 'source_timestamp' in result
        assert 'status_code' in result

    async def test_modbus_exception_response(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test handling of Modbus exception responses."""
        # Configure meter to return exception for specific address
        mock_modbus_steam_meter.set_exception_response(
            address=999,
            exception_code=2  # Illegal Data Address
        )

        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        result = await steam_quality_controller.read_registers(
            start_address=999,
            count=1
        )

        assert result['status'] == 'error'
        assert result['exception_code'] == 2

    async def test_opcua_bad_status_code_handling(
        self,
        mock_opcua_steam_meter,
        steam_quality_controller
    ):
        """Test handling of OPC-UA bad status codes."""
        mock_opcua_steam_meter.set_node_status(
            'ns=2;s=Steam.Quality.Invalid',
            status_code='Bad_NodeIdUnknown'
        )

        await steam_quality_controller.connect_meter({
            'protocol': 'opcua',
            'endpoint': f'opc.tcp://{mock_opcua_steam_meter.host}:{mock_opcua_steam_meter.port}'
        })

        result = await steam_quality_controller.read_opcua_node(
            'ns=2;s=Steam.Quality.Invalid'
        )

        assert result['status_code'] == 'Bad_NodeIdUnknown'
        assert result['quality'] == 'BAD'

    async def test_communication_timeout_handling(
        self,
        mock_slow_meter,
        steam_quality_controller
    ):
        """Test handling of communication timeouts."""
        mock_slow_meter.set_response_delay(5.0)  # 5 second delay

        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_slow_meter.host,
            'port': mock_slow_meter.port,
            'timeout_seconds': 1.0
        })

        result = await steam_quality_controller.read_steam_quality()

        assert result['status'] == 'timeout'


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.performance
class TestSteamMeterPerformance:
    """Performance tests for steam meter integration."""

    async def test_meter_polling_latency(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller,
        performance_monitor
    ):
        """Test meter polling latency meets target (<50ms)."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        performance_monitor.start()

        poll_times = []
        for _ in range(100):
            start = asyncio.get_event_loop().time()
            await steam_quality_controller.read_steam_quality()
            elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
            poll_times.append(elapsed_ms)
            performance_monitor.record_metric('poll_latency_ms', elapsed_ms)

        avg_latency = sum(poll_times) / len(poll_times)
        max_latency = max(poll_times)

        print(f"\n=== Steam Meter Polling Performance ===")
        print(f"Average Latency: {avg_latency:.2f}ms")
        print(f"Max Latency: {max_latency:.2f}ms")

        assert avg_latency < 50.0, f"Average latency {avg_latency}ms exceeds 50ms target"

    async def test_high_frequency_polling(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test high-frequency polling stability (10 Hz)."""
        await steam_quality_controller.connect_meter({
            'protocol': 'modbus_tcp',
            'host': mock_modbus_steam_meter.host,
            'port': mock_modbus_steam_meter.port
        })

        readings = []
        poll_interval = 0.1  # 100ms = 10 Hz

        for _ in range(50):
            result = await steam_quality_controller.read_steam_quality()
            readings.append(result)
            await asyncio.sleep(poll_interval)

        # All readings should be successful
        success_count = sum(1 for r in readings if r['quality'] == 'GOOD')
        success_rate = success_count / len(readings)

        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} below 95% target"

    async def test_concurrent_meter_access(
        self,
        mock_modbus_steam_meter,
        steam_quality_controller
    ):
        """Test concurrent access to multiple meters."""
        # Simulate connecting to multiple meters
        meter_ports = [502, 503, 504]

        async def read_from_meter(port):
            result = await steam_quality_controller.read_from_port(port)
            return result

        tasks = [read_from_meter(port) for port in meter_ports]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All concurrent reads should complete
        assert len(results) == len(meter_ports)
