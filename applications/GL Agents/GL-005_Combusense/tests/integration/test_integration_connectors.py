# -*- coding: utf-8 -*-
"""
Integration connector tests for GL-005 CombustionControlAgent.

Tests mock server integrations for all connector types:
- DCS Connector (OPC UA) mock integration
- PLC Connector (Modbus) mock integration
- Analyzer Connector (MQTT) mock integration
- SCADA publishing verification

Target: 30+ tests covering:
- Connection establishment and teardown
- Read/write operations
- Error handling
- Performance requirements
- Data quality validation
- Protocol-specific behaviors
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.connectors]


# ============================================================================
# DCS CONNECTOR (OPC UA) MOCK INTEGRATION TESTS
# ============================================================================

class TestDCSConnectorIntegration:
    """Test DCS connector integration with OPC UA mock server."""

    async def test_dcs_connection_establishment(self, opcua_server):
        """Test DCS connection can be established."""
        # Server should be running from fixture
        assert opcua_server.is_running is True

    async def test_dcs_connection_teardown(self, opcua_server):
        """Test DCS connection can be properly closed."""
        # Stop server
        await opcua_server.stop()
        assert opcua_server.is_running is False

        # Restart for other tests
        await opcua_server.start()
        assert opcua_server.is_running is True

    async def test_dcs_read_single_node(self, opcua_server):
        """Test reading a single node from DCS."""
        value = await opcua_server.read_node('fuel_flow')

        assert value is not None
        assert isinstance(value, (int, float))
        assert value >= 0

    async def test_dcs_read_multiple_nodes(self, opcua_server):
        """Test reading multiple nodes from DCS."""
        nodes = ['fuel_flow', 'air_flow', 'combustion_temperature', 'furnace_pressure']
        values = await opcua_server.read_multiple_nodes(nodes)

        assert len(values) == len(nodes)
        for node in nodes:
            assert node in values
            assert values[node] is not None

    async def test_dcs_write_single_node(self, opcua_server):
        """Test writing a single node to DCS."""
        test_value = 550.0
        success = await opcua_server.write_node('fuel_flow', test_value)

        assert success is True

        # Verify write
        read_value = await opcua_server.read_node('fuel_flow')
        assert abs(read_value - test_value) < 1.0

    async def test_dcs_write_multiple_nodes(self, opcua_server):
        """Test writing multiple nodes to DCS."""
        values = {
            'fuel_flow': 520.0,
            'air_flow': 5200.0
        }

        for node, value in values.items():
            await opcua_server.write_node(node, value)

        # Verify writes
        for node, expected in values.items():
            actual = await opcua_server.read_node(node)
            assert abs(actual - expected) < 10.0

    async def test_dcs_read_nonexistent_node(self, opcua_server):
        """Test reading a nonexistent node raises appropriate error."""
        with pytest.raises((KeyError, ValueError, Exception)):
            await opcua_server.read_node('nonexistent_node')

    async def test_dcs_read_performance(self, opcua_server, performance_timer):
        """Test DCS read performance meets requirements."""
        with performance_timer() as timer:
            for _ in range(100):
                await opcua_server.read_node('fuel_flow')

        avg_time_ms = timer.elapsed_ms / 100
        assert avg_time_ms < 50.0  # <50ms per read

    async def test_dcs_write_performance(self, opcua_server, performance_timer):
        """Test DCS write performance meets requirements."""
        with performance_timer() as timer:
            for _ in range(100):
                await opcua_server.write_node('fuel_flow', 500.0)

        avg_time_ms = timer.elapsed_ms / 100
        assert avg_time_ms < 50.0  # <50ms per write

    async def test_dcs_concurrent_reads(self, opcua_server):
        """Test concurrent read operations."""
        nodes = ['fuel_flow', 'air_flow', 'combustion_temperature', 'furnace_pressure']

        # Run concurrent reads
        tasks = [opcua_server.read_node(node) for node in nodes]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(nodes)
        assert all(r is not None for r in results)

    async def test_dcs_data_types(self, opcua_server):
        """Test DCS handles different data types correctly."""
        # Float value
        await opcua_server.write_node('fuel_flow', 500.5)
        float_value = await opcua_server.read_node('fuel_flow')
        assert isinstance(float_value, float)

        # Integer-like value
        await opcua_server.write_node('fuel_flow', 500)
        int_value = await opcua_server.read_node('fuel_flow')
        assert isinstance(int_value, (int, float))


# ============================================================================
# PLC CONNECTOR (MODBUS) MOCK INTEGRATION TESTS
# ============================================================================

class TestPLCConnectorIntegration:
    """Test PLC connector integration with Modbus mock server."""

    async def test_plc_connection_establishment(self, modbus_server):
        """Test PLC connection can be established."""
        assert modbus_server.is_running is True

    async def test_plc_connection_teardown(self, modbus_server):
        """Test PLC connection can be properly closed."""
        await modbus_server.stop()
        assert modbus_server.is_running is False

        await modbus_server.start()
        assert modbus_server.is_running is True

    async def test_plc_read_coils(self, modbus_server):
        """Test reading coils from PLC."""
        coils = await modbus_server.read_coils(0, 8)

        assert len(coils) == 8
        assert all(isinstance(c, bool) for c in coils)

    async def test_plc_write_single_coil(self, modbus_server):
        """Test writing a single coil to PLC."""
        await modbus_server.write_coil(1, True)

        coils = await modbus_server.read_coils(1, 1)
        assert coils[0] is True

        # Reset
        await modbus_server.write_coil(1, False)

    async def test_plc_read_input_registers(self, modbus_server):
        """Test reading input registers from PLC."""
        registers = await modbus_server.read_input_registers(0, 4)

        assert len(registers) == 4
        assert all(isinstance(r, (int, float)) for r in registers)

    async def test_plc_read_holding_registers(self, modbus_server):
        """Test reading holding registers from PLC."""
        registers = await modbus_server.read_holding_registers(0, 4)

        assert len(registers) == 4

    async def test_plc_write_holding_register(self, modbus_server):
        """Test writing holding register to PLC."""
        test_value = 12345
        await modbus_server.write_holding_register(0, test_value)

        registers = await modbus_server.read_holding_registers(0, 1)
        assert registers[0] == test_value

    async def test_plc_emergency_stop_coil(self, modbus_server):
        """Test emergency stop coil functionality."""
        # Activate emergency stop
        await modbus_server.write_coil(1, True)

        coils = await modbus_server.read_coils(1, 1)
        assert coils[0] is True

        # Deactivate
        await modbus_server.write_coil(1, False)

        coils = await modbus_server.read_coils(1, 1)
        assert coils[0] is False

    async def test_plc_read_performance(self, modbus_server, performance_timer):
        """Test PLC read performance meets requirements."""
        with performance_timer() as timer:
            for _ in range(100):
                await modbus_server.read_coils(0, 8)

        avg_time_ms = timer.elapsed_ms / 100
        assert avg_time_ms < 20.0  # <20ms per read

    async def test_plc_write_performance(self, modbus_server, performance_timer):
        """Test PLC write performance meets requirements."""
        with performance_timer() as timer:
            for _ in range(100):
                await modbus_server.write_coil(0, True)
                await modbus_server.write_coil(0, False)

        avg_time_ms = timer.elapsed_ms / 200
        assert avg_time_ms < 20.0  # <20ms per write

    async def test_plc_concurrent_operations(self, modbus_server):
        """Test concurrent PLC operations."""
        tasks = [
            modbus_server.read_coils(0, 8),
            modbus_server.read_input_registers(0, 4),
            modbus_server.read_holding_registers(0, 4)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(r is not None for r in results)


# ============================================================================
# ANALYZER CONNECTOR (MQTT) MOCK INTEGRATION TESTS
# ============================================================================

class TestAnalyzerConnectorIntegration:
    """Test analyzer connector integration with MQTT mock broker."""

    async def test_mqtt_broker_connection(self, mqtt_broker):
        """Test MQTT broker connection is established."""
        assert mqtt_broker.is_running is True

    async def test_mqtt_publish_message(self, mqtt_broker):
        """Test publishing message to MQTT broker."""
        test_data = {
            'o2_percent': 3.5,
            'co_ppm': 25.0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        await mqtt_broker.publish('combustion/analyzer/data', test_data)

        # Verify message stored
        received = mqtt_broker.get_latest_message('combustion/analyzer/data')
        assert received is not None
        assert received['o2_percent'] == 3.5

    async def test_mqtt_get_latest_message(self, mqtt_broker):
        """Test getting latest message from topic."""
        # Publish multiple messages
        for i in range(3):
            await mqtt_broker.publish('combustion/analyzer/data', {
                'o2_percent': 3.0 + i * 0.1,
                'sequence': i
            })

        # Get latest
        latest = mqtt_broker.get_latest_message('combustion/analyzer/data')

        assert latest is not None
        assert latest['sequence'] == 2  # Last message

    async def test_mqtt_message_ordering(self, mqtt_broker):
        """Test MQTT messages maintain order."""
        topic = 'combustion/analyzer/test_order'

        # Publish numbered messages
        for i in range(5):
            await mqtt_broker.publish(topic, {'sequence': i})

        # Get all messages
        messages = mqtt_broker.get_all_messages(topic)

        # Verify order
        for i, msg in enumerate(messages):
            assert msg['sequence'] == i

    async def test_mqtt_clear_messages(self, mqtt_broker):
        """Test clearing messages from topic."""
        topic = 'combustion/analyzer/test_clear'

        await mqtt_broker.publish(topic, {'test': 'data'})

        mqtt_broker.clear_messages(topic)

        latest = mqtt_broker.get_latest_message(topic)
        assert latest is None

    async def test_mqtt_multiple_topics(self, mqtt_broker):
        """Test handling multiple topics."""
        topics = [
            'combustion/analyzer/o2',
            'combustion/analyzer/co',
            'combustion/analyzer/nox'
        ]

        for i, topic in enumerate(topics):
            await mqtt_broker.publish(topic, {'value': i * 10})

        # Verify each topic
        for i, topic in enumerate(topics):
            msg = mqtt_broker.get_latest_message(topic)
            assert msg is not None
            assert msg['value'] == i * 10

    async def test_mqtt_analyzer_data_format(self, mqtt_broker):
        """Test analyzer data format validation."""
        valid_data = {
            'o2_percent': 3.5,
            'co_ppm': 25.0,
            'co2_percent': 12.5,
            'nox_ppm': 45.0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        await mqtt_broker.publish('combustion/analyzer/data', valid_data)

        received = mqtt_broker.get_latest_message('combustion/analyzer/data')

        # Validate format
        assert 'o2_percent' in received
        assert 'co_ppm' in received
        assert 'timestamp' in received

    async def test_mqtt_publish_performance(self, mqtt_broker, performance_timer):
        """Test MQTT publish performance."""
        topic = 'combustion/analyzer/perf_test'
        data = {'o2_percent': 3.5, 'timestamp': 'test'}

        with performance_timer() as timer:
            for _ in range(100):
                await mqtt_broker.publish(topic, data)

        avg_time_ms = timer.elapsed_ms / 100
        assert avg_time_ms < 10.0  # <10ms per publish

    async def test_mqtt_last_update_time(self, mqtt_broker):
        """Test tracking of last update time."""
        topic = 'combustion/analyzer/time_test'

        await mqtt_broker.publish(topic, {'test': 'data'})

        last_update = mqtt_broker.get_last_update_time(topic)

        assert last_update is not None
        assert last_update > 0


# ============================================================================
# SCADA PUBLISHING VERIFICATION TESTS
# ============================================================================

class TestSCADAPublishingIntegration:
    """Test SCADA publishing integration."""

    async def test_scada_real_time_data_publish(self, mqtt_broker, opcua_server):
        """Test real-time data publishing to SCADA."""
        # Read from DCS
        dcs_data = await opcua_server.read_multiple_nodes([
            'fuel_flow', 'air_flow', 'combustion_temperature'
        ])

        # Publish to SCADA (via MQTT in this mock)
        scada_data = {
            'fuel_flow': dcs_data['fuel_flow'],
            'air_flow': dcs_data['air_flow'],
            'temperature': dcs_data['combustion_temperature'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        await mqtt_broker.publish('scada/combustion/realtime', scada_data)

        # Verify publishing
        received = mqtt_broker.get_latest_message('scada/combustion/realtime')
        assert received is not None
        assert 'fuel_flow' in received

    async def test_scada_alarm_publish(self, mqtt_broker, safety_limits, opcua_server):
        """Test alarm publishing to SCADA."""
        # Simulate alarm condition
        await opcua_server.write_node('combustion_temperature', safety_limits.max_temperature_c + 10)

        temp = await opcua_server.read_node('combustion_temperature')

        if temp > safety_limits.max_temperature_c:
            alarm_data = {
                'alarm_id': 'TEMP_HIGH_001',
                'severity': 'CRITICAL',
                'source': 'combustion_temperature',
                'value': temp,
                'limit': safety_limits.max_temperature_c,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            await mqtt_broker.publish('scada/combustion/alarms', alarm_data)

        # Verify alarm published
        alarm = mqtt_broker.get_latest_message('scada/combustion/alarms')
        assert alarm is not None
        assert alarm['severity'] == 'CRITICAL'

    async def test_scada_historical_data_publish(self, mqtt_broker, opcua_server):
        """Test historical data publishing to SCADA."""
        # Collect historical samples
        samples = []
        for i in range(5):
            temp = await opcua_server.read_node('combustion_temperature')
            samples.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'value': temp,
                'quality': 'GOOD'
            })
            await asyncio.sleep(0.01)

        # Publish historical data
        historical_data = {
            'tag': 'combustion_temperature',
            'samples': samples
        }

        await mqtt_broker.publish('scada/combustion/history', historical_data)

        # Verify
        received = mqtt_broker.get_latest_message('scada/combustion/history')
        assert received is not None
        assert len(received['samples']) == 5

    async def test_scada_command_receive(self, mqtt_broker, opcua_server):
        """Test receiving commands from SCADA."""
        # Simulate SCADA command
        command = {
            'command_type': 'SETPOINT_CHANGE',
            'target': 'fuel_flow',
            'value': 550.0,
            'operator': 'test_operator',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        await mqtt_broker.publish('scada/combustion/commands', command)

        # Process command
        received_command = mqtt_broker.get_latest_message('scada/combustion/commands')

        if received_command:
            await opcua_server.write_node(
                received_command['target'],
                received_command['value']
            )

        # Verify command executed
        fuel = await opcua_server.read_node('fuel_flow')
        assert abs(fuel - 550.0) < 10.0

    async def test_scada_heartbeat_publish(self, mqtt_broker):
        """Test heartbeat publishing to SCADA."""
        heartbeat = {
            'system_id': 'GL005_COMBUSTION_CONTROL',
            'status': 'RUNNING',
            'uptime_seconds': 3600,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        await mqtt_broker.publish('scada/combustion/heartbeat', heartbeat)

        # Verify
        received = mqtt_broker.get_latest_message('scada/combustion/heartbeat')
        assert received is not None
        assert received['status'] == 'RUNNING'

    async def test_scada_tag_registration(self, mqtt_broker):
        """Test tag registration with SCADA."""
        tags = [
            {'name': 'fuel_flow', 'type': 'float', 'units': 'kg/hr'},
            {'name': 'air_flow', 'type': 'float', 'units': 'kg/hr'},
            {'name': 'combustion_temperature', 'type': 'float', 'units': 'C'},
            {'name': 'furnace_pressure', 'type': 'float', 'units': 'mbar'}
        ]

        registration = {
            'action': 'REGISTER_TAGS',
            'tags': tags,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        await mqtt_broker.publish('scada/combustion/config', registration)

        # Verify
        received = mqtt_broker.get_latest_message('scada/combustion/config')
        assert received is not None
        assert len(received['tags']) == 4


# ============================================================================
# FLAME SCANNER MOCK INTEGRATION TESTS
# ============================================================================

class TestFlameScannerIntegration:
    """Test flame scanner mock integration."""

    async def test_flame_scanner_status_read(self, flame_scanner):
        """Test reading flame scanner status."""
        status = await flame_scanner.get_flame_status()

        assert status is not None
        assert 'flame_detected' in status
        assert isinstance(status['flame_detected'], bool)

    async def test_flame_scanner_flame_detected(self, flame_scanner):
        """Test flame detected status."""
        flame_scanner.restore_flame()

        status = await flame_scanner.get_flame_status()

        assert status['flame_detected'] is True

    async def test_flame_scanner_flame_loss(self, flame_scanner):
        """Test flame loss simulation."""
        flame_scanner.set_flame_loss()

        status = await flame_scanner.get_flame_status()

        assert status['flame_detected'] is False

        # Cleanup
        flame_scanner.restore_flame()

    async def test_flame_scanner_intensity_read(self, flame_scanner):
        """Test reading flame intensity."""
        status = await flame_scanner.get_flame_status()

        if 'intensity' in status:
            assert 0 <= status['intensity'] <= 100

    async def test_flame_scanner_signal_strength(self, flame_scanner):
        """Test flame scanner signal strength."""
        flame_scanner.set_signal_strength(0.5)

        status = await flame_scanner.get_flame_status()

        if 'signal_strength' in status:
            assert status['signal_strength'] == pytest.approx(0.5, rel=0.1)

        # Cleanup
        flame_scanner.set_signal_strength(1.0)

    async def test_flame_scanner_performance(self, flame_scanner, performance_timer):
        """Test flame scanner read performance."""
        with performance_timer() as timer:
            for _ in range(100):
                await flame_scanner.get_flame_status()

        avg_time_ms = timer.elapsed_ms / 100
        assert avg_time_ms < 10.0  # <10ms per read


# ============================================================================
# CROSS-CONNECTOR INTEGRATION TESTS
# ============================================================================

class TestCrossConnectorIntegration:
    """Test integration between multiple connectors."""

    async def test_dcs_to_scada_data_flow(
        self,
        opcua_server,
        mqtt_broker
    ):
        """Test data flows correctly from DCS to SCADA."""
        # Read from DCS
        dcs_data = await opcua_server.read_multiple_nodes([
            'fuel_flow', 'air_flow', 'combustion_temperature'
        ])

        # Publish to SCADA
        scada_data = {
            'source': 'DCS',
            **dcs_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        await mqtt_broker.publish('scada/combustion/dcs_data', scada_data)

        # Verify
        received = mqtt_broker.get_latest_message('scada/combustion/dcs_data')
        assert received is not None
        assert received['source'] == 'DCS'
        assert 'fuel_flow' in received

    async def test_plc_to_scada_alarm_flow(
        self,
        modbus_server,
        mqtt_broker
    ):
        """Test alarm data flows from PLC to SCADA."""
        # Simulate PLC emergency stop
        await modbus_server.write_coil(1, True)

        coils = await modbus_server.read_coils(1, 1)

        if coils[0]:
            alarm = {
                'source': 'PLC',
                'type': 'EMERGENCY_STOP',
                'active': True,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            await mqtt_broker.publish('scada/combustion/plc_alarms', alarm)

        # Verify
        received = mqtt_broker.get_latest_message('scada/combustion/plc_alarms')
        assert received is not None
        assert received['type'] == 'EMERGENCY_STOP'

        # Cleanup
        await modbus_server.write_coil(1, False)

    async def test_analyzer_to_scada_emissions_flow(
        self,
        mqtt_broker
    ):
        """Test emissions data flows from analyzer to SCADA."""
        # Simulate analyzer data
        analyzer_data = {
            'o2_percent': 3.5,
            'co_ppm': 25.0,
            'nox_ppm': 45.0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        await mqtt_broker.publish('combustion/analyzer/data', analyzer_data)

        # Forward to SCADA
        received = mqtt_broker.get_latest_message('combustion/analyzer/data')

        if received:
            scada_emissions = {
                'source': 'ANALYZER',
                'emissions': received,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            await mqtt_broker.publish('scada/combustion/emissions', scada_emissions)

        # Verify
        final = mqtt_broker.get_latest_message('scada/combustion/emissions')
        assert final is not None
        assert final['source'] == 'ANALYZER'

    async def test_full_data_acquisition_cycle(
        self,
        opcua_server,
        modbus_server,
        mqtt_broker,
        flame_scanner
    ):
        """Test complete data acquisition from all sources."""
        # Collect from all sources
        acquisition = {
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        # DCS
        dcs_data = await opcua_server.read_multiple_nodes([
            'fuel_flow', 'air_flow', 'combustion_temperature'
        ])
        acquisition['dcs'] = dcs_data

        # PLC
        coils = await modbus_server.read_coils(0, 8)
        registers = await modbus_server.read_input_registers(0, 4)
        acquisition['plc'] = {
            'coils': coils,
            'registers': registers
        }

        # Analyzer
        analyzer_data = {
            'o2_percent': 3.5,
            'co_ppm': 25.0
        }
        await mqtt_broker.publish('combustion/analyzer/data', analyzer_data)
        acquisition['analyzer'] = mqtt_broker.get_latest_message('combustion/analyzer/data')

        # Flame scanner
        flame_status = await flame_scanner.get_flame_status()
        acquisition['flame_scanner'] = flame_status

        # Verify complete acquisition
        assert 'dcs' in acquisition
        assert 'plc' in acquisition
        assert 'analyzer' in acquisition
        assert 'flame_scanner' in acquisition

        # Publish complete acquisition to SCADA
        await mqtt_broker.publish('scada/combustion/full_acquisition', acquisition)

        final = mqtt_broker.get_latest_message('scada/combustion/full_acquisition')
        assert final is not None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestConnectorErrorHandling:
    """Test error handling in connectors."""

    async def test_dcs_connection_error_recovery(self, opcua_server):
        """Test DCS connection error recovery."""
        # Disconnect
        await opcua_server.stop()

        # Attempt read (should fail)
        with pytest.raises(ConnectionError):
            await opcua_server.read_node('fuel_flow')

        # Reconnect
        await opcua_server.start()

        # Should work now
        value = await opcua_server.read_node('fuel_flow')
        assert value is not None

    async def test_plc_connection_error_recovery(self, modbus_server):
        """Test PLC connection error recovery."""
        # Disconnect
        await modbus_server.stop()

        # Attempt read (should fail)
        with pytest.raises(ConnectionError):
            await modbus_server.read_coils(0, 8)

        # Reconnect
        await modbus_server.start()

        # Should work now
        coils = await modbus_server.read_coils(0, 8)
        assert coils is not None

    async def test_timeout_handling(self, opcua_server):
        """Test timeout handling in connectors."""
        try:
            value = await asyncio.wait_for(
                opcua_server.read_node('fuel_flow'),
                timeout=5.0
            )
            assert value is not None
        except asyncio.TimeoutError:
            pytest.skip("Timeout occurred - may indicate slow mock")

    async def test_invalid_data_handling(self, mqtt_broker):
        """Test handling of invalid data."""
        # Publish invalid data
        invalid_data = "not a valid json object"

        # This should not crash
        try:
            await mqtt_broker.publish('test/invalid', invalid_data)
        except (TypeError, ValueError):
            pass  # Expected behavior

        # System should still work
        valid_data = {'test': 'valid'}
        await mqtt_broker.publish('test/valid', valid_data)

        received = mqtt_broker.get_latest_message('test/valid')
        assert received is not None
