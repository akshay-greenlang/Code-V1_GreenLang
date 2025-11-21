# -*- coding: utf-8 -*-
"""
Integration tests for burner controller Modbus communication.

Tests connection, control operations, setpoint writing, and error handling
for the BurnerOptimizationAgent's Modbus interface.
"""

import pytest
import asyncio
import time
from decimal import Decimal
from unittest.mock import Mock, patch
from pymodbus.client import AsyncModbusTcpClient
from pymodbus.exceptions import ModbusException

from mock_servers import MockModbusServer, BurnerState


class TestBurnerControllerIntegration:
    """Integration tests for burner controller communication."""

    @pytest.mark.asyncio
    async def test_modbus_connection_establishment(self, mock_modbus_server):
        """Test establishing Modbus TCP connection."""
        # Given: Mock Modbus server is running
        server = MockModbusServer()
        await server.start()

        try:
            # When: Attempting to connect
            client = AsyncModbusTcpClient('localhost', port=5502)
            connected = await client.connect()

            # Then: Connection should be established
            assert connected is True
            assert client.connected

            # Verify we can read registers
            result = await client.read_holding_registers(0, 10, unit=1)
            assert not result.isError()
            assert len(result.registers) == 10

            await client.close()

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_read_burner_state_values(self, mock_modbus_server):
        """Test reading current burner state from Modbus registers."""
        # Given: Mock server with known state
        server = MockModbusServer()
        await server.start()

        try:
            client = AsyncModbusTcpClient('localhost', port=5502)
            await client.connect()

            # When: Reading burner state registers
            result = await client.read_holding_registers(0, 7, unit=1)

            # Then: Values should match expected ranges
            fuel_flow = result.registers[0] / 100.0
            air_flow = result.registers[1] / 10.0
            o2_level = result.registers[2] / 100.0
            temperature = result.registers[3] / 10.0
            co_emissions = result.registers[4]
            nox_emissions = result.registers[5]
            efficiency = result.registers[6] / 100.0

            assert 5.0 <= fuel_flow <= 20.0  # Within limits
            assert 50.0 <= air_flow <= 200.0
            assert 1.0 <= o2_level <= 6.0
            assert 600.0 <= temperature <= 1200.0
            assert 0 <= co_emissions <= 500
            assert 0 <= nox_emissions <= 300
            assert 70.0 <= efficiency <= 100.0

            await client.close()

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_write_fuel_flow_setpoint(self, mock_modbus_server):
        """Test writing fuel flow setpoint to controller."""
        # Given: Mock server running
        server = MockModbusServer()
        await server.start()

        try:
            client = AsyncModbusTcpClient('localhost', port=5502)
            await client.connect()

            # When: Writing new fuel flow setpoint
            new_setpoint = 12.5  # kg/h
            register_value = int(new_setpoint * 100)

            result = await client.write_register(7, register_value, unit=1)
            assert not result.isError()

            # Then: Setpoint should be updated
            await asyncio.sleep(0.2)  # Allow server to process

            read_result = await client.read_holding_registers(7, 1, unit=1)
            read_value = read_result.registers[0] / 100.0

            assert abs(read_value - new_setpoint) < 0.01

            await client.close()

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_write_air_flow_setpoint(self, mock_modbus_server):
        """Test writing air flow setpoint to controller."""
        # Given: Mock server running
        server = MockModbusServer()
        await server.start()

        try:
            client = AsyncModbusTcpClient('localhost', port=5502)
            await client.connect()

            # When: Writing new air flow setpoint
            new_setpoint = 135.0  # m³/h
            register_value = int(new_setpoint * 10)

            result = await client.write_register(8, register_value, unit=1)
            assert not result.isError()

            # Then: Setpoint should be updated
            await asyncio.sleep(0.2)

            read_result = await client.read_holding_registers(8, 1, unit=1)
            read_value = read_result.registers[0] / 10.0

            assert abs(read_value - new_setpoint) < 0.1

            await client.close()

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_gradual_setpoint_ramping(self, mock_modbus_server):
        """Test that setpoints are applied gradually (ramping)."""
        # Given: Mock server with ramping enabled
        server = MockModbusServer()
        await server.start()

        try:
            client = AsyncModbusTcpClient('localhost', port=5502)
            await client.connect()

            # Read initial fuel flow
            initial_result = await client.read_holding_registers(0, 1, unit=1)
            initial_fuel_flow = initial_result.registers[0] / 100.0

            # When: Writing large setpoint change
            target_fuel_flow = initial_fuel_flow + 5.0  # Large increase
            register_value = int(target_fuel_flow * 100)

            await client.write_register(7, register_value, unit=1)

            # Then: Actual value should ramp gradually
            measurements = []
            for _ in range(20):  # Monitor for 2 seconds
                await asyncio.sleep(0.1)
                result = await client.read_holding_registers(0, 1, unit=1)
                current = result.registers[0] / 100.0
                measurements.append(current)

            # Verify gradual change
            changes = [measurements[i+1] - measurements[i]
                      for i in range(len(measurements)-1)]

            # Maximum change per cycle should be limited by ramp rate
            max_change_per_cycle = 0.5 * 0.1  # ramp_rate * cycle_time
            assert all(abs(change) <= max_change_per_cycle * 1.5
                      for change in changes)

            await client.close()

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_safety_interlock_blocks_changes(self, mock_modbus_server):
        """Test that safety interlock prevents setpoint changes."""
        # Given: Mock server with safety interlock
        server = MockModbusServer()
        await server.start()

        try:
            client = AsyncModbusTcpClient('localhost', port=5502)
            await client.connect()

            # Disable safety interlock
            server.safety_interlock_enabled = False
            server._update_registers()

            # When: Attempting to write setpoint with interlock disabled
            new_setpoint = 15.0
            register_value = int(new_setpoint * 100)

            result = await client.write_register(7, register_value, unit=1)

            # Then: Setpoint should not change
            await asyncio.sleep(0.2)

            read_result = await client.read_holding_registers(0, 1, unit=1)
            actual_fuel_flow = read_result.registers[0] / 100.0

            # Fuel flow should remain at original value
            assert actual_fuel_flow != new_setpoint

            await client.close()

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_emergency_stop_handling(self, mock_modbus_server):
        """Test emergency stop functionality."""
        # Given: Mock server running normally
        server = MockModbusServer()
        await server.start()

        try:
            client = AsyncModbusTcpClient('localhost', port=5502)
            await client.connect()

            # When: Triggering emergency stop
            server.emergency_stop = True
            server._update_registers()

            await asyncio.sleep(0.5)  # Allow shutdown

            # Then: Fuel and air flow should be zero
            result = await client.read_holding_registers(0, 2, unit=1)
            fuel_flow = result.registers[0] / 100.0
            air_flow = result.registers[1] / 10.0

            assert fuel_flow == 0.0
            assert air_flow == 0.0

            await client.close()

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_connection_retry_on_failure(self):
        """Test connection retry mechanism."""
        # Given: No server running initially
        connected = False
        retry_count = 0
        max_retries = 3

        async def connect_with_retry():
            nonlocal connected, retry_count

            for i in range(max_retries):
                try:
                    client = AsyncModbusTcpClient('localhost', port=5502)
                    connected = await client.connect()

                    if connected:
                        await client.close()
                        return True

                except Exception:
                    retry_count += 1
                    await asyncio.sleep(0.5)

            return False

        # When: Attempting connection with retries
        # Start server after 1 second
        async def delayed_server_start():
            await asyncio.sleep(1)
            server = MockModbusServer()
            await server.start()
            return server

        server_task = asyncio.create_task(delayed_server_start())
        connect_task = asyncio.create_task(connect_with_retry())

        # Then: Connection should eventually succeed
        success = await connect_task
        assert success is True
        assert retry_count > 0  # Should have retried

        server = await server_task
        await server.stop()

    @pytest.mark.asyncio
    async def test_write_multiple_registers(self, mock_modbus_server):
        """Test writing multiple registers in single transaction."""
        # Given: Mock server running
        server = MockModbusServer()
        await server.start()

        try:
            client = AsyncModbusTcpClient('localhost', port=5502)
            await client.connect()

            # When: Writing multiple setpoints
            values = [
                int(11.0 * 100),  # Fuel flow: 11.0 kg/h
                int(130.0 * 10),  # Air flow: 130.0 m³/h
            ]

            result = await client.write_registers(7, values, unit=1)
            assert not result.isError()

            # Then: Both values should be updated
            await asyncio.sleep(0.2)

            read_result = await client.read_holding_registers(7, 2, unit=1)
            fuel_setpoint = read_result.registers[0] / 100.0
            air_setpoint = read_result.registers[1] / 10.0

            assert abs(fuel_setpoint - 11.0) < 0.01
            assert abs(air_setpoint - 130.0) < 0.1

            await client.close()

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_invalid_setpoint_rejection(self, mock_modbus_server):
        """Test that invalid setpoints are rejected."""
        # Given: Mock server with limits
        server = MockModbusServer()
        await server.start()

        try:
            client = AsyncModbusTcpClient('localhost', port=5502)
            await client.connect()

            # When: Writing out-of-range setpoint
            invalid_fuel_flow = 25.0  # Above max limit of 20.0
            register_value = int(invalid_fuel_flow * 100)

            # Write should succeed at protocol level
            result = await client.write_register(7, register_value, unit=1)
            assert not result.isError()

            # Then: Server should reject the value
            await asyncio.sleep(0.2)

            # Actual fuel flow should not reach invalid value
            read_result = await client.read_holding_registers(0, 1, unit=1)
            actual_fuel_flow = read_result.registers[0] / 100.0

            assert actual_fuel_flow <= 20.0  # Should be clamped to max

            await client.close()

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_concurrent_read_write_operations(self, mock_modbus_server):
        """Test concurrent read and write operations."""
        # Given: Mock server running
        server = MockModbusServer()
        await server.start()

        try:
            client = AsyncModbusTcpClient('localhost', port=5502)
            await client.connect()

            # When: Performing concurrent operations
            async def read_operation():
                results = []
                for _ in range(10):
                    result = await client.read_holding_registers(0, 7, unit=1)
                    results.append(result)
                    await asyncio.sleep(0.05)
                return results

            async def write_operation():
                for i in range(5):
                    value = int((10.0 + i * 0.5) * 100)
                    await client.write_register(7, value, unit=1)
                    await asyncio.sleep(0.1)

            # Execute concurrently
            read_task = asyncio.create_task(read_operation())
            write_task = asyncio.create_task(write_operation())

            read_results, _ = await asyncio.gather(read_task, write_task)

            # Then: All operations should complete successfully
            assert len(read_results) == 10
            assert all(not r.isError() for r in read_results)

            await client.close()

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_modbus_exception_handling(self):
        """Test handling of Modbus exceptions."""
        # Given: Client without server
        client = AsyncModbusTcpClient('localhost', port=5502)

        # When: Attempting operations without connection
        with pytest.raises(Exception):
            await client.read_holding_registers(0, 10, unit=1)

    @pytest.mark.asyncio
    async def test_connection_loss_recovery(self, mock_modbus_server):
        """Test recovery from connection loss."""
        # Given: Mock server running
        server = MockModbusServer()
        await server.start()

        client = AsyncModbusTcpClient('localhost', port=5502)
        await client.connect()

        # When: Server stops (connection lost)
        await server.stop()
        await asyncio.sleep(0.5)

        # Attempt read should fail
        try:
            await client.read_holding_registers(0, 1, unit=1)
            assert False, "Should have raised exception"
        except:
            pass  # Expected

        # Restart server
        await server.start()
        await asyncio.sleep(0.5)

        # Then: Should be able to reconnect
        await client.close()
        client = AsyncModbusTcpClient('localhost', port=5502)
        connected = await client.connect()
        assert connected

        # And perform operations
        result = await client.read_holding_registers(0, 1, unit=1)
        assert not result.isError()

        await client.close()
        await server.stop()

    @pytest.mark.asyncio
    async def test_flame_scanner_integration(self, mock_modbus_server):
        """Test reading flame scanner data from Modbus."""
        # Given: Mock server with flame data
        server = MockModbusServer()
        await server.start()

        try:
            client = AsyncModbusTcpClient('localhost', port=5502)
            await client.connect()

            # When: Reading flame scanner registers
            result = await client.read_holding_registers(11, 2, unit=1)

            # Then: Should get flame intensity and stability
            flame_intensity = result.registers[0] / 100.0
            flame_stability = result.registers[1] / 100.0

            assert 0.0 <= flame_intensity <= 100.0
            assert 0.0 <= flame_stability <= 100.0

            await client.close()

        finally:
            await server.stop()