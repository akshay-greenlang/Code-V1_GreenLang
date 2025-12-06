# -*- coding: utf-8 -*-
"""
Modbus Integration Tests for Process Heat Agents
=================================================

Comprehensive integration tests for Modbus TCP/RTU gateway functionality:
- Holding register read/write (function codes 03, 06, 16)
- Input register read (function code 04)
- Coil read/write (function codes 01, 05, 15)
- Discrete input read (function code 02)
- Connection pooling
- Error handling (timeout, invalid address, communication errors)

Test Coverage Target: 85%+

References:
- greenlang/infrastructure/protocols/modbus_gateway.py

Author: GreenLang Test Engineering Team
Date: December 2025
"""

import asyncio
import pytest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from tests.integration.protocols.conftest import (
    MockModbusServer,
    MockModbusRegister,
)


# =============================================================================
# Test Class: Modbus Connection Tests
# =============================================================================


class TestModbusConnection:
    """Test Modbus connection establishment and management."""

    @pytest.mark.asyncio
    async def test_connection_establishment(self, mock_modbus_server):
        """Test successful connection to Modbus server."""
        server = mock_modbus_server

        connected = await server.connect()

        assert connected is True
        assert server.connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, connected_mock_modbus_server):
        """Test disconnection from Modbus server."""
        server = connected_mock_modbus_server

        assert server.connected is True

        await server.disconnect()

        assert server.connected is False

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, mock_modbus_server):
        """Test connection error handling."""
        server = mock_modbus_server
        server.simulate_error()

        with pytest.raises(ConnectionError, match="Simulated connection error"):
            await server.connect()

    @pytest.mark.asyncio
    async def test_connection_with_latency(self, mock_modbus_server):
        """Test connection with simulated latency."""
        server = mock_modbus_server
        server.set_latency(100)  # 100ms latency

        start_time = datetime.utcnow()
        await server.connect()
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000

        assert elapsed >= 100

    @pytest.mark.asyncio
    async def test_connection_timeout(self, mock_modbus_server):
        """Test connection timeout handling."""
        server = mock_modbus_server
        server.enable_timeout_mode()

        with pytest.raises(TimeoutError, match="timeout"):
            await asyncio.wait_for(server.connect(), timeout=0.1)


# =============================================================================
# Test Class: Holding Register Tests
# =============================================================================


class TestModbusHoldingRegisters:
    """Test Modbus holding register read/write operations."""

    @pytest.mark.asyncio
    async def test_read_single_holding_register(self, connected_mock_modbus_server):
        """Test reading a single holding register."""
        server = connected_mock_modbus_server

        values = await server.read_holding_registers(address=0, count=1)

        assert len(values) == 1
        assert values[0] == 855  # Temperature (scale 0.1 = 85.5 C)

    @pytest.mark.asyncio
    async def test_read_multiple_holding_registers(self, connected_mock_modbus_server):
        """Test reading multiple holding registers."""
        server = connected_mock_modbus_server

        values = await server.read_holding_registers(address=0, count=4)

        assert len(values) == 4
        assert values[0] == 855   # Temperature
        assert values[1] == 25    # Pressure
        assert values[2] == 1500  # Flow rate high
        assert values[3] == 0     # Flow rate low

    @pytest.mark.asyncio
    async def test_write_single_holding_register(self, connected_mock_modbus_server):
        """Test writing a single holding register."""
        server = connected_mock_modbus_server

        # Write new temperature value (90.0 C = 900 raw)
        provenance_hash = await server.write_register(address=0, value=900)

        # Verify write
        values = await server.read_holding_registers(address=0, count=1)
        assert values[0] == 900

        # Verify provenance hash
        assert len(provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_write_multiple_holding_registers(self, connected_mock_modbus_server):
        """Test writing multiple holding registers."""
        server = connected_mock_modbus_server

        # Write new values
        new_values = [950, 30, 2000, 100]  # Temp, Pressure, Flow high, Flow low
        provenance_hash = await server.write_registers(address=0, values=new_values)

        # Verify write
        values = await server.read_holding_registers(address=0, count=4)
        assert values == new_values

    @pytest.mark.asyncio
    async def test_read_emission_factor_register(self, connected_mock_modbus_server):
        """Test reading emission factor register."""
        server = connected_mock_modbus_server

        values = await server.read_holding_registers(address=10, count=1)

        # Raw value 268, scale 0.01 = 2.68 kg CO2/kg
        assert values[0] == 268

    @pytest.mark.asyncio
    async def test_read_efficiency_register(self, connected_mock_modbus_server):
        """Test reading efficiency register."""
        server = connected_mock_modbus_server

        values = await server.read_holding_registers(address=11, count=1)

        # Raw value 92 = 92% efficiency
        assert values[0] == 92

    @pytest.mark.asyncio
    async def test_write_register_invalid_value(self, connected_mock_modbus_server):
        """Test writing invalid register value raises error."""
        server = connected_mock_modbus_server

        # Value exceeds 16-bit range
        with pytest.raises(ValueError, match="Invalid"):
            await server.write_register(address=0, value=70000)

    @pytest.mark.asyncio
    async def test_write_register_negative_value(self, connected_mock_modbus_server):
        """Test writing negative register value raises error."""
        server = connected_mock_modbus_server

        with pytest.raises(ValueError, match="Invalid"):
            await server.write_register(address=0, value=-100)

    @pytest.mark.asyncio
    async def test_read_holding_registers_with_latency(self, connected_mock_modbus_server):
        """Test reading registers with latency."""
        server = connected_mock_modbus_server
        server.set_latency(50)

        start_time = datetime.utcnow()
        await server.read_holding_registers(address=0, count=1)
        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000

        assert elapsed >= 50


# =============================================================================
# Test Class: Input Register Tests
# =============================================================================


class TestModbusInputRegisters:
    """Test Modbus input register read operations."""

    @pytest.mark.asyncio
    async def test_read_single_input_register(self, connected_mock_modbus_server):
        """Test reading a single input register."""
        server = connected_mock_modbus_server

        values = await server.read_input_registers(address=0, count=1)

        assert len(values) == 1
        assert values[0] == 857  # Actual temperature (85.7 C)

    @pytest.mark.asyncio
    async def test_read_multiple_input_registers(self, connected_mock_modbus_server):
        """Test reading multiple input registers."""
        server = connected_mock_modbus_server

        values = await server.read_input_registers(address=0, count=4)

        assert len(values) == 4
        assert values[0] == 857   # Temperature
        assert values[1] == 24    # Pressure
        assert values[2] == 1495  # Flow high
        assert values[3] == 0     # Flow low

    @pytest.mark.asyncio
    async def test_input_registers_are_readonly(self, connected_mock_modbus_server):
        """Test input registers are read-only (no write method for them)."""
        server = connected_mock_modbus_server

        # Input registers can only be read, not written
        # The write_register method only affects holding registers
        await server.write_register(address=0, value=999)

        # Input register at address 0 should still be original value
        input_values = await server.read_input_registers(address=0, count=1)
        assert input_values[0] == 857  # Unchanged

        # Holding register at address 0 should be updated
        holding_values = await server.read_holding_registers(address=0, count=1)
        assert holding_values[0] == 999

    @pytest.mark.asyncio
    async def test_compare_input_vs_holding_registers(self, connected_mock_modbus_server):
        """Test comparing input registers (actual) vs holding registers (setpoint)."""
        server = connected_mock_modbus_server

        # Read setpoints (holding)
        holding = await server.read_holding_registers(address=0, count=2)

        # Read actuals (input)
        input_vals = await server.read_input_registers(address=0, count=2)

        # Calculate deviation
        temp_setpoint = holding[0] * 0.1  # Scale factor
        temp_actual = input_vals[0] * 0.1

        temp_deviation = abs(temp_actual - temp_setpoint)

        # Should be within acceptable range
        assert temp_deviation < 1.0  # Less than 1 degree deviation


# =============================================================================
# Test Class: Coil Tests
# =============================================================================


class TestModbusCoils:
    """Test Modbus coil read/write operations."""

    @pytest.mark.asyncio
    async def test_read_single_coil(self, connected_mock_modbus_server):
        """Test reading a single coil."""
        server = connected_mock_modbus_server

        values = await server.read_coils(address=0, count=1)

        assert len(values) == 1
        assert values[0] is True  # System running

    @pytest.mark.asyncio
    async def test_read_multiple_coils(self, connected_mock_modbus_server):
        """Test reading multiple coils."""
        server = connected_mock_modbus_server

        values = await server.read_coils(address=0, count=6)

        assert len(values) == 6
        assert values[0] is True   # System running
        assert values[1] is False  # Alarm active
        assert values[2] is True   # Pump 1 on
        assert values[3] is False  # Pump 2 on
        assert values[4] is True   # Heater on
        assert values[5] is False  # Emergency stop

    @pytest.mark.asyncio
    async def test_write_single_coil(self, connected_mock_modbus_server):
        """Test writing a single coil."""
        server = connected_mock_modbus_server

        # Turn on alarm
        provenance_hash = await server.write_coil(address=1, value=True)

        # Verify write
        values = await server.read_coils(address=1, count=1)
        assert values[0] is True

        # Verify provenance hash
        assert len(provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_write_multiple_coils(self, connected_mock_modbus_server):
        """Test writing multiple coils."""
        server = connected_mock_modbus_server

        # Turn on both pumps
        provenance_hash = await server.write_coils(address=2, values=[True, True])

        # Verify write
        values = await server.read_coils(address=2, count=2)
        assert values == [True, True]

    @pytest.mark.asyncio
    async def test_toggle_coil(self, connected_mock_modbus_server):
        """Test toggling a coil value."""
        server = connected_mock_modbus_server

        # Read initial value
        initial = await server.read_coils(address=3, count=1)
        initial_value = initial[0]

        # Toggle
        await server.write_coil(address=3, value=not initial_value)

        # Verify toggled
        toggled = await server.read_coils(address=3, count=1)
        assert toggled[0] != initial_value

    @pytest.mark.asyncio
    async def test_emergency_stop_coil(self, connected_mock_modbus_server):
        """Test emergency stop coil handling."""
        server = connected_mock_modbus_server

        # Emergency stop should be off initially
        values = await server.read_coils(address=5, count=1)
        assert values[0] is False

        # Trigger emergency stop
        await server.write_coil(address=5, value=True)

        # Verify triggered
        values = await server.read_coils(address=5, count=1)
        assert values[0] is True


# =============================================================================
# Test Class: Discrete Input Tests
# =============================================================================


class TestModbusDiscreteInputs:
    """Test Modbus discrete input read operations."""

    @pytest.mark.asyncio
    async def test_read_single_discrete_input(self, connected_mock_modbus_server):
        """Test reading a single discrete input."""
        server = connected_mock_modbus_server

        values = await server.read_discrete_inputs(address=0, count=1)

        assert len(values) == 1
        assert values[0] is True  # Emergency stop released

    @pytest.mark.asyncio
    async def test_read_multiple_discrete_inputs(self, connected_mock_modbus_server):
        """Test reading multiple discrete inputs."""
        server = connected_mock_modbus_server

        values = await server.read_discrete_inputs(address=0, count=4)

        assert len(values) == 4
        assert values[0] is True   # Emergency stop released
        assert values[1] is True   # Safety circuit OK
        assert values[2] is False  # High temperature alarm
        assert values[3] is False  # Low pressure alarm

    @pytest.mark.asyncio
    async def test_safety_interlocks(self, connected_mock_modbus_server):
        """Test safety interlock discrete inputs."""
        server = connected_mock_modbus_server

        # Read all safety inputs
        safety_inputs = await server.read_discrete_inputs(address=0, count=2)

        # Both should be OK for system to run
        emergency_stop_released = safety_inputs[0]
        safety_circuit_ok = safety_inputs[1]

        assert emergency_stop_released is True
        assert safety_circuit_ok is True

        # System can run if both are true
        system_can_run = emergency_stop_released and safety_circuit_ok
        assert system_can_run is True

    @pytest.mark.asyncio
    async def test_alarm_inputs(self, connected_mock_modbus_server):
        """Test alarm discrete inputs."""
        server = connected_mock_modbus_server

        # Read alarm inputs
        alarms = await server.read_discrete_inputs(address=2, count=2)

        high_temp_alarm = alarms[0]
        low_pressure_alarm = alarms[1]

        # Both should be inactive
        assert high_temp_alarm is False
        assert low_pressure_alarm is False


# =============================================================================
# Test Class: Error Handling Tests
# =============================================================================


class TestModbusErrorHandling:
    """Test Modbus error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_operation_when_not_connected(self, mock_modbus_server):
        """Test operations fail when not connected."""
        server = mock_modbus_server

        with pytest.raises(ConnectionError, match="Not connected"):
            await server.read_holding_registers(address=0, count=1)

    @pytest.mark.asyncio
    async def test_invalid_address_error(self, connected_mock_modbus_server):
        """Test invalid address error handling."""
        server = connected_mock_modbus_server
        server.enable_invalid_address_mode()

        with pytest.raises(ValueError, match="Invalid address"):
            await server.read_holding_registers(address=0, count=1)

    @pytest.mark.asyncio
    async def test_address_out_of_range(self, connected_mock_modbus_server):
        """Test address out of range error."""
        server = connected_mock_modbus_server

        with pytest.raises(ValueError, match="out of range"):
            await server.read_holding_registers(address=-1, count=1)

    @pytest.mark.asyncio
    async def test_error_mode_operations(self, connected_mock_modbus_server):
        """Test operations fail in error mode."""
        server = connected_mock_modbus_server
        server.simulate_error()

        with pytest.raises(Exception, match="Simulated Modbus error"):
            await server.read_holding_registers(address=0, count=1)

    @pytest.mark.asyncio
    async def test_clear_error_restores_operations(self, connected_mock_modbus_server):
        """Test clearing error mode restores operations."""
        server = connected_mock_modbus_server

        server.simulate_error()

        with pytest.raises(Exception):
            await server.read_holding_registers(address=0, count=1)

        server.clear_error()

        # Should work now
        values = await server.read_holding_registers(address=0, count=1)
        assert values is not None

    @pytest.mark.asyncio
    async def test_write_invalid_values_in_batch(self, connected_mock_modbus_server):
        """Test batch write with invalid values."""
        server = connected_mock_modbus_server

        # Include an invalid value (> 65535)
        invalid_values = [100, 200, 70000, 400]

        with pytest.raises(ValueError, match="Invalid value"):
            await server.write_registers(address=0, values=invalid_values)


# =============================================================================
# Test Class: Register Map Integration
# =============================================================================


class TestModbusRegisterMap:
    """Test Modbus register map for process heat scenarios."""

    @pytest.mark.asyncio
    async def test_process_heat_register_map(
        self,
        connected_mock_modbus_server,
        sample_modbus_register_map
    ):
        """Test reading process heat data via register map."""
        server = connected_mock_modbus_server
        register_map = sample_modbus_register_map

        data = {}
        for reg in register_map:
            if reg["data_type"] == "uint16":
                values = await server.read_holding_registers(
                    address=reg["address"],
                    count=1
                )
                scaled_value = values[0] * reg["scale"]
                data[reg["name"]] = scaled_value
            elif reg["data_type"] == "uint32":
                values = await server.read_holding_registers(
                    address=reg["address"],
                    count=2
                )
                combined = (values[0] << 16) | values[1]
                data[reg["name"]] = combined * reg["scale"]

        assert data["temperature"] == pytest.approx(85.5, rel=0.01)
        assert data["pressure"] == pytest.approx(2.5, rel=0.01)
        assert data["emission_factor"] == pytest.approx(2.68, rel=0.01)
        assert data["efficiency"] == pytest.approx(0.92, rel=0.01)

    @pytest.mark.asyncio
    async def test_calculate_emissions_from_modbus(self, connected_mock_modbus_server):
        """Test emissions calculation from Modbus data."""
        server = connected_mock_modbus_server

        # Read fuel consumption (address 20, scale 0.1)
        fuel_raw = await server.read_holding_registers(address=20, count=1)
        fuel_consumption = fuel_raw[0] * 0.1  # 45.2 kg/h

        # Read emission factor (address 10, scale 0.01)
        ef_raw = await server.read_holding_registers(address=10, count=1)
        emission_factor = ef_raw[0] * 0.01  # 2.68 kg CO2/kg

        # Calculate emissions
        calculated_emissions = fuel_consumption * emission_factor

        # Read stored emissions (address 21, scale 0.01)
        stored_raw = await server.read_holding_registers(address=21, count=1)
        stored_emissions = stored_raw[0] * 0.01  # 121.14 kg CO2

        # Should match
        assert calculated_emissions == pytest.approx(stored_emissions, rel=0.01)

    @pytest.mark.asyncio
    async def test_status_register(self, connected_mock_modbus_server):
        """Test status register interpretation."""
        server = connected_mock_modbus_server

        values = await server.read_holding_registers(address=100, count=1)
        status = values[0]

        # Status 1 = Running
        assert status == 1

        # Write new status (0 = Stopped)
        await server.write_register(address=100, value=0)

        values = await server.read_holding_registers(address=100, count=1)
        assert values[0] == 0


# =============================================================================
# Test Class: Modbus Performance Tests
# =============================================================================


@pytest.mark.performance
class TestModbusPerformance:
    """Performance tests for Modbus operations."""

    @pytest.mark.asyncio
    async def test_read_latency(self, connected_mock_modbus_server, performance_timer):
        """Test register read latency."""
        server = connected_mock_modbus_server

        for _ in range(100):
            performance_timer.start()
            await server.read_holding_registers(address=0, count=1)
            performance_timer.stop()

        assert performance_timer.average_ms < 10

    @pytest.mark.asyncio
    async def test_batch_read_throughput(
        self,
        connected_mock_modbus_server,
        throughput_calculator
    ):
        """Test batch read throughput."""
        server = connected_mock_modbus_server

        throughput_calculator.start()

        for _ in range(100):
            values = await server.read_holding_registers(address=0, count=10)
            throughput_calculator.record_message(len(values) * 2)  # 2 bytes per register

        stats = throughput_calculator.get_throughput()
        assert stats["messages_per_sec"] > 100

    @pytest.mark.asyncio
    async def test_write_latency(self, connected_mock_modbus_server, performance_timer):
        """Test register write latency."""
        server = connected_mock_modbus_server

        for i in range(50):
            performance_timer.start()
            await server.write_register(address=0, value=800 + i)
            performance_timer.stop()

        assert performance_timer.average_ms < 20

    @pytest.mark.asyncio
    async def test_coil_operations_throughput(
        self,
        connected_mock_modbus_server,
        throughput_calculator
    ):
        """Test coil operation throughput."""
        server = connected_mock_modbus_server

        throughput_calculator.start()

        for _ in range(100):
            await server.read_coils(address=0, count=8)
            throughput_calculator.record_message(1)

        stats = throughput_calculator.get_throughput()
        assert stats["messages_per_sec"] > 100


# =============================================================================
# Test Class: Modbus Statistics Tests
# =============================================================================


class TestModbusStatistics:
    """Test Modbus server statistics."""

    @pytest.mark.asyncio
    async def test_server_statistics(self, connected_mock_modbus_server):
        """Test server statistics."""
        server = connected_mock_modbus_server

        stats = server.get_statistics()

        assert stats["connected"] is True
        assert stats["host"] == "localhost"
        assert stats["port"] == 502
        assert stats["holding_register_count"] > 0
        assert stats["input_register_count"] > 0
        assert stats["coil_count"] > 0
        assert stats["discrete_input_count"] > 0

    @pytest.mark.asyncio
    async def test_statistics_error_mode_tracking(self, connected_mock_modbus_server):
        """Test statistics track error mode."""
        server = connected_mock_modbus_server

        stats = server.get_statistics()
        assert stats["error_mode"] is False

        server.simulate_error()

        stats = server.get_statistics()
        assert stats["error_mode"] is True

    @pytest.mark.asyncio
    async def test_statistics_latency_tracking(self, connected_mock_modbus_server):
        """Test statistics track latency setting."""
        server = connected_mock_modbus_server

        stats = server.get_statistics()
        assert stats["latency_ms"] == 0

        server.set_latency(100)

        stats = server.get_statistics()
        assert stats["latency_ms"] == 100


# =============================================================================
# Test Class: Modbus Process Heat Integration
# =============================================================================


class TestModbusProcessHeatIntegration:
    """Integration tests for process heat data via Modbus."""

    @pytest.mark.asyncio
    async def test_complete_process_heat_data_cycle(self, connected_mock_modbus_server):
        """Test complete process heat data read cycle."""
        server = connected_mock_modbus_server

        # Read all holding registers for process data
        process_data = await server.read_holding_registers(address=0, count=22)

        # Parse process data
        temperature = process_data[0] * 0.1
        pressure = process_data[1] * 0.1
        flow_rate = (process_data[2] << 16) | process_data[3]
        emission_factor = process_data[10] * 0.01
        efficiency = process_data[11] * 0.01
        fuel_consumption = process_data[20] * 0.1
        total_emissions = process_data[21] * 0.01

        # Validate ranges
        assert 50 <= temperature <= 150  # Valid temp range
        assert 0 < pressure <= 10  # Valid pressure range
        assert flow_rate > 0  # Positive flow
        assert 0 < emission_factor <= 5  # Valid EF range
        assert 0 < efficiency <= 1  # Valid efficiency
        assert fuel_consumption > 0
        assert total_emissions > 0

    @pytest.mark.asyncio
    async def test_setpoint_adjustment_workflow(self, connected_mock_modbus_server):
        """Test setpoint adjustment workflow."""
        server = connected_mock_modbus_server

        # Read current setpoint
        current = await server.read_holding_registers(address=0, count=1)
        current_temp_setpoint = current[0] * 0.1

        # Calculate new setpoint (increase by 5 degrees)
        new_setpoint = current_temp_setpoint + 5
        new_raw = int(new_setpoint / 0.1)

        # Write new setpoint
        await server.write_register(address=0, value=new_raw)

        # Verify
        updated = await server.read_holding_registers(address=0, count=1)
        updated_setpoint = updated[0] * 0.1

        assert updated_setpoint == pytest.approx(new_setpoint, rel=0.01)

    @pytest.mark.asyncio
    async def test_safety_check_workflow(self, connected_mock_modbus_server):
        """Test safety check workflow using coils and discrete inputs."""
        server = connected_mock_modbus_server

        # Check safety interlocks
        safety_inputs = await server.read_discrete_inputs(address=0, count=4)
        emergency_stop_ok = safety_inputs[0]
        safety_circuit_ok = safety_inputs[1]
        high_temp_alarm = safety_inputs[2]
        low_pressure_alarm = safety_inputs[3]

        # Check system coil
        system_coils = await server.read_coils(address=0, count=2)
        system_running = system_coils[0]
        alarm_active = system_coils[1]

        # Determine if system can run
        can_run = (
            emergency_stop_ok and
            safety_circuit_ok and
            not high_temp_alarm and
            not low_pressure_alarm
        )

        assert can_run is True
        assert system_running is True
        assert alarm_active is False

    @pytest.mark.asyncio
    async def test_provenance_tracking_across_writes(self, connected_mock_modbus_server):
        """Test provenance hash generation across multiple writes."""
        server = connected_mock_modbus_server

        provenance_hashes = []

        # Multiple writes
        for i in range(5):
            hash_val = await server.write_register(address=0, value=800 + i * 10)
            provenance_hashes.append(hash_val)

        # All hashes should be unique
        assert len(set(provenance_hashes)) == 5

        # All hashes should be valid SHA-256
        for h in provenance_hashes:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)
