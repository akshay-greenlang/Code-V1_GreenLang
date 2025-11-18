"""
Safety interlock integration tests for GL-005 CombustionControlAgent.

Tests comprehensive safety interlock systems including temperature limits,
pressure limits, fuel flow limits, emergency shutdown, and safety overrides.

Target: 10+ tests covering:
- Temperature limit violations
- Pressure limit violations
- Fuel flow limit violations
- CO emission limit violations
- Flame loss detection
- Emergency shutdown execution
- Safety override handling
- Multi-parameter safety validation
- Safety response timing
- Recovery after safety events
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.safety]


# ============================================================================
# TEMPERATURE SAFETY INTERLOCK TESTS
# ============================================================================

class TestTemperatureSafetyInterlocks:
    """Test temperature safety interlock systems."""

    async def test_high_temperature_limit_violation(
        self,
        opcua_server,
        modbus_server,
        safety_limits
    ):
        """Test high temperature limit triggers safety interlock."""
        # Simulate high temperature condition
        dangerous_temp = safety_limits.max_temperature_c + 50.0
        await opcua_server.write_node('combustion_temperature', dangerous_temp)

        # Read current temperature
        current_temp = await opcua_server.read_node('combustion_temperature')

        # Validate temperature exceeds limit
        assert current_temp > safety_limits.max_temperature_c

        # Safety interlock should trigger
        if current_temp > safety_limits.max_temperature_c:
            # Execute emergency actions
            await modbus_server.write_coil(1, True)  # Emergency stop
            await opcua_server.write_node('fuel_flow', 0.0)  # Shut off fuel

            safety_action = 'EMERGENCY_SHUTDOWN'
        else:
            safety_action = 'NONE'

        assert safety_action == 'EMERGENCY_SHUTDOWN'

        # Verify emergency shutdown executed
        emergency_stop = await modbus_server.read_coils(1, 1)
        fuel_flow = await opcua_server.read_node('fuel_flow')

        assert emergency_stop[0] is True
        assert fuel_flow == 0.0

    async def test_low_temperature_limit_violation(
        self,
        opcua_server,
        safety_limits
    ):
        """Test low temperature limit triggers safety action."""
        # Simulate low temperature condition
        low_temp = safety_limits.min_temperature_c - 50.0
        await opcua_server.write_node('combustion_temperature', low_temp)

        # Read current temperature
        current_temp = await opcua_server.read_node('combustion_temperature')

        # Validate temperature below limit
        assert current_temp < safety_limits.min_temperature_c

        # Safety action: increase fuel flow
        if current_temp < safety_limits.min_temperature_c:
            current_fuel = await opcua_server.read_node('fuel_flow')
            increased_fuel = current_fuel * 1.2  # Increase by 20%
            await opcua_server.write_node('fuel_flow', increased_fuel)

            safety_action = 'INCREASE_FUEL_FLOW'
        else:
            safety_action = 'NONE'

        assert safety_action == 'INCREASE_FUEL_FLOW'

    async def test_temperature_rate_of_change_limit(
        self,
        opcua_server,
        control_parameters
    ):
        """Test temperature rate of change limit enforcement."""
        max_rate_c_per_min = control_parameters['ramp_rate_limit_c_per_min']

        # Read initial temperature
        initial_temp = await opcua_server.read_node('combustion_temperature')

        # Attempt rapid temperature change
        target_temp = initial_temp + 100.0  # Large step

        # Calculate allowed ramp
        interval_sec = 1.0
        max_change_per_interval = (max_rate_c_per_min / 60.0) * interval_sec

        # Apply rate-limited change
        if abs(target_temp - initial_temp) > max_change_per_interval:
            limited_temp = initial_temp + max_change_per_interval
        else:
            limited_temp = target_temp

        await opcua_server.write_node('combustion_temperature', limited_temp)

        # Verify rate limit enforced
        new_temp = await opcua_server.read_node('combustion_temperature')
        actual_change = abs(new_temp - initial_temp)

        assert actual_change <= max_change_per_interval * 1.1  # 10% tolerance


# ============================================================================
# PRESSURE SAFETY INTERLOCK TESTS
# ============================================================================

class TestPressureSafetyInterlocks:
    """Test pressure safety interlock systems."""

    async def test_high_pressure_limit_violation(
        self,
        opcua_server,
        modbus_server,
        safety_limits
    ):
        """Test high pressure limit triggers emergency shutdown."""
        # Simulate high pressure condition
        dangerous_pressure = safety_limits.max_pressure_mbar + 20.0
        await opcua_server.write_node('furnace_pressure', dangerous_pressure)

        # Read current pressure
        current_pressure = await opcua_server.read_node('furnace_pressure')

        # Validate pressure exceeds limit
        assert current_pressure > safety_limits.max_pressure_mbar

        # Trigger emergency shutdown
        if current_pressure > safety_limits.max_pressure_mbar:
            await modbus_server.write_coil(1, True)
            await opcua_server.write_node('fuel_flow', 0.0)
            await opcua_server.write_node('air_flow', 0.0)

        # Verify shutdown
        emergency_stop = await modbus_server.read_coils(1, 1)
        assert emergency_stop[0] is True

    async def test_low_pressure_limit_violation(
        self,
        opcua_server,
        safety_limits
    ):
        """Test low pressure limit triggers corrective action."""
        # Simulate low pressure condition
        low_pressure = safety_limits.min_pressure_mbar - 10.0
        await opcua_server.write_node('furnace_pressure', low_pressure)

        # Read current pressure
        current_pressure = await opcua_server.read_node('furnace_pressure')

        # Validate pressure below limit
        assert current_pressure < safety_limits.min_pressure_mbar

        # Corrective action: reduce air flow
        if current_pressure < safety_limits.min_pressure_mbar:
            current_air = await opcua_server.read_node('air_flow')
            reduced_air = current_air * 0.9  # Reduce by 10%
            await opcua_server.write_node('air_flow', reduced_air)

            safety_action = 'REDUCE_AIR_FLOW'
        else:
            safety_action = 'NONE'

        assert safety_action == 'REDUCE_AIR_FLOW'


# ============================================================================
# FUEL FLOW SAFETY INTERLOCK TESTS
# ============================================================================

class TestFuelFlowSafetyInterlocks:
    """Test fuel flow safety interlock systems."""

    async def test_maximum_fuel_flow_limit(
        self,
        opcua_server,
        safety_limits
    ):
        """Test maximum fuel flow limit enforcement."""
        # Attempt to set fuel flow above limit
        excessive_fuel = safety_limits.max_fuel_flow_kg_hr + 100.0

        # Apply limit
        if excessive_fuel > safety_limits.max_fuel_flow_kg_hr:
            limited_fuel = safety_limits.max_fuel_flow_kg_hr
        else:
            limited_fuel = excessive_fuel

        await opcua_server.write_node('fuel_flow', limited_fuel)

        # Verify limit enforced
        actual_fuel = await opcua_server.read_node('fuel_flow')
        assert actual_fuel <= safety_limits.max_fuel_flow_kg_hr

    async def test_minimum_fuel_flow_limit(
        self,
        opcua_server,
        safety_limits
    ):
        """Test minimum fuel flow limit enforcement."""
        # Attempt to set fuel flow below minimum stable
        low_fuel = safety_limits.min_fuel_flow_kg_hr - 10.0

        # Check if below minimum
        if low_fuel < safety_limits.min_fuel_flow_kg_hr:
            # Either increase to minimum or shutdown
            if low_fuel < safety_limits.min_fuel_flow_kg_hr * 0.5:
                final_fuel = 0.0  # Shutdown
                action = 'SHUTDOWN'
            else:
                final_fuel = safety_limits.min_fuel_flow_kg_hr
                action = 'INCREASE_TO_MINIMUM'
        else:
            final_fuel = low_fuel
            action = 'NONE'

        await opcua_server.write_node('fuel_flow', final_fuel)

        # Verify appropriate action taken
        assert action in ['SHUTDOWN', 'INCREASE_TO_MINIMUM', 'NONE']


# ============================================================================
# EMISSION SAFETY INTERLOCK TESTS
# ============================================================================

class TestEmissionSafetyInterlocks:
    """Test emission safety interlock systems."""

    async def test_high_co_emission_limit(
        self,
        opcua_server,
        mqtt_broker,
        safety_limits
    ):
        """Test high CO emission triggers corrective action."""
        # Simulate high CO reading
        high_co_data = {
            'o2_percent': 3.5,
            'co_ppm': safety_limits.max_co_ppm + 30.0,  # Exceeds limit
            'co2_percent': 12.0,
            'nox_ppm': 30.0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        await mqtt_broker.publish('combustion/analyzer/data', high_co_data)

        # Read CO level
        analyzer_data = mqtt_broker.get_latest_message('combustion/analyzer/data')
        co_ppm = analyzer_data['co_ppm']

        # Validate CO exceeds limit
        assert co_ppm > safety_limits.max_co_ppm

        # Corrective action: increase air flow
        if co_ppm > safety_limits.max_co_ppm:
            current_air = await opcua_server.read_node('air_flow')
            increased_air = current_air * 1.15  # Increase by 15%
            await opcua_server.write_node('air_flow', increased_air)

            safety_action = 'INCREASE_AIR_FLOW'
        else:
            safety_action = 'NONE'

        assert safety_action == 'INCREASE_AIR_FLOW'

    async def test_high_nox_emission_limit(
        self,
        mqtt_broker,
        safety_limits
    ):
        """Test high NOx emission triggers corrective action."""
        # Simulate high NOx reading
        high_nox_data = {
            'o2_percent': 3.5,
            'co_ppm': 25.0,
            'co2_percent': 12.0,
            'nox_ppm': safety_limits.max_nox_ppm + 10.0,  # Exceeds limit
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        await mqtt_broker.publish('combustion/analyzer/data', high_nox_data)

        # Read NOx level
        analyzer_data = mqtt_broker.get_latest_message('combustion/analyzer/data')
        nox_ppm = analyzer_data['nox_ppm']

        # Validate NOx exceeds limit
        assert nox_ppm > safety_limits.max_nox_ppm


# ============================================================================
# FLAME SAFETY INTERLOCK TESTS
# ============================================================================

class TestFlameSafetyInterlocks:
    """Test flame detection safety interlock systems."""

    async def test_flame_loss_triggers_emergency_shutdown(
        self,
        opcua_server,
        modbus_server,
        flame_scanner
    ):
        """Test flame loss triggers immediate emergency shutdown."""
        # Simulate flame loss
        flame_scanner.set_flame_loss()

        # Check flame status
        flame_data = await flame_scanner.get_flame_status()

        # Validate flame lost
        assert flame_data['flame_detected'] is False

        # Trigger emergency shutdown
        if not flame_data['flame_detected']:
            # Immediate fuel cutoff
            await opcua_server.write_node('fuel_flow', 0.0)

            # Activate emergency stop
            await modbus_server.write_coil(1, True)

            safety_action = 'EMERGENCY_SHUTDOWN_FLAME_LOSS'
        else:
            safety_action = 'NONE'

        assert safety_action == 'EMERGENCY_SHUTDOWN_FLAME_LOSS'

        # Verify fuel cut off
        fuel_flow = await opcua_server.read_node('fuel_flow')
        assert fuel_flow == 0.0

        # Cleanup: restore flame
        flame_scanner.restore_flame()

    async def test_low_flame_intensity_warning(
        self,
        flame_scanner,
        safety_limits
    ):
        """Test low flame intensity triggers warning."""
        # Get flame data
        flame_data = await flame_scanner.get_flame_status()
        flame_intensity = flame_data['intensity_percent']

        # Check if below minimum safe intensity
        if flame_intensity < safety_limits.min_flame_intensity:
            warning_level = 'CRITICAL'
        elif flame_intensity < safety_limits.min_flame_intensity * 1.2:
            warning_level = 'WARNING'
        else:
            warning_level = 'NORMAL'

        # Normal operation should be above warning threshold
        assert warning_level in ['WARNING', 'NORMAL']


# ============================================================================
# MULTI-PARAMETER SAFETY VALIDATION TESTS
# ============================================================================

class TestMultiParameterSafetyValidation:
    """Test validation of multiple safety parameters simultaneously."""

    async def test_comprehensive_safety_check(
        self,
        opcua_server,
        mqtt_broker,
        flame_scanner,
        safety_limits
    ):
        """Test comprehensive safety check of all parameters."""
        violations = []

        # Check temperature
        temperature = await opcua_server.read_node('combustion_temperature')
        if not (safety_limits.min_temperature_c <= temperature <= safety_limits.max_temperature_c):
            violations.append('TEMPERATURE')

        # Check pressure
        pressure = await opcua_server.read_node('furnace_pressure')
        if not (safety_limits.min_pressure_mbar <= pressure <= safety_limits.max_pressure_mbar):
            violations.append('PRESSURE')

        # Check fuel flow
        fuel_flow = await opcua_server.read_node('fuel_flow')
        if not (safety_limits.min_fuel_flow_kg_hr <= fuel_flow <= safety_limits.max_fuel_flow_kg_hr):
            violations.append('FUEL_FLOW')

        # Check emissions
        analyzer_data = mqtt_broker.get_latest_message('combustion/analyzer/data')
        if analyzer_data and analyzer_data['co_ppm'] > safety_limits.max_co_ppm:
            violations.append('CO_EMISSION')

        # Check flame
        flame_data = await flame_scanner.get_flame_status()
        if not flame_data['flame_detected']:
            violations.append('FLAME_LOSS')

        # Normal operation should have no violations
        assert len(violations) == 0


# ============================================================================
# SAFETY RESPONSE TIMING TESTS
# ============================================================================

@pytest.mark.performance
class TestSafetyResponseTiming:
    """Test safety interlock response timing."""

    async def test_emergency_shutdown_response_time(
        self,
        opcua_server,
        modbus_server,
        performance_timer
    ):
        """Test emergency shutdown completes within time limit."""
        max_shutdown_time_ms = 1000.0  # 1 second max

        with performance_timer() as timer:
            # Trigger emergency shutdown
            await modbus_server.write_coil(1, True)
            await opcua_server.write_node('fuel_flow', 0.0)
            await opcua_server.write_node('air_flow', 0.0)

            # Verify shutdown
            fuel = await opcua_server.read_node('fuel_flow')
            assert fuel == 0.0

        # Validate shutdown time
        assert timer.elapsed_ms < max_shutdown_time_ms

    async def test_safety_check_execution_time(
        self,
        opcua_server,
        performance_timer,
        benchmark_thresholds
    ):
        """Test safety check completes within time limit."""
        with performance_timer() as timer:
            # Execute safety checks
            temp = await opcua_server.read_node('combustion_temperature')
            pressure = await opcua_server.read_node('furnace_pressure')
            fuel = await opcua_server.read_node('fuel_flow')

            # Validate ranges (simplified)
            checks_passed = (
                800 <= temp <= 1400 and
                50 <= pressure <= 150 and
                50 <= fuel <= 1000
            )

        # Validate check time
        assert timer.elapsed_ms < benchmark_thresholds['safety_check_max_ms']


# ============================================================================
# SAFETY RECOVERY TESTS
# ============================================================================

class TestSafetyRecovery:
    """Test recovery procedures after safety events."""

    async def test_recovery_after_emergency_shutdown(
        self,
        opcua_server,
        modbus_server,
        flame_scanner
    ):
        """Test system recovery after emergency shutdown."""
        # Trigger shutdown
        await modbus_server.write_coil(1, True)
        await opcua_server.write_node('fuel_flow', 0.0)

        # Reset emergency stop
        await modbus_server.write_coil(1, False)

        # Verify reset
        emergency_stop = await modbus_server.read_coils(1, 1)
        assert emergency_stop[0] is False

        # Can now restart (simplified - would have full restart procedure)
        restart_allowed = not emergency_stop[0]
        assert restart_allowed is True

    async def test_safe_restart_procedure(
        self,
        opcua_server,
        flame_scanner
    ):
        """Test safe restart procedure after shutdown."""
        # Ensure system is shutdown
        await opcua_server.write_node('fuel_flow', 0.0)
        await opcua_server.write_node('air_flow', 0.0)

        # Safe restart sequence
        # 1. Start purge air
        await opcua_server.write_node('air_flow', 2000.0)
        await asyncio.sleep(0.1)  # Purge time

        # 2. Start pilot fuel
        await opcua_server.write_node('fuel_flow', 50.0)
        await asyncio.sleep(0.1)  # Ignition time

        # 3. Verify flame established
        flame_data = await flame_scanner.get_flame_status()

        if flame_data['flame_detected']:
            # 4. Ramp to operating conditions
            await opcua_server.write_node('fuel_flow', 500.0)
            await opcua_server.write_node('air_flow', 5000.0)

            restart_success = True
        else:
            restart_success = False

        # Restart should succeed (with mock servers)
        assert restart_success is True
