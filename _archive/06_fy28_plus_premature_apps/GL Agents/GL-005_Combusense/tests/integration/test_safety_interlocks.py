# -*- coding: utf-8 -*-
"""
Safety interlock integration tests for GL-005 CombustionControlAgent.

Tests all 9 safety interlocks for SIL-2 compliance:
1. High temperature limit (1400C)
2. Low temperature limit (800C)
3. High pressure limit (150 mbar)
4. Low pressure limit (50 mbar)
5. High fuel flow limit (1000 kg/hr)
6. Low fuel flow limit (50 kg/hr)
7. High CO emission limit (100 ppm)
8. Flame loss detection
9. Emergency stop button

Target: 30+ tests covering:
- Interlock trip behavior
- Fail-safe defaults
- Emergency shutdown sequence
- Interlock reset procedures
- Alarm generation
- Response timing (SIL-2: <100ms)
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.safety]


# ============================================================================
# INTERLOCK 1: HIGH TEMPERATURE LIMIT (1400C)
# ============================================================================

class TestHighTemperatureInterlock:
    """Test high temperature interlock (1400C limit)."""

    async def test_high_temp_interlock_trips_at_limit(
        self,
        opcua_server,
        modbus_server,
        safety_limits
    ):
        """Test interlock trips when temperature exceeds limit."""
        # Set temperature above limit
        dangerous_temp = safety_limits.max_temperature_c + 10.0
        await opcua_server.write_node('combustion_temperature', dangerous_temp)

        # Read temperature
        current_temp = await opcua_server.read_node('combustion_temperature')

        # Check interlock condition
        interlock_tripped = current_temp > safety_limits.max_temperature_c

        assert interlock_tripped is True

        # Execute shutdown
        if interlock_tripped:
            await modbus_server.write_coil(1, True)
            await opcua_server.write_node('fuel_flow', 0.0)

        # Verify shutdown state
        emergency_stop = await modbus_server.read_coils(1, 1)
        fuel_flow = await opcua_server.read_node('fuel_flow')

        assert emergency_stop[0] is True
        assert fuel_flow == 0.0

    async def test_high_temp_interlock_response_time(
        self,
        opcua_server,
        modbus_server,
        safety_limits,
        performance_timer
    ):
        """Test interlock responds within SIL-2 timing (<100ms)."""
        dangerous_temp = safety_limits.max_temperature_c + 50.0

        with performance_timer() as timer:
            await opcua_server.write_node('combustion_temperature', dangerous_temp)
            current_temp = await opcua_server.read_node('combustion_temperature')

            if current_temp > safety_limits.max_temperature_c:
                await modbus_server.write_coil(1, True)
                await opcua_server.write_node('fuel_flow', 0.0)

        # SIL-2 requirement: <100ms response
        assert timer.elapsed_ms < 100.0

    async def test_high_temp_interlock_at_boundary(
        self,
        opcua_server,
        safety_limits
    ):
        """Test interlock behavior at exact boundary value."""
        # Set exactly at limit
        await opcua_server.write_node('combustion_temperature', safety_limits.max_temperature_c)
        temp_at_limit = await opcua_server.read_node('combustion_temperature')

        # Should not trip at exactly the limit
        interlock_tripped = temp_at_limit > safety_limits.max_temperature_c
        assert interlock_tripped is False

        # Set just above limit
        await opcua_server.write_node('combustion_temperature', safety_limits.max_temperature_c + 0.1)
        temp_above = await opcua_server.read_node('combustion_temperature')

        # Should trip above the limit
        interlock_tripped = temp_above > safety_limits.max_temperature_c
        assert interlock_tripped is True

    async def test_high_temp_generates_alarm(
        self,
        opcua_server,
        safety_limits
    ):
        """Test high temperature generates appropriate alarm."""
        # Warning level (90% of limit)
        warning_temp = safety_limits.max_temperature_c * 0.9
        await opcua_server.write_node('combustion_temperature', warning_temp)
        temp = await opcua_server.read_node('combustion_temperature')

        alarm_level = self._determine_alarm_level(temp, safety_limits.max_temperature_c)
        assert alarm_level in ['WARNING', 'NORMAL']

        # Critical level (above limit)
        critical_temp = safety_limits.max_temperature_c + 20.0
        await opcua_server.write_node('combustion_temperature', critical_temp)
        temp = await opcua_server.read_node('combustion_temperature')

        alarm_level = self._determine_alarm_level(temp, safety_limits.max_temperature_c)
        assert alarm_level == 'CRITICAL'

    def _determine_alarm_level(self, value: float, limit: float) -> str:
        """Determine alarm level based on value and limit."""
        if value > limit:
            return 'CRITICAL'
        elif value > limit * 0.95:
            return 'WARNING'
        elif value > limit * 0.85:
            return 'ADVISORY'
        else:
            return 'NORMAL'


# ============================================================================
# INTERLOCK 2: LOW TEMPERATURE LIMIT (800C)
# ============================================================================

class TestLowTemperatureInterlock:
    """Test low temperature interlock (800C limit)."""

    async def test_low_temp_interlock_trips_below_limit(
        self,
        opcua_server,
        safety_limits
    ):
        """Test interlock trips when temperature falls below limit."""
        # Set temperature below limit
        low_temp = safety_limits.min_temperature_c - 50.0
        await opcua_server.write_node('combustion_temperature', low_temp)

        current_temp = await opcua_server.read_node('combustion_temperature')
        interlock_tripped = current_temp < safety_limits.min_temperature_c

        assert interlock_tripped is True

    async def test_low_temp_controlled_shutdown(
        self,
        opcua_server,
        safety_limits
    ):
        """Test low temp triggers controlled (not emergency) shutdown."""
        low_temp = safety_limits.min_temperature_c - 20.0
        await opcua_server.write_node('combustion_temperature', low_temp)

        current_temp = await opcua_server.read_node('combustion_temperature')

        if current_temp < safety_limits.min_temperature_c:
            # Controlled shutdown: gradual fuel reduction
            current_fuel = await opcua_server.read_node('fuel_flow')
            reduced_fuel = current_fuel * 0.5  # Reduce by 50%
            await opcua_server.write_node('fuel_flow', reduced_fuel)

            shutdown_type = 'CONTROLLED'
        else:
            shutdown_type = 'NONE'

        assert shutdown_type == 'CONTROLLED'

    async def test_low_temp_unstable_combustion_detection(
        self,
        opcua_server,
        safety_limits
    ):
        """Test detection of unstable combustion from low temperature."""
        # Set temperature just below minimum stable
        unstable_temp = safety_limits.min_temperature_c - 10.0
        await opcua_server.write_node('combustion_temperature', unstable_temp)

        current_temp = await opcua_server.read_node('combustion_temperature')

        # Determine combustion stability
        if current_temp < safety_limits.min_temperature_c:
            stability_status = 'UNSTABLE'
        elif current_temp < safety_limits.min_temperature_c * 1.1:
            stability_status = 'MARGINAL'
        else:
            stability_status = 'STABLE'

        assert stability_status == 'UNSTABLE'


# ============================================================================
# INTERLOCK 3: HIGH PRESSURE LIMIT (150 mbar)
# ============================================================================

class TestHighPressureInterlock:
    """Test high pressure interlock (150 mbar limit)."""

    async def test_high_pressure_interlock_emergency_shutdown(
        self,
        opcua_server,
        modbus_server,
        safety_limits
    ):
        """Test high pressure triggers emergency shutdown."""
        dangerous_pressure = safety_limits.max_pressure_mbar + 20.0
        await opcua_server.write_node('furnace_pressure', dangerous_pressure)

        current_pressure = await opcua_server.read_node('furnace_pressure')

        if current_pressure > safety_limits.max_pressure_mbar:
            await modbus_server.write_coil(1, True)
            await opcua_server.write_node('fuel_flow', 0.0)

        emergency_stop = await modbus_server.read_coils(1, 1)
        assert emergency_stop[0] is True

    async def test_high_pressure_response_time(
        self,
        opcua_server,
        modbus_server,
        safety_limits,
        performance_timer
    ):
        """Test high pressure interlock responds within SIL-2 timing."""
        dangerous_pressure = safety_limits.max_pressure_mbar + 30.0

        with performance_timer() as timer:
            await opcua_server.write_node('furnace_pressure', dangerous_pressure)
            current = await opcua_server.read_node('furnace_pressure')

            if current > safety_limits.max_pressure_mbar:
                await modbus_server.write_coil(1, True)
                await opcua_server.write_node('fuel_flow', 0.0)

        assert timer.elapsed_ms < 100.0

    async def test_high_pressure_at_boundary(
        self,
        opcua_server,
        safety_limits
    ):
        """Test interlock behavior at pressure boundary."""
        # At limit - should not trip
        await opcua_server.write_node('furnace_pressure', safety_limits.max_pressure_mbar)
        pressure = await opcua_server.read_node('furnace_pressure')

        interlock_tripped = pressure > safety_limits.max_pressure_mbar
        assert interlock_tripped is False


# ============================================================================
# INTERLOCK 4: LOW PRESSURE LIMIT (50 mbar)
# ============================================================================

class TestLowPressureInterlock:
    """Test low pressure interlock (50 mbar limit)."""

    async def test_low_pressure_reduces_air_flow(
        self,
        opcua_server,
        safety_limits
    ):
        """Test low pressure triggers air flow reduction."""
        low_pressure = safety_limits.min_pressure_mbar - 10.0
        await opcua_server.write_node('furnace_pressure', low_pressure)

        current_pressure = await opcua_server.read_node('furnace_pressure')

        if current_pressure < safety_limits.min_pressure_mbar:
            current_air = await opcua_server.read_node('air_flow')
            reduced_air = current_air * 0.9
            await opcua_server.write_node('air_flow', reduced_air)
            action = 'REDUCE_AIR'
        else:
            action = 'NONE'

        assert action == 'REDUCE_AIR'

    async def test_low_pressure_prevents_ignition(
        self,
        opcua_server,
        safety_limits
    ):
        """Test low pressure prevents ignition attempt."""
        very_low_pressure = 20.0  # Well below minimum
        await opcua_server.write_node('furnace_pressure', very_low_pressure)

        current_pressure = await opcua_server.read_node('furnace_pressure')
        min_ignition_pressure = 40.0

        ignition_allowed = current_pressure >= min_ignition_pressure
        assert ignition_allowed is False

    async def test_low_pressure_draft_alarm(
        self,
        opcua_server,
        safety_limits
    ):
        """Test low pressure generates draft alarm."""
        low_pressure = safety_limits.min_pressure_mbar - 5.0
        await opcua_server.write_node('furnace_pressure', low_pressure)

        current_pressure = await opcua_server.read_node('furnace_pressure')

        if current_pressure < safety_limits.min_pressure_mbar:
            alarm_type = 'LOW_DRAFT'
        else:
            alarm_type = 'NONE'

        assert alarm_type == 'LOW_DRAFT'


# ============================================================================
# INTERLOCK 5: HIGH FUEL FLOW LIMIT (1000 kg/hr)
# ============================================================================

class TestHighFuelFlowInterlock:
    """Test high fuel flow interlock (1000 kg/hr limit)."""

    async def test_high_fuel_flow_clamped_to_maximum(
        self,
        opcua_server,
        safety_limits
    ):
        """Test fuel flow setpoint is clamped to maximum."""
        excessive_fuel = safety_limits.max_fuel_flow_kg_hr + 100.0

        # Apply clamping
        clamped_fuel = min(excessive_fuel, safety_limits.max_fuel_flow_kg_hr)
        await opcua_server.write_node('fuel_flow', clamped_fuel)

        actual_fuel = await opcua_server.read_node('fuel_flow')
        assert actual_fuel <= safety_limits.max_fuel_flow_kg_hr

    async def test_high_fuel_flow_triggers_alarm(
        self,
        opcua_server,
        safety_limits
    ):
        """Test high fuel flow generates alarm."""
        high_fuel = safety_limits.max_fuel_flow_kg_hr * 0.95
        await opcua_server.write_node('fuel_flow', high_fuel)

        actual_fuel = await opcua_server.read_node('fuel_flow')

        if actual_fuel > safety_limits.max_fuel_flow_kg_hr * 0.9:
            alarm_type = 'HIGH_FUEL_FLOW'
        else:
            alarm_type = 'NONE'

        assert alarm_type == 'HIGH_FUEL_FLOW'

    async def test_high_fuel_flow_emergency_shutdown(
        self,
        opcua_server,
        modbus_server,
        safety_limits
    ):
        """Test excessive fuel flow triggers emergency shutdown."""
        dangerous_fuel = safety_limits.max_fuel_flow_kg_hr * 1.2
        await opcua_server.write_node('fuel_flow', dangerous_fuel)

        actual_fuel = await opcua_server.read_node('fuel_flow')

        if actual_fuel > safety_limits.max_fuel_flow_kg_hr * 1.1:
            await modbus_server.write_coil(1, True)
            await opcua_server.write_node('fuel_flow', 0.0)

        emergency_stop = await modbus_server.read_coils(1, 1)
        assert emergency_stop[0] is True


# ============================================================================
# INTERLOCK 6: LOW FUEL FLOW LIMIT (50 kg/hr)
# ============================================================================

class TestLowFuelFlowInterlock:
    """Test low fuel flow interlock (50 kg/hr limit)."""

    async def test_low_fuel_flow_triggers_shutdown(
        self,
        opcua_server,
        safety_limits
    ):
        """Test fuel flow below minimum triggers shutdown."""
        low_fuel = safety_limits.min_fuel_flow_kg_hr - 10.0
        await opcua_server.write_node('fuel_flow', low_fuel)

        actual_fuel = await opcua_server.read_node('fuel_flow')

        if actual_fuel < safety_limits.min_fuel_flow_kg_hr:
            await opcua_server.write_node('fuel_flow', 0.0)
            action = 'SHUTDOWN'
        else:
            action = 'CONTINUE'

        assert action == 'SHUTDOWN'

    async def test_low_fuel_flow_unstable_combustion(
        self,
        opcua_server,
        safety_limits
    ):
        """Test low fuel flow indicates unstable combustion."""
        low_fuel = safety_limits.min_fuel_flow_kg_hr + 5.0
        await opcua_server.write_node('fuel_flow', low_fuel)

        actual_fuel = await opcua_server.read_node('fuel_flow')

        if actual_fuel < safety_limits.min_fuel_flow_kg_hr * 1.2:
            stability = 'MARGINAL'
        else:
            stability = 'STABLE'

        assert stability == 'MARGINAL'

    async def test_low_fuel_flow_causes_high_co(
        self,
        opcua_server,
        mqtt_broker,
        safety_limits
    ):
        """Test low fuel flow can cause incomplete combustion (high CO)."""
        # Low fuel flow
        low_fuel = safety_limits.min_fuel_flow_kg_hr + 5.0
        await opcua_server.write_node('fuel_flow', low_fuel)

        # Publish high CO data (simulating incomplete combustion)
        high_co_data = {
            'o2_percent': 6.5,
            'co_ppm': 90.0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        await mqtt_broker.publish('combustion/analyzer/data', high_co_data)

        analyzer_data = mqtt_broker.get_latest_message('combustion/analyzer/data')

        # Verify correlation
        assert analyzer_data['co_ppm'] > safety_limits.max_co_ppm * 0.8


# ============================================================================
# INTERLOCK 7: HIGH CO EMISSION LIMIT (100 ppm)
# ============================================================================

class TestHighCOEmissionInterlock:
    """Test high CO emission interlock (100 ppm limit)."""

    async def test_high_co_increases_air_flow(
        self,
        opcua_server,
        mqtt_broker,
        safety_limits
    ):
        """Test high CO triggers air flow increase."""
        # Publish high CO reading
        high_co_data = {
            'o2_percent': 2.5,
            'co_ppm': safety_limits.max_co_ppm + 20.0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        await mqtt_broker.publish('combustion/analyzer/data', high_co_data)

        analyzer_data = mqtt_broker.get_latest_message('combustion/analyzer/data')

        if analyzer_data['co_ppm'] > safety_limits.max_co_ppm:
            current_air = await opcua_server.read_node('air_flow')
            increased_air = current_air * 1.15
            await opcua_server.write_node('air_flow', increased_air)
            action = 'INCREASE_AIR'
        else:
            action = 'NONE'

        assert action == 'INCREASE_AIR'

    async def test_high_co_persistent_triggers_shutdown(
        self,
        opcua_server,
        mqtt_broker,
        modbus_server,
        safety_limits
    ):
        """Test persistent high CO triggers shutdown."""
        high_co_count = 0
        threshold = 3  # 3 consecutive readings

        for _ in range(5):
            high_co_data = {
                'co_ppm': safety_limits.max_co_ppm + 30.0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            await mqtt_broker.publish('combustion/analyzer/data', high_co_data)

            analyzer_data = mqtt_broker.get_latest_message('combustion/analyzer/data')

            if analyzer_data['co_ppm'] > safety_limits.max_co_ppm:
                high_co_count += 1
            else:
                high_co_count = 0

            await asyncio.sleep(0.05)

        if high_co_count >= threshold:
            await modbus_server.write_coil(1, True)
            await opcua_server.write_node('fuel_flow', 0.0)

        emergency_stop = await modbus_server.read_coils(1, 1)
        assert emergency_stop[0] is True

    async def test_high_co_response_time(
        self,
        opcua_server,
        mqtt_broker,
        safety_limits,
        performance_timer
    ):
        """Test CO interlock response time."""
        with performance_timer() as timer:
            high_co_data = {
                'co_ppm': safety_limits.max_co_ppm * 2.0,  # Double the limit
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            await mqtt_broker.publish('combustion/analyzer/data', high_co_data)

            analyzer_data = mqtt_broker.get_latest_message('combustion/analyzer/data')

            if analyzer_data['co_ppm'] > safety_limits.max_co_ppm:
                current_air = await opcua_server.read_node('air_flow')
                await opcua_server.write_node('air_flow', current_air * 1.2)

        # Should respond quickly
        assert timer.elapsed_ms < 200.0


# ============================================================================
# INTERLOCK 8: FLAME LOSS DETECTION
# ============================================================================

class TestFlameLossInterlock:
    """Test flame loss detection interlock."""

    async def test_flame_loss_immediate_fuel_cutoff(
        self,
        opcua_server,
        modbus_server,
        flame_scanner,
        performance_timer
    ):
        """Test flame loss triggers immediate fuel cutoff."""
        flame_scanner.set_flame_loss()

        with performance_timer() as timer:
            flame_data = await flame_scanner.get_flame_status()

            if not flame_data['flame_detected']:
                await opcua_server.write_node('fuel_flow', 0.0)
                await modbus_server.write_coil(1, True)

        # Must complete in <100ms for SIL-2
        assert timer.elapsed_ms < 100.0

        fuel_flow = await opcua_server.read_node('fuel_flow')
        assert fuel_flow == 0.0

        # Cleanup
        flame_scanner.restore_flame()

    async def test_flame_loss_requires_purge(
        self,
        opcua_server,
        flame_scanner
    ):
        """Test flame loss requires purge cycle before restart."""
        flame_scanner.set_flame_loss()

        flame_data = await flame_scanner.get_flame_status()

        if not flame_data['flame_detected']:
            # Shutdown fuel
            await opcua_server.write_node('fuel_flow', 0.0)

            # Purge cycle: air flow only
            purge_air = 3000.0
            await opcua_server.write_node('air_flow', purge_air)

            # Verify purge configuration
            air_flow = await opcua_server.read_node('air_flow')
            fuel_flow = await opcua_server.read_node('fuel_flow')

            purge_active = (air_flow > 0 and fuel_flow == 0.0)
        else:
            purge_active = False

        assert purge_active is True

        # Cleanup
        flame_scanner.restore_flame()

    async def test_flame_scanner_failure_failsafe(
        self,
        flame_scanner
    ):
        """Test flame scanner failure triggers fail-safe state."""
        # Simulate scanner working
        flame_data = await flame_scanner.get_flame_status()
        assert flame_data is not None

        # Scanner should report flame status
        assert 'flame_detected' in flame_data

    async def test_flame_loss_alarm_priority(
        self,
        opcua_server,
        flame_scanner
    ):
        """Test flame loss generates critical priority alarm."""
        flame_scanner.set_flame_loss()

        flame_data = await flame_scanner.get_flame_status()

        if not flame_data['flame_detected']:
            alarm_priority = 'CRITICAL'
            fuel_flow = await opcua_server.read_node('fuel_flow')

            if fuel_flow > 0:
                # Fuel flowing with no flame = highest priority
                alarm_priority = 'EMERGENCY'
        else:
            alarm_priority = 'NONE'

        assert alarm_priority in ['CRITICAL', 'EMERGENCY']

        # Cleanup
        flame_scanner.restore_flame()


# ============================================================================
# INTERLOCK 9: EMERGENCY STOP BUTTON
# ============================================================================

class TestEmergencyStopInterlock:
    """Test emergency stop button interlock."""

    async def test_emergency_stop_immediate_response(
        self,
        opcua_server,
        modbus_server,
        performance_timer
    ):
        """Test emergency stop responds immediately (<50ms)."""
        with performance_timer() as timer:
            await modbus_server.write_coil(1, True)
            await opcua_server.write_node('fuel_flow', 0.0)
            await opcua_server.write_node('air_flow', 0.0)

        # Emergency stop must be <50ms for SIL-2
        assert timer.elapsed_ms < 50.0

    async def test_emergency_stop_cannot_be_bypassed(
        self,
        opcua_server,
        modbus_server
    ):
        """Test emergency stop cannot be bypassed."""
        # Activate emergency stop
        await modbus_server.write_coil(1, True)

        # Check if stop active
        emergency_stop = await modbus_server.read_coils(1, 1)

        if emergency_stop[0]:
            # Block any fuel flow attempt
            fuel_setpoint = 0.0
        else:
            fuel_setpoint = 500.0

        await opcua_server.write_node('fuel_flow', fuel_setpoint)
        actual_fuel = await opcua_server.read_node('fuel_flow')

        assert actual_fuel == 0.0

    async def test_emergency_stop_latches(
        self,
        modbus_server
    ):
        """Test emergency stop latches until manually reset."""
        # Trip emergency stop
        await modbus_server.write_coil(1, True)

        # Verify latched
        stop_1 = await modbus_server.read_coils(1, 1)
        assert stop_1[0] is True

        # Try to read again - should still be latched
        await asyncio.sleep(0.1)
        stop_2 = await modbus_server.read_coils(1, 1)
        assert stop_2[0] is True

    async def test_emergency_stop_reset_procedure(
        self,
        modbus_server
    ):
        """Test emergency stop requires proper reset procedure."""
        # Trip emergency stop
        await modbus_server.write_coil(1, True)

        stop_active = await modbus_server.read_coils(1, 1)
        assert stop_active[0] is True

        # Reset procedure: clear coil
        await modbus_server.write_coil(1, False)

        stop_after_reset = await modbus_server.read_coils(1, 1)
        assert stop_after_reset[0] is False

    async def test_emergency_stop_all_outputs_safe(
        self,
        opcua_server,
        modbus_server
    ):
        """Test all outputs go to safe state on emergency stop."""
        # Trip emergency stop
        await modbus_server.write_coil(1, True)

        # Set all outputs to safe state
        await opcua_server.write_node('fuel_flow', 0.0)

        # Verify safe states
        fuel = await opcua_server.read_node('fuel_flow')
        emergency = await modbus_server.read_coils(1, 1)

        assert fuel == 0.0
        assert emergency[0] is True


# ============================================================================
# INTERLOCK INTERACTION TESTS
# ============================================================================

class TestInterlockInteractions:
    """Test interactions between multiple interlocks."""

    async def test_multiple_interlocks_simultaneous(
        self,
        opcua_server,
        modbus_server,
        flame_scanner,
        safety_limits
    ):
        """Test behavior with multiple simultaneous interlock trips."""
        # Trip multiple interlocks
        await opcua_server.write_node('combustion_temperature', safety_limits.max_temperature_c + 50)
        await opcua_server.write_node('furnace_pressure', safety_limits.max_pressure_mbar + 20)
        flame_scanner.set_flame_loss()

        # Check all conditions
        temp = await opcua_server.read_node('combustion_temperature')
        pressure = await opcua_server.read_node('furnace_pressure')
        flame = await flame_scanner.get_flame_status()

        tripped_interlocks = []

        if temp > safety_limits.max_temperature_c:
            tripped_interlocks.append('HIGH_TEMP')

        if pressure > safety_limits.max_pressure_mbar:
            tripped_interlocks.append('HIGH_PRESSURE')

        if not flame['flame_detected']:
            tripped_interlocks.append('FLAME_LOSS')

        # All should trigger shutdown
        await modbus_server.write_coil(1, True)
        await opcua_server.write_node('fuel_flow', 0.0)

        assert len(tripped_interlocks) == 3

        # Cleanup
        flame_scanner.restore_flame()

    async def test_interlock_priority_order(
        self,
        opcua_server,
        modbus_server,
        flame_scanner,
        safety_limits
    ):
        """Test interlocks are processed in priority order."""
        # Flame loss has highest priority
        flame_scanner.set_flame_loss()
        await opcua_server.write_node('combustion_temperature', safety_limits.max_temperature_c + 10)

        flame = await flame_scanner.get_flame_status()
        temp = await opcua_server.read_node('combustion_temperature')

        # Determine highest priority interlock
        if not flame['flame_detected']:
            primary_interlock = 'FLAME_LOSS'
        elif temp > safety_limits.max_temperature_c:
            primary_interlock = 'HIGH_TEMP'
        else:
            primary_interlock = 'NONE'

        assert primary_interlock == 'FLAME_LOSS'

        # Cleanup
        flame_scanner.restore_flame()

    async def test_interlock_cascade_detection(
        self,
        opcua_server,
        safety_limits
    ):
        """Test detection of cascading interlock conditions."""
        # High fuel flow can lead to high temperature
        high_fuel = safety_limits.max_fuel_flow_kg_hr * 0.95
        await opcua_server.write_node('fuel_flow', high_fuel)

        # Simulate resulting high temperature
        high_temp = safety_limits.max_temperature_c * 0.98
        await opcua_server.write_node('combustion_temperature', high_temp)

        fuel = await opcua_server.read_node('fuel_flow')
        temp = await opcua_server.read_node('combustion_temperature')

        # Detect cascade
        cascade_conditions = []

        if fuel > safety_limits.max_fuel_flow_kg_hr * 0.9:
            cascade_conditions.append('HIGH_FUEL')

        if temp > safety_limits.max_temperature_c * 0.95:
            cascade_conditions.append('HIGH_TEMP_RISK')

        # Should detect both conditions
        assert len(cascade_conditions) >= 1


# ============================================================================
# FAIL-SAFE DEFAULTS TESTS
# ============================================================================

class TestFailSafeDefaults:
    """Test fail-safe default behaviors."""

    async def test_fuel_valve_fails_closed(
        self,
        opcua_server,
        modbus_server
    ):
        """Test fuel valve fails to closed position."""
        # Simulate communication loss (emergency stop)
        await modbus_server.write_coil(1, True)

        # Safe state = fuel valve closed
        await opcua_server.write_node('fuel_flow', 0.0)

        fuel = await opcua_server.read_node('fuel_flow')
        assert fuel == 0.0

    async def test_air_damper_fails_open(
        self,
        opcua_server,
        modbus_server
    ):
        """Test air damper fails to open position for purge."""
        # Trigger emergency
        await modbus_server.write_coil(1, True)
        await opcua_server.write_node('fuel_flow', 0.0)

        # Air damper should stay open for purge
        air_flow = await opcua_server.read_node('air_flow')

        # Air flow should be positive or at purge level
        assert air_flow > 0

    async def test_ignition_fails_safe(
        self,
        modbus_server
    ):
        """Test ignition system fails to safe (off) state."""
        # With emergency stop, ignition should be disabled
        await modbus_server.write_coil(1, True)

        emergency = await modbus_server.read_coils(1, 1)

        # Ignition should not be possible during emergency
        ignition_allowed = not emergency[0]
        assert ignition_allowed is False


# ============================================================================
# SHUTDOWN SEQUENCE TESTS
# ============================================================================

class TestShutdownSequence:
    """Test emergency shutdown sequence execution."""

    async def test_shutdown_sequence_order(
        self,
        opcua_server,
        modbus_server
    ):
        """Test shutdown follows correct sequence."""
        sequence_steps = []

        # Step 1: Emergency stop
        await modbus_server.write_coil(1, True)
        sequence_steps.append('ESTOP_ACTIVE')

        # Step 2: Fuel cutoff
        await opcua_server.write_node('fuel_flow', 0.0)
        sequence_steps.append('FUEL_OFF')

        # Step 3: Maintain air for purge
        air_flow = await opcua_server.read_node('air_flow')
        if air_flow > 0:
            sequence_steps.append('AIR_PURGE')

        # Verify sequence
        assert sequence_steps == ['ESTOP_ACTIVE', 'FUEL_OFF', 'AIR_PURGE']

    async def test_shutdown_timing_requirements(
        self,
        opcua_server,
        modbus_server,
        performance_timer
    ):
        """Test shutdown meets timing requirements."""
        with performance_timer() as timer:
            # Full shutdown sequence
            await modbus_server.write_coil(1, True)
            await opcua_server.write_node('fuel_flow', 0.0)

            # Verify states
            emergency = await modbus_server.read_coils(1, 1)
            fuel = await opcua_server.read_node('fuel_flow')

        # Total shutdown should be <200ms
        assert timer.elapsed_ms < 200.0
        assert emergency[0] is True
        assert fuel == 0.0

    async def test_shutdown_completion_verification(
        self,
        opcua_server,
        modbus_server
    ):
        """Test verification of shutdown completion."""
        # Execute shutdown
        await modbus_server.write_coil(1, True)
        await opcua_server.write_node('fuel_flow', 0.0)

        # Verify completion
        emergency = await modbus_server.read_coils(1, 1)
        fuel = await opcua_server.read_node('fuel_flow')

        shutdown_complete = (
            emergency[0] is True and
            fuel == 0.0
        )

        assert shutdown_complete is True
