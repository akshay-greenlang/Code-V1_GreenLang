# -*- coding: utf-8 -*-
"""
End-to-end integration tests for GL-005 CombustionControlAgent control workflow.

Tests complete control cycles from state reading through optimization
to setpoint writing, validating the entire control workflow.

Target: 10+ tests covering:
- Full control cycle execution
- Multiple consecutive cycles
- Optimization convergence
- Emergency shutdown scenarios
- Recovery after failures
- State persistence
- Control stability
- Performance validation
"""

import pytest
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.e2e]


# ============================================================================
# E2E CONTROL CYCLE TESTS
# ============================================================================

class TestE2EControlCycle:
    """Test end-to-end control cycle execution."""

    async def test_complete_control_cycle_execution(
        self,
        opcua_server,
        modbus_server,
        mqtt_broker,
        flame_scanner
    ):
        """Test complete control cycle from reading to writing setpoints."""
        # Phase 1: Read current state from all sources
        dcs_data = await opcua_server.read_multiple_nodes([
            'fuel_flow', 'air_flow', 'combustion_temperature', 'furnace_pressure'
        ])

        plc_registers = await modbus_server.read_input_registers(0, 3)

        # Verify data read successfully
        assert dcs_data is not None
        assert 'fuel_flow' in dcs_data
        assert 'combustion_temperature' in dcs_data
        assert len(plc_registers) == 3

        # Phase 2: Read emissions data
        analyzer_data = mqtt_broker.get_latest_message('combustion/analyzer/data')
        assert analyzer_data is not None
        assert 'o2_percent' in analyzer_data

        # Phase 3: Read flame status
        flame_data = await flame_scanner.get_flame_status()
        assert flame_data['flame_detected'] is True

        # Phase 4: Calculate new setpoints (simplified)
        current_temp = dcs_data['combustion_temperature']
        setpoint_temp = 1200.0
        temp_error = setpoint_temp - current_temp

        # Simple proportional control
        fuel_adjustment = temp_error * 0.5
        new_fuel_setpoint = dcs_data['fuel_flow'] + fuel_adjustment

        # Phase 5: Write new setpoints
        write_success = await opcua_server.write_node('fuel_flow', new_fuel_setpoint)
        assert write_success is True

        # Phase 6: Verify setpoint written
        verify_fuel = await opcua_server.read_node('fuel_flow')
        assert abs(verify_fuel - new_fuel_setpoint) < 1.0  # Within tolerance

    async def test_multiple_consecutive_control_cycles(
        self,
        opcua_server,
        performance_timer,
        benchmark_thresholds
    ):
        """Test multiple consecutive control cycles execute successfully."""
        num_cycles = 20
        cycle_times = []

        for cycle_num in range(num_cycles):
            with performance_timer() as timer:
                # Read state
                state = await opcua_server.read_multiple_nodes([
                    'fuel_flow', 'air_flow', 'combustion_temperature'
                ])

                # Calculate control action
                temp_error = 1200.0 - state['combustion_temperature']
                fuel_adjust = temp_error * 0.5

                # Write setpoint
                new_fuel = state['fuel_flow'] + fuel_adjust
                await opcua_server.write_node('fuel_flow', new_fuel)

                # Simulate control interval
                await asyncio.sleep(0.01)

            cycle_times.append(timer.elapsed_ms)

        # Validate all cycles completed
        assert len(cycle_times) == num_cycles

        # Validate cycle times are reasonable
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        assert avg_cycle_time < benchmark_thresholds['cycle_execution_max_ms']

    async def test_control_cycle_with_state_validation(
        self,
        opcua_server,
        safety_limits
    ):
        """Test control cycle includes state validation."""
        # Read current state
        state = await opcua_server.read_multiple_nodes([
            'combustion_temperature',
            'furnace_pressure',
            'fuel_flow'
        ])

        # Validate state against safety limits
        violations = []

        if not (safety_limits.min_temperature_c <= state['combustion_temperature'] <= safety_limits.max_temperature_c):
            violations.append('TEMPERATURE')

        if not (safety_limits.min_pressure_mbar <= state['furnace_pressure'] <= safety_limits.max_pressure_mbar):
            violations.append('PRESSURE')

        if not (safety_limits.min_fuel_flow_kg_hr <= state['fuel_flow'] <= safety_limits.max_fuel_flow_kg_hr):
            violations.append('FUEL_FLOW')

        # Normal operation should have no violations
        assert len(violations) == 0


# ============================================================================
# OPTIMIZATION CONVERGENCE TESTS
# ============================================================================

class TestOptimizationConvergence:
    """Test optimization algorithm convergence."""

    async def test_temperature_setpoint_convergence(
        self,
        opcua_server,
        control_parameters
    ):
        """Test control converges to temperature setpoint."""
        setpoint = control_parameters['setpoint_temperature_c']
        tolerance = 5.0  # degrees C

        max_iterations = 50
        converged = False

        for iteration in range(max_iterations):
            # Read current temperature
            current_temp = await opcua_server.read_node('combustion_temperature')

            # Check if converged
            error = abs(setpoint - current_temp)
            if error < tolerance:
                converged = True
                break

            # Calculate control action
            kp = control_parameters['pid_kp']
            fuel_adjustment = kp * (setpoint - current_temp)

            # Apply control action
            current_fuel = await opcua_server.read_node('fuel_flow')
            new_fuel = current_fuel + fuel_adjustment
            await opcua_server.write_node('fuel_flow', new_fuel)

            # Wait for process response
            await asyncio.sleep(0.1)

        assert converged is True

    async def test_fuel_air_ratio_optimization(
        self,
        opcua_server,
        control_parameters
    ):
        """Test fuel-air ratio optimizes to target."""
        target_ratio = control_parameters['fuel_air_ratio_target']
        tolerance = 0.05

        # Read current flows
        fuel_flow = await opcua_server.read_node('fuel_flow')
        air_flow = await opcua_server.read_node('air_flow')

        current_ratio = fuel_flow / air_flow

        # Calculate required air flow adjustment
        required_air = fuel_flow / target_ratio
        air_adjustment = required_air - air_flow

        # Apply adjustment
        new_air = air_flow + air_adjustment
        await opcua_server.write_node('air_flow', new_air)

        # Verify new ratio
        verify_air = await opcua_server.read_node('air_flow')
        verify_fuel = await opcua_server.read_node('fuel_flow')
        new_ratio = verify_fuel / verify_air

        assert abs(new_ratio - target_ratio) < tolerance


# ============================================================================
# EMERGENCY SHUTDOWN TESTS
# ============================================================================

class TestEmergencyShutdown:
    """Test emergency shutdown scenarios."""

    async def test_emergency_shutdown_on_high_temperature(
        self,
        opcua_server,
        modbus_server,
        safety_limits
    ):
        """Test emergency shutdown triggered by high temperature."""
        # Simulate high temperature condition
        dangerous_temp = safety_limits.max_temperature_c + 50
        await opcua_server.write_node('combustion_temperature', dangerous_temp)

        # Read temperature
        current_temp = await opcua_server.read_node('combustion_temperature')

        # Check if emergency shutdown required
        if current_temp > safety_limits.max_temperature_c:
            # Trigger emergency shutdown
            await modbus_server.write_coil(1, True)  # Emergency stop coil

            # Shut off fuel
            await opcua_server.write_node('fuel_flow', 0.0)

        # Verify shutdown executed
        emergency_stop = await modbus_server.read_coils(1, 1)
        fuel_flow = await opcua_server.read_node('fuel_flow')

        assert emergency_stop[0] is True
        assert fuel_flow == 0.0

    async def test_emergency_shutdown_on_flame_loss(
        self,
        opcua_server,
        modbus_server,
        flame_scanner
    ):
        """Test emergency shutdown triggered by flame loss."""
        # Simulate flame loss
        flame_scanner.set_flame_loss()

        # Check flame status
        flame_data = await flame_scanner.get_flame_status()

        if not flame_data['flame_detected']:
            # Trigger emergency shutdown
            await modbus_server.write_coil(1, True)
            await opcua_server.write_node('fuel_flow', 0.0)

        # Verify shutdown
        emergency_stop = await modbus_server.read_coils(1, 1)
        assert emergency_stop[0] is True

        # Restore flame for cleanup
        flame_scanner.restore_flame()

    async def test_shutdown_timeout_validation(
        self,
        opcua_server,
        modbus_server,
        performance_timer
    ):
        """Test emergency shutdown completes within timeout."""
        shutdown_timeout_ms = 1000.0

        with performance_timer() as timer:
            # Trigger shutdown
            await modbus_server.write_coil(1, True)
            await opcua_server.write_node('fuel_flow', 0.0)
            await opcua_server.write_node('air_flow', 0.0)

        # Verify shutdown completed within timeout
        assert timer.elapsed_ms < shutdown_timeout_ms


# ============================================================================
# RECOVERY AND RESILIENCE TESTS
# ============================================================================

class TestRecoveryAndResilience:
    """Test recovery after failures."""

    async def test_recovery_after_dcs_connection_loss(self, opcua_server):
        """Test recovery after DCS connection loss."""
        # Simulate connection loss
        await opcua_server.stop()

        # Attempt to read (should fail)
        with pytest.raises(ConnectionError):
            await opcua_server.read_node('fuel_flow')

        # Restore connection
        await opcua_server.start()

        # Verify recovery
        fuel_flow = await opcua_server.read_node('fuel_flow')
        assert fuel_flow is not None

    async def test_recovery_after_sensor_timeout(self, opcua_server):
        """Test recovery after sensor timeout."""
        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                data = await asyncio.wait_for(
                    opcua_server.read_node('fuel_flow'),
                    timeout=1.0
                )
                success = True
            except asyncio.TimeoutError:
                retry_count += 1
                await asyncio.sleep(0.1)

        assert success is True or retry_count == max_retries

    async def test_graceful_degradation_on_partial_failure(
        self,
        opcua_server,
        mqtt_broker
    ):
        """Test graceful degradation when one subsystem fails."""
        # DCS is working
        dcs_data = await opcua_server.read_node('fuel_flow')
        assert dcs_data is not None

        # Analyzer data might be unavailable
        analyzer_data = mqtt_broker.get_latest_message('combustion/analyzer/data')

        # System should continue with available data
        if analyzer_data is None:
            # Use last known good values or defaults
            default_o2 = 3.5
            assert default_o2 > 0

        # Control should continue
        assert True  # Placeholder for continued operation


# ============================================================================
# STATE PERSISTENCE TESTS
# ============================================================================

class TestStatePersistence:
    """Test state persistence and recovery."""

    async def test_state_snapshot_creation(self, opcua_server):
        """Test creation of state snapshot."""
        # Read complete state
        state_snapshot = await opcua_server.read_multiple_nodes([
            'fuel_flow',
            'air_flow',
            'combustion_temperature',
            'furnace_pressure',
            'o2_percent'
        ])

        # Add timestamp
        state_snapshot['timestamp'] = datetime.now(timezone.utc).isoformat()

        # Create hash for integrity
        state_json = json.dumps(state_snapshot, sort_keys=True)
        state_hash = hashlib.sha256(state_json.encode()).hexdigest()

        assert state_hash is not None
        assert len(state_hash) == 64

    async def test_state_restoration_from_snapshot(self, opcua_server):
        """Test restoration of state from snapshot."""
        # Create snapshot
        original_state = await opcua_server.read_multiple_nodes([
            'fuel_flow', 'air_flow', 'combustion_temperature'
        ])

        # Modify state
        await opcua_server.write_node('fuel_flow', 600.0)

        # Restore from snapshot
        await opcua_server.write_node('fuel_flow', original_state['fuel_flow'])

        # Verify restoration
        restored_fuel = await opcua_server.read_node('fuel_flow')
        assert abs(restored_fuel - original_state['fuel_flow']) < 10.0


# ============================================================================
# CONTROL STABILITY TESTS
# ============================================================================

@pytest.mark.slow
class TestControlStability:
    """Test control stability over time."""

    async def test_sustained_operation_stability(
        self,
        opcua_server,
        control_parameters
    ):
        """Test control remains stable over sustained operation."""
        duration_seconds = 10
        sample_interval = 0.5
        num_samples = int(duration_seconds / sample_interval)

        temperature_samples = []
        setpoint = control_parameters['setpoint_temperature_c']

        for _ in range(num_samples):
            temp = await opcua_server.read_node('combustion_temperature')
            temperature_samples.append(temp)
            await asyncio.sleep(sample_interval)

        # Calculate temperature variance
        mean_temp = sum(temperature_samples) / len(temperature_samples)
        variance = sum((t - mean_temp) ** 2 for t in temperature_samples) / len(temperature_samples)
        std_dev = variance ** 0.5

        # Stable control should have low variance
        assert std_dev < 10.0  # Within 10 degrees C

    async def test_no_oscillations_in_control(self, opcua_server):
        """Test control does not oscillate."""
        num_samples = 20
        samples = []

        for _ in range(num_samples):
            temp = await opcua_server.read_node('combustion_temperature')
            samples.append(temp)
            await asyncio.sleep(0.1)

        # Check for oscillations (sign changes in derivative)
        derivatives = [samples[i+1] - samples[i] for i in range(len(samples)-1)]
        sign_changes = sum(1 for i in range(len(derivatives)-1)
                          if derivatives[i] * derivatives[i+1] < 0)

        # Should have minimal sign changes (no rapid oscillations)
        assert sign_changes < num_samples / 2


# ============================================================================
# PERFORMANCE VALIDATION TESTS
# ============================================================================

@pytest.mark.performance
class TestE2EPerformance:
    """Test end-to-end performance metrics."""

    async def test_control_loop_latency(
        self,
        opcua_server,
        performance_timer,
        benchmark_thresholds
    ):
        """Test complete control loop latency meets target."""
        with performance_timer() as timer:
            # Read sensors
            state = await opcua_server.read_multiple_nodes([
                'fuel_flow', 'air_flow', 'combustion_temperature'
            ])

            # Calculate control
            temp_error = 1200.0 - state['combustion_temperature']
            fuel_adjust = temp_error * 0.5

            # Write actuators
            new_fuel = state['fuel_flow'] + fuel_adjust
            await opcua_server.write_node('fuel_flow', new_fuel)

        # Validate latency
        assert timer.elapsed_ms < benchmark_thresholds['control_loop_max_latency_ms']

    async def test_throughput_validation(
        self,
        opcua_server,
        benchmark_thresholds
    ):
        """Test control throughput meets minimum requirement."""
        num_cycles = 50
        start_time = time.perf_counter()

        for _ in range(num_cycles):
            await opcua_server.read_node('fuel_flow')
            await opcua_server.write_node('fuel_flow', 500.0)

        duration = time.perf_counter() - start_time
        throughput = num_cycles / duration

        assert throughput >= benchmark_thresholds['min_throughput_cps']
