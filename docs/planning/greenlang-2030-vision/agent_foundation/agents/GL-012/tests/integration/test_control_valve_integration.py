# -*- coding: utf-8 -*-
"""
Control Valve Integration Tests for GL-012 SteamQualityController

Comprehensive integration tests for steam control valve operations including:
- Valve position control (0-100%)
- Valve status reading and diagnostics
- Emergency close functionality
- Safety interlock behavior
- Actuator response testing
- Mock actuator response simulation

Test Count: 30+ tests
Coverage Target: 90%+

Standards: IEC 61131-3 (Safety), ISA-75.01 (Control Valve Sizing)

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from decimal import Decimal

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.control_valve]


# =============================================================================
# VALVE POSITION CONTROL TESTS
# =============================================================================

class TestValvePositionControl:
    """Test valve position control operations."""

    async def test_set_valve_position_valid_range(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test setting valve position within valid range (0-100%)."""
        test_positions = [0.0, 25.0, 50.0, 75.0, 100.0]

        for target_position in test_positions:
            result = await steam_quality_controller.set_valve_position(
                valve_id='CV-STEAM-001',
                position_percent=target_position
            )

            assert result['status'] == 'success'
            assert result['commanded_position'] == target_position
            assert abs(result['actual_position'] - target_position) < 1.0  # 1% tolerance

    async def test_set_valve_position_out_of_range_high(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test rejection of position above 100%."""
        result = await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=105.0
        )

        assert result['status'] == 'error'
        assert 'out_of_range' in result['error_code'].lower()

    async def test_set_valve_position_out_of_range_low(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test rejection of negative position."""
        result = await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=-5.0
        )

        assert result['status'] == 'error'
        assert 'out_of_range' in result['error_code'].lower()

    async def test_valve_position_ramp_rate_limit(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test valve position change respects ramp rate limit."""
        # Configure ramp rate limit
        steam_quality_controller.configure_valve(
            valve_id='CV-STEAM-001',
            ramp_rate_percent_per_second=10.0  # 10% per second max
        )

        # Set initial position
        await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=0.0
        )

        # Request large position change
        start_time = asyncio.get_event_loop().time()
        result = await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=100.0,
            wait_for_position=True
        )
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should take approximately 10 seconds for 100% travel at 10%/sec
        assert elapsed >= 9.0, f"Position change too fast: {elapsed}s"

    async def test_valve_position_precision(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test valve positioning precision (0.1% resolution)."""
        test_positions = [12.3, 45.7, 78.9, 33.33]

        for target_position in test_positions:
            result = await steam_quality_controller.set_valve_position(
                valve_id='CV-STEAM-001',
                position_percent=target_position
            )

            assert abs(result['actual_position'] - target_position) < 0.5

    async def test_valve_position_feedback_loop(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test valve position feedback correction loop."""
        # Simulate position feedback mismatch
        mock_control_valve.set_position_offset(2.0)  # 2% offset

        result = await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=50.0,
            enable_feedback_correction=True
        )

        # Feedback correction should compensate for offset
        assert result['feedback_correction_applied'] is True
        assert abs(result['actual_position'] - 50.0) < 1.0


# =============================================================================
# VALVE STATUS READING TESTS
# =============================================================================

class TestValveStatusReading:
    """Test valve status reading operations."""

    async def test_read_valve_position(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading current valve position."""
        # Set known position first
        await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=45.0
        )

        status = await steam_quality_controller.read_valve_status('CV-STEAM-001')

        assert 'position_percent' in status
        assert abs(status['position_percent'] - 45.0) < 1.0

    async def test_read_valve_operating_mode(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading valve operating mode."""
        status = await steam_quality_controller.read_valve_status('CV-STEAM-001')

        assert 'operating_mode' in status
        assert status['operating_mode'] in ['AUTO', 'MANUAL', 'LOCAL', 'REMOTE']

    async def test_read_valve_actuator_state(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading actuator state."""
        status = await steam_quality_controller.read_valve_status('CV-STEAM-001')

        assert 'actuator_state' in status
        assert status['actuator_state'] in [
            'IDLE', 'OPENING', 'CLOSING', 'FAULT', 'CALIBRATING'
        ]

    async def test_read_valve_limit_switches(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading valve limit switch status."""
        # Move to fully closed
        await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=0.0,
            wait_for_position=True
        )

        status = await steam_quality_controller.read_valve_status('CV-STEAM-001')

        assert 'limit_switches' in status
        assert status['limit_switches']['closed'] is True
        assert status['limit_switches']['open'] is False

    async def test_read_valve_torque(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading actuator torque."""
        status = await steam_quality_controller.read_valve_status('CV-STEAM-001')

        assert 'actuator_torque_percent' in status
        assert 0 <= status['actuator_torque_percent'] <= 100

    async def test_read_valve_travel_time(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading valve travel time metrics."""
        status = await steam_quality_controller.read_valve_status('CV-STEAM-001')

        assert 'stroke_time_seconds' in status
        assert status['stroke_time_seconds'] > 0

    async def test_read_valve_cycle_count(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading valve cycle count for maintenance tracking."""
        status = await steam_quality_controller.read_valve_status('CV-STEAM-001')

        assert 'cycle_count' in status
        assert status['cycle_count'] >= 0


# =============================================================================
# EMERGENCY CLOSE FUNCTIONALITY TESTS
# =============================================================================

class TestEmergencyClose:
    """Test emergency close functionality."""

    async def test_emergency_close_execution(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test emergency close command execution."""
        # Set valve to open position first
        await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=75.0,
            wait_for_position=True
        )

        # Execute emergency close
        result = await steam_quality_controller.emergency_close(
            valve_id='CV-STEAM-001',
            reason='HIGH_PRESSURE_TRIP'
        )

        assert result['status'] == 'success'
        assert result['emergency_close_executed'] is True
        assert result['final_position'] == 0.0

    async def test_emergency_close_response_time(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test emergency close completes within time limit (<2 seconds)."""
        await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=100.0,
            wait_for_position=True
        )

        start_time = asyncio.get_event_loop().time()
        result = await steam_quality_controller.emergency_close(
            valve_id='CV-STEAM-001',
            reason='SAFETY_TRIP'
        )
        elapsed = asyncio.get_event_loop().time() - start_time

        assert elapsed < 2.0, f"Emergency close took {elapsed}s, exceeds 2s limit"
        assert result['response_time_ms'] < 2000

    async def test_emergency_close_overrides_normal_operation(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test emergency close overrides ongoing position command."""
        # Start slow position change
        position_task = asyncio.create_task(
            steam_quality_controller.set_valve_position(
                valve_id='CV-STEAM-001',
                position_percent=100.0,
                wait_for_position=True
            )
        )

        await asyncio.sleep(0.5)  # Let it start moving

        # Emergency close should interrupt
        result = await steam_quality_controller.emergency_close(
            valve_id='CV-STEAM-001',
            reason='EMERGENCY_STOP'
        )

        assert result['status'] == 'success'

        # Cancel the ongoing task
        position_task.cancel()
        try:
            await position_task
        except asyncio.CancelledError:
            pass

    async def test_emergency_close_lockout(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test valve lockout after emergency close."""
        await steam_quality_controller.emergency_close(
            valve_id='CV-STEAM-001',
            reason='SAFETY_TRIP'
        )

        # Attempt normal operation - should be blocked
        result = await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=50.0
        )

        assert result['status'] == 'blocked'
        assert 'lockout' in result['reason'].lower()

    async def test_emergency_close_reset_required(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reset required after emergency close."""
        await steam_quality_controller.emergency_close(
            valve_id='CV-STEAM-001',
            reason='SAFETY_TRIP'
        )

        status = await steam_quality_controller.read_valve_status('CV-STEAM-001')

        assert status['reset_required'] is True
        assert status['locked_out'] is True

    async def test_emergency_close_acknowledgment(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test emergency close acknowledgment and reset."""
        await steam_quality_controller.emergency_close(
            valve_id='CV-STEAM-001',
            reason='SAFETY_TRIP'
        )

        # Acknowledge and reset
        reset_result = await steam_quality_controller.acknowledge_emergency(
            valve_id='CV-STEAM-001',
            operator_id='OP-001',
            acknowledgment_code='ACK-12345'
        )

        assert reset_result['status'] == 'success'
        assert reset_result['lockout_cleared'] is True

        # Normal operation should now be possible
        position_result = await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=50.0
        )

        assert position_result['status'] == 'success'


# =============================================================================
# DIAGNOSTIC READING TESTS
# =============================================================================

class TestValveDiagnostics:
    """Test valve diagnostic reading operations."""

    async def test_read_valve_health_status(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading overall valve health status."""
        diagnostics = await steam_quality_controller.read_valve_diagnostics(
            'CV-STEAM-001'
        )

        assert 'health_status' in diagnostics
        assert diagnostics['health_status'] in ['GOOD', 'WARNING', 'FAULT']

    async def test_read_actuator_temperature(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading actuator motor temperature."""
        diagnostics = await steam_quality_controller.read_valve_diagnostics(
            'CV-STEAM-001'
        )

        assert 'actuator_temperature_c' in diagnostics
        assert diagnostics['actuator_temperature_c'] > 0
        assert diagnostics['actuator_temperature_c'] < 150  # Reasonable limit

    async def test_read_valve_friction_signature(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading valve friction signature for predictive maintenance."""
        diagnostics = await steam_quality_controller.read_valve_diagnostics(
            'CV-STEAM-001'
        )

        assert 'friction_signature' in diagnostics
        assert 'static_friction' in diagnostics['friction_signature']
        assert 'dynamic_friction' in diagnostics['friction_signature']

    async def test_read_deadband_analysis(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading valve deadband analysis."""
        diagnostics = await steam_quality_controller.read_valve_diagnostics(
            'CV-STEAM-001'
        )

        assert 'deadband_percent' in diagnostics
        assert diagnostics['deadband_percent'] >= 0
        assert diagnostics['deadband_percent'] < 10  # Should be less than 10%

    async def test_read_positioner_status(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading smart positioner status."""
        diagnostics = await steam_quality_controller.read_valve_diagnostics(
            'CV-STEAM-001'
        )

        assert 'positioner' in diagnostics
        assert 'calibration_status' in diagnostics['positioner']
        assert 'supply_pressure_bar' in diagnostics['positioner']

    async def test_read_valve_alerts(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading active valve alerts."""
        diagnostics = await steam_quality_controller.read_valve_diagnostics(
            'CV-STEAM-001'
        )

        assert 'active_alerts' in diagnostics
        assert isinstance(diagnostics['active_alerts'], list)

    async def test_read_maintenance_prediction(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test reading predictive maintenance data."""
        diagnostics = await steam_quality_controller.read_valve_diagnostics(
            'CV-STEAM-001'
        )

        assert 'maintenance_prediction' in diagnostics
        assert 'remaining_life_percent' in diagnostics['maintenance_prediction']
        assert 'next_service_date' in diagnostics['maintenance_prediction']


# =============================================================================
# SAFETY INTERLOCK BEHAVIOR TESTS
# =============================================================================

class TestSafetyInterlockBehavior:
    """Test valve safety interlock behavior."""

    async def test_high_pressure_interlock(
        self,
        mock_control_valve,
        steam_quality_controller,
        mock_pressure_sensor
    ):
        """Test valve closes on high pressure interlock."""
        # Set valve to open
        await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=80.0,
            wait_for_position=True
        )

        # Simulate high pressure condition
        mock_pressure_sensor.set_value(150.0)  # Above limit

        # Trigger interlock check
        result = await steam_quality_controller.check_safety_interlocks()

        assert result['interlock_triggered'] is True
        assert 'HIGH_PRESSURE' in result['active_interlocks']

        # Valve should be closed
        status = await steam_quality_controller.read_valve_status('CV-STEAM-001')
        assert status['position_percent'] == 0.0

    async def test_low_pressure_interlock(
        self,
        mock_control_valve,
        steam_quality_controller,
        mock_pressure_sensor
    ):
        """Test valve behavior on low pressure interlock."""
        mock_pressure_sensor.set_value(5.0)  # Below minimum

        result = await steam_quality_controller.check_safety_interlocks()

        assert 'LOW_PRESSURE' in result['active_interlocks']

    async def test_high_temperature_interlock(
        self,
        mock_control_valve,
        steam_quality_controller,
        mock_temperature_sensor
    ):
        """Test valve closes on high temperature interlock."""
        mock_temperature_sensor.set_value(650.0)  # Above limit

        result = await steam_quality_controller.check_safety_interlocks()

        assert 'HIGH_TEMPERATURE' in result['active_interlocks']

    async def test_multiple_interlock_priority(
        self,
        mock_control_valve,
        steam_quality_controller,
        mock_pressure_sensor,
        mock_temperature_sensor
    ):
        """Test multiple interlock priority handling."""
        # Trigger multiple interlocks
        mock_pressure_sensor.set_value(150.0)  # High pressure
        mock_temperature_sensor.set_value(650.0)  # High temperature

        result = await steam_quality_controller.check_safety_interlocks()

        assert len(result['active_interlocks']) >= 2
        # Highest priority should be processed first
        assert result['highest_priority_interlock'] is not None

    async def test_interlock_bypass_requires_authorization(
        self,
        mock_control_valve,
        steam_quality_controller,
        mock_pressure_sensor
    ):
        """Test interlock bypass requires proper authorization."""
        mock_pressure_sensor.set_value(150.0)  # Trigger interlock

        # Attempt bypass without authorization
        bypass_result = await steam_quality_controller.bypass_interlock(
            interlock_id='HIGH_PRESSURE',
            operator_id='OP-001',
            authorization_code=None
        )

        assert bypass_result['status'] == 'denied'
        assert 'authorization_required' in bypass_result['reason'].lower()

    async def test_authorized_interlock_bypass(
        self,
        mock_control_valve,
        steam_quality_controller,
        mock_pressure_sensor
    ):
        """Test authorized interlock bypass."""
        mock_pressure_sensor.set_value(150.0)  # Trigger interlock

        # Bypass with proper authorization
        bypass_result = await steam_quality_controller.bypass_interlock(
            interlock_id='HIGH_PRESSURE',
            operator_id='SUPERVISOR-001',
            authorization_code='BYPASS-AUTH-12345',
            reason='Controlled maintenance procedure'
        )

        assert bypass_result['status'] == 'success'
        assert bypass_result['bypass_active'] is True
        assert bypass_result['expires_at'] is not None

    async def test_interlock_bypass_timeout(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test interlock bypass automatic timeout."""
        # Activate bypass with short timeout
        await steam_quality_controller.bypass_interlock(
            interlock_id='HIGH_PRESSURE',
            operator_id='SUPERVISOR-001',
            authorization_code='BYPASS-AUTH-12345',
            timeout_seconds=1.0
        )

        # Wait for timeout
        await asyncio.sleep(1.5)

        # Bypass should have expired
        status = await steam_quality_controller.get_bypass_status('HIGH_PRESSURE')
        assert status['bypass_active'] is False

    async def test_fail_safe_valve_action(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test fail-safe valve action on communication loss."""
        # Simulate communication loss
        mock_control_valve.simulate_communication_loss()

        await asyncio.sleep(0.5)

        # Valve should fail to safe position (closed)
        status = await steam_quality_controller.read_valve_status('CV-STEAM-001')

        assert status['fail_safe_triggered'] is True
        assert status['position_percent'] == 0.0

    async def test_interlock_status_logging(
        self,
        mock_control_valve,
        steam_quality_controller,
        mock_pressure_sensor
    ):
        """Test interlock events are properly logged."""
        mock_pressure_sensor.set_value(150.0)  # Trigger interlock

        await steam_quality_controller.check_safety_interlocks()

        # Get interlock log
        log = await steam_quality_controller.get_interlock_log(
            start_time=datetime.now(timezone.utc) - timedelta(minutes=5)
        )

        assert len(log) > 0
        assert log[-1]['interlock_type'] == 'HIGH_PRESSURE'
        assert 'timestamp' in log[-1]
        assert 'action_taken' in log[-1]


# =============================================================================
# ACTUATOR RESPONSE TESTS
# =============================================================================

class TestActuatorResponse:
    """Test actuator response simulation."""

    async def test_actuator_step_response(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test actuator step response characteristics."""
        # Set to known starting position
        await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=0.0,
            wait_for_position=True
        )

        # Step to 50%
        start_time = asyncio.get_event_loop().time()
        await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=50.0,
            wait_for_position=True
        )
        step_time = asyncio.get_event_loop().time() - start_time

        # Record step response
        assert step_time > 0

    async def test_actuator_dead_time(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test actuator dead time measurement."""
        mock_control_valve.set_dead_time(0.2)  # 200ms dead time

        start_time = asyncio.get_event_loop().time()

        await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=50.0
        )

        # Read position until it starts moving
        while True:
            status = await steam_quality_controller.read_valve_status('CV-STEAM-001')
            if status['actuator_state'] == 'OPENING':
                break
            if asyncio.get_event_loop().time() - start_time > 5.0:
                break
            await asyncio.sleep(0.05)

        dead_time = asyncio.get_event_loop().time() - start_time
        assert dead_time >= 0.15  # Should be at least 150ms

    async def test_actuator_stall_detection(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test actuator stall detection."""
        mock_control_valve.simulate_stall()

        result = await steam_quality_controller.set_valve_position(
            valve_id='CV-STEAM-001',
            position_percent=50.0,
            wait_for_position=True,
            timeout_seconds=5.0
        )

        assert result['status'] == 'fault'
        assert 'stall' in result['fault_type'].lower()

    async def test_actuator_overcurrent_detection(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test actuator overcurrent fault detection."""
        mock_control_valve.simulate_overcurrent()

        diagnostics = await steam_quality_controller.read_valve_diagnostics(
            'CV-STEAM-001'
        )

        assert diagnostics['health_status'] == 'FAULT'
        assert 'OVERCURRENT' in diagnostics['active_alerts']


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.performance
class TestControlValvePerformance:
    """Performance tests for control valve operations."""

    async def test_valve_command_latency(
        self,
        mock_control_valve,
        steam_quality_controller,
        performance_monitor
    ):
        """Test valve command latency meets target (<100ms)."""
        performance_monitor.start()

        latencies = []
        for i in range(50):
            position = (i * 2) % 100

            start = asyncio.get_event_loop().time()
            await steam_quality_controller.set_valve_position(
                valve_id='CV-STEAM-001',
                position_percent=float(position)
            )
            latency_ms = (asyncio.get_event_loop().time() - start) * 1000
            latencies.append(latency_ms)
            performance_monitor.record_metric('command_latency_ms', latency_ms)

        avg_latency = sum(latencies) / len(latencies)

        print(f"\n=== Valve Command Latency ===")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"Max: {max(latencies):.2f}ms")

        assert avg_latency < 100.0, f"Average latency {avg_latency}ms exceeds 100ms target"

    async def test_status_read_throughput(
        self,
        mock_control_valve,
        steam_quality_controller
    ):
        """Test valve status read throughput."""
        num_reads = 100
        start_time = asyncio.get_event_loop().time()

        for _ in range(num_reads):
            await steam_quality_controller.read_valve_status('CV-STEAM-001')

        elapsed = asyncio.get_event_loop().time() - start_time
        reads_per_second = num_reads / elapsed

        print(f"\n=== Status Read Throughput ===")
        print(f"Reads per second: {reads_per_second:.1f}")

        assert reads_per_second >= 50, f"Throughput {reads_per_second}/s below 50/s target"

    async def test_concurrent_valve_control(
        self,
        steam_quality_controller
    ):
        """Test concurrent control of multiple valves."""
        valve_ids = ['CV-STEAM-001', 'CV-STEAM-002', 'CV-STEAM-003']

        async def control_valve(valve_id, position):
            return await steam_quality_controller.set_valve_position(
                valve_id=valve_id,
                position_percent=position
            )

        tasks = [
            control_valve(valve_id, 50.0 + i * 10)
            for i, valve_id in enumerate(valve_ids)
        ]

        results = await asyncio.gather(*tasks)

        # All commands should succeed
        assert all(r['status'] == 'success' for r in results)
