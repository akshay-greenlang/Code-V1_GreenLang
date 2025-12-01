# -*- coding: utf-8 -*-
"""
Desuperheater Integration Tests for GL-012 SteamQualityController

Comprehensive integration tests for desuperheater control systems including:
- Spray water injection rate control
- Temperature regulation loops (PID control)
- Spray valve coordination
- Water supply monitoring
- Fault handling and recovery

Test Count: 28+ tests
Coverage Target: 90%+

Standards: ASME PTC 4.4 (Steam), ISA-5.1 (Instrumentation)

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from decimal import Decimal
import math

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.desuperheater]


# =============================================================================
# INJECTION RATE CONTROL TESTS
# =============================================================================

class TestInjectionRateControl:
    """Test spray water injection rate control."""

    async def test_set_injection_rate_valid_range(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test setting injection rate within valid range."""
        test_rates = [0.0, 5.0, 10.0, 25.0, 50.0]  # kg/s

        for target_rate in test_rates:
            result = await steam_quality_controller.set_injection_rate(
                desuperheater_id='DSH-001',
                rate_kg_s=target_rate
            )

            assert result['status'] == 'success'
            assert abs(result['actual_rate_kg_s'] - target_rate) < 0.5

    async def test_injection_rate_exceeds_maximum(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test rejection of injection rate above maximum."""
        # Get maximum rate from config
        max_rate = await steam_quality_controller.get_max_injection_rate('DSH-001')

        result = await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=max_rate + 10.0
        )

        assert result['status'] == 'error'
        assert 'exceeds_maximum' in result['error_code'].lower()

    async def test_injection_rate_negative_rejected(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test rejection of negative injection rate."""
        result = await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=-5.0
        )

        assert result['status'] == 'error'
        assert 'invalid_value' in result['error_code'].lower()

    async def test_injection_rate_ramp_control(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test injection rate change with ramp control."""
        # Configure ramp rate
        steam_quality_controller.configure_desuperheater(
            desuperheater_id='DSH-001',
            ramp_rate_kg_s_per_second=5.0
        )

        # Set initial rate
        await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=0.0
        )

        # Request large rate change
        start_time = asyncio.get_event_loop().time()
        result = await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=25.0,
            wait_for_setpoint=True
        )
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should take ~5 seconds (25 kg/s at 5 kg/s/s)
        assert elapsed >= 4.5

    async def test_injection_rate_feedback_reading(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test reading actual injection rate from flow sensor."""
        await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=15.0
        )

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert 'actual_injection_rate_kg_s' in status
        assert abs(status['actual_injection_rate_kg_s'] - 15.0) < 1.0

    async def test_injection_mass_balance_calculation(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test mass balance calculation for desuperheater."""
        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        steam_inlet_flow = status['steam_inlet_flow_kg_s']
        injection_rate = status['actual_injection_rate_kg_s']
        steam_outlet_flow = status['steam_outlet_flow_kg_s']

        # Mass balance: outlet = inlet + injection
        expected_outlet = steam_inlet_flow + injection_rate
        mass_balance_error = abs(steam_outlet_flow - expected_outlet)

        assert mass_balance_error < 0.5, f"Mass balance error: {mass_balance_error} kg/s"


# =============================================================================
# TEMPERATURE REGULATION LOOP TESTS
# =============================================================================

class TestTemperatureRegulationLoop:
    """Test temperature regulation control loops."""

    async def test_temperature_setpoint_control(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test temperature setpoint control."""
        target_temp = 400.0  # degrees C

        result = await steam_quality_controller.set_outlet_temperature(
            desuperheater_id='DSH-001',
            temperature_setpoint_c=target_temp
        )

        assert result['status'] == 'success'
        assert result['setpoint_c'] == target_temp

    async def test_pid_controller_response(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test PID controller temperature response."""
        # Set target temperature
        await steam_quality_controller.set_outlet_temperature(
            desuperheater_id='DSH-001',
            temperature_setpoint_c=400.0
        )

        # Simulate disturbance (inlet temp increase)
        mock_desuperheater.set_inlet_temperature(550.0)

        # Allow controller to respond
        await asyncio.sleep(2.0)

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        # Controller should increase injection to compensate
        assert status['controller_output'] > 0

    async def test_temperature_control_within_tolerance(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test temperature control maintains setpoint within tolerance."""
        setpoint = 400.0
        tolerance = 5.0  # +/- 5 degrees C

        await steam_quality_controller.set_outlet_temperature(
            desuperheater_id='DSH-001',
            temperature_setpoint_c=setpoint
        )

        # Wait for steady state
        await asyncio.sleep(3.0)

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        temp_error = abs(status['outlet_temperature_c'] - setpoint)
        assert temp_error <= tolerance, f"Temperature error {temp_error}C exceeds tolerance"

    async def test_cascade_control_configuration(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test cascade control loop configuration."""
        # Configure cascade control (temperature -> injection rate)
        result = await steam_quality_controller.configure_cascade_control(
            desuperheater_id='DSH-001',
            primary_pv='outlet_temperature',
            secondary_pv='injection_rate',
            primary_tuning={'kp': 2.0, 'ki': 0.5, 'kd': 0.1},
            secondary_tuning={'kp': 1.0, 'ki': 0.2, 'kd': 0.0}
        )

        assert result['status'] == 'success'
        assert result['control_mode'] == 'CASCADE'

    async def test_anti_windup_protection(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test PID anti-windup protection during saturation."""
        # Set unreachable setpoint to cause controller saturation
        await steam_quality_controller.set_outlet_temperature(
            desuperheater_id='DSH-001',
            temperature_setpoint_c=200.0  # Very low, likely unreachable
        )

        # Run controller for a while
        await asyncio.sleep(2.0)

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        # Integral term should be limited
        assert status['controller_integral_limited'] is True
        assert status['controller_output'] <= 100.0  # Should not exceed 100%

    async def test_feedforward_compensation(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test feedforward compensation for load changes."""
        # Enable feedforward
        await steam_quality_controller.configure_feedforward(
            desuperheater_id='DSH-001',
            enabled=True,
            steam_flow_gain=0.5,
            inlet_temp_gain=0.3
        )

        # Simulate load change
        mock_desuperheater.set_steam_flow(100.0)  # Increase flow
        await asyncio.sleep(1.0)

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        # Feedforward should have preemptively adjusted injection
        assert status['feedforward_contribution'] > 0


# =============================================================================
# SPRAY VALVE COORDINATION TESTS
# =============================================================================

class TestSprayValveCoordination:
    """Test spray valve coordination operations."""

    async def test_primary_secondary_valve_sequencing(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test primary/secondary valve sequencing."""
        # Low demand - primary valve only
        await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=5.0
        )

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert status['primary_valve_position'] > 0
        assert status['secondary_valve_position'] == 0

        # High demand - both valves
        await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=40.0
        )

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert status['primary_valve_position'] == 100.0  # Fully open
        assert status['secondary_valve_position'] > 0

    async def test_spray_nozzle_selection(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test spray nozzle selection for optimal atomization."""
        # Configure multiple nozzles
        await steam_quality_controller.configure_spray_nozzles(
            desuperheater_id='DSH-001',
            nozzles=[
                {'id': 'N1', 'capacity_kg_s': 10.0},
                {'id': 'N2', 'capacity_kg_s': 20.0},
                {'id': 'N3', 'capacity_kg_s': 30.0}
            ]
        )

        # Request specific injection rate
        await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=25.0
        )

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        # Should select optimal nozzle combination
        assert 'active_nozzles' in status
        assert len(status['active_nozzles']) > 0

    async def test_valve_position_limits(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test spray valve position limits are respected."""
        # Configure position limits
        await steam_quality_controller.configure_valve_limits(
            desuperheater_id='DSH-001',
            valve_id='SPRAY-VALVE-001',
            min_position_percent=5.0,  # Minimum for atomization
            max_position_percent=95.0  # Maximum for wear protection
        )

        # Request very low injection
        await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=0.5
        )

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        # Valve should be at minimum (not fully closed)
        assert status['spray_valve_position'] >= 5.0

    async def test_valve_split_range_control(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test split-range control between multiple spray valves."""
        # Configure split range
        await steam_quality_controller.configure_split_range(
            desuperheater_id='DSH-001',
            valves=[
                {'id': 'V1', 'range_start': 0, 'range_end': 50},
                {'id': 'V2', 'range_start': 50, 'range_end': 100}
            ]
        )

        # 75% output should have V1 at 100%, V2 at 50%
        await steam_quality_controller.set_controller_output(
            desuperheater_id='DSH-001',
            output_percent=75.0
        )

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert status['valve_positions']['V1'] == 100.0
        assert abs(status['valve_positions']['V2'] - 50.0) < 5.0


# =============================================================================
# WATER SUPPLY MONITORING TESTS
# =============================================================================

class TestWaterSupplyMonitoring:
    """Test spray water supply monitoring."""

    async def test_water_pressure_monitoring(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test spray water supply pressure monitoring."""
        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert 'water_supply_pressure_bar' in status
        assert status['water_supply_pressure_bar'] > 0

    async def test_low_water_pressure_alarm(
        self,
        mock_desuperheater,
        steam_quality_controller,
        mock_water_supply
    ):
        """Test low water pressure alarm generation."""
        # Simulate low pressure
        mock_water_supply.set_pressure(5.0)  # Below minimum

        await asyncio.sleep(0.5)

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert status['water_pressure_alarm'] is True
        assert 'LOW_WATER_PRESSURE' in status['active_alarms']

    async def test_water_temperature_monitoring(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test spray water temperature monitoring."""
        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert 'water_supply_temperature_c' in status
        # Water should be subcooled
        assert status['water_supply_temperature_c'] < 100.0

    async def test_water_flow_measurement(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test spray water flow measurement."""
        await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=20.0
        )

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert 'water_flow_rate_kg_s' in status
        assert abs(status['water_flow_rate_kg_s'] - 20.0) < 2.0

    async def test_water_quality_monitoring(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test spray water quality monitoring."""
        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert 'water_quality' in status
        assert 'conductivity_us_cm' in status['water_quality']
        assert 'ph' in status['water_quality']

    async def test_strainer_differential_pressure(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test water strainer differential pressure monitoring."""
        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert 'strainer_dp_bar' in status

        # High DP indicates clogged strainer
        if status['strainer_dp_bar'] > 1.0:
            assert 'STRAINER_CLOGGED' in status['active_alarms']

    async def test_pump_status_monitoring(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test spray water pump status monitoring."""
        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert 'pump_status' in status
        assert status['pump_status'] in ['RUNNING', 'STOPPED', 'FAULT']


# =============================================================================
# FAULT HANDLING TESTS
# =============================================================================

class TestDesuperheaterFaultHandling:
    """Test desuperheater fault handling and recovery."""

    async def test_spray_valve_stuck_detection(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test detection of stuck spray valve."""
        mock_desuperheater.simulate_stuck_valve()

        # Attempt to change position
        result = await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=20.0,
            timeout_seconds=5.0
        )

        assert result['status'] == 'fault'
        assert 'valve_stuck' in result['fault_type'].lower()

    async def test_temperature_sensor_failure(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test handling of temperature sensor failure."""
        mock_desuperheater.simulate_sensor_failure('outlet_temperature')

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert status['outlet_temperature_quality'] == 'BAD'
        assert 'SENSOR_FAILURE' in status['active_alarms']

    async def test_fallback_to_manual_control(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test automatic fallback to manual control on sensor failure."""
        mock_desuperheater.simulate_sensor_failure('outlet_temperature')

        await asyncio.sleep(1.0)

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert status['control_mode'] == 'MANUAL'
        assert status['auto_fallback_active'] is True

    async def test_water_supply_failure_response(
        self,
        mock_desuperheater,
        steam_quality_controller,
        mock_water_supply
    ):
        """Test response to water supply failure."""
        mock_water_supply.simulate_failure()

        await asyncio.sleep(0.5)

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        # Spray should be shut off
        assert status['spray_valve_position'] == 0.0
        assert 'WATER_SUPPLY_FAILURE' in status['active_alarms']

    async def test_thermal_shock_protection(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test thermal shock protection on cold start."""
        # Simulate cold desuperheater
        mock_desuperheater.set_outlet_temperature(100.0)  # Cold

        # Attempt rapid injection
        result = await steam_quality_controller.set_injection_rate(
            desuperheater_id='DSH-001',
            rate_kg_s=50.0  # High rate
        )

        # Should limit rate to prevent thermal shock
        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert status['actual_injection_rate_kg_s'] < 50.0
        assert status['thermal_protection_active'] is True

    async def test_water_hammer_prevention(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test water hammer prevention during valve cycling."""
        # Rapid on/off cycling should be prevented
        cycle_times = []

        for i in range(5):
            await steam_quality_controller.set_injection_rate(
                desuperheater_id='DSH-001',
                rate_kg_s=20.0 if i % 2 == 0 else 0.0
            )
            cycle_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)

        # Controller should impose minimum cycle time
        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert status.get('cycle_limiting_active', False) or \
               status.get('min_cycle_time_enforced', False)

    async def test_nozzle_blockage_detection(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test spray nozzle blockage detection."""
        mock_desuperheater.simulate_nozzle_blockage('N1')

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        assert 'NOZZLE_BLOCKED' in status['active_alarms']
        assert 'N1' in status['blocked_nozzles']

    async def test_fault_recovery_procedure(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test fault recovery procedure."""
        # Simulate and clear fault
        mock_desuperheater.simulate_stuck_valve()
        await asyncio.sleep(0.5)

        mock_desuperheater.clear_faults()

        # Acknowledge fault
        result = await steam_quality_controller.acknowledge_fault(
            desuperheater_id='DSH-001',
            fault_id='VALVE_STUCK_001',
            operator_id='OP-001'
        )

        assert result['status'] == 'success'

        # Reset to auto control
        result = await steam_quality_controller.reset_to_auto(
            desuperheater_id='DSH-001'
        )

        assert result['status'] == 'success'
        assert result['control_mode'] == 'AUTO'

    async def test_alarm_priority_classification(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test alarm priority classification."""
        # Trigger multiple alarms
        mock_desuperheater.simulate_sensor_failure('outlet_temperature')
        mock_desuperheater.set_inlet_temperature(700.0)  # High temp

        status = await steam_quality_controller.read_desuperheater_status('DSH-001')

        alarms = status['active_alarms']

        # Check priority classification
        for alarm in alarms:
            assert 'priority' in alarm
            assert alarm['priority'] in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.performance
class TestDesuperheaterPerformance:
    """Performance tests for desuperheater control."""

    async def test_temperature_control_response_time(
        self,
        mock_desuperheater,
        steam_quality_controller,
        performance_monitor
    ):
        """Test temperature control response time to setpoint change."""
        performance_monitor.start()

        # Set initial setpoint
        await steam_quality_controller.set_outlet_temperature(
            desuperheater_id='DSH-001',
            temperature_setpoint_c=450.0
        )
        await asyncio.sleep(2.0)  # Allow to stabilize

        # Step change
        start_time = asyncio.get_event_loop().time()
        await steam_quality_controller.set_outlet_temperature(
            desuperheater_id='DSH-001',
            temperature_setpoint_c=400.0
        )

        # Wait for within 5% of setpoint
        target = 400.0
        tolerance = 20.0  # 5%

        while True:
            status = await steam_quality_controller.read_desuperheater_status('DSH-001')
            error = abs(status['outlet_temperature_c'] - target)

            if error <= tolerance:
                break

            if asyncio.get_event_loop().time() - start_time > 60.0:
                break

            await asyncio.sleep(0.5)

        response_time = asyncio.get_event_loop().time() - start_time

        print(f"\n=== Temperature Response Time ===")
        print(f"Response time: {response_time:.1f}s")

        assert response_time < 30.0, f"Response time {response_time}s exceeds 30s target"

    async def test_control_loop_execution_rate(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test control loop execution rate (10 Hz minimum)."""
        execution_count = 0
        start_time = asyncio.get_event_loop().time()

        # Monitor control loop executions for 1 second
        while asyncio.get_event_loop().time() - start_time < 1.0:
            await steam_quality_controller.execute_control_loop('DSH-001')
            execution_count += 1
            await asyncio.sleep(0.01)  # Allow other tasks

        print(f"\n=== Control Loop Rate ===")
        print(f"Executions per second: {execution_count}")

        assert execution_count >= 10, f"Control rate {execution_count}/s below 10 Hz target"

    async def test_disturbance_rejection(
        self,
        mock_desuperheater,
        steam_quality_controller
    ):
        """Test disturbance rejection performance."""
        # Establish steady state at setpoint
        await steam_quality_controller.set_outlet_temperature(
            desuperheater_id='DSH-001',
            temperature_setpoint_c=400.0
        )
        await asyncio.sleep(3.0)

        # Apply disturbance
        mock_desuperheater.apply_disturbance(
            parameter='inlet_temperature',
            magnitude=50.0  # +50C step
        )

        # Measure peak deviation
        max_deviation = 0
        for _ in range(30):
            status = await steam_quality_controller.read_desuperheater_status('DSH-001')
            deviation = abs(status['outlet_temperature_c'] - 400.0)
            max_deviation = max(max_deviation, deviation)
            await asyncio.sleep(0.5)

        print(f"\n=== Disturbance Rejection ===")
        print(f"Max deviation: {max_deviation:.1f}C")

        assert max_deviation < 30.0, f"Max deviation {max_deviation}C exceeds 30C limit"
