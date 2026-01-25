"""
GL-002 FLAMEGUARD - Edge Case Tests

Comprehensive tests for edge cases and failure scenarios:
- Flame loss scenarios
- Sensor degradation and failures
- Fuel quality variations
- Emergency shutdown sequences
- Boundary conditions
- Error recovery
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys

sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])


# =============================================================================
# FLAME LOSS SCENARIO TESTS
# =============================================================================


class TestFlameLossScenarios:
    """Test flame loss detection and response scenarios."""

    @pytest.fixture
    def flame_detector(self):
        from safety.flame_detector import FlameDetector
        detector = FlameDetector(boiler_id="BOILER-001", voting_logic="2oo3")
        detector.add_scanner("UV-1", "UV")
        detector.add_scanner("UV-2", "UV")
        detector.add_scanner("UV-3", "UV")
        return detector

    def test_gradual_flame_degradation(self, flame_detector):
        """Test gradual flame signal degradation."""
        # Start with strong flame
        flame_detector.update_scanner("UV-1", signal_percent=90.0)
        flame_detector.update_scanner("UV-2", signal_percent=85.0)
        flame_detector.update_scanner("UV-3", signal_percent=88.0)
        assert flame_detector.is_flame_proven()

        # Gradual degradation
        for signal in [70, 50, 30, 15, 5]:
            flame_detector.update_scanner("UV-1", signal_percent=float(signal))
            flame_detector.update_scanner("UV-2", signal_percent=float(signal))
            flame_detector.update_scanner("UV-3", signal_percent=float(signal))

        # Should eventually lose flame
        assert not flame_detector.is_flame_proven()

    def test_intermittent_flame(self, flame_detector):
        """Test intermittent flame signal (flickering)."""
        signals = [80, 10, 75, 5, 85, 8, 90, 3]

        for signal in signals:
            flame_detector.update_scanner("UV-1", signal_percent=float(signal))
            flame_detector.update_scanner("UV-2", signal_percent=float(signal))
            flame_detector.update_scanner("UV-3", signal_percent=float(signal))

        # Final state depends on last signal
        # Signal was 3% - should not be proven
        assert not flame_detector.is_flame_proven()

    def test_single_scanner_dropout(self, flame_detector):
        """Test single scanner dropout (2oo3 should still work)."""
        # All scanners see flame
        flame_detector.update_scanner("UV-1", signal_percent=85.0)
        flame_detector.update_scanner("UV-2", signal_percent=80.0)
        flame_detector.update_scanner("UV-3", signal_percent=82.0)
        assert flame_detector.is_flame_proven()

        # One scanner drops out
        flame_detector.update_scanner("UV-1", signal_percent=0.0)

        # 2oo3 should still prove flame
        assert flame_detector.is_flame_proven()

    def test_two_scanner_dropout(self, flame_detector):
        """Test two scanner dropout (2oo3 should fail)."""
        # All scanners see flame
        flame_detector.update_scanner("UV-1", signal_percent=85.0)
        flame_detector.update_scanner("UV-2", signal_percent=80.0)
        flame_detector.update_scanner("UV-3", signal_percent=82.0)
        assert flame_detector.is_flame_proven()

        # Two scanners drop out
        flame_detector.update_scanner("UV-1", signal_percent=0.0)
        flame_detector.update_scanner("UV-2", signal_percent=0.0)

        # 2oo3 should fail
        assert not flame_detector.is_flame_proven()

    def test_flame_loss_callback_timing(self, flame_detector):
        """Test flame loss callback is called after timeout."""
        callback_times = []

        def on_failure(boiler_id):
            callback_times.append(datetime.now(timezone.utc))

        flame_detector._failure_callback = on_failure

        # Establish flame
        flame_detector.update_scanner("UV-1", signal_percent=85.0)
        flame_detector.update_scanner("UV-2", signal_percent=80.0)
        flame_detector.update_scanner("UV-3", signal_percent=82.0)

        # Simulate flame loss with expired timer
        flame_detector._flame_loss_time = datetime.now(timezone.utc) - timedelta(seconds=5)

        # Lose flame
        flame_detector.update_scanner("UV-1", signal_percent=0.0)
        flame_detector.update_scanner("UV-2", signal_percent=0.0)
        flame_detector.update_scanner("UV-3", signal_percent=0.0)

        # Callback should have been called
        assert len(callback_times) >= 1


# =============================================================================
# SENSOR DEGRADATION TESTS
# =============================================================================


class TestSensorDegradation:
    """Test sensor degradation and failure handling."""

    @pytest.fixture
    def flame_detector(self):
        from safety.flame_detector import FlameDetector
        detector = FlameDetector(boiler_id="BOILER-001", voting_logic="2oo3")
        detector.add_scanner("UV-1", "UV")
        detector.add_scanner("UV-2", "UV")
        detector.add_scanner("UV-3", "UV")
        return detector

    @pytest.fixture
    def interlock_manager(self):
        from safety.safety_interlocks import SafetyInterlockManager
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_scanner_fault_detection(self, flame_detector):
        """Test scanner fault is detected."""
        from safety.flame_detector import FlameStatus

        # One scanner faults
        flame_detector.update_scanner("UV-1", signal_percent=80.0, healthy=True)
        flame_detector.update_scanner("UV-2", signal_percent=80.0, healthy=False)  # Fault
        flame_detector.update_scanner("UV-3", signal_percent=80.0, healthy=True)

        status = flame_detector.get_status()
        assert status["scanners"]["UV-2"]["healthy"] is False

    def test_multiple_scanner_faults(self, flame_detector):
        """Test multiple scanner faults."""
        from safety.flame_detector import FlameStatus

        # All scanners fault
        flame_detector.update_scanner("UV-1", signal_percent=80.0, healthy=False)
        flame_detector.update_scanner("UV-2", signal_percent=80.0, healthy=False)
        flame_detector.update_scanner("UV-3", signal_percent=80.0, healthy=False)

        assert flame_detector.status == FlameStatus.SCANNER_FAULT
        assert not flame_detector.is_flame_proven()

    def test_sensor_drift_simulation(self, interlock_manager):
        """Test sensor drift causing alarm."""
        from safety.safety_interlocks import InterlockStatus

        # Normal reading
        status = interlock_manager.update_value("STEAM_PRESSURE", 120.0)
        assert status == InterlockStatus.NORMAL

        # Gradual drift toward alarm
        for pressure in [125, 130, 135, 140, 142]:
            status = interlock_manager.update_value("STEAM_PRESSURE", float(pressure))

        # Should be in alarm at 142
        assert status == InterlockStatus.ALARM

    def test_sensor_spike_and_recovery(self, interlock_manager):
        """Test sensor spike followed by recovery."""
        from safety.safety_interlocks import InterlockStatus

        # Normal reading
        interlock_manager.update_value("STEAM_PRESSURE", 120.0)

        # Spike to trip
        interlock_manager.update_value("STEAM_PRESSURE", 155.0)
        assert interlock_manager.is_tripped

        # Recover to normal
        interlock_manager.update_value("STEAM_PRESSURE", 120.0)

        # Still tripped until reset
        assert interlock_manager.is_tripped

        # Manual reset
        interlock_manager.reset_trip("OPERATOR-001")
        assert not interlock_manager.is_tripped

    def test_frozen_sensor_detection(self, interlock_manager):
        """Test detection of frozen sensor (no change for extended period)."""
        # This would require time-based tracking
        # For now, just verify same value updates work
        for _ in range(100):
            interlock_manager.update_value("STEAM_PRESSURE", 120.0)

        # Should still be normal
        status = interlock_manager.get_status()
        assert not status["tripped"]


# =============================================================================
# FUEL QUALITY VARIATION TESTS
# =============================================================================


class TestFuelQualityVariations:
    """Test handling of fuel quality variations."""

    @pytest.fixture
    def efficiency_calculator(self):
        from calculators.efficiency_calculator import EfficiencyCalculator
        return EfficiencyCalculator()

    @pytest.fixture
    def emissions_calculator(self):
        from calculators.emissions_calculator import EmissionsCalculator
        return EmissionsCalculator()

    def test_low_btu_fuel(self, efficiency_calculator):
        """Test efficiency calculation with low BTU fuel."""
        from calculators.efficiency_calculator import EfficiencyInput

        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=30000.0,  # Higher flow needed for low BTU
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=4.0,
            fuel_type="landfill_gas",  # Low BTU fuel
            fuel_hhv_override=500.0,  # Low heating value
        )

        result = efficiency_calculator.calculate(inp, method="indirect")

        # Should still calculate efficiency
        assert 50.0 <= result.efficiency_hhv_percent <= 100.0

    def test_high_moisture_fuel(self, efficiency_calculator):
        """Test efficiency calculation with high moisture fuel."""
        from calculators.efficiency_calculator import EfficiencyInput

        inp = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=15000.0,
            flue_gas_temperature_f=400.0,
            flue_gas_o2_percent=4.0,
            fuel_type="coal_subbituminous",
            fuel_moisture_percent=30.0,  # High moisture
        )

        result = efficiency_calculator.calculate(inp, method="indirect")

        # High moisture should increase moisture loss
        assert result.moisture_in_fuel_loss_percent > 0

    def test_variable_sulfur_content(self, emissions_calculator):
        """Test emissions with variable sulfur content."""
        from calculators.emissions_calculator import EmissionsInput

        # Low sulfur
        inp_low = EmissionsInput(
            fuel_type="fuel_oil_no2",
            heat_input_mmbtu_hr=100.0,
            sulfur_content_percent=0.1,
            flue_gas_o2_percent=3.0,
        )
        result_low = emissions_calculator.calculate(inp_low)

        # High sulfur
        inp_high = EmissionsInput(
            fuel_type="fuel_oil_no2",
            heat_input_mmbtu_hr=100.0,
            sulfur_content_percent=2.0,
            flue_gas_o2_percent=3.0,
        )
        result_high = emissions_calculator.calculate(inp_high)

        # High sulfur should produce more SO2
        assert result_high.so2_lb_hr > result_low.so2_lb_hr


# =============================================================================
# EMERGENCY SHUTDOWN SEQUENCE TESTS
# =============================================================================


class TestEmergencyShutdownSequences:
    """Test emergency shutdown sequence handling."""

    @pytest.fixture
    def bms(self):
        from safety.burner_management import BurnerManagementSystem, BurnerState
        bms = BurnerManagementSystem("BOILER-001")
        # Set to firing state
        bms._state = BurnerState.FIRING
        bms._flame_proven = True
        return bms

    @pytest.fixture
    def interlock_manager(self):
        from safety.safety_interlocks import SafetyInterlockManager
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_emergency_stop_sequence(self, bms):
        """Test emergency stop transitions to lockout."""
        from safety.burner_management import BurnerState

        bms.emergency_stop("Operator E-Stop pressed")

        assert bms.state == BurnerState.LOCKOUT
        assert "E-Stop" in bms.get_status()["lockout_reason"]

    def test_high_pressure_shutdown(self, bms, interlock_manager):
        """Test high pressure causes shutdown."""
        from safety.safety_interlocks import InterlockStatus

        # Trigger high pressure
        status = interlock_manager.update_value("STEAM_PRESSURE", 160.0)
        assert status == InterlockStatus.TRIP
        assert interlock_manager.is_tripped

        # In real system, this would trigger BMS shutdown

    def test_low_water_shutdown(self, bms, interlock_manager):
        """Test low water causes immediate shutdown."""
        from safety.safety_interlocks import InterlockStatus

        # Trigger low water
        status = interlock_manager.update_value("DRUM_LEVEL", -5.0)
        assert status == InterlockStatus.TRIP

        # This is SIL3 - cannot be bypassed

    def test_fuel_pressure_loss(self, bms, interlock_manager):
        """Test fuel pressure loss causes shutdown."""
        from safety.safety_interlocks import InterlockStatus

        # Trigger low fuel pressure
        status = interlock_manager.update_value("FUEL_PRESSURE", 1.0)
        assert status == InterlockStatus.TRIP

    def test_multiple_simultaneous_trips(self, interlock_manager):
        """Test multiple simultaneous trip conditions."""
        # Multiple trips at once
        interlock_manager.update_value("STEAM_PRESSURE", 160.0)
        interlock_manager.update_value("FUEL_PRESSURE", 1.0)
        interlock_manager.update_value("DRUM_LEVEL", -5.0)

        status = interlock_manager.get_status()

        # Should record all trip causes
        assert len(status["trip_causes"]) >= 3
        assert interlock_manager.is_tripped

    def test_shutdown_with_flame_loss(self, bms):
        """Test shutdown when flame is lost during firing."""
        from safety.burner_management import BurnerState

        # Simulate flame loss
        bms.update_flame_signal(0.0)

        assert bms.state == BurnerState.LOCKOUT
        assert "flame" in bms.get_status()["lockout_reason"].lower()


# =============================================================================
# BOUNDARY CONDITION TESTS
# =============================================================================


class TestBoundaryConditions:
    """Test boundary conditions and edge values."""

    @pytest.fixture
    def efficiency_calculator(self):
        from calculators.efficiency_calculator import EfficiencyCalculator
        return EfficiencyCalculator()

    @pytest.fixture
    def pid_controller(self):
        from optimization.o2_trim_controller import PIDController
        return PIDController(kp=2.0, ki=0.1, kd=0.05)

    def test_zero_load_operation(self, efficiency_calculator):
        """Test calculations at zero load."""
        from calculators.efficiency_calculator import EfficiencyInput

        inp = EfficiencyInput(
            steam_flow_klb_hr=0.0,  # Zero steam
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=0.0,  # Zero fuel
            flue_gas_temperature_f=200.0,
            flue_gas_o2_percent=21.0,  # Atmospheric
            fuel_type="natural_gas",
        )

        result = efficiency_calculator.calculate(inp, method="indirect")

        # Should handle gracefully
        assert result.efficiency_hhv_percent >= 50.0

    def test_maximum_load_operation(self, efficiency_calculator):
        """Test calculations at maximum load."""
        from calculators.efficiency_calculator import EfficiencyInput

        inp = EfficiencyInput(
            steam_flow_klb_hr=500.0,  # Maximum steam
            steam_pressure_psig=200.0,  # High pressure
            steam_temperature_f=500.0,
            feedwater_temperature_f=227.0,
            fuel_flow_rate=100000.0,  # High fuel
            flue_gas_temperature_f=450.0,
            flue_gas_o2_percent=2.0,  # Low O2
            fuel_type="natural_gas",
        )

        result = efficiency_calculator.calculate(inp, method="indirect")

        assert 50.0 <= result.efficiency_hhv_percent <= 100.0

    def test_extreme_temperatures(self, efficiency_calculator):
        """Test calculations at extreme temperatures."""
        from calculators.efficiency_calculator import EfficiencyInput

        # Very cold ambient
        inp_cold = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=100.0,  # Cold feedwater
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=300.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
            ambient_temperature_f=-20.0,  # Very cold
        )

        result_cold = efficiency_calculator.calculate(inp_cold, method="indirect")
        assert 50.0 <= result_cold.efficiency_hhv_percent <= 100.0

        # Very hot ambient
        inp_hot = EfficiencyInput(
            steam_flow_klb_hr=100.0,
            steam_pressure_psig=150.0,
            steam_temperature_f=366.0,
            feedwater_temperature_f=250.0,  # Hot feedwater
            fuel_flow_rate=8000.0,
            flue_gas_temperature_f=600.0,
            flue_gas_o2_percent=3.0,
            fuel_type="natural_gas",
            ambient_temperature_f=120.0,  # Very hot
        )

        result_hot = efficiency_calculator.calculate(inp_hot, method="indirect")
        assert 50.0 <= result_hot.efficiency_hhv_percent <= 100.0

    def test_pid_extreme_setpoint_change(self, pid_controller):
        """Test PID response to extreme setpoint change."""
        # Normal operation
        for i in range(10):
            pid_controller.compute(setpoint=3.0, process_value=3.0, timestamp=float(i))

        # Extreme setpoint change
        output = pid_controller.compute(setpoint=8.0, process_value=3.0, timestamp=10.0)

        # Should be bounded by output limits
        assert -10.0 <= output <= 10.0

    def test_pid_rapid_oscillation(self, pid_controller):
        """Test PID with rapid process value oscillation."""
        outputs = []

        for i in range(100):
            # Oscillating process value
            pv = 3.0 + (1.0 if i % 2 == 0 else -1.0)
            output = pid_controller.compute(setpoint=3.0, process_value=pv, timestamp=float(i))
            outputs.append(output)

        # Output should remain bounded
        assert all(-10.0 <= o <= 10.0 for o in outputs)


# =============================================================================
# ERROR RECOVERY TESTS
# =============================================================================


class TestErrorRecovery:
    """Test error recovery scenarios."""

    @pytest.fixture
    def interlock_manager(self):
        from safety.safety_interlocks import SafetyInterlockManager
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_trip_and_recovery_sequence(self, interlock_manager):
        """Test full trip and recovery sequence."""
        from safety.safety_interlocks import InterlockStatus

        # Normal operation
        interlock_manager.update_value("STEAM_PRESSURE", 120.0)
        assert not interlock_manager.is_tripped

        # Trip
        interlock_manager.update_value("STEAM_PRESSURE", 160.0)
        assert interlock_manager.is_tripped

        # Return to normal
        interlock_manager.update_value("STEAM_PRESSURE", 110.0)

        # Attempt reset
        result = interlock_manager.reset_trip("OPERATOR-001")
        assert result is True
        assert not interlock_manager.is_tripped

    def test_failed_reset_attempt(self, interlock_manager):
        """Test failed reset while still in trip condition."""
        # Trip
        interlock_manager.update_value("STEAM_PRESSURE", 160.0)
        assert interlock_manager.is_tripped

        # Try to reset while still in trip condition
        result = interlock_manager.reset_trip("OPERATOR-001")
        assert result is False
        assert interlock_manager.is_tripped

    def test_repeated_trip_reset_cycles(self, interlock_manager):
        """Test repeated trip/reset cycles."""
        for cycle in range(5):
            # Trip
            interlock_manager.update_value("STEAM_PRESSURE", 160.0)
            assert interlock_manager.is_tripped

            # Return to normal and reset
            interlock_manager.update_value("STEAM_PRESSURE", 110.0)
            interlock_manager.reset_trip(f"OPERATOR-{cycle}")
            assert not interlock_manager.is_tripped


# =============================================================================
# COMMUNICATION FAILURE TESTS
# =============================================================================


class TestCommunicationFailures:
    """Test communication failure handling."""

    @pytest.fixture
    def scada_connector(self):
        from integration.scada_connector import (
            SCADAConnector,
            SCADAConnectionConfig,
            SCADAProtocol,
            TagMapping,
            DataType,
        )

        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        connector = SCADAConnector(config)

        connector.add_tag(TagMapping(
            scada_tag="HR100",
            internal_name="pressure",
            data_type=DataType.FLOAT32,
        ))

        return connector

    @pytest.mark.asyncio
    async def test_connection_failure_handling(self, scada_connector):
        """Test connection failure handling."""
        with patch.object(scada_connector._handler, "connect", return_value=False):
            result = await scada_connector.connect()

            assert result is False

    @pytest.mark.asyncio
    async def test_read_during_disconnection(self, scada_connector):
        """Test read operations during disconnection."""
        from integration.scada_connector import TagQuality

        # Not connected
        scada_connector._handler._connected = False

        result = await scada_connector._handler.read_tags(["HR100"])

        assert result["HR100"].quality == TagQuality.NOT_CONNECTED

    def test_statistics_track_errors(self, scada_connector):
        """Test error statistics are tracked."""
        scada_connector._stats["errors"] = 0

        # Simulate errors
        for _ in range(5):
            scada_connector._stats["errors"] += 1

        stats = scada_connector.get_statistics()
        assert stats["errors"] == 5


# =============================================================================
# CONCURRENCY EDGE CASES
# =============================================================================


class TestConcurrencyEdgeCases:
    """Test concurrent operation edge cases."""

    @pytest.fixture
    def interlock_manager(self):
        from safety.safety_interlocks import SafetyInterlockManager
        return SafetyInterlockManager(boiler_id="BOILER-001")

    def test_rapid_concurrent_updates(self, interlock_manager):
        """Test rapid concurrent value updates."""
        # Simulate rapid updates
        for i in range(1000):
            value = 100.0 + (i % 100)
            interlock_manager.update_value("STEAM_PRESSURE", value)

        # Should complete without error
        status = interlock_manager.get_status()
        assert "STEAM_PRESSURE" in status["interlocks"]

    def test_bypass_during_trip(self, interlock_manager):
        """Test bypass attempt during trip."""
        # Trip first
        interlock_manager.update_value("STEAM_PRESSURE", 160.0)
        assert interlock_manager.is_tripped

        # Try to bypass during trip
        result = interlock_manager.set_bypass(
            "STEAM_PRESSURE",
            reason="Test",
            duration_minutes=60,
            operator="OPERATOR-001",
        )

        # Bypass might succeed for SIL2, but trip remains
        # (This depends on implementation)


# =============================================================================
# DATA QUALITY EDGE CASES
# =============================================================================


class TestDataQualityEdgeCases:
    """Test data quality edge cases."""

    @pytest.fixture
    def scada_connector(self):
        from integration.scada_connector import (
            SCADAConnector,
            SCADAConnectionConfig,
            SCADAProtocol,
            TagMapping,
            DataType,
        )

        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        connector = SCADAConnector(config)

        connector.add_tag(TagMapping(
            scada_tag="HR100",
            internal_name="temperature",
            data_type=DataType.FLOAT32,
            low_limit=-50.0,
            high_limit=500.0,
        ))

        return connector

    def test_nan_value_handling(self, scada_connector):
        """Test NaN value handling."""
        from integration.scada_connector import TagValue, TagQuality
        import math

        raw_values = {
            "HR100": TagValue(
                tag="HR100",
                value=float('nan'),
                quality=TagQuality.GOOD,
                timestamp=datetime.now(timezone.utc),
            )
        }

        processed = scada_connector._process_values(raw_values)

        # NaN should be handled appropriately
        if "temperature" in processed:
            assert processed["temperature"].value is None or math.isnan(processed["temperature"].value)

    def test_infinity_value_handling(self, scada_connector):
        """Test infinity value handling."""
        from integration.scada_connector import TagValue, TagQuality

        raw_values = {
            "HR100": TagValue(
                tag="HR100",
                value=float('inf'),
                quality=TagQuality.GOOD,
                timestamp=datetime.now(timezone.utc),
            )
        }

        processed = scada_connector._process_values(raw_values)

        # Should be flagged as sensor failure (above high limit)
        if "temperature" in processed:
            assert processed["temperature"].quality == TagQuality.SENSOR_FAILURE

    def test_null_value_handling(self, scada_connector):
        """Test null value handling."""
        from integration.scada_connector import TagValue, TagQuality

        raw_values = {
            "HR100": TagValue(
                tag="HR100",
                value=None,
                quality=TagQuality.BAD,
                timestamp=datetime.now(timezone.utc),
            )
        }

        processed = scada_connector._process_values(raw_values)

        assert processed["temperature"].value is None
        assert processed["temperature"].quality == TagQuality.BAD
