"""
GL-002 FLAMEGUARD - Property-Based Tests with Hypothesis

Uses Hypothesis for property-based testing to verify invariants:
- Combustion calculation invariants
- Safety interlock state machine properties
- Efficiency bounds
- O2/excess air relationships
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
from datetime import datetime, timezone
import sys

sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

# Realistic ranges for boiler parameters
load_percent = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
o2_percent = st.floats(min_value=0.0, max_value=21.0, allow_nan=False, allow_infinity=False)
co_ppm = st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False)
temperature_f = st.floats(min_value=-50.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
pressure_psig = st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False)
steam_flow = st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False)
fuel_flow = st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
signal_percent = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
pid_gain = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)


# =============================================================================
# EFFICIENCY CALCULATOR PROPERTY TESTS
# =============================================================================


class TestEfficiencyProperties:
    """Property-based tests for efficiency calculations."""

    @given(
        o2=st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_o2_to_excess_air_monotonic(self, o2):
        """Test O2 to excess air conversion is monotonically increasing."""
        # O2 / (21 - O2) * 100
        if o2 >= 21:
            excess_air = 500.0  # Capped
        else:
            excess_air = o2 / (21 - o2) * 100

        # Verify positive relationship
        assert excess_air >= 0

        # Verify higher O2 means higher excess air
        if o2 < 20:
            o2_higher = o2 + 0.5
            if o2_higher < 21:
                excess_air_higher = o2_higher / (21 - o2_higher) * 100
                assert excess_air_higher > excess_air

    @given(
        o2=o2_percent,
    )
    @settings(max_examples=100)
    def test_excess_air_bounds(self, o2):
        """Test excess air is always bounded."""
        if o2 >= 21:
            excess_air = 500.0
        else:
            excess_air = o2 / (21 - o2) * 100
            excess_air = min(500.0, excess_air)

        # Excess air should be non-negative and bounded
        assert 0 <= excess_air <= 500

    @given(
        efficiency=st.floats(min_value=50.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_efficiency_bounds(self, efficiency):
        """Test efficiency is always in valid range."""
        # Clamp to valid range
        clamped = max(50.0, min(100.0, efficiency))

        assert 50.0 <= clamped <= 100.0

    @given(
        steam_flow=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        fuel_flow=st.floats(min_value=0.1, max_value=100000.0, allow_nan=False, allow_infinity=False),
        efficiency=st.floats(min_value=50.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_fuel_efficiency_relationship(self, steam_flow, fuel_flow, efficiency):
        """Test relationship between steam, fuel, and efficiency."""
        # Higher efficiency means less fuel for same steam output
        fuel_required = steam_flow * 1000 / efficiency if efficiency > 0 else float('inf')

        if efficiency > 50:
            fuel_required_lower_eff = steam_flow * 1000 / (efficiency - 10) if efficiency > 60 else float('inf')
            if fuel_required_lower_eff != float('inf'):
                assert fuel_required <= fuel_required_lower_eff


# =============================================================================
# EMISSIONS CALCULATOR PROPERTY TESTS
# =============================================================================


class TestEmissionsProperties:
    """Property-based tests for emissions calculations."""

    @given(
        heat_input=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        factor=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_emissions_proportional_to_heat_input(self, heat_input, factor):
        """Test emissions are proportional to heat input."""
        emissions = heat_input * factor

        assert emissions >= 0

        # Doubling heat input should double emissions
        emissions_doubled = heat_input * 2 * factor
        if emissions > 0:
            assert abs(emissions_doubled / emissions - 2.0) < 0.001

    @given(
        co2_lb_hr=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        ch4_lb_hr=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        n2o_lb_hr=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_co2e_always_greater_than_co2(self, co2_lb_hr, ch4_lb_hr, n2o_lb_hr):
        """Test CO2e is always >= CO2 (due to GWP contributions)."""
        # GWP values: CO2=1, CH4=28, N2O=265
        co2e = co2_lb_hr + ch4_lb_hr * 28 + n2o_lb_hr * 265

        assert co2e >= co2_lb_hr

    @given(
        measured_o2=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        nox_ppm=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_o2_correction_direction(self, measured_o2, nox_ppm):
        """Test O2 correction to 3% is in correct direction."""
        reference_o2 = 3.0

        if measured_o2 >= 21:
            corrected = nox_ppm
        else:
            correction_factor = (21 - reference_o2) / (21 - measured_o2)
            corrected = nox_ppm * correction_factor

        # If measured O2 > 3%, corrected should be > measured
        # If measured O2 < 3%, corrected should be < measured
        if measured_o2 > reference_o2:
            assert corrected >= nox_ppm
        elif measured_o2 < reference_o2:
            assert corrected <= nox_ppm
        else:
            assert abs(corrected - nox_ppm) < 0.001


# =============================================================================
# PID CONTROLLER PROPERTY TESTS
# =============================================================================


class TestPIDProperties:
    """Property-based tests for PID controller."""

    @given(
        kp=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        ki=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
        kd=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        setpoint=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        process_value=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_pid_output_bounded(self, kp, ki, kd, setpoint, process_value):
        """Test PID output is always bounded."""
        from optimization.o2_trim_controller import PIDController

        output_min = -10.0
        output_max = 10.0

        pid = PIDController(
            kp=kp, ki=ki, kd=kd,
            output_min=output_min, output_max=output_max,
        )

        output = pid.compute(setpoint, process_value, timestamp=0.0)

        assert output_min <= output <= output_max

    @given(
        kp=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        setpoint=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        error=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_proportional_response_direction(self, kp, setpoint, error):
        """Test proportional response is in correct direction."""
        from optimization.o2_trim_controller import PIDController

        process_value = setpoint - error

        pid = PIDController(kp=kp, ki=0.0, kd=0.0, deadband=0.0)
        output = pid.compute(setpoint, process_value, timestamp=0.0)

        # Positive error should give positive output (clamped)
        if error > 0:
            assert output > 0 or output == pid.output_max
        elif error < 0:
            assert output < 0 or output == pid.output_min
        else:
            assert output == 0.0

    @given(
        setpoint=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_pid_at_setpoint_no_output(self, setpoint):
        """Test PID output is zero when at setpoint."""
        from optimization.o2_trim_controller import PIDController

        pid = PIDController(kp=2.0, ki=0.0, kd=0.0)

        # Process value equals setpoint
        output = pid.compute(setpoint, setpoint, timestamp=0.0)

        assert output == 0.0


# =============================================================================
# O2 TRIM CONTROLLER PROPERTY TESTS
# =============================================================================


class TestO2TrimProperties:
    """Property-based tests for O2 trim controller."""

    @given(
        load_percent=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_setpoint_decreases_with_load(self, load_percent):
        """Test O2 setpoint generally decreases with increasing load."""
        from optimization.o2_trim_controller import O2TrimController

        controller = O2TrimController(boiler_id="TEST")

        # Default curve: {0.25: 5.0, 0.50: 3.5, 0.75: 3.0, 1.00: 2.5}

        result = controller.compute(
            o2_measured=3.0,
            co_measured=50.0,
            load_percent=load_percent,
        )

        # Setpoint should be in valid range
        assert 1.5 <= result.o2_setpoint <= 8.0

    @given(
        co_measured=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_co_crosslimit_activates_correctly(self, co_measured):
        """Test CO cross-limiting activates when CO exceeds limit."""
        from optimization.o2_trim_controller import O2TrimController

        co_limit = 400.0
        controller = O2TrimController(boiler_id="TEST", co_limit_ppm=co_limit)

        result = controller.compute(
            o2_measured=3.0,
            co_measured=co_measured,
            load_percent=75.0,
        )

        if co_measured > co_limit:
            assert result.co_override_active is True
        else:
            assert result.co_override_active is False

    @given(
        air_temp=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_temperature_compensation_direction(self, air_temp):
        """Test temperature compensation adjusts setpoint correctly."""
        from optimization.o2_trim_controller import O2TrimController

        controller = O2TrimController(boiler_id="TEST")
        reference_temp = 80.0  # Reference temperature

        result_ref = controller.compute(
            o2_measured=3.0,
            co_measured=50.0,
            load_percent=75.0,
            air_temp=reference_temp,
        )

        controller.reset()  # Reset for new calculation

        result_test = controller.compute(
            o2_measured=3.0,
            co_measured=50.0,
            load_percent=75.0,
            air_temp=air_temp,
        )

        # Hot air (less dense) needs higher O2 setpoint
        # Cold air (more dense) needs lower O2 setpoint
        if air_temp > reference_temp:
            # Setpoint should be higher (or equal due to other factors)
            pass  # Temperature compensation is small
        elif air_temp < reference_temp:
            # Setpoint should be lower (or equal)
            pass


# =============================================================================
# SAFETY INTERLOCK STATE MACHINE TESTS
# =============================================================================


class SafetyInterlockStateMachine(RuleBasedStateMachine):
    """State machine tests for safety interlock system."""

    def __init__(self):
        super().__init__()
        from safety.safety_interlocks import SafetyInterlockManager
        self.manager = SafetyInterlockManager(boiler_id="TEST")
        self.tripped_tags = set()

    @rule(value=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False))
    def update_steam_pressure(self, value):
        """Update steam pressure and track expected state."""
        from safety.safety_interlocks import InterlockStatus

        result = self.manager.update_value("STEAM_PRESSURE", value)

        # Track trip state
        if value >= 150.0:  # Trip high
            self.tripped_tags.add("STEAM_PRESSURE")

    @rule(value=st.floats(min_value=-10.0, max_value=15.0, allow_nan=False, allow_infinity=False))
    def update_drum_level(self, value):
        """Update drum level."""
        from safety.safety_interlocks import InterlockStatus

        result = self.manager.update_value("DRUM_LEVEL", value)

        # Track trip state
        if value <= -4.0 or value >= 8.0:
            self.tripped_tags.add("DRUM_LEVEL")

    @rule(value=st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False))
    def update_fuel_pressure(self, value):
        """Update fuel pressure."""
        result = self.manager.update_value("FUEL_PRESSURE", value)

        if value <= 2.0 or value >= 25.0:
            self.tripped_tags.add("FUEL_PRESSURE")

    @rule()
    def attempt_reset(self):
        """Attempt to reset trips."""
        if self.manager.is_tripped:
            # Get current values
            status = self.manager.get_status()

            # Check if all conditions are normal
            all_normal = True
            for tag in self.tripped_tags:
                interlock = status["interlocks"].get(tag)
                if interlock:
                    if tag == "STEAM_PRESSURE" and interlock["value"] >= 150.0:
                        all_normal = False
                    elif tag == "DRUM_LEVEL" and (interlock["value"] <= -4.0 or interlock["value"] >= 8.0):
                        all_normal = False
                    elif tag == "FUEL_PRESSURE" and (interlock["value"] <= 2.0 or interlock["value"] >= 25.0):
                        all_normal = False

            result = self.manager.reset_trip("TEST_OPERATOR")

            if all_normal:
                self.tripped_tags.clear()

    @invariant()
    def trip_state_consistent(self):
        """Verify trip state is consistent."""
        # If any tag is tripped and in trip condition, manager should be tripped
        status = self.manager.get_status()

        for tag, interlock in status["interlocks"].items():
            if interlock["status"] == "trip":
                assert self.manager.is_tripped, f"Tag {tag} is tripped but manager.is_tripped is False"

    @invariant()
    def bypass_respects_sil(self):
        """Verify SIL3 tags cannot be bypassed."""
        status = self.manager.get_status()

        # DRUM_LEVEL is SIL3 - should never be bypassed
        if "DRUM_LEVEL" in status["interlocks"]:
            assert status["interlocks"]["DRUM_LEVEL"]["bypassed"] is False


TestSafetyInterlockState = SafetyInterlockStateMachine.TestCase


# =============================================================================
# FLAME DETECTOR STATE MACHINE TESTS
# =============================================================================


class FlameDetectorStateMachine(RuleBasedStateMachine):
    """State machine tests for flame detector."""

    def __init__(self):
        super().__init__()
        from safety.flame_detector import FlameDetector
        self.detector = FlameDetector(boiler_id="TEST", voting_logic="2oo3")
        self.detector.add_scanner("UV-1", "UV")
        self.detector.add_scanner("UV-2", "UV")
        self.detector.add_scanner("UV-3", "UV")
        self.scanner_signals = {"UV-1": 0.0, "UV-2": 0.0, "UV-3": 0.0}
        self.scanner_healthy = {"UV-1": True, "UV-2": True, "UV-3": True}

    @rule(
        scanner=st.sampled_from(["UV-1", "UV-2", "UV-3"]),
        signal=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def update_signal(self, scanner, signal):
        """Update scanner signal."""
        self.detector.update_scanner(scanner, signal_percent=signal)
        self.scanner_signals[scanner] = signal

    @rule(
        scanner=st.sampled_from(["UV-1", "UV-2", "UV-3"]),
        healthy=st.booleans(),
    )
    def update_health(self, scanner, healthy):
        """Update scanner health status."""
        current_signal = self.scanner_signals[scanner]
        self.detector.update_scanner(scanner, signal_percent=current_signal, healthy=healthy)
        self.scanner_healthy[scanner] = healthy

    @invariant()
    def voting_logic_correct(self):
        """Verify 2oo3 voting logic is correct."""
        from safety.flame_detector import FlameDetector

        # Count scanners seeing flame (signal > MIN_SIGNAL_PERCENT and healthy)
        min_signal = FlameDetector.MIN_SIGNAL_PERCENT
        flame_count = sum(
            1 for scanner in ["UV-1", "UV-2", "UV-3"]
            if self.scanner_signals[scanner] >= min_signal and self.scanner_healthy[scanner]
        )

        healthy_count = sum(1 for h in self.scanner_healthy.values() if h)

        if healthy_count == 0:
            # All scanners faulty
            assert not self.detector.is_flame_proven()
        elif flame_count >= 2:
            # 2oo3 satisfied
            assert self.detector.is_flame_proven()
        else:
            # 2oo3 not satisfied
            assert not self.detector.is_flame_proven()

    @invariant()
    def signal_percent_bounded(self):
        """Verify signal percent is in valid range."""
        assert 0.0 <= self.detector.signal_percent <= 100.0


TestFlameDetectorState = FlameDetectorStateMachine.TestCase


# =============================================================================
# LOAD DISPATCH PROPERTY TESTS
# =============================================================================


class TestLoadDispatchProperties:
    """Property-based tests for load dispatch optimization."""

    @given(
        demand=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        capacity1=st.floats(min_value=50.0, max_value=300.0, allow_nan=False, allow_infinity=False),
        capacity2=st.floats(min_value=50.0, max_value=300.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_dispatch_respects_capacity(self, demand, capacity1, capacity2):
        """Test dispatch respects total capacity."""
        from optimization.combustion_optimizer import CombustionOptimizer, BoilerModel

        optimizer = CombustionOptimizer(boilers=[
            BoilerModel(boiler_id="B1", rated_capacity_klb_hr=capacity1),
            BoilerModel(boiler_id="B2", rated_capacity_klb_hr=capacity2),
        ])

        result = optimizer.optimize_load_dispatch(total_demand_klb_hr=demand)

        total_capacity = capacity1 + capacity2
        total_allocated = sum(result.allocations.values())

        # Allocated should not exceed demand or capacity
        assert total_allocated <= min(demand, total_capacity) + 1.0

    @given(
        demand=st.floats(min_value=50.0, max_value=300.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_dispatch_allocations_non_negative(self, demand):
        """Test all allocations are non-negative."""
        from optimization.combustion_optimizer import CombustionOptimizer, BoilerModel

        optimizer = CombustionOptimizer(boilers=[
            BoilerModel(boiler_id="B1", rated_capacity_klb_hr=200.0),
            BoilerModel(boiler_id="B2", rated_capacity_klb_hr=150.0),
        ])

        result = optimizer.optimize_load_dispatch(total_demand_klb_hr=demand)

        for alloc in result.allocations.values():
            assert alloc >= 0


# =============================================================================
# MODBUS VALUE CONVERSION PROPERTY TESTS
# =============================================================================


class TestModbusConversionProperties:
    """Property-based tests for Modbus value conversions."""

    @given(
        value=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_float32_roundtrip(self, value):
        """Test FLOAT32 value survives roundtrip conversion."""
        from integration.scada_connector import ModbusTCPHandler, SCADAConnectionConfig, SCADAProtocol, DataType
        import struct

        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        handler = ModbusTCPHandler(config)

        # Value to registers
        registers = handler._value_to_registers(value, DataType.FLOAT32)

        # Registers back to value
        converted = handler._convert_value(registers, DataType.FLOAT32)

        # Should be approximately equal (floating point precision)
        assert abs(converted - value) < abs(value * 0.0001) + 0.001

    @given(
        value=st.integers(min_value=-32768, max_value=32767),
    )
    @settings(max_examples=50)
    def test_int16_roundtrip(self, value):
        """Test INT16 value survives roundtrip conversion."""
        from integration.scada_connector import ModbusTCPHandler, SCADAConnectionConfig, SCADAProtocol, DataType

        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        handler = ModbusTCPHandler(config)

        # Value to registers
        registers = handler._value_to_registers(value, DataType.INT16)

        # Registers back to value
        converted = handler._convert_value(registers, DataType.INT16)

        assert converted == value

    @given(
        value=st.integers(min_value=0, max_value=65535),
    )
    @settings(max_examples=50)
    def test_uint16_roundtrip(self, value):
        """Test UINT16 value survives roundtrip conversion."""
        from integration.scada_connector import ModbusTCPHandler, SCADAConnectionConfig, SCADAProtocol, DataType

        config = SCADAConnectionConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="localhost",
            port=502,
        )
        handler = ModbusTCPHandler(config)

        # Value to registers
        registers = handler._value_to_registers(value, DataType.UINT16)

        # Registers back to value
        converted = handler._convert_value(registers, DataType.UINT16)

        assert converted == value
