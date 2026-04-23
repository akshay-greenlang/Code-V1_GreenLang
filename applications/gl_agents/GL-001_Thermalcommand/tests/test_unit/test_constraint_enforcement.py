"""
Unit Tests: Constraint Enforcement

Tests all constraint enforcement logic for ThermalCommand agent including:
- Operational constraints (temperature, pressure, flow limits)
- Safety constraints (SIS interlocks, permissives)
- Equipment constraints (capacity, ramp rates)
- Process constraints (mass balance, energy balance)
- Regulatory constraints (emissions limits, efficiency minimums)

Reference: GL-001 Specification Section 11.1
Target Coverage: 85%+
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Constraint Classes (Simulated Production Code)
# =============================================================================

class ConstraintViolation(Exception):
    """Raised when a constraint is violated."""
    def __init__(self, constraint_name: str, actual_value: Any, limit: Any,
                 severity: str = "error", message: str = ""):
        self.constraint_name = constraint_name
        self.actual_value = actual_value
        self.limit = limit
        self.severity = severity
        super().__init__(f"Constraint '{constraint_name}' violated: {actual_value} vs limit {limit}. {message}")


class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SAFETY = "safety"


@dataclass
class ConstraintResult:
    """Result of constraint evaluation."""
    name: str
    passed: bool
    actual_value: Any
    limit: Any
    severity: ConstraintSeverity
    message: str = ""


class OperationalConstraints:
    """Enforces operational constraints for thermal systems."""

    # Temperature limits (Celsius)
    TEMP_MIN = 0.0
    TEMP_MAX = 1200.0
    TEMP_PROCESS_MIN = 100.0
    TEMP_PROCESS_MAX = 600.0

    # Pressure limits (bar)
    PRESSURE_MIN = 0.0
    PRESSURE_MAX = 100.0
    PRESSURE_OPERATING_MAX = 50.0

    # Flow rate limits (m3/h)
    FLOW_MIN = 0.0
    FLOW_MAX = 10000.0

    @classmethod
    def check_temperature(cls, temperature: float, context: str = "general") -> ConstraintResult:
        """Check temperature against operational limits."""
        if context == "process":
            min_limit = cls.TEMP_PROCESS_MIN
            max_limit = cls.TEMP_PROCESS_MAX
        else:
            min_limit = cls.TEMP_MIN
            max_limit = cls.TEMP_MAX

        if temperature < min_limit:
            return ConstraintResult(
                name="temperature_min",
                passed=False,
                actual_value=temperature,
                limit=min_limit,
                severity=ConstraintSeverity.ERROR,
                message=f"Temperature {temperature}C below minimum {min_limit}C"
            )

        if temperature > max_limit:
            return ConstraintResult(
                name="temperature_max",
                passed=False,
                actual_value=temperature,
                limit=max_limit,
                severity=ConstraintSeverity.CRITICAL if temperature > cls.TEMP_MAX else ConstraintSeverity.ERROR,
                message=f"Temperature {temperature}C exceeds maximum {max_limit}C"
            )

        return ConstraintResult(
            name="temperature",
            passed=True,
            actual_value=temperature,
            limit=(min_limit, max_limit),
            severity=ConstraintSeverity.WARNING
        )

    @classmethod
    def check_pressure(cls, pressure: float, is_operating: bool = True) -> ConstraintResult:
        """Check pressure against operational limits."""
        max_limit = cls.PRESSURE_OPERATING_MAX if is_operating else cls.PRESSURE_MAX

        if pressure < cls.PRESSURE_MIN:
            return ConstraintResult(
                name="pressure_min",
                passed=False,
                actual_value=pressure,
                limit=cls.PRESSURE_MIN,
                severity=ConstraintSeverity.ERROR,
                message=f"Pressure {pressure} bar below minimum"
            )

        if pressure > max_limit:
            severity = ConstraintSeverity.SAFETY if pressure > cls.PRESSURE_MAX else ConstraintSeverity.ERROR
            return ConstraintResult(
                name="pressure_max",
                passed=False,
                actual_value=pressure,
                limit=max_limit,
                severity=severity,
                message=f"Pressure {pressure} bar exceeds limit {max_limit} bar"
            )

        return ConstraintResult(
            name="pressure",
            passed=True,
            actual_value=pressure,
            limit=(cls.PRESSURE_MIN, max_limit),
            severity=ConstraintSeverity.WARNING
        )

    @classmethod
    def check_flow_rate(cls, flow_rate: float, max_capacity: Optional[float] = None) -> ConstraintResult:
        """Check flow rate against operational limits."""
        max_limit = max_capacity if max_capacity is not None else cls.FLOW_MAX

        if flow_rate < cls.FLOW_MIN:
            return ConstraintResult(
                name="flow_rate_min",
                passed=False,
                actual_value=flow_rate,
                limit=cls.FLOW_MIN,
                severity=ConstraintSeverity.ERROR,
                message=f"Flow rate {flow_rate} m3/h is negative"
            )

        if flow_rate > max_limit:
            return ConstraintResult(
                name="flow_rate_max",
                passed=False,
                actual_value=flow_rate,
                limit=max_limit,
                severity=ConstraintSeverity.ERROR,
                message=f"Flow rate {flow_rate} m3/h exceeds capacity {max_limit} m3/h"
            )

        return ConstraintResult(
            name="flow_rate",
            passed=True,
            actual_value=flow_rate,
            limit=(cls.FLOW_MIN, max_limit),
            severity=ConstraintSeverity.WARNING
        )


class SafetyConstraints:
    """Enforces safety constraints including SIS interlocks."""

    # Safety limits
    EMERGENCY_TEMP_MAX = 650.0  # Emergency shutdown temperature
    EMERGENCY_PRESSURE_MAX = 60.0  # Emergency shutdown pressure
    LOW_WATER_LEVEL = 0.2  # Minimum water level (fraction)
    HIGH_WATER_LEVEL = 0.9  # Maximum water level (fraction)

    @classmethod
    def check_emergency_temperature(cls, temperature: float) -> ConstraintResult:
        """Check if temperature exceeds emergency shutdown limit."""
        if temperature > cls.EMERGENCY_TEMP_MAX:
            return ConstraintResult(
                name="emergency_temperature",
                passed=False,
                actual_value=temperature,
                limit=cls.EMERGENCY_TEMP_MAX,
                severity=ConstraintSeverity.SAFETY,
                message=f"EMERGENCY: Temperature {temperature}C exceeds safety limit {cls.EMERGENCY_TEMP_MAX}C"
            )

        return ConstraintResult(
            name="emergency_temperature",
            passed=True,
            actual_value=temperature,
            limit=cls.EMERGENCY_TEMP_MAX,
            severity=ConstraintSeverity.WARNING
        )

    @classmethod
    def check_emergency_pressure(cls, pressure: float) -> ConstraintResult:
        """Check if pressure exceeds emergency shutdown limit."""
        if pressure > cls.EMERGENCY_PRESSURE_MAX:
            return ConstraintResult(
                name="emergency_pressure",
                passed=False,
                actual_value=pressure,
                limit=cls.EMERGENCY_PRESSURE_MAX,
                severity=ConstraintSeverity.SAFETY,
                message=f"EMERGENCY: Pressure {pressure} bar exceeds safety limit {cls.EMERGENCY_PRESSURE_MAX} bar"
            )

        return ConstraintResult(
            name="emergency_pressure",
            passed=True,
            actual_value=pressure,
            limit=cls.EMERGENCY_PRESSURE_MAX,
            severity=ConstraintSeverity.WARNING
        )

    @classmethod
    def check_water_level(cls, level: float) -> ConstraintResult:
        """Check boiler water level safety."""
        if level < cls.LOW_WATER_LEVEL:
            return ConstraintResult(
                name="low_water_level",
                passed=False,
                actual_value=level,
                limit=cls.LOW_WATER_LEVEL,
                severity=ConstraintSeverity.SAFETY,
                message=f"SAFETY: Water level {level*100:.1f}% below minimum {cls.LOW_WATER_LEVEL*100:.1f}%"
            )

        if level > cls.HIGH_WATER_LEVEL:
            return ConstraintResult(
                name="high_water_level",
                passed=False,
                actual_value=level,
                limit=cls.HIGH_WATER_LEVEL,
                severity=ConstraintSeverity.ERROR,
                message=f"Water level {level*100:.1f}% above maximum {cls.HIGH_WATER_LEVEL*100:.1f}%"
            )

        return ConstraintResult(
            name="water_level",
            passed=True,
            actual_value=level,
            limit=(cls.LOW_WATER_LEVEL, cls.HIGH_WATER_LEVEL),
            severity=ConstraintSeverity.WARNING
        )

    @classmethod
    def check_permissive(cls, permissive_name: str, conditions: Dict[str, bool]) -> ConstraintResult:
        """Check if all permissive conditions are met."""
        failed_conditions = [name for name, met in conditions.items() if not met]

        if failed_conditions:
            return ConstraintResult(
                name=f"permissive_{permissive_name}",
                passed=False,
                actual_value=conditions,
                limit="all_true",
                severity=ConstraintSeverity.SAFETY,
                message=f"Permissive '{permissive_name}' blocked by: {', '.join(failed_conditions)}"
            )

        return ConstraintResult(
            name=f"permissive_{permissive_name}",
            passed=True,
            actual_value=conditions,
            limit="all_true",
            severity=ConstraintSeverity.WARNING
        )


class EquipmentConstraints:
    """Enforces equipment-specific constraints."""

    @staticmethod
    def check_capacity(equipment_id: str, current_load: float, max_capacity: float,
                      min_load: float = 0.0) -> ConstraintResult:
        """Check if equipment is within capacity limits."""
        if current_load < min_load:
            return ConstraintResult(
                name=f"{equipment_id}_min_load",
                passed=False,
                actual_value=current_load,
                limit=min_load,
                severity=ConstraintSeverity.ERROR,
                message=f"{equipment_id} load {current_load} below minimum {min_load}"
            )

        if current_load > max_capacity:
            return ConstraintResult(
                name=f"{equipment_id}_max_capacity",
                passed=False,
                actual_value=current_load,
                limit=max_capacity,
                severity=ConstraintSeverity.ERROR,
                message=f"{equipment_id} load {current_load} exceeds capacity {max_capacity}"
            )

        return ConstraintResult(
            name=f"{equipment_id}_capacity",
            passed=True,
            actual_value=current_load,
            limit=(min_load, max_capacity),
            severity=ConstraintSeverity.WARNING
        )

    @staticmethod
    def check_ramp_rate(equipment_id: str, current_value: float, previous_value: float,
                       max_ramp_rate: float, time_delta: float) -> ConstraintResult:
        """Check if equipment change rate is within limits."""
        if time_delta <= 0:
            return ConstraintResult(
                name=f"{equipment_id}_ramp_rate",
                passed=True,
                actual_value=0,
                limit=max_ramp_rate,
                severity=ConstraintSeverity.WARNING
            )

        actual_rate = abs(current_value - previous_value) / time_delta

        if actual_rate > max_ramp_rate:
            return ConstraintResult(
                name=f"{equipment_id}_ramp_rate",
                passed=False,
                actual_value=actual_rate,
                limit=max_ramp_rate,
                severity=ConstraintSeverity.ERROR,
                message=f"{equipment_id} ramp rate {actual_rate:.2f}/min exceeds limit {max_ramp_rate}/min"
            )

        return ConstraintResult(
            name=f"{equipment_id}_ramp_rate",
            passed=True,
            actual_value=actual_rate,
            limit=max_ramp_rate,
            severity=ConstraintSeverity.WARNING
        )


class ProcessConstraints:
    """Enforces process constraints like mass and energy balance."""

    BALANCE_TOLERANCE = 0.05  # 5% tolerance for balance checks

    @classmethod
    def check_mass_balance(cls, inputs: List[float], outputs: List[float],
                          accumulation: float = 0.0) -> ConstraintResult:
        """Check mass balance: inputs = outputs + accumulation."""
        total_in = sum(inputs)
        total_out = sum(outputs)
        imbalance = total_in - total_out - accumulation

        # Calculate relative error
        if total_in > 0:
            relative_error = abs(imbalance) / total_in
        else:
            relative_error = 0 if imbalance == 0 else float('inf')

        if relative_error > cls.BALANCE_TOLERANCE:
            return ConstraintResult(
                name="mass_balance",
                passed=False,
                actual_value=relative_error,
                limit=cls.BALANCE_TOLERANCE,
                severity=ConstraintSeverity.ERROR,
                message=f"Mass imbalance: {relative_error*100:.1f}% (limit: {cls.BALANCE_TOLERANCE*100}%)"
            )

        return ConstraintResult(
            name="mass_balance",
            passed=True,
            actual_value=relative_error,
            limit=cls.BALANCE_TOLERANCE,
            severity=ConstraintSeverity.WARNING
        )

    @classmethod
    def check_energy_balance(cls, energy_in: float, energy_out: float,
                            losses: float = 0.0) -> ConstraintResult:
        """Check energy balance: energy_in = energy_out + losses."""
        imbalance = energy_in - energy_out - losses

        if energy_in > 0:
            relative_error = abs(imbalance) / energy_in
        else:
            relative_error = 0 if imbalance == 0 else float('inf')

        if relative_error > cls.BALANCE_TOLERANCE:
            return ConstraintResult(
                name="energy_balance",
                passed=False,
                actual_value=relative_error,
                limit=cls.BALANCE_TOLERANCE,
                severity=ConstraintSeverity.ERROR,
                message=f"Energy imbalance: {relative_error*100:.1f}%"
            )

        return ConstraintResult(
            name="energy_balance",
            passed=True,
            actual_value=relative_error,
            limit=cls.BALANCE_TOLERANCE,
            severity=ConstraintSeverity.WARNING
        )


class RegulatoryConstraints:
    """Enforces regulatory constraints like emissions and efficiency."""

    # Emissions limits (kg CO2 per MWh)
    CO2_LIMIT = 400.0
    NOX_LIMIT = 0.5  # kg/MWh
    SO2_LIMIT = 0.3  # kg/MWh

    # Efficiency minimums
    MIN_BOILER_EFFICIENCY = 0.80
    MIN_SYSTEM_EFFICIENCY = 0.75

    @classmethod
    def check_co2_emissions(cls, co2_rate: float) -> ConstraintResult:
        """Check CO2 emissions against regulatory limit."""
        if co2_rate > cls.CO2_LIMIT:
            return ConstraintResult(
                name="co2_emissions",
                passed=False,
                actual_value=co2_rate,
                limit=cls.CO2_LIMIT,
                severity=ConstraintSeverity.ERROR,
                message=f"CO2 emissions {co2_rate} kg/MWh exceed limit {cls.CO2_LIMIT} kg/MWh"
            )

        return ConstraintResult(
            name="co2_emissions",
            passed=True,
            actual_value=co2_rate,
            limit=cls.CO2_LIMIT,
            severity=ConstraintSeverity.WARNING
        )

    @classmethod
    def check_nox_emissions(cls, nox_rate: float) -> ConstraintResult:
        """Check NOx emissions against regulatory limit."""
        if nox_rate > cls.NOX_LIMIT:
            return ConstraintResult(
                name="nox_emissions",
                passed=False,
                actual_value=nox_rate,
                limit=cls.NOX_LIMIT,
                severity=ConstraintSeverity.ERROR,
                message=f"NOx emissions {nox_rate} kg/MWh exceed limit {cls.NOX_LIMIT} kg/MWh"
            )

        return ConstraintResult(
            name="nox_emissions",
            passed=True,
            actual_value=nox_rate,
            limit=cls.NOX_LIMIT,
            severity=ConstraintSeverity.WARNING
        )

    @classmethod
    def check_efficiency(cls, efficiency: float, equipment_type: str = "boiler") -> ConstraintResult:
        """Check efficiency against minimum requirements."""
        if equipment_type == "boiler":
            min_efficiency = cls.MIN_BOILER_EFFICIENCY
        else:
            min_efficiency = cls.MIN_SYSTEM_EFFICIENCY

        if efficiency < min_efficiency:
            return ConstraintResult(
                name=f"{equipment_type}_efficiency",
                passed=False,
                actual_value=efficiency,
                limit=min_efficiency,
                severity=ConstraintSeverity.ERROR,
                message=f"{equipment_type} efficiency {efficiency*100:.1f}% below minimum {min_efficiency*100:.1f}%"
            )

        return ConstraintResult(
            name=f"{equipment_type}_efficiency",
            passed=True,
            actual_value=efficiency,
            limit=min_efficiency,
            severity=ConstraintSeverity.WARNING
        )


# =============================================================================
# Test Classes
# =============================================================================

class TestOperationalConstraints:
    """Test suite for operational constraints."""

    @pytest.mark.parametrize("temperature,expected_pass", [
        (0, True),
        (500, True),
        (1200, True),
        (-10, False),
        (1300, False),
    ])
    def test_temperature_general_limits(self, temperature, expected_pass):
        """Test general temperature limits."""
        result = OperationalConstraints.check_temperature(temperature)
        assert result.passed == expected_pass

    @pytest.mark.parametrize("temperature,expected_pass", [
        (100, True),
        (350, True),
        (600, True),
        (50, False),
        (700, False),
    ])
    def test_temperature_process_limits(self, temperature, expected_pass):
        """Test process temperature limits."""
        result = OperationalConstraints.check_temperature(temperature, context="process")
        assert result.passed == expected_pass

    def test_temperature_below_min_severity(self):
        """Test that below-min temperature has ERROR severity."""
        result = OperationalConstraints.check_temperature(-10)
        assert result.severity == ConstraintSeverity.ERROR

    def test_temperature_above_max_severity(self):
        """Test that above-max temperature has CRITICAL severity."""
        result = OperationalConstraints.check_temperature(1300)
        assert result.severity == ConstraintSeverity.CRITICAL

    @pytest.mark.parametrize("pressure,is_operating,expected_pass", [
        (0, True, True),
        (25, True, True),
        (50, True, True),
        (51, True, False),  # Above operating limit
        (75, False, True),  # OK for non-operating
        (101, False, False),  # Above absolute max
        (-5, True, False),  # Negative
    ])
    def test_pressure_limits(self, pressure, is_operating, expected_pass):
        """Test pressure limits."""
        result = OperationalConstraints.check_pressure(pressure, is_operating)
        assert result.passed == expected_pass

    def test_pressure_safety_severity(self):
        """Test that pressure above absolute max has SAFETY severity."""
        result = OperationalConstraints.check_pressure(150, is_operating=False)
        assert result.severity == ConstraintSeverity.SAFETY

    @pytest.mark.parametrize("flow_rate,max_capacity,expected_pass", [
        (0, None, True),
        (500, None, True),
        (10000, None, True),
        (10001, None, False),  # Above default max
        (-10, None, False),  # Negative
        (500, 400, False),  # Above custom capacity
        (300, 400, True),
    ])
    def test_flow_rate_limits(self, flow_rate, max_capacity, expected_pass):
        """Test flow rate limits."""
        result = OperationalConstraints.check_flow_rate(flow_rate, max_capacity)
        assert result.passed == expected_pass


class TestSafetyConstraints:
    """Test suite for safety constraints."""

    @pytest.mark.parametrize("temperature,expected_pass", [
        (400, True),
        (650, True),  # At limit
        (651, False),  # Above limit
        (700, False),
    ])
    def test_emergency_temperature(self, temperature, expected_pass):
        """Test emergency temperature limits."""
        result = SafetyConstraints.check_emergency_temperature(temperature)
        assert result.passed == expected_pass

    def test_emergency_temperature_severity(self):
        """Test that emergency temperature violation has SAFETY severity."""
        result = SafetyConstraints.check_emergency_temperature(700)
        assert result.severity == ConstraintSeverity.SAFETY
        assert "EMERGENCY" in result.message

    @pytest.mark.parametrize("pressure,expected_pass", [
        (30, True),
        (60, True),  # At limit
        (61, False),  # Above limit
        (100, False),
    ])
    def test_emergency_pressure(self, pressure, expected_pass):
        """Test emergency pressure limits."""
        result = SafetyConstraints.check_emergency_pressure(pressure)
        assert result.passed == expected_pass

    @pytest.mark.parametrize("level,expected_pass", [
        (0.5, True),  # Normal
        (0.2, True),  # At low limit
        (0.9, True),  # At high limit
        (0.15, False),  # Below low limit
        (0.95, False),  # Above high limit
    ])
    def test_water_level(self, level, expected_pass):
        """Test water level safety limits."""
        result = SafetyConstraints.check_water_level(level)
        assert result.passed == expected_pass

    def test_low_water_level_severity(self):
        """Test that low water level has SAFETY severity."""
        result = SafetyConstraints.check_water_level(0.1)
        assert result.severity == ConstraintSeverity.SAFETY
        assert "SAFETY" in result.message

    def test_permissive_all_conditions_met(self):
        """Test permissive passes when all conditions met."""
        conditions = {
            "fuel_available": True,
            "air_flow_ok": True,
            "safety_valves_ok": True
        }
        result = SafetyConstraints.check_permissive("boiler_start", conditions)
        assert result.passed == True

    def test_permissive_some_conditions_failed(self):
        """Test permissive fails when some conditions not met."""
        conditions = {
            "fuel_available": True,
            "air_flow_ok": False,
            "safety_valves_ok": True
        }
        result = SafetyConstraints.check_permissive("boiler_start", conditions)
        assert result.passed == False
        assert "air_flow_ok" in result.message

    def test_permissive_multiple_failures_reported(self):
        """Test that multiple failed conditions are reported."""
        conditions = {
            "condition_a": False,
            "condition_b": False,
            "condition_c": True
        }
        result = SafetyConstraints.check_permissive("test", conditions)
        assert "condition_a" in result.message
        assert "condition_b" in result.message


class TestEquipmentConstraints:
    """Test suite for equipment constraints."""

    @pytest.mark.parametrize("current_load,max_capacity,min_load,expected_pass", [
        (50, 100, 10, True),  # Normal operation
        (100, 100, 10, True),  # At max
        (10, 100, 10, True),  # At min
        (5, 100, 10, False),  # Below min
        (110, 100, 10, False),  # Above max
        (0, 100, 0, True),  # Zero with zero min
    ])
    def test_capacity_limits(self, current_load, max_capacity, min_load, expected_pass):
        """Test equipment capacity limits."""
        result = EquipmentConstraints.check_capacity(
            "BOILER_001", current_load, max_capacity, min_load
        )
        assert result.passed == expected_pass

    def test_capacity_equipment_id_in_name(self):
        """Test that equipment ID is in constraint name."""
        result = EquipmentConstraints.check_capacity("PUMP_001", 50, 100)
        assert "PUMP_001" in result.name

    @pytest.mark.parametrize("current,previous,max_rate,time_delta,expected_pass", [
        (50, 40, 20, 1, True),  # 10/min, within limit
        (50, 40, 5, 1, False),  # 10/min, exceeds limit
        (50, 50, 10, 1, True),  # No change
        (100, 50, 100, 1, True),  # At limit
        (100, 50, 49, 1, False),  # Just over limit
    ])
    def test_ramp_rate_limits(self, current, previous, max_rate, time_delta, expected_pass):
        """Test equipment ramp rate limits."""
        result = EquipmentConstraints.check_ramp_rate(
            "BOILER_001", current, previous, max_rate, time_delta
        )
        assert result.passed == expected_pass

    def test_ramp_rate_zero_time_delta(self):
        """Test ramp rate with zero time delta."""
        result = EquipmentConstraints.check_ramp_rate(
            "BOILER_001", 100, 50, 10, 0
        )
        assert result.passed == True  # Should pass (no time elapsed)


class TestProcessConstraints:
    """Test suite for process constraints."""

    @pytest.mark.parametrize("inputs,outputs,accumulation,expected_pass", [
        ([100], [100], 0, True),  # Perfect balance
        ([100, 50], [150], 0, True),  # Multiple inputs
        ([100], [95], 5, True),  # With accumulation
        ([100], [90], 0, False),  # 10% imbalance
        ([100], [80], 0, False),  # 20% imbalance
        ([0], [0], 0, True),  # Zero case
    ])
    def test_mass_balance(self, inputs, outputs, accumulation, expected_pass):
        """Test mass balance constraints."""
        result = ProcessConstraints.check_mass_balance(inputs, outputs, accumulation)
        assert result.passed == expected_pass

    @pytest.mark.parametrize("energy_in,energy_out,losses,expected_pass", [
        (1000, 1000, 0, True),  # Perfect balance
        (1000, 850, 150, True),  # With losses
        (1000, 800, 150, False),  # 5% imbalance (50/1000)
        (1000, 700, 100, False),  # 20% imbalance
        (0, 0, 0, True),  # Zero case
    ])
    def test_energy_balance(self, energy_in, energy_out, losses, expected_pass):
        """Test energy balance constraints."""
        result = ProcessConstraints.check_energy_balance(energy_in, energy_out, losses)
        assert result.passed == expected_pass

    def test_mass_balance_within_tolerance(self):
        """Test mass balance within 5% tolerance passes."""
        # 4% imbalance should pass
        result = ProcessConstraints.check_mass_balance([100], [96], 0)
        assert result.passed == True

    def test_mass_balance_exceeds_tolerance(self):
        """Test mass balance exceeding 5% tolerance fails."""
        # 6% imbalance should fail
        result = ProcessConstraints.check_mass_balance([100], [94], 0)
        assert result.passed == False


class TestRegulatoryConstraints:
    """Test suite for regulatory constraints."""

    @pytest.mark.parametrize("co2_rate,expected_pass", [
        (200, True),
        (400, True),  # At limit
        (401, False),  # Above limit
        (500, False),
    ])
    def test_co2_emissions(self, co2_rate, expected_pass):
        """Test CO2 emissions limits."""
        result = RegulatoryConstraints.check_co2_emissions(co2_rate)
        assert result.passed == expected_pass

    @pytest.mark.parametrize("nox_rate,expected_pass", [
        (0.2, True),
        (0.5, True),  # At limit
        (0.6, False),  # Above limit
    ])
    def test_nox_emissions(self, nox_rate, expected_pass):
        """Test NOx emissions limits."""
        result = RegulatoryConstraints.check_nox_emissions(nox_rate)
        assert result.passed == expected_pass

    @pytest.mark.parametrize("efficiency,equipment_type,expected_pass", [
        (0.85, "boiler", True),
        (0.80, "boiler", True),  # At limit
        (0.79, "boiler", False),  # Below limit
        (0.80, "system", True),
        (0.75, "system", True),  # At limit
        (0.74, "system", False),  # Below limit
    ])
    def test_efficiency_limits(self, efficiency, equipment_type, expected_pass):
        """Test efficiency limits by equipment type."""
        result = RegulatoryConstraints.check_efficiency(efficiency, equipment_type)
        assert result.passed == expected_pass


class TestConstraintSeverityLevels:
    """Test that correct severity levels are assigned."""

    def test_warning_severity_for_passing_constraints(self):
        """Test that passing constraints have WARNING severity."""
        result = OperationalConstraints.check_temperature(500)
        assert result.passed == True
        assert result.severity == ConstraintSeverity.WARNING

    def test_error_severity_for_operational_violations(self):
        """Test that operational violations have ERROR severity."""
        result = OperationalConstraints.check_temperature(-10)
        assert result.passed == False
        assert result.severity == ConstraintSeverity.ERROR

    def test_safety_severity_for_safety_violations(self):
        """Test that safety violations have SAFETY severity."""
        result = SafetyConstraints.check_emergency_temperature(700)
        assert result.passed == False
        assert result.severity == ConstraintSeverity.SAFETY

    def test_critical_severity_for_extreme_violations(self):
        """Test that extreme violations have CRITICAL severity."""
        result = OperationalConstraints.check_temperature(1500)
        assert result.passed == False
        assert result.severity == ConstraintSeverity.CRITICAL


class TestConstraintMessages:
    """Test that constraint messages are informative."""

    def test_message_contains_actual_value(self):
        """Test that message contains actual value."""
        result = OperationalConstraints.check_temperature(1300)
        assert "1300" in result.message

    def test_message_contains_limit(self):
        """Test that message contains limit."""
        result = SafetyConstraints.check_emergency_pressure(100)
        assert "60" in result.message

    def test_message_contains_equipment_id(self):
        """Test that message contains equipment ID."""
        result = EquipmentConstraints.check_capacity("BOILER_001", 150, 100)
        assert "BOILER_001" in result.message
