"""
GreenLang Input Validation Engine
==================================

Comprehensive validation for calculation inputs ensuring:
- Range checking (physical limits)
- Unit validation (correct units for parameters)
- Physical feasibility (thermodynamic constraints)
- Constraint verification (cross-parameter validation)

Copyright (c) 2024 GreenLang. All rights reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .calculation_engine import (
    FormulaDefinition,
    ParameterDefinition,
    UnitCategory,
    UnitConverter,
)


# =============================================================================
# Validation Types
# =============================================================================

class ValidationSeverity(Enum):
    """Severity level for validation messages."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationErrorCode(Enum):
    """Standardized validation error codes."""
    # Range errors
    BELOW_MINIMUM = "RANGE_001"
    ABOVE_MAXIMUM = "RANGE_002"
    OUT_OF_VALID_RANGE = "RANGE_003"
    NEGATIVE_VALUE = "RANGE_004"
    ZERO_VALUE = "RANGE_005"

    # Type errors
    INVALID_TYPE = "TYPE_001"
    NAN_VALUE = "TYPE_002"
    INF_VALUE = "TYPE_003"
    MISSING_VALUE = "TYPE_004"

    # Unit errors
    UNKNOWN_UNIT = "UNIT_001"
    INCOMPATIBLE_UNITS = "UNIT_002"
    UNIT_MISMATCH = "UNIT_003"

    # Physical feasibility errors
    THERMODYNAMIC_VIOLATION = "PHYS_001"
    MASS_BALANCE_VIOLATION = "PHYS_002"
    ENERGY_BALANCE_VIOLATION = "PHYS_003"
    PHASE_CONSTRAINT_VIOLATION = "PHYS_004"
    SUPERSONIC_FLOW = "PHYS_005"
    NEGATIVE_ENTROPY = "PHYS_006"
    IMPOSSIBLE_EFFICIENCY = "PHYS_007"

    # Constraint errors
    CONSTRAINT_VIOLATION = "CONST_001"
    DEPENDENCY_MISSING = "CONST_002"
    MUTUAL_EXCLUSION = "CONST_003"
    SUM_CONSTRAINT = "CONST_004"


@dataclass
class ValidationError:
    """Detailed validation error information."""
    code: ValidationErrorCode
    message: str
    parameter: str
    value: Any
    severity: ValidationSeverity = ValidationSeverity.ERROR
    constraint: Optional[str] = None
    suggested_range: Optional[Tuple[float, float]] = None
    suggested_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code.value,
            "message": self.message,
            "parameter": self.parameter,
            "value": self.value,
            "severity": self.severity.value,
            "constraint": self.constraint,
            "suggested_range": self.suggested_range,
            "suggested_value": self.suggested_value,
        }


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)

    def add_error(self, error: ValidationError):
        """Add a validation error."""
        if error.severity == ValidationSeverity.ERROR:
            self.errors.append(error)
            self.is_valid = False
        elif error.severity == ValidationSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.info.append(error)

    def merge(self, other: ValidationResult):
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if not other.is_valid:
            self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
        }


# =============================================================================
# Physical Constants for Validation
# =============================================================================

class PhysicalConstants:
    """Physical constants for validation."""

    # Temperature limits
    ABSOLUTE_ZERO_K = 0.0
    ABSOLUTE_ZERO_C = -273.15
    ABSOLUTE_ZERO_F = -459.67

    # Pressure limits
    MIN_PRESSURE_PA = 0.0
    VACUUM_LIMIT_PA = 1e-10

    # Water/Steam
    WATER_CRITICAL_TEMP_K = 647.096
    WATER_CRITICAL_PRESSURE_MPA = 22.064
    WATER_TRIPLE_POINT_TEMP_K = 273.16
    WATER_TRIPLE_POINT_PRESSURE_PA = 611.657

    # Efficiency limits
    MAX_CARNOT_EFFICIENCY = 1.0
    MIN_EFFICIENCY = 0.0

    # Common bounds
    MAX_REASONABLE_TEMP_K = 10000.0  # Extreme combustion
    MAX_REASONABLE_PRESSURE_MPA = 1000.0  # Ultra-high pressure
    MAX_REASONABLE_VELOCITY_MS = 340.0 * 3  # ~Mach 3

    # Composition limits
    MIN_MOLE_FRACTION = 0.0
    MAX_MOLE_FRACTION = 1.0


# =============================================================================
# Range Validator
# =============================================================================

class RangeValidator:
    """
    Validates parameter values against physical and operational ranges.

    Provides:
    - Absolute physical limits (e.g., T > 0 K)
    - Operational limits (e.g., typical equipment ranges)
    - Warning thresholds for unusual values
    """

    # Standard ranges for common parameters
    STANDARD_RANGES = {
        # Temperature (K)
        "temperature_k": (0.01, 10000.0),
        "temperature_c": (-273.14, 9726.85),
        "temperature_f": (-459.66, 17540.33),
        "ambient_temperature_k": (173.0, 373.0),  # -100C to 100C
        "stack_temperature_k": (373.0, 873.0),  # 100C to 600C

        # Pressure
        "pressure_pa": (0.0, 1e12),
        "pressure_kpa": (0.0, 1e9),
        "pressure_mpa": (0.0, 1000.0),
        "pressure_bar": (0.0, 10000.0),
        "pressure_psi": (0.0, 145038.0),
        "atmospheric_pressure_kpa": (80.0, 110.0),

        # Mass flow
        "mass_flow_kg_s": (0.0, 1e6),
        "mass_flow_kg_h": (0.0, 3.6e9),

        # Energy
        "energy_kj": (0.0, 1e15),
        "power_kw": (0.0, 1e9),
        "heat_duty_kw": (0.0, 1e9),

        # Efficiency
        "efficiency": (0.0, 1.0),
        "efficiency_percent": (0.0, 100.0),

        # Composition
        "mole_fraction": (0.0, 1.0),
        "mass_fraction": (0.0, 1.0),
        "volume_fraction": (0.0, 1.0),
        "percent": (0.0, 100.0),

        # Combustion
        "excess_air_percent": (0.0, 500.0),
        "air_fuel_ratio": (0.0, 100.0),
        "oxygen_percent_dry": (0.0, 21.0),
        "co2_percent_dry": (0.0, 25.0),

        # Heat transfer
        "heat_transfer_coeff_w_m2k": (0.1, 100000.0),
        "thermal_conductivity_w_mk": (0.001, 500.0),
        "fouling_factor_m2k_w": (0.0, 0.01),

        # Dimensionless numbers
        "reynolds_number": (0.0, 1e10),
        "prandtl_number": (0.001, 10000.0),
        "nusselt_number": (0.1, 100000.0),

        # Safety
        "relief_pressure_ratio": (1.0, 2.0),
        "purge_time_minutes": (0.0, 1440.0),  # Max 24 hours
        "sif_probability": (0.0, 1.0),
    }

    @classmethod
    def validate(
        cls,
        parameter_name: str,
        value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_zero: bool = True,
        allow_negative: bool = True,
    ) -> ValidationResult:
        """
        Validate parameter value against range constraints.

        Args:
            parameter_name: Name of the parameter
            value: Value to validate
            min_value: Minimum allowed value (overrides standard)
            max_value: Maximum allowed value (overrides standard)
            allow_zero: Whether zero is allowed
            allow_negative: Whether negative values are allowed

        Returns:
            ValidationResult with any errors/warnings
        """
        result = ValidationResult(is_valid=True)

        # Check for NaN/Inf
        if math.isnan(value):
            result.add_error(ValidationError(
                code=ValidationErrorCode.NAN_VALUE,
                message=f"Parameter '{parameter_name}' is NaN",
                parameter=parameter_name,
                value=value,
            ))
            return result

        if math.isinf(value):
            result.add_error(ValidationError(
                code=ValidationErrorCode.INF_VALUE,
                message=f"Parameter '{parameter_name}' is infinite",
                parameter=parameter_name,
                value=value,
            ))
            return result

        # Check zero
        if not allow_zero and value == 0:
            result.add_error(ValidationError(
                code=ValidationErrorCode.ZERO_VALUE,
                message=f"Parameter '{parameter_name}' cannot be zero",
                parameter=parameter_name,
                value=value,
            ))

        # Check negative
        if not allow_negative and value < 0:
            result.add_error(ValidationError(
                code=ValidationErrorCode.NEGATIVE_VALUE,
                message=f"Parameter '{parameter_name}' cannot be negative",
                parameter=parameter_name,
                value=value,
            ))

        # Get standard range if not specified
        std_range = cls.STANDARD_RANGES.get(parameter_name)

        if min_value is None and std_range:
            min_value = std_range[0]
        if max_value is None and std_range:
            max_value = std_range[1]

        # Check minimum
        if min_value is not None and value < min_value:
            result.add_error(ValidationError(
                code=ValidationErrorCode.BELOW_MINIMUM,
                message=f"Parameter '{parameter_name}' value {value} is below minimum {min_value}",
                parameter=parameter_name,
                value=value,
                suggested_range=(min_value, max_value),
            ))

        # Check maximum
        if max_value is not None and value > max_value:
            result.add_error(ValidationError(
                code=ValidationErrorCode.ABOVE_MAXIMUM,
                message=f"Parameter '{parameter_name}' value {value} is above maximum {max_value}",
                parameter=parameter_name,
                value=value,
                suggested_range=(min_value, max_value),
            ))

        return result

    @classmethod
    def validate_temperature(
        cls,
        value: float,
        unit: str = "K",
        parameter_name: str = "temperature"
    ) -> ValidationResult:
        """Validate temperature with unit awareness."""
        result = ValidationResult(is_valid=True)

        # Convert to Kelvin for validation
        if unit == "K":
            value_k = value
        elif unit == "C":
            value_k = value + 273.15
        elif unit == "F":
            value_k = (value - 32) * 5/9 + 273.15
        elif unit == "R":
            value_k = value * 5/9
        else:
            value_k = value

        if value_k < PhysicalConstants.ABSOLUTE_ZERO_K:
            result.add_error(ValidationError(
                code=ValidationErrorCode.THERMODYNAMIC_VIOLATION,
                message=f"Temperature {value}{unit} is below absolute zero",
                parameter=parameter_name,
                value=value,
                suggested_range=(PhysicalConstants.ABSOLUTE_ZERO_K, None),
            ))

        if value_k > PhysicalConstants.MAX_REASONABLE_TEMP_K:
            result.add_error(ValidationError(
                code=ValidationErrorCode.ABOVE_MAXIMUM,
                message=f"Temperature {value}{unit} exceeds reasonable maximum",
                parameter=parameter_name,
                value=value,
                severity=ValidationSeverity.WARNING,
            ))

        return result

    @classmethod
    def validate_pressure(
        cls,
        value: float,
        unit: str = "Pa",
        allow_vacuum: bool = True,
        parameter_name: str = "pressure"
    ) -> ValidationResult:
        """Validate pressure with unit awareness."""
        result = ValidationResult(is_valid=True)

        # Negative pressure is non-physical (except gauge)
        if value < 0 and "g" not in unit.lower():  # Not gauge pressure
            result.add_error(ValidationError(
                code=ValidationErrorCode.NEGATIVE_VALUE,
                message=f"Absolute pressure cannot be negative: {value} {unit}",
                parameter=parameter_name,
                value=value,
            ))

        if not allow_vacuum and value < PhysicalConstants.VACUUM_LIMIT_PA:
            result.add_error(ValidationError(
                code=ValidationErrorCode.BELOW_MINIMUM,
                message=f"Pressure {value} {unit} is below vacuum limit",
                parameter=parameter_name,
                value=value,
            ))

        return result


# =============================================================================
# Unit Validator
# =============================================================================

class UnitValidator:
    """
    Validates units and unit compatibility.

    Ensures:
    - Units are recognized
    - Units match expected categories
    - Unit conversions are possible
    """

    def __init__(self):
        """Initialize with unit converter."""
        self._converter = UnitConverter()

    def validate_unit(
        self,
        unit: str,
        expected_category: Optional[UnitCategory] = None
    ) -> ValidationResult:
        """
        Validate that a unit is recognized and matches expected category.

        Args:
            unit: Unit symbol to validate
            expected_category: Expected unit category

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        unit_def = self._converter.get_unit_info(unit)

        if unit_def is None:
            result.add_error(ValidationError(
                code=ValidationErrorCode.UNKNOWN_UNIT,
                message=f"Unknown unit: '{unit}'",
                parameter="unit",
                value=unit,
            ))
            return result

        if expected_category and unit_def.category != expected_category:
            result.add_error(ValidationError(
                code=ValidationErrorCode.UNIT_MISMATCH,
                message=(
                    f"Unit '{unit}' is {unit_def.category.value}, "
                    f"expected {expected_category.value}"
                ),
                parameter="unit",
                value=unit,
            ))

        return result

    def validate_unit_compatibility(
        self,
        from_unit: str,
        to_unit: str
    ) -> ValidationResult:
        """Validate that two units can be converted between."""
        result = ValidationResult(is_valid=True)

        from_def = self._converter.get_unit_info(from_unit)
        to_def = self._converter.get_unit_info(to_unit)

        if from_def is None:
            result.add_error(ValidationError(
                code=ValidationErrorCode.UNKNOWN_UNIT,
                message=f"Unknown unit: '{from_unit}'",
                parameter="from_unit",
                value=from_unit,
            ))

        if to_def is None:
            result.add_error(ValidationError(
                code=ValidationErrorCode.UNKNOWN_UNIT,
                message=f"Unknown unit: '{to_unit}'",
                parameter="to_unit",
                value=to_unit,
            ))

        if from_def and to_def and from_def.category != to_def.category:
            result.add_error(ValidationError(
                code=ValidationErrorCode.INCOMPATIBLE_UNITS,
                message=(
                    f"Cannot convert between {from_unit} ({from_def.category.value}) "
                    f"and {to_unit} ({to_def.category.value})"
                ),
                parameter="units",
                value=(from_unit, to_unit),
            ))

        return result


# =============================================================================
# Physical Feasibility Validator
# =============================================================================

class PhysicalFeasibilityValidator:
    """
    Validates physical feasibility of parameter combinations.

    Checks thermodynamic constraints:
    - Phase boundaries (water/steam)
    - Energy conservation
    - Mass balance
    - Second law of thermodynamics
    """

    @staticmethod
    def validate_steam_conditions(
        temperature_k: float,
        pressure_mpa: float
    ) -> ValidationResult:
        """
        Validate steam conditions against water phase diagram.

        Args:
            temperature_k: Temperature in Kelvin
            pressure_mpa: Pressure in MPa

        Returns:
            ValidationResult with phase information
        """
        result = ValidationResult(is_valid=True)

        # Check critical point
        if (temperature_k > PhysicalConstants.WATER_CRITICAL_TEMP_K and
                pressure_mpa > PhysicalConstants.WATER_CRITICAL_PRESSURE_MPA):
            result.add_error(ValidationError(
                code=ValidationErrorCode.PHASE_CONSTRAINT_VIOLATION,
                message="Conditions are supercritical - use appropriate equations",
                parameter="steam_conditions",
                value=(temperature_k, pressure_mpa),
                severity=ValidationSeverity.WARNING,
            ))

        # Check triple point
        if (temperature_k < PhysicalConstants.WATER_TRIPLE_POINT_TEMP_K and
                pressure_mpa * 1e6 < PhysicalConstants.WATER_TRIPLE_POINT_PRESSURE_PA):
            result.add_error(ValidationError(
                code=ValidationErrorCode.PHASE_CONSTRAINT_VIOLATION,
                message="Conditions below triple point - ice formation possible",
                parameter="steam_conditions",
                value=(temperature_k, pressure_mpa),
                severity=ValidationSeverity.WARNING,
            ))

        return result

    @staticmethod
    def validate_carnot_limit(
        efficiency: float,
        hot_temp_k: float,
        cold_temp_k: float
    ) -> ValidationResult:
        """
        Validate efficiency against Carnot limit.

        Args:
            efficiency: Actual efficiency (0-1)
            hot_temp_k: Hot reservoir temperature (K)
            cold_temp_k: Cold reservoir temperature (K)

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if hot_temp_k <= cold_temp_k:
            result.add_error(ValidationError(
                code=ValidationErrorCode.THERMODYNAMIC_VIOLATION,
                message="Hot temperature must exceed cold temperature",
                parameter="temperatures",
                value=(hot_temp_k, cold_temp_k),
            ))
            return result

        carnot_efficiency = 1 - (cold_temp_k / hot_temp_k)

        if efficiency > carnot_efficiency:
            result.add_error(ValidationError(
                code=ValidationErrorCode.IMPOSSIBLE_EFFICIENCY,
                message=(
                    f"Efficiency {efficiency:.3f} exceeds Carnot limit "
                    f"{carnot_efficiency:.3f}"
                ),
                parameter="efficiency",
                value=efficiency,
                suggested_range=(0, carnot_efficiency),
            ))

        if efficiency < 0:
            result.add_error(ValidationError(
                code=ValidationErrorCode.NEGATIVE_VALUE,
                message="Efficiency cannot be negative",
                parameter="efficiency",
                value=efficiency,
            ))

        if efficiency > 1:
            result.add_error(ValidationError(
                code=ValidationErrorCode.IMPOSSIBLE_EFFICIENCY,
                message="Efficiency cannot exceed 100%",
                parameter="efficiency",
                value=efficiency,
            ))

        return result

    @staticmethod
    def validate_mass_balance(
        inputs: Dict[str, float],
        outputs: Dict[str, float],
        tolerance: float = 0.001
    ) -> ValidationResult:
        """
        Validate mass balance (inputs = outputs).

        Args:
            inputs: Input mass flows
            outputs: Output mass flows
            tolerance: Relative tolerance for imbalance

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        total_input = sum(inputs.values())
        total_output = sum(outputs.values())

        if total_input == 0 and total_output == 0:
            return result

        imbalance = abs(total_input - total_output)
        relative_imbalance = imbalance / max(total_input, total_output, 1e-10)

        if relative_imbalance > tolerance:
            result.add_error(ValidationError(
                code=ValidationErrorCode.MASS_BALANCE_VIOLATION,
                message=(
                    f"Mass imbalance: input={total_input:.4f}, "
                    f"output={total_output:.4f}, "
                    f"error={relative_imbalance*100:.2f}%"
                ),
                parameter="mass_balance",
                value={"inputs": inputs, "outputs": outputs},
            ))

        return result

    @staticmethod
    def validate_energy_balance(
        energy_in: float,
        energy_out: float,
        heat_loss: float = 0,
        tolerance: float = 0.01
    ) -> ValidationResult:
        """
        Validate energy balance (input = output + losses).

        Args:
            energy_in: Total energy input
            energy_out: Total energy output
            heat_loss: Heat losses
            tolerance: Relative tolerance

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if energy_in == 0:
            if energy_out > 0 or heat_loss > 0:
                result.add_error(ValidationError(
                    code=ValidationErrorCode.ENERGY_BALANCE_VIOLATION,
                    message="Energy output without input violates first law",
                    parameter="energy_balance",
                    value={"in": energy_in, "out": energy_out, "loss": heat_loss},
                ))
            return result

        expected_output = energy_in - heat_loss
        imbalance = abs(energy_out - expected_output)
        relative_imbalance = imbalance / energy_in

        if relative_imbalance > tolerance:
            result.add_error(ValidationError(
                code=ValidationErrorCode.ENERGY_BALANCE_VIOLATION,
                message=(
                    f"Energy imbalance: in={energy_in:.2f}, "
                    f"out={energy_out:.2f}, "
                    f"loss={heat_loss:.2f}, "
                    f"error={relative_imbalance*100:.2f}%"
                ),
                parameter="energy_balance",
                value={"in": energy_in, "out": energy_out, "loss": heat_loss},
            ))

        return result

    @staticmethod
    def validate_composition(
        components: Dict[str, float],
        parameter_name: str = "composition"
    ) -> ValidationResult:
        """
        Validate composition sums to 1 (or 100 for percent).

        Args:
            components: Component fractions/percentages
            parameter_name: Name for error messages

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if not components:
            return result

        total = sum(components.values())

        # Check for negative values
        for comp, value in components.items():
            if value < 0:
                result.add_error(ValidationError(
                    code=ValidationErrorCode.NEGATIVE_VALUE,
                    message=f"Composition component '{comp}' cannot be negative",
                    parameter=parameter_name,
                    value=value,
                ))

        # Check sum (allow for both fraction and percent)
        if abs(total - 1.0) > 0.001 and abs(total - 100.0) > 0.1:
            result.add_error(ValidationError(
                code=ValidationErrorCode.SUM_CONSTRAINT,
                message=f"Composition sum {total:.4f} does not equal 1.0 or 100%",
                parameter=parameter_name,
                value=total,
            ))

        return result


# =============================================================================
# Constraint Validator
# =============================================================================

class ConstraintValidator:
    """
    Validates cross-parameter constraints and dependencies.

    Handles:
    - Parameter dependencies
    - Mutual exclusion
    - Conditional requirements
    - Complex constraints
    """

    @staticmethod
    def validate_dependency(
        parameters: Dict[str, Any],
        dependent: str,
        requires: List[str]
    ) -> ValidationResult:
        """
        Validate that dependent parameter has all required parameters.

        Args:
            parameters: All parameters
            dependent: Parameter that has dependencies
            requires: List of required parameters

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        if dependent not in parameters:
            return result  # Dependency not used

        for required in requires:
            if required not in parameters or parameters[required] is None:
                result.add_error(ValidationError(
                    code=ValidationErrorCode.DEPENDENCY_MISSING,
                    message=f"Parameter '{dependent}' requires '{required}'",
                    parameter=dependent,
                    value=parameters.get(dependent),
                    constraint=f"requires: {requires}",
                ))

        return result

    @staticmethod
    def validate_mutual_exclusion(
        parameters: Dict[str, Any],
        exclusive_groups: List[List[str]]
    ) -> ValidationResult:
        """
        Validate mutual exclusion constraints.

        Args:
            parameters: All parameters
            exclusive_groups: Groups of mutually exclusive parameters

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        for group in exclusive_groups:
            present = [p for p in group if p in parameters and parameters[p] is not None]
            if len(present) > 1:
                result.add_error(ValidationError(
                    code=ValidationErrorCode.MUTUAL_EXCLUSION,
                    message=f"Parameters {present} are mutually exclusive",
                    parameter=present[0],
                    value=parameters[present[0]],
                    constraint=f"exclusive: {group}",
                ))

        return result

    @staticmethod
    def validate_at_least_one(
        parameters: Dict[str, Any],
        group: List[str],
        group_name: str = "parameters"
    ) -> ValidationResult:
        """
        Validate at least one parameter from group is present.

        Args:
            parameters: All parameters
            group: Group of parameters (at least one required)
            group_name: Name for error messages

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        present = [p for p in group if p in parameters and parameters[p] is not None]

        if not present:
            result.add_error(ValidationError(
                code=ValidationErrorCode.DEPENDENCY_MISSING,
                message=f"At least one of {group_name} must be provided: {group}",
                parameter=group_name,
                value=None,
                constraint=f"at_least_one: {group}",
            ))

        return result

    @staticmethod
    def validate_custom_constraint(
        parameters: Dict[str, Any],
        constraint_fn: Callable[[Dict[str, Any]], Tuple[bool, str]],
        constraint_name: str
    ) -> ValidationResult:
        """
        Validate using custom constraint function.

        Args:
            parameters: All parameters
            constraint_fn: Function returning (is_valid, message)
            constraint_name: Name of constraint for errors

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        is_valid, message = constraint_fn(parameters)

        if not is_valid:
            result.add_error(ValidationError(
                code=ValidationErrorCode.CONSTRAINT_VIOLATION,
                message=message,
                parameter=constraint_name,
                value=parameters,
                constraint=constraint_name,
            ))

        return result


# =============================================================================
# Combined Validation Engine
# =============================================================================

class ValidationEngine:
    """
    Comprehensive validation engine combining all validators.

    Provides a single interface for validating formula inputs
    against all constraint types.
    """

    def __init__(self):
        """Initialize validation engine."""
        self.range_validator = RangeValidator()
        self.unit_validator = UnitValidator()
        self.physical_validator = PhysicalFeasibilityValidator()
        self.constraint_validator = ConstraintValidator()

    def validate_formula_input(
        self,
        formula: FormulaDefinition,
        parameters: Dict[str, float],
        parameter_units: Optional[Dict[str, str]] = None
    ) -> ValidationResult:
        """
        Comprehensive validation of formula input.

        Args:
            formula: Formula definition
            parameters: Parameter values
            parameter_units: Units for each parameter

        Returns:
            Combined ValidationResult
        """
        result = ValidationResult(is_valid=True)
        param_defs = {p.name: p for p in formula.parameters}

        # Validate each parameter
        for param_name, value in parameters.items():
            param_def = param_defs.get(param_name)
            if not param_def:
                continue

            # Range validation
            range_result = self.range_validator.validate(
                parameter_name=param_name,
                value=value,
                min_value=param_def.min_value,
                max_value=param_def.max_value,
            )
            result.merge(range_result)

            # Unit validation
            if parameter_units and param_name in parameter_units:
                unit_result = self.unit_validator.validate_unit(
                    parameter_units[param_name],
                    param_def.category,
                )
                result.merge(unit_result)

        # Validate required parameters
        for param_def in formula.parameters:
            if param_def.required and param_def.name not in parameters:
                if param_def.default_value is None:
                    result.add_error(ValidationError(
                        code=ValidationErrorCode.MISSING_VALUE,
                        message=f"Missing required parameter: {param_def.name}",
                        parameter=param_def.name,
                        value=None,
                    ))

        return result

    def validate_thermodynamic_input(
        self,
        parameters: Dict[str, float],
        check_steam: bool = False,
        check_carnot: bool = False,
    ) -> ValidationResult:
        """
        Validate thermodynamic constraints.

        Args:
            parameters: Parameter values
            check_steam: Validate steam conditions
            check_carnot: Validate Carnot efficiency

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Steam conditions
        if check_steam:
            temp_k = parameters.get("temperature_k")
            press_mpa = parameters.get("pressure_mpa")
            if temp_k is not None and press_mpa is not None:
                steam_result = self.physical_validator.validate_steam_conditions(
                    temp_k, press_mpa
                )
                result.merge(steam_result)

        # Carnot efficiency
        if check_carnot:
            efficiency = parameters.get("efficiency")
            hot_temp = parameters.get("hot_temperature_k")
            cold_temp = parameters.get("cold_temperature_k")
            if all(v is not None for v in [efficiency, hot_temp, cold_temp]):
                carnot_result = self.physical_validator.validate_carnot_limit(
                    efficiency, hot_temp, cold_temp
                )
                result.merge(carnot_result)

        return result

    def validate_combustion_input(
        self,
        fuel_composition: Optional[Dict[str, float]] = None,
        excess_air: Optional[float] = None,
        flue_gas_composition: Optional[Dict[str, float]] = None,
    ) -> ValidationResult:
        """
        Validate combustion calculation inputs.

        Args:
            fuel_composition: Fuel composition (mass or mole fractions)
            excess_air: Excess air percentage
            flue_gas_composition: Flue gas composition

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Fuel composition
        if fuel_composition:
            comp_result = self.physical_validator.validate_composition(
                fuel_composition, "fuel_composition"
            )
            result.merge(comp_result)

        # Excess air
        if excess_air is not None:
            ea_result = self.range_validator.validate(
                "excess_air_percent",
                excess_air,
                min_value=0,
                max_value=500,
            )
            result.merge(ea_result)

        # Flue gas composition
        if flue_gas_composition:
            fg_result = self.physical_validator.validate_composition(
                flue_gas_composition, "flue_gas_composition"
            )
            result.merge(fg_result)

            # Check O2 vs CO2 consistency
            o2 = flue_gas_composition.get("O2", 0)
            co2 = flue_gas_composition.get("CO2", 0)
            if o2 > 0.21 or o2 > 21:  # Fraction or percent
                result.add_error(ValidationError(
                    code=ValidationErrorCode.THERMODYNAMIC_VIOLATION,
                    message="Flue gas O2 cannot exceed air O2 content",
                    parameter="flue_gas_O2",
                    value=o2,
                ))

        return result
