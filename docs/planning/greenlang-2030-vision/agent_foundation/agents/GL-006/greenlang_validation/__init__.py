# -*- coding: utf-8 -*-
"""
GreenLang Validation Framework for GL-006 HeatRecoveryMaximizer.

This module provides specialized validation utilities for thermodynamic calculations,
economic analysis, and operational constraints in heat recovery systems.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import math

# Re-export core validation
from greenlang_core.validation import (
    ValidationResult,
    ValidationError,
    ValidationSeverity,
    ValidationCategory,
    Validator,
    ValidationContext,
)


class ThermodynamicValidationCode(str, Enum):
    """Thermodynamic validation error codes."""
    TEMPERATURE_BELOW_ABSOLUTE_ZERO = "THERMO_001"
    TEMPERATURE_APPROACH_VIOLATION = "THERMO_002"
    ENERGY_BALANCE_VIOLATION = "THERMO_003"
    ENTROPY_DECREASE_VIOLATION = "THERMO_004"
    INVALID_HEAT_TRANSFER_COEFFICIENT = "THERMO_005"
    PRESSURE_DROP_EXCEEDED = "THERMO_006"
    EFFECTIVENESS_OUT_OF_RANGE = "THERMO_007"
    PINCH_VIOLATION = "THERMO_008"
    EXERGY_DESTRUCTION_NEGATIVE = "THERMO_009"
    MASS_BALANCE_VIOLATION = "THERMO_010"


@dataclass
class ThermodynamicConstraints:
    """Thermodynamic validation constraints."""
    min_temperature_approach_c: float = 10.0
    max_temperature_c: float = 1000.0
    min_temperature_c: float = -273.15
    max_pressure_bar: float = 100.0
    min_pressure_bar: float = 0.001
    max_pressure_drop_bar: float = 0.5
    min_effectiveness: float = 0.1
    max_effectiveness: float = 0.99
    energy_balance_tolerance: float = 0.01
    mass_balance_tolerance: float = 0.001


class ThermodynamicValidator:
    """
    Specialized validator for thermodynamic calculations.

    Provides validation for heat recovery calculations ensuring physical
    consistency and adherence to thermodynamic laws.
    """

    def __init__(self, constraints: Optional[ThermodynamicConstraints] = None):
        """Initialize with optional custom constraints."""
        self.constraints = constraints or ThermodynamicConstraints()

    def validate_temperature(
        self,
        temperature_c: float,
        field_name: str = "temperature"
    ) -> ValidationResult:
        """Validate temperature value."""
        result = ValidationResult()

        if temperature_c < self.constraints.min_temperature_c:
            result.add_error(ValidationError(
                message=f"{field_name} ({temperature_c}C) is below absolute zero",
                code=ThermodynamicValidationCode.TEMPERATURE_BELOW_ABSOLUTE_ZERO.value,
                field=field_name,
                value=temperature_c,
                category=ValidationCategory.THERMODYNAMIC,
            ))

        if temperature_c > self.constraints.max_temperature_c:
            result.add_error(ValidationError(
                message=f"{field_name} ({temperature_c}C) exceeds maximum ({self.constraints.max_temperature_c}C)",
                code=ThermodynamicValidationCode.TEMPERATURE_BELOW_ABSOLUTE_ZERO.value,
                field=field_name,
                value=temperature_c,
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.THERMODYNAMIC,
            ))

        return result

    def validate_temperature_approach(
        self,
        hot_outlet_c: float,
        cold_inlet_c: float,
        field_name: str = "temperature_approach"
    ) -> ValidationResult:
        """Validate minimum temperature approach."""
        result = ValidationResult()
        approach = hot_outlet_c - cold_inlet_c

        if approach < self.constraints.min_temperature_approach_c:
            result.add_error(ValidationError(
                message=f"Temperature approach ({approach:.2f}C) is below minimum ({self.constraints.min_temperature_approach_c}C)",
                code=ThermodynamicValidationCode.TEMPERATURE_APPROACH_VIOLATION.value,
                field=field_name,
                value=approach,
                category=ValidationCategory.THERMODYNAMIC,
            ))

        return result

    def validate_energy_balance(
        self,
        heat_in_kw: float,
        heat_out_kw: float,
        field_name: str = "energy_balance"
    ) -> ValidationResult:
        """Validate energy balance."""
        result = ValidationResult()

        if heat_in_kw <= 0 or heat_out_kw <= 0:
            if heat_in_kw > 0 or heat_out_kw > 0:  # One side has heat
                imbalance = abs(heat_in_kw - heat_out_kw)
                relative_error = imbalance / max(heat_in_kw, heat_out_kw, 1)

                if relative_error > self.constraints.energy_balance_tolerance:
                    result.add_error(ValidationError(
                        message=f"Energy balance violation: {relative_error*100:.2f}% imbalance",
                        code=ThermodynamicValidationCode.ENERGY_BALANCE_VIOLATION.value,
                        field=field_name,
                        value={"heat_in": heat_in_kw, "heat_out": heat_out_kw},
                        category=ValidationCategory.THERMODYNAMIC,
                    ))
        else:
            imbalance = abs(heat_in_kw - heat_out_kw)
            relative_error = imbalance / max(heat_in_kw, heat_out_kw)

            if relative_error > self.constraints.energy_balance_tolerance:
                result.add_error(ValidationError(
                    message=f"Energy balance violation: {relative_error*100:.2f}% imbalance",
                    code=ThermodynamicValidationCode.ENERGY_BALANCE_VIOLATION.value,
                    field=field_name,
                    value={"heat_in": heat_in_kw, "heat_out": heat_out_kw},
                    category=ValidationCategory.THERMODYNAMIC,
                ))

        return result

    def validate_effectiveness(
        self,
        effectiveness: float,
        field_name: str = "effectiveness"
    ) -> ValidationResult:
        """Validate heat exchanger effectiveness."""
        result = ValidationResult()

        if effectiveness < self.constraints.min_effectiveness:
            result.add_error(ValidationError(
                message=f"Effectiveness ({effectiveness:.3f}) is below minimum ({self.constraints.min_effectiveness})",
                code=ThermodynamicValidationCode.EFFECTIVENESS_OUT_OF_RANGE.value,
                field=field_name,
                value=effectiveness,
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.THERMODYNAMIC,
            ))

        if effectiveness > self.constraints.max_effectiveness:
            result.add_error(ValidationError(
                message=f"Effectiveness ({effectiveness:.3f}) exceeds physical maximum ({self.constraints.max_effectiveness})",
                code=ThermodynamicValidationCode.EFFECTIVENESS_OUT_OF_RANGE.value,
                field=field_name,
                value=effectiveness,
                category=ValidationCategory.THERMODYNAMIC,
            ))

        return result

    def validate_heat_exchanger(
        self,
        hot_inlet_c: float,
        hot_outlet_c: float,
        cold_inlet_c: float,
        cold_outlet_c: float,
        hot_flow_kg_s: float,
        cold_flow_kg_s: float,
        hot_cp_kj_kg_k: float = 4.18,
        cold_cp_kj_kg_k: float = 4.18,
    ) -> ValidationResult:
        """
        Comprehensive heat exchanger validation.

        Validates temperature profiles, energy balance, and thermodynamic consistency.
        """
        result = ValidationResult()

        # Validate temperatures
        for temp, name in [
            (hot_inlet_c, "hot_inlet"),
            (hot_outlet_c, "hot_outlet"),
            (cold_inlet_c, "cold_inlet"),
            (cold_outlet_c, "cold_outlet"),
        ]:
            result.merge(self.validate_temperature(temp, name))

        # Validate temperature profiles
        if hot_outlet_c > hot_inlet_c:
            result.add_error(ValidationError(
                message="Hot outlet cannot be higher than hot inlet",
                code="THERMO_PROFILE_001",
                field="hot_temperature_profile",
                category=ValidationCategory.THERMODYNAMIC,
            ))

        if cold_outlet_c < cold_inlet_c:
            result.add_error(ValidationError(
                message="Cold outlet cannot be lower than cold inlet",
                code="THERMO_PROFILE_002",
                field="cold_temperature_profile",
                category=ValidationCategory.THERMODYNAMIC,
            ))

        # Validate temperature approach
        result.merge(self.validate_temperature_approach(hot_outlet_c, cold_inlet_c))

        # Calculate and validate energy balance
        hot_duty_kw = hot_flow_kg_s * hot_cp_kj_kg_k * (hot_inlet_c - hot_outlet_c)
        cold_duty_kw = cold_flow_kg_s * cold_cp_kj_kg_k * (cold_outlet_c - cold_inlet_c)
        result.merge(self.validate_energy_balance(hot_duty_kw, cold_duty_kw))

        # Calculate and validate effectiveness
        c_hot = hot_flow_kg_s * hot_cp_kj_kg_k
        c_cold = cold_flow_kg_s * cold_cp_kj_kg_k
        c_min = min(c_hot, c_cold)

        if c_min > 0:
            q_max = c_min * (hot_inlet_c - cold_inlet_c)
            if q_max > 0:
                effectiveness = hot_duty_kw / q_max
                result.merge(self.validate_effectiveness(effectiveness))

        return result


class EconomicValidator:
    """Validator for economic calculations."""

    def __init__(self):
        """Initialize economic validator."""
        self.min_roi = -1.0  # -100%
        self.max_roi = 10.0  # 1000%
        self.min_payback_years = 0.1
        self.max_payback_years = 30.0
        self.min_npv_factor = -1e9
        self.max_npv_factor = 1e9

    def validate_roi(self, roi: float, field_name: str = "roi") -> ValidationResult:
        """Validate return on investment."""
        result = ValidationResult()

        if roi < self.min_roi:
            result.add_error(ValidationError(
                message=f"ROI ({roi*100:.1f}%) is below minimum ({self.min_roi*100:.1f}%)",
                code="ECON_ROI_001",
                field=field_name,
                value=roi,
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.ECONOMIC,
            ))

        if roi > self.max_roi:
            result.add_error(ValidationError(
                message=f"ROI ({roi*100:.1f}%) exceeds realistic maximum - verify calculations",
                code="ECON_ROI_002",
                field=field_name,
                value=roi,
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.ECONOMIC,
            ))

        return result

    def validate_payback(self, payback_years: float, field_name: str = "payback") -> ValidationResult:
        """Validate payback period."""
        result = ValidationResult()

        if payback_years < self.min_payback_years:
            result.add_error(ValidationError(
                message=f"Payback ({payback_years:.2f} years) is unrealistically short",
                code="ECON_PAYBACK_001",
                field=field_name,
                value=payback_years,
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.ECONOMIC,
            ))

        if payback_years > self.max_payback_years:
            result.add_error(ValidationError(
                message=f"Payback ({payback_years:.2f} years) exceeds typical project lifetime",
                code="ECON_PAYBACK_002",
                field=field_name,
                value=payback_years,
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.ECONOMIC,
            ))

        return result


__all__ = [
    # Re-exports from core
    'ValidationResult',
    'ValidationError',
    'ValidationSeverity',
    'ValidationCategory',
    'Validator',
    'ValidationContext',
    # Module-specific
    'ThermodynamicValidationCode',
    'ThermodynamicConstraints',
    'ThermodynamicValidator',
    'EconomicValidator',
]
