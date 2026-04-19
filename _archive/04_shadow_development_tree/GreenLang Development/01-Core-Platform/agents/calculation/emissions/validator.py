# -*- coding: utf-8 -*-
"""
Calculation Validator

Validates calculation inputs and outputs for data quality and reasonableness.

Validation Checks:
1. Input validation (negative values, outliers, unit compatibility)
2. Output validation (NaN/Inf, magnitude checks)
3. Provenance validation (hash integrity)
4. Data quality assessment
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from decimal import Decimal
from greenlang.utilities.determinism import FinancialDecimal
from .core_calculator import (
    CalculationRequest,
    CalculationResult,
    CalculationStatus,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


@dataclass
class ValidationResult:
    """
    Result of validation check.

    Attributes:
        is_valid: Overall validation result
        errors: Critical errors (calculation should not proceed)
        warnings: Non-critical warnings (calculation can proceed but flagged)
        info: Informational messages
        checks_performed: List of validation checks performed
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    checks_performed: List[str] = field(default_factory=list)

    def add_error(self, message: str):
        """Add error and mark as invalid"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """Add warning (doesn't affect validity)"""
        self.warnings.append(message)

    def add_info(self, message: str):
        """Add informational message"""
        self.info.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'checks_performed': self.checks_performed,
        }


class CalculationValidator:
    """
    Validates calculations for data quality and reasonableness.

    ZERO-HALLUCINATION: Pure rule-based validation, no LLM involved.
    """

    # Reasonable emission ranges (kg CO2e per unit)
    EMISSION_FACTOR_RANGES = {
        # Fuels (kg CO2e per unit)
        'diesel': {'min': 2.0, 'max': 12.0, 'unit': 'per gallon or per liter'},
        'gasoline': {'min': 1.5, 'max': 10.0, 'unit': 'per gallon or per liter'},
        'natural_gas': {'min': 1.0, 'max': 60.0, 'unit': 'per therm or per MMBtu'},
        'coal': {'min': 0.5, 'max': 3.0, 'unit': 'per kg'},

        # Electricity (kg CO2e per kWh)
        'electricity': {'min': 0.0, 'max': 1.5, 'unit': 'per kWh'},

        # Materials (kg CO2e per kg)
        'steel': {'min': 0.3, 'max': 3.0, 'unit': 'per kg'},
        'aluminum': {'min': 0.4, 'max': 15.0, 'unit': 'per kg'},
        'cement': {'min': 0.5, 'max': 1.2, 'unit': 'per kg'},
    }

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.

        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode

    def validate_request(self, request: CalculationRequest) -> ValidationResult:
        """
        Validate calculation request before processing.

        Args:
            request: CalculationRequest to validate

        Returns:
            ValidationResult with any issues found

        Example:
            >>> from greenlang.calculation import CalculationValidator, CalculationRequest
            >>> validator = CalculationValidator()
            >>> request = CalculationRequest(
            ...     factor_id='diesel',
            ...     activity_amount=100,
            ...     activity_unit='gallons'
            ... )
            >>> result = validator.validate_request(request)
            >>> if not result.is_valid:
            ...     print(f"Validation errors: {result.errors}")
        """
        result = ValidationResult(is_valid=True)

        # Check 1: Activity amount is non-negative
        result.checks_performed.append('activity_amount_non_negative')
        if request.activity_amount < 0:
            result.add_error(
                f"Activity amount cannot be negative: {request.activity_amount}"
            )

        # Check 2: Activity amount is not unreasonably large
        result.checks_performed.append('activity_amount_reasonable')
        if request.activity_amount > 1e9:
            result.add_warning(
                f"Activity amount is very large: {request.activity_amount}. "
                "Please verify this is correct."
            )

        # Check 3: Activity amount is not zero (informational)
        result.checks_performed.append('activity_amount_non_zero')
        if request.activity_amount == 0:
            result.add_info(
                "Activity amount is zero - emissions will be zero"
            )

        # Check 4: Factor ID is not empty
        result.checks_performed.append('factor_id_not_empty')
        if not request.factor_id or request.factor_id.strip() == '':
            result.add_error("Factor ID cannot be empty")

        # Check 5: Unit is not empty
        result.checks_performed.append('unit_not_empty')
        if not request.activity_unit or request.activity_unit.strip() == '':
            result.add_error("Activity unit cannot be empty")

        # Check 6: Detect potential unit mismatches
        result.checks_performed.append('unit_factor_compatibility')
        self._check_unit_factor_compatibility(request, result)

        # In strict mode, warnings become errors
        if self.strict_mode and result.warnings:
            for warning in result.warnings:
                result.add_error(f"[STRICT MODE] {warning}")
            result.warnings.clear()

        return result

    def validate_result(self, calculation: CalculationResult) -> ValidationResult:
        """
        Validate calculation result after processing.

        Args:
            calculation: CalculationResult to validate

        Returns:
            ValidationResult with any issues found

        Example:
            >>> validator = CalculationValidator()
            >>> # ... perform calculation ...
            >>> result = validator.validate_result(calculation_result)
            >>> if not result.is_valid:
            ...     print(f"Result validation failed: {result.errors}")
        """
        result = ValidationResult(is_valid=True)

        # Check 1: Emissions are not NaN or Inf
        result.checks_performed.append('emissions_not_nan_inf')
        if calculation.emissions_kg_co2e != calculation.emissions_kg_co2e:  # NaN check
            result.add_error("Emissions result is NaN (not a number)")

        try:
            if abs(FinancialDecimal.from_string(calculation.emissions_kg_co2e)) == FinancialDecimal.from_string('inf'):
                result.add_error("Emissions result is infinite")
        except (ValueError, OverflowError):
            result.add_error("Emissions result is invalid")

        # Check 2: Emissions are non-negative
        result.checks_performed.append('emissions_non_negative')
        if calculation.emissions_kg_co2e < 0:
            result.add_error(
                f"Emissions cannot be negative: {calculation.emissions_kg_co2e} kg CO2e"
            )

        # Check 3: Provenance hash exists
        result.checks_performed.append('provenance_hash_exists')
        if not calculation.provenance_hash:
            result.add_error("Missing provenance hash - cannot verify calculation integrity")

        # Check 4: Verify provenance hash
        result.checks_performed.append('provenance_hash_valid')
        if calculation.provenance_hash and not calculation.verify_provenance():
            result.add_error(
                "Provenance hash mismatch - calculation may have been tampered with"
            )

        # Check 5: Calculation steps documented
        result.checks_performed.append('calculation_steps_documented')
        if not calculation.calculation_steps:
            result.add_warning("No calculation steps documented in audit trail")

        # Check 6: Factor resolution exists
        result.checks_performed.append('factor_resolution_exists')
        if not calculation.factor_resolution:
            result.add_warning("No emission factor resolution information")

        # Check 7: Factor source URI exists
        result.checks_performed.append('factor_source_uri')
        if calculation.factor_resolution and not calculation.factor_resolution.uri:
            result.add_warning(
                "Emission factor missing source URI - cannot verify provenance"
            )

        # Check 8: Reasonable emission magnitude
        result.checks_performed.append('emissions_reasonable_magnitude')
        self._check_emission_reasonableness(calculation, result)

        # Check 9: Calculation completed successfully
        result.checks_performed.append('calculation_status')
        if calculation.status == CalculationStatus.FAILED:
            result.add_error(
                f"Calculation failed: {', '.join(calculation.errors)}"
            )

        # In strict mode, warnings become errors
        if self.strict_mode and result.warnings:
            for warning in result.warnings:
                result.add_error(f"[STRICT MODE] {warning}")
            result.warnings.clear()

        return result

    def _check_unit_factor_compatibility(
        self,
        request: CalculationRequest,
        result: ValidationResult
    ):
        """Check if unit is likely compatible with factor"""
        factor_id = request.factor_id.lower()
        unit = request.activity_unit.lower()

        # Common incompatibilities
        incompatible_pairs = [
            (['diesel', 'gasoline', 'fuel'], ['kwh', 'mwh'], 'Fuel factors typically use volume or mass units (gallons, liters, kg)'),
            (['electricity', 'grid'], ['gallons', 'liters'], 'Electricity factors use energy units (kWh, MWh)'),
            (['steel', 'aluminum', 'cement'], ['gallons', 'liters'], 'Material factors typically use mass units (kg, tonnes)'),
        ]

        for factor_keywords, unit_keywords, message in incompatible_pairs:
            if any(kw in factor_id for kw in factor_keywords):
                if any(kw in unit for kw in unit_keywords):
                    result.add_warning(
                        f"Potential unit mismatch: {message}"
                    )

    def _check_emission_reasonableness(
        self,
        calculation: CalculationResult,
        result: ValidationResult
    ):
        """Check if emission result is within reasonable range"""
        emissions = FinancialDecimal.from_string(calculation.emissions_kg_co2e)
        activity = FinancialDecimal.from_string(calculation.request.activity_amount)

        if emissions == 0:
            return  # Zero emissions is valid (e.g., renewable energy)

        # Calculate emissions per unit of activity
        emissions_per_unit = emissions / activity if activity > 0 else 0

        # Check against known ranges
        factor_id = calculation.request.factor_id.lower()

        for factor_type, range_info in self.EMISSION_FACTOR_RANGES.items():
            if factor_type in factor_id:
                if emissions_per_unit < range_info['min']:
                    result.add_warning(
                        f"Emissions per unit ({emissions_per_unit:.3f} kg CO2e) "
                        f"is below typical range for {factor_type} "
                        f"(min: {range_info['min']} {range_info['unit']})"
                    )
                elif emissions_per_unit > range_info['max']:
                    result.add_warning(
                        f"Emissions per unit ({emissions_per_unit:.3f} kg CO2e) "
                        f"is above typical range for {factor_type} "
                        f"(max: {range_info['max']} {range_info['unit']})"
                    )
                break

        # Check for extremely large emissions (>1000 tonnes CO2e)
        if emissions > 1_000_000:  # 1000 tonnes
            result.add_warning(
                f"Emissions are very large: {emissions:,.0f} kg CO2e "
                f"({emissions/1000:,.0f} tonnes CO2e). Please verify inputs."
            )

    def validate_batch(
        self,
        calculations: List[CalculationResult]
    ) -> Dict[str, Any]:
        """
        Validate batch of calculations.

        Args:
            calculations: List of CalculationResults

        Returns:
            Dictionary with batch validation summary

        Example:
            >>> validator = CalculationValidator()
            >>> batch_validation = validator.validate_batch(calculations)
            >>> print(f"Valid: {batch_validation['valid_count']}/{batch_validation['total_count']}")
        """
        total = len(calculations)
        valid = 0
        invalid = 0
        all_errors = []
        all_warnings = []

        for calc in calculations:
            validation = self.validate_result(calc)
            if validation.is_valid:
                valid += 1
            else:
                invalid += 1

            all_errors.extend(validation.errors)
            all_warnings.extend(validation.warnings)

        return {
            'total_count': total,
            'valid_count': valid,
            'invalid_count': invalid,
            'validation_rate': valid / total if total > 0 else 0,
            'total_errors': len(all_errors),
            'total_warnings': len(all_warnings),
            'errors': all_errors[:10],  # First 10 errors
            'warnings': all_warnings[:10],  # First 10 warnings
        }
