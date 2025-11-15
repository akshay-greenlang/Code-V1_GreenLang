"""
Calculation Validators - Ensure Regulatory Compliance

Validates calculations meet regulatory requirements:
- Precision requirements
- Materiality thresholds
- Reproducibility verification
- Data quality checks
- Regulatory-specific rules (GHG Protocol, CBAM, CSRD, etc.)
"""

from typing import Dict, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field
from enum import Enum

from .calculation_engine import CalculationResult


class ValidationSeverity(str, Enum):
    """Validation message severity."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationMessage(BaseModel):
    """Validation message."""

    severity: ValidationSeverity = Field(..., description="Message severity")
    code: str = Field(..., description="Validation code")
    message: str = Field(..., description="Human-readable message")
    field: Optional[str] = Field(None, description="Related field")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")


class ValidationResult(BaseModel):
    """Calculation validation result."""

    is_valid: bool = Field(..., description="Overall validation status")
    errors: List[ValidationMessage] = Field(default_factory=list, description="Validation errors")
    warnings: List[ValidationMessage] = Field(default_factory=list, description="Validation warnings")
    info: List[ValidationMessage] = Field(default_factory=list, description="Informational messages")

    @property
    def error_count(self) -> int:
        """Get error count."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Get warning count."""
        return len(self.warnings)

    def add_error(self, code: str, message: str, field: Optional[str] = None, **metadata):
        """Add validation error."""
        self.errors.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code=code,
            message=message,
            field=field,
            metadata=metadata
        ))
        self.is_valid = False

    def add_warning(self, code: str, message: str, field: Optional[str] = None, **metadata):
        """Add validation warning."""
        self.warnings.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code=code,
            message=message,
            field=field,
            metadata=metadata
        ))

    def add_info(self, code: str, message: str, field: Optional[str] = None, **metadata):
        """Add informational message."""
        self.info.append(ValidationMessage(
            severity=ValidationSeverity.INFO,
            code=code,
            message=message,
            field=field,
            metadata=metadata
        ))


class CalculationValidator:
    """
    Validate calculations meet regulatory requirements.

    Implements validation rules for:
    - GHG Protocol
    - CBAM (Carbon Border Adjustment Mechanism)
    - CSRD (Corporate Sustainability Reporting Directive)
    - EU Taxonomy
    - ISO 14064
    """

    # Regulatory precision requirements
    PRECISION_REQUIREMENTS = {
        'ghg_protocol': {
            'scope1': 2,
            'scope2': 2,
            'scope3': 2,
            'default': 2
        },
        'cbam': {
            'embedded_emissions': 3,
            'default': 3
        },
        'csrd': {
            'emissions': 2,
            'financial': 2,
            'default': 2
        }
    }

    # Materiality thresholds (tCO2e)
    MATERIALITY_THRESHOLDS = {
        'ghp_protocol': {
            'scope1': Decimal('500'),  # 500 tCO2e
            'scope2': Decimal('500'),
            'scope3_category': Decimal('100')
        },
        'csrd': {
            'scope1': Decimal('1000'),  # 1000 tCO2e
            'scope2': Decimal('1000'),
            'scope3_total': Decimal('10000')
        }
    }

    def __init__(self):
        """Initialize validator."""
        pass

    def validate_result(
        self,
        result: CalculationResult,
        standard: str = "ghg_protocol",
        check_reproducibility: bool = True
    ) -> ValidationResult:
        """
        Validate calculation result.

        Args:
            result: Calculation result to validate
            standard: Regulatory standard to validate against
            check_reproducibility: Whether to test reproducibility

        Returns:
            ValidationResult with all validation messages
        """
        validation = ValidationResult(is_valid=True)

        # Check 1: Provenance hash present
        if not result.provenance_hash:
            validation.add_error(
                code="MISSING_PROVENANCE",
                message="Calculation missing provenance hash - audit trail incomplete"
            )

        # Check 2: Calculation steps documented
        if not result.calculation_steps:
            validation.add_error(
                code="MISSING_STEPS",
                message="No calculation steps documented - cannot audit calculation"
            )

        # Check 3: Output value is valid
        if result.output_value < 0:
            validation.add_error(
                code="NEGATIVE_EMISSIONS",
                message="Negative emission value detected (physically impossible)",
                field="output_value",
                value=float(result.output_value)
            )

        # Check 4: Precision meets regulatory requirements
        self._validate_precision(result, standard, validation)

        # Check 5: Data quality checks
        self._validate_data_quality(result, validation)

        # Check 6: Materiality assessment
        self._validate_materiality(result, standard, validation)

        # Check 7: Uncertainty within acceptable range
        self._validate_uncertainty(result, validation)

        # Check 8: Standard-specific validations
        if standard == "ghg_protocol":
            self._validate_ghg_protocol(result, validation)
        elif standard == "cbam":
            self._validate_cbam(result, validation)
        elif standard == "csrd":
            self._validate_csrd(result, validation)

        # Check 9: Reproducibility (optional, expensive)
        if check_reproducibility and not validation.errors:
            # This would re-run the calculation - skipped in base validation
            validation.add_info(
                code="REPRODUCIBILITY_CHECK",
                message="Reproducibility check recommended for final results"
            )

        return validation

    def _validate_precision(
        self,
        result: CalculationResult,
        standard: str,
        validation: ValidationResult
    ) -> None:
        """Validate precision meets regulatory requirements."""
        # Determine required precision
        precision_rules = self.PRECISION_REQUIREMENTS.get(standard, {})
        required_precision = precision_rules.get('default', 2)

        # Extract formula category from formula_id
        if 'scope1' in result.formula_id.lower():
            required_precision = precision_rules.get('scope1', required_precision)
        elif 'scope2' in result.formula_id.lower():
            required_precision = precision_rules.get('scope2', required_precision)
        elif 'scope3' in result.formula_id.lower():
            required_precision = precision_rules.get('scope3', required_precision)

        # Count decimal places in result
        value_str = str(result.output_value)
        if '.' in value_str:
            decimal_places = len(value_str.split('.')[1])
        else:
            decimal_places = 0

        if decimal_places > required_precision:
            validation.add_warning(
                code="EXCESS_PRECISION",
                message=f"Result precision ({decimal_places} decimals) exceeds "
                        f"regulatory requirement ({required_precision} decimals) for {standard}",
                field="output_value",
                required=required_precision,
                actual=decimal_places
            )

    def _validate_data_quality(
        self,
        result: CalculationResult,
        validation: ValidationResult
    ) -> None:
        """Validate data quality of emission factors used."""
        low_quality_factors = [
            f for f in result.emission_factors_used
            if f.get('data_quality') == 'low'
        ]

        if low_quality_factors:
            validation.add_warning(
                code="LOW_QUALITY_FACTORS",
                message=f"{len(low_quality_factors)} emission factors have low data quality",
                count=len(low_quality_factors),
                factors=[f.get('factor_id') for f in low_quality_factors]
            )

        # Check for outdated emission factors (>5 years old)
        from datetime import datetime
        current_year = datetime.now().year

        outdated_factors = [
            f for f in result.emission_factors_used
            if current_year - f.get('source_year', current_year) > 5
        ]

        if outdated_factors:
            validation.add_warning(
                code="OUTDATED_FACTORS",
                message=f"{len(outdated_factors)} emission factors are >5 years old",
                count=len(outdated_factors),
                factors=[f.get('factor_id') for f in outdated_factors]
            )

    def _validate_materiality(
        self,
        result: CalculationResult,
        standard: str,
        validation: ValidationResult
    ) -> None:
        """Validate materiality of emissions."""
        thresholds = self.MATERIALITY_THRESHOLDS.get(standard, {})

        # Convert output to tCO2e if not already
        if result.output_unit == "kg_co2e":
            value_tco2e = result.output_value / 1000
        elif result.output_unit == "t_co2e":
            value_tco2e = result.output_value
        else:
            # Unknown unit, skip materiality check
            validation.add_info(
                code="MATERIALITY_UNKNOWN_UNIT",
                message=f"Cannot assess materiality for unit: {result.output_unit}"
            )
            return

        # Determine applicable threshold
        threshold = None
        if 'scope1' in result.formula_id.lower():
            threshold = thresholds.get('scope1')
        elif 'scope2' in result.formula_id.lower():
            threshold = thresholds.get('scope2')

        if threshold and value_tco2e < threshold:
            validation.add_info(
                code="BELOW_MATERIALITY",
                message=f"Emissions ({value_tco2e} tCO2e) below materiality threshold ({threshold} tCO2e)",
                value=float(value_tco2e),
                threshold=float(threshold)
            )

    def _validate_uncertainty(
        self,
        result: CalculationResult,
        validation: ValidationResult
    ) -> None:
        """Validate uncertainty is within acceptable range."""
        if result.uncertainty_percentage:
            # GHG Protocol recommends <10% uncertainty for Tier 1
            if result.uncertainty_percentage > 10:
                validation.add_warning(
                    code="HIGH_UNCERTAINTY",
                    message=f"Uncertainty ({result.uncertainty_percentage:.1f}%) exceeds 10% threshold",
                    uncertainty=result.uncertainty_percentage
                )
            elif result.uncertainty_percentage > 20:
                validation.add_error(
                    code="EXCESSIVE_UNCERTAINTY",
                    message=f"Uncertainty ({result.uncertainty_percentage:.1f}%) exceeds 20% - data quality too low",
                    uncertainty=result.uncertainty_percentage
                )

    def _validate_ghg_protocol(
        self,
        result: CalculationResult,
        validation: ValidationResult
    ) -> None:
        """GHG Protocol specific validations."""
        # Check emission factors have credible sources
        non_credible_sources = [
            f for f in result.emission_factors_used
            if f.get('source') not in ['DEFRA', 'EPA', 'Ecoinvent', 'IEA', 'IPCC']
        ]

        if non_credible_sources:
            validation.add_warning(
                code="GHG_NON_STANDARD_SOURCE",
                message=f"{len(non_credible_sources)} emission factors from non-standard sources",
                sources=[f.get('source') for f in non_credible_sources]
            )

    def _validate_cbam(
        self,
        result: CalculationResult,
        validation: ValidationResult
    ) -> None:
        """CBAM (Carbon Border Adjustment Mechanism) specific validations."""
        # CBAM requires 3 decimal places minimum
        value_str = str(result.output_value)
        decimal_places = len(value_str.split('.')[1]) if '.' in value_str else 0

        if decimal_places < 3:
            validation.add_error(
                code="CBAM_INSUFFICIENT_PRECISION",
                message="CBAM requires minimum 3 decimal places for embedded emissions",
                required=3,
                actual=decimal_places
            )

        # CBAM requires specific emission factor sources
        if result.emission_factors_used:
            validation.add_info(
                code="CBAM_VERIFY_SOURCES",
                message="Verify emission factors are from EU-approved sources for CBAM compliance"
            )

    def _validate_csrd(
        self,
        result: CalculationResult,
        validation: ValidationResult
    ) -> None:
        """CSRD (Corporate Sustainability Reporting Directive) specific validations."""
        # CSRD requires complete provenance
        if not result.provenance_hash:
            validation.add_error(
                code="CSRD_MISSING_PROVENANCE",
                message="CSRD requires complete audit trail with provenance hash"
            )

        # CSRD requires uncertainty disclosure
        if result.uncertainty_percentage is None:
            validation.add_warning(
                code="CSRD_MISSING_UNCERTAINTY",
                message="CSRD recommends uncertainty disclosure for emissions data"
            )


# Example usage
if __name__ == "__main__":
    from decimal import Decimal
    from .calculation_engine import CalculationResult, CalculationStep

    # Create sample result for validation
    sample_result = CalculationResult(
        formula_id="scope1_stationary_combustion",
        formula_version="1.0",
        output_value=Decimal("2690.000"),
        output_unit="kg_co2e",
        calculation_steps=[],
        provenance_hash="abc123...",
        calculation_time_ms=5.5,
        input_parameters={"fuel_quantity": 1000, "fuel_type": "diesel"},
        emission_factors_used=[
            {
                'factor_id': 'defra_2024_diesel',
                'material_or_fuel': 'diesel',
                'factor_co2e': 2.69,
                'unit': 'kg_co2e_per_liter',
                'region': 'GB',
                'source': 'DEFRA',
                'source_year': 2024,
                'data_quality': 'high'
            }
        ],
        uncertainty_percentage=5.0
    )

    # Validate
    validator = CalculationValidator()
    validation_result = validator.validate_result(sample_result, standard="ghg_protocol")

    print(f"Validation Result:")
    print(f"  Valid: {validation_result.is_valid}")
    print(f"  Errors: {validation_result.error_count}")
    print(f"  Warnings: {validation_result.warning_count}")

    for error in validation_result.errors:
        print(f"  ERROR [{error.code}]: {error.message}")

    for warning in validation_result.warnings:
        print(f"  WARNING [{warning.code}]: {warning.message}")
