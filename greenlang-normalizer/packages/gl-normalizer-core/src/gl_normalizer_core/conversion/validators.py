"""
Conversion validation functions for GL-FOUND-X-003 Unit Conversion Engine.

This module provides pure, deterministic validation functions for unit
conversions in sustainability reporting. All functions have no side effects
and no I/O operations.

Validation capabilities include:
- Conversion path validation (checking if conversion is supported)
- Reference condition validation (temperature, pressure ranges)
- GWP version validation for CO2e conversions
- Basis validation for energy conversions
- Dimension compatibility validation

Example:
    >>> from gl_normalizer_core.conversion.validators import validate_gwp_version
    >>> result = validate_gwp_version("AR6")
    >>> print(result.is_valid)
    True
"""

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

from gl_normalizer_core.errors.codes import GLNORMErrorCode
from gl_normalizer_core.conversion.contexts import (
    ConversionContext,
    GWPVersion,
    EnergyBasis,
    MassBasis,
    PressureMode,
)


# Valid GWP versions
VALID_GWP_VERSIONS: FrozenSet[str] = frozenset({v.value for v in GWPVersion})

# Valid energy bases
VALID_ENERGY_BASES: FrozenSet[str] = frozenset({v.value for v in EnergyBasis})

# Valid mass bases
VALID_MASS_BASES: FrozenSet[str] = frozenset({v.value for v in MassBasis})

# Valid pressure modes
VALID_PRESSURE_MODES: FrozenSet[str] = frozenset({v.value for v in PressureMode})

# GL Canonical Units by dimension
GL_CANONICAL_UNITS: dict = {
    "energy": "MJ",
    "mass": "kg",
    "emissions": "kgCO2e",
    "volume": "m3",
    "pressure": "kPa_abs",
    "temperature_celsius": "degC",
    "temperature_absolute": "K",
}

# Units requiring reference conditions
UNITS_REQUIRING_REFERENCE_CONDITIONS: FrozenSet[str] = frozenset({
    "Nm3",  # Normal cubic meter
    "Sm3",  # Standard cubic meter
    "scf",  # Standard cubic feet
    "NCM",  # Normal cubic meter
    "SCM",  # Standard cubic meter
})

# Units requiring GWP version
UNITS_REQUIRING_GWP: FrozenSet[str] = frozenset({
    "kgCO2e",
    "tCO2e",
    "gCO2e",
    "lbCO2e",
    "kg_CO2e",
    "t_CO2e",
    "g_CO2e",
    "lb_CO2e",
    "kgCO2eq",
    "tCO2eq",
})

# Units requiring basis (LHV/HHV)
UNITS_REQUIRING_BASIS: FrozenSet[str] = frozenset({
    "MJ_LHV",
    "MJ_HHV",
    "GJ_LHV",
    "GJ_HHV",
    "kWh_LHV",
    "kWh_HHV",
    "BTU_LHV",
    "BTU_HHV",
    "therm_LHV",
    "therm_HHV",
})

# Physical limits for validation
MIN_TEMPERATURE_C: float = -273.15  # Absolute zero in Celsius
MAX_TEMPERATURE_C: float = 10000.0  # Reasonable upper limit
MIN_PRESSURE_KPA: float = 0.0  # Zero pressure
MAX_PRESSURE_KPA: float = 1000000.0  # 10,000 atm (reasonable limit)

# Dimension compatibility mapping
COMPATIBLE_DIMENSIONS: dict = {
    "energy": {"energy", "heat", "work"},
    "mass": {"mass"},
    "volume": {"volume"},
    "temperature": {"temperature", "temperature_difference"},
    "pressure": {"pressure"},
    "length": {"length", "distance"},
    "time": {"time", "duration"},
    "emissions": {"emissions", "mass"},  # CO2e can convert to mass
}


@dataclass(frozen=True)
class ValidationResult:
    """
    Immutable result of a validation operation.

    Attributes:
        is_valid: Whether the validation passed.
        error_code: Error code if validation failed.
        error_message: Detailed error message if validation failed.
        warnings: List of warning messages (non-fatal issues).
        context: Additional context information.
    """

    is_valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    context: Optional[dict] = None

    @classmethod
    def success(
        cls,
        warnings: Optional[List[str]] = None,
    ) -> "ValidationResult":
        """
        Create a successful validation result.

        Args:
            warnings: Optional list of warning messages.

        Returns:
            ValidationResult indicating success.
        """
        return cls(
            is_valid=True,
            warnings=tuple(warnings) if warnings else (),
        )

    @classmethod
    def failure(
        cls,
        error_code: str,
        error_message: str,
        warnings: Optional[List[str]] = None,
        context: Optional[dict] = None,
    ) -> "ValidationResult":
        """
        Create a failed validation result.

        Args:
            error_code: Error code from GLNORMErrorCode.
            error_message: Detailed error message.
            warnings: Optional list of warning messages.
            context: Optional additional context.

        Returns:
            ValidationResult indicating failure.
        """
        return cls(
            is_valid=False,
            error_code=error_code,
            error_message=error_message,
            warnings=tuple(warnings) if warnings else (),
            context=context,
        )


def validate_gwp_version(gwp_version: Optional[str]) -> ValidationResult:
    """
    Validate a GWP version string.

    Args:
        gwp_version: GWP version to validate (AR4, AR5, AR6).

    Returns:
        ValidationResult indicating whether the version is valid.

    Example:
        >>> result = validate_gwp_version("AR6")
        >>> print(result.is_valid)
        True
        >>> result = validate_gwp_version("AR7")
        >>> print(result.is_valid)
        False
    """
    if gwp_version is None:
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E305_GWP_VERSION_MISSING.value,
            error_message="GWP version is required but not provided.",
            context={"valid_versions": list(VALID_GWP_VERSIONS)},
        )

    if gwp_version not in VALID_GWP_VERSIONS:
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E305_GWP_VERSION_MISSING.value,
            error_message=(
                f"Invalid GWP version '{gwp_version}'. "
                f"Must be one of: {', '.join(sorted(VALID_GWP_VERSIONS))}"
            ),
            context={
                "provided": gwp_version,
                "valid_versions": list(VALID_GWP_VERSIONS),
            },
        )

    warnings = []
    if gwp_version == GWPVersion.AR4.value:
        warnings.append(
            "GWP version AR4 is outdated. Consider using AR6 for current compliance."
        )

    return ValidationResult.success(warnings=warnings if warnings else None)


def validate_basis(basis: Optional[str]) -> ValidationResult:
    """
    Validate an energy basis string.

    Args:
        basis: Energy basis to validate (LHV or HHV).

    Returns:
        ValidationResult indicating whether the basis is valid.

    Example:
        >>> result = validate_basis("LHV")
        >>> print(result.is_valid)
        True
    """
    if basis is None:
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E306_BASIS_MISSING.value,
            error_message="Energy basis is required but not provided.",
            context={"valid_bases": list(VALID_ENERGY_BASES)},
        )

    if basis not in VALID_ENERGY_BASES:
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E306_BASIS_MISSING.value,
            error_message=(
                f"Invalid energy basis '{basis}'. "
                f"Must be one of: {', '.join(sorted(VALID_ENERGY_BASES))}"
            ),
            context={
                "provided": basis,
                "valid_bases": list(VALID_ENERGY_BASES),
            },
        )

    return ValidationResult.success()


def validate_reference_conditions(
    temperature_c: Optional[float],
    pressure_kpa: Optional[float],
) -> ValidationResult:
    """
    Validate reference conditions for gas volume conversions.

    Args:
        temperature_c: Reference temperature in degrees Celsius.
        pressure_kpa: Reference pressure in kPa (absolute).

    Returns:
        ValidationResult indicating whether the conditions are valid.

    Example:
        >>> result = validate_reference_conditions(15.0, 101.325)
        >>> print(result.is_valid)
        True
    """
    warnings = []

    # Check if both are provided
    if temperature_c is None or pressure_kpa is None:
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E301_MISSING_REFERENCE_CONDITIONS.value,
            error_message=(
                "Reference conditions (temperature and pressure) are required "
                "for gas volume conversions but were not fully provided."
            ),
            context={
                "temperature_provided": temperature_c is not None,
                "pressure_provided": pressure_kpa is not None,
            },
        )

    # Validate temperature range
    if temperature_c < MIN_TEMPERATURE_C:
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E302_INVALID_REFERENCE_CONDITIONS.value,
            error_message=(
                f"Reference temperature {temperature_c} C is below absolute zero "
                f"({MIN_TEMPERATURE_C} C)."
            ),
            context={
                "temperature_c": temperature_c,
                "minimum": MIN_TEMPERATURE_C,
            },
        )

    if temperature_c > MAX_TEMPERATURE_C:
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E302_INVALID_REFERENCE_CONDITIONS.value,
            error_message=(
                f"Reference temperature {temperature_c} C exceeds reasonable limit "
                f"({MAX_TEMPERATURE_C} C)."
            ),
            context={
                "temperature_c": temperature_c,
                "maximum": MAX_TEMPERATURE_C,
            },
        )

    # Validate pressure range
    if pressure_kpa <= MIN_PRESSURE_KPA:
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E302_INVALID_REFERENCE_CONDITIONS.value,
            error_message=(
                f"Reference pressure {pressure_kpa} kPa must be positive."
            ),
            context={
                "pressure_kpa": pressure_kpa,
                "minimum": MIN_PRESSURE_KPA,
            },
        )

    if pressure_kpa > MAX_PRESSURE_KPA:
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E302_INVALID_REFERENCE_CONDITIONS.value,
            error_message=(
                f"Reference pressure {pressure_kpa} kPa exceeds reasonable limit "
                f"({MAX_PRESSURE_KPA} kPa)."
            ),
            context={
                "pressure_kpa": pressure_kpa,
                "maximum": MAX_PRESSURE_KPA,
            },
        )

    # Add warnings for unusual conditions
    if temperature_c < -50 or temperature_c > 100:
        warnings.append(
            f"Reference temperature {temperature_c} C is outside typical range (-50 to 100 C)."
        )

    if pressure_kpa < 50 or pressure_kpa > 200:
        warnings.append(
            f"Reference pressure {pressure_kpa} kPa is outside typical range (50 to 200 kPa)."
        )

    return ValidationResult.success(warnings=warnings if warnings else None)


def validate_conversion_path(
    from_unit: str,
    to_unit: str,
    context: Optional[ConversionContext] = None,
) -> ValidationResult:
    """
    Validate that a conversion path exists between two units.

    This function checks whether a conversion between two units is possible,
    considering the provided context (GWP version, basis, reference conditions).

    Args:
        from_unit: Source unit string.
        to_unit: Target unit string.
        context: Optional conversion context with required parameters.

    Returns:
        ValidationResult indicating whether the conversion is valid.

    Example:
        >>> result = validate_conversion_path("kWh", "MJ")
        >>> print(result.is_valid)
        True
    """
    warnings = []

    # Check if units are the same (trivial conversion)
    if from_unit == to_unit:
        return ValidationResult.success()

    # Check if source unit requires reference conditions
    if _unit_requires_reference_conditions(from_unit) or _unit_requires_reference_conditions(to_unit):
        if context is None or not context.has_reference_conditions:
            return ValidationResult.failure(
                error_code=GLNORMErrorCode.E301_MISSING_REFERENCE_CONDITIONS.value,
                error_message=(
                    f"Conversion from '{from_unit}' to '{to_unit}' requires "
                    "reference conditions (temperature and pressure)."
                ),
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "requires": "reference_conditions",
                },
            )

    # Check if units require GWP version
    if _unit_requires_gwp(from_unit) or _unit_requires_gwp(to_unit):
        if context is None or not context.has_gwp_version:
            return ValidationResult.failure(
                error_code=GLNORMErrorCode.E305_GWP_VERSION_MISSING.value,
                error_message=(
                    f"Conversion involving CO2e units ('{from_unit}' to '{to_unit}') "
                    "requires GWP version to be specified in context."
                ),
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "requires": "gwp_version",
                },
            )

    # Check if units require basis
    if _unit_requires_basis(from_unit) or _unit_requires_basis(to_unit):
        if context is None or not context.has_basis:
            return ValidationResult.failure(
                error_code=GLNORMErrorCode.E306_BASIS_MISSING.value,
                error_message=(
                    f"Conversion involving basis-specific units ('{from_unit}' to '{to_unit}') "
                    "requires energy basis (LHV/HHV) to be specified in context."
                ),
                context={
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "requires": "basis",
                },
            )

    return ValidationResult.success(warnings=warnings if warnings else None)


def validate_context_for_conversion(
    from_unit: str,
    to_unit: str,
    context: ConversionContext,
) -> ValidationResult:
    """
    Validate that a context has all required parameters for a specific conversion.

    This is a comprehensive validation that checks all context parameters
    that may be required for a given conversion path.

    Args:
        from_unit: Source unit string.
        to_unit: Target unit string.
        context: Conversion context to validate.

    Returns:
        ValidationResult with all validation errors and warnings.

    Example:
        >>> context = ConversionContext(gwp_version="AR6")
        >>> result = validate_context_for_conversion("kgCH4", "kgCO2e", context)
        >>> print(result.is_valid)
        True
    """
    all_warnings = []
    errors = []

    # Check reference conditions if needed
    if _unit_requires_reference_conditions(from_unit) or _unit_requires_reference_conditions(to_unit):
        ref_result = validate_reference_conditions(
            context.temperature_ref,
            context.pressure_ref,
        )
        if not ref_result.is_valid:
            errors.append(ref_result.error_message)
        if ref_result.warnings:
            all_warnings.extend(ref_result.warnings)

    # Check GWP version if needed
    if _unit_requires_gwp(from_unit) or _unit_requires_gwp(to_unit):
        gwp_result = validate_gwp_version(context.gwp_version)
        if not gwp_result.is_valid:
            errors.append(gwp_result.error_message)
        if gwp_result.warnings:
            all_warnings.extend(gwp_result.warnings)

    # Check basis if needed
    if _unit_requires_basis(from_unit) or _unit_requires_basis(to_unit):
        basis_result = validate_basis(context.basis)
        if not basis_result.is_valid:
            errors.append(basis_result.error_message)

    if errors:
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E300_CONVERSION_NOT_SUPPORTED.value,
            error_message="; ".join(errors),
            warnings=all_warnings,
            context={
                "from_unit": from_unit,
                "to_unit": to_unit,
                "errors": errors,
            },
        )

    return ValidationResult.success(warnings=all_warnings if all_warnings else None)


def validate_dimension_compatibility(
    source_dimension: str,
    target_dimension: str,
) -> ValidationResult:
    """
    Validate that two dimensions are compatible for conversion.

    Args:
        source_dimension: Dimension of the source unit.
        target_dimension: Dimension of the target unit.

    Returns:
        ValidationResult indicating whether the dimensions are compatible.

    Example:
        >>> result = validate_dimension_compatibility("energy", "heat")
        >>> print(result.is_valid)
        True
    """
    # Same dimension is always compatible
    if source_dimension == target_dimension:
        return ValidationResult.success()

    # Check compatibility mapping
    source_compat = COMPATIBLE_DIMENSIONS.get(source_dimension, {source_dimension})
    if target_dimension in source_compat:
        return ValidationResult.success()

    target_compat = COMPATIBLE_DIMENSIONS.get(target_dimension, {target_dimension})
    if source_dimension in target_compat:
        return ValidationResult.success()

    return ValidationResult.failure(
        error_code=GLNORMErrorCode.E200_DIMENSION_MISMATCH.value,
        error_message=(
            f"Dimensions '{source_dimension}' and '{target_dimension}' are not compatible. "
            "Cannot convert between these unit types."
        ),
        context={
            "source_dimension": source_dimension,
            "target_dimension": target_dimension,
            "source_compatible_with": list(source_compat),
            "target_compatible_with": list(target_compat),
        },
    )


def validate_numeric_value(value: float) -> ValidationResult:
    """
    Validate that a numeric value is suitable for conversion.

    Args:
        value: Numeric value to validate.

    Returns:
        ValidationResult indicating whether the value is valid.

    Example:
        >>> result = validate_numeric_value(100.0)
        >>> print(result.is_valid)
        True
    """
    import math

    if math.isnan(value):
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E304_PRECISION_OVERFLOW.value,
            error_message="Value is NaN (not a number).",
            context={"value": "NaN"},
        )

    if math.isinf(value):
        return ValidationResult.failure(
            error_code=GLNORMErrorCode.E304_PRECISION_OVERFLOW.value,
            error_message=f"Value is infinite: {value}",
            context={"value": str(value)},
        )

    return ValidationResult.success()


def _unit_requires_reference_conditions(unit: str) -> bool:
    """
    Check if a unit requires reference conditions.

    Args:
        unit: Unit string to check.

    Returns:
        True if the unit requires reference conditions.
    """
    # Normalize unit for comparison
    unit_normalized = unit.replace("_", "").replace("-", "").upper()
    for ref_unit in UNITS_REQUIRING_REFERENCE_CONDITIONS:
        if ref_unit.replace("_", "").replace("-", "").upper() in unit_normalized:
            return True
    return False


def _unit_requires_gwp(unit: str) -> bool:
    """
    Check if a unit requires GWP version.

    Args:
        unit: Unit string to check.

    Returns:
        True if the unit requires GWP version.
    """
    # Check for CO2e indicators
    unit_upper = unit.upper()
    return "CO2E" in unit_upper or "CO2EQ" in unit_upper


def _unit_requires_basis(unit: str) -> bool:
    """
    Check if a unit requires energy basis.

    Args:
        unit: Unit string to check.

    Returns:
        True if the unit requires energy basis.
    """
    unit_upper = unit.upper()
    return "_LHV" in unit_upper or "_HHV" in unit_upper


__all__ = [
    "ValidationResult",
    "validate_gwp_version",
    "validate_basis",
    "validate_reference_conditions",
    "validate_conversion_path",
    "validate_context_for_conversion",
    "validate_dimension_compatibility",
    "validate_numeric_value",
    "VALID_GWP_VERSIONS",
    "VALID_ENERGY_BASES",
    "VALID_MASS_BASES",
    "VALID_PRESSURE_MODES",
    "GL_CANONICAL_UNITS",
    "UNITS_REQUIRING_REFERENCE_CONDITIONS",
    "UNITS_REQUIRING_GWP",
    "UNITS_REQUIRING_BASIS",
]
