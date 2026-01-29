"""
Compatibility Checking for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides functions for checking dimensional compatibility between
units, identifying missing context for conditional conversions, and returning
structured compatibility errors with GLNORM-E2xx codes.

Key Features:
    - Pure functions for deterministic compatibility checking
    - Structured error responses with GLNORM error codes
    - Context requirement detection (GWP version, reference conditions, basis)
    - Conversion path validation

All functions are pure and deterministic - they do not modify state or
rely on external mutable data.

Example:
    >>> from gl_normalizer_core.dimensions.compatibility import (
    ...     check_compatibility,
    ...     get_missing_context,
    ... )
    >>> result = check_compatibility("kWh", "MJ")
    >>> result.is_compatible
    True
    >>> result = check_compatibility("kWh", "kg")
    >>> result.is_compatible
    False
    >>> result.error_code
    'GLNORM-E202'
"""

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field

from gl_normalizer_core.errors.codes import GLNORMErrorCode
from gl_normalizer_core.dimensions.constants import (
    GL_CANONICAL_DIMENSIONS,
    CONTEXT_DEPENDENT_UNITS,
    DERIVED_DIMENSION_DEFINITIONS,
)
from gl_normalizer_core.dimensions.dimension import Dimension


class ContextRequirement(str, Enum):
    """
    Types of context that may be required for unit conversion.

    Attributes:
        REFERENCE_CONDITIONS: Temperature and pressure for volume conversions
        GWP_VERSION: Global Warming Potential version for CO2e conversions
        ENERGY_BASIS: HHV/LHV basis for energy content conversions
        NONE: No additional context required
    """

    REFERENCE_CONDITIONS = "reference_conditions"
    GWP_VERSION = "gwp_version"
    ENERGY_BASIS = "energy_basis"
    NONE = "none"


class MissingContextInfo(BaseModel):
    """
    Information about missing context required for conversion.

    Attributes:
        requirement: Type of context requirement
        field_name: Name of the field that should be provided
        description: Human-readable description of the requirement
        example_values: Example valid values for the field
        documentation_url: Link to documentation about this requirement
    """

    requirement: ContextRequirement = Field(
        ...,
        description="Type of context requirement",
    )
    field_name: str = Field(
        ...,
        description="Name of the field that should be provided",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the requirement",
    )
    example_values: List[str] = Field(
        default_factory=list,
        description="Example valid values for the field",
    )
    documentation_url: Optional[str] = Field(
        default=None,
        description="Link to documentation about this requirement",
    )

    model_config = {"frozen": True}


class CompatibilityResult(BaseModel):
    """
    Result of a compatibility check between two units.

    Attributes:
        is_compatible: Whether the units are dimensionally compatible
        source_unit: The source unit string
        target_unit: The target unit string
        source_dimension: Dimension of the source unit
        target_dimension: Dimension of the target unit
        error_code: GLNORM error code if incompatible
        error_message: Human-readable error message if incompatible
        missing_context: List of missing context requirements
        hint: Suggestion for resolving incompatibility
        details: Additional details about the compatibility check
    """

    is_compatible: bool = Field(
        ...,
        description="Whether the units are dimensionally compatible",
    )
    source_unit: str = Field(
        ...,
        description="The source unit string",
    )
    target_unit: str = Field(
        ...,
        description="The target unit string",
    )
    source_dimension: Optional[Dict[str, int]] = Field(
        default=None,
        description="Dimension exponents of the source unit",
    )
    target_dimension: Optional[Dict[str, int]] = Field(
        default=None,
        description="Dimension exponents of the target unit",
    )
    error_code: Optional[str] = Field(
        default=None,
        description="GLNORM error code if incompatible",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Human-readable error message if incompatible",
    )
    missing_context: List[MissingContextInfo] = Field(
        default_factory=list,
        description="List of missing context requirements",
    )
    hint: Optional[str] = Field(
        default=None,
        description="Suggestion for resolving incompatibility",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the compatibility check",
    )

    model_config = {"frozen": True}


class CompatibilityError(BaseModel):
    """
    Structured error for compatibility failures.

    This model follows the GLNORM-E2xx error code taxonomy for
    dimension-related errors.

    Attributes:
        code: GLNORM error code (e.g., GLNORM-E200)
        message: Human-readable error message
        source_unit: The source unit that failed
        target_unit: The target unit that failed
        source_dimension: Dimension of the source unit
        target_dimension: Dimension of the target unit
        expected: Expected dimension or condition
        actual: Actual dimension or condition found
        hint: Suggestion for resolving the error
        documentation_url: Link to relevant documentation
    """

    code: str = Field(
        ...,
        pattern=r"^GLNORM-E2\d{2}$",
        description="GLNORM error code for dimension errors",
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
    )
    source_unit: str = Field(
        ...,
        description="The source unit that failed",
    )
    target_unit: str = Field(
        ...,
        description="The target unit that failed",
    )
    source_dimension: Optional[str] = Field(
        default=None,
        description="String representation of source dimension",
    )
    target_dimension: Optional[str] = Field(
        default=None,
        description="String representation of target dimension",
    )
    expected: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Expected dimension or condition",
    )
    actual: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Actual dimension or condition found",
    )
    hint: Optional[str] = Field(
        default=None,
        description="Suggestion for resolving the error",
    )
    documentation_url: Optional[str] = Field(
        default="https://docs.greenlang.io/normalizer/dimensions",
        description="Link to documentation",
    )

    model_config = {"frozen": True}


# =============================================================================
# Pure Functions for Compatibility Checking
# =============================================================================


def get_unit_dimension(unit_str: str) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """
    Get the dimension exponents for a unit string.

    This is a pure function that looks up the dimension from the
    GL_CANONICAL_DIMENSIONS registry.

    Args:
        unit_str: Unit string to look up

    Returns:
        Tuple of (dimension exponents dict, error message if not found)

    Example:
        >>> exponents, error = get_unit_dimension("kWh")
        >>> print(exponents)
        {'mass': 1, 'length': 2, 'time': -2}
    """
    # Normalize unit string
    normalized = unit_str.strip()

    # Direct lookup
    if normalized in GL_CANONICAL_DIMENSIONS:
        return dict(GL_CANONICAL_DIMENSIONS[normalized]), None

    # Try common variations
    variations = [
        normalized.replace("^", ""),  # m^2 -> m2
        normalized.replace(" ", ""),  # kg CO2e -> kgCO2e
        normalized.replace("-", ""),  # kg-CO2e -> kgCO2e
    ]

    for var in variations:
        if var in GL_CANONICAL_DIMENSIONS:
            return dict(GL_CANONICAL_DIMENSIONS[var]), None

    return None, f"Unknown unit '{unit_str}'"


def check_compatibility(
    source_unit: str,
    target_unit: str,
) -> CompatibilityResult:
    """
    Check if two units are dimensionally compatible for conversion.

    This is a pure function that determines whether a conversion
    between two units is possible based on their dimensions.

    Args:
        source_unit: Source unit string
        target_unit: Target unit string

    Returns:
        CompatibilityResult with compatibility status and any errors

    Example:
        >>> result = check_compatibility("kWh", "MJ")
        >>> result.is_compatible
        True
        >>> result = check_compatibility("kWh", "kg")
        >>> result.is_compatible
        False
        >>> result.error_code
        'GLNORM-E202'
    """
    # Get source dimension
    source_dim, source_error = get_unit_dimension(source_unit)
    if source_error:
        return CompatibilityResult(
            is_compatible=False,
            source_unit=source_unit,
            target_unit=target_unit,
            source_dimension=None,
            target_dimension=None,
            error_code=GLNORMErrorCode.E201_DIMENSION_UNKNOWN.value,
            error_message=f"Cannot determine dimension for source unit: {source_error}",
            hint=f"Check that '{source_unit}' is a valid unit in the registry",
        )

    # Get target dimension
    target_dim, target_error = get_unit_dimension(target_unit)
    if target_error:
        return CompatibilityResult(
            is_compatible=False,
            source_unit=source_unit,
            target_unit=target_unit,
            source_dimension=source_dim,
            target_dimension=None,
            error_code=GLNORMErrorCode.E201_DIMENSION_UNKNOWN.value,
            error_message=f"Cannot determine dimension for target unit: {target_error}",
            hint=f"Check that '{target_unit}' is a valid unit in the registry",
        )

    # Compare dimensions
    if source_dim == target_dim:
        # Check for missing context
        missing_context = get_missing_context(source_unit, target_unit)

        return CompatibilityResult(
            is_compatible=True,
            source_unit=source_unit,
            target_unit=target_unit,
            source_dimension=source_dim,
            target_dimension=target_dim,
            missing_context=missing_context,
            details={
                "dimension_match": True,
                "context_required": len(missing_context) > 0,
            },
        )

    # Dimensions don't match - create detailed error
    source_dim_obj = Dimension.from_exponents(source_dim)
    target_dim_obj = Dimension.from_exponents(target_dim)

    return CompatibilityResult(
        is_compatible=False,
        source_unit=source_unit,
        target_unit=target_unit,
        source_dimension=source_dim,
        target_dimension=target_dim,
        error_code=GLNORMErrorCode.E202_DIMENSION_INCOMPATIBLE.value,
        error_message=(
            f"Dimensions are incompatible: "
            f"'{source_unit}' has dimension {source_dim_obj}, "
            f"'{target_unit}' has dimension {target_dim_obj}"
        ),
        hint=_get_incompatibility_hint(source_dim_obj, target_dim_obj),
        details={
            "source_display": str(source_dim_obj),
            "target_display": str(target_dim_obj),
        },
    )


def are_compatible(unit1: str, unit2: str) -> bool:
    """
    Simple boolean check if two units are dimensionally compatible.

    This is a convenience function that wraps check_compatibility.

    Args:
        unit1: First unit string
        unit2: Second unit string

    Returns:
        True if units are compatible, False otherwise

    Example:
        >>> are_compatible("kWh", "MJ")
        True
        >>> are_compatible("kWh", "kg")
        False
    """
    result = check_compatibility(unit1, unit2)
    return result.is_compatible


def get_missing_context(
    source_unit: str,
    target_unit: str,
) -> List[MissingContextInfo]:
    """
    Identify missing context required for conditional conversions.

    Some conversions require additional context such as:
    - Reference conditions (temperature/pressure) for Nm3, scf
    - GWP version for CO2e conversions
    - Energy basis (HHV/LHV) for energy content conversions

    Args:
        source_unit: Source unit string
        target_unit: Target unit string

    Returns:
        List of missing context requirements

    Example:
        >>> missing = get_missing_context("kgCH4", "kgCO2e")
        >>> len(missing) > 0
        True
        >>> missing[0].requirement
        <ContextRequirement.GWP_VERSION: 'gwp_version'>
    """
    missing: List[MissingContextInfo] = []

    # Check source unit for context requirements
    if source_unit in CONTEXT_DEPENDENT_UNITS:
        req_type = CONTEXT_DEPENDENT_UNITS[source_unit]
        missing.append(_create_context_requirement(req_type, source_unit, "source"))

    # Check target unit for context requirements
    if target_unit in CONTEXT_DEPENDENT_UNITS:
        req_type = CONTEXT_DEPENDENT_UNITS[target_unit]
        # Avoid duplicates
        existing_types = {m.requirement.value for m in missing}
        if req_type not in existing_types:
            missing.append(_create_context_requirement(req_type, target_unit, "target"))

    return missing


def _create_context_requirement(
    req_type: str,
    unit: str,
    context: str,
) -> MissingContextInfo:
    """
    Create a MissingContextInfo for a context requirement type.

    Args:
        req_type: The type of requirement (from CONTEXT_DEPENDENT_UNITS)
        unit: The unit that requires this context
        context: "source" or "target"

    Returns:
        MissingContextInfo instance
    """
    if req_type == "reference_conditions":
        return MissingContextInfo(
            requirement=ContextRequirement.REFERENCE_CONDITIONS,
            field_name="reference_conditions",
            description=(
                f"Reference temperature and pressure required for {context} unit '{unit}'. "
                "Volume-based units like Nm3 and scf require standard conditions."
            ),
            example_values=[
                '{"temperature_C": 0, "pressure_kPa": 101.325}',
                '{"temperature_C": 15, "pressure_kPa": 101.325}',
                '{"temperature_C": 20, "pressure_kPa": 101.325}',
            ],
            documentation_url="https://docs.greenlang.io/normalizer/reference-conditions",
        )

    elif req_type == "gwp_version":
        return MissingContextInfo(
            requirement=ContextRequirement.GWP_VERSION,
            field_name="gwp_version",
            description=(
                f"Global Warming Potential (GWP) version required for {context} unit '{unit}'. "
                "CO2e conversions depend on the IPCC assessment report version."
            ),
            example_values=["AR5", "AR6", "AR4"],
            documentation_url="https://docs.greenlang.io/normalizer/gwp",
        )

    elif req_type == "energy_basis":
        return MissingContextInfo(
            requirement=ContextRequirement.ENERGY_BASIS,
            field_name="energy_basis",
            description=(
                f"Energy basis (HHV or LHV) required for {context} unit '{unit}'. "
                "Energy content values differ based on measurement basis."
            ),
            example_values=["HHV", "LHV", "NCV", "GCV"],
            documentation_url="https://docs.greenlang.io/normalizer/energy-basis",
        )

    # Default case
    return MissingContextInfo(
        requirement=ContextRequirement.NONE,
        field_name="unknown",
        description=f"Unknown context requirement '{req_type}' for unit '{unit}'",
    )


def _get_incompatibility_hint(
    source_dim: Dimension,
    target_dim: Dimension,
) -> str:
    """
    Generate a helpful hint for dimension incompatibility.

    Args:
        source_dim: Source Dimension object
        target_dim: Target Dimension object

    Returns:
        Human-readable hint string
    """
    # Find matching derived dimensions for source and target
    source_match = _find_dimension_name(source_dim)
    target_match = _find_dimension_name(target_dim)

    if source_match and target_match:
        return (
            f"Cannot convert between {source_match} ({source_dim}) and "
            f"{target_match} ({target_dim}). These are fundamentally different "
            f"physical quantities."
        )

    return (
        f"Source dimension {source_dim} is not compatible with "
        f"target dimension {target_dim}. Units must have the same "
        f"physical dimension for conversion."
    )


def _find_dimension_name(dim: Dimension) -> Optional[str]:
    """
    Find the name of a dimension from the derived definitions.

    Args:
        dim: Dimension to look up

    Returns:
        Name if found, None otherwise
    """
    # Check if it's a named dimension
    if dim.name:
        return dim.name

    # Search in derived definitions
    for name, exponents in DERIVED_DIMENSION_DEFINITIONS.items():
        if dim.exponents == exponents:
            return name

    # Check base dimensions
    if dim.is_base and len(dim.exponents) == 1:
        return list(dim.exponents.keys())[0]

    return None


def create_compatibility_error(
    source_unit: str,
    target_unit: str,
    error_code: GLNORMErrorCode,
    message: Optional[str] = None,
) -> CompatibilityError:
    """
    Create a structured CompatibilityError.

    Factory function for creating compatibility errors with proper
    GLNORM-E2xx codes and all required fields.

    Args:
        source_unit: Source unit string
        target_unit: Target unit string
        error_code: GLNORM error code (must be E2xx)
        message: Optional custom message (auto-generated if not provided)

    Returns:
        CompatibilityError instance

    Example:
        >>> error = create_compatibility_error(
        ...     "kWh", "kg",
        ...     GLNORMErrorCode.E202_DIMENSION_INCOMPATIBLE
        ... )
        >>> error.code
        'GLNORM-E202'
    """
    source_dim, _ = get_unit_dimension(source_unit)
    target_dim, _ = get_unit_dimension(target_unit)

    source_dim_str = None
    target_dim_str = None

    if source_dim:
        source_dim_str = str(Dimension.from_exponents(source_dim))
    if target_dim:
        target_dim_str = str(Dimension.from_exponents(target_dim))

    if message is None:
        message = _generate_error_message(error_code, source_unit, target_unit)

    return CompatibilityError(
        code=error_code.value,
        message=message,
        source_unit=source_unit,
        target_unit=target_unit,
        source_dimension=source_dim_str,
        target_dimension=target_dim_str,
        expected={"dimension": target_dim_str} if target_dim_str else None,
        actual={"dimension": source_dim_str} if source_dim_str else None,
        hint=_get_error_hint(error_code, source_unit, target_unit),
    )


def _generate_error_message(
    error_code: GLNORMErrorCode,
    source_unit: str,
    target_unit: str,
) -> str:
    """Generate an error message based on the error code."""
    messages = {
        GLNORMErrorCode.E200_DIMENSION_MISMATCH: (
            f"Dimension mismatch: cannot convert '{source_unit}' to '{target_unit}'"
        ),
        GLNORMErrorCode.E201_DIMENSION_UNKNOWN: (
            f"Cannot determine dimension for unit '{source_unit}' or '{target_unit}'"
        ),
        GLNORMErrorCode.E202_DIMENSION_INCOMPATIBLE: (
            f"Dimensions incompatible: '{source_unit}' and '{target_unit}' "
            f"have different physical dimensions"
        ),
        GLNORMErrorCode.E203_DIMENSIONLESS_EXPECTED: (
            f"Expected dimensionless quantity but got '{source_unit}'"
        ),
        GLNORMErrorCode.E204_DIMENSION_EXPECTED: (
            f"Expected dimensioned quantity but got dimensionless"
        ),
    }
    return messages.get(error_code, f"Dimension error for '{source_unit}' to '{target_unit}'")


def _get_error_hint(
    error_code: GLNORMErrorCode,
    source_unit: str,
    target_unit: str,
) -> str:
    """Generate a helpful hint based on the error code."""
    hints = {
        GLNORMErrorCode.E200_DIMENSION_MISMATCH: (
            f"Ensure source and target units have the same dimension. "
            f"For example, both should be energy units (kWh, MJ) or mass units (kg, t)."
        ),
        GLNORMErrorCode.E201_DIMENSION_UNKNOWN: (
            f"Check that the unit is registered in the GreenLang unit registry. "
            f"Common units include: kWh, MJ, kg, m3, kgCO2e."
        ),
        GLNORMErrorCode.E202_DIMENSION_INCOMPATIBLE: (
            f"These units represent different physical quantities and cannot be converted. "
            f"Verify the expected_dimension matches your input."
        ),
        GLNORMErrorCode.E203_DIMENSIONLESS_EXPECTED: (
            f"Use a dimensionless quantity like %, ppm, or a ratio."
        ),
        GLNORMErrorCode.E204_DIMENSION_EXPECTED: (
            f"Provide a unit with the expected dimension."
        ),
    }
    return hints.get(error_code, "Check unit dimensions and try again.")


def validate_expected_dimension(
    unit_str: str,
    expected_dimension: str,
) -> CompatibilityResult:
    """
    Validate that a unit has the expected dimension.

    This function is used by the normalizer to verify that input units
    match the expected dimension from the schema context.

    Args:
        unit_str: Unit string to validate
        expected_dimension: Expected dimension name (e.g., "energy", "mass")

    Returns:
        CompatibilityResult indicating if the unit has the expected dimension

    Example:
        >>> result = validate_expected_dimension("kWh", "energy")
        >>> result.is_compatible
        True
        >>> result = validate_expected_dimension("kg", "energy")
        >>> result.is_compatible
        False
    """
    # Get the unit's dimension
    unit_dim, error = get_unit_dimension(unit_str)
    if error:
        return CompatibilityResult(
            is_compatible=False,
            source_unit=unit_str,
            target_unit=expected_dimension,
            error_code=GLNORMErrorCode.E201_DIMENSION_UNKNOWN.value,
            error_message=error,
            hint=f"Check that '{unit_str}' is a valid unit",
        )

    # Get the expected dimension
    try:
        expected_dim_obj = Dimension.from_name(expected_dimension)
    except ValueError as e:
        return CompatibilityResult(
            is_compatible=False,
            source_unit=unit_str,
            target_unit=expected_dimension,
            error_code=GLNORMErrorCode.E201_DIMENSION_UNKNOWN.value,
            error_message=f"Unknown expected dimension: {expected_dimension}",
            hint=str(e),
        )

    # Compare
    unit_dim_obj = Dimension.from_exponents(unit_dim)

    if unit_dim_obj.exponents == expected_dim_obj.exponents:
        return CompatibilityResult(
            is_compatible=True,
            source_unit=unit_str,
            target_unit=expected_dimension,
            source_dimension=unit_dim,
            target_dimension=expected_dim_obj.exponents,
        )

    return CompatibilityResult(
        is_compatible=False,
        source_unit=unit_str,
        target_unit=expected_dimension,
        source_dimension=unit_dim,
        target_dimension=dict(expected_dim_obj.exponents),
        error_code=GLNORMErrorCode.E200_DIMENSION_MISMATCH.value,
        error_message=(
            f"Dimension mismatch: unit '{unit_str}' has dimension {unit_dim_obj}, "
            f"but expected '{expected_dimension}' ({expected_dim_obj})"
        ),
        hint=(
            f"Use a unit with dimension '{expected_dimension}'. "
            f"Example units: {_get_example_units_for_dimension(expected_dimension)}"
        ),
    )


def _get_example_units_for_dimension(dimension_name: str) -> str:
    """Get example units for a dimension name."""
    examples = {
        "energy": "kWh, MJ, GJ, BTU",
        "mass": "kg, g, t, lb",
        "volume": "m3, L, gal, bbl",
        "power": "kW, MW, W",
        "emissions": "kgCO2e, tCO2e, gCO2e",
        "length": "m, km, ft, mi",
        "time": "s, min, h, day",
        "temperature": "K, degC, degF",
        "pressure": "Pa, kPa, bar, psi",
    }
    return examples.get(dimension_name, "check documentation for valid units")
