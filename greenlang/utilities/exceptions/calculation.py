"""
GreenLang Calculation Exceptions - Emission and Measurement Calculation Errors

This module provides exception classes for calculation-related errors
in the zero-hallucination calculation engine including emission calculations,
unit conversions, factor lookups, methodology selection, and organizational
boundary determination.

Features:
- Emission calculation failures (Scope 1, 2, 3)
- Unit conversion errors
- Missing or invalid emission factors
- Methodology selection and application errors
- Organizational and operational boundary errors

Author: GreenLang Team
Date: 2026-04-02
"""

from typing import Any, Dict, Optional

from greenlang.exceptions.base import GreenLangException


class CalculationException(GreenLangException):
    """Base exception for calculation-related errors.

    Raised when the deterministic calculation engine encounters
    errors in emission calculations, unit conversions, or factor
    lookups. These are always zero-hallucination paths where LLM
    calls are never used for numeric computation.
    """
    ERROR_PREFIX = "GL_CALC"


class EmissionCalculationError(CalculationException):
    """Emission calculation failed.

    Raised when a GHG emission calculation cannot be completed due to
    invalid inputs, arithmetic errors, or missing intermediate values.

    Example:
        >>> raise EmissionCalculationError(
        ...     message="Division by zero in intensity calculation",
        ...     scope="scope_1",
        ...     calculation_step="intensity_ratio",
        ...     context={"numerator": 1500.0, "denominator": 0.0}
        ... )
    """

    def __init__(
        self,
        message: str,
        scope: Optional[str] = None,
        calculation_step: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize emission calculation error.

        Args:
            message: Error message
            scope: GHG scope (scope_1, scope_2, scope_3)
            calculation_step: Step in the calculation pipeline that failed
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if scope:
            context["scope"] = scope
        if calculation_step:
            context["calculation_step"] = calculation_step
        super().__init__(message, agent_name=agent_name, context=context)


class UnitConversionError(CalculationException):
    """Unit conversion failed.

    Raised when converting between measurement units fails due to
    incompatible units, missing conversion factors, or invalid values.

    Example:
        >>> raise UnitConversionError(
        ...     message="Cannot convert gallons to kWh",
        ...     source_unit="gallon",
        ...     target_unit="kWh",
        ...     context={"value": 100.0, "reason": "Incompatible dimensions"}
        ... )
    """

    def __init__(
        self,
        message: str,
        source_unit: Optional[str] = None,
        target_unit: Optional[str] = None,
        value: Optional[float] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize unit conversion error.

        Args:
            message: Error message
            source_unit: Original unit of measurement
            target_unit: Target unit of measurement
            value: Value being converted
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if source_unit:
            context["source_unit"] = source_unit
        if target_unit:
            context["target_unit"] = target_unit
        if value is not None:
            context["value"] = value
        super().__init__(message, agent_name=agent_name, context=context)


class FactorNotFoundError(CalculationException):
    """Required emission factor not found.

    Raised when a calculation requires an emission factor that cannot be
    located in any configured data source. Distinct from EmissionFactorError
    (integration layer) in that this is raised within the calculation engine
    when the factor is needed for a deterministic computation.

    Example:
        >>> raise FactorNotFoundError(
        ...     message="No emission factor for refrigerant R-410A in region EU",
        ...     factor_type="gwp",
        ...     lookup_key="R-410A",
        ...     context={"region": "EU", "year": 2026}
        ... )
    """

    def __init__(
        self,
        message: str,
        factor_type: Optional[str] = None,
        lookup_key: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize factor not found error.

        Args:
            message: Error message
            factor_type: Type of factor (gwp, emission_factor, conversion_factor)
            lookup_key: Key used for the factor lookup
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if factor_type:
            context["factor_type"] = factor_type
        if lookup_key:
            context["lookup_key"] = lookup_key
        super().__init__(message, agent_name=agent_name, context=context)


class MethodologyError(CalculationException):
    """Calculation methodology error.

    Raised when the selected GHG accounting methodology cannot be applied
    to the given data, or when methodology requirements are not met.

    Example:
        >>> raise MethodologyError(
        ...     message="Spend-based method requires economic data",
        ...     methodology="spend_based",
        ...     ghg_category="scope_3_cat_1",
        ...     context={"missing_data": ["spend_amount", "currency"]}
        ... )
    """

    def __init__(
        self,
        message: str,
        methodology: Optional[str] = None,
        ghg_category: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize methodology error.

        Args:
            message: Error message
            methodology: Name of the methodology (spend_based, activity_based, etc.)
            ghg_category: GHG Protocol category (scope_1, scope_3_cat_1, etc.)
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if methodology:
            context["methodology"] = methodology
        if ghg_category:
            context["ghg_category"] = ghg_category
        super().__init__(message, agent_name=agent_name, context=context)


class BoundaryError(CalculationException):
    """Organizational or operational boundary error.

    Raised when boundary definition, consolidation approach, or scope
    delineation is invalid or inconsistent.

    Example:
        >>> raise BoundaryError(
        ...     message="Operational control boundary excludes joint venture",
        ...     boundary_type="operational_control",
        ...     entity="JV-Solar-Farm-01",
        ...     context={"ownership_pct": 45.0}
        ... )
    """

    def __init__(
        self,
        message: str,
        boundary_type: Optional[str] = None,
        entity: Optional[str] = None,
        agent_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize boundary error.

        Args:
            message: Error message
            boundary_type: Boundary approach (operational_control, equity_share,
                           financial_control)
            entity: Entity or facility affected
            agent_name: Name of agent
            context: Error context
        """
        context = context or {}
        if boundary_type:
            context["boundary_type"] = boundary_type
        if entity:
            context["entity"] = entity
        super().__init__(message, agent_name=agent_name, context=context)


__all__ = [
    'CalculationException',
    'EmissionCalculationError',
    'UnitConversionError',
    'FactorNotFoundError',
    'MethodologyError',
    'BoundaryError',
]
