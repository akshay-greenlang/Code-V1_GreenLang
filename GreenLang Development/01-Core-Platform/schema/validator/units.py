# -*- coding: utf-8 -*-
"""
Unit Validator for GL-FOUND-X-002.

This module implements unit validation and conversion, integrating with
the GreenLang unit catalog and converter. It validates that numeric values
with units meet schema requirements.

Key Features:
    - Detect missing units when schema requires units
    - Detect incompatible units (e.g., kg vs kWh)
    - Validate unit is in allowed catalog
    - Support multiple input formats (object, string, separate fields)
    - Convert to canonical units for normalization

Supported Input Formats:
    - Object form: { "value": 10, "unit": "kWh" }
    - String form: "10 kWh", "10.5 kg", "-5 C"
    - Separate fields: energy_value, energy_unit
    - Raw numeric (with implicit unit from schema)

Error Codes:
    - GLSCHEMA-E300: UNIT_MISSING - Required unit not provided
    - GLSCHEMA-E301: UNIT_INCOMPATIBLE - Unit dimension mismatch
    - GLSCHEMA-E302: UNIT_NONCANONICAL - Unit not in canonical form (warning)
    - GLSCHEMA-E303: UNIT_UNKNOWN - Unit not in catalog

Example:
    >>> from greenlang.schema.validator.units import UnitValidator
    >>> from greenlang.schema.units.catalog import UnitCatalog
    >>> catalog = UnitCatalog()
    >>> validator = UnitValidator(catalog, ValidationOptions())
    >>> findings, normalized = validator.validate(
    ...     {"value": 1000, "unit": "Wh"},
    ...     UnitSpecIR(path="/energy", dimension="energy", canonical="kWh"),
    ...     "/energy"
    ... )
    >>> normalized.value
    1.0
    >>> normalized.unit
    "kWh"

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 2.3
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from greenlang.schema.compiler.ir import UnitSpecIR
from greenlang.schema.errors import ErrorCode, format_error_message
from greenlang.schema.models.config import ValidationOptions, ValidationProfile
from greenlang.schema.models.finding import Finding, FindingHint, Severity
from greenlang.schema.units.catalog import UnitCatalog


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Regex pattern for parsing string form: "10 kWh", "10.5 kg", "-5 C"
# Captures: optional sign, number (int or float), optional whitespace, unit
UNIT_STRING_PATTERN = re.compile(
    r"^\s*"  # Leading whitespace
    r"(?P<value>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"  # Number (int, float, scientific)
    r"\s+"  # Required whitespace between value and unit
    r"(?P<unit>[^\s]+)"  # Unit (non-whitespace characters)
    r"\s*$"  # Trailing whitespace
)


# =============================================================================
# NORMALIZED UNIT MODEL
# =============================================================================


class NormalizedUnit(BaseModel):
    """
    Normalized unit value with conversion provenance.

    Represents a numeric value with unit after validation and optional
    conversion to canonical form. Preserves original values for audit trail.

    Attributes:
        value: The numeric value (possibly converted to canonical)
        unit: The unit symbol (possibly converted to canonical)
        original_value: The original numeric value before conversion
        original_unit: The original unit symbol before conversion
        conversion_factor: Factor used for conversion (1.0 if no conversion)
        was_converted: Whether the value was converted to canonical unit

    Example:
        >>> normalized = NormalizedUnit(
        ...     value=1.0,
        ...     unit="kWh",
        ...     original_value=1000.0,
        ...     original_unit="Wh",
        ...     conversion_factor=0.001,
        ...     was_converted=True
        ... )
    """

    value: float = Field(
        ...,
        description="The numeric value (possibly converted)"
    )
    unit: str = Field(
        ...,
        description="The unit symbol (possibly converted)"
    )
    original_value: float = Field(
        ...,
        description="The original numeric value before conversion"
    )
    original_unit: str = Field(
        ...,
        description="The original unit symbol before conversion"
    )
    conversion_factor: float = Field(
        default=1.0,
        description="Conversion factor used (1.0 if no conversion)"
    )
    was_converted: bool = Field(
        default=False,
        description="Whether the value was converted to canonical unit"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for payload normalization.

        Returns:
            Dictionary with value and unit fields
        """
        return {
            "value": self.value,
            "unit": self.unit,
        }

    def to_dict_with_meta(self) -> Dict[str, Any]:
        """
        Convert to dictionary with metadata for audit trail.

        Returns:
            Dictionary with value, unit, and _meta fields
        """
        result = {
            "value": self.value,
            "unit": self.unit,
        }

        if self.was_converted:
            result["_meta"] = {
                "original_value": self.original_value,
                "original_unit": self.original_unit,
                "conversion_factor": self.conversion_factor,
            }

        return result


# =============================================================================
# UNIT VALIDATOR
# =============================================================================


class UnitValidator:
    """
    Validates units against schema specifications.

    Performs comprehensive unit validation including:
    - Parsing unit values from multiple input formats
    - Checking unit presence (when required by schema)
    - Validating dimensional compatibility
    - Checking unit is in allowed list
    - Converting to canonical unit for normalization

    Attributes:
        catalog: Unit catalog for lookups and conversions
        options: Validation options (profile, normalize, etc.)

    Example:
        >>> catalog = UnitCatalog()
        >>> options = ValidationOptions()
        >>> validator = UnitValidator(catalog, options)
        >>> findings, normalized = validator.validate(
        ...     {"value": 100, "unit": "kWh"},
        ...     UnitSpecIR(path="/energy", dimension="energy", canonical="kWh"),
        ...     "/energy"
        ... )
    """

    def __init__(
        self,
        catalog: UnitCatalog,
        options: ValidationOptions
    ):
        """
        Initialize the unit validator.

        Args:
            catalog: Unit catalog for lookups and conversions
            options: Validation options
        """
        self.catalog = catalog
        self.options = options

        logger.debug(
            f"UnitValidator initialized with profile={options.profile.value}"
        )

    def validate(
        self,
        value: Any,
        unit_spec: UnitSpecIR,
        path: str
    ) -> Tuple[List[Finding], Optional[NormalizedUnit]]:
        """
        Validate unit and optionally normalize to canonical.

        Handles multiple input formats:
        - Object form: {"value": 10, "unit": "kWh"}
        - String form: "10 kWh"
        - Raw numeric: 10 (unit taken from schema default if available)

        Args:
            value: The value to validate (various formats)
            unit_spec: Unit specification from compiled schema IR
            path: JSON Pointer path for error reporting

        Returns:
            Tuple of (findings, normalized_unit)
            - findings: List of validation errors/warnings
            - normalized_unit: NormalizedUnit if valid, None if validation failed

        Example:
            >>> findings, normalized = validator.validate(
            ...     {"value": 1000, "unit": "Wh"},
            ...     UnitSpecIR(path="/energy", dimension="energy", canonical="kWh"),
            ...     "/energy"
            ... )
        """
        findings: List[Finding] = []
        start_time = logger.isEnabledFor(logging.DEBUG)

        # Step 1: Parse the value into numeric + unit
        numeric_value, unit_symbol = self._parse_unit_value(value)

        # Step 2: Check if we got a valid numeric value
        if numeric_value is None:
            # Could not parse numeric value - this is a type error, not unit error
            # Let the structural validator handle this
            logger.debug(f"Could not parse numeric value from {value} at {path}")
            return findings, None

        # Step 3: Validate unit presence
        presence_findings = self._validate_unit_presence(
            unit_symbol,
            unit_spec,
            path
        )
        findings.extend(presence_findings)

        # If unit is missing and required, we can't proceed
        if unit_symbol is None and presence_findings:
            return findings, None

        # If unit is None but no findings, use canonical as default
        if unit_symbol is None:
            unit_symbol = unit_spec.canonical

        # Step 4: Validate unit is known in catalog
        unknown_findings = self._validate_unit_known(
            unit_symbol,
            unit_spec,
            path
        )
        findings.extend(unknown_findings)

        # If unit is unknown, we can't validate further
        if unknown_findings:
            return findings, None

        # Step 5: Validate dimensional compatibility
        compatibility_findings = self._validate_dimension(
            unit_symbol,
            unit_spec.dimension,
            path
        )
        findings.extend(compatibility_findings)

        # If dimension is incompatible, we can't convert
        if compatibility_findings:
            return findings, None

        # Step 6: Check if unit is in allowed list (if specified)
        allowed_findings = self._validate_unit_allowed(
            unit_symbol,
            unit_spec,
            path
        )
        findings.extend(allowed_findings)

        # Step 7: Check if unit is non-canonical (warning in strict mode)
        canonical_findings = self._validate_canonical(
            unit_symbol,
            unit_spec,
            path
        )
        findings.extend(canonical_findings)

        # Step 8: Normalize to canonical unit
        normalized = self._normalize_unit(
            numeric_value,
            unit_symbol,
            unit_spec.canonical
        )

        logger.debug(
            f"Unit validation at {path}: {numeric_value} {unit_symbol} -> "
            f"{normalized.value} {normalized.unit}"
        )

        return findings, normalized

    # -------------------------------------------------------------------------
    # Parsing Methods
    # -------------------------------------------------------------------------

    def _parse_unit_value(
        self,
        value: Any
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse unit value from various input formats.

        Supported formats:
        - Object: {"value": 10, "unit": "kWh"} -> (10.0, "kWh")
        - String: "10 kWh" -> (10.0, "kWh")
        - Raw numeric: 10 -> (10.0, None)
        - Numeric with explicit None unit: (10, None)

        Args:
            value: Input value in any supported format

        Returns:
            Tuple of (numeric_value, unit_symbol)
            Either or both may be None if parsing fails
        """
        # Case 1: Dictionary/object form
        if isinstance(value, dict):
            return self._parse_object_form(value)

        # Case 2: String form
        if isinstance(value, str):
            return self._parse_string_form(value)

        # Case 3: Raw numeric
        if isinstance(value, (int, float)):
            return float(value), None

        # Unknown format
        logger.debug(f"Unknown unit value format: {type(value)}")
        return None, None

    def _parse_object_form(
        self,
        value: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse object form: {"value": 10, "unit": "kWh"}.

        Also supports variations:
        - {"amount": 10, "unit": "kWh"}
        - {"quantity": 10, "units": "kWh"}

        Args:
            value: Dictionary with value and unit fields

        Returns:
            Tuple of (numeric_value, unit_symbol)
        """
        # Try various field names for value
        numeric_value: Optional[float] = None
        for value_key in ["value", "amount", "quantity", "val"]:
            if value_key in value:
                raw_value = value[value_key]
                if isinstance(raw_value, (int, float)):
                    numeric_value = float(raw_value)
                    break
                elif isinstance(raw_value, str):
                    try:
                        numeric_value = float(raw_value)
                        break
                    except ValueError:
                        pass

        # Try various field names for unit
        unit_symbol: Optional[str] = None
        for unit_key in ["unit", "units", "uom"]:
            if unit_key in value:
                raw_unit = value[unit_key]
                if isinstance(raw_unit, str):
                    unit_symbol = raw_unit.strip()
                    break

        return numeric_value, unit_symbol

    def _parse_string_form(
        self,
        value: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse string form: "10 kWh", "10.5 kg", "-5 C".

        Args:
            value: String in format "NUMBER UNIT"

        Returns:
            Tuple of (numeric_value, unit_symbol)
        """
        match = UNIT_STRING_PATTERN.match(value)
        if match:
            try:
                numeric_value = float(match.group("value"))
                unit_symbol = match.group("unit")
                return numeric_value, unit_symbol
            except ValueError:
                pass

        # Try parsing as just a number
        try:
            numeric_value = float(value.strip())
            return numeric_value, None
        except ValueError:
            pass

        return None, None

    # -------------------------------------------------------------------------
    # Validation Methods
    # -------------------------------------------------------------------------

    def _validate_unit_presence(
        self,
        unit: Optional[str],
        unit_spec: UnitSpecIR,
        path: str
    ) -> List[Finding]:
        """
        Validate that a unit is present when required.

        A unit is required when the schema specifies a unit_spec with dimension.

        Args:
            unit: The unit symbol (may be None)
            unit_spec: Unit specification from schema
            path: JSON Pointer path

        Returns:
            List of findings (empty if valid)
        """
        findings: List[Finding] = []

        # If no unit provided and dimension is specified, unit is required
        if unit is None or unit.strip() == "":
            allowed_units = self.catalog.list_units_for_dimension(unit_spec.dimension)
            allowed_str = ", ".join(allowed_units[:10])
            if len(allowed_units) > 10:
                allowed_str += f", ... ({len(allowed_units)} total)"

            findings.append(Finding(
                code=ErrorCode.UNIT_MISSING.value,
                severity=Severity.ERROR,
                path=path,
                message=format_error_message(
                    ErrorCode.UNIT_MISSING,
                    path=path
                ),
                expected={
                    "dimension": unit_spec.dimension,
                    "canonical": unit_spec.canonical,
                    "format": "{ value: <number>, unit: <string> } or '<number> <unit>'"
                },
                actual=None,
                hint=FindingHint(
                    category="unit_missing",
                    suggested_values=allowed_units[:5],
                    docs_url="https://docs.greenlang.dev/schema/units"
                )
            ))

        return findings

    def _validate_unit_known(
        self,
        unit: str,
        unit_spec: UnitSpecIR,
        path: str
    ) -> List[Finding]:
        """
        Validate that the unit is known in the catalog.

        Args:
            unit: The unit symbol
            unit_spec: Unit specification from schema
            path: JSON Pointer path

        Returns:
            List of findings (empty if valid)
        """
        findings: List[Finding] = []

        if not self.catalog.is_known_unit(unit):
            known_units = self.catalog.list_units_for_dimension(unit_spec.dimension)
            known_str = ", ".join(known_units[:10])

            findings.append(Finding(
                code=ErrorCode.UNIT_UNKNOWN.value,
                severity=Severity.ERROR,
                path=path,
                message=format_error_message(
                    ErrorCode.UNIT_UNKNOWN,
                    unit=unit,
                    path=path
                ),
                expected={
                    "dimension": unit_spec.dimension,
                    "known_units": known_units[:10]
                },
                actual={"unit": unit},
                hint=FindingHint(
                    category="unit_unknown",
                    suggested_values=known_units[:5],
                    docs_url="https://docs.greenlang.dev/schema/units"
                )
            ))

        return findings

    def _validate_dimension(
        self,
        unit: str,
        expected_dimension: str,
        path: str
    ) -> List[Finding]:
        """
        Validate unit has expected dimension.

        Args:
            unit: The unit symbol
            expected_dimension: Expected dimension name
            path: JSON Pointer path

        Returns:
            List of findings (empty if compatible)
        """
        findings: List[Finding] = []

        actual_dimension = self.catalog.get_unit_dimension(unit)

        if actual_dimension is None:
            # Unit not in catalog - already handled by _validate_unit_known
            return findings

        if actual_dimension != expected_dimension:
            compatible_units = self.catalog.list_units_for_dimension(expected_dimension)

            findings.append(Finding(
                code=ErrorCode.UNIT_INCOMPATIBLE.value,
                severity=Severity.ERROR,
                path=path,
                message=format_error_message(
                    ErrorCode.UNIT_INCOMPATIBLE,
                    unit=unit,
                    path=path,
                    expected_dimension=expected_dimension
                ),
                expected={
                    "dimension": expected_dimension,
                    "compatible_units": compatible_units[:10]
                },
                actual={
                    "unit": unit,
                    "dimension": actual_dimension
                },
                hint=FindingHint(
                    category="unit_incompatible",
                    suggested_values=compatible_units[:5],
                    docs_url="https://docs.greenlang.dev/schema/units"
                )
            ))

        return findings

    def _validate_unit_allowed(
        self,
        unit: str,
        unit_spec: UnitSpecIR,
        path: str
    ) -> List[Finding]:
        """
        Validate unit is in allowed list (if specified).

        Args:
            unit: The unit symbol
            unit_spec: Unit specification from schema
            path: JSON Pointer path

        Returns:
            List of findings (empty if allowed or no restrictions)
        """
        findings: List[Finding] = []

        # Check if allowed list is specified and non-empty
        if unit_spec.allowed and not unit_spec.is_unit_allowed(unit):
            findings.append(Finding(
                code=ErrorCode.UNIT_INCOMPATIBLE.value,
                severity=Severity.ERROR,
                path=path,
                message=f"Unit '{unit}' at path '{path}' is not in the allowed list",
                expected={
                    "allowed_units": list(unit_spec.allowed)
                },
                actual={"unit": unit},
                hint=FindingHint(
                    category="unit_not_allowed",
                    suggested_values=list(unit_spec.allowed)[:5]
                )
            ))

        return findings

    def _validate_canonical(
        self,
        unit: str,
        unit_spec: UnitSpecIR,
        path: str
    ) -> List[Finding]:
        """
        Check if unit is non-canonical (warning in strict mode).

        Args:
            unit: The unit symbol
            unit_spec: Unit specification from schema
            path: JSON Pointer path

        Returns:
            List of findings (empty or warning if non-canonical)
        """
        findings: List[Finding] = []

        # Resolve to canonical symbol for comparison
        resolved_unit = self.catalog.get_unit(unit)
        if resolved_unit is None:
            return findings

        canonical_unit = unit_spec.canonical

        # Check if unit is not canonical (in strict mode, emit warning)
        if (
            self.options.profile == ValidationProfile.STRICT
            and resolved_unit.symbol != canonical_unit
            and not resolved_unit.is_canonical
        ):
            findings.append(Finding(
                code=ErrorCode.UNIT_NONCANONICAL.value,
                severity=Severity.WARNING,
                path=path,
                message=format_error_message(
                    ErrorCode.UNIT_NONCANONICAL,
                    unit=unit,
                    path=path
                ),
                expected={"canonical_unit": canonical_unit},
                actual={"unit": unit},
                hint=FindingHint(
                    category="unit_noncanonical",
                    suggested_values=[canonical_unit]
                )
            ))

        return findings

    # -------------------------------------------------------------------------
    # Normalization Methods
    # -------------------------------------------------------------------------

    def _normalize_unit(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> NormalizedUnit:
        """
        Normalize value to canonical unit.

        Args:
            value: Numeric value
            from_unit: Source unit symbol
            to_unit: Target canonical unit

        Returns:
            NormalizedUnit with converted value and provenance
        """
        # Check if conversion is needed
        if from_unit == to_unit:
            return NormalizedUnit(
                value=value,
                unit=to_unit,
                original_value=value,
                original_unit=from_unit,
                conversion_factor=1.0,
                was_converted=False
            )

        # Check if both units are compatible
        if not self.catalog.is_compatible(from_unit, to_unit):
            # Units are incompatible - return original (error already reported)
            return NormalizedUnit(
                value=value,
                unit=from_unit,
                original_value=value,
                original_unit=from_unit,
                conversion_factor=1.0,
                was_converted=False
            )

        try:
            # Convert to canonical unit
            converted_value = self.catalog.convert(value, from_unit, to_unit)
            conversion_factor = self.catalog.get_conversion_factor(from_unit, to_unit)

            return NormalizedUnit(
                value=converted_value,
                unit=to_unit,
                original_value=value,
                original_unit=from_unit,
                conversion_factor=conversion_factor,
                was_converted=True
            )

        except ValueError as e:
            # Conversion failed - return original
            logger.warning(f"Unit conversion failed: {e}")
            return NormalizedUnit(
                value=value,
                unit=from_unit,
                original_value=value,
                original_unit=from_unit,
                conversion_factor=1.0,
                was_converted=False
            )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_unit_finding(
    code: ErrorCode,
    path: str,
    unit: Optional[str] = None,
    dimension: Optional[str] = None,
    expected_dimension: Optional[str] = None,
    allowed_units: Optional[List[str]] = None,
    canonical_unit: Optional[str] = None,
) -> Finding:
    """
    Create a unit-related validation finding.

    Helper function for creating standardized unit validation findings.

    Args:
        code: Error code (UNIT_MISSING, UNIT_INCOMPATIBLE, etc.)
        path: JSON Pointer path
        unit: The unit symbol (if available)
        dimension: Actual dimension (if available)
        expected_dimension: Expected dimension
        allowed_units: List of allowed units
        canonical_unit: Canonical unit for normalization

    Returns:
        Finding with appropriate severity, message, and hints
    """
    severity = Severity.ERROR
    if code == ErrorCode.UNIT_NONCANONICAL:
        severity = Severity.WARNING

    # Build expected dict
    expected: Dict[str, Any] = {}
    if expected_dimension:
        expected["dimension"] = expected_dimension
    if allowed_units:
        expected["allowed_units"] = allowed_units[:10]
    if canonical_unit:
        expected["canonical_unit"] = canonical_unit

    # Build actual dict
    actual: Optional[Dict[str, Any]] = None
    if unit:
        actual = {"unit": unit}
        if dimension:
            actual["dimension"] = dimension

    # Build hint
    suggested_values = allowed_units[:5] if allowed_units else []
    if canonical_unit and canonical_unit not in suggested_values:
        suggested_values.insert(0, canonical_unit)

    return Finding(
        code=code.value,
        severity=severity,
        path=path,
        message=format_error_message(
            code,
            path=path,
            unit=unit or "(missing)",
            expected_dimension=expected_dimension or "(any)",
            canonical_unit=canonical_unit or "(none)"
        ),
        expected=expected if expected else None,
        actual=actual,
        hint=FindingHint(
            category=f"unit_{code.name.lower().replace('unit_', '')}",
            suggested_values=suggested_values
        ) if suggested_values else None
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "NormalizedUnit",
    "UnitValidator",
    "create_unit_finding",
]
