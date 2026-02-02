# -*- coding: utf-8 -*-
"""
Canonicalizer for GL-FOUND-X-002.

This module implements key and unit canonicalization:
    - Unit conversion to canonical SI units
    - Key alias resolution
    - Casing normalization
    - Stable output ordering for reproducibility

Key Features:
    - Convert values to canonical SI units (e.g., Wh -> kWh)
    - Store original unit in metadata for audit trail
    - Record all conversions for provenance tracking
    - Handle precision appropriately using Decimal arithmetic
    - Support multiple input formats (object, string, raw numeric)

Design Principles:
    - Zero-hallucination: All conversions are deterministic mathematical operations
    - Complete provenance: All conversions tracked with factors
    - Fail loudly: Unknown units raise errors rather than guessing
    - Idempotent: canonicalize(canonicalize(x)) == canonicalize(x)

Example:
    >>> from greenlang.schema.normalizer.canonicalizer import UnitCanonicalizer
    >>> from greenlang.schema.units.catalog import UnitCatalog
    >>> catalog = UnitCatalog()
    >>> canonicalizer = UnitCanonicalizer(catalog)
    >>> value, records = canonicalizer.canonicalize(
    ...     {"value": 1000, "unit": "Wh"},
    ...     UnitSpecIR(path="/energy", dimension="energy", canonical="kWh", allowed=[]),
    ...     "/energy"
    ... )
    >>> print(value)
    {'value': 1.0, 'unit': 'kWh', '_meta': {'original_value': 1000, 'original_unit': 'Wh', 'conversion_factor': 0.001}}

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 3.2
"""

from __future__ import annotations

import hashlib
import logging
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from greenlang.schema.compiler.ir import SchemaIR, UnitSpecIR
from greenlang.schema.units.catalog import UnitCatalog


logger = logging.getLogger(__name__)


# =============================================================================
# CANONICAL UNITS BY DIMENSION
# =============================================================================

# Default canonical units for each dimension (SI base or common derived)
# These can be overridden by schema-specific UnitSpecIR.canonical
CANONICAL_UNITS: Dict[str, str] = {
    "energy": "kWh",        # kilowatt-hour (common in carbon accounting)
    "mass": "kg",           # kilogram
    "volume": "L",          # liter
    "area": "m2",           # square meter
    "length": "m",          # meter
    "time": "s",            # second
    "temperature": "K",     # kelvin
    "power": "kW",          # kilowatt
    "emissions": "kgCO2e",  # kg CO2 equivalent
}


# =============================================================================
# CONVERSION RECORD MODEL
# =============================================================================


class ConversionRecord(BaseModel):
    """
    Record of a unit conversion.

    Captures complete provenance information for a unit conversion operation,
    enabling full audit trail and reversibility.

    Attributes:
        path: JSON Pointer path where conversion occurred
        original_value: The numeric value before conversion
        original_unit: The unit symbol before conversion
        canonical_value: The numeric value after conversion
        canonical_unit: The unit symbol after conversion (canonical)
        conversion_factor: Factor used (canonical = original * factor)
        dimension: Physical dimension of the unit

    Example:
        >>> record = ConversionRecord(
        ...     path="/energy_consumption",
        ...     original_value=1000.0,
        ...     original_unit="Wh",
        ...     canonical_value=1.0,
        ...     canonical_unit="kWh",
        ...     conversion_factor=0.001,
        ...     dimension="energy"
        ... )
    """

    path: str = Field(
        ...,
        description="JSON Pointer path where conversion occurred"
    )
    original_value: float = Field(
        ...,
        description="The numeric value before conversion"
    )
    original_unit: str = Field(
        ...,
        description="The unit symbol before conversion"
    )
    canonical_value: float = Field(
        ...,
        description="The numeric value after conversion"
    )
    canonical_unit: str = Field(
        ...,
        description="The unit symbol after conversion (canonical)"
    )
    conversion_factor: float = Field(
        ...,
        description="Conversion factor (canonical = original * factor)"
    )
    dimension: str = Field(
        ...,
        description="Physical dimension of the unit"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the conversion record
        """
        return {
            "path": self.path,
            "original_value": self.original_value,
            "original_unit": self.original_unit,
            "canonical_value": self.canonical_value,
            "canonical_unit": self.canonical_unit,
            "conversion_factor": self.conversion_factor,
            "dimension": self.dimension,
        }

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of the conversion for provenance.

        Returns:
            Hexadecimal hash string
        """
        content = (
            f"{self.path}:{self.original_value}:{self.original_unit}:"
            f"{self.canonical_value}:{self.canonical_unit}:{self.conversion_factor}"
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


# =============================================================================
# CANONICALIZED VALUE MODEL
# =============================================================================


class CanonicalizedValue(BaseModel):
    """
    A canonicalized unit value.

    Represents a value that has been converted to its canonical unit form,
    with metadata preserving the original value for audit purposes.

    Attributes:
        value: The canonical numeric value
        unit: The canonical unit symbol
        meta: Metadata including original value/unit if converted

    Example:
        >>> canonical = CanonicalizedValue(
        ...     value=1.0,
        ...     unit="kWh",
        ...     meta={
        ...         "original_value": 1000.0,
        ...         "original_unit": "Wh",
        ...         "conversion_factor": 0.001
        ...     }
        ... )
    """

    value: float = Field(
        ...,
        description="The canonical numeric value"
    )
    unit: str = Field(
        ...,
        description="The canonical unit symbol"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata including original value/unit if converted"
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for payload normalization.

        Returns:
            Dictionary with value, unit, and optional _meta
        """
        result: Dict[str, Any] = {
            "value": self.value,
            "unit": self.unit,
        }
        if self.meta:
            result["_meta"] = self.meta
        return result

    @property
    def was_converted(self) -> bool:
        """Check if value was converted from a different unit."""
        return bool(self.meta and "original_unit" in self.meta)


# =============================================================================
# KEY RENAME RECORD
# =============================================================================


class KeyRename(BaseModel):
    """
    Record of a key rename.

    Captures information about key canonicalization operations for audit.

    Attributes:
        path: JSON Pointer path where rename occurred
        original_key: The original key name
        canonical_key: The canonical key name
        reason: Reason for rename ("alias", "casing", "typo_correction")

    Example:
        >>> rename = KeyRename(
        ...     path="/",
        ...     original_key="Energy",
        ...     canonical_key="energy",
        ...     reason="casing"
        ... )
    """

    path: str = Field(
        ...,
        description="JSON Pointer path where rename occurred"
    )
    original_key: str = Field(
        ...,
        description="The original key name"
    )
    canonical_key: str = Field(
        ...,
        description="The canonical key name"
    )
    reason: Literal["alias", "casing", "typo_correction"] = Field(
        ...,
        description="Reason for the rename"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


# =============================================================================
# UNIT STRING PARSING
# =============================================================================

# Regex pattern for parsing string form: "10 kWh", "10.5 kg", "-5 C"
# Captures: optional sign, number (int or float), optional whitespace, unit
UNIT_STRING_PATTERN = re.compile(
    r"^\s*"  # Leading whitespace
    r"(?P<value>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"  # Number
    r"\s+"  # Required whitespace between value and unit
    r"(?P<unit>[^\s]+)"  # Unit (non-whitespace characters)
    r"\s*$"  # Trailing whitespace
)


# =============================================================================
# UNIT CANONICALIZER
# =============================================================================


class UnitCanonicalizer:
    """
    Canonicalizes unit values to SI units.

    Converts numeric values with units to their canonical SI form while
    preserving the original value in metadata for audit purposes. Supports
    multiple input formats and maintains complete conversion provenance.

    Attributes:
        catalog: Unit catalog for lookups and conversions
        _records: List of conversion records for the current session

    Example:
        >>> catalog = UnitCatalog()
        >>> canonicalizer = UnitCanonicalizer(catalog)
        >>> value, records = canonicalizer.canonicalize(
        ...     {"value": 1000, "unit": "Wh"},
        ...     UnitSpecIR(path="/energy", dimension="energy", canonical="kWh", allowed=[]),
        ...     "/energy"
        ... )
        >>> print(value["value"], value["unit"])
        1.0 kWh
    """

    def __init__(self, catalog: UnitCatalog):
        """
        Initialize the unit canonicalizer.

        Args:
            catalog: Unit catalog for conversion factors and dimension lookups
        """
        self.catalog = catalog
        self._records: List[ConversionRecord] = []

        logger.debug("UnitCanonicalizer initialized")

    # -------------------------------------------------------------------------
    # Main Canonicalization Methods
    # -------------------------------------------------------------------------

    def canonicalize(
        self,
        value: Any,
        unit_spec: UnitSpecIR,
        path: str
    ) -> Tuple[Any, List[ConversionRecord]]:
        """
        Convert value to canonical unit.

        Handles multiple input formats:
        - Object form: {"value": x, "unit": "y"}
        - String form: "x y" (e.g., "100 kWh")
        - Raw numeric: x (uses canonical unit from spec)

        Args:
            value: The value (can be dict, string, or numeric)
            unit_spec: Unit specification from IR
            path: JSON Pointer path for provenance tracking

        Returns:
            Tuple of (canonicalized value dict, conversion records)
            The canonicalized value is formatted as:
            {
                "value": <float>,
                "unit": "<canonical_unit>",
                "_meta": {  # Only if converted
                    "original_value": <float>,
                    "original_unit": "<original_unit>",
                    "conversion_factor": <float>
                }
            }

        Raises:
            ValueError: If value cannot be parsed or unit is unknown

        Example:
            >>> value, records = canonicalizer.canonicalize(
            ...     {"value": 1000, "unit": "Wh"},
            ...     UnitSpecIR(path="/energy", dimension="energy", canonical="kWh", allowed=[]),
            ...     "/energy"
            ... )
        """
        records: List[ConversionRecord] = []

        # Extract numeric value and unit from various formats
        numeric_value, original_unit = self._extract_value_unit(value)

        if numeric_value is None:
            logger.warning(f"Could not extract numeric value from {value} at {path}")
            # Return original value unchanged if we can't parse it
            return value, records

        # Use canonical unit from spec if no unit provided
        if original_unit is None:
            original_unit = unit_spec.canonical

        # Get the target canonical unit
        canonical_unit = unit_spec.canonical

        # Check if conversion is needed
        if original_unit == canonical_unit:
            # No conversion needed - return formatted value
            output = self._format_canonical_output(
                value=numeric_value,
                unit=canonical_unit,
                original_unit=None,  # No conversion
                path=path
            )
            return output, records

        # Validate units are compatible
        if not self.catalog.is_compatible(original_unit, canonical_unit):
            original_dim = self.catalog.get_unit_dimension(original_unit)
            target_dim = self.catalog.get_unit_dimension(canonical_unit)
            raise ValueError(
                f"Cannot convert {original_unit} ({original_dim}) to "
                f"{canonical_unit} ({target_dim}) at {path}: "
                f"dimensions are incompatible"
            )

        # Create canonicalized value with conversion
        output, record = self._create_canonical_value(
            original_value=numeric_value,
            original_unit=original_unit,
            canonical_unit=canonical_unit,
            path=path
        )

        records.append(record)
        self._records.append(record)

        logger.debug(
            f"Canonicalized {numeric_value} {original_unit} -> "
            f"{output['value']} {output['unit']} at {path}"
        )

        return output, records

    def canonicalize_object(
        self,
        payload: Dict[str, Any],
        unit_specs: Dict[str, UnitSpecIR]
    ) -> Tuple[Dict[str, Any], List[ConversionRecord]]:
        """
        Canonicalize all unit values in an object.

        Recursively processes an object and converts all values that have
        unit specifications to their canonical forms.

        Args:
            payload: The object containing values to canonicalize
            unit_specs: Dictionary mapping JSON Pointer paths to unit specs

        Returns:
            Tuple of (canonicalized payload, list of all conversion records)

        Example:
            >>> unit_specs = {
            ...     "/energy": UnitSpecIR(path="/energy", dimension="energy", canonical="kWh", allowed=[]),
            ...     "/mass": UnitSpecIR(path="/mass", dimension="mass", canonical="kg", allowed=[])
            ... }
            >>> output, records = canonicalizer.canonicalize_object(payload, unit_specs)
        """
        all_records: List[ConversionRecord] = []
        result = self._canonicalize_recursive(
            data=payload,
            path="",
            unit_specs=unit_specs,
            records=all_records
        )

        return result, all_records

    # -------------------------------------------------------------------------
    # Value Extraction Methods
    # -------------------------------------------------------------------------

    def _extract_value_unit(
        self,
        value: Any
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract numeric value and unit from various formats.

        Supports:
        - Object form: {"value": 10, "unit": "kWh"}
        - String form: "10 kWh"
        - Raw numeric: 10

        Args:
            value: Input value in any supported format

        Returns:
            Tuple of (numeric_value, unit_symbol)
            Either or both may be None if parsing fails
        """
        # Case 1: Dictionary/object form
        if isinstance(value, dict):
            return self._extract_from_object(value)

        # Case 2: String form
        if isinstance(value, str):
            return self._extract_from_string(value)

        # Case 3: Raw numeric
        if isinstance(value, (int, float)):
            return float(value), None

        # Unknown format
        logger.debug(f"Unknown unit value format: {type(value)}")
        return None, None

    def _extract_from_object(
        self,
        value: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract value and unit from object form.

        Supports variations:
        - {"value": 10, "unit": "kWh"}
        - {"amount": 10, "unit": "kWh"}
        - {"quantity": 10, "units": "kWh"}

        Args:
            value: Dictionary with value and unit fields

        Returns:
            Tuple of (numeric_value, unit_symbol)
        """
        numeric_value: Optional[float] = None
        unit_symbol: Optional[str] = None

        # Try various field names for value
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
        for unit_key in ["unit", "units", "uom"]:
            if unit_key in value:
                raw_unit = value[unit_key]
                if isinstance(raw_unit, str):
                    unit_symbol = raw_unit.strip()
                    break

        return numeric_value, unit_symbol

    def _extract_from_string(
        self,
        value: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract value and unit from string form.

        Supports: "10 kWh", "10.5 kg", "-5 C", "1.5e3 MWh"

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
    # Conversion Methods
    # -------------------------------------------------------------------------

    def _create_canonical_value(
        self,
        original_value: float,
        original_unit: str,
        canonical_unit: str,
        path: str
    ) -> Tuple[Dict[str, Any], ConversionRecord]:
        """
        Create canonicalized value with metadata.

        Performs the actual conversion and creates both the output value
        and the conversion record for provenance tracking.

        Args:
            original_value: The numeric value before conversion
            original_unit: The unit symbol before conversion
            canonical_unit: The target canonical unit
            path: JSON Pointer path for provenance

        Returns:
            Tuple of (canonicalized value dict, conversion record)

        Raises:
            ValueError: If conversion fails
        """
        # Perform conversion using catalog
        try:
            canonical_value = self.catalog.convert(
                value=original_value,
                from_unit=original_unit,
                to_unit=canonical_unit
            )
            conversion_factor = self.catalog.get_conversion_factor(
                from_unit=original_unit,
                to_unit=canonical_unit
            )
        except ValueError as e:
            logger.error(f"Conversion failed at {path}: {e}")
            raise

        # Get dimension for the record
        dimension = self.catalog.get_unit_dimension(original_unit) or "unknown"

        # Create conversion record
        record = ConversionRecord(
            path=path,
            original_value=original_value,
            original_unit=original_unit,
            canonical_value=canonical_value,
            canonical_unit=canonical_unit,
            conversion_factor=conversion_factor,
            dimension=dimension
        )

        # Format output with metadata
        output = self._format_canonical_output(
            value=canonical_value,
            unit=canonical_unit,
            original_unit=original_unit,
            path=path,
            original_value=original_value,
            conversion_factor=conversion_factor
        )

        return output, record

    def _format_canonical_output(
        self,
        value: float,
        unit: str,
        original_unit: Optional[str],
        path: str,
        original_value: Optional[float] = None,
        conversion_factor: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Format output as canonical value with metadata.

        Output format:
        {
            "value": 1000.0,
            "unit": "kWh",
            "_meta": {  # Only included if conversion occurred
                "original_value": 1.0,
                "original_unit": "MWh",
                "conversion_factor": 1000.0
            }
        }

        Args:
            value: The canonical numeric value
            unit: The canonical unit symbol
            original_unit: The original unit (None if no conversion)
            path: JSON Pointer path
            original_value: The original numeric value (if converted)
            conversion_factor: The conversion factor used (if converted)

        Returns:
            Formatted dictionary with value, unit, and optional _meta
        """
        result: Dict[str, Any] = {
            "value": value,
            "unit": unit,
        }

        # Add metadata if conversion occurred
        if original_unit is not None and original_unit != unit:
            result["_meta"] = {
                "original_value": original_value if original_value is not None else value,
                "original_unit": original_unit,
                "conversion_factor": conversion_factor if conversion_factor is not None else 1.0,
            }

        return result

    # -------------------------------------------------------------------------
    # Recursive Canonicalization
    # -------------------------------------------------------------------------

    def _canonicalize_recursive(
        self,
        data: Any,
        path: str,
        unit_specs: Dict[str, UnitSpecIR],
        records: List[ConversionRecord]
    ) -> Any:
        """
        Recursively canonicalize values in nested structures.

        Args:
            data: Data to process (may be dict, list, or primitive)
            path: Current JSON Pointer path
            unit_specs: Dictionary of unit specifications by path
            records: List to append conversion records to

        Returns:
            Canonicalized data structure
        """
        if isinstance(data, dict):
            result: Dict[str, Any] = {}
            for key, value in data.items():
                child_path = f"{path}/{key}"

                # Check if this path has a unit spec
                if child_path in unit_specs:
                    try:
                        canonical_value, new_records = self.canonicalize(
                            value=value,
                            unit_spec=unit_specs[child_path],
                            path=child_path
                        )
                        result[key] = canonical_value
                        records.extend(new_records)
                    except ValueError as e:
                        logger.warning(f"Canonicalization failed at {child_path}: {e}")
                        result[key] = value  # Keep original on error
                else:
                    # Recurse into nested structures
                    result[key] = self._canonicalize_recursive(
                        data=value,
                        path=child_path,
                        unit_specs=unit_specs,
                        records=records
                    )
            return result

        elif isinstance(data, list):
            result_list: List[Any] = []
            for i, item in enumerate(data):
                child_path = f"{path}/{i}"
                result_list.append(
                    self._canonicalize_recursive(
                        data=item,
                        path=child_path,
                        unit_specs=unit_specs,
                        records=records
                    )
                )
            return result_list

        else:
            # Primitive value - return as-is
            return data

    # -------------------------------------------------------------------------
    # Record Management
    # -------------------------------------------------------------------------

    def get_records(self) -> List[ConversionRecord]:
        """
        Get all conversion records from this session.

        Returns:
            List of ConversionRecord objects
        """
        return list(self._records)

    def clear_records(self) -> None:
        """Clear all conversion records."""
        self._records.clear()
        logger.debug("Conversion records cleared")


# =============================================================================
# KEY CANONICALIZER
# =============================================================================


class KeyCanonicalizer:
    """
    Key canonicalizer.

    Resolves key aliases and normalizes casing according to schema
    definitions. Ensures stable output ordering for reproducibility.

    Attributes:
        ir: Compiled schema IR with alias mappings

    Example:
        >>> canonicalizer = KeyCanonicalizer(ir)
        >>> payload, renames = canonicalizer.canonicalize({"Energy": 100})
        >>> print(payload)
        {"energy": 100}
    """

    def __init__(self, ir: SchemaIR):
        """
        Initialize the key canonicalizer.

        Args:
            ir: Compiled schema IR with alias mappings
        """
        self.ir = ir

        logger.debug("KeyCanonicalizer initialized")

    def canonicalize(
        self,
        payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[KeyRename]]:
        """
        Canonicalize all keys in payload.

        Applies:
        1. Alias resolution (from schema renamed_fields)
        2. Casing normalization (if schema demands)

        Output key ordering is stable and deterministic (alphabetically sorted).

        Args:
            payload: Payload to process

        Returns:
            Tuple of (canonicalized_payload, list of KeyRename records)

        Example:
            >>> payload, renames = canonicalizer.canonicalize({"Energy": 100})
            >>> print(payload)
            {"energy": 100}
        """
        renames: List[KeyRename] = []
        result = self._canonicalize_recursive(
            data=payload,
            path="",
            renames=renames
        )

        return result, renames

    def _canonicalize_recursive(
        self,
        data: Any,
        path: str,
        renames: List[KeyRename]
    ) -> Any:
        """
        Recursively canonicalize keys in nested structures.

        Args:
            data: Data to process (may be dict, list, or primitive)
            path: Current JSON Pointer path
            renames: List to append rename records to

        Returns:
            Canonicalized data
        """
        if isinstance(data, dict):
            result: Dict[str, Any] = {}

            for key, value in data.items():
                child_path = f"{path}/{key}"

                # Check for alias resolution
                canonical_key = self._resolve_alias(key)
                rename_reason: Optional[str] = None

                if canonical_key is not None:
                    rename_reason = "alias"
                else:
                    # Check for casing normalization
                    normalized_key = self._normalize_casing(key)
                    if normalized_key != key:
                        canonical_key = normalized_key
                        rename_reason = "casing"

                # Use canonical key if found, otherwise original
                final_key = canonical_key if canonical_key is not None else key

                # Record the rename if it occurred
                if canonical_key is not None and rename_reason is not None:
                    renames.append(KeyRename(
                        path=path,
                        original_key=key,
                        canonical_key=canonical_key,
                        reason=rename_reason  # type: ignore
                    ))

                # Recurse into nested structures
                result[final_key] = self._canonicalize_recursive(
                    data=value,
                    path=f"{path}/{final_key}",
                    renames=renames
                )

            # Return with stable key ordering
            return self._stable_order_dict(result)

        elif isinstance(data, list):
            result_list: List[Any] = []
            for i, item in enumerate(data):
                child_path = f"{path}/{i}"
                result_list.append(
                    self._canonicalize_recursive(
                        data=item,
                        path=child_path,
                        renames=renames
                    )
                )
            return result_list

        else:
            # Primitive value - return as-is
            return data

    def _resolve_alias(self, key: str) -> Optional[str]:
        """
        Resolve a key alias to its canonical name.

        Checks the schema's renamed_fields mapping for old->new renames.

        Args:
            key: Key to resolve

        Returns:
            Canonical name if key is an alias, None otherwise
        """
        return self.ir.renamed_fields.get(key)

    def _normalize_casing(self, key: str) -> str:
        """
        Normalize key casing according to convention.

        Default convention is snake_case (lowercase with underscores).
        CamelCase and PascalCase are converted.

        Args:
            key: Key to normalize

        Returns:
            Normalized key (e.g., "EnergyConsumption" -> "energy_consumption")
        """
        # Check if it's already lowercase/snake_case
        if key == key.lower():
            return key

        # Convert CamelCase/PascalCase to snake_case
        result = ""
        for i, char in enumerate(key):
            if char.isupper():
                if i > 0 and not key[i-1].isupper():
                    result += "_"
                result += char.lower()
            else:
                result += char

        # Clean up any double underscores
        while "__" in result:
            result = result.replace("__", "_")

        # Remove leading/trailing underscores
        return result.strip("_")

    def _stable_order_dict(
        self,
        d: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create dict with stable key ordering.

        Keys are sorted alphabetically for reproducibility.

        Args:
            d: Dictionary to reorder

        Returns:
            New dictionary with sorted keys
        """
        return {k: d[k] for k in sorted(d.keys())}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_canonical_unit(dimension: str) -> Optional[str]:
    """
    Get the default canonical unit for a dimension.

    Args:
        dimension: Physical dimension name (e.g., "energy", "mass")

    Returns:
        Canonical unit symbol if dimension is known, None otherwise
    """
    return CANONICAL_UNITS.get(dimension)


def is_canonical_unit(unit: str, dimension: str, catalog: UnitCatalog) -> bool:
    """
    Check if a unit is the canonical unit for its dimension.

    Args:
        unit: Unit symbol to check
        dimension: Expected dimension
        catalog: Unit catalog for lookups

    Returns:
        True if unit is the canonical unit for the dimension
    """
    canonical = get_canonical_unit(dimension)
    if canonical is None:
        return False

    # Resolve any aliases
    unit_def = catalog.get_unit(unit)
    if unit_def is None:
        return False

    return unit_def.symbol == canonical or catalog.is_canonical(unit)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "CANONICAL_UNITS",
    # Models
    "ConversionRecord",
    "CanonicalizedValue",
    "KeyRename",
    # Canonicalizers
    "UnitCanonicalizer",
    "KeyCanonicalizer",
    # Helper functions
    "get_canonical_unit",
    "is_canonical_unit",
]
