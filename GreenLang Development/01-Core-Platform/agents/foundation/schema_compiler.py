# -*- coding: utf-8 -*-
"""
GL-FOUND-X-002: Schema Compiler & Validator
===========================================

The core schema validation and compilation engine for GreenLang Climate OS.
Validates input payloads against GreenLang schemas with zero-hallucination compliance.

Capabilities:
    - JSON Schema validation (Draft-07 specification)
    - Unit consistency checking (kgCO2e, tCO2e, MWh, etc.)
    - Machine-fixable error hints with actionable suggestions
    - Schema registry for managing GreenLang schemas
    - Safe type coercion with validation tracking
    - Complete provenance tracking for audit trails

Zero-Hallucination Guarantees:
    - All validation results have complete lineage
    - Deterministic validation with same inputs
    - All coercions tracked and versioned
    - All error hints based on schema rules, not inferred

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import re
import time
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.governance.validation.framework import (
    ValidationError,
    ValidationResult,
    ValidationSeverity,
)
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Type Definitions
# =============================================================================

# Supported unit families for consistency checking
UNIT_FAMILIES = {
    "mass_co2e": {"kgCO2e", "tCO2e", "gCO2e", "MtCO2e", "GtCO2e"},
    "mass": {"kg", "g", "t", "mt", "lb", "ton"},
    "energy": {"kWh", "MWh", "GWh", "TWh", "J", "kJ", "MJ", "GJ", "TJ", "BTU", "therm"},
    "volume": {"L", "m3", "gal", "barrel", "ft3"},
    "area": {"m2", "ft2", "ha", "acre", "km2"},
    "distance": {"km", "m", "mi", "ft", "nm"},
    "time": {"s", "min", "h", "d", "wk", "mo", "yr"},
    "currency": {"USD", "EUR", "GBP", "JPY", "CNY", "CHF", "CAD", "AUD"},
    "percentage": {"%", "percent", "pct"},
    "dimensionless": {"count", "unit", "each", "pcs"},
}

# Unit conversion factors to base units within each family
UNIT_CONVERSIONS = {
    # Mass CO2e to kgCO2e
    "gCO2e": 0.001,
    "kgCO2e": 1.0,
    "tCO2e": 1000.0,
    "MtCO2e": 1_000_000_000.0,
    "GtCO2e": 1_000_000_000_000.0,
    # Mass to kg
    "g": 0.001,
    "kg": 1.0,
    "t": 1000.0,
    "mt": 1000.0,
    "lb": 0.453592,
    "ton": 907.185,
    # Energy to kWh
    "J": 2.7778e-7,
    "kJ": 2.7778e-4,
    "MJ": 0.27778,
    "GJ": 277.78,
    "TJ": 277780.0,
    "kWh": 1.0,
    "MWh": 1000.0,
    "GWh": 1_000_000.0,
    "TWh": 1_000_000_000.0,
    "BTU": 2.931e-4,
    "therm": 29.3071,
}


class SchemaType(str, Enum):
    """Types of schemas supported by the registry."""
    JSON_SCHEMA = "json_schema"
    PYDANTIC = "pydantic"
    CUSTOM = "custom"


class CoercionType(str, Enum):
    """Types of type coercion supported."""
    STRING_TO_INT = "string_to_int"
    STRING_TO_FLOAT = "string_to_float"
    STRING_TO_BOOL = "string_to_bool"
    STRING_TO_DECIMAL = "string_to_decimal"
    INT_TO_FLOAT = "int_to_float"
    FLOAT_TO_INT = "float_to_int"
    NONE_TO_DEFAULT = "none_to_default"
    LIST_WRAP = "list_wrap"


class FixSuggestionType(str, Enum):
    """Types of fix suggestions for validation errors."""
    TYPE_COERCION = "type_coercion"
    VALUE_RANGE = "value_range"
    PATTERN_MATCH = "pattern_match"
    REQUIRED_FIELD = "required_field"
    UNIT_CONVERSION = "unit_conversion"
    FORMAT_CORRECTION = "format_correction"
    ENUM_SUGGESTION = "enum_suggestion"


# =============================================================================
# Data Models
# =============================================================================

class FixSuggestion(BaseModel):
    """A machine-fixable suggestion for correcting validation errors."""
    suggestion_type: FixSuggestionType = Field(
        ..., description="Type of fix suggestion"
    )
    field: str = Field(..., description="Field path requiring fix")
    original_value: Any = Field(None, description="Original invalid value")
    suggested_value: Any = Field(None, description="Suggested corrected value")
    description: str = Field(..., description="Human-readable fix description")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in suggestion (0-1)"
    )
    auto_fixable: bool = Field(
        default=False, description="Whether fix can be auto-applied"
    )
    code_snippet: Optional[str] = Field(
        None, description="Code snippet to implement fix"
    )


class UnitInfo(BaseModel):
    """Information about a unit and its family."""
    unit: str = Field(..., description="Unit symbol")
    family: str = Field(..., description="Unit family (mass_co2e, energy, etc.)")
    base_unit: str = Field(..., description="Base unit for this family")
    conversion_factor: Optional[float] = Field(
        None, description="Factor to convert to base unit"
    )


class CoercionRecord(BaseModel):
    """Record of a type coercion operation for audit trails."""
    field: str = Field(..., description="Field that was coerced")
    coercion_type: CoercionType = Field(..., description="Type of coercion")
    original_value: Any = Field(..., description="Value before coercion")
    original_type: str = Field(..., description="Type before coercion")
    coerced_value: Any = Field(..., description="Value after coercion")
    coerced_type: str = Field(..., description="Type after coercion")
    timestamp: datetime = Field(
        default_factory=DeterministicClock.now, description="When coercion occurred"
    )


class SchemaRegistryEntry(BaseModel):
    """An entry in the schema registry."""
    schema_id: str = Field(..., description="Unique schema identifier")
    schema_name: str = Field(..., description="Human-readable schema name")
    schema_version: str = Field(default="1.0.0", description="Schema version")
    schema_type: SchemaType = Field(
        default=SchemaType.JSON_SCHEMA, description="Type of schema"
    )
    schema_content: Dict[str, Any] = Field(..., description="Schema definition")
    description: str = Field(default="", description="Schema description")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    created_at: datetime = Field(
        default_factory=DeterministicClock.now, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=DeterministicClock.now, description="Last update timestamp"
    )
    content_hash: str = Field(default="", description="SHA-256 hash of schema content")

    @model_validator(mode='after')
    def compute_content_hash(self):
        """Compute content hash if not provided."""
        if not self.content_hash:
            json_str = json.dumps(self.schema_content, sort_keys=True, default=str)
            self.content_hash = hashlib.sha256(json_str.encode()).hexdigest()
        return self


class SchemaCompilerInput(BaseModel):
    """Input data model for SchemaCompilerAgent."""
    payload: Dict[str, Any] = Field(..., description="Data payload to validate")
    schema_id: Optional[str] = Field(
        None, description="Schema ID to validate against"
    )
    inline_schema: Optional[Dict[str, Any]] = Field(
        None, description="Inline schema definition"
    )
    enable_coercion: bool = Field(
        default=True, description="Enable automatic type coercion"
    )
    enable_unit_check: bool = Field(
        default=True, description="Enable unit consistency checking"
    )
    strict_mode: bool = Field(
        default=False, description="Fail on warnings in strict mode"
    )
    generate_fixes: bool = Field(
        default=True, description="Generate machine-fixable suggestions"
    )
    unit_fields: Optional[Dict[str, str]] = Field(
        None, description="Map of field paths to expected unit families"
    )

    @model_validator(mode='after')
    def validate_schema_provided(self):
        """Ensure at least one schema source is provided."""
        # This will be checked in the agent's validation
        return self


class SchemaCompilerOutput(BaseModel):
    """Output data model for SchemaCompilerAgent."""
    is_valid: bool = Field(..., description="Whether validation passed")
    validation_result: ValidationResult = Field(
        ..., description="Detailed validation result"
    )
    coerced_payload: Optional[Dict[str, Any]] = Field(
        None, description="Payload with coerced values"
    )
    coercion_records: List[CoercionRecord] = Field(
        default_factory=list, description="Records of all coercions"
    )
    fix_suggestions: List[FixSuggestion] = Field(
        default_factory=list, description="Machine-fixable suggestions"
    )
    unit_validations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Unit consistency check results"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., description="Processing duration")
    schema_used: str = Field(..., description="Schema ID or 'inline' if provided")


# =============================================================================
# Schema Registry
# =============================================================================

class SchemaRegistry:
    """
    Registry for managing GreenLang schemas.

    Provides storage, retrieval, and versioning of JSON schemas
    used for validation throughout the GreenLang platform.

    Example:
        registry = SchemaRegistry()
        registry.register("emissions-input", emissions_schema, "1.0.0")
        schema = registry.get("emissions-input")
    """

    def __init__(self):
        """Initialize the schema registry."""
        self._schemas: Dict[str, Dict[str, SchemaRegistryEntry]] = {}
        self._default_versions: Dict[str, str] = {}
        self._initialize_builtin_schemas()

    def _initialize_builtin_schemas(self):
        """Initialize built-in GreenLang schemas."""
        # Emissions data schema
        self.register(
            schema_id="gl-emissions-input",
            schema_name="GreenLang Emissions Input",
            schema_content={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "emissions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "string"},
                                "fuel_type": {"type": "string"},
                                "quantity": {"type": "number", "minimum": 0},
                                "unit": {"type": "string"},
                                "co2e_emissions_kg": {"type": "number"},
                                "scope": {"type": "integer", "enum": [1, 2, 3]},
                                "category": {"type": "string"},
                            },
                            "required": ["fuel_type", "co2e_emissions_kg"],
                        },
                    },
                    "reporting_period": {
                        "type": "object",
                        "properties": {
                            "start_date": {"type": "string", "format": "date"},
                            "end_date": {"type": "string", "format": "date"},
                        },
                    },
                    "organization_id": {"type": "string"},
                },
                "required": ["emissions"],
            },
            version="1.0.0",
            description="Standard input schema for emissions data",
            tags=["emissions", "input", "core"],
        )

        # Activity data schema
        self.register(
            schema_id="gl-activity-data",
            schema_name="GreenLang Activity Data",
            schema_content={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "activity_type": {"type": "string"},
                    "quantity": {"type": "number", "minimum": 0},
                    "unit": {"type": "string"},
                    "emission_factor": {"type": "number", "minimum": 0},
                    "emission_factor_unit": {"type": "string"},
                    "source": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["activity_type", "quantity", "unit"],
            },
            version="1.0.0",
            description="Schema for activity data inputs",
            tags=["activity", "input", "core"],
        )

        # Calculation result schema
        self.register(
            schema_id="gl-calculation-result",
            schema_name="GreenLang Calculation Result",
            schema_content={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "total_emissions": {"type": "number"},
                    "unit": {"type": "string", "enum": ["kgCO2e", "tCO2e"]},
                    "breakdown": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "value": {"type": "number"},
                                "percentage": {"type": "number"},
                            },
                        },
                    },
                    "provenance_hash": {"type": "string"},
                    "calculation_timestamp": {"type": "string", "format": "date-time"},
                },
                "required": ["total_emissions", "unit", "provenance_hash"],
            },
            version="1.0.0",
            description="Schema for calculation output results",
            tags=["calculation", "output", "core"],
        )

    def register(
        self,
        schema_id: str,
        schema_name: str,
        schema_content: Dict[str, Any],
        version: str = "1.0.0",
        description: str = "",
        tags: Optional[List[str]] = None,
        schema_type: SchemaType = SchemaType.JSON_SCHEMA,
    ) -> SchemaRegistryEntry:
        """
        Register a schema in the registry.

        Args:
            schema_id: Unique identifier for the schema
            schema_name: Human-readable name
            schema_content: The schema definition
            version: Schema version (semver)
            description: Schema description
            tags: Optional tags for categorization
            schema_type: Type of schema

        Returns:
            The registered schema entry
        """
        entry = SchemaRegistryEntry(
            schema_id=schema_id,
            schema_name=schema_name,
            schema_version=version,
            schema_type=schema_type,
            schema_content=schema_content,
            description=description,
            tags=tags or [],
        )

        if schema_id not in self._schemas:
            self._schemas[schema_id] = {}
            self._default_versions[schema_id] = version

        self._schemas[schema_id][version] = entry
        logger.info(f"Registered schema: {schema_id} v{version}")

        return entry

    def get(
        self,
        schema_id: str,
        version: Optional[str] = None
    ) -> Optional[SchemaRegistryEntry]:
        """
        Get a schema from the registry.

        Args:
            schema_id: Schema identifier
            version: Specific version (uses default if not provided)

        Returns:
            Schema entry or None if not found
        """
        if schema_id not in self._schemas:
            return None

        if version is None:
            version = self._default_versions.get(schema_id)

        return self._schemas[schema_id].get(version)

    def get_schema_content(
        self,
        schema_id: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get just the schema content."""
        entry = self.get(schema_id, version)
        return entry.schema_content if entry else None

    def list_schemas(
        self,
        tags: Optional[List[str]] = None
    ) -> List[SchemaRegistryEntry]:
        """
        List all schemas, optionally filtered by tags.

        Args:
            tags: Optional tags to filter by

        Returns:
            List of schema entries
        """
        entries = []
        for schema_versions in self._schemas.values():
            for entry in schema_versions.values():
                if tags is None or any(t in entry.tags for t in tags):
                    entries.append(entry)
        return entries

    def get_versions(self, schema_id: str) -> List[str]:
        """Get all versions of a schema."""
        if schema_id not in self._schemas:
            return []
        return list(self._schemas[schema_id].keys())

    def set_default_version(self, schema_id: str, version: str):
        """Set the default version for a schema."""
        if schema_id in self._schemas and version in self._schemas[schema_id]:
            self._default_versions[schema_id] = version

    def unregister(self, schema_id: str, version: Optional[str] = None):
        """Remove a schema from the registry."""
        if schema_id not in self._schemas:
            return

        if version:
            if version in self._schemas[schema_id]:
                del self._schemas[schema_id][version]
        else:
            del self._schemas[schema_id]
            if schema_id in self._default_versions:
                del self._default_versions[schema_id]


# =============================================================================
# Type Coercion Engine
# =============================================================================

class TypeCoercionEngine:
    """
    Engine for safe type coercion with complete tracking.

    Performs type conversions that are safe and reversible,
    tracking all coercions for audit trails.
    """

    # Patterns for string parsing
    BOOL_TRUE_VALUES = {"true", "yes", "1", "on", "enabled", "t", "y"}
    BOOL_FALSE_VALUES = {"false", "no", "0", "off", "disabled", "f", "n"}
    INT_PATTERN = re.compile(r"^-?\d+$")
    FLOAT_PATTERN = re.compile(r"^-?\d+\.?\d*([eE][+-]?\d+)?$")

    def __init__(self):
        """Initialize the coercion engine."""
        self._coercion_records: List[CoercionRecord] = []

    def clear_records(self):
        """Clear coercion records."""
        self._coercion_records = []

    def get_records(self) -> List[CoercionRecord]:
        """Get all coercion records."""
        return self._coercion_records.copy()

    def coerce(
        self,
        value: Any,
        target_type: str,
        field: str,
        allow_lossy: bool = False
    ) -> Tuple[Any, bool, Optional[CoercionRecord]]:
        """
        Attempt to coerce a value to the target type.

        Args:
            value: Value to coerce
            target_type: Target JSON Schema type
            field: Field path for tracking
            allow_lossy: Allow potentially lossy conversions

        Returns:
            Tuple of (coerced_value, success, coercion_record)
        """
        original_type = type(value).__name__

        # No coercion needed
        if self._matches_type(value, target_type):
            return value, True, None

        coercion_type = None
        coerced_value = None
        success = False

        try:
            if target_type == "integer":
                coerced_value, coercion_type = self._coerce_to_integer(
                    value, allow_lossy
                )
                success = coerced_value is not None
            elif target_type == "number":
                coerced_value, coercion_type = self._coerce_to_number(value)
                success = coerced_value is not None
            elif target_type == "string":
                coerced_value = str(value)
                coercion_type = CoercionType.STRING_TO_INT  # Generic
                success = True
            elif target_type == "boolean":
                coerced_value, coercion_type = self._coerce_to_boolean(value)
                success = coerced_value is not None
            elif target_type == "array":
                if not isinstance(value, list):
                    coerced_value = [value]
                    coercion_type = CoercionType.LIST_WRAP
                    success = True
        except (ValueError, TypeError, InvalidOperation):
            success = False

        if success and coercion_type:
            record = CoercionRecord(
                field=field,
                coercion_type=coercion_type,
                original_value=value,
                original_type=original_type,
                coerced_value=coerced_value,
                coerced_type=type(coerced_value).__name__,
            )
            self._coercion_records.append(record)
            return coerced_value, True, record

        return value, False, None

    def _matches_type(self, value: Any, json_type: Union[str, List[str]]) -> bool:
        """Check if value matches JSON Schema type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        # Handle union types (e.g., ["string", "null"])
        if isinstance(json_type, list):
            return any(self._matches_type(value, t) for t in json_type)

        expected = type_mapping.get(json_type)
        if expected is None:
            return True

        # Special case: boolean is subclass of int in Python
        if json_type == "integer" and isinstance(value, bool):
            return False

        return isinstance(value, expected)

    def _coerce_to_integer(
        self,
        value: Any,
        allow_lossy: bool
    ) -> Tuple[Optional[int], Optional[CoercionType]]:
        """Coerce value to integer."""
        if isinstance(value, str):
            value = value.strip()
            if self.INT_PATTERN.match(value):
                return int(value), CoercionType.STRING_TO_INT
            elif self.FLOAT_PATTERN.match(value):
                float_val = float(value)
                if float_val.is_integer() or allow_lossy:
                    return int(float_val), CoercionType.STRING_TO_INT
        elif isinstance(value, float):
            if value.is_integer() or allow_lossy:
                return int(value), CoercionType.FLOAT_TO_INT
        return None, None

    def _coerce_to_number(
        self,
        value: Any
    ) -> Tuple[Optional[float], Optional[CoercionType]]:
        """Coerce value to number (float)."""
        if isinstance(value, str):
            value = value.strip()
            if self.FLOAT_PATTERN.match(value):
                return float(value), CoercionType.STRING_TO_FLOAT
        elif isinstance(value, int) and not isinstance(value, bool):
            return float(value), CoercionType.INT_TO_FLOAT
        return None, None

    def _coerce_to_boolean(
        self,
        value: Any
    ) -> Tuple[Optional[bool], Optional[CoercionType]]:
        """Coerce value to boolean."""
        if isinstance(value, str):
            lower_value = value.lower().strip()
            if lower_value in self.BOOL_TRUE_VALUES:
                return True, CoercionType.STRING_TO_BOOL
            elif lower_value in self.BOOL_FALSE_VALUES:
                return False, CoercionType.STRING_TO_BOOL
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value), CoercionType.STRING_TO_BOOL
        return None, None


# =============================================================================
# Unit Consistency Checker
# =============================================================================

class UnitConsistencyChecker:
    """
    Checker for unit consistency in emission calculations.

    Ensures units across related fields are compatible and from
    the same unit family (e.g., mass_co2e, energy).
    """

    def __init__(self):
        """Initialize the unit checker."""
        self._unit_to_family: Dict[str, str] = {}
        self._initialize_unit_mapping()

    def _initialize_unit_mapping(self):
        """Build reverse mapping from units to families."""
        for family, units in UNIT_FAMILIES.items():
            for unit in units:
                self._unit_to_family[unit] = family
                self._unit_to_family[unit.lower()] = family

    def get_unit_info(self, unit: str) -> Optional[UnitInfo]:
        """
        Get information about a unit.

        Args:
            unit: Unit symbol

        Returns:
            UnitInfo or None if unknown unit
        """
        family = self._unit_to_family.get(unit) or self._unit_to_family.get(unit.lower())
        if not family:
            return None

        # Determine base unit for family
        base_units = {
            "mass_co2e": "kgCO2e",
            "mass": "kg",
            "energy": "kWh",
            "volume": "L",
            "area": "m2",
            "distance": "km",
            "time": "h",
            "currency": "USD",
            "percentage": "%",
            "dimensionless": "unit",
        }
        base_unit = base_units.get(family, "unit")
        conversion_factor = UNIT_CONVERSIONS.get(unit)

        return UnitInfo(
            unit=unit,
            family=family,
            base_unit=base_unit,
            conversion_factor=conversion_factor,
        )

    def check_consistency(
        self,
        units: List[Tuple[str, str]],
        expected_family: Optional[str] = None
    ) -> ValidationResult:
        """
        Check if a list of units are consistent.

        Args:
            units: List of (field_path, unit) tuples
            expected_family: Expected unit family

        Returns:
            ValidationResult with any inconsistencies
        """
        result = ValidationResult(valid=True)

        if not units:
            return result

        families_found: Dict[str, List[str]] = {}

        for field_path, unit in units:
            info = self.get_unit_info(unit)
            if info is None:
                error = ValidationError(
                    field=field_path,
                    message=f"Unknown unit '{unit}'",
                    severity=ValidationSeverity.WARNING,
                    validator="unit_consistency",
                    value=unit,
                )
                result.add_error(error)
                continue

            if info.family not in families_found:
                families_found[info.family] = []
            families_found[info.family].append(f"{field_path}={unit}")

        # Check for expected family
        if expected_family and expected_family not in families_found:
            found_families = list(families_found.keys())
            error = ValidationError(
                field="units",
                message=f"Expected unit family '{expected_family}', found: {found_families}",
                severity=ValidationSeverity.ERROR,
                validator="unit_consistency",
                expected=expected_family,
                value=found_families,
            )
            result.add_error(error)

        # Check for mixed families
        if len(families_found) > 1:
            error = ValidationError(
                field="units",
                message=f"Mixed unit families detected: {dict(families_found)}",
                severity=ValidationSeverity.WARNING,
                validator="unit_consistency",
                value=dict(families_found),
            )
            result.add_error(error)

        return result

    def suggest_conversion(
        self,
        from_unit: str,
        to_unit: str
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest a unit conversion if possible.

        Args:
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Conversion details or None if not possible
        """
        from_info = self.get_unit_info(from_unit)
        to_info = self.get_unit_info(to_unit)

        if not from_info or not to_info:
            return None

        if from_info.family != to_info.family:
            return None

        from_factor = UNIT_CONVERSIONS.get(from_unit, 1.0)
        to_factor = UNIT_CONVERSIONS.get(to_unit, 1.0)

        conversion_factor = from_factor / to_factor

        return {
            "from_unit": from_unit,
            "to_unit": to_unit,
            "conversion_factor": conversion_factor,
            "formula": f"{from_unit} * {conversion_factor} = {to_unit}",
        }


# =============================================================================
# Fix Suggestion Generator
# =============================================================================

class FixSuggestionGenerator:
    """
    Generator for machine-fixable error suggestions.

    Analyzes validation errors and generates actionable
    suggestions that can be automatically applied.
    """

    def __init__(self, coercion_engine: TypeCoercionEngine):
        """Initialize the suggestion generator."""
        self._coercion_engine = coercion_engine
        self._unit_checker = UnitConsistencyChecker()

    def generate_suggestions(
        self,
        error: ValidationError,
        schema: Dict[str, Any],
        value: Any
    ) -> List[FixSuggestion]:
        """
        Generate fix suggestions for a validation error.

        Args:
            error: The validation error
            schema: Schema that caused the error
            value: The invalid value

        Returns:
            List of fix suggestions
        """
        suggestions = []

        # Type mismatch suggestions
        if "type" in error.message.lower():
            type_suggestions = self._generate_type_suggestions(error, schema, value)
            suggestions.extend(type_suggestions)

        # Required field suggestions
        if "required" in error.message.lower():
            required_suggestions = self._generate_required_suggestions(error, schema)
            suggestions.extend(required_suggestions)

        # Enum value suggestions
        if "enum" in error.message.lower() or "is not one of" in error.message.lower():
            enum_suggestions = self._generate_enum_suggestions(error, schema, value)
            suggestions.extend(enum_suggestions)

        # Range/minimum/maximum suggestions
        if any(kw in error.message.lower() for kw in ["minimum", "maximum", "range"]):
            range_suggestions = self._generate_range_suggestions(error, schema, value)
            suggestions.extend(range_suggestions)

        # Pattern mismatch suggestions
        if "pattern" in error.message.lower():
            pattern_suggestions = self._generate_pattern_suggestions(error, schema, value)
            suggestions.extend(pattern_suggestions)

        return suggestions

    def _generate_type_suggestions(
        self,
        error: ValidationError,
        schema: Dict[str, Any],
        value: Any
    ) -> List[FixSuggestion]:
        """Generate suggestions for type mismatches."""
        suggestions = []
        expected_type = schema.get("type", "unknown")

        # Try coercion
        coerced, success, _ = self._coercion_engine.coerce(
            value, expected_type, error.field
        )

        if success:
            suggestions.append(FixSuggestion(
                suggestion_type=FixSuggestionType.TYPE_COERCION,
                field=error.field,
                original_value=value,
                suggested_value=coerced,
                description=f"Convert {type(value).__name__} to {expected_type}",
                confidence=0.95,
                auto_fixable=True,
                code_snippet=f'data["{error.field}"] = {repr(coerced)}',
            ))
        else:
            # Provide manual fix suggestion
            default_values = {
                "string": '""',
                "integer": "0",
                "number": "0.0",
                "boolean": "false",
                "array": "[]",
                "object": "{}",
            }
            suggestions.append(FixSuggestion(
                suggestion_type=FixSuggestionType.TYPE_COERCION,
                field=error.field,
                original_value=value,
                suggested_value=None,
                description=f"Value must be of type {expected_type}. "
                           f"Example: {default_values.get(expected_type, 'null')}",
                confidence=0.5,
                auto_fixable=False,
            ))

        return suggestions

    def _generate_required_suggestions(
        self,
        error: ValidationError,
        schema: Dict[str, Any]
    ) -> List[FixSuggestion]:
        """Generate suggestions for missing required fields."""
        suggestions = []

        # Get field schema if possible
        properties = schema.get("properties", {})
        field_schema = properties.get(error.field, {})
        field_type = field_schema.get("type", "string")

        # Suggest default value based on type
        default_values = {
            "string": "",
            "integer": 0,
            "number": 0.0,
            "boolean": False,
            "array": [],
            "object": {},
        }

        default_value = field_schema.get("default", default_values.get(field_type))

        suggestions.append(FixSuggestion(
            suggestion_type=FixSuggestionType.REQUIRED_FIELD,
            field=error.field,
            original_value=None,
            suggested_value=default_value,
            description=f"Add required field '{error.field}' with type {field_type}",
            confidence=0.8,
            auto_fixable=True if default_value is not None else False,
            code_snippet=f'data["{error.field}"] = {repr(default_value)}',
        ))

        return suggestions

    def _generate_enum_suggestions(
        self,
        error: ValidationError,
        schema: Dict[str, Any],
        value: Any
    ) -> List[FixSuggestion]:
        """Generate suggestions for invalid enum values."""
        suggestions = []

        enum_values = schema.get("enum", [])
        if not enum_values:
            return suggestions

        # Find closest match using simple string similarity
        if isinstance(value, str):
            closest = self._find_closest_match(value, [str(e) for e in enum_values])
            if closest:
                suggestions.append(FixSuggestion(
                    suggestion_type=FixSuggestionType.ENUM_SUGGESTION,
                    field=error.field,
                    original_value=value,
                    suggested_value=closest,
                    description=f"Did you mean '{closest}'? Valid values: {enum_values}",
                    confidence=0.7,
                    auto_fixable=True,
                    code_snippet=f'data["{error.field}"] = {repr(closest)}',
                ))
        else:
            suggestions.append(FixSuggestion(
                suggestion_type=FixSuggestionType.ENUM_SUGGESTION,
                field=error.field,
                original_value=value,
                suggested_value=enum_values[0] if enum_values else None,
                description=f"Value must be one of: {enum_values}",
                confidence=0.5,
                auto_fixable=False,
            ))

        return suggestions

    def _generate_range_suggestions(
        self,
        error: ValidationError,
        schema: Dict[str, Any],
        value: Any
    ) -> List[FixSuggestion]:
        """Generate suggestions for out-of-range values."""
        suggestions = []

        minimum = schema.get("minimum")
        maximum = schema.get("maximum")

        suggested_value = value
        if minimum is not None and isinstance(value, (int, float)) and value < minimum:
            suggested_value = minimum
        elif maximum is not None and isinstance(value, (int, float)) and value > maximum:
            suggested_value = maximum

        if suggested_value != value:
            range_str = ""
            if minimum is not None and maximum is not None:
                range_str = f"[{minimum}, {maximum}]"
            elif minimum is not None:
                range_str = f">= {minimum}"
            elif maximum is not None:
                range_str = f"<= {maximum}"

            suggestions.append(FixSuggestion(
                suggestion_type=FixSuggestionType.VALUE_RANGE,
                field=error.field,
                original_value=value,
                suggested_value=suggested_value,
                description=f"Value must be in range {range_str}",
                confidence=0.9,
                auto_fixable=True,
                code_snippet=f'data["{error.field}"] = {suggested_value}',
            ))

        return suggestions

    def _generate_pattern_suggestions(
        self,
        error: ValidationError,
        schema: Dict[str, Any],
        value: Any
    ) -> List[FixSuggestion]:
        """Generate suggestions for pattern mismatches."""
        suggestions = []

        pattern = schema.get("pattern", "")

        suggestions.append(FixSuggestion(
            suggestion_type=FixSuggestionType.PATTERN_MATCH,
            field=error.field,
            original_value=value,
            suggested_value=None,
            description=f"Value must match pattern: {pattern}",
            confidence=0.3,
            auto_fixable=False,
        ))

        return suggestions

    def _find_closest_match(
        self,
        value: str,
        candidates: List[str]
    ) -> Optional[str]:
        """Find closest string match using simple similarity."""
        if not candidates:
            return None

        value_lower = value.lower()
        best_match = None
        best_score = 0

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Simple substring matching
            if value_lower in candidate_lower or candidate_lower in value_lower:
                score = len(value_lower) / max(len(candidate_lower), 1)
                if score > best_score:
                    best_score = score
                    best_match = candidate

            # Prefix matching
            common_prefix = 0
            for i in range(min(len(value_lower), len(candidate_lower))):
                if value_lower[i] == candidate_lower[i]:
                    common_prefix += 1
                else:
                    break
            prefix_score = common_prefix / max(len(candidate_lower), 1)
            if prefix_score > best_score and prefix_score > 0.5:
                best_score = prefix_score
                best_match = candidate

        return best_match if best_score > 0.3 else None


# =============================================================================
# Schema Compiler Agent
# =============================================================================

class SchemaCompilerAgent(BaseAgent):
    """
    GL-FOUND-X-002: Schema Compiler & Validator

    The core schema validation and compilation engine for GreenLang Climate OS.
    Validates input payloads against GreenLang schemas with comprehensive
    error reporting and machine-fixable suggestions.

    Zero-Hallucination Guarantees:
        - All validation results have complete lineage
        - Deterministic validation with same inputs
        - All coercions tracked and versioned
        - All error hints based on schema rules, not inferred

    Capabilities:
        - JSON Schema validation (Draft-07)
        - Unit consistency checking
        - Machine-fixable error hints
        - Schema registry management
        - Safe type coercion

    Example:
        agent = SchemaCompilerAgent()
        result = agent.run({
            "payload": {"emissions": [{"fuel_type": "gas", "co2e_emissions_kg": "100"}]},
            "schema_id": "gl-emissions-input",
            "enable_coercion": True
        })
    """

    AGENT_ID = "GL-FOUND-X-002"
    AGENT_NAME = "Schema Compiler & Validator"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Schema Compiler Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Schema validation and compilation engine for GreenLang",
                version=self.VERSION,
                parameters={
                    "enable_coercion": True,
                    "enable_unit_check": True,
                    "strict_mode": False,
                    "generate_fixes": True,
                }
            )
        super().__init__(config)

        # Initialize components
        self._schema_registry = SchemaRegistry()
        self._coercion_engine = TypeCoercionEngine()
        self._unit_checker = UnitConsistencyChecker()
        self._fix_generator = FixSuggestionGenerator(self._coercion_engine)

        # Try to import jsonschema for full validation
        try:
            import jsonschema
            from jsonschema import Draft7Validator
            self._jsonschema_available = True
            self._Draft7Validator = Draft7Validator
        except ImportError:
            self._jsonschema_available = False
            self._Draft7Validator = None
            logger.warning("jsonschema not available, using basic validation")

    @property
    def schema_registry(self) -> SchemaRegistry:
        """Get the schema registry."""
        return self._schema_registry

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data before processing.

        Args:
            input_data: Input data to validate

        Returns:
            True if valid
        """
        if not isinstance(input_data, dict):
            return False

        if "payload" not in input_data:
            return False

        # Must have either schema_id or schema (support both "schema" and "inline_schema")
        has_schema = (
            "schema_id" in input_data or
            "schema" in input_data or
            "inline_schema" in input_data
        )
        if not has_schema:
            return False

        return True

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute schema validation.

        Args:
            input_data: Input containing payload and schema info

        Returns:
            AgentResult with validation results
        """
        start_time = time.time()

        try:
            # Parse input
            payload = input_data.get("payload", {})
            schema_id = input_data.get("schema_id")
            # Support both "schema" (legacy) and "inline_schema" (new)
            inline_schema = input_data.get("inline_schema") or input_data.get("schema")
            enable_coercion = input_data.get(
                "enable_coercion",
                self.config.parameters.get("enable_coercion", True)
            )
            enable_unit_check = input_data.get(
                "enable_unit_check",
                self.config.parameters.get("enable_unit_check", True)
            )
            strict_mode = input_data.get(
                "strict_mode",
                self.config.parameters.get("strict_mode", False)
            )
            generate_fixes = input_data.get(
                "generate_fixes",
                self.config.parameters.get("generate_fixes", True)
            )
            unit_fields = input_data.get("unit_fields", {})

            # Get schema
            schema: Dict[str, Any] = {}
            schema_used = "unknown"

            if inline_schema:
                schema = inline_schema
                schema_used = "inline"
            elif schema_id:
                schema_entry = self._schema_registry.get(schema_id)
                if schema_entry:
                    schema = schema_entry.schema_content
                    schema_used = f"{schema_id}@{schema_entry.schema_version}"
                else:
                    return AgentResult(
                        success=False,
                        error=f"Schema not found: {schema_id}",
                        data={"schema_id": schema_id}
                    )

            # Clear previous coercion records
            self._coercion_engine.clear_records()

            # Perform validation
            validation_result = self._validate_against_schema(payload, schema)

            # Perform type coercion if enabled
            coerced_payload = None
            if enable_coercion:
                coerced_payload = self._apply_coercion(payload, schema)

            # Check unit consistency if enabled
            unit_validations = []
            if enable_unit_check:
                unit_result = self._check_units(payload, unit_fields)
                validation_result.merge(unit_result)
                unit_validations = self._extract_unit_validations(payload, unit_fields)

            # Generate fix suggestions if enabled
            fix_suggestions = []
            if generate_fixes and not validation_result.valid:
                fix_suggestions = self._generate_all_fixes(
                    validation_result.errors, schema, payload
                )

            # Determine final validity
            is_valid = validation_result.valid
            if strict_mode and validation_result.warnings:
                is_valid = False

            # Calculate provenance hash
            provenance_data = {
                "input_payload": payload,
                "schema_used": schema_used,
                "coercion_enabled": enable_coercion,
                "validation_result": validation_result.valid,
            }
            provenance_hash = self._compute_provenance_hash(provenance_data)

            processing_time = (time.time() - start_time) * 1000

            # Build output
            output = SchemaCompilerOutput(
                is_valid=is_valid,
                validation_result=validation_result,
                coerced_payload=coerced_payload,
                coercion_records=self._coercion_engine.get_records(),
                fix_suggestions=fix_suggestions,
                unit_validations=unit_validations,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time,
                schema_used=schema_used,
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
                metadata={
                    "agent": self.AGENT_ID,
                    "version": self.VERSION,
                    "schema_used": schema_used,
                    "is_valid": is_valid,
                    "error_count": len(validation_result.errors),
                    "warning_count": len(validation_result.warnings),
                    "coercion_count": len(self._coercion_engine.get_records()),
                    "fix_suggestion_count": len(fix_suggestions),
                }
            )

        except Exception as e:
            logger.error(f"Schema validation failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                data={"exception_type": type(e).__name__}
            )

    def _validate_against_schema(
        self,
        payload: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate payload against JSON schema.

        Args:
            payload: Data to validate
            schema: JSON Schema

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        if not self._jsonschema_available:
            # Basic validation without jsonschema
            return self._basic_schema_validation(payload, schema)

        try:
            validator = self._Draft7Validator(schema)
            errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)

            for error in errors:
                field_path = ".".join(str(p) for p in error.path) if error.path else "root"

                validation_error = ValidationError(
                    field=field_path,
                    message=error.message,
                    severity=ValidationSeverity.ERROR,
                    validator="json_schema_draft07",
                    value=error.instance if hasattr(error, 'instance') else None,
                    expected=str(error.schema) if hasattr(error, 'schema') else None,
                    location=f"$.{field_path}",
                )
                result.add_error(validation_error)

        except Exception as e:
            logger.error(f"JSON Schema validation error: {e}")
            result.add_error(ValidationError(
                field="__schema__",
                message=f"Schema validation failed: {str(e)}",
                severity=ValidationSeverity.ERROR,
                validator="json_schema_draft07",
            ))

        return result

    def _basic_schema_validation(
        self,
        payload: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Basic validation without jsonschema library.

        Args:
            payload: Data to validate
            schema: JSON Schema

        Returns:
            ValidationResult
        """
        result = ValidationResult(valid=True)

        schema_type = schema.get("type")

        # Check root type
        if schema_type:
            if not self._check_type(payload, schema_type):
                result.add_error(ValidationError(
                    field="root",
                    message=f"Expected type {schema_type}, got {type(payload).__name__}",
                    severity=ValidationSeverity.ERROR,
                    validator="basic_schema",
                    value=payload,
                    expected=schema_type,
                ))
                return result

        # Check required properties
        if schema_type == "object" and isinstance(payload, dict):
            required = schema.get("required", [])
            for field in required:
                if field not in payload:
                    result.add_error(ValidationError(
                        field=field,
                        message=f"Required field '{field}' is missing",
                        severity=ValidationSeverity.ERROR,
                        validator="basic_schema",
                        expected="required",
                    ))

            # Validate properties
            properties = schema.get("properties", {})
            for field, field_schema in properties.items():
                if field in payload:
                    field_result = self._validate_field(
                        payload[field], field_schema, field
                    )
                    result.merge(field_result)

        return result

    def _validate_field(
        self,
        value: Any,
        field_schema: Dict[str, Any],
        field_path: str
    ) -> ValidationResult:
        """Validate a single field against its schema."""
        result = ValidationResult(valid=True)

        field_type = field_schema.get("type")
        if field_type and not self._check_type(value, field_type):
            result.add_error(ValidationError(
                field=field_path,
                message=f"Expected type {field_type}, got {type(value).__name__}",
                severity=ValidationSeverity.ERROR,
                validator="basic_schema",
                value=value,
                expected=field_type,
            ))

        # Check minimum/maximum for numbers
        if isinstance(value, (int, float)):
            minimum = field_schema.get("minimum")
            maximum = field_schema.get("maximum")
            if minimum is not None and value < minimum:
                result.add_error(ValidationError(
                    field=field_path,
                    message=f"Value {value} is less than minimum {minimum}",
                    severity=ValidationSeverity.ERROR,
                    validator="basic_schema",
                    value=value,
                    expected=f">= {minimum}",
                ))
            if maximum is not None and value > maximum:
                result.add_error(ValidationError(
                    field=field_path,
                    message=f"Value {value} is greater than maximum {maximum}",
                    severity=ValidationSeverity.ERROR,
                    validator="basic_schema",
                    value=value,
                    expected=f"<= {maximum}",
                ))

        # Check enum
        enum_values = field_schema.get("enum")
        if enum_values is not None and value not in enum_values:
            result.add_error(ValidationError(
                field=field_path,
                message=f"Value {value} is not one of {enum_values}",
                severity=ValidationSeverity.ERROR,
                validator="basic_schema",
                value=value,
                expected=enum_values,
            ))

        return result

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON Schema type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        expected = type_mapping.get(expected_type)
        if expected is None:
            return True

        # Special case: boolean is subclass of int
        if expected_type == "integer" and isinstance(value, bool):
            return False

        return isinstance(value, expected)

    def _apply_coercion(
        self,
        payload: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply type coercion to payload based on schema.

        Args:
            payload: Original payload
            schema: JSON Schema

        Returns:
            Coerced payload
        """
        import copy
        coerced = copy.deepcopy(payload)

        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in coerced:
                expected_type = field_schema.get("type")
                if expected_type:
                    coerced_value, success, _ = self._coercion_engine.coerce(
                        coerced[field], expected_type, field
                    )
                    if success:
                        coerced[field] = coerced_value

        # Handle nested arrays
        if "items" in schema and isinstance(coerced, list):
            items_schema = schema["items"]
            for i, item in enumerate(coerced):
                if isinstance(item, dict):
                    coerced[i] = self._apply_coercion(item, items_schema)

        return coerced

    def _check_units(
        self,
        payload: Dict[str, Any],
        unit_fields: Dict[str, str]
    ) -> ValidationResult:
        """
        Check unit consistency in payload.

        Args:
            payload: Data payload
            unit_fields: Map of field paths to expected unit families

        Returns:
            ValidationResult with unit issues
        """
        result = ValidationResult(valid=True)

        # Extract units from payload
        units: List[Tuple[str, str]] = []
        self._extract_units(payload, "", units)

        # Check each unit field
        for field_path, expected_family in unit_fields.items():
            field_units = [(f, u) for f, u in units if f.startswith(field_path)]
            unit_result = self._unit_checker.check_consistency(
                field_units, expected_family
            )
            result.merge(unit_result)

        # General consistency check if no specific fields
        if not unit_fields and units:
            unit_result = self._unit_checker.check_consistency(units)
            result.merge(unit_result)

        return result

    def _extract_units(
        self,
        data: Any,
        path: str,
        units: List[Tuple[str, str]]
    ):
        """Recursively extract unit fields from data."""
        if isinstance(data, dict):
            # Look for unit fields
            for key in ["unit", "units", "emission_unit", "emission_factor_unit"]:
                if key in data and isinstance(data[key], str):
                    field_path = f"{path}.{key}" if path else key
                    units.append((field_path, data[key]))

            # Recurse into nested dicts
            for key, value in data.items():
                nested_path = f"{path}.{key}" if path else key
                self._extract_units(value, nested_path, units)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                nested_path = f"{path}[{i}]"
                self._extract_units(item, nested_path, units)

    def _extract_unit_validations(
        self,
        payload: Dict[str, Any],
        unit_fields: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Extract unit validation details."""
        validations = []
        units: List[Tuple[str, str]] = []
        self._extract_units(payload, "", units)

        for field_path, unit in units:
            info = self._unit_checker.get_unit_info(unit)
            validations.append({
                "field": field_path,
                "unit": unit,
                "valid": info is not None,
                "family": info.family if info else "unknown",
                "base_unit": info.base_unit if info else None,
            })

        return validations

    def _generate_all_fixes(
        self,
        errors: List[ValidationError],
        schema: Dict[str, Any],
        payload: Dict[str, Any]
    ) -> List[FixSuggestion]:
        """Generate fix suggestions for all errors."""
        suggestions = []
        properties = schema.get("properties", {})

        for error in errors:
            field_schema = properties.get(error.field, schema)
            value = payload.get(error.field) if isinstance(payload, dict) else None

            field_suggestions = self._fix_generator.generate_suggestions(
                error, field_schema, value
            )
            suggestions.extend(field_suggestions)

        return suggestions

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # ==========================================================================
    # Public API Methods
    # ==========================================================================

    def validate(
        self,
        payload: Dict[str, Any],
        schema_id: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SchemaCompilerOutput:
        """
        Convenience method for validation.

        Args:
            payload: Data to validate
            schema_id: Schema ID from registry
            schema: Inline schema
            **kwargs: Additional options

        Returns:
            SchemaCompilerOutput
        """
        input_data = {
            "payload": payload,
            **kwargs
        }
        if schema_id:
            input_data["schema_id"] = schema_id
        if schema:
            input_data["schema"] = schema

        result = self.run(input_data)
        if result.success:
            return SchemaCompilerOutput(**result.data)
        else:
            return SchemaCompilerOutput(
                is_valid=False,
                validation_result=ValidationResult(valid=False),
                provenance_hash="error",
                processing_time_ms=0,
                schema_used="error",
            )

    def register_schema(
        self,
        schema_id: str,
        schema_name: str,
        schema_content: Dict[str, Any],
        version: str = "1.0.0",
        **kwargs
    ) -> SchemaRegistryEntry:
        """
        Register a new schema.

        Args:
            schema_id: Unique schema identifier
            schema_name: Human-readable name
            schema_content: Schema definition
            version: Schema version
            **kwargs: Additional options

        Returns:
            Registered schema entry
        """
        return self._schema_registry.register(
            schema_id=schema_id,
            schema_name=schema_name,
            schema_content=schema_content,
            version=version,
            **kwargs
        )

    def get_schema(
        self,
        schema_id: str,
        version: Optional[str] = None
    ) -> Optional[SchemaRegistryEntry]:
        """Get a schema from the registry."""
        return self._schema_registry.get(schema_id, version)

    def list_schemas(
        self,
        tags: Optional[List[str]] = None
    ) -> List[SchemaRegistryEntry]:
        """List all registered schemas."""
        return self._schema_registry.list_schemas(tags)

    def get_unit_info(self, unit: str) -> Optional[UnitInfo]:
        """Get information about a unit."""
        return self._unit_checker.get_unit_info(unit)

    def suggest_unit_conversion(
        self,
        from_unit: str,
        to_unit: str
    ) -> Optional[Dict[str, Any]]:
        """Get unit conversion suggestion."""
        return self._unit_checker.suggest_conversion(from_unit, to_unit)
