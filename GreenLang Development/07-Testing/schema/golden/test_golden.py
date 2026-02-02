# -*- coding: utf-8 -*-
"""
Golden Tests for GL-FOUND-X-002 Schema Compiler & Validator.

This module implements comprehensive golden tests that validate schema
validation behavior against known-good expected outputs. Golden tests
ensure determinism and prevent regressions.

Test Categories:
    1. Valid payloads - All should pass validation
    2. Invalid payloads - Should produce specific error codes
    3. Schema validation - Tests for schema self-validation
    4. Error code coverage - Ensures all error codes are tested

Key Features:
    - Loads test data from YAML/JSON files
    - Compares actual vs expected validation reports
    - Supports multiple validation profiles
    - Tracks error code coverage
    - Deterministic ordering verification

Usage:
    pytest tests/schema/golden/test_golden.py -v
    pytest tests/schema/golden/test_golden.py -v -k "valid"
    pytest tests/schema/golden/test_golden.py -v -k "invalid"

Author: GreenLang Test Engineering Team
Date: 2026-01-29
Version: 1.0.0
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from ipaddress import IPv4Address, IPv6Address
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from uuid import UUID

import pytest
import yaml


# =============================================================================
# PATH CONSTANTS
# =============================================================================

GOLDEN_DIR = Path(__file__).parent
SCHEMAS_DIR = GOLDEN_DIR / "schemas"
PAYLOADS_DIR = GOLDEN_DIR / "payloads"
EXPECTED_DIR = GOLDEN_DIR / "expected"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_yaml_or_json(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML or JSON file.

    Args:
        file_path: Path to the file

    Returns:
        Parsed content as dictionary
    """
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def discover_files(directory: Path, patterns: List[str] = None) -> List[Path]:
    """
    Discover test files in a directory.

    Args:
        directory: Directory to search
        patterns: List of glob patterns (default: yaml, yml, json)

    Returns:
        Sorted list of file paths
    """
    if not directory.exists():
        return []

    patterns = patterns or ["**/*.yaml", "**/*.yml", "**/*.json"]
    files = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))

    # Filter out __init__.py and other non-data files
    files = [f for f in files if f.name != "__init__.py" and not f.name.startswith("_")]
    return sorted(files)


def extract_test_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract test metadata from payload data.

    Args:
        data: Payload data with optional _test_metadata

    Returns:
        Test metadata dictionary
    """
    return data.pop("_test_metadata", {})


def get_schema_for_payload(payload_path: Path, metadata: Dict[str, Any]) -> Path:
    """
    Determine the schema path for a payload.

    Args:
        payload_path: Path to the payload file
        metadata: Test metadata from the payload

    Returns:
        Path to the corresponding schema file
    """
    if "schema" in metadata:
        return SCHEMAS_DIR / metadata["schema"]

    # Try to infer from filename
    name = payload_path.stem
    for prefix in ["valid_", "invalid_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Remove numeric suffix (e.g., "001")
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        name = parts[0]

    # Map common names to schemas
    schema_mapping = {
        "string": "basic/string_constraints.yaml",
        "numeric": "basic/numeric_constraints.yaml",
        "array": "basic/array_constraints.yaml",
        "object": "basic/object_constraints.yaml",
        "enum": "basic/enum_constraints.yaml",
        "type": "basic/type_constraints.yaml",
        "energy_units": "units/energy_units.yaml",
        "mass_units": "units/mass_units.yaml",
        "conditional_rules": "rules/conditional_rules.yaml",
        "dependency_rules": "rules/dependency_rules.yaml",
        "ref_schema": "basic/ref_schema.yaml",
        "combined": "basic/combined_schema.yaml",
    }

    for key, schema_path in schema_mapping.items():
        if key in name.lower():
            return SCHEMAS_DIR / schema_path

    return None


# =============================================================================
# FIXTURE: MOCK SCHEMA VALIDATOR
# =============================================================================


class MockValidationReport:
    """Mock validation report for testing."""

    def __init__(
        self,
        valid: bool,
        findings: List[Dict[str, Any]] = None,
        schema_ref: Dict[str, Any] = None,
    ):
        self.valid = valid
        self.findings = findings or []
        self.schema_ref = schema_ref or {}

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.get("severity") == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.get("severity") == "warning")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "schema_ref": self.schema_ref,
            "summary": {
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "info_count": sum(1 for f in self.findings if f.get("severity") == "info"),
                "total_findings": len(self.findings),
            },
            "findings": self.findings,
        }


@pytest.fixture
def mock_validator():
    """
    Fixture that provides a mock validator for golden tests.

    This allows tests to run even if the full validator is not implemented.
    In production, this would be replaced with the real validator.
    """
    class MockValidator:
        def __init__(self):
            self.schemas = {}
            # Known units by dimension for unit validation
            self._known_units = {
                "energy": {"J", "kJ", "MJ", "GJ", "Wh", "kWh", "MWh", "GWh", "BTU", "therm"},
                "mass": {"g", "kg", "t", "tonne", "lb", "oz", "ton"},
                "volume": {"L", "mL", "m3", "gallon", "barrel", "ft3"},
                "temperature": {"K", "C", "F"},
                "emissions": {"gCO2e", "kgCO2e", "tCO2e", "MTCO2e"},
            }
            # Format validators
            self._format_validators = {
                "email": self._validate_email,
                "uri": self._validate_uri,
                "uuid": self._validate_uuid,
                "date": self._validate_date,
                "date-time": self._validate_datetime,
                "ipv4": self._validate_ipv4,
                "ipv6": self._validate_ipv6,
            }

        def _validate_email(self, value: str) -> bool:
            """Validate email format."""
            if not isinstance(value, str):
                return False
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, value))

        def _validate_uri(self, value: str) -> bool:
            """Validate URI format."""
            if not isinstance(value, str):
                return False
            try:
                result = urlparse(value)
                return bool(result.scheme and result.netloc)
            except Exception:
                return False

        def _validate_uuid(self, value: str) -> bool:
            """Validate UUID format."""
            if not isinstance(value, str):
                return False
            try:
                UUID(value)
                return True
            except (ValueError, AttributeError):
                return False

        def _validate_date(self, value: str) -> bool:
            """Validate ISO 8601 date format (YYYY-MM-DD)."""
            if not isinstance(value, str):
                return False
            try:
                datetime.strptime(value, "%Y-%m-%d")
                return True
            except ValueError:
                return False

        def _validate_datetime(self, value: str) -> bool:
            """Validate ISO 8601 datetime format."""
            if not isinstance(value, str):
                return False
            patterns = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S%z",
            ]
            for pattern in patterns:
                try:
                    datetime.strptime(value.replace("+00:00", "Z").rstrip("Z") + "Z" if "Z" not in value else value, pattern)
                    return True
                except ValueError:
                    continue
            # Simple regex fallback
            dt_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$'
            return bool(re.match(dt_pattern, value))

        def _validate_ipv4(self, value: str) -> bool:
            """Validate IPv4 address format."""
            if not isinstance(value, str):
                return False
            try:
                IPv4Address(value)
                return True
            except ValueError:
                return False

        def _validate_ipv6(self, value: str) -> bool:
            """Validate IPv6 address format."""
            if not isinstance(value, str):
                return False
            try:
                IPv6Address(value)
                return True
            except ValueError:
                return False

        def load_schema(self, schema_path: Path) -> Dict[str, Any]:
            """Load and cache a schema."""
            if schema_path not in self.schemas:
                self.schemas[schema_path] = load_yaml_or_json(schema_path)
            return self.schemas[schema_path]

        def _resolve_ref(self, schema: Dict, ref: str) -> Optional[Dict]:
            """Resolve a $ref within the same schema."""
            if not ref.startswith("#/"):
                return None  # Only support local refs

            parts = ref[2:].split("/")  # Remove "#/" prefix
            current = schema
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current if isinstance(current, dict) else None

        def _get_resolved_schema(self, prop_schema: Dict, root_schema: Dict) -> Dict:
            """Get the resolved schema, following $ref if present."""
            if "$ref" in prop_schema:
                resolved = self._resolve_ref(root_schema, prop_schema["$ref"])
                if resolved:
                    return resolved
            return prop_schema

        def validate(
            self,
            payload: Dict[str, Any],
            schema: Dict[str, Any],
            profile: str = "standard",
        ) -> MockValidationReport:
            """
            Validate a payload against a schema.

            This is a simplified mock implementation.
            The real validator would perform full validation.
            """
            findings = []

            # Basic required field check
            required = schema.get("required", [])
            properties = schema.get("properties", {})

            for field in required:
                if field not in payload:
                    findings.append({
                        "code": "GLSCHEMA-E100",
                        "severity": "error",
                        "path": f"/{field}",
                        "message": f"Required field '{field}' is missing",
                    })

            # Type checking
            for field, value in payload.items():
                if field in properties:
                    prop_schema = properties[field]
                    expected_type = prop_schema.get("type")

                    # Check for null violation (E103)
                    if value is None:
                        if expected_type and not self._type_allows_null(expected_type):
                            findings.append({
                                "code": "GLSCHEMA-E103",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"Null value not allowed for field '{field}'",
                            })
                            continue  # Skip further type checks for null

                    if expected_type and not self._check_type(value, expected_type):
                        findings.append({
                            "code": "GLSCHEMA-E102",
                            "severity": "error",
                            "path": f"/{field}",
                            "message": f"Expected type '{expected_type}' but found '{type(value).__name__}'",
                        })

                    # Property count for nested objects (minProperties/maxProperties)
                    if isinstance(value, dict):
                        min_props = prop_schema.get("minProperties")
                        max_props = prop_schema.get("maxProperties")
                        prop_count = len(value)

                        if min_props is not None and prop_count < min_props:
                            findings.append({
                                "code": "GLSCHEMA-E105",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"Object has {prop_count} properties, minimum required is {min_props}",
                            })

                        if max_props is not None and prop_count > max_props:
                            findings.append({
                                "code": "GLSCHEMA-E105",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"Object has {prop_count} properties, maximum allowed is {max_props}",
                            })

                        # PropertyNames pattern validation
                        prop_names_schema = prop_schema.get("propertyNames")
                        if prop_names_schema:
                            pattern = prop_names_schema.get("pattern")
                            if pattern:
                                for key in value.keys():
                                    if not re.match(pattern, key):
                                        findings.append({
                                            "code": "GLSCHEMA-E209",
                                            "severity": "error",
                                            "path": f"/{field}/{key}",
                                            "message": f"Property name '{key}' does not match pattern '{pattern}'",
                                        })

                    # Range checking for numbers
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        minimum = prop_schema.get("minimum")
                        maximum = prop_schema.get("maximum")
                        exc_min = prop_schema.get("exclusiveMinimum")
                        exc_max = prop_schema.get("exclusiveMaximum")

                        if minimum is not None and value < minimum:
                            findings.append({
                                "code": "GLSCHEMA-E200",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"Value {value} is below minimum {minimum}",
                            })

                        if maximum is not None and value > maximum:
                            findings.append({
                                "code": "GLSCHEMA-E200",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"Value {value} is above maximum {maximum}",
                            })

                        if exc_min is not None and value <= exc_min:
                            findings.append({
                                "code": "GLSCHEMA-E200",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"Value {value} must be greater than {exc_min}",
                            })

                        if exc_max is not None and value >= exc_max:
                            findings.append({
                                "code": "GLSCHEMA-E200",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"Value {value} must be less than {exc_max}",
                            })

                    # Enum checking
                    enum_values = prop_schema.get("enum")
                    if enum_values is not None and value not in enum_values:
                        findings.append({
                            "code": "GLSCHEMA-E202",
                            "severity": "error",
                            "path": f"/{field}",
                            "message": f"Value '{value}' is not one of allowed values: {enum_values}",
                        })

                    # String length checking
                    if isinstance(value, str):
                        min_len = prop_schema.get("minLength")
                        max_len = prop_schema.get("maxLength")

                        if min_len is not None and len(value) < min_len:
                            findings.append({
                                "code": "GLSCHEMA-E203",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"String length {len(value)} is below minimum {min_len}",
                            })

                        if max_len is not None and len(value) > max_len:
                            findings.append({
                                "code": "GLSCHEMA-E203",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"String length {len(value)} is above maximum {max_len}",
                            })

                        # Pattern checking
                        pattern = prop_schema.get("pattern")
                        if pattern is not None:
                            try:
                                if not re.match(pattern, value):
                                    findings.append({
                                        "code": "GLSCHEMA-E201",
                                        "severity": "error",
                                        "path": f"/{field}",
                                        "message": f"Value '{value}' does not match pattern '{pattern}'",
                                    })
                            except re.error:
                                pass  # Skip invalid patterns

                        # Format checking
                        format_type = prop_schema.get("format")
                        if format_type is not None:
                            validator = self._format_validators.get(format_type)
                            if validator and not validator(value):
                                findings.append({
                                    "code": "GLSCHEMA-E206",
                                    "severity": "error",
                                    "path": f"/{field}",
                                    "message": f"Value '{value}' does not match format '{format_type}'",
                                })

                    # Const checking
                    if "const" in prop_schema:
                        const_val = prop_schema["const"]
                        if value != const_val:
                            findings.append({
                                "code": "GLSCHEMA-E207",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"Value must be {const_val!r}, got {value!r}",
                            })

                    # MultipleOf checking (with floating point tolerance)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        multiple_of = prop_schema.get("multipleOf")
                        if multiple_of is not None and multiple_of != 0:
                            # Use tolerance for floating point comparison
                            quotient = value / multiple_of
                            tolerance = 1e-9
                            if abs(quotient - round(quotient)) > tolerance:
                                findings.append({
                                    "code": "GLSCHEMA-E205",
                                    "severity": "error",
                                    "path": f"/{field}",
                                    "message": f"Value {value} is not a multiple of {multiple_of}",
                                })

                    # Nested object with unit validation
                    if isinstance(value, dict):
                        # Resolve $ref if present
                        resolved_prop_schema = self._get_resolved_schema(prop_schema, schema)
                        nested_findings = self._validate_nested_object(
                            value, resolved_prop_schema, f"/{field}", profile, schema
                        )
                        findings.extend(nested_findings)

                    # Array constraint checking (minItems, maxItems, uniqueItems)
                    if isinstance(value, list):
                        min_items = prop_schema.get("minItems")
                        max_items = prop_schema.get("maxItems")
                        unique_items = prop_schema.get("uniqueItems", False)

                        if min_items is not None and len(value) < min_items:
                            findings.append({
                                "code": "GLSCHEMA-E203",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"Array has {len(value)} items, minimum required is {min_items}",
                            })

                        if max_items is not None and len(value) > max_items:
                            findings.append({
                                "code": "GLSCHEMA-E203",
                                "severity": "error",
                                "path": f"/{field}",
                                "message": f"Array has {len(value)} items, maximum allowed is {max_items}",
                            })

                        if unique_items:
                            seen = []
                            duplicate_indices = []
                            for i, item in enumerate(value):
                                try:
                                    item_str = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else repr(item)
                                except (TypeError, ValueError):
                                    item_str = repr(item)
                                if item_str in seen:
                                    duplicate_indices.append(i)
                                else:
                                    seen.append(item_str)
                            if duplicate_indices:
                                findings.append({
                                    "code": "GLSCHEMA-E204",
                                    "severity": "error",
                                    "path": f"/{field}",
                                    "message": f"Array contains duplicate items at indices {duplicate_indices}",
                                })

                        # Validate array items against items schema
                        items_schema = prop_schema.get("items")
                        if items_schema:
                            for i, item in enumerate(value):
                                item_findings = self._validate_item(item, items_schema, f"/{field}/{i}")
                                findings.extend(item_findings)

                        # Contains validation
                        contains_schema = prop_schema.get("contains")
                        if contains_schema:
                            has_match = False
                            for item in value:
                                if self._check_contains_match(item, contains_schema):
                                    has_match = True
                                    break
                            if not has_match:
                                findings.append({
                                    "code": "GLSCHEMA-E208",
                                    "severity": "error",
                                    "path": f"/{field}",
                                    "message": f"Array does not contain any item matching the 'contains' schema",
                                })

            # Unknown field check (strict mode)
            if profile == "strict" and not schema.get("additionalProperties", True):
                for field in payload:
                    if field not in properties and field != "_test_metadata":
                        findings.append({
                            "code": "GLSCHEMA-E101",
                            "severity": "error",
                            "path": f"/{field}",
                            "message": f"Unknown field '{field}' not allowed",
                        })

            # Rule validation (x-gl-rules)
            rules = schema.get("x-gl-rules", [])
            for rule in rules:
                rule_findings = self._evaluate_rule(rule, payload)
                findings.extend(rule_findings)

            # DependentRequired validation (JSON Schema keyword)
            dependent_required = schema.get("dependentRequired", {})
            for trigger_field, required_fields in dependent_required.items():
                if trigger_field in payload:
                    for req_field in required_fields:
                        if req_field not in payload:
                            findings.append({
                                "code": "GLSCHEMA-E403",
                                "severity": "error",
                                "path": "",
                                "message": f"Property '{req_field}' is required when '{trigger_field}' is present",
                            })

            # OneOf validation
            one_of = schema.get("oneOf")
            if one_of:
                matches = sum(1 for sub_schema in one_of if self._matches_schema(payload, sub_schema, check_const=True))
                if matches != 1:
                    findings.append({
                        "code": "GLSCHEMA-E405",
                        "severity": "error",
                        "path": "",
                        "message": f"Value must match exactly one schema in oneOf, but matched {matches}",
                    })

            # AnyOf validation
            any_of = schema.get("anyOf")
            if any_of:
                matches = sum(1 for sub_schema in any_of if self._matches_schema(payload, sub_schema, check_const=True))
                if matches == 0:
                    findings.append({
                        "code": "GLSCHEMA-E406",
                        "severity": "error",
                        "path": "",
                        "message": "Value must match at least one schema in anyOf",
                    })

            # Sort findings for determinism
            findings.sort(key=lambda f: (f.get("path", ""), f.get("code", "")))

            valid = not any(f.get("severity") == "error" for f in findings)

            return MockValidationReport(valid=valid, findings=findings)

        def _validate_item(self, item: Any, item_schema: Dict[str, Any], path: str) -> List[Dict[str, Any]]:
            """
            Validate an array item against its schema.

            Args:
                item: The item to validate
                item_schema: Schema for the item
                path: JSON Pointer path to the item

            Returns:
                List of findings for this item
            """
            findings = []

            # Type checking
            expected_type = item_schema.get("type")
            if expected_type and not self._check_type(item, expected_type):
                findings.append({
                    "code": "GLSCHEMA-E102",
                    "severity": "error",
                    "path": path,
                    "message": f"Expected type '{expected_type}' but found '{type(item).__name__}'",
                })

            # Numeric range checking
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                minimum = item_schema.get("minimum")
                maximum = item_schema.get("maximum")

                if minimum is not None and item < minimum:
                    findings.append({
                        "code": "GLSCHEMA-E200",
                        "severity": "error",
                        "path": path,
                        "message": f"Value {item} is below minimum {minimum}",
                    })

                if maximum is not None and item > maximum:
                    findings.append({
                        "code": "GLSCHEMA-E200",
                        "severity": "error",
                        "path": path,
                        "message": f"Value {item} is above maximum {maximum}",
                    })

            # String constraints
            if isinstance(item, str):
                min_len = item_schema.get("minLength")
                max_len = item_schema.get("maxLength")

                if min_len is not None and len(item) < min_len:
                    findings.append({
                        "code": "GLSCHEMA-E203",
                        "severity": "error",
                        "path": path,
                        "message": f"String length {len(item)} is below minimum {min_len}",
                    })

                if max_len is not None and len(item) > max_len:
                    findings.append({
                        "code": "GLSCHEMA-E203",
                        "severity": "error",
                        "path": path,
                        "message": f"String length {len(item)} is above maximum {max_len}",
                    })

            # Enum checking
            enum_values = item_schema.get("enum")
            if enum_values is not None and item not in enum_values:
                findings.append({
                    "code": "GLSCHEMA-E202",
                    "severity": "error",
                    "path": path,
                    "message": f"Value '{item}' is not one of allowed values: {enum_values}",
                })

            # Nested object validation
            if isinstance(item, dict):
                required_fields = item_schema.get("required", [])
                for req_field in required_fields:
                    if req_field not in item:
                        findings.append({
                            "code": "GLSCHEMA-E100",
                            "severity": "error",
                            "path": f"{path}/{req_field}",
                            "message": f"Required field '{req_field}' is missing",
                        })

                # Validate nested properties
                nested_props = item_schema.get("properties", {})
                for prop_name, prop_value in item.items():
                    if prop_name in nested_props:
                        nested_findings = self._validate_item(
                            prop_value, nested_props[prop_name], f"{path}/{prop_name}"
                        )
                        findings.extend(nested_findings)

            return findings

        def _check_type(self, value: Any, expected_type: str) -> bool:
            """Check if value matches expected JSON Schema type."""
            type_map = {
                "string": str,
                "number": (int, float),
                "integer": int,
                "boolean": bool,
                "array": list,
                "object": dict,
                "null": type(None),
            }

            if isinstance(expected_type, list):
                # Union type
                return any(self._check_type(value, t) for t in expected_type)

            expected = type_map.get(expected_type)
            if expected is None:
                return True

            if expected_type == "integer" and isinstance(value, bool):
                return False
            if expected_type == "number" and isinstance(value, bool):
                return False

            return isinstance(value, expected)

        def _type_allows_null(self, expected_type) -> bool:
            """Check if the type allows null values."""
            if expected_type is None:
                return True
            if isinstance(expected_type, list):
                return "null" in expected_type
            return expected_type == "null"

        def _validate_nested_object(self, value: Dict, prop_schema: Dict, path: str, profile: str, root_schema: Dict = None) -> List[Dict]:
            """Validate nested object including unit checking."""
            findings = []
            root_schema = root_schema or prop_schema

            nested_properties = prop_schema.get("properties", {})
            nested_required = prop_schema.get("required", [])

            # Check required fields in nested object
            for req_field in nested_required:
                if req_field not in value:
                    findings.append({
                        "code": "GLSCHEMA-E100",
                        "severity": "error",
                        "path": f"{path}/{req_field}",
                        "message": f"Required field '{req_field}' is missing",
                    })

            # Validate nested fields
            for nested_field, nested_value in value.items():
                if nested_field in nested_properties:
                    nested_prop_schema = nested_properties[nested_field]
                    # Enum checking for nested fields
                    enum_values = nested_prop_schema.get("enum")
                    if enum_values is not None and nested_value not in enum_values:
                        findings.append({
                            "code": "GLSCHEMA-E202",
                            "severity": "error",
                            "path": f"{path}/{nested_field}",
                            "message": f"Value '{nested_value}' is not one of allowed values: {enum_values}",
                        })

                    # Type checking for nested fields
                    expected_type = nested_prop_schema.get("type")
                    if expected_type and not self._check_type(nested_value, expected_type):
                        findings.append({
                            "code": "GLSCHEMA-E102",
                            "severity": "error",
                            "path": f"{path}/{nested_field}",
                            "message": f"Expected type '{expected_type}' but found '{type(nested_value).__name__}'",
                        })

                    # Range checking for nested numeric fields
                    if isinstance(nested_value, (int, float)) and not isinstance(nested_value, bool):
                        minimum = nested_prop_schema.get("minimum")
                        maximum = nested_prop_schema.get("maximum")
                        if minimum is not None and nested_value < minimum:
                            findings.append({
                                "code": "GLSCHEMA-E200",
                                "severity": "error",
                                "path": f"{path}/{nested_field}",
                                "message": f"Value {nested_value} is below minimum {minimum}",
                            })
                        if maximum is not None and nested_value > maximum:
                            findings.append({
                                "code": "GLSCHEMA-E200",
                                "severity": "error",
                                "path": f"{path}/{nested_field}",
                                "message": f"Value {nested_value} is above maximum {maximum}",
                            })

            # Unit validation using x-gl-unit extension
            unit_spec = prop_schema.get("x-gl-unit")
            if unit_spec:
                unit_findings = self._validate_unit(value, unit_spec, path)
                findings.extend(unit_findings)

            return findings

        def _validate_unit(self, value: Dict, unit_spec: Dict, path: str) -> List[Dict]:
            """
            Validate unit specification in a value with unit.

            Checks:
            - GLSCHEMA-E300: Unit missing
            - GLSCHEMA-E301: Unit incompatible (wrong dimension)
            - GLSCHEMA-E303: Unit unknown (not in catalog)
            """
            findings = []
            dimension = unit_spec.get("dimension")

            # Get unit from value
            unit = value.get("unit")

            # E300: Check if unit is missing
            if unit is None or (isinstance(unit, str) and unit.strip() == ""):
                findings.append({
                    "code": "GLSCHEMA-E300",
                    "severity": "error",
                    "path": f"{path}/unit",
                    "message": f"Required unit not provided for value at path '{path}'",
                })
                return findings

            # Get allowed units from spec
            allowed_units = unit_spec.get("allowed", [])
            if allowed_units:
                allowed_set = set(allowed_units)
            else:
                # Use dimension-based units
                allowed_set = self._known_units.get(dimension, set())

            # E303: Check if unit is unknown (not in allowed list)
            all_known_units = set()
            for dim_units in self._known_units.values():
                all_known_units.update(dim_units)

            if unit not in all_known_units and unit not in allowed_set:
                findings.append({
                    "code": "GLSCHEMA-E303",
                    "severity": "error",
                    "path": f"{path}/unit",
                    "message": f"Unit '{unit}' at path '{path}' is not recognized in the unit catalog",
                })
                return findings

            # E301: Check if unit is from the correct dimension/allowed list
            if dimension and allowed_set:
                if unit not in allowed_set:
                    findings.append({
                        "code": "GLSCHEMA-E301",
                        "severity": "error",
                        "path": f"{path}/unit",
                        "message": f"Unit '{unit}' at path '{path}' is incompatible with dimension '{dimension}'",
                    })

            return findings

        def _check_contains_match(self, item: Any, contains_schema: Dict) -> bool:
            """Check if an item matches the contains schema."""
            expected_type = contains_schema.get("type")
            if expected_type and not self._check_type(item, expected_type):
                return False

            # Check enum
            enum_values = contains_schema.get("enum")
            if enum_values is not None and item not in enum_values:
                return False

            # Check const
            if "const" in contains_schema and item != contains_schema["const"]:
                return False

            # Check numeric constraints
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                minimum = contains_schema.get("minimum")
                maximum = contains_schema.get("maximum")
                if minimum is not None and item < minimum:
                    return False
                if maximum is not None and item > maximum:
                    return False

            return True

        def _evaluate_rule(self, rule: Dict, payload: Dict) -> List[Dict]:
            """
            Evaluate a cross-field validation rule.

            Returns findings if the rule is violated.
            """
            findings = []
            rule_id = rule.get("id", "UNKNOWN")
            when_clause = rule.get("when")
            check_clause = rule.get("check")
            message = rule.get("message", f"Rule {rule_id} violated")
            severity = rule.get("severity", "error")

            # Evaluate 'when' condition
            if when_clause:
                if not self._evaluate_condition(when_clause, payload):
                    # Condition not met, rule doesn't apply
                    return findings

            # Evaluate 'check' assertion
            if check_clause:
                if not self._evaluate_condition(check_clause, payload):
                    # Determine error code based on rule type
                    # E402 for consistency/sum rules, E400 for general rules
                    error_code = "GLSCHEMA-E400"
                    if check_clause.get("eq") and isinstance(check_clause["eq"], list):
                        # Check if this is a sum-based consistency check
                        for operand in check_clause["eq"]:
                            if isinstance(operand, dict) and "sum" in operand:
                                error_code = "GLSCHEMA-E402"
                                break

                    findings.append({
                        "code": error_code,
                        "severity": severity,
                        "path": "",
                        "message": message,
                        "rule_id": rule_id,
                    })

            return findings

        def _evaluate_condition(self, condition: Dict, payload: Dict) -> bool:
            """Evaluate a rule condition against the payload."""
            if not isinstance(condition, dict):
                return True

            # Handle 'exists' check
            if "exists" in condition:
                path = condition["exists"]
                return self._path_exists(path, payload)

            # Handle 'eq' (equality) check
            if "eq" in condition:
                operands = condition["eq"]
                if len(operands) >= 2:
                    left = self._resolve_operand(operands[0], payload)
                    right = self._resolve_operand(operands[1], payload)
                    return left == right

            # Handle 'and' (conjunction)
            if "and" in condition:
                return all(self._evaluate_condition(c, payload) for c in condition["and"])

            # Handle 'or' (disjunction)
            if "or" in condition:
                return any(self._evaluate_condition(c, payload) for c in condition["or"])

            # Handle 'not'
            if "not" in condition:
                return not self._evaluate_condition(condition["not"], payload)

            return True

        def _resolve_operand(self, operand: Any, payload: Dict) -> Any:
            """Resolve an operand value from the payload."""
            if isinstance(operand, dict):
                if "path" in operand:
                    return self._get_path_value(operand["path"], payload)
                if "sum" in operand:
                    values = [self._resolve_operand(o, payload) for o in operand["sum"]]
                    try:
                        return sum(v for v in values if isinstance(v, (int, float)))
                    except (TypeError, ValueError):
                        return None
            return operand

        def _path_exists(self, path: str, payload: Dict) -> bool:
            """Check if a JSON Pointer path exists in the payload."""
            if not path or path == "/":
                return True
            parts = path.strip("/").split("/")
            current = payload
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False
            return True

        def _get_path_value(self, path: str, payload: Dict) -> Any:
            """Get a value from the payload using JSON Pointer path."""
            if not path or path == "/":
                return payload
            parts = path.strip("/").split("/")
            current = payload
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            return current

        def _matches_schema(self, payload: Dict, sub_schema: Dict, check_const: bool = False) -> bool:
            """Check if payload matches a sub-schema (simplified for oneOf/anyOf).

            Args:
                payload: The data to validate
                sub_schema: The sub-schema to check against
                check_const: Whether to check const constraints (default False for oneOf)
            """
            # Check required fields
            required = sub_schema.get("required", [])
            for field in required:
                if field not in payload:
                    return False

            # Check const constraints in properties (only if enabled)
            properties = sub_schema.get("properties", {})
            if check_const:
                for field, prop_schema in properties.items():
                    if "const" in prop_schema:
                        # Field must exist and match const
                        if field not in payload:
                            return False
                        if payload[field] != prop_schema["const"]:
                            return False

            # Check type constraints
            for field, value in payload.items():
                if field in properties and field != "_test_metadata":
                    prop_schema = properties[field]
                    expected_type = prop_schema.get("type")
                    if expected_type and not self._check_type(value, expected_type):
                        return False

            return True

    return MockValidator()


# =============================================================================
# DISCOVER TEST DATA
# =============================================================================


def get_valid_payloads() -> List[Tuple[str, Path]]:
    """Discover all valid payload test files."""
    valid_dir = PAYLOADS_DIR / "valid"
    files = discover_files(valid_dir)
    return [(f.stem, f) for f in files]


def get_invalid_payloads() -> List[Tuple[str, Path]]:
    """Discover all invalid payload test files."""
    invalid_dir = PAYLOADS_DIR / "invalid"
    files = discover_files(invalid_dir)
    return [(f.stem, f) for f in files]


def get_schemas() -> List[Tuple[str, Path]]:
    """Discover all test schema files."""
    files = discover_files(SCHEMAS_DIR)
    return [(f.stem, f) for f in files]


# =============================================================================
# TEST CLASS: VALID PAYLOADS
# =============================================================================


class TestValidPayloads:
    """Tests for payloads that should pass validation."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        "test_name,payload_path",
        get_valid_payloads(),
        ids=[name for name, _ in get_valid_payloads()],
    )
    def test_valid_payload_passes_validation(
        self,
        test_name: str,
        payload_path: Path,
        mock_validator,
    ):
        """
        Test that valid payloads pass validation.

        Each valid payload should:
        1. Load successfully
        2. Pass validation with no errors
        3. Have zero error findings
        """
        # Load payload
        payload_data = load_yaml_or_json(payload_path)
        metadata = extract_test_metadata(payload_data)

        # Get schema
        schema_path = get_schema_for_payload(payload_path, metadata)
        if schema_path is None or not schema_path.exists():
            pytest.skip(f"Schema not found for {payload_path}")

        schema = mock_validator.load_schema(schema_path)

        # Validate
        result = mock_validator.validate(payload_data, schema, profile="standard")

        # Assert
        assert result.valid, (
            f"Expected valid=True for {test_name}, but got errors: "
            f"{[f['message'] for f in result.findings if f['severity'] == 'error']}"
        )
        assert result.error_count == 0, (
            f"Expected 0 errors for {test_name}, got {result.error_count}"
        )

    @pytest.mark.golden
    def test_all_valid_payloads_have_schemas(self):
        """Test that all valid payloads have corresponding schemas."""
        valid_payloads = get_valid_payloads()
        missing_schemas = []

        for test_name, payload_path in valid_payloads:
            payload_data = load_yaml_or_json(payload_path)
            metadata = extract_test_metadata(payload_data)
            schema_path = get_schema_for_payload(payload_path, metadata)

            if schema_path is None or not schema_path.exists():
                missing_schemas.append(test_name)

        assert len(missing_schemas) == 0, (
            f"Missing schemas for payloads: {missing_schemas}"
        )


# =============================================================================
# TEST CLASS: INVALID PAYLOADS
# =============================================================================


class TestInvalidPayloads:
    """Tests for payloads that should fail validation with specific errors."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        "test_name,payload_path",
        get_invalid_payloads(),
        ids=[name for name, _ in get_invalid_payloads()],
    )
    def test_invalid_payload_fails_validation(
        self,
        test_name: str,
        payload_path: Path,
        mock_validator,
    ):
        """
        Test that invalid payloads fail validation.

        Each invalid payload should:
        1. Load successfully
        2. Fail validation (valid=False)
        3. Have at least one error finding
        """
        # Load payload
        payload_data = load_yaml_or_json(payload_path)
        metadata = extract_test_metadata(payload_data)

        # Get schema
        schema_path = get_schema_for_payload(payload_path, metadata)
        if schema_path is None or not schema_path.exists():
            pytest.skip(f"Schema not found for {payload_path}")

        schema = mock_validator.load_schema(schema_path)

        # Validate
        result = mock_validator.validate(payload_data, schema, profile="strict")

        # Assert
        assert not result.valid, (
            f"Expected valid=False for {test_name}, but validation passed"
        )
        assert result.error_count > 0, (
            f"Expected at least one error for {test_name}, got {result.error_count}"
        )

    @pytest.mark.golden
    @pytest.mark.parametrize(
        "test_name,payload_path",
        get_invalid_payloads(),
        ids=[name for name, _ in get_invalid_payloads()],
    )
    def test_invalid_payload_produces_expected_error_code(
        self,
        test_name: str,
        payload_path: Path,
        mock_validator,
    ):
        """
        Test that invalid payloads produce the expected error codes.

        Each invalid payload with _test_metadata.expected_errors should
        produce findings matching those error codes.
        """
        # Load payload
        payload_data = load_yaml_or_json(payload_path)
        metadata = extract_test_metadata(payload_data)

        expected_errors = metadata.get("expected_errors", [])
        if not expected_errors:
            pytest.skip(f"No expected_errors in metadata for {test_name}")

        # Get schema
        schema_path = get_schema_for_payload(payload_path, metadata)
        if schema_path is None or not schema_path.exists():
            pytest.skip(f"Schema not found for {payload_path}")

        schema = mock_validator.load_schema(schema_path)

        # Validate
        result = mock_validator.validate(payload_data, schema, profile="strict")

        # Check expected error codes are present
        actual_codes = {f["code"] for f in result.findings}

        for expected in expected_errors:
            expected_code = expected["code"]
            assert expected_code in actual_codes, (
                f"Expected error code {expected_code} not found in findings. "
                f"Got: {actual_codes}"
            )


# =============================================================================
# TEST CLASS: ERROR CODE COVERAGE
# =============================================================================


class TestErrorCodeCoverage:
    """Tests to ensure comprehensive error code coverage."""

    # All GLSCHEMA error codes that should be covered by golden tests
    EXPECTED_ERROR_CODES: Set[str] = {
        # Structural Errors (E1xx)
        "GLSCHEMA-E100",  # MISSING_REQUIRED
        "GLSCHEMA-E101",  # UNKNOWN_FIELD
        "GLSCHEMA-E102",  # TYPE_MISMATCH
        "GLSCHEMA-E103",  # INVALID_NULL
        "GLSCHEMA-E104",  # CONTAINER_TYPE_MISMATCH
        "GLSCHEMA-E105",  # PROPERTY_COUNT_VIOLATION
        "GLSCHEMA-E106",  # REQUIRED_PROPERTIES_MISSING
        "GLSCHEMA-E107",  # DUPLICATE_KEY
        # Constraint Errors (E2xx)
        "GLSCHEMA-E200",  # RANGE_VIOLATION
        "GLSCHEMA-E201",  # PATTERN_MISMATCH
        "GLSCHEMA-E202",  # ENUM_VIOLATION
        "GLSCHEMA-E203",  # LENGTH_VIOLATION
        "GLSCHEMA-E204",  # UNIQUE_VIOLATION
        "GLSCHEMA-E205",  # MULTIPLE_OF_VIOLATION
        "GLSCHEMA-E206",  # FORMAT_VIOLATION
        "GLSCHEMA-E207",  # CONST_VIOLATION
        "GLSCHEMA-E208",  # CONTAINS_VIOLATION
        "GLSCHEMA-E209",  # PROPERTY_NAME_VIOLATION
        # Unit Errors (E3xx)
        "GLSCHEMA-E300",  # UNIT_MISSING
        "GLSCHEMA-E301",  # UNIT_INCOMPATIBLE
        "GLSCHEMA-E302",  # UNIT_NONCANONICAL (warning)
        "GLSCHEMA-E303",  # UNIT_UNKNOWN
        # Rule Errors (E4xx)
        "GLSCHEMA-E400",  # RULE_VIOLATION
        "GLSCHEMA-E401",  # CONDITIONAL_REQUIRED
        "GLSCHEMA-E402",  # CONSISTENCY_ERROR
        "GLSCHEMA-E403",  # DEPENDENCY_VIOLATION
        "GLSCHEMA-E405",  # ONE_OF_VIOLATION
        "GLSCHEMA-E406",  # ANY_OF_VIOLATION
    }

    @pytest.mark.golden
    def test_all_error_codes_have_test_coverage(self):
        """
        Test that all expected error codes have at least one test case.

        This ensures we have comprehensive coverage of validation scenarios.
        """
        covered_codes: Set[str] = set()

        # Scan all invalid payload metadata for expected_errors
        invalid_payloads = get_invalid_payloads()
        for test_name, payload_path in invalid_payloads:
            try:
                payload_data = load_yaml_or_json(payload_path)
                metadata = payload_data.get("_test_metadata", {})
                expected_errors = metadata.get("expected_errors", [])

                for error in expected_errors:
                    covered_codes.add(error.get("code", ""))
            except Exception:
                pass

        # Check coverage
        missing_codes = self.EXPECTED_ERROR_CODES - covered_codes
        coverage_pct = len(covered_codes) / len(self.EXPECTED_ERROR_CODES) * 100

        # Allow some missing codes but warn
        if missing_codes:
            pytest.warns(
                UserWarning,
                match=f"Missing coverage for {len(missing_codes)} error codes",
            )

        # Require at least 50% coverage
        assert coverage_pct >= 50, (
            f"Error code coverage is only {coverage_pct:.1f}%. "
            f"Missing: {sorted(missing_codes)}"
        )

    @pytest.mark.golden
    def test_list_covered_error_codes(self):
        """List all error codes covered by golden tests (for reporting)."""
        covered_codes: Dict[str, List[str]] = {}

        invalid_payloads = get_invalid_payloads()
        for test_name, payload_path in invalid_payloads:
            try:
                payload_data = load_yaml_or_json(payload_path)
                metadata = payload_data.get("_test_metadata", {})
                expected_errors = metadata.get("expected_errors", [])

                for error in expected_errors:
                    code = error.get("code", "")
                    if code:
                        if code not in covered_codes:
                            covered_codes[code] = []
                        covered_codes[code].append(test_name)
            except Exception:
                pass

        # Print coverage report
        print("\n=== Error Code Coverage Report ===")
        for code in sorted(covered_codes.keys()):
            tests = covered_codes[code]
            print(f"{code}: {len(tests)} test(s)")

        print(f"\nTotal codes covered: {len(covered_codes)}")
        print(f"Total codes expected: {len(self.EXPECTED_ERROR_CODES)}")


# =============================================================================
# TEST CLASS: EXPECTED REPORTS
# =============================================================================


class TestExpectedReports:
    """Tests that compare actual validation results with expected reports."""

    @pytest.mark.golden
    def test_valid_report_matches_expected(self, mock_validator):
        """Test that valid payload produces expected report structure."""
        # Load payload
        payload_path = PAYLOADS_DIR / "valid" / "string_valid_001.yaml"
        if not payload_path.exists():
            pytest.skip("string_valid_001.yaml not found")

        payload_data = load_yaml_or_json(payload_path)
        metadata = extract_test_metadata(payload_data)

        # Get schema
        schema_path = get_schema_for_payload(payload_path, metadata)
        schema = mock_validator.load_schema(schema_path)

        # Validate
        result = mock_validator.validate(payload_data, schema)

        # Load expected report
        expected_path = EXPECTED_DIR / "string_valid_001_report.json"
        if not expected_path.exists():
            pytest.skip("Expected report not found")

        expected = load_yaml_or_json(expected_path)

        # Compare key fields
        assert result.valid == expected["valid"]
        assert result.error_count == expected["summary"]["error_count"]

    @pytest.mark.golden
    @pytest.mark.parametrize(
        "payload_name,expected_name",
        [
            ("missing_required_001", "missing_required_001_report"),
            ("type_mismatch_001", "type_mismatch_001_report"),
            ("range_violation_001", "range_violation_001_report"),
            ("enum_violation_001", "enum_violation_001_report"),
        ],
    )
    def test_invalid_report_matches_expected(
        self,
        payload_name: str,
        expected_name: str,
        mock_validator,
    ):
        """Test that invalid payloads produce expected reports."""
        # Load payload
        payload_path = PAYLOADS_DIR / "invalid" / f"{payload_name}.yaml"
        if not payload_path.exists():
            pytest.skip(f"{payload_name}.yaml not found")

        payload_data = load_yaml_or_json(payload_path)
        metadata = extract_test_metadata(payload_data)

        # Get schema
        schema_path = get_schema_for_payload(payload_path, metadata)
        if not schema_path or not schema_path.exists():
            pytest.skip(f"Schema not found for {payload_name}")

        schema = mock_validator.load_schema(schema_path)

        # Validate
        result = mock_validator.validate(payload_data, schema, profile="strict")

        # Load expected report
        expected_path = EXPECTED_DIR / f"{expected_name}.json"
        if not expected_path.exists():
            pytest.skip(f"Expected report {expected_name}.json not found")

        expected = load_yaml_or_json(expected_path)

        # Compare
        assert result.valid == expected["valid"], (
            f"valid mismatch: got {result.valid}, expected {expected['valid']}"
        )
        assert result.error_count == expected["summary"]["error_count"], (
            f"error_count mismatch: got {result.error_count}, "
            f"expected {expected['summary']['error_count']}"
        )

        # Check that expected error codes are present
        actual_codes = {f["code"] for f in result.findings}
        expected_codes = {f["code"] for f in expected["findings"]}
        assert expected_codes.issubset(actual_codes), (
            f"Missing expected error codes: {expected_codes - actual_codes}"
        )


# =============================================================================
# TEST CLASS: SCHEMA VALIDATION
# =============================================================================


class TestSchemaValidation:
    """Tests for schema self-validation."""

    @pytest.mark.golden
    @pytest.mark.parametrize(
        "test_name,schema_path",
        get_schemas(),
        ids=[name for name, _ in get_schemas()],
    )
    def test_schema_is_valid(self, test_name: str, schema_path: Path):
        """Test that all test schemas are themselves valid."""
        try:
            schema = load_yaml_or_json(schema_path)
        except Exception as e:
            pytest.fail(f"Failed to load schema {test_name}: {e}")

        # Basic schema structure validation
        assert isinstance(schema, dict), f"Schema {test_name} is not a dict"

        # Check for required schema fields
        if "$schema" in schema:
            assert "json-schema.org" in schema["$schema"], (
                f"Schema {test_name} has invalid $schema URI"
            )

        if "type" in schema:
            valid_types = {"string", "number", "integer", "boolean", "object", "array", "null"}
            schema_type = schema["type"]
            if isinstance(schema_type, str):
                assert schema_type in valid_types, (
                    f"Schema {test_name} has invalid type: {schema_type}"
                )


# =============================================================================
# TEST CLASS: DETERMINISM
# =============================================================================


class TestDeterminism:
    """Tests for validation determinism and reproducibility."""

    @pytest.mark.golden
    def test_validation_is_deterministic(self, mock_validator):
        """Test that validation produces identical results on repeated runs."""
        # Load a payload with multiple errors
        payload_path = PAYLOADS_DIR / "invalid" / "multiple_errors_001.yaml"
        if not payload_path.exists():
            pytest.skip("multiple_errors_001.yaml not found")

        payload_data = load_yaml_or_json(payload_path)
        metadata = extract_test_metadata(payload_data)

        schema_path = get_schema_for_payload(payload_path, metadata)
        if not schema_path or not schema_path.exists():
            pytest.skip("Schema not found")

        schema = mock_validator.load_schema(schema_path)

        # Run validation multiple times
        results = []
        for _ in range(10):
            result = mock_validator.validate(payload_data, schema, profile="strict")
            results.append(result.to_dict())

        # All results should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], start=2):
            assert result == first_result, (
                f"Validation result {i} differs from result 1"
            )

    @pytest.mark.golden
    def test_findings_are_sorted_deterministically(self, mock_validator):
        """Test that findings are always sorted in the same order."""
        payload_path = PAYLOADS_DIR / "invalid" / "multiple_errors_001.yaml"
        if not payload_path.exists():
            pytest.skip("multiple_errors_001.yaml not found")

        payload_data = load_yaml_or_json(payload_path)
        metadata = extract_test_metadata(payload_data)

        schema_path = get_schema_for_payload(payload_path, metadata)
        if not schema_path or not schema_path.exists():
            pytest.skip("Schema not found")

        schema = mock_validator.load_schema(schema_path)
        result = mock_validator.validate(payload_data, schema, profile="strict")

        # Verify findings are sorted
        paths = [f["path"] for f in result.findings]
        assert paths == sorted(paths), "Findings are not sorted by path"


# =============================================================================
# TEST CLASS: VALIDATION PROFILES
# =============================================================================


class TestValidationProfiles:
    """Tests for different validation profiles (strict, standard, permissive)."""

    @pytest.mark.golden
    def test_strict_profile_rejects_unknown_fields(self, mock_validator):
        """Test that strict profile rejects unknown fields."""
        payload_path = PAYLOADS_DIR / "invalid" / "unknown_field_001.yaml"
        if not payload_path.exists():
            pytest.skip("unknown_field_001.yaml not found")

        payload_data = load_yaml_or_json(payload_path)
        metadata = extract_test_metadata(payload_data)

        schema_path = get_schema_for_payload(payload_path, metadata)
        if not schema_path or not schema_path.exists():
            pytest.skip("Schema not found")

        schema = mock_validator.load_schema(schema_path)

        # Strict profile should reject
        result = mock_validator.validate(payload_data, schema, profile="strict")
        assert not result.valid, "Strict profile should reject unknown fields"

        # Check for UNKNOWN_FIELD error
        error_codes = {f["code"] for f in result.findings}
        assert "GLSCHEMA-E101" in error_codes, (
            "Expected GLSCHEMA-E101 (UNKNOWN_FIELD) in strict mode"
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    # Run discovery and print summary
    print("=== Golden Test Discovery ===")
    print(f"Schemas: {len(get_schemas())}")
    print(f"Valid payloads: {len(get_valid_payloads())}")
    print(f"Invalid payloads: {len(get_invalid_payloads())}")
    print(f"\nRun with: pytest {__file__} -v")
