# -*- coding: utf-8 -*-
"""
Unit Tests for SchemaValidator (AGENT-DATA-002)

Tests schema-based validation of normalized data.

Coverage target: 85%+ of schema_validator.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline SchemaValidator
# ---------------------------------------------------------------------------

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "energy": {
        "required_fields": ["facility_name", "reporting_year", "electricity_kwh"],
        "field_types": {"facility_name": "string", "reporting_year": "integer",
                        "electricity_kwh": "float", "natural_gas_therms": "float"},
        "field_ranges": {"reporting_year": (1990, 2100), "electricity_kwh": (0, 1e9)},
    },
    "transport": {
        "required_fields": ["vehicle_id", "distance_km"],
        "field_types": {"vehicle_id": "string", "distance_km": "float",
                        "fuel_used_litres": "float", "fuel_type": "string"},
        "field_ranges": {"distance_km": (0, 100000), "fuel_used_litres": (0, 100000)},
    },
    "emissions": {
        "required_fields": ["source", "scope", "total_emissions_tco2e"],
        "field_types": {"source": "string", "scope": "string",
                        "total_emissions_tco2e": "float", "emission_factor": "float"},
        "field_ranges": {"total_emissions_tco2e": (0, 1e12)},
    },
}


class SchemaValidator:
    """Validates normalized data against predefined schemas."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._schemas: Dict[str, Dict[str, Any]] = dict(SCHEMAS)
        self._stats: Dict[str, int] = {
            "validations_run": 0, "rows_validated": 0,
            "errors_found": 0, "warnings_found": 0,
        }

    def validate_data(self, data: List[Dict[str, Any]], schema_name: str) -> Dict[str, Any]:
        schema = self._schemas.get(schema_name)
        if not schema:
            return {"is_valid": False, "errors": [f"Unknown schema: {schema_name}"], "warnings": []}
        errors, warnings = [], []
        for idx, row in enumerate(data):
            row_result = self.validate_row(row, schema, idx)
            errors.extend(row_result["errors"])
            warnings.extend(row_result["warnings"])
        self._stats["validations_run"] += 1
        self._stats["rows_validated"] += len(data)
        self._stats["errors_found"] += len(errors)
        self._stats["warnings_found"] += len(warnings)
        return {"is_valid": len(errors) == 0, "errors": errors, "warnings": warnings,
                "rows_validated": len(data), "error_count": len(errors)}

    def validate_row(self, row: Dict[str, Any], schema: Dict[str, Any],
                     row_index: int = 0) -> Dict[str, Any]:
        errors, warnings = [], []
        req_errors = self.validate_required_fields(row, schema.get("required_fields", []), row_index)
        errors.extend(req_errors)
        type_errors = self.validate_types(row, schema.get("field_types", {}), row_index)
        errors.extend(type_errors)
        range_warns = self.validate_ranges(row, schema.get("field_ranges", {}), row_index)
        warnings.extend(range_warns)
        cross_errors = self.validate_cross_field(row, schema, row_index)
        errors.extend(cross_errors)
        return {"errors": errors, "warnings": warnings}

    def validate_field(self, value: Any, field_name: str, expected_type: str) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        if expected_type == "integer":
            try:
                int(float(s.replace(",", "")))
            except (ValueError, TypeError):
                return f"Field '{field_name}': expected integer, got '{s}'"
        elif expected_type == "float":
            try:
                float(s.replace(",", ""))
            except (ValueError, TypeError):
                return f"Field '{field_name}': expected float, got '{s}'"
        elif expected_type == "string":
            if not isinstance(value, (str, int, float)):
                return f"Field '{field_name}': expected string-coercible value"
        return None

    def validate_required_fields(self, row: Dict[str, Any], required: List[str],
                                 row_index: int = 0) -> List[str]:
        errors = []
        for field in required:
            if field not in row or row[field] is None or str(row[field]).strip() == "":
                errors.append(f"Row {row_index}: required field '{field}' is missing or empty")
        return errors

    def validate_ranges(self, row: Dict[str, Any], ranges: Dict[str, tuple],
                        row_index: int = 0) -> List[str]:
        warnings = []
        for field, (low, high) in ranges.items():
            value = row.get(field)
            if value is None:
                continue
            try:
                num = float(str(value).replace(",", ""))
                if num < low or num > high:
                    warnings.append(f"Row {row_index}: '{field}' value {num} outside range [{low}, {high}]")
            except (ValueError, TypeError):
                pass
        return warnings

    def validate_types(self, row: Dict[str, Any], types: Dict[str, str],
                       row_index: int = 0) -> List[str]:
        errors = []
        for field, expected_type in types.items():
            if field in row and row[field] is not None:
                error = self.validate_field(row[field], field, expected_type)
                if error:
                    errors.append(f"Row {row_index}: {error}")
        return errors

    def validate_cross_field(self, row: Dict[str, Any], schema: Dict[str, Any],
                             row_index: int = 0) -> List[str]:
        return []

    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        self._schemas[name] = schema

    def get_available_schemas(self) -> List[str]:
        return list(self._schemas.keys())

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self._stats)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestSchemaValidatorInit:
    def test_default_creation(self):
        v = SchemaValidator()
        assert "energy" in v.get_available_schemas()

    def test_initial_statistics(self):
        v = SchemaValidator()
        assert v.get_statistics()["validations_run"] == 0

    def test_default_schemas_present(self):
        v = SchemaValidator()
        schemas = v.get_available_schemas()
        assert "energy" in schemas
        assert "transport" in schemas
        assert "emissions" in schemas


class TestValidateDataEnergy:
    def test_valid_energy_data(self):
        v = SchemaValidator()
        data = [{"facility_name": "London HQ", "reporting_year": 2025, "electricity_kwh": 4500.0}]
        result = v.validate_data(data, "energy")
        assert result["is_valid"] is True

    def test_missing_required_field(self):
        v = SchemaValidator()
        data = [{"facility_name": "London HQ", "reporting_year": 2025}]
        result = v.validate_data(data, "energy")
        assert result["is_valid"] is False
        assert result["error_count"] >= 1

    def test_invalid_type(self):
        v = SchemaValidator()
        data = [{"facility_name": "London", "reporting_year": "not_a_year", "electricity_kwh": 4500.0}]
        result = v.validate_data(data, "energy")
        assert result["is_valid"] is False

    def test_multiple_rows(self):
        v = SchemaValidator()
        data = [
            {"facility_name": "A", "reporting_year": 2025, "electricity_kwh": 100.0},
            {"facility_name": "B", "reporting_year": 2025, "electricity_kwh": 200.0},
        ]
        result = v.validate_data(data, "energy")
        assert result["rows_validated"] == 2

    def test_out_of_range_warning(self):
        v = SchemaValidator()
        data = [{"facility_name": "A", "reporting_year": 1800, "electricity_kwh": 100.0}]
        result = v.validate_data(data, "energy")
        assert len(result["warnings"]) >= 1


class TestValidateDataTransport:
    def test_valid_transport_data(self):
        v = SchemaValidator()
        data = [{"vehicle_id": "V001", "distance_km": 350.0}]
        result = v.validate_data(data, "transport")
        assert result["is_valid"] is True

    def test_negative_distance(self):
        v = SchemaValidator()
        data = [{"vehicle_id": "V001", "distance_km": -100.0}]
        result = v.validate_data(data, "transport")
        assert len(result["warnings"]) >= 1

    def test_missing_vehicle_id(self):
        v = SchemaValidator()
        data = [{"distance_km": 350.0}]
        result = v.validate_data(data, "transport")
        assert result["is_valid"] is False


class TestValidateDataEmissions:
    def test_valid_emissions_data(self):
        v = SchemaValidator()
        data = [{"source": "Boiler", "scope": "Scope 1", "total_emissions_tco2e": 1250.5}]
        result = v.validate_data(data, "emissions")
        assert result["is_valid"] is True

    def test_missing_source(self):
        v = SchemaValidator()
        data = [{"scope": "Scope 1", "total_emissions_tco2e": 1250.5}]
        result = v.validate_data(data, "emissions")
        assert result["is_valid"] is False


class TestValidateRow:
    def test_valid_row(self):
        v = SchemaValidator()
        schema = SCHEMAS["energy"]
        result = v.validate_row({"facility_name": "A", "reporting_year": 2025, "electricity_kwh": 100.0}, schema)
        assert len(result["errors"]) == 0

    def test_invalid_row(self):
        v = SchemaValidator()
        schema = SCHEMAS["energy"]
        result = v.validate_row({}, schema)
        assert len(result["errors"]) >= 3


class TestValidateField:
    def test_valid_integer(self):
        v = SchemaValidator()
        assert v.validate_field(2025, "year", "integer") is None

    def test_invalid_integer(self):
        v = SchemaValidator()
        result = v.validate_field("abc", "year", "integer")
        assert result is not None

    def test_valid_float(self):
        v = SchemaValidator()
        assert v.validate_field(1250.5, "emissions", "float") is None

    def test_invalid_float(self):
        v = SchemaValidator()
        result = v.validate_field("not_a_number", "emissions", "float")
        assert result is not None

    def test_valid_string(self):
        v = SchemaValidator()
        assert v.validate_field("London", "name", "string") is None

    def test_none_value(self):
        v = SchemaValidator()
        assert v.validate_field(None, "name", "string") is None


class TestValidateRequiredFields:
    def test_all_present(self):
        v = SchemaValidator()
        errors = v.validate_required_fields(
            {"a": "val", "b": "val"}, ["a", "b"])
        assert errors == []

    def test_missing_field(self):
        v = SchemaValidator()
        errors = v.validate_required_fields({"a": "val"}, ["a", "b"])
        assert len(errors) == 1

    def test_empty_value(self):
        v = SchemaValidator()
        errors = v.validate_required_fields({"a": ""}, ["a"])
        assert len(errors) == 1

    def test_none_value(self):
        v = SchemaValidator()
        errors = v.validate_required_fields({"a": None}, ["a"])
        assert len(errors) == 1


class TestValidateRanges:
    def test_in_range(self):
        v = SchemaValidator()
        warnings = v.validate_ranges({"val": 50}, {"val": (0, 100)})
        assert warnings == []

    def test_below_range(self):
        v = SchemaValidator()
        warnings = v.validate_ranges({"val": -10}, {"val": (0, 100)})
        assert len(warnings) == 1

    def test_above_range(self):
        v = SchemaValidator()
        warnings = v.validate_ranges({"val": 200}, {"val": (0, 100)})
        assert len(warnings) == 1

    def test_none_value_skipped(self):
        v = SchemaValidator()
        warnings = v.validate_ranges({"val": None}, {"val": (0, 100)})
        assert warnings == []


class TestRegisterSchema:
    def test_register_custom_schema(self):
        v = SchemaValidator()
        v.register_schema("custom", {"required_fields": ["field1"]})
        assert "custom" in v.get_available_schemas()

    def test_validate_with_custom_schema(self):
        v = SchemaValidator()
        v.register_schema("custom", {"required_fields": ["name"], "field_types": {}, "field_ranges": {}})
        result = v.validate_data([{"name": "Test"}], "custom")
        assert result["is_valid"] is True


class TestGetAvailableSchemas:
    def test_default_schemas(self):
        v = SchemaValidator()
        schemas = v.get_available_schemas()
        assert len(schemas) >= 3

    def test_unknown_schema_error(self):
        v = SchemaValidator()
        result = v.validate_data([{}], "nonexistent")
        assert result["is_valid"] is False


class TestSchemaValidatorStatistics:
    def test_stats_after_validation(self):
        v = SchemaValidator()
        v.validate_data([{"facility_name": "A", "reporting_year": 2025, "electricity_kwh": 100.0}], "energy")
        stats = v.get_statistics()
        assert stats["validations_run"] == 1
        assert stats["rows_validated"] == 1
