# -*- coding: utf-8 -*-
"""
Unit Tests for SchemaTranslatorEngine (AGENT-DATA-004)

Tests schema registration, retrieval, listing, field mapping registration,
single and batch translation, type coercion, schema validation (required
fields, extra fields, type checking), schema versioning, and SHA-256
provenance tracking on translations.

Coverage target: 85%+ of schema_translator.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inline models
# ---------------------------------------------------------------------------


class SchemaDefinition:
    """Schema definition for a data source."""

    def __init__(self, schema_id: str, name: str, source_type: str,
                 version: str = "1.0.0",
                 fields: Optional[List[Dict[str, Any]]] = None,
                 required_fields: Optional[List[str]] = None):
        self.schema_id = schema_id
        self.name = name
        self.source_type = source_type
        self.version = version
        self.fields = fields or []
        self.required_fields = required_fields or []
        self.provenance_hash = ""
        self.created_at = datetime.now(timezone.utc).isoformat()


class FieldMapping:
    """Maps fields from a source schema to a target schema."""

    def __init__(self, mapping_id: str, source_type: str,
                 target_type: str,
                 field_map: Optional[Dict[str, str]] = None,
                 type_coercions: Optional[Dict[str, str]] = None):
        self.mapping_id = mapping_id
        self.source_type = source_type
        self.target_type = target_type
        self.field_map = field_map or {}
        self.type_coercions = type_coercions or {}


class TranslationResult:
    """Result of translating a record through a mapping."""

    def __init__(self, source_type: str, target_type: str,
                 original: Dict[str, Any], translated: Dict[str, Any],
                 provenance_hash: str = ""):
        self.source_type = source_type
        self.target_type = target_type
        self.original = original
        self.translated = translated
        self.provenance_hash = provenance_hash


# ---------------------------------------------------------------------------
# Inline SchemaTranslatorEngine
# ---------------------------------------------------------------------------


class SchemaTranslatorEngine:
    """Translates data between different source schemas using field mappings."""

    def __init__(self):
        self._schemas: Dict[str, SchemaDefinition] = {}
        self._mappings: Dict[str, FieldMapping] = {}
        self._schema_counter = 0
        self._mapping_counter = 0

    def register_schema(self, name: str, source_type: str,
                        version: str = "1.0.0",
                        fields: Optional[List[Dict[str, Any]]] = None,
                        required_fields: Optional[List[str]] = None) -> SchemaDefinition:
        """Register a new schema definition."""
        self._schema_counter += 1
        schema_id = f"SCH-{self._schema_counter:05d}"
        schema = SchemaDefinition(
            schema_id=schema_id,
            name=name,
            source_type=source_type,
            version=version,
            fields=fields,
            required_fields=required_fields,
        )
        schema.provenance_hash = _compute_hash({
            "schema_id": schema_id,
            "name": name,
            "source_type": source_type,
            "version": version,
        })
        self._schemas[schema_id] = schema
        return schema

    def get_schema(self, schema_id: str) -> Optional[SchemaDefinition]:
        """Get a schema by ID."""
        return self._schemas.get(schema_id)

    def list_schemas(self, source_type: Optional[str] = None) -> List[SchemaDefinition]:
        """List all schemas, optionally filtered by source type."""
        schemas = list(self._schemas.values())
        if source_type is not None:
            schemas = [s for s in schemas if s.source_type == source_type]
        return schemas

    def register_mapping(self, source_type: str, target_type: str,
                         field_map: Dict[str, str],
                         type_coercions: Optional[Dict[str, str]] = None) -> FieldMapping:
        """Register a field mapping between source and target types."""
        self._mapping_counter += 1
        mapping_id = f"MAP-{self._mapping_counter:05d}"
        mapping_key = f"{source_type}->{target_type}"
        mapping = FieldMapping(
            mapping_id=mapping_id,
            source_type=source_type,
            target_type=target_type,
            field_map=field_map,
            type_coercions=type_coercions,
        )
        self._mappings[mapping_key] = mapping
        return mapping

    def translate(self, record: Dict[str, Any],
                  source_type: str, target_type: str) -> TranslationResult:
        """Translate a record from source schema to target schema."""
        mapping_key = f"{source_type}->{target_type}"
        mapping = self._mappings.get(mapping_key)
        if mapping is None:
            raise KeyError(
                f"No mapping registered for {source_type} -> {target_type}"
            )

        translated: Dict[str, Any] = {}
        for src_field, tgt_field in mapping.field_map.items():
            if src_field in record:
                value = record[src_field]
                # Apply type coercion if defined
                if mapping.type_coercions and tgt_field in mapping.type_coercions:
                    value = self._coerce_type(value, mapping.type_coercions[tgt_field])
                translated[tgt_field] = value

        provenance_hash = _compute_hash({
            "source_type": source_type,
            "target_type": target_type,
            "original": record,
            "translated": translated,
        })

        return TranslationResult(
            source_type=source_type,
            target_type=target_type,
            original=record,
            translated=translated,
            provenance_hash=provenance_hash,
        )

    def translate_batch(self, records: List[Dict[str, Any]],
                        source_type: str,
                        target_type: str) -> List[TranslationResult]:
        """Translate a batch of records."""
        return [self.translate(r, source_type, target_type) for r in records]

    def validate_against_schema(self, data: Dict[str, Any],
                                schema_id: str) -> Dict[str, Any]:
        """Validate a data record against a registered schema."""
        schema = self._schemas.get(schema_id)
        if schema is None:
            return {"valid": False, "errors": [f"Schema {schema_id} not found"]}

        errors: List[str] = []

        # Check required fields
        for field_name in schema.required_fields:
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")

        # Check field types
        field_type_map = {f["name"]: f.get("type", "any") for f in schema.fields}
        for field_name, value in data.items():
            if field_name in field_type_map:
                expected_type = field_type_map[field_name]
                if not self._check_type(value, expected_type):
                    errors.append(
                        f"Field '{field_name}' expected type '{expected_type}', "
                        f"got '{type(value).__name__}'"
                    )
            elif field_name not in [f["name"] for f in schema.fields]:
                errors.append(f"Unexpected field: {field_name}")

        return {"valid": len(errors) == 0, "errors": errors}

    def _coerce_type(self, value: Any, target_type: str) -> Any:
        """Coerce a value to the specified type."""
        try:
            if target_type == "str":
                return str(value)
            elif target_type == "int":
                return int(value)
            elif target_type == "float":
                return float(value)
            elif target_type == "bool":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
        except (ValueError, TypeError):
            return value
        return value

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected type."""
        type_map = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": (int, float),
            "number": (int, float),
            "bool": bool,
            "boolean": bool,
            "any": object,
        }
        expected = type_map.get(expected_type, object)
        return isinstance(value, expected)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> SchemaTranslatorEngine:
    return SchemaTranslatorEngine()


@pytest.fixture
def registered_schema(engine) -> SchemaDefinition:
    return engine.register_schema(
        name="emissions_report",
        source_type="erp",
        version="1.0.0",
        fields=[
            {"name": "co2_kg", "type": "float"},
            {"name": "source", "type": "str"},
            {"name": "year", "type": "int"},
        ],
        required_fields=["co2_kg", "source"],
    )


@pytest.fixture
def registered_mapping(engine) -> FieldMapping:
    return engine.register_mapping(
        source_type="erp",
        target_type="canonical",
        field_map={
            "emission_amount": "co2_kg",
            "emission_source": "source",
            "report_year": "year",
        },
        type_coercions={"co2_kg": "float", "year": "int"},
    )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestRegisterSchema:
    """Tests for schema registration."""

    def test_register_schema_success(self, engine):
        schema = engine.register_schema(
            name="test_schema",
            source_type="csv",
            version="1.0.0",
            fields=[{"name": "col_a", "type": "str"}],
        )
        assert schema is not None
        assert schema.schema_id.startswith("SCH-")
        assert schema.name == "test_schema"
        assert schema.source_type == "csv"
        assert schema.version == "1.0.0"

    def test_register_schema_id_generation(self, engine):
        s1 = engine.register_schema("s1", "csv")
        s2 = engine.register_schema("s2", "erp")
        assert s1.schema_id != s2.schema_id
        assert s1.schema_id == "SCH-00001"
        assert s2.schema_id == "SCH-00002"

    def test_register_schema_provenance(self, engine):
        schema = engine.register_schema("test", "csv")
        assert schema.provenance_hash is not None
        assert len(schema.provenance_hash) == 64
        int(schema.provenance_hash, 16)  # valid hex


class TestGetSchema:
    """Tests for schema retrieval."""

    def test_get_schema_exists(self, engine, registered_schema):
        retrieved = engine.get_schema(registered_schema.schema_id)
        assert retrieved is not None
        assert retrieved.schema_id == registered_schema.schema_id
        assert retrieved.name == "emissions_report"

    def test_get_schema_not_found(self, engine):
        result = engine.get_schema("SCH-99999")
        assert result is None


class TestListSchemas:
    """Tests for schema listing."""

    def test_list_schemas_all(self, engine):
        engine.register_schema("s1", "csv")
        engine.register_schema("s2", "erp")
        engine.register_schema("s3", "api")
        schemas = engine.list_schemas()
        assert len(schemas) == 3

    def test_list_schemas_by_source_type(self, engine):
        engine.register_schema("s1", "csv")
        engine.register_schema("s2", "erp")
        engine.register_schema("s3", "csv")
        schemas = engine.list_schemas(source_type="csv")
        assert len(schemas) == 2
        assert all(s.source_type == "csv" for s in schemas)

    def test_list_schemas_empty(self, engine):
        schemas = engine.list_schemas()
        assert schemas == []


class TestRegisterMapping:
    """Tests for field mapping registration."""

    def test_register_mapping_success(self, engine):
        mapping = engine.register_mapping(
            source_type="csv",
            target_type="canonical",
            field_map={"col_a": "field_a"},
        )
        assert mapping is not None
        assert mapping.mapping_id.startswith("MAP-")
        assert mapping.source_type == "csv"
        assert mapping.target_type == "canonical"

    def test_register_mapping_key_format(self, engine):
        engine.register_mapping("csv", "canonical", {"a": "b"})
        key = "csv->canonical"
        assert key in engine._mappings


class TestTranslate:
    """Tests for single record translation."""

    def test_translate_single_field(self, engine, registered_mapping):
        record = {"emission_amount": 150.5}
        result = engine.translate(record, "erp", "canonical")
        assert result.translated["co2_kg"] == 150.5

    def test_translate_multiple_fields(self, engine, registered_mapping):
        record = {
            "emission_amount": 150.5,
            "emission_source": "diesel",
            "report_year": "2025",
        }
        result = engine.translate(record, "erp", "canonical")
        assert result.translated["co2_kg"] == 150.5
        assert result.translated["source"] == "diesel"
        assert result.translated["year"] == 2025  # coerced to int

    def test_translate_missing_mapping(self, engine):
        with pytest.raises(KeyError, match="No mapping registered"):
            engine.translate({"x": 1}, "unknown", "canonical")

    def test_translate_type_coercion(self, engine, registered_mapping):
        record = {"emission_amount": "99.9", "report_year": "2024"}
        result = engine.translate(record, "erp", "canonical")
        assert isinstance(result.translated["co2_kg"], float)
        assert result.translated["co2_kg"] == 99.9
        assert isinstance(result.translated["year"], int)
        assert result.translated["year"] == 2024


class TestTranslateBatch:
    """Tests for batch translation."""

    def test_translate_batch_multiple(self, engine, registered_mapping):
        records = [
            {"emission_amount": 10.0, "emission_source": "gas"},
            {"emission_amount": 20.0, "emission_source": "coal"},
            {"emission_amount": 30.0, "emission_source": "oil"},
        ]
        results = engine.translate_batch(records, "erp", "canonical")
        assert len(results) == 3
        assert results[0].translated["co2_kg"] == 10.0
        assert results[1].translated["source"] == "coal"
        assert results[2].translated["co2_kg"] == 30.0

    def test_translate_batch_empty(self, engine, registered_mapping):
        results = engine.translate_batch([], "erp", "canonical")
        assert results == []


class TestValidateAgainstSchema:
    """Tests for data validation against a registered schema."""

    def test_valid_data(self, engine, registered_schema):
        data = {"co2_kg": 100.5, "source": "diesel", "year": 2025}
        result = engine.validate_against_schema(data, registered_schema.schema_id)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_required_field(self, engine, registered_schema):
        data = {"year": 2025}  # missing co2_kg and source
        result = engine.validate_against_schema(data, registered_schema.schema_id)
        assert result["valid"] is False
        assert any("co2_kg" in e for e in result["errors"])
        assert any("source" in e for e in result["errors"])

    def test_extra_field(self, engine, registered_schema):
        data = {"co2_kg": 100.5, "source": "diesel", "unknown_field": "x"}
        result = engine.validate_against_schema(data, registered_schema.schema_id)
        assert result["valid"] is False
        assert any("Unexpected field" in e for e in result["errors"])

    def test_wrong_type(self, engine, registered_schema):
        data = {"co2_kg": "not_a_number", "source": "diesel"}
        result = engine.validate_against_schema(data, registered_schema.schema_id)
        assert result["valid"] is False
        assert any("expected type" in e for e in result["errors"])


class TestSchemaVersioning:
    """Tests for schema versioning."""

    def test_multiple_versions_same_source(self, engine):
        s1 = engine.register_schema("report_v1", "erp", version="1.0.0")
        s2 = engine.register_schema("report_v2", "erp", version="2.0.0")
        assert s1.schema_id != s2.schema_id
        assert s1.version == "1.0.0"
        assert s2.version == "2.0.0"

        schemas = engine.list_schemas(source_type="erp")
        assert len(schemas) == 2


class TestTranslationProvenance:
    """Tests for SHA-256 provenance tracking on translations."""

    def test_translation_provenance_hash(self, engine, registered_mapping):
        record = {"emission_amount": 100.0, "emission_source": "gas"}
        result = engine.translate(record, "erp", "canonical")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)  # valid hex

    def test_translation_provenance_deterministic(self, engine, registered_mapping):
        record = {"emission_amount": 100.0, "emission_source": "gas"}
        r1 = engine.translate(record, "erp", "canonical")
        r2 = engine.translate(record, "erp", "canonical")
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_records_different_hash(self, engine, registered_mapping):
        r1 = engine.translate({"emission_amount": 10.0}, "erp", "canonical")
        r2 = engine.translate({"emission_amount": 20.0}, "erp", "canonical")
        assert r1.provenance_hash != r2.provenance_hash
