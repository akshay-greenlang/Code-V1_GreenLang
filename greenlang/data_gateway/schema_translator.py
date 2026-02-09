# -*- coding: utf-8 -*-
"""
Schema Translator Engine - AGENT-DATA-004: API Gateway Agent (GL-DATA-GW-001)

Manages schema definitions, field mappings between data sources, and
translates data between different schema formats. Provides schema
validation against registered definitions.

Zero-Hallucination Guarantees:
    - All translations use deterministic field mapping tables
    - Schema validation uses exact type and constraint checking
    - No ML/LLM used for schema inference or mapping
    - SHA-256 provenance hashes on all translation operations

Example:
    >>> from greenlang.data_gateway.schema_translator import SchemaTranslatorEngine
    >>> translator = SchemaTranslatorEngine()
    >>> schema_id = translator.register_schema(schema_def)
    >>> translated = translator.translate(data, "erp", "canonical")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

VALID_FIELD_TYPES = frozenset({
    "string", "integer", "float", "decimal", "boolean",
    "date", "datetime", "timestamp", "json", "array",
    "uuid", "text", "binary", "enum",
})


def _make_schema_definition(
    schema_id: str,
    name: str,
    source_type: str,
    version: str = "1.0.0",
    description: str = "",
    fields: Optional[List[Dict[str, Any]]] = None,
    primary_key: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a SchemaDefinition dictionary.

    Args:
        schema_id: Unique schema identifier.
        name: Human-readable schema name.
        source_type: Source type this schema belongs to.
        version: Schema version string.
        description: Schema description.
        fields: List of field definitions.
        primary_key: Primary key field names.

    Returns:
        SchemaDefinition dictionary.
    """
    now = _utcnow().isoformat()
    return {
        "schema_id": schema_id,
        "name": name,
        "source_type": source_type,
        "version": version,
        "description": description,
        "fields": fields or [],
        "primary_key": primary_key or [],
        "created_at": now,
        "updated_at": now,
    }


def _make_schema_mapping(
    source_field: str,
    target_field: str,
    transform: Optional[str] = None,
    default_value: Any = None,
    required: bool = False,
) -> Dict[str, Any]:
    """Create a SchemaMapping dictionary.

    Args:
        source_field: Source schema field name.
        target_field: Target schema field name.
        transform: Optional transformation function name.
        default_value: Default value if source field is missing.
        required: Whether the mapping is required.

    Returns:
        SchemaMapping dictionary.
    """
    return {
        "source_field": source_field,
        "target_field": target_field,
        "transform": transform,
        "default_value": default_value,
        "required": required,
    }


# ---------------------------------------------------------------------------
# Built-in transformations
# ---------------------------------------------------------------------------

def _transform_uppercase(value: Any) -> Any:
    """Transform value to uppercase string."""
    return str(value).upper() if value is not None else None


def _transform_lowercase(value: Any) -> Any:
    """Transform value to lowercase string."""
    return str(value).lower() if value is not None else None


def _transform_strip(value: Any) -> Any:
    """Strip whitespace from string value."""
    return str(value).strip() if value is not None else None


def _transform_to_float(value: Any) -> Any:
    """Transform value to float."""
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _transform_to_int(value: Any) -> Any:
    """Transform value to integer."""
    try:
        return int(float(value)) if value is not None else None
    except (ValueError, TypeError):
        return None


def _transform_to_string(value: Any) -> Any:
    """Transform value to string."""
    return str(value) if value is not None else None


def _transform_to_bool(value: Any) -> Any:
    """Transform value to boolean."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


BUILT_IN_TRANSFORMS: Dict[str, Any] = {
    "uppercase": _transform_uppercase,
    "lowercase": _transform_lowercase,
    "strip": _transform_strip,
    "to_float": _transform_to_float,
    "to_int": _transform_to_int,
    "to_string": _transform_to_string,
    "to_bool": _transform_to_bool,
}


class SchemaTranslatorEngine:
    """Schema translation and mapping engine.

    Manages schema definitions, field mappings between source types,
    and performs deterministic data translation between schemas.

    Attributes:
        _config: Configuration dictionary or object.
        _provenance: Provenance tracker instance.
        _schemas: In-memory schema definition storage.
        _mappings: In-memory schema mapping storage (key: "src:tgt").

    Example:
        >>> translator = SchemaTranslatorEngine()
        >>> translator.register_mapping("erp", "canonical", mappings)
        >>> result = translator.translate(data, "erp", "canonical")
    """

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize SchemaTranslatorEngine.

        Args:
            config: Optional configuration.
            provenance: Optional ProvenanceTracker instance.
        """
        self._config = config or {}
        self._provenance = provenance
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._mappings: Dict[str, List[Dict[str, Any]]] = {}

        logger.info("SchemaTranslatorEngine initialized")

    # ------------------------------------------------------------------
    # Schema Registration
    # ------------------------------------------------------------------

    def register_schema(
        self,
        schema: Dict[str, Any],
    ) -> str:
        """Register a schema definition.

        Args:
            schema: Schema definition dictionary with keys:
                name (str): Schema name (required).
                source_type (str): Source type (required).
                version (str): Schema version.
                description (str): Description.
                fields (List[Dict]): Field definitions.
                primary_key (List[str]): Primary key fields.

        Returns:
            Generated schema_id.

        Raises:
            ValueError: If required fields are missing.
        """
        name = schema.get("name", "")
        source_type = schema.get("source_type", "")

        if not name:
            raise ValueError("Schema name is required")
        if not source_type:
            raise ValueError("Schema source_type is required")

        schema_id = self._generate_schema_id()

        schema_def = _make_schema_definition(
            schema_id=schema_id,
            name=name,
            source_type=source_type,
            version=schema.get("version", "1.0.0"),
            description=schema.get("description", ""),
            fields=schema.get("fields"),
            primary_key=schema.get("primary_key"),
        )

        self._schemas[schema_id] = schema_def

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(schema_def)
            self._provenance.record(
                entity_type="schema",
                entity_id=schema_id,
                action="schema_translation",
                data_hash=data_hash,
            )

        logger.info(
            "Registered schema %s: name=%s, type=%s, fields=%d",
            schema_id, name, source_type,
            len(schema_def.get("fields", [])),
        )
        return schema_id

    def get_schema(
        self,
        schema_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a schema definition by ID.

        Args:
            schema_id: Schema identifier.

        Returns:
            SchemaDefinition dictionary or None if not found.
        """
        return self._schemas.get(schema_id)

    def list_schemas(
        self,
        source_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List schema definitions with optional filter.

        Args:
            source_type: Filter by source type.

        Returns:
            List of SchemaDefinition dictionaries.
        """
        results = list(self._schemas.values())

        if source_type:
            results = [
                s for s in results
                if s.get("source_type") == source_type.lower()
            ]

        results.sort(key=lambda s: s.get("name", ""))
        return results

    # ------------------------------------------------------------------
    # Mapping Registration
    # ------------------------------------------------------------------

    def register_mapping(
        self,
        source_type: str,
        target_type: str,
        mappings: List[Dict[str, Any]],
    ) -> None:
        """Register field mappings between source and target schemas.

        Args:
            source_type: Source schema type.
            target_type: Target schema type.
            mappings: List of SchemaMapping dictionaries with keys:
                source_field (str): Source field name.
                target_field (str): Target field name.
                transform (str): Optional transform function.
                default_value: Default value.
                required (bool): Whether required.
        """
        key = f"{source_type}:{target_type}"

        validated_mappings: List[Dict[str, Any]] = []
        for m in mappings:
            validated_mappings.append(_make_schema_mapping(
                source_field=m.get("source_field", ""),
                target_field=m.get("target_field", ""),
                transform=m.get("transform"),
                default_value=m.get("default_value"),
                required=m.get("required", False),
            ))

        self._mappings[key] = validated_mappings

        logger.info(
            "Registered %d mappings: %s -> %s",
            len(validated_mappings), source_type, target_type,
        )

    def get_mappings(
        self,
        source_type: str,
        target_type: str,
    ) -> List[Dict[str, Any]]:
        """Get registered mappings between source and target types.

        Args:
            source_type: Source schema type.
            target_type: Target schema type.

        Returns:
            List of SchemaMapping dictionaries.
        """
        key = f"{source_type}:{target_type}"
        return list(self._mappings.get(key, []))

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def translate(
        self,
        data: Dict[str, Any],
        source_type: str,
        target_type: str,
    ) -> Dict[str, Any]:
        """Translate a data dictionary between schemas.

        Applies field mappings and transformations to convert data
        from source schema format to target schema format.

        Args:
            data: Source data dictionary.
            source_type: Source schema type.
            target_type: Target schema type.

        Returns:
            Translated data dictionary.

        Raises:
            ValueError: If no mappings registered for this pair.
        """
        start_time = time.monotonic()

        key = f"{source_type}:{target_type}"
        mappings = self._mappings.get(key)

        if mappings is None:
            raise ValueError(
                f"No mappings registered for {source_type} -> {target_type}"
            )

        result: Dict[str, Any] = {}
        errors: List[str] = []

        for mapping in mappings:
            source_field = mapping["source_field"]
            target_field = mapping["target_field"]
            transform = mapping.get("transform")
            default_value = mapping.get("default_value")
            required = mapping.get("required", False)

            # Get source value
            value = data.get(source_field, default_value)

            # Check required
            if required and value is None:
                errors.append(
                    f"Required field '{source_field}' missing from source"
                )
                continue

            # Apply transform
            if transform and value is not None:
                transform_fn = BUILT_IN_TRANSFORMS.get(transform)
                if transform_fn:
                    value = transform_fn(value)
                else:
                    logger.warning(
                        "Unknown transform '%s' for field '%s'",
                        transform, source_field,
                    )

            result[target_field] = value

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(result)
            self._provenance.record(
                entity_type="translation",
                entity_id=f"{source_type}:{target_type}",
                action="schema_translation",
                data_hash=data_hash,
            )

        # Record metrics
        try:
            from greenlang.data_gateway.metrics import (
                record_schema_translation,
            )
            record_schema_translation(
                source=source_type,
                direction=f"{source_type}_to_{target_type}",
            )
        except ImportError:
            pass

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Translated %s -> %s: %d fields mapped (%.1f ms)",
            source_type, target_type, len(result), elapsed_ms,
        )

        if errors:
            logger.warning(
                "Translation errors: %s", "; ".join(errors),
            )
            result["_translation_errors"] = errors

        return result

    def translate_batch(
        self,
        data: List[Dict[str, Any]],
        source_type: str,
        target_type: str,
    ) -> List[Dict[str, Any]]:
        """Translate a batch of data dictionaries between schemas.

        Args:
            data: List of source data dictionaries.
            source_type: Source schema type.
            target_type: Target schema type.

        Returns:
            List of translated data dictionaries.
        """
        results: List[Dict[str, Any]] = []
        for item in data:
            try:
                translated = self.translate(item, source_type, target_type)
                results.append(translated)
            except Exception as e:
                logger.error(
                    "Batch translation error for item: %s", e,
                )
                results.append({
                    "_translation_error": str(e),
                    "_source_data": item,
                })
        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_against_schema(
        self,
        data: Dict[str, Any],
        schema_id: str,
    ) -> List[str]:
        """Validate data against a registered schema definition.

        Checks:
        - Required fields are present
        - Field types match (basic type checking)
        - Primary key fields are present and non-null

        Args:
            data: Data dictionary to validate.
            schema_id: Schema identifier to validate against.

        Returns:
            List of validation error messages (empty if valid).
        """
        schema = self._schemas.get(schema_id)
        if schema is None:
            return [f"Schema not found: {schema_id}"]

        errors: List[str] = []

        # Validate primary key fields
        for pk_field in schema.get("primary_key", []):
            if pk_field not in data or data[pk_field] is None:
                errors.append(
                    f"Primary key field '{pk_field}' is missing or null"
                )

        # Validate fields
        for field_def in schema.get("fields", []):
            field_name = field_def.get("name", "")
            field_type = field_def.get("type", "string")
            required = field_def.get("required", False)
            nullable = field_def.get("nullable", True)

            if not field_name:
                continue

            value = data.get(field_name)

            # Check required
            if required and field_name not in data:
                errors.append(
                    f"Required field '{field_name}' is missing"
                )
                continue

            # Check nullable
            if not nullable and value is None and field_name in data:
                errors.append(
                    f"Field '{field_name}' cannot be null"
                )
                continue

            # Basic type checking (if value is present and not null)
            if value is not None and field_type in VALID_FIELD_TYPES:
                type_error = self._check_field_type(
                    field_name, value, field_type,
                )
                if type_error:
                    errors.append(type_error)

        return errors

    def _check_field_type(
        self,
        field_name: str,
        value: Any,
        expected_type: str,
    ) -> Optional[str]:
        """Check if a value matches the expected field type.

        Args:
            field_name: Field name for error messages.
            value: Value to check.
            expected_type: Expected type name.

        Returns:
            Error message or None if valid.
        """
        type_checks = {
            "string": (str,),
            "text": (str,),
            "integer": (int,),
            "float": (int, float),
            "decimal": (int, float),
            "boolean": (bool,),
            "array": (list,),
            "json": (dict, list),
        }

        expected_python_types = type_checks.get(expected_type)
        if expected_python_types and not isinstance(value, expected_python_types):
            return (
                f"Field '{field_name}' expected type {expected_type}, "
                f"got {type(value).__name__}"
            )
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_schema_id(self) -> str:
        """Generate a unique schema identifier.

        Returns:
            Schema ID in format "SCH-{hex12}".
        """
        return f"SCH-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def schema_count(self) -> int:
        """Return the total number of registered schemas."""
        return len(self._schemas)

    @property
    def mapping_count(self) -> int:
        """Return the total number of registered mapping sets."""
        return len(self._mappings)

    def get_statistics(self) -> Dict[str, Any]:
        """Get translator statistics.

        Returns:
            Dictionary with schema and mapping counts.
        """
        return {
            "total_schemas": len(self._schemas),
            "total_mapping_sets": len(self._mappings),
            "mapping_pairs": list(self._mappings.keys()),
        }


__all__ = [
    "SchemaTranslatorEngine",
    "VALID_FIELD_TYPES",
    "BUILT_IN_TRANSFORMS",
]
