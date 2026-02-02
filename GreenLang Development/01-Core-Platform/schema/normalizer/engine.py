# -*- coding: utf-8 -*-
"""
Normalization Engine for GL-FOUND-X-002.

This module implements the main normalization engine that orchestrates all
payload transformations to produce canonical form. The normalization process
is deterministic, idempotent, and non-destructive.

Normalization Order:
    1. Key canonicalization (aliases, casing)
    2. Default application
    3. Type coercion
    4. Unit canonicalization
    5. Add _meta block with audit trail

Design Principles:
    - Deterministic: Same input -> same output, always
    - Idempotent: normalize(normalize(x)) == normalize(x)
    - Non-destructive: Never loses semantic information
    - Zero-hallucination: All transformations are schema-driven
    - Full provenance: Complete audit trail in _meta block

Example:
    >>> from greenlang.schema.normalizer.engine import NormalizationEngine, normalize
    >>> from greenlang.schema.compiler.ir import SchemaIR
    >>> from greenlang.schema.units.catalog import UnitCatalog
    >>> from greenlang.schema.models.config import ValidationOptions
    >>>
    >>> engine = NormalizationEngine(ir, UnitCatalog(), ValidationOptions())
    >>> result = engine.normalize({"Energy": "100 Wh"})
    >>> print(result.normalized)
    {"energy": {"value": 0.1, "unit": "kWh", "_meta": {...}}}
    >>> print(result.is_modified)
    True

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 3.4
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from ..compiler.ir import SchemaIR, UnitSpecIR, PropertyIR
from ..models.config import ValidationOptions, CoercionPolicy
from ..models.schema_ref import SchemaRef
from ..units.catalog import UnitCatalog
from .coercions import CoercionEngine, CoercionRecord, get_python_type_name
from .canonicalizer import UnitCanonicalizer, ConversionRecord
from .keys import KeyCanonicalizer, KeyRename

logger = logging.getLogger(__name__)


# =============================================================================
# NORMALIZATION META MODEL
# =============================================================================


class NormalizationMeta(BaseModel):
    """
    Metadata about normalization operations.

    Captures complete provenance information for all transformations applied
    during normalization, enabling full audit trail and reproducibility.

    Attributes:
        schema_ref: Reference to the schema used for normalization
        normalized_at: Timestamp when normalization was performed (UTC)
        coercions: List of type coercion records
        conversions: List of unit conversion records
        renames: List of key rename records
        defaults_applied: List of JSON Pointer paths where defaults were applied
        provenance_hash: SHA-256 hash of the normalized payload for integrity

    Example:
        >>> meta = NormalizationMeta(
        ...     schema_ref=SchemaRef(schema_id="test", version="1.0.0"),
        ...     normalized_at=datetime.now(timezone.utc),
        ...     coercions=[],
        ...     conversions=[],
        ...     renames=[],
        ...     defaults_applied=["/optional_field"]
        ... )
    """

    schema_ref: Optional[SchemaRef] = Field(
        default=None,
        description="Reference to the schema used for normalization"
    )
    normalized_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when normalization was performed (UTC)"
    )
    coercions: List[CoercionRecord] = Field(
        default_factory=list,
        description="List of type coercion records"
    )
    conversions: List[ConversionRecord] = Field(
        default_factory=list,
        description="List of unit conversion records"
    )
    renames: List[KeyRename] = Field(
        default_factory=list,
        description="List of key rename records"
    )
    defaults_applied: List[str] = Field(
        default_factory=list,
        description="List of paths where defaults were applied"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the normalized payload"
    )

    model_config = ConfigDict(
        frozen=False,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for embedding in normalized payload.

        Returns:
            Dictionary representation suitable for _meta block
        """
        result: Dict[str, Any] = {
            "normalized_at": self.normalized_at.isoformat(),
        }

        if self.schema_ref:
            result["schema_ref"] = self.schema_ref.to_uri()

        if self.coercions:
            result["coercions"] = [c.to_dict() for c in self.coercions]

        if self.conversions:
            result["conversions"] = [c.to_dict() for c in self.conversions]

        if self.renames:
            result["renames"] = [r.to_dict() for r in self.renames]

        if self.defaults_applied:
            result["defaults_applied"] = self.defaults_applied

        if self.provenance_hash:
            result["provenance_hash"] = self.provenance_hash

        return result

    def has_changes(self) -> bool:
        """Check if any normalization changes were made."""
        return bool(
            self.coercions
            or self.conversions
            or self.renames
            or self.defaults_applied
        )


# =============================================================================
# NORMALIZATION RESULT MODEL
# =============================================================================


class NormalizationResult(BaseModel):
    """
    Result of normalization operation.

    Encapsulates the complete outcome of a normalization run, including
    the normalized payload, metadata, and modification status.

    Attributes:
        normalized: The normalized payload in canonical form
        meta: Metadata about normalization operations
        is_modified: Whether any changes were made to the original payload

    Example:
        >>> result = engine.normalize(payload)
        >>> if result.is_modified:
        ...     print(f"Applied {len(result.meta.coercions)} coercions")
        >>> normalized = result.normalized
    """

    normalized: Dict[str, Any] = Field(
        ...,
        description="The normalized payload in canonical form"
    )
    meta: NormalizationMeta = Field(
        ...,
        description="Metadata about normalization operations"
    )
    is_modified: bool = Field(
        ...,
        description="Whether any changes were made to the original payload"
    )

    model_config = ConfigDict(
        frozen=False,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @property
    def coercion_count(self) -> int:
        """Get number of coercions applied."""
        return len(self.meta.coercions)

    @property
    def conversion_count(self) -> int:
        """Get number of unit conversions applied."""
        return len(self.meta.conversions)

    @property
    def rename_count(self) -> int:
        """Get number of key renames applied."""
        return len(self.meta.renames)

    @property
    def default_count(self) -> int:
        """Get number of defaults applied."""
        return len(self.meta.defaults_applied)


# =============================================================================
# NORMALIZATION ENGINE
# =============================================================================


class NormalizationEngine:
    """
    Main normalization engine.

    Orchestrates the complete normalization process, transforming payloads
    to canonical form according to schema definitions. The normalization
    is deterministic, idempotent, and maintains complete audit trail.

    Normalization Order:
        1. Key canonicalization (aliases, casing)
        2. Default application for missing optional fields
        3. Type coercion (string to number, etc.)
        4. Unit canonicalization (convert to SI)
        5. Add _meta block with audit trail

    Attributes:
        ir: Compiled schema intermediate representation
        catalog: Unit catalog for conversions
        options: Validation options controlling normalization behavior

    Example:
        >>> engine = NormalizationEngine(ir, catalog, options)
        >>> result = engine.normalize(
        ...     payload={"Energy": "100 Wh"},
        ...     schema_ref=SchemaRef(schema_id="test", version="1.0.0")
        ... )
        >>> print(result.normalized)
        {"energy": {"value": 0.1, "unit": "kWh"}, "_meta": {...}}
    """

    def __init__(
        self,
        ir: SchemaIR,
        catalog: UnitCatalog,
        options: ValidationOptions
    ):
        """
        Initialize the NormalizationEngine.

        Args:
            ir: Compiled schema intermediate representation
            catalog: Unit catalog for unit lookups and conversions
            options: Validation options controlling normalization behavior
        """
        self.ir = ir
        self.catalog = catalog
        self.options = options

        # Initialize sub-engines
        self._coercion_engine = CoercionEngine(options.coercion_policy)
        self._unit_canonicalizer = UnitCanonicalizer(catalog)
        self._key_canonicalizer = KeyCanonicalizer(ir)

        logger.debug(
            f"NormalizationEngine initialized for schema {ir.schema_id}@{ir.version}"
        )

    # -------------------------------------------------------------------------
    # Main Normalization Method
    # -------------------------------------------------------------------------

    def normalize(
        self,
        payload: Dict[str, Any],
        schema_ref: Optional[SchemaRef] = None
    ) -> NormalizationResult:
        """
        Normalize payload to canonical form.

        Performs these transformations in order:
        1. Key canonicalization (resolves aliases, normalizes casing)
        2. Default application (applies schema defaults for missing fields)
        3. Type coercion (converts strings to numbers, booleans, etc.)
        4. Unit canonicalization (converts to SI canonical units)
        5. Add _meta block (audit trail of all transformations)

        The normalization is:
        - Deterministic: Same input always produces same output
        - Idempotent: normalize(normalize(x)) == normalize(x)
        - Non-destructive: Never loses semantic information

        Args:
            payload: The payload dictionary to normalize
            schema_ref: Optional schema reference for provenance tracking

        Returns:
            NormalizationResult containing:
            - normalized: The normalized payload
            - meta: Metadata about transformations
            - is_modified: Whether any changes were made

        Raises:
            ValueError: If payload is not a dictionary

        Example:
            >>> result = engine.normalize({"energyConsumption": "100 kWh"})
            >>> print(result.normalized)
            {"energy_consumption": {"value": 100.0, "unit": "kWh"}}
        """
        if not isinstance(payload, dict):
            raise ValueError(
                f"Payload must be a dictionary, got {type(payload).__name__}"
            )

        start_time = time.time()

        # Create deep copy to avoid modifying original
        working_payload = copy.deepcopy(payload)

        # Initialize metadata
        meta = NormalizationMeta(
            schema_ref=schema_ref,
            normalized_at=datetime.now(timezone.utc),
        )

        # Skip normalization if disabled in options
        if not self.options.normalize:
            logger.debug("Normalization disabled, returning original payload")
            meta.provenance_hash = self._compute_provenance_hash(working_payload)
            return NormalizationResult(
                normalized=working_payload,
                meta=meta,
                is_modified=False
            )

        try:
            # Step 1: Key canonicalization
            working_payload, key_renames = self._canonicalize_keys(working_payload)
            meta.renames = key_renames

            # Step 2: Apply defaults
            working_payload, defaults_applied = self._apply_defaults(working_payload)
            meta.defaults_applied = defaults_applied

            # Step 3: Type coercion
            working_payload, coercions = self._coerce_types(working_payload)
            meta.coercions = coercions

            # Step 4: Unit canonicalization
            working_payload, conversions = self._canonicalize_units(working_payload)
            meta.conversions = conversions

            # Compute provenance hash
            meta.provenance_hash = self._compute_provenance_hash(working_payload)

            # Step 5: Add _meta block if changes were made
            working_payload = self._add_meta_block(working_payload, meta)

            # Determine if any modifications were made
            is_modified = meta.has_changes()

            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Normalization completed in {elapsed_ms:.2f}ms: "
                f"{len(key_renames)} renames, {len(defaults_applied)} defaults, "
                f"{len(coercions)} coercions, {len(conversions)} conversions"
            )

            return NormalizationResult(
                normalized=working_payload,
                meta=meta,
                is_modified=is_modified
            )

        except Exception as e:
            logger.error(f"Normalization failed: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Step 1: Key Canonicalization
    # -------------------------------------------------------------------------

    def _canonicalize_keys(
        self,
        payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[KeyRename]]:
        """
        Step 1: Canonicalize keys.

        Resolves key aliases defined in schema and normalizes casing
        to the expected convention (default: snake_case).

        Args:
            payload: The payload to process

        Returns:
            Tuple of (canonicalized payload, list of KeyRename records)
        """
        logger.debug("Step 1: Canonicalizing keys")

        try:
            result, renames = self._key_canonicalizer.canonicalize(payload)
            if renames:
                logger.debug(f"Applied {len(renames)} key renames")
            return result, renames
        except Exception as e:
            logger.warning(f"Key canonicalization failed: {e}")
            return payload, []

    # -------------------------------------------------------------------------
    # Step 2: Default Application
    # -------------------------------------------------------------------------

    def _apply_defaults(
        self,
        payload: Dict[str, Any],
        path: str = ""
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Step 2: Apply schema defaults for missing optional fields.

        Only applies defaults if options.normalize is True. Defaults are
        applied recursively for nested objects.

        Args:
            payload: The payload to process
            path: Current JSON Pointer path (for recursion)

        Returns:
            Tuple of (payload with defaults, list of paths where defaults applied)
        """
        logger.debug("Step 2: Applying defaults")

        defaults_applied: List[str] = []
        result = copy.copy(payload)

        # Apply defaults at current level
        for prop_path, prop_ir in self.ir.properties.items():
            # Check if this property belongs at current level
            if not self._is_direct_child_path(prop_path, path):
                continue

            # Extract key name from path
            key_name = self._extract_key_from_path(prop_path)
            if not key_name:
                continue

            # Check if key is missing and has default
            if key_name not in result:
                has_default, default_value = self._get_property_default(prop_path)
                if has_default and self._should_apply_default(prop_path):
                    result[key_name] = copy.deepcopy(default_value)
                    defaults_applied.append(prop_path)
                    logger.debug(f"Applied default at {prop_path}: {default_value}")

        # Recursively apply defaults to nested objects
        for key, value in result.items():
            if isinstance(value, dict):
                child_path = f"{path}/{key}"
                result[key], child_defaults = self._apply_defaults(value, child_path)
                defaults_applied.extend(child_defaults)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        item_path = f"{path}/{key}/{i}"
                        result[key][i], item_defaults = self._apply_defaults(
                            item, item_path
                        )
                        defaults_applied.extend(item_defaults)

        return result, defaults_applied

    def _get_property_default(
        self,
        path: str
    ) -> Tuple[bool, Any]:
        """
        Get default value for property from IR.

        Args:
            path: JSON Pointer path to the property

        Returns:
            Tuple of (has_default, default_value)
        """
        prop_ir = self.ir.get_property(path)
        if prop_ir is None:
            return False, None

        if prop_ir.has_default:
            return True, prop_ir.default_value

        return False, None

    def _should_apply_default(
        self,
        path: str
    ) -> bool:
        """
        Check if default should be applied for this path.

        Defaults are applied for optional fields only.
        Required fields must be present (validation will catch missing required).

        Args:
            path: JSON Pointer path to the property

        Returns:
            True if default should be applied
        """
        # Don't apply defaults to required fields - let validation handle it
        if path in self.ir.required_paths:
            return False

        return True

    # -------------------------------------------------------------------------
    # Step 3: Type Coercion
    # -------------------------------------------------------------------------

    def _coerce_types(
        self,
        payload: Dict[str, Any],
        path: str = ""
    ) -> Tuple[Dict[str, Any], List[CoercionRecord]]:
        """
        Step 3: Coerce types according to schema.

        Performs safe type coercions based on schema type expectations.
        Only coercions allowed by the coercion policy are applied.

        Args:
            payload: The payload to process
            path: Current JSON Pointer path (for recursion)

        Returns:
            Tuple of (coerced payload, list of CoercionRecord)
        """
        logger.debug("Step 3: Coercing types")

        # Clear previous records from engine
        self._coercion_engine.clear_records()

        # Skip if coercion is disabled
        if self.options.coercion_policy == CoercionPolicy.OFF:
            logger.debug("Type coercion disabled")
            return payload, []

        result = self._coerce_types_recursive(payload, path)
        coercions = self._coercion_engine.get_records()

        if coercions:
            logger.debug(f"Applied {len(coercions)} type coercions")

        return result, coercions

    def _coerce_types_recursive(
        self,
        data: Any,
        path: str
    ) -> Any:
        """
        Recursively coerce types in nested structures.

        Args:
            data: Data to process
            path: Current JSON Pointer path

        Returns:
            Coerced data
        """
        if isinstance(data, dict):
            result: Dict[str, Any] = {}
            for key, value in data.items():
                child_path = f"{path}/{key}"
                result[key] = self._coerce_types_recursive(value, child_path)
            return result

        elif isinstance(data, list):
            result_list: List[Any] = []
            for i, item in enumerate(data):
                item_path = f"{path}/{i}"
                result_list.append(self._coerce_types_recursive(item, item_path))
            return result_list

        else:
            # Primitive value - attempt coercion based on schema type
            return self._coerce_value(data, path)

    def _coerce_value(
        self,
        value: Any,
        path: str
    ) -> Any:
        """
        Coerce a single value based on schema type expectation.

        Args:
            value: The value to coerce
            path: JSON Pointer path to the value

        Returns:
            Coerced value (or original if no coercion needed/possible)
        """
        # Get expected type from schema
        prop_ir = self.ir.get_property(path)
        if prop_ir is None or prop_ir.type is None:
            return value

        target_type = prop_ir.type

        # Handle union types (e.g., ["string", "null"])
        if isinstance(target_type, list):
            # Try each type in order
            for t in target_type:
                result = self._coercion_engine.coerce(value, t, path)
                if result.success:
                    return result.value
            return value

        # Single type
        result = self._coercion_engine.coerce(value, target_type, path)
        return result.value if result.success else value

    # -------------------------------------------------------------------------
    # Step 4: Unit Canonicalization
    # -------------------------------------------------------------------------

    def _canonicalize_units(
        self,
        payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[ConversionRecord]]:
        """
        Step 4: Canonicalize units to SI.

        Converts all unit values to their canonical SI form while
        preserving original values in metadata.

        Args:
            payload: The payload to process

        Returns:
            Tuple of (canonicalized payload, list of ConversionRecord)
        """
        logger.debug("Step 4: Canonicalizing units")

        # Clear previous records
        self._unit_canonicalizer.clear_records()

        # Get unit specs from IR
        if not self.ir.unit_specs:
            logger.debug("No unit specifications in schema")
            return payload, []

        # Canonicalize using unit canonicalizer
        result, conversions = self._unit_canonicalizer.canonicalize_object(
            payload, self.ir.unit_specs
        )

        if conversions:
            logger.debug(f"Applied {len(conversions)} unit conversions")

        return result, conversions

    # -------------------------------------------------------------------------
    # Step 5: Add Meta Block
    # -------------------------------------------------------------------------

    def _add_meta_block(
        self,
        payload: Dict[str, Any],
        meta: NormalizationMeta
    ) -> Dict[str, Any]:
        """
        Step 5: Add _meta block with normalization audit trail.

        Only adds the _meta block if there were any changes or if
        the schema requires it.

        Args:
            payload: The normalized payload
            meta: Normalization metadata

        Returns:
            Payload with _meta block added (if applicable)
        """
        # Only add _meta if there were changes
        if not meta.has_changes():
            logger.debug("No changes made, skipping _meta block")
            return payload

        logger.debug("Step 5: Adding _meta block")

        result = copy.copy(payload)
        result["_meta"] = meta.to_dict()

        return result

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _compute_provenance_hash(self, payload: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of the normalized payload for provenance.

        Args:
            payload: The payload to hash

        Returns:
            Hexadecimal SHA-256 hash string
        """
        # Serialize deterministically (sorted keys)
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _is_direct_child_path(self, prop_path: str, parent_path: str) -> bool:
        """
        Check if a property path is a direct child of a parent path.

        Args:
            prop_path: Property path (e.g., "/foo/bar")
            parent_path: Parent path (e.g., "/foo" or "")

        Returns:
            True if prop_path is a direct child of parent_path
        """
        if parent_path == "":
            # Root level - check for single segment
            parts = prop_path.strip("/").split("/")
            return len(parts) == 1
        else:
            # Nested - check for exactly one more segment
            if not prop_path.startswith(parent_path + "/"):
                return False
            remainder = prop_path[len(parent_path) + 1:]
            return "/" not in remainder

    def _extract_key_from_path(self, path: str) -> Optional[str]:
        """
        Extract the key name from a JSON Pointer path.

        Args:
            path: JSON Pointer path (e.g., "/foo/bar")

        Returns:
            Key name (e.g., "bar") or None if invalid
        """
        if not path:
            return None

        parts = path.strip("/").split("/")
        return parts[-1] if parts else None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def normalize(
    payload: Dict[str, Any],
    ir: SchemaIR,
    catalog: Optional[UnitCatalog] = None,
    options: Optional[ValidationOptions] = None
) -> NormalizationResult:
    """
    Convenience function to normalize a payload.

    Creates a NormalizationEngine and normalizes the payload in one call.
    Useful for simple normalization without managing engine lifecycle.

    Args:
        payload: The payload dictionary to normalize
        ir: Compiled schema intermediate representation
        catalog: Unit catalog for conversions (creates default if None)
        options: Validation options (creates default if None)

    Returns:
        NormalizationResult with normalized payload and metadata

    Example:
        >>> result = normalize(payload, compiled_ir)
        >>> if result.is_modified:
        ...     print(f"Applied {len(result.meta.coercions)} coercions")
        >>> normalized = result.normalized
    """
    # Create defaults if not provided
    if catalog is None:
        catalog = UnitCatalog()

    if options is None:
        options = ValidationOptions()

    # Create engine and normalize
    engine = NormalizationEngine(ir, catalog, options)

    # Create schema ref from IR
    schema_ref = SchemaRef(
        schema_id=ir.schema_id,
        version=ir.version
    )

    return engine.normalize(payload, schema_ref)


# =============================================================================
# PROPERTY TEST HELPER
# =============================================================================


def is_normalization_idempotent(
    payload: Dict[str, Any],
    ir: SchemaIR,
    catalog: UnitCatalog,
    options: ValidationOptions
) -> bool:
    """
    Test that normalization is idempotent.

    Verifies that normalize(normalize(x)) == normalize(x), which is
    a critical property for correctness.

    This function is designed for use in property-based testing
    with Hypothesis or similar frameworks.

    Args:
        payload: The payload to test
        ir: Compiled schema IR
        catalog: Unit catalog
        options: Validation options

    Returns:
        True if normalization is idempotent for this payload

    Example:
        >>> from hypothesis import given
        >>> from hypothesis.strategies import dictionaries, text, integers
        >>>
        >>> @given(dictionaries(text(), integers()))
        >>> def test_idempotent(payload):
        ...     assert is_normalization_idempotent(payload, ir, catalog, options)
    """
    engine = NormalizationEngine(ir, catalog, options)

    # First normalization
    result1 = engine.normalize(payload)

    # Second normalization (of the normalized result, excluding _meta)
    payload2 = _remove_meta_block(result1.normalized)
    result2 = engine.normalize(payload2)

    # Compare (excluding _meta blocks and timestamps)
    normalized1 = _remove_meta_block(result1.normalized)
    normalized2 = _remove_meta_block(result2.normalized)

    return normalized1 == normalized2


def _remove_meta_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove _meta blocks from payload for comparison.

    Args:
        payload: Payload to process

    Returns:
        Payload with all _meta blocks removed
    """
    result: Dict[str, Any] = {}

    for key, value in payload.items():
        if key == "_meta":
            continue

        if isinstance(value, dict):
            result[key] = _remove_meta_block(value)
        elif isinstance(value, list):
            result[key] = [
                _remove_meta_block(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    return result


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Models
    "NormalizationMeta",
    "NormalizationResult",
    # Engine
    "NormalizationEngine",
    # Convenience functions
    "normalize",
    # Property test helpers
    "is_normalization_idempotent",
]
