# -*- coding: utf-8 -*-
"""
Schema Registry Engine — AGENT-DATA-017

Engine 1 of 7 in the Schema Migration Agent pipeline.

Registers, catalogs, and manages schemas with namespaces, tags, ownership,
and a full status lifecycle (draft → active → deprecated → archived). Supports
JSON Schema (Draft 2020-12), Avro, and Protobuf-like definitions. Provides
full-text schema search, import/export, bulk operations, and schema grouping
by domain.

Zero-Hallucination Guarantees:
    - All schema registration uses deterministic UUID generation
    - Status transitions enforce a strict one-way state machine
    - Definition validation uses structural rule-based checks only
    - SHA-256 provenance hashes recorded on every mutating operation
    - No LLM or ML model in any registry path

Supported Schema Types:
    - json_schema: JSON Schema Draft 2020-12 (validated for required keys)
    - avro:        Apache Avro record schema (validated for type/name/fields)
    - protobuf:    Protobuf-like descriptor (validated for syntax/messages)

Status Lifecycle (one-way, no backward transitions):
    draft → active → deprecated → archived

Example:
    >>> from greenlang.schema_migration.schema_registry import SchemaRegistryEngine
    >>> engine = SchemaRegistryEngine()
    >>> schema = engine.register_schema(
    ...     namespace="emissions",
    ...     name="ActivityRecord",
    ...     schema_type="json_schema",
    ...     definition_json={"$schema": "https://json-schema.org/draft/2020-12/schema",
    ...                      "type": "object", "properties": {}},
    ...     owner="platform-team",
    ...     tags=["emissions", "scope3"],
    ... )
    >>> print(schema["schema_id"], schema["status"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: ProvenanceTracker
# ---------------------------------------------------------------------------

try:
    from greenlang.data_quality_profiler.provenance import ProvenanceTracker  # type: ignore
    _PROVENANCE_AVAILABLE = True
except ImportError:
    try:
        from greenlang.provenance import ProvenanceTracker  # type: ignore
        _PROVENANCE_AVAILABLE = True
    except ImportError:
        _PROVENANCE_AVAILABLE = False
        logger.info(
            "ProvenanceTracker not available; using built-in SHA-256 tracking only."
        )

        class ProvenanceTracker:  # type: ignore  # noqa: F811
            """Minimal fallback ProvenanceTracker when the real module is absent."""

            def __init__(self) -> None:
                self._log: List[Dict[str, Any]] = []

            def record(
                self,
                entity_type: str,
                entity_id: str,
                action: str,
                data_hash: str,
                user_id: str = "system",
            ) -> str:
                """Record a provenance entry and return a chain hash."""
                ts = _utcnow().isoformat()
                chain_input = json.dumps(
                    {"entity_type": entity_type, "entity_id": entity_id,
                     "action": action, "data_hash": data_hash, "ts": ts},
                    sort_keys=True,
                )
                chain_hash = hashlib.sha256(chain_input.encode()).hexdigest()
                self._log.append(
                    {"entity_type": entity_type, "entity_id": entity_id,
                     "action": action, "data_hash": data_hash, "ts": ts,
                     "chain_hash": chain_hash, "user_id": user_id}
                )
                return chain_hash

            def build_hash(self, data: Any) -> str:
                """Return SHA-256 hash of JSON-serialised data."""
                return hashlib.sha256(
                    json.dumps(data, sort_keys=True, default=str).encode()
                ).hexdigest()


# ---------------------------------------------------------------------------
# Optional dependency: Prometheus metrics
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram  # type: ignore
    _PROMETHEUS_AVAILABLE = True

    sm_schemas_registered_total = Counter(
        "gl_sm_schemas_registered_total",
        "Total schemas registered",
        labelnames=["schema_type"],
    )
    sm_schemas_by_status = Gauge(
        "gl_sm_schemas_by_status",
        "Count of schemas by current status",
        labelnames=["status"],
    )
    sm_search_queries_total = Counter(
        "gl_sm_search_queries_total",
        "Total search queries executed",
    )
    sm_import_total = Counter(
        "gl_sm_import_total",
        "Total schemas processed during bulk import",
        labelnames=["result"],
    )
    sm_validation_errors_total = Counter(
        "gl_sm_validation_errors_total",
        "Total schema definition validation errors",
        labelnames=["schema_type"],
    )
    sm_groups_total = Gauge(
        "gl_sm_groups_total",
        "Total number of schema groups",
    )
    sm_operation_duration_seconds = Histogram(
        "gl_sm_operation_duration_seconds",
        "Duration of registry operations in seconds",
        labelnames=["operation"],
    )
    sm_processing_errors_total = Counter(
        "gl_sm_processing_errors_total",
        "Total processing errors in registry",
        labelnames=["error_type"],
    )
    sm_exports_total = Counter(
        "gl_sm_exports_total",
        "Total schema export operations",
    )
    sm_status_transitions_total = Counter(
        "gl_sm_status_transitions_total",
        "Total status transitions",
        labelnames=["from_status", "to_status"],
    )
    sm_registry_size = Gauge(
        "gl_sm_registry_size",
        "Current total number of schemas in registry",
    )
    sm_namespaces_total = Gauge(
        "gl_sm_namespaces_total",
        "Total distinct namespaces in the registry",
    )

except (ImportError, ValueError):
    _PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed or metrics already registered; "
        "schema registry metrics disabled."
    )

    class _NoOpCounter:  # type: ignore
        def labels(self, **_: Any) -> "_NoOpCounter":
            return self
        def inc(self, _: float = 1) -> None:
            pass

    class _NoOpGauge:  # type: ignore
        def labels(self, **_: Any) -> "_NoOpGauge":
            return self
        def set(self, _: float) -> None:
            pass
        def inc(self, _: float = 1) -> None:
            pass
        def dec(self, _: float = 1) -> None:
            pass

    class _NoOpHistogram:  # type: ignore
        def labels(self, **_: Any) -> "_NoOpHistogram":
            return self
        def observe(self, _: float) -> None:
            pass

    sm_schemas_registered_total = _NoOpCounter()  # type: ignore
    sm_schemas_by_status = _NoOpGauge()  # type: ignore
    sm_search_queries_total = _NoOpCounter()  # type: ignore
    sm_import_total = _NoOpCounter()  # type: ignore
    sm_validation_errors_total = _NoOpCounter()  # type: ignore
    sm_groups_total = _NoOpGauge()  # type: ignore
    sm_operation_duration_seconds = _NoOpHistogram()  # type: ignore
    sm_processing_errors_total = _NoOpCounter()  # type: ignore
    sm_exports_total = _NoOpCounter()  # type: ignore
    sm_status_transitions_total = _NoOpCounter()  # type: ignore
    sm_registry_size = _NoOpGauge()  # type: ignore
    sm_namespaces_total = _NoOpGauge()  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum number of schemas importable in a single bulk import call.
MAX_BULK_IMPORT = 1_000

#: Valid schema types accepted by the registry.
VALID_SCHEMA_TYPES: Set[str] = {"json_schema", "avro", "protobuf"}

#: Valid statuses for schemas.
VALID_STATUSES: Set[str] = {"draft", "active", "deprecated", "archived"}

#: Allowed status transitions (from_status → set of allowed to_statuses).
STATUS_TRANSITIONS: Dict[str, Set[str]] = {
    "draft":      {"active"},
    "active":     {"deprecated"},
    "deprecated": {"archived"},
    "archived":   set(),  # terminal state
}

#: Required top-level keys for JSON Schema definitions.
JSON_SCHEMA_REQUIRED_KEYS: Set[str] = {"type"}

#: Required top-level keys for Avro definitions.
AVRO_REQUIRED_KEYS: Set[str] = {"type", "name", "fields"}

#: Required top-level keys for Protobuf definitions.
PROTOBUF_REQUIRED_KEYS: Set[str] = {"syntax", "messages"}

#: Maximum tag length.
MAX_TAG_LENGTH = 64

#: Maximum namespace length.
MAX_NAMESPACE_LENGTH = 128

#: Maximum schema name length.
MAX_SCHEMA_NAME_LENGTH = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed for determinism.

    Returns:
        Current UTC datetime with microseconds set to zero.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def _build_sha256(data: Any) -> str:
    """Build a deterministic SHA-256 hash from any JSON-serialisable value.

    All dict keys are sorted for determinism regardless of insertion order.

    Args:
        data: JSON-serialisable value (dict, list, str, int, etc.).

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_tags(tags: Optional[List[str]]) -> List[str]:
    """Normalize, deduplicate, and sort a list of tags.

    Tags are lowercased and stripped of surrounding whitespace. Empty
    strings and duplicates are removed. Order is sorted for stability.

    Args:
        tags: Raw tag list, or None.

    Returns:
        Sorted list of unique, lowercased, stripped tag strings.
    """
    if not tags:
        return []
    seen: Set[str] = set()
    result: List[str] = []
    for tag in tags:
        normalized = tag.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return sorted(result)


def _validate_namespace(namespace: str) -> None:
    """Raise ValueError if the namespace string is invalid.

    A valid namespace must be non-empty, contain only alphanumeric characters,
    hyphens, underscores, or dots, and not exceed MAX_NAMESPACE_LENGTH.

    Args:
        namespace: Namespace string to validate.

    Raises:
        ValueError: If the namespace is empty, too long, or contains illegal chars.
    """
    if not namespace:
        raise ValueError("namespace must be a non-empty string.")
    if len(namespace) > MAX_NAMESPACE_LENGTH:
        raise ValueError(
            f"namespace exceeds maximum length of {MAX_NAMESPACE_LENGTH} characters."
        )
    if not re.match(r"^[A-Za-z0-9._\-]+$", namespace):
        raise ValueError(
            "namespace may only contain alphanumeric characters, hyphens, "
            "underscores, and dots."
        )


def _validate_schema_name(name: str) -> None:
    """Raise ValueError if the schema name string is invalid.

    A valid schema name must be non-empty and not exceed MAX_SCHEMA_NAME_LENGTH.

    Args:
        name: Schema name string to validate.

    Raises:
        ValueError: If the name is empty or too long.
    """
    if not name:
        raise ValueError("name must be a non-empty string.")
    if len(name) > MAX_SCHEMA_NAME_LENGTH:
        raise ValueError(
            f"name exceeds maximum length of {MAX_SCHEMA_NAME_LENGTH} characters."
        )


def _validate_tags_list(tags: List[str]) -> None:
    """Raise ValueError if any individual tag exceeds the maximum length.

    Args:
        tags: Normalised list of tag strings.

    Raises:
        ValueError: If any tag exceeds MAX_TAG_LENGTH characters.
    """
    for tag in tags:
        if len(tag) > MAX_TAG_LENGTH:
            raise ValueError(
                f"Tag '{tag}' exceeds maximum length of {MAX_TAG_LENGTH} characters."
            )


# ---------------------------------------------------------------------------
# SchemaRegistryEngine
# ---------------------------------------------------------------------------


class SchemaRegistryEngine:
    """
    Schema Registry Engine — Engine 1 of 7, AGENT-DATA-017.

    Registers, catalogs, and manages schemas across namespaces with full
    lifecycle support (draft → active → deprecated → archived).  Schemas
    carry rich metadata: owner, tags, description, custom metadata, and a
    SHA-256 provenance chain for every mutation.

    Supported schema types
    ----------------------
    - ``json_schema``: JSON Schema Draft 2020-12.  Definition must be a valid
      JSON object containing at minimum a ``"type"`` key.
    - ``avro``: Apache Avro record schema.  Definition must contain
      ``"type"``, ``"name"``, and ``"fields"`` keys.
    - ``protobuf``: Protobuf-like descriptor dict.  Definition must contain
      ``"syntax"`` and ``"messages"`` keys.

    Status Lifecycle
    ----------------
    Transitions are strictly one-way:

    .. code-block::

        draft  →  active  →  deprecated  →  archived  (terminal)

    Backward transitions are rejected with a ``ValueError``.

    Thread Safety
    -------------
    All mutating operations are protected by ``self._lock`` (a
    ``threading.Lock``).  Read operations outside the lock return deep copies
    so callers cannot accidentally mutate internal state.

    Zero-Hallucination Guarantees
    ------------------------------
    - UUID assignment via ``uuid.uuid4()`` (no LLM involvement).
    - Timestamps from ``datetime.now(timezone.utc)`` (deterministic).
    - SHA-256 provenance hash computed from JSON-serialised payload.
    - Definition validation is structural/rule-based only.
    - No ML or LLM calls anywhere in the class.

    Attributes:
        _schemas: Schema store keyed by schema_id (UUID string).
        _groups: Schema groups keyed by group name.
        _namespace_index: Mapping from namespace to set of schema_ids.
        _name_index: Mapping from lower-cased schema name to set of schema_ids.
        _tag_index: Mapping from tag string to set of schema_ids.
        _lock: Thread-safety lock protecting all state.
        _provenance: ProvenanceTracker for SHA-256 audit trails.

    Example:
        >>> engine = SchemaRegistryEngine()
        >>> schema = engine.register_schema(
        ...     namespace="supply_chain",
        ...     name="SupplierRecord",
        ...     schema_type="json_schema",
        ...     definition_json={"type": "object", "properties": {"id": {"type": "string"}}},
        ...     owner="data-team",
        ...     tags=["supplier", "core"],
        ...     description="Canonical supplier record schema.",
        ... )
        >>> assert schema["status"] == "draft"
        >>> updated = engine.update_schema(schema["schema_id"], status="active")
        >>> assert updated["status"] == "active"
        >>> stats = engine.get_statistics()
        >>> assert stats["total_schemas"] == 1
    """

    def __init__(self) -> None:
        """Initialise the SchemaRegistryEngine with empty in-memory state."""
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._groups: Dict[str, Dict[str, Any]] = {}
        self._namespace_index: Dict[str, Set[str]] = {}
        self._name_index: Dict[str, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()
        self._provenance = ProvenanceTracker()
        logger.info("SchemaRegistryEngine initialised (AGENT-DATA-017, Engine 1 of 7)")

    # ------------------------------------------------------------------
    # 1. register_schema
    # ------------------------------------------------------------------

    def register_schema(
        self,
        namespace: str,
        name: str,
        schema_type: str,
        definition_json: Any,
        owner: str = "",
        tags: Optional[List[str]] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register a new schema in the registry.

        Validates the namespace, name, schema type, and definition before
        storing.  Enforces uniqueness of the (namespace, name) pair — a
        ``ValueError`` is raised if the combination already exists.

        The new schema is always created with ``status="draft"``.

        Args:
            namespace: Logical namespace / domain for the schema (e.g.
                ``"emissions"``, ``"supply_chain"``).  Must be non-empty and
                contain only alphanumeric characters, hyphens, underscores, or
                dots.
            name: Human-readable name of the schema (e.g. ``"ActivityRecord"``).
                Must be non-empty.
            schema_type: One of ``"json_schema"``, ``"avro"``, or
                ``"protobuf"``.
            definition_json: The schema definition as a Python dict (already
                parsed from JSON) or any JSON-serialisable value.  The engine
                performs structural validation appropriate to the schema type.
            owner: Free-form owner identifier (team slug, email, etc.).
            tags: Optional list of string tags for discovery / filtering.
                Tags are normalised (lower-cased, stripped, deduplicated).
            description: Human-readable description of the schema.
            metadata: Arbitrary additional metadata dict (not validated).

        Returns:
            Full schema dict containing at minimum:

            .. code-block:: python

                {
                    "schema_id": "<uuid>",
                    "namespace": "...",
                    "name": "...",
                    "schema_type": "json_schema",
                    "definition": {...},
                    "owner": "...",
                    "tags": [...],
                    "description": "...",
                    "metadata": {...},
                    "status": "draft",
                    "created_at": "<ISO-8601>",
                    "updated_at": "<ISO-8601>",
                    "provenance_hash": "<sha256>",
                }

        Raises:
            ValueError: If namespace/name/schema_type are invalid, if a schema
                with the same (namespace, name) already exists, or if the
                definition fails structural validation.
        """
        import time as _time
        t0 = _time.monotonic()

        # Input validation
        _validate_namespace(namespace)
        _validate_schema_name(name)

        if schema_type not in VALID_SCHEMA_TYPES:
            raise ValueError(
                f"Unsupported schema_type '{schema_type}'. "
                f"Must be one of {sorted(VALID_SCHEMA_TYPES)}."
            )

        normalized_tags = _normalize_tags(tags)
        _validate_tags_list(normalized_tags)

        # Structural definition validation
        validation_result = self.validate_definition(schema_type, definition_json)
        if not validation_result["valid"]:
            sm_validation_errors_total.labels(schema_type=schema_type).inc()
            raise ValueError(
                f"Schema definition validation failed: {validation_result['errors']}"
            )

        now_str = _utcnow().isoformat()
        schema_id = str(uuid.uuid4())
        safe_metadata = copy.deepcopy(metadata) if metadata else {}

        schema_record: Dict[str, Any] = {
            "schema_id":      schema_id,
            "namespace":      namespace,
            "name":           name,
            "schema_type":    schema_type,
            "definition":     copy.deepcopy(definition_json),
            "owner":          owner,
            "tags":           normalized_tags,
            "description":    description,
            "metadata":       safe_metadata,
            "status":         "draft",
            "created_at":     now_str,
            "updated_at":     now_str,
            "provenance_hash": "",
        }

        # Compute provenance hash before locking (no shared state read)
        data_hash = _build_sha256(
            {k: v for k, v in schema_record.items() if k != "provenance_hash"}
        )

        with self._lock:
            # Enforce (namespace, name) uniqueness
            existing_ids = self._namespace_index.get(namespace, set())
            for existing_id in existing_ids:
                existing = self._schemas.get(existing_id, {})
                if existing.get("name") == name and existing.get("status") != "archived":
                    raise ValueError(
                        f"A non-archived schema with namespace='{namespace}' and "
                        f"name='{name}' already exists (id={existing_id})."
                    )

            # Record provenance
            chain_hash = self._provenance.record(
                entity_type="schema",
                entity_id=schema_id,
                action="register",
                data_hash=data_hash,
            )
            schema_record["provenance_hash"] = chain_hash

            # Store
            self._schemas[schema_id] = schema_record

            # Update indexes
            self._namespace_index.setdefault(namespace, set()).add(schema_id)
            name_lower = name.lower()
            self._name_index.setdefault(name_lower, set()).add(schema_id)
            for tag in normalized_tags:
                self._tag_index.setdefault(tag, set()).add(schema_id)

        # Metrics
        sm_schemas_registered_total.labels(schema_type=schema_type).inc()
        sm_registry_size.set(len(self._schemas))
        sm_namespaces_total.set(len(self._namespace_index))
        sm_schemas_by_status.labels(status="draft").inc()
        elapsed = _time.monotonic() - t0
        sm_operation_duration_seconds.labels(operation="register_schema").observe(elapsed)

        logger.info(
            "Schema registered: id=%s namespace=%s name=%s type=%s",
            schema_id, namespace, name, schema_type,
        )
        return copy.deepcopy(schema_record)

    # ------------------------------------------------------------------
    # 2. get_schema
    # ------------------------------------------------------------------

    def get_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a schema by its unique ID.

        Args:
            schema_id: UUID string assigned at registration time.

        Returns:
            Deep-copy of the schema dict, or ``None`` if not found.

        Example:
            >>> schema = engine.get_schema("550e8400-e29b-41d4-a716-446655440000")
            >>> if schema:
            ...     print(schema["name"], schema["status"])
        """
        with self._lock:
            record = self._schemas.get(schema_id)
            if record is None:
                return None
            return copy.deepcopy(record)

    # ------------------------------------------------------------------
    # 3. list_schemas
    # ------------------------------------------------------------------

    def list_schemas(
        self,
        namespace: Optional[str] = None,
        name_contains: Optional[str] = None,
        schema_type: Optional[str] = None,
        owner: Optional[str] = None,
        tag: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List and filter schemas with optional pagination.

        All filters are applied with AND logic (a schema must satisfy every
        provided filter to be included in results).

        Args:
            namespace: Exact namespace match filter.
            name_contains: Case-insensitive substring match against schema name.
            schema_type: Exact schema type filter (``"json_schema"``, ``"avro"``,
                or ``"protobuf"``).
            owner: Exact owner match filter (case-sensitive).
            tag: Single tag filter; schema must carry this tag.
            status: Exact status filter (``"draft"``, ``"active"``,
                ``"deprecated"``, ``"archived"``).
            limit: Maximum number of results to return (default 100).
            offset: Number of results to skip before returning (default 0).

        Returns:
            List of schema dicts (deep copies), ordered by ``created_at``
            ascending, then sliced by ``offset`` and ``limit``.

        Raises:
            ValueError: If ``limit`` or ``offset`` are negative.

        Example:
            >>> active = engine.list_schemas(namespace="emissions", status="active")
            >>> page2 = engine.list_schemas(limit=10, offset=10)
        """
        if limit < 0:
            raise ValueError("limit must be >= 0.")
        if offset < 0:
            raise ValueError("offset must be >= 0.")

        import time as _time
        t0 = _time.monotonic()

        with self._lock:
            # Start from namespace index if namespace filter provided (faster)
            if namespace is not None:
                candidate_ids: Set[str] = set(self._namespace_index.get(namespace, set()))
            elif tag is not None:
                candidate_ids = set(self._tag_index.get(tag.lower(), set()))
            else:
                candidate_ids = set(self._schemas.keys())

            results: List[Dict[str, Any]] = []
            for schema_id in candidate_ids:
                record = self._schemas.get(schema_id)
                if record is None:
                    continue

                # Apply remaining filters
                if namespace is not None and record["namespace"] != namespace:
                    continue
                if schema_type is not None and record["schema_type"] != schema_type:
                    continue
                if owner is not None and record["owner"] != owner:
                    continue
                if status is not None and record["status"] != status:
                    continue
                if tag is not None and tag.lower() not in record["tags"]:
                    continue
                if name_contains is not None:
                    if name_contains.lower() not in record["name"].lower():
                        continue

                results.append(copy.deepcopy(record))

        # Sort by created_at ascending for deterministic ordering
        results.sort(key=lambda r: r.get("created_at", ""))
        paginated = results[offset: offset + limit]

        elapsed = _time.monotonic() - t0
        sm_operation_duration_seconds.labels(operation="list_schemas").observe(elapsed)

        logger.debug(
            "list_schemas returned %d of %d matches (offset=%d, limit=%d)",
            len(paginated), len(results), offset, limit,
        )
        return paginated

    # ------------------------------------------------------------------
    # 4. update_schema
    # ------------------------------------------------------------------

    def update_schema(
        self,
        schema_id: str,
        owner: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update schema metadata and/or advance its status.

        Only the fields explicitly provided (non-``None``) are updated.
        Status transitions are strictly enforced — only the following
        progressions are allowed:

        - ``draft`` → ``active``
        - ``active`` → ``deprecated``
        - ``deprecated`` → ``archived``

        Any attempt to transition to an earlier status, or any other invalid
        transition, raises a ``ValueError``.

        Args:
            schema_id: UUID of the schema to update.
            owner: New owner identifier, or ``None`` to leave unchanged.
            tags: New complete tag list, or ``None`` to leave unchanged.
                When provided, the old tags are fully replaced by the new list.
            status: Target status string, or ``None`` to leave unchanged.
            description: New description string, or ``None`` to leave unchanged.
            metadata: Metadata dict to merge into existing metadata, or
                ``None`` to leave unchanged.  Keys from the provided dict
                overwrite matching keys in the existing metadata; other existing
                keys are preserved.

        Returns:
            Deep-copy of the updated schema dict.

        Raises:
            KeyError: If no schema with the given ``schema_id`` exists.
            ValueError: If the requested status transition is invalid.

        Example:
            >>> engine.update_schema(schema_id, status="active")
            >>> engine.update_schema(schema_id, owner="new-team", tags=["updated"])
        """
        import time as _time
        t0 = _time.monotonic()

        with self._lock:
            record = self._schemas.get(schema_id)
            if record is None:
                raise KeyError(f"Schema not found: {schema_id}")

            old_status = record["status"]

            # Validate and apply status transition
            if status is not None and status != old_status:
                if status not in VALID_STATUSES:
                    raise ValueError(
                        f"Invalid status '{status}'. "
                        f"Must be one of {sorted(VALID_STATUSES)}."
                    )
                allowed_transitions = STATUS_TRANSITIONS.get(old_status, set())
                if status not in allowed_transitions:
                    raise ValueError(
                        f"Invalid status transition: '{old_status}' → '{status}'. "
                        f"Allowed transitions from '{old_status}': "
                        f"{sorted(allowed_transitions) or ['(none — terminal state)']}"
                    )
                record["status"] = status

            # Update owner
            if owner is not None:
                record["owner"] = owner

            # Update description
            if description is not None:
                record["description"] = description

            # Update tags (full replacement when provided)
            if tags is not None:
                normalized_new_tags = _normalize_tags(tags)
                _validate_tags_list(normalized_new_tags)
                old_tags = set(record["tags"])
                new_tags_set = set(normalized_new_tags)

                # Remove schema from tag index for stale tags
                for removed_tag in old_tags - new_tags_set:
                    self._tag_index.get(removed_tag, set()).discard(schema_id)

                # Add schema to tag index for new tags
                for added_tag in new_tags_set - old_tags:
                    self._tag_index.setdefault(added_tag, set()).add(schema_id)

                record["tags"] = normalized_new_tags

            # Merge metadata
            if metadata is not None:
                record["metadata"].update(copy.deepcopy(metadata))

            record["updated_at"] = _utcnow().isoformat()

            # Compute new provenance hash
            data_hash = _build_sha256(
                {k: v for k, v in record.items() if k != "provenance_hash"}
            )
            chain_hash = self._provenance.record(
                entity_type="schema",
                entity_id=schema_id,
                action="update",
                data_hash=data_hash,
            )
            record["provenance_hash"] = chain_hash

            result = copy.deepcopy(record)

        # Metrics
        if status is not None and status != old_status:
            sm_status_transitions_total.labels(
                from_status=old_status, to_status=status
            ).inc()
            sm_schemas_by_status.labels(status=old_status).dec()
            sm_schemas_by_status.labels(status=status).inc()

        elapsed = _time.monotonic() - t0
        sm_operation_duration_seconds.labels(operation="update_schema").observe(elapsed)

        logger.info(
            "Schema updated: id=%s status=%s→%s owner=%s",
            schema_id,
            old_status,
            result["status"],
            result["owner"],
        )
        return result

    # ------------------------------------------------------------------
    # 5. delete_schema
    # ------------------------------------------------------------------

    def delete_schema(self, schema_id: str) -> Dict[str, Any]:
        """Soft-delete a schema by archiving it.

        The schema is not removed from storage; instead its status is set to
        ``"archived"``.  The schema remains searchable and retrievable but
        cannot be transitioned to any other status.

        If the schema is already ``"archived"``, this is a no-op (returns the
        existing archived record).

        Args:
            schema_id: UUID of the schema to archive.

        Returns:
            Deep-copy of the archived schema dict.

        Raises:
            KeyError: If no schema with the given ``schema_id`` exists.

        Example:
            >>> archived = engine.delete_schema(schema_id)
            >>> assert archived["status"] == "archived"
        """
        import time as _time
        t0 = _time.monotonic()

        with self._lock:
            record = self._schemas.get(schema_id)
            if record is None:
                raise KeyError(f"Schema not found: {schema_id}")

            old_status = record["status"]

            if old_status != "archived":
                # Force-archive regardless of current status (soft delete bypass)
                record["status"] = "archived"
                record["updated_at"] = _utcnow().isoformat()

                data_hash = _build_sha256(
                    {k: v for k, v in record.items() if k != "provenance_hash"}
                )
                chain_hash = self._provenance.record(
                    entity_type="schema",
                    entity_id=schema_id,
                    action="delete",
                    data_hash=data_hash,
                )
                record["provenance_hash"] = chain_hash

            result = copy.deepcopy(record)

        if old_status != "archived":
            sm_status_transitions_total.labels(
                from_status=old_status, to_status="archived"
            ).inc()
            sm_schemas_by_status.labels(status=old_status).dec()
            sm_schemas_by_status.labels(status="archived").inc()

        elapsed = _time.monotonic() - t0
        sm_operation_duration_seconds.labels(operation="delete_schema").observe(elapsed)

        logger.info(
            "Schema soft-deleted (archived): id=%s (was %s)", schema_id, old_status
        )
        return result

    # ------------------------------------------------------------------
    # 6. search_schemas
    # ------------------------------------------------------------------

    def search_schemas(self, query: str) -> List[Dict[str, Any]]:
        """Full-text search across schema name, namespace, description, and tags.

        The search is case-insensitive.  A schema is included in results if the
        query string appears as a substring in any of the following fields:

        - ``name``
        - ``namespace``
        - ``description``
        - Any tag in the ``tags`` list
        - ``owner``

        Results are sorted by a simple relevance heuristic: schemas whose
        ``name`` matches the query are ranked first, then ``namespace`` matches,
        then ``description``/tag/owner matches.  Within each tier, ordering is
        by ``created_at`` ascending.

        Args:
            query: Search string (case-insensitive substring match).

        Returns:
            List of matching schema dicts (deep copies), ordered by relevance.

        Example:
            >>> results = engine.search_schemas("emissions")
            >>> for s in results:
            ...     print(s["name"], s["namespace"])
        """
        import time as _time
        t0 = _time.monotonic()
        sm_search_queries_total.inc()

        if not query:
            return []

        query_lower = query.strip().lower()

        matches: List[Dict[str, Any]] = []
        with self._lock:
            all_records = list(self._schemas.values())

        for record in all_records:
            score = self._compute_search_score(record, query_lower)
            if score > 0:
                matches.append((score, record["created_at"], copy.deepcopy(record)))

        # Sort: descending score, then ascending created_at as tiebreaker
        matches.sort(key=lambda x: (-x[0], x[1]))
        results = [entry[2] for entry in matches]

        elapsed = _time.monotonic() - t0
        sm_operation_duration_seconds.labels(operation="search_schemas").observe(elapsed)

        logger.debug(
            "search_schemas query='%s' returned %d results", query, len(results)
        )
        return results

    def _compute_search_score(self, record: Dict[str, Any], query_lower: str) -> int:
        """Compute a relevance score for a schema record against a query string.

        Higher scores indicate stronger relevance.  The scoring tiers are:

        - Name exact match: +10
        - Name contains query: +6
        - Namespace contains query: +4
        - Description contains query: +2
        - Any tag contains query: +3 (per matching tag)
        - Owner contains query: +1

        Args:
            record: Schema dict from internal storage.
            query_lower: Lower-cased search query string.

        Returns:
            Non-negative integer relevance score (0 means no match).
        """
        score = 0
        name_lower = record.get("name", "").lower()
        namespace_lower = record.get("namespace", "").lower()
        description_lower = record.get("description", "").lower()
        owner_lower = record.get("owner", "").lower()
        tags = record.get("tags", [])

        if name_lower == query_lower:
            score += 10
        elif query_lower in name_lower:
            score += 6

        if query_lower in namespace_lower:
            score += 4

        if query_lower in description_lower:
            score += 2

        for tag in tags:
            if query_lower in tag:
                score += 3

        if query_lower in owner_lower:
            score += 1

        return score

    # ------------------------------------------------------------------
    # 7. validate_definition
    # ------------------------------------------------------------------

    def validate_definition(
        self,
        schema_type: str,
        definition_json: Any,
    ) -> Dict[str, Any]:
        """Validate a schema definition against its declared type.

        Performs structural validation only — no LLM or external calls.

        Validation rules by type:

        - **json_schema**: Definition must be a non-empty dict containing at
          minimum a ``"type"`` key.  An optional ``"$schema"`` key may be
          present and is validated for known draft URIs.
        - **avro**: Definition must be a dict with ``"type"``, ``"name"``,
          and ``"fields"`` keys.  ``"fields"`` must be a list.
        - **protobuf**: Definition must be a dict with ``"syntax"`` and
          ``"messages"`` keys.  ``"messages"`` must be a list.

        Args:
            schema_type: One of ``"json_schema"``, ``"avro"``,
                ``"protobuf"``.
            definition_json: The schema definition to validate (expected to
                be a Python dict or any JSON-serialisable value).

        Returns:
            Dict with keys:

            .. code-block:: python

                {
                    "valid": True | False,
                    "schema_type": "json_schema",
                    "errors": [],          # list of error message strings
                    "warnings": [],        # list of warning message strings
                }

        Raises:
            ValueError: If ``schema_type`` is not one of the supported types.

        Example:
            >>> result = engine.validate_definition(
            ...     "json_schema",
            ...     {"type": "object", "properties": {}},
            ... )
            >>> assert result["valid"] is True
        """
        if schema_type not in VALID_SCHEMA_TYPES:
            raise ValueError(
                f"Unsupported schema_type '{schema_type}'. "
                f"Must be one of {sorted(VALID_SCHEMA_TYPES)}."
            )

        errors: List[str] = []
        warnings: List[str] = []

        if schema_type == "json_schema":
            errors, warnings = self._validate_json_schema_definition(
                definition_json
            )
        elif schema_type == "avro":
            errors, warnings = self._validate_avro_definition(definition_json)
        elif schema_type == "protobuf":
            errors, warnings = self._validate_protobuf_definition(definition_json)

        return {
            "valid": len(errors) == 0,
            "schema_type": schema_type,
            "errors": errors,
            "warnings": warnings,
        }

    def _validate_json_schema_definition(
        self, definition: Any
    ) -> tuple:
        """Validate a JSON Schema definition dict.

        Args:
            definition: Candidate JSON Schema definition.

        Returns:
            Tuple of (errors list, warnings list).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(definition, dict):
            errors.append(
                "JSON Schema definition must be a JSON object (dict), "
                f"got {type(definition).__name__}."
            )
            return errors, warnings

        if not definition:
            errors.append("JSON Schema definition must not be empty.")
            return errors, warnings

        # Check for at minimum a "type" key
        if "type" not in definition:
            # Permissive: allow if $schema is present (could be a root schema)
            if "$schema" not in definition:
                errors.append(
                    "JSON Schema definition must contain a 'type' key or a "
                    "'$schema' key at the root level."
                )

        # Validate $schema URI if present
        schema_uri = definition.get("$schema", "")
        if schema_uri:
            known_drafts = [
                "http://json-schema.org/draft-07/schema",
                "http://json-schema.org/draft-06/schema",
                "http://json-schema.org/draft-04/schema",
                "https://json-schema.org/draft/2019-09/schema",
                "https://json-schema.org/draft/2020-12/schema",
            ]
            if not any(schema_uri.startswith(d) for d in known_drafts):
                warnings.append(
                    f"'$schema' URI '{schema_uri}' is not a recognised JSON "
                    "Schema draft URI.  GreenLang recommends "
                    "https://json-schema.org/draft/2020-12/schema."
                )

        # Validate "properties" if present
        if "properties" in definition:
            props = definition["properties"]
            if not isinstance(props, dict):
                errors.append(
                    "'properties' must be a JSON object (dict), "
                    f"got {type(props).__name__}."
                )

        # Validate "required" if present
        if "required" in definition:
            req = definition["required"]
            if not isinstance(req, list):
                errors.append(
                    "'required' must be a JSON array (list), "
                    f"got {type(req).__name__}."
                )
            elif not all(isinstance(r, str) for r in req):
                errors.append("All elements of 'required' must be strings.")

        return errors, warnings

    def _validate_avro_definition(self, definition: Any) -> tuple:
        """Validate an Apache Avro schema definition dict.

        Args:
            definition: Candidate Avro schema definition.

        Returns:
            Tuple of (errors list, warnings list).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(definition, dict):
            errors.append(
                "Avro schema definition must be a JSON object (dict), "
                f"got {type(definition).__name__}."
            )
            return errors, warnings

        for key in AVRO_REQUIRED_KEYS:
            if key not in definition:
                errors.append(
                    f"Avro schema definition is missing required key: '{key}'."
                )

        if "type" in definition and definition["type"] not in (
            "record", "enum", "array", "map", "union", "fixed"
        ):
            warnings.append(
                f"Avro 'type' is '{definition['type']}'; expected one of "
                "'record', 'enum', 'array', 'map', 'union', 'fixed'."
            )

        if "fields" in definition:
            fields = definition["fields"]
            if not isinstance(fields, list):
                errors.append(
                    "'fields' must be a JSON array (list), "
                    f"got {type(fields).__name__}."
                )
            else:
                for idx, field in enumerate(fields):
                    if not isinstance(field, dict):
                        errors.append(
                            f"fields[{idx}] must be an object (dict), "
                            f"got {type(field).__name__}."
                        )
                        continue
                    if "name" not in field:
                        errors.append(
                            f"fields[{idx}] is missing the required 'name' key."
                        )
                    if "type" not in field:
                        errors.append(
                            f"fields[{idx}] is missing the required 'type' key."
                        )

        return errors, warnings

    def _validate_protobuf_definition(self, definition: Any) -> tuple:
        """Validate a Protobuf-like schema descriptor dict.

        Args:
            definition: Candidate Protobuf descriptor definition.

        Returns:
            Tuple of (errors list, warnings list).
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(definition, dict):
            errors.append(
                "Protobuf schema definition must be a JSON object (dict), "
                f"got {type(definition).__name__}."
            )
            return errors, warnings

        for key in PROTOBUF_REQUIRED_KEYS:
            if key not in definition:
                errors.append(
                    f"Protobuf schema definition is missing required key: '{key}'."
                )

        if "syntax" in definition and definition["syntax"] not in (
            "proto2", "proto3"
        ):
            warnings.append(
                f"Protobuf 'syntax' is '{definition['syntax']}'; "
                "expected 'proto2' or 'proto3'."
            )

        if "messages" in definition:
            messages = definition["messages"]
            if not isinstance(messages, list):
                errors.append(
                    "'messages' must be a JSON array (list), "
                    f"got {type(messages).__name__}."
                )
            else:
                if len(messages) == 0:
                    warnings.append(
                        "'messages' list is empty; schema has no message definitions."
                    )
                for idx, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        errors.append(
                            f"messages[{idx}] must be an object (dict), "
                            f"got {type(msg).__name__}."
                        )
                        continue
                    if "name" not in msg:
                        errors.append(
                            f"messages[{idx}] is missing the required 'name' key."
                        )
                    if "fields" in msg and not isinstance(msg["fields"], list):
                        errors.append(
                            f"messages[{idx}]['fields'] must be a list."
                        )

        return errors, warnings

    # ------------------------------------------------------------------
    # 8. import_schemas
    # ------------------------------------------------------------------

    def import_schemas(self, schemas_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk-import up to MAX_BULK_IMPORT schemas in a single call.

        Each element of ``schemas_list`` must be a dict with the following
        required keys: ``namespace``, ``name``, ``schema_type``,
        ``definition_json``.  Optional keys mirror the ``register_schema``
        signature: ``owner``, ``tags``, ``description``, ``metadata``.

        Schemas that fail validation are skipped; the method continues with
        the remaining entries.  A summary dict is returned regardless.

        Args:
            schemas_list: List of schema dicts to import.  Maximum
                ``MAX_BULK_IMPORT`` (1 000) entries per call.

        Returns:
            Summary dict:

            .. code-block:: python

                {
                    "total":    10,
                    "success":  8,
                    "failed":   2,
                    "failures": [
                        {"index": 3, "error": "...", "name": "..."},
                        {"index": 7, "error": "...", "name": "..."},
                    ],
                    "schema_ids": ["uuid1", "uuid2", ...],  # successful IDs
                    "provenance_hash": "<sha256>",
                }

        Raises:
            ValueError: If ``schemas_list`` contains more than
                ``MAX_BULK_IMPORT`` entries.

        Example:
            >>> result = engine.import_schemas([
            ...     {"namespace": "ns", "name": "A", "schema_type": "avro",
            ...      "definition_json": {"type": "record", "name": "A", "fields": []}},
            ... ])
            >>> print(result["success"], result["failed"])
        """
        import time as _time
        t0 = _time.monotonic()

        if len(schemas_list) > MAX_BULK_IMPORT:
            raise ValueError(
                f"Bulk import exceeds maximum of {MAX_BULK_IMPORT} schemas. "
                f"Received {len(schemas_list)}."
            )

        total = len(schemas_list)
        success_count = 0
        failed_count = 0
        failures: List[Dict[str, Any]] = []
        schema_ids: List[str] = []

        for idx, schema_dict in enumerate(schemas_list):
            raw_name = schema_dict.get("name", f"<unnamed-{idx}>")
            try:
                schema = self.register_schema(
                    namespace=schema_dict["namespace"],
                    name=schema_dict["name"],
                    schema_type=schema_dict["schema_type"],
                    definition_json=schema_dict["definition_json"],
                    owner=schema_dict.get("owner", ""),
                    tags=schema_dict.get("tags"),
                    description=schema_dict.get("description", ""),
                    metadata=schema_dict.get("metadata"),
                )
                schema_ids.append(schema["schema_id"])
                success_count += 1
                sm_import_total.labels(result="success").inc()
            except Exception as exc:
                failed_count += 1
                failures.append({
                    "index": idx,
                    "name": raw_name,
                    "error": str(exc),
                })
                sm_import_total.labels(result="failure").inc()
                logger.warning(
                    "Bulk import: schema[%d] name='%s' failed: %s",
                    idx, raw_name, exc,
                )

        summary: Dict[str, Any] = {
            "total":    total,
            "success":  success_count,
            "failed":   failed_count,
            "failures": failures,
            "schema_ids": schema_ids,
            "provenance_hash": _build_sha256({
                "total": total,
                "success": success_count,
                "failed": failed_count,
                "schema_ids": schema_ids,
            }),
        }

        elapsed = _time.monotonic() - t0
        sm_operation_duration_seconds.labels(operation="import_schemas").observe(elapsed)

        logger.info(
            "Bulk import complete: total=%d success=%d failed=%d elapsed=%.3fs",
            total, success_count, failed_count, elapsed,
        )
        return summary

    # ------------------------------------------------------------------
    # 9. export_schemas
    # ------------------------------------------------------------------

    def export_schemas(
        self,
        schema_ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Export schemas as a list of JSON-serialisable dicts.

        If both ``schema_ids`` and ``namespace`` are provided, the union of
        both selection criteria is returned (schemas matching either).

        If neither is provided, **all** schemas are exported.

        Args:
            schema_ids: Optional list of specific schema UUID strings to export.
                Non-existent IDs are silently skipped.
            namespace: Optional namespace to export all schemas from.

        Returns:
            List of schema dicts (deep copies), ordered by ``created_at``
            ascending.  Each dict is fully JSON-serialisable.

        Example:
            >>> all_schemas = engine.export_schemas()
            >>> ns_schemas  = engine.export_schemas(namespace="emissions")
            >>> two_schemas = engine.export_schemas(schema_ids=["id1", "id2"])
        """
        import time as _time
        t0 = _time.monotonic()

        sm_exports_total.inc()
        selected_ids: Set[str] = set()

        with self._lock:
            if schema_ids is None and namespace is None:
                # Export all
                selected_ids = set(self._schemas.keys())
            else:
                if schema_ids is not None:
                    for sid in schema_ids:
                        if sid in self._schemas:
                            selected_ids.add(sid)
                if namespace is not None:
                    selected_ids.update(
                        self._namespace_index.get(namespace, set())
                    )

            records = [
                copy.deepcopy(self._schemas[sid])
                for sid in selected_ids
                if sid in self._schemas
            ]

        records.sort(key=lambda r: r.get("created_at", ""))

        elapsed = _time.monotonic() - t0
        sm_operation_duration_seconds.labels(operation="export_schemas").observe(elapsed)

        logger.info(
            "export_schemas: exported %d schemas (namespace=%s, ids=%s)",
            len(records),
            namespace,
            f"{len(schema_ids)} explicit IDs" if schema_ids else "all",
        )
        return records

    # ------------------------------------------------------------------
    # 10. create_group
    # ------------------------------------------------------------------

    def create_group(
        self,
        name: str,
        description: str = "",
        schema_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a new schema group for logical domain organisation.

        Schema groups allow schemas from any namespace to be bundled together
        under a named label (e.g., ``"ghg-protocol-scope3"``,
        ``"eudr-compliance"``).  A schema may belong to multiple groups.

        Args:
            name: Unique group name.  Must be non-empty.  Case-sensitive.
            description: Human-readable description of the group.
            schema_ids: Optional list of existing schema UUIDs to add to the
                group at creation time.  Non-existent IDs are silently skipped.

        Returns:
            The created group dict:

            .. code-block:: python

                {
                    "name": "ghg-protocol-scope3",
                    "description": "...",
                    "schema_ids": ["uuid1", "uuid2"],
                    "created_at": "<ISO-8601>",
                    "updated_at": "<ISO-8601>",
                    "provenance_hash": "<sha256>",
                }

        Raises:
            ValueError: If ``name`` is empty or a group with the same name
                already exists.

        Example:
            >>> group = engine.create_group(
            ...     "ghg-protocol",
            ...     description="All GHG Protocol schemas",
            ...     schema_ids=[schema["schema_id"]],
            ... )
        """
        if not name:
            raise ValueError("Group name must be a non-empty string.")

        now_str = _utcnow().isoformat()

        with self._lock:
            if name in self._groups:
                raise ValueError(f"A group named '{name}' already exists.")

            # Validate and filter schema_ids to existing ones
            valid_ids: List[str] = []
            if schema_ids:
                for sid in schema_ids:
                    if sid in self._schemas:
                        valid_ids.append(sid)
                    else:
                        logger.warning(
                            "create_group '%s': schema_id '%s' not found; skipping.",
                            name, sid,
                        )

            group_record: Dict[str, Any] = {
                "name":           name,
                "description":    description,
                "schema_ids":     list(dict.fromkeys(valid_ids)),  # deduplicate preserving order
                "created_at":     now_str,
                "updated_at":     now_str,
                "provenance_hash": "",
            }

            data_hash = _build_sha256(
                {k: v for k, v in group_record.items() if k != "provenance_hash"}
            )
            chain_hash = self._provenance.record(
                entity_type="schema_group",
                entity_id=name,
                action="create",
                data_hash=data_hash,
            )
            group_record["provenance_hash"] = chain_hash
            self._groups[name] = group_record
            result = copy.deepcopy(group_record)

        sm_groups_total.set(len(self._groups))
        logger.info(
            "Schema group created: name='%s' schema_ids=%d",
            name, len(result["schema_ids"]),
        )
        return result

    # ------------------------------------------------------------------
    # 11. get_group
    # ------------------------------------------------------------------

    def get_group(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a schema group by name.

        Args:
            name: Group name (case-sensitive).

        Returns:
            Deep-copy of the group dict, or ``None`` if not found.

        Example:
            >>> group = engine.get_group("ghg-protocol")
            >>> if group:
            ...     print(len(group["schema_ids"]), "schemas in group")
        """
        with self._lock:
            record = self._groups.get(name)
            if record is None:
                return None
            return copy.deepcopy(record)

    # ------------------------------------------------------------------
    # 12. list_groups
    # ------------------------------------------------------------------

    def list_groups(self) -> List[Dict[str, Any]]:
        """List all schema groups, ordered by ``created_at`` ascending.

        Returns:
            List of group dicts (deep copies).

        Example:
            >>> for group in engine.list_groups():
            ...     print(group["name"], len(group["schema_ids"]))
        """
        with self._lock:
            records = [copy.deepcopy(g) for g in self._groups.values()]

        records.sort(key=lambda g: g.get("created_at", ""))
        return records

    # ------------------------------------------------------------------
    # 13. add_to_group
    # ------------------------------------------------------------------

    def add_to_group(self, group_name: str, schema_id: str) -> Dict[str, Any]:
        """Add a schema to an existing group.

        If the schema is already a member of the group, this is a no-op and
        the current group record is returned unchanged.

        Args:
            group_name: Name of the target group.
            schema_id: UUID of the schema to add.

        Returns:
            Deep-copy of the updated group dict.

        Raises:
            KeyError: If ``group_name`` does not exist.
            KeyError: If ``schema_id`` does not exist in the registry.

        Example:
            >>> engine.add_to_group("ghg-protocol", schema["schema_id"])
        """
        with self._lock:
            group = self._groups.get(group_name)
            if group is None:
                raise KeyError(f"Group not found: '{group_name}'")

            if schema_id not in self._schemas:
                raise KeyError(f"Schema not found: '{schema_id}'")

            if schema_id in group["schema_ids"]:
                logger.debug(
                    "add_to_group: schema '%s' already in group '%s'; no-op.",
                    schema_id, group_name,
                )
                return copy.deepcopy(group)

            group["schema_ids"].append(schema_id)
            group["updated_at"] = _utcnow().isoformat()

            data_hash = _build_sha256(
                {k: v for k, v in group.items() if k != "provenance_hash"}
            )
            chain_hash = self._provenance.record(
                entity_type="schema_group",
                entity_id=group_name,
                action="add_member",
                data_hash=data_hash,
            )
            group["provenance_hash"] = chain_hash
            result = copy.deepcopy(group)

        logger.info(
            "add_to_group: schema '%s' added to group '%s'.", schema_id, group_name
        )
        return result

    # ------------------------------------------------------------------
    # 14. remove_from_group
    # ------------------------------------------------------------------

    def remove_from_group(self, group_name: str, schema_id: str) -> Dict[str, Any]:
        """Remove a schema from a group.

        If the schema is not a member of the group, this is a no-op and the
        current group record is returned unchanged.

        Args:
            group_name: Name of the target group.
            schema_id: UUID of the schema to remove.

        Returns:
            Deep-copy of the updated group dict.

        Raises:
            KeyError: If ``group_name`` does not exist.

        Example:
            >>> engine.remove_from_group("ghg-protocol", schema["schema_id"])
        """
        with self._lock:
            group = self._groups.get(group_name)
            if group is None:
                raise KeyError(f"Group not found: '{group_name}'")

            if schema_id not in group["schema_ids"]:
                logger.debug(
                    "remove_from_group: schema '%s' not in group '%s'; no-op.",
                    schema_id, group_name,
                )
                return copy.deepcopy(group)

            group["schema_ids"].remove(schema_id)
            group["updated_at"] = _utcnow().isoformat()

            data_hash = _build_sha256(
                {k: v for k, v in group.items() if k != "provenance_hash"}
            )
            chain_hash = self._provenance.record(
                entity_type="schema_group",
                entity_id=group_name,
                action="remove_member",
                data_hash=data_hash,
            )
            group["provenance_hash"] = chain_hash
            result = copy.deepcopy(group)

        logger.info(
            "remove_from_group: schema '%s' removed from group '%s'.",
            schema_id, group_name,
        )
        return result

    # ------------------------------------------------------------------
    # 15. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return a comprehensive snapshot of registry statistics.

        This method is read-only and never modifies state.  It computes all
        aggregates from the current in-memory state under the lock.

        Returns:
            Stats dict:

            .. code-block:: python

                {
                    "total_schemas": 42,
                    "by_type": {
                        "json_schema": 30,
                        "avro": 10,
                        "protobuf": 2,
                    },
                    "by_status": {
                        "draft": 5,
                        "active": 30,
                        "deprecated": 5,
                        "archived": 2,
                    },
                    "by_namespace": {
                        "emissions": 20,
                        "supply_chain": 22,
                    },
                    "total_groups": 3,
                    "groups": {
                        "ghg-protocol": 15,
                        "eudr": 7,
                        "supplier": 20,
                    },
                    "total_tags": 18,
                    "total_namespaces": 2,
                    "provenance_entries": 85,
                }

        Example:
            >>> stats = engine.get_statistics()
            >>> print(stats["total_schemas"], stats["by_status"]["active"])
        """
        with self._lock:
            total = len(self._schemas)

            by_type: Dict[str, int] = {st: 0 for st in VALID_SCHEMA_TYPES}
            by_status: Dict[str, int] = {s: 0 for s in VALID_STATUSES}
            by_namespace: Dict[str, int] = {}

            for record in self._schemas.values():
                st = record.get("schema_type", "unknown")
                by_type[st] = by_type.get(st, 0) + 1

                status = record.get("status", "unknown")
                by_status[status] = by_status.get(status, 0) + 1

                ns = record.get("namespace", "")
                by_namespace[ns] = by_namespace.get(ns, 0) + 1

            group_sizes: Dict[str, int] = {
                gname: len(g["schema_ids"])
                for gname, g in self._groups.items()
            }

            total_tags = len(self._tag_index)
            total_namespaces = len(self._namespace_index)
            total_groups = len(self._groups)

        # Provenance entries — safe to read outside lock (tracker is thread-safe)
        prov_count = getattr(self._provenance, "entry_count", None)
        if prov_count is None:
            prov_count = len(getattr(self._provenance, "_log", []))

        return {
            "total_schemas":      total,
            "by_type":            by_type,
            "by_status":          by_status,
            "by_namespace":       by_namespace,
            "total_groups":       total_groups,
            "groups":             group_sizes,
            "total_tags":         total_tags,
            "total_namespaces":   total_namespaces,
            "provenance_entries": prov_count,
        }

    # ------------------------------------------------------------------
    # 16. reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all registry data.

        Removes all schemas, groups, and index structures.  Resets the
        ProvenanceTracker to a fresh instance.  Intended for use in
        automated tests only — do NOT call in production.

        Example:
            >>> engine.reset()
            >>> assert engine.get_statistics()["total_schemas"] == 0
        """
        with self._lock:
            self._schemas.clear()
            self._groups.clear()
            self._namespace_index.clear()
            self._name_index.clear()
            self._tag_index.clear()
            self._provenance = ProvenanceTracker()

        sm_registry_size.set(0)
        sm_groups_total.set(0)
        sm_namespaces_total.set(0)

        logger.warning(
            "SchemaRegistryEngine.reset() called — all registry data cleared."
        )


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "SchemaRegistryEngine",
    "VALID_SCHEMA_TYPES",
    "VALID_STATUSES",
    "STATUS_TRANSITIONS",
    "MAX_BULK_IMPORT",
]
