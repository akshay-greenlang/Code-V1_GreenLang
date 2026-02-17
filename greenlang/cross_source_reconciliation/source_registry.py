# -*- coding: utf-8 -*-
"""
Source Registry Engine - AGENT-DATA-015 Cross-Source Reconciliation

Pure-Python engine for managing data sources with metadata, schema
mappings, credibility scoring, tolerance rules, and health metrics.
Provides source registration, schema alignment with fuzzy column
matching (Jaro-Winkler similarity), multi-factor credibility
computation, source ranking, and tolerance rule management for
cross-source reconciliation.

Zero-Hallucination: All calculations use deterministic Python
arithmetic. No LLM calls for numeric computations. Credibility
scores are derived from measurable data quality factors with
explicit, auditable formulas.

Engine 1 of 7 in the Cross-Source Reconciliation pipeline.

Example:
    >>> from greenlang.cross_source_reconciliation.source_registry import (
    ...     SourceRegistryEngine,
    ... )
    >>> engine = SourceRegistryEngine()
    >>> source = engine.register_source(
    ...     name="ERP System",
    ...     source_type="erp",
    ...     priority=80,
    ... )
    >>> print(source.id, source.status)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.cross_source_reconciliation.models import (
        SourceType,
        SourceStatus,
        FieldType,
        CredibilityFactor,
        TemporalGranularity,
        SourceDefinition,
        SchemaMapping,
        SourceCredibility,
        SourceHealthMetrics,
        ToleranceRule,
        SUPPORTED_UNITS,
        SUPPORTED_CURRENCIES,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.info(
        "cross_source_reconciliation.models not available; "
        "source registry cannot function without models"
    )

# ---------------------------------------------------------------------------
# Metrics import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.cross_source_reconciliation.metrics import (
        inc_jobs_processed as _inc_jobs_processed,
        observe_duration as _observe_duration_raw,
        inc_errors as _inc_errors,
    )
    _METRICS_AVAILABLE = True

    def inc_jobs_processed(status: str) -> None:
        """Delegate to metrics module inc_jobs_processed."""
        _inc_jobs_processed(status)

    def observe_duration(operation: str, duration: float) -> None:
        """Delegate to metrics module observe_duration (duration only)."""
        _observe_duration_raw(duration)

    def inc_errors(error_type: str) -> None:
        """Delegate to metrics module inc_errors."""
        _inc_errors(error_type)

except ImportError:
    _METRICS_AVAILABLE = False

    def inc_jobs_processed(status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    def observe_duration(operation: str, duration: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    def inc_errors(error_type: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is not available."""

    logger.info(
        "cross_source_reconciliation.metrics not available; "
        "source registry metrics disabled"
    )

# ---------------------------------------------------------------------------
# Provenance import (graceful fallback with inline tracker)
# ---------------------------------------------------------------------------

try:
    from greenlang.cross_source_reconciliation.provenance import (
        ProvenanceTracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    logger.info(
        "cross_source_reconciliation.provenance not available; "
        "using inline ProvenanceTracker"
    )

    class ProvenanceTracker:  # type: ignore[no-redef]
        """Minimal inline provenance tracker for standalone operation.

        Provides SHA-256 chain hashing without external dependencies.
        """

        GENESIS_HASH = hashlib.sha256(
            b"greenlang-cross-source-reconciliation-genesis"
        ).hexdigest()

        def __init__(self) -> None:
            """Initialize with genesis hash."""
            self._last_chain_hash: str = self.GENESIS_HASH
            self._chain: List[Dict[str, Any]] = []
            self._lock = threading.Lock()

        def hash_record(self, data: Dict[str, Any]) -> str:
            """Compute deterministic SHA-256 hash of a data record.

            Args:
                data: Dictionary to hash.

            Returns:
                Hex-encoded SHA-256 hash string.
            """
            serialized = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        def add_to_chain(
            self,
            operation: str,
            input_hash: str,
            output_hash: str,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> str:
            """Add a chain link and return the new chain hash.

            Args:
                operation: Name of the operation performed.
                input_hash: SHA-256 hash of the operation input.
                output_hash: SHA-256 hash of the operation output.
                metadata: Optional additional metadata.

            Returns:
                New chain hash linking this entry to the previous.
            """
            timestamp = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )
            with self._lock:
                combined = json.dumps({
                    "previous": self._last_chain_hash,
                    "input": input_hash,
                    "output": output_hash,
                    "operation": operation,
                    "timestamp": timestamp,
                }, sort_keys=True)
                chain_hash = hashlib.sha256(
                    combined.encode("utf-8"),
                ).hexdigest()

                self._chain.append({
                    "operation": operation,
                    "input_hash": input_hash,
                    "output_hash": output_hash,
                    "chain_hash": chain_hash,
                    "timestamp": timestamp,
                    "metadata": metadata or {},
                })
                self._last_chain_hash = chain_hash

            return chain_hash

        def get_chain(self) -> List[Dict[str, Any]]:
            """Return the full provenance chain.

            Returns:
                List of provenance entries, oldest first.
            """
            with self._lock:
                return list(self._chain)

        @property
        def entry_count(self) -> int:
            """Return the total number of provenance entries."""
            with self._lock:
                return len(self._chain)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to a range.

    Args:
        value: Value to clamp.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Returns:
        Clamped value.
    """
    return max(low, min(high, value))


def _safe_mean(values: List[float]) -> float:
    """Compute arithmetic mean, returning 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        Arithmetic mean or 0.0.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Jaro-Winkler similarity (pure Python, no external dependency)
# ---------------------------------------------------------------------------


def _jaro_similarity(s1: str, s2: str) -> float:
    """Compute Jaro similarity between two strings.

    Implements the Jaro string similarity metric as described in
    the original 1989 paper. Returns a value between 0.0 (no
    similarity) and 1.0 (exact match).

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Jaro similarity score (0.0-1.0).
    """
    if s1 == s2:
        return 1.0

    len1 = len(s1)
    len2 = len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    # Maximum matching distance
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Find matching characters
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len1
        + matches / len2
        + (matches - transpositions / 2.0) / matches
    ) / 3.0

    return jaro


def _jaro_winkler_similarity(
    s1: str,
    s2: str,
    prefix_weight: float = 0.1,
) -> float:
    """Compute Jaro-Winkler similarity between two strings.

    Extends Jaro similarity with a prefix bonus for strings that
    share a common prefix (up to 4 characters).

    Args:
        s1: First string.
        s2: Second string.
        prefix_weight: Winkler prefix scaling factor (default 0.1).

    Returns:
        Jaro-Winkler similarity score (0.0-1.0).

    Example:
        >>> _jaro_winkler_similarity("emission", "emissions")
        0.985...
    """
    jaro = _jaro_similarity(s1, s2)

    # Compute common prefix length (max 4)
    prefix_len = 0
    max_prefix = min(4, len(s1), len(s2))
    for i in range(max_prefix):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * prefix_weight * (1.0 - jaro)


# ---------------------------------------------------------------------------
# Certification score lookup
# ---------------------------------------------------------------------------

#: Default certification scores by source type. Official registries
#: receive the highest score (1.0), ERP systems are well-trusted (0.8),
#: API and database sources are moderately trusted (0.7), while
#: questionnaires (0.5) and manual entry (0.3) are less reliable.
CERTIFICATION_SCORES: Dict[str, float] = {
    "registry": 1.0,
    "erp": 0.8,
    "utility": 0.75,
    "meter": 0.75,
    "api": 0.7,
    "iot": 0.65,
    "spreadsheet": 0.55,
    "questionnaire": 0.5,
    "manual": 0.3,
    "other": 0.4,
}

#: Default credibility factor weights for weighted average computation.
DEFAULT_CREDIBILITY_WEIGHTS: Dict[str, float] = {
    "completeness": 0.25,
    "timeliness": 0.20,
    "consistency": 0.20,
    "accuracy": 0.20,
    "certification": 0.15,
}

#: Mapping of refresh_cadence string to approximate timedelta for
#: timeliness scoring.
CADENCE_TIMEDELTAS: Dict[str, timedelta] = {
    "realtime": timedelta(minutes=5),
    "hourly": timedelta(hours=1),
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
    "monthly": timedelta(days=30),
    "quarterly": timedelta(days=91),
    "annual": timedelta(days=365),
}


# ---------------------------------------------------------------------------
# Unit conversion helpers using models.SUPPORTED_UNITS
# ---------------------------------------------------------------------------


def _get_conversion_factor_from_map(
    unit_from: str,
    unit_to: str,
) -> Optional[float]:
    """Get conversion factor from the SUPPORTED_UNITS lookup map.

    The models.SUPPORTED_UNITS dictionary uses keys like ``"kg_to_tonnes"``
    mapping to their multiplicative conversion factor.

    Args:
        unit_from: Source unit string.
        unit_to: Target unit string.

    Returns:
        Conversion factor, or None if conversion is not in the map.

    Example:
        >>> _get_conversion_factor_from_map("kg", "tonnes")
        0.001
    """
    if not _MODELS_AVAILABLE:
        return None

    key = f"{unit_from}_to_{unit_to}"
    factor = SUPPORTED_UNITS.get(key)
    if factor is not None:
        return factor

    # Try reverse
    reverse_key = f"{unit_to}_to_{unit_from}"
    reverse_factor = SUPPORTED_UNITS.get(reverse_key)
    if reverse_factor is not None and reverse_factor != 0.0:
        return 1.0 / reverse_factor

    return None


# ---------------------------------------------------------------------------
# Default tolerance rules
# ---------------------------------------------------------------------------

#: Default tolerance rules applied when no source-pair-specific rules exist.
DEFAULT_TOLERANCE_DEFS: List[Dict[str, Any]] = [
    {
        "field_name": "*_numeric",
        "field_type": "numeric",
        "tolerance_abs": 0.01,
        "tolerance_pct": 5.0,
        "description": "Default numeric tolerance: 0.01 abs or 5% rel",
    },
    {
        "field_name": "*_currency",
        "field_type": "currency",
        "tolerance_abs": 0.01,
        "tolerance_pct": 2.0,
        "description": "Default currency tolerance: 0.01 abs or 2% rel",
    },
    {
        "field_name": "*_date",
        "field_type": "date",
        "tolerance_abs": None,
        "tolerance_pct": None,
        "description": "Default date tolerance: exact match",
    },
    {
        "field_name": "*_string",
        "field_type": "string",
        "tolerance_abs": None,
        "tolerance_pct": None,
        "description": "Default string tolerance: exact match",
    },
]


# ---------------------------------------------------------------------------
# Internal wrapper for source state beyond Pydantic model
# ---------------------------------------------------------------------------


class _SourceState:
    """Internal mutable state tracked per source beyond the Pydantic model.

    The Pydantic SourceDefinition model uses ``extra='forbid'``, so
    additional operational fields (version, last_refresh, record_count,
    provenance_hash) are tracked here.

    Attributes:
        definition: The Pydantic SourceDefinition.
        version: Monotonic version counter.
        last_refresh: Timestamp of last data refresh.
        record_count: Number of records contributed.
        provenance_hash: Latest SHA-256 provenance hash.
        updated_at: When the source was last updated.
    """

    __slots__ = (
        "definition", "version", "last_refresh", "record_count",
        "provenance_hash", "updated_at",
    )

    def __init__(self, definition: SourceDefinition) -> None:
        """Initialize source state from a SourceDefinition.

        Args:
            definition: The Pydantic SourceDefinition model instance.
        """
        self.definition = definition
        self.version: int = 1
        self.last_refresh: Optional[datetime] = None
        self.record_count: int = 0
        self.provenance_hash: str = ""
        self.updated_at: datetime = _utcnow()


# ---------------------------------------------------------------------------
# SourceRegistryEngine
# ---------------------------------------------------------------------------


class SourceRegistryEngine:
    """Pure-Python engine for managing data sources in cross-source reconciliation.

    Manages the lifecycle of data sources including registration, schema
    mapping, credibility scoring, health monitoring, source ranking, and
    tolerance rule management. Each mutation is tracked with SHA-256
    provenance hashing for complete audit trails.

    The engine stores all state in-memory for high-performance access
    during reconciliation runs. For persistent storage, the service layer
    serialises state to PostgreSQL.

    Attributes:
        _sources: Registered source states keyed by source id.
        _schema_maps: Schema mappings keyed by source id.
        _credibility_cache: Cached credibility scores keyed by source id.
        _tolerance_rules: Tolerance rules keyed by source pair key.
        _provenance: SHA-256 provenance tracker.
        _lock: Thread-safety lock for concurrent access.

    Example:
        >>> engine = SourceRegistryEngine()
        >>> src = engine.register_source(
        ...     name="SAP ERP",
        ...     source_type="erp",
        ...     priority=90,
        ...     schema_info={"columns": ["spend", "vendor", "date"]},
        ... )
        >>> assert src.status == SourceStatus.ACTIVE
        >>> assert 0.0 <= src.credibility_score <= 1.0
    """

    def __init__(self) -> None:
        """Initialize SourceRegistryEngine with empty state."""
        self._sources: Dict[str, _SourceState] = {}
        self._schema_maps: Dict[str, List[SchemaMapping]] = {}
        self._credibility_cache: Dict[str, SourceCredibility] = {}
        self._tolerance_rules: Dict[str, List[ToleranceRule]] = {}
        self._provenance = ProvenanceTracker()
        self._lock = threading.Lock()

        logger.info("SourceRegistryEngine initialized")

    # ------------------------------------------------------------------
    # 1. register_source
    # ------------------------------------------------------------------

    def register_source(
        self,
        name: str,
        source_type: str = "other",
        priority: int = 50,
        schema_info: Optional[Dict[str, Any]] = None,
        refresh_cadence: str = "daily",
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> SourceDefinition:
        """Register a new data source in the registry.

        Creates a SourceDefinition with a unique ID, validates the
        priority range, assigns a default credibility score of 0.5,
        sets the status to active, and records provenance.

        Args:
            name: Human-readable name for the source.
            source_type: Source type classification. Must be one of the
                SourceType enum values: erp, utility, meter,
                questionnaire, spreadsheet, api, iot, registry,
                manual, other. Defaults to "other".
            priority: Priority ranking from 1 (lowest) to 100 (highest).
                Higher-priority sources are preferred during conflict
                resolution. Defaults to 50.
            schema_info: Optional dictionary describing the source schema
                (column names, types, constraints). Used for schema
                alignment and mapping.
            refresh_cadence: Expected refresh interval string. One of:
                realtime, hourly, daily, weekly, monthly, quarterly,
                annual. Defaults to "daily".
            description: Optional human-readable description.
            tags: Optional list of tags for categorisation.

        Returns:
            SourceDefinition Pydantic model with populated id, status,
            and timestamps.

        Raises:
            ValueError: If priority is outside the 1-100 range, name is
                empty, or source_type is not a recognised value.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> src = engine.register_source(
            ...     name="ERP System",
            ...     source_type="erp",
            ...     priority=80,
            ...     schema_info={"columns": ["amount", "date"]},
            ... )
            >>> assert src.id
            >>> assert src.status == SourceStatus.ACTIVE
        """
        start = time.time()

        # Validate name
        if not name or not name.strip():
            raise ValueError("Source name must not be empty")

        # Validate priority
        if not 1 <= priority <= 100:
            raise ValueError(
                f"Priority must be between 1 and 100, got {priority}"
            )

        # Resolve source type enum
        source_type_enum = self._resolve_source_type(source_type)

        # Build SourceDefinition (Pydantic model)
        source = SourceDefinition(
            name=name.strip(),
            source_type=source_type_enum,
            priority=priority,
            credibility_score=0.5,
            schema_info=schema_info or {},
            refresh_cadence=refresh_cadence.strip(),
            description=description.strip() if description else "",
            tags=list(tags) if tags else [],
            status=SourceStatus.ACTIVE,
        )

        source_id = source.id

        # Build internal state wrapper
        state = _SourceState(source)

        # Compute provenance hash
        input_hash = self._provenance.hash_record({
            "name": source.name,
            "source_type": source.source_type.value,
            "priority": source.priority,
        })
        output_hash = self._provenance.hash_record({
            "source_id": source_id,
            "status": source.status.value,
            "credibility_score": source.credibility_score,
        })
        provenance_hash = self._provenance.add_to_chain(
            operation="register_source",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "source_id": source_id,
                "name": source.name,
                "source_type": source.source_type.value,
                "priority": source.priority,
            },
        )
        state.provenance_hash = provenance_hash

        # Store
        with self._lock:
            self._sources[source_id] = state

        elapsed = time.time() - start
        inc_jobs_processed("registered")
        observe_duration("register_source", elapsed)

        logger.info(
            "Source registered: id=%s, name=%s, type=%s, priority=%d, "
            "%.3fms",
            source_id[:12], source.name, source.source_type.value,
            source.priority, elapsed * 1000,
        )
        return source

    # ------------------------------------------------------------------
    # 2. update_source
    # ------------------------------------------------------------------

    def update_source(
        self,
        source_id: str,
        **kwargs: Any,
    ) -> SourceDefinition:
        """Update specified fields of an existing source.

        Only the fields provided in ``kwargs`` are updated. All other
        fields retain their current values. The internal version counter
        is incremented and a new provenance hash is computed.

        Supported fields:
            name, source_type, priority, schema_info, refresh_cadence,
            description, tags, status, credibility_score.

        Args:
            source_id: ID of the source to update.
            **kwargs: Field-name/value pairs to update.

        Returns:
            New SourceDefinition with updated values.

        Raises:
            KeyError: If source_id is not found.
            ValueError: If updated values fail validation.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> src = engine.register_source(name="ERP", source_type="erp")
            >>> updated = engine.update_source(src.id, priority=90)
            >>> assert updated.priority == 90
        """
        start = time.time()

        with self._lock:
            if source_id not in self._sources:
                raise KeyError(f"Source not found: {source_id}")
            state = self._sources[source_id]
            old_def = state.definition

        # Validate updatable fields
        updatable = {
            "name", "source_type", "priority", "schema_info",
            "refresh_cadence", "description", "tags", "status",
            "credibility_score",
        }
        # Also accept internal state fields
        internal_updatable = {"last_refresh", "record_count"}
        all_updatable = updatable | internal_updatable
        invalid_keys = set(kwargs.keys()) - all_updatable
        if invalid_keys:
            raise ValueError(
                f"Cannot update fields: {sorted(invalid_keys)}. "
                f"Updatable: {sorted(all_updatable)}"
            )

        # Build new model data from existing
        model_data = old_def.model_dump()

        # Track changes
        changes: Dict[str, Any] = {}

        # Process each update
        for key, value in kwargs.items():
            if key in internal_updatable:
                # Internal state field
                old_val = getattr(state, key, None)
                if old_val != value:
                    setattr(state, key, value)
                    changes[key] = {"old": str(old_val), "new": str(value)}
                continue

            # Pydantic model field
            if key == "source_type":
                value = self._resolve_source_type(value)

            if key == "status":
                value = self._resolve_source_status(value)

            old_val = model_data.get(key)
            if old_val != value:
                model_data[key] = value
                changes[key] = {"old": str(old_val), "new": str(value)}

        # Preserve the original id
        model_data["id"] = source_id

        # Create new SourceDefinition with updated values
        new_def = SourceDefinition(**model_data)

        # Update state
        state.definition = new_def
        state.version += 1
        state.updated_at = _utcnow()

        # Compute provenance
        input_hash = self._provenance.hash_record({
            "source_id": source_id,
            "changes": {k: v["new"] for k, v in changes.items()},
            "version": state.version,
        })
        output_hash = self._provenance.hash_record({
            "source_id": source_id,
            "status": new_def.status.value,
            "version": state.version,
        })
        provenance_hash = self._provenance.add_to_chain(
            operation="update_source",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "source_id": source_id,
                "changes": list(changes.keys()),
                "version": state.version,
            },
        )
        state.provenance_hash = provenance_hash

        with self._lock:
            self._sources[source_id] = state

        elapsed = time.time() - start
        inc_jobs_processed("updated")
        observe_duration("update_source", elapsed)

        logger.info(
            "Source updated: id=%s, fields=%s, version=%d, %.3fms",
            source_id[:12], list(changes.keys()),
            state.version, elapsed * 1000,
        )
        return new_def

    # ------------------------------------------------------------------
    # 3. get_source
    # ------------------------------------------------------------------

    def get_source(
        self,
        source_id: str,
    ) -> Optional[SourceDefinition]:
        """Retrieve a registered source by its ID.

        Args:
            source_id: Unique identifier of the source to retrieve.

        Returns:
            SourceDefinition if found, None otherwise.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> src = engine.register_source(name="Test", source_type="api")
            >>> retrieved = engine.get_source(src.id)
            >>> assert retrieved is not None
            >>> assert retrieved.name == "Test"
        """
        with self._lock:
            state = self._sources.get(source_id)

        if state is None:
            logger.debug("Source not found: id=%s", source_id[:12])
            return None

        logger.debug(
            "Source retrieved: id=%s, name=%s",
            source_id[:12], state.definition.name,
        )
        return state.definition

    # ------------------------------------------------------------------
    # 4. list_sources
    # ------------------------------------------------------------------

    def list_sources(
        self,
        source_type: Optional[str] = None,
        status: Optional[str] = None,
        min_priority: Optional[int] = None,
    ) -> List[SourceDefinition]:
        """List registered sources with optional filtering.

        Filters are combined with AND logic: a source must match all
        specified filters to be included in the result.

        Args:
            source_type: Filter by source type (e.g. "erp", "api").
                None to include all types.
            status: Filter by status (e.g. "active", "inactive").
                None to include all statuses.
            min_priority: Filter by minimum priority (inclusive).
                None to include all priorities.

        Returns:
            List of SourceDefinition objects matching all filters,
            sorted by priority descending then by name ascending.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> engine.register_source(name="A", source_type="erp", priority=90)
            >>> engine.register_source(name="B", source_type="api", priority=70)
            >>> erp_sources = engine.list_sources(source_type="erp")
            >>> assert len(erp_sources) == 1
        """
        with self._lock:
            definitions = [
                st.definition for st in self._sources.values()
            ]

        # Apply filters
        if source_type is not None:
            st_lower = source_type.lower().strip()
            definitions = [
                d for d in definitions
                if d.source_type.value == st_lower
            ]

        if status is not None:
            st_lower = status.lower().strip()
            definitions = [
                d for d in definitions
                if d.status.value == st_lower
            ]

        if min_priority is not None:
            definitions = [
                d for d in definitions
                if d.priority >= min_priority
            ]

        # Sort: priority desc, then name asc
        definitions.sort(key=lambda d: (-d.priority, d.name))

        logger.debug(
            "Listed sources: total=%d, filters=(type=%s, status=%s, "
            "min_priority=%s)",
            len(definitions), source_type, status, min_priority,
        )
        return definitions

    # ------------------------------------------------------------------
    # 5. deactivate_source
    # ------------------------------------------------------------------

    def deactivate_source(self, source_id: str) -> bool:
        """Deactivate a registered source.

        Sets the source status to inactive. Deactivated sources are
        excluded from reconciliation runs but remain in the registry
        for audit purposes.

        Args:
            source_id: ID of the source to deactivate.

        Returns:
            True if the source was successfully deactivated, False if
            the source was not found or was already inactive.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> src = engine.register_source(name="Test", source_type="api")
            >>> assert engine.deactivate_source(src.id) is True
            >>> assert engine.get_source(src.id).status == SourceStatus.INACTIVE
        """
        start = time.time()

        with self._lock:
            if source_id not in self._sources:
                logger.debug(
                    "Cannot deactivate: source not found id=%s",
                    source_id[:12],
                )
                return False

            state = self._sources[source_id]
            old_def = state.definition

            if old_def.status == SourceStatus.INACTIVE:
                logger.debug(
                    "Source already inactive: id=%s", source_id[:12],
                )
                return False

            old_status = old_def.status.value

        # Rebuild definition with inactive status
        self.update_source(source_id, status=SourceStatus.INACTIVE)

        elapsed = time.time() - start

        # Additional provenance for deactivation event
        input_hash = self._provenance.hash_record({
            "source_id": source_id,
            "old_status": old_status,
        })
        output_hash = self._provenance.hash_record({
            "source_id": source_id,
            "new_status": "inactive",
        })
        self._provenance.add_to_chain(
            operation="deactivate_source",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "source_id": source_id,
                "old_status": old_status,
            },
        )

        inc_jobs_processed("deactivated")
        observe_duration("deactivate_source", elapsed)

        logger.info(
            "Source deactivated: id=%s, was=%s, %.3fms",
            source_id[:12], old_status, elapsed * 1000,
        )
        return True

    # ------------------------------------------------------------------
    # 6. register_schema_mapping
    # ------------------------------------------------------------------

    def register_schema_mapping(
        self,
        source_id: str,
        mappings: List[SchemaMapping],
    ) -> List[SchemaMapping]:
        """Register schema mappings for a source.

        Maps source columns to canonical columns with optional unit
        conversion and date format alignment. Replaces any existing
        mappings for the source.

        Unit conversions are logged when ``unit_from`` and ``unit_to``
        are provided. The actual conversion factor lookup uses the
        SUPPORTED_UNITS map from models.

        Args:
            source_id: ID of the source to map.
            mappings: List of SchemaMapping objects defining column-level
                mappings from source to canonical schema.

        Returns:
            List of SchemaMapping objects as stored.

        Raises:
            KeyError: If source_id is not found.
            ValueError: If mappings list is empty or contains duplicate
                source_column entries.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> src = engine.register_source(name="ERP", source_type="erp")
            >>> maps = engine.register_schema_mapping(src.id, [
            ...     SchemaMapping(
            ...         source_column="weight_kg",
            ...         canonical_column="weight",
            ...         unit_from="kg",
            ...         unit_to="tonnes",
            ...     ),
            ... ])
            >>> assert len(maps) == 1
        """
        start = time.time()

        with self._lock:
            if source_id not in self._sources:
                raise KeyError(f"Source not found: {source_id}")

        if not mappings:
            raise ValueError("Mappings list must not be empty")

        # Check for duplicate source columns
        source_cols = [m.source_column for m in mappings]
        seen: Set[str] = set()
        duplicates: Set[str] = set()
        for c in source_cols:
            if c in seen:
                duplicates.add(c)
            seen.add(c)
        if duplicates:
            raise ValueError(
                f"Duplicate source_column entries: {duplicates}"
            )

        # Log unit conversions if applicable
        for mapping in mappings:
            if mapping.unit_from and mapping.unit_to:
                factor = _get_conversion_factor_from_map(
                    mapping.unit_from, mapping.unit_to,
                )
                if factor is not None:
                    logger.debug(
                        "Unit conversion available: %s -> %s = %.6f",
                        mapping.unit_from, mapping.unit_to, factor,
                    )
                else:
                    logger.debug(
                        "No built-in conversion: %s -> %s",
                        mapping.unit_from, mapping.unit_to,
                    )

        # Provenance for mapping registration
        input_hash = self._provenance.hash_record({
            "source_id": source_id,
            "mapping_count": len(mappings),
            "columns": [m.source_column for m in mappings],
        })
        output_hash = self._provenance.hash_record({
            "source_id": source_id,
            "registered_count": len(mappings),
        })
        self._provenance.add_to_chain(
            operation="register_schema_mapping",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "source_id": source_id,
                "mapping_count": len(mappings),
            },
        )

        # Store mappings (replace existing)
        with self._lock:
            self._schema_maps[source_id] = list(mappings)

        elapsed = time.time() - start
        inc_jobs_processed("schema_mapped")
        observe_duration("register_schema_mapping", elapsed)

        logger.info(
            "Schema mappings registered: source=%s, mappings=%d, %.3fms",
            source_id[:12], len(mappings), elapsed * 1000,
        )
        return list(mappings)

    # ------------------------------------------------------------------
    # 7. get_schema_mapping
    # ------------------------------------------------------------------

    def get_schema_mapping(
        self,
        source_id: str,
    ) -> List[SchemaMapping]:
        """Retrieve schema mappings for a source.

        Args:
            source_id: ID of the source whose mappings to retrieve.

        Returns:
            List of SchemaMapping objects for the source. Returns an
            empty list if no mappings are registered.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> src = engine.register_source(name="Test", source_type="api")
            >>> assert engine.get_schema_mapping(src.id) == []
        """
        with self._lock:
            mappings = self._schema_maps.get(source_id, [])

        logger.debug(
            "Schema mappings retrieved: source=%s, count=%d",
            source_id[:12], len(mappings),
        )
        return list(mappings)

    # ------------------------------------------------------------------
    # 8. align_schemas
    # ------------------------------------------------------------------

    def align_schemas(
        self,
        source_ids: List[str],
        similarity_threshold: float = 0.85,
    ) -> Dict[str, List[SchemaMapping]]:
        """Auto-discover common columns across multiple sources.

        Performs fuzzy column name matching using Jaro-Winkler similarity
        to suggest schema mappings. For each source, compares its schema
        columns against a canonical column set derived from the union of
        all source schemas. Columns with similarity above the threshold
        are suggested as mappings.

        Args:
            source_ids: List of source IDs to align.
            similarity_threshold: Minimum Jaro-Winkler similarity score
                (0.0-1.0) for a column match. Defaults to 0.85.

        Returns:
            Dictionary mapping source_id to a list of suggested
            SchemaMapping objects.

        Raises:
            KeyError: If any source_id is not found.
            ValueError: If fewer than 2 source_ids are provided or if
                similarity_threshold is outside 0.0-1.0.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> s1 = engine.register_source(
            ...     name="A", source_type="erp",
            ...     schema_info={"columns": ["emission_amount", "date"]},
            ... )
            >>> s2 = engine.register_source(
            ...     name="B", source_type="api",
            ...     schema_info={"columns": ["emissions_amount", "date"]},
            ... )
            >>> aligned = engine.align_schemas([s1.id, s2.id])
            >>> assert len(aligned) == 2
        """
        start = time.time()

        if len(source_ids) < 2:
            raise ValueError(
                "At least 2 source_ids are required for schema alignment"
            )

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be 0.0-1.0, "
                f"got {similarity_threshold}"
            )

        # Collect schemas
        source_schemas: Dict[str, List[str]] = {}
        with self._lock:
            for sid in source_ids:
                if sid not in self._sources:
                    raise KeyError(f"Source not found: {sid}")
                source_def = self._sources[sid].definition
                columns = self._extract_columns(source_def.schema_info)
                source_schemas[sid] = columns

        # Build canonical column set: union of all columns
        all_columns: Set[str] = set()
        for columns in source_schemas.values():
            all_columns.update(columns)

        canonical_columns = sorted(all_columns)

        # For each source, find best matches for each of its columns
        result: Dict[str, List[SchemaMapping]] = {}

        for sid in source_ids:
            suggested: List[SchemaMapping] = []
            source_columns = source_schemas[sid]

            for src_col in source_columns:
                best_match: Optional[str] = None
                best_score = 0.0

                # Normalise for comparison
                src_norm = self._normalize_column_name(src_col)

                for can_col in canonical_columns:
                    can_norm = self._normalize_column_name(can_col)

                    # Exact match
                    if src_norm == can_norm:
                        best_match = can_col
                        best_score = 1.0
                        break

                    # Jaro-Winkler similarity
                    score = _jaro_winkler_similarity(src_norm, can_norm)
                    if score > best_score:
                        best_score = score
                        best_match = can_col

                if (
                    best_match is not None
                    and best_score >= similarity_threshold
                ):
                    mapping = SchemaMapping(
                        source_column=src_col,
                        canonical_column=best_match,
                    )
                    suggested.append(mapping)

                    logger.debug(
                        "Schema align match: source=%s, %s -> %s "
                        "(score=%.3f)",
                        sid[:12], src_col, best_match, best_score,
                    )

            result[sid] = suggested

        # Provenance for alignment operation
        input_hash = self._provenance.hash_record({
            "source_ids": sorted(source_ids),
            "threshold": similarity_threshold,
        })
        total_suggestions = sum(len(v) for v in result.values())
        output_hash = self._provenance.hash_record({
            "total_suggestions": total_suggestions,
            "source_count": len(source_ids),
        })
        self._provenance.add_to_chain(
            operation="align_schemas",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "source_count": len(source_ids),
                "total_suggestions": total_suggestions,
                "threshold": similarity_threshold,
            },
        )

        elapsed = time.time() - start
        observe_duration("align_schemas", elapsed)

        logger.info(
            "Schema alignment complete: sources=%d, suggestions=%d, "
            "threshold=%.2f, %.3fms",
            len(source_ids), total_suggestions,
            similarity_threshold, elapsed * 1000,
        )
        return result

    # ------------------------------------------------------------------
    # 9. compute_credibility
    # ------------------------------------------------------------------

    def compute_credibility(
        self,
        source_id: str,
        records: List[Dict[str, Any]],
        reference_records: Optional[List[Dict[str, Any]]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> SourceCredibility:
        """Compute a multi-factor credibility score for a source.

        Evaluates five credibility factors:

        1. **Completeness** (default weight 0.25): Fraction of non-null
           fields across all records. A fully complete dataset scores 1.0.

        2. **Timeliness** (default weight 0.20): Freshness of the source
           data relative to its configured refresh cadence. A recently
           refreshed source scores 1.0; score decays exponentially as
           staleness increases.

        3. **Consistency** (default weight 0.20): Internal consistency
           measured by format uniformity and value stability.

        4. **Accuracy** (default weight 0.20): Deviation from consensus
           when reference_records from other sources are provided.
           Without reference data, defaults to 0.5 (neutral).

        5. **Certification** (default weight 0.15): Static score based
           on source type. Registry sources score 1.0, ERP 0.8,
           questionnaire 0.5, manual 0.3.

        Args:
            source_id: ID of the source to assess.
            records: List of record dictionaries from the source.
            reference_records: Optional records from other sources for
                accuracy comparison.
            weights: Optional custom weights for each factor. Keys must
                be credibility factor names. Weights are normalised to
                sum to 1.0.

        Returns:
            SourceCredibility with individual factor scores and
            weighted overall score.

        Raises:
            KeyError: If source_id is not found.
            ValueError: If records list is empty.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> src = engine.register_source(name="ERP", source_type="erp")
            >>> cred = engine.compute_credibility(
            ...     src.id,
            ...     [{"amount": 100, "date": "2025-01-01"}],
            ... )
            >>> assert 0.0 <= cred.overall_score <= 1.0
        """
        start = time.time()

        with self._lock:
            if source_id not in self._sources:
                raise KeyError(f"Source not found: {source_id}")
            state = self._sources[source_id]
            source_def = state.definition

        if not records:
            raise ValueError("Records list must not be empty")

        # Resolve weights
        w = dict(DEFAULT_CREDIBILITY_WEIGHTS)
        if weights:
            for k, v in weights.items():
                if k in w:
                    w[k] = v
        # Normalise weights to sum to 1.0
        total_weight = sum(w.values())
        if total_weight > 0:
            w = {k: v / total_weight for k, v in w.items()}

        # Factor 1: Completeness
        completeness = self._compute_completeness(records)

        # Factor 2: Timeliness
        timeliness = self._compute_timeliness(state)

        # Factor 3: Consistency
        consistency = self._compute_consistency(records)

        # Factor 4: Accuracy
        accuracy = self._compute_accuracy(records, reference_records)

        # Factor 5: Certification
        certification = CERTIFICATION_SCORES.get(
            source_def.source_type.value, 0.4,
        )

        # Weighted average
        overall = (
            w.get("completeness", 0.25) * completeness
            + w.get("timeliness", 0.20) * timeliness
            + w.get("consistency", 0.20) * consistency
            + w.get("accuracy", 0.20) * accuracy
            + w.get("certification", 0.15) * certification
        )
        overall = _clamp(overall, 0.0, 1.0)

        credibility = SourceCredibility(
            source_id=source_id,
            completeness_score=round(completeness, 4),
            timeliness_score=round(timeliness, 4),
            consistency_score=round(consistency, 4),
            accuracy_score=round(accuracy, 4),
            certification_score=round(certification, 4),
            overall_score=round(overall, 4),
            sample_size=len(records),
        )

        # Provenance
        input_hash = self._provenance.hash_record({
            "source_id": source_id,
            "record_count": len(records),
            "has_reference": reference_records is not None,
        })
        output_hash = self._provenance.hash_record({
            "overall_score": credibility.overall_score,
            "completeness": credibility.completeness_score,
            "timeliness": credibility.timeliness_score,
            "consistency": credibility.consistency_score,
            "accuracy": credibility.accuracy_score,
            "certification": credibility.certification_score,
        })
        self._provenance.add_to_chain(
            operation="compute_credibility",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "source_id": source_id,
                "overall_score": credibility.overall_score,
                "record_count": len(records),
            },
        )

        # Cache credibility and update source credibility score
        with self._lock:
            self._credibility_cache[source_id] = credibility
            if source_id in self._sources:
                # Update credibility_score on the definition
                old_def = self._sources[source_id].definition
                model_data = old_def.model_dump()
                model_data["credibility_score"] = round(overall, 4)
                self._sources[source_id].definition = SourceDefinition(
                    **model_data,
                )

        elapsed = time.time() - start
        inc_jobs_processed("credibility_computed")
        observe_duration("compute_credibility", elapsed)

        logger.info(
            "Credibility computed: source=%s, overall=%.3f "
            "(comp=%.3f, time=%.3f, cons=%.3f, acc=%.3f, cert=%.3f), "
            "records=%d, %.3fms",
            source_id[:12], overall, completeness, timeliness,
            consistency, accuracy, certification,
            len(records), elapsed * 1000,
        )
        return credibility

    # ------------------------------------------------------------------
    # 10. rank_sources
    # ------------------------------------------------------------------

    def rank_sources(
        self,
        source_ids: List[str],
    ) -> List[Tuple[str, float]]:
        """Rank sources by combined priority and credibility score.

        Computes a ranking score as ``(priority/100) * credibility_score``
        for each source and returns them sorted in descending order
        (highest-ranked first).

        Args:
            source_ids: List of source IDs to rank.

        Returns:
            List of (source_id, ranking_score) tuples sorted by
            ranking_score descending. Sources not found are excluded.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> s1 = engine.register_source(name="A", source_type="erp", priority=90)
            >>> s2 = engine.register_source(name="B", source_type="manual", priority=30)
            >>> ranked = engine.rank_sources([s1.id, s2.id])
            >>> assert ranked[0][0] == s1.id
        """
        start = time.time()

        rankings: List[Tuple[str, float]] = []

        with self._lock:
            for sid in source_ids:
                state = self._sources.get(sid)
                if state is None:
                    logger.debug(
                        "Rank: skipping unknown source %s", sid[:12],
                    )
                    continue

                source_def = state.definition
                # Ranking = priority (1-100) normalized * credibility (0-1)
                norm_priority = source_def.priority / 100.0
                score = norm_priority * source_def.credibility_score
                rankings.append((sid, round(score, 4)))

        # Sort descending by score
        rankings.sort(key=lambda x: -x[1])

        elapsed = time.time() - start
        observe_duration("rank_sources", elapsed)

        logger.info(
            "Sources ranked: count=%d, top=%s (%.4f), %.3fms",
            len(rankings),
            rankings[0][0][:12] if rankings else "none",
            rankings[0][1] if rankings else 0.0,
            elapsed * 1000,
        )
        return rankings

    # ------------------------------------------------------------------
    # 11. get_source_health
    # ------------------------------------------------------------------

    def get_source_health(
        self,
        source_id: str,
    ) -> SourceHealthMetrics:
        """Compute health metrics for a registered source.

        Aggregates the source's contribution statistics, discrepancy
        rate, missing value rate, and average credibility score.

        Args:
            source_id: ID of the source to assess.

        Returns:
            SourceHealthMetrics with populated fields. Returns default
            metrics (zeros) if the source has no credibility history.

        Raises:
            KeyError: If source_id is not found.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> src = engine.register_source(name="Test", source_type="erp")
            >>> health = engine.get_source_health(src.id)
            >>> assert health.source_id == src.id
        """
        start = time.time()

        with self._lock:
            if source_id not in self._sources:
                raise KeyError(f"Source not found: {source_id}")
            state = self._sources[source_id]
            cached_cred = self._credibility_cache.get(source_id)

        # Derive health from credibility cache and state
        avg_credibility = 0.0
        missing_rate = 0.0
        discrepancy_rate = 0.0

        if cached_cred is not None:
            avg_credibility = cached_cred.overall_score
            # Missing rate is inverse of completeness
            missing_rate = round(
                1.0 - cached_cred.completeness_score, 4,
            )
            # Discrepancy rate derived from accuracy
            discrepancy_rate = round(
                1.0 - cached_cred.accuracy_score, 4,
            )

        health = SourceHealthMetrics(
            source_id=source_id,
            records_contributed=state.record_count,
            discrepancy_rate=discrepancy_rate,
            missing_rate=missing_rate,
            avg_credibility=round(avg_credibility, 4),
            last_refresh=state.last_refresh,
        )

        # Provenance
        input_hash = self._provenance.hash_record({
            "source_id": source_id,
            "record_count": state.record_count,
        })
        output_hash = self._provenance.hash_record({
            "avg_credibility": health.avg_credibility,
            "missing_rate": health.missing_rate,
            "discrepancy_rate": health.discrepancy_rate,
        })
        self._provenance.add_to_chain(
            operation="get_source_health",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "source_id": source_id,
                "records_contributed": health.records_contributed,
            },
        )

        elapsed = time.time() - start
        observe_duration("get_source_health", elapsed)

        logger.info(
            "Source health computed: source=%s, records=%d, "
            "credibility=%.3f, missing=%.3f, discrepancy=%.3f, %.3fms",
            source_id[:12], health.records_contributed,
            health.avg_credibility, health.missing_rate,
            health.discrepancy_rate, elapsed * 1000,
        )
        return health

    # ------------------------------------------------------------------
    # 12. set_tolerance_rules
    # ------------------------------------------------------------------

    def set_tolerance_rules(
        self,
        source_pair_key: str,
        rules: List[ToleranceRule],
    ) -> None:
        """Store tolerance rules for a specific source pair.

        Tolerance rules define acceptable differences between values
        from two sources during reconciliation. They are keyed by
        a source pair key (typically ``"sourceA_id:sourceB_id"``).

        Args:
            source_pair_key: Key identifying the source pair. Convention
                is ``"<source_a_id>:<source_b_id>"`` with IDs sorted
                alphabetically.
            rules: List of ToleranceRule objects to store.

        Raises:
            ValueError: If source_pair_key is empty or rules is empty.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> s1 = engine.register_source(name="A", source_type="erp")
            >>> s2 = engine.register_source(name="B", source_type="api")
            >>> pair_key = f"{s1.id}:{s2.id}"
            >>> engine.set_tolerance_rules(pair_key, [
            ...     ToleranceRule(
            ...         field_name="amount",
            ...         tolerance_pct=5.0,
            ...     ),
            ... ])
        """
        start = time.time()

        if not source_pair_key or not source_pair_key.strip():
            raise ValueError("source_pair_key must not be empty")

        if not rules:
            raise ValueError("Rules list must not be empty")

        # Provenance
        input_hash = self._provenance.hash_record({
            "pair_key": source_pair_key,
            "rule_count": len(rules),
            "fields": [r.field_name for r in rules],
        })
        output_hash = self._provenance.hash_record({
            "pair_key": source_pair_key,
            "stored_count": len(rules),
        })
        self._provenance.add_to_chain(
            operation="set_tolerance_rules",
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={
                "pair_key": source_pair_key,
                "rule_count": len(rules),
            },
        )

        with self._lock:
            self._tolerance_rules[source_pair_key.strip()] = list(rules)

        elapsed = time.time() - start
        inc_jobs_processed("tolerance_set")
        observe_duration("set_tolerance_rules", elapsed)

        logger.info(
            "Tolerance rules set: pair=%s, rules=%d, %.3fms",
            source_pair_key[:24], len(rules), elapsed * 1000,
        )

    # ------------------------------------------------------------------
    # 13. get_tolerance_rules
    # ------------------------------------------------------------------

    def get_tolerance_rules(
        self,
        source_a_id: str,
        source_b_id: str,
    ) -> List[ToleranceRule]:
        """Retrieve tolerance rules for a source pair.

        Checks both orderings of the pair key (a:b and b:a). If no
        source-pair-specific rules exist, returns default tolerance
        rules.

        Args:
            source_a_id: ID of the first source.
            source_b_id: ID of the second source.

        Returns:
            List of ToleranceRule objects for this pair, or default
            rules if no pair-specific rules are defined.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> s1 = engine.register_source(name="A", source_type="erp")
            >>> s2 = engine.register_source(name="B", source_type="api")
            >>> rules = engine.get_tolerance_rules(s1.id, s2.id)
            >>> assert len(rules) > 0  # defaults are returned
        """
        key_ab = f"{source_a_id}:{source_b_id}"
        key_ba = f"{source_b_id}:{source_a_id}"

        with self._lock:
            rules = self._tolerance_rules.get(key_ab)
            if rules is not None:
                logger.debug(
                    "Tolerance rules found for pair %s: %d rules",
                    key_ab[:24], len(rules),
                )
                return list(rules)

            rules = self._tolerance_rules.get(key_ba)
            if rules is not None:
                logger.debug(
                    "Tolerance rules found for pair %s: %d rules",
                    key_ba[:24], len(rules),
                )
                return list(rules)

        # Return default rules
        defaults = self._build_default_tolerance_rules()
        logger.debug(
            "No pair-specific rules for %s:%s; returning %d defaults",
            source_a_id[:12], source_b_id[:12], len(defaults),
        )
        return defaults

    # ------------------------------------------------------------------
    # 14. _normalize_value
    # ------------------------------------------------------------------

    def _normalize_value(
        self,
        value: Any,
        field_type: str,
        schema_mapping: Optional[SchemaMapping] = None,
    ) -> Any:
        """Normalise a value using unit conversion, date formatting, or string normalization.

        Applies transformations defined in the schema mapping to convert
        the value from the source's representation to the canonical
        representation.

        Args:
            value: Raw value from the source.
            field_type: Type classification of the field (string, numeric,
                date, currency, unit_value, boolean, categorical).
            schema_mapping: Optional SchemaMapping containing conversion
                rules. If None, the value is returned as-is.

        Returns:
            Normalised value in the canonical representation. Returns
            the original value if no conversion is applicable.

        Example:
            >>> engine = SourceRegistryEngine()
            >>> result = engine._normalize_value(
            ...     1500.0, "numeric",
            ...     SchemaMapping(
            ...         source_column="weight",
            ...         canonical_column="weight_tonnes",
            ...         unit_from="kg",
            ...         unit_to="tonnes",
            ...     ),
            ... )
        """
        if value is None:
            return None

        if schema_mapping is None:
            return value

        # Numeric/unit conversion
        if field_type in ("numeric", "unit_value", "currency"):
            return self._normalize_numeric(value, schema_mapping)

        # Date format normalization
        if field_type == "date":
            return self._normalize_date(value, schema_mapping)

        # String normalization
        if field_type in ("string", "categorical"):
            return self._normalize_string(value)

        # Default: return as-is
        return value

    # ------------------------------------------------------------------
    # 15. _compute_provenance
    # ------------------------------------------------------------------

    def _compute_provenance(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
    ) -> str:
        """Compute SHA-256 provenance hash for an operation.

        Creates a chain link recording the operation with input/output
        hashes for complete audit trail.

        Args:
            operation: Name of the operation performed.
            input_data: Input data (dict or any JSON-serialisable type).
            output_data: Output data (dict or any JSON-serialisable type).

        Returns:
            SHA-256 chain hash string.
        """
        if isinstance(input_data, dict):
            input_hash = self._provenance.hash_record(input_data)
        else:
            input_hash = self._provenance.hash_record(
                {"data": str(input_data)},
            )

        if isinstance(output_data, dict):
            output_hash = self._provenance.hash_record(output_data)
        else:
            output_hash = self._provenance.hash_record(
                {"data": str(output_data)},
            )

        return self._provenance.add_to_chain(
            operation=operation,
            input_hash=input_hash,
            output_hash=output_hash,
            metadata={"operation": operation},
        )

    # ------------------------------------------------------------------
    # Private: Completeness computation
    # ------------------------------------------------------------------

    def _compute_completeness(
        self,
        records: List[Dict[str, Any]],
    ) -> float:
        """Compute completeness score as fraction of non-null fields.

        Scans all fields across all records and counts the fraction
        that are non-null and non-empty.

        Args:
            records: List of record dictionaries.

        Returns:
            Completeness score between 0.0 and 1.0.
        """
        if not records:
            return 0.0

        total_fields = 0
        non_null_fields = 0

        for record in records:
            for key, value in record.items():
                total_fields += 1
                if value is not None and value != "":
                    if isinstance(value, float) and math.isnan(value):
                        continue
                    non_null_fields += 1

        if total_fields == 0:
            return 0.0

        return non_null_fields / total_fields

    # ------------------------------------------------------------------
    # Private: Timeliness computation
    # ------------------------------------------------------------------

    def _compute_timeliness(self, state: _SourceState) -> float:
        """Compute timeliness score based on last refresh vs cadence.

        Uses exponential decay: score = exp(-staleness / cadence).
        Sources with no last_refresh get 0.5 (neutral).

        Args:
            state: The internal source state.

        Returns:
            Timeliness score between 0.0 and 1.0.
        """
        if state.last_refresh is None:
            return 0.5

        cadence_str = state.definition.refresh_cadence
        cadence_td = CADENCE_TIMEDELTAS.get(cadence_str)

        if cadence_td is None:
            return 0.5

        now = _utcnow()
        elapsed = now - state.last_refresh
        cadence_seconds = cadence_td.total_seconds()

        if cadence_seconds <= 0:
            return 0.5

        elapsed_seconds = elapsed.total_seconds()

        # Staleness ratio: how many cadence periods have elapsed
        staleness = elapsed_seconds / cadence_seconds

        # Exponential decay: score = exp(-staleness)
        score = math.exp(-staleness)

        return _clamp(score, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Private: Consistency computation
    # ------------------------------------------------------------------

    def _compute_consistency(
        self,
        records: List[Dict[str, Any]],
    ) -> float:
        """Compute internal consistency score for a set of records.

        Evaluates consistency across two dimensions:
        1. Type uniformity: Do all values for a column share the same type?
        2. Value stability: Are values within reasonable ranges?

        Args:
            records: List of record dictionaries.

        Returns:
            Consistency score between 0.0 and 1.0.
        """
        if not records or len(records) < 2:
            return 0.5

        # Build column-level type profiles
        column_types: Dict[str, Dict[str, int]] = {}
        for record in records:
            for key, value in record.items():
                if key not in column_types:
                    column_types[key] = {}
                type_name = type(value).__name__
                column_types[key][type_name] = (
                    column_types[key].get(type_name, 0) + 1
                )

        if not column_types:
            return 0.5

        # Type uniformity: fraction of columns where dominant type
        # covers >80% of values
        type_scores: List[float] = []
        for col, type_counts in column_types.items():
            total = sum(type_counts.values())
            if total == 0:
                continue
            dominant = max(type_counts.values())
            type_scores.append(dominant / total)

        type_uniformity = _safe_mean(type_scores)

        # Value stability: check numeric columns for outlier fraction
        stability_scores: List[float] = []
        for record_set_key in column_types:
            values: List[float] = []
            for record in records:
                val = record.get(record_set_key)
                if isinstance(val, (int, float)) and not (
                    isinstance(val, float) and math.isnan(val)
                ):
                    values.append(float(val))

            if len(values) >= 3:
                stability = self._compute_value_stability(values)
                stability_scores.append(stability)

        value_stability = (
            _safe_mean(stability_scores) if stability_scores else 0.7
        )

        # Combined score (weighted)
        consistency = 0.6 * type_uniformity + 0.4 * value_stability
        return _clamp(consistency, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Private: Accuracy computation
    # ------------------------------------------------------------------

    def _compute_accuracy(
        self,
        records: List[Dict[str, Any]],
        reference_records: Optional[List[Dict[str, Any]]],
    ) -> float:
        """Compute accuracy score relative to reference (consensus) data.

        When reference records are provided, compares numeric columns
        and computes the average relative agreement. Without reference
        data, returns 0.5 (neutral).

        Args:
            records: Source records to evaluate.
            reference_records: Optional reference records for comparison.

        Returns:
            Accuracy score between 0.0 and 1.0.
        """
        if reference_records is None or not reference_records:
            return 0.5

        if not records:
            return 0.0

        # Find common numeric columns
        source_cols = set(records[0].keys()) if records else set()
        ref_cols = (
            set(reference_records[0].keys()) if reference_records else set()
        )
        common_cols = source_cols & ref_cols

        if not common_cols:
            return 0.5

        agreement_scores: List[float] = []

        for col in common_cols:
            source_vals = self._extract_numeric_values(records, col)
            ref_vals = self._extract_numeric_values(
                reference_records, col,
            )

            if not source_vals or not ref_vals:
                continue

            # Compare mean values
            source_mean = _safe_mean(source_vals)
            ref_mean = _safe_mean(ref_vals)

            if ref_mean == 0.0 and source_mean == 0.0:
                agreement_scores.append(1.0)
                continue

            if ref_mean == 0.0:
                agreement_scores.append(0.5)
                continue

            relative_error = abs(source_mean - ref_mean) / abs(ref_mean)
            agreement = max(0.0, 1.0 - relative_error)
            agreement_scores.append(agreement)

        if not agreement_scores:
            return 0.5

        return _clamp(_safe_mean(agreement_scores), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Private: Value stability helper
    # ------------------------------------------------------------------

    def _compute_value_stability(
        self,
        values: List[float],
    ) -> float:
        """Compute value stability using coefficient of variation.

        Low CV indicates stable values (high score). High CV indicates
        unstable values (lower score, floored at 0.3).

        Args:
            values: List of numeric values (at least 3).

        Returns:
            Stability score between 0.3 and 1.0.
        """
        if len(values) < 2:
            return 0.7

        mean = _safe_mean(values)
        if mean == 0.0:
            return 0.7

        variance = _safe_mean([(v - mean) ** 2 for v in values])
        std = math.sqrt(variance)
        cv = std / abs(mean)

        # Map CV to stability score
        if cv <= 0.0:
            return 1.0
        elif cv <= 1.0:
            return 1.0 - 0.5 * cv
        else:
            return max(0.3, 0.5 - 0.1 * (cv - 1.0))

    # ------------------------------------------------------------------
    # Private: Numeric value extraction
    # ------------------------------------------------------------------

    def _extract_numeric_values(
        self,
        records: List[Dict[str, Any]],
        column: str,
    ) -> List[float]:
        """Extract numeric values from a column across records.

        Args:
            records: List of record dictionaries.
            column: Column name to extract.

        Returns:
            List of float values (NaN and None excluded).
        """
        values: List[float] = []
        for record in records:
            val = record.get(column)
            if isinstance(val, (int, float)):
                if isinstance(val, float) and math.isnan(val):
                    continue
                values.append(float(val))
        return values

    # ------------------------------------------------------------------
    # Private: Column extraction from schema_info
    # ------------------------------------------------------------------

    def _extract_columns(
        self,
        schema_info: Dict[str, Any],
    ) -> List[str]:
        """Extract column names from a schema_info dictionary.

        Supports multiple schema_info formats:
        - ``{"columns": ["col1", "col2"]}``
        - ``{"fields": {"col1": {...}, "col2": {...}}}``
        - ``{"col1": "type1", "col2": "type2"}``

        Args:
            schema_info: Schema metadata dictionary.

        Returns:
            List of column name strings.
        """
        if not schema_info:
            return []

        # Format 1: explicit columns list
        if "columns" in schema_info:
            cols = schema_info["columns"]
            if isinstance(cols, list):
                return [str(c) for c in cols]

        # Format 2: fields dict
        if "fields" in schema_info:
            fields = schema_info["fields"]
            if isinstance(fields, dict):
                return list(fields.keys())

        # Format 3: direct column->type mapping (exclude metadata keys)
        metadata_keys = {
            "columns", "fields", "description", "version",
            "name", "type", "format", "schema_version",
        }
        cols = [
            k for k in schema_info.keys()
            if k not in metadata_keys
        ]
        if cols:
            return cols

        return []

    # ------------------------------------------------------------------
    # Private: Column name normalization
    # ------------------------------------------------------------------

    def _normalize_column_name(self, name: str) -> str:
        """Normalise a column name for fuzzy matching.

        Converts to lowercase, replaces common separators with
        underscores, and strips leading/trailing whitespace.

        Args:
            name: Raw column name.

        Returns:
            Normalised column name string.
        """
        result = name.lower().strip()
        for sep in ("-", ".", " ", "/"):
            result = result.replace(sep, "_")
        while "__" in result:
            result = result.replace("__", "_")
        result = result.strip("_")
        return result

    # ------------------------------------------------------------------
    # Private: Numeric normalization
    # ------------------------------------------------------------------

    def _normalize_numeric(
        self,
        value: Any,
        mapping: SchemaMapping,
    ) -> Any:
        """Apply numeric unit conversion using SUPPORTED_UNITS lookup.

        Args:
            value: Raw numeric value.
            mapping: SchemaMapping with unit_from and unit_to.

        Returns:
            Converted numeric value or original if not numeric.
        """
        if value is None:
            return None

        try:
            numeric_val = float(value)
        except (TypeError, ValueError):
            logger.debug(
                "Cannot convert to numeric: %s (column=%s)",
                value, mapping.source_column,
            )
            return value

        if math.isnan(numeric_val):
            return None

        # Look up conversion factor
        if mapping.unit_from and mapping.unit_to:
            factor = _get_conversion_factor_from_map(
                mapping.unit_from, mapping.unit_to,
            )
            if factor is not None:
                converted = numeric_val * factor
                logger.debug(
                    "Numeric normalized: %s -> %.6f (%s->%s, factor=%.6f)",
                    value, converted, mapping.unit_from,
                    mapping.unit_to, factor,
                )
                return converted

        return numeric_val

    # ------------------------------------------------------------------
    # Private: Date normalization
    # ------------------------------------------------------------------

    def _normalize_date(
        self,
        value: Any,
        mapping: SchemaMapping,
    ) -> Any:
        """Normalise date/datetime values to ISO format.

        Parses the value using the source format (date_format) and
        returns an ISO-formatted string.

        Args:
            value: Raw date/datetime value (string or datetime).
            mapping: SchemaMapping with optional date_format.

        Returns:
            ISO-formatted date string, or original value if parsing fails.
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            return value.isoformat()

        if not isinstance(value, str):
            return value

        fmt_from = mapping.date_format

        if fmt_from:
            try:
                dt = datetime.strptime(value, fmt_from)
                return dt.isoformat()
            except ValueError:
                logger.debug(
                    "Date parse failed: %s with format %s (column=%s)",
                    value, fmt_from, mapping.source_column,
                )

        # Try common formats
        common_formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%Y/%m/%d",
            "%d.%m.%Y",
        ]
        for fmt in common_formats:
            try:
                dt = datetime.strptime(value, fmt)
                return dt.isoformat()
            except ValueError:
                continue

        logger.debug(
            "Cannot parse date: %s (column=%s)",
            value, mapping.source_column,
        )
        return value

    # ------------------------------------------------------------------
    # Private: String normalization
    # ------------------------------------------------------------------

    def _normalize_string(self, value: Any) -> Any:
        """Normalise a string value by stripping whitespace.

        Args:
            value: Raw string value.

        Returns:
            Stripped string, or original value if not a string.
        """
        if isinstance(value, str):
            return value.strip()
        return value

    # ------------------------------------------------------------------
    # Private: Default tolerance rules builder
    # ------------------------------------------------------------------

    def _build_default_tolerance_rules(self) -> List[ToleranceRule]:
        """Build default ToleranceRule objects from DEFAULT_TOLERANCE_DEFS.

        Returns:
            List of ToleranceRule objects with default values.
        """
        rules: List[ToleranceRule] = []

        for rule_dict in DEFAULT_TOLERANCE_DEFS:
            field_type_str = rule_dict.get("field_type", "numeric")
            try:
                field_type_enum = FieldType(field_type_str)
            except ValueError:
                field_type_enum = FieldType.NUMERIC

            rule = ToleranceRule(
                field_name=rule_dict.get("field_name", "*"),
                field_type=field_type_enum,
                tolerance_abs=rule_dict.get("tolerance_abs"),
                tolerance_pct=rule_dict.get("tolerance_pct"),
            )
            rules.append(rule)

        return rules

    # ------------------------------------------------------------------
    # Private: Resolve enums from string
    # ------------------------------------------------------------------

    def _resolve_source_type(self, source_type: Any) -> SourceType:
        """Resolve a source_type string or enum to a SourceType enum.

        Args:
            source_type: String or SourceType enum value.

        Returns:
            SourceType enum.

        Raises:
            ValueError: If the string is not a valid SourceType value.
        """
        if isinstance(source_type, SourceType):
            return source_type

        st_lower = str(source_type).lower().strip()
        try:
            return SourceType(st_lower)
        except ValueError:
            valid = [t.value for t in SourceType]
            raise ValueError(
                f"Unrecognised source_type: {source_type!r}. "
                f"Must be one of: {valid}"
            )

    def _resolve_source_status(self, status: Any) -> SourceStatus:
        """Resolve a status string or enum to a SourceStatus enum.

        Args:
            status: String or SourceStatus enum value.

        Returns:
            SourceStatus enum.

        Raises:
            ValueError: If the string is not a valid SourceStatus value.
        """
        if isinstance(status, SourceStatus):
            return status

        st_lower = str(status).lower().strip()
        try:
            return SourceStatus(st_lower)
        except ValueError:
            valid = [s.value for s in SourceStatus]
            raise ValueError(
                f"Invalid status: {status!r}. "
                f"Must be one of: {valid}"
            )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def source_count(self) -> int:
        """Return the total number of registered sources."""
        with self._lock:
            return len(self._sources)

    @property
    def active_source_count(self) -> int:
        """Return the number of active sources."""
        with self._lock:
            return sum(
                1 for s in self._sources.values()
                if s.definition.status == SourceStatus.ACTIVE
            )

    @property
    def provenance_chain_length(self) -> int:
        """Return the number of entries in the provenance chain."""
        return self._provenance.entry_count

    def get_source_state(self, source_id: str) -> Optional[_SourceState]:
        """Return the internal source state for advanced introspection.

        Args:
            source_id: Unique identifier of the source.

        Returns:
            Internal _SourceState or None if not found.
        """
        with self._lock:
            return self._sources.get(source_id)

    def get_all_source_ids(self) -> List[str]:
        """Return all registered source IDs.

        Returns:
            List of source ID strings, sorted alphabetically.
        """
        with self._lock:
            return sorted(self._sources.keys())

    def clear(self) -> None:
        """Clear all registry state.

        Removes all sources, schema mappings, credibility caches,
        and tolerance rules. Primarily for testing.
        """
        with self._lock:
            self._sources.clear()
            self._schema_maps.clear()
            self._credibility_cache.clear()
            self._tolerance_rules.clear()

        logger.info("SourceRegistryEngine state cleared")


# ---------------------------------------------------------------------------
# __all__ export list
# ---------------------------------------------------------------------------

__all__ = ["SourceRegistryEngine"]
