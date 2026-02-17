# -*- coding: utf-8 -*-
"""
Asset Registry Engine - AGENT-DATA-018: Data Lineage Tracker

Engine 1 of 7 in the Data Lineage Tracker pipeline.

Pure-Python engine for managing data assets as lineage graph nodes. Provides
asset registration, retrieval, update, deletion (soft and hard), search with
regex pattern matching, bulk operations, import/export, and aggregate
statistics. Each asset is assigned a unique UUID and tracked with SHA-256
provenance hashing for complete audit trails.

Data assets represent the nodes in the lineage graph. They can be datasets,
fields, agents, pipelines, reports, metrics, or external sources. Each asset
carries rich metadata including qualified name, display name, owner, tags,
classification level, schema reference, and free-form metadata.

Zero-Hallucination Guarantees:
    - All IDs are deterministic UUID-4 values (no LLM involvement)
    - Timestamps from ``datetime.now(timezone.utc)`` (deterministic)
    - SHA-256 provenance hashes recorded on every mutating operation
    - Index maintenance uses explicit set operations only
    - Statistics are derived from in-memory data structures
    - No ML or LLM calls anywhere in this engine

Thread Safety:
    All mutating and read operations are protected by ``self._lock``
    (a ``threading.Lock``). Callers receive plain dict copies so they
    cannot accidentally mutate internal state.

Example:
    >>> from greenlang.data_lineage_tracker.asset_registry import (
    ...     AssetRegistryEngine,
    ... )
    >>> engine = AssetRegistryEngine()
    >>> asset = engine.register_asset(
    ...     qualified_name="emissions.scope3.spend_data",
    ...     asset_type="dataset",
    ...     display_name="Scope 3 Spend Data",
    ...     owner="data-team",
    ...     tags=["scope3", "spend"],
    ...     classification="confidential",
    ... )
    >>> print(asset["asset_id"], asset["status"])
    >>> assert engine.get_statistics()["total_assets"] == 1

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-018 Data Lineage Tracker (GL-DATA-X-021)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: config
# ---------------------------------------------------------------------------

try:
    from greenlang.data_lineage_tracker.config import get_config  # type: ignore
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    logger.info(
        "data_lineage_tracker.config not available; "
        "using default configuration values"
    )

    def get_config() -> None:  # type: ignore[misc]
        """No-op fallback when config module is not available."""
        return None

# ---------------------------------------------------------------------------
# Optional dependency: ProvenanceTracker
# ---------------------------------------------------------------------------

try:
    from greenlang.data_lineage_tracker.provenance import (
        ProvenanceTracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    logger.info(
        "data_lineage_tracker.provenance not available; "
        "using inline ProvenanceTracker"
    )

    class ProvenanceTracker:  # type: ignore[no-redef]
        """Minimal inline provenance tracker for standalone operation.

        Provides SHA-256 chain hashing without external dependencies.
        """

        GENESIS_HASH = hashlib.sha256(
            b"greenlang-data-lineage-genesis"
        ).hexdigest()

        def __init__(self) -> None:
            """Initialize with genesis hash."""
            self._last_chain_hash: str = self.GENESIS_HASH
            self._chain: List[Dict[str, Any]] = []
            self._lock = threading.Lock()

        def record(
            self,
            entity_type: str,
            entity_id: str,
            action: str,
            metadata: Optional[Any] = None,
        ) -> Any:
            """Record a provenance entry and return a stub entry.

            Args:
                entity_type: Type of entity being tracked.
                entity_id: Unique identifier for the entity.
                action: Action performed.
                metadata: Optional payload to hash.

            Returns:
                A stub object with a ``hash_value`` attribute.
            """
            ts = _utcnow().isoformat()
            if metadata is None:
                serialized = "null"
            else:
                serialized = json.dumps(metadata, sort_keys=True, default=str)
            data_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

            with self._lock:
                combined = json.dumps({
                    "action": action,
                    "data_hash": data_hash,
                    "parent_hash": self._last_chain_hash,
                    "timestamp": ts,
                }, sort_keys=True)
                chain_hash = hashlib.sha256(
                    combined.encode("utf-8"),
                ).hexdigest()

                self._chain.append({
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "action": action,
                    "hash_value": chain_hash,
                    "parent_hash": self._last_chain_hash,
                    "timestamp": ts,
                    "metadata": {"data_hash": data_hash},
                })
                self._last_chain_hash = chain_hash

            class _StubEntry:
                def __init__(self, hv: str) -> None:
                    self.hash_value = hv

            return _StubEntry(chain_hash)

        def build_hash(self, data: Any) -> str:
            """Return SHA-256 hash of JSON-serialized data."""
            return hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()

        @property
        def entry_count(self) -> int:
            """Return the total number of provenance entries."""
            with self._lock:
                return len(self._chain)

        def export_chain(self) -> List[Dict[str, Any]]:
            """Return the full provenance chain for audit."""
            with self._lock:
                return list(self._chain)

        def reset(self) -> None:
            """Clear all provenance state."""
            with self._lock:
                self._chain.clear()
                self._last_chain_hash = self.GENESIS_HASH

# ---------------------------------------------------------------------------
# Optional dependency: Prometheus metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.data_lineage_tracker.metrics import (
        record_asset_registered,
        observe_processing_duration,
        PROMETHEUS_AVAILABLE,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    logger.info(
        "data_lineage_tracker.metrics not available; "
        "asset registry metrics disabled"
    )

    def record_asset_registered(  # type: ignore[misc]
        asset_type: str,
        classification: str,
    ) -> None:
        """No-op fallback when metrics module is not available."""

    def observe_processing_duration(  # type: ignore[misc]
        operation: str,
        seconds: float,
    ) -> None:
        """No-op fallback when metrics module is not available."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid asset types for lineage graph nodes.
VALID_ASSET_TYPES: Set[str] = {
    "dataset",
    "field",
    "agent",
    "pipeline",
    "report",
    "metric",
    "external_source",
}

#: Valid data classification levels (increasing sensitivity).
VALID_CLASSIFICATIONS: Set[str] = {
    "public",
    "internal",
    "confidential",
    "restricted",
}

#: Valid asset statuses.
VALID_STATUSES: Set[str] = {
    "active",
    "deprecated",
    "archived",
}

#: Maximum number of assets importable in a single bulk call.
MAX_BULK_IMPORT: int = 5_000

#: Maximum qualified name length.
MAX_QUALIFIED_NAME_LENGTH: int = 512

#: Maximum display name length.
MAX_DISPLAY_NAME_LENGTH: int = 256

#: Maximum tag length.
MAX_TAG_LENGTH: int = 64

#: Maximum description length.
MAX_DESCRIPTION_LENGTH: int = 4_096


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
    """Build a deterministic SHA-256 hash from any JSON-serializable value.

    All dict keys are sorted for determinism regardless of insertion order.

    Args:
        data: JSON-serializable value (dict, list, str, int, etc.).

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
        if not isinstance(tag, str):
            continue
        normalized = tag.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return sorted(result)


def _validate_qualified_name(qualified_name: str) -> None:
    """Validate that a qualified name is non-empty and within length limits.

    A valid qualified name must be non-empty after stripping and not
    exceed MAX_QUALIFIED_NAME_LENGTH characters. It should follow a
    dot-separated naming convention (e.g. ``emissions.scope3.spend_data``)
    but this is not strictly enforced.

    Args:
        qualified_name: Qualified name string to validate.

    Raises:
        ValueError: If the qualified name is empty or too long.
    """
    if not qualified_name or not qualified_name.strip():
        raise ValueError("qualified_name must be a non-empty string.")
    if len(qualified_name.strip()) > MAX_QUALIFIED_NAME_LENGTH:
        raise ValueError(
            f"qualified_name exceeds maximum length of "
            f"{MAX_QUALIFIED_NAME_LENGTH} characters."
        )


def _validate_tags_list(tags: List[str]) -> None:
    """Raise ValueError if any individual tag exceeds the maximum length.

    Args:
        tags: Normalized list of tag strings.

    Raises:
        ValueError: If any tag exceeds MAX_TAG_LENGTH characters.
    """
    for tag in tags:
        if len(tag) > MAX_TAG_LENGTH:
            raise ValueError(
                f"Tag '{tag}' exceeds maximum length of "
                f"{MAX_TAG_LENGTH} characters."
            )


# ---------------------------------------------------------------------------
# AssetRegistryEngine
# ---------------------------------------------------------------------------


class AssetRegistryEngine:
    """Pure-Python engine for managing data assets as lineage graph nodes.

    Engine 1 of 7 in the Data Lineage Tracker pipeline (AGENT-DATA-018).

    Manages the full lifecycle of data assets including registration,
    retrieval by ID or qualified name, update, soft/hard deletion, search
    with regex pattern matching, bulk operations, import/export, and
    aggregate statistics. Every mutation is tracked with SHA-256 provenance
    hashing for complete audit trails.

    Assets are identified both by a unique UUID (``asset_id``) and by a
    ``qualified_name`` (e.g. ``"emissions.scope3.spend_data"``). The
    qualified name must be unique across the registry and serves as a
    human-friendly lookup key.

    Indexes are maintained for fast lookup by asset type, owner, tag, and
    qualified name. All indexes are kept consistent under the thread-safety
    lock.

    Zero-Hallucination Guarantees:
        - UUID assignment via ``uuid.uuid4()`` (no LLM involvement)
        - Timestamps from ``datetime.now(timezone.utc)`` (deterministic)
        - SHA-256 provenance hash computed from JSON-serialized payloads
        - All lookups use explicit dict/set operations
        - No ML or LLM calls anywhere in the class

    Attributes:
        _assets: Asset store keyed by asset_id (UUID string).
        _name_index: Mapping from qualified_name to asset_id for O(1)
            name-based lookup.
        _type_index: Mapping from asset_type to set of asset_ids for
            fast type-filtered queries.
        _owner_index: Mapping from owner string to set of asset_ids.
        _tag_index: Mapping from tag string to set of asset_ids.
        _lock: Thread-safety lock protecting all state.
        _provenance: ProvenanceTracker for SHA-256 audit trails.

    Example:
        >>> engine = AssetRegistryEngine()
        >>> asset = engine.register_asset(
        ...     qualified_name="supply_chain.supplier_records",
        ...     asset_type="dataset",
        ...     display_name="Supplier Records",
        ...     owner="data-team",
        ...     tags=["supplier", "core"],
        ...     classification="confidential",
        ... )
        >>> assert asset["status"] == "active"
        >>> retrieved = engine.get_asset(asset["asset_id"])
        >>> assert retrieved is not None
        >>> stats = engine.get_statistics()
        >>> assert stats["total_assets"] == 1
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize AssetRegistryEngine with empty in-memory state.

        Sets up the asset store, all lookup indexes, and the provenance
        tracker. If no ProvenanceTracker is provided, a new default
        instance is created.

        Args:
            provenance: Optional ProvenanceTracker instance. When None,
                a new tracker is created internally.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> assert engine.get_statistics()["total_assets"] == 0
        """
        self._assets: Dict[str, dict] = {}
        self._name_index: Dict[str, str] = {}
        self._type_index: Dict[str, Set[str]] = {}
        self._owner_index: Dict[str, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}
        self._lock: threading.Lock = threading.Lock()
        self._provenance: ProvenanceTracker = provenance if provenance is not None else ProvenanceTracker()

        logger.info(
            "AssetRegistryEngine initialized (AGENT-DATA-018, Engine 1 of 7)"
        )

    # ------------------------------------------------------------------
    # 1. register_asset
    # ------------------------------------------------------------------

    def register_asset(
        self,
        qualified_name: str,
        asset_type: str,
        display_name: Optional[str] = None,
        owner: str = "system",
        tags: Optional[List[str]] = None,
        classification: str = "internal",
        schema_ref: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Register a new data asset as a node in the lineage graph.

        Creates an asset record with a unique UUID, validates all input
        parameters, sets the initial status to ``"active"``, updates all
        lookup indexes, records a provenance entry, and emits a
        Prometheus metric.

        The ``qualified_name`` must be unique across the entire registry.
        Attempting to register a duplicate qualified name raises a
        ``ValueError``.

        Args:
            qualified_name: Globally unique dot-separated name for the
                asset (e.g. ``"emissions.scope3.spend_data"``). Must be
                non-empty and at most 512 characters.
            asset_type: Type classification for the asset. Must be one of:
                ``dataset``, ``field``, ``agent``, ``pipeline``,
                ``report``, ``metric``, ``external_source``.
            display_name: Optional human-readable display name. Defaults
                to the ``qualified_name`` if not provided.
            owner: Team or individual responsible for this asset.
                Defaults to ``"system"``.
            tags: Optional list of string tags for discovery and filtering.
                Tags are normalized (lowercased, stripped, deduplicated,
                sorted).
            classification: Data classification level. Must be one of:
                ``public``, ``internal``, ``confidential``, ``restricted``.
                Defaults to ``"internal"``.
            schema_ref: Optional reference to a schema definition (e.g.
                a schema registry UUID or URI).
            description: Human-readable description of the asset. Maximum
                4096 characters.
            metadata: Optional dictionary of additional metadata. Not
                validated structurally.

        Returns:
            Complete asset dict containing:

            .. code-block:: python

                {
                    "asset_id": "<uuid>",
                    "qualified_name": "...",
                    "asset_type": "dataset",
                    "display_name": "...",
                    "owner": "...",
                    "tags": [...],
                    "classification": "internal",
                    "schema_ref": None,
                    "description": "...",
                    "metadata": {...},
                    "status": "active",
                    "version": 1,
                    "created_at": "<ISO-8601>",
                    "updated_at": "<ISO-8601>",
                    "provenance_hash": "<sha256>",
                }

        Raises:
            ValueError: If ``qualified_name`` is empty, too long, or
                already exists; if ``asset_type`` is not valid; if
                ``classification`` is not valid; if any tag exceeds the
                maximum length; if ``description`` exceeds the maximum
                length.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> asset = engine.register_asset(
            ...     qualified_name="finance.erp.spend",
            ...     asset_type="dataset",
            ...     owner="finance-team",
            ...     tags=["erp", "spend"],
            ... )
            >>> assert asset["asset_id"]
            >>> assert asset["status"] == "active"
        """
        t0 = time.monotonic()

        # -- Input validation --
        _validate_qualified_name(qualified_name)
        clean_name = qualified_name.strip()

        asset_type_lower = asset_type.strip().lower()
        if asset_type_lower not in VALID_ASSET_TYPES:
            raise ValueError(
                f"Invalid asset_type: {asset_type!r}. "
                f"Must be one of: {sorted(VALID_ASSET_TYPES)}"
            )

        classification_lower = classification.strip().lower()
        if classification_lower not in VALID_CLASSIFICATIONS:
            raise ValueError(
                f"Invalid classification: {classification!r}. "
                f"Must be one of: {sorted(VALID_CLASSIFICATIONS)}"
            )

        normalized_tags = _normalize_tags(tags)
        _validate_tags_list(normalized_tags)

        if description and len(description) > MAX_DESCRIPTION_LENGTH:
            raise ValueError(
                f"Description exceeds maximum length of "
                f"{MAX_DESCRIPTION_LENGTH} characters."
            )

        clean_display = display_name.strip() if display_name else clean_name
        if clean_display and len(clean_display) > MAX_DISPLAY_NAME_LENGTH:
            raise ValueError(
                f"display_name exceeds maximum length of "
                f"{MAX_DISPLAY_NAME_LENGTH} characters."
            )

        clean_owner = owner.strip() if owner else "system"

        # -- Build asset record --
        asset_id = str(uuid.uuid4())
        now_str = _utcnow().isoformat()

        asset_record: dict = {
            "asset_id": asset_id,
            "qualified_name": clean_name,
            "asset_type": asset_type_lower,
            "display_name": clean_display,
            "owner": clean_owner,
            "tags": normalized_tags,
            "classification": classification_lower,
            "schema_ref": schema_ref,
            "description": description.strip() if description else "",
            "metadata": copy.deepcopy(metadata) if metadata else {},
            "status": "active",
            "version": 1,
            "created_at": now_str,
            "updated_at": now_str,
            "provenance_hash": "",
        }

        # -- Compute provenance hash before locking --
        data_hash = _build_sha256(
            {k: v for k, v in asset_record.items() if k != "provenance_hash"}
        )

        with self._lock:
            # Enforce qualified_name uniqueness
            if clean_name in self._name_index:
                existing_id = self._name_index[clean_name]
                existing = self._assets.get(existing_id)
                if existing and existing.get("status") != "archived":
                    raise ValueError(
                        f"An asset with qualified_name={clean_name!r} "
                        f"already exists (id={existing_id})."
                    )
                # If the existing asset is archived, allow re-registration
                # by removing the stale index entry
                del self._name_index[clean_name]

            # Record provenance
            entry = self._provenance.record(
                entity_type="lineage_asset",
                entity_id=asset_id,
                action="asset_registered",
                metadata={"data_hash": data_hash, "qualified_name": clean_name},
            )
            asset_record["provenance_hash"] = entry.hash_value

            # Store asset
            self._assets[asset_id] = asset_record

            # Update indexes
            self._name_index[clean_name] = asset_id
            self._type_index.setdefault(asset_type_lower, set()).add(asset_id)
            self._owner_index.setdefault(clean_owner, set()).add(asset_id)
            for tag in normalized_tags:
                self._tag_index.setdefault(tag, set()).add(asset_id)

        # Metrics
        elapsed = time.monotonic() - t0
        record_asset_registered(asset_type_lower, classification_lower)
        observe_processing_duration("asset_register", elapsed)

        logger.info(
            "Asset registered: id=%s qualified_name=%s type=%s "
            "classification=%s owner=%s tags=%d elapsed=%.3fms",
            asset_id,
            clean_name,
            asset_type_lower,
            classification_lower,
            clean_owner,
            len(normalized_tags),
            elapsed * 1000,
        )
        return copy.deepcopy(asset_record)

    # ------------------------------------------------------------------
    # 2. get_asset
    # ------------------------------------------------------------------

    def get_asset(self, asset_id: str) -> Optional[dict]:
        """Retrieve a registered asset by its unique UUID.

        Args:
            asset_id: UUID string assigned at registration time.

        Returns:
            Deep-copy of the asset dict, or ``None`` if no asset with
            the given ID exists.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> asset = engine.register_asset(
            ...     qualified_name="test.asset",
            ...     asset_type="dataset",
            ... )
            >>> retrieved = engine.get_asset(asset["asset_id"])
            >>> assert retrieved is not None
            >>> assert retrieved["qualified_name"] == "test.asset"
        """
        with self._lock:
            record = self._assets.get(asset_id)
            if record is None:
                logger.debug("Asset not found by id: %s", asset_id)
                return None
            return copy.deepcopy(record)

    # ------------------------------------------------------------------
    # 3. get_asset_by_name
    # ------------------------------------------------------------------

    def get_asset_by_name(self, qualified_name: str) -> Optional[dict]:
        """Retrieve a registered asset by its qualified name.

        Performs an O(1) lookup using the name index.

        Args:
            qualified_name: Dot-separated qualified name of the asset
                (e.g. ``"emissions.scope3.spend_data"``).

        Returns:
            Deep-copy of the asset dict, or ``None`` if no asset with
            the given qualified name exists.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> engine.register_asset(
            ...     qualified_name="test.asset",
            ...     asset_type="dataset",
            ... )
            >>> asset = engine.get_asset_by_name("test.asset")
            >>> assert asset is not None
            >>> assert asset["asset_type"] == "dataset"
        """
        clean_name = qualified_name.strip() if qualified_name else ""
        with self._lock:
            asset_id = self._name_index.get(clean_name)
            if asset_id is None:
                logger.debug(
                    "Asset not found by qualified_name: %s", clean_name
                )
                return None
            record = self._assets.get(asset_id)
            if record is None:
                logger.debug(
                    "Asset index inconsistency: name=%s -> id=%s not in store",
                    clean_name,
                    asset_id,
                )
                return None
            return copy.deepcopy(record)

    # ------------------------------------------------------------------
    # 4. update_asset
    # ------------------------------------------------------------------

    def update_asset(self, asset_id: str, **kwargs: Any) -> Optional[dict]:
        """Update specified fields of an existing asset.

        Only the fields explicitly provided in ``kwargs`` are updated.
        All other fields retain their current values. The internal
        version counter is incremented and a new provenance hash is
        computed.

        Supported updatable fields:
            ``display_name``, ``owner``, ``tags``, ``classification``,
            ``status``, ``description``, ``metadata``.

        When ``tags`` is updated the tag index is fully reconciled
        (stale tags removed, new tags added). When ``owner`` is updated
        the owner index is likewise reconciled.

        Args:
            asset_id: UUID of the asset to update.
            **kwargs: Field-name/value pairs to update. Only the fields
                listed above are accepted.

        Returns:
            Deep-copy of the updated asset dict, or ``None`` if the
            asset was not found.

        Raises:
            ValueError: If an unsupported field is provided, or if a
                field value fails validation (e.g. invalid classification
                or status).

        Example:
            >>> engine = AssetRegistryEngine()
            >>> asset = engine.register_asset(
            ...     qualified_name="test.asset",
            ...     asset_type="dataset",
            ... )
            >>> updated = engine.update_asset(
            ...     asset["asset_id"],
            ...     owner="new-team",
            ...     classification="confidential",
            ... )
            >>> assert updated["owner"] == "new-team"
            >>> assert updated["version"] == 2
        """
        t0 = time.monotonic()

        updatable_fields: Set[str] = {
            "display_name",
            "owner",
            "tags",
            "classification",
            "status",
            "description",
            "metadata",
        }

        invalid_keys = set(kwargs.keys()) - updatable_fields
        if invalid_keys:
            raise ValueError(
                f"Cannot update fields: {sorted(invalid_keys)}. "
                f"Updatable fields: {sorted(updatable_fields)}"
            )

        with self._lock:
            record = self._assets.get(asset_id)
            if record is None:
                logger.debug("Update failed: asset not found id=%s", asset_id)
                return None

            changes: Dict[str, Any] = {}

            for key, value in kwargs.items():
                old_value = record.get(key)

                # Validate specific fields
                if key == "classification":
                    value_lower = str(value).strip().lower()
                    if value_lower not in VALID_CLASSIFICATIONS:
                        raise ValueError(
                            f"Invalid classification: {value!r}. "
                            f"Must be one of: {sorted(VALID_CLASSIFICATIONS)}"
                        )
                    value = value_lower

                if key == "status":
                    value_lower = str(value).strip().lower()
                    if value_lower not in VALID_STATUSES:
                        raise ValueError(
                            f"Invalid status: {value!r}. "
                            f"Must be one of: {sorted(VALID_STATUSES)}"
                        )
                    value = value_lower

                if key == "display_name":
                    if value is not None:
                        value = str(value).strip()
                        if len(value) > MAX_DISPLAY_NAME_LENGTH:
                            raise ValueError(
                                f"display_name exceeds maximum length of "
                                f"{MAX_DISPLAY_NAME_LENGTH} characters."
                            )

                if key == "description":
                    if value is not None:
                        value = str(value).strip()
                        if len(value) > MAX_DESCRIPTION_LENGTH:
                            raise ValueError(
                                f"description exceeds maximum length of "
                                f"{MAX_DESCRIPTION_LENGTH} characters."
                            )

                if key == "tags":
                    if value is not None:
                        normalized = _normalize_tags(value)
                        _validate_tags_list(normalized)

                        # Reconcile tag index
                        old_tags = set(record.get("tags", []))
                        new_tags = set(normalized)

                        for removed_tag in old_tags - new_tags:
                            self._tag_index.get(removed_tag, set()).discard(
                                asset_id
                            )

                        for added_tag in new_tags - old_tags:
                            self._tag_index.setdefault(
                                added_tag, set()
                            ).add(asset_id)

                        value = normalized

                if key == "owner":
                    if value is not None:
                        value = str(value).strip()

                        # Reconcile owner index
                        old_owner = record.get("owner", "")
                        if old_owner != value:
                            self._owner_index.get(
                                old_owner, set()
                            ).discard(asset_id)
                            self._owner_index.setdefault(
                                value, set()
                            ).add(asset_id)

                if key == "metadata":
                    if value is not None and isinstance(value, dict):
                        value = copy.deepcopy(value)

                if old_value != value:
                    changes[key] = {"old": str(old_value), "new": str(value)}
                    record[key] = value

            # Bump version and update timestamp
            record["version"] = record.get("version", 1) + 1
            record["updated_at"] = _utcnow().isoformat()

            # Compute new provenance hash
            data_hash = _build_sha256(
                {k: v for k, v in record.items() if k != "provenance_hash"}
            )
            entry = self._provenance.record(
                entity_type="lineage_asset",
                entity_id=asset_id,
                action="asset_updated",
                metadata={
                    "data_hash": data_hash,
                    "changes": list(changes.keys()),
                    "version": record["version"],
                },
            )
            record["provenance_hash"] = entry.hash_value

            result = copy.deepcopy(record)

        elapsed = time.monotonic() - t0
        observe_processing_duration("asset_update", elapsed)

        logger.info(
            "Asset updated: id=%s fields=%s version=%d elapsed=%.3fms",
            asset_id,
            list(changes.keys()),
            result["version"],
            elapsed * 1000,
        )
        return result

    # ------------------------------------------------------------------
    # 5. delete_asset
    # ------------------------------------------------------------------

    def delete_asset(self, asset_id: str, hard: bool = False) -> bool:
        """Delete an asset from the registry.

        Supports both soft and hard deletion:

        - **Soft delete** (``hard=False``, default): Sets the asset status
          to ``"archived"``. The asset remains in the store and is
          retrievable, but will be excluded from most searches by default.

        - **Hard delete** (``hard=True``): Permanently removes the asset
          from the store and all indexes. The asset is no longer
          retrievable.

        In both cases a provenance entry is recorded for audit.

        Args:
            asset_id: UUID of the asset to delete.
            hard: If ``True``, permanently remove the asset. If ``False``
                (default), soft-delete by archiving.

        Returns:
            ``True`` if the asset was found and deleted (soft or hard).
            ``False`` if no asset with the given ID exists.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> asset = engine.register_asset(
            ...     qualified_name="test.delete",
            ...     asset_type="dataset",
            ... )
            >>> # Soft delete
            >>> assert engine.delete_asset(asset["asset_id"]) is True
            >>> archived = engine.get_asset(asset["asset_id"])
            >>> assert archived["status"] == "archived"
            >>> # Hard delete
            >>> assert engine.delete_asset(asset["asset_id"], hard=True) is True
            >>> assert engine.get_asset(asset["asset_id"]) is None
        """
        t0 = time.monotonic()

        with self._lock:
            record = self._assets.get(asset_id)
            if record is None:
                logger.debug(
                    "Delete failed: asset not found id=%s", asset_id
                )
                return False

            old_status = record.get("status", "active")
            qualified_name = record.get("qualified_name", "")

            if hard:
                # Hard delete: remove from all indexes and store
                self._name_index.pop(qualified_name, None)

                asset_type = record.get("asset_type", "")
                self._type_index.get(asset_type, set()).discard(asset_id)

                owner = record.get("owner", "")
                self._owner_index.get(owner, set()).discard(asset_id)

                for tag in record.get("tags", []):
                    self._tag_index.get(tag, set()).discard(asset_id)

                del self._assets[asset_id]
                action = "asset_hard_deleted"
            else:
                # Soft delete: set status to archived
                record["status"] = "archived"
                record["updated_at"] = _utcnow().isoformat()
                record["version"] = record.get("version", 1) + 1
                action = "asset_soft_deleted"

            # Record provenance
            data_hash = _build_sha256({
                "asset_id": asset_id,
                "qualified_name": qualified_name,
                "action": action,
                "old_status": old_status,
            })
            entry = self._provenance.record(
                entity_type="lineage_asset",
                entity_id=asset_id,
                action=action,
                metadata={
                    "data_hash": data_hash,
                    "qualified_name": qualified_name,
                    "hard": hard,
                },
            )

            if not hard:
                record["provenance_hash"] = entry.hash_value

        elapsed = time.monotonic() - t0
        observe_processing_duration("asset_delete", elapsed)

        logger.info(
            "Asset %s: id=%s qualified_name=%s (was %s) elapsed=%.3fms",
            "hard-deleted" if hard else "soft-deleted (archived)",
            asset_id,
            qualified_name,
            old_status,
            elapsed * 1000,
        )
        return True

    # ------------------------------------------------------------------
    # 6. search_assets
    # ------------------------------------------------------------------

    def search_assets(
        self,
        asset_type: Optional[str] = None,
        owner: Optional[str] = None,
        classification: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        name_pattern: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """Search and filter assets with optional pagination.

        All filters are applied with AND logic: an asset must satisfy
        every provided filter to be included in the result. Tags are
        also applied with AND logic -- an asset must carry all specified
        tags.

        When ``name_pattern`` is provided it is compiled as a Python
        regex and matched against the asset's ``qualified_name`` using
        ``re.search`` (case-insensitive). Invalid regex patterns are
        logged and silently ignored (no results filtered by pattern).

        Results are sorted by ``qualified_name`` ascending for
        deterministic ordering, then sliced by ``offset`` and ``limit``.

        Args:
            asset_type: Filter by exact asset type (e.g. ``"dataset"``).
                None to include all types.
            owner: Filter by exact owner (case-sensitive). None to
                include all owners.
            classification: Filter by exact classification level. None
                to include all levels.
            status: Filter by exact status. None to include all statuses.
            tags: Filter by tags with AND logic; asset must carry all
                specified tags. None to skip tag filtering.
            name_pattern: Regex pattern to match against qualified_name
                (case-insensitive ``re.search``). None to skip pattern
                filtering.
            limit: Maximum number of results to return. Defaults to 100.
            offset: Number of results to skip before returning. Defaults
                to 0.

        Returns:
            List of matching asset dicts (deep copies), sorted by
            ``qualified_name`` ascending, then paginated.

        Raises:
            ValueError: If ``limit`` or ``offset`` are negative.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> engine.register_asset(
            ...     qualified_name="a.dataset",
            ...     asset_type="dataset",
            ...     owner="team-a",
            ... )
            >>> engine.register_asset(
            ...     qualified_name="b.pipeline",
            ...     asset_type="pipeline",
            ...     owner="team-b",
            ... )
            >>> datasets = engine.search_assets(asset_type="dataset")
            >>> assert len(datasets) == 1
            >>> assert datasets[0]["asset_type"] == "dataset"
        """
        t0 = time.monotonic()

        if limit < 0:
            raise ValueError("limit must be >= 0.")
        if offset < 0:
            raise ValueError("offset must be >= 0.")

        # Compile regex pattern if provided
        compiled_pattern = None
        if name_pattern is not None:
            try:
                compiled_pattern = re.compile(name_pattern, re.IGNORECASE)
            except re.error as exc:
                logger.warning(
                    "Invalid regex pattern %r: %s; ignoring pattern filter",
                    name_pattern,
                    exc,
                )

        with self._lock:
            # Start with a candidate set, narrowing by index when possible
            candidate_ids: Optional[Set[str]] = None

            if asset_type is not None:
                type_lower = asset_type.strip().lower()
                type_ids = self._type_index.get(type_lower, set())
                candidate_ids = set(type_ids)

            if owner is not None:
                owner_ids = self._owner_index.get(owner.strip(), set())
                if candidate_ids is None:
                    candidate_ids = set(owner_ids)
                else:
                    candidate_ids &= owner_ids

            if tags is not None and tags:
                normalized_filter_tags = _normalize_tags(tags)
                for tag in normalized_filter_tags:
                    tag_ids = self._tag_index.get(tag, set())
                    if candidate_ids is None:
                        candidate_ids = set(tag_ids)
                    else:
                        candidate_ids &= tag_ids

            # Fall back to all assets if no index filter narrowed the set
            if candidate_ids is None:
                candidate_ids = set(self._assets.keys())

            # Apply remaining filters
            results: List[dict] = []
            for aid in candidate_ids:
                record = self._assets.get(aid)
                if record is None:
                    continue

                # Classification filter
                if classification is not None:
                    if record.get("classification") != classification.strip().lower():
                        continue

                # Status filter
                if status is not None:
                    if record.get("status") != status.strip().lower():
                        continue

                # Regex name pattern filter
                if compiled_pattern is not None:
                    qname = record.get("qualified_name", "")
                    if not compiled_pattern.search(qname):
                        continue

                results.append(copy.deepcopy(record))

        # Sort by qualified_name ascending
        results.sort(key=lambda r: r.get("qualified_name", ""))

        # Apply pagination
        paginated = results[offset: offset + limit] if limit > 0 else results[offset:]

        elapsed = time.monotonic() - t0
        observe_processing_duration("asset_search", elapsed)

        logger.debug(
            "search_assets: returned %d of %d matches "
            "(type=%s, owner=%s, classification=%s, status=%s, "
            "tags=%s, pattern=%s, limit=%d, offset=%d) elapsed=%.3fms",
            len(paginated),
            len(results),
            asset_type,
            owner,
            classification,
            status,
            tags,
            name_pattern,
            limit,
            offset,
            elapsed * 1000,
        )
        return paginated

    # ------------------------------------------------------------------
    # 7. list_asset_types
    # ------------------------------------------------------------------

    def list_asset_types(self) -> Dict[str, int]:
        """Return a count of assets grouped by asset type.

        The result includes only asset types that have at least one
        registered asset (types with zero assets are omitted).

        Returns:
            Dictionary mapping asset type strings to the number of
            assets of that type. Sorted alphabetically by type.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> engine.register_asset(
            ...     qualified_name="a", asset_type="dataset",
            ... )
            >>> engine.register_asset(
            ...     qualified_name="b", asset_type="dataset",
            ... )
            >>> engine.register_asset(
            ...     qualified_name="c", asset_type="pipeline",
            ... )
            >>> types = engine.list_asset_types()
            >>> assert types == {"dataset": 2, "pipeline": 1}
        """
        with self._lock:
            counts: Dict[str, int] = {}
            for asset_type, asset_ids in self._type_index.items():
                # Only count assets that are actually in the store
                active_count = sum(
                    1 for aid in asset_ids if aid in self._assets
                )
                if active_count > 0:
                    counts[asset_type] = active_count

        logger.debug("list_asset_types: %s", counts)
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------------
    # 8. list_owners
    # ------------------------------------------------------------------

    def list_owners(self) -> Dict[str, int]:
        """Return a count of assets grouped by owner.

        The result includes only owners that have at least one registered
        asset (owners with zero assets are omitted).

        Returns:
            Dictionary mapping owner strings to the number of assets
            they own. Sorted alphabetically by owner.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> engine.register_asset(
            ...     qualified_name="a", asset_type="dataset", owner="team-a",
            ... )
            >>> engine.register_asset(
            ...     qualified_name="b", asset_type="dataset", owner="team-b",
            ... )
            >>> owners = engine.list_owners()
            >>> assert owners == {"team-a": 1, "team-b": 1}
        """
        with self._lock:
            counts: Dict[str, int] = {}
            for owner, asset_ids in self._owner_index.items():
                active_count = sum(
                    1 for aid in asset_ids if aid in self._assets
                )
                if active_count > 0:
                    counts[owner] = active_count

        logger.debug("list_owners: %s", counts)
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------------
    # 9. bulk_register
    # ------------------------------------------------------------------

    def bulk_register(self, assets: List[dict]) -> dict:
        """Register multiple assets in a single batch operation.

        Each element of ``assets`` must be a dict with the keys accepted
        by ``register_asset``. At minimum ``qualified_name`` and
        ``asset_type`` are required.

        Assets that fail validation are skipped and recorded in the
        ``errors`` list of the result. Successfully registered assets
        are included in the ``registered`` count.

        Args:
            assets: List of asset dictionaries. Maximum
                ``MAX_BULK_IMPORT`` (5000) entries per call.

        Returns:
            Summary dict:

            .. code-block:: python

                {
                    "registered": 8,
                    "failed": 2,
                    "errors": [
                        {"index": 3, "qualified_name": "...", "error": "..."},
                        {"index": 7, "qualified_name": "...", "error": "..."},
                    ],
                    "asset_ids": ["uuid1", "uuid2", ...],
                    "provenance_hash": "<sha256>",
                }

        Raises:
            ValueError: If ``assets`` is empty or exceeds MAX_BULK_IMPORT.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> result = engine.bulk_register([
            ...     {"qualified_name": "a", "asset_type": "dataset"},
            ...     {"qualified_name": "b", "asset_type": "pipeline"},
            ... ])
            >>> assert result["registered"] == 2
            >>> assert result["failed"] == 0
        """
        t0 = time.monotonic()

        if not assets:
            raise ValueError("Assets list must not be empty.")
        if len(assets) > MAX_BULK_IMPORT:
            raise ValueError(
                f"Bulk register exceeds maximum of {MAX_BULK_IMPORT} assets. "
                f"Received {len(assets)}."
            )

        registered_count = 0
        failed_count = 0
        errors: List[Dict[str, Any]] = []
        asset_ids: List[str] = []

        for idx, asset_dict in enumerate(assets):
            raw_name = asset_dict.get("qualified_name", f"<unnamed-{idx}>")
            try:
                result = self.register_asset(
                    qualified_name=asset_dict.get("qualified_name", ""),
                    asset_type=asset_dict.get("asset_type", ""),
                    display_name=asset_dict.get("display_name"),
                    owner=asset_dict.get("owner", "system"),
                    tags=asset_dict.get("tags"),
                    classification=asset_dict.get("classification", "internal"),
                    schema_ref=asset_dict.get("schema_ref"),
                    description=asset_dict.get("description", ""),
                    metadata=asset_dict.get("metadata"),
                )
                asset_ids.append(result["asset_id"])
                registered_count += 1
            except Exception as exc:
                failed_count += 1
                errors.append({
                    "index": idx,
                    "qualified_name": str(raw_name),
                    "error": str(exc),
                })
                logger.warning(
                    "Bulk register: entry[%d] name=%r failed: %s",
                    idx,
                    raw_name,
                    exc,
                )

        summary_hash = _build_sha256({
            "registered": registered_count,
            "failed": failed_count,
            "asset_ids": asset_ids,
        })

        summary: dict = {
            "registered": registered_count,
            "failed": failed_count,
            "errors": errors,
            "asset_ids": asset_ids,
            "provenance_hash": summary_hash,
        }

        elapsed = time.monotonic() - t0
        observe_processing_duration("asset_bulk_register", elapsed)

        logger.info(
            "Bulk register complete: registered=%d failed=%d elapsed=%.3fms",
            registered_count,
            failed_count,
            elapsed * 1000,
        )
        return summary

    # ------------------------------------------------------------------
    # 10. export_assets
    # ------------------------------------------------------------------

    def export_assets(
        self,
        asset_type: Optional[str] = None,
    ) -> List[dict]:
        """Export assets as a list of JSON-serializable dictionaries.

        If ``asset_type`` is provided, only assets of that type are
        exported. Otherwise all assets are exported.

        Results are sorted by ``qualified_name`` ascending for
        deterministic output.

        Args:
            asset_type: Optional asset type filter. None to export all.

        Returns:
            List of asset dicts (deep copies), ordered by
            ``qualified_name`` ascending.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> engine.register_asset(
            ...     qualified_name="a", asset_type="dataset",
            ... )
            >>> exported = engine.export_assets()
            >>> assert len(exported) == 1
            >>> assert exported[0]["qualified_name"] == "a"
        """
        t0 = time.monotonic()

        with self._lock:
            if asset_type is not None:
                type_lower = asset_type.strip().lower()
                candidate_ids = self._type_index.get(type_lower, set())
                records = [
                    copy.deepcopy(self._assets[aid])
                    for aid in candidate_ids
                    if aid in self._assets
                ]
            else:
                records = [
                    copy.deepcopy(record)
                    for record in self._assets.values()
                ]

        records.sort(key=lambda r: r.get("qualified_name", ""))

        elapsed = time.monotonic() - t0
        observe_processing_duration("asset_export", elapsed)

        logger.info(
            "export_assets: exported %d assets (type=%s) elapsed=%.3fms",
            len(records),
            asset_type,
            elapsed * 1000,
        )
        return records

    # ------------------------------------------------------------------
    # 11. import_assets
    # ------------------------------------------------------------------

    def import_assets(self, assets: List[dict]) -> dict:
        """Import assets from a list of previously exported dictionaries.

        Each element must contain at minimum ``qualified_name`` and
        ``asset_type``. Assets that fail validation are skipped.

        This method delegates to ``bulk_register`` internally but
        provides a distinct API surface for symmetry with
        ``export_assets``.

        Args:
            assets: List of asset dicts to import. Maximum
                ``MAX_BULK_IMPORT`` entries per call.

        Returns:
            Summary dict with counts:

            .. code-block:: python

                {
                    "imported": 8,
                    "failed": 2,
                    "errors": [...],
                    "asset_ids": [...],
                    "provenance_hash": "<sha256>",
                }

        Raises:
            ValueError: If ``assets`` is empty or exceeds MAX_BULK_IMPORT.

        Example:
            >>> engine_a = AssetRegistryEngine()
            >>> engine_a.register_asset(
            ...     qualified_name="a", asset_type="dataset",
            ... )
            >>> exported = engine_a.export_assets()
            >>> engine_b = AssetRegistryEngine()
            >>> result = engine_b.import_assets(exported)
            >>> assert result["imported"] == 1
        """
        t0 = time.monotonic()

        if not assets:
            raise ValueError("Assets list must not be empty.")
        if len(assets) > MAX_BULK_IMPORT:
            raise ValueError(
                f"Import exceeds maximum of {MAX_BULK_IMPORT} assets. "
                f"Received {len(assets)}."
            )

        imported_count = 0
        failed_count = 0
        errors: List[Dict[str, Any]] = []
        asset_ids: List[str] = []

        for idx, asset_dict in enumerate(assets):
            raw_name = asset_dict.get(
                "qualified_name", f"<unnamed-{idx}>"
            )
            try:
                result = self.register_asset(
                    qualified_name=asset_dict.get("qualified_name", ""),
                    asset_type=asset_dict.get("asset_type", ""),
                    display_name=asset_dict.get("display_name"),
                    owner=asset_dict.get("owner", "system"),
                    tags=asset_dict.get("tags"),
                    classification=asset_dict.get(
                        "classification", "internal"
                    ),
                    schema_ref=asset_dict.get("schema_ref"),
                    description=asset_dict.get("description", ""),
                    metadata=asset_dict.get("metadata"),
                )
                asset_ids.append(result["asset_id"])
                imported_count += 1
            except Exception as exc:
                failed_count += 1
                errors.append({
                    "index": idx,
                    "qualified_name": str(raw_name),
                    "error": str(exc),
                })
                logger.warning(
                    "Import: entry[%d] name=%r failed: %s",
                    idx,
                    raw_name,
                    exc,
                )

        summary_hash = _build_sha256({
            "imported": imported_count,
            "failed": failed_count,
            "asset_ids": asset_ids,
        })

        summary: dict = {
            "imported": imported_count,
            "failed": failed_count,
            "errors": errors,
            "asset_ids": asset_ids,
            "provenance_hash": summary_hash,
        }

        elapsed = time.monotonic() - t0
        observe_processing_duration("asset_import", elapsed)

        logger.info(
            "Import complete: imported=%d failed=%d elapsed=%.3fms",
            imported_count,
            failed_count,
            elapsed * 1000,
        )
        return summary

    # ------------------------------------------------------------------
    # 12. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return a comprehensive snapshot of registry statistics.

        Computes aggregate counts across all registered assets grouped
        by type, classification, status, and owner. This method is
        read-only and never modifies state.

        Returns:
            Stats dict with the following structure:

            .. code-block:: python

                {
                    "total_assets": 42,
                    "by_type": {
                        "dataset": 20,
                        "field": 10,
                        "pipeline": 5,
                        ...
                    },
                    "by_classification": {
                        "public": 5,
                        "internal": 20,
                        "confidential": 15,
                        "restricted": 2,
                    },
                    "by_status": {
                        "active": 38,
                        "deprecated": 2,
                        "archived": 2,
                    },
                    "by_owner": {
                        "data-team": 25,
                        "finance-team": 17,
                    },
                    "total_tags": 18,
                    "provenance_entries": 85,
                }

        Example:
            >>> engine = AssetRegistryEngine()
            >>> engine.register_asset(
            ...     qualified_name="a",
            ...     asset_type="dataset",
            ...     classification="confidential",
            ... )
            >>> stats = engine.get_statistics()
            >>> assert stats["total_assets"] == 1
            >>> assert stats["by_type"]["dataset"] == 1
            >>> assert stats["by_classification"]["confidential"] == 1
        """
        with self._lock:
            total = len(self._assets)

            by_type: Dict[str, int] = {}
            by_classification: Dict[str, int] = {}
            by_status: Dict[str, int] = {}
            by_owner: Dict[str, int] = {}

            for record in self._assets.values():
                atype = record.get("asset_type", "unknown")
                by_type[atype] = by_type.get(atype, 0) + 1

                aclass = record.get("classification", "unknown")
                by_classification[aclass] = by_classification.get(aclass, 0) + 1

                astatus = record.get("status", "unknown")
                by_status[astatus] = by_status.get(astatus, 0) + 1

                aowner = record.get("owner", "unknown")
                by_owner[aowner] = by_owner.get(aowner, 0) + 1

            total_tags = len(self._tag_index)

        # Provenance entries - safe to read outside lock (tracker is thread-safe)
        prov_count = getattr(self._provenance, "entry_count", 0)

        stats: dict = {
            "total_assets": total,
            "by_type": dict(sorted(by_type.items())),
            "by_classification": dict(sorted(by_classification.items())),
            "by_status": dict(sorted(by_status.items())),
            "by_owner": dict(sorted(by_owner.items())),
            "total_tags": total_tags,
            "provenance_entries": prov_count,
        }

        logger.debug(
            "get_statistics: total=%d types=%d owners=%d tags=%d",
            total,
            len(by_type),
            len(by_owner),
            total_tags,
        )
        return stats

    # ------------------------------------------------------------------
    # 13. clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all registry state.

        Removes all assets and their indexes. Resets the provenance
        tracker to a fresh state. This method is intended for use in
        automated tests only. Do NOT call in production.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> engine.register_asset(
            ...     qualified_name="test",
            ...     asset_type="dataset",
            ... )
            >>> assert engine.get_statistics()["total_assets"] == 1
            >>> engine.clear()
            >>> assert engine.get_statistics()["total_assets"] == 0
        """
        with self._lock:
            self._assets.clear()
            self._name_index.clear()
            self._type_index.clear()
            self._owner_index.clear()
            self._tag_index.clear()

        # Reset provenance tracker
        if hasattr(self._provenance, "reset"):
            self._provenance.reset()
        else:
            self._provenance = ProvenanceTracker()

        logger.warning(
            "AssetRegistryEngine.clear() called - all registry data cleared."
        )

    # ------------------------------------------------------------------
    # Introspection helpers (properties)
    # ------------------------------------------------------------------

    @property
    def asset_count(self) -> int:
        """Return the total number of registered assets.

        Returns:
            Integer count of assets currently in the registry.
        """
        with self._lock:
            return len(self._assets)

    @property
    def provenance_chain_length(self) -> int:
        """Return the number of entries in the provenance chain.

        Returns:
            Integer count of provenance entries recorded.
        """
        return getattr(self._provenance, "entry_count", 0)

    def list_assets(self, limit: int = 10000) -> List[dict]:
        """Return all registered assets as a list of dictionaries.

        Convenience method used by LineageTrackerPipelineEngine to
        synchronise the lineage graph from the asset registry.

        Args:
            limit: Maximum number of assets to return. Defaults to 10000.

        Returns:
            List of asset dictionaries (deep copies), sorted by
            ``qualified_name``.
        """
        return self.search_assets(limit=limit)

    def get_all_asset_ids(self) -> List[str]:
        """Return all registered asset IDs.

        Returns:
            Sorted list of asset UUID strings.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> engine.register_asset(
            ...     qualified_name="a", asset_type="dataset",
            ... )
            >>> ids = engine.get_all_asset_ids()
            >>> assert len(ids) == 1
        """
        with self._lock:
            return sorted(self._assets.keys())

    def get_provenance_chain(self) -> List[Dict[str, Any]]:
        """Return the full provenance chain for audit.

        Returns:
            List of provenance entries in insertion order (oldest first).
            Each entry is a dictionary suitable for JSON serialization.

        Example:
            >>> engine = AssetRegistryEngine()
            >>> engine.register_asset(
            ...     qualified_name="a", asset_type="dataset",
            ... )
            >>> chain = engine.get_provenance_chain()
            >>> assert len(chain) >= 1
        """
        if hasattr(self._provenance, "export_chain"):
            entries = self._provenance.export_chain()
            # Convert ProvenanceEntry objects to dicts if needed
            result: List[Dict[str, Any]] = []
            for entry in entries:
                if isinstance(entry, dict):
                    result.append(entry)
                elif hasattr(entry, "to_dict"):
                    result.append(entry.to_dict())
                else:
                    result.append({"hash_value": str(entry)})
            return result
        return []


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = ["AssetRegistryEngine"]
