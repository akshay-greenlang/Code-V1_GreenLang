"""
PACK-050 GHG Consolidation Pack - Entity Registry Engine
====================================================================

Manages the authoritative registry of all corporate entities
(subsidiaries, joint ventures, associates, divisions, SPVs,
franchises, branches) within a multi-entity corporate structure.
Provides lifecycle management, hierarchical tree construction,
entity search/filtering, and structural validation.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 3): Setting
      Organizational Boundaries requires a complete enumeration
      of all entities in the corporate structure.
    - IAS 27 / IFRS 10: Consolidated Financial Statements -
      defines subsidiary, associate, and JV relationships.
    - GHG Protocol Corporate Standard (Chapter 5): Tracking
      emissions over time requires entity lifecycle tracking.
    - ESRS E1-6: Gross Scopes 1, 2 and 3 require entity-level
      disaggregation for consolidation.

Capabilities:
    - Register entities with full metadata (legal name,
      jurisdiction, incorporation date, LEI, ISIN, sector code)
    - Entity lifecycle management (active, dormant, acquired,
      divested, merged, liquidated)
    - Hierarchical tree structure with parent-child relationships
    - Entity search and filtering by any attribute
    - Tree traversal (ancestors, descendants, siblings)
    - Hierarchy validation (no circular references, valid parents)
    - Registry statistics and analytics

Zero-Hallucination:
    - All aggregations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation or classification path
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-050 GHG Consolidation
Engine:  1 of 5
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash, excluding volatile fields."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("created_at", "updated_at", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert any value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> Decimal:
    """Round a value to two decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EntityType(str, Enum):
    """Types of corporate entities in a GHG reporting structure."""
    SUBSIDIARY = "SUBSIDIARY"
    JOINT_VENTURE = "JOINT_VENTURE"
    ASSOCIATE = "ASSOCIATE"
    DIVISION = "DIVISION"
    SPV = "SPV"
    FRANCHISE = "FRANCHISE"
    BRANCH = "BRANCH"
    HOLDING_COMPANY = "HOLDING_COMPANY"
    PARENT = "PARENT"
    OTHER = "OTHER"


class EntityStatus(str, Enum):
    """Entity lifecycle statuses."""
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    ACQUIRED = "ACQUIRED"
    DIVESTED = "DIVESTED"
    MERGED = "MERGED"
    LIQUIDATED = "LIQUIDATED"
    UNDER_FORMATION = "UNDER_FORMATION"
    SUSPENDED = "SUSPENDED"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class EntityRecord(BaseModel):
    """Represents a single corporate entity in the registry.

    An entity is any legal or operational unit (subsidiary, JV,
    associate, division, SPV, franchise, branch) that may be
    part of the GHG reporting boundary.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    entity_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for the entity.",
    )
    legal_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Official legal name of the entity.",
    )
    trading_name: Optional[str] = Field(
        None,
        max_length=500,
        description="Trading or brand name if different from legal name.",
    )
    entity_type: str = Field(
        ...,
        description="Type of corporate entity (maps to EntityType enum).",
    )
    status: str = Field(
        default="ACTIVE",
        description="Current lifecycle status of the entity.",
    )
    jurisdiction: Optional[str] = Field(
        None,
        max_length=100,
        description="Legal jurisdiction (e.g. 'Delaware, US', 'England and Wales').",
    )
    country: Optional[str] = Field(
        None,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code.",
    )
    incorporation_date: Optional[date] = Field(
        None,
        description="Date the entity was incorporated or formed.",
    )
    lei: Optional[str] = Field(
        None,
        min_length=20,
        max_length=20,
        description="Legal Entity Identifier (20-character alphanumeric).",
    )
    isin: Optional[str] = Field(
        None,
        min_length=12,
        max_length=12,
        description="International Securities Identification Number.",
    )
    sector_code: Optional[str] = Field(
        None,
        max_length=20,
        description="Industry sector code (e.g. NACE, SIC, GICS).",
    )
    sector_name: Optional[str] = Field(
        None,
        max_length=255,
        description="Human-readable sector name.",
    )
    parent_entity_id: Optional[str] = Field(
        None,
        description="ID of the parent entity (None for top-level).",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Freeform tags for filtering.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the entity record was created.",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the entity record was last updated.",
    )

    @field_validator("entity_type")
    @classmethod
    def _validate_entity_type(cls, v: str) -> str:
        """Validate entity_type is a recognised value."""
        valid = {et.value for et in EntityType}
        if v.upper() not in valid:
            logger.warning(
                "Entity type '%s' not in standard taxonomy; accepted as OTHER.",
                v,
            )
            return "OTHER"
        return v.upper()

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        """Validate status is a recognised value."""
        valid = {es.value for es in EntityStatus}
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid entity status '{v}'. Must be one of {sorted(valid)}."
            )
        return v.upper()


class EntityHierarchy(BaseModel):
    """Represents the full hierarchical tree of corporate entities.

    Contains the root entity and all descendants organised as
    a tree structure for traversal and analysis.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    hierarchy_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this hierarchy snapshot.",
    )
    root_entity_id: str = Field(
        ...,
        description="ID of the top-level (root) entity.",
    )
    tree: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Adjacency list: parent_id -> [child_ids].",
    )
    total_entities: int = Field(
        default=0,
        description="Total entities in the hierarchy.",
    )
    max_depth: int = Field(
        default=0,
        description="Maximum depth of the tree.",
    )
    entity_type_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of entities by type.",
    )
    status_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of entities by status.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of this snapshot.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )


class EntitySearchResult(BaseModel):
    """Result of an entity search operation."""
    model_config = ConfigDict(validate_default=True)

    query: Dict[str, Any] = Field(
        default_factory=dict,
        description="The search criteria used.",
    )
    total_results: int = Field(
        default=0,
        description="Number of matching entities.",
    )
    entities: List[EntityRecord] = Field(
        default_factory=list,
        description="Matching entity records.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )


class EntityRegistryStats(BaseModel):
    """Aggregate statistics for the entity registry."""
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    total_entities: int = Field(
        default=0,
        description="Total number of entities.",
    )
    active_entities: int = Field(
        default=0,
        description="Number of active entities.",
    )
    entity_type_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of entities by type.",
    )
    status_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of entities by status.",
    )
    countries_covered: int = Field(
        default=0,
        description="Number of distinct countries.",
    )
    country_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of entities by country.",
    )
    entities_with_lei: int = Field(
        default=0,
        description="Entities that have an LEI.",
    )
    top_level_entities: int = Field(
        default=0,
        description="Entities with no parent.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of this snapshot.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EntityRegistryEngine:
    """Manages the authoritative corporate entity registry.

    Provides CRUD operations for entity records, hierarchical
    tree construction, search/filtering, lifecycle management,
    and structural validation. All aggregations use deterministic
    arithmetic with SHA-256 provenance hashing on every result.

    Attributes:
        _entities: Internal dict mapping entity_id to EntityRecord.
        _change_log: Append-only list of change events.

    Example:
        >>> engine = EntityRegistryEngine()
        >>> entity = engine.register_entity({
        ...     "legal_name": "Acme Corp",
        ...     "entity_type": "PARENT",
        ...     "jurisdiction": "Delaware, US",
        ...     "country": "US",
        ... })
        >>> assert entity.status == "ACTIVE"
    """

    def __init__(self) -> None:
        """Initialise the EntityRegistryEngine with empty state."""
        self._entities: Dict[str, EntityRecord] = {}
        self._change_log: List[Dict[str, Any]] = []
        logger.info("EntityRegistryEngine v%s initialised.", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def register_entity(self, entity_data: Dict[str, Any]) -> EntityRecord:
        """Register a new corporate entity in the registry.

        Validates the incoming data, assigns a unique entity_id if
        not provided, and stores the record.

        Args:
            entity_data: Dictionary of entity attributes. Must include
                at minimum: legal_name, entity_type.

        Returns:
            The created EntityRecord with generated entity_id.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        logger.info(
            "Registering entity '%s'.",
            entity_data.get("legal_name", "N/A"),
        )

        if "entity_id" not in entity_data or not entity_data["entity_id"]:
            entity_data["entity_id"] = _new_uuid()

        # Validate parent exists if specified
        parent_id = entity_data.get("parent_entity_id")
        if parent_id and parent_id not in self._entities:
            raise ValueError(
                f"Parent entity '{parent_id}' not found in registry."
            )

        now = _utcnow()
        entity_data["created_at"] = now
        entity_data["updated_at"] = now

        entity = EntityRecord(**entity_data)
        self._entities[entity.entity_id] = entity

        self._change_log.append({
            "event": "ENTITY_REGISTERED",
            "entity_id": entity.entity_id,
            "legal_name": entity.legal_name,
            "entity_type": entity.entity_type,
            "timestamp": now.isoformat(),
        })

        logger.info(
            "Entity '%s' registered (id=%s, type=%s).",
            entity.legal_name,
            entity.entity_id,
            entity.entity_type,
        )
        return entity

    def update_entity(
        self, entity_id: str, updates: Dict[str, Any],
    ) -> EntityRecord:
        """Update an existing entity record.

        Applies partial updates to the entity fields. Immutable
        fields (entity_id, created_at) are silently ignored.

        Args:
            entity_id: The ID of the entity to update.
            updates: Dictionary of fields to update.

        Returns:
            The updated EntityRecord.

        Raises:
            KeyError: If the entity_id does not exist.
            ValueError: If update values are invalid.
        """
        if entity_id not in self._entities:
            raise KeyError(f"Entity '{entity_id}' not found in registry.")

        entity = self._entities[entity_id]
        logger.info("Updating entity '%s' (id=%s).", entity.legal_name, entity_id)

        immutable_fields = {"entity_id", "created_at"}
        current_data = entity.model_dump()
        changes_applied: Dict[str, Any] = {}

        for key, value in updates.items():
            if key in immutable_fields:
                logger.warning("Ignoring immutable field '%s' in update.", key)
                continue
            if key in current_data:
                old_value = current_data[key]
                current_data[key] = value
                changes_applied[key] = {"old": old_value, "new": value}
            else:
                logger.warning("Unknown field '%s' ignored in update.", key)

        current_data["updated_at"] = _utcnow()
        updated_entity = EntityRecord(**current_data)
        self._entities[entity_id] = updated_entity

        self._change_log.append({
            "event": "ENTITY_UPDATED",
            "entity_id": entity_id,
            "changes": changes_applied,
            "timestamp": _utcnow().isoformat(),
        })

        logger.info(
            "Entity '%s' updated with %d field(s).",
            entity_id, len(changes_applied),
        )
        return updated_entity

    def deactivate_entity(
        self,
        entity_id: str,
        new_status: str,
        reason: str,
    ) -> EntityRecord:
        """Deactivate an entity by changing its lifecycle status.

        Sets the entity status to the specified non-active status
        (DORMANT, DIVESTED, MERGED, LIQUIDATED, SUSPENDED).

        Args:
            entity_id: The entity to deactivate.
            new_status: Target status (must be non-active).
            reason: Reason for deactivation (audit trail).

        Returns:
            The updated EntityRecord.

        Raises:
            KeyError: If entity not found.
            ValueError: If new_status is ACTIVE or invalid.
        """
        if entity_id not in self._entities:
            raise KeyError(f"Entity '{entity_id}' not found in registry.")

        new_status_upper = new_status.upper()
        if new_status_upper == EntityStatus.ACTIVE.value:
            raise ValueError("Cannot deactivate to ACTIVE status.")

        valid = {es.value for es in EntityStatus}
        if new_status_upper not in valid:
            raise ValueError(
                f"Invalid status '{new_status}'. Must be one of {sorted(valid)}."
            )

        entity = self._entities[entity_id]
        logger.info(
            "Deactivating entity '%s' (id=%s) to %s: %s.",
            entity.legal_name, entity_id, new_status_upper, reason,
        )

        updated = self.update_entity(entity_id, {"status": new_status_upper})

        self._change_log.append({
            "event": "ENTITY_DEACTIVATED",
            "entity_id": entity_id,
            "new_status": new_status_upper,
            "reason": reason,
            "timestamp": _utcnow().isoformat(),
        })

        return updated

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def get_entity_tree(
        self,
        root_entity_id: Optional[str] = None,
    ) -> EntityHierarchy:
        """Build the hierarchical tree of corporate entities.

        Constructs an adjacency-list representation of the entity
        hierarchy, computes depth, and provides distribution analytics.

        Args:
            root_entity_id: Optional root entity. If None, auto-detects
                the entity with no parent.

        Returns:
            EntityHierarchy with tree structure and analytics.

        Raises:
            ValueError: If no root entity can be determined.
        """
        logger.info("Building entity hierarchy tree.")

        # Determine root
        if root_entity_id is None:
            roots = [
                e for e in self._entities.values()
                if e.parent_entity_id is None
            ]
            if not roots:
                raise ValueError("No root entity found (all entities have a parent).")
            root_entity_id = roots[0].entity_id

        if root_entity_id not in self._entities:
            raise KeyError(f"Root entity '{root_entity_id}' not found.")

        # Build adjacency list
        tree: Dict[str, List[str]] = {}
        for entity in self._entities.values():
            parent = entity.parent_entity_id or "__ROOT__"
            if parent not in tree:
                tree[parent] = []
            tree[parent].append(entity.entity_id)

        # Compute max depth via BFS
        max_depth = self._compute_tree_depth(root_entity_id, tree)

        # Distribution analytics
        type_dist: Dict[str, int] = {}
        status_dist: Dict[str, int] = {}
        for entity in self._entities.values():
            etype = entity.entity_type
            type_dist[etype] = type_dist.get(etype, 0) + 1
            estatus = entity.status
            status_dist[estatus] = status_dist.get(estatus, 0) + 1

        result = EntityHierarchy(
            root_entity_id=root_entity_id,
            tree=tree,
            total_entities=len(self._entities),
            max_depth=max_depth,
            entity_type_distribution=type_dist,
            status_distribution=status_dist,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Entity hierarchy built: %d entities, max depth=%d.",
            result.total_entities, max_depth,
        )
        return result

    def _compute_tree_depth(
        self,
        root_id: str,
        tree: Dict[str, List[str]],
    ) -> int:
        """Compute the maximum depth of the entity tree via BFS.

        Args:
            root_id: Root entity ID.
            tree: Adjacency list mapping parent to children.

        Returns:
            Maximum depth (root = depth 0).
        """
        max_depth = 0
        queue: List[Tuple[str, int]] = [(root_id, 0)]
        visited: Set[str] = set()

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            max_depth = max(max_depth, depth)

            children = tree.get(current_id, [])
            for child_id in children:
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        return max_depth

    def get_ancestors(self, entity_id: str) -> List[EntityRecord]:
        """Get all ancestor entities from entity up to the root.

        Args:
            entity_id: The entity to trace ancestors for.

        Returns:
            List of ancestors from immediate parent to root.

        Raises:
            KeyError: If entity not found.
        """
        if entity_id not in self._entities:
            raise KeyError(f"Entity '{entity_id}' not found.")

        ancestors: List[EntityRecord] = []
        visited: Set[str] = set()
        current = self._entities[entity_id]

        while current.parent_entity_id is not None:
            if current.parent_entity_id in visited:
                logger.error(
                    "Circular reference detected at entity '%s'.",
                    current.parent_entity_id,
                )
                break
            visited.add(current.parent_entity_id)
            parent = self._entities.get(current.parent_entity_id)
            if parent is None:
                break
            ancestors.append(parent)
            current = parent

        return ancestors

    def get_descendants(self, entity_id: str) -> List[EntityRecord]:
        """Get all descendant entities below the given entity.

        Args:
            entity_id: The entity to find descendants for.

        Returns:
            List of all descendant entities (breadth-first order).

        Raises:
            KeyError: If entity not found.
        """
        if entity_id not in self._entities:
            raise KeyError(f"Entity '{entity_id}' not found.")

        # Build parent-to-children map
        children_map: Dict[str, List[str]] = {}
        for entity in self._entities.values():
            if entity.parent_entity_id:
                if entity.parent_entity_id not in children_map:
                    children_map[entity.parent_entity_id] = []
                children_map[entity.parent_entity_id].append(entity.entity_id)

        descendants: List[EntityRecord] = []
        queue: List[str] = list(children_map.get(entity_id, []))
        visited: Set[str] = {entity_id}

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)
            if current_id in self._entities:
                descendants.append(self._entities[current_id])
            for child in children_map.get(current_id, []):
                if child not in visited:
                    queue.append(child)

        return descendants

    def get_children(self, entity_id: str) -> List[EntityRecord]:
        """Get the direct children of an entity.

        Args:
            entity_id: The parent entity ID.

        Returns:
            List of direct child EntityRecords.

        Raises:
            KeyError: If entity not found.
        """
        if entity_id not in self._entities:
            raise KeyError(f"Entity '{entity_id}' not found.")

        return [
            e for e in self._entities.values()
            if e.parent_entity_id == entity_id
        ]

    def get_siblings(self, entity_id: str) -> List[EntityRecord]:
        """Get sibling entities (same parent, excluding self).

        Args:
            entity_id: The entity to find siblings for.

        Returns:
            List of sibling EntityRecords.

        Raises:
            KeyError: If entity not found.
        """
        if entity_id not in self._entities:
            raise KeyError(f"Entity '{entity_id}' not found.")

        entity = self._entities[entity_id]
        parent_id = entity.parent_entity_id

        return [
            e for e in self._entities.values()
            if e.parent_entity_id == parent_id and e.entity_id != entity_id
        ]

    # ------------------------------------------------------------------
    # Search & Filter
    # ------------------------------------------------------------------

    def search_entities(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> EntitySearchResult:
        """Search entities by arbitrary criteria.

        Supported filter keys:
            - legal_name: str (substring match, case-insensitive)
            - entity_type: str or List[str]
            - status: str or List[str]
            - country: str or List[str]
            - jurisdiction: str (substring match)
            - parent_entity_id: str
            - has_lei: bool
            - sector_code: str
            - tag: str (matches if tag is in entity.tags)

        Args:
            filters: Dictionary of filter criteria.

        Returns:
            EntitySearchResult with matching entities.
        """
        if not filters:
            entities = list(self._entities.values())
        else:
            entities = [
                e for e in self._entities.values()
                if self._matches_filters(e, filters)
            ]

        result = EntitySearchResult(
            query=filters or {},
            total_results=len(entities),
            entities=entities,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Entity search returned %d result(s) for filters=%s.",
            len(entities), filters,
        )
        return result

    def _matches_filters(
        self,
        entity: EntityRecord,
        filters: Dict[str, Any],
    ) -> bool:
        """Check if an entity matches all filter criteria.

        Args:
            entity: The entity to check.
            filters: The filter criteria.

        Returns:
            True if all filters match.
        """
        for key, value in filters.items():
            if key == "legal_name":
                if value.lower() not in entity.legal_name.lower():
                    return False

            elif key == "entity_type":
                acceptable = value if isinstance(value, list) else [value]
                acceptable_upper = [v.upper() for v in acceptable]
                if entity.entity_type.upper() not in acceptable_upper:
                    return False

            elif key == "status":
                acceptable = value if isinstance(value, list) else [value]
                acceptable_upper = [v.upper() for v in acceptable]
                if entity.status.upper() not in acceptable_upper:
                    return False

            elif key == "country":
                acceptable = value if isinstance(value, list) else [value]
                if entity.country not in acceptable:
                    return False

            elif key == "jurisdiction":
                if entity.jurisdiction is None:
                    return False
                if value.lower() not in entity.jurisdiction.lower():
                    return False

            elif key == "parent_entity_id":
                if entity.parent_entity_id != value:
                    return False

            elif key == "has_lei":
                has_lei = entity.lei is not None
                if has_lei != value:
                    return False

            elif key == "sector_code":
                if entity.sector_code != value:
                    return False

            elif key == "tag":
                if value not in entity.tags:
                    return False

            else:
                logger.warning("Unknown filter key '%s' ignored.", key)

        return True

    # ------------------------------------------------------------------
    # Hierarchy Validation
    # ------------------------------------------------------------------

    def validate_hierarchy(self) -> Dict[str, Any]:
        """Validate the entity hierarchy for structural integrity.

        Checks:
            1. No circular references in parent chains
            2. All parent_entity_id values reference existing entities
            3. Exactly one root entity (no parent)
            4. No orphaned subtrees

        Returns:
            Dictionary with validation results including:
                is_valid, errors, warnings, stats.
        """
        logger.info("Validating entity hierarchy.")

        errors: List[str] = []
        warnings: List[str] = []

        # Check 1: Validate parent references exist
        for entity in self._entities.values():
            if entity.parent_entity_id is not None:
                if entity.parent_entity_id not in self._entities:
                    errors.append(
                        f"Entity '{entity.entity_id}' references non-existent "
                        f"parent '{entity.parent_entity_id}'."
                    )

        # Check 2: Detect circular references
        for entity in self._entities.values():
            visited: Set[str] = set()
            current = entity
            while current.parent_entity_id is not None:
                if current.entity_id in visited:
                    errors.append(
                        f"Circular reference detected involving "
                        f"entity '{current.entity_id}'."
                    )
                    break
                visited.add(current.entity_id)
                parent = self._entities.get(current.parent_entity_id)
                if parent is None:
                    break
                current = parent

        # Check 3: Count root entities
        roots = [
            e for e in self._entities.values()
            if e.parent_entity_id is None
        ]
        if len(roots) == 0:
            errors.append("No root entity found (all entities have a parent).")
        elif len(roots) > 1:
            root_names = [r.legal_name for r in roots]
            warnings.append(
                f"Multiple root entities found: {root_names}. "
                f"Expected exactly one."
            )

        # Check 4: Detect orphaned subtrees
        if roots:
            reachable: Set[str] = set()
            for root in roots:
                self._collect_reachable(root.entity_id, reachable)
            unreachable = set(self._entities.keys()) - reachable
            if unreachable:
                warnings.append(
                    f"{len(unreachable)} entity(ies) not reachable from "
                    f"any root: {sorted(list(unreachable)[:5])}."
                )

        is_valid = len(errors) == 0

        result = {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "total_entities": len(self._entities),
            "root_count": len(roots),
            "provenance_hash": _compute_hash({
                "errors": errors,
                "warnings": warnings,
            }),
        }

        logger.info(
            "Hierarchy validation: valid=%s, %d error(s), %d warning(s).",
            is_valid, len(errors), len(warnings),
        )
        return result

    def _collect_reachable(
        self,
        entity_id: str,
        visited: Set[str],
    ) -> None:
        """Collect all entities reachable from the given entity via DFS.

        Args:
            entity_id: Starting entity.
            visited: Set to accumulate reachable entity IDs.
        """
        if entity_id in visited:
            return
        visited.add(entity_id)
        for entity in self._entities.values():
            if entity.parent_entity_id == entity_id:
                self._collect_reachable(entity.entity_id, visited)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_registry_stats(self) -> EntityRegistryStats:
        """Generate aggregate statistics for the entity registry.

        Returns:
            EntityRegistryStats with counts and distributions.
        """
        logger.info("Generating entity registry statistics.")

        entities = list(self._entities.values())
        active_count = sum(1 for e in entities if e.status == EntityStatus.ACTIVE.value)

        type_dist: Dict[str, int] = {}
        status_dist: Dict[str, int] = {}
        country_dist: Dict[str, int] = {}
        lei_count = 0
        top_level_count = 0

        for entity in entities:
            etype = entity.entity_type
            type_dist[etype] = type_dist.get(etype, 0) + 1

            estatus = entity.status
            status_dist[estatus] = status_dist.get(estatus, 0) + 1

            if entity.country:
                country_dist[entity.country] = country_dist.get(entity.country, 0) + 1

            if entity.lei is not None:
                lei_count += 1

            if entity.parent_entity_id is None:
                top_level_count += 1

        result = EntityRegistryStats(
            total_entities=len(entities),
            active_entities=active_count,
            entity_type_distribution=type_dist,
            status_distribution=status_dist,
            countries_covered=len(country_dist),
            country_distribution=country_dist,
            entities_with_lei=lei_count,
            top_level_entities=top_level_count,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Registry stats: %d total, %d active, %d countries.",
            result.total_entities, result.active_entities,
            result.countries_covered,
        )
        return result

    # ------------------------------------------------------------------
    # Accessors & Utilities
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> EntityRecord:
        """Retrieve an entity by ID.

        Args:
            entity_id: The entity ID.

        Returns:
            The EntityRecord.

        Raises:
            KeyError: If not found.
        """
        if entity_id not in self._entities:
            raise KeyError(f"Entity '{entity_id}' not found.")
        return self._entities[entity_id]

    def get_all_entities(self) -> List[EntityRecord]:
        """Return all entities in the registry.

        Returns:
            List of all EntityRecords.
        """
        return list(self._entities.values())

    def get_active_entities(self) -> List[EntityRecord]:
        """Return only active entities.

        Returns:
            List of active EntityRecords.
        """
        return [
            e for e in self._entities.values()
            if e.status == EntityStatus.ACTIVE.value
        ]

    def get_entity_count(self) -> int:
        """Return total number of entities in registry.

        Returns:
            Integer count of all entities.
        """
        return len(self._entities)

    def get_change_log(self) -> List[Dict[str, Any]]:
        """Return the complete change log.

        Returns:
            List of change log entries in chronological order.
        """
        return list(self._change_log)

    def export_registry(self) -> Dict[str, Any]:
        """Export the entire registry as a serialisable dictionary.

        Returns:
            Dictionary with entities and metadata.
        """
        entities_data = [
            e.model_dump(mode="json") for e in self._entities.values()
        ]

        export = {
            "version": _MODULE_VERSION,
            "exported_at": _utcnow().isoformat(),
            "total_entities": len(self._entities),
            "entities": entities_data,
        }
        export["provenance_hash"] = _compute_hash(export)
        return export

    def import_entities(self, entity_dicts: List[Dict[str, Any]]) -> int:
        """Bulk import entities from a list of dictionaries.

        Skips entities whose entity_id already exists. Returns the
        number of entities successfully imported.

        Args:
            entity_dicts: List of entity data dictionaries.

        Returns:
            Number of entities imported.
        """
        imported = 0
        existing_ids = set(self._entities.keys())

        for ed in entity_dicts:
            eid = ed.get("entity_id", "")
            if eid in existing_ids:
                logger.warning("Skipping duplicate entity_id '%s'.", eid)
                continue
            try:
                entity = self.register_entity(ed)
                existing_ids.add(entity.entity_id)
                imported += 1
            except (ValueError, TypeError) as exc:
                logger.error(
                    "Failed to import entity '%s': %s.",
                    ed.get("legal_name", "N/A"), exc,
                )

        logger.info("Imported %d of %d entities.", imported, len(entity_dicts))
        return imported
