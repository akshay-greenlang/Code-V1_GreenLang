"""
Event Versioning for GreenLang
==============================

TASK-130: Event Versioning Implementation

This module provides schema versioning and migration capabilities for events,
enabling backward compatibility and smooth schema evolution.

Features:
- Schema version management
- Backward compatibility checking
- Event upgrade/downgrade transformers
- Version negotiation between producers/consumers
- Schema registry integration
- Migration path documentation generator

Example:
    >>> from greenlang.infrastructure.events import EventVersionManager, SchemaRegistry
    >>> registry = SchemaRegistry()
    >>> manager = EventVersionManager(registry)
    >>>
    >>> # Register schema versions
    >>> manager.register_schema("EmissionCalculated", v1_schema)
    >>> manager.register_schema("EmissionCalculated", v2_schema)
    >>>
    >>> # Transform events between versions
    >>> v2_event = await manager.upgrade(v1_event, target_version=2)

Author: GreenLang Infrastructure Team
Created: 2025-12-07
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator

from greenlang.infrastructure.events.event_schema import BaseEvent, EventMetadata

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class CompatibilityMode(str, Enum):
    """Schema compatibility modes."""
    NONE = "none"                     # No compatibility checking
    BACKWARD = "backward"             # New schema can read old data
    FORWARD = "forward"               # Old schema can read new data
    FULL = "full"                     # Both backward and forward
    BACKWARD_TRANSITIVE = "backward_transitive"  # All previous versions
    FORWARD_TRANSITIVE = "forward_transitive"    # All future versions
    FULL_TRANSITIVE = "full_transitive"          # All versions


class FieldChange(str, Enum):
    """Types of field changes."""
    ADDED = "added"
    REMOVED = "removed"
    RENAMED = "renamed"
    TYPE_CHANGED = "type_changed"
    OPTIONAL_CHANGED = "optional_changed"
    DEFAULT_CHANGED = "default_changed"


class MigrationDirection(str, Enum):
    """Migration direction."""
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"


# =============================================================================
# Schema Models
# =============================================================================


class SchemaField(BaseModel):
    """Definition of a schema field."""
    name: str = Field(..., description="Field name")
    field_type: str = Field(..., description="Field type (string, int, float, bool, object, array)")
    required: bool = Field(default=True, description="Whether field is required")
    default: Optional[Any] = Field(default=None, description="Default value")
    description: str = Field(default="", description="Field description")
    deprecated: bool = Field(default=False)
    deprecated_since: Optional[int] = Field(default=None)
    replaced_by: Optional[str] = Field(default=None)

    def to_avro_field(self) -> Dict[str, Any]:
        """Convert to Avro field definition."""
        avro_type = self._to_avro_type()

        if not self.required:
            avro_type = ["null", avro_type]

        field_def = {
            "name": self.name,
            "type": avro_type,
        }

        if self.default is not None:
            field_def["default"] = self.default
        elif not self.required:
            field_def["default"] = None

        if self.description:
            field_def["doc"] = self.description

        return field_def

    def _to_avro_type(self) -> Union[str, Dict]:
        """Convert field type to Avro type."""
        type_map = {
            "string": "string",
            "int": "int",
            "long": "long",
            "float": "float",
            "double": "double",
            "bool": "boolean",
            "boolean": "boolean",
            "bytes": "bytes",
        }
        return type_map.get(self.field_type, "string")


class EventSchema(BaseModel):
    """Schema definition for an event type."""
    schema_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str = Field(..., description="Event type identifier")
    version: int = Field(..., ge=1, description="Schema version")
    namespace: str = Field(default="com.greenlang.events")
    description: str = Field(default="")
    fields: List[SchemaField] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(default=None)
    deprecated: bool = Field(default=False)
    deprecated_message: Optional[str] = Field(default=None)
    compatibility_mode: CompatibilityMode = Field(default=CompatibilityMode.BACKWARD)
    provenance_hash: str = Field(default="")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate schema hash."""
        data = {
            "event_type": self.event_type,
            "version": self.version,
            "fields": [f.dict() for f in self.fields]
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def to_avro_schema(self) -> Dict[str, Any]:
        """Convert to Avro schema format."""
        return {
            "type": "record",
            "name": self.event_type.replace(".", "_"),
            "namespace": self.namespace,
            "doc": self.description,
            "fields": [f.to_avro_field() for f in self.fields]
        }

    def get_field(self, name: str) -> Optional[SchemaField]:
        """Get a field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def get_required_fields(self) -> List[SchemaField]:
        """Get all required fields."""
        return [f for f in self.fields if f.required]

    def get_optional_fields(self) -> List[SchemaField]:
        """Get all optional fields."""
        return [f for f in self.fields if not f.required]


class SchemaChange(BaseModel):
    """Represents a change between schema versions."""
    change_type: FieldChange = Field(..., description="Type of change")
    field_name: str = Field(..., description="Affected field name")
    old_value: Optional[Any] = Field(default=None)
    new_value: Optional[Any] = Field(default=None)
    breaking: bool = Field(default=False, description="Whether change is breaking")
    migration_hint: Optional[str] = Field(default=None)


class SchemaDiff(BaseModel):
    """Difference between two schema versions."""
    source_version: int = Field(..., description="Source schema version")
    target_version: int = Field(..., description="Target schema version")
    changes: List[SchemaChange] = Field(default_factory=list)
    is_compatible: bool = Field(default=True)
    breaking_changes: List[SchemaChange] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @property
    def has_breaking_changes(self) -> bool:
        """Check if there are breaking changes."""
        return len(self.breaking_changes) > 0


class MigrationPath(BaseModel):
    """Migration path between schema versions."""
    event_type: str = Field(..., description="Event type")
    source_version: int = Field(..., description="Source version")
    target_version: int = Field(..., description="Target version")
    steps: List[Tuple[int, int]] = Field(default_factory=list)
    transformers: List[str] = Field(default_factory=list)
    estimated_complexity: str = Field(default="low")  # low, medium, high


# =============================================================================
# Transformers
# =============================================================================


class EventTransformer(ABC):
    """Base class for event transformers."""

    @abstractmethod
    def transform(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform event data."""
        pass

    @property
    @abstractmethod
    def source_version(self) -> int:
        """Source version this transformer handles."""
        pass

    @property
    @abstractmethod
    def target_version(self) -> int:
        """Target version this transformer produces."""
        pass


class UpgradeTransformer(EventTransformer):
    """Transformer for upgrading events to newer versions."""

    def __init__(
        self,
        source_version: int,
        target_version: int,
        transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        self._source_version = source_version
        self._target_version = target_version
        self._transform_fn = transform_fn

    @property
    def source_version(self) -> int:
        return self._source_version

    @property
    def target_version(self) -> int:
        return self._target_version

    def transform(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._transform_fn(event_data)


class DowngradeTransformer(EventTransformer):
    """Transformer for downgrading events to older versions."""

    def __init__(
        self,
        source_version: int,
        target_version: int,
        transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        self._source_version = source_version
        self._target_version = target_version
        self._transform_fn = transform_fn

    @property
    def source_version(self) -> int:
        return self._source_version

    @property
    def target_version(self) -> int:
        return self._target_version

    def transform(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._transform_fn(event_data)


class TransformerChain:
    """Chain of transformers for multi-version migrations."""

    def __init__(self, transformers: List[EventTransformer]):
        self.transformers = transformers

    def transform(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all transformers in sequence."""
        result = event_data.copy()
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result


# =============================================================================
# Schema Registry
# =============================================================================


class SchemaRegistry:
    """
    Schema Registry for event schemas.

    Stores and manages schema versions for all event types,
    with support for compatibility checking and schema evolution.

    Example:
        >>> registry = SchemaRegistry()
        >>> registry.register(emission_schema_v1)
        >>> registry.register(emission_schema_v2)
        >>> schema = registry.get_schema("EmissionCalculated", version=2)
    """

    def __init__(self):
        """Initialize the schema registry."""
        # schemas[event_type][version] = EventSchema
        self._schemas: Dict[str, Dict[int, EventSchema]] = {}
        self._compatibility_modes: Dict[str, CompatibilityMode] = {}
        self._lock = asyncio.Lock()

        logger.info("SchemaRegistry initialized")

    async def register(
        self,
        schema: EventSchema,
        check_compatibility: bool = True
    ) -> bool:
        """
        Register a new schema version.

        Args:
            schema: Schema to register
            check_compatibility: Whether to check compatibility

        Returns:
            True if registered successfully

        Raises:
            ValueError: If schema is incompatible
        """
        async with self._lock:
            event_type = schema.event_type

            if event_type not in self._schemas:
                self._schemas[event_type] = {}
                self._compatibility_modes[event_type] = schema.compatibility_mode

            # Check if version already exists
            if schema.version in self._schemas[event_type]:
                existing = self._schemas[event_type][schema.version]
                if existing.provenance_hash != schema.provenance_hash:
                    raise ValueError(
                        f"Schema version {schema.version} already exists "
                        f"for {event_type} with different definition"
                    )
                return True  # Same schema already registered

            # Check compatibility with previous version
            if check_compatibility and schema.version > 1:
                prev_version = schema.version - 1
                if prev_version in self._schemas[event_type]:
                    prev_schema = self._schemas[event_type][prev_version]
                    compatible = await self._check_compatibility(
                        prev_schema, schema
                    )
                    if not compatible:
                        raise ValueError(
                            f"Schema version {schema.version} is not compatible "
                            f"with version {prev_version}"
                        )

            self._schemas[event_type][schema.version] = schema
            logger.info(f"Registered schema: {event_type} v{schema.version}")

            return True

    async def get_schema(
        self,
        event_type: str,
        version: Optional[int] = None
    ) -> Optional[EventSchema]:
        """
        Get a schema by event type and version.

        Args:
            event_type: Event type
            version: Schema version (None for latest)

        Returns:
            EventSchema or None
        """
        async with self._lock:
            if event_type not in self._schemas:
                return None

            versions = self._schemas[event_type]

            if version is None:
                # Get latest
                version = max(versions.keys()) if versions else None

            if version is None:
                return None

            return versions.get(version)

    async def get_latest_version(self, event_type: str) -> Optional[int]:
        """Get the latest version number for an event type."""
        async with self._lock:
            if event_type not in self._schemas:
                return None
            versions = self._schemas[event_type]
            return max(versions.keys()) if versions else None

    async def get_all_versions(self, event_type: str) -> List[int]:
        """Get all version numbers for an event type."""
        async with self._lock:
            if event_type not in self._schemas:
                return []
            return sorted(self._schemas[event_type].keys())

    async def list_event_types(self) -> List[str]:
        """List all registered event types."""
        async with self._lock:
            return list(self._schemas.keys())

    async def _check_compatibility(
        self,
        old_schema: EventSchema,
        new_schema: EventSchema
    ) -> bool:
        """Check if new schema is compatible with old schema."""
        mode = self._compatibility_modes.get(
            old_schema.event_type,
            CompatibilityMode.BACKWARD
        )

        if mode == CompatibilityMode.NONE:
            return True

        # Get field names
        old_fields = {f.name: f for f in old_schema.fields}
        new_fields = {f.name: f for f in new_schema.fields}

        # Check backward compatibility
        if mode in [
            CompatibilityMode.BACKWARD,
            CompatibilityMode.FULL,
            CompatibilityMode.BACKWARD_TRANSITIVE,
            CompatibilityMode.FULL_TRANSITIVE
        ]:
            # All old required fields must exist in new schema
            for name, field in old_fields.items():
                if field.required and name not in new_fields:
                    logger.warning(f"Backward incompatible: required field '{name}' removed")
                    return False

                # Type changes are breaking
                if name in new_fields:
                    if field.field_type != new_fields[name].field_type:
                        logger.warning(f"Backward incompatible: field '{name}' type changed")
                        return False

        # Check forward compatibility
        if mode in [
            CompatibilityMode.FORWARD,
            CompatibilityMode.FULL,
            CompatibilityMode.FORWARD_TRANSITIVE,
            CompatibilityMode.FULL_TRANSITIVE
        ]:
            # All new required fields must have defaults or be optional
            for name, field in new_fields.items():
                if field.required and name not in old_fields:
                    if field.default is None:
                        logger.warning(
                            f"Forward incompatible: new required field '{name}' has no default"
                        )
                        return False

        return True

    async def set_compatibility_mode(
        self,
        event_type: str,
        mode: CompatibilityMode
    ) -> None:
        """Set compatibility mode for an event type."""
        async with self._lock:
            self._compatibility_modes[event_type] = mode

    async def delete_schema(
        self,
        event_type: str,
        version: int
    ) -> bool:
        """Delete a schema version."""
        async with self._lock:
            if event_type not in self._schemas:
                return False

            if version not in self._schemas[event_type]:
                return False

            del self._schemas[event_type][version]

            if not self._schemas[event_type]:
                del self._schemas[event_type]
                del self._compatibility_modes[event_type]

            logger.info(f"Deleted schema: {event_type} v{version}")
            return True


# =============================================================================
# Schema Differ
# =============================================================================


class SchemaDiffer:
    """Compares schemas and identifies differences."""

    def diff(self, old_schema: EventSchema, new_schema: EventSchema) -> SchemaDiff:
        """
        Compare two schemas and return differences.

        Args:
            old_schema: Older schema version
            new_schema: Newer schema version

        Returns:
            SchemaDiff with all changes
        """
        changes = []
        breaking_changes = []
        warnings = []

        old_fields = {f.name: f for f in old_schema.fields}
        new_fields = {f.name: f for f in new_schema.fields}

        # Find added fields
        for name, field in new_fields.items():
            if name not in old_fields:
                change = SchemaChange(
                    change_type=FieldChange.ADDED,
                    field_name=name,
                    new_value=field.dict(),
                    breaking=field.required and field.default is None,
                    migration_hint=f"Add field '{name}' with default value"
                )
                changes.append(change)
                if change.breaking:
                    breaking_changes.append(change)
                    warnings.append(f"New required field '{name}' without default is breaking")

        # Find removed fields
        for name, field in old_fields.items():
            if name not in new_fields:
                change = SchemaChange(
                    change_type=FieldChange.REMOVED,
                    field_name=name,
                    old_value=field.dict(),
                    breaking=field.required,
                    migration_hint=f"Remove field '{name}' from processing"
                )
                changes.append(change)
                if change.breaking:
                    breaking_changes.append(change)
                    warnings.append(f"Required field '{name}' was removed")

        # Find modified fields
        for name in set(old_fields.keys()) & set(new_fields.keys()):
            old_field = old_fields[name]
            new_field = new_fields[name]

            # Type change
            if old_field.field_type != new_field.field_type:
                change = SchemaChange(
                    change_type=FieldChange.TYPE_CHANGED,
                    field_name=name,
                    old_value=old_field.field_type,
                    new_value=new_field.field_type,
                    breaking=True,
                    migration_hint=f"Convert field '{name}' from {old_field.field_type} to {new_field.field_type}"
                )
                changes.append(change)
                breaking_changes.append(change)
                warnings.append(f"Field '{name}' type changed from {old_field.field_type} to {new_field.field_type}")

            # Required/optional change
            if old_field.required != new_field.required:
                change = SchemaChange(
                    change_type=FieldChange.OPTIONAL_CHANGED,
                    field_name=name,
                    old_value=old_field.required,
                    new_value=new_field.required,
                    breaking=new_field.required and not old_field.required,
                    migration_hint=f"Field '{name}' optionality changed"
                )
                changes.append(change)
                if change.breaking:
                    breaking_changes.append(change)

            # Default value change
            if old_field.default != new_field.default:
                change = SchemaChange(
                    change_type=FieldChange.DEFAULT_CHANGED,
                    field_name=name,
                    old_value=old_field.default,
                    new_value=new_field.default,
                    breaking=False
                )
                changes.append(change)

        return SchemaDiff(
            source_version=old_schema.version,
            target_version=new_schema.version,
            changes=changes,
            is_compatible=len(breaking_changes) == 0,
            breaking_changes=breaking_changes,
            warnings=warnings
        )


# =============================================================================
# Event Version Manager
# =============================================================================


class EventVersionManager:
    """
    Event Version Manager for GreenLang.

    Manages schema versions, compatibility checking, and event transformation
    between versions.

    Attributes:
        registry: Schema registry
        differ: Schema differ

    Example:
        >>> registry = SchemaRegistry()
        >>> manager = EventVersionManager(registry)
        >>>
        >>> # Register upgrade transformer
        >>> manager.register_transformer(v1_to_v2_transformer)
        >>>
        >>> # Upgrade event to latest version
        >>> upgraded = await manager.upgrade(event, target_version=2)
    """

    def __init__(self, registry: Optional[SchemaRegistry] = None):
        """
        Initialize the version manager.

        Args:
            registry: Schema registry (creates new one if not provided)
        """
        self.registry = registry or SchemaRegistry()
        self.differ = SchemaDiffer()

        # Transformers: transformers[event_type][(source, target)] = transformer
        self._transformers: Dict[str, Dict[Tuple[int, int], EventTransformer]] = {}

        # Version negotiation callbacks
        self._negotiation_callbacks: List[Callable] = []

        logger.info("EventVersionManager initialized")

    async def register_schema(
        self,
        schema: EventSchema,
        check_compatibility: bool = True
    ) -> bool:
        """
        Register a new schema version.

        Args:
            schema: Schema to register
            check_compatibility: Whether to check compatibility

        Returns:
            True if registered
        """
        return await self.registry.register(schema, check_compatibility)

    def register_transformer(
        self,
        event_type: str,
        transformer: EventTransformer
    ) -> None:
        """
        Register an event transformer.

        Args:
            event_type: Event type
            transformer: Transformer to register
        """
        if event_type not in self._transformers:
            self._transformers[event_type] = {}

        key = (transformer.source_version, transformer.target_version)
        self._transformers[event_type][key] = transformer

        logger.info(
            f"Registered transformer: {event_type} "
            f"v{transformer.source_version} -> v{transformer.target_version}"
        )

    def register_upgrade(
        self,
        event_type: str,
        source_version: int,
        target_version: int,
        transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """
        Register an upgrade transformer function.

        Args:
            event_type: Event type
            source_version: Source version
            target_version: Target version
            transform_fn: Transformation function
        """
        transformer = UpgradeTransformer(source_version, target_version, transform_fn)
        self.register_transformer(event_type, transformer)

    def register_downgrade(
        self,
        event_type: str,
        source_version: int,
        target_version: int,
        transform_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """
        Register a downgrade transformer function.

        Args:
            event_type: Event type
            source_version: Source version
            target_version: Target version
            transform_fn: Transformation function
        """
        transformer = DowngradeTransformer(source_version, target_version, transform_fn)
        self.register_transformer(event_type, transformer)

    async def upgrade(
        self,
        event: BaseEvent,
        target_version: Optional[int] = None
    ) -> BaseEvent:
        """
        Upgrade an event to a target version.

        Args:
            event: Event to upgrade
            target_version: Target version (None for latest)

        Returns:
            Upgraded event
        """
        event_type = event.event_type
        current_version = event.metadata.version

        if target_version is None:
            target_version = await self.registry.get_latest_version(event_type)

        if target_version is None or current_version >= target_version:
            return event  # Already at target or higher

        # Find transformation path
        path = await self._find_transformation_path(
            event_type, current_version, target_version, MigrationDirection.UPGRADE
        )

        if not path:
            raise ValueError(
                f"No upgrade path from v{current_version} to v{target_version} "
                f"for {event_type}"
            )

        # Apply transformations
        data = event.data.copy()
        for source, target in path:
            key = (source, target)
            transformer = self._transformers.get(event_type, {}).get(key)
            if transformer:
                data = transformer.transform(data)

        # Create upgraded event
        new_metadata = EventMetadata(
            event_id=event.metadata.event_id,
            correlation_id=event.metadata.correlation_id,
            causation_id=event.metadata.causation_id,
            timestamp=event.metadata.timestamp,
            version=target_version,
            source=event.metadata.source,
            priority=event.metadata.priority,
        )

        return BaseEvent(
            event_type=event_type,
            metadata=new_metadata,
            data=data
        )

    async def downgrade(
        self,
        event: BaseEvent,
        target_version: int
    ) -> BaseEvent:
        """
        Downgrade an event to a target version.

        Args:
            event: Event to downgrade
            target_version: Target version

        Returns:
            Downgraded event
        """
        event_type = event.event_type
        current_version = event.metadata.version

        if current_version <= target_version:
            return event  # Already at target or lower

        # Find transformation path
        path = await self._find_transformation_path(
            event_type, current_version, target_version, MigrationDirection.DOWNGRADE
        )

        if not path:
            raise ValueError(
                f"No downgrade path from v{current_version} to v{target_version} "
                f"for {event_type}"
            )

        # Apply transformations
        data = event.data.copy()
        for source, target in path:
            key = (source, target)
            transformer = self._transformers.get(event_type, {}).get(key)
            if transformer:
                data = transformer.transform(data)

        # Create downgraded event
        new_metadata = EventMetadata(
            event_id=event.metadata.event_id,
            correlation_id=event.metadata.correlation_id,
            causation_id=event.metadata.causation_id,
            timestamp=event.metadata.timestamp,
            version=target_version,
            source=event.metadata.source,
            priority=event.metadata.priority,
        )

        return BaseEvent(
            event_type=event_type,
            metadata=new_metadata,
            data=data
        )

    async def _find_transformation_path(
        self,
        event_type: str,
        source_version: int,
        target_version: int,
        direction: MigrationDirection
    ) -> List[Tuple[int, int]]:
        """Find a transformation path between versions."""
        if source_version == target_version:
            return []

        transformers = self._transformers.get(event_type, {})

        # Build graph
        graph: Dict[int, List[int]] = {}
        for (src, tgt) in transformers.keys():
            if src not in graph:
                graph[src] = []
            graph[src].append(tgt)

        # BFS to find path
        from collections import deque

        queue = deque([(source_version, [])])
        visited = {source_version}

        while queue:
            current, path = queue.popleft()

            if current == target_version:
                return path

            for next_version in graph.get(current, []):
                # Check direction
                if direction == MigrationDirection.UPGRADE and next_version <= current:
                    continue
                if direction == MigrationDirection.DOWNGRADE and next_version >= current:
                    continue

                if next_version not in visited:
                    visited.add(next_version)
                    new_path = path + [(current, next_version)]
                    queue.append((next_version, new_path))

        return []

    async def get_schema_diff(
        self,
        event_type: str,
        from_version: int,
        to_version: int
    ) -> SchemaDiff:
        """
        Get the diff between two schema versions.

        Args:
            event_type: Event type
            from_version: Source version
            to_version: Target version

        Returns:
            SchemaDiff with changes
        """
        from_schema = await self.registry.get_schema(event_type, from_version)
        to_schema = await self.registry.get_schema(event_type, to_version)

        if not from_schema or not to_schema:
            raise ValueError(f"Schema not found for {event_type}")

        return self.differ.diff(from_schema, to_schema)

    async def get_migration_path(
        self,
        event_type: str,
        from_version: int,
        to_version: int
    ) -> MigrationPath:
        """
        Get the migration path between versions.

        Args:
            event_type: Event type
            from_version: Source version
            to_version: Target version

        Returns:
            MigrationPath with steps and transformers
        """
        direction = (
            MigrationDirection.UPGRADE
            if to_version > from_version
            else MigrationDirection.DOWNGRADE
        )

        steps = await self._find_transformation_path(
            event_type, from_version, to_version, direction
        )

        transformer_names = []
        for src, tgt in steps:
            key = (src, tgt)
            transformer = self._transformers.get(event_type, {}).get(key)
            if transformer:
                transformer_names.append(f"{type(transformer).__name__}")

        # Estimate complexity
        complexity = "low"
        if len(steps) > 3:
            complexity = "medium"
        if len(steps) > 5:
            complexity = "high"

        return MigrationPath(
            event_type=event_type,
            source_version=from_version,
            target_version=to_version,
            steps=steps,
            transformers=transformer_names,
            estimated_complexity=complexity
        )

    async def negotiate_version(
        self,
        event_type: str,
        producer_version: int,
        consumer_versions: List[int]
    ) -> int:
        """
        Negotiate the best version for producer and consumers.

        Args:
            event_type: Event type
            producer_version: Version producer can produce
            consumer_versions: Versions consumers can handle

        Returns:
            Negotiated version
        """
        # Find the highest version all consumers can handle
        # that the producer can produce (with transformations)
        available_versions = await self.registry.get_all_versions(event_type)

        candidates = []
        for version in available_versions:
            # Check if producer can reach this version
            if version <= producer_version:
                # Check if all consumers can handle it
                if all(v >= version or v in consumer_versions for v in consumer_versions):
                    candidates.append(version)

        if not candidates:
            raise ValueError("No compatible version found")

        return max(candidates)

    async def generate_migration_docs(
        self,
        event_type: str
    ) -> str:
        """
        Generate migration documentation for an event type.

        Args:
            event_type: Event type

        Returns:
            Markdown documentation
        """
        versions = await self.registry.get_all_versions(event_type)

        if not versions:
            return f"# {event_type}\n\nNo schema versions registered."

        lines = [
            f"# {event_type} Schema Migration Guide",
            "",
            "## Version History",
            ""
        ]

        for version in versions:
            schema = await self.registry.get_schema(event_type, version)
            if schema:
                lines.append(f"### Version {version}")
                lines.append(f"- Created: {schema.created_at.isoformat()}")
                lines.append(f"- Deprecated: {schema.deprecated}")
                if schema.description:
                    lines.append(f"- Description: {schema.description}")
                lines.append("")

                lines.append("#### Fields")
                for field in schema.fields:
                    required = "required" if field.required else "optional"
                    lines.append(f"- `{field.name}` ({field.field_type}, {required})")
                    if field.description:
                        lines.append(f"  - {field.description}")
                    if field.deprecated:
                        lines.append(f"  - **Deprecated** since v{field.deprecated_since}")
                lines.append("")

        # Add migration paths
        if len(versions) > 1:
            lines.append("## Migration Paths")
            lines.append("")

            for i in range(len(versions) - 1):
                from_v = versions[i]
                to_v = versions[i + 1]

                diff = await self.get_schema_diff(event_type, from_v, to_v)

                lines.append(f"### v{from_v} to v{to_v}")
                lines.append("")

                if diff.breaking_changes:
                    lines.append("**Breaking Changes:**")
                    for change in diff.breaking_changes:
                        lines.append(f"- {change.change_type.value}: `{change.field_name}`")
                        if change.migration_hint:
                            lines.append(f"  - Hint: {change.migration_hint}")
                    lines.append("")

                if diff.changes:
                    lines.append("**All Changes:**")
                    for change in diff.changes:
                        lines.append(f"- {change.change_type.value}: `{change.field_name}`")
                    lines.append("")

        return "\n".join(lines)


# =============================================================================
# FastAPI Router
# =============================================================================


def create_versioning_router(manager: EventVersionManager):
    """
    Create FastAPI router for schema versioning.

    Args:
        manager: EventVersionManager instance

    Returns:
        FastAPI APIRouter
    """
    try:
        from fastapi import APIRouter, HTTPException, Query, status
        from fastapi.responses import PlainTextResponse
    except ImportError:
        logger.warning("FastAPI not available, skipping router creation")
        return None

    router = APIRouter(prefix="/api/v1/schemas", tags=["Event Schemas"])

    @router.get("/event-types")
    async def list_event_types():
        """List all registered event types."""
        types = await manager.registry.list_event_types()
        return {"event_types": types}

    @router.get("/{event_type}/versions")
    async def get_versions(event_type: str):
        """Get all versions for an event type."""
        versions = await manager.registry.get_all_versions(event_type)
        if not versions:
            raise HTTPException(status_code=404, detail="Event type not found")
        return {"event_type": event_type, "versions": versions}

    @router.get("/{event_type}/schema")
    async def get_schema(
        event_type: str,
        version: Optional[int] = Query(None)
    ):
        """Get schema for an event type."""
        schema = await manager.registry.get_schema(event_type, version)
        if not schema:
            raise HTTPException(status_code=404, detail="Schema not found")
        return schema.dict()

    @router.get("/{event_type}/diff")
    async def get_diff(
        event_type: str,
        from_version: int = Query(...),
        to_version: int = Query(...)
    ):
        """Get diff between two schema versions."""
        try:
            diff = await manager.get_schema_diff(event_type, from_version, to_version)
            return diff.dict()
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.get("/{event_type}/migration-path")
    async def get_migration_path(
        event_type: str,
        from_version: int = Query(...),
        to_version: int = Query(...)
    ):
        """Get migration path between versions."""
        path = await manager.get_migration_path(event_type, from_version, to_version)
        return path.dict()

    @router.get("/{event_type}/docs", response_class=PlainTextResponse)
    async def get_migration_docs(event_type: str):
        """Get migration documentation."""
        docs = await manager.generate_migration_docs(event_type)
        return docs

    @router.post("/{event_type}/negotiate")
    async def negotiate_version(
        event_type: str,
        producer_version: int = Query(...),
        consumer_versions: List[int] = Query(...)
    ):
        """Negotiate compatible version."""
        try:
            version = await manager.negotiate_version(
                event_type, producer_version, consumer_versions
            )
            return {"negotiated_version": version}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return router
