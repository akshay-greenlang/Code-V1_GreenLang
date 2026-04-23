"""
GL-001 ThermalCommand: Schema Registry

This module implements a schema registry for the ThermalCommand
ProcessHeatOrchestrator system, providing:

1. Schema version management
2. Compatibility checking (forward/backward)
3. Schema migration support
4. Schema discovery and documentation
5. Runtime schema validation

The registry follows semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes (incompatible schema changes)
- MINOR: Backward-compatible additions
- PATCH: Backward-compatible bug fixes

Standards Compliance:
- JSON Schema Draft 2020-12
- Apache Avro compatibility rules
- OpenAPI 3.1 for API schemas

Author: GreenLang Data Integration Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel, Field


# =============================================================================
# Enumerations
# =============================================================================

class CompatibilityMode(str, Enum):
    """Schema compatibility modes."""
    NONE = "none"                   # No compatibility checking
    BACKWARD = "backward"           # New schema can read old data
    FORWARD = "forward"             # Old schema can read new data
    FULL = "full"                   # Both backward and forward
    BACKWARD_TRANSITIVE = "backward_transitive"  # Backward through all versions
    FORWARD_TRANSITIVE = "forward_transitive"    # Forward through all versions
    FULL_TRANSITIVE = "full_transitive"          # Full through all versions


class SchemaStatus(str, Enum):
    """Schema lifecycle status."""
    DRAFT = "draft"                 # Under development
    ACTIVE = "active"               # In production use
    DEPRECATED = "deprecated"       # Scheduled for removal
    RETIRED = "retired"             # No longer supported


class ChangeType(str, Enum):
    """Types of schema changes."""
    FIELD_ADDED = "field_added"
    FIELD_REMOVED = "field_removed"
    FIELD_RENAMED = "field_renamed"
    FIELD_TYPE_CHANGED = "field_type_changed"
    FIELD_REQUIRED_CHANGED = "field_required_changed"
    DEFAULT_CHANGED = "default_changed"
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"
    ENUM_VALUE_ADDED = "enum_value_added"
    ENUM_VALUE_REMOVED = "enum_value_removed"


# =============================================================================
# Schema Version Model
# =============================================================================

@dataclass
class SemanticVersion:
    """Semantic version representation."""

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse version string (e.g., '1.2.3')."""
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3))
        )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)

    def __gt__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __ge__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    def is_compatible_with(self, other: "SemanticVersion", mode: CompatibilityMode) -> bool:
        """Check if this version is compatible with another."""
        if mode == CompatibilityMode.NONE:
            return True

        # Major version difference is always incompatible
        if self.major != other.major:
            return False

        if mode in (CompatibilityMode.BACKWARD, CompatibilityMode.BACKWARD_TRANSITIVE):
            # Newer can read older (minor/patch can increase)
            return self >= other

        if mode in (CompatibilityMode.FORWARD, CompatibilityMode.FORWARD_TRANSITIVE):
            # Older can read newer (minor/patch can decrease)
            return self <= other

        if mode in (CompatibilityMode.FULL, CompatibilityMode.FULL_TRANSITIVE):
            # Same major and minor, any patch
            return self.minor == other.minor

        return True

    def bump_major(self) -> "SemanticVersion":
        """Create new version with bumped major."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemanticVersion":
        """Create new version with bumped minor."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemanticVersion":
        """Create new version with bumped patch."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)


# =============================================================================
# Schema Change Model
# =============================================================================

@dataclass
class SchemaChange:
    """Represents a change between schema versions."""

    change_type: ChangeType
    field_path: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    is_breaking: bool = False
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_type": self.change_type.value,
            "field_path": self.field_path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "is_breaking": self.is_breaking,
            "description": self.description,
        }


# =============================================================================
# Schema Definition Model
# =============================================================================

@dataclass
class SchemaDefinition:
    """Complete schema definition with metadata."""

    name: str
    version: SemanticVersion
    schema_type: str  # 'pydantic', 'json_schema', 'avro', 'protobuf'
    schema_content: Dict[str, Any]
    status: SchemaStatus = SchemaStatus.ACTIVE
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    deprecated_at: Optional[datetime] = None
    deprecated_reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    @property
    def schema_id(self) -> str:
        """Get unique schema identifier."""
        return f"{self.name}:{self.version}"

    @property
    def fingerprint(self) -> str:
        """Get schema fingerprint (hash)."""
        content_str = json.dumps(self.schema_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": str(self.version),
            "schema_type": self.schema_type,
            "schema_content": self.schema_content,
            "status": self.status.value,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
            "deprecated_reason": self.deprecated_reason,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "fingerprint": self.fingerprint,
        }


# =============================================================================
# Migration Definition
# =============================================================================

@dataclass
class SchemaMigration:
    """Schema migration definition."""

    name: str
    from_version: SemanticVersion
    to_version: SemanticVersion
    changes: List[SchemaChange]
    upgrade_script: Optional[str] = None
    downgrade_script: Optional[str] = None
    is_reversible: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def migration_id(self) -> str:
        """Get migration identifier."""
        return f"{self.name}:{self.from_version}->{self.to_version}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "from_version": str(self.from_version),
            "to_version": str(self.to_version),
            "changes": [c.to_dict() for c in self.changes],
            "upgrade_script": self.upgrade_script,
            "downgrade_script": self.downgrade_script,
            "is_reversible": self.is_reversible,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Compatibility Checker
# =============================================================================

class CompatibilityChecker:
    """
    Checks compatibility between schema versions.

    Follows Confluent Schema Registry compatibility rules.
    """

    # Changes that break backward compatibility
    BACKWARD_BREAKING_CHANGES = {
        ChangeType.FIELD_REMOVED,        # Can't read old data with missing field
        ChangeType.FIELD_TYPE_CHANGED,   # Type mismatch
        ChangeType.ENUM_VALUE_REMOVED,   # Old value no longer valid
    }

    # Changes that break forward compatibility
    FORWARD_BREAKING_CHANGES = {
        ChangeType.FIELD_ADDED,          # Old schema doesn't know new field
        ChangeType.FIELD_TYPE_CHANGED,   # Type mismatch
        ChangeType.ENUM_VALUE_ADDED,     # New value not recognized
    }

    def __init__(self, mode: CompatibilityMode = CompatibilityMode.BACKWARD):
        """Initialize compatibility checker."""
        self.mode = mode

    def check_compatibility(
        self,
        old_schema: SchemaDefinition,
        new_schema: SchemaDefinition
    ) -> Tuple[bool, List[SchemaChange]]:
        """
        Check if new schema is compatible with old schema.

        Args:
            old_schema: Previous schema version
            new_schema: New schema version

        Returns:
            Tuple of (is_compatible, list of changes)
        """
        changes = self._detect_changes(old_schema, new_schema)
        breaking_changes = self._find_breaking_changes(changes)

        is_compatible = len(breaking_changes) == 0
        return is_compatible, changes

    def _detect_changes(
        self,
        old_schema: SchemaDefinition,
        new_schema: SchemaDefinition
    ) -> List[SchemaChange]:
        """Detect changes between schemas."""
        changes = []

        old_content = old_schema.schema_content
        new_content = new_schema.schema_content

        # Get field sets
        old_fields = self._extract_fields(old_content)
        new_fields = self._extract_fields(new_content)

        old_field_names = set(old_fields.keys())
        new_field_names = set(new_fields.keys())

        # Detect removed fields
        for field_name in old_field_names - new_field_names:
            changes.append(SchemaChange(
                change_type=ChangeType.FIELD_REMOVED,
                field_path=field_name,
                old_value=old_fields[field_name],
                is_breaking=True,
                description=f"Field '{field_name}' was removed"
            ))

        # Detect added fields
        for field_name in new_field_names - old_field_names:
            field_info = new_fields[field_name]
            is_required = field_info.get("required", False)
            has_default = "default" in field_info

            is_breaking = is_required and not has_default

            changes.append(SchemaChange(
                change_type=ChangeType.FIELD_ADDED,
                field_path=field_name,
                new_value=field_info,
                is_breaking=is_breaking,
                description=f"Field '{field_name}' was added"
            ))

        # Detect modified fields
        for field_name in old_field_names & new_field_names:
            old_field = old_fields[field_name]
            new_field = new_fields[field_name]

            # Check type changes
            old_type = old_field.get("type")
            new_type = new_field.get("type")
            if old_type != new_type:
                changes.append(SchemaChange(
                    change_type=ChangeType.FIELD_TYPE_CHANGED,
                    field_path=field_name,
                    old_value=old_type,
                    new_value=new_type,
                    is_breaking=True,
                    description=f"Type of '{field_name}' changed from {old_type} to {new_type}"
                ))

            # Check required changes
            old_required = old_field.get("required", False)
            new_required = new_field.get("required", False)
            if old_required != new_required:
                changes.append(SchemaChange(
                    change_type=ChangeType.FIELD_REQUIRED_CHANGED,
                    field_path=field_name,
                    old_value=old_required,
                    new_value=new_required,
                    is_breaking=new_required and not old_required,
                    description=f"Required status of '{field_name}' changed"
                ))

            # Check default changes
            old_default = old_field.get("default")
            new_default = new_field.get("default")
            if old_default != new_default:
                changes.append(SchemaChange(
                    change_type=ChangeType.DEFAULT_CHANGED,
                    field_path=field_name,
                    old_value=old_default,
                    new_value=new_default,
                    is_breaking=False,
                    description=f"Default of '{field_name}' changed"
                ))

            # Check enum value changes
            if old_type == "enum" and new_type == "enum":
                old_values = set(old_field.get("enum", []))
                new_values = set(new_field.get("enum", []))

                for removed in old_values - new_values:
                    changes.append(SchemaChange(
                        change_type=ChangeType.ENUM_VALUE_REMOVED,
                        field_path=field_name,
                        old_value=removed,
                        is_breaking=True,
                        description=f"Enum value '{removed}' removed from '{field_name}'"
                    ))

                for added in new_values - old_values:
                    changes.append(SchemaChange(
                        change_type=ChangeType.ENUM_VALUE_ADDED,
                        field_path=field_name,
                        new_value=added,
                        is_breaking=False,
                        description=f"Enum value '{added}' added to '{field_name}'"
                    ))

        return changes

    def _extract_fields(self, schema_content: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract field definitions from schema content."""
        fields = {}

        # Handle Pydantic/JSON Schema style
        if "properties" in schema_content:
            required_fields = set(schema_content.get("required", []))
            for name, props in schema_content["properties"].items():
                fields[name] = {
                    **props,
                    "required": name in required_fields
                }

        # Handle flat field definitions
        elif "fields" in schema_content:
            for field_def in schema_content["fields"]:
                name = field_def.get("name")
                if name:
                    fields[name] = field_def

        return fields

    def _find_breaking_changes(self, changes: List[SchemaChange]) -> List[SchemaChange]:
        """Find breaking changes based on compatibility mode."""
        breaking = []

        for change in changes:
            is_breaking = False

            if self.mode in (CompatibilityMode.BACKWARD, CompatibilityMode.BACKWARD_TRANSITIVE):
                if change.change_type in self.BACKWARD_BREAKING_CHANGES:
                    is_breaking = True

            if self.mode in (CompatibilityMode.FORWARD, CompatibilityMode.FORWARD_TRANSITIVE):
                if change.change_type in self.FORWARD_BREAKING_CHANGES:
                    is_breaking = True

            if self.mode in (CompatibilityMode.FULL, CompatibilityMode.FULL_TRANSITIVE):
                if (change.change_type in self.BACKWARD_BREAKING_CHANGES or
                    change.change_type in self.FORWARD_BREAKING_CHANGES):
                    is_breaking = True

            if is_breaking:
                breaking.append(change)

        return breaking


# =============================================================================
# Schema Migrator
# =============================================================================

class SchemaMigrator:
    """
    Handles schema migration operations.

    Supports:
    - Automatic data transformation between versions
    - Migration script execution
    - Rollback support
    """

    def __init__(self):
        """Initialize schema migrator."""
        self._migrations: Dict[str, SchemaMigration] = {}
        self._transform_functions: Dict[str, Callable] = {}

    def register_migration(self, migration: SchemaMigration):
        """Register a migration."""
        self._migrations[migration.migration_id] = migration

    def register_transform(
        self,
        from_version: str,
        to_version: str,
        transform_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """
        Register a data transformation function.

        Args:
            from_version: Source version string
            to_version: Target version string
            transform_func: Function to transform data
        """
        key = f"{from_version}->{to_version}"
        self._transform_functions[key] = transform_func

    def migrate_data(
        self,
        data: Dict[str, Any],
        from_version: SemanticVersion,
        to_version: SemanticVersion,
        schema_name: str
    ) -> Dict[str, Any]:
        """
        Migrate data from one schema version to another.

        Args:
            data: Data to migrate
            from_version: Source version
            to_version: Target version
            schema_name: Schema name

        Returns:
            Migrated data
        """
        if from_version == to_version:
            return data

        # Find migration path
        path = self._find_migration_path(schema_name, from_version, to_version)
        if not path:
            raise ValueError(
                f"No migration path found from {from_version} to {to_version}"
            )

        # Apply migrations
        result = data.copy()
        for migration in path:
            result = self._apply_migration(result, migration)

        return result

    def _find_migration_path(
        self,
        schema_name: str,
        from_version: SemanticVersion,
        to_version: SemanticVersion
    ) -> List[SchemaMigration]:
        """Find migration path between versions."""
        # Simple direct path lookup
        migration_id = f"{schema_name}:{from_version}->{to_version}"
        if migration_id in self._migrations:
            return [self._migrations[migration_id]]

        # Try to find incremental path
        path = []
        current = from_version

        if to_version > from_version:
            # Forward migration
            while current < to_version:
                # Try patch bump
                next_version = current.bump_patch()
                migration_id = f"{schema_name}:{current}->{next_version}"
                if migration_id in self._migrations:
                    path.append(self._migrations[migration_id])
                    current = next_version
                    continue

                # Try minor bump
                next_version = current.bump_minor()
                migration_id = f"{schema_name}:{current}->{next_version}"
                if migration_id in self._migrations:
                    path.append(self._migrations[migration_id])
                    current = next_version
                    continue

                # No path found
                return []

        return path

    def _apply_migration(
        self,
        data: Dict[str, Any],
        migration: SchemaMigration
    ) -> Dict[str, Any]:
        """Apply a single migration to data."""
        result = data.copy()

        # Check for custom transform function
        transform_key = f"{migration.from_version}->{migration.to_version}"
        if transform_key in self._transform_functions:
            return self._transform_functions[transform_key](result)

        # Apply changes automatically
        for change in migration.changes:
            if change.change_type == ChangeType.FIELD_REMOVED:
                # Remove field
                self._remove_field(result, change.field_path)

            elif change.change_type == ChangeType.FIELD_ADDED:
                # Add field with default
                if change.new_value and "default" in change.new_value:
                    self._set_field(result, change.field_path, change.new_value["default"])

            elif change.change_type == ChangeType.FIELD_RENAMED:
                # Rename field
                old_value = self._get_field(result, change.old_value)
                if old_value is not None:
                    self._remove_field(result, change.old_value)
                    self._set_field(result, change.new_value, old_value)

        return result

    def _get_field(self, data: Dict, path: str) -> Any:
        """Get nested field value."""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _set_field(self, data: Dict, path: str, value: Any):
        """Set nested field value."""
        keys = path.split(".")
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def _remove_field(self, data: Dict, path: str):
        """Remove nested field."""
        keys = path.split(".")
        current = data
        for key in keys[:-1]:
            if key not in current:
                return
            current = current[key]
        if keys[-1] in current:
            del current[keys[-1]]


# =============================================================================
# Schema Registry
# =============================================================================

class SchemaRegistry:
    """
    Central schema registry for ThermalCommand.

    Provides:
    - Schema registration and lookup
    - Version management
    - Compatibility checking
    - Migration support
    - Schema discovery
    """

    def __init__(self, compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARD):
        """
        Initialize schema registry.

        Args:
            compatibility_mode: Default compatibility mode
        """
        self._schemas: Dict[str, Dict[str, SchemaDefinition]] = {}  # name -> version -> schema
        self._latest_versions: Dict[str, SemanticVersion] = {}
        self.compatibility_mode = compatibility_mode
        self.compatibility_checker = CompatibilityChecker(compatibility_mode)
        self.migrator = SchemaMigrator()

        # Register built-in schemas
        self._register_builtin_schemas()

    def _register_builtin_schemas(self):
        """Register built-in ThermalCommand schemas."""
        from .domain_schemas import DOMAIN_SCHEMAS

        for schema_name, schema_class in DOMAIN_SCHEMAS.items():
            # Generate JSON schema from Pydantic model
            json_schema = schema_class.model_json_schema()

            self.register_schema(
                name=schema_name,
                version="1.0.0",
                schema_type="pydantic",
                schema_content=json_schema,
                description=schema_class.__doc__ or f"{schema_name} schema",
                tags=["domain", "builtin"],
            )

    def register_schema(
        self,
        name: str,
        version: str,
        schema_type: str,
        schema_content: Dict[str, Any],
        description: str = "",
        status: SchemaStatus = SchemaStatus.ACTIVE,
        created_by: str = "system",
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        check_compatibility: bool = True
    ) -> SchemaDefinition:
        """
        Register a new schema version.

        Args:
            name: Schema name
            version: Version string (e.g., '1.0.0')
            schema_type: Schema type ('pydantic', 'json_schema', etc.)
            schema_content: Schema definition content
            description: Schema description
            status: Schema status
            created_by: Creator identifier
            tags: Optional tags
            dependencies: Optional schema dependencies
            check_compatibility: Whether to check compatibility

        Returns:
            Registered SchemaDefinition

        Raises:
            ValueError: If compatibility check fails
        """
        sem_version = SemanticVersion.parse(version)

        # Create schema definition
        schema_def = SchemaDefinition(
            name=name,
            version=sem_version,
            schema_type=schema_type,
            schema_content=schema_content,
            status=status,
            description=description,
            created_by=created_by,
            tags=tags or [],
            dependencies=dependencies or [],
        )

        # Initialize schema storage if needed
        if name not in self._schemas:
            self._schemas[name] = {}

        # Check compatibility with latest version
        if check_compatibility and name in self._latest_versions:
            latest = self.get_schema(name)
            if latest and self.compatibility_mode != CompatibilityMode.NONE:
                is_compatible, changes = self.compatibility_checker.check_compatibility(
                    latest, schema_def
                )
                if not is_compatible:
                    breaking = [c for c in changes if c.is_breaking]
                    raise ValueError(
                        f"Schema {name}:{version} is not compatible with "
                        f"{name}:{latest.version}. Breaking changes: "
                        f"{[c.description for c in breaking]}"
                    )

                # Register migration
                migration = SchemaMigration(
                    name=name,
                    from_version=latest.version,
                    to_version=sem_version,
                    changes=changes,
                )
                self.migrator.register_migration(migration)

        # Store schema
        self._schemas[name][str(sem_version)] = schema_def

        # Update latest version
        if name not in self._latest_versions or sem_version > self._latest_versions[name]:
            self._latest_versions[name] = sem_version

        return schema_def

    def get_schema(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[SchemaDefinition]:
        """
        Get schema by name and optional version.

        Args:
            name: Schema name
            version: Optional version (default: latest)

        Returns:
            SchemaDefinition or None
        """
        if name not in self._schemas:
            return None

        if version:
            return self._schemas[name].get(version)

        # Return latest version
        if name in self._latest_versions:
            latest_ver = str(self._latest_versions[name])
            return self._schemas[name].get(latest_ver)

        return None

    def get_all_versions(self, name: str) -> List[SchemaDefinition]:
        """
        Get all versions of a schema.

        Args:
            name: Schema name

        Returns:
            List of schema definitions sorted by version
        """
        if name not in self._schemas:
            return []

        schemas = list(self._schemas[name].values())
        return sorted(schemas, key=lambda s: s.version)

    def get_latest_version(self, name: str) -> Optional[str]:
        """
        Get latest version string for a schema.

        Args:
            name: Schema name

        Returns:
            Version string or None
        """
        if name in self._latest_versions:
            return str(self._latest_versions[name])
        return None

    def deprecate_schema(
        self,
        name: str,
        version: str,
        reason: str
    ) -> bool:
        """
        Mark a schema version as deprecated.

        Args:
            name: Schema name
            version: Version to deprecate
            reason: Deprecation reason

        Returns:
            True if successful
        """
        schema = self.get_schema(name, version)
        if not schema:
            return False

        schema.status = SchemaStatus.DEPRECATED
        schema.deprecated_at = datetime.now(timezone.utc)
        schema.deprecated_reason = reason
        return True

    def check_compatibility(
        self,
        name: str,
        new_schema_content: Dict[str, Any],
        new_version: Optional[str] = None
    ) -> Tuple[bool, List[SchemaChange]]:
        """
        Check if new schema is compatible with existing versions.

        Args:
            name: Schema name
            new_schema_content: New schema content
            new_version: Optional new version

        Returns:
            Tuple of (is_compatible, list of changes)
        """
        latest = self.get_schema(name)
        if not latest:
            return True, []

        new_schema = SchemaDefinition(
            name=name,
            version=SemanticVersion.parse(new_version) if new_version else latest.version.bump_patch(),
            schema_type=latest.schema_type,
            schema_content=new_schema_content,
        )

        return self.compatibility_checker.check_compatibility(latest, new_schema)

    def migrate_data(
        self,
        name: str,
        data: Dict[str, Any],
        from_version: str,
        to_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Migrate data between schema versions.

        Args:
            name: Schema name
            data: Data to migrate
            from_version: Source version
            to_version: Target version (default: latest)

        Returns:
            Migrated data
        """
        from_ver = SemanticVersion.parse(from_version)
        to_ver = (
            SemanticVersion.parse(to_version) if to_version
            else self._latest_versions.get(name)
        )

        if not to_ver:
            raise ValueError(f"No target version for schema {name}")

        return self.migrator.migrate_data(data, from_ver, to_ver, name)

    def list_schemas(
        self,
        status: Optional[SchemaStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[str]:
        """
        List registered schema names.

        Args:
            status: Optional status filter
            tags: Optional tag filter

        Returns:
            List of schema names
        """
        result = []

        for name in self._schemas:
            latest = self.get_schema(name)
            if not latest:
                continue

            if status and latest.status != status:
                continue

            if tags and not all(t in latest.tags for t in tags):
                continue

            result.append(name)

        return sorted(result)

    def export_registry(self) -> Dict[str, Any]:
        """
        Export registry to dictionary.

        Returns:
            Registry export dictionary
        """
        return {
            "compatibility_mode": self.compatibility_mode.value,
            "schema_count": len(self._schemas),
            "schemas": {
                name: {
                    "latest_version": str(self._latest_versions.get(name, "")),
                    "versions": {
                        ver: schema.to_dict()
                        for ver, schema in versions.items()
                    }
                }
                for name, versions in self._schemas.items()
            },
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

    def validate_data(
        self,
        name: str,
        data: Dict[str, Any],
        version: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate data against a schema.

        Args:
            name: Schema name
            data: Data to validate
            version: Optional schema version

        Returns:
            Tuple of (is_valid, list of errors)
        """
        schema = self.get_schema(name, version)
        if not schema:
            return False, [f"Schema not found: {name}:{version or 'latest'}"]

        errors = []

        # Basic validation against schema content
        schema_content = schema.schema_content
        properties = schema_content.get("properties", {})
        required = set(schema_content.get("required", []))

        # Check required fields
        for field_name in required:
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")

        # Check field types
        for field_name, value in data.items():
            if field_name in properties:
                field_schema = properties[field_name]
                expected_type = field_schema.get("type")

                if expected_type:
                    type_valid = self._check_type(value, expected_type)
                    if not type_valid:
                        errors.append(
                            f"Field '{field_name}' has wrong type: "
                            f"expected {expected_type}, got {type(value).__name__}"
                        )

        is_valid = len(errors) == 0
        return is_valid, errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON Schema type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected = type_mapping.get(expected_type)
        if expected:
            return isinstance(value, expected)

        return True


# =============================================================================
# Singleton Instance
# =============================================================================

_schema_registry: Optional[SchemaRegistry] = None


def get_schema_registry() -> SchemaRegistry:
    """
    Get the global schema registry instance.

    Returns:
        SchemaRegistry singleton instance
    """
    global _schema_registry
    if _schema_registry is None:
        _schema_registry = SchemaRegistry()
    return _schema_registry


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "CompatibilityMode",
    "SchemaStatus",
    "ChangeType",
    # Models
    "SemanticVersion",
    "SchemaChange",
    "SchemaDefinition",
    "SchemaMigration",
    # Classes
    "CompatibilityChecker",
    "SchemaMigrator",
    "SchemaRegistry",
    # Functions
    "get_schema_registry",
]
