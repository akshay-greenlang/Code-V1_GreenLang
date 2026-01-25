"""
Schema Registry for GL-003 UNIFIEDSTEAM

Provides schema versioning, compatibility checking, and registry
management for Kafka message schemas.

Author: GL-003 Data Engineering Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class SchemaCompatibility(Enum):
    """Schema compatibility modes."""
    BACKWARD = "backward"  # New schema can read old data
    FORWARD = "forward"    # Old schema can read new data
    FULL = "full"          # Both backward and forward compatible
    NONE = "none"          # No compatibility checking


class SchemaType(Enum):
    """Schema format types."""
    AVRO = "avro"
    JSON_SCHEMA = "json_schema"
    PROTOBUF = "protobuf"


@dataclass
class SchemaVersion:
    """
    Schema version with metadata.

    Represents a specific version of a schema.
    """
    schema_id: int
    subject: str
    version: int
    schema_type: SchemaType
    schema_definition: Dict[str, Any]
    fingerprint: str
    created_at: datetime
    created_by: str = "system"
    deprecated: bool = False
    deprecation_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_id": self.schema_id,
            "subject": self.subject,
            "version": self.version,
            "schema_type": self.schema_type.value,
            "schema_definition": self.schema_definition,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "deprecated": self.deprecated,
            "deprecation_reason": self.deprecation_reason,
        }


@dataclass
class Subject:
    """
    Schema subject (topic).

    Groups schema versions for a specific Kafka topic.
    """
    name: str
    compatibility: SchemaCompatibility
    schema_type: SchemaType
    created_at: datetime
    versions: Dict[int, SchemaVersion] = field(default_factory=dict)
    latest_version: int = 0
    description: str = ""
    owner: str = ""
    tags: Set[str] = field(default_factory=set)

    def get_latest(self) -> Optional[SchemaVersion]:
        """Get latest schema version."""
        if self.latest_version in self.versions:
            return self.versions[self.latest_version]
        return None

    def get_version(self, version: int) -> Optional[SchemaVersion]:
        """Get specific schema version."""
        return self.versions.get(version)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "compatibility": self.compatibility.value,
            "schema_type": self.schema_type.value,
            "created_at": self.created_at.isoformat(),
            "versions": {v: sv.to_dict() for v, sv in self.versions.items()},
            "latest_version": self.latest_version,
            "description": self.description,
            "owner": self.owner,
            "tags": list(self.tags),
        }


class SchemaRegistry:
    """
    Schema registry for GL-003 UNIFIEDSTEAM.

    Manages schema versions, compatibility checking, and evolution
    for Kafka message schemas.

    Example:
        >>> registry = SchemaRegistry()
        >>> registry.create_subject("gl003.site1.utilities.raw")
        >>> version = registry.register_schema(
        ...     "gl003.site1.utilities.raw",
        ...     get_raw_signal_avro(),
        ... )
    """

    def __init__(
        self,
        default_compatibility: SchemaCompatibility = SchemaCompatibility.BACKWARD,
    ):
        """
        Initialize schema registry.

        Args:
            default_compatibility: Default compatibility mode for new subjects
        """
        self.default_compatibility = default_compatibility
        self._subjects: Dict[str, Subject] = {}
        self._schemas_by_id: Dict[int, SchemaVersion] = {}
        self._next_schema_id = 1
        self._audit_log: List[Dict[str, Any]] = []

    def create_subject(
        self,
        name: str,
        schema_type: SchemaType = SchemaType.AVRO,
        compatibility: Optional[SchemaCompatibility] = None,
        description: str = "",
        owner: str = "",
        tags: Optional[Set[str]] = None,
    ) -> Subject:
        """
        Create a new subject.

        Args:
            name: Subject name (typically topic name)
            schema_type: Type of schema (avro, json, protobuf)
            compatibility: Compatibility mode
            description: Subject description
            owner: Subject owner
            tags: Subject tags

        Returns:
            Created Subject

        Raises:
            ValueError: If subject already exists
        """
        if name in self._subjects:
            raise ValueError(f"Subject already exists: {name}")

        subject = Subject(
            name=name,
            compatibility=compatibility or self.default_compatibility,
            schema_type=schema_type,
            created_at=datetime.now(timezone.utc),
            description=description,
            owner=owner,
            tags=tags or set(),
        )

        self._subjects[name] = subject

        self._log_action("create_subject", {
            "subject": name,
            "compatibility": subject.compatibility.value,
        })

        logger.info(f"Created subject: {name}")
        return subject

    def register_schema(
        self,
        subject_name: str,
        schema_definition: Dict[str, Any],
        created_by: str = "system",
    ) -> SchemaVersion:
        """
        Register a new schema version.

        Args:
            subject_name: Subject to register under
            schema_definition: Schema definition
            created_by: User registering schema

        Returns:
            Registered SchemaVersion

        Raises:
            KeyError: If subject doesn't exist
            ValueError: If schema is incompatible
        """
        if subject_name not in self._subjects:
            raise KeyError(f"Subject not found: {subject_name}")

        subject = self._subjects[subject_name]

        # Calculate fingerprint
        fingerprint = self._compute_fingerprint(schema_definition)

        # Check if schema already exists
        for version in subject.versions.values():
            if version.fingerprint == fingerprint:
                logger.info(f"Schema already registered: {subject_name} v{version.version}")
                return version

        # Check compatibility
        if subject.latest_version > 0:
            latest = subject.get_latest()
            if not self._check_compatibility(
                subject.compatibility,
                latest.schema_definition,
                schema_definition,
            ):
                raise ValueError(
                    f"Schema is not {subject.compatibility.value} compatible "
                    f"with version {subject.latest_version}"
                )

        # Create new version
        new_version = subject.latest_version + 1
        schema_id = self._next_schema_id
        self._next_schema_id += 1

        schema_version = SchemaVersion(
            schema_id=schema_id,
            subject=subject_name,
            version=new_version,
            schema_type=subject.schema_type,
            schema_definition=schema_definition,
            fingerprint=fingerprint,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
        )

        subject.versions[new_version] = schema_version
        subject.latest_version = new_version
        self._schemas_by_id[schema_id] = schema_version

        self._log_action("register_schema", {
            "subject": subject_name,
            "version": new_version,
            "schema_id": schema_id,
        })

        logger.info(f"Registered schema: {subject_name} v{new_version} (id={schema_id})")
        return schema_version

    def get_schema(self, schema_id: int) -> Optional[SchemaVersion]:
        """
        Get schema by ID.

        Args:
            schema_id: Schema ID

        Returns:
            SchemaVersion or None
        """
        return self._schemas_by_id.get(schema_id)

    def get_latest_schema(self, subject_name: str) -> Optional[SchemaVersion]:
        """
        Get latest schema for subject.

        Args:
            subject_name: Subject name

        Returns:
            Latest SchemaVersion or None
        """
        if subject_name not in self._subjects:
            return None
        return self._subjects[subject_name].get_latest()

    def get_schema_version(
        self,
        subject_name: str,
        version: int,
    ) -> Optional[SchemaVersion]:
        """
        Get specific schema version.

        Args:
            subject_name: Subject name
            version: Version number

        Returns:
            SchemaVersion or None
        """
        if subject_name not in self._subjects:
            return None
        return self._subjects[subject_name].get_version(version)

    def list_subjects(self) -> List[str]:
        """List all subject names."""
        return list(self._subjects.keys())

    def list_versions(self, subject_name: str) -> List[int]:
        """
        List all versions for a subject.

        Args:
            subject_name: Subject name

        Returns:
            List of version numbers
        """
        if subject_name not in self._subjects:
            return []
        return sorted(self._subjects[subject_name].versions.keys())

    def deprecate_schema(
        self,
        subject_name: str,
        version: int,
        reason: str = "",
        deprecated_by: str = "system",
    ) -> SchemaVersion:
        """
        Deprecate a schema version.

        Args:
            subject_name: Subject name
            version: Version to deprecate
            reason: Deprecation reason
            deprecated_by: User deprecating

        Returns:
            Updated SchemaVersion
        """
        schema = self.get_schema_version(subject_name, version)
        if not schema:
            raise KeyError(f"Schema not found: {subject_name} v{version}")

        schema.deprecated = True
        schema.deprecation_reason = reason

        self._log_action("deprecate_schema", {
            "subject": subject_name,
            "version": version,
            "reason": reason,
            "deprecated_by": deprecated_by,
        })

        logger.warning(f"Deprecated schema: {subject_name} v{version} - {reason}")
        return schema

    def update_compatibility(
        self,
        subject_name: str,
        compatibility: SchemaCompatibility,
    ) -> Subject:
        """
        Update compatibility mode for a subject.

        Args:
            subject_name: Subject name
            compatibility: New compatibility mode

        Returns:
            Updated Subject
        """
        if subject_name not in self._subjects:
            raise KeyError(f"Subject not found: {subject_name}")

        subject = self._subjects[subject_name]
        old_compat = subject.compatibility
        subject.compatibility = compatibility

        self._log_action("update_compatibility", {
            "subject": subject_name,
            "old": old_compat.value,
            "new": compatibility.value,
        })

        return subject

    def test_compatibility(
        self,
        subject_name: str,
        schema_definition: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Test if a schema is compatible with subject.

        Args:
            subject_name: Subject name
            schema_definition: Schema to test

        Returns:
            Compatibility test results
        """
        if subject_name not in self._subjects:
            return {
                "compatible": False,
                "error": f"Subject not found: {subject_name}",
            }

        subject = self._subjects[subject_name]
        latest = subject.get_latest()

        if not latest:
            return {
                "compatible": True,
                "message": "No existing schema - first version",
            }

        is_compatible = self._check_compatibility(
            subject.compatibility,
            latest.schema_definition,
            schema_definition,
        )

        return {
            "compatible": is_compatible,
            "compatibility_mode": subject.compatibility.value,
            "latest_version": subject.latest_version,
        }

    def _check_compatibility(
        self,
        mode: SchemaCompatibility,
        old_schema: Dict[str, Any],
        new_schema: Dict[str, Any],
    ) -> bool:
        """
        Check schema compatibility.

        Simplified implementation - in production use Avro tools.

        Args:
            mode: Compatibility mode
            old_schema: Existing schema
            new_schema: New schema to check

        Returns:
            True if compatible
        """
        if mode == SchemaCompatibility.NONE:
            return True

        # Simplified compatibility check
        # In production, use fastavro or confluent-kafka
        old_fields = set()
        new_fields = set()

        if "fields" in old_schema:
            old_fields = {f["name"] for f in old_schema["fields"]}
        if "fields" in new_schema:
            new_fields = {f["name"] for f in new_schema["fields"]}

        if mode == SchemaCompatibility.BACKWARD:
            # New schema can read old data
            # Old fields must exist in new schema (or have defaults)
            return old_fields.issubset(new_fields)

        elif mode == SchemaCompatibility.FORWARD:
            # Old schema can read new data
            # New fields must exist in old schema (or have defaults)
            return new_fields.issubset(old_fields)

        elif mode == SchemaCompatibility.FULL:
            # Both directions
            return old_fields == new_fields

        return True

    def _compute_fingerprint(self, schema: Dict[str, Any]) -> str:
        """Compute schema fingerprint."""
        canonical = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]

    def export_registry(self) -> Dict[str, Any]:
        """Export complete registry state."""
        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "subjects": {
                name: subject.to_dict()
                for name, subject in self._subjects.items()
            },
            "next_schema_id": self._next_schema_id,
        }

    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log action to audit trail."""
        self._audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **details,
        })

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit log."""
        return self._audit_log.copy()


# Pre-configured subjects for GL-003
def setup_gl003_subjects(registry: SchemaRegistry, site: str = "site1") -> None:
    """
    Set up standard GL-003 Kafka subjects.

    Args:
        registry: Schema registry instance
        site: Site identifier
    """
    from .avro_schemas import (
        get_raw_signal_avro,
        get_validated_signal_avro,
        get_feature_avro,
        get_computed_avro,
        get_recommendation_avro,
        get_event_avro,
    )

    subjects = [
        (f"gl003.{site}.utilities.raw", get_raw_signal_avro(), "Raw OT signals"),
        (f"gl003.{site}.utilities.validated", get_validated_signal_avro(), "Validated signals"),
        (f"gl003.{site}.utilities.features", get_feature_avro(), "ML features"),
        (f"gl003.{site}.utilities.computed", get_computed_avro(), "Computed properties"),
        (f"gl003.{site}.utilities.recommendations", get_recommendation_avro(), "Recommendations"),
        (f"gl003.{site}.utilities.events", get_event_avro(), "Events and alarms"),
    ]

    for subject_name, schema, description in subjects:
        registry.create_subject(
            name=subject_name,
            description=description,
            owner="GL-003",
            tags={"gl003", "steam", site},
        )
        registry.register_schema(subject_name, schema)

    logger.info(f"Set up {len(subjects)} GL-003 subjects for site {site}")
