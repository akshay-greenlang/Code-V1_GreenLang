"""
Feature Store for GL-003 UNIFIEDSTEAM

Provides feature definition, versioning, lineage tracking, and
governance for ML features used in steam system models.

Author: GL-003 MLOps Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Feature data types."""
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    TIMESTAMP = "timestamp"
    ARRAY = "array"
    EMBEDDING = "embedding"


class FeatureSource(Enum):
    """Sources of features."""
    RAW_SENSOR = "raw_sensor"
    DERIVED = "derived"
    AGGREGATED = "aggregated"
    EXTERNAL = "external"
    MODEL_OUTPUT = "model_output"


class FeatureStatus(Enum):
    """Feature lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class FeatureLineage:
    """
    Feature lineage tracking.

    Records where features come from and how they're computed.
    """
    source_type: FeatureSource
    source_tags: List[str]
    transformation: str
    dependencies: List[str]
    computation_sql: Optional[str] = None
    computation_python: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_type": self.source_type.value,
            "source_tags": self.source_tags,
            "transformation": self.transformation,
            "dependencies": self.dependencies,
            "computation_sql": self.computation_sql,
            "computation_python": self.computation_python,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class FeatureStatistics:
    """
    Feature statistics for monitoring and validation.

    Captures distribution characteristics.
    """
    computed_at: datetime
    sample_count: int
    null_count: int
    null_pct: float

    # Numeric statistics
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    median: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None

    # Categorical statistics
    unique_count: Optional[int] = None
    top_values: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "computed_at": self.computed_at.isoformat(),
            "sample_count": self.sample_count,
            "null_count": self.null_count,
            "null_pct": self.null_pct,
        }

        for metric in ["mean", "std", "min_value", "max_value",
                      "median", "p25", "p75", "unique_count"]:
            value = getattr(self, metric)
            if value is not None:
                result[metric] = value

        if self.top_values:
            result["top_values"] = self.top_values

        return result


@dataclass
class FeatureDefinition:
    """
    Feature definition with metadata and governance.

    Complete specification of a feature for ML models.
    """
    feature_id: str
    name: str
    description: str
    feature_type: FeatureType
    source: FeatureSource
    status: FeatureStatus

    # Versioning
    version: str
    created_at: datetime
    created_by: str
    last_updated: datetime

    # Technical details
    data_type: str  # e.g., "float64", "int32"
    nullable: bool = True
    default_value: Optional[Any] = None

    # Validation
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    validation_regex: Optional[str] = None

    # Lineage
    lineage: Optional[FeatureLineage] = None

    # Statistics
    statistics: Optional[FeatureStatistics] = None

    # Governance
    owner: str = ""
    team: str = ""
    tags: Set[str] = field(default_factory=set)
    models_using: Set[str] = field(default_factory=set)

    # Documentation
    business_description: str = ""
    example_values: List[Any] = field(default_factory=list)
    notes: str = ""

    def validate_value(self, value: Any) -> bool:
        """
        Validate a feature value.

        Args:
            value: Value to validate

        Returns:
            True if valid
        """
        if value is None:
            return self.nullable

        if self.feature_type == FeatureType.FLOAT:
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

        elif self.feature_type == FeatureType.INTEGER:
            if not isinstance(value, int):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

        elif self.feature_type == FeatureType.CATEGORICAL:
            if self.allowed_values and value not in self.allowed_values:
                return False

        elif self.feature_type == FeatureType.BOOLEAN:
            if not isinstance(value, bool):
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "description": self.description,
            "feature_type": self.feature_type.value,
            "source": self.source.value,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "last_updated": self.last_updated.isoformat(),
            "technical": {
                "data_type": self.data_type,
                "nullable": self.nullable,
                "default_value": self.default_value,
            },
            "validation": {
                "min_value": self.min_value,
                "max_value": self.max_value,
                "allowed_values": self.allowed_values,
            },
            "lineage": self.lineage.to_dict() if self.lineage else None,
            "statistics": self.statistics.to_dict() if self.statistics else None,
            "governance": {
                "owner": self.owner,
                "team": self.team,
                "tags": list(self.tags),
                "models_using": list(self.models_using),
            },
            "documentation": {
                "business_description": self.business_description,
                "example_values": self.example_values,
                "notes": self.notes,
            },
        }


@dataclass
class FeatureGroup:
    """
    Group of related features.

    Organizes features by domain or use case.
    """
    group_id: str
    name: str
    description: str
    created_at: datetime
    created_by: str

    # Features
    features: Dict[str, FeatureDefinition] = field(default_factory=dict)

    # Metadata
    entity_type: str = ""  # e.g., "steam_trap", "header", "boiler"
    refresh_frequency: str = ""  # e.g., "1m", "5m", "1h"
    owner: str = ""
    tags: Set[str] = field(default_factory=set)

    def add_feature(self, feature: FeatureDefinition):
        """Add feature to group."""
        self.features[feature.feature_id] = feature

    def get_feature(self, feature_id: str) -> Optional[FeatureDefinition]:
        """Get feature by ID."""
        return self.features.get(feature_id)

    def list_features(self) -> List[FeatureDefinition]:
        """List all features in group."""
        return list(self.features.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "group_id": self.group_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "features": {
                fid: f.to_dict() for fid, f in self.features.items()
            },
            "entity_type": self.entity_type,
            "refresh_frequency": self.refresh_frequency,
            "owner": self.owner,
            "tags": list(self.tags),
        }


class FeatureStore:
    """
    Centralized feature store for GL-003 UNIFIEDSTEAM.

    Manages feature definitions, groups, lineage, and governance.

    Example:
        >>> store = FeatureStore()
        >>> feature = store.register_feature(
        ...     feature_id="acoustic_rms_db",
        ...     name="Acoustic RMS Level",
        ...     feature_type=FeatureType.FLOAT,
        ...     source=FeatureSource.RAW_SENSOR,
        ... )
    """

    def __init__(self):
        """Initialize feature store."""
        self._features: Dict[str, FeatureDefinition] = {}
        self._groups: Dict[str, FeatureGroup] = {}
        self._audit_log: List[Dict[str, Any]] = []

    def register_feature(
        self,
        feature_id: str,
        name: str,
        description: str,
        feature_type: FeatureType,
        source: FeatureSource,
        data_type: str = "float64",
        owner: str = "",
        team: str = "",
        created_by: str = "system",
        version: str = "1.0.0",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allowed_values: Optional[List[Any]] = None,
        lineage: Optional[FeatureLineage] = None,
        tags: Optional[Set[str]] = None,
    ) -> FeatureDefinition:
        """
        Register a new feature.

        Args:
            feature_id: Unique feature identifier
            name: Human-readable name
            description: Feature description
            feature_type: Type of feature
            source: Source of feature
            data_type: Technical data type
            owner: Feature owner
            team: Owning team
            created_by: User creating feature
            version: Feature version
            min_value: Minimum valid value
            max_value: Maximum valid value
            allowed_values: List of allowed values
            lineage: Feature lineage information
            tags: Feature tags

        Returns:
            FeatureDefinition
        """
        if feature_id in self._features:
            raise ValueError(f"Feature already exists: {feature_id}")

        now = datetime.now(timezone.utc)

        feature = FeatureDefinition(
            feature_id=feature_id,
            name=name,
            description=description,
            feature_type=feature_type,
            source=source,
            status=FeatureStatus.DRAFT,
            version=version,
            created_at=now,
            created_by=created_by,
            last_updated=now,
            data_type=data_type,
            min_value=min_value,
            max_value=max_value,
            allowed_values=allowed_values,
            lineage=lineage,
            owner=owner,
            team=team,
            tags=tags or set(),
        )

        self._features[feature_id] = feature

        self._log_action("register_feature", {
            "feature_id": feature_id,
            "name": name,
            "type": feature_type.value,
        })

        logger.info(f"Registered feature: {feature_id} ({name})")
        return feature

    def create_group(
        self,
        group_id: str,
        name: str,
        description: str,
        entity_type: str = "",
        refresh_frequency: str = "",
        owner: str = "",
        created_by: str = "system",
        tags: Optional[Set[str]] = None,
    ) -> FeatureGroup:
        """
        Create a feature group.

        Args:
            group_id: Unique group identifier
            name: Group name
            description: Group description
            entity_type: Entity type for features
            refresh_frequency: How often features refresh
            owner: Group owner
            created_by: User creating group
            tags: Group tags

        Returns:
            FeatureGroup
        """
        if group_id in self._groups:
            raise ValueError(f"Group already exists: {group_id}")

        group = FeatureGroup(
            group_id=group_id,
            name=name,
            description=description,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            entity_type=entity_type,
            refresh_frequency=refresh_frequency,
            owner=owner,
            tags=tags or set(),
        )

        self._groups[group_id] = group

        self._log_action("create_group", {
            "group_id": group_id,
            "name": name,
        })

        return group

    def add_feature_to_group(
        self,
        feature_id: str,
        group_id: str,
    ):
        """
        Add a feature to a group.

        Args:
            feature_id: Feature to add
            group_id: Group to add to
        """
        if feature_id not in self._features:
            raise KeyError(f"Feature not found: {feature_id}")
        if group_id not in self._groups:
            raise KeyError(f"Group not found: {group_id}")

        feature = self._features[feature_id]
        group = self._groups[group_id]
        group.add_feature(feature)

        self._log_action("add_to_group", {
            "feature_id": feature_id,
            "group_id": group_id,
        })

    def activate_feature(
        self,
        feature_id: str,
        activated_by: str = "system",
    ) -> FeatureDefinition:
        """
        Activate a feature for use.

        Args:
            feature_id: Feature to activate
            activated_by: User activating

        Returns:
            Updated FeatureDefinition
        """
        if feature_id not in self._features:
            raise KeyError(f"Feature not found: {feature_id}")

        feature = self._features[feature_id]
        feature.status = FeatureStatus.ACTIVE
        feature.last_updated = datetime.now(timezone.utc)

        self._log_action("activate_feature", {
            "feature_id": feature_id,
            "activated_by": activated_by,
        })

        return feature

    def deprecate_feature(
        self,
        feature_id: str,
        reason: str = "",
        deprecated_by: str = "system",
    ) -> FeatureDefinition:
        """
        Deprecate a feature.

        Args:
            feature_id: Feature to deprecate
            reason: Deprecation reason
            deprecated_by: User deprecating

        Returns:
            Updated FeatureDefinition
        """
        if feature_id not in self._features:
            raise KeyError(f"Feature not found: {feature_id}")

        feature = self._features[feature_id]
        feature.status = FeatureStatus.DEPRECATED
        feature.last_updated = datetime.now(timezone.utc)
        feature.notes = f"Deprecated: {reason}"

        self._log_action("deprecate_feature", {
            "feature_id": feature_id,
            "reason": reason,
            "deprecated_by": deprecated_by,
        })

        logger.warning(f"Feature deprecated: {feature_id} - {reason}")
        return feature

    def register_model_usage(
        self,
        feature_id: str,
        model_id: str,
    ):
        """
        Register that a model uses a feature.

        Args:
            feature_id: Feature being used
            model_id: Model using the feature
        """
        if feature_id not in self._features:
            raise KeyError(f"Feature not found: {feature_id}")

        feature = self._features[feature_id]
        feature.models_using.add(model_id)

    def update_statistics(
        self,
        feature_id: str,
        statistics: FeatureStatistics,
    ):
        """
        Update feature statistics.

        Args:
            feature_id: Feature to update
            statistics: New statistics
        """
        if feature_id not in self._features:
            raise KeyError(f"Feature not found: {feature_id}")

        self._features[feature_id].statistics = statistics

    def get_feature(self, feature_id: str) -> Optional[FeatureDefinition]:
        """Get feature by ID."""
        return self._features.get(feature_id)

    def get_group(self, group_id: str) -> Optional[FeatureGroup]:
        """Get group by ID."""
        return self._groups.get(group_id)

    def list_features(
        self,
        status: Optional[FeatureStatus] = None,
        source: Optional[FeatureSource] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[FeatureDefinition]:
        """
        List features with optional filtering.

        Args:
            status: Filter by status
            source: Filter by source
            tags: Filter by tags (all must match)

        Returns:
            List of matching features
        """
        features = list(self._features.values())

        if status:
            features = [f for f in features if f.status == status]
        if source:
            features = [f for f in features if f.source == source]
        if tags:
            features = [f for f in features if tags.issubset(f.tags)]

        return features

    def get_feature_vector(
        self,
        feature_ids: List[str],
    ) -> Dict[str, FeatureDefinition]:
        """
        Get multiple features as a vector.

        Args:
            feature_ids: List of feature IDs

        Returns:
            Dictionary of features
        """
        return {
            fid: self._features[fid]
            for fid in feature_ids
            if fid in self._features
        }

    def validate_feature_vector(
        self,
        values: Dict[str, Any],
    ) -> Dict[str, bool]:
        """
        Validate a feature vector.

        Args:
            values: Dictionary of feature values

        Returns:
            Dictionary of validation results
        """
        results = {}
        for feature_id, value in values.items():
            if feature_id in self._features:
                feature = self._features[feature_id]
                results[feature_id] = feature.validate_value(value)
            else:
                results[feature_id] = False
        return results

    def export_catalog(self) -> Dict[str, Any]:
        """Export complete feature catalog."""
        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "features": {
                fid: f.to_dict() for fid, f in self._features.items()
            },
            "groups": {
                gid: g.to_dict() for gid, g in self._groups.items()
            },
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


# Pre-defined features for GL-003
def create_steam_trap_features(store: FeatureStore) -> FeatureGroup:
    """Create standard steam trap feature group."""
    group = store.create_group(
        group_id="steam_trap_features",
        name="Steam Trap Features",
        description="Features for steam trap failure prediction",
        entity_type="steam_trap",
        refresh_frequency="5m",
        owner="ML Team",
    )

    # Acoustic features
    store.register_feature(
        feature_id="acoustic_rms_db",
        name="Acoustic RMS Level",
        description="Root mean square acoustic level in decibels",
        feature_type=FeatureType.FLOAT,
        source=FeatureSource.RAW_SENSOR,
        data_type="float64",
        min_value=20.0,
        max_value=120.0,
        tags={"acoustic", "sensor", "trap"},
    )
    store.add_feature_to_group("acoustic_rms_db", "steam_trap_features")

    store.register_feature(
        feature_id="acoustic_peak_freq_hz",
        name="Acoustic Peak Frequency",
        description="Peak frequency in acoustic spectrum",
        feature_type=FeatureType.FLOAT,
        source=FeatureSource.DERIVED,
        data_type="float64",
        min_value=0.0,
        max_value=50000.0,
        tags={"acoustic", "derived", "trap"},
    )
    store.add_feature_to_group("acoustic_peak_freq_hz", "steam_trap_features")

    # Process features
    store.register_feature(
        feature_id="inlet_pressure_kpa",
        name="Inlet Pressure",
        description="Steam trap inlet pressure in kPa",
        feature_type=FeatureType.FLOAT,
        source=FeatureSource.RAW_SENSOR,
        data_type="float64",
        min_value=100.0,
        max_value=4000.0,
        tags={"process", "sensor", "trap"},
    )
    store.add_feature_to_group("inlet_pressure_kpa", "steam_trap_features")

    store.register_feature(
        feature_id="pressure_differential_kpa",
        name="Pressure Differential",
        description="Pressure drop across steam trap",
        feature_type=FeatureType.FLOAT,
        source=FeatureSource.DERIVED,
        data_type="float64",
        min_value=0.0,
        max_value=2000.0,
        tags={"process", "derived", "trap"},
    )
    store.add_feature_to_group("pressure_differential_kpa", "steam_trap_features")

    # Activate features
    for fid in ["acoustic_rms_db", "acoustic_peak_freq_hz",
                "inlet_pressure_kpa", "pressure_differential_kpa"]:
        store.activate_feature(fid)

    return group
