# -*- coding: utf-8 -*-
"""
Model Registry for Fouling Prediction - GL-014 Exchangerpro Agent.

Provides model lifecycle management:
- Model versioning and artifact storage
- Training data snapshot IDs
- Feature schema versions
- Performance metrics tracking
- Model promotion and rollback

All operations maintain complete audit trail with provenance tracking.

Author: GreenLang AI Team
Version: 1.0.0
"""

import logging
import hashlib
import json
import time
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ModelStatus(str, Enum):
    """Model lifecycle status."""
    DRAFT = "draft"  # In development
    STAGING = "staging"  # Ready for validation
    PRODUCTION = "production"  # Active in production
    DEPRECATED = "deprecated"  # No longer recommended
    ARCHIVED = "archived"  # Historical reference only


class ModelStage(str, Enum):
    """Model deployment stage."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureSchema:
    """Feature schema definition."""

    version: str
    created_at: datetime
    feature_names: List[str]
    feature_types: Dict[str, str]  # feature_name -> dtype
    feature_descriptions: Dict[str, str] = field(default_factory=dict)

    # Statistics from training data
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)
    feature_mins: Dict[str, float] = field(default_factory=dict)
    feature_maxs: Dict[str, float] = field(default_factory=dict)

    # Validation constraints
    required_features: List[str] = field(default_factory=list)
    optional_features: List[str] = field(default_factory=list)

    # Provenance
    provenance_hash: str = ""

    def validate_features(self, feature_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input features against schema.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check required features
        for feat in self.required_features:
            if feat not in feature_dict:
                errors.append(f"Missing required feature: {feat}")

        # Check types (basic validation)
        for feat, value in feature_dict.items():
            if feat in self.feature_types:
                expected_type = self.feature_types[feat]
                if expected_type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"Feature {feat} should be numeric, got {type(value)}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSchema":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class TrainingSnapshot:
    """Training data snapshot metadata."""

    snapshot_id: str
    created_at: datetime

    # Data statistics
    n_samples: int
    n_features: int
    date_range_start: datetime
    date_range_end: datetime

    # Data sources
    data_sources: List[str] = field(default_factory=list)
    exchanger_ids: List[str] = field(default_factory=list)

    # Target statistics
    target_name: str = ""
    target_mean: float = 0.0
    target_std: float = 0.0
    target_min: float = 0.0
    target_max: float = 0.0

    # Quality metrics
    data_quality_score: float = 1.0
    missing_value_ratio: float = 0.0

    # Storage
    storage_path: str = ""
    storage_format: str = "parquet"  # parquet, csv, pickle

    # Provenance
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["date_range_start"] = self.date_range_start.isoformat()
        data["date_range_end"] = self.date_range_end.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingSnapshot":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["date_range_start"] = datetime.fromisoformat(data["date_range_start"])
        data["date_range_end"] = datetime.fromisoformat(data["date_range_end"])
        return cls(**data)


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""

    # Regression metrics
    mae: float = 0.0  # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Squared Error
    mape: float = 0.0  # Mean Absolute Percentage Error
    r2: float = 0.0  # R-squared

    # Quantile metrics
    pinball_loss: Dict[str, float] = field(default_factory=dict)  # By quantile
    coverage: Dict[str, float] = field(default_factory=dict)  # CI coverage

    # Business metrics
    warning_precision: float = 0.0  # Precision for warning predictions
    warning_recall: float = 0.0  # Recall for warning predictions
    critical_precision: float = 0.0
    critical_recall: float = 0.0

    # Timing metrics
    avg_inference_time_ms: float = 0.0
    p95_inference_time_ms: float = 0.0

    # Validation metadata
    validation_date: Optional[datetime] = None
    validation_dataset_id: str = ""
    n_validation_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.validation_date:
            data["validation_date"] = self.validation_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create from dictionary."""
        if data.get("validation_date"):
            data["validation_date"] = datetime.fromisoformat(data["validation_date"])
        return cls(**data)


@dataclass
class ModelVersion:
    """Model version metadata."""

    version_id: str
    model_name: str
    version_number: str  # Semantic versioning: MAJOR.MINOR.PATCH

    # Status
    status: ModelStatus = ModelStatus.DRAFT
    stage: ModelStage = ModelStage.DEVELOPMENT

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    promoted_at: Optional[datetime] = None

    # References
    feature_schema_version: str = ""
    training_snapshot_id: str = ""

    # Model type and configuration
    model_type: str = ""  # xgboost, lightgbm, gradient_boosting
    model_config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Performance
    metrics: Optional[PerformanceMetrics] = None

    # Storage
    artifact_path: str = ""
    artifact_size_bytes: int = 0
    artifact_hash: str = ""

    # Description and notes
    description: str = ""
    changelog: str = ""
    tags: List[str] = field(default_factory=list)

    # Author
    created_by: str = ""

    # Provenance
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "version_number": self.version_number,
            "status": self.status.value,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "feature_schema_version": self.feature_schema_version,
            "training_snapshot_id": self.training_snapshot_id,
            "model_type": self.model_type,
            "model_config": self.model_config,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "artifact_path": self.artifact_path,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_hash": self.artifact_hash,
            "description": self.description,
            "changelog": self.changelog,
            "tags": self.tags,
            "created_by": self.created_by,
            "provenance_hash": self.provenance_hash,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        data["status"] = ModelStatus(data["status"])
        data["stage"] = ModelStage(data["stage"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if data.get("promoted_at"):
            data["promoted_at"] = datetime.fromisoformat(data["promoted_at"])
        if data.get("metrics"):
            data["metrics"] = PerformanceMetrics.from_dict(data["metrics"])
        return cls(**data)


@dataclass
class ModelArtifact:
    """Model artifact with serialized model and metadata."""

    version: ModelVersion
    model_bytes: bytes
    feature_schema: FeatureSchema
    training_snapshot: Optional[TrainingSnapshot] = None

    # Additional metadata
    dependencies: Dict[str, str] = field(default_factory=dict)  # package -> version
    custom_objects: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelRegistryConfig:
    """Configuration for model registry."""

    # Storage paths
    registry_path: str = "./model_registry"
    artifact_path: str = "./model_registry/artifacts"
    metadata_path: str = "./model_registry/metadata"

    # Versioning
    auto_increment_version: bool = True

    # Retention
    max_versions_per_model: int = 10
    archive_after_days: int = 90

    # Validation requirements
    require_metrics_for_promotion: bool = True
    min_validation_samples: int = 100


# =============================================================================
# Model Registry
# =============================================================================

class FoulingModelRegistry:
    """
    Model registry for fouling prediction models.

    Provides:
    - Model versioning and lifecycle management
    - Artifact storage and retrieval
    - Feature schema versioning
    - Training data snapshot tracking
    - Performance metrics storage
    - Model promotion and rollback

    Example:
        >>> registry = FoulingModelRegistry(config)
        >>> version = registry.register_model(
        ...     model=trained_model,
        ...     model_name="ua_predictor_7d",
        ...     feature_schema=schema,
        ...     training_snapshot=snapshot,
        ...     metrics=metrics,
        ... )
        >>> registry.promote_to_production(version.version_id)
        >>> prod_model = registry.get_production_model("ua_predictor_7d")
    """

    def __init__(self, config: Optional[ModelRegistryConfig] = None):
        """
        Initialize model registry.

        Args:
            config: Registry configuration
        """
        self.config = config or ModelRegistryConfig()

        # In-memory storage (would be database in production)
        self._versions: Dict[str, ModelVersion] = {}
        self._artifacts: Dict[str, bytes] = {}
        self._schemas: Dict[str, FeatureSchema] = {}
        self._snapshots: Dict[str, TrainingSnapshot] = {}

        # Index for quick lookups
        self._model_versions: Dict[str, List[str]] = {}  # model_name -> [version_ids]
        self._production_versions: Dict[str, str] = {}  # model_name -> version_id

        # Ensure directories exist
        self._ensure_directories()

        logger.info(f"FoulingModelRegistry initialized at {self.config.registry_path}")

    def _ensure_directories(self) -> None:
        """Ensure storage directories exist."""
        Path(self.config.registry_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.artifact_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.metadata_path).mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model: Any,
        model_name: str,
        feature_schema: FeatureSchema,
        training_snapshot: Optional[TrainingSnapshot] = None,
        metrics: Optional[PerformanceMetrics] = None,
        model_type: str = "unknown",
        model_config: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        created_by: str = "system",
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model: Trained model object
            model_name: Name of the model
            feature_schema: Feature schema for this model
            training_snapshot: Training data snapshot metadata
            metrics: Performance metrics
            model_type: Type of model (xgboost, lightgbm, etc.)
            model_config: Model configuration
            hyperparameters: Model hyperparameters
            description: Model description
            tags: Optional tags
            created_by: Creator identifier

        Returns:
            ModelVersion metadata
        """
        start_time = time.time()

        # Generate version number
        version_number = self._get_next_version(model_name)

        # Generate version ID
        version_id = self._generate_version_id(model_name, version_number)

        # Serialize model
        model_bytes = pickle.dumps(model)
        artifact_hash = hashlib.sha256(model_bytes).hexdigest()

        # Register schema if new
        if feature_schema.version not in self._schemas:
            self._schemas[feature_schema.version] = feature_schema

        # Register snapshot if provided
        if training_snapshot and training_snapshot.snapshot_id not in self._snapshots:
            self._snapshots[training_snapshot.snapshot_id] = training_snapshot

        # Create version metadata
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            version_number=version_number,
            status=ModelStatus.DRAFT,
            stage=ModelStage.DEVELOPMENT,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            feature_schema_version=feature_schema.version,
            training_snapshot_id=training_snapshot.snapshot_id if training_snapshot else "",
            model_type=model_type,
            model_config=model_config or {},
            hyperparameters=hyperparameters or {},
            metrics=metrics,
            artifact_path=f"{self.config.artifact_path}/{version_id}.pkl",
            artifact_size_bytes=len(model_bytes),
            artifact_hash=artifact_hash,
            description=description,
            tags=tags or [],
            created_by=created_by,
            provenance_hash=self._compute_provenance_hash(
                model_name, version_number, artifact_hash
            ),
        )

        # Store
        self._versions[version_id] = version
        self._artifacts[version_id] = model_bytes

        # Update index
        if model_name not in self._model_versions:
            self._model_versions[model_name] = []
        self._model_versions[model_name].append(version_id)

        # Save to disk
        self._save_version_metadata(version)
        self._save_artifact(version_id, model_bytes)

        computation_time = (time.time() - start_time) * 1000

        logger.info(
            f"Registered model {model_name} version {version_number} "
            f"(id={version_id}) in {computation_time:.1f}ms"
        )

        return version

    def get_model(
        self,
        version_id: str,
    ) -> Tuple[Any, ModelVersion, FeatureSchema]:
        """
        Get a model by version ID.

        Args:
            version_id: Model version ID

        Returns:
            Tuple of (model, version_metadata, feature_schema)
        """
        if version_id not in self._versions:
            # Try loading from disk
            version = self._load_version_metadata(version_id)
            if version is None:
                raise ValueError(f"Version {version_id} not found")
            self._versions[version_id] = version

        version = self._versions[version_id]

        # Load artifact
        if version_id not in self._artifacts:
            model_bytes = self._load_artifact(version_id)
            if model_bytes is None:
                raise ValueError(f"Artifact for {version_id} not found")
            self._artifacts[version_id] = model_bytes

        model = pickle.loads(self._artifacts[version_id])

        # Get schema
        schema = self._schemas.get(version.feature_schema_version)
        if schema is None:
            schema = self._load_schema(version.feature_schema_version)

        return model, version, schema

    def get_production_model(
        self,
        model_name: str,
    ) -> Tuple[Any, ModelVersion, FeatureSchema]:
        """
        Get the current production model for a model name.

        Args:
            model_name: Name of the model

        Returns:
            Tuple of (model, version_metadata, feature_schema)
        """
        if model_name not in self._production_versions:
            raise ValueError(f"No production model for {model_name}")

        version_id = self._production_versions[model_name]
        return self.get_model(version_id)

    def promote_to_staging(
        self,
        version_id: str,
    ) -> ModelVersion:
        """
        Promote a model version to staging.

        Args:
            version_id: Model version ID

        Returns:
            Updated ModelVersion
        """
        if version_id not in self._versions:
            raise ValueError(f"Version {version_id} not found")

        version = self._versions[version_id]

        if version.status not in [ModelStatus.DRAFT]:
            raise ValueError(
                f"Cannot promote from {version.status} to staging"
            )

        version.status = ModelStatus.STAGING
        version.stage = ModelStage.STAGING
        version.updated_at = datetime.utcnow()

        self._save_version_metadata(version)

        logger.info(f"Promoted {version_id} to staging")

        return version

    def promote_to_production(
        self,
        version_id: str,
        require_metrics: Optional[bool] = None,
    ) -> ModelVersion:
        """
        Promote a model version to production.

        Args:
            version_id: Model version ID
            require_metrics: Override config requirement for metrics

        Returns:
            Updated ModelVersion
        """
        if version_id not in self._versions:
            raise ValueError(f"Version {version_id} not found")

        version = self._versions[version_id]

        # Check requirements
        require_metrics = require_metrics if require_metrics is not None \
            else self.config.require_metrics_for_promotion

        if require_metrics and version.metrics is None:
            raise ValueError("Cannot promote to production without metrics")

        if version.status not in [ModelStatus.STAGING, ModelStatus.DRAFT]:
            raise ValueError(
                f"Cannot promote from {version.status} to production"
            )

        # Deprecate current production version
        if version.model_name in self._production_versions:
            old_version_id = self._production_versions[version.model_name]
            if old_version_id != version_id:
                old_version = self._versions[old_version_id]
                old_version.status = ModelStatus.DEPRECATED
                old_version.updated_at = datetime.utcnow()
                self._save_version_metadata(old_version)

        # Update version
        version.status = ModelStatus.PRODUCTION
        version.stage = ModelStage.PRODUCTION
        version.promoted_at = datetime.utcnow()
        version.updated_at = datetime.utcnow()

        # Update production index
        self._production_versions[version.model_name] = version_id

        self._save_version_metadata(version)

        logger.info(
            f"Promoted {version_id} to production for {version.model_name}"
        )

        return version

    def rollback(
        self,
        model_name: str,
        to_version_id: str,
    ) -> ModelVersion:
        """
        Rollback to a previous model version.

        Args:
            model_name: Name of the model
            to_version_id: Version ID to rollback to

        Returns:
            Updated ModelVersion (the rolled-back-to version)
        """
        if to_version_id not in self._versions:
            raise ValueError(f"Version {to_version_id} not found")

        version = self._versions[to_version_id]

        if version.model_name != model_name:
            raise ValueError(
                f"Version {to_version_id} is not for model {model_name}"
            )

        # Promote the target version
        return self.promote_to_production(to_version_id, require_metrics=False)

    def update_metrics(
        self,
        version_id: str,
        metrics: PerformanceMetrics,
    ) -> ModelVersion:
        """
        Update metrics for a model version.

        Args:
            version_id: Model version ID
            metrics: New performance metrics

        Returns:
            Updated ModelVersion
        """
        if version_id not in self._versions:
            raise ValueError(f"Version {version_id} not found")

        version = self._versions[version_id]
        version.metrics = metrics
        version.updated_at = datetime.utcnow()

        self._save_version_metadata(version)

        logger.info(f"Updated metrics for {version_id}")

        return version

    def list_versions(
        self,
        model_name: str,
        status: Optional[ModelStatus] = None,
        limit: int = 10,
    ) -> List[ModelVersion]:
        """
        List model versions.

        Args:
            model_name: Name of the model
            status: Optional status filter
            limit: Maximum number of versions to return

        Returns:
            List of ModelVersion
        """
        if model_name not in self._model_versions:
            return []

        versions = [
            self._versions[vid]
            for vid in self._model_versions[model_name]
            if vid in self._versions
        ]

        if status:
            versions = [v for v in versions if v.status == status]

        # Sort by created_at descending
        versions.sort(key=lambda v: v.created_at, reverse=True)

        return versions[:limit]

    def register_feature_schema(
        self,
        feature_names: List[str],
        feature_types: Dict[str, str],
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
        required_features: Optional[List[str]] = None,
        descriptions: Optional[Dict[str, str]] = None,
    ) -> FeatureSchema:
        """
        Register a new feature schema.

        Args:
            feature_names: List of feature names
            feature_types: Dict mapping feature name to type
            feature_stats: Dict with mean, std, min, max per feature
            required_features: List of required features
            descriptions: Feature descriptions

        Returns:
            FeatureSchema
        """
        version = f"1.{len(self._schemas)}.0"

        feature_stats = feature_stats or {}

        schema = FeatureSchema(
            version=version,
            created_at=datetime.utcnow(),
            feature_names=feature_names,
            feature_types=feature_types,
            feature_descriptions=descriptions or {},
            feature_means={k: v.get("mean", 0.0) for k, v in feature_stats.items()},
            feature_stds={k: v.get("std", 1.0) for k, v in feature_stats.items()},
            feature_mins={k: v.get("min", 0.0) for k, v in feature_stats.items()},
            feature_maxs={k: v.get("max", 1.0) for k, v in feature_stats.items()},
            required_features=required_features or feature_names,
            optional_features=[f for f in feature_names if f not in (required_features or [])],
            provenance_hash=hashlib.sha256(
                f"{version}{','.join(feature_names)}".encode()
            ).hexdigest(),
        )

        self._schemas[version] = schema
        self._save_schema(schema)

        logger.info(f"Registered feature schema version {version}")

        return schema

    def register_training_snapshot(
        self,
        n_samples: int,
        n_features: int,
        date_range_start: datetime,
        date_range_end: datetime,
        data_sources: List[str],
        target_name: str,
        target_stats: Dict[str, float],
        exchanger_ids: Optional[List[str]] = None,
        data_quality_score: float = 1.0,
        storage_path: str = "",
    ) -> TrainingSnapshot:
        """
        Register a training data snapshot.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            date_range_start: Start of data date range
            date_range_end: End of data date range
            data_sources: List of data source identifiers
            target_name: Name of target variable
            target_stats: Dict with mean, std, min, max
            exchanger_ids: List of exchanger IDs in data
            data_quality_score: Quality score (0-1)
            storage_path: Path where data is stored

        Returns:
            TrainingSnapshot
        """
        snapshot_id = f"snap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        snapshot = TrainingSnapshot(
            snapshot_id=snapshot_id,
            created_at=datetime.utcnow(),
            n_samples=n_samples,
            n_features=n_features,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            data_sources=data_sources,
            exchanger_ids=exchanger_ids or [],
            target_name=target_name,
            target_mean=target_stats.get("mean", 0.0),
            target_std=target_stats.get("std", 1.0),
            target_min=target_stats.get("min", 0.0),
            target_max=target_stats.get("max", 1.0),
            data_quality_score=data_quality_score,
            storage_path=storage_path,
            provenance_hash=hashlib.sha256(
                f"{snapshot_id}{n_samples}{date_range_start}".encode()
            ).hexdigest(),
        )

        self._snapshots[snapshot_id] = snapshot
        self._save_snapshot(snapshot)

        logger.info(f"Registered training snapshot {snapshot_id}")

        return snapshot

    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str,
    ) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            version_id_1: First version ID
            version_id_2: Second version ID

        Returns:
            Comparison results
        """
        v1 = self._versions.get(version_id_1)
        v2 = self._versions.get(version_id_2)

        if v1 is None or v2 is None:
            raise ValueError("One or both versions not found")

        comparison = {
            "version_1": version_id_1,
            "version_2": version_id_2,
            "model_name": v1.model_name,
            "schema_changed": v1.feature_schema_version != v2.feature_schema_version,
            "metrics_comparison": {},
        }

        if v1.metrics and v2.metrics:
            comparison["metrics_comparison"] = {
                "mae_diff": v2.metrics.mae - v1.metrics.mae,
                "rmse_diff": v2.metrics.rmse - v1.metrics.rmse,
                "r2_diff": v2.metrics.r2 - v1.metrics.r2,
                "improvement": v2.metrics.mae < v1.metrics.mae,
            }

        return comparison

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _get_next_version(self, model_name: str) -> str:
        """Get next version number for a model."""
        if model_name not in self._model_versions:
            return "1.0.0"

        versions = self._model_versions[model_name]
        if not versions:
            return "1.0.0"

        # Get latest version and increment
        latest = self._versions.get(versions[-1])
        if latest is None:
            return "1.0.0"

        parts = latest.version_number.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        if self.config.auto_increment_version:
            patch += 1
            if patch >= 100:
                patch = 0
                minor += 1
            if minor >= 100:
                minor = 0
                major += 1

        return f"{major}.{minor}.{patch}"

    def _generate_version_id(self, model_name: str, version_number: str) -> str:
        """Generate unique version ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        hash_input = f"{model_name}{version_number}{timestamp}"
        short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        return f"{model_name}_{version_number.replace('.', '_')}_{short_hash}"

    def _compute_provenance_hash(
        self,
        model_name: str,
        version_number: str,
        artifact_hash: str,
    ) -> str:
        """Compute provenance hash."""
        content = f"{model_name}|{version_number}|{artifact_hash}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _save_version_metadata(self, version: ModelVersion) -> None:
        """Save version metadata to disk."""
        path = Path(self.config.metadata_path) / f"{version.version_id}.json"
        with open(path, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

    def _load_version_metadata(self, version_id: str) -> Optional[ModelVersion]:
        """Load version metadata from disk."""
        path = Path(self.config.metadata_path) / f"{version_id}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return ModelVersion.from_dict(data)

    def _save_artifact(self, version_id: str, model_bytes: bytes) -> None:
        """Save model artifact to disk."""
        path = Path(self.config.artifact_path) / f"{version_id}.pkl"
        with open(path, "wb") as f:
            f.write(model_bytes)

    def _load_artifact(self, version_id: str) -> Optional[bytes]:
        """Load model artifact from disk."""
        path = Path(self.config.artifact_path) / f"{version_id}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return f.read()

    def _save_schema(self, schema: FeatureSchema) -> None:
        """Save feature schema to disk."""
        path = Path(self.config.metadata_path) / f"schema_{schema.version}.json"
        with open(path, "w") as f:
            json.dump(schema.to_dict(), f, indent=2)

    def _load_schema(self, version: str) -> Optional[FeatureSchema]:
        """Load feature schema from disk."""
        path = Path(self.config.metadata_path) / f"schema_{version}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return FeatureSchema.from_dict(data)

    def _save_snapshot(self, snapshot: TrainingSnapshot) -> None:
        """Save training snapshot to disk."""
        path = Path(self.config.metadata_path) / f"snapshot_{snapshot.snapshot_id}.json"
        with open(path, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_versions": len(self._versions),
            "total_models": len(self._model_versions),
            "production_models": len(self._production_versions),
            "schemas": len(self._schemas),
            "snapshots": len(self._snapshots),
            "models": list(self._model_versions.keys()),
        }
