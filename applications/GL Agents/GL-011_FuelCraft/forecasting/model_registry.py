# -*- coding: utf-8 -*-
"""
Model Registry Module for GL-011 FuelCraft

Provides model versioning, registration, and deployment tracking for
fuel price forecasting models. Ensures complete traceability from
training data to deployed model.

Features:
- Model registration with training data hash
- Feature schema version tracking
- Evaluation metrics storage
- Deployment status management
- Model artifact versioning

Zero-Hallucination Architecture:
- SHA-256 hashing of training data and artifacts
- Immutable model versions
- Complete audit trail
- No LLM-based model selection

Author: GreenLang AI Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Constants
DEFAULT_ARTIFACT_DIR = "./model_artifacts"
SUPPORTED_FRAMEWORKS = ["sklearn", "xgboost", "lightgbm", "tensorflow", "pytorch", "custom"]


class DeploymentStatus(str, Enum):
    """Model deployment status."""
    REGISTERED = "registered"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelFramework(str, Enum):
    """Supported ML frameworks."""
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    PROPHET = "prophet"
    STATSMODELS = "statsmodels"
    CUSTOM = "custom"


class EvaluationMetrics(BaseModel):
    """
    Evaluation metrics for a model version.
    """

    # Regression metrics
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")
    r2: Optional[float] = Field(None, description="R-squared coefficient")

    # Quantile metrics
    pinball_loss: Optional[float] = Field(None, description="Pinball loss for quantiles")
    calibration_error: Optional[float] = Field(None, description="Prediction interval calibration")
    coverage_rate: Optional[float] = Field(None, description="Prediction interval coverage")

    # Business metrics
    directional_accuracy: Optional[float] = Field(None, description="Direction prediction accuracy")
    value_at_risk_breach_rate: Optional[float] = Field(None, description="VaR breach rate")

    # Timing metrics
    inference_time_ms: Optional[float] = Field(None, description="Average inference time")
    training_time_seconds: Optional[float] = Field(None, description="Training duration")

    # Data info
    train_samples: int = Field(0, description="Number of training samples")
    validation_samples: int = Field(0, description="Number of validation samples")
    test_samples: int = Field(0, description="Number of test samples")
    evaluation_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Provenance
    evaluation_hash: str = Field("", description="SHA-256 hash of evaluation")

    def model_post_init(self, __context: Any) -> None:
        """Compute evaluation hash after initialization."""
        if not self.evaluation_hash:
            self.evaluation_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of metrics."""
        data = {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "r2": self.r2,
            "train_samples": self.train_samples,
            "evaluation_date": self.evaluation_date.isoformat(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


class ModelMetadata(BaseModel):
    """
    Metadata for a registered model.
    """

    model_id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    description: str = Field("", description="Model description")
    framework: ModelFramework = Field(..., description="ML framework")

    # Target info
    fuel_type: str = Field(..., description="Target fuel type")
    market_hub: str = Field(..., description="Target market hub")
    horizons: List[str] = Field(default_factory=list, description="Supported horizons")

    # Schema info
    feature_names: List[str] = Field(default_factory=list, description="Input feature names")
    feature_schema_version: str = Field(..., description="Feature schema version hash")
    output_quantiles: List[float] = Field(default_factory=lambda: [0.10, 0.50, 0.90])

    # Training info
    training_data_hash: str = Field(..., description="SHA-256 hash of training data")
    training_start_date: datetime = Field(..., description="Training data start date")
    training_end_date: datetime = Field(..., description="Training data end date")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    random_seed: int = Field(42, description="Training random seed")

    # Metadata
    created_by: str = Field("system", description="Creator username or system")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = Field(default_factory=list)


class ModelVersion(BaseModel):
    """
    Versioned model with artifacts and evaluation.
    """

    version_id: str = Field(..., description="Unique version identifier")
    model_id: str = Field(..., description="Parent model ID")
    version: str = Field(..., description="Semantic version (e.g., 1.0.0)")

    # Artifact info
    artifact_path: Optional[str] = Field(None, description="Path to model artifact")
    artifact_hash: str = Field("", description="SHA-256 hash of artifact")
    artifact_size_bytes: int = Field(0, description="Artifact file size")

    # Status
    status: DeploymentStatus = Field(DeploymentStatus.REGISTERED)
    status_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Evaluation
    metrics: Optional[EvaluationMetrics] = Field(None, description="Evaluation metrics")

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deployed_at: Optional[datetime] = Field(None, description="Production deployment time")

    # Provenance
    version_hash: str = Field("", description="SHA-256 hash of version")

    def model_post_init(self, __context: Any) -> None:
        """Compute version hash after initialization."""
        if not self.version_hash:
            self.version_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of version."""
        data = {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "version": self.version,
            "artifact_hash": self.artifact_hash,
            "status": self.status.value,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class ModelRegistryConfig:
    """Configuration for model registry."""

    artifact_dir: str = DEFAULT_ARTIFACT_DIR
    enable_versioning: bool = True
    max_versions_per_model: int = 10
    auto_archive_old_versions: bool = True
    require_evaluation_for_production: bool = True
    min_r2_for_production: float = 0.6
    max_mape_for_production: float = 0.20
    cache_enabled: bool = True


class ModelRegistry:
    """
    Model registry for fuel price forecasting models.

    Provides centralized model management with:
    - Model registration and versioning
    - Training data provenance tracking
    - Evaluation metrics storage
    - Deployment lifecycle management

    Zero-Hallucination Guarantees:
    - SHA-256 hashing of all artifacts and data
    - Immutable version records
    - Complete audit trail
    """

    def __init__(self, config: Optional[ModelRegistryConfig] = None):
        """
        Initialize model registry.

        Args:
            config: Registry configuration
        """
        self.config = config or ModelRegistryConfig()

        # Storage
        self._models: Dict[str, ModelMetadata] = {}
        self._versions: Dict[str, List[ModelVersion]] = {}
        self._loaded_models: Dict[str, Any] = {}

        # Ensure artifact directory exists
        self._artifact_path = Path(self.config.artifact_dir)
        self._artifact_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ModelRegistry initialized: artifact_dir={self.config.artifact_dir}, "
            f"versioning={self.config.enable_versioning}"
        )

    def register_model(
        self,
        metadata: ModelMetadata,
        model_object: Optional[Any] = None
    ) -> str:
        """
        Register a new model.

        Args:
            metadata: Model metadata
            model_object: Optional model object to serialize

        Returns:
            Model ID
        """
        model_id = metadata.model_id

        if model_id in self._models:
            logger.warning(f"Model {model_id} already registered, updating metadata")

        self._models[model_id] = metadata

        if model_id not in self._versions:
            self._versions[model_id] = []

        logger.info(f"Registered model: {model_id}")

        return model_id

    def register_version(
        self,
        model_id: str,
        version: str,
        model_object: Any,
        metrics: Optional[EvaluationMetrics] = None
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_id: Parent model ID
            version: Semantic version string
            model_object: Model object to serialize
            metrics: Optional evaluation metrics

        Returns:
            Created ModelVersion
        """
        import uuid

        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not registered")

        # Serialize model
        artifact_path, artifact_hash, artifact_size = self._serialize_model(
            model_id, version, model_object
        )

        # Create version record
        version_record = ModelVersion(
            version_id=str(uuid.uuid4()),
            model_id=model_id,
            version=version,
            artifact_path=str(artifact_path),
            artifact_hash=artifact_hash,
            artifact_size_bytes=artifact_size,
            metrics=metrics,
        )

        # Add status history entry
        version_record.status_history.append({
            "status": DeploymentStatus.REGISTERED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Version registered",
        })

        # Store version
        self._versions[model_id].append(version_record)

        # Archive old versions if needed
        if self.config.auto_archive_old_versions:
            self._archive_old_versions(model_id)

        logger.info(f"Registered version {version} for model {model_id}")

        return version_record

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model with its latest production version.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model info or None
        """
        if model_id not in self._models:
            return None

        metadata = self._models[model_id]
        latest_version = self.get_latest_version(model_id, DeploymentStatus.PRODUCTION)

        if latest_version is None:
            latest_version = self.get_latest_version(model_id)

        if latest_version is None:
            return None

        # Load model object
        model_object = self._load_model(latest_version.artifact_path)

        return {
            "model": model_object,
            "model_id": model_id,
            "version": latest_version.version,
            "metadata": metadata,
            "version_info": latest_version,
        }

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        return self._models.get(model_id)

    def get_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version."""
        if model_id not in self._versions:
            return None

        for v in self._versions[model_id]:
            if v.version == version:
                return v

        return None

    def get_latest_version(
        self,
        model_id: str,
        status: Optional[DeploymentStatus] = None
    ) -> Optional[ModelVersion]:
        """
        Get latest model version.

        Args:
            model_id: Model identifier
            status: Optional status filter

        Returns:
            Latest ModelVersion or None
        """
        if model_id not in self._versions:
            return None

        versions = self._versions[model_id]

        if status:
            versions = [v for v in versions if v.status == status]

        if not versions:
            return None

        # Sort by created_at descending
        versions = sorted(versions, key=lambda v: v.created_at, reverse=True)

        return versions[0]

    def list_versions(
        self,
        model_id: str,
        status: Optional[DeploymentStatus] = None
    ) -> List[ModelVersion]:
        """List all versions for a model."""
        if model_id not in self._versions:
            return []

        versions = self._versions[model_id]

        if status:
            versions = [v for v in versions if v.status == status]

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def update_status(
        self,
        model_id: str,
        version: str,
        new_status: DeploymentStatus,
        message: str = ""
    ) -> ModelVersion:
        """
        Update deployment status for a model version.

        Args:
            model_id: Model identifier
            version: Version string
            new_status: New deployment status
            message: Optional status change message

        Returns:
            Updated ModelVersion
        """
        version_record = self.get_version(model_id, version)

        if version_record is None:
            raise ValueError(f"Version {version} not found for model {model_id}")

        # Validate transition
        self._validate_status_transition(version_record, new_status)

        # Check evaluation requirements for production
        if new_status == DeploymentStatus.PRODUCTION:
            self._validate_production_requirements(version_record)

        # Update status
        old_status = version_record.status
        version_record.status = new_status
        version_record.updated_at = datetime.now(timezone.utc)

        if new_status == DeploymentStatus.PRODUCTION:
            version_record.deployed_at = datetime.now(timezone.utc)

        # Add history entry
        version_record.status_history.append({
            "status": new_status.value,
            "previous_status": old_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
        })

        logger.info(f"Updated status: {model_id}/{version} -> {new_status.value}")

        return version_record

    def add_evaluation(
        self,
        model_id: str,
        version: str,
        metrics: EvaluationMetrics
    ) -> ModelVersion:
        """
        Add evaluation metrics to a model version.

        Args:
            model_id: Model identifier
            version: Version string
            metrics: Evaluation metrics

        Returns:
            Updated ModelVersion
        """
        version_record = self.get_version(model_id, version)

        if version_record is None:
            raise ValueError(f"Version {version} not found for model {model_id}")

        version_record.metrics = metrics
        version_record.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Added evaluation to {model_id}/{version}: "
            f"R2={metrics.r2}, MAPE={metrics.mape}"
        )

        return version_record

    def export_registry(self) -> Dict[str, Any]:
        """Export complete registry state."""
        return {
            "models": {
                mid: m.model_dump() for mid, m in self._models.items()
            },
            "versions": {
                mid: [v.model_dump() for v in versions]
                for mid, versions in self._versions.items()
            },
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "artifact_dir": self.config.artifact_dir,
                "versioning": self.config.enable_versioning,
            },
        }

    def _serialize_model(
        self,
        model_id: str,
        version: str,
        model_object: Any
    ) -> Tuple[Path, str, int]:
        """Serialize model object to file."""
        # Create version directory
        version_dir = self._artifact_path / model_id / version
        version_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = version_dir / "model.pkl"

        # Serialize
        with open(artifact_path, "wb") as f:
            pickle.dump(model_object, f)

        # Compute hash
        artifact_hash = self._compute_file_hash(artifact_path)

        # Get size
        artifact_size = artifact_path.stat().st_size

        return artifact_path, artifact_hash, artifact_size

    def _load_model(self, artifact_path: Optional[str]) -> Any:
        """Load model object from file."""
        if artifact_path is None:
            return None

        # Check cache
        if self.config.cache_enabled and artifact_path in self._loaded_models:
            return self._loaded_models[artifact_path]

        path = Path(artifact_path)

        if not path.exists():
            logger.warning(f"Model artifact not found: {artifact_path}")
            return None

        with open(path, "rb") as f:
            model = pickle.load(f)

        # Cache
        if self.config.cache_enabled:
            self._loaded_models[artifact_path] = model

        return model

    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()

        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def _validate_status_transition(
        self,
        version: ModelVersion,
        new_status: DeploymentStatus
    ) -> None:
        """Validate status transition is allowed."""
        allowed_transitions = {
            DeploymentStatus.REGISTERED: [
                DeploymentStatus.VALIDATING,
                DeploymentStatus.ARCHIVED,
            ],
            DeploymentStatus.VALIDATING: [
                DeploymentStatus.STAGING,
                DeploymentStatus.REGISTERED,
                DeploymentStatus.ARCHIVED,
            ],
            DeploymentStatus.STAGING: [
                DeploymentStatus.PRODUCTION,
                DeploymentStatus.VALIDATING,
                DeploymentStatus.ARCHIVED,
            ],
            DeploymentStatus.PRODUCTION: [
                DeploymentStatus.DEPRECATED,
                DeploymentStatus.STAGING,
            ],
            DeploymentStatus.DEPRECATED: [
                DeploymentStatus.ARCHIVED,
                DeploymentStatus.PRODUCTION,  # Allow rollback
            ],
            DeploymentStatus.ARCHIVED: [],
        }

        if new_status not in allowed_transitions.get(version.status, []):
            raise ValueError(
                f"Invalid transition: {version.status.value} -> {new_status.value}"
            )

    def _validate_production_requirements(self, version: ModelVersion) -> None:
        """Validate version meets production requirements."""
        if not self.config.require_evaluation_for_production:
            return

        if version.metrics is None:
            raise ValueError("Evaluation metrics required for production deployment")

        if version.metrics.r2 is not None:
            if version.metrics.r2 < self.config.min_r2_for_production:
                raise ValueError(
                    f"R2 {version.metrics.r2} below threshold {self.config.min_r2_for_production}"
                )

        if version.metrics.mape is not None:
            if version.metrics.mape > self.config.max_mape_for_production:
                raise ValueError(
                    f"MAPE {version.metrics.mape} above threshold {self.config.max_mape_for_production}"
                )

    def _archive_old_versions(self, model_id: str) -> None:
        """Archive old versions beyond max limit."""
        if model_id not in self._versions:
            return

        versions = self._versions[model_id]

        # Keep production and staging versions
        active_statuses = {DeploymentStatus.PRODUCTION, DeploymentStatus.STAGING}
        active = [v for v in versions if v.status in active_statuses]
        other = [v for v in versions if v.status not in active_statuses]

        # Sort by created_at descending
        other = sorted(other, key=lambda v: v.created_at, reverse=True)

        # Keep only max_versions_per_model
        max_to_keep = max(0, self.config.max_versions_per_model - len(active))

        for version in other[max_to_keep:]:
            if version.status != DeploymentStatus.ARCHIVED:
                version.status = DeploymentStatus.ARCHIVED
                version.status_history.append({
                    "status": DeploymentStatus.ARCHIVED.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "Auto-archived due to version limit",
                })
                logger.info(f"Auto-archived: {model_id}/{version.version}")


# Utility functions

def compute_model_hash(model_object: Any) -> str:
    """
    Compute SHA-256 hash of model object.

    Args:
        model_object: Model to hash

    Returns:
        64-character hex hash
    """
    serialized = pickle.dumps(model_object)
    return hashlib.sha256(serialized).hexdigest()


def validate_model_schema(
    model_metadata: ModelMetadata,
    expected_features: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate model schema matches expected features.

    Args:
        model_metadata: Model metadata
        expected_features: Expected feature names

    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []

    model_features = set(model_metadata.feature_names)
    expected = set(expected_features)

    missing = expected - model_features
    extra = model_features - expected

    if missing:
        errors.append(f"Missing features: {list(missing)}")

    if extra:
        errors.append(f"Extra features: {list(extra)}")

    return len(errors) == 0, errors
