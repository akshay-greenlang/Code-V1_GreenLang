"""
Model Registry for GL-003 UNIFIEDSTEAM

Provides centralized model artifact management, versioning, and
deployment tracking for ML models in the steam optimization system.

Author: GL-003 MLOps Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import hashlib
import json
import logging

from .model_cards import ModelCard, ModelType

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class DeploymentStatus(Enum):
    """Deployment status states."""
    NOT_DEPLOYED = "not_deployed"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    CANARY = "canary"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class ArtifactType(Enum):
    """Types of model artifacts."""
    MODEL_WEIGHTS = "model_weights"
    MODEL_CONFIG = "model_config"
    FEATURE_SCALER = "feature_scaler"
    LABEL_ENCODER = "label_encoder"
    PREPROCESSING_PIPELINE = "preprocessing_pipeline"
    INFERENCE_SCRIPT = "inference_script"
    EVALUATION_REPORT = "evaluation_report"
    MODEL_CARD = "model_card"


@dataclass
class ModelArtifact:
    """
    Model artifact with metadata and provenance.

    Represents a single artifact associated with a model version.
    """
    artifact_id: str
    artifact_type: ArtifactType
    file_path: str
    file_size_bytes: int
    content_hash: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ModelVersion:
    """
    Model version with artifacts and deployment history.

    Represents a specific version of a registered model.
    """
    version_id: str
    model_id: str
    version: str
    stage: ModelStage
    created_at: datetime
    created_by: str

    # Artifacts
    artifacts: List[ModelArtifact] = field(default_factory=list)

    # Training information
    training_run_id: Optional[str] = None
    training_dataset_version: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Deployment
    deployment_status: DeploymentStatus = DeploymentStatus.NOT_DEPLOYED
    deployed_at: Optional[datetime] = None
    deployed_by: Optional[str] = None
    deployment_environment: Optional[str] = None

    # Model card
    model_card: Optional[ModelCard] = None

    # Tags and notes
    tags: Set[str] = field(default_factory=set)
    description: str = ""
    release_notes: str = ""

    # Lineage
    parent_version_id: Optional[str] = None
    derived_from_experiment: Optional[str] = None

    def get_artifact(self, artifact_type: ArtifactType) -> Optional[ModelArtifact]:
        """Get artifact by type."""
        for artifact in self.artifacts:
            if artifact.artifact_type == artifact_type:
                return artifact
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "version": self.version,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "training": {
                "run_id": self.training_run_id,
                "dataset_version": self.training_dataset_version,
                "hyperparameters": self.hyperparameters,
                "metrics": self.metrics,
            },
            "deployment": {
                "status": self.deployment_status.value,
                "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
                "deployed_by": self.deployed_by,
                "environment": self.deployment_environment,
            },
            "tags": list(self.tags),
            "description": self.description,
            "release_notes": self.release_notes,
            "lineage": {
                "parent_version_id": self.parent_version_id,
                "derived_from_experiment": self.derived_from_experiment,
            },
        }


@dataclass
class RegisteredModel:
    """
    Registered model with version history.

    Top-level model registration with all versions.
    """
    model_id: str
    name: str
    model_type: ModelType
    description: str
    created_at: datetime
    created_by: str
    last_updated: datetime

    # Ownership
    owner: str
    team: str

    # Versions
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    latest_version: Optional[str] = None
    production_version: Optional[str] = None
    staging_version: Optional[str] = None

    # Tags
    tags: Set[str] = field(default_factory=set)

    def get_version(self, version: str) -> Optional[ModelVersion]:
        """Get specific version."""
        return self.versions.get(version)

    def get_latest(self) -> Optional[ModelVersion]:
        """Get latest version."""
        if self.latest_version:
            return self.versions.get(self.latest_version)
        return None

    def get_production(self) -> Optional[ModelVersion]:
        """Get production version."""
        if self.production_version:
            return self.versions.get(self.production_version)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "model_type": self.model_type.value,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "last_updated": self.last_updated.isoformat(),
            "owner": self.owner,
            "team": self.team,
            "versions": {v: mv.to_dict() for v, mv in self.versions.items()},
            "latest_version": self.latest_version,
            "production_version": self.production_version,
            "staging_version": self.staging_version,
            "tags": list(self.tags),
        }


class ModelRegistry:
    """
    Centralized model registry for GL-003 UNIFIEDSTEAM.

    Manages model registration, versioning, artifact storage,
    and deployment tracking.

    Example:
        >>> registry = ModelRegistry()
        >>> model = registry.register_model(
        ...     model_id="trap-classifier-001",
        ...     name="Steam Trap Failure Classifier",
        ...     model_type=ModelType.TRAP_FAILURE_CLASSIFIER,
        ...     owner="ML Team",
        ... )
        >>> version = registry.create_version(
        ...     model_id="trap-classifier-001",
        ...     version="1.0.0",
        ...     artifacts=[...],
        ... )
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize model registry.

        Args:
            storage_path: Path for artifact storage
        """
        self.storage_path = storage_path or "/models"
        self._models: Dict[str, RegisteredModel] = {}
        self._audit_log: List[Dict[str, Any]] = []

    def register_model(
        self,
        model_id: str,
        name: str,
        model_type: ModelType,
        description: str = "",
        owner: str = "",
        team: str = "",
        created_by: str = "system",
        tags: Optional[Set[str]] = None,
    ) -> RegisteredModel:
        """
        Register a new model.

        Args:
            model_id: Unique model identifier
            name: Human-readable model name
            model_type: Type of model
            description: Model description
            owner: Model owner
            team: Owning team
            created_by: User creating the registration
            tags: Optional tags

        Returns:
            RegisteredModel

        Raises:
            ValueError: If model_id already exists
        """
        if model_id in self._models:
            raise ValueError(f"Model already registered: {model_id}")

        now = datetime.now(timezone.utc)

        model = RegisteredModel(
            model_id=model_id,
            name=name,
            model_type=model_type,
            description=description,
            created_at=now,
            created_by=created_by,
            last_updated=now,
            owner=owner,
            team=team,
            tags=tags or set(),
        )

        self._models[model_id] = model

        self._log_action("register_model", {
            "model_id": model_id,
            "name": name,
            "model_type": model_type.value,
        })

        logger.info(f"Registered model: {model_id} ({name})")
        return model

    def create_version(
        self,
        model_id: str,
        version: str,
        created_by: str = "system",
        artifacts: Optional[List[ModelArtifact]] = None,
        training_run_id: Optional[str] = None,
        training_dataset_version: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        model_card: Optional[ModelCard] = None,
        description: str = "",
        release_notes: str = "",
        tags: Optional[Set[str]] = None,
        parent_version_id: Optional[str] = None,
    ) -> ModelVersion:
        """
        Create a new model version.

        Args:
            model_id: ID of registered model
            version: Version string (e.g., "1.0.0")
            created_by: User creating the version
            artifacts: List of model artifacts
            training_run_id: ID of training run
            training_dataset_version: Version of training dataset
            hyperparameters: Model hyperparameters
            metrics: Evaluation metrics
            model_card: Model card documentation
            description: Version description
            release_notes: Release notes
            tags: Version tags
            parent_version_id: ID of parent version

        Returns:
            ModelVersion

        Raises:
            KeyError: If model_id not found
            ValueError: If version already exists
        """
        if model_id not in self._models:
            raise KeyError(f"Model not found: {model_id}")

        model = self._models[model_id]

        if version in model.versions:
            raise ValueError(f"Version already exists: {version}")

        now = datetime.now(timezone.utc)
        version_id = f"{model_id}-v{version.replace('.', '-')}"

        model_version = ModelVersion(
            version_id=version_id,
            model_id=model_id,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            created_at=now,
            created_by=created_by,
            artifacts=artifacts or [],
            training_run_id=training_run_id,
            training_dataset_version=training_dataset_version,
            hyperparameters=hyperparameters or {},
            metrics=metrics or {},
            model_card=model_card,
            description=description,
            release_notes=release_notes,
            tags=tags or set(),
            parent_version_id=parent_version_id,
        )

        model.versions[version] = model_version
        model.latest_version = version
        model.last_updated = now

        self._log_action("create_version", {
            "model_id": model_id,
            "version": version,
            "version_id": version_id,
        })

        logger.info(f"Created version {version} for model {model_id}")
        return model_version

    def add_artifact(
        self,
        model_id: str,
        version: str,
        artifact_type: ArtifactType,
        file_path: str,
        file_size_bytes: int,
        content_hash: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelArtifact:
        """
        Add artifact to a model version.

        Args:
            model_id: Model ID
            version: Version string
            artifact_type: Type of artifact
            file_path: Path to artifact file
            file_size_bytes: Size of artifact
            content_hash: SHA256 hash of content
            metadata: Additional metadata

        Returns:
            ModelArtifact
        """
        model_version = self.get_version(model_id, version)
        if not model_version:
            raise KeyError(f"Version not found: {model_id}:{version}")

        artifact_id = f"{model_version.version_id}-{artifact_type.value}"

        artifact = ModelArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            content_hash=content_hash,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        model_version.artifacts.append(artifact)

        self._log_action("add_artifact", {
            "model_id": model_id,
            "version": version,
            "artifact_type": artifact_type.value,
        })

        return artifact

    def transition_stage(
        self,
        model_id: str,
        version: str,
        target_stage: ModelStage,
        transitioned_by: str = "system",
    ) -> ModelVersion:
        """
        Transition a model version to a new stage.

        Args:
            model_id: Model ID
            version: Version string
            target_stage: Target stage
            transitioned_by: User making transition

        Returns:
            Updated ModelVersion
        """
        model_version = self.get_version(model_id, version)
        if not model_version:
            raise KeyError(f"Version not found: {model_id}:{version}")

        model = self._models[model_id]
        old_stage = model_version.stage
        model_version.stage = target_stage

        # Update model's stage pointers
        if target_stage == ModelStage.PRODUCTION:
            model.production_version = version
        elif target_stage == ModelStage.STAGING:
            model.staging_version = version

        model.last_updated = datetime.now(timezone.utc)

        self._log_action("transition_stage", {
            "model_id": model_id,
            "version": version,
            "from_stage": old_stage.value,
            "to_stage": target_stage.value,
            "transitioned_by": transitioned_by,
        })

        logger.info(
            f"Transitioned {model_id}:{version} from {old_stage.value} "
            f"to {target_stage.value}"
        )
        return model_version

    def update_deployment_status(
        self,
        model_id: str,
        version: str,
        status: DeploymentStatus,
        environment: str = "",
        deployed_by: str = "system",
    ) -> ModelVersion:
        """
        Update deployment status of a model version.

        Args:
            model_id: Model ID
            version: Version string
            status: New deployment status
            environment: Deployment environment
            deployed_by: User deploying

        Returns:
            Updated ModelVersion
        """
        model_version = self.get_version(model_id, version)
        if not model_version:
            raise KeyError(f"Version not found: {model_id}:{version}")

        model_version.deployment_status = status
        model_version.deployment_environment = environment

        if status == DeploymentStatus.DEPLOYED:
            model_version.deployed_at = datetime.now(timezone.utc)
            model_version.deployed_by = deployed_by

        self._log_action("update_deployment", {
            "model_id": model_id,
            "version": version,
            "status": status.value,
            "environment": environment,
        })

        return model_version

    def get_model(self, model_id: str) -> Optional[RegisteredModel]:
        """Get registered model by ID."""
        return self._models.get(model_id)

    def get_version(
        self,
        model_id: str,
        version: str,
    ) -> Optional[ModelVersion]:
        """Get specific model version."""
        model = self.get_model(model_id)
        if model:
            return model.get_version(version)
        return None

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[RegisteredModel]:
        """
        List registered models with optional filtering.

        Args:
            model_type: Filter by model type
            tags: Filter by tags (all must match)

        Returns:
            List of matching RegisteredModels
        """
        models = list(self._models.values())

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if tags:
            models = [m for m in models if tags.issubset(m.tags)]

        return models

    def list_versions(
        self,
        model_id: str,
        stage: Optional[ModelStage] = None,
    ) -> List[ModelVersion]:
        """
        List versions of a model.

        Args:
            model_id: Model ID
            stage: Filter by stage

        Returns:
            List of ModelVersions
        """
        model = self.get_model(model_id)
        if not model:
            return []

        versions = list(model.versions.values())

        if stage:
            versions = [v for v in versions if v.stage == stage]

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def search_by_metrics(
        self,
        model_id: str,
        metric_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> List[ModelVersion]:
        """
        Search versions by metric values.

        Args:
            model_id: Model ID
            metric_name: Name of metric to search
            min_value: Minimum metric value
            max_value: Maximum metric value

        Returns:
            List of matching ModelVersions
        """
        versions = self.list_versions(model_id)
        results = []

        for version in versions:
            if metric_name in version.metrics:
                value = version.metrics[metric_name]
                if min_value is not None and value < min_value:
                    continue
                if max_value is not None and value > max_value:
                    continue
                results.append(version)

        return results

    def export_registry(self) -> Dict[str, Any]:
        """Export complete registry state."""
        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "models": {
                model_id: model.to_dict()
                for model_id, model in self._models.items()
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
