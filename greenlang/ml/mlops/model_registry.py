# -*- coding: utf-8 -*-
"""
Model Registry Module

This module provides MLflow-based model registry capabilities for GreenLang,
enabling model versioning, deployment staging, and lifecycle management
with complete provenance tracking.

The model registry is critical for maintaining reproducibility and
auditability of ML models used in regulatory compliance.

Example:
    >>> from greenlang.ml.mlops import ModelRegistry
    >>> registry = ModelRegistry()
    >>> registry.register_model(model, "emission_predictor")
    >>> production_model = registry.get_model("emission_predictor", stage="Production")
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import hashlib
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model deployment stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelFramework(str, Enum):
    """Supported ML frameworks."""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ONNX = "onnx"
    CUSTOM = "custom"


class ModelRegistryConfig(BaseModel):
    """Configuration for model registry."""

    tracking_uri: str = Field(
        default="./mlruns",
        description="MLflow tracking URI"
    )
    artifact_location: str = Field(
        default="./mlartifacts",
        description="Artifact storage location"
    )
    default_experiment: str = Field(
        default="greenlang_models",
        description="Default experiment name"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    enable_signature_validation: bool = Field(
        default=True,
        description="Validate model signatures"
    )
    auto_log_metrics: bool = Field(
        default=True,
        description="Auto-log training metrics"
    )


class ModelMetadata(BaseModel):
    """Metadata for a registered model."""

    model_name: str = Field(
        ...,
        description="Model name"
    )
    version: str = Field(
        ...,
        description="Model version"
    )
    framework: ModelFramework = Field(
        ...,
        description="ML framework"
    )
    stage: ModelStage = Field(
        default=ModelStage.NONE,
        description="Deployment stage"
    )
    description: Optional[str] = Field(
        default=None,
        description="Model description"
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Model tags"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Model performance metrics"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model parameters"
    )
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Input data schema"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Output data schema"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 provenance hash"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    run_id: Optional[str] = Field(
        default=None,
        description="MLflow run ID"
    )
    artifact_path: Optional[str] = Field(
        default=None,
        description="Path to model artifacts"
    )


class ModelVersion(BaseModel):
    """Model version information."""

    version: str = Field(
        ...,
        description="Version string"
    )
    stage: ModelStage = Field(
        ...,
        description="Deployment stage"
    )
    created_at: datetime = Field(
        ...,
        description="Creation timestamp"
    )
    description: Optional[str] = Field(
        default=None,
        description="Version description"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Version metrics"
    )


class ModelRegistry:
    """
    Model Registry for GreenLang.

    This class provides MLflow-based model registry capabilities,
    enabling model versioning, deployment staging, and lifecycle
    management with complete provenance tracking.

    Key capabilities:
    - Model registration and versioning
    - Stage transitions (Staging -> Production)
    - Model retrieval by version or stage
    - Provenance tracking for audit trails
    - Signature validation
    - Metrics logging

    Attributes:
        config: Registry configuration
        _models: In-memory model cache
        _metadata: Model metadata store
        _mlflow_client: MLflow client (when available)

    Example:
        >>> registry = ModelRegistry()
        >>> # Register a new model
        >>> metadata = registry.register_model(
        ...     model,
        ...     name="emission_predictor",
        ...     framework=ModelFramework.SKLEARN,
        ...     metrics={"rmse": 0.05, "r2": 0.95}
        ... )
        >>> # Promote to production
        >>> registry.transition_stage(
        ...     "emission_predictor",
        ...     metadata.version,
        ...     ModelStage.PRODUCTION
        ... )
        >>> # Load production model
        >>> model = registry.get_model("emission_predictor", stage=ModelStage.PRODUCTION)
    """

    def __init__(self, config: Optional[ModelRegistryConfig] = None):
        """
        Initialize model registry.

        Args:
            config: Registry configuration
        """
        self.config = config or ModelRegistryConfig()
        self._models: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, List[ModelMetadata]] = {}
        self._mlflow_client = None
        self._initialized = False

        # Create artifact directory
        Path(self.config.artifact_location).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ModelRegistry initialized: tracking_uri={self.config.tracking_uri}"
        )

    def _initialize_mlflow(self) -> bool:
        """Initialize MLflow if available."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.default_experiment)

            self._mlflow_client = MlflowClient()
            self._initialized = True

            logger.info("MLflow initialized successfully")
            return True

        except ImportError:
            logger.warning(
                "MLflow not installed. Using fallback registry. "
                "Install with: pip install mlflow"
            )
            self._initialized = True
            return False

    def _calculate_provenance(
        self,
        model: Any,
        name: str,
        metrics: Dict[str, float]
    ) -> str:
        """Calculate SHA-256 provenance hash for model."""
        # Create hash from model characteristics
        model_str = str(type(model).__name__)

        if hasattr(model, "get_params"):
            model_str += str(model.get_params())
        elif hasattr(model, "state_dict"):
            # PyTorch
            import torch
            state = model.state_dict()
            model_str += str(sum(v.sum().item() for v in state.values()))

        metrics_str = str(sorted(metrics.items()))
        combined = f"{name}|{model_str}|{metrics_str}"

        return hashlib.sha256(combined.encode()).hexdigest()

    def _generate_version(self, name: str) -> str:
        """Generate next version number for model."""
        if name not in self._metadata:
            return "1"

        versions = [m.version for m in self._metadata[name]]
        try:
            max_version = max(int(v) for v in versions if v.isdigit())
            return str(max_version + 1)
        except ValueError:
            return str(len(versions) + 1)

    def _save_model_artifact(
        self,
        model: Any,
        name: str,
        version: str,
        framework: ModelFramework
    ) -> str:
        """Save model to artifact location."""
        import pickle

        artifact_dir = Path(self.config.artifact_location) / name / version
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifact_dir / "model.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def _load_model_artifact(self, artifact_path: str) -> Any:
        """Load model from artifact location."""
        import pickle

        with open(artifact_path, "rb") as f:
            return pickle.load(f)

    def register_model(
        self,
        model: Any,
        name: str,
        framework: ModelFramework = ModelFramework.SKLEARN,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> ModelMetadata:
        """
        Register a model in the registry.

        Args:
            model: ML model to register
            name: Model name
            framework: ML framework
            description: Model description
            tags: Model tags
            metrics: Performance metrics
            parameters: Model parameters
            input_schema: Input data schema
            output_schema: Output data schema

        Returns:
            ModelMetadata for registered model

        Example:
            >>> metadata = registry.register_model(
            ...     model,
            ...     name="carbon_predictor",
            ...     framework=ModelFramework.SKLEARN,
            ...     metrics={"mae": 0.05}
            ... )
        """
        if not self._initialized:
            self._initialize_mlflow()

        tags = tags or {}
        metrics = metrics or {}
        parameters = parameters or {}

        # Generate version
        version = self._generate_version(name)

        # Calculate provenance
        provenance_hash = self._calculate_provenance(model, name, metrics)

        # Save model artifact
        artifact_path = self._save_model_artifact(model, name, version, framework)

        # Create metadata
        metadata = ModelMetadata(
            model_name=name,
            version=version,
            framework=framework,
            stage=ModelStage.NONE,
            description=description,
            tags=tags,
            metrics=metrics,
            parameters=parameters,
            input_schema=input_schema,
            output_schema=output_schema,
            provenance_hash=provenance_hash,
            artifact_path=artifact_path
        )

        # Store in registry
        if name not in self._metadata:
            self._metadata[name] = []
        self._metadata[name].append(metadata)

        if name not in self._models:
            self._models[name] = {}
        self._models[name][version] = model

        # Log to MLflow if available
        if self._mlflow_client is not None:
            self._log_to_mlflow(model, metadata)

        logger.info(
            f"Model registered: {name} v{version}, "
            f"provenance: {provenance_hash[:16]}..."
        )

        return metadata

    def _log_to_mlflow(
        self,
        model: Any,
        metadata: ModelMetadata
    ) -> None:
        """Log model to MLflow."""
        try:
            import mlflow

            with mlflow.start_run(run_name=f"{metadata.model_name}_v{metadata.version}"):
                # Log parameters
                for key, value in metadata.parameters.items():
                    mlflow.log_param(key, value)

                # Log metrics
                for key, value in metadata.metrics.items():
                    mlflow.log_metric(key, value)

                # Log tags
                for key, value in metadata.tags.items():
                    mlflow.set_tag(key, value)

                mlflow.set_tag("provenance_hash", metadata.provenance_hash)
                mlflow.set_tag("framework", metadata.framework.value)

                # Log model
                if metadata.framework == ModelFramework.SKLEARN:
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                        registered_model_name=metadata.model_name
                    )
                elif metadata.framework == ModelFramework.PYTORCH:
                    mlflow.pytorch.log_model(
                        model,
                        "model",
                        registered_model_name=metadata.model_name
                    )
                else:
                    # Generic logging
                    mlflow.pyfunc.log_model(
                        "model",
                        python_model=model,
                        registered_model_name=metadata.model_name
                    )

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Optional[Any]:
        """
        Retrieve a model from the registry.

        Args:
            name: Model name
            version: Specific version (optional)
            stage: Deployment stage (optional)

        Returns:
            Model or None if not found

        Example:
            >>> model = registry.get_model("emission_predictor", stage=ModelStage.PRODUCTION)
        """
        if name not in self._metadata:
            logger.warning(f"Model not found: {name}")
            return None

        # Find matching version
        if version is not None:
            for meta in self._metadata[name]:
                if meta.version == version:
                    return self._models.get(name, {}).get(version)

        if stage is not None:
            for meta in reversed(self._metadata[name]):  # Latest first
                if meta.stage == stage:
                    return self._models.get(name, {}).get(meta.version)

        # Return latest version
        if self._metadata[name]:
            latest = self._metadata[name][-1]
            return self._models.get(name, {}).get(latest.version)

        return None

    def get_metadata(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """Get metadata for a model."""
        if name not in self._metadata:
            return None

        if version is not None:
            for meta in self._metadata[name]:
                if meta.version == version:
                    return meta

        # Return latest
        if self._metadata[name]:
            return self._metadata[name][-1]

        return None

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._metadata.keys())

    def list_versions(
        self,
        name: str
    ) -> List[ModelVersion]:
        """List all versions of a model."""
        if name not in self._metadata:
            return []

        return [
            ModelVersion(
                version=m.version,
                stage=m.stage,
                created_at=m.created_at,
                description=m.description,
                metrics=m.metrics
            )
            for m in self._metadata[name]
        ]

    def transition_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage,
        archive_existing: bool = True
    ) -> bool:
        """
        Transition model to a new stage.

        Args:
            name: Model name
            version: Model version
            stage: Target stage
            archive_existing: Archive current production model

        Returns:
            Success status

        Example:
            >>> registry.transition_stage(
            ...     "emission_predictor",
            ...     "5",
            ...     ModelStage.PRODUCTION
            ... )
        """
        if name not in self._metadata:
            logger.error(f"Model not found: {name}")
            return False

        # Find version
        target_meta = None
        for meta in self._metadata[name]:
            if meta.version == version:
                target_meta = meta
                break

        if target_meta is None:
            logger.error(f"Version not found: {name} v{version}")
            return False

        # Archive existing production model
        if archive_existing and stage == ModelStage.PRODUCTION:
            for meta in self._metadata[name]:
                if meta.stage == ModelStage.PRODUCTION:
                    meta.stage = ModelStage.ARCHIVED
                    meta.updated_at = datetime.utcnow()
                    logger.info(f"Archived {name} v{meta.version}")

        # Update stage
        target_meta.stage = stage
        target_meta.updated_at = datetime.utcnow()

        logger.info(f"Transitioned {name} v{version} to {stage.value}")

        return True

    def delete_model(
        self,
        name: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Delete a model or specific version.

        Args:
            name: Model name
            version: Specific version (None = all versions)

        Returns:
            Success status
        """
        if name not in self._metadata:
            return False

        if version is not None:
            # Delete specific version
            self._metadata[name] = [
                m for m in self._metadata[name] if m.version != version
            ]
            if name in self._models and version in self._models[name]:
                del self._models[name][version]
            logger.info(f"Deleted {name} v{version}")
        else:
            # Delete all versions
            del self._metadata[name]
            if name in self._models:
                del self._models[name]
            logger.info(f"Deleted all versions of {name}")

        return True

    def compare_models(
        self,
        name: str,
        versions: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple model versions.

        Args:
            name: Model name
            versions: List of versions to compare

        Returns:
            Comparison dictionary
        """
        comparison = {}

        for version in versions:
            meta = self.get_metadata(name, version)
            if meta:
                comparison[version] = {
                    "metrics": meta.metrics,
                    "parameters": meta.parameters,
                    "stage": meta.stage.value,
                    "created_at": meta.created_at.isoformat()
                }

        return comparison


# Unit test stubs
class TestModelRegistry:
    """Unit tests for ModelRegistry."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        registry = ModelRegistry()
        assert registry.config.tracking_uri == "./mlruns"

    def test_register_model(self):
        """Test model registration."""
        registry = ModelRegistry()

        class MockModel:
            def get_params(self):
                return {"n_estimators": 100}

        metadata = registry.register_model(
            MockModel(),
            name="test_model",
            metrics={"accuracy": 0.95}
        )

        assert metadata.model_name == "test_model"
        assert metadata.version == "1"
        assert metadata.metrics["accuracy"] == 0.95
        assert len(metadata.provenance_hash) == 64

    def test_version_increment(self):
        """Test version auto-increment."""
        registry = ModelRegistry()

        class MockModel:
            pass

        v1 = registry.register_model(MockModel(), "test")
        v2 = registry.register_model(MockModel(), "test")
        v3 = registry.register_model(MockModel(), "test")

        assert v1.version == "1"
        assert v2.version == "2"
        assert v3.version == "3"

    def test_stage_transition(self):
        """Test stage transition."""
        registry = ModelRegistry()

        class MockModel:
            pass

        meta = registry.register_model(MockModel(), "test")
        assert meta.stage == ModelStage.NONE

        registry.transition_stage("test", "1", ModelStage.PRODUCTION)
        updated = registry.get_metadata("test", "1")
        assert updated.stage == ModelStage.PRODUCTION

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        registry = ModelRegistry()

        class MockModel:
            def get_params(self):
                return {"a": 1}

        model = MockModel()
        hash1 = registry._calculate_provenance(model, "test", {"m": 0.5})
        hash2 = registry._calculate_provenance(model, "test", {"m": 0.5})

        assert hash1 == hash2
