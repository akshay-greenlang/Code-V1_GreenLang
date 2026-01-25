# -*- coding: utf-8 -*-
"""
ModelRegistry - MLOps Model Lifecycle Management

This module implements a model registry for managing ML model versions,
tracking performance, and handling model deployment lifecycle.

Key Features:
    - Model registration with metadata
    - Model loading and versioning
    - Performance tracking over time
    - Model archival and rollback

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    REGISTERED = "registered"
    VALIDATING = "validating"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class ModelType(str, Enum):
    """Types of models in the registry."""
    STABILITY_PREDICTOR = "stability_predictor"
    EMISSIONS_PREDICTOR = "emissions_predictor"
    SOFT_SENSOR = "soft_sensor"
    PERFORMANCE_SURROGATE = "performance_surrogate"
    DRIFT_DETECTOR = "drift_detector"
    CUSTOM = "custom"


class BaseModel(Protocol):
    """Protocol for models that can be registered."""
    @property
    def model_id(self) -> str: ...
    @property
    def is_fitted(self) -> bool: ...
    def save_model(self, path: Path) -> None: ...


class ModelMetadata(BaseModel):
    """Metadata for a registered model."""
    model_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(..., description="Human-readable model name")
    model_type: ModelType = Field(..., description="Type of model")
    version: str = Field(default="1.0.0", description="Semantic version")
    status: ModelStatus = Field(default=ModelStatus.REGISTERED)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = Field(default="system", description="Creator identifier")
    description: str = Field(default="", description="Model description")
    framework: str = Field(default="sklearn", description="ML framework used")
    algorithm: str = Field(default="", description="Algorithm name")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_data_hash: str = Field(default="", description="Hash of training data")
    n_training_samples: int = Field(default=0, ge=0)
    feature_names: List[str] = Field(default_factory=list)
    target_names: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    file_path: str = Field(default="", description="Path to model file")
    file_hash: str = Field(default="", description="SHA-256 hash of model file")


class PerformanceMetrics(BaseModel):
    """Performance metrics for a model."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str = Field(...)
    n_predictions: int = Field(default=0, ge=0)
    mean_latency_ms: float = Field(default=0.0, ge=0.0)
    p99_latency_ms: float = Field(default=0.0, ge=0.0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy: Optional[float] = Field(default=None)
    mae: Optional[float] = Field(default=None)
    rmse: Optional[float] = Field(default=None)
    r2: Optional[float] = Field(default=None)
    custom_metrics: Dict[str, float] = Field(default_factory=dict)


class ModelRegistry:
    """
    Registry for managing ML model lifecycle.

    The ModelRegistry provides:
    1. Model registration with metadata tracking
    2. Model versioning and loading
    3. Active model management by type
    4. Performance tracking over time
    5. Model archival and cleanup

    Example:
        >>> registry = ModelRegistry(storage_path=Path("./models"))
        >>> metadata = ModelMetadata(name="stability_v1", model_type=ModelType.STABILITY_PREDICTOR)
        >>> model_id = registry.register_model(my_model, metadata)
        >>> loaded_model = registry.load_model(model_id)
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_versions_per_type: int = 5
    ):
        """
        Initialize ModelRegistry.

        Args:
            storage_path: Directory for model storage
            max_versions_per_type: Maximum versions to keep per model type
        """
        self.storage_path = storage_path or Path("./model_registry")
        self.max_versions_per_type = max_versions_per_type

        # Create storage directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "models").mkdir(exist_ok=True)
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        (self.storage_path / "archive").mkdir(exist_ok=True)

        # In-memory registry
        self._registry: Dict[str, ModelMetadata] = {}
        self._active_models: Dict[ModelType, str] = {}
        self._performance_history: Dict[str, List[PerformanceMetrics]] = {}

        # Load existing registry
        self._load_registry()

        logger.info(f"ModelRegistry initialized at {self.storage_path}")

    def register_model(
        self,
        model: Any,
        metadata: ModelMetadata
    ) -> str:
        """
        Register a new model in the registry.

        Args:
            model: Model object to register (must implement save_model)
            metadata: Model metadata

        Returns:
            Unique model ID
        """
        start_time = time.time()

        # Generate model ID if not provided
        if not metadata.model_id:
            metadata.model_id = str(uuid4())

        model_id = metadata.model_id

        # Save model file
        model_path = self.storage_path / "models" / f"{model_id}.pkl"

        if hasattr(model, 'save_model'):
            model.save_model(model_path)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Compute file hash
        with open(model_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        metadata.file_path = str(model_path)
        metadata.file_hash = file_hash
        metadata.updated_at = datetime.now(timezone.utc)

        # Save metadata
        metadata_path = self.storage_path / "metadata" / f"{model_id}.json"
        with open(metadata_path, 'w') as f:
            f.write(metadata.json(indent=2))

        # Update registry
        self._registry[model_id] = metadata
        self._performance_history[model_id] = []

        # Auto-activate if first of type
        if metadata.model_type not in self._active_models:
            self._active_models[metadata.model_type] = model_id
            metadata.status = ModelStatus.ACTIVE
            self._save_metadata(metadata)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Model registered: id={model_id}, type={metadata.model_type.value}, "
            f"version={metadata.version}, time={elapsed_ms:.1f}ms"
        )

        return model_id

    def load_model(self, model_id: str) -> Any:
        """
        Load a model from the registry.

        Args:
            model_id: Model identifier

        Returns:
            Loaded model object

        Raises:
            ValueError: If model not found
        """
        if model_id not in self._registry:
            raise ValueError(f"Model {model_id} not found in registry")

        metadata = self._registry[model_id]
        model_path = Path(metadata.file_path)

        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")

        # Verify file hash
        with open(model_path, 'rb') as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()

        if current_hash != metadata.file_hash:
            logger.warning(f"Model file hash mismatch for {model_id}")

        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Model loaded: id={model_id}, type={metadata.model_type.value}")
        return model

    def get_active_model(self, model_type: Union[str, ModelType]) -> Any:
        """
        Get the currently active model for a type.

        Args:
            model_type: Type of model to get

        Returns:
            Active model object

        Raises:
            ValueError: If no active model for type
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        if model_type not in self._active_models:
            raise ValueError(f"No active model for type {model_type.value}")

        model_id = self._active_models[model_type]
        return self.load_model(model_id)

    def set_active_model(self, model_id: str) -> None:
        """
        Set a model as the active model for its type.

        Args:
            model_id: Model identifier to activate
        """
        if model_id not in self._registry:
            raise ValueError(f"Model {model_id} not found in registry")

        metadata = self._registry[model_id]

        # Deactivate current active model
        if metadata.model_type in self._active_models:
            old_id = self._active_models[metadata.model_type]
            if old_id in self._registry:
                old_metadata = self._registry[old_id]
                old_metadata.status = ModelStatus.DEPRECATED
                self._save_metadata(old_metadata)

        # Activate new model
        self._active_models[metadata.model_type] = model_id
        metadata.status = ModelStatus.ACTIVE
        metadata.updated_at = datetime.now(timezone.utc)
        self._save_metadata(metadata)

        logger.info(f"Model {model_id} set as active for {metadata.model_type.value}")

    def track_model_performance(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Track performance metrics for a model.

        Args:
            model_id: Model identifier
            metrics: Dictionary of metric name to value
        """
        if model_id not in self._registry:
            logger.warning(f"Model {model_id} not found, skipping metrics")
            return

        perf_metrics = PerformanceMetrics(
            model_id=model_id,
            n_predictions=metrics.get('n_predictions', 0),
            mean_latency_ms=metrics.get('mean_latency_ms', 0.0),
            p99_latency_ms=metrics.get('p99_latency_ms', 0.0),
            error_rate=metrics.get('error_rate', 0.0),
            accuracy=metrics.get('accuracy'),
            mae=metrics.get('mae'),
            rmse=metrics.get('rmse'),
            r2=metrics.get('r2'),
            custom_metrics={k: v for k, v in metrics.items()
                          if k not in ['n_predictions', 'mean_latency_ms', 'p99_latency_ms',
                                      'error_rate', 'accuracy', 'mae', 'rmse', 'r2']}
        )

        if model_id not in self._performance_history:
            self._performance_history[model_id] = []

        self._performance_history[model_id].append(perf_metrics)

        # Keep only last 1000 metrics
        if len(self._performance_history[model_id]) > 1000:
            self._performance_history[model_id] = self._performance_history[model_id][-1000:]

        logger.debug(f"Performance metrics tracked for model {model_id}")

    def get_model_performance(
        self,
        model_id: str,
        limit: int = 100
    ) -> List[PerformanceMetrics]:
        """
        Get performance history for a model.

        Args:
            model_id: Model identifier
            limit: Maximum number of records to return

        Returns:
            List of PerformanceMetrics
        """
        if model_id not in self._performance_history:
            return []

        return self._performance_history[model_id][-limit:]

    def archive_model(self, model_id: str, reason: str = "") -> None:
        """
        Archive a model, removing it from active use.

        Args:
            model_id: Model identifier
            reason: Reason for archival
        """
        if model_id not in self._registry:
            raise ValueError(f"Model {model_id} not found in registry")

        metadata = self._registry[model_id]

        # Cannot archive active models
        if metadata.status == ModelStatus.ACTIVE:
            raise ValueError(f"Cannot archive active model {model_id}. Set another model as active first.")

        # Move model file to archive
        model_path = Path(metadata.file_path)
        if model_path.exists():
            archive_path = self.storage_path / "archive" / f"{model_id}.pkl"
            shutil.move(str(model_path), str(archive_path))
            metadata.file_path = str(archive_path)

        metadata.status = ModelStatus.ARCHIVED
        metadata.updated_at = datetime.now(timezone.utc)
        metadata.tags.append(f"archived_reason:{reason}")
        self._save_metadata(metadata)

        logger.info(f"Model {model_id} archived: {reason}")

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None
    ) -> List[ModelMetadata]:
        """
        List models in the registry.

        Args:
            model_type: Filter by model type
            status: Filter by status

        Returns:
            List of ModelMetadata
        """
        models = list(self._registry.values())

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if status:
            models = [m for m in models if m.status == status]

        # Sort by creation date (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)

        return models

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model."""
        return self._registry.get(model_id)

    def delete_model(self, model_id: str, force: bool = False) -> None:
        """
        Delete a model from the registry.

        Args:
            model_id: Model identifier
            force: Force deletion even if active
        """
        if model_id not in self._registry:
            raise ValueError(f"Model {model_id} not found")

        metadata = self._registry[model_id]

        if metadata.status == ModelStatus.ACTIVE and not force:
            raise ValueError(f"Cannot delete active model {model_id}. Use force=True or archive first.")

        # Delete files
        model_path = Path(metadata.file_path)
        if model_path.exists():
            model_path.unlink()

        metadata_path = self.storage_path / "metadata" / f"{model_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()

        # Remove from registry
        del self._registry[model_id]
        if model_id in self._performance_history:
            del self._performance_history[model_id]

        # Remove from active if applicable
        for mtype, mid in list(self._active_models.items()):
            if mid == model_id:
                del self._active_models[mtype]

        logger.info(f"Model {model_id} deleted")

    def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Save metadata to disk."""
        metadata_path = self.storage_path / "metadata" / f"{metadata.model_id}.json"
        with open(metadata_path, 'w') as f:
            f.write(metadata.json(indent=2))

    def _load_registry(self) -> None:
        """Load registry from disk."""
        metadata_dir = self.storage_path / "metadata"
        if not metadata_dir.exists():
            return

        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                metadata = ModelMetadata(**data)
                self._registry[metadata.model_id] = metadata

                if metadata.status == ModelStatus.ACTIVE:
                    self._active_models[metadata.model_type] = metadata.model_id

                self._performance_history[metadata.model_id] = []

            except Exception as e:
                logger.error(f"Failed to load metadata from {metadata_file}: {e}")

        logger.info(f"Loaded {len(self._registry)} models from registry")

    @property
    def registered_models(self) -> int:
        """Number of registered models."""
        return len(self._registry)

    @property
    def active_model_ids(self) -> Dict[str, str]:
        """Dictionary of model type to active model ID."""
        return {k.value: v for k, v in self._active_models.items()}
