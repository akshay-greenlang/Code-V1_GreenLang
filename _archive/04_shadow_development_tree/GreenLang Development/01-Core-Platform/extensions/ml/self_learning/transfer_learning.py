# -*- coding: utf-8 -*-
"""
Transfer Learning Pipeline Module

This module provides transfer learning capabilities for GreenLang Process Heat
agents, enabling pre-trained model adaptation to specific domains (boiler,
furnace, steam) with frozen layer management and domain-specific head replacement.

Transfer learning reduces training time and data requirements by leveraging
knowledge from pre-trained models, critical for Process Heat applications
where labeled data may be limited but domain knowledge is transferable.

Example:
    >>> from greenlang.ml.self_learning import TransferLearningPipeline
    >>> pipeline = TransferLearningPipeline(pretrained_model, domain="boiler")
    >>> pipeline.freeze_backbone()
    >>> pipeline.replace_head(num_classes=5)
    >>> result = pipeline.fine_tune(new_data, epochs=10)
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
from collections import OrderedDict
import copy

logger = logging.getLogger(__name__)


class ProcessHeatDomain(str, Enum):
    """Process Heat application domains."""
    BOILER = "boiler"
    FURNACE = "furnace"
    STEAM = "steam"
    CHP = "chp"  # Combined Heat and Power
    HEAT_EXCHANGER = "heat_exchanger"
    GENERIC = "generic"


class TransferStrategy(str, Enum):
    """Transfer learning strategies."""
    FREEZE_ALL = "freeze_all"  # Freeze entire backbone
    FREEZE_EARLY = "freeze_early"  # Freeze early layers only
    GRADUAL_UNFREEZE = "gradual_unfreeze"  # Progressively unfreeze
    DISCRIMINATIVE_LR = "discriminative_lr"  # Different LR per layer
    FEATURE_EXTRACTION = "feature_extraction"  # Use as fixed feature extractor


class TransferLearningConfig(BaseModel):
    """Configuration for transfer learning pipeline."""

    source_domain: ProcessHeatDomain = Field(
        default=ProcessHeatDomain.GENERIC,
        description="Source domain of pretrained model"
    )
    target_domain: ProcessHeatDomain = Field(
        default=ProcessHeatDomain.BOILER,
        description="Target domain for fine-tuning"
    )
    strategy: TransferStrategy = Field(
        default=TransferStrategy.FREEZE_EARLY,
        description="Transfer learning strategy"
    )
    freeze_layers: int = Field(
        default=3,
        ge=0,
        description="Number of layers to freeze from the start"
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0,
        description="Base learning rate for fine-tuning"
    )
    lr_decay_factor: float = Field(
        default=0.5,
        gt=0,
        le=1.0,
        description="LR decay factor for discriminative LR"
    )
    n_epochs: int = Field(
        default=50,
        ge=1,
        description="Number of fine-tuning epochs"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for fine-tuning"
    )
    unfreeze_schedule: List[int] = Field(
        default=[10, 20, 30],
        description="Epochs at which to unfreeze layers (gradual unfreeze)"
    )
    early_stopping_patience: int = Field(
        default=5,
        ge=1,
        description="Patience for early stopping"
    )
    validation_split: float = Field(
        default=0.2,
        ge=0,
        lt=1.0,
        description="Validation split ratio"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class PretrainedModelInfo(BaseModel):
    """Information about a pretrained model."""

    model_id: str = Field(
        ...,
        description="Unique model identifier"
    )
    domain: ProcessHeatDomain = Field(
        ...,
        description="Domain the model was trained on"
    )
    architecture: str = Field(
        ...,
        description="Model architecture name"
    )
    input_dim: int = Field(
        ...,
        description="Input dimension"
    )
    output_dim: int = Field(
        ...,
        description="Output dimension"
    )
    n_layers: int = Field(
        ...,
        description="Number of layers"
    )
    n_parameters: int = Field(
        ...,
        description="Total trainable parameters"
    )
    pretrain_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Metrics from pretraining"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash of model weights"
    )


class TransferMetrics(BaseModel):
    """Metrics tracked during transfer learning."""

    source_performance: float = Field(
        default=0.0,
        description="Performance on source domain before transfer"
    )
    target_performance_before: float = Field(
        default=0.0,
        description="Performance on target before fine-tuning"
    )
    target_performance_after: float = Field(
        default=0.0,
        description="Performance on target after fine-tuning"
    )
    transfer_gain: float = Field(
        default=0.0,
        description="Improvement from transfer learning"
    )
    negative_transfer: bool = Field(
        default=False,
        description="Whether negative transfer occurred"
    )
    convergence_epoch: int = Field(
        default=0,
        description="Epoch at which convergence occurred"
    )
    layers_frozen: int = Field(
        default=0,
        description="Number of frozen layers at end"
    )
    layers_unfrozen: int = Field(
        default=0,
        description="Number of trainable layers"
    )


class FineTuningResult(BaseModel):
    """Result from fine-tuning process."""

    success: bool = Field(
        ...,
        description="Whether fine-tuning succeeded"
    )
    metrics: TransferMetrics = Field(
        ...,
        description="Transfer learning metrics"
    )
    training_history: List[Dict[str, float]] = Field(
        ...,
        description="Training history per epoch"
    )
    best_epoch: int = Field(
        ...,
        description="Epoch with best validation performance"
    )
    best_val_loss: float = Field(
        ...,
        description="Best validation loss achieved"
    )
    final_lr: float = Field(
        ...,
        description="Final learning rate used"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time"
    )


class TransferLearningPipeline:
    """
    Transfer Learning Pipeline for GreenLang Process Heat agents.

    This class provides comprehensive transfer learning capabilities,
    enabling adaptation of pretrained models to specific Process Heat
    domains (boiler, furnace, steam) with sophisticated layer management.

    Key capabilities:
    - Pre-trained model loading for Process Heat domains
    - Multiple transfer strategies (freeze, gradual unfreeze, discriminative LR)
    - Feature extraction and frozen layer management
    - Domain-specific head replacement
    - Transfer metrics tracking
    - Negative transfer detection
    - Provenance tracking

    Attributes:
        model: Neural network model
        config: Transfer learning configuration
        pretrained_info: Information about pretrained model
        _frozen_layers: Set of frozen layer names
        _layer_lrs: Per-layer learning rates

    Example:
        >>> # Load pretrained model for generic process heat
        >>> pretrained = load_pretrained_model("process_heat_v1")
        >>> pipeline = TransferLearningPipeline(
        ...     model=pretrained,
        ...     config=TransferLearningConfig(
        ...         source_domain=ProcessHeatDomain.GENERIC,
        ...         target_domain=ProcessHeatDomain.BOILER,
        ...         strategy=TransferStrategy.GRADUAL_UNFREEZE
        ...     )
        ... )
        >>> # Freeze backbone and replace head for boiler-specific output
        >>> pipeline.freeze_backbone()
        >>> pipeline.replace_head(num_classes=5)
        >>> # Fine-tune on boiler data
        >>> result = pipeline.fine_tune(boiler_X, boiler_y)
        >>> print(f"Transfer gain: {result.metrics.transfer_gain:.2%}")
    """

    # Registry of pretrained models per domain
    PRETRAINED_REGISTRY: Dict[ProcessHeatDomain, List[str]] = {
        ProcessHeatDomain.GENERIC: [
            "process_heat_base_v1",
            "process_heat_base_v2",
        ],
        ProcessHeatDomain.BOILER: [
            "boiler_efficiency_v1",
            "boiler_emissions_v1",
        ],
        ProcessHeatDomain.FURNACE: [
            "furnace_thermal_v1",
            "furnace_combustion_v1",
        ],
        ProcessHeatDomain.STEAM: [
            "steam_quality_v1",
            "steam_flow_v1",
        ],
        ProcessHeatDomain.CHP: [
            "chp_optimization_v1",
        ],
        ProcessHeatDomain.HEAT_EXCHANGER: [
            "heat_exchanger_efficiency_v1",
        ],
    }

    def __init__(
        self,
        model: Any,
        config: Optional[TransferLearningConfig] = None,
        pretrained_info: Optional[PretrainedModelInfo] = None
    ):
        """
        Initialize transfer learning pipeline.

        Args:
            model: Pre-trained neural network model
            config: Transfer learning configuration
            pretrained_info: Information about the pretrained model
        """
        self.model = model
        self.config = config or TransferLearningConfig()
        self.pretrained_info = pretrained_info

        # Track frozen layers
        self._frozen_layers: set = set()
        self._layer_lrs: Dict[str, float] = {}
        self._original_head: Any = None
        self._head_replaced: bool = False

        # Training state
        self._optimizer = None
        self._scheduler = None
        self._best_weights = None
        self._best_val_loss = float('inf')

        # Metrics tracking
        self._training_history: List[Dict[str, float]] = []
        self._transfer_metrics = TransferMetrics()

        np.random.seed(self.config.random_state)

        logger.info(
            f"TransferLearningPipeline initialized: "
            f"{self.config.source_domain} -> {self.config.target_domain}"
        )

    @classmethod
    def load_pretrained(
        cls,
        model_id: str,
        domain: ProcessHeatDomain = ProcessHeatDomain.GENERIC,
        config: Optional[TransferLearningConfig] = None
    ) -> "TransferLearningPipeline":
        """
        Load a pretrained model from registry.

        Args:
            model_id: Identifier of pretrained model
            domain: Domain of the pretrained model
            config: Optional configuration override

        Returns:
            TransferLearningPipeline with loaded model

        Example:
            >>> pipeline = TransferLearningPipeline.load_pretrained(
            ...     "process_heat_base_v1",
            ...     domain=ProcessHeatDomain.GENERIC
            ... )
        """
        logger.info(f"Loading pretrained model: {model_id}")

        try:
            import torch
            import torch.nn as nn

            # Create a default model architecture for demo
            # In production, this would load from model registry
            model = nn.Sequential(
                nn.Linear(50, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10)  # Default output
            )

            # Calculate model info
            n_params = sum(p.numel() for p in model.parameters())
            n_layers = len(list(model.children()))

            pretrained_info = PretrainedModelInfo(
                model_id=model_id,
                domain=domain,
                architecture="ProcessHeatMLP",
                input_dim=50,
                output_dim=10,
                n_layers=n_layers,
                n_parameters=n_params,
                pretrain_metrics={"accuracy": 0.92, "loss": 0.15},
                provenance_hash=cls._calculate_model_hash(model)
            )

            return cls(
                model=model,
                config=config,
                pretrained_info=pretrained_info
            )

        except ImportError:
            logger.warning("PyTorch not available, creating mock model")
            return cls(model=None, config=config)

    @staticmethod
    def _calculate_model_hash(model: Any) -> str:
        """Calculate SHA-256 hash of model weights."""
        try:
            import torch
            params_bytes = b""
            for param in model.parameters():
                params_bytes += param.data.cpu().numpy().tobytes()
            return hashlib.sha256(params_bytes).hexdigest()
        except Exception:
            return hashlib.sha256(str(model).encode()).hexdigest()

    def _get_layer_names(self) -> List[str]:
        """Get ordered list of layer names."""
        try:
            import torch
            return [name for name, _ in self.model.named_children()]
        except Exception:
            return []

    def _get_layer_count(self) -> int:
        """Get total number of layers."""
        try:
            return len(list(self.model.children()))
        except Exception:
            return 0

    def freeze_layer(self, layer_name: str) -> None:
        """
        Freeze a specific layer by name.

        Args:
            layer_name: Name of layer to freeze
        """
        try:
            import torch
            for name, module in self.model.named_modules():
                if name == layer_name or name.startswith(layer_name):
                    for param in module.parameters():
                        param.requires_grad = False
                    self._frozen_layers.add(layer_name)
                    logger.debug(f"Frozen layer: {layer_name}")
        except Exception as e:
            logger.error(f"Failed to freeze layer {layer_name}: {e}")

    def unfreeze_layer(self, layer_name: str) -> None:
        """
        Unfreeze a specific layer by name.

        Args:
            layer_name: Name of layer to unfreeze
        """
        try:
            import torch
            for name, module in self.model.named_modules():
                if name == layer_name or name.startswith(layer_name):
                    for param in module.parameters():
                        param.requires_grad = True
                    self._frozen_layers.discard(layer_name)
                    logger.debug(f"Unfrozen layer: {layer_name}")
        except Exception as e:
            logger.error(f"Failed to unfreeze layer {layer_name}: {e}")

    def freeze_backbone(self, n_layers: Optional[int] = None) -> int:
        """
        Freeze backbone layers (all except final head).

        Args:
            n_layers: Number of layers to freeze (from start).
                     If None, uses config value.

        Returns:
            Number of layers frozen

        Example:
            >>> pipeline.freeze_backbone(n_layers=5)
        """
        n_layers = n_layers or self.config.freeze_layers
        layer_names = self._get_layer_names()

        frozen_count = 0
        for i, name in enumerate(layer_names):
            if i < n_layers:
                self.freeze_layer(name)
                frozen_count += 1

        self._transfer_metrics.layers_frozen = frozen_count
        self._transfer_metrics.layers_unfrozen = len(layer_names) - frozen_count

        logger.info(f"Frozen {frozen_count} backbone layers")
        return frozen_count

    def unfreeze_all(self) -> None:
        """Unfreeze all layers."""
        try:
            import torch
            for param in self.model.parameters():
                param.requires_grad = True
            self._frozen_layers.clear()
            logger.info("All layers unfrozen")
        except Exception as e:
            logger.error(f"Failed to unfreeze layers: {e}")

    def freeze_all_except_head(self) -> None:
        """Freeze all layers except the final classification head."""
        layer_names = self._get_layer_names()
        if len(layer_names) > 0:
            # Freeze all but last layer
            for name in layer_names[:-1]:
                self.freeze_layer(name)
            logger.info(f"Frozen all layers except head ({layer_names[-1]})")

    def replace_head(
        self,
        num_classes: int,
        hidden_dim: Optional[int] = None
    ) -> None:
        """
        Replace the classification head for target domain.

        Args:
            num_classes: Number of output classes for target domain
            hidden_dim: Optional hidden dimension for new head

        Example:
            >>> # Replace head for boiler efficiency classification
            >>> pipeline.replace_head(num_classes=5)  # 5 efficiency levels
        """
        try:
            import torch
            import torch.nn as nn

            # Store original head
            layer_names = self._get_layer_names()
            if layer_names:
                self._original_head = getattr(self.model, layer_names[-1])

            # Determine input features to head
            if hasattr(self._original_head, 'in_features'):
                in_features = self._original_head.in_features
            else:
                # Default for MLP
                in_features = 32

            # Create new head
            if hidden_dim:
                new_head = nn.Sequential(
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, num_classes)
                )
            else:
                new_head = nn.Linear(in_features, num_classes)

            # Replace the head
            if layer_names:
                setattr(self.model, layer_names[-1], new_head)

            self._head_replaced = True
            logger.info(
                f"Replaced head: {in_features} -> {num_classes} classes"
            )

        except ImportError:
            logger.warning("PyTorch required for head replacement")
        except Exception as e:
            logger.error(f"Failed to replace head: {e}")

    def get_feature_extractor(self) -> Any:
        """
        Get the model as a feature extractor (without head).

        Returns:
            Model with head removed

        Example:
            >>> extractor = pipeline.get_feature_extractor()
            >>> features = extractor(input_data)
        """
        try:
            import torch
            import torch.nn as nn

            layer_names = self._get_layer_names()
            if len(layer_names) > 1:
                # Create sequential without last layer
                children = list(self.model.children())[:-1]
                return nn.Sequential(*children)
            return self.model
        except ImportError:
            return self.model

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from input data using frozen backbone.

        Args:
            X: Input data

        Returns:
            Extracted features

        Example:
            >>> features = pipeline.extract_features(boiler_data)
        """
        try:
            import torch

            feature_extractor = self.get_feature_extractor()
            feature_extractor.eval()

            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                features = feature_extractor(X_tensor)
                return features.numpy()
        except ImportError:
            logger.warning("PyTorch required for feature extraction")
            return X
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return X

    def set_discriminative_lrs(self, base_lr: Optional[float] = None) -> Dict[str, float]:
        """
        Set discriminative learning rates (lower LR for earlier layers).

        Args:
            base_lr: Base learning rate for final layer

        Returns:
            Dictionary of layer-wise learning rates

        Example:
            >>> lrs = pipeline.set_discriminative_lrs(base_lr=0.001)
            >>> # Earlier layers get smaller LR
        """
        base_lr = base_lr or self.config.learning_rate
        layer_names = self._get_layer_names()

        self._layer_lrs = {}
        for i, name in enumerate(reversed(layer_names)):
            # Decay LR for earlier layers
            layer_lr = base_lr * (self.config.lr_decay_factor ** i)
            self._layer_lrs[name] = layer_lr

        logger.info(f"Set discriminative LRs: {self._layer_lrs}")
        return self._layer_lrs

    def _create_optimizer(self) -> Any:
        """Create optimizer with appropriate learning rates."""
        try:
            import torch.optim as optim

            if self.config.strategy == TransferStrategy.DISCRIMINATIVE_LR:
                # Create parameter groups with different LRs
                param_groups = []
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        lr = self._layer_lrs.get(name.split('.')[0], self.config.learning_rate)
                        param_groups.append({'params': [param], 'lr': lr})
                return optim.Adam(param_groups)
            else:
                # Standard optimizer for trainable parameters
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                return optim.Adam(trainable, lr=self.config.learning_rate)
        except ImportError:
            return None

    def _evaluate_target_performance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Evaluate model performance on target domain data."""
        try:
            import torch
            import torch.nn.functional as F

            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y)

                output = self.model(X_tensor)

                if len(output.shape) > 1 and output.shape[1] > 1:
                    # Classification
                    predictions = output.argmax(dim=1)
                    accuracy = (predictions == y_tensor).float().mean().item()
                    return accuracy
                else:
                    # Regression - use negative MSE as "performance"
                    mse = F.mse_loss(output.squeeze(), y_tensor.float())
                    return -mse.item()
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0

    def fine_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> FineTuningResult:
        """
        Fine-tune the model on target domain data.

        Args:
            X: Training features
            y: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            FineTuningResult with metrics and history

        Example:
            >>> result = pipeline.fine_tune(boiler_X, boiler_y)
            >>> if result.success:
            ...     print(f"Transfer gain: {result.metrics.transfer_gain:.2%}")
        """
        start_time = datetime.utcnow()

        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError(
                "PyTorch required for fine-tuning. Install with: pip install torch"
            )

        logger.info(
            f"Starting fine-tuning: {len(X)} samples, "
            f"strategy={self.config.strategy}"
        )

        # Create validation split if not provided
        if X_val is None:
            split_idx = int(len(X) * (1 - self.config.validation_split))
            indices = np.random.permutation(len(X))
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, y_train = X, y

        # Evaluate before fine-tuning
        self._transfer_metrics.target_performance_before = self._evaluate_target_performance(
            X_val, y_val
        )

        # Apply transfer strategy
        if self.config.strategy == TransferStrategy.FREEZE_ALL:
            self.freeze_all_except_head()
        elif self.config.strategy == TransferStrategy.FREEZE_EARLY:
            self.freeze_backbone()
        elif self.config.strategy == TransferStrategy.DISCRIMINATIVE_LR:
            self.set_discriminative_lrs()
        elif self.config.strategy == TransferStrategy.FEATURE_EXTRACTION:
            self.freeze_all_except_head()

        # Prepare data loaders
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(
            y_train,
            dtype=torch.long if y_train.dtype in [np.int32, np.int64] else torch.float32
        )

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Create optimizer
        self._optimizer = self._create_optimizer()
        if self._optimizer is None:
            raise RuntimeError("Failed to create optimizer")

        # Loss function
        is_classification = y_train.dtype in [np.int32, np.int64]
        criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

        # Training loop
        self._training_history = []
        patience_counter = 0
        best_epoch = 0

        for epoch in range(self.config.n_epochs):
            # Gradual unfreezing
            if self.config.strategy == TransferStrategy.GRADUAL_UNFREEZE:
                if epoch in self.config.unfreeze_schedule:
                    layer_names = self._get_layer_names()
                    unfreeze_idx = self.config.unfreeze_schedule.index(epoch)
                    if unfreeze_idx < len(layer_names):
                        self.unfreeze_layer(layer_names[-(unfreeze_idx + 2)])
                        # Rebuild optimizer with new trainable params
                        self._optimizer = self._create_optimizer()

            # Training epoch
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                self._optimizer.zero_grad()

                output = self.model(batch_X)

                if is_classification:
                    loss = criterion(output, batch_y)
                else:
                    loss = criterion(output.squeeze(), batch_y)

                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches

            # Validation
            val_loss = self._evaluate_validation_loss(
                X_val, y_val, criterion, is_classification
            )
            val_performance = self._evaluate_target_performance(X_val, y_val)

            self._training_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_performance": val_performance
            })

            # Early stopping check
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_weights = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.n_epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )

        # Restore best weights
        if self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)

        # Evaluate after fine-tuning
        self._transfer_metrics.target_performance_after = self._evaluate_target_performance(
            X_val, y_val
        )
        self._transfer_metrics.transfer_gain = (
            self._transfer_metrics.target_performance_after -
            self._transfer_metrics.target_performance_before
        )
        self._transfer_metrics.negative_transfer = self._transfer_metrics.transfer_gain < 0
        self._transfer_metrics.convergence_epoch = best_epoch

        # Calculate provenance
        provenance_hash = self._calculate_provenance(X, y)

        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        logger.info(
            f"Fine-tuning complete. Best epoch: {best_epoch}, "
            f"Transfer gain: {self._transfer_metrics.transfer_gain:.4f}"
        )

        if self._transfer_metrics.negative_transfer:
            logger.warning("Negative transfer detected! Consider different strategy.")

        return FineTuningResult(
            success=not self._transfer_metrics.negative_transfer,
            metrics=self._transfer_metrics,
            training_history=self._training_history,
            best_epoch=best_epoch,
            best_val_loss=self._best_val_loss,
            final_lr=self.config.learning_rate,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms
        )

    def _evaluate_validation_loss(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        criterion: Any,
        is_classification: bool
    ) -> float:
        """Evaluate validation loss."""
        try:
            import torch

            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_tensor = torch.tensor(
                    y_val,
                    dtype=torch.long if is_classification else torch.float32
                )

                output = self.model(X_tensor)

                if is_classification:
                    loss = criterion(output, y_tensor)
                else:
                    loss = criterion(output.squeeze(), y_tensor)

                return loss.item()
        except Exception as e:
            logger.error(f"Validation evaluation failed: {e}")
            return float('inf')

    def _calculate_provenance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data_hash = hashlib.sha256(X.tobytes()).hexdigest()[:16]
        model_hash = self._calculate_model_hash(self.model)[:16]
        config_hash = hashlib.sha256(self.config.json().encode()).hexdigest()[:16]

        combined = f"{data_hash}|{model_hash}|{config_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_transfer_metrics(self) -> TransferMetrics:
        """Get current transfer learning metrics."""
        return self._transfer_metrics

    def get_training_history(self) -> List[Dict[str, float]]:
        """Get training history."""
        return self._training_history.copy()

    def get_frozen_layers(self) -> List[str]:
        """Get list of frozen layer names."""
        return list(self._frozen_layers)

    def get_trainable_parameters(self) -> int:
        """Get count of trainable parameters."""
        try:
            return sum(
                p.numel() for p in self.model.parameters()
                if p.requires_grad
            )
        except Exception:
            return 0

    def get_total_parameters(self) -> int:
        """Get total parameter count."""
        try:
            return sum(p.numel() for p in self.model.parameters())
        except Exception:
            return 0

    def save_model(self, path: str) -> None:
        """
        Save fine-tuned model to disk.

        Args:
            path: Path to save model
        """
        try:
            import torch
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.dict(),
                'pretrained_info': self.pretrained_info.dict() if self.pretrained_info else None,
                'transfer_metrics': self._transfer_metrics.dict(),
                'training_history': self._training_history
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, path: str) -> None:
        """
        Load fine-tuned model from disk.

        Args:
            path: Path to load model from
        """
        try:
            import torch
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.config = TransferLearningConfig(**checkpoint['config'])
            if checkpoint['pretrained_info']:
                self.pretrained_info = PretrainedModelInfo(**checkpoint['pretrained_info'])
            self._transfer_metrics = TransferMetrics(**checkpoint['transfer_metrics'])
            self._training_history = checkpoint['training_history']
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


# Factory functions for common transfer learning scenarios
def create_boiler_transfer_pipeline(
    pretrained_model: Any,
    num_efficiency_classes: int = 5
) -> TransferLearningPipeline:
    """
    Create transfer learning pipeline for boiler efficiency prediction.

    Args:
        pretrained_model: Pre-trained process heat model
        num_efficiency_classes: Number of efficiency classes

    Returns:
        Configured TransferLearningPipeline
    """
    config = TransferLearningConfig(
        source_domain=ProcessHeatDomain.GENERIC,
        target_domain=ProcessHeatDomain.BOILER,
        strategy=TransferStrategy.GRADUAL_UNFREEZE,
        freeze_layers=3
    )
    pipeline = TransferLearningPipeline(pretrained_model, config)
    pipeline.freeze_backbone()
    pipeline.replace_head(num_classes=num_efficiency_classes)
    return pipeline


def create_furnace_transfer_pipeline(
    pretrained_model: Any,
    output_dim: int = 1  # Temperature prediction
) -> TransferLearningPipeline:
    """
    Create transfer learning pipeline for furnace temperature prediction.

    Args:
        pretrained_model: Pre-trained process heat model
        output_dim: Output dimension

    Returns:
        Configured TransferLearningPipeline
    """
    config = TransferLearningConfig(
        source_domain=ProcessHeatDomain.GENERIC,
        target_domain=ProcessHeatDomain.FURNACE,
        strategy=TransferStrategy.DISCRIMINATIVE_LR,
        freeze_layers=2
    )
    pipeline = TransferLearningPipeline(pretrained_model, config)
    pipeline.replace_head(num_classes=output_dim)
    return pipeline


def create_steam_transfer_pipeline(
    pretrained_model: Any,
    num_quality_classes: int = 3
) -> TransferLearningPipeline:
    """
    Create transfer learning pipeline for steam quality classification.

    Args:
        pretrained_model: Pre-trained process heat model
        num_quality_classes: Number of quality classes

    Returns:
        Configured TransferLearningPipeline
    """
    config = TransferLearningConfig(
        source_domain=ProcessHeatDomain.GENERIC,
        target_domain=ProcessHeatDomain.STEAM,
        strategy=TransferStrategy.FREEZE_EARLY,
        freeze_layers=4
    )
    pipeline = TransferLearningPipeline(pretrained_model, config)
    pipeline.freeze_backbone()
    pipeline.replace_head(num_classes=num_quality_classes)
    return pipeline


# Unit test stubs
class TestTransferLearningPipeline:
    """Unit tests for TransferLearningPipeline."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        pipeline = TransferLearningPipeline(model=None)
        assert pipeline.config.source_domain == ProcessHeatDomain.GENERIC
        assert pipeline.config.target_domain == ProcessHeatDomain.BOILER

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = TransferLearningConfig(
            source_domain=ProcessHeatDomain.FURNACE,
            target_domain=ProcessHeatDomain.STEAM,
            strategy=TransferStrategy.DISCRIMINATIVE_LR
        )
        pipeline = TransferLearningPipeline(model=None, config=config)
        assert pipeline.config.target_domain == ProcessHeatDomain.STEAM

    def test_model_hash_deterministic(self):
        """Test model hash is deterministic."""
        try:
            import torch.nn as nn
            model = nn.Linear(10, 5)
            hash1 = TransferLearningPipeline._calculate_model_hash(model)
            hash2 = TransferLearningPipeline._calculate_model_hash(model)
            assert hash1 == hash2
        except ImportError:
            pass  # PyTorch not available

    def test_transfer_metrics_defaults(self):
        """Test transfer metrics default values."""
        metrics = TransferMetrics()
        assert metrics.source_performance == 0.0
        assert not metrics.negative_transfer

    def test_pretrained_registry(self):
        """Test pretrained model registry."""
        assert ProcessHeatDomain.BOILER in TransferLearningPipeline.PRETRAINED_REGISTRY
        assert len(TransferLearningPipeline.PRETRAINED_REGISTRY[ProcessHeatDomain.BOILER]) > 0
