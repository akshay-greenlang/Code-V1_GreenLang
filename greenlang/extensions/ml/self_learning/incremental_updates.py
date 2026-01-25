# -*- coding: utf-8 -*-
"""
Incremental Model Updates Module

This module provides warm-start training and incremental update capabilities
for GreenLang Process Heat models, enabling efficient model updates without
full retraining while maintaining compatibility with online_learner.py.

Incremental updates are critical for production systems that need to
incorporate new data quickly while minimizing computational resources
and maintaining model stability.

Example:
    >>> from greenlang.ml.self_learning import IncrementalUpdater
    >>> updater = IncrementalUpdater(model)
    >>> updater.warm_start_fit(new_data, epochs=5)
    >>> if updater.detect_degradation():
    ...     updater.rollback()
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import copy
import pickle
import os

logger = logging.getLogger(__name__)


class UpdateFrequency(str, Enum):
    """Update frequency options."""
    CONTINUOUS = "continuous"  # Update on every batch
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    ON_DRIFT = "on_drift"  # Update only when drift detected
    ON_THRESHOLD = "on_threshold"  # Update when data threshold reached
    MANUAL = "manual"


class UpdateMode(str, Enum):
    """Model update modes."""
    WARM_START = "warm_start"  # Continue training from current weights
    PARTIAL_FIT = "partial_fit"  # Incremental learning API
    FINE_TUNE = "fine_tune"  # Limited epochs with lower LR
    SLIDING_WINDOW = "sliding_window"  # Train on recent data only


class IncrementalUpdateConfig(BaseModel):
    """Configuration for incremental model updates."""

    update_frequency: UpdateFrequency = Field(
        default=UpdateFrequency.DAILY,
        description="How often to update the model"
    )
    update_mode: UpdateMode = Field(
        default=UpdateMode.WARM_START,
        description="How to update the model"
    )
    min_samples_for_update: int = Field(
        default=100,
        ge=10,
        description="Minimum samples before triggering update"
    )
    max_epochs_per_update: int = Field(
        default=10,
        ge=1,
        description="Maximum epochs per incremental update"
    )
    learning_rate: float = Field(
        default=0.0001,
        gt=0,
        description="Learning rate for warm-start (typically lower than initial)"
    )
    lr_decay_per_update: float = Field(
        default=0.95,
        gt=0,
        le=1.0,
        description="LR decay factor for each update"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for training"
    )
    validation_split: float = Field(
        default=0.2,
        ge=0,
        lt=1.0,
        description="Validation split for update"
    )
    sliding_window_size: int = Field(
        default=10000,
        ge=100,
        description="Size of sliding window (samples)"
    )
    checkpoint_frequency: int = Field(
        default=5,
        ge=1,
        description="Create checkpoint every N updates"
    )
    max_checkpoints: int = Field(
        default=10,
        ge=1,
        description="Maximum checkpoints to keep"
    )
    degradation_threshold: float = Field(
        default=0.05,
        gt=0,
        description="Performance drop threshold for rollback"
    )
    auto_rollback: bool = Field(
        default=True,
        description="Automatically rollback on degradation"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed"
    )


class CheckpointInfo(BaseModel):
    """Information about a model checkpoint."""

    checkpoint_id: str = Field(
        ...,
        description="Unique checkpoint identifier"
    )
    update_number: int = Field(
        ...,
        description="Update number this checkpoint was created"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When checkpoint was created"
    )
    samples_seen: int = Field(
        ...,
        description="Total samples seen at checkpoint"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics at checkpoint"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash of model weights"
    )


class UpdateResult(BaseModel):
    """Result from an incremental update."""

    success: bool = Field(
        ...,
        description="Whether update succeeded"
    )
    update_number: int = Field(
        ...,
        description="Cumulative update number"
    )
    samples_used: int = Field(
        ...,
        description="Samples used in this update"
    )
    epochs_completed: int = Field(
        ...,
        description="Epochs completed"
    )
    performance_before: float = Field(
        ...,
        description="Performance before update"
    )
    performance_after: float = Field(
        ...,
        description="Performance after update"
    )
    performance_change: float = Field(
        ...,
        description="Change in performance"
    )
    rolled_back: bool = Field(
        default=False,
        description="Whether update was rolled back"
    )
    checkpoint_created: bool = Field(
        default=False,
        description="Whether checkpoint was created"
    )
    learning_rate_used: float = Field(
        ...,
        description="Learning rate used for this update"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing time"
    )


class UpdateSchedule(BaseModel):
    """Schedule for model updates."""

    next_update: datetime = Field(
        ...,
        description="Next scheduled update time"
    )
    last_update: Optional[datetime] = Field(
        default=None,
        description="Last update time"
    )
    updates_today: int = Field(
        default=0,
        description="Updates performed today"
    )
    samples_pending: int = Field(
        default=0,
        description="Samples waiting for update"
    )
    is_due: bool = Field(
        default=False,
        description="Whether update is due"
    )


class IncrementalUpdater:
    """
    Incremental Model Updater for GreenLang Process Heat agents.

    This class provides efficient incremental update capabilities,
    enabling models to learn from new data without full retraining
    while maintaining stability and performance guarantees.

    Key capabilities:
    - Warm-start training from existing weights
    - Partial fit interface compatible with online_learner.py
    - Model checkpoint management
    - Update frequency scheduling
    - Rollback on performance degradation
    - Sliding window updates
    - Provenance tracking

    Attributes:
        model: Neural network model
        config: Update configuration
        _checkpoints: List of model checkpoints
        _update_history: History of updates
        _data_buffer: Buffer for incoming data

    Example:
        >>> updater = IncrementalUpdater(
        ...     model=emission_model,
        ...     config=IncrementalUpdateConfig(
        ...         update_frequency=UpdateFrequency.DAILY,
        ...         update_mode=UpdateMode.WARM_START
        ...     )
        ... )
        >>> # Add new data as it arrives
        >>> updater.add_data(new_X, new_y)
        >>> # Check if update is due
        >>> if updater.is_update_due():
        ...     result = updater.update()
        ...     if result.rolled_back:
        ...         print("Update rolled back due to degradation")
    """

    def __init__(
        self,
        model: Any,
        config: Optional[IncrementalUpdateConfig] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize incremental updater.

        Args:
            model: Model to update incrementally
            config: Update configuration
            checkpoint_dir: Directory for storing checkpoints
        """
        self.model = model
        self.config = config or IncrementalUpdateConfig()
        self.checkpoint_dir = checkpoint_dir

        # Data buffer for pending updates
        self._data_buffer_X: List[np.ndarray] = []
        self._data_buffer_y: List[np.ndarray] = []

        # Sliding window data
        self._sliding_window_X: deque = deque(maxlen=self.config.sliding_window_size)
        self._sliding_window_y: deque = deque(maxlen=self.config.sliding_window_size)

        # Checkpoints
        self._checkpoints: List[Tuple[CheckpointInfo, bytes]] = []
        self._current_weights: Optional[bytes] = None

        # Update tracking
        self._update_count: int = 0
        self._samples_seen: int = 0
        self._current_lr: float = self.config.learning_rate
        self._last_update: Optional[datetime] = None
        self._update_history: List[UpdateResult] = []

        # Performance tracking
        self._baseline_performance: Optional[float] = None
        self._current_performance: Optional[float] = None

        np.random.seed(self.config.random_state)

        # Store initial weights
        self._save_current_weights()

        logger.info(
            f"IncrementalUpdater initialized: "
            f"frequency={self.config.update_frequency}, "
            f"mode={self.config.update_mode}"
        )

    def _save_current_weights(self) -> None:
        """Save current model weights to memory."""
        try:
            import torch
            self._current_weights = pickle.dumps(self.model.state_dict())
        except Exception:
            self._current_weights = pickle.dumps(self.model)

    def _restore_weights(self, weights_bytes: bytes) -> None:
        """Restore model weights from bytes."""
        try:
            import torch
            state_dict = pickle.loads(weights_bytes)
            self.model.load_state_dict(state_dict)
        except Exception:
            self.model = pickle.loads(weights_bytes)

    def _calculate_weights_hash(self) -> str:
        """Calculate SHA-256 hash of current weights."""
        try:
            import torch
            params_bytes = b""
            for param in self.model.parameters():
                params_bytes += param.data.cpu().numpy().tobytes()
            return hashlib.sha256(params_bytes).hexdigest()
        except Exception:
            return hashlib.sha256(str(self.model).encode()).hexdigest()

    def add_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        add_to_window: bool = True
    ) -> None:
        """
        Add new data to the update buffer.

        Args:
            X: Input features
            y: Target values
            add_to_window: Also add to sliding window

        Example:
            >>> updater.add_data(new_features, new_labels)
        """
        self._data_buffer_X.append(X)
        self._data_buffer_y.append(y)

        if add_to_window:
            for i in range(len(X)):
                self._sliding_window_X.append(X[i])
                self._sliding_window_y.append(y[i])

        logger.debug(f"Added {len(X)} samples to buffer")

    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return sum(len(x) for x in self._data_buffer_X)

    def get_window_size(self) -> int:
        """Get current sliding window size."""
        return len(self._sliding_window_X)

    def clear_buffer(self) -> None:
        """Clear the data buffer."""
        self._data_buffer_X.clear()
        self._data_buffer_y.clear()
        logger.debug("Data buffer cleared")

    def is_update_due(self) -> bool:
        """
        Check if an update is due based on schedule and data.

        Returns:
            Whether update should be performed

        Example:
            >>> if updater.is_update_due():
            ...     updater.update()
        """
        buffer_size = self.get_buffer_size()

        # Check minimum samples threshold
        if buffer_size < self.config.min_samples_for_update:
            return False

        # Check frequency
        if self.config.update_frequency == UpdateFrequency.CONTINUOUS:
            return True

        elif self.config.update_frequency == UpdateFrequency.ON_THRESHOLD:
            return buffer_size >= self.config.min_samples_for_update

        elif self.config.update_frequency == UpdateFrequency.MANUAL:
            return False

        elif self._last_update is None:
            return True

        # Time-based frequencies
        now = datetime.utcnow()
        elapsed = now - self._last_update

        if self.config.update_frequency == UpdateFrequency.HOURLY:
            return elapsed >= timedelta(hours=1)
        elif self.config.update_frequency == UpdateFrequency.DAILY:
            return elapsed >= timedelta(days=1)
        elif self.config.update_frequency == UpdateFrequency.WEEKLY:
            return elapsed >= timedelta(weeks=1)

        return False

    def get_schedule(self) -> UpdateSchedule:
        """Get current update schedule status."""
        now = datetime.utcnow()

        if self._last_update is None:
            next_update = now
        else:
            if self.config.update_frequency == UpdateFrequency.HOURLY:
                next_update = self._last_update + timedelta(hours=1)
            elif self.config.update_frequency == UpdateFrequency.DAILY:
                next_update = self._last_update + timedelta(days=1)
            elif self.config.update_frequency == UpdateFrequency.WEEKLY:
                next_update = self._last_update + timedelta(weeks=1)
            else:
                next_update = now

        return UpdateSchedule(
            next_update=next_update,
            last_update=self._last_update,
            updates_today=self._get_updates_today(),
            samples_pending=self.get_buffer_size(),
            is_due=self.is_update_due()
        )

    def _get_updates_today(self) -> int:
        """Count updates performed today."""
        today = datetime.utcnow().date()
        return sum(
            1 for r in self._update_history
            if hasattr(r, 'timestamp') and r.timestamp.date() == today
        )

    def _evaluate_performance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Evaluate model performance on data."""
        try:
            import torch
            import torch.nn.functional as F

            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y)

                output = self.model(X_tensor)

                if len(output.shape) > 1 and output.shape[1] > 1:
                    # Classification accuracy
                    predictions = output.argmax(dim=1)
                    accuracy = (predictions == y_tensor).float().mean().item()
                    return accuracy
                else:
                    # Regression - use negative MSE
                    mse = F.mse_loss(output.squeeze(), y_tensor.float())
                    return -mse.item()
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return 0.0

    def warm_start_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> UpdateResult:
        """
        Perform warm-start training from current weights.

        Args:
            X: Training features
            y: Training labels
            epochs: Override epochs per update
            learning_rate: Override learning rate

        Returns:
            UpdateResult with details

        Example:
            >>> result = updater.warm_start_fit(new_X, new_y, epochs=5)
        """
        return self._do_update(X, y, epochs, learning_rate)

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> UpdateResult:
        """
        Partial fit interface compatible with online_learner.py.

        This provides a scikit-learn-like interface for incremental
        learning, making it easy to integrate with existing pipelines.

        Args:
            X: Training features
            y: Training labels

        Returns:
            UpdateResult

        Example:
            >>> # Compatible with online_learner patterns
            >>> for X_batch, y_batch in data_stream:
            ...     updater.partial_fit(X_batch, y_batch)
        """
        # Add to buffer and update if threshold reached
        self.add_data(X, y)

        if self.get_buffer_size() >= self.config.min_samples_for_update:
            return self.update()

        # Return a minimal result if no update performed
        return UpdateResult(
            success=True,
            update_number=self._update_count,
            samples_used=0,
            epochs_completed=0,
            performance_before=self._current_performance or 0.0,
            performance_after=self._current_performance or 0.0,
            performance_change=0.0,
            learning_rate_used=self._current_lr,
            provenance_hash=self._calculate_weights_hash(),
            processing_time_ms=0.0
        )

    def update(self) -> UpdateResult:
        """
        Perform scheduled update using buffered data.

        Returns:
            UpdateResult with details

        Example:
            >>> if updater.is_update_due():
            ...     result = updater.update()
        """
        if self.get_buffer_size() == 0:
            logger.warning("No data in buffer for update")
            return UpdateResult(
                success=False,
                update_number=self._update_count,
                samples_used=0,
                epochs_completed=0,
                performance_before=self._current_performance or 0.0,
                performance_after=self._current_performance or 0.0,
                performance_change=0.0,
                learning_rate_used=self._current_lr,
                provenance_hash=self._calculate_weights_hash(),
                processing_time_ms=0.0
            )

        # Concatenate buffer data
        X = np.vstack(self._data_buffer_X)
        y = np.concatenate(self._data_buffer_y)

        # Perform update based on mode
        if self.config.update_mode == UpdateMode.SLIDING_WINDOW:
            # Use sliding window data
            X = np.array(list(self._sliding_window_X))
            y = np.array(list(self._sliding_window_y))

        result = self._do_update(X, y)

        # Clear buffer after update
        self.clear_buffer()

        return result

    def _do_update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> UpdateResult:
        """Perform the actual update."""
        start_time = datetime.utcnow()

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError("PyTorch required for incremental updates")

        epochs = epochs or self.config.max_epochs_per_update
        lr = learning_rate or self._current_lr

        logger.info(
            f"Starting update {self._update_count + 1}: "
            f"{len(X)} samples, {epochs} epochs, lr={lr}"
        )

        # Store weights for potential rollback
        self._save_current_weights()
        weights_before = self._current_weights

        # Create validation split
        split_idx = int(len(X) * (1 - self.config.validation_split))
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Evaluate before update
        performance_before = self._evaluate_performance(X_val, y_val)
        if self._baseline_performance is None:
            self._baseline_performance = performance_before

        # Prepare data
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(
            y_train,
            dtype=torch.long if y_train.dtype in [np.int32, np.int64] else torch.float32
        )

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Create optimizer for trainable parameters only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            trainable_params = list(self.model.parameters())

        optimizer = optim.Adam(trainable_params, lr=lr)

        # Loss function
        is_classification = y_train.dtype in [np.int32, np.int64]
        criterion = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

        # Training loop
        self.model.train()
        epochs_completed = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()

                output = self.model(batch_X)

                if is_classification:
                    loss = criterion(output, batch_y)
                else:
                    loss = criterion(output.squeeze(), batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epochs_completed = epoch + 1

        # Evaluate after update
        performance_after = self._evaluate_performance(X_val, y_val)
        performance_change = performance_after - performance_before

        # Check for degradation
        rolled_back = False
        if (self.config.auto_rollback and
            performance_change < -self.config.degradation_threshold):

            logger.warning(
                f"Performance degraded by {abs(performance_change):.4f}, "
                f"rolling back"
            )
            self._restore_weights(weights_before)
            rolled_back = True
            performance_after = performance_before

        # Update state
        self._update_count += 1
        self._samples_seen += len(X)
        self._last_update = datetime.utcnow()
        self._current_performance = performance_after

        # Decay learning rate
        self._current_lr *= self.config.lr_decay_per_update

        # Create checkpoint if scheduled
        checkpoint_created = False
        if self._update_count % self.config.checkpoint_frequency == 0:
            self._create_checkpoint()
            checkpoint_created = True

        # Calculate provenance
        provenance_hash = self._calculate_provenance(X, y, performance_after)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = UpdateResult(
            success=not rolled_back,
            update_number=self._update_count,
            samples_used=len(X),
            epochs_completed=epochs_completed,
            performance_before=performance_before,
            performance_after=performance_after,
            performance_change=performance_change,
            rolled_back=rolled_back,
            checkpoint_created=checkpoint_created,
            learning_rate_used=lr,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

        self._update_history.append(result)

        logger.info(
            f"Update {self._update_count} complete: "
            f"perf {performance_before:.4f} -> {performance_after:.4f}"
        )

        return result

    def _calculate_provenance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        performance: float
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data_hash = hashlib.sha256(X.tobytes()).hexdigest()[:16]
        weights_hash = self._calculate_weights_hash()[:16]
        combined = f"{data_hash}|{weights_hash}|{performance}|{self._update_count}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _create_checkpoint(self) -> CheckpointInfo:
        """Create a model checkpoint."""
        self._save_current_weights()

        checkpoint_id = f"ckpt_{self._update_count}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            update_number=self._update_count,
            timestamp=datetime.utcnow(),
            samples_seen=self._samples_seen,
            performance_metrics={
                "current": self._current_performance or 0.0,
                "baseline": self._baseline_performance or 0.0
            },
            provenance_hash=self._calculate_weights_hash()
        )

        self._checkpoints.append((info, self._current_weights))

        # Prune old checkpoints
        while len(self._checkpoints) > self.config.max_checkpoints:
            self._checkpoints.pop(0)

        # Optionally save to disk
        if self.checkpoint_dir:
            self._save_checkpoint_to_disk(info, self._current_weights)

        logger.info(f"Created checkpoint: {checkpoint_id}")
        return info

    def _save_checkpoint_to_disk(
        self,
        info: CheckpointInfo,
        weights: bytes
    ) -> None:
        """Save checkpoint to disk."""
        if not self.checkpoint_dir:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save weights
        weights_path = os.path.join(
            self.checkpoint_dir,
            f"{info.checkpoint_id}_weights.pkl"
        )
        with open(weights_path, 'wb') as f:
            f.write(weights)

        # Save info
        info_path = os.path.join(
            self.checkpoint_dir,
            f"{info.checkpoint_id}_info.json"
        )
        with open(info_path, 'w') as f:
            f.write(info.json())

    def get_checkpoints(self) -> List[CheckpointInfo]:
        """Get list of available checkpoints."""
        return [info for info, _ in self._checkpoints]

    def rollback(
        self,
        checkpoint_id: Optional[str] = None,
        steps: int = 1
    ) -> bool:
        """
        Rollback model to a previous checkpoint.

        Args:
            checkpoint_id: Specific checkpoint to rollback to
            steps: Number of updates to rollback (if no checkpoint_id)

        Returns:
            Whether rollback succeeded

        Example:
            >>> # Rollback to previous checkpoint
            >>> updater.rollback(steps=1)
            >>> # Rollback to specific checkpoint
            >>> updater.rollback(checkpoint_id="ckpt_5_20240101_120000")
        """
        if not self._checkpoints:
            logger.warning("No checkpoints available for rollback")
            return False

        if checkpoint_id:
            # Find specific checkpoint
            for info, weights in self._checkpoints:
                if info.checkpoint_id == checkpoint_id:
                    self._restore_weights(weights)
                    self._current_performance = info.performance_metrics.get("current", 0.0)
                    logger.info(f"Rolled back to checkpoint: {checkpoint_id}")
                    return True
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return False
        else:
            # Rollback by steps
            if steps > len(self._checkpoints):
                steps = len(self._checkpoints)

            target_idx = len(self._checkpoints) - steps
            info, weights = self._checkpoints[target_idx]
            self._restore_weights(weights)
            self._current_performance = info.performance_metrics.get("current", 0.0)
            logger.info(f"Rolled back {steps} step(s) to: {info.checkpoint_id}")
            return True

    def detect_degradation(
        self,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> bool:
        """
        Detect if model has degraded from baseline.

        Args:
            X_test: Test features (uses sliding window if not provided)
            y_test: Test labels

        Returns:
            Whether degradation is detected

        Example:
            >>> if updater.detect_degradation():
            ...     updater.rollback()
        """
        if X_test is None:
            if len(self._sliding_window_X) < 100:
                return False
            X_test = np.array(list(self._sliding_window_X)[-100:])
            y_test = np.array(list(self._sliding_window_y)[-100:])

        current = self._evaluate_performance(X_test, y_test)
        baseline = self._baseline_performance or current

        degradation = baseline - current
        is_degraded = degradation > self.config.degradation_threshold

        if is_degraded:
            logger.warning(
                f"Performance degradation detected: "
                f"baseline={baseline:.4f}, current={current:.4f}, "
                f"drop={degradation:.4f}"
            )

        return is_degraded

    def get_update_history(
        self,
        limit: Optional[int] = None
    ) -> List[UpdateResult]:
        """Get update history."""
        if limit:
            return self._update_history[-limit:]
        return self._update_history.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get updater statistics."""
        return {
            "update_count": self._update_count,
            "samples_seen": self._samples_seen,
            "current_lr": self._current_lr,
            "buffer_size": self.get_buffer_size(),
            "window_size": self.get_window_size(),
            "checkpoints": len(self._checkpoints),
            "baseline_performance": self._baseline_performance,
            "current_performance": self._current_performance,
            "last_update": self._last_update.isoformat() if self._last_update else None
        }

    def reset(self) -> None:
        """Reset updater state (not model weights)."""
        self.clear_buffer()
        self._sliding_window_X.clear()
        self._sliding_window_y.clear()
        self._update_history.clear()
        self._update_count = 0
        self._samples_seen = 0
        self._current_lr = self.config.learning_rate
        self._last_update = None
        self._baseline_performance = None
        self._current_performance = None
        logger.info("IncrementalUpdater reset")


# Factory functions
def create_daily_updater(
    model: Any,
    min_samples: int = 100
) -> IncrementalUpdater:
    """Create an incremental updater with daily schedule."""
    config = IncrementalUpdateConfig(
        update_frequency=UpdateFrequency.DAILY,
        update_mode=UpdateMode.WARM_START,
        min_samples_for_update=min_samples
    )
    return IncrementalUpdater(model, config)


def create_continuous_updater(
    model: Any,
    batch_threshold: int = 50
) -> IncrementalUpdater:
    """Create an incremental updater with continuous updates."""
    config = IncrementalUpdateConfig(
        update_frequency=UpdateFrequency.CONTINUOUS,
        update_mode=UpdateMode.PARTIAL_FIT,
        min_samples_for_update=batch_threshold,
        max_epochs_per_update=1
    )
    return IncrementalUpdater(model, config)


def create_sliding_window_updater(
    model: Any,
    window_size: int = 10000
) -> IncrementalUpdater:
    """Create an updater using sliding window of recent data."""
    config = IncrementalUpdateConfig(
        update_frequency=UpdateFrequency.DAILY,
        update_mode=UpdateMode.SLIDING_WINDOW,
        sliding_window_size=window_size
    )
    return IncrementalUpdater(model, config)


# Unit test stubs
class TestIncrementalUpdater:
    """Unit tests for IncrementalUpdater."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        updater = IncrementalUpdater(model=None)
        assert updater.config.update_frequency == UpdateFrequency.DAILY
        assert updater.config.update_mode == UpdateMode.WARM_START

    def test_buffer_management(self):
        """Test data buffer management."""
        updater = IncrementalUpdater(model=None)

        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)

        updater.add_data(X, y)
        assert updater.get_buffer_size() == 50

        updater.clear_buffer()
        assert updater.get_buffer_size() == 0

    def test_sliding_window(self):
        """Test sliding window functionality."""
        config = IncrementalUpdateConfig(sliding_window_size=100)
        updater = IncrementalUpdater(model=None, config=config)

        # Add more than window size
        for i in range(15):
            X = np.random.randn(10, 5)
            y = np.random.randint(0, 2, 10)
            updater.add_data(X, y)

        assert updater.get_window_size() == 100

    def test_update_schedule(self):
        """Test update schedule calculation."""
        config = IncrementalUpdateConfig(
            update_frequency=UpdateFrequency.DAILY,
            min_samples_for_update=10
        )
        updater = IncrementalUpdater(model=None, config=config)

        # No data - not due
        assert not updater.is_update_due()

        # Add data - should be due (first update)
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 2, 20)
        updater.add_data(X, y)
        assert updater.is_update_due()

    def test_checkpoint_management(self):
        """Test checkpoint creation and retrieval."""
        config = IncrementalUpdateConfig(max_checkpoints=3)
        updater = IncrementalUpdater(model=None, config=config)

        # Manually create checkpoints
        for i in range(5):
            updater._update_count = i + 1
            updater._create_checkpoint()

        # Should only keep last 3
        assert len(updater._checkpoints) == 3

    def test_statistics(self):
        """Test statistics gathering."""
        updater = IncrementalUpdater(model=None)
        stats = updater.get_statistics()

        assert "update_count" in stats
        assert "buffer_size" in stats
        assert "current_lr" in stats
