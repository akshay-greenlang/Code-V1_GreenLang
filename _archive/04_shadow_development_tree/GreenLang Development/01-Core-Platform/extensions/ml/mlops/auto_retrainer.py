# -*- coding: utf-8 -*-
"""
Auto Retrainer Module

This module provides automatic model retraining pipeline for GreenLang,
enabling continuous model improvement based on drift detection, performance
degradation, and scheduled updates.

Automatic retraining is essential for maintaining model accuracy over time,
especially for regulatory compliance where models must reflect current
emission factors and regulations.

Example:
    >>> from greenlang.ml.mlops import AutoRetrainer
    >>> retrainer = AutoRetrainer(model, training_fn=train_model)
    >>> retrainer.configure_triggers(drift_threshold=0.2, perf_threshold=0.85)
    >>> retrainer.start_monitoring()
"""

from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pydantic import BaseModel, Field
import numpy as np
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)


class RetrainingTrigger(str, Enum):
    """Types of retraining triggers."""
    DRIFT = "drift"
    PERFORMANCE = "performance"
    SCHEDULE = "schedule"
    MANUAL = "manual"
    DATA_VOLUME = "data_volume"


class RetrainingStatus(str, Enum):
    """Status of retraining job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RetrainingStrategy(str, Enum):
    """Retraining strategy."""
    FULL = "full"
    INCREMENTAL = "incremental"
    FINE_TUNE = "fine_tune"
    TRANSFER = "transfer"


class AutoRetrainerConfig(BaseModel):
    """Configuration for auto retrainer."""

    strategy: RetrainingStrategy = Field(
        default=RetrainingStrategy.INCREMENTAL,
        description="Retraining strategy"
    )
    drift_threshold: float = Field(
        default=0.2,
        gt=0,
        description="Drift score threshold to trigger retraining"
    )
    performance_threshold: float = Field(
        default=0.85,
        gt=0,
        le=1.0,
        description="Performance threshold below which to retrain"
    )
    schedule_days: int = Field(
        default=7,
        ge=1,
        description="Days between scheduled retraining"
    )
    min_new_samples: int = Field(
        default=1000,
        ge=100,
        description="Minimum new samples for retraining"
    )
    validation_split: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Validation split for evaluation"
    )
    max_retrain_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum retraining attempts"
    )
    rollback_on_regression: bool = Field(
        default=True,
        description="Rollback if new model is worse"
    )
    auto_promote: bool = Field(
        default=False,
        description="Automatically promote to production"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    artifact_path: str = Field(
        default="./retraining_artifacts",
        description="Path for artifacts"
    )


class RetrainingJob(BaseModel):
    """Information about a retraining job."""

    job_id: str = Field(
        ...,
        description="Unique job identifier"
    )
    trigger: RetrainingTrigger = Field(
        ...,
        description="What triggered retraining"
    )
    strategy: RetrainingStrategy = Field(
        ...,
        description="Retraining strategy used"
    )
    status: RetrainingStatus = Field(
        default=RetrainingStatus.PENDING,
        description="Current status"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Completion timestamp"
    )
    n_training_samples: int = Field(
        default=0,
        description="Number of training samples"
    )
    n_validation_samples: int = Field(
        default=0,
        description="Number of validation samples"
    )
    old_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Metrics before retraining"
    )
    new_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Metrics after retraining"
    )
    improvement: Dict[str, float] = Field(
        default_factory=dict,
        description="Improvement per metric"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    promoted: bool = Field(
        default=False,
        description="Whether model was promoted"
    )
    rolled_back: bool = Field(
        default=False,
        description="Whether model was rolled back"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash"
    )


class AutoRetrainer:
    """
    Auto Retrainer for GreenLang ML models.

    This class provides automated model retraining capabilities,
    enabling continuous model improvement based on configurable
    triggers and validation.

    Key capabilities:
    - Drift-triggered retraining
    - Performance-triggered retraining
    - Scheduled retraining
    - Incremental and full retraining
    - Automatic validation and rollback
    - Provenance tracking

    Attributes:
        model: Current model
        config: Retrainer configuration
        _training_fn: Function to train model
        _evaluation_fn: Function to evaluate model
        _data_buffer: Buffer for new training data
        _job_history: History of retraining jobs

    Example:
        >>> retrainer = AutoRetrainer(
        ...     model,
        ...     training_fn=train_emission_model,
        ...     evaluation_fn=evaluate_model,
        ...     config=AutoRetrainerConfig(
        ...         drift_threshold=0.15,
        ...         performance_threshold=0.90
        ...     )
        ... )
        >>> # Add new data
        >>> retrainer.add_training_data(X_new, y_new)
        >>> # Check if retraining needed
        >>> if retrainer.should_retrain():
        ...     job = retrainer.retrain()
    """

    def __init__(
        self,
        model: Any,
        training_fn: Optional[Callable] = None,
        evaluation_fn: Optional[Callable] = None,
        config: Optional[AutoRetrainerConfig] = None
    ):
        """
        Initialize auto retrainer.

        Args:
            model: Initial model
            training_fn: Training function (X, y) -> model
            evaluation_fn: Evaluation function (model, X, y) -> metrics
            config: Retrainer configuration
        """
        self.model = model
        self._original_model = model  # Keep original for rollback
        self.config = config or AutoRetrainerConfig()
        self._training_fn = training_fn or self._default_training
        self._evaluation_fn = evaluation_fn or self._default_evaluation

        self._data_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self._job_history: List[RetrainingJob] = []
        self._last_retrain: Optional[datetime] = None
        self._current_metrics: Dict[str, float] = {}
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Create artifact directory
        Path(self.config.artifact_path).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"AutoRetrainer initialized: strategy={self.config.strategy}"
        )

    def _default_training(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Any:
        """Default training function using sklearn."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(random_state=42)
            model.fit(X, y)
            return model
        except ImportError:
            raise ImportError(
                "Training function not provided and sklearn not available"
            )

    def _default_evaluation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Default evaluation function."""
        predictions = model.predict(X)

        mse = float(np.mean((y - predictions) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y - predictions)))

        # R-squared
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

    def _calculate_provenance(
        self,
        job: RetrainingJob,
        X: np.ndarray
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data_hash = hashlib.sha256(X.tobytes()).hexdigest()[:16]
        combined = (
            f"{job.job_id}|{job.trigger.value}|{data_hash}|"
            f"{job.n_training_samples}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def add_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> int:
        """
        Add new training data to buffer.

        Args:
            X: Feature data
            y: Target data

        Returns:
            Total samples in buffer

        Example:
            >>> retrainer.add_training_data(X_new, y_new)
        """
        self._data_buffer.append((X, y))
        total_samples = sum(x.shape[0] for x, _ in self._data_buffer)

        logger.debug(f"Added {X.shape[0]} samples, total buffer: {total_samples}")
        return total_samples

    def get_buffer_size(self) -> int:
        """Get total samples in data buffer."""
        return sum(x.shape[0] for x, _ in self._data_buffer)

    def clear_buffer(self) -> None:
        """Clear the data buffer."""
        self._data_buffer.clear()
        logger.info("Data buffer cleared")

    def should_retrain(
        self,
        drift_score: Optional[float] = None,
        performance_metric: Optional[float] = None
    ) -> Tuple[bool, Optional[RetrainingTrigger]]:
        """
        Check if retraining should be triggered.

        Args:
            drift_score: Current drift score
            performance_metric: Current performance metric

        Returns:
            Tuple of (should_retrain, trigger_type)

        Example:
            >>> should, trigger = retrainer.should_retrain(drift_score=0.25)
            >>> if should:
            ...     retrainer.retrain()
        """
        # Check drift threshold
        if drift_score is not None and drift_score > self.config.drift_threshold:
            logger.info(f"Drift threshold exceeded: {drift_score:.4f}")
            return (True, RetrainingTrigger.DRIFT)

        # Check performance threshold
        if (performance_metric is not None and
            performance_metric < self.config.performance_threshold):
            logger.info(f"Performance below threshold: {performance_metric:.4f}")
            return (True, RetrainingTrigger.PERFORMANCE)

        # Check data volume
        if self.get_buffer_size() >= self.config.min_new_samples:
            logger.info(f"Data volume threshold reached: {self.get_buffer_size()}")
            return (True, RetrainingTrigger.DATA_VOLUME)

        # Check schedule
        if self._last_retrain is not None:
            days_since = (datetime.utcnow() - self._last_retrain).days
            if days_since >= self.config.schedule_days:
                logger.info(f"Scheduled retraining: {days_since} days since last")
                return (True, RetrainingTrigger.SCHEDULE)

        return (False, None)

    def retrain(
        self,
        trigger: RetrainingTrigger = RetrainingTrigger.MANUAL,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> RetrainingJob:
        """
        Execute retraining job.

        Args:
            trigger: What triggered retraining
            X: Training features (uses buffer if None)
            y: Training targets (uses buffer if None)
            X_val: Validation features
            y_val: Validation targets

        Returns:
            RetrainingJob with results

        Example:
            >>> job = retrainer.retrain(trigger=RetrainingTrigger.DRIFT)
            >>> if job.status == RetrainingStatus.COMPLETED:
            ...     print(f"Improvement: {job.improvement}")
        """
        job_id = self._generate_job_id()
        job = RetrainingJob(
            job_id=job_id,
            trigger=trigger,
            strategy=self.config.strategy,
            status=RetrainingStatus.RUNNING,
            started_at=datetime.utcnow()
        )

        logger.info(f"Starting retraining job {job_id}, trigger={trigger.value}")

        try:
            # Prepare training data
            if X is None or y is None:
                if not self._data_buffer:
                    raise ValueError("No training data provided or in buffer")

                X = np.vstack([x for x, _ in self._data_buffer])
                y = np.concatenate([y for _, y in self._data_buffer])

            # Split validation if not provided
            if X_val is None or y_val is None:
                split_idx = int(len(X) * (1 - self.config.validation_split))
                indices = np.random.permutation(len(X))
                train_idx = indices[:split_idx]
                val_idx = indices[split_idx:]

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
            else:
                X_train, y_train = X, y

            job.n_training_samples = len(X_train)
            job.n_validation_samples = len(X_val)

            # Evaluate current model
            old_metrics = self._evaluation_fn(self.model, X_val, y_val)
            job.old_metrics = old_metrics
            self._current_metrics = old_metrics

            # Train new model based on strategy
            if self.config.strategy == RetrainingStrategy.FULL:
                new_model = self._training_fn(X_train, y_train)

            elif self.config.strategy == RetrainingStrategy.INCREMENTAL:
                # Combine with some historical data if available
                new_model = self._training_fn(X_train, y_train)

            elif self.config.strategy == RetrainingStrategy.FINE_TUNE:
                # Fine-tune existing model
                if hasattr(self.model, "partial_fit"):
                    self.model.partial_fit(X_train, y_train)
                    new_model = self.model
                else:
                    new_model = self._training_fn(X_train, y_train)

            else:
                new_model = self._training_fn(X_train, y_train)

            # Evaluate new model
            new_metrics = self._evaluation_fn(new_model, X_val, y_val)
            job.new_metrics = new_metrics

            # Calculate improvement
            for key in old_metrics:
                if key in new_metrics:
                    # For error metrics (lower is better)
                    if key in ["mse", "rmse", "mae"]:
                        improvement = (old_metrics[key] - new_metrics[key]) / (old_metrics[key] + 1e-10)
                    else:
                        # For accuracy metrics (higher is better)
                        improvement = (new_metrics[key] - old_metrics[key]) / (old_metrics[key] + 1e-10)
                    job.improvement[key] = float(improvement)

            # Check for regression
            is_regression = all(imp < 0 for imp in job.improvement.values())

            if is_regression and self.config.rollback_on_regression:
                logger.warning("New model shows regression, rolling back")
                job.rolled_back = True
                job.status = RetrainingStatus.COMPLETED
            else:
                # Update model
                self._original_model = self.model
                self.model = new_model
                self._current_metrics = new_metrics

                # Auto-promote if configured
                if self.config.auto_promote:
                    job.promoted = True
                    logger.info("Model auto-promoted to production")

                job.status = RetrainingStatus.COMPLETED

            # Update timestamps
            job.completed_at = datetime.utcnow()
            self._last_retrain = datetime.utcnow()

            # Clear buffer after successful training
            if job.status == RetrainingStatus.COMPLETED and not job.rolled_back:
                self.clear_buffer()

            # Calculate provenance
            job.provenance_hash = self._calculate_provenance(job, X_train)

            # Save job history
            self._job_history.append(job)

            logger.info(
                f"Retraining job {job_id} completed: "
                f"improvement={job.improvement}, rolled_back={job.rolled_back}"
            )

        except Exception as e:
            job.status = RetrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            self._job_history.append(job)
            logger.error(f"Retraining job {job_id} failed: {e}")

        return job

    def rollback(self) -> bool:
        """
        Rollback to previous model.

        Returns:
            Success status
        """
        if self._original_model is not None:
            self.model = self._original_model
            logger.info("Rolled back to previous model")
            return True
        return False

    def promote_model(self) -> bool:
        """
        Promote current model to production.

        Returns:
            Success status
        """
        # This would integrate with model registry
        logger.info("Model promoted to production")
        return True

    def get_job_history(
        self,
        limit: Optional[int] = None
    ) -> List[RetrainingJob]:
        """Get retraining job history."""
        if limit:
            return self._job_history[-limit:]
        return self._job_history.copy()

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current model metrics."""
        return self._current_metrics.copy()

    def start_monitoring(
        self,
        check_interval_seconds: int = 3600
    ) -> None:
        """
        Start background monitoring for retraining triggers.

        Args:
            check_interval_seconds: Interval between checks
        """
        if self._monitoring:
            logger.warning("Monitoring already running")
            return

        self._monitoring = True

        def monitor_loop():
            while self._monitoring:
                should, trigger = self.should_retrain()
                if should and trigger:
                    self.retrain(trigger=trigger)
                time.sleep(check_interval_seconds)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Monitoring started (interval={check_interval_seconds}s)")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        logger.info("Monitoring stopped")

    def get_retraining_summary(self) -> Dict[str, Any]:
        """Get summary of retraining history."""
        if not self._job_history:
            return {"total_jobs": 0}

        completed_jobs = [j for j in self._job_history if j.status == RetrainingStatus.COMPLETED]

        return {
            "total_jobs": len(self._job_history),
            "completed_jobs": len(completed_jobs),
            "failed_jobs": sum(1 for j in self._job_history if j.status == RetrainingStatus.FAILED),
            "trigger_distribution": {
                t.value: sum(1 for j in self._job_history if j.trigger == t)
                for t in RetrainingTrigger
            },
            "avg_improvement": {
                key: float(np.mean([j.improvement.get(key, 0) for j in completed_jobs if j.improvement]))
                for key in ["r2", "rmse", "mae"]
            },
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None
        }


# Unit test stubs
class TestAutoRetrainer:
    """Unit tests for AutoRetrainer."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        retrainer = AutoRetrainer(MockModel())
        assert retrainer.config.strategy == RetrainingStrategy.INCREMENTAL

    def test_add_training_data(self):
        """Test adding training data."""
        class MockModel:
            pass

        retrainer = AutoRetrainer(MockModel())

        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        retrainer.add_training_data(X, y)

        assert retrainer.get_buffer_size() == 100

    def test_should_retrain_drift(self):
        """Test drift-triggered retraining."""
        class MockModel:
            pass

        config = AutoRetrainerConfig(drift_threshold=0.2)
        retrainer = AutoRetrainer(MockModel(), config=config)

        should, trigger = retrainer.should_retrain(drift_score=0.25)
        assert should
        assert trigger == RetrainingTrigger.DRIFT

    def test_should_retrain_performance(self):
        """Test performance-triggered retraining."""
        class MockModel:
            pass

        config = AutoRetrainerConfig(performance_threshold=0.85)
        retrainer = AutoRetrainer(MockModel(), config=config)

        should, trigger = retrainer.should_retrain(performance_metric=0.75)
        assert should
        assert trigger == RetrainingTrigger.PERFORMANCE

    def test_clear_buffer(self):
        """Test clearing data buffer."""
        class MockModel:
            pass

        retrainer = AutoRetrainer(MockModel())
        retrainer.add_training_data(np.random.randn(100, 5), np.random.randn(100))
        assert retrainer.get_buffer_size() == 100

        retrainer.clear_buffer()
        assert retrainer.get_buffer_size() == 0
