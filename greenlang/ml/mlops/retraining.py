"""
Auto-Retraining Pipeline - Automated Model Retraining with Triggers.

This module provides automated model retraining capabilities for GreenLang
Process Heat agents, supporting drift-based, schedule-based, and data volume
triggers with comprehensive validation before deployment.

Example:
    >>> from greenlang.ml.mlops.retraining import RetrainingPipeline
    >>> pipeline = RetrainingPipeline()
    >>> pipeline.configure_triggers(drift_threshold=0.2, schedule="0 0 * * 0")
    >>> needed, reason = pipeline.check_retraining_needed("heat_model")
    >>> if needed:
    ...     pipeline.trigger_retraining("heat_model", training_config)
"""

import hashlib
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import MLOpsConfig, get_config
from .drift_detection import DriftDetector
from .schemas import (
    ModelStage,
    RetrainingConfig,
    RetrainingResult,
    RetrainingTrigger,
    RetrainingTriggerType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Cron Parser (Simple Implementation)
# =============================================================================

class SimpleCronParser:
    """
    Simple cron expression parser.

    Supports basic cron format: minute hour day month weekday
    Example: "0 0 * * 0" = Sunday at midnight
    """

    def __init__(self, expression: str):
        """
        Initialize cron parser.

        Args:
            expression: Cron expression string.
        """
        self.expression = expression
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")

        self.minute = self._parse_field(parts[0], 0, 59)
        self.hour = self._parse_field(parts[1], 0, 23)
        self.day = self._parse_field(parts[2], 1, 31)
        self.month = self._parse_field(parts[3], 1, 12)
        self.weekday = self._parse_field(parts[4], 0, 6)

    def _parse_field(self, field: str, min_val: int, max_val: int) -> set:
        """Parse a single cron field."""
        if field == "*":
            return set(range(min_val, max_val + 1))

        if "/" in field:
            # Step values like */5
            base, step = field.split("/")
            step = int(step)
            if base == "*":
                return set(range(min_val, max_val + 1, step))

        if "," in field:
            # List of values
            return set(int(v) for v in field.split(","))

        if "-" in field:
            # Range
            start, end = field.split("-")
            return set(range(int(start), int(end) + 1))

        # Single value
        return {int(field)}

    def matches(self, dt: datetime) -> bool:
        """Check if datetime matches cron expression."""
        return (
            dt.minute in self.minute
            and dt.hour in self.hour
            and dt.day in self.day
            and dt.month in self.month
            and dt.weekday() in self.weekday
        )

    def next_run(self, after: datetime) -> datetime:
        """Calculate next run time after given datetime."""
        current = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Simple iteration (not efficient but works)
        for _ in range(60 * 24 * 31):  # Max 31 days lookahead
            if self.matches(current):
                return current
            current += timedelta(minutes=1)

        return current


# =============================================================================
# Retraining Pipeline Implementation
# =============================================================================

class RetrainingPipeline:
    """
    Automated model retraining pipeline with configurable triggers.

    This class provides comprehensive retraining automation including:
    - Drift-based triggers (when data distribution shifts)
    - Schedule-based triggers (cron expressions)
    - Data volume triggers (when enough new data accumulates)
    - Performance-based triggers (when model accuracy degrades)
    - Validation before deployment

    Attributes:
        config: MLOps configuration
        storage_path: Path for retraining data storage
        triggers: Configured retraining triggers

    Example:
        >>> pipeline = RetrainingPipeline()
        >>> pipeline.configure_triggers(
        ...     drift_threshold=0.2,
        ...     schedule="0 0 * * 0",  # Weekly Sunday midnight
        ...     min_samples=5000
        ... )
        >>> needed, reason = pipeline.check_retraining_needed("model_name")
        >>> if needed:
        ...     result = pipeline.trigger_retraining("model_name", config)
    """

    def __init__(self, config: Optional[MLOpsConfig] = None):
        """
        Initialize RetrainingPipeline.

        Args:
            config: MLOps configuration. If None, uses default configuration.
        """
        self.config = config or get_config()
        self.storage_path = Path(self.config.retraining.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._triggers: Dict[str, List[RetrainingTrigger]] = {}
        self._retraining_history: Dict[str, List[RetrainingResult]] = {}
        self._data_counters: Dict[str, int] = {}
        self._last_retraining: Dict[str, datetime] = {}
        self._drift_detector = DriftDetector(config)

        # Training function registry
        self._training_functions: Dict[str, Callable] = {}

        # Load existing state
        self._load_state()

        logger.info("RetrainingPipeline initialized")

    def _load_state(self) -> None:
        """Load pipeline state from storage."""
        state_file = self.storage_path / "pipeline_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                    self._data_counters = state.get("data_counters", {})
                    self._last_retraining = {
                        k: datetime.fromisoformat(v)
                        for k, v in state.get("last_retraining", {}).items()
                    }

                    # Load triggers
                    for model_name, triggers in state.get("triggers", {}).items():
                        self._triggers[model_name] = [
                            RetrainingTrigger(**t) for t in triggers
                        ]

                logger.info("Loaded pipeline state from storage")
            except Exception as e:
                logger.warning(f"Failed to load pipeline state: {e}")

    def _save_state(self) -> None:
        """Save pipeline state to storage."""
        state_file = self.storage_path / "pipeline_state.json"
        state = {
            "data_counters": self._data_counters,
            "last_retraining": {
                k: v.isoformat() for k, v in self._last_retraining.items()
            },
            "triggers": {
                k: [t.dict() for t in v] for k, v in self._triggers.items()
            },
        }
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def register_training_function(
        self, model_name: str, training_fn: Callable
    ) -> None:
        """
        Register a training function for a model.

        Args:
            model_name: Name of the model.
            training_fn: Function that trains the model.
                         Signature: (config: dict) -> Tuple[model, metrics]
        """
        self._training_functions[model_name] = training_fn
        logger.info(f"Registered training function for {model_name}")

    def configure_triggers(
        self,
        model_name: str,
        drift_threshold: Optional[float] = None,
        schedule: Optional[str] = None,
        min_samples: Optional[int] = None,
        performance_threshold: Optional[float] = None,
    ) -> None:
        """
        Configure retraining triggers for a model.

        Args:
            model_name: Name of the model.
            drift_threshold: Drift score threshold to trigger retraining.
            schedule: Cron expression for scheduled retraining.
            min_samples: Minimum new samples to trigger retraining.
            performance_threshold: Performance degradation threshold.
        """
        triggers = []

        if drift_threshold is not None:
            triggers.append(
                RetrainingTrigger(
                    trigger_type=RetrainingTriggerType.DRIFT,
                    enabled=True,
                    drift_threshold=drift_threshold,
                )
            )

        if schedule is not None:
            # Validate cron expression
            try:
                SimpleCronParser(schedule)
            except ValueError as e:
                raise ValueError(f"Invalid cron expression: {e}")

            triggers.append(
                RetrainingTrigger(
                    trigger_type=RetrainingTriggerType.SCHEDULE,
                    enabled=True,
                    schedule=schedule,
                )
            )

        if min_samples is not None:
            triggers.append(
                RetrainingTrigger(
                    trigger_type=RetrainingTriggerType.DATA_VOLUME,
                    enabled=True,
                    min_samples=min_samples,
                )
            )

        if performance_threshold is not None:
            triggers.append(
                RetrainingTrigger(
                    trigger_type=RetrainingTriggerType.PERFORMANCE,
                    enabled=True,
                    performance_threshold=performance_threshold,
                )
            )

        with self._lock:
            self._triggers[model_name] = triggers
            self._save_state()

        logger.info(f"Configured {len(triggers)} triggers for {model_name}")

    def record_new_data(self, model_name: str, count: int = 1) -> None:
        """
        Record new data samples for data volume trigger.

        Args:
            model_name: Name of the model.
            count: Number of new samples.
        """
        with self._lock:
            self._data_counters[model_name] = (
                self._data_counters.get(model_name, 0) + count
            )
            self._save_state()

    def check_retraining_needed(
        self,
        model_name: str,
        current_drift_score: Optional[float] = None,
        current_performance: Optional[Dict[str, float]] = None,
        baseline_performance: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        """
        Check if retraining is needed for a model.

        Args:
            model_name: Name of the model.
            current_drift_score: Current drift score (if available).
            current_performance: Current performance metrics.
            baseline_performance: Baseline performance metrics.

        Returns:
            Tuple of (needs_retraining: bool, reason: str).
        """
        if model_name not in self._triggers:
            return False, "No triggers configured for this model"

        triggers = self._triggers[model_name]

        # Check cooldown period
        if model_name in self._last_retraining:
            cooldown_hours = self.config.retraining.cooldown_period_hours
            cooldown_end = self._last_retraining[model_name] + timedelta(
                hours=cooldown_hours
            )
            if datetime.utcnow() < cooldown_end:
                remaining = (cooldown_end - datetime.utcnow()).total_seconds() / 3600
                return False, f"Cooldown period active ({remaining:.1f} hours remaining)"

        for trigger in triggers:
            if not trigger.enabled:
                continue

            if trigger.trigger_type == RetrainingTriggerType.DRIFT:
                if current_drift_score is not None:
                    if current_drift_score > trigger.drift_threshold:
                        return True, (
                            f"Drift threshold exceeded: {current_drift_score:.3f} > "
                            f"{trigger.drift_threshold}"
                        )

            elif trigger.trigger_type == RetrainingTriggerType.SCHEDULE:
                if trigger.schedule:
                    cron = SimpleCronParser(trigger.schedule)
                    now = datetime.utcnow()

                    # Check if we're within the scheduled window (5 minutes)
                    if cron.matches(now) or cron.matches(
                        now - timedelta(minutes=5)
                    ):
                        # Check if we already retrained in this window
                        if model_name in self._last_retraining:
                            last = self._last_retraining[model_name]
                            if (now - last).total_seconds() < 600:
                                continue

                        return True, f"Scheduled retraining: {trigger.schedule}"

            elif trigger.trigger_type == RetrainingTriggerType.DATA_VOLUME:
                new_samples = self._data_counters.get(model_name, 0)
                if new_samples >= trigger.min_samples:
                    return True, (
                        f"Data volume threshold reached: {new_samples} >= "
                        f"{trigger.min_samples}"
                    )

            elif trigger.trigger_type == RetrainingTriggerType.PERFORMANCE:
                if current_performance and baseline_performance:
                    # Check for significant degradation
                    for metric, current in current_performance.items():
                        baseline = baseline_performance.get(metric)
                        if baseline is not None:
                            # For error metrics (lower is better)
                            if metric in ["mae", "rmse", "mse", "mape"]:
                                degradation = (current - baseline) / max(baseline, 1e-10)
                                if degradation > trigger.performance_threshold:
                                    return True, (
                                        f"Performance degraded: {metric} increased by "
                                        f"{degradation*100:.1f}%"
                                    )
                            # For accuracy metrics (higher is better)
                            else:
                                degradation = (baseline - current) / max(baseline, 1e-10)
                                if degradation > trigger.performance_threshold:
                                    return True, (
                                        f"Performance degraded: {metric} decreased by "
                                        f"{degradation*100:.1f}%"
                                    )

        return False, "No trigger conditions met"

    def trigger_retraining(
        self,
        model_name: str,
        training_config: Dict[str, Any],
        training_data: Optional[np.ndarray] = None,
        training_labels: Optional[np.ndarray] = None,
        validation_data: Optional[np.ndarray] = None,
        validation_labels: Optional[np.ndarray] = None,
        trigger_reason: str = "Manual trigger",
    ) -> RetrainingResult:
        """
        Trigger model retraining.

        Args:
            model_name: Name of the model to retrain.
            training_config: Training configuration/hyperparameters.
            training_data: Training features (if not using registered function).
            training_labels: Training labels.
            validation_data: Validation features.
            validation_labels: Validation labels.
            trigger_reason: Reason for retraining.

        Returns:
            RetrainingResult with training outcome.

        Raises:
            ValueError: If no training function registered and no data provided.
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting retraining for {model_name}: {trigger_reason}")

        # Determine trigger type from reason
        trigger_type = RetrainingTriggerType.MANUAL
        if "drift" in trigger_reason.lower():
            trigger_type = RetrainingTriggerType.DRIFT
        elif "schedule" in trigger_reason.lower():
            trigger_type = RetrainingTriggerType.SCHEDULE
        elif "data volume" in trigger_reason.lower():
            trigger_type = RetrainingTriggerType.DATA_VOLUME
        elif "performance" in trigger_reason.lower():
            trigger_type = RetrainingTriggerType.PERFORMANCE

        try:
            # Get training function
            if model_name in self._training_functions:
                training_fn = self._training_functions[model_name]
                new_model, new_metrics = training_fn(training_config)
            elif training_data is not None and training_labels is not None:
                # Use provided data with default sklearn-style training
                new_model, new_metrics = self._default_training(
                    training_config, training_data, training_labels
                )
            else:
                raise ValueError(
                    f"No training function registered for {model_name} "
                    "and no training data provided"
                )

            training_completed = datetime.utcnow()
            training_duration = (
                training_completed - start_time
            ).total_seconds()

            # Get old model metrics (placeholder - would need model registry integration)
            old_model_metrics = training_config.get("baseline_metrics", {})

            # Calculate improvements
            improvements = {}
            for metric, new_value in new_metrics.items():
                old_value = old_model_metrics.get(metric)
                if old_value is not None:
                    if metric in ["mae", "rmse", "mse", "mape"]:
                        # For error metrics, improvement is reduction
                        improvements[metric] = (old_value - new_value) / max(
                            old_value, 1e-10
                        )
                    else:
                        # For accuracy metrics, improvement is increase
                        improvements[metric] = (new_value - old_value) / max(
                            old_value, 1e-10
                        )

            # Validate new model if validation data provided
            validation_passed = True
            if validation_data is not None and validation_labels is not None:
                validation_passed = self.validate_new_model(
                    old_model=None,  # Would need old model from registry
                    new_model=new_model,
                    validation_data=validation_data,
                    validation_labels=validation_labels,
                    min_improvement=self.config.retraining.default_min_improvement,
                    baseline_metrics=old_model_metrics,
                )

            # Determine if should deploy
            should_deploy = (
                validation_passed
                and self.config.retraining.default_validation_split > 0
            )

            # Generate version
            new_version = self._generate_version(model_name)
            old_version = training_config.get("current_version", "0.0.0")

            result = RetrainingResult(
                retraining_id=self._generate_retraining_id(),
                model_name=model_name,
                old_version=old_version,
                new_version=new_version,
                trigger_type=trigger_type,
                trigger_reason=trigger_reason,
                training_started_at=start_time,
                training_completed_at=training_completed,
                training_duration_seconds=training_duration,
                old_model_metrics=old_model_metrics,
                new_model_metrics=new_metrics,
                improvement=improvements,
                validation_passed=validation_passed,
                deployed=should_deploy and self.config.retraining.default_validation_split > 0,
                deployment_stage=ModelStage.STAGING if should_deploy else None,
            )

            # Update state
            with self._lock:
                self._last_retraining[model_name] = datetime.utcnow()
                if model_name in self._data_counters:
                    self._data_counters[model_name] = 0  # Reset counter

                if model_name not in self._retraining_history:
                    self._retraining_history[model_name] = []
                self._retraining_history[model_name].append(result)

                self._save_state()

            # Save result
            self._save_result(result)

            logger.info(
                f"Retraining completed for {model_name}: "
                f"validation_passed={validation_passed}, "
                f"duration={training_duration:.1f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Retraining failed for {model_name}: {e}", exc_info=True)
            raise

    def _default_training(
        self,
        config: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Default training implementation using sklearn-style API.

        Args:
            config: Training configuration.
            X: Training features.
            y: Training labels.

        Returns:
            Tuple of (trained_model, metrics).
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            # Split data
            val_split = config.get("validation_split", 0.2)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split, random_state=42
            )

            # Get hyperparameters
            n_estimators = config.get("n_estimators", 100)
            max_depth = config.get("max_depth", 5)
            learning_rate = config.get("learning_rate", 0.1)

            # Train model
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )
            model.fit(X_train, y_train)

            # Calculate metrics
            y_pred = model.predict(X_val)
            metrics = {
                "mae": float(mean_absolute_error(y_val, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
            }

            return model, metrics

        except ImportError:
            raise RuntimeError(
                "sklearn required for default training. "
                "Install with: pip install scikit-learn"
            )

    def validate_new_model(
        self,
        old_model: Optional[Any],
        new_model: Any,
        validation_data: np.ndarray,
        validation_labels: np.ndarray,
        min_improvement: float = 0.0,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Validate new model meets quality thresholds.

        Args:
            old_model: Previous model (optional).
            new_model: New trained model.
            validation_data: Validation features.
            validation_labels: Validation labels.
            min_improvement: Minimum required improvement.
            baseline_metrics: Baseline metrics to compare against.

        Returns:
            True if validation passes, False otherwise.
        """
        logger.info("Validating new model...")

        # Get predictions from new model
        if hasattr(new_model, "predict"):
            new_preds = new_model.predict(validation_data)
        else:
            raise ValueError("Model must have predict method")

        # Calculate new model metrics
        errors = np.abs(validation_labels - new_preds)
        new_mae = float(np.mean(errors))
        new_rmse = float(np.sqrt(np.mean((validation_labels - new_preds) ** 2)))

        logger.info(f"New model metrics: MAE={new_mae:.4f}, RMSE={new_rmse:.4f}")

        # Compare to baseline
        if baseline_metrics:
            baseline_mae = baseline_metrics.get("mae")
            if baseline_mae is not None:
                improvement = (baseline_mae - new_mae) / max(baseline_mae, 1e-10)
                if improvement < min_improvement:
                    logger.warning(
                        f"Validation failed: MAE improvement {improvement:.2%} < "
                        f"required {min_improvement:.2%}"
                    )
                    return False

        # Compare to old model if provided
        if old_model is not None:
            if hasattr(old_model, "predict"):
                old_preds = old_model.predict(validation_data)
                old_mae = float(np.mean(np.abs(validation_labels - old_preds)))

                if new_mae > old_mae * (1 + min_improvement):
                    logger.warning(
                        f"Validation failed: new MAE {new_mae:.4f} > "
                        f"old MAE {old_mae:.4f}"
                    )
                    return False

        logger.info("Model validation passed")
        return True

    def _generate_retraining_id(self) -> str:
        """Generate unique retraining identifier."""
        timestamp = datetime.utcnow().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]

    def _generate_version(self, model_name: str) -> str:
        """Generate next version number for model."""
        history = self._retraining_history.get(model_name, [])
        if not history:
            return "1.0.0"

        # Increment patch version
        last_version = history[-1].new_version
        parts = last_version.split(".")
        if len(parts) >= 3:
            parts[-1] = str(int(parts[-1]) + 1)
            return ".".join(parts)
        return f"{last_version}.1"

    def _save_result(self, result: RetrainingResult) -> None:
        """Save retraining result to storage."""
        result_path = (
            self.storage_path / f"retraining_{result.retraining_id}.json"
        )
        with open(result_path, "w") as f:
            f.write(result.json(indent=2))

    def get_retraining_history(
        self, model_name: str, limit: int = 10
    ) -> List[RetrainingResult]:
        """
        Get retraining history for a model.

        Args:
            model_name: Name of the model.
            limit: Maximum number of results.

        Returns:
            List of RetrainingResult objects.
        """
        history = self._retraining_history.get(model_name, [])
        return history[-limit:]

    def get_next_scheduled_retraining(
        self, model_name: str
    ) -> Optional[datetime]:
        """
        Get next scheduled retraining time for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Next scheduled datetime or None if no schedule configured.
        """
        triggers = self._triggers.get(model_name, [])

        for trigger in triggers:
            if trigger.trigger_type == RetrainingTriggerType.SCHEDULE:
                if trigger.schedule and trigger.enabled:
                    cron = SimpleCronParser(trigger.schedule)
                    return cron.next_run(datetime.utcnow())

        return None

    def disable_trigger(
        self, model_name: str, trigger_type: RetrainingTriggerType
    ) -> bool:
        """
        Disable a specific trigger for a model.

        Args:
            model_name: Name of the model.
            trigger_type: Type of trigger to disable.

        Returns:
            True if trigger was found and disabled.
        """
        if model_name not in self._triggers:
            return False

        with self._lock:
            for trigger in self._triggers[model_name]:
                if trigger.trigger_type == trigger_type:
                    trigger.enabled = False
                    self._save_state()
                    logger.info(f"Disabled {trigger_type} trigger for {model_name}")
                    return True

        return False

    def enable_trigger(
        self, model_name: str, trigger_type: RetrainingTriggerType
    ) -> bool:
        """
        Enable a specific trigger for a model.

        Args:
            model_name: Name of the model.
            trigger_type: Type of trigger to enable.

        Returns:
            True if trigger was found and enabled.
        """
        if model_name not in self._triggers:
            return False

        with self._lock:
            for trigger in self._triggers[model_name]:
                if trigger.trigger_type == trigger_type:
                    trigger.enabled = True
                    self._save_state()
                    logger.info(f"Enabled {trigger_type} trigger for {model_name}")
                    return True

        return False

    def get_trigger_status(self, model_name: str) -> Dict[str, Any]:
        """
        Get status of all triggers for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary with trigger status information.
        """
        triggers = self._triggers.get(model_name, [])

        return {
            "model_name": model_name,
            "triggers": [
                {
                    "type": t.trigger_type.value,
                    "enabled": t.enabled,
                    "threshold": t.drift_threshold or t.min_samples or t.performance_threshold,
                    "schedule": t.schedule,
                }
                for t in triggers
            ],
            "new_data_count": self._data_counters.get(model_name, 0),
            "last_retraining": (
                self._last_retraining[model_name].isoformat()
                if model_name in self._last_retraining
                else None
            ),
            "next_scheduled": (
                self.get_next_scheduled_retraining(model_name).isoformat()
                if self.get_next_scheduled_retraining(model_name)
                else None
            ),
        }
