# -*- coding: utf-8 -*-
"""
TASK-070: Graceful Degradation

This module provides graceful degradation capabilities for GreenLang Process
Heat ML models, including fallback model selection, ensemble voting under
partial failure, feature subset models, rule-based backup predictions, and
degradation status reporting.

Graceful degradation ensures system reliability when primary models fail or
inputs are corrupted, critical for safety-critical Process Heat applications.

Example:
    >>> from greenlang.ml.robustness import GracefulDegradationManager
    >>> manager = GracefulDegradationManager(
    ...     primary_model=main_model,
    ...     fallback_models=[simple_model, rule_based_model],
    ...     config=DegradationConfig()
    ... )
    >>> result = manager.predict_safe(X)
    >>> if result.degraded:
    ...     print(f"Using fallback: {result.model_used}")
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pydantic import BaseModel, Field
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
import copy

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class DegradationLevel(str, Enum):
    """Levels of degradation."""
    NONE = "none"  # Primary model functioning normally
    MINOR = "minor"  # Minor issues, using enhanced features
    PARTIAL = "partial"  # Some fallback engaged
    SIGNIFICANT = "significant"  # Major fallback required
    FULL = "full"  # Complete fallback to rules
    EMERGENCY = "emergency"  # Emergency conservative values


class FallbackReason(str, Enum):
    """Reasons for fallback activation."""
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    ANOMALOUS_INPUT = "anomalous_input"
    MISSING_FEATURES = "missing_features"
    CONFIDENCE_LOW = "confidence_low"
    DRIFT_DETECTED = "drift_detected"
    MANUAL_OVERRIDE = "manual_override"


class VotingStrategy(str, Enum):
    """Ensemble voting strategies."""
    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED = "weighted"
    MAJORITY = "majority"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


# =============================================================================
# Configuration
# =============================================================================

class DegradationConfig(BaseModel):
    """Configuration for graceful degradation."""

    # Fallback settings
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries before fallback"
    )
    timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Timeout for model predictions"
    )

    # Confidence settings
    min_confidence: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Minimum confidence for primary model"
    )
    confidence_threshold_fallback: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Confidence threshold to trigger fallback"
    )

    # Ensemble settings
    voting_strategy: VotingStrategy = Field(
        default=VotingStrategy.WEIGHTED,
        description="Voting strategy for ensemble"
    )
    require_minimum_models: int = Field(
        default=1,
        ge=1,
        description="Minimum models required for prediction"
    )

    # Feature subset settings
    enable_feature_subset: bool = Field(
        default=True,
        description="Enable feature subset fallback"
    )
    core_feature_indices: List[int] = Field(
        default_factory=list,
        description="Indices of core (required) features"
    )

    # Rule-based settings
    enable_rule_based: bool = Field(
        default=True,
        description="Enable rule-based backup"
    )
    conservative_mode: bool = Field(
        default=True,
        description="Use conservative values in emergency"
    )

    # Process Heat specific defaults
    default_efficiency: float = Field(
        default=75.0,
        description="Default efficiency % for emergency"
    )
    default_temperature: float = Field(
        default=200.0,
        description="Default temperature C for emergency"
    )

    # Monitoring
    log_degradation: bool = Field(
        default=True,
        description="Log all degradation events"
    )
    alert_on_degradation: bool = Field(
        default=True,
        description="Send alerts on degradation"
    )

    # Provenance
    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance"
    )


# =============================================================================
# Model Wrappers
# =============================================================================

class ModelWrapper(BaseModel):
    """Wrapper for a fallback model with metadata."""

    name: str = Field(..., description="Model name")
    priority: int = Field(
        default=0,
        description="Priority (lower = higher priority)"
    )
    model_type: str = Field(
        default="ml",
        description="Model type (ml, rule, simple)"
    )

    # Capabilities
    supports_confidence: bool = Field(
        default=False,
        description="Model provides confidence scores"
    )
    required_features: Optional[List[int]] = Field(
        default=None,
        description="Required feature indices"
    )

    # Performance
    expected_latency_ms: float = Field(
        default=100.0,
        description="Expected latency in ms"
    )
    reliability_score: float = Field(
        default=0.9,
        ge=0,
        le=1,
        description="Historical reliability (0-1)"
    )

    # Accuracy
    accuracy_score: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Expected accuracy relative to primary"
    )

    class Config:
        arbitrary_types_allowed = True


class RuleBasedModel:
    """
    Rule-based backup model for Process Heat predictions.

    Uses engineering rules and correlations when ML models fail.
    """

    def __init__(
        self,
        rules: Optional[Dict[str, Callable]] = None,
        default_values: Optional[Dict[str, float]] = None
    ):
        """
        Initialize rule-based model.

        Args:
            rules: Dict of output_name -> rule function(inputs) -> value
            default_values: Default values for each output
        """
        self.rules = rules or {}
        self.default_values = default_values or {
            "efficiency": 75.0,
            "temperature": 200.0,
            "heat_duty": 10.0,
            "emissions": 50.0
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make rule-based predictions."""
        n_samples = len(X)
        n_outputs = len(self.default_values)

        predictions = np.zeros((n_samples, n_outputs))

        for i, (output_name, default_val) in enumerate(self.default_values.items()):
            if output_name in self.rules:
                try:
                    for j in range(n_samples):
                        predictions[j, i] = self.rules[output_name](X[j])
                except Exception:
                    predictions[:, i] = default_val
            else:
                predictions[:, i] = default_val

        return predictions

    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Rule-based models have fixed low confidence."""
        return np.full(len(X), 0.5)


class FeatureSubsetModel:
    """
    Model trained on subset of features for robustness.

    Falls back when some features are missing or corrupted.
    """

    def __init__(
        self,
        base_model: Any,
        feature_indices: List[int],
        name: str = "feature_subset"
    ):
        """
        Initialize feature subset model.

        Args:
            base_model: Base model (already trained on subset)
            feature_indices: Indices of features this model uses
            name: Model name
        """
        self.model = base_model
        self.feature_indices = feature_indices
        self.name = name

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using feature subset."""
        X_subset = X[:, self.feature_indices]
        return self.model.predict(X_subset)

    def check_features(self, X: np.ndarray) -> bool:
        """Check if required features are available and valid."""
        for idx in self.feature_indices:
            if idx >= X.shape[1]:
                return False
            if np.any(np.isnan(X[:, idx])) or np.any(np.isinf(X[:, idx])):
                return False
        return True


# =============================================================================
# Result Models
# =============================================================================

class DegradationEvent(BaseModel):
    """Record of a degradation event."""

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    level: DegradationLevel = Field(..., description="Degradation level")
    reason: FallbackReason = Field(..., description="Reason for degradation")
    model_from: str = Field(..., description="Original model")
    model_to: str = Field(..., description="Fallback model")
    sample_index: Optional[int] = Field(
        default=None,
        description="Sample index if applicable"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message"
    )
    recovery_time_ms: Optional[float] = Field(
        default=None,
        description="Time to fallback in ms"
    )


class PredictionResult(BaseModel):
    """Result from safe prediction with degradation info."""

    # Predictions
    predictions: List[float] = Field(..., description="Predictions")
    confidence_scores: Optional[List[float]] = Field(
        default=None,
        description="Confidence scores"
    )

    # Degradation status
    degraded: bool = Field(..., description="Whether degradation occurred")
    degradation_level: DegradationLevel = Field(
        ...,
        description="Current degradation level"
    )
    model_used: str = Field(..., description="Model that made predictions")
    models_attempted: List[str] = Field(
        default_factory=list,
        description="Models attempted before success"
    )

    # Events
    degradation_events: List[DegradationEvent] = Field(
        default_factory=list,
        description="Degradation events"
    )

    # Timing
    total_latency_ms: float = Field(..., description="Total latency")
    retries: int = Field(default=0, description="Number of retries")

    # Reliability
    estimated_accuracy: float = Field(
        default=1.0,
        description="Estimated accuracy (relative to primary)"
    )
    reliability_score: float = Field(
        default=1.0,
        description="Overall reliability score"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash")


class DegradationStatus(BaseModel):
    """Overall degradation status report."""

    current_level: DegradationLevel = Field(
        ...,
        description="Current degradation level"
    )
    primary_model_healthy: bool = Field(
        ...,
        description="Primary model status"
    )
    available_models: List[str] = Field(
        ...,
        description="Available fallback models"
    )
    unavailable_models: List[str] = Field(
        default_factory=list,
        description="Unavailable models"
    )

    # Statistics
    total_predictions: int = Field(..., description="Total predictions")
    degraded_predictions: int = Field(
        ...,
        description="Predictions using fallback"
    )
    degradation_rate: float = Field(..., description="Degradation rate")

    # Recent events
    recent_events: List[DegradationEvent] = Field(
        default_factory=list,
        description="Recent degradation events"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Status timestamp"
    )


# =============================================================================
# Graceful Degradation Manager
# =============================================================================

class GracefulDegradationManager:
    """
    Graceful Degradation Manager for Process Heat ML Models.

    This manager ensures reliable predictions by:
    - Automatic fallback to backup models on failure
    - Ensemble voting under partial failure
    - Feature subset models for corrupted inputs
    - Rule-based backup predictions
    - Comprehensive degradation status reporting

    All calculations are deterministic for reproducibility.

    Attributes:
        primary_model: Primary ML model
        fallback_models: List of fallback models
        config: Degradation configuration

    Example:
        >>> manager = GracefulDegradationManager(
        ...     primary_model=main_model,
        ...     fallback_models=[simple_model],
        ...     config=DegradationConfig()
        ... )
        >>> result = manager.predict_safe(X)
        >>> if result.degraded:
        ...     logger.warning(f"Degraded to {result.model_used}")
    """

    def __init__(
        self,
        primary_model: Any,
        fallback_models: Optional[List[Any]] = None,
        model_metadata: Optional[List[ModelWrapper]] = None,
        config: Optional[DegradationConfig] = None,
        rule_model: Optional[RuleBasedModel] = None
    ):
        """
        Initialize graceful degradation manager.

        Args:
            primary_model: Primary prediction model
            fallback_models: List of fallback models (in priority order)
            model_metadata: Metadata for fallback models
            config: Degradation configuration
            rule_model: Rule-based backup model
        """
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.config = config or DegradationConfig()
        self.rule_model = rule_model or self._create_default_rule_model()

        # Model metadata
        self.model_metadata = model_metadata or [
            ModelWrapper(
                name=f"fallback_{i}",
                priority=i,
                accuracy_score=0.9 - 0.1 * i
            )
            for i in range(len(self.fallback_models))
        ]

        # Feature subset models
        self._feature_subset_models: List[FeatureSubsetModel] = []

        # State tracking
        self._total_predictions = 0
        self._degraded_predictions = 0
        self._degradation_events: List[DegradationEvent] = []
        self._model_health: Dict[str, bool] = {"primary": True}

        for i in range(len(self.fallback_models)):
            self._model_health[f"fallback_{i}"] = True

        logger.info(
            f"GracefulDegradationManager initialized: "
            f"fallbacks={len(self.fallback_models)}"
        )

    def _create_default_rule_model(self) -> RuleBasedModel:
        """Create default rule-based model for Process Heat."""
        rules = {
            # Simple efficiency rule based on load
            "efficiency": lambda x: min(95, max(60, 85 - 0.1 * abs(x[0] - 100))),
            # Temperature approximation
            "temperature": lambda x: 150 + 0.5 * x[0] if len(x) > 0 else 200,
        }

        default_values = {
            "efficiency": self.config.default_efficiency,
            "temperature": self.config.default_temperature,
            "heat_duty": 10.0,
            "emissions": 50.0
        }

        return RuleBasedModel(rules=rules, default_values=default_values)

    def add_feature_subset_model(
        self,
        model: Any,
        feature_indices: List[int],
        name: str = "feature_subset"
    ):
        """Add a feature subset model for fallback."""
        self._feature_subset_models.append(
            FeatureSubsetModel(model, feature_indices, name)
        )
        self._model_health[name] = True

    def _try_predict(
        self,
        model: Any,
        X: np.ndarray,
        model_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """
        Try to make prediction with a model.

        Returns:
            Tuple of (predictions, confidence, error_message)
        """
        try:
            predictions = model.predict(X)
            predictions = np.atleast_1d(predictions).flatten()

            # Get confidence if available
            confidence = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)
                    confidence = np.max(proba, axis=1)
                except Exception:
                    pass
            elif hasattr(model, "get_confidence"):
                confidence = model.get_confidence(X)

            # Validate predictions
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                return None, None, "Invalid predictions (NaN or Inf)"

            return predictions, confidence, None

        except Exception as e:
            return None, None, str(e)

    def _select_fallback(
        self,
        X: np.ndarray,
        failed_models: List[str]
    ) -> Tuple[Any, str, float]:
        """
        Select best available fallback model.

        Returns:
            Tuple of (model, model_name, accuracy_score)
        """
        # Try fallback models in priority order
        for i, (model, metadata) in enumerate(
            zip(self.fallback_models, self.model_metadata)
        ):
            model_name = metadata.name
            if model_name in failed_models:
                continue

            if not self._model_health.get(model_name, True):
                continue

            # Check feature requirements
            if metadata.required_features:
                if not all(
                    idx < X.shape[1] and not np.any(np.isnan(X[:, idx]))
                    for idx in metadata.required_features
                ):
                    continue

            return model, model_name, metadata.accuracy_score

        # Try feature subset models
        for subset_model in self._feature_subset_models:
            if subset_model.name in failed_models:
                continue

            if subset_model.check_features(X):
                return subset_model, subset_model.name, 0.7

        # Fall back to rule-based
        if self.config.enable_rule_based:
            return self.rule_model, "rule_based", 0.5

        raise RuntimeError("No available fallback models")

    def _ensemble_predict(
        self,
        predictions: List[np.ndarray],
        confidences: List[Optional[np.ndarray]],
        model_weights: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine predictions using ensemble voting.

        Returns:
            Tuple of (ensemble predictions, ensemble confidence)
        """
        n_samples = len(predictions[0])
        n_models = len(predictions)

        # Stack predictions
        pred_matrix = np.column_stack(predictions)

        if self.config.voting_strategy == VotingStrategy.MEAN:
            ensemble_pred = np.mean(pred_matrix, axis=1)

        elif self.config.voting_strategy == VotingStrategy.MEDIAN:
            ensemble_pred = np.median(pred_matrix, axis=1)

        elif self.config.voting_strategy == VotingStrategy.WEIGHTED:
            weights = np.array(model_weights) / sum(model_weights)
            ensemble_pred = np.sum(pred_matrix * weights, axis=1)

        elif self.config.voting_strategy == VotingStrategy.CONFIDENCE_WEIGHTED:
            # Use confidence as weights
            conf_matrix = np.column_stack([
                c if c is not None else np.ones(n_samples) * 0.5
                for c in confidences
            ])
            weights = conf_matrix / (np.sum(conf_matrix, axis=1, keepdims=True) + 1e-10)
            ensemble_pred = np.sum(pred_matrix * weights, axis=1)

        else:  # MAJORITY - for classification
            ensemble_pred = np.mean(pred_matrix, axis=1)

        # Ensemble confidence
        if any(c is not None for c in confidences):
            valid_confs = [c for c in confidences if c is not None]
            ensemble_conf = np.mean(np.column_stack(valid_confs), axis=1)
        else:
            # Estimate confidence from prediction agreement
            pred_std = np.std(pred_matrix, axis=1)
            pred_range = np.ptp(pred_matrix, axis=1)
            ensemble_conf = 1.0 / (1.0 + pred_std / (np.mean(np.abs(predictions[0])) + 1e-10))

        return ensemble_pred, ensemble_conf

    def predict_safe(
        self,
        X: np.ndarray,
        return_all: bool = False
    ) -> PredictionResult:
        """
        Make safe predictions with automatic fallback.

        Args:
            X: Input features
            return_all: Return predictions from all available models

        Returns:
            PredictionResult with predictions and degradation info

        Example:
            >>> result = manager.predict_safe(X)
            >>> if result.degraded:
            ...     logger.warning(f"Using fallback: {result.model_used}")
        """
        import time
        start_time = time.time()

        X = np.atleast_2d(X)
        n_samples = len(X)

        models_attempted = []
        degradation_events = []
        failed_models = []

        predictions = None
        confidence = None
        model_used = "primary"
        accuracy_score = 1.0
        retries = 0

        # Try primary model
        for attempt in range(self.config.max_retries + 1):
            preds, conf, error = self._try_predict(
                self.primary_model, X, "primary"
            )

            if preds is not None:
                # Check confidence threshold
                if conf is not None and np.mean(conf) < self.config.confidence_threshold_fallback:
                    degradation_events.append(DegradationEvent(
                        level=DegradationLevel.MINOR,
                        reason=FallbackReason.CONFIDENCE_LOW,
                        model_from="primary",
                        model_to="primary",
                        error_message=f"Low confidence: {np.mean(conf):.2f}"
                    ))

                predictions = preds
                confidence = conf
                break

            retries = attempt
            models_attempted.append("primary")

            if error:
                degradation_events.append(DegradationEvent(
                    level=DegradationLevel.PARTIAL,
                    reason=FallbackReason.MODEL_ERROR,
                    model_from="primary",
                    model_to="unknown",
                    error_message=error
                ))

        # If primary failed, try fallbacks
        if predictions is None:
            failed_models.append("primary")
            self._model_health["primary"] = False

            # Try fallback models
            while predictions is None:
                try:
                    fallback, fallback_name, acc = self._select_fallback(X, failed_models)
                except RuntimeError:
                    # No fallbacks available - use emergency values
                    predictions = self._emergency_predictions(n_samples)
                    confidence = np.full(n_samples, 0.1)
                    model_used = "emergency"
                    accuracy_score = 0.3

                    degradation_events.append(DegradationEvent(
                        level=DegradationLevel.EMERGENCY,
                        reason=FallbackReason.MODEL_ERROR,
                        model_from=failed_models[-1] if failed_models else "primary",
                        model_to="emergency",
                        error_message="All models failed, using emergency values"
                    ))
                    break

                preds, conf, error = self._try_predict(fallback, X, fallback_name)

                if preds is not None:
                    predictions = preds
                    confidence = conf
                    model_used = fallback_name
                    accuracy_score = acc

                    degradation_events.append(DegradationEvent(
                        level=DegradationLevel.PARTIAL if acc > 0.7 else DegradationLevel.SIGNIFICANT,
                        reason=FallbackReason.MODEL_ERROR,
                        model_from=failed_models[-1] if failed_models else "primary",
                        model_to=fallback_name
                    ))
                    break
                else:
                    failed_models.append(fallback_name)
                    models_attempted.append(fallback_name)
                    self._model_health[fallback_name] = False

        # Determine degradation level
        if model_used == "primary":
            degradation_level = DegradationLevel.NONE
            degraded = False
        elif model_used == "emergency":
            degradation_level = DegradationLevel.EMERGENCY
            degraded = True
        elif model_used == "rule_based":
            degradation_level = DegradationLevel.FULL
            degraded = True
        elif accuracy_score < 0.7:
            degradation_level = DegradationLevel.SIGNIFICANT
            degraded = True
        else:
            degradation_level = DegradationLevel.PARTIAL
            degraded = True

        # Update statistics
        self._total_predictions += n_samples
        if degraded:
            self._degraded_predictions += n_samples
        self._degradation_events.extend(degradation_events)

        # Calculate latency
        total_latency_ms = (time.time() - start_time) * 1000

        # Log degradation if configured
        if self.config.log_degradation and degraded:
            logger.warning(
                f"Prediction degraded to {model_used}, "
                f"accuracy={accuracy_score:.2f}, "
                f"latency={total_latency_ms:.1f}ms"
            )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            predictions, model_used, len(degradation_events)
        )

        # Calculate reliability score
        reliability_score = accuracy_score * (1.0 - len(failed_models) * 0.1)
        reliability_score = max(0.1, reliability_score)

        return PredictionResult(
            predictions=predictions.tolist(),
            confidence_scores=confidence.tolist() if confidence is not None else None,
            degraded=degraded,
            degradation_level=degradation_level,
            model_used=model_used,
            models_attempted=models_attempted,
            degradation_events=degradation_events,
            total_latency_ms=total_latency_ms,
            retries=retries,
            estimated_accuracy=accuracy_score,
            reliability_score=reliability_score,
            provenance_hash=provenance_hash
        )

    def _emergency_predictions(self, n_samples: int) -> np.ndarray:
        """Generate emergency conservative predictions."""
        if self.config.conservative_mode:
            # Use conservative (safe) default values
            return np.full(n_samples, self.config.default_efficiency)
        else:
            return np.zeros(n_samples)

    def get_status(self) -> DegradationStatus:
        """
        Get current degradation status.

        Returns:
            Comprehensive status report
        """
        available_models = [
            name for name, healthy in self._model_health.items()
            if healthy
        ]
        unavailable_models = [
            name for name, healthy in self._model_health.items()
            if not healthy
        ]

        # Determine current level
        if self._model_health.get("primary", False):
            current_level = DegradationLevel.NONE
        elif len(available_models) > len(unavailable_models):
            current_level = DegradationLevel.PARTIAL
        elif "rule_based" in [m.name for m in self._feature_subset_models]:
            current_level = DegradationLevel.FULL
        else:
            current_level = DegradationLevel.SIGNIFICANT

        degradation_rate = (
            self._degraded_predictions / self._total_predictions
            if self._total_predictions > 0 else 0.0
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_level, degradation_rate, unavailable_models
        )

        return DegradationStatus(
            current_level=current_level,
            primary_model_healthy=self._model_health.get("primary", False),
            available_models=available_models,
            unavailable_models=unavailable_models,
            total_predictions=self._total_predictions,
            degraded_predictions=self._degraded_predictions,
            degradation_rate=degradation_rate,
            recent_events=self._degradation_events[-10:],
            recommendations=recommendations,
            timestamp=datetime.utcnow()
        )

    def _generate_recommendations(
        self,
        current_level: DegradationLevel,
        degradation_rate: float,
        unavailable_models: List[str]
    ) -> List[str]:
        """Generate recommendations based on status."""
        recommendations = []

        if current_level == DegradationLevel.EMERGENCY:
            recommendations.append(
                "CRITICAL: System in emergency mode. Immediate investigation required."
            )

        if "primary" in unavailable_models:
            recommendations.append(
                "Primary model unhealthy. Investigate model service and restart if needed."
            )

        if degradation_rate > 0.1:
            recommendations.append(
                f"High degradation rate ({degradation_rate:.1%}). "
                "Consider model retraining or data quality review."
            )

        if len(unavailable_models) > len(self.fallback_models) / 2:
            recommendations.append(
                "Multiple fallback models unavailable. "
                "System resilience is compromised."
            )

        if not recommendations:
            recommendations.append("System operating normally.")

        return recommendations

    def reset_health(self, model_name: Optional[str] = None):
        """Reset model health status."""
        if model_name:
            self._model_health[model_name] = True
        else:
            for name in self._model_health:
                self._model_health[name] = True

    def _calculate_provenance(
        self,
        predictions: np.ndarray,
        model_used: str,
        n_events: int
    ) -> str:
        """Calculate SHA-256 provenance hash (deterministic)."""
        pred_hash = hashlib.sha256(predictions.tobytes()).hexdigest()[:16]
        provenance_data = f"{pred_hash}|{model_used}|{n_events}"
        return hashlib.sha256(provenance_data.encode()).hexdigest()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_process_heat_degradation_manager(
    primary_model: Any,
    fallback_models: Optional[List[Any]] = None
) -> GracefulDegradationManager:
    """
    Create a GracefulDegradationManager configured for Process Heat.

    Args:
        primary_model: Primary prediction model
        fallback_models: Optional fallback models

    Returns:
        Configured degradation manager
    """
    config = DegradationConfig(
        max_retries=2,
        timeout_seconds=3.0,
        min_confidence=0.7,
        confidence_threshold_fallback=0.5,
        voting_strategy=VotingStrategy.WEIGHTED,
        enable_rule_based=True,
        conservative_mode=True,
        default_efficiency=75.0,
        default_temperature=200.0
    )

    # Create rule-based model with Process Heat rules
    rule_model = RuleBasedModel(
        rules={
            "efficiency": lambda x: np.clip(85 - 0.1 * abs(x[0] - 100), 60, 95),
            "temperature": lambda x: np.clip(150 + 0.5 * x[0], 100, 500),
        },
        default_values={
            "efficiency": 75.0,
            "temperature": 200.0,
            "heat_duty": 10.0,
            "emissions": 50.0
        }
    )

    return GracefulDegradationManager(
        primary_model=primary_model,
        fallback_models=fallback_models or [],
        config=config,
        rule_model=rule_model
    )


# =============================================================================
# Unit Tests
# =============================================================================

class TestGracefulDegradationManager:
    """Unit tests for GracefulDegradationManager."""

    def test_primary_model_success(self):
        """Test successful prediction with primary model."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        manager = GracefulDegradationManager(primary_model=MockModel())

        X = np.random.randn(10, 3)
        result = manager.predict_safe(X)

        assert not result.degraded
        assert result.model_used == "primary"
        assert result.degradation_level == DegradationLevel.NONE
        assert len(result.predictions) == 10

    def test_fallback_on_primary_failure(self):
        """Test fallback when primary model fails."""
        class FailingModel:
            def predict(self, X):
                raise RuntimeError("Model failed")

        class BackupModel:
            def predict(self, X):
                return np.ones(len(X))

        config = DegradationConfig(max_retries=0)
        manager = GracefulDegradationManager(
            primary_model=FailingModel(),
            fallback_models=[BackupModel()],
            model_metadata=[ModelWrapper(name="backup", priority=0, accuracy_score=0.8)],
            config=config
        )

        X = np.random.randn(5, 3)
        result = manager.predict_safe(X)

        assert result.degraded
        assert result.model_used == "backup"
        assert len(result.degradation_events) > 0

    def test_rule_based_fallback(self):
        """Test fallback to rule-based model."""
        class FailingModel:
            def predict(self, X):
                raise RuntimeError("Failed")

        config = DegradationConfig(
            max_retries=0,
            enable_rule_based=True
        )
        manager = GracefulDegradationManager(
            primary_model=FailingModel(),
            config=config
        )

        X = np.array([[100.0, 50.0, 25.0]])
        result = manager.predict_safe(X)

        assert result.degraded
        assert result.model_used == "rule_based"
        assert len(result.predictions) == 1

    def test_status_reporting(self):
        """Test status reporting."""
        class MockModel:
            def predict(self, X):
                return np.ones(len(X))

        manager = GracefulDegradationManager(primary_model=MockModel())

        # Make some predictions
        X = np.random.randn(10, 3)
        manager.predict_safe(X)

        status = manager.get_status()

        assert status.total_predictions == 10
        assert status.primary_model_healthy
        assert "primary" in status.available_models

    def test_emergency_mode(self):
        """Test emergency mode when all models fail."""
        class FailingModel:
            def predict(self, X):
                raise RuntimeError("Failed")

        config = DegradationConfig(
            max_retries=0,
            enable_rule_based=False  # Disable rule-based
        )

        # Override rule model to also fail
        class FailingRuleModel:
            def predict(self, X):
                raise RuntimeError("Rules failed")

        manager = GracefulDegradationManager(
            primary_model=FailingModel(),
            config=config,
            rule_model=FailingRuleModel()
        )
        # Re-enable rule-based to use the failing one
        manager.config.enable_rule_based = True

        X = np.random.randn(5, 3)
        result = manager.predict_safe(X)

        # Should fall through to emergency
        assert result.degraded
        assert result.degradation_level in [
            DegradationLevel.SIGNIFICANT,
            DegradationLevel.FULL,
            DegradationLevel.EMERGENCY
        ]

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        class MockModel:
            def predict(self, X):
                return np.ones(len(X))

        manager = GracefulDegradationManager(primary_model=MockModel())

        predictions = np.array([1.0, 2.0, 3.0])
        hash1 = manager._calculate_provenance(predictions, "primary", 0)
        hash2 = manager._calculate_provenance(predictions, "primary", 0)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_confidence_threshold(self):
        """Test confidence threshold triggers fallback warning."""
        class LowConfidenceModel:
            def predict(self, X):
                return np.ones(len(X))

            def get_confidence(self, X):
                return np.full(len(X), 0.3)  # Low confidence

        config = DegradationConfig(
            confidence_threshold_fallback=0.5
        )
        manager = GracefulDegradationManager(
            primary_model=LowConfidenceModel(),
            config=config
        )

        X = np.random.randn(5, 3)
        result = manager.predict_safe(X)

        # Should still use primary but with degradation event
        assert len(result.degradation_events) > 0
