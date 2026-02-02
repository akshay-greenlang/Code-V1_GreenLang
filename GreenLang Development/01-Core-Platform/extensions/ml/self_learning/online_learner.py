# -*- coding: utf-8 -*-
"""
Online Learner Module

This module provides online/incremental learning capabilities using River
(formerly creme) and scikit-multiflow, enabling models to learn from
streaming data without retraining from scratch.

Online learning is critical for GreenLang agents that need to adapt to
changing emission factors, new regulations, and evolving data patterns
while maintaining provenance and determinism.

Example:
    >>> from greenlang.ml.self_learning import OnlineLearner
    >>> learner = OnlineLearner(model_type="hoeffding_tree")
    >>> for X, y in emission_data_stream:
    ...     prediction = learner.predict_one(X)
    ...     learner.learn_one(X, y)
"""

from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class OnlineModelType(str, Enum):
    """Supported online learning model types."""
    HOEFFDING_TREE = "hoeffding_tree"
    HOEFFDING_ADAPTIVE_TREE = "hoeffding_adaptive_tree"
    ADAPTIVE_RANDOM_FOREST = "adaptive_random_forest"
    PASSIVE_AGGRESSIVE = "passive_aggressive"
    PERCEPTRON = "perceptron"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    ARIMA = "arima"


class OnlineLearnerConfig(BaseModel):
    """Configuration for online learner."""

    model_type: OnlineModelType = Field(
        default=OnlineModelType.HOEFFDING_ADAPTIVE_TREE,
        description="Type of online learning model"
    )
    learning_rate: float = Field(
        default=0.01,
        gt=0,
        le=1.0,
        description="Learning rate for gradient-based models"
    )
    grace_period: int = Field(
        default=200,
        ge=10,
        description="Grace period before tree splits"
    )
    split_confidence: float = Field(
        default=0.0000001,
        gt=0,
        description="Confidence for splitting (Hoeffding bound)"
    )
    max_depth: Optional[int] = Field(
        default=None,
        description="Maximum tree depth"
    )
    n_estimators: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of estimators for ensemble"
    )
    window_size: int = Field(
        default=1000,
        ge=100,
        description="Window size for drift detection"
    )
    enable_drift_detection: bool = Field(
        default=True,
        description="Enable concept drift detection"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class OnlineLearnerMetrics(BaseModel):
    """Metrics for online learner."""

    samples_seen: int = Field(
        default=0,
        description="Total samples processed"
    )
    accuracy: float = Field(
        default=0.0,
        description="Running accuracy"
    )
    mae: float = Field(
        default=0.0,
        description="Mean absolute error (regression)"
    )
    rmse: float = Field(
        default=0.0,
        description="Root mean squared error"
    )
    drift_detected_count: int = Field(
        default=0,
        description="Number of drifts detected"
    )
    last_drift_at: Optional[int] = Field(
        default=None,
        description="Sample index of last drift"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Total processing time"
    )


class OnlineLearnerResult(BaseModel):
    """Result from online learning prediction."""

    prediction: Union[float, int, str] = Field(
        ...,
        description="Model prediction"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Prediction confidence"
    )
    drift_warning: bool = Field(
        default=False,
        description="Whether drift warning is active"
    )
    drift_detected: bool = Field(
        default=False,
        description="Whether drift was detected"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp"
    )


class OnlineLearner:
    """
    Online Learner for GreenLang agents.

    This class provides online/incremental learning capabilities using
    River library, enabling models to continuously learn from streaming
    data while maintaining prediction quality and detecting concept drift.

    Key capabilities:
    - Incremental learning from single samples
    - Concept drift detection and adaptation
    - Multiple model types (trees, ensembles, linear)
    - Provenance tracking for audit trails
    - Performance monitoring

    Attributes:
        config: Learner configuration
        model: Internal River model
        metrics: Running performance metrics
        _drift_detector: Concept drift detector
        _performance_window: Sliding window of predictions

    Example:
        >>> learner = OnlineLearner(
        ...     config=OnlineLearnerConfig(
        ...         model_type=OnlineModelType.ADAPTIVE_RANDOM_FOREST,
        ...         enable_drift_detection=True
        ...     )
        ... )
        >>> for x, y in emission_stream:
        ...     pred = learner.predict_one(x)
        ...     learner.learn_one(x, y)
        ...     if learner.drift_detected:
        ...         print("Drift detected!")
    """

    def __init__(self, config: Optional[OnlineLearnerConfig] = None):
        """
        Initialize online learner.

        Args:
            config: Learner configuration
        """
        self.config = config or OnlineLearnerConfig()
        self.model = None
        self.metrics = OnlineLearnerMetrics()
        self._drift_detector = None
        self._performance_window: deque = deque(maxlen=self.config.window_size)
        self._prediction_window: deque = deque(maxlen=self.config.window_size)
        self._initialized = False

        logger.info(
            f"OnlineLearner initialized with model_type={self.config.model_type}"
        )

    def _initialize_model(self) -> None:
        """Initialize the online learning model."""
        try:
            from river import tree, ensemble, linear_model, drift
        except ImportError:
            raise ImportError(
                "River is required. Install with: pip install river"
            )

        if self.config.model_type == OnlineModelType.HOEFFDING_TREE:
            self.model = tree.HoeffdingTreeClassifier(
                grace_period=self.config.grace_period,
                split_confidence=self.config.split_confidence,
                max_depth=self.config.max_depth
            )

        elif self.config.model_type == OnlineModelType.HOEFFDING_ADAPTIVE_TREE:
            self.model = tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=self.config.grace_period,
                split_confidence=self.config.split_confidence,
                max_depth=self.config.max_depth
            )

        elif self.config.model_type == OnlineModelType.ADAPTIVE_RANDOM_FOREST:
            self.model = ensemble.AdaptiveRandomForestClassifier(
                n_models=self.config.n_estimators,
                seed=self.config.random_state
            )

        elif self.config.model_type == OnlineModelType.PASSIVE_AGGRESSIVE:
            self.model = linear_model.PAClassifier()

        elif self.config.model_type == OnlineModelType.PERCEPTRON:
            self.model = linear_model.Perceptron()

        elif self.config.model_type == OnlineModelType.LOGISTIC_REGRESSION:
            self.model = linear_model.LogisticRegression(
                optimizer=linear_model.optim.SGD(self.config.learning_rate)
            )

        elif self.config.model_type == OnlineModelType.LINEAR_REGRESSION:
            self.model = linear_model.LinearRegression(
                optimizer=linear_model.optim.SGD(self.config.learning_rate)
            )

        else:
            # Default to Hoeffding Adaptive Tree
            self.model = tree.HoeffdingAdaptiveTreeClassifier()

        # Initialize drift detector
        if self.config.enable_drift_detection:
            self._drift_detector = drift.ADWIN()

        self._initialized = True
        logger.info(f"Model initialized: {type(self.model).__name__}")

    def _calculate_provenance(
        self,
        x: Dict[str, Any],
        prediction: Any
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        input_str = str(sorted(x.items()))
        combined = f"{input_str}|{prediction}|{self.metrics.samples_seen}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def predict_one(
        self,
        x: Dict[str, Any]
    ) -> OnlineLearnerResult:
        """
        Make prediction for a single sample.

        Args:
            x: Feature dictionary

        Returns:
            OnlineLearnerResult with prediction

        Example:
            >>> result = learner.predict_one({"fuel_type": "diesel", "quantity": 100})
            >>> print(f"Prediction: {result.prediction}")
        """
        start_time = datetime.utcnow()

        if not self._initialized:
            self._initialize_model()

        # Make prediction
        if hasattr(self.model, "predict_proba_one"):
            proba = self.model.predict_proba_one(x)
            if proba:
                prediction = max(proba, key=proba.get)
                confidence = proba.get(prediction, 0.0)
            else:
                prediction = 0
                confidence = 0.0
        elif hasattr(self.model, "predict_one"):
            prediction = self.model.predict_one(x)
            confidence = None
        else:
            prediction = 0
            confidence = None

        # Check for drift
        drift_warning = False
        drift_detected = False

        if self._drift_detector is not None and len(self._prediction_window) > 0:
            # Use prediction error as drift signal
            drift_detected = self._drift_detector.drift_detected

        # Calculate provenance
        provenance_hash = self._calculate_provenance(x, prediction)

        # Update processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.metrics.processing_time_ms += processing_time

        return OnlineLearnerResult(
            prediction=prediction if prediction is not None else 0,
            confidence=confidence,
            drift_warning=drift_warning,
            drift_detected=drift_detected,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )

    def learn_one(
        self,
        x: Dict[str, Any],
        y: Union[int, float, str]
    ) -> None:
        """
        Update model with a single sample.

        Args:
            x: Feature dictionary
            y: Target value

        Example:
            >>> learner.learn_one({"fuel_type": "diesel", "quantity": 100}, 150.5)
        """
        if not self._initialized:
            self._initialize_model()

        # Get prediction before learning (for drift detection)
        if hasattr(self.model, "predict_one"):
            pred = self.model.predict_one(x)
            if pred is not None:
                error = 0 if pred == y else 1  # For classification
                if isinstance(y, (int, float)) and isinstance(pred, (int, float)):
                    error = abs(y - pred)  # For regression

                self._prediction_window.append(error)

                # Update drift detector
                if self._drift_detector is not None:
                    self._drift_detector.update(error)
                    if self._drift_detector.drift_detected:
                        self.metrics.drift_detected_count += 1
                        self.metrics.last_drift_at = self.metrics.samples_seen
                        logger.warning(
                            f"Concept drift detected at sample {self.metrics.samples_seen}"
                        )

        # Learn from sample
        self.model.learn_one(x, y)

        # Update metrics
        self.metrics.samples_seen += 1

        # Update running accuracy/error
        if len(self._prediction_window) > 0:
            if isinstance(y, (int, float)):
                errors = list(self._prediction_window)
                self.metrics.mae = float(np.mean(errors))
                self.metrics.rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
            else:
                # Classification accuracy
                correct = sum(1 for e in self._prediction_window if e == 0)
                self.metrics.accuracy = correct / len(self._prediction_window)

    def learn_many(
        self,
        X: List[Dict[str, Any]],
        y: List[Union[int, float, str]]
    ) -> None:
        """
        Update model with multiple samples.

        Args:
            X: List of feature dictionaries
            y: List of target values
        """
        for x, target in zip(X, y):
            self.learn_one(x, target)

    def predict_many(
        self,
        X: List[Dict[str, Any]]
    ) -> List[OnlineLearnerResult]:
        """
        Make predictions for multiple samples.

        Args:
            X: List of feature dictionaries

        Returns:
            List of OnlineLearnerResult
        """
        return [self.predict_one(x) for x in X]

    def get_metrics(self) -> OnlineLearnerMetrics:
        """Get current performance metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = OnlineLearnerMetrics()
        self._performance_window.clear()
        self._prediction_window.clear()

    def reset_model(self) -> None:
        """Reset the model to initial state."""
        self._initialized = False
        self.model = None
        self._drift_detector = None
        self.reset_metrics()
        logger.info("Model reset to initial state")

    @property
    def drift_detected(self) -> bool:
        """Check if drift is currently detected."""
        if self._drift_detector is not None:
            return self._drift_detector.drift_detected
        return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self._initialized:
            return {"initialized": False}

        info = {
            "initialized": True,
            "model_type": self.config.model_type.value,
            "model_class": type(self.model).__name__,
            "samples_seen": self.metrics.samples_seen,
            "drift_count": self.metrics.drift_detected_count
        }

        # Add tree-specific info
        if hasattr(self.model, "n_nodes"):
            info["n_nodes"] = self.model.n_nodes

        if hasattr(self.model, "height"):
            info["height"] = self.model.height

        return info

    def export_model(self) -> bytes:
        """Export model to bytes for persistence."""
        import pickle
        return pickle.dumps({
            "model": self.model,
            "config": self.config.dict(),
            "metrics": self.metrics.dict()
        })

    def import_model(self, data: bytes) -> None:
        """Import model from bytes."""
        import pickle
        loaded = pickle.loads(data)
        self.model = loaded["model"]
        self.config = OnlineLearnerConfig(**loaded["config"])
        self.metrics = OnlineLearnerMetrics(**loaded["metrics"])
        self._initialized = True


# Unit test stubs
class TestOnlineLearner:
    """Unit tests for OnlineLearner."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        learner = OnlineLearner()
        assert learner.config.model_type == OnlineModelType.HOEFFDING_ADAPTIVE_TREE
        assert not learner._initialized

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = OnlineLearnerConfig(
            model_type=OnlineModelType.ADAPTIVE_RANDOM_FOREST,
            n_estimators=20
        )
        learner = OnlineLearner(config)
        assert learner.config.model_type == OnlineModelType.ADAPTIVE_RANDOM_FOREST

    def test_predict_initializes_model(self):
        """Test that prediction initializes model."""
        learner = OnlineLearner()
        assert not learner._initialized

        # Mock prediction
        try:
            result = learner.predict_one({"x": 1.0})
            assert learner._initialized
        except ImportError:
            pass  # River not installed

    def test_metrics_tracking(self):
        """Test metrics are tracked correctly."""
        learner = OnlineLearner()
        assert learner.metrics.samples_seen == 0

        try:
            learner.learn_one({"x": 1.0}, 1)
            assert learner.metrics.samples_seen == 1
        except ImportError:
            pass  # River not installed

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        learner = OnlineLearner()

        x = {"feature_a": 1.0, "feature_b": 2.0}
        hash1 = learner._calculate_provenance(x, 1)
        hash2 = learner._calculate_provenance(x, 1)

        assert hash1 == hash2

    def test_reset_model(self):
        """Test model reset."""
        learner = OnlineLearner()
        learner.metrics.samples_seen = 100
        learner._initialized = True

        learner.reset_model()

        assert not learner._initialized
        assert learner.metrics.samples_seen == 0
