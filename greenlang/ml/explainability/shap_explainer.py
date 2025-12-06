# -*- coding: utf-8 -*-
"""
SHAP Explainer Module

This module provides SHAP (SHapley Additive exPlanations) integration for
GreenLang ML models, enabling feature importance analysis and model interpretability.

SHAP values represent the contribution of each feature to the prediction,
providing a principled approach to explaining ML model decisions with
provenance tracking for audit trails.

Example:
    >>> from greenlang.ml.explainability import SHAPExplainer
    >>> explainer = SHAPExplainer(model, explainer_type="tree")
    >>> result = explainer.explain(X_sample)
    >>> print(result.feature_importance)
"""

from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ExplainerType(str, Enum):
    """Supported SHAP explainer types."""
    TREE = "tree"
    KERNEL = "kernel"
    LINEAR = "linear"
    DEEP = "deep"
    GRADIENT = "gradient"
    PARTITION = "partition"


class SHAPExplainerConfig(BaseModel):
    """Configuration for SHAP explainer."""

    explainer_type: ExplainerType = Field(
        default=ExplainerType.KERNEL,
        description="Type of SHAP explainer to use"
    )
    n_samples: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Number of samples for KernelSHAP"
    )
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Names of features for explanations"
    )
    background_samples: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of background samples for reference"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking for audit trails"
    )
    max_display_features: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of features to display"
    )

    @validator("feature_names")
    def validate_feature_names(cls, v):
        """Validate feature names are unique."""
        if v is not None and len(v) != len(set(v)):
            raise ValueError("Feature names must be unique")
        return v


class SHAPResult(BaseModel):
    """Result from SHAP explanation."""

    shap_values: List[List[float]] = Field(
        ...,
        description="SHAP values matrix (samples x features)"
    )
    base_value: float = Field(
        ...,
        description="Base/expected value of the model"
    )
    feature_importance: Dict[str, float] = Field(
        ...,
        description="Mean absolute SHAP values per feature"
    )
    feature_names: List[str] = Field(
        ...,
        description="Names of features"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        ...,
        description="Processing duration in milliseconds"
    )
    explainer_type: str = Field(
        ...,
        description="Type of explainer used"
    )
    n_samples: int = Field(
        ...,
        description="Number of samples explained"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of explanation generation"
    )


class SHAPExplainer:
    """
    SHAP Explainer for GreenLang ML models.

    This class provides SHAP-based explanations for ML model predictions,
    integrating with GreenLang's zero-hallucination principles by providing
    mathematically grounded feature importance values.

    SHAP values are computed using game-theoretic principles (Shapley values),
    ensuring fair attribution of feature contributions to predictions.

    Attributes:
        model: The ML model to explain (must have predict or predict_proba method)
        config: Configuration for the explainer
        _explainer: Internal SHAP explainer instance
        _background_data: Background data for reference

    Example:
        >>> model = train_emission_model(X_train, y_train)
        >>> explainer = SHAPExplainer(model, config=SHAPExplainerConfig(
        ...     explainer_type=ExplainerType.TREE,
        ...     feature_names=["fuel_type", "quantity", "region"]
        ... ))
        >>> result = explainer.explain(X_test[:10])
        >>> print(f"Top feature: {result.feature_importance}")
    """

    def __init__(
        self,
        model: Any,
        config: Optional[SHAPExplainerConfig] = None,
        background_data: Optional[np.ndarray] = None
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: ML model with predict/predict_proba method
            config: Explainer configuration
            background_data: Background data for KernelSHAP reference
        """
        self.model = model
        self.config = config or SHAPExplainerConfig()
        self._background_data = background_data
        self._explainer = None
        self._initialized = False

        logger.info(
            f"SHAPExplainer initialized with type={self.config.explainer_type}"
        )

    def _get_prediction_function(self) -> Callable:
        """
        Get the prediction function from the model.

        Returns:
            Callable prediction function
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba
        elif hasattr(self.model, "predict"):
            return self.model.predict
        else:
            raise ValueError(
                "Model must have 'predict' or 'predict_proba' method"
            )

    def _initialize_explainer(self, X: np.ndarray) -> None:
        """
        Initialize the appropriate SHAP explainer.

        Args:
            X: Sample data for initialization
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is required. Install with: pip install shap"
            )

        predict_fn = self._get_prediction_function()

        # Select background data
        if self._background_data is not None:
            background = self._background_data
        else:
            # Use kmeans to select representative background samples
            n_bg = min(self.config.background_samples, len(X))
            background = shap.kmeans(X, n_bg)

        # Initialize explainer based on type
        if self.config.explainer_type == ExplainerType.TREE:
            try:
                self._explainer = shap.TreeExplainer(self.model)
            except Exception:
                logger.warning(
                    "TreeExplainer failed, falling back to KernelExplainer"
                )
                self._explainer = shap.KernelExplainer(predict_fn, background)

        elif self.config.explainer_type == ExplainerType.LINEAR:
            try:
                self._explainer = shap.LinearExplainer(self.model, background)
            except Exception:
                logger.warning(
                    "LinearExplainer failed, falling back to KernelExplainer"
                )
                self._explainer = shap.KernelExplainer(predict_fn, background)

        elif self.config.explainer_type == ExplainerType.DEEP:
            try:
                self._explainer = shap.DeepExplainer(self.model, background)
            except Exception:
                logger.warning(
                    "DeepExplainer failed, falling back to KernelExplainer"
                )
                self._explainer = shap.KernelExplainer(predict_fn, background)

        elif self.config.explainer_type == ExplainerType.GRADIENT:
            try:
                self._explainer = shap.GradientExplainer(self.model, background)
            except Exception:
                logger.warning(
                    "GradientExplainer failed, falling back to KernelExplainer"
                )
                self._explainer = shap.KernelExplainer(predict_fn, background)

        elif self.config.explainer_type == ExplainerType.PARTITION:
            try:
                self._explainer = shap.PartitionExplainer(predict_fn, background)
            except Exception:
                logger.warning(
                    "PartitionExplainer failed, falling back to KernelExplainer"
                )
                self._explainer = shap.KernelExplainer(predict_fn, background)

        else:
            # Default to KernelExplainer (model-agnostic)
            self._explainer = shap.KernelExplainer(predict_fn, background)

        self._initialized = True
        logger.info(f"SHAP explainer initialized: {type(self._explainer).__name__}")

    def _calculate_provenance(
        self,
        X: np.ndarray,
        shap_values: np.ndarray
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Args:
            X: Input data
            shap_values: Computed SHAP values

        Returns:
            SHA-256 hash string
        """
        # Create deterministic string representation
        input_str = np.array2string(X, precision=8, separator=",")
        output_str = np.array2string(shap_values, precision=8, separator=",")
        combined = f"{input_str}|{output_str}|{self.config.explainer_type}"

        return hashlib.sha256(combined.encode()).hexdigest()

    def _compute_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Compute mean absolute SHAP values per feature.

        Args:
            shap_values: SHAP values matrix
            feature_names: Names of features

        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Mean absolute SHAP value per feature
        mean_abs = np.abs(shap_values).mean(axis=0)

        # Handle multi-output case
        if len(mean_abs.shape) > 1:
            mean_abs = mean_abs.mean(axis=-1)

        # Create sorted dictionary
        importance = {
            name: float(score)
            for name, score in zip(feature_names, mean_abs)
        }

        # Sort by importance (descending)
        importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        return importance

    def explain(
        self,
        X: Union[np.ndarray, List[List[float]]],
        feature_names: Optional[List[str]] = None
    ) -> SHAPResult:
        """
        Generate SHAP explanations for input samples.

        This method computes SHAP values for each feature and sample,
        providing interpretable explanations for model predictions.

        Args:
            X: Input data to explain (samples x features)
            feature_names: Optional feature names (overrides config)

        Returns:
            SHAPResult containing SHAP values and feature importance

        Raises:
            ValueError: If input data is invalid
            ImportError: If SHAP is not installed

        Example:
            >>> result = explainer.explain(X_test[:5])
            >>> for name, importance in result.feature_importance.items():
            ...     print(f"{name}: {importance:.4f}")
        """
        start_time = datetime.utcnow()

        # Convert to numpy array if needed
        if isinstance(X, list):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Determine feature names
        if feature_names is not None:
            names = feature_names
        elif self.config.feature_names is not None:
            names = self.config.feature_names
        else:
            names = [f"feature_{i}" for i in range(X.shape[1])]

        # Validate feature count
        if len(names) != X.shape[1]:
            raise ValueError(
                f"Feature names count ({len(names)}) does not match "
                f"input features ({X.shape[1]})"
            )

        # Initialize explainer if needed
        if not self._initialized:
            self._initialize_explainer(X)

        # Compute SHAP values
        logger.info(f"Computing SHAP values for {X.shape[0]} samples")
        shap_values = self._explainer.shap_values(X)

        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class case - use mean across classes
            shap_values = np.mean(np.array(shap_values), axis=0)

        # Get expected value
        if hasattr(self._explainer, "expected_value"):
            expected_value = self._explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = float(np.mean(expected_value))
            else:
                expected_value = float(expected_value)
        else:
            expected_value = 0.0

        # Compute feature importance
        importance = self._compute_feature_importance(shap_values, names)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(X, shap_values)

        # Calculate processing time
        processing_time_ms = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        logger.info(
            f"SHAP explanation completed in {processing_time_ms:.2f}ms, "
            f"provenance: {provenance_hash[:16]}..."
        )

        return SHAPResult(
            shap_values=shap_values.tolist(),
            base_value=expected_value,
            feature_importance=importance,
            feature_names=names,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time_ms,
            explainer_type=self.config.explainer_type.value,
            n_samples=X.shape[0],
            timestamp=datetime.utcnow()
        )

    def explain_single(
        self,
        x: Union[np.ndarray, List[float]],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single sample.

        Args:
            x: Single sample to explain
            feature_names: Optional feature names

        Returns:
            Dictionary with feature contributions
        """
        if isinstance(x, list):
            x = np.array(x)

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        result = self.explain(x, feature_names)

        # Return simplified single-sample result
        return {
            "contributions": dict(
                zip(
                    result.feature_names,
                    result.shap_values[0]
                )
            ),
            "base_value": result.base_value,
            "feature_importance": result.feature_importance,
            "provenance_hash": result.provenance_hash
        }

    def get_summary_plot_data(
        self,
        X: np.ndarray,
        max_features: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get data for generating summary plot.

        Args:
            X: Input data to explain
            max_features: Maximum features to include

        Returns:
            Dictionary with plot data
        """
        result = self.explain(X)

        max_feat = max_features or self.config.max_display_features

        # Get top features
        top_features = list(result.feature_importance.keys())[:max_feat]

        # Get indices
        feature_indices = [
            result.feature_names.index(f) for f in top_features
        ]

        shap_array = np.array(result.shap_values)

        return {
            "shap_values": shap_array[:, feature_indices].tolist(),
            "feature_names": top_features,
            "feature_values": X[:, feature_indices].tolist(),
            "base_value": result.base_value
        }

    def get_waterfall_data(
        self,
        x: np.ndarray,
        max_features: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get data for generating waterfall plot for single prediction.

        Args:
            x: Single sample
            max_features: Maximum features to include

        Returns:
            Dictionary with waterfall plot data
        """
        result = self.explain_single(x)

        max_feat = max_features or self.config.max_display_features

        contributions = result["contributions"]

        # Sort by absolute contribution
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:max_feat]

        return {
            "base_value": result["base_value"],
            "features": [f[0] for f in sorted_features],
            "contributions": [f[1] for f in sorted_features],
            "cumulative": self._compute_cumulative(
                result["base_value"],
                [f[1] for f in sorted_features]
            )
        }

    def _compute_cumulative(
        self,
        base: float,
        contributions: List[float]
    ) -> List[float]:
        """Compute cumulative values for waterfall plot."""
        cumulative = [base]
        for c in contributions:
            cumulative.append(cumulative[-1] + c)
        return cumulative


# Unit test stubs
class TestSHAPExplainer:
    """Unit tests for SHAPExplainer."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        # Mock model
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        explainer = SHAPExplainer(MockModel())
        assert explainer.config.explainer_type == ExplainerType.KERNEL
        assert not explainer._initialized

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        config = SHAPExplainerConfig(
            explainer_type=ExplainerType.TREE,
            n_samples=50,
            feature_names=["a", "b", "c"]
        )

        explainer = SHAPExplainer(MockModel(), config=config)
        assert explainer.config.explainer_type == ExplainerType.TREE
        assert explainer.config.n_samples == 50

    def test_feature_importance_calculation(self):
        """Test feature importance calculation."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        explainer = SHAPExplainer(MockModel())

        # Mock SHAP values
        shap_values = np.array([
            [0.1, 0.5, 0.2],
            [0.2, 0.4, 0.1],
            [0.15, 0.45, 0.15]
        ])

        importance = explainer._compute_feature_importance(
            shap_values,
            ["feat_a", "feat_b", "feat_c"]
        )

        # feat_b should be most important
        assert list(importance.keys())[0] == "feat_b"

    def test_provenance_hash_deterministic(self):
        """Test that provenance hash is deterministic."""
        class MockModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        explainer = SHAPExplainer(MockModel())

        X = np.array([[1.0, 2.0, 3.0]])
        shap_values = np.array([[0.1, 0.2, 0.3]])

        hash1 = explainer._calculate_provenance(X, shap_values)
        hash2 = explainer._calculate_provenance(X, shap_values)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
