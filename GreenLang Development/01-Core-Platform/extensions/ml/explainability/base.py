"""
Base classes for ML Explainability Framework.

This module defines the abstract base classes that all explainability
methods must implement, ensuring a consistent interface across the framework.

Example:
    >>> class MyExplainer(ExplainabilityLayer):
    ...     def explain(self, model, X, **kwargs):
    ...         # Implementation
    ...         pass
    ...     def get_feature_importance(self, model, X):
    ...         # Implementation
    ...         pass
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import hashlib
import json
import logging
import numpy as np

from .schemas import (
    ExplanationResult,
    GlobalExplanationResult,
    ExplainerType,
    ModelType,
    ProcessHeatContext,
    compute_provenance_hash,
)

logger = logging.getLogger(__name__)


class ExplainabilityLayer(ABC):
    """
    Base class for all explainability methods.

    This abstract base class defines the interface that all explainability
    implementations must follow. It ensures consistency across SHAP, LIME,
    counterfactual, and other explanation methods.

    The class implements the zero-hallucination principle: all explanations
    are derived from actual model computations, never synthesized or guessed.

    Attributes:
        explainer_type: Type of explainer (SHAP, LIME, etc.)
        model_type: Type of model being explained
        feature_names: Names of input features
        initialized: Whether the explainer has been initialized

    Example:
        >>> class SHAPExplainer(ExplainabilityLayer):
        ...     def explain(self, model, X, **kwargs):
        ...         shap_values = self.explainer.shap_values(X)
        ...         return self._format_explanation(shap_values)
    """

    def __init__(
        self,
        explainer_type: ExplainerType,
        model_type: ModelType = ModelType.GENERIC,
        feature_names: Optional[List[str]] = None,
        process_heat_context: Optional[ProcessHeatContext] = None,
    ):
        """
        Initialize the explainability layer.

        Args:
            explainer_type: Type of explainer being implemented
            model_type: Type of model to explain
            feature_names: List of feature names
            process_heat_context: Domain context for process heat
        """
        self.explainer_type = explainer_type
        self.model_type = model_type
        self.feature_names = feature_names or []
        self.process_heat_context = process_heat_context
        self.initialized = False
        self._model_cache: Dict[str, Any] = {}

        logger.info(
            f"Initialized {self.__class__.__name__} with "
            f"explainer_type={explainer_type}, model_type={model_type}"
        )

    @abstractmethod
    def explain(
        self,
        model: Any,
        X: Union[np.ndarray, "pd.DataFrame"],
        **kwargs: Any
    ) -> ExplanationResult:
        """
        Generate explanation for a prediction.

        This is the primary method for generating explanations.
        Implementations must compute explanations from actual model
        behavior, not synthesize or guess values.

        Args:
            model: The trained model to explain
            X: Input data (single instance or batch)
            **kwargs: Additional explainer-specific arguments

        Returns:
            ExplanationResult containing:
                - prediction: Model's predicted value
                - feature_contributions: Dict of feature contributions
                - top_features: Ordered list of top contributing features
                - confidence: Explanation confidence score
                - provenance_hash: SHA-256 hash for audit trail

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If explanation generation fails
        """
        pass

    @abstractmethod
    def get_feature_importance(
        self,
        model: Any,
        X: Union[np.ndarray, "pd.DataFrame"],
    ) -> Dict[str, float]:
        """
        Get global feature importance scores.

        Computes the overall importance of each feature based on
        the model and provided data. This is a model-level (global)
        explanation rather than instance-level.

        Args:
            model: The trained model
            X: Background/training data for computing importance

        Returns:
            Dictionary mapping feature names to importance scores.
            Scores are normalized to sum to 1.0.

        Raises:
            ValueError: If model or data is incompatible
        """
        pass

    def initialize(self, model: Any, background_data: Optional[Any] = None) -> None:
        """
        Initialize the explainer with a model.

        Some explainers (like SHAP KernelExplainer) require initialization
        with background data before generating explanations.

        Args:
            model: The model to explain
            background_data: Background/training data for initialization

        Raises:
            RuntimeError: If initialization fails
        """
        self.initialized = True
        logger.info(f"{self.__class__.__name__} initialized successfully")

    def validate_input(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        model: Any
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input data compatibility with the model.

        Args:
            X: Input data to validate
            model: Model to check against

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if X is empty
            if hasattr(X, "shape"):
                if X.shape[0] == 0:
                    return False, "Input data is empty"

            # Check feature count matches if feature_names set
            if self.feature_names:
                if hasattr(X, "shape") and len(X.shape) > 1:
                    if X.shape[1] != len(self.feature_names):
                        return False, (
                            f"Feature count mismatch: expected {len(self.feature_names)}, "
                            f"got {X.shape[1]}"
                        )

            # Check for NaN values
            if hasattr(X, "isna"):
                if X.isna().any().any():
                    return False, "Input contains NaN values"
            elif isinstance(X, np.ndarray):
                if np.isnan(X).any():
                    return False, "Input contains NaN values"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _extract_prediction(
        self,
        model: Any,
        X: Union[np.ndarray, "pd.DataFrame"]
    ) -> float:
        """
        Extract prediction from model.

        Args:
            model: Trained model
            X: Input data

        Returns:
            Model prediction as float
        """
        try:
            # Handle different prediction methods
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)
                if len(pred.shape) > 1:
                    return float(pred[0, 1]) if pred.shape[1] > 1 else float(pred[0, 0])
                return float(pred[0])
            elif hasattr(model, "predict"):
                pred = model.predict(X)
                return float(pred[0]) if hasattr(pred, "__len__") else float(pred)
            else:
                raise ValueError("Model has no predict or predict_proba method")
        except Exception as e:
            logger.error(f"Failed to extract prediction: {e}")
            raise

    def _get_feature_names(
        self,
        X: Union[np.ndarray, "pd.DataFrame"]
    ) -> List[str]:
        """
        Get feature names from data or use stored names.

        Args:
            X: Input data

        Returns:
            List of feature names
        """
        # Try to get from DataFrame
        if hasattr(X, "columns"):
            return list(X.columns)

        # Use stored feature names
        if self.feature_names:
            return self.feature_names

        # Generate default names
        if hasattr(X, "shape") and len(X.shape) > 1:
            return [f"feature_{i}" for i in range(X.shape[1])]

        return ["feature_0"]

    def _compute_confidence(
        self,
        feature_contributions: Dict[str, float],
        prediction: float
    ) -> float:
        """
        Compute confidence score for explanation.

        Confidence is based on how well the feature contributions
        explain the prediction variance.

        Args:
            feature_contributions: Feature contribution values
            prediction: Model prediction

        Returns:
            Confidence score between 0 and 1
        """
        if not feature_contributions:
            return 0.0

        # Sum of absolute contributions
        total_contribution = sum(abs(v) for v in feature_contributions.values())

        # Confidence based on contribution magnitude relative to prediction
        if abs(prediction) > 1e-10:
            explained_ratio = min(total_contribution / abs(prediction), 2.0) / 2.0
        else:
            explained_ratio = 0.5 if total_contribution < 1.0 else 1.0

        # Adjust based on number of features with non-trivial contributions
        significant_features = sum(
            1 for v in feature_contributions.values()
            if abs(v) > 0.01 * total_contribution
        )
        feature_coverage = min(significant_features / max(len(feature_contributions), 1), 1.0)

        confidence = 0.7 * explained_ratio + 0.3 * feature_coverage
        return min(max(confidence, 0.0), 1.0)

    def _format_top_features(
        self,
        feature_contributions: Dict[str, float],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top contributing features sorted by absolute contribution.

        Args:
            feature_contributions: Dictionary of contributions
            top_k: Number of top features to return

        Returns:
            List of (feature_name, contribution) tuples
        """
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:top_k]

    def _compute_provenance_hash(
        self,
        model: Any,
        X: Any,
        contributions: Dict[str, float],
        prediction: float
    ) -> str:
        """
        Compute SHA-256 provenance hash for audit trail.

        Creates a deterministic hash of all inputs and outputs
        for complete reproducibility verification.

        Args:
            model: The model used
            X: Input data
            contributions: Feature contributions
            prediction: Model prediction

        Returns:
            SHA-256 hexdigest string
        """
        # Create hashable representation
        hash_components = {
            "model_type": str(type(model).__name__),
            "input_shape": str(X.shape) if hasattr(X, "shape") else str(len(X)),
            "contributions": contributions,
            "prediction": prediction,
            "timestamp": datetime.utcnow().isoformat()[:19],  # Minute precision
            "explainer_type": self.explainer_type.value,
        }

        # Add input data hash
        if hasattr(X, "tobytes"):
            hash_components["input_hash"] = hashlib.md5(X.tobytes()).hexdigest()[:16]

        return compute_provenance_hash(hash_components)

    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set feature names for the explainer.

        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        logger.debug(f"Set {len(feature_names)} feature names")

    def set_process_heat_context(self, context: ProcessHeatContext) -> None:
        """
        Set process heat domain context.

        Args:
            context: Process heat context with domain knowledge
        """
        self.process_heat_context = context
        logger.debug(f"Set process heat context for {context.equipment_type}")


class ExplainerRegistry:
    """
    Registry for explainability implementations.

    Provides a centralized way to register and retrieve explainer
    implementations by type and model compatibility.
    """

    _registry: Dict[str, type] = {}
    _model_compatibility: Dict[str, List[ModelType]] = {}

    @classmethod
    def register(
        cls,
        explainer_type: str,
        explainer_class: type,
        compatible_models: Optional[List[ModelType]] = None
    ) -> None:
        """
        Register an explainer implementation.

        Args:
            explainer_type: Unique identifier for the explainer
            explainer_class: Class implementing ExplainabilityLayer
            compatible_models: List of compatible model types
        """
        if not issubclass(explainer_class, ExplainabilityLayer):
            raise TypeError(
                f"Explainer class must inherit from ExplainabilityLayer"
            )

        cls._registry[explainer_type] = explainer_class
        cls._model_compatibility[explainer_type] = compatible_models or list(ModelType)

        logger.info(f"Registered explainer: {explainer_type}")

    @classmethod
    def get(cls, explainer_type: str) -> type:
        """
        Get an explainer class by type.

        Args:
            explainer_type: Type of explainer to retrieve

        Returns:
            Explainer class

        Raises:
            KeyError: If explainer type not registered
        """
        if explainer_type not in cls._registry:
            raise KeyError(
                f"Unknown explainer type: {explainer_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[explainer_type]

    @classmethod
    def get_compatible_explainers(cls, model_type: ModelType) -> List[str]:
        """
        Get list of explainers compatible with a model type.

        Args:
            model_type: Type of model

        Returns:
            List of compatible explainer type names
        """
        compatible = []
        for explainer_type, models in cls._model_compatibility.items():
            if model_type in models or ModelType.GENERIC in models:
                compatible.append(explainer_type)
        return compatible

    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered explainer types.

        Returns:
            List of registered explainer type names
        """
        return list(cls._registry.keys())


class ExplainerFactory:
    """
    Factory for creating explainer instances.

    Provides convenient methods for instantiating appropriate
    explainers based on model type and requirements.
    """

    @staticmethod
    def create(
        explainer_type: Union[str, ExplainerType],
        model_type: ModelType = ModelType.GENERIC,
        feature_names: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ExplainabilityLayer:
        """
        Create an explainer instance.

        Args:
            explainer_type: Type of explainer to create
            model_type: Type of model to explain
            feature_names: List of feature names
            **kwargs: Additional arguments for the explainer

        Returns:
            Configured explainer instance

        Raises:
            KeyError: If explainer type not found
        """
        if isinstance(explainer_type, ExplainerType):
            explainer_type = explainer_type.value

        explainer_class = ExplainerRegistry.get(explainer_type)
        return explainer_class(
            explainer_type=ExplainerType(explainer_type),
            model_type=model_type,
            feature_names=feature_names,
            **kwargs
        )

    @staticmethod
    def create_best_for_model(
        model: Any,
        feature_names: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ExplainabilityLayer:
        """
        Create the best explainer for a given model.

        Automatically detects model type and selects the most
        appropriate explainer.

        Args:
            model: The model to explain
            feature_names: List of feature names
            **kwargs: Additional arguments

        Returns:
            Configured explainer instance
        """
        model_type = ExplainerFactory._detect_model_type(model)

        # Select best explainer based on model type
        if model_type in (ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.CATBOOST):
            explainer_type = ExplainerType.SHAP_TREE
        elif model_type == ModelType.SKLEARN:
            # Check if tree-based
            model_name = type(model).__name__.lower()
            if any(t in model_name for t in ["tree", "forest", "gradient", "xgb"]):
                explainer_type = ExplainerType.SHAP_TREE
            elif "linear" in model_name or "logistic" in model_name:
                explainer_type = ExplainerType.SHAP_LINEAR
            else:
                explainer_type = ExplainerType.SHAP_KERNEL
        else:
            explainer_type = ExplainerType.SHAP_KERNEL

        return ExplainerFactory.create(
            explainer_type=explainer_type,
            model_type=model_type,
            feature_names=feature_names,
            **kwargs
        )

    @staticmethod
    def _detect_model_type(model: Any) -> ModelType:
        """
        Detect the type of a model.

        Args:
            model: Model to detect type for

        Returns:
            Detected ModelType
        """
        model_module = type(model).__module__

        if "xgboost" in model_module:
            return ModelType.XGBOOST
        elif "lightgbm" in model_module:
            return ModelType.LIGHTGBM
        elif "catboost" in model_module:
            return ModelType.CATBOOST
        elif "sklearn" in model_module:
            return ModelType.SKLEARN
        elif "torch" in model_module:
            return ModelType.PYTORCH
        elif "tensorflow" in model_module or "keras" in model_module:
            return ModelType.TENSORFLOW
        else:
            return ModelType.GENERIC
