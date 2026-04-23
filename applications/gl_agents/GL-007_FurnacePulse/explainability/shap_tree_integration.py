"""
GL-007 FurnacePulse - SHAP TreeExplainer Integration

Production-grade SHAP TreeExplainer integration for furnace monitoring predictions.
Provides fast, exact SHAP value computation for tree-based models (XGBoost, LightGBM,
CatBoost, RandomForest, GradientBoosting).

Key Features:
    - explain_hotspot_prediction(): SHAP explanations for hotspot severity
    - explain_rul_prediction(): SHAP explanations for remaining useful life
    - Waterfall plot generation for individual predictions
    - Force plot generation for feature contribution visualization
    - Batch explanation for multiple predictions
    - Provenance tracking with SHA-256 hashes

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017

Zero-Hallucination: All SHAP values are computed deterministically from trained models.
No LLM involvement in numeric calculations.

Example:
    >>> explainer = SHAPTreeIntegration(model=xgb_model, feature_names=feature_names)
    >>> result = explainer.explain_hotspot_prediction(sensor_readings, hotspot_prediction)
    >>> waterfall_data = explainer.generate_waterfall_plot(result)
    >>> force_data = explainer.generate_force_plot(result)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import hashlib
import json
import logging
import io
import base64

import numpy as np

# SHAP library import with version check
try:
    import shap
    from shap import TreeExplainer, Explanation
    SHAP_VERSION = shap.__version__
    HAS_SHAP = True
    if tuple(map(int, SHAP_VERSION.split('.')[:2])) < (0, 42):
        logging.warning(f"SHAP version {SHAP_VERSION} is older than recommended 0.42+")
except ImportError:
    HAS_SHAP = False
    SHAP_VERSION = None
    TreeExplainer = None
    Explanation = None

# Optional matplotlib for plot generation
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

logger = logging.getLogger(__name__)


class PredictionType(str, Enum):
    """Type of prediction being explained."""
    HOTSPOT = "hotspot"
    RUL = "remaining_useful_life"
    EFFICIENCY = "efficiency"
    ANOMALY = "anomaly"


class ExplanationType(str, Enum):
    """Type of SHAP explanation output."""
    LOCAL = "local"  # Individual prediction
    GLOBAL = "global"  # Model-wide feature importance
    BATCH = "batch"  # Multiple predictions


@dataclass
class FeatureContribution:
    """
    Single feature's contribution to a prediction.

    Attributes:
        feature_name: Name of the input feature
        feature_value: Actual value of the feature
        shap_value: SHAP contribution value
        direction: "positive" or "negative" contribution
        contribution_percent: Percentage of total absolute contribution
        rank: Importance rank (1 = most important)
        engineering_interpretation: Human-readable explanation
    """
    feature_name: str
    feature_value: float
    shap_value: float
    direction: str
    contribution_percent: float
    rank: int
    engineering_interpretation: str


@dataclass
class SHAPExplanationResult:
    """
    Complete SHAP explanation result for a prediction.

    Attributes:
        prediction_type: Type of prediction (HOTSPOT, RUL, etc.)
        prediction_id: Unique identifier for this prediction
        predicted_value: Model output value
        base_value: Expected value (model output without features)
        feature_names: Ordered list of feature names
        feature_values: Ordered list of feature values
        shap_values: SHAP values for each feature
        contributions: Ranked list of feature contributions
        interaction_effects: Top feature interaction pairs
        timestamp: When explanation was generated
        computation_hash: SHA-256 hash for provenance
        model_info: Model metadata (type, version, etc.)
        explanation_time_ms: Time to compute explanation
    """
    prediction_type: PredictionType
    prediction_id: str
    predicted_value: float
    base_value: float
    feature_names: List[str]
    feature_values: np.ndarray
    shap_values: np.ndarray
    contributions: List[FeatureContribution]
    interaction_effects: Dict[str, Dict[str, float]]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    computation_hash: str = ""
    model_info: Dict[str, Any] = field(default_factory=dict)
    explanation_time_ms: float = 0.0


@dataclass
class WaterfallPlotData:
    """
    Data structure for waterfall plot visualization.

    The waterfall shows how each feature pushes the prediction
    from the base value to the final predicted value.
    """
    base_value: float
    predicted_value: float
    features: List[Dict[str, Any]]
    plot_image_base64: Optional[str] = None


@dataclass
class ForcePlotData:
    """
    Data structure for force plot visualization.

    The force plot shows features pushing prediction higher (red)
    or lower (blue) from the base value.
    """
    base_value: float
    predicted_value: float
    positive_features: List[Dict[str, Any]]
    negative_features: List[Dict[str, Any]]
    plot_html: Optional[str] = None


class SHAPTreeIntegration:
    """
    SHAP TreeExplainer integration for furnace monitoring predictions.

    Provides production-grade SHAP explanations using TreeExplainer for
    fast, exact computation of SHAP values. Supports hotspot detection
    and RUL prediction models.

    Features:
        - Exact SHAP values for tree-based models
        - Waterfall and force plot generation
        - Batch explanation support
        - Provenance tracking with SHA-256
        - Engineering-focused interpretations

    Supported Models:
        - XGBoost (xgboost.XGBRegressor, XGBClassifier)
        - LightGBM (lightgbm.LGBMRegressor, LGBMClassifier)
        - CatBoost (catboost.CatBoostRegressor, CatBoostClassifier)
        - Scikit-learn (RandomForestRegressor, GradientBoostingRegressor)

    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=100).fit(X, y)
        >>> explainer = SHAPTreeIntegration(
        ...     model=model,
        ...     feature_names=["tmt_zone1", "tmt_zone2", "flow_rate"]
        ... )
        >>> result = explainer.explain_hotspot_prediction(
        ...     sensor_readings={"tmt_zone1": 850.0, "tmt_zone2": 820.0, "flow_rate": 1500.0},
        ...     hotspot_prediction={"severity_score": 0.75, "prediction_id": "HS-001"}
        ... )
    """

    VERSION = "1.0.0"

    # Feature category patterns for engineering interpretation
    FEATURE_PATTERNS = {
        "tmt": "Tube metal temperature",
        "temp": "Temperature measurement",
        "flow": "Flow rate",
        "pressure": "Pressure reading",
        "flame": "Flame characteristic",
        "vibration": "Vibration level",
        "o2": "Oxygen concentration",
        "co2": "CO2 concentration",
        "nox": "NOx emission",
        "efficiency": "Efficiency metric",
        "gradient": "Temperature gradient",
        "rate": "Rate of change",
        "delta": "Difference/deviation",
        "hours": "Operating time",
        "cycles": "Thermal cycles",
    }

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        model_version: str = "1.0.0",
        model_type: str = "auto",
        background_data: Optional[np.ndarray] = None,
        max_display_features: int = 10,
        compute_interactions: bool = True,
    ) -> None:
        """
        Initialize SHAP TreeExplainer integration.

        Args:
            model: Trained tree-based model (XGBoost, LightGBM, RandomForest, etc.)
            feature_names: List of feature names matching model input order
            model_version: Version string for the model
            model_type: Model type ("xgboost", "lightgbm", "sklearn", "auto")
            background_data: Optional background dataset for SHAP (uses model defaults if None)
            max_display_features: Maximum features to display in plots
            compute_interactions: Whether to compute SHAP interaction values

        Raises:
            ImportError: If SHAP library is not available
            ValueError: If model is not a supported tree-based model
        """
        if not HAS_SHAP:
            raise ImportError(
                "SHAP library is required. Install with: pip install shap>=0.42"
            )

        self.model = model
        self.feature_names = feature_names
        self.model_version = model_version
        self.model_type = model_type if model_type != "auto" else self._detect_model_type()
        self.background_data = background_data
        self.max_display_features = max_display_features
        self.compute_interactions = compute_interactions

        # Initialize TreeExplainer
        self._explainer = self._create_explainer()
        self._explanation_count = 0

        logger.info(
            f"SHAPTreeIntegration initialized: model_type={self.model_type}, "
            f"features={len(feature_names)}, shap_version={SHAP_VERSION}"
        )

    def _detect_model_type(self) -> str:
        """Detect model type from class name."""
        class_name = type(self.model).__name__.lower()

        if "xgb" in class_name:
            return "xgboost"
        elif "lgb" in class_name or "lightgbm" in class_name:
            return "lightgbm"
        elif "catboost" in class_name:
            return "catboost"
        elif "forest" in class_name or "gradient" in class_name or "tree" in class_name:
            return "sklearn"
        else:
            logger.warning(f"Unknown model type: {class_name}, using sklearn fallback")
            return "sklearn"

    def _create_explainer(self) -> "TreeExplainer":
        """Create SHAP TreeExplainer for the model."""
        try:
            if self.background_data is not None:
                explainer = shap.TreeExplainer(
                    self.model,
                    data=self.background_data,
                    feature_perturbation="interventional"
                )
            else:
                explainer = shap.TreeExplainer(self.model)

            logger.info("TreeExplainer created successfully")
            return explainer

        except Exception as e:
            logger.error(f"Failed to create TreeExplainer: {e}")
            raise ValueError(f"Model is not supported by TreeExplainer: {e}")

    def explain_hotspot_prediction(
        self,
        sensor_readings: Dict[str, float],
        hotspot_prediction: Dict[str, Any],
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> SHAPExplanationResult:
        """
        Explain a hotspot detection prediction using SHAP TreeExplainer.

        This method provides complete feature attribution for hotspot severity
        predictions, identifying which sensor readings most contributed to
        the detection.

        Args:
            sensor_readings: Dictionary of sensor tag IDs to current values
                Example: {"tmt_zone1": 850.0, "tmt_zone2": 820.0, "flow_rate": 1500.0}
            hotspot_prediction: Hotspot prediction result containing:
                - severity_score: Predicted severity (0.0 - 1.0)
                - prediction_id: Unique prediction identifier
                - Additional fields (location, confidence, etc.)
            additional_context: Optional context for interpretation

        Returns:
            SHAPExplanationResult with complete feature attribution

        Raises:
            ValueError: If sensor readings don't match expected features
        """
        import time
        start_time = time.perf_counter()

        # Extract prediction metadata
        prediction_id = hotspot_prediction.get("prediction_id", self._generate_id())
        predicted_value = hotspot_prediction.get("severity_score", 0.0)

        # Build feature vector in correct order
        feature_values = self._build_feature_vector(sensor_readings)

        # Compute SHAP values using TreeExplainer
        shap_result = self._compute_shap_values(feature_values)

        # Extract base value and SHAP values
        base_value = float(shap_result.base_values)
        shap_values = shap_result.values

        # Build feature contributions
        contributions = self._build_contributions(
            feature_values, shap_values, PredictionType.HOTSPOT
        )

        # Compute interactions if enabled
        interaction_effects = {}
        if self.compute_interactions:
            interaction_effects = self._compute_top_interactions(
                feature_values, shap_values
            )

        # Compute provenance hash
        computation_hash = self._compute_provenance_hash(
            feature_values, shap_values, base_value, predicted_value
        )

        explanation_time_ms = (time.perf_counter() - start_time) * 1000
        self._explanation_count += 1

        result = SHAPExplanationResult(
            prediction_type=PredictionType.HOTSPOT,
            prediction_id=prediction_id,
            predicted_value=predicted_value,
            base_value=base_value,
            feature_names=self.feature_names.copy(),
            feature_values=feature_values,
            shap_values=shap_values,
            contributions=contributions,
            interaction_effects=interaction_effects,
            computation_hash=computation_hash,
            model_info={
                "model_type": self.model_type,
                "model_version": self.model_version,
                "shap_version": SHAP_VERSION,
                "explainer_version": self.VERSION,
            },
            explanation_time_ms=round(explanation_time_ms, 2),
        )

        logger.debug(
            f"Hotspot explanation completed: prediction_id={prediction_id}, "
            f"time={explanation_time_ms:.2f}ms"
        )

        return result

    def explain_rul_prediction(
        self,
        sensor_readings: Dict[str, float],
        rul_prediction: Dict[str, Any],
        component_id: str = "furnace_tube",
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> SHAPExplanationResult:
        """
        Explain a Remaining Useful Life (RUL) prediction using SHAP TreeExplainer.

        This method provides complete feature attribution for RUL predictions,
        identifying which factors most influence the predicted remaining life.

        Args:
            sensor_readings: Dictionary of sensor tag IDs to current values
                Example: {"operating_hours": 45000, "tmt_max": 920.0, "thermal_cycles": 1500}
            rul_prediction: RUL prediction result containing:
                - rul_hours: Predicted remaining useful life in hours
                - prediction_id: Unique prediction identifier
                - confidence: Prediction confidence (optional)
            component_id: ID of the component being assessed
            additional_context: Optional context for interpretation

        Returns:
            SHAPExplanationResult with complete feature attribution

        Raises:
            ValueError: If sensor readings don't match expected features
        """
        import time
        start_time = time.perf_counter()

        # Extract prediction metadata
        prediction_id = rul_prediction.get("prediction_id", self._generate_id())
        predicted_value = rul_prediction.get("rul_hours", 0.0)

        # Build feature vector in correct order
        feature_values = self._build_feature_vector(sensor_readings)

        # Compute SHAP values using TreeExplainer
        shap_result = self._compute_shap_values(feature_values)

        # Extract base value and SHAP values
        base_value = float(shap_result.base_values)
        shap_values = shap_result.values

        # Build feature contributions with RUL-specific interpretations
        contributions = self._build_contributions(
            feature_values, shap_values, PredictionType.RUL
        )

        # Compute interactions if enabled
        interaction_effects = {}
        if self.compute_interactions:
            interaction_effects = self._compute_top_interactions(
                feature_values, shap_values
            )

        # Compute provenance hash
        computation_hash = self._compute_provenance_hash(
            feature_values, shap_values, base_value, predicted_value
        )

        explanation_time_ms = (time.perf_counter() - start_time) * 1000
        self._explanation_count += 1

        result = SHAPExplanationResult(
            prediction_type=PredictionType.RUL,
            prediction_id=prediction_id,
            predicted_value=predicted_value,
            base_value=base_value,
            feature_names=self.feature_names.copy(),
            feature_values=feature_values,
            shap_values=shap_values,
            contributions=contributions,
            interaction_effects=interaction_effects,
            computation_hash=computation_hash,
            model_info={
                "model_type": self.model_type,
                "model_version": self.model_version,
                "shap_version": SHAP_VERSION,
                "explainer_version": self.VERSION,
                "component_id": component_id,
            },
            explanation_time_ms=round(explanation_time_ms, 2),
        )

        logger.debug(
            f"RUL explanation completed: prediction_id={prediction_id}, "
            f"rul_hours={predicted_value}, time={explanation_time_ms:.2f}ms"
        )

        return result

    def explain_batch(
        self,
        sensor_readings_batch: List[Dict[str, float]],
        predictions_batch: List[Dict[str, Any]],
        prediction_type: PredictionType,
    ) -> List[SHAPExplanationResult]:
        """
        Explain a batch of predictions efficiently.

        Batch processing is more efficient than individual explanations
        as it leverages vectorized SHAP computation.

        Args:
            sensor_readings_batch: List of sensor reading dictionaries
            predictions_batch: List of prediction result dictionaries
            prediction_type: Type of predictions (HOTSPOT or RUL)

        Returns:
            List of SHAPExplanationResult objects
        """
        if len(sensor_readings_batch) != len(predictions_batch):
            raise ValueError("Sensor readings and predictions must have same length")

        import time
        start_time = time.perf_counter()

        # Build feature matrix
        feature_matrix = np.array([
            self._build_feature_vector(readings)
            for readings in sensor_readings_batch
        ])

        # Compute SHAP values for entire batch
        shap_results = self._explainer(feature_matrix)

        results = []
        for i, (readings, prediction) in enumerate(zip(sensor_readings_batch, predictions_batch)):
            prediction_id = prediction.get("prediction_id", self._generate_id())

            if prediction_type == PredictionType.HOTSPOT:
                predicted_value = prediction.get("severity_score", 0.0)
            else:
                predicted_value = prediction.get("rul_hours", 0.0)

            base_value = float(shap_results.base_values[i])
            shap_values = shap_results.values[i]

            contributions = self._build_contributions(
                feature_matrix[i], shap_values, prediction_type
            )

            computation_hash = self._compute_provenance_hash(
                feature_matrix[i], shap_values, base_value, predicted_value
            )

            results.append(SHAPExplanationResult(
                prediction_type=prediction_type,
                prediction_id=prediction_id,
                predicted_value=predicted_value,
                base_value=base_value,
                feature_names=self.feature_names.copy(),
                feature_values=feature_matrix[i],
                shap_values=shap_values,
                contributions=contributions,
                interaction_effects={},  # Skip interactions for batch
                computation_hash=computation_hash,
                model_info={
                    "model_type": self.model_type,
                    "model_version": self.model_version,
                    "shap_version": SHAP_VERSION,
                    "batch_index": i,
                    "batch_size": len(sensor_readings_batch),
                },
            ))

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Batch explanation completed: {len(results)} predictions, "
            f"time={elapsed_ms:.2f}ms ({elapsed_ms/len(results):.2f}ms/prediction)"
        )

        return results

    def generate_waterfall_plot(
        self,
        explanation: SHAPExplanationResult,
        max_display: Optional[int] = None,
        show_values: bool = True,
    ) -> WaterfallPlotData:
        """
        Generate waterfall plot data for an explanation.

        The waterfall plot shows how each feature pushes the prediction
        from the base value to the final predicted value.

        Args:
            explanation: SHAP explanation result
            max_display: Maximum features to display (default: max_display_features)
            show_values: Whether to include feature values in labels

        Returns:
            WaterfallPlotData with plot data and optional base64 image
        """
        max_display = max_display or self.max_display_features

        # Sort features by absolute SHAP value
        sorted_indices = np.argsort(np.abs(explanation.shap_values))[::-1]
        top_indices = sorted_indices[:max_display]

        features = []
        for idx in top_indices:
            feature_name = explanation.feature_names[idx]
            feature_value = float(explanation.feature_values[idx])
            shap_value = float(explanation.shap_values[idx])

            features.append({
                "name": feature_name,
                "value": round(feature_value, 4),
                "shap_value": round(shap_value, 6),
                "direction": "positive" if shap_value > 0 else "negative",
            })

        plot_data = WaterfallPlotData(
            base_value=explanation.base_value,
            predicted_value=explanation.predicted_value,
            features=features,
        )

        # Generate plot image if matplotlib available
        if HAS_MATPLOTLIB and plt is not None:
            try:
                # Create SHAP Explanation object for plotting
                shap_exp = shap.Explanation(
                    values=explanation.shap_values,
                    base_values=explanation.base_value,
                    data=explanation.feature_values,
                    feature_names=explanation.feature_names,
                )

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_exp, max_display=max_display, show=False)

                # Save to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                plot_data.plot_image_base64 = base64.b64encode(buffer.read()).decode()
                plt.close(fig)

            except Exception as e:
                logger.warning(f"Failed to generate waterfall plot image: {e}")

        return plot_data

    def generate_force_plot(
        self,
        explanation: SHAPExplanationResult,
        link: str = "identity",
    ) -> ForcePlotData:
        """
        Generate force plot data for an explanation.

        The force plot shows features pushing prediction higher (positive/red)
        or lower (negative/blue) from the base value.

        Args:
            explanation: SHAP explanation result
            link: Link function ("identity" or "logit")

        Returns:
            ForcePlotData with separated positive/negative contributions
        """
        positive_features = []
        negative_features = []

        # Sort by absolute value for display
        sorted_indices = np.argsort(np.abs(explanation.shap_values))[::-1]

        for idx in sorted_indices[:self.max_display_features]:
            feature_name = explanation.feature_names[idx]
            feature_value = float(explanation.feature_values[idx])
            shap_value = float(explanation.shap_values[idx])

            feature_data = {
                "name": feature_name,
                "value": round(feature_value, 4),
                "shap_value": round(shap_value, 6),
                "contribution_abs": round(abs(shap_value), 6),
            }

            if shap_value > 0:
                positive_features.append(feature_data)
            else:
                negative_features.append(feature_data)

        plot_data = ForcePlotData(
            base_value=explanation.base_value,
            predicted_value=explanation.predicted_value,
            positive_features=positive_features,
            negative_features=negative_features,
        )

        # Generate force plot HTML if SHAP supports it
        if HAS_SHAP:
            try:
                shap_exp = shap.Explanation(
                    values=explanation.shap_values,
                    base_values=explanation.base_value,
                    data=explanation.feature_values,
                    feature_names=explanation.feature_names,
                )

                force_plot = shap.plots.force(shap_exp, matplotlib=False, show=False)
                if hasattr(force_plot, 'html'):
                    plot_data.plot_html = force_plot.html()

            except Exception as e:
                logger.debug(f"Force plot HTML generation skipped: {e}")

        return plot_data

    def get_global_feature_importance(
        self,
        sample_data: Optional[np.ndarray] = None,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Compute global feature importance across multiple samples.

        Args:
            sample_data: Sample data for computing importance (uses background if None)
            num_samples: Number of samples to use if generating

        Returns:
            Dictionary mapping feature names to mean absolute SHAP values
        """
        if sample_data is None:
            if self.background_data is not None:
                sample_data = self.background_data[:num_samples]
            else:
                logger.warning("No sample data for global importance, using current model")
                return {}

        # Compute SHAP values for all samples
        shap_results = self._explainer(sample_data)
        mean_abs_shap = np.mean(np.abs(shap_results.values), axis=0)

        importance = {}
        for i, name in enumerate(self.feature_names):
            importance[name] = round(float(mean_abs_shap[i]), 6)

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance

    def _build_feature_vector(self, sensor_readings: Dict[str, float]) -> np.ndarray:
        """Build feature vector from sensor readings in correct order."""
        values = []
        missing_features = []

        for name in self.feature_names:
            if name in sensor_readings:
                values.append(float(sensor_readings[name]))
            else:
                missing_features.append(name)
                values.append(0.0)  # Default for missing

        if missing_features:
            logger.warning(f"Missing features (using 0.0): {missing_features}")

        return np.array(values)

    def _compute_shap_values(self, feature_values: np.ndarray) -> Any:
        """Compute SHAP values using TreeExplainer."""
        # Reshape for single prediction
        if feature_values.ndim == 1:
            feature_values = feature_values.reshape(1, -1)

        shap_result = self._explainer(feature_values)

        # Handle single prediction output
        if shap_result.values.ndim > 1:
            return shap.Explanation(
                values=shap_result.values[0],
                base_values=shap_result.base_values[0] if hasattr(shap_result.base_values, '__len__') else shap_result.base_values,
                data=shap_result.data[0] if hasattr(shap_result.data, '__len__') else shap_result.data,
                feature_names=self.feature_names,
            )

        return shap_result

    def _build_contributions(
        self,
        feature_values: np.ndarray,
        shap_values: np.ndarray,
        prediction_type: PredictionType,
    ) -> List[FeatureContribution]:
        """Build ranked list of feature contributions."""
        total_abs_shap = np.sum(np.abs(shap_values))
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]

        contributions = []
        for rank, idx in enumerate(sorted_indices, 1):
            feature_name = self.feature_names[idx]
            feature_value = float(feature_values[idx])
            shap_value = float(shap_values[idx])
            direction = "positive" if shap_value > 0 else "negative"

            contribution_pct = (
                abs(shap_value) / total_abs_shap * 100
                if total_abs_shap > 0 else 0.0
            )

            interpretation = self._generate_interpretation(
                feature_name, shap_value, feature_value, prediction_type
            )

            contributions.append(FeatureContribution(
                feature_name=feature_name,
                feature_value=round(feature_value, 4),
                shap_value=round(shap_value, 6),
                direction=direction,
                contribution_percent=round(contribution_pct, 2),
                rank=rank,
                engineering_interpretation=interpretation,
            ))

        return contributions

    def _compute_top_interactions(
        self,
        feature_values: np.ndarray,
        shap_values: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """Compute top feature interaction effects."""
        interactions = {}

        # Get top features by absolute SHAP
        top_indices = np.argsort(np.abs(shap_values))[-top_k:]

        for i in top_indices:
            feature_i = self.feature_names[i]
            interactions[feature_i] = {}

            for j in top_indices:
                if i != j:
                    feature_j = self.feature_names[j]
                    # Approximate interaction as normalized product
                    interaction_val = (
                        shap_values[i] * shap_values[j] /
                        (abs(shap_values[i]) + abs(shap_values[j]) + 1e-10)
                    )
                    interactions[feature_i][feature_j] = round(float(interaction_val), 6)

        return interactions

    def _generate_interpretation(
        self,
        feature_name: str,
        shap_value: float,
        feature_value: float,
        prediction_type: PredictionType,
    ) -> str:
        """Generate engineering interpretation for a feature contribution."""
        direction = "increases" if shap_value > 0 else "decreases"
        name_lower = feature_name.lower()

        # Find matching pattern
        feature_desc = "Feature"
        for pattern, description in self.FEATURE_PATTERNS.items():
            if pattern in name_lower:
                feature_desc = description
                break

        # Build interpretation based on prediction type
        if prediction_type == PredictionType.HOTSPOT:
            if shap_value > 0:
                return (
                    f"{feature_desc} ({feature_value:.2f}) {direction} hotspot severity. "
                    f"Higher values indicate increased thermal stress risk."
                )
            else:
                return (
                    f"{feature_desc} ({feature_value:.2f}) {direction} hotspot severity. "
                    f"This value helps maintain safe thermal conditions."
                )

        elif prediction_type == PredictionType.RUL:
            if shap_value > 0:
                return (
                    f"{feature_desc} ({feature_value:.2f}) {direction} predicted RUL. "
                    f"This condition contributes to extended component life."
                )
            else:
                return (
                    f"{feature_desc} ({feature_value:.2f}) {direction} predicted RUL. "
                    f"This condition accelerates degradation and reduces remaining life."
                )

        else:
            return f"{feature_desc} ({feature_value:.2f}) {direction} the prediction."

    def _compute_provenance_hash(
        self,
        feature_values: np.ndarray,
        shap_values: np.ndarray,
        base_value: float,
        predicted_value: float,
    ) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        data = {
            "feature_values": feature_values.tolist(),
            "shap_values": shap_values.tolist(),
            "base_value": base_value,
            "predicted_value": predicted_value,
            "model_version": self.model_version,
            "explainer_version": self.VERSION,
            "shap_version": SHAP_VERSION,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _generate_id(self) -> str:
        """Generate unique explanation ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        self._explanation_count += 1
        data = f"{timestamp}_{self._explanation_count}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def get_stats(self) -> Dict[str, Any]:
        """Get explainer statistics."""
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "shap_version": SHAP_VERSION,
            "explainer_version": self.VERSION,
            "num_features": len(self.feature_names),
            "explanations_generated": self._explanation_count,
            "has_background_data": self.background_data is not None,
            "compute_interactions": self.compute_interactions,
        }


def create_shap_explainer_for_furnace(
    model: Any,
    feature_names: Optional[List[str]] = None,
    model_version: str = "1.0.0",
) -> SHAPTreeIntegration:
    """
    Factory function to create SHAP explainer with furnace-specific defaults.

    Args:
        model: Trained tree-based model
        feature_names: Feature names (auto-generated if None)
        model_version: Model version string

    Returns:
        Configured SHAPTreeIntegration instance
    """
    # Default furnace feature names if not provided
    if feature_names is None:
        feature_names = [
            "tmt_zone1_avg", "tmt_zone2_avg", "tmt_zone3_avg", "tmt_zone4_avg",
            "tmt_max", "tmt_rate_of_rise", "tmt_gradient",
            "fuel_flow", "air_flow", "excess_air_percent",
            "stack_temp", "flue_o2", "flue_co2",
            "draft_firebox", "draft_stack",
            "operating_hours", "thermal_cycles", "days_since_maintenance",
        ]

    return SHAPTreeIntegration(
        model=model,
        feature_names=feature_names,
        model_version=model_version,
        compute_interactions=True,
    )
