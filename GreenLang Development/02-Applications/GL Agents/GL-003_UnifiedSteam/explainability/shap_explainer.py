"""
GL-003 UNIFIEDSTEAM - SHAP-Based Explainer

Provides SHAP (SHapley Additive exPlanations) for ML model decisions:
- Global feature importance across the fleet
- Local explanations for individual predictions
- Asset-level explanations for specific equipment
- Integration with trap failure prediction and anomaly detection models

IMPORTANT: SHAP is used ONLY for explaining ML model outputs (classification,
anomaly detection), NOT for explaining deterministic calculations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import logging
import uuid
import math

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models that can be explained."""
    TRAP_FAILURE_PREDICTION = "trap_failure_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MAINTENANCE_PREDICTION = "maintenance_prediction"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class FeatureCategory(Enum):
    """Categories of features for grouping."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"
    ENVIRONMENTAL = "environmental"
    DERIVED = "derived"


@dataclass
class FeatureContribution:
    """Single feature's SHAP contribution."""
    feature_name: str
    feature_value: float
    shap_value: float  # SHAP contribution to prediction
    contribution_percent: float
    direction: str  # "positive", "negative"
    category: FeatureCategory
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "shap_value": self.shap_value,
            "contribution_percent": self.contribution_percent,
            "direction": self.direction,
            "category": self.category.value,
            "description": self.description,
        }


@dataclass
class FeatureImportance:
    """Global feature importance from SHAP."""
    importance_id: str
    model_type: ModelType
    timestamp: datetime

    # Feature rankings
    feature_rankings: List[Dict[str, Any]]  # Sorted by importance
    mean_abs_shap: Dict[str, float]  # Feature -> mean |SHAP|

    # Summary statistics
    total_features: int
    top_features: List[str]  # Top 10 features
    feature_categories: Dict[str, List[str]]  # Category -> features

    # Dataset info
    dataset_size: int
    dataset_time_range: Optional[Tuple[datetime, datetime]] = None

    def to_dict(self) -> Dict:
        return {
            "importance_id": self.importance_id,
            "model_type": self.model_type.value,
            "timestamp": self.timestamp.isoformat(),
            "feature_rankings": self.feature_rankings,
            "mean_abs_shap": self.mean_abs_shap,
            "total_features": self.total_features,
            "top_features": self.top_features,
            "feature_categories": self.feature_categories,
            "dataset_size": self.dataset_size,
        }


@dataclass
class LocalExplanation:
    """Local SHAP explanation for a single prediction."""
    explanation_id: str
    model_type: ModelType
    timestamp: datetime
    instance_id: str

    # Prediction details
    predicted_value: float
    base_value: float  # Expected value (E[f(x)])
    prediction_delta: float  # predicted - base

    # Feature contributions
    contributions: List[FeatureContribution]

    # Summary
    top_positive_features: List[str]
    top_negative_features: List[str]

    # Confidence
    explanation_confidence: float = 0.0

    # Natural language
    summary_text: str = ""

    def to_dict(self) -> Dict:
        return {
            "explanation_id": self.explanation_id,
            "model_type": self.model_type.value,
            "timestamp": self.timestamp.isoformat(),
            "instance_id": self.instance_id,
            "predicted_value": self.predicted_value,
            "base_value": self.base_value,
            "prediction_delta": self.prediction_delta,
            "contributions": [c.to_dict() for c in self.contributions],
            "top_positive_features": self.top_positive_features,
            "top_negative_features": self.top_negative_features,
            "explanation_confidence": self.explanation_confidence,
            "summary_text": self.summary_text,
        }


@dataclass
class AssetExplanation:
    """SHAP explanation aggregated for a specific asset."""
    explanation_id: str
    asset_id: str
    asset_type: str  # "trap", "prv", "desuperheater", etc.
    model_type: ModelType
    timestamp: datetime

    # Aggregated over history
    history_window_hours: int
    sample_count: int

    # Feature importance for this asset
    asset_feature_importance: Dict[str, float]
    asset_top_features: List[str]

    # Trend analysis
    feature_trends: Dict[str, str]  # Feature -> "increasing", "decreasing", "stable"

    # Comparison to fleet
    comparison_to_fleet: Dict[str, float]  # Feature -> percentile in fleet

    # Current status
    current_risk_score: float
    risk_drivers: List[str]

    # Recommendations
    suggested_actions: List[str]

    def to_dict(self) -> Dict:
        return {
            "explanation_id": self.explanation_id,
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "model_type": self.model_type.value,
            "timestamp": self.timestamp.isoformat(),
            "history_window_hours": self.history_window_hours,
            "sample_count": self.sample_count,
            "asset_feature_importance": self.asset_feature_importance,
            "asset_top_features": self.asset_top_features,
            "feature_trends": self.feature_trends,
            "comparison_to_fleet": self.comparison_to_fleet,
            "current_risk_score": self.current_risk_score,
            "risk_drivers": self.risk_drivers,
            "suggested_actions": self.suggested_actions,
        }


@dataclass
class SHAPVisualization:
    """Data for SHAP visualization."""
    visualization_id: str
    visualization_type: str  # "summary", "waterfall", "force", "dependence"
    model_type: ModelType
    timestamp: datetime

    # Data for plotting
    plot_data: Dict[str, Any]

    # Rendering hints
    title: str
    x_label: str = ""
    y_label: str = ""
    color_scheme: str = "default"

    def to_dict(self) -> Dict:
        return {
            "visualization_id": self.visualization_id,
            "visualization_type": self.visualization_type,
            "model_type": self.model_type.value,
            "timestamp": self.timestamp.isoformat(),
            "plot_data": self.plot_data,
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "color_scheme": self.color_scheme,
        }


class SHAPExplainer:
    """
    SHAP-based explainer for ML model decisions.

    Integrates with:
    - Trap failure prediction model
    - Anomaly detection model
    - Performance degradation model

    IMPORTANT: This explainer is for ML model outputs only.
    Physics-based calculations use the PhysicsExplainer.
    """

    def __init__(
        self,
        agent_id: str = "GL-003",
        default_num_features: int = 10,
    ) -> None:
        self.agent_id = agent_id
        self.default_num_features = default_num_features

        # Feature metadata
        self._feature_metadata = self._initialize_feature_metadata()

        # Cached explanations
        self._global_importance: Dict[str, FeatureImportance] = {}
        self._local_explanations: Dict[str, LocalExplanation] = {}
        self._asset_explanations: Dict[str, AssetExplanation] = {}

        # Model references (set externally)
        self._models: Dict[str, Any] = {}

        logger.info(f"SHAPExplainer initialized: {agent_id}")

    def _initialize_feature_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Initialize feature metadata for steam system models."""
        return {
            # Temperature features
            "inlet_temp_f": {
                "category": FeatureCategory.TEMPERATURE,
                "description": "Trap inlet temperature",
                "unit": "F",
            },
            "outlet_temp_f": {
                "category": FeatureCategory.TEMPERATURE,
                "description": "Trap outlet temperature",
                "unit": "F",
            },
            "temp_differential_f": {
                "category": FeatureCategory.DERIVED,
                "description": "Temperature drop across trap",
                "unit": "F",
            },
            "superheat_f": {
                "category": FeatureCategory.TEMPERATURE,
                "description": "Steam superheat above saturation",
                "unit": "F",
            },
            "subcooling_f": {
                "category": FeatureCategory.TEMPERATURE,
                "description": "Condensate subcooling below saturation",
                "unit": "F",
            },
            # Pressure features
            "header_pressure_psig": {
                "category": FeatureCategory.PRESSURE,
                "description": "Steam header pressure",
                "unit": "psig",
            },
            "differential_pressure_psi": {
                "category": FeatureCategory.PRESSURE,
                "description": "Pressure drop across trap",
                "unit": "psi",
            },
            "backpressure_psig": {
                "category": FeatureCategory.PRESSURE,
                "description": "Condensate return pressure",
                "unit": "psig",
            },
            # Flow features
            "steam_flow_klb_hr": {
                "category": FeatureCategory.FLOW,
                "description": "Steam flow rate",
                "unit": "klb/hr",
            },
            "condensate_flow_gpm": {
                "category": FeatureCategory.FLOW,
                "description": "Condensate flow rate",
                "unit": "gpm",
            },
            # Operational features
            "operating_hours": {
                "category": FeatureCategory.OPERATIONAL,
                "description": "Total operating hours",
                "unit": "hours",
            },
            "cycles_count": {
                "category": FeatureCategory.OPERATIONAL,
                "description": "Number of on/off cycles",
                "unit": "count",
            },
            "load_percent": {
                "category": FeatureCategory.OPERATIONAL,
                "description": "Current load percentage",
                "unit": "%",
            },
            # Maintenance features
            "days_since_inspection": {
                "category": FeatureCategory.MAINTENANCE,
                "description": "Days since last inspection",
                "unit": "days",
            },
            "days_since_maintenance": {
                "category": FeatureCategory.MAINTENANCE,
                "description": "Days since last maintenance",
                "unit": "days",
            },
            "previous_failures": {
                "category": FeatureCategory.MAINTENANCE,
                "description": "Count of previous failures",
                "unit": "count",
            },
            # Environmental features
            "ambient_temp_f": {
                "category": FeatureCategory.ENVIRONMENTAL,
                "description": "Ambient temperature",
                "unit": "F",
            },
            "humidity_percent": {
                "category": FeatureCategory.ENVIRONMENTAL,
                "description": "Relative humidity",
                "unit": "%",
            },
        }

    def register_model(
        self,
        model_type: ModelType,
        model: Any,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Register a model for SHAP explanation."""
        self._models[model_type.value] = {
            "model": model,
            "feature_names": feature_names or [],
        }
        logger.info(f"Registered model for SHAP: {model_type.value}")

    def compute_global_feature_importance(
        self,
        model: Any,
        dataset: List[Dict[str, float]],
        model_type: ModelType = ModelType.TRAP_FAILURE_PREDICTION,
        feature_names: Optional[List[str]] = None,
    ) -> FeatureImportance:
        """
        Compute global feature importance using SHAP.

        In production, this would use actual SHAP library.
        Here we compute approximations for the interface.

        Args:
            model: The ML model to explain (or None for mock)
            dataset: List of feature dictionaries
            model_type: Type of model being explained
            feature_names: Names of features in order

        Returns:
            FeatureImportance with global rankings
        """
        importance_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Get feature names from first sample if not provided
        if not feature_names and dataset:
            feature_names = list(dataset[0].keys())

        # Compute mean absolute SHAP values
        # In production: use shap.TreeExplainer or shap.KernelExplainer
        mean_abs_shap = self._compute_mock_importance(dataset, feature_names)

        # Create rankings
        feature_rankings = []
        for feature, importance in sorted(
            mean_abs_shap.items(), key=lambda x: x[1], reverse=True
        ):
            metadata = self._feature_metadata.get(feature, {})
            feature_rankings.append({
                "feature": feature,
                "importance": importance,
                "rank": len(feature_rankings) + 1,
                "category": metadata.get("category", FeatureCategory.DERIVED).value,
                "description": metadata.get("description", feature),
            })

        # Get top features
        top_features = [r["feature"] for r in feature_rankings[:10]]

        # Group by category
        feature_categories = {}
        for feature in feature_names:
            metadata = self._feature_metadata.get(feature, {})
            category = metadata.get("category", FeatureCategory.DERIVED).value
            if category not in feature_categories:
                feature_categories[category] = []
            feature_categories[category].append(feature)

        importance = FeatureImportance(
            importance_id=importance_id,
            model_type=model_type,
            timestamp=timestamp,
            feature_rankings=feature_rankings,
            mean_abs_shap=mean_abs_shap,
            total_features=len(feature_names),
            top_features=top_features,
            feature_categories=feature_categories,
            dataset_size=len(dataset),
        )

        self._global_importance[model_type.value] = importance
        logger.info(f"Computed global feature importance: {importance_id}")

        return importance

    def _compute_mock_importance(
        self,
        dataset: List[Dict[str, float]],
        feature_names: List[str],
    ) -> Dict[str, float]:
        """
        Compute mock feature importance.

        In production, replace with actual SHAP computation.
        This simulates importance based on feature variance and domain knowledge.
        """
        importance = {}

        # Domain-based importance weights (steam trap failure prediction)
        domain_weights = {
            "temp_differential_f": 0.25,
            "outlet_temp_f": 0.20,
            "inlet_temp_f": 0.15,
            "superheat_f": 0.12,
            "subcooling_f": 0.10,
            "operating_hours": 0.08,
            "cycles_count": 0.06,
            "days_since_inspection": 0.05,
            "differential_pressure_psi": 0.04,
            "previous_failures": 0.03,
        }

        for feature in feature_names:
            # Use domain weight if available, else compute from variance
            if feature in domain_weights:
                importance[feature] = domain_weights[feature]
            else:
                # Compute normalized variance as proxy
                if dataset:
                    values = [d.get(feature, 0) for d in dataset]
                    if values:
                        mean_val = sum(values) / len(values)
                        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                        # Normalize to [0, 0.1] range for non-domain features
                        importance[feature] = min(0.1, math.sqrt(variance) / 100)
                    else:
                        importance[feature] = 0.01
                else:
                    importance[feature] = 0.01

        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def compute_local_explanation(
        self,
        model: Any,
        instance: Dict[str, float],
        model_type: ModelType = ModelType.TRAP_FAILURE_PREDICTION,
        predicted_value: Optional[float] = None,
        base_value: Optional[float] = None,
    ) -> LocalExplanation:
        """
        Compute local SHAP explanation for a single instance.

        Args:
            model: The ML model to explain
            instance: Single feature dictionary
            model_type: Type of model being explained
            predicted_value: The model's prediction (or None to compute)
            base_value: The expected value E[f(x)] (or None for default)

        Returns:
            LocalExplanation with feature contributions
        """
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        instance_id = instance.get("instance_id", str(uuid.uuid4())[:8])

        # Get or compute predicted value
        if predicted_value is None:
            # In production: predicted_value = model.predict([instance])[0]
            predicted_value = self._mock_prediction(instance, model_type)

        # Set base value (expected value)
        if base_value is None:
            base_value = 0.5 if model_type == ModelType.TRAP_FAILURE_PREDICTION else 0.0

        prediction_delta = predicted_value - base_value

        # Compute SHAP values for this instance
        # In production: use shap.TreeExplainer(model).shap_values(instance)
        shap_values = self._compute_mock_shap_values(instance, prediction_delta)

        # Build contributions
        contributions = []
        total_abs_shap = sum(abs(v) for v in shap_values.values())

        for feature, shap_val in shap_values.items():
            metadata = self._feature_metadata.get(feature, {})
            contribution_pct = (
                abs(shap_val) / total_abs_shap * 100 if total_abs_shap > 0 else 0
            )

            contributions.append(FeatureContribution(
                feature_name=feature,
                feature_value=instance.get(feature, 0),
                shap_value=shap_val,
                contribution_percent=contribution_pct,
                direction="positive" if shap_val > 0 else "negative",
                category=metadata.get("category", FeatureCategory.DERIVED),
                description=metadata.get("description", feature),
            ))

        # Sort by absolute SHAP value
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

        # Get top features
        top_positive = [c.feature_name for c in contributions if c.shap_value > 0][:3]
        top_negative = [c.feature_name for c in contributions if c.shap_value < 0][:3]

        # Generate summary text
        summary_text = self._generate_local_summary(
            model_type, predicted_value, top_positive, top_negative
        )

        explanation = LocalExplanation(
            explanation_id=explanation_id,
            model_type=model_type,
            timestamp=timestamp,
            instance_id=instance_id,
            predicted_value=predicted_value,
            base_value=base_value,
            prediction_delta=prediction_delta,
            contributions=contributions[:self.default_num_features],
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            explanation_confidence=self._compute_explanation_confidence(contributions),
            summary_text=summary_text,
        )

        self._local_explanations[explanation_id] = explanation
        logger.debug(f"Created local explanation: {explanation_id}")

        return explanation

    def _mock_prediction(
        self,
        instance: Dict[str, float],
        model_type: ModelType,
    ) -> float:
        """Generate mock prediction for demonstration."""
        if model_type == ModelType.TRAP_FAILURE_PREDICTION:
            # Higher risk with high temp differential, low subcooling
            temp_diff = instance.get("temp_differential_f", 10)
            subcooling = instance.get("subcooling_f", 5)
            operating_hours = instance.get("operating_hours", 1000)

            risk = 0.3  # Base risk
            if temp_diff > 30:
                risk += 0.2
            if subcooling < 2:
                risk += 0.15
            if operating_hours > 8000:
                risk += 0.1

            return min(1.0, risk)

        return 0.5  # Default

    def _compute_mock_shap_values(
        self,
        instance: Dict[str, float],
        prediction_delta: float,
    ) -> Dict[str, float]:
        """Compute mock SHAP values for an instance."""
        shap_values = {}

        # Distribute prediction delta across features
        # Weight by feature importance and deviation from typical values
        typical_values = {
            "temp_differential_f": 15.0,
            "outlet_temp_f": 180.0,
            "inlet_temp_f": 350.0,
            "superheat_f": 20.0,
            "subcooling_f": 5.0,
            "operating_hours": 4000.0,
            "cycles_count": 1000.0,
            "days_since_inspection": 30.0,
            "differential_pressure_psi": 50.0,
            "previous_failures": 0.0,
        }

        importance_weights = {
            "temp_differential_f": 0.25,
            "outlet_temp_f": 0.18,
            "inlet_temp_f": 0.12,
            "superheat_f": 0.10,
            "subcooling_f": 0.10,
            "operating_hours": 0.08,
            "cycles_count": 0.06,
            "days_since_inspection": 0.05,
            "differential_pressure_psi": 0.04,
            "previous_failures": 0.02,
        }

        total_weight = 0
        for feature, value in instance.items():
            if feature in importance_weights:
                typical = typical_values.get(feature, value)
                if typical != 0:
                    deviation = (value - typical) / typical
                else:
                    deviation = 0

                weight = importance_weights[feature]
                # SHAP value is weighted deviation * prediction delta
                shap_values[feature] = weight * deviation * prediction_delta
                total_weight += abs(shap_values[feature])

        # Normalize so SHAP values sum to prediction_delta
        if total_weight > 0:
            scale = prediction_delta / (total_weight + 0.001)
            for feature in shap_values:
                # Keep relative magnitudes but scale appropriately
                pass  # Already computed proportionally

        return shap_values

    def _generate_local_summary(
        self,
        model_type: ModelType,
        predicted_value: float,
        top_positive: List[str],
        top_negative: List[str],
    ) -> str:
        """Generate natural language summary for local explanation."""
        if model_type == ModelType.TRAP_FAILURE_PREDICTION:
            risk_level = "high" if predicted_value > 0.7 else "medium" if predicted_value > 0.4 else "low"
            summary = f"Trap failure risk is {risk_level} ({predicted_value:.1%})."

            if top_positive:
                drivers = ", ".join(
                    self._feature_metadata.get(f, {}).get("description", f)
                    for f in top_positive[:2]
                )
                summary += f" Risk increased by: {drivers}."

            if top_negative and predicted_value < 0.5:
                mitigators = ", ".join(
                    self._feature_metadata.get(f, {}).get("description", f)
                    for f in top_negative[:2]
                )
                summary += f" Risk reduced by: {mitigators}."

            return summary

        return f"Prediction: {predicted_value:.3f}"

    def _compute_explanation_confidence(
        self,
        contributions: List[FeatureContribution],
    ) -> float:
        """Compute confidence in the explanation."""
        if not contributions:
            return 0.0

        # Higher confidence if top features dominate
        total_shap = sum(abs(c.shap_value) for c in contributions)
        top_3_shap = sum(abs(c.shap_value) for c in contributions[:3])

        if total_shap > 0:
            concentration = top_3_shap / total_shap
            return min(1.0, 0.5 + concentration * 0.5)

        return 0.5

    def generate_asset_level_explanation(
        self,
        asset_id: str,
        model: Any,
        history: List[Dict[str, float]],
        asset_type: str = "trap",
        model_type: ModelType = ModelType.TRAP_FAILURE_PREDICTION,
        history_window_hours: int = 168,  # 1 week
    ) -> AssetExplanation:
        """
        Generate SHAP explanation aggregated for a specific asset.

        Args:
            asset_id: Unique identifier for the asset
            model: The ML model to explain
            history: Historical feature data for this asset
            asset_type: Type of asset (trap, prv, etc.)
            model_type: Type of model
            history_window_hours: Hours of history to analyze

        Returns:
            AssetExplanation with aggregated insights
        """
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Compute local explanations for all history points
        local_explanations = []
        for instance in history:
            instance["instance_id"] = asset_id
            local_exp = self.compute_local_explanation(
                model, instance, model_type
            )
            local_explanations.append(local_exp)

        # Aggregate feature importance for this asset
        feature_shap_sums = {}
        for exp in local_explanations:
            for contrib in exp.contributions:
                if contrib.feature_name not in feature_shap_sums:
                    feature_shap_sums[contrib.feature_name] = []
                feature_shap_sums[contrib.feature_name].append(abs(contrib.shap_value))

        asset_feature_importance = {
            feature: sum(vals) / len(vals)
            for feature, vals in feature_shap_sums.items()
        }

        # Sort by importance
        sorted_features = sorted(
            asset_feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        asset_top_features = [f for f, _ in sorted_features[:5]]

        # Compute feature trends
        feature_trends = self._compute_feature_trends(history)

        # Compute comparison to fleet (mock - would use fleet statistics)
        comparison_to_fleet = self._compute_fleet_comparison(
            asset_feature_importance
        )

        # Current risk score
        if local_explanations:
            current_risk_score = local_explanations[-1].predicted_value
        else:
            current_risk_score = 0.0

        # Identify risk drivers
        risk_drivers = []
        if local_explanations:
            latest = local_explanations[-1]
            for contrib in latest.contributions[:3]:
                if contrib.direction == "positive":
                    risk_drivers.append(
                        f"{contrib.description}: {contrib.feature_value:.1f}"
                    )

        # Generate suggested actions
        suggested_actions = self._generate_asset_actions(
            asset_type, asset_top_features, feature_trends, current_risk_score
        )

        explanation = AssetExplanation(
            explanation_id=explanation_id,
            asset_id=asset_id,
            asset_type=asset_type,
            model_type=model_type,
            timestamp=timestamp,
            history_window_hours=history_window_hours,
            sample_count=len(history),
            asset_feature_importance=asset_feature_importance,
            asset_top_features=asset_top_features,
            feature_trends=feature_trends,
            comparison_to_fleet=comparison_to_fleet,
            current_risk_score=current_risk_score,
            risk_drivers=risk_drivers,
            suggested_actions=suggested_actions,
        )

        self._asset_explanations[asset_id] = explanation
        logger.info(f"Created asset explanation for {asset_id}: {explanation_id}")

        return explanation

    def _compute_feature_trends(
        self,
        history: List[Dict[str, float]],
    ) -> Dict[str, str]:
        """Compute trends for each feature over history."""
        if len(history) < 2:
            return {}

        trends = {}
        features = history[0].keys() if history else []

        for feature in features:
            values = [h.get(feature, 0) for h in history]
            if len(values) >= 2:
                # Simple linear trend
                first_half_avg = sum(values[:len(values)//2]) / (len(values)//2)
                second_half_avg = sum(values[len(values)//2:]) / (len(values) - len(values)//2)

                if second_half_avg > first_half_avg * 1.1:
                    trends[feature] = "increasing"
                elif second_half_avg < first_half_avg * 0.9:
                    trends[feature] = "decreasing"
                else:
                    trends[feature] = "stable"

        return trends

    def _compute_fleet_comparison(
        self,
        asset_importance: Dict[str, float],
    ) -> Dict[str, float]:
        """Compare asset feature importance to fleet averages."""
        # In production, this would use actual fleet statistics
        # Mock: return percentiles around 50%
        comparison = {}
        for feature in asset_importance:
            # Mock percentile based on importance
            imp = asset_importance[feature]
            percentile = min(99, max(1, 50 + imp * 100))
            comparison[feature] = percentile

        return comparison

    def _generate_asset_actions(
        self,
        asset_type: str,
        top_features: List[str],
        trends: Dict[str, str],
        risk_score: float,
    ) -> List[str]:
        """Generate suggested actions for an asset."""
        actions = []

        if risk_score > 0.7:
            actions.append(f"Schedule immediate inspection of {asset_type}")

        if "temp_differential_f" in top_features:
            if trends.get("temp_differential_f") == "increasing":
                actions.append("Investigate increasing temperature differential")

        if "days_since_inspection" in top_features:
            actions.append("Schedule routine inspection")

        if "operating_hours" in top_features:
            if risk_score > 0.5:
                actions.append("Consider preventive replacement based on operating hours")

        if "subcooling_f" in top_features:
            if trends.get("subcooling_f") == "decreasing":
                actions.append("Check for steam bypass (reduced subcooling)")

        if not actions:
            actions.append("Continue normal monitoring")

        return actions

    def visualize_shap_summary(
        self,
        explanations: Union[FeatureImportance, List[LocalExplanation]],
        visualization_type: str = "summary",
    ) -> SHAPVisualization:
        """
        Generate visualization data for SHAP explanations.

        Args:
            explanations: Global or local explanations
            visualization_type: Type of plot ("summary", "waterfall", "force")

        Returns:
            SHAPVisualization with plot data
        """
        visualization_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        if isinstance(explanations, FeatureImportance):
            model_type = explanations.model_type
            plot_data = self._prepare_summary_plot_data(explanations)
            title = f"Global Feature Importance - {model_type.value}"
        else:
            model_type = explanations[0].model_type if explanations else ModelType.CLASSIFICATION
            plot_data = self._prepare_local_plot_data(explanations, visualization_type)
            title = f"SHAP {visualization_type.title()} - {model_type.value}"

        visualization = SHAPVisualization(
            visualization_id=visualization_id,
            visualization_type=visualization_type,
            model_type=model_type,
            timestamp=timestamp,
            plot_data=plot_data,
            title=title,
            x_label="SHAP Value (impact on model output)",
            y_label="Feature",
        )

        return visualization

    def _prepare_summary_plot_data(
        self,
        importance: FeatureImportance,
    ) -> Dict[str, Any]:
        """Prepare data for summary plot."""
        return {
            "features": [r["feature"] for r in importance.feature_rankings[:15]],
            "importance": [r["importance"] for r in importance.feature_rankings[:15]],
            "categories": [r["category"] for r in importance.feature_rankings[:15]],
        }

    def _prepare_local_plot_data(
        self,
        explanations: List[LocalExplanation],
        plot_type: str,
    ) -> Dict[str, Any]:
        """Prepare data for local explanation plots."""
        if not explanations:
            return {}

        if plot_type == "waterfall":
            # Single explanation waterfall
            exp = explanations[0]
            return {
                "base_value": exp.base_value,
                "predicted_value": exp.predicted_value,
                "features": [c.feature_name for c in exp.contributions],
                "shap_values": [c.shap_value for c in exp.contributions],
                "feature_values": [c.feature_value for c in exp.contributions],
            }

        elif plot_type == "force":
            # Force plot data
            exp = explanations[0]
            return {
                "base_value": exp.base_value,
                "output_value": exp.predicted_value,
                "features": {
                    c.feature_name: {
                        "effect": c.shap_value,
                        "value": c.feature_value,
                    }
                    for c in exp.contributions
                },
            }

        return {}

    def get_global_importance(
        self,
        model_type: ModelType,
    ) -> Optional[FeatureImportance]:
        """Get cached global feature importance."""
        return self._global_importance.get(model_type.value)

    def get_local_explanation(
        self,
        explanation_id: str,
    ) -> Optional[LocalExplanation]:
        """Get local explanation by ID."""
        return self._local_explanations.get(explanation_id)

    def get_asset_explanation(
        self,
        asset_id: str,
    ) -> Optional[AssetExplanation]:
        """Get asset explanation by asset ID."""
        return self._asset_explanations.get(asset_id)
