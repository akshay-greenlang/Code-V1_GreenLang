"""
GL-003 UNIFIEDSTEAM - LIME-Based Explainer

Provides LIME (Local Interpretable Model-agnostic Explanations) for single events:
- Single prediction explanations with local linear approximations
- Anomaly detection explanations
- Counterfactual generation (what would need to change)

LIME is particularly useful for:
- Explaining individual anomaly alerts
- Understanding why a specific trap was flagged
- Generating "what-if" counterfactual scenarios

IMPORTANT: LIME is used ONLY for ML model explanations, NOT for physics calculations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum
import logging
import uuid
import math
import random

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies in steam systems."""
    TEMPERATURE_ANOMALY = "temperature_anomaly"
    PRESSURE_ANOMALY = "pressure_anomaly"
    FLOW_ANOMALY = "flow_anomaly"
    EFFICIENCY_ANOMALY = "efficiency_anomaly"
    TRAP_FAILURE = "trap_failure"
    LEAK_DETECTED = "leak_detected"
    FOULING_DETECTED = "fouling_detected"
    CONTROL_ANOMALY = "control_anomaly"


class CounterfactualType(Enum):
    """Types of counterfactual outcomes."""
    REDUCE_RISK = "reduce_risk"
    ACHIEVE_TARGET = "achieve_target"
    AVOID_ANOMALY = "avoid_anomaly"
    OPTIMIZE_PERFORMANCE = "optimize_performance"


@dataclass
class LIMEFeatureWeight:
    """Single feature weight from LIME."""
    feature_name: str
    feature_value: float
    weight: float  # Linear coefficient in local model
    contribution: float  # weight * (value - baseline)
    importance_rank: int
    direction: str  # "positive", "negative"
    description: str = ""
    baseline_value: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "weight": self.weight,
            "contribution": self.contribution,
            "importance_rank": self.importance_rank,
            "direction": self.direction,
            "description": self.description,
            "baseline_value": self.baseline_value,
        }


@dataclass
class LIMEExplanation:
    """LIME explanation for a single prediction."""
    explanation_id: str
    timestamp: datetime
    instance_id: str

    # Prediction
    predicted_value: float
    predicted_class: Optional[str] = None
    prediction_probability: float = 0.0

    # Local linear model
    intercept: float = 0.0
    feature_weights: List[LIMEFeatureWeight] = field(default_factory=list)

    # Model quality
    local_model_r2: float = 0.0  # R-squared of local linear model
    sample_size: int = 0  # Number of perturbed samples used

    # Summary
    top_positive_features: List[str] = field(default_factory=list)
    top_negative_features: List[str] = field(default_factory=list)
    explanation_text: str = ""

    # Confidence
    explanation_confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp.isoformat(),
            "instance_id": self.instance_id,
            "predicted_value": self.predicted_value,
            "predicted_class": self.predicted_class,
            "prediction_probability": self.prediction_probability,
            "intercept": self.intercept,
            "feature_weights": [fw.to_dict() for fw in self.feature_weights],
            "local_model_r2": self.local_model_r2,
            "sample_size": self.sample_size,
            "top_positive_features": self.top_positive_features,
            "top_negative_features": self.top_negative_features,
            "explanation_text": self.explanation_text,
            "explanation_confidence": self.explanation_confidence,
        }


@dataclass
class AnomalyContext:
    """Context information for an anomaly."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: str  # "low", "medium", "high", "critical"
    detected_at: datetime

    # Location
    asset_id: str
    asset_type: str
    location: str

    # Anomaly details
    anomaly_score: float
    threshold_used: float
    deviation_percent: float

    # Historical context
    hours_since_last_anomaly: Optional[float] = None
    similar_anomalies_last_30d: int = 0

    def to_dict(self) -> Dict:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
            "asset_id": self.asset_id,
            "asset_type": self.asset_type,
            "location": self.location,
            "anomaly_score": self.anomaly_score,
            "threshold_used": self.threshold_used,
            "deviation_percent": self.deviation_percent,
            "hours_since_last_anomaly": self.hours_since_last_anomaly,
            "similar_anomalies_last_30d": self.similar_anomalies_last_30d,
        }


@dataclass
class AnomalyExplanation:
    """Explanation for an anomaly detection result."""
    explanation_id: str
    timestamp: datetime

    # Anomaly context
    context: AnomalyContext

    # LIME explanation
    lime_explanation: LIMEExplanation

    # Anomaly-specific interpretation
    primary_cause: str
    contributing_factors: List[str]
    normal_range: Dict[str, Tuple[float, float]]  # Feature -> (min, max)
    current_values: Dict[str, float]
    out_of_range_features: List[str]

    # Recommended actions
    recommended_actions: List[str]
    urgency: str  # "immediate", "soon", "scheduled", "monitor"

    # Similar past anomalies
    similar_past_anomalies: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.to_dict(),
            "lime_explanation": self.lime_explanation.to_dict(),
            "primary_cause": self.primary_cause,
            "contributing_factors": self.contributing_factors,
            "normal_range": {k: list(v) for k, v in self.normal_range.items()},
            "current_values": self.current_values,
            "out_of_range_features": self.out_of_range_features,
            "recommended_actions": self.recommended_actions,
            "urgency": self.urgency,
            "similar_past_anomalies": self.similar_past_anomalies,
        }


@dataclass
class FeatureChange:
    """A single feature change in a counterfactual."""
    feature_name: str
    current_value: float
    suggested_value: float
    change_amount: float
    change_percent: float
    feasibility: str  # "easy", "moderate", "difficult", "infeasible"
    implementation_note: str = ""

    def to_dict(self) -> Dict:
        return {
            "feature_name": self.feature_name,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "change_amount": self.change_amount,
            "change_percent": self.change_percent,
            "feasibility": self.feasibility,
            "implementation_note": self.implementation_note,
        }


@dataclass
class Counterfactual:
    """Counterfactual explanation - what changes would achieve target outcome."""
    counterfactual_id: str
    timestamp: datetime
    instance_id: str
    counterfactual_type: CounterfactualType

    # Current state
    current_prediction: float
    current_class: Optional[str] = None

    # Target state
    target_prediction: float
    target_class: Optional[str] = None

    # Required changes
    required_changes: List[FeatureChange] = field(default_factory=list)
    total_change_cost: float = 0.0  # Aggregate "cost" of changes

    # Achieved outcome
    counterfactual_prediction: float = 0.0
    prediction_achieved: bool = False

    # Uncertainty
    confidence: float = 0.0
    uncertainty_range: Tuple[float, float] = (0.0, 0.0)

    # Natural language
    explanation_text: str = ""
    action_summary: str = ""

    def to_dict(self) -> Dict:
        return {
            "counterfactual_id": self.counterfactual_id,
            "timestamp": self.timestamp.isoformat(),
            "instance_id": self.instance_id,
            "counterfactual_type": self.counterfactual_type.value,
            "current_prediction": self.current_prediction,
            "current_class": self.current_class,
            "target_prediction": self.target_prediction,
            "target_class": self.target_class,
            "required_changes": [c.to_dict() for c in self.required_changes],
            "total_change_cost": self.total_change_cost,
            "counterfactual_prediction": self.counterfactual_prediction,
            "prediction_achieved": self.prediction_achieved,
            "confidence": self.confidence,
            "uncertainty_range": list(self.uncertainty_range),
            "explanation_text": self.explanation_text,
            "action_summary": self.action_summary,
        }


class LIMEExplainer:
    """
    LIME-based explainer for single-event explanations.

    Features:
    - Single prediction explanations
    - Anomaly detection explanations
    - Counterfactual generation

    IMPORTANT: This explains ML model outputs only.
    Physics calculations use the PhysicsExplainer.
    """

    def __init__(
        self,
        agent_id: str = "GL-003",
        num_samples: int = 1000,
        num_features: int = 10,
    ) -> None:
        self.agent_id = agent_id
        self.num_samples = num_samples
        self.num_features = num_features

        # Feature metadata
        self._feature_metadata = self._initialize_feature_metadata()

        # Normal operating ranges for anomaly detection
        self._normal_ranges = self._initialize_normal_ranges()

        # Feature change feasibility
        self._feasibility_map = self._initialize_feasibility_map()

        # Cached explanations
        self._lime_explanations: Dict[str, LIMEExplanation] = {}
        self._anomaly_explanations: Dict[str, AnomalyExplanation] = {}
        self._counterfactuals: Dict[str, Counterfactual] = {}

        logger.info(f"LIMEExplainer initialized: {agent_id}")

    def _initialize_feature_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Initialize feature metadata."""
        return {
            "inlet_temp_f": {
                "description": "Trap inlet temperature",
                "unit": "F",
                "typical_range": (300, 400),
            },
            "outlet_temp_f": {
                "description": "Trap outlet temperature",
                "unit": "F",
                "typical_range": (150, 220),
            },
            "temp_differential_f": {
                "description": "Temperature drop across trap",
                "unit": "F",
                "typical_range": (100, 200),
            },
            "superheat_f": {
                "description": "Steam superheat",
                "unit": "F",
                "typical_range": (10, 50),
            },
            "subcooling_f": {
                "description": "Condensate subcooling",
                "unit": "F",
                "typical_range": (2, 15),
            },
            "header_pressure_psig": {
                "description": "Header pressure",
                "unit": "psig",
                "typical_range": (100, 300),
            },
            "differential_pressure_psi": {
                "description": "Pressure drop",
                "unit": "psi",
                "typical_range": (20, 80),
            },
            "operating_hours": {
                "description": "Operating hours",
                "unit": "hours",
                "typical_range": (0, 10000),
            },
            "days_since_inspection": {
                "description": "Days since inspection",
                "unit": "days",
                "typical_range": (0, 90),
            },
            "steam_flow_klb_hr": {
                "description": "Steam flow rate",
                "unit": "klb/hr",
                "typical_range": (10, 100),
            },
        }

    def _initialize_normal_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Initialize normal operating ranges for anomaly detection."""
        return {
            "inlet_temp_f": (280, 420),
            "outlet_temp_f": (140, 250),
            "temp_differential_f": (80, 220),
            "superheat_f": (5, 60),
            "subcooling_f": (1, 20),
            "header_pressure_psig": (80, 350),
            "differential_pressure_psi": (15, 100),
            "operating_hours": (0, 12000),
            "days_since_inspection": (0, 120),
        }

    def _initialize_feasibility_map(self) -> Dict[str, Dict[str, str]]:
        """Map features to change feasibility."""
        return {
            "inlet_temp_f": {
                "feasibility": "moderate",
                "note": "Adjust boiler/header conditions",
            },
            "outlet_temp_f": {
                "feasibility": "difficult",
                "note": "Requires trap replacement or process change",
            },
            "temp_differential_f": {
                "feasibility": "difficult",
                "note": "Result of system conditions",
            },
            "superheat_f": {
                "feasibility": "moderate",
                "note": "Adjust desuperheater setpoint",
            },
            "subcooling_f": {
                "feasibility": "moderate",
                "note": "Check trap operation",
            },
            "header_pressure_psig": {
                "feasibility": "moderate",
                "note": "Adjust pressure setpoint",
            },
            "differential_pressure_psi": {
                "feasibility": "easy",
                "note": "Adjust valve or check for blockage",
            },
            "operating_hours": {
                "feasibility": "infeasible",
                "note": "Cannot change operating history",
            },
            "days_since_inspection": {
                "feasibility": "easy",
                "note": "Schedule inspection",
            },
        }

    def explain_single_prediction(
        self,
        model: Any,
        instance: Dict[str, float],
        num_features: Optional[int] = None,
        predicted_value: Optional[float] = None,
    ) -> LIMEExplanation:
        """
        Generate LIME explanation for a single prediction.

        Args:
            model: The ML model to explain (or None for mock)
            instance: Feature dictionary for the instance
            num_features: Number of top features to include
            predicted_value: Pre-computed prediction (or None to compute)

        Returns:
            LIMEExplanation with local linear approximation
        """
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        instance_id = instance.get("instance_id", str(uuid.uuid4())[:8])
        num_features = num_features or self.num_features

        # Get or compute prediction
        if predicted_value is None:
            predicted_value = self._mock_prediction(instance)

        # Compute LIME weights through perturbation sampling
        # In production: use lime.lime_tabular.LimeTabularExplainer
        weights, intercept, r2 = self._compute_lime_weights(
            model, instance, predicted_value
        )

        # Build feature weight objects
        feature_weights = []
        sorted_features = sorted(
            weights.items(), key=lambda x: abs(x[1]), reverse=True
        )

        for rank, (feature, weight) in enumerate(sorted_features[:num_features], 1):
            value = instance.get(feature, 0)
            metadata = self._feature_metadata.get(feature, {})
            baseline = sum(metadata.get("typical_range", (value, value))) / 2

            contribution = weight * (value - baseline)

            feature_weights.append(LIMEFeatureWeight(
                feature_name=feature,
                feature_value=value,
                weight=weight,
                contribution=contribution,
                importance_rank=rank,
                direction="positive" if weight > 0 else "negative",
                description=metadata.get("description", feature),
                baseline_value=baseline,
            ))

        # Identify top features
        top_positive = [
            fw.feature_name for fw in feature_weights
            if fw.direction == "positive"
        ][:3]
        top_negative = [
            fw.feature_name for fw in feature_weights
            if fw.direction == "negative"
        ][:3]

        # Generate explanation text
        explanation_text = self._generate_lime_text(
            predicted_value, feature_weights[:5]
        )

        # Compute confidence based on local model fit
        confidence = min(1.0, 0.5 + r2 * 0.5)

        explanation = LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=timestamp,
            instance_id=instance_id,
            predicted_value=predicted_value,
            predicted_class=self._value_to_class(predicted_value),
            prediction_probability=predicted_value,
            intercept=intercept,
            feature_weights=feature_weights,
            local_model_r2=r2,
            sample_size=self.num_samples,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            explanation_text=explanation_text,
            explanation_confidence=confidence,
        )

        self._lime_explanations[explanation_id] = explanation
        logger.debug(f"Created LIME explanation: {explanation_id}")

        return explanation

    def _mock_prediction(self, instance: Dict[str, float]) -> float:
        """Generate mock prediction for demonstration."""
        # Simple risk model based on key features
        risk = 0.3

        temp_diff = instance.get("temp_differential_f", 100)
        if temp_diff > 180:
            risk += 0.2
        elif temp_diff < 80:
            risk += 0.15

        subcooling = instance.get("subcooling_f", 5)
        if subcooling < 2:
            risk += 0.2

        operating_hours = instance.get("operating_hours", 4000)
        if operating_hours > 8000:
            risk += 0.1

        days_since = instance.get("days_since_inspection", 30)
        if days_since > 90:
            risk += 0.1

        return min(1.0, max(0.0, risk))

    def _compute_lime_weights(
        self,
        model: Any,
        instance: Dict[str, float],
        predicted_value: float,
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Compute LIME weights through perturbation.

        In production, use lime library. This is a simplified approximation.

        Returns:
            (weights dict, intercept, r-squared)
        """
        # Mock LIME weights based on feature importance and instance values
        weights = {}

        # Domain-based importance for steam traps
        importance = {
            "temp_differential_f": 0.30,
            "subcooling_f": 0.20,
            "outlet_temp_f": 0.15,
            "operating_hours": 0.10,
            "days_since_inspection": 0.08,
            "inlet_temp_f": 0.07,
            "header_pressure_psig": 0.05,
            "differential_pressure_psi": 0.03,
            "superheat_f": 0.02,
        }

        for feature, value in instance.items():
            if feature in importance:
                # Weight proportional to importance
                # Sign based on whether high values increase risk
                base_weight = importance[feature]

                if feature in ["temp_differential_f", "operating_hours", "days_since_inspection"]:
                    # Higher values increase risk
                    weights[feature] = base_weight * 0.01
                elif feature in ["subcooling_f"]:
                    # Lower values increase risk
                    weights[feature] = -base_weight * 0.02
                else:
                    weights[feature] = base_weight * 0.005

        # Intercept is the base prediction
        intercept = predicted_value * 0.3

        # Mock R-squared (higher is better local fit)
        r2 = 0.75 + random.uniform(0, 0.2)

        return weights, intercept, r2

    def _value_to_class(self, value: float) -> str:
        """Convert prediction value to class label."""
        if value >= 0.7:
            return "high_risk"
        elif value >= 0.4:
            return "medium_risk"
        else:
            return "low_risk"

    def _generate_lime_text(
        self,
        predicted_value: float,
        top_weights: List[LIMEFeatureWeight],
    ) -> str:
        """Generate explanation text from LIME weights."""
        risk_class = self._value_to_class(predicted_value)
        text = f"Prediction: {risk_class} ({predicted_value:.1%} risk score). "

        positive_factors = [
            fw for fw in top_weights if fw.direction == "positive"
        ][:2]
        negative_factors = [
            fw for fw in top_weights if fw.direction == "negative"
        ][:2]

        if positive_factors:
            factors = ", ".join(fw.description for fw in positive_factors)
            text += f"Risk drivers: {factors}. "

        if negative_factors:
            factors = ", ".join(fw.description for fw in negative_factors)
            text += f"Risk reducers: {factors}."

        return text

    def explain_anomaly_detection(
        self,
        anomaly_result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> AnomalyExplanation:
        """
        Generate explanation for an anomaly detection result.

        Args:
            anomaly_result: Result from anomaly detection model
            context: Additional context (asset info, history)

        Returns:
            AnomalyExplanation with cause analysis and recommendations
        """
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Build anomaly context
        anomaly_context = AnomalyContext(
            anomaly_id=anomaly_result.get("anomaly_id", str(uuid.uuid4())[:8]),
            anomaly_type=AnomalyType(
                anomaly_result.get("anomaly_type", "temperature_anomaly")
            ),
            severity=anomaly_result.get("severity", "medium"),
            detected_at=datetime.fromisoformat(
                anomaly_result.get("detected_at", timestamp.isoformat())
            ),
            asset_id=context.get("asset_id", "unknown"),
            asset_type=context.get("asset_type", "trap"),
            location=context.get("location", "unknown"),
            anomaly_score=anomaly_result.get("anomaly_score", 0.5),
            threshold_used=anomaly_result.get("threshold", 0.8),
            deviation_percent=anomaly_result.get("deviation_percent", 0.0),
            hours_since_last_anomaly=context.get("hours_since_last_anomaly"),
            similar_anomalies_last_30d=context.get("similar_anomalies_30d", 0),
        )

        # Get instance features
        instance = anomaly_result.get("features", {})

        # Generate LIME explanation for the anomaly detection
        lime_explanation = self.explain_single_prediction(
            model=None,
            instance=instance,
            predicted_value=anomaly_context.anomaly_score,
        )

        # Identify out-of-range features
        current_values = instance
        out_of_range = []
        for feature, value in current_values.items():
            if feature in self._normal_ranges:
                min_val, max_val = self._normal_ranges[feature]
                if value < min_val or value > max_val:
                    out_of_range.append(feature)

        # Determine primary cause
        primary_cause = self._determine_primary_cause(
            anomaly_context.anomaly_type, lime_explanation.feature_weights
        )

        # Contributing factors
        contributing_factors = [
            fw.description for fw in lime_explanation.feature_weights[:3]
            if fw.contribution > 0
        ]

        # Generate recommendations
        recommended_actions = self._generate_anomaly_recommendations(
            anomaly_context, out_of_range, lime_explanation
        )

        # Determine urgency
        urgency = self._determine_urgency(anomaly_context)

        explanation = AnomalyExplanation(
            explanation_id=explanation_id,
            timestamp=timestamp,
            context=anomaly_context,
            lime_explanation=lime_explanation,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            normal_range=self._normal_ranges,
            current_values=current_values,
            out_of_range_features=out_of_range,
            recommended_actions=recommended_actions,
            urgency=urgency,
        )

        self._anomaly_explanations[explanation_id] = explanation
        logger.info(f"Created anomaly explanation: {explanation_id}")

        return explanation

    def _determine_primary_cause(
        self,
        anomaly_type: AnomalyType,
        feature_weights: List[LIMEFeatureWeight],
    ) -> str:
        """Determine the primary cause of an anomaly."""
        # Map anomaly types to likely causes
        type_causes = {
            AnomalyType.TEMPERATURE_ANOMALY: "Unusual temperature pattern detected",
            AnomalyType.PRESSURE_ANOMALY: "Pressure deviation from normal range",
            AnomalyType.FLOW_ANOMALY: "Abnormal flow rate detected",
            AnomalyType.TRAP_FAILURE: "Steam trap malfunction likely",
            AnomalyType.LEAK_DETECTED: "Possible steam or condensate leak",
            AnomalyType.FOULING_DETECTED: "Heat transfer surface fouling",
            AnomalyType.CONTROL_ANOMALY: "Control system abnormality",
        }

        base_cause = type_causes.get(anomaly_type, "Abnormal system behavior")

        # Enhance with top feature
        if feature_weights:
            top_feature = feature_weights[0]
            if top_feature.contribution > 0.1:
                return f"{base_cause} - driven by {top_feature.description}"

        return base_cause

    def _generate_anomaly_recommendations(
        self,
        context: AnomalyContext,
        out_of_range: List[str],
        lime_exp: LIMEExplanation,
    ) -> List[str]:
        """Generate recommendations for an anomaly."""
        recommendations = []

        # Based on severity
        if context.severity == "critical":
            recommendations.append("IMMEDIATE: Dispatch operator to inspect")
        elif context.severity == "high":
            recommendations.append("Schedule inspection within 24 hours")

        # Based on anomaly type
        type_actions = {
            AnomalyType.TRAP_FAILURE: "Test trap with ultrasonic or temperature survey",
            AnomalyType.LEAK_DETECTED: "Perform visual inspection for steam leaks",
            AnomalyType.TEMPERATURE_ANOMALY: "Verify temperature sensor calibration",
            AnomalyType.PRESSURE_ANOMALY: "Check pressure gauge and PRV operation",
            AnomalyType.FOULING_DETECTED: "Schedule heat exchanger cleaning",
        }
        if context.anomaly_type in type_actions:
            recommendations.append(type_actions[context.anomaly_type])

        # Based on out-of-range features
        if "subcooling_f" in out_of_range:
            recommendations.append("Check for steam blow-through at trap")
        if "temp_differential_f" in out_of_range:
            recommendations.append("Verify condensate discharge is normal")
        if "days_since_inspection" in out_of_range:
            recommendations.append("Update inspection schedule for this trap group")

        # Default
        if not recommendations:
            recommendations.append("Monitor for continued anomalies")

        return recommendations

    def _determine_urgency(self, context: AnomalyContext) -> str:
        """Determine urgency of response needed."""
        if context.severity == "critical":
            return "immediate"
        elif context.severity == "high":
            return "soon"
        elif context.similar_anomalies_last_30d > 3:
            return "soon"
        elif context.severity == "medium":
            return "scheduled"
        else:
            return "monitor"

    def generate_counterfactual(
        self,
        instance: Dict[str, float],
        target_outcome: float,
        model: Any = None,
        counterfactual_type: CounterfactualType = CounterfactualType.REDUCE_RISK,
        max_changes: int = 5,
    ) -> Counterfactual:
        """
        Generate counterfactual: what changes would achieve target outcome.

        Args:
            instance: Current feature values
            target_outcome: Desired prediction value
            model: ML model (or None for mock)
            counterfactual_type: Type of counterfactual goal
            max_changes: Maximum number of features to change

        Returns:
            Counterfactual with required changes
        """
        counterfactual_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        instance_id = instance.get("instance_id", str(uuid.uuid4())[:8])

        # Get current prediction
        current_prediction = self._mock_prediction(instance)

        # Determine changes needed
        required_changes = self._compute_required_changes(
            instance, current_prediction, target_outcome, max_changes
        )

        # Compute total change cost
        total_cost = sum(
            abs(c.change_percent) for c in required_changes
            if c.feasibility != "infeasible"
        )

        # Verify counterfactual achieves target
        cf_instance = instance.copy()
        for change in required_changes:
            cf_instance[change.feature_name] = change.suggested_value
        cf_prediction = self._mock_prediction(cf_instance)

        prediction_achieved = abs(cf_prediction - target_outcome) < 0.1

        # Generate explanation text
        explanation_text = self._generate_cf_text(
            current_prediction, target_outcome, required_changes
        )
        action_summary = self._generate_action_summary(required_changes)

        # Estimate uncertainty
        uncertainty_range = (
            max(0, cf_prediction - 0.1),
            min(1, cf_prediction + 0.1),
        )

        counterfactual = Counterfactual(
            counterfactual_id=counterfactual_id,
            timestamp=timestamp,
            instance_id=instance_id,
            counterfactual_type=counterfactual_type,
            current_prediction=current_prediction,
            current_class=self._value_to_class(current_prediction),
            target_prediction=target_outcome,
            target_class=self._value_to_class(target_outcome),
            required_changes=required_changes,
            total_change_cost=total_cost,
            counterfactual_prediction=cf_prediction,
            prediction_achieved=prediction_achieved,
            confidence=0.7 if prediction_achieved else 0.4,
            uncertainty_range=uncertainty_range,
            explanation_text=explanation_text,
            action_summary=action_summary,
        )

        self._counterfactuals[counterfactual_id] = counterfactual
        logger.info(f"Created counterfactual: {counterfactual_id}")

        return counterfactual

    def _compute_required_changes(
        self,
        instance: Dict[str, float],
        current: float,
        target: float,
        max_changes: int,
    ) -> List[FeatureChange]:
        """Compute required feature changes to achieve target."""
        changes = []
        delta_needed = target - current

        # Priority features for reducing risk
        if delta_needed < 0:  # Need to reduce risk
            priority_features = [
                ("days_since_inspection", -0.5),  # Reduce days
                ("subcooling_f", 0.3),  # Increase subcooling
                ("operating_hours", -0.1),  # Can't change
            ]
        else:  # Need to increase (unusual case)
            priority_features = []

        for feature, impact in priority_features[:max_changes]:
            if feature not in instance:
                continue

            current_value = instance[feature]
            feasibility_info = self._feasibility_map.get(feature, {})
            feasibility = feasibility_info.get("feasibility", "moderate")
            note = feasibility_info.get("note", "")

            if feasibility == "infeasible":
                continue

            # Calculate suggested change
            # Target is proportional to needed delta
            if impact < 0:  # Decrease feature
                suggested = current_value * (1 + delta_needed * 0.5)
            else:  # Increase feature
                suggested = current_value * (1 - delta_needed * 0.5)

            # Clip to reasonable ranges
            metadata = self._feature_metadata.get(feature, {})
            typical_range = metadata.get("typical_range", (0, current_value * 2))
            suggested = max(typical_range[0], min(typical_range[1], suggested))

            change_amount = suggested - current_value
            change_percent = (
                (change_amount / current_value * 100) if current_value != 0 else 0
            )

            changes.append(FeatureChange(
                feature_name=feature,
                current_value=current_value,
                suggested_value=suggested,
                change_amount=change_amount,
                change_percent=change_percent,
                feasibility=feasibility,
                implementation_note=note,
            ))

        # Sort by feasibility
        feasibility_order = {"easy": 0, "moderate": 1, "difficult": 2, "infeasible": 3}
        changes.sort(key=lambda c: feasibility_order.get(c.feasibility, 2))

        return changes[:max_changes]

    def _generate_cf_text(
        self,
        current: float,
        target: float,
        changes: List[FeatureChange],
    ) -> str:
        """Generate counterfactual explanation text."""
        current_class = self._value_to_class(current)
        target_class = self._value_to_class(target)

        text = f"To move from {current_class} to {target_class}: "

        feasible_changes = [c for c in changes if c.feasibility != "infeasible"]
        if feasible_changes:
            change_strs = [
                f"{c.feature_name} from {c.current_value:.1f} to {c.suggested_value:.1f}"
                for c in feasible_changes[:3]
            ]
            text += "Adjust " + ", ".join(change_strs) + "."
        else:
            text += "No feasible changes identified."

        return text

    def _generate_action_summary(self, changes: List[FeatureChange]) -> str:
        """Generate action summary for counterfactual."""
        easy = [c for c in changes if c.feasibility == "easy"]
        moderate = [c for c in changes if c.feasibility == "moderate"]

        summary_parts = []
        if easy:
            summary_parts.append(f"{len(easy)} easy change(s)")
        if moderate:
            summary_parts.append(f"{len(moderate)} moderate change(s)")

        if summary_parts:
            return "Recommended: " + ", ".join(summary_parts)
        return "No easy changes available"

    def get_lime_explanation(
        self,
        explanation_id: str,
    ) -> Optional[LIMEExplanation]:
        """Get LIME explanation by ID."""
        return self._lime_explanations.get(explanation_id)

    def get_anomaly_explanation(
        self,
        explanation_id: str,
    ) -> Optional[AnomalyExplanation]:
        """Get anomaly explanation by ID."""
        return self._anomaly_explanations.get(explanation_id)

    def get_counterfactual(
        self,
        counterfactual_id: str,
    ) -> Optional[Counterfactual]:
        """Get counterfactual by ID."""
        return self._counterfactuals.get(counterfactual_id)
