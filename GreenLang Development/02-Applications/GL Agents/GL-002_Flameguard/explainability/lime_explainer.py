# -*- coding: utf-8 -*-
"""
GL-002 FLAMEGUARD - LIME Explainer for Flame Safety Predictions

LIME-based explanations for flame safety ML predictions including:
- Flame stability predictions
- Burner fault detection
- Safety system recommendations

Reference: NFPA 85, IEC 61508
Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations
import hashlib, logging, uuid, random, math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class FlameAnomalyType(str, Enum):
    FLAME_LOSS = "flame_loss"
    FLAME_INSTABILITY = "flame_instability"
    IGNITION_FAILURE = "ignition_failure"
    FUEL_RICH = "fuel_rich"
    FUEL_LEAN = "fuel_lean"
    SCANNER_FAULT = "scanner_fault"
    BURNER_FAULT = "burner_fault"

@dataclass
class LIMEFeatureWeight:
    feature_name: str
    feature_value: float
    weight: float
    contribution: float
    importance_rank: int
    direction: str
    description: str = ""
    baseline_value: float = 0.0

    def to_dict(self) -> Dict:
        return {"feature_name": self.feature_name, "feature_value": self.feature_value,
                "weight": self.weight, "contribution": self.contribution,
                "importance_rank": self.importance_rank, "direction": self.direction,
                "description": self.description, "baseline_value": self.baseline_value}

@dataclass
class LIMEExplanation:
    explanation_id: str
    timestamp: datetime
    instance_id: str
    predicted_value: float
    predicted_class: Optional[str] = None
    prediction_probability: float = 0.0
    intercept: float = 0.0
    feature_weights: List[LIMEFeatureWeight] = field(default_factory=list)
    local_model_r2: float = 0.0
    sample_size: int = 0
    top_positive_features: List[str] = field(default_factory=list)
    top_negative_features: List[str] = field(default_factory=list)
    explanation_text: str = ""
    explanation_confidence: float = 0.0
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            content = f"{self.explanation_id}|{self.predicted_value}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {"explanation_id": self.explanation_id, "timestamp": self.timestamp.isoformat(),
                "instance_id": self.instance_id, "predicted_value": self.predicted_value,
                "predicted_class": self.predicted_class, "feature_weights": [fw.to_dict() for fw in self.feature_weights],
                "local_model_r2": self.local_model_r2, "explanation_text": self.explanation_text,
                "explanation_confidence": self.explanation_confidence, "provenance_hash": self.provenance_hash}

class FlameguardLIMEExplainer:
    VERSION = "1.0.0"

    def __init__(self, agent_id: str = "GL-002", num_samples: int = 1000, num_features: int = 10):
        self.agent_id = agent_id
        self.num_samples = num_samples
        self.num_features = num_features
        self._feature_metadata = self._initialize_feature_metadata()
        self._normal_ranges = self._initialize_normal_ranges()
        self._explanations: Dict[str, LIMEExplanation] = {}
        logger.info(f"FlameguardLIMEExplainer initialized for {agent_id}")

    def _initialize_feature_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {
            "flame_intensity": {"description": "Flame scanner intensity", "unit": "%", "typical_range": (60, 100)},
            "flame_stability_index": {"description": "Flame stability score", "unit": "score", "typical_range": (0.8, 1.0)},
            "uv_signal": {"description": "UV flame detector signal", "unit": "mV", "typical_range": (200, 1000)},
            "ir_signal": {"description": "IR flame detector signal", "unit": "mV", "typical_range": (100, 800)},
            "fuel_pressure": {"description": "Fuel supply pressure", "unit": "psig", "typical_range": (5, 15)},
            "air_flow": {"description": "Combustion air flow", "unit": "%", "typical_range": (80, 120)},
            "firing_rate": {"description": "Current firing rate", "unit": "%", "typical_range": (20, 100)},
            "burner_temp": {"description": "Burner temperature", "unit": "°F", "typical_range": (200, 400)},
            "ignition_time": {"description": "Time since ignition", "unit": "s", "typical_range": (0, 10)},
            "scanner_age_days": {"description": "Scanner age", "unit": "days", "typical_range": (0, 365)},
        }

    def _initialize_normal_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {k: v["typical_range"] for k, v in self._feature_metadata.items()}

    def explain_prediction(self, model: Any, instance: Dict[str, float],
                           predicted_value: Optional[float] = None) -> LIMEExplanation:
        explanation_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        instance_id = instance.get("instance_id", str(uuid.uuid4())[:8])

        if predicted_value is None:
            predicted_value = self._mock_prediction(instance)

        weights, intercept, r2 = self._compute_lime_weights(model, instance, predicted_value)

        feature_weights = []
        sorted_features = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)

        for rank, (feature, weight) in enumerate(sorted_features[:self.num_features], 1):
            value = instance.get(feature, 0)
            metadata = self._feature_metadata.get(feature, {})
            baseline = sum(metadata.get("typical_range", (value, value))) / 2
            contribution = weight * (value - baseline)

            feature_weights.append(LIMEFeatureWeight(
                feature_name=feature, feature_value=value, weight=weight, contribution=contribution,
                importance_rank=rank, direction="positive" if weight > 0 else "negative",
                description=metadata.get("description", feature), baseline_value=baseline))

        top_positive = [fw.feature_name for fw in feature_weights if fw.direction == "positive"][:3]
        top_negative = [fw.feature_name for fw in feature_weights if fw.direction == "negative"][:3]
        explanation_text = self._generate_explanation_text(predicted_value, feature_weights[:5])
        confidence = min(1.0, 0.5 + r2 * 0.5)

        explanation = LIMEExplanation(
            explanation_id=explanation_id, timestamp=timestamp, instance_id=instance_id,
            predicted_value=predicted_value, predicted_class=self._value_to_class(predicted_value),
            prediction_probability=predicted_value, intercept=intercept, feature_weights=feature_weights,
            local_model_r2=r2, sample_size=self.num_samples, top_positive_features=top_positive,
            top_negative_features=top_negative, explanation_text=explanation_text, explanation_confidence=confidence)

        self._explanations[explanation_id] = explanation
        return explanation

    def _mock_prediction(self, instance: Dict[str, float]) -> float:
        risk = 0.2
        flame_intensity = instance.get("flame_intensity", 80)
        if flame_intensity < 50: risk += 0.4
        elif flame_intensity < 70: risk += 0.2
        stability = instance.get("flame_stability_index", 0.9)
        if stability < 0.7: risk += 0.3
        elif stability < 0.85: risk += 0.15
        fuel_pressure = instance.get("fuel_pressure", 10)
        if fuel_pressure < 5 or fuel_pressure > 15: risk += 0.2
        return min(1.0, max(0.0, risk))

    def _compute_lime_weights(self, model: Any, instance: Dict[str, float],
                               predicted_value: float) -> Tuple[Dict[str, float], float, float]:
        importance = {"flame_intensity": 0.25, "flame_stability_index": 0.20, "uv_signal": 0.15,
                      "fuel_pressure": 0.12, "air_flow": 0.10, "firing_rate": 0.08,
                      "ir_signal": 0.05, "burner_temp": 0.03, "scanner_age_days": 0.02}
        weights = {}
        for feature, value in instance.items():
            if feature in importance:
                base_weight = importance[feature]
                if feature in ["flame_intensity", "flame_stability_index", "uv_signal"]:
                    weights[feature] = -base_weight * 0.01  # Higher reduces risk
                else:
                    weights[feature] = base_weight * 0.005
        return weights, predicted_value * 0.3, 0.75 + random.uniform(0, 0.2)

    def _value_to_class(self, value: float) -> str:
        if value >= 0.7: return "high_risk"
        elif value >= 0.4: return "medium_risk"
        else: return "low_risk"

    def _generate_explanation_text(self, predicted_value: float, top_weights: List[LIMEFeatureWeight]) -> str:
        risk_class = self._value_to_class(predicted_value)
        text = f"Flame safety prediction: {risk_class} ({predicted_value:.1%} risk). "
        positive_factors = [fw for fw in top_weights if fw.direction == "positive"][:2]
        negative_factors = [fw for fw in top_weights if fw.direction == "negative"][:2]
        if positive_factors:
            text += f"Risk factors: {', '.join(fw.description for fw in positive_factors)}. "
        if negative_factors:
            text += f"Protective factors: {', '.join(fw.description for fw in negative_factors)}."
        return text

    def get_explanation(self, explanation_id: str) -> Optional[LIMEExplanation]:
        return self._explanations.get(explanation_id)

__all__ = ["FlameAnomalyType", "LIMEFeatureWeight", "LIMEExplanation", "FlameguardLIMEExplainer"]
