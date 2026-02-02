# LIMEThermalExplainer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from datetime import datetime, timezone
import numpy as np
import hashlib, json, logging

try:
    import lime, lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExergyExplanationType(Enum):
    PHYSICAL_EXERGY = "physical_exergy"
    EXERGY_DESTRUCTION = "exergy_destruction"
    EXERGY_EFFICIENCY = "exergy_efficiency"

class PerturbationStrategy(Enum):
    GAUSSIAN = "gaussian"
    THERMODYNAMIC = "thermodynamic"

@dataclass
class ExergyFeatureWeight:
    feature_name: str
    feature_value: float
    weight: float
    contribution: float
    condition: str
    unit: str = ""
    thermodynamic_category: str = ""
    physical_interpretation: str = ""

@dataclass
class LocalExergyModel:
    coefficients: Dict[str, float]
    intercept: float
    score: float
    feature_names: List[str]
    model_complexity: int

@dataclass
class ConsistencyCheckResult:
    is_consistent: bool
    correlation_coefficient: float
    top_features_overlap: float
    direction_agreement: float
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LIMEExergyExplanation:
    explanation_type: ExergyExplanationType
    instance_values: Dict[str, float]
    predicted_value: float
    feature_weights: List[ExergyFeatureWeight]
    local_model: LocalExergyModel
    num_features: int
    num_samples: int
    kernel_width: float
    perturbation_strategy: PerturbationStrategy
    local_fidelity: float
    timestamp: str = ""
    provenance_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def feature_ranking(self):
        return sorted([(fw.feature_name, abs(fw.weight)) for fw in self.feature_weights], key=lambda x: x[1], reverse=True)

class LIMEThermalExplainer:
    VERSION = "1.0.0"
    THERMODYNAMIC_CATEGORIES = {"inlet_temperature": "thermal", "outlet_temperature": "thermal", "h": "thermodynamic", "s": "thermodynamic", "mass_flow_rate": "flow", "exergy_in": "exergy", "efficiency": "performance"}
    FEATURE_UNITS = {"inlet_temperature": "K", "h": "kJ/kg", "s": "kJ/(kg*K)", "mass_flow_rate": "kg/s", "efficiency": "%"}

    def __init__(self, training_data, feature_names, perturbation_strategy=PerturbationStrategy.THERMODYNAMIC, categorical_features=None, kernel_width=None):
        if not LIME_AVAILABLE:
            raise ImportError("LIME required")
        self.training_data = training_data
        self.feature_names = feature_names
        self.perturbation_strategy = perturbation_strategy
        self.kernel_width = kernel_width
        self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_data, feature_names, mode="regression", categorical_features=categorical_features or [], kernel_width=kernel_width, verbose=False)

    def explain_exergy_prediction(self, model, instance, explanation_type=ExergyExplanationType.EXERGY_EFFICIENCY, num_features=10, num_samples=5000):
        if isinstance(instance, dict):
            instance_array = np.array([instance[k] for k in self.feature_names])
            instance_dict = instance
        else:
            instance_array = np.array(instance)
            instance_dict = {name: float(instance_array[i]) for i, name in enumerate(self.feature_names)}
        predict_fn = model.predict if hasattr(model, "predict") else model
        prediction = predict_fn(instance_array.reshape(1, -1))
        predicted_value = float(prediction[0]) if hasattr(prediction, "__len__") else float(prediction)
        explanation = self._lime_explainer.explain_instance(instance_array, predict_fn, num_features=num_features, num_samples=num_samples)
        feature_weights = [ExergyFeatureWeight(self._parse_condition(c, instance_dict)[0], self._parse_condition(c, instance_dict)[1], w, w, c, self.FEATURE_UNITS.get(self._parse_condition(c, instance_dict)[0], ""), self.THERMODYNAMIC_CATEGORIES.get(self._parse_condition(c, instance_dict)[0], "other"), "") for c, w in explanation.as_list()]
        local_model = LocalExergyModel({fw.feature_name: fw.weight for fw in feature_weights}, explanation.intercept[0] if hasattr(explanation, "intercept") else 0.0, explanation.score if hasattr(explanation, "score") else 0.0, [fw.feature_name for fw in feature_weights], len(feature_weights))
        return LIMEExergyExplanation(explanation_type, instance_dict, predicted_value, feature_weights, local_model, num_features, num_samples, self.kernel_width or 0.75*np.sqrt(len(self.feature_names)), self.perturbation_strategy, explanation.score if hasattr(explanation, "score") else 0.0, datetime.now(timezone.utc).isoformat(), hashlib.sha256(json.dumps({"p": predicted_value}).encode()).hexdigest(), {"version": self.VERSION})

    def check_consistency_with_shap(self, lime_explanation, shap_values, shap_feature_names, threshold=0.7):
        lime_weights = {fw.feature_name: fw.weight for fw in lime_explanation.feature_weights}
        shap_dict = {name: shap_values[i] for i, name in enumerate(shap_feature_names)}
        common = set(lime_weights.keys()) & set(shap_dict.keys())
        if len(common) < 3:
            return ConsistencyCheckResult(False, 0.0, 0.0, 0.0, {"error": "Too few common features"})
        lime_vec = np.array([lime_weights.get(f, 0) for f in common])
        shap_vec = np.array([shap_dict.get(f, 0) for f in common])
        correlation = np.corrcoef(lime_vec, shap_vec)[0, 1] if np.std(lime_vec) > 0 and np.std(shap_vec) > 0 else 0.0
        is_consistent = correlation >= threshold
        return ConsistencyCheckResult(is_consistent, float(correlation), 0.0, 0.0, {"common": len(common)})

    def _parse_condition(self, condition, instance_dict):
        for f in self.feature_names:
            if f in condition:
                return f, instance_dict.get(f, 0.0)
        return condition[:20], 0.0

def explain_exergy_instance(model, instance, training_data, feature_names, num_features=10):
    return LIMEThermalExplainer(training_data, feature_names).explain_exergy_prediction(model, instance, num_features=num_features)

__all__ = ["LIMEThermalExplainer", "LIMEExergyExplanation", "ExergyFeatureWeight", "LocalExergyModel", "ConsistencyCheckResult", "ExergyExplanationType", "PerturbationStrategy", "explain_exergy_instance"]
