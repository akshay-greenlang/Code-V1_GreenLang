# GL-012 LIME Quality Explainer
# This file was created by GL-BackendDeveloper
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
from pydantic import BaseModel, Field
import numpy as np
import hashlib, json, uuid, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class FeatureSelectionMethod(str, Enum):
    AUTO = "auto"
    FORWARD = "forward"
    LASSO = "lasso"
    RIDGE = "ridge"
    HIGHEST_WEIGHTS = "highest_weights"
    NONE = "none"


class KernelType(str, Enum):
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"
    COSINE = "cosine"


class ConsistencyStatus(str, Enum):
    CONSISTENT = "consistent"
    PARTIALLY_CONSISTENT = "partially_consistent"
    INCONSISTENT = "inconsistent"
    NOT_CHECKED = "not_checked"
    ERROR = "error"


class ExplanationMode(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class LIMEConfig(BaseModel):
    n_samples: int = Field(default=500, ge=100, le=10000)
    kernel_width: float = Field(default=0.75, gt=0.0, le=10.0)
    feature_selection: FeatureSelectionMethod = Field(default=FeatureSelectionMethod.AUTO)
    num_features: int = Field(default=10, ge=1, le=50)
    random_seed: int = Field(default=42, ge=0)
    discretize_continuous: bool = Field(default=True)
    kernel_type: KernelType = Field(default=KernelType.EXPONENTIAL)
    consistency_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    mode: ExplanationMode = Field(default=ExplanationMode.REGRESSION)
    regularization_weight: float = Field(default=1.0, ge=0.0)


class FeatureExplanation(BaseModel):
    feature_name: str = Field(...)
    weight: float = Field(...)
    feature_value: float = Field(...)
    contribution: float = Field(...)
    rank: int = Field(..., ge=1)
    direction: str = Field(...)
    physical_unit: Optional[str] = Field(None)
    description: Optional[str] = Field(None)


class LIMEExplanation(BaseModel):
    explanation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature_names: List[str] = Field(...)
    feature_weights: Dict[str, float] = Field(...)
    feature_explanations: List[FeatureExplanation] = Field(default_factory=list)
    local_prediction: float = Field(...)
    original_prediction: float = Field(...)
    intercept: float = Field(default=0.0)
    score: float = Field(..., ge=0.0, le=1.0)
    consistency_status: ConsistencyStatus = Field(default=ConsistencyStatus.NOT_CHECKED)
    consistency_score: Optional[float] = Field(None)
    inconsistent_features: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)
    header_id: str = Field(...)
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    agent_id: str = Field(default="GL-012")
    version: str = Field(default="1.0.0")

    def get_top_features(self, n: int = 5) -> List[FeatureExplanation]:
        return sorted(self.feature_explanations, key=lambda x: abs(x.weight), reverse=True)[:n]


QUALITY_FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "pressure_mpa": {"description": "Steam pressure", "unit": "MPa", "typical_range": (0.1, 20.0), "sensitivity": "high"},
    "temperature_c": {"description": "Steam temperature", "unit": "C", "typical_range": (100.0, 600.0), "sensitivity": "high"},
    "superheat_c": {"description": "Superheat", "unit": "C", "typical_range": (0.0, 100.0), "sensitivity": "high"},
    "drum_level_pct": {"description": "Boiler drum level", "unit": "%", "typical_range": (0.0, 100.0), "sensitivity": "high"},
    "enthalpy_kj_kg": {"description": "Specific enthalpy", "unit": "kJ/kg", "typical_range": (400.0, 3500.0), "sensitivity": "high"},
    "flow_rate_kg_s": {"description": "Mass flow rate", "unit": "kg/s", "typical_range": (0.1, 1000.0), "sensitivity": "medium"},
}


class LIMEQualityExplainer:
    VERSION = "1.0.0"

    def __init__(self, config: Optional[LIMEConfig] = None, agent_id: str = "GL-012", shap_explainer: Optional[Any] = None):
        self.config = config or LIMEConfig()
        self.agent_id = agent_id
        self.shap_explainer = shap_explainer
        self.rng = np.random.RandomState(self.config.random_seed)

    def set_shap_explainer(self, shap_explainer: Any) -> None:
        self.shap_explainer = shap_explainer

    def explain_quality_prediction(self, features: Dict[str, float], predicted_quality: float, header_id: str,
                                   instance_id: Optional[str] = None, prediction_fn: Optional[Callable] = None) -> LIMEExplanation:
        start_time = datetime.now(timezone.utc)
        feature_names = list(features.keys())
        feature_values = np.array([features[name] for name in feature_names])
        perturbed_data, weights = self._generate_perturbations(feature_values, feature_names)
        if prediction_fn:
            perturbed_predictions = prediction_fn(perturbed_data)
        else:
            perturbed_predictions = self._approximate_predictions(perturbed_data, feature_values, predicted_quality, feature_names)
        lime_weights, intercept, r2_score = self._fit_local_model(perturbed_data, perturbed_predictions, weights, feature_names)
        local_prediction = intercept + sum([lime_weights.get(name, 0) * features[name] for name in feature_names])
        feature_explanations = self._build_feature_explanations(feature_names, features, lime_weights)
        provenance_hash = self._compute_provenance_hash(features, predicted_quality, lime_weights, header_id)
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return LIMEExplanation(
            explanation_id=instance_id or str(uuid.uuid4()), timestamp=start_time, feature_names=feature_names,
            feature_weights=lime_weights, feature_explanations=feature_explanations, local_prediction=float(local_prediction),
            original_prediction=predicted_quality, intercept=float(intercept), score=float(r2_score),
            consistency_status=ConsistencyStatus.NOT_CHECKED, provenance_hash=provenance_hash, header_id=header_id,
            config_snapshot=self.config.model_dump(), processing_time_ms=processing_time_ms, agent_id=self.agent_id, version=self.VERSION
        )

    def explain_dryness_fraction(self, features: Dict[str, float], dryness_fraction: float, header_id: str,
                                  prediction_fn: Optional[Callable] = None) -> LIMEExplanation:
        return self.explain_quality_prediction(features, dryness_fraction, header_id, prediction_fn=prediction_fn)

    def _generate_perturbations(self, original: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n_features, n_samples = len(original), self.config.n_samples
        perturbations = np.zeros((n_samples, n_features))
        for i, name in enumerate(feature_names):
            metadata = QUALITY_FEATURE_METADATA.get(name, {})
            typical_range = metadata.get("typical_range", (0, 1))
            quartiles = np.linspace(typical_range[0], typical_range[1], 5)
            perturbations[:, i] = self.rng.choice(quartiles, size=n_samples)
        distances = np.sqrt(np.sum(((perturbations - original) / (original + 1e-10)) ** 2, axis=1))
        return perturbations, np.exp(-(distances ** 2) / (self.config.kernel_width ** 2))

    def _approximate_predictions(self, perturbed_data: np.ndarray, original: np.ndarray,
                                  original_prediction: float, feature_names: List[str]) -> np.ndarray:
        predictions = np.full(len(perturbed_data), original_prediction)
        for i, name in enumerate(feature_names):
            metadata = QUALITY_FEATURE_METADATA.get(name, {})
            sensitivity = {"high": 0.05, "medium": 0.02, "low": 0.01}.get(metadata.get("sensitivity", "medium"), 0.02)
            typical_range = metadata.get("typical_range", (0, 1))
            range_size = typical_range[1] - typical_range[0]
            if range_size > 0:
                predictions += sensitivity * (perturbed_data[:, i] - original[i]) / range_size
        return np.clip(predictions, 0.0, 1.0)

    def _fit_local_model(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                         feature_names: List[str]) -> Tuple[Dict[str, float], float, float]:
        sqrt_weights = np.sqrt(weights)
        X_weighted = X * sqrt_weights[:, np.newaxis]
        y_weighted = y * sqrt_weights
        X_with_intercept = np.column_stack([np.ones(len(X)), X_weighted])
        n_features = X.shape[1]
        reg_matrix = self.config.regularization_weight * np.eye(n_features + 1)
        reg_matrix[0, 0] = 0
        try:
            XtX = X_with_intercept.T @ X_with_intercept + reg_matrix
            Xty = X_with_intercept.T @ y_weighted
            coefficients = np.linalg.solve(XtX, Xty)
        except:
            coefficients = np.linalg.lstsq(X_with_intercept, y_weighted, rcond=None)[0]
        intercept, feature_weights = coefficients[0], coefficients[1:]
        y_pred = X_with_intercept @ coefficients
        ss_res = np.sum(weights * (y_weighted - y_pred) ** 2)
        ss_tot = np.sum(weights * (y_weighted - np.average(y_weighted, weights=weights)) ** 2)
        r2_score = max(0.0, min(1.0, 1 - (ss_res / (ss_tot + 1e-10))))
        return dict(zip(feature_names, feature_weights)), intercept, r2_score

    def _build_feature_explanations(self, feature_names: List[str], features: Dict[str, float],
                                     weights: Dict[str, float]) -> List[FeatureExplanation]:
        explanations = []
        sorted_items = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for rank, (name, weight) in enumerate(sorted_items, start=1):
            value = features.get(name, 0.0)
            metadata = QUALITY_FEATURE_METADATA.get(name, {})
            explanations.append(FeatureExplanation(
                feature_name=name, weight=weight, feature_value=value, contribution=weight * value,
                rank=rank, direction="positive" if weight > 0 else "negative",
                physical_unit=metadata.get("unit"), description=metadata.get("description")
            ))
        return explanations

    def _compute_provenance_hash(self, features: Dict[str, float], prediction: float,
                                  weights: Dict[str, float], header_id: str) -> str:
        provenance_data = {"features": features, "prediction": prediction, "weights": weights,
                           "header_id": header_id, "config": self.config.model_dump(),
                           "agent_id": self.agent_id, "version": self.VERSION}
        return hashlib.sha256(json.dumps(provenance_data, sort_keys=True, default=str).encode()).hexdigest()


def create_lime_explainer(n_samples: int = 500, kernel_width: float = 0.75, random_seed: int = 42,
                           agent_id: str = "GL-012", shap_explainer: Optional[Any] = None, **kwargs) -> LIMEQualityExplainer:
    config = LIMEConfig(n_samples=n_samples, kernel_width=kernel_width, random_seed=random_seed, **kwargs)
    return LIMEQualityExplainer(config=config, agent_id=agent_id, shap_explainer=shap_explainer)


__all__ = [
    "FeatureSelectionMethod", "KernelType", "ConsistencyStatus", "ExplanationMode",
    "LIMEConfig", "FeatureExplanation", "LIMEExplanation", "QUALITY_FEATURE_METADATA",
    "LIMEQualityExplainer", "create_lime_explainer",
]
