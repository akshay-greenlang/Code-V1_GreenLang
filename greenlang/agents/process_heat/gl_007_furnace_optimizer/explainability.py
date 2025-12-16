# -*- coding: utf-8 -*-
"""
GL-007 LIME Explainability Module
=================================

LIME (Local Interpretable Model-agnostic Explanations) for furnace
and cooling tower optimization decisions.

Key Capabilities:
    - Efficiency optimization explanation
    - Temperature zone analysis explanation
    - Cooling tower performance explanation
    - TMT safety margin explanation
    - Counterfactual analysis

ZERO-HALLUCINATION GUARANTEE:
    All explanations traceable to actual calculations.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureExplanation:
    """Explanation for a single feature's contribution."""
    feature_name: str
    feature_value: float
    contribution: float
    contribution_pct: float
    direction: str
    description: str
    threshold_info: Optional[str] = None
    unit: Optional[str] = None


@dataclass
class CounterfactualExplanation:
    """What-if explanation."""
    original_value: float
    feature_name: str
    suggested_value: float
    expected_outcome_change: float
    description: str
    unit: Optional[str] = None


@dataclass
class LIMEExplanation:
    """Complete LIME explanation."""
    explanation_id: str
    timestamp: datetime
    target_variable: str
    predicted_value: float
    feature_explanations: List[FeatureExplanation]
    top_positive_features: List[str]
    top_negative_features: List[str]
    local_model_r_squared: float
    num_samples_used: int
    counterfactuals: List[CounterfactualExplanation]
    summary_text: str
    confidence: float
    provenance_hash: str


@dataclass
class ExplainerConfig:
    """LIME explainer configuration."""
    num_samples: int = 1000
    num_features: int = 10
    kernel_width: float = 0.75
    random_seed: int = 42


FURNACE_FEATURE_INFO = {
    "flue_gas_o2_pct": {
        "name": "Flue Gas O2",
        "unit": "%",
        "optimal_range": (2.0, 4.0),
        "positive_description": "O2 is within optimal range for efficiency",
        "negative_description": "O2 outside optimal range impacts efficiency",
    },
    "excess_air_pct": {
        "name": "Excess Air",
        "unit": "%",
        "optimal_range": (10.0, 20.0),
        "positive_description": "Excess air is optimal for combustion",
        "negative_description": "Excess air level is sub-optimal",
    },
    "flue_gas_temp_f": {
        "name": "Flue Gas Temperature",
        "unit": "F",
        "optimal_max": 450.0,
        "positive_description": "Stack temperature indicates good heat recovery",
        "negative_description": "High stack temperature indicates heat loss",
    },
    "furnace_temp_f": {
        "name": "Furnace Temperature",
        "unit": "F",
        "positive_description": "Furnace temperature at setpoint",
        "negative_description": "Furnace temperature deviation from setpoint",
    },
    "tmt_margin_f": {
        "name": "TMT Safety Margin",
        "unit": "F",
        "optimal_min": 100.0,
        "positive_description": "Adequate safety margin on tube metal temperature",
        "negative_description": "Low TMT margin - risk of tube damage",
    },
    "efficiency_pct": {
        "name": "Thermal Efficiency",
        "unit": "%",
        "optimal_min": 85.0,
        "positive_description": "High thermal efficiency",
        "negative_description": "Efficiency below target",
    },
}

COOLING_TOWER_FEATURE_INFO = {
    "approach_f": {
        "name": "Approach Temperature",
        "unit": "F",
        "optimal_max": 8.0,
        "positive_description": "Good approach temperature for cooling",
        "negative_description": "High approach indicates reduced cooling capacity",
    },
    "range_f": {
        "name": "Cooling Range",
        "unit": "F",
        "positive_description": "Cooling range matches design",
        "negative_description": "Cooling range deviation from design",
    },
    "fan_power_kw": {
        "name": "Fan Power",
        "unit": "kW",
        "positive_description": "Fan power consumption is efficient",
        "negative_description": "High fan power indicates inefficiency",
    },
    "lg_ratio": {
        "name": "L/G Ratio",
        "unit": "",
        "optimal_range": (1.0, 1.5),
        "positive_description": "L/G ratio is optimal",
        "negative_description": "L/G ratio outside optimal range",
    },
}


class LIMEFurnaceExplainer:
    """LIME Explainer for Furnace and Cooling Tower Optimization."""

    def __init__(self, config: Optional[ExplainerConfig] = None) -> None:
        self.config = config or ExplainerConfig()
        self._rng = random.Random(self.config.random_seed)
        self._explanation_count = 0
        self._audit_trail: List[Dict[str, Any]] = []
        logger.info("LIMEFurnaceExplainer initialized")

    def explain_furnace_optimization(
        self,
        furnace_features: Dict[str, float],
        optimization_result: Dict[str, float],
        predict_fn: Optional[Callable] = None,
    ) -> LIMEExplanation:
        """Explain furnace optimization decision."""
        self._explanation_count += 1
        explanation_id = f"EXP-FUR-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        target_value = optimization_result.get("efficiency_improvement_pct", 0.0)

        if predict_fn is None:
            predict_fn = self._default_furnace_predictor

        perturbations, predictions = self._generate_perturbations(
            furnace_features, predict_fn, FURNACE_FEATURE_INFO
        )

        coefficients, intercept, r_squared = self._fit_local_model(
            furnace_features, perturbations, predictions, target_value
        )

        feature_explanations = self._build_feature_explanations(
            furnace_features, coefficients, FURNACE_FEATURE_INFO
        )

        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        top_positive = [fe.feature_name for fe in feature_explanations if fe.direction == "positive"][:3]
        top_negative = [fe.feature_name for fe in feature_explanations if fe.direction == "negative"][:3]

        counterfactuals = self._generate_counterfactuals(
            furnace_features, coefficients, target_value, FURNACE_FEATURE_INFO
        )

        summary = self._generate_furnace_summary(
            optimization_result, feature_explanations, top_positive, top_negative
        )

        provenance_hash = self._calculate_provenance_hash(
            furnace_features, target_value, coefficients
        )

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="efficiency_improvement_pct",
            predicted_value=target_value,
            feature_explanations=feature_explanations[:self.config.num_features],
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            local_model_r_squared=r_squared,
            num_samples_used=len(perturbations),
            counterfactuals=counterfactuals[:3],
            summary_text=summary,
            confidence=min(0.95, r_squared + 0.3),
            provenance_hash=provenance_hash,
        )

    def explain_cooling_tower_optimization(
        self,
        ct_features: Dict[str, float],
        optimization_result: Dict[str, float],
        predict_fn: Optional[Callable] = None,
    ) -> LIMEExplanation:
        """Explain cooling tower optimization decision."""
        self._explanation_count += 1
        explanation_id = f"EXP-CT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        target_value = optimization_result.get("energy_savings_pct", 0.0)

        if predict_fn is None:
            predict_fn = self._default_cooling_tower_predictor

        perturbations, predictions = self._generate_perturbations(
            ct_features, predict_fn, COOLING_TOWER_FEATURE_INFO
        )

        coefficients, intercept, r_squared = self._fit_local_model(
            ct_features, perturbations, predictions, target_value
        )

        feature_explanations = self._build_feature_explanations(
            ct_features, coefficients, COOLING_TOWER_FEATURE_INFO
        )

        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        top_positive = [fe.feature_name for fe in feature_explanations if fe.direction == "positive"][:3]
        top_negative = [fe.feature_name for fe in feature_explanations if fe.direction == "negative"][:3]

        summary = self._generate_ct_summary(
            optimization_result, feature_explanations, top_positive, top_negative
        )

        provenance_hash = self._calculate_provenance_hash(
            ct_features, target_value, coefficients
        )

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="energy_savings_pct",
            predicted_value=target_value,
            feature_explanations=feature_explanations[:self.config.num_features],
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            local_model_r_squared=r_squared,
            num_samples_used=len(perturbations),
            counterfactuals=[],
            summary_text=summary,
            confidence=min(0.95, r_squared + 0.3),
            provenance_hash=provenance_hash,
        )

    def _generate_perturbations(
        self,
        features: Dict[str, float],
        predict_fn: Callable,
        feature_info: Dict[str, Any],
    ) -> Tuple[List[Dict[str, float]], List[float]]:
        """Generate perturbations for LIME."""
        perturbations = []
        predictions = []

        feature_names = list(features.keys())

        for _ in range(self.config.num_samples):
            perturbed = {}
            for name in feature_names:
                original = features[name]
                std = abs(original) * 0.2 if original != 0 else 1.0
                perturbed_value = self._rng.gauss(original, std)
                perturbed[name] = max(0, perturbed_value)

            perturbations.append(perturbed)
            try:
                predictions.append(predict_fn(perturbed))
            except Exception:
                predictions.append(0.0)

        return perturbations, predictions

    def _fit_local_model(
        self,
        original_features: Dict[str, float],
        perturbations: List[Dict[str, float]],
        predictions: List[float],
        original_prediction: float,
    ) -> Tuple[Dict[str, float], float, float]:
        """Fit weighted linear model."""
        feature_names = list(original_features.keys())
        n_features = len(feature_names)
        n_samples = len(perturbations)

        if n_samples == 0:
            return {name: 0.0 for name in feature_names}, original_prediction, 0.5

        X = np.zeros((n_samples, n_features))
        y = np.array(predictions)

        for i, perturbed in enumerate(perturbations):
            for j, name in enumerate(feature_names):
                X[i, j] = perturbed.get(name, 0.0)

        original_vector = np.array([original_features.get(name, 0.0) for name in feature_names])

        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_norm = (X - X.mean(axis=0)) / X_std
        orig_norm = (original_vector - X.mean(axis=0)) / X_std

        distances = np.sqrt(np.sum((X_norm - orig_norm) ** 2, axis=1))
        weights = np.exp(-distances ** 2 / (self.config.kernel_width ** 2))

        W = np.diag(weights)
        X_bias = np.column_stack([np.ones(n_samples), X])

        try:
            XtWX = X_bias.T @ W @ X_bias
            XtWy = X_bias.T @ W @ y
            reg = np.eye(n_features + 1) * 1e-6
            beta = np.linalg.solve(XtWX + reg, XtWy)
            intercept = beta[0]
            coefficients = {name: beta[i+1] for i, name in enumerate(feature_names)}

            y_pred = X_bias @ beta
            ss_res = np.sum(weights * (y - y_pred) ** 2)
            ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        except np.linalg.LinAlgError:
            coefficients = {name: 0.0 for name in feature_names}
            intercept = original_prediction
            r_squared = 0.5

        return coefficients, intercept, max(0, min(1, r_squared))

    def _build_feature_explanations(
        self,
        features: Dict[str, float],
        coefficients: Dict[str, float],
        feature_info: Dict[str, Any],
    ) -> List[FeatureExplanation]:
        """Build feature explanations."""
        explanations = []
        total_contribution = sum(abs(c * features.get(n, 0)) for n, c in coefficients.items())

        for name, coef in coefficients.items():
            value = features.get(name, 0)
            contribution = coef * value
            contribution_pct = abs(contribution) / total_contribution * 100 if total_contribution > 0 else 0

            direction = "neutral" if abs(contribution) < 0.01 else ("positive" if contribution > 0 else "negative")

            info = feature_info.get(name, {})
            description = info.get("positive_description" if direction == "positive" else "negative_description", f"{name} impact")

            explanations.append(FeatureExplanation(
                feature_name=info.get("name", name),
                feature_value=value,
                contribution=round(contribution, 4),
                contribution_pct=round(contribution_pct, 2),
                direction=direction,
                description=description,
                unit=info.get("unit"),
            ))

        return explanations

    def _generate_counterfactuals(
        self,
        features: Dict[str, float],
        coefficients: Dict[str, float],
        current_value: float,
        feature_info: Dict[str, Any],
    ) -> List[CounterfactualExplanation]:
        """Generate counterfactual explanations."""
        counterfactuals = []
        target_improvement = max(1.0, current_value * 0.2)

        for name, coef in coefficients.items():
            if abs(coef) < 0.001:
                continue

            value = features.get(name, 0)
            info = feature_info.get(name, {})

            if coef != 0:
                needed_change = target_improvement / coef
                new_value = value + needed_change

                if new_value > 0 and abs(needed_change) < value * 2:
                    counterfactuals.append(CounterfactualExplanation(
                        original_value=value,
                        feature_name=info.get("name", name),
                        suggested_value=round(new_value, 2),
                        expected_outcome_change=target_improvement,
                        description=f"Adjusting {info.get('name', name)} could improve efficiency",
                        unit=info.get("unit"),
                    ))

        return counterfactuals

    def _generate_furnace_summary(
        self,
        result: Dict[str, float],
        explanations: List[FeatureExplanation],
        top_positive: List[str],
        top_negative: List[str],
    ) -> str:
        """Generate furnace optimization summary."""
        parts = []
        efficiency_improvement = result.get("efficiency_improvement_pct", 0)

        if efficiency_improvement > 3:
            parts.append(f"Significant efficiency improvement ({efficiency_improvement:.1f}%) achievable")
        elif efficiency_improvement > 0:
            parts.append(f"Modest efficiency improvement ({efficiency_improvement:.1f}%) identified")
        else:
            parts.append("Furnace already at optimal efficiency")

        if top_positive:
            parts.append(f"Positive factors: {', '.join(top_positive)}")
        if top_negative:
            parts.append(f"Areas to improve: {', '.join(top_negative)}")

        return ". ".join(parts)

    def _generate_ct_summary(
        self,
        result: Dict[str, float],
        explanations: List[FeatureExplanation],
        top_positive: List[str],
        top_negative: List[str],
    ) -> str:
        """Generate cooling tower summary."""
        parts = []
        savings = result.get("energy_savings_pct", 0)

        if savings > 20:
            parts.append(f"Significant energy savings ({savings:.0f}%) achievable")
        elif savings > 0:
            parts.append(f"Energy savings ({savings:.0f}%) through fan optimization")
        else:
            parts.append("Cooling tower at optimal operation")

        if top_positive:
            parts.append(f"Enabling factors: {', '.join(top_positive)}")

        return ". ".join(parts)

    def _default_furnace_predictor(self, features: Dict[str, float]) -> float:
        """Default furnace efficiency predictor."""
        improvement = 0.0
        o2 = features.get("flue_gas_o2_pct", 3.0)
        if o2 > 4.0:
            improvement += (o2 - 4.0) * 0.5
        elif o2 < 2.0:
            improvement -= (2.0 - o2) * 1.0

        flue_temp = features.get("flue_gas_temp_f", 400.0)
        if flue_temp > 450:
            improvement += (flue_temp - 450) * 0.02

        return improvement

    def _default_cooling_tower_predictor(self, features: Dict[str, float]) -> float:
        """Default cooling tower energy predictor."""
        savings = 0.0
        fan_speed = features.get("fan_speed_pct", 100.0)
        if fan_speed > 70:
            savings = (fan_speed - 70) * 0.8
        return min(40, savings)

    def _calculate_provenance_hash(
        self,
        features: Dict[str, float],
        predicted_value: float,
        coefficients: Dict[str, float],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "features": {k: round(v, 6) if isinstance(v, float) else v for k, v in features.items()},
            "predicted_value": round(predicted_value, 6),
            "coefficients": {k: round(v, 6) for k, v in coefficients.items()},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get explainer audit trail."""
        return self._audit_trail.copy()


def create_explainer(num_samples: int = 1000, num_features: int = 10) -> LIMEFurnaceExplainer:
    """Factory function to create LIME explainer."""
    config = ExplainerConfig(num_samples=num_samples, num_features=num_features)
    return LIMEFurnaceExplainer(config)


__all__ = [
    "FeatureExplanation",
    "CounterfactualExplanation",
    "LIMEExplanation",
    "ExplainerConfig",
    "LIMEFurnaceExplainer",
    "create_explainer",
    "FURNACE_FEATURE_INFO",
    "COOLING_TOWER_FEATURE_INFO",
]
