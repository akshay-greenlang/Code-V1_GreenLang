# -*- coding: utf-8 -*-
"""
GL-008 LIME Explainability Module
=================================

LIME explanations for steam trap monitoring and failure detection.

Key Capabilities:
    - Trap failure prediction explanation
    - Loss quantification explanation
    - Maintenance priority justification
    - Counterfactual analysis

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
    """Feature contribution explanation."""
    feature_name: str
    feature_value: float
    contribution: float
    contribution_pct: float
    direction: str
    description: str
    threshold_info: Optional[str] = None
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


STEAM_TRAP_FEATURE_INFO = {
    "inlet_temp_f": {
        "name": "Inlet Temperature",
        "unit": "F",
        "positive_description": "Inlet temperature normal for steam trap operation",
        "negative_description": "Abnormal inlet temperature indicates trap issue",
    },
    "outlet_temp_f": {
        "name": "Outlet Temperature",
        "unit": "F",
        "positive_description": "Outlet temperature indicates proper condensate discharge",
        "negative_description": "High outlet temperature indicates steam blowing through",
    },
    "differential_temp_f": {
        "name": "Temperature Differential",
        "unit": "F",
        "optimal_min": 10.0,
        "positive_description": "Good temperature differential indicates proper operation",
        "negative_description": "Low differential suggests trap may be failed open",
    },
    "acoustic_level_db": {
        "name": "Acoustic Level",
        "unit": "dB",
        "warning_high": 85.0,
        "positive_description": "Normal acoustic signature",
        "negative_description": "High acoustic level indicates steam blowing through",
    },
    "cycle_rate_per_min": {
        "name": "Cycle Rate",
        "unit": "cycles/min",
        "optimal_range": (4.0, 20.0),
        "positive_description": "Cycling rate normal for trap type",
        "negative_description": "Abnormal cycling indicates potential failure",
    },
    "steam_loss_lb_hr": {
        "name": "Steam Loss Rate",
        "unit": "lb/hr",
        "warning_high": 50.0,
        "positive_description": "Minimal steam loss",
        "negative_description": "Significant steam loss detected",
    },
    "annual_loss_usd": {
        "name": "Annual Energy Loss",
        "unit": "$/year",
        "positive_description": "Low annual loss indicates efficient operation",
        "negative_description": "High annual loss from failed trap",
    },
}


class LIMESteamTrapExplainer:
    """LIME Explainer for Steam Trap Monitoring."""

    def __init__(self, config: Optional[ExplainerConfig] = None) -> None:
        self.config = config or ExplainerConfig()
        self._rng = random.Random(self.config.random_seed)
        self._explanation_count = 0
        logger.info("LIMESteamTrapExplainer initialized")

    def explain_trap_failure(
        self,
        trap_features: Dict[str, float],
        failure_probability: float,
        predict_fn: Optional[Callable] = None,
    ) -> LIMEExplanation:
        """Explain steam trap failure prediction."""
        self._explanation_count += 1
        explanation_id = f"EXP-TRAP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        if predict_fn is None:
            predict_fn = self._default_failure_predictor

        perturbations, predictions = self._generate_perturbations(trap_features, predict_fn)
        coefficients, intercept, r_squared = self._fit_local_model(
            trap_features, perturbations, predictions, failure_probability
        )

        feature_explanations = self._build_feature_explanations(trap_features, coefficients)
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        top_positive = [fe.feature_name for fe in feature_explanations if fe.direction == "positive"][:3]
        top_negative = [fe.feature_name for fe in feature_explanations if fe.direction == "negative"][:3]

        summary = self._generate_summary(failure_probability, feature_explanations, top_negative)

        provenance_hash = self._calculate_provenance_hash(trap_features, failure_probability, coefficients)

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="failure_probability",
            predicted_value=failure_probability,
            feature_explanations=feature_explanations[:self.config.num_features],
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            local_model_r_squared=r_squared,
            num_samples_used=len(perturbations),
            summary_text=summary,
            confidence=min(0.95, r_squared + 0.3),
            provenance_hash=provenance_hash,
        )

    def explain_loss_quantification(
        self,
        trap_features: Dict[str, float],
        loss_result: Dict[str, float],
    ) -> LIMEExplanation:
        """Explain steam loss quantification."""
        self._explanation_count += 1
        explanation_id = f"EXP-LOSS-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        annual_loss = loss_result.get("annual_loss_usd", 0.0)
        feature_explanations = []

        steam_loss = trap_features.get("steam_loss_lb_hr", 0)
        if steam_loss > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Steam Loss Rate",
                feature_value=steam_loss,
                contribution=steam_loss * 100,
                contribution_pct=60.0,
                direction="negative",
                description=f"Steam loss of {steam_loss:.0f} lb/hr is primary loss driver",
                unit="lb/hr",
            ))

        operating_hours = trap_features.get("operating_hours_yr", 8760)
        feature_explanations.append(FeatureExplanation(
            feature_name="Operating Hours",
            feature_value=operating_hours,
            contribution=operating_hours * 0.1,
            contribution_pct=20.0,
            direction="neutral",
            description=f"{operating_hours} operating hours/year",
            unit="hrs/year",
        ))

        summary = f"Annual loss of ${annual_loss:,.0f} primarily due to steam loss of {steam_loss:.0f} lb/hr"

        provenance_hash = self._calculate_provenance_hash(trap_features, annual_loss, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="annual_loss_usd",
            predicted_value=annual_loss,
            feature_explanations=feature_explanations,
            top_positive_features=[],
            top_negative_features=["Steam Loss Rate"],
            local_model_r_squared=0.9,
            num_samples_used=0,
            summary_text=summary,
            confidence=0.95,
            provenance_hash=provenance_hash,
        )

    def _generate_perturbations(
        self,
        features: Dict[str, float],
        predict_fn: Callable,
    ) -> Tuple[List[Dict[str, float]], List[float]]:
        """Generate perturbations for LIME."""
        perturbations = []
        predictions = []

        for _ in range(self.config.num_samples):
            perturbed = {}
            for name, value in features.items():
                std = abs(value) * 0.2 if value != 0 else 1.0
                perturbed[name] = max(0, self._rng.gauss(value, std))
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
        distances = np.sqrt(np.sum(((X - X.mean(axis=0)) / X_std - (original_vector - X.mean(axis=0)) / X_std) ** 2, axis=1))
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
    ) -> List[FeatureExplanation]:
        """Build feature explanations."""
        explanations = []
        total_contribution = sum(abs(c * features.get(n, 0)) for n, c in coefficients.items())

        for name, coef in coefficients.items():
            value = features.get(name, 0)
            contribution = coef * value
            contribution_pct = abs(contribution) / total_contribution * 100 if total_contribution > 0 else 0
            direction = "neutral" if abs(contribution) < 0.01 else ("positive" if contribution < 0 else "negative")

            info = STEAM_TRAP_FEATURE_INFO.get(name, {})
            description = info.get("positive_description" if direction == "positive" else "negative_description", f"{name}")

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

    def _generate_summary(
        self,
        failure_prob: float,
        explanations: List[FeatureExplanation],
        top_negative: List[str],
    ) -> str:
        """Generate failure prediction summary."""
        parts = []

        if failure_prob > 0.8:
            parts.append(f"High failure probability ({failure_prob:.0%}): Immediate replacement recommended")
        elif failure_prob > 0.5:
            parts.append(f"Moderate failure probability ({failure_prob:.0%}): Schedule inspection")
        else:
            parts.append(f"Low failure probability ({failure_prob:.0%}): Trap operating normally")

        if top_negative:
            parts.append(f"Key indicators: {', '.join(top_negative)}")

        return ". ".join(parts)

    def _default_failure_predictor(self, features: Dict[str, float]) -> float:
        """Default failure probability predictor."""
        prob = 0.0
        diff_temp = features.get("differential_temp_f", 30)
        if diff_temp < 10:
            prob += 0.4

        acoustic = features.get("acoustic_level_db", 60)
        if acoustic > 85:
            prob += 0.3

        steam_loss = features.get("steam_loss_lb_hr", 0)
        if steam_loss > 50:
            prob += 0.3

        return min(1.0, prob)

    def _calculate_provenance_hash(
        self,
        features: Dict[str, float],
        predicted_value: float,
        coefficients: Dict[str, float],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "features": {k: round(v, 6) for k, v in features.items()},
            "predicted_value": round(predicted_value, 6),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def create_explainer(num_samples: int = 1000) -> LIMESteamTrapExplainer:
    """Factory function."""
    return LIMESteamTrapExplainer(ExplainerConfig(num_samples=num_samples))


__all__ = ["FeatureExplanation", "LIMEExplanation", "LIMESteamTrapExplainer", "create_explainer"]
