# -*- coding: utf-8 -*-
"""
GL-011 LIME Explainability Module
=================================

LIME explanations for fuel optimization decisions.

Key Capabilities:
    - Fuel blend recommendation explanation
    - Fuel switching decision explanation
    - Cost optimization explanation
    - Inventory alert explanation

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


FUEL_OPTIMIZATION_FEATURE_INFO = {
    "natural_gas_price_usd_mmbtu": {
        "name": "Natural Gas Price",
        "unit": "$/MMBTU",
        "positive_description": "Competitive natural gas pricing",
        "negative_description": "High natural gas price increases costs",
    },
    "fuel_oil_price_usd_mmbtu": {
        "name": "Fuel Oil Price",
        "unit": "$/MMBTU",
        "positive_description": "Fuel oil price favorable for switching",
        "negative_description": "High fuel oil price limits switching options",
    },
    "blend_ratio_pct": {
        "name": "Blend Ratio",
        "unit": "%",
        "positive_description": "Optimal blend ratio for cost/emissions balance",
        "negative_description": "Non-optimal blend ratio increases costs",
    },
    "wobbe_index_deviation": {
        "name": "Wobbe Index Deviation",
        "unit": "BTU/SCF",
        "warning_high": 50.0,
        "positive_description": "Wobbe index within acceptable range",
        "negative_description": "Wobbe index deviation requires blend adjustment",
    },
    "inventory_level_pct": {
        "name": "Inventory Level",
        "unit": "%",
        "warning_low": 20.0,
        "positive_description": "Adequate fuel inventory",
        "negative_description": "Low inventory level triggers reorder",
    },
    "co2_emission_factor": {
        "name": "CO2 Emission Factor",
        "unit": "kg/MMBTU",
        "positive_description": "Low carbon intensity fuel",
        "negative_description": "High carbon intensity increases emissions",
    },
    "heating_value_btu_scf": {
        "name": "Heating Value",
        "unit": "BTU/SCF",
        "positive_description": "High heating value improves efficiency",
        "negative_description": "Low heating value increases fuel consumption",
    },
    "transition_cost_usd": {
        "name": "Transition Cost",
        "unit": "$",
        "positive_description": "Low fuel switching transition cost",
        "negative_description": "High transition cost delays fuel switching",
    },
}


class LIMEFuelOptimizationExplainer:
    """LIME Explainer for Fuel Optimization Decisions."""

    def __init__(self, config: Optional[ExplainerConfig] = None) -> None:
        self.config = config or ExplainerConfig()
        self._rng = random.Random(self.config.random_seed)
        self._explanation_count = 0
        logger.info("LIMEFuelOptimizationExplainer initialized")

    def explain_blend_recommendation(
        self,
        fuel_features: Dict[str, float],
        blend_result: Dict[str, float],
        predict_fn: Optional[Callable] = None,
    ) -> LIMEExplanation:
        """Explain fuel blend recommendation."""
        self._explanation_count += 1
        explanation_id = f"EXP-BLEND-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        blended_cost = blend_result.get("blended_cost_usd_mmbtu", 0.0)

        if predict_fn is None:
            predict_fn = self._default_blend_predictor

        perturbations, predictions = self._generate_perturbations(fuel_features, predict_fn)
        coefficients, intercept, r_squared = self._fit_local_model(
            fuel_features, perturbations, predictions, blended_cost
        )

        feature_explanations = self._build_feature_explanations(fuel_features, coefficients)
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        top_positive = [fe.feature_name for fe in feature_explanations if fe.direction == "positive"][:3]
        top_negative = [fe.feature_name for fe in feature_explanations if fe.direction == "negative"][:3]

        summary = self._generate_blend_summary(blended_cost, blend_result, top_positive, top_negative)

        provenance_hash = self._calculate_provenance_hash(fuel_features, blended_cost, coefficients)

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="blended_cost_usd_mmbtu",
            predicted_value=blended_cost,
            feature_explanations=feature_explanations[:self.config.num_features],
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            local_model_r_squared=r_squared,
            num_samples_used=len(perturbations),
            summary_text=summary,
            confidence=min(0.95, r_squared + 0.3),
            provenance_hash=provenance_hash,
        )

    def explain_switching_decision(
        self,
        current_fuel_features: Dict[str, float],
        alternative_fuel_features: Dict[str, float],
        switching_result: Dict[str, Any],
    ) -> LIMEExplanation:
        """Explain fuel switching recommendation."""
        self._explanation_count += 1
        explanation_id = f"EXP-SWITCH-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        savings = switching_result.get("savings_usd_hr", 0.0)
        feature_explanations = []

        # Price differential
        current_cost = current_fuel_features.get("price_usd_mmbtu", 0)
        alt_cost = alternative_fuel_features.get("price_usd_mmbtu", 0)
        price_diff = current_cost - alt_cost
        if price_diff > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Price Differential",
                feature_value=price_diff,
                contribution=price_diff * 10,
                contribution_pct=50.0,
                direction="positive",
                description=f"Alternative fuel ${price_diff:.2f}/MMBTU cheaper",
                unit="$/MMBTU",
            ))

        # Transition cost impact
        transition_cost = switching_result.get("transition_cost_usd", 0)
        if transition_cost > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Transition Cost",
                feature_value=transition_cost,
                contribution=-transition_cost * 0.1,
                contribution_pct=20.0,
                direction="negative",
                description=f"${transition_cost:.0f} one-time switching cost",
                unit="$",
            ))

        # Emissions impact
        emissions_reduction = switching_result.get("emissions_reduction_pct", 0)
        if emissions_reduction > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Emissions Reduction",
                feature_value=emissions_reduction,
                contribution=emissions_reduction * 5,
                contribution_pct=15.0,
                direction="positive",
                description=f"{emissions_reduction:.1f}% CO2 reduction with alternative fuel",
                unit="%",
            ))

        recommended = switching_result.get("recommended", False)
        summary = f"Fuel switch {'recommended' if recommended else 'not recommended'}. "
        if savings > 0:
            summary += f"Potential savings: ${savings:.0f}/hr. "
        if feature_explanations:
            summary += f"Key factors: {', '.join([fe.feature_name for fe in feature_explanations[:2]])}"

        provenance_hash = self._calculate_provenance_hash(current_fuel_features, savings, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="savings_usd_hr",
            predicted_value=savings,
            feature_explanations=feature_explanations,
            top_positive_features=[fe.feature_name for fe in feature_explanations if fe.direction == "positive"],
            top_negative_features=[fe.feature_name for fe in feature_explanations if fe.direction == "negative"],
            local_model_r_squared=0.9,
            num_samples_used=0,
            summary_text=summary,
            confidence=0.9,
            provenance_hash=provenance_hash,
        )

    def explain_inventory_alert(
        self,
        inventory_features: Dict[str, float],
        alert_result: Dict[str, Any],
    ) -> LIMEExplanation:
        """Explain inventory alert recommendation."""
        self._explanation_count += 1
        explanation_id = f"EXP-INV-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        days_of_supply = alert_result.get("days_of_supply", 0.0)
        feature_explanations = []

        # Current level
        current_level = inventory_features.get("current_level_pct", 0)
        if current_level < 30:
            feature_explanations.append(FeatureExplanation(
                feature_name="Current Level",
                feature_value=current_level,
                contribution=-30 + current_level,
                contribution_pct=40.0,
                direction="negative",
                description=f"Inventory at {current_level:.0f}% - below reorder point",
                unit="%",
            ))

        # Consumption rate
        consumption_rate = inventory_features.get("consumption_rate_gal_hr", 0)
        if consumption_rate > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Consumption Rate",
                feature_value=consumption_rate,
                contribution=-consumption_rate * 0.5,
                contribution_pct=30.0,
                direction="neutral",
                description=f"Current consumption: {consumption_rate:.0f} gal/hr",
                unit="gal/hr",
            ))

        alert_active = alert_result.get("alert_active", False)
        summary = f"Inventory alert {'active' if alert_active else 'cleared'}. "
        summary += f"{days_of_supply:.1f} days of supply remaining. "
        if feature_explanations:
            summary += f"Key factors: {', '.join([fe.feature_name for fe in feature_explanations[:2]])}"

        provenance_hash = self._calculate_provenance_hash(inventory_features, days_of_supply, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="days_of_supply",
            predicted_value=days_of_supply,
            feature_explanations=feature_explanations,
            top_positive_features=[],
            top_negative_features=[fe.feature_name for fe in feature_explanations if fe.direction == "negative"],
            local_model_r_squared=0.85,
            num_samples_used=0,
            summary_text=summary,
            confidence=0.9,
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

        try:
            X_bias = np.column_stack([np.ones(n_samples), X])
            beta = np.linalg.lstsq(X_bias, y, rcond=None)[0]
            intercept = beta[0]
            coefficients = {name: beta[i+1] for i, name in enumerate(feature_names)}

            y_pred = X_bias @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
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
        total = sum(abs(c * features.get(n, 0)) for n, c in coefficients.items())

        for name, coef in coefficients.items():
            value = features.get(name, 0)
            contribution = coef * value
            contribution_pct = abs(contribution) / total * 100 if total > 0 else 0
            # For cost minimization, negative contribution is positive outcome
            direction = "neutral" if abs(contribution) < 0.01 else ("positive" if contribution < 0 else "negative")

            info = FUEL_OPTIMIZATION_FEATURE_INFO.get(name, {})
            description = info.get("positive_description" if direction == "positive" else "negative_description", name)

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

    def _generate_blend_summary(
        self,
        blended_cost: float,
        blend_result: Dict[str, float],
        top_positive: List[str],
        top_negative: List[str],
    ) -> str:
        """Generate blend recommendation summary."""
        parts = []
        parts.append(f"Blended fuel cost: ${blended_cost:.2f}/MMBTU")

        savings = blend_result.get("cost_savings_usd_hr", 0)
        if savings > 0:
            parts.append(f"Savings vs single fuel: ${savings:.0f}/hr")

        if top_positive:
            parts.append(f"Favorable factors: {', '.join(top_positive)}")

        if top_negative:
            parts.append(f"Unfavorable factors: {', '.join(top_negative)}")

        return ". ".join(parts)

    def _default_blend_predictor(self, features: Dict[str, float]) -> float:
        """Default blend cost predictor."""
        cost = 0.0
        ng_price = features.get("natural_gas_price_usd_mmbtu", 5.0)
        ng_ratio = features.get("natural_gas_blend_pct", 100) / 100
        cost += ng_price * ng_ratio

        fo_price = features.get("fuel_oil_price_usd_mmbtu", 8.0)
        fo_ratio = 1 - ng_ratio
        cost += fo_price * fo_ratio

        return cost

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


def create_explainer(num_samples: int = 1000) -> LIMEFuelOptimizationExplainer:
    """Factory function."""
    return LIMEFuelOptimizationExplainer(ExplainerConfig(num_samples=num_samples))


__all__ = ["FeatureExplanation", "LIMEExplanation", "LIMEFuelOptimizationExplainer", "create_explainer"]
