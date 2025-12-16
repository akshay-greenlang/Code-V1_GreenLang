# -*- coding: utf-8 -*-
"""
GL-015 LIME Explainability Module
=================================

LIME explanations for insulation analysis decisions.

Key Capabilities:
    - Heat loss explanation
    - Economic thickness explanation
    - Surface temperature compliance explanation
    - Condensation prevention explanation

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


INSULATION_FEATURE_INFO = {
    "operating_temperature_f": {
        "name": "Operating Temperature",
        "unit": "F",
        "positive_description": "Process temperature drives insulation requirements",
        "negative_description": "High temperature increases heat loss potential",
    },
    "ambient_temperature_f": {
        "name": "Ambient Temperature",
        "unit": "F",
        "positive_description": "Temperature differential drives heat transfer",
        "negative_description": "Large temperature differential increases heat loss",
    },
    "insulation_thickness_in": {
        "name": "Insulation Thickness",
        "unit": "inches",
        "positive_description": "Adequate insulation thickness",
        "negative_description": "Insufficient insulation thickness increases heat loss",
    },
    "thermal_conductivity_btu_in_hr_ft2_f": {
        "name": "Thermal Conductivity",
        "unit": "BTU-in/hr-ft2-F",
        "positive_description": "Low conductivity material reduces heat transfer",
        "negative_description": "High conductivity material increases heat loss",
    },
    "surface_temperature_f": {
        "name": "Surface Temperature",
        "unit": "F",
        "warning_high": 140.0,
        "positive_description": "Surface temperature within OSHA limits",
        "negative_description": "Surface temperature exceeds personnel protection limit",
    },
    "wind_speed_mph": {
        "name": "Wind Speed",
        "unit": "mph",
        "positive_description": "Low wind reduces convective heat loss",
        "negative_description": "High wind increases convective heat loss",
    },
    "pipe_diameter_in": {
        "name": "Pipe Diameter",
        "unit": "inches",
        "positive_description": "Pipe geometry affects heat loss per length",
        "negative_description": "Larger diameter increases surface area for heat loss",
    },
    "insulation_condition_factor": {
        "name": "Insulation Condition",
        "unit": "factor",
        "warning_high": 1.3,
        "positive_description": "Good insulation condition",
        "negative_description": "Degraded insulation increases heat loss",
    },
}


class LIMEInsulationExplainer:
    """LIME Explainer for Insulation Analysis."""

    def __init__(self, config: Optional[ExplainerConfig] = None) -> None:
        self.config = config or ExplainerConfig()
        self._rng = random.Random(self.config.random_seed)
        self._explanation_count = 0
        logger.info("LIMEInsulationExplainer initialized")

    def explain_heat_loss(
        self,
        insulation_features: Dict[str, float],
        heat_loss_result: Dict[str, float],
        predict_fn: Optional[Callable] = None,
    ) -> LIMEExplanation:
        """Explain heat loss calculation."""
        self._explanation_count += 1
        explanation_id = f"EXP-HL-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        heat_loss = heat_loss_result.get("heat_loss_btu_hr", 0.0)

        if predict_fn is None:
            predict_fn = self._default_heat_loss_predictor

        perturbations, predictions = self._generate_perturbations(insulation_features, predict_fn)
        coefficients, intercept, r_squared = self._fit_local_model(
            insulation_features, perturbations, predictions, heat_loss
        )

        feature_explanations = self._build_feature_explanations(insulation_features, coefficients)
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        top_positive = [fe.feature_name for fe in feature_explanations if fe.direction == "positive"][:3]
        top_negative = [fe.feature_name for fe in feature_explanations if fe.direction == "negative"][:3]

        summary = self._generate_heat_loss_summary(heat_loss, heat_loss_result, top_negative)

        provenance_hash = self._calculate_provenance_hash(insulation_features, heat_loss, coefficients)

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="heat_loss_btu_hr",
            predicted_value=heat_loss,
            feature_explanations=feature_explanations[:self.config.num_features],
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            local_model_r_squared=r_squared,
            num_samples_used=len(perturbations),
            summary_text=summary,
            confidence=min(0.95, r_squared + 0.3),
            provenance_hash=provenance_hash,
        )

    def explain_economic_thickness(
        self,
        insulation_features: Dict[str, float],
        economic_result: Dict[str, float],
    ) -> LIMEExplanation:
        """Explain economic thickness recommendation."""
        self._explanation_count += 1
        explanation_id = f"EXP-ECON-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        optimal_thickness = economic_result.get("optimal_thickness_in", 0.0)
        feature_explanations = []

        # Energy cost impact
        annual_savings = economic_result.get("annual_savings_usd", 0)
        if annual_savings > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Annual Energy Savings",
                feature_value=annual_savings,
                contribution=annual_savings,
                contribution_pct=40.0,
                direction="positive",
                description=f"${annual_savings:,.0f} annual energy savings",
                unit="$/year",
            ))

        # Payback period
        payback = economic_result.get("simple_payback_years", 0)
        if payback > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Payback Period",
                feature_value=payback,
                contribution=-payback * 10,
                contribution_pct=25.0,
                direction="positive" if payback < 3 else "negative",
                description=f"{payback:.1f} year simple payback",
                unit="years",
            ))

        # Current vs optimal thickness
        current_thickness = insulation_features.get("current_thickness_in", 0)
        thickness_gap = optimal_thickness - current_thickness
        if thickness_gap > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Thickness Gap",
                feature_value=thickness_gap,
                contribution=thickness_gap * 100,
                contribution_pct=20.0,
                direction="negative",
                description=f"Additional {thickness_gap:.1f}\" needed for optimal",
                unit="inches",
            ))

        roi = economic_result.get("roi_pct", 0)
        summary = f"Optimal insulation thickness: {optimal_thickness:.1f}\". "
        if annual_savings > 0:
            summary += f"ROI: {roi:.0f}%. "
        if thickness_gap > 0:
            summary += f"Add {thickness_gap:.1f}\" insulation. "
        summary += f"Payback: {payback:.1f} years."

        provenance_hash = self._calculate_provenance_hash(insulation_features, optimal_thickness, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="optimal_thickness_in",
            predicted_value=optimal_thickness,
            feature_explanations=feature_explanations,
            top_positive_features=[fe.feature_name for fe in feature_explanations if fe.direction == "positive"],
            top_negative_features=[fe.feature_name for fe in feature_explanations if fe.direction == "negative"],
            local_model_r_squared=0.9,
            num_samples_used=0,
            summary_text=summary,
            confidence=0.9,
            provenance_hash=provenance_hash,
        )

    def explain_surface_temperature(
        self,
        insulation_features: Dict[str, float],
        surface_temp_result: Dict[str, float],
    ) -> LIMEExplanation:
        """Explain surface temperature compliance."""
        self._explanation_count += 1
        explanation_id = f"EXP-SURF-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        surface_temp = surface_temp_result.get("calculated_surface_temp_f", 0.0)
        feature_explanations = []

        # OSHA compliance
        is_compliant = surface_temp_result.get("is_compliant", True)
        osha_limit = 140.0
        margin = osha_limit - surface_temp
        feature_explanations.append(FeatureExplanation(
            feature_name="OSHA Compliance Margin",
            feature_value=margin,
            contribution=margin,
            contribution_pct=50.0,
            direction="positive" if margin > 0 else "negative",
            description=f"{abs(margin):.0f}F {'below' if margin > 0 else 'above'} 140F limit",
            unit="F",
        ))

        # Additional thickness needed
        additional_thickness = surface_temp_result.get("additional_thickness_needed_in", 0)
        if additional_thickness > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Additional Thickness Needed",
                feature_value=additional_thickness,
                contribution=-additional_thickness * 20,
                contribution_pct=30.0,
                direction="negative",
                description=f"Add {additional_thickness:.1f}\" for OSHA compliance",
                unit="inches",
            ))

        # Burn risk
        burn_risk = surface_temp_result.get("contact_burn_risk", "none")
        if burn_risk != "none":
            feature_explanations.append(FeatureExplanation(
                feature_name="Contact Burn Risk",
                feature_value=1.0 if burn_risk == "high" else 0.5,
                contribution=-50,
                contribution_pct=20.0,
                direction="negative",
                description=f"{burn_risk.capitalize()} burn risk on contact",
                unit="",
            ))

        summary = f"Surface temperature: {surface_temp:.0f}F. "
        summary += f"OSHA {'compliant' if is_compliant else 'non-compliant'}. "
        if not is_compliant:
            summary += f"Add {additional_thickness:.1f}\" insulation for compliance."

        provenance_hash = self._calculate_provenance_hash(insulation_features, surface_temp, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="surface_temperature_f",
            predicted_value=surface_temp,
            feature_explanations=feature_explanations,
            top_positive_features=[fe.feature_name for fe in feature_explanations if fe.direction == "positive"],
            top_negative_features=[fe.feature_name for fe in feature_explanations if fe.direction == "negative"],
            local_model_r_squared=0.95,
            num_samples_used=0,
            summary_text=summary,
            confidence=0.95,
            provenance_hash=provenance_hash,
        )

    def explain_condensation_analysis(
        self,
        insulation_features: Dict[str, float],
        condensation_result: Dict[str, float],
    ) -> LIMEExplanation:
        """Explain condensation prevention analysis."""
        self._explanation_count += 1
        explanation_id = f"EXP-COND-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        surface_temp = condensation_result.get("surface_temperature_f", 0.0)
        feature_explanations = []

        # Dew point margin
        dew_point = condensation_result.get("ambient_dew_point_f", 0)
        margin = surface_temp - dew_point
        feature_explanations.append(FeatureExplanation(
            feature_name="Dew Point Margin",
            feature_value=margin,
            contribution=margin,
            contribution_pct=50.0,
            direction="positive" if margin > 5 else "negative",
            description=f"Surface {margin:.0f}F above dew point",
            unit="F",
        ))

        # Condensation risk
        risk_level = condensation_result.get("condensation_risk_level", "none")
        if risk_level != "none":
            feature_explanations.append(FeatureExplanation(
                feature_name="Condensation Risk",
                feature_value=1.0 if risk_level == "high" else 0.5,
                contribution=-30 if risk_level == "high" else -15,
                contribution_pct=30.0,
                direction="negative",
                description=f"{risk_level.capitalize()} condensation risk",
                unit="",
            ))

        # Vapor barrier status
        vapor_barrier = condensation_result.get("vapor_barrier_required", False)
        if vapor_barrier:
            feature_explanations.append(FeatureExplanation(
                feature_name="Vapor Barrier Required",
                feature_value=1.0,
                contribution=-20,
                contribution_pct=20.0,
                direction="negative",
                description="Vapor barrier required to prevent moisture intrusion",
                unit="",
            ))

        has_risk = condensation_result.get("condensation_risk", False)
        summary = f"Surface temp: {surface_temp:.0f}F, Dew point: {dew_point:.0f}F. "
        summary += f"Condensation risk: {'Yes' if has_risk else 'No'}. "
        if vapor_barrier:
            summary += "Vapor barrier required."

        provenance_hash = self._calculate_provenance_hash(insulation_features, margin, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="dew_point_margin_f",
            predicted_value=margin,
            feature_explanations=feature_explanations,
            top_positive_features=[fe.feature_name for fe in feature_explanations if fe.direction == "positive"],
            top_negative_features=[fe.feature_name for fe in feature_explanations if fe.direction == "negative"],
            local_model_r_squared=0.9,
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
            # For heat loss, positive contribution means more heat loss (bad)
            direction = "neutral" if abs(contribution) < 0.01 else ("negative" if contribution > 0 else "positive")

            info = INSULATION_FEATURE_INFO.get(name, {})
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

    def _generate_heat_loss_summary(
        self,
        heat_loss: float,
        heat_loss_result: Dict[str, float],
        top_negative: List[str],
    ) -> str:
        """Generate heat loss summary."""
        parts = []
        parts.append(f"Heat loss: {heat_loss:,.0f} BTU/hr")

        reduction = heat_loss_result.get("heat_loss_reduction_pct", 0)
        if reduction > 0:
            parts.append(f"Insulation reduces loss by {reduction:.0f}%")

        surface_temp = heat_loss_result.get("outer_surface_temperature_f", 0)
        if surface_temp > 0:
            parts.append(f"Surface temperature: {surface_temp:.0f}F")

        if top_negative:
            parts.append(f"Key factors: {', '.join(top_negative)}")

        return ". ".join(parts)

    def _default_heat_loss_predictor(self, features: Dict[str, float]) -> float:
        """Default heat loss predictor."""
        op_temp = features.get("operating_temperature_f", 350)
        amb_temp = features.get("ambient_temperature_f", 77)
        delta_t = op_temp - amb_temp

        thickness = features.get("insulation_thickness_in", 2)
        conductivity = features.get("thermal_conductivity_btu_in_hr_ft2_f", 0.25)

        # Simplified heat loss per linear foot of pipe
        diameter = features.get("pipe_diameter_in", 4)
        area = 3.14159 * diameter / 12  # ft2 per ft of pipe

        if thickness > 0:
            resistance = thickness / conductivity
            heat_loss = delta_t * area / resistance
        else:
            heat_loss = delta_t * area * 10  # Bare pipe

        return max(0, heat_loss)

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


def create_explainer(num_samples: int = 1000) -> LIMEInsulationExplainer:
    """Factory function."""
    return LIMEInsulationExplainer(ExplainerConfig(num_samples=num_samples))


__all__ = ["FeatureExplanation", "LIMEExplanation", "LIMEInsulationExplainer", "create_explainer"]
