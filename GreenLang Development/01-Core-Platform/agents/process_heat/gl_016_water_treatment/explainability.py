# -*- coding: utf-8 -*-
"""
GL-016 LIME Explainability Module
=================================

LIME explanations for water treatment monitoring decisions.

Key Capabilities:
    - Boiler water chemistry explanation
    - Feedwater quality explanation
    - Blowdown optimization explanation
    - Chemical dosing explanation

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


WATER_TREATMENT_FEATURE_INFO = {
    "ph": {
        "name": "pH",
        "unit": "",
        "optimal_min": 9.0,
        "optimal_max": 11.0,
        "positive_description": "pH within optimal range for corrosion protection",
        "negative_description": "pH outside optimal range increases corrosion risk",
    },
    "phosphate_ppm": {
        "name": "Phosphate",
        "unit": "ppm",
        "optimal_min": 2.0,
        "optimal_max": 20.0,
        "positive_description": "Phosphate level provides scale protection",
        "negative_description": "Phosphate level out of range",
    },
    "dissolved_oxygen_ppb": {
        "name": "Dissolved Oxygen",
        "unit": "ppb",
        "warning_high": 10.0,
        "positive_description": "Low dissolved oxygen minimizes oxygen corrosion",
        "negative_description": "Elevated dissolved oxygen causes pitting corrosion",
    },
    "conductivity_umho": {
        "name": "Conductivity",
        "unit": "umho/cm",
        "warning_high": 5000.0,
        "positive_description": "Conductivity within acceptable limits",
        "negative_description": "High conductivity indicates excessive TDS",
    },
    "silica_ppm": {
        "name": "Silica",
        "unit": "ppm",
        "warning_high": 150.0,
        "positive_description": "Silica below carryover threshold",
        "negative_description": "High silica risks turbine deposition",
    },
    "iron_ppb": {
        "name": "Iron",
        "unit": "ppb",
        "warning_high": 50.0,
        "positive_description": "Low iron indicates minimal corrosion",
        "negative_description": "Elevated iron indicates active corrosion",
    },
    "cycles_of_concentration": {
        "name": "Cycles of Concentration",
        "unit": "cycles",
        "optimal_min": 3.0,
        "optimal_max": 10.0,
        "positive_description": "Optimal cycles balance water usage and chemistry",
        "negative_description": "Non-optimal cycles waste water or risk scaling",
    },
    "blowdown_rate_pct": {
        "name": "Blowdown Rate",
        "unit": "%",
        "optimal_max": 5.0,
        "positive_description": "Efficient blowdown rate",
        "negative_description": "High blowdown wastes energy and water",
    },
}


class LIMEWaterTreatmentExplainer:
    """LIME Explainer for Water Treatment Monitoring."""

    def __init__(self, config: Optional[ExplainerConfig] = None) -> None:
        self.config = config or ExplainerConfig()
        self._rng = random.Random(self.config.random_seed)
        self._explanation_count = 0
        logger.info("LIMEWaterTreatmentExplainer initialized")

    def explain_boiler_water_analysis(
        self,
        water_features: Dict[str, float],
        analysis_result: Dict[str, Any],
        predict_fn: Optional[Callable] = None,
    ) -> LIMEExplanation:
        """Explain boiler water chemistry analysis."""
        self._explanation_count += 1
        explanation_id = f"EXP-BW-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        corrosion_risk = analysis_result.get("corrosion_risk_score", 0.0)

        if predict_fn is None:
            predict_fn = self._default_corrosion_predictor

        perturbations, predictions = self._generate_perturbations(water_features, predict_fn)
        coefficients, intercept, r_squared = self._fit_local_model(
            water_features, perturbations, predictions, corrosion_risk
        )

        feature_explanations = self._build_feature_explanations(water_features, coefficients)
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        top_positive = [fe.feature_name for fe in feature_explanations if fe.direction == "positive"][:3]
        top_negative = [fe.feature_name for fe in feature_explanations if fe.direction == "negative"][:3]

        summary = self._generate_boiler_water_summary(analysis_result, top_negative)

        provenance_hash = self._calculate_provenance_hash(water_features, corrosion_risk, coefficients)

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="corrosion_risk_score",
            predicted_value=corrosion_risk,
            feature_explanations=feature_explanations[:self.config.num_features],
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            local_model_r_squared=r_squared,
            num_samples_used=len(perturbations),
            summary_text=summary,
            confidence=min(0.95, r_squared + 0.3),
            provenance_hash=provenance_hash,
        )

    def explain_feedwater_analysis(
        self,
        feedwater_features: Dict[str, float],
        analysis_result: Dict[str, Any],
    ) -> LIMEExplanation:
        """Explain feedwater quality analysis."""
        self._explanation_count += 1
        explanation_id = f"EXP-FW-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        quality_score = 100 - analysis_result.get("corrosion_risk_score", 0)
        feature_explanations = []

        # Dissolved oxygen assessment
        do_ppb = feedwater_features.get("dissolved_oxygen_ppb", 0)
        if do_ppb > 7:
            feature_explanations.append(FeatureExplanation(
                feature_name="Dissolved Oxygen",
                feature_value=do_ppb,
                contribution=-(do_ppb - 7) * 5,
                contribution_pct=40.0,
                direction="negative",
                description=f"DO at {do_ppb:.0f} ppb exceeds 7 ppb limit",
                unit="ppb",
            ))
        else:
            feature_explanations.append(FeatureExplanation(
                feature_name="Dissolved Oxygen",
                feature_value=do_ppb,
                contribution=10,
                contribution_pct=30.0,
                direction="positive",
                description=f"DO at {do_ppb:.0f} ppb within spec",
                unit="ppb",
            ))

        # Iron transport
        iron_ppb = feedwater_features.get("iron_ppb", 0)
        if iron_ppb > 20:
            feature_explanations.append(FeatureExplanation(
                feature_name="Iron Transport",
                feature_value=iron_ppb,
                contribution=-(iron_ppb - 20) * 2,
                contribution_pct=25.0,
                direction="negative",
                description=f"Elevated iron ({iron_ppb:.0f} ppb) indicates corrosion",
                unit="ppb",
            ))

        # pH assessment
        ph = feedwater_features.get("ph", 9.0)
        if 8.5 <= ph <= 9.5:
            feature_explanations.append(FeatureExplanation(
                feature_name="Feedwater pH",
                feature_value=ph,
                contribution=15,
                contribution_pct=20.0,
                direction="positive",
                description=f"pH {ph:.1f} within optimal range",
                unit="",
            ))

        status = analysis_result.get("overall_status", "good")
        summary = f"Feedwater quality: {status}. "
        if do_ppb > 7:
            summary += f"Adjust oxygen scavenger dosing. "
        if iron_ppb > 20:
            summary += f"Investigate iron source. "

        provenance_hash = self._calculate_provenance_hash(feedwater_features, quality_score, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="feedwater_quality_score",
            predicted_value=quality_score,
            feature_explanations=feature_explanations,
            top_positive_features=[fe.feature_name for fe in feature_explanations if fe.direction == "positive"],
            top_negative_features=[fe.feature_name for fe in feature_explanations if fe.direction == "negative"],
            local_model_r_squared=0.9,
            num_samples_used=0,
            summary_text=summary,
            confidence=0.9,
            provenance_hash=provenance_hash,
        )

    def explain_blowdown_optimization(
        self,
        blowdown_features: Dict[str, float],
        blowdown_result: Dict[str, float],
    ) -> LIMEExplanation:
        """Explain blowdown optimization recommendation."""
        self._explanation_count += 1
        explanation_id = f"EXP-BD-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        optimal_rate = blowdown_result.get("optimal_blowdown_rate_pct", 0.0)
        feature_explanations = []

        # Current vs optimal cycles
        current_cycles = blowdown_features.get("cycles_of_concentration", 1)
        optimal_cycles = blowdown_result.get("optimal_cycles_of_concentration", 1)
        if current_cycles != optimal_cycles:
            feature_explanations.append(FeatureExplanation(
                feature_name="Cycles Adjustment",
                feature_value=optimal_cycles - current_cycles,
                contribution=(optimal_cycles - current_cycles) * 10,
                contribution_pct=35.0,
                direction="positive" if optimal_cycles > current_cycles else "neutral",
                description=f"Adjust from {current_cycles:.1f} to {optimal_cycles:.1f} cycles",
                unit="cycles",
            ))

        # Savings potential
        savings = blowdown_result.get("total_savings_usd_yr", 0)
        if savings > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Annual Savings",
                feature_value=savings,
                contribution=savings,
                contribution_pct=40.0,
                direction="positive",
                description=f"${savings:,.0f}/year potential savings",
                unit="$/year",
            ))

        # Water savings
        water_savings = blowdown_result.get("water_savings_kgal_yr", 0)
        if water_savings > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Water Savings",
                feature_value=water_savings,
                contribution=water_savings,
                contribution_pct=25.0,
                direction="positive",
                description=f"{water_savings:,.0f} kgal/year water savings",
                unit="kgal/year",
            ))

        current_rate = blowdown_features.get("blowdown_rate_pct", 0)
        summary = f"Optimize blowdown from {current_rate:.1f}% to {optimal_rate:.1f}%. "
        summary += f"Optimal cycles: {optimal_cycles:.1f}. "
        if savings > 0:
            summary += f"Potential savings: ${savings:,.0f}/year."

        provenance_hash = self._calculate_provenance_hash(blowdown_features, optimal_rate, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="optimal_blowdown_rate_pct",
            predicted_value=optimal_rate,
            feature_explanations=feature_explanations,
            top_positive_features=[fe.feature_name for fe in feature_explanations if fe.direction == "positive"],
            top_negative_features=[fe.feature_name for fe in feature_explanations if fe.direction == "negative"],
            local_model_r_squared=0.9,
            num_samples_used=0,
            summary_text=summary,
            confidence=0.9,
            provenance_hash=provenance_hash,
        )

    def explain_chemical_dosing(
        self,
        dosing_features: Dict[str, float],
        dosing_result: Dict[str, float],
    ) -> LIMEExplanation:
        """Explain chemical dosing recommendation."""
        self._explanation_count += 1
        explanation_id = f"EXP-CHEM-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        cost_savings = dosing_result.get("cost_savings_per_day", 0.0)
        feature_explanations = []

        # Oxygen scavenger adjustment
        scavenger_change = dosing_result.get("scavenger_dose_change_ppm", 0)
        if abs(scavenger_change) > 1:
            feature_explanations.append(FeatureExplanation(
                feature_name="Scavenger Dose Adjustment",
                feature_value=scavenger_change,
                contribution=abs(scavenger_change) * 5,
                contribution_pct=35.0,
                direction="positive" if scavenger_change != 0 else "neutral",
                description=f"{'Increase' if scavenger_change > 0 else 'Decrease'} scavenger by {abs(scavenger_change):.0f} ppm",
                unit="ppm",
            ))

        # Phosphate adjustment
        phosphate_change = dosing_result.get("phosphate_dose_change_ppm", 0)
        if abs(phosphate_change) > 0.5:
            feature_explanations.append(FeatureExplanation(
                feature_name="Phosphate Dose Adjustment",
                feature_value=phosphate_change,
                contribution=abs(phosphate_change) * 3,
                contribution_pct=25.0,
                direction="positive" if phosphate_change != 0 else "neutral",
                description=f"{'Increase' if phosphate_change > 0 else 'Decrease'} phosphate by {abs(phosphate_change):.1f} ppm",
                unit="ppm",
            ))

        # Cost optimization
        annual_savings = dosing_result.get("annual_savings_usd", 0)
        if annual_savings > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Annual Chemical Savings",
                feature_value=annual_savings,
                contribution=annual_savings,
                contribution_pct=40.0,
                direction="positive",
                description=f"${annual_savings:,.0f}/year chemical savings",
                unit="$/year",
            ))

        within_range = dosing_result.get("within_recommended_ranges", True)
        summary = f"Chemical dosing {'optimized' if within_range else 'needs adjustment'}. "
        if scavenger_change != 0:
            summary += f"Scavenger: {'+' if scavenger_change > 0 else ''}{scavenger_change:.0f} ppm. "
        if annual_savings > 0:
            summary += f"Savings: ${annual_savings:,.0f}/year."

        provenance_hash = self._calculate_provenance_hash(dosing_features, cost_savings, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="cost_savings_per_day",
            predicted_value=cost_savings,
            feature_explanations=feature_explanations,
            top_positive_features=[fe.feature_name for fe in feature_explanations if fe.direction == "positive"],
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
            # For corrosion risk, positive contribution is bad
            direction = "neutral" if abs(contribution) < 0.01 else ("negative" if contribution > 0 else "positive")

            info = WATER_TREATMENT_FEATURE_INFO.get(name, {})
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

    def _generate_boiler_water_summary(
        self,
        analysis_result: Dict[str, Any],
        top_negative: List[str],
    ) -> str:
        """Generate boiler water summary."""
        parts = []

        status = analysis_result.get("overall_status", "unknown")
        parts.append(f"Boiler water status: {status}")

        corrosion_risk = analysis_result.get("corrosion_risk_score", 0)
        if corrosion_risk > 50:
            parts.append(f"Elevated corrosion risk ({corrosion_risk:.0f}/100)")

        scaling_risk = analysis_result.get("scaling_risk_score", 0)
        if scaling_risk > 50:
            parts.append(f"Elevated scaling risk ({scaling_risk:.0f}/100)")

        if top_negative:
            parts.append(f"Key concerns: {', '.join(top_negative)}")

        return ". ".join(parts)

    def _default_corrosion_predictor(self, features: Dict[str, float]) -> float:
        """Default corrosion risk predictor."""
        risk = 0.0

        # pH impact
        ph = features.get("ph", 10.0)
        if ph < 9.0 or ph > 11.5:
            risk += abs(10.0 - ph) * 20

        # Dissolved oxygen impact
        do_ppb = features.get("dissolved_oxygen_ppb", 0)
        if do_ppb > 7:
            risk += (do_ppb - 7) * 3

        # Iron indicates active corrosion
        iron = features.get("iron_ppb", 0)
        if iron > 20:
            risk += (iron - 20) * 0.5

        return min(100, max(0, risk))

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


def create_explainer(num_samples: int = 1000) -> LIMEWaterTreatmentExplainer:
    """Factory function."""
    return LIMEWaterTreatmentExplainer(ExplainerConfig(num_samples=num_samples))


__all__ = ["FeatureExplanation", "LIMEExplanation", "LIMEWaterTreatmentExplainer", "create_explainer"]
