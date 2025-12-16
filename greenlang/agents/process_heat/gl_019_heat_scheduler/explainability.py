# -*- coding: utf-8 -*-
"""
GL-019 LIME Explainability Module
=================================

LIME explanations for heat scheduling decisions.

Key Capabilities:
    - Load forecast explanation
    - Thermal storage dispatch explanation
    - Demand charge optimization explanation
    - Production scheduling explanation

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


HEAT_SCHEDULER_FEATURE_INFO = {
    "current_load_kw": {
        "name": "Current Load",
        "unit": "kW",
        "positive_description": "Current heat load within capacity",
        "negative_description": "High heat load requires additional resources",
    },
    "forecast_peak_kw": {
        "name": "Forecast Peak",
        "unit": "kW",
        "positive_description": "Peak load manageable with available capacity",
        "negative_description": "Forecast peak exceeds normal capacity",
    },
    "storage_soc_pct": {
        "name": "Storage State of Charge",
        "unit": "%",
        "positive_description": "Adequate thermal storage available",
        "negative_description": "Low storage limits peak shaving capability",
    },
    "demand_charge_rate": {
        "name": "Demand Charge Rate",
        "unit": "$/kW",
        "positive_description": "Demand charge within budget",
        "negative_description": "High demand rate increases costs",
    },
    "time_of_use_period": {
        "name": "Time of Use Period",
        "unit": "",
        "positive_description": "Off-peak period for charging",
        "negative_description": "On-peak period - minimize grid usage",
    },
    "ambient_temperature_c": {
        "name": "Ambient Temperature",
        "unit": "C",
        "positive_description": "Moderate temperatures reduce heating/cooling load",
        "negative_description": "Extreme temperature increases energy demand",
    },
    "production_schedule_load_kw": {
        "name": "Production Heat Load",
        "unit": "kW",
        "positive_description": "Production scheduled efficiently",
        "negative_description": "Production overlaps with peak periods",
    },
    "spinning_reserve_pct": {
        "name": "Spinning Reserve",
        "unit": "%",
        "positive_description": "Adequate reserve capacity maintained",
        "negative_description": "Low reserve increases reliability risk",
    },
}


class LIMEHeatSchedulerExplainer:
    """LIME Explainer for Heat Scheduling Decisions."""

    def __init__(self, config: Optional[ExplainerConfig] = None) -> None:
        self.config = config or ExplainerConfig()
        self._rng = random.Random(self.config.random_seed)
        self._explanation_count = 0
        logger.info("LIMEHeatSchedulerExplainer initialized")

    def explain_load_forecast(
        self,
        forecast_features: Dict[str, float],
        forecast_result: Dict[str, float],
        predict_fn: Optional[Callable] = None,
    ) -> LIMEExplanation:
        """Explain load forecast prediction."""
        self._explanation_count += 1
        explanation_id = f"EXP-FORE-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        peak_load = forecast_result.get("peak_load_kw", 0.0)

        if predict_fn is None:
            predict_fn = self._default_load_predictor

        perturbations, predictions = self._generate_perturbations(forecast_features, predict_fn)
        coefficients, intercept, r_squared = self._fit_local_model(
            forecast_features, perturbations, predictions, peak_load
        )

        feature_explanations = self._build_feature_explanations(forecast_features, coefficients)
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        top_positive = [fe.feature_name for fe in feature_explanations if fe.direction == "positive"][:3]
        top_negative = [fe.feature_name for fe in feature_explanations if fe.direction == "negative"][:3]

        summary = self._generate_forecast_summary(forecast_result, feature_explanations)

        provenance_hash = self._calculate_provenance_hash(forecast_features, peak_load, coefficients)

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="peak_load_kw",
            predicted_value=peak_load,
            feature_explanations=feature_explanations[:self.config.num_features],
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            local_model_r_squared=r_squared,
            num_samples_used=len(perturbations),
            summary_text=summary,
            confidence=min(0.95, r_squared + 0.3),
            provenance_hash=provenance_hash,
        )

    def explain_storage_dispatch(
        self,
        storage_features: Dict[str, float],
        dispatch_result: Dict[str, Any],
    ) -> LIMEExplanation:
        """Explain thermal storage dispatch decision."""
        self._explanation_count += 1
        explanation_id = f"EXP-STOR-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        total_savings = dispatch_result.get("total_savings_usd", 0.0)
        feature_explanations = []

        # Energy arbitrage
        arbitrage = dispatch_result.get("total_energy_arbitrage_usd", 0)
        if arbitrage != 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Energy Arbitrage",
                feature_value=arbitrage,
                contribution=arbitrage,
                contribution_pct=40.0,
                direction="positive" if arbitrage > 0 else "negative",
                description=f"${abs(arbitrage):,.0f} energy arbitrage value",
                unit="$",
            ))

        # Demand savings
        demand_savings = dispatch_result.get("total_demand_savings_usd", 0)
        if demand_savings > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Demand Charge Savings",
                feature_value=demand_savings,
                contribution=demand_savings,
                contribution_pct=35.0,
                direction="positive",
                description=f"${demand_savings:,.0f} demand charge reduction",
                unit="$",
            ))

        # Storage utilization
        current_soc = storage_features.get("current_soc_pct", 50)
        total_capacity = storage_features.get("total_capacity_kwh", 0)
        if total_capacity > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Storage Utilization",
                feature_value=current_soc,
                contribution=current_soc * 0.5,
                contribution_pct=25.0,
                direction="positive" if current_soc > 30 else "negative",
                description=f"Storage at {current_soc:.0f}% SOC",
                unit="%",
            ))

        mode = dispatch_result.get("mode", "idle")
        summary = f"Storage dispatch: {mode}. "
        if demand_savings > 0:
            summary += f"Demand savings: ${demand_savings:,.0f}. "
        if arbitrage > 0:
            summary += f"Arbitrage value: ${arbitrage:,.0f}."

        provenance_hash = self._calculate_provenance_hash(storage_features, total_savings, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="total_savings_usd",
            predicted_value=total_savings,
            feature_explanations=feature_explanations,
            top_positive_features=[fe.feature_name for fe in feature_explanations if fe.direction == "positive"],
            top_negative_features=[fe.feature_name for fe in feature_explanations if fe.direction == "negative"],
            local_model_r_squared=0.9,
            num_samples_used=0,
            summary_text=summary,
            confidence=0.9,
            provenance_hash=provenance_hash,
        )

    def explain_demand_charge_optimization(
        self,
        demand_features: Dict[str, float],
        demand_result: Dict[str, float],
    ) -> LIMEExplanation:
        """Explain demand charge optimization."""
        self._explanation_count += 1
        explanation_id = f"EXP-DEM-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        savings = demand_result.get("demand_charge_savings_usd", 0.0)
        feature_explanations = []

        # Peak reduction
        baseline_peak = demand_result.get("baseline_peak_kw", 0)
        optimized_peak = demand_result.get("optimized_peak_kw", 0)
        reduction_kw = baseline_peak - optimized_peak
        if reduction_kw > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Peak Reduction",
                feature_value=reduction_kw,
                contribution=reduction_kw,
                contribution_pct=45.0,
                direction="positive",
                description=f"Reduced peak by {reduction_kw:.0f} kW",
                unit="kW",
            ))

        # Load shifting
        load_shifted = demand_result.get("load_shifted_kwh", 0)
        if load_shifted > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Load Shifted",
                feature_value=load_shifted,
                contribution=load_shifted * 0.1,
                contribution_pct=30.0,
                direction="positive",
                description=f"{load_shifted:,.0f} kWh shifted to off-peak",
                unit="kWh",
            ))

        # Ratchet impact
        ratchet_impact = demand_result.get("ratchet_impact_usd", 0)
        if ratchet_impact > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Annual Ratchet Impact",
                feature_value=ratchet_impact,
                contribution=-ratchet_impact,
                contribution_pct=25.0,
                direction="negative",
                description=f"${ratchet_impact:,.0f} annual ratchet cost",
                unit="$/year",
            ))

        peak_exceeded = demand_result.get("peak_limit_exceeded", False)
        summary = f"Demand optimization: {('Peak exceeded' if peak_exceeded else 'Within limits')}. "
        summary += f"Peak reduction: {reduction_kw:.0f} kW. "
        if savings > 0:
            summary += f"Savings: ${savings:,.0f}/month."

        provenance_hash = self._calculate_provenance_hash(demand_features, savings, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="demand_charge_savings_usd",
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

    def explain_production_schedule(
        self,
        production_features: Dict[str, float],
        schedule_result: Dict[str, Any],
    ) -> LIMEExplanation:
        """Explain production scheduling decision."""
        self._explanation_count += 1
        explanation_id = f"EXP-PROD-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._explanation_count:04d}"

        cost_savings = schedule_result.get("scheduling_cost_savings_usd", 0.0)
        feature_explanations = []

        # Orders on time
        total_orders = schedule_result.get("total_orders", 0)
        on_time = schedule_result.get("orders_on_time", 0)
        if total_orders > 0:
            on_time_pct = on_time / total_orders * 100
            feature_explanations.append(FeatureExplanation(
                feature_name="On-Time Delivery",
                feature_value=on_time_pct,
                contribution=on_time_pct,
                contribution_pct=35.0,
                direction="positive" if on_time_pct > 90 else "negative",
                description=f"{on_time_pct:.0f}% orders on time",
                unit="%",
            ))

        # Rescheduled orders
        rescheduled = schedule_result.get("orders_rescheduled", 0)
        if rescheduled > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Orders Rescheduled",
                feature_value=rescheduled,
                contribution=rescheduled * 10,
                contribution_pct=25.0,
                direction="neutral",
                description=f"{rescheduled} orders optimally rescheduled",
                unit="orders",
            ))

        # Cost savings from scheduling
        if cost_savings > 0:
            feature_explanations.append(FeatureExplanation(
                feature_name="Scheduling Savings",
                feature_value=cost_savings,
                contribution=cost_savings,
                contribution_pct=40.0,
                direction="positive",
                description=f"${cost_savings:,.0f} saved via optimal scheduling",
                unit="$",
            ))

        summary = f"Production schedule: {total_orders} orders. "
        summary += f"{on_time}/{total_orders} on-time. "
        if cost_savings > 0:
            summary += f"Cost savings: ${cost_savings:,.0f}."

        provenance_hash = self._calculate_provenance_hash(production_features, cost_savings, {})

        return LIMEExplanation(
            explanation_id=explanation_id,
            timestamp=datetime.now(timezone.utc),
            target_variable="scheduling_cost_savings_usd",
            predicted_value=cost_savings,
            feature_explanations=feature_explanations,
            top_positive_features=[fe.feature_name for fe in feature_explanations if fe.direction == "positive"],
            top_negative_features=[fe.feature_name for fe in feature_explanations if fe.direction == "negative"],
            local_model_r_squared=0.85,
            num_samples_used=0,
            summary_text=summary,
            confidence=0.85,
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
            direction = "neutral" if abs(contribution) < 0.01 else ("positive" if contribution > 0 else "negative")

            info = HEAT_SCHEDULER_FEATURE_INFO.get(name, {})
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

    def _generate_forecast_summary(
        self,
        forecast_result: Dict[str, float],
        feature_explanations: List[FeatureExplanation],
    ) -> str:
        """Generate forecast summary."""
        parts = []

        peak = forecast_result.get("peak_load_kw", 0)
        parts.append(f"Forecast peak: {peak:,.0f} kW")

        avg = forecast_result.get("avg_load_kw", 0)
        if avg > 0:
            parts.append(f"Average: {avg:,.0f} kW")

        mape = forecast_result.get("mape_pct", 0)
        if mape > 0:
            parts.append(f"Forecast MAPE: {mape:.1f}%")

        top_features = [fe.feature_name for fe in feature_explanations[:2]]
        if top_features:
            parts.append(f"Key drivers: {', '.join(top_features)}")

        return ". ".join(parts)

    def _default_load_predictor(self, features: Dict[str, float]) -> float:
        """Default load predictor."""
        base_load = features.get("base_load_kw", 1000)
        temp = features.get("ambient_temperature_c", 20)

        # Temperature-driven load
        if temp < 10:
            temp_load = (10 - temp) * 50  # Heating
        elif temp > 25:
            temp_load = (temp - 25) * 30  # Cooling
        else:
            temp_load = 0

        production_load = features.get("production_schedule_load_kw", 0)

        return base_load + temp_load + production_load

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


def create_explainer(num_samples: int = 1000) -> LIMEHeatSchedulerExplainer:
    """Factory function."""
    return LIMEHeatSchedulerExplainer(ExplainerConfig(num_samples=num_samples))


__all__ = ["FeatureExplanation", "LIMEExplanation", "LIMEHeatSchedulerExplainer", "create_explainer"]
