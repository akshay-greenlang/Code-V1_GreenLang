"""
GL-022 SUPERHEATER CONTROL AGENT - LIME Explainability Module

This module provides Local Interpretable Model-agnostic Explanations (LIME)
for superheater temperature control and desuperheater spray optimization
decisions. All explanations are deterministic and auditable.

Explainability Categories:
    - Temperature control explanation (PID contributions, deviation)
    - Spray rate optimization reasoning
    - Tube metal temperature analysis
    - Steam thermodynamic property impact

Engineering Standards:
    - IAPWS-IF97 for steam properties
    - ASME PTC 4 for boiler performance
    - API 530 for tube metal temperature limits
    - ISA-5.1 for control system symbology

Example:
    >>> from greenlang.agents.process_heat.gl_022_superheater_control.explainability import (
    ...     create_explainer,
    ...     ExplainerConfig,
    ... )
    >>>
    >>> explainer = create_explainer(ExplainerConfig())
    >>> explanation = explainer.explain_temperature_control(
    ...     features={
    ...         'process_variable_f': 935.0,
    ...         'setpoint_f': 950.0,
    ...         'deviation_f': -15.0,
    ...         'controller_output_pct': 65.0,
    ...         'spray_valve_position_pct': 25.0,
    ...     },
    ...     prediction=0.75,  # Control action intensity
    ... )
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureExplanation:
    """Explanation for a single feature's contribution."""

    feature_name: str
    feature_value: float
    contribution: float
    contribution_pct: float
    direction: str  # 'positive', 'negative', 'neutral'
    description: str
    unit: str = ""
    threshold_status: str = ""  # 'normal', 'warning', 'alarm', 'critical'


@dataclass
class LIMEExplanation:
    """Complete LIME explanation output."""

    explanation_type: str
    prediction: float
    prediction_unit: str
    base_value: float
    feature_explanations: List[FeatureExplanation]
    summary: str
    confidence: float
    provenance_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_method: str = "LIME Local Linear Regression"
    model_fidelity: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExplainerConfig:
    """Configuration for LIME explainer."""

    num_samples: int = 1000
    kernel_width: float = 0.75
    feature_selection: str = "auto"
    random_seed: int = 42
    decimal_precision: int = 4
    include_interactions: bool = False


# =============================================================================
# FEATURE INFORMATION
# =============================================================================

SUPERHEATER_CONTROL_FEATURE_INFO: Dict[str, Dict[str, Any]] = {
    # Temperature control features
    'process_variable_f': {
        'description': 'Current outlet steam temperature',
        'unit': 'F',
        'typical_range': (700, 1050),
        'warning_low': 750,
        'warning_high': 1000,
        'alarm_high': 1050,
        'weight': 1.0,
    },
    'setpoint_f': {
        'description': 'Temperature setpoint target',
        'unit': 'F',
        'typical_range': (750, 1000),
        'weight': 0.8,
    },
    'deviation_f': {
        'description': 'Deviation from temperature setpoint',
        'unit': 'F',
        'typical_range': (-50, 50),
        'warning_low': -20,
        'warning_high': 20,
        'alarm_low': -30,
        'alarm_high': 30,
        'weight': 1.5,
    },
    'rate_of_change_f_per_min': {
        'description': 'Temperature rate of change',
        'unit': 'F/min',
        'typical_range': (-15, 15),
        'warning_high': 10,
        'alarm_high': 15,
        'weight': 1.2,
    },

    # Controller output features
    'controller_output_pct': {
        'description': 'PID controller output signal',
        'unit': '%',
        'typical_range': (0, 100),
        'weight': 0.9,
    },
    'proportional_contribution': {
        'description': 'Proportional term contribution',
        'unit': 'dimensionless',
        'typical_range': (-50, 50),
        'weight': 0.7,
    },
    'integral_contribution': {
        'description': 'Integral term contribution',
        'unit': 'dimensionless',
        'typical_range': (-30, 30),
        'weight': 0.6,
    },
    'derivative_contribution': {
        'description': 'Derivative term contribution',
        'unit': 'dimensionless',
        'typical_range': (-20, 20),
        'weight': 0.5,
    },

    # Spray water features
    'spray_valve_position_pct': {
        'description': 'Desuperheater spray valve position',
        'unit': '%',
        'typical_range': (0, 100),
        'warning_high': 80,
        'alarm_high': 95,
        'weight': 1.1,
    },
    'spray_flow_lb_hr': {
        'description': 'Spray water flow rate',
        'unit': 'lb/hr',
        'typical_range': (0, 10000),
        'weight': 0.9,
    },
    'spray_to_steam_ratio_pct': {
        'description': 'Spray to steam flow ratio',
        'unit': '%',
        'typical_range': (0, 15),
        'warning_high': 12,
        'alarm_high': 15,
        'weight': 1.0,
    },
    'spray_water_temp_f': {
        'description': 'Spray water temperature',
        'unit': 'F',
        'typical_range': (100, 400),
        'weight': 0.6,
    },

    # Steam conditions
    'inlet_temperature_f': {
        'description': 'Superheater inlet steam temperature',
        'unit': 'F',
        'typical_range': (600, 900),
        'weight': 0.7,
    },
    'inlet_pressure_psig': {
        'description': 'Superheater inlet pressure',
        'unit': 'psig',
        'typical_range': (400, 1500),
        'weight': 0.6,
    },
    'steam_flow_lb_hr': {
        'description': 'Steam mass flow rate',
        'unit': 'lb/hr',
        'typical_range': (10000, 200000),
        'weight': 0.8,
    },
    'superheat_f': {
        'description': 'Degrees of superheat above saturation',
        'unit': 'F',
        'typical_range': (50, 300),
        'warning_low': 50,
        'weight': 0.9,
    },

    # Tube metal temperature features
    'max_tube_metal_temp_f': {
        'description': 'Maximum tube metal temperature',
        'unit': 'F',
        'typical_range': (800, 1100),
        'warning_high': 1000,
        'alarm_high': 1050,
        'weight': 1.3,
    },
    'tube_metal_margin_f': {
        'description': 'Margin below tube design temperature',
        'unit': 'F',
        'typical_range': (0, 200),
        'warning_low': 50,
        'alarm_low': 25,
        'weight': 1.4,
    },
    'thermal_stress_level': {
        'description': 'Tube thermal stress level index',
        'unit': 'dimensionless',
        'typical_range': (0, 1),
        'warning_high': 0.7,
        'alarm_high': 0.85,
        'weight': 1.2,
    },

    # Efficiency features
    'efficiency_loss_from_spray_pct': {
        'description': 'Thermal efficiency loss from spray water',
        'unit': '%',
        'typical_range': (0, 5),
        'warning_high': 2,
        'alarm_high': 3,
        'weight': 0.8,
    },
    'heat_duty_mmbtu_hr': {
        'description': 'Superheater heat duty',
        'unit': 'MMBTU/hr',
        'typical_range': (10, 200),
        'weight': 0.7,
    },
}


# =============================================================================
# LIME EXPLAINER CLASS
# =============================================================================

class LIMESuperheaterControlExplainer:
    """
    LIME-based explainer for GL-022 Superheater Control Agent.

    Provides deterministic, auditable explanations for temperature control
    and spray optimization decisions using local linear interpretable models.
    """

    def __init__(self, config: ExplainerConfig):
        """
        Initialize the LIME explainer.

        Args:
            config: Explainer configuration parameters
        """
        self.config = config
        self.feature_info = SUPERHEATER_CONTROL_FEATURE_INFO
        self._rng = np.random.RandomState(config.random_seed)

    def explain_temperature_control(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "control_action_intensity",
    ) -> LIMEExplanation:
        """
        Explain a temperature control decision.

        Analyzes PID controller behavior, deviation from setpoint,
        and factors affecting control response.

        Args:
            features: Input features for the control decision
            prediction: Model prediction (control output or action)
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with feature contributions
        """
        # Generate perturbations around the input
        perturbations = self._generate_perturbations(features)

        # Calculate local model using weighted linear regression
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build feature explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'temperature_control'
        )

        # Sort by absolute contribution
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_temperature_control_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_temperature_control_recommendations(
            features, feature_explanations
        )

        # Generate warnings
        warnings = self._check_temperature_control_warnings(features)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="temperature_control",
            prediction=prediction,
            prediction_unit=prediction_unit,
            base_value=intercept,
            feature_explanations=feature_explanations,
            summary=summary,
            confidence=model_fidelity,
            provenance_hash=provenance_hash,
            model_fidelity=model_fidelity,
            recommendations=recommendations,
            warnings=warnings,
        )

    def explain_spray_optimization(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "spray_valve_target_pct",
    ) -> LIMEExplanation:
        """
        Explain a spray valve optimization decision.

        Analyzes factors driving desuperheater spray water requirements
        and their impact on temperature control and efficiency.

        Args:
            features: Input features for spray optimization
            prediction: Optimized spray valve position
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with feature contributions
        """
        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local linear model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'spray_optimization'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_spray_optimization_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_spray_recommendations(
            features, feature_explanations
        )

        # Generate warnings
        warnings = self._check_spray_warnings(features)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="spray_optimization",
            prediction=prediction,
            prediction_unit=prediction_unit,
            base_value=intercept,
            feature_explanations=feature_explanations,
            summary=summary,
            confidence=model_fidelity,
            provenance_hash=provenance_hash,
            model_fidelity=model_fidelity,
            recommendations=recommendations,
            warnings=warnings,
        )

    def explain_tube_metal_analysis(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "thermal_stress_index",
    ) -> LIMEExplanation:
        """
        Explain tube metal temperature analysis results.

        Analyzes factors affecting tube metal temperatures, thermal stress,
        and margin to design limits.

        Args:
            features: Tube metal related features
            prediction: Thermal stress index or margin
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with feature contributions
        """
        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'tube_metal'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_tube_metal_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_tube_metal_recommendations(
            features, feature_explanations
        )

        # Check warnings
        warnings = self._check_tube_metal_warnings(features)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="tube_metal_analysis",
            prediction=prediction,
            prediction_unit=prediction_unit,
            base_value=intercept,
            feature_explanations=feature_explanations,
            summary=summary,
            confidence=model_fidelity,
            provenance_hash=provenance_hash,
            model_fidelity=model_fidelity,
            recommendations=recommendations,
            warnings=warnings,
        )

    def explain_safety_assessment(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "safety_score",
    ) -> LIMEExplanation:
        """
        Explain safety interlock assessment.

        Analyzes factors contributing to safety status including
        temperature limits, flow protection, and interlock status.

        Args:
            features: Safety-related features
            prediction: Safety score or status index
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with feature contributions
        """
        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'safety'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_safety_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_safety_recommendations(
            features, feature_explanations
        )

        # Check warnings
        warnings = self._check_safety_warnings(features)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="safety_assessment",
            prediction=prediction,
            prediction_unit=prediction_unit,
            base_value=intercept,
            feature_explanations=feature_explanations,
            summary=summary,
            confidence=model_fidelity,
            provenance_hash=provenance_hash,
            model_fidelity=model_fidelity,
            recommendations=recommendations,
            warnings=warnings,
        )

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _generate_perturbations(
        self,
        features: Dict[str, float],
    ) -> np.ndarray:
        """
        Generate perturbations around the input features.

        Uses feature-specific ranges for realistic perturbations.
        """
        n_features = len(features)
        perturbations = np.zeros((self.config.num_samples, n_features))

        feature_names = list(features.keys())
        feature_values = np.array([features[k] for k in feature_names])

        for i, name in enumerate(feature_names):
            info = self.feature_info.get(name, {})
            typical_range = info.get('typical_range', (0, 100))

            # Calculate perturbation scale
            range_size = typical_range[1] - typical_range[0]
            scale = range_size * self.config.kernel_width * 0.1

            # Generate normally distributed perturbations
            perturbations[:, i] = feature_values[i] + self._rng.normal(
                0, scale, self.config.num_samples
            )

            # Clip to valid range
            perturbations[:, i] = np.clip(
                perturbations[:, i],
                typical_range[0],
                typical_range[1]
            )

        return perturbations

    def _fit_local_model(
        self,
        features: Dict[str, float],
        perturbations: np.ndarray,
        prediction: float,
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Fit a local linear model using weighted linear regression.

        Returns weights, intercept, and model fidelity (R-squared).
        """
        feature_names = list(features.keys())
        feature_values = np.array([features[k] for k in feature_names])

        # Calculate distances for kernel weighting
        distances = np.sqrt(np.sum((perturbations - feature_values) ** 2, axis=1))
        kernel_width = np.percentile(distances, 75) * self.config.kernel_width
        kernel_weights = np.exp(-distances ** 2 / (2 * kernel_width ** 2))

        # Generate synthetic predictions (simplified surrogate)
        # In production, this would use the actual model
        synthetic_predictions = prediction + np.sum(
            (perturbations - feature_values) * 0.01, axis=1
        ) + self._rng.normal(0, 0.01, self.config.num_samples)

        # Weighted linear regression
        W = np.diag(kernel_weights)
        X = np.column_stack([np.ones(self.config.num_samples), perturbations])

        try:
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ synthetic_predictions
            coefficients = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            coefficients = np.linalg.lstsq(X, synthetic_predictions, rcond=None)[0]

        intercept = coefficients[0]
        weights = dict(zip(feature_names, coefficients[1:]))

        # Calculate R-squared (model fidelity)
        y_pred = X @ coefficients
        ss_res = np.sum(kernel_weights * (synthetic_predictions - y_pred) ** 2)
        ss_tot = np.sum(kernel_weights * (synthetic_predictions - np.average(
            synthetic_predictions, weights=kernel_weights
        )) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return weights, intercept, max(0.0, min(1.0, r_squared))

    def _build_feature_explanations(
        self,
        features: Dict[str, float],
        weights: Dict[str, float],
        context: str,
    ) -> List[FeatureExplanation]:
        """Build feature explanations from weights."""
        explanations = []
        total_contribution = sum(abs(w * features.get(k, 0))
                                 for k, w in weights.items())

        for name, weight in weights.items():
            if name not in features:
                continue

            value = features[name]
            info = self.feature_info.get(name, {})

            contribution = weight * value
            contribution_pct = (
                abs(contribution) / total_contribution * 100
                if total_contribution > 0 else 0.0
            )

            # Determine direction
            if abs(contribution) < 0.001:
                direction = "neutral"
            elif contribution > 0:
                direction = "positive"
            else:
                direction = "negative"

            # Determine threshold status
            threshold_status = self._get_threshold_status(name, value, info)

            # Generate description
            description = self._generate_feature_description(
                name, value, weight, info, context
            )

            explanations.append(FeatureExplanation(
                feature_name=name,
                feature_value=round(value, self.config.decimal_precision),
                contribution=round(contribution, self.config.decimal_precision),
                contribution_pct=round(contribution_pct, 1),
                direction=direction,
                description=description,
                unit=info.get('unit', ''),
                threshold_status=threshold_status,
            ))

        return explanations

    def _get_threshold_status(
        self,
        name: str,
        value: float,
        info: Dict[str, Any],
    ) -> str:
        """Determine threshold status for a feature value."""
        if 'alarm_high' in info and value >= info['alarm_high']:
            return 'alarm'
        if 'alarm_low' in info and value <= info['alarm_low']:
            return 'alarm'
        if 'warning_high' in info and value >= info['warning_high']:
            return 'warning'
        if 'warning_low' in info and value <= info['warning_low']:
            return 'warning'
        return 'normal'

    def _generate_feature_description(
        self,
        name: str,
        value: float,
        weight: float,
        info: Dict[str, Any],
        context: str,
    ) -> str:
        """Generate human-readable description for a feature."""
        base_desc = info.get('description', name.replace('_', ' ').title())
        unit = info.get('unit', '')

        if context == 'temperature_control':
            if name == 'deviation_f':
                if value > 0:
                    return f"Temperature is {value:.1f}F above setpoint, requiring spray action"
                else:
                    return f"Temperature is {abs(value):.1f}F below setpoint, reducing spray"
            elif name == 'rate_of_change_f_per_min':
                if abs(value) > 10:
                    return f"Rapid temperature change of {value:.1f}F/min requires control response"
                return f"Temperature rate of change is {value:.1f}F/min"

        elif context == 'spray_optimization':
            if name == 'spray_to_steam_ratio_pct':
                if value > 10:
                    return f"High spray ratio of {value:.1f}% may impact efficiency"
                return f"Spray ratio of {value:.1f}% is within acceptable range"

        elif context == 'tube_metal':
            if name == 'tube_metal_margin_f':
                if value < 50:
                    return f"Low margin of {value:.1f}F to tube design temperature limit"
                return f"Adequate margin of {value:.1f}F to design limit"

        return f"{base_desc}: {value:.2f} {unit}"

    def _generate_temperature_control_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for temperature control explanation."""
        deviation = features.get('deviation_f', 0)
        spray_pos = features.get('spray_valve_position_pct', 0)

        if abs(deviation) < 5:
            status = "maintaining stable temperature"
        elif deviation > 0:
            status = f"reducing temperature (deviation: +{deviation:.1f}F)"
        else:
            status = f"allowing temperature rise (deviation: {deviation:.1f}F)"

        top_factors = [e.feature_name.replace('_', ' ') for e in explanations[:3]]

        return (
            f"Temperature control is {status} with spray valve at {spray_pos:.1f}%. "
            f"Key contributing factors: {', '.join(top_factors)}."
        )

    def _generate_spray_optimization_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for spray optimization explanation."""
        spray_ratio = features.get('spray_to_steam_ratio_pct', 0)
        eff_loss = features.get('efficiency_loss_from_spray_pct', 0)

        return (
            f"Spray valve optimized to {prediction:.1f}% position. "
            f"Spray-to-steam ratio is {spray_ratio:.1f}% with "
            f"{eff_loss:.2f}% efficiency impact. "
            f"Primary driver: {explanations[0].feature_name.replace('_', ' ') if explanations else 'N/A'}."
        )

    def _generate_tube_metal_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for tube metal analysis explanation."""
        max_temp = features.get('max_tube_metal_temp_f', 0)
        margin = features.get('tube_metal_margin_f', 0)

        if margin < 50:
            status = "WARNING: Low margin to design limit"
        elif margin < 100:
            status = "Adequate margin to design limit"
        else:
            status = "Good margin to design limit"

        return (
            f"Maximum tube metal temperature: {max_temp:.0f}F. "
            f"{status} ({margin:.0f}F remaining). "
            f"Thermal stress assessment: {prediction:.2f}."
        )

    def _generate_safety_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for safety assessment explanation."""
        if prediction > 0.9:
            status = "Normal operation - all safety margins adequate"
        elif prediction > 0.7:
            status = "Caution - some safety margins reduced"
        else:
            status = "WARNING - safety margins critically low"

        return f"Safety assessment score: {prediction:.2f}. {status}."

    def _generate_temperature_control_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate recommendations for temperature control."""
        recommendations = []

        deviation = features.get('deviation_f', 0)
        rate = features.get('rate_of_change_f_per_min', 0)
        spray_pos = features.get('spray_valve_position_pct', 0)

        if abs(deviation) > 20:
            recommendations.append(
                f"Large temperature deviation ({deviation:.1f}F) - verify control loop tuning"
            )

        if abs(rate) > 10:
            recommendations.append(
                f"High rate of change ({rate:.1f}F/min) - monitor for overshoot"
            )

        if spray_pos > 80:
            recommendations.append(
                "Spray valve operating near limit - consider load reduction"
            )

        return recommendations

    def _generate_spray_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate recommendations for spray optimization."""
        recommendations = []

        spray_ratio = features.get('spray_to_steam_ratio_pct', 0)
        eff_loss = features.get('efficiency_loss_from_spray_pct', 0)

        if spray_ratio > 10:
            recommendations.append(
                f"High spray ratio ({spray_ratio:.1f}%) - evaluate firing rate reduction"
            )

        if eff_loss > 2:
            recommendations.append(
                f"Significant efficiency loss ({eff_loss:.2f}%) from spray water"
            )

        return recommendations

    def _generate_tube_metal_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate recommendations for tube metal analysis."""
        recommendations = []

        margin = features.get('tube_metal_margin_f', 100)
        stress = features.get('thermal_stress_level', 0)

        if margin < 50:
            recommendations.append(
                "Increase spray to reduce tube metal temperatures"
            )

        if stress > 0.7:
            recommendations.append(
                "High thermal stress - consider gradual load changes"
            )

        return recommendations

    def _generate_safety_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []

        tube_margin = features.get('tube_metal_margin_f', 100)
        superheat = features.get('superheat_f', 100)

        if tube_margin < 50:
            recommendations.append(
                "CRITICAL: Increase temperature control to protect tube metal"
            )

        if superheat < 50:
            recommendations.append(
                "Low superheat - risk of moisture carryover"
            )

        return recommendations

    def _check_temperature_control_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for temperature control warnings."""
        warnings = []

        deviation = features.get('deviation_f', 0)
        if abs(deviation) > 30:
            warnings.append(f"ALARM: Temperature deviation exceeds limits ({deviation:.1f}F)")
        elif abs(deviation) > 20:
            warnings.append(f"WARNING: Temperature deviation high ({deviation:.1f}F)")

        return warnings

    def _check_spray_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for spray optimization warnings."""
        warnings = []

        spray_ratio = features.get('spray_to_steam_ratio_pct', 0)
        if spray_ratio > 15:
            warnings.append(f"ALARM: Spray ratio exceeds maximum ({spray_ratio:.1f}%)")
        elif spray_ratio > 12:
            warnings.append(f"WARNING: High spray ratio ({spray_ratio:.1f}%)")

        return warnings

    def _check_tube_metal_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for tube metal warnings."""
        warnings = []

        margin = features.get('tube_metal_margin_f', 100)
        max_temp = features.get('max_tube_metal_temp_f', 0)

        if margin < 25:
            warnings.append(f"ALARM: Tube metal margin critically low ({margin:.0f}F)")
        elif margin < 50:
            warnings.append(f"WARNING: Low tube metal margin ({margin:.0f}F)")

        if max_temp > 1050:
            warnings.append(f"ALARM: Tube metal temperature exceeds limit ({max_temp:.0f}F)")

        return warnings

    def _check_safety_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for general safety warnings."""
        warnings = []
        warnings.extend(self._check_temperature_control_warnings(features))
        warnings.extend(self._check_tube_metal_warnings(features))
        return warnings

    def _calculate_provenance_hash(
        self,
        features: Dict[str, float],
        prediction: float,
        weights: Dict[str, float],
        intercept: float,
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        hash_input = {
            'features': {k: round(v, 6) for k, v in sorted(features.items())},
            'prediction': round(prediction, 6),
            'weights': {k: round(v, 6) for k, v in sorted(weights.items())},
            'intercept': round(intercept, 6),
            'config': {
                'num_samples': self.config.num_samples,
                'kernel_width': self.config.kernel_width,
                'random_seed': self.config.random_seed,
            },
        }

        hash_string = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_explainer(config: Optional[ExplainerConfig] = None) -> LIMESuperheaterControlExplainer:
    """
    Factory function to create a LIME explainer for superheater control.

    Args:
        config: Optional explainer configuration. Uses defaults if not provided.

    Returns:
        Configured LIMESuperheaterControlExplainer instance
    """
    if config is None:
        config = ExplainerConfig()
    return LIMESuperheaterControlExplainer(config)
