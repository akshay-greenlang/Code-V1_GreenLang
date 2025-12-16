"""
GL-024 AIR PREHEATER AGENT - LIME Explainability Module

This module provides Local Interpretable Model-agnostic Explanations (LIME)
for air preheater optimization and monitoring decisions. All explanations
are deterministic and auditable.

Explainability Categories:
    - Heat transfer performance explanation (effectiveness, LMTD drivers)
    - Leakage analysis explanation (O2 differential, seal condition)
    - Cold-end protection explanation (acid dew point, corrosion risk)
    - Fouling assessment explanation (cleanliness factor drivers)
    - Optimization decision explanation (setpoint recommendations)

Engineering Standards:
    - ASME PTC 4.3 for Air Heaters Performance Test Code
    - API 560 for Fired Heaters
    - NFPA 85/86 for Combustion Safety
    - VDI 2055 for Thermal Insulation

Air Preheater Types:
    - Regenerative (Ljungstrom rotating, Rothemuhle stationary)
    - Recuperative (tubular, plate)
    - Heat pipe

Example:
    >>> from greenlang.agents.process_heat.gl_024_air_preheater.explainability import (
    ...     create_explainer,
    ...     ExplainerConfig,
    ... )
    >>>
    >>> explainer = create_explainer(ExplainerConfig())
    >>> explanation = explainer.explain_heat_transfer_performance(
    ...     features={
    ...         'gas_inlet_temp_c': 350.0,
    ...         'gas_outlet_temp_c': 150.0,
    ...         'air_inlet_temp_c': 30.0,
    ...         'air_outlet_temp_c': 280.0,
    ...         'effectiveness': 0.72,
    ...     },
    ...     prediction=0.72,
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

AIR_PREHEATER_FEATURE_INFO: Dict[str, Dict[str, Any]] = {
    # Gas-side temperature features
    'gas_inlet_temp_c': {
        'description': 'Flue gas inlet temperature (hot side in)',
        'unit': 'C',
        'typical_range': (200, 500),
        'warning_low': 180,
        'warning_high': 450,
        'alarm_high': 500,
        'weight': 1.0,
    },
    'gas_outlet_temp_c': {
        'description': 'Flue gas outlet temperature (cold end)',
        'unit': 'C',
        'typical_range': (100, 200),
        'warning_low': 90,
        'warning_high': 180,
        'alarm_low': 80,
        'weight': 1.2,
    },
    'gas_flow_kg_s': {
        'description': 'Flue gas mass flow rate',
        'unit': 'kg/s',
        'typical_range': (10, 500),
        'weight': 0.8,
    },
    'gas_temp_drop_c': {
        'description': 'Flue gas temperature drop across APH',
        'unit': 'C',
        'typical_range': (100, 300),
        'warning_low': 80,
        'weight': 0.9,
    },

    # Air-side temperature features
    'air_inlet_temp_c': {
        'description': 'Combustion air inlet temperature (ambient)',
        'unit': 'C',
        'typical_range': (-20, 50),
        'warning_low': -10,
        'warning_high': 45,
        'weight': 0.7,
    },
    'air_outlet_temp_c': {
        'description': 'Preheated combustion air outlet temperature',
        'unit': 'C',
        'typical_range': (150, 350),
        'warning_low': 120,
        'warning_high': 320,
        'weight': 1.0,
    },
    'air_flow_kg_s': {
        'description': 'Combustion air mass flow rate',
        'unit': 'kg/s',
        'typical_range': (10, 500),
        'weight': 0.8,
    },
    'air_temp_rise_c': {
        'description': 'Air temperature rise across APH',
        'unit': 'C',
        'typical_range': (150, 300),
        'warning_low': 100,
        'weight': 0.9,
    },

    # O2 concentration features (for leakage detection)
    'o2_inlet_pct': {
        'description': 'O2 concentration at APH gas inlet',
        'unit': '%',
        'typical_range': (2.0, 6.0),
        'warning_high': 5.0,
        'alarm_high': 7.0,
        'weight': 1.1,
    },
    'o2_outlet_pct': {
        'description': 'O2 concentration at APH gas outlet',
        'unit': '%',
        'typical_range': (2.5, 8.0),
        'warning_high': 6.5,
        'alarm_high': 9.0,
        'weight': 1.1,
    },
    'o2_rise_pct': {
        'description': 'O2 rise across APH (leakage indicator)',
        'unit': '%',
        'typical_range': (0.3, 2.0),
        'warning_high': 1.5,
        'alarm_high': 2.5,
        'weight': 1.3,
    },

    # SO2/SO3 and acid dew point features
    'so2_ppm': {
        'description': 'SO2 concentration in flue gas',
        'unit': 'ppm',
        'typical_range': (0, 2000),
        'warning_high': 1500,
        'alarm_high': 2500,
        'weight': 0.8,
    },
    'so3_ppm': {
        'description': 'SO3 concentration in flue gas',
        'unit': 'ppm',
        'typical_range': (0, 50),
        'warning_high': 30,
        'alarm_high': 50,
        'weight': 1.0,
    },
    'fuel_sulfur_pct': {
        'description': 'Fuel sulfur content',
        'unit': '%',
        'typical_range': (0, 4.0),
        'warning_high': 2.5,
        'alarm_high': 4.0,
        'weight': 1.1,
    },
    'acid_dew_point_c': {
        'description': 'Calculated acid dew point temperature',
        'unit': 'C',
        'typical_range': (100, 160),
        'weight': 1.2,
    },

    # Pressure drop features
    'gas_side_dp_mbar': {
        'description': 'Gas-side pressure drop across APH',
        'unit': 'mbar',
        'typical_range': (5, 30),
        'warning_high': 25,
        'alarm_high': 35,
        'weight': 0.9,
    },
    'air_side_dp_mbar': {
        'description': 'Air-side pressure drop across APH',
        'unit': 'mbar',
        'typical_range': (5, 25),
        'warning_high': 20,
        'alarm_high': 30,
        'weight': 0.9,
    },
    'dp_ratio': {
        'description': 'Ratio of current to design pressure drop',
        'unit': 'dimensionless',
        'typical_range': (0.8, 1.5),
        'warning_high': 1.3,
        'alarm_high': 1.5,
        'weight': 1.0,
    },

    # Leakage features
    'leakage_pct': {
        'description': 'Air-to-gas leakage percentage',
        'unit': '%',
        'typical_range': (3, 15),
        'warning_high': 10,
        'alarm_high': 15,
        'weight': 1.2,
    },
    'seal_gap_mm': {
        'description': 'Radial seal gap measurement',
        'unit': 'mm',
        'typical_range': (2, 10),
        'warning_high': 8,
        'alarm_high': 12,
        'weight': 0.9,
    },
    'leakage_heat_loss_kw': {
        'description': 'Heat loss due to air leakage',
        'unit': 'kW',
        'typical_range': (0, 500),
        'warning_high': 300,
        'alarm_high': 500,
        'weight': 1.0,
    },

    # Effectiveness and cleanliness features
    'effectiveness': {
        'description': 'Heat exchanger thermal effectiveness',
        'unit': 'dimensionless',
        'typical_range': (0.50, 0.85),
        'warning_low': 0.55,
        'alarm_low': 0.45,
        'weight': 1.3,
    },
    'design_effectiveness': {
        'description': 'Design thermal effectiveness',
        'unit': 'dimensionless',
        'typical_range': (0.65, 0.85),
        'weight': 0.8,
    },
    'cleanliness_factor': {
        'description': 'APH cleanliness factor (actual/design UA)',
        'unit': 'dimensionless',
        'typical_range': (0.70, 1.00),
        'warning_low': 0.75,
        'alarm_low': 0.65,
        'weight': 1.2,
    },
    'effectiveness_degradation_pct': {
        'description': 'Effectiveness degradation from design',
        'unit': '%',
        'typical_range': (0, 25),
        'warning_high': 15,
        'alarm_high': 25,
        'weight': 1.1,
    },

    # Cold end temperature margin
    'cold_end_temp_margin_c': {
        'description': 'Margin above acid dew point temperature',
        'unit': 'C',
        'typical_range': (15, 60),
        'warning_low': 20,
        'alarm_low': 10,
        'weight': 1.4,
    },

    # Operating conditions
    'boiler_load_pct': {
        'description': 'Boiler load percentage',
        'unit': '%',
        'typical_range': (40, 100),
        'warning_low': 50,
        'weight': 0.7,
    },
    'rotation_speed_rpm': {
        'description': 'Regenerative APH rotation speed',
        'unit': 'rpm',
        'typical_range': (0.8, 1.5),
        'warning_low': 0.7,
        'warning_high': 1.6,
        'weight': 0.6,
    },

    # Heat recovery metrics
    'heat_recovery_kw': {
        'description': 'Heat recovered by air preheater',
        'unit': 'kW',
        'typical_range': (100, 10000),
        'weight': 1.0,
    },
    'lmtd_c': {
        'description': 'Log mean temperature difference',
        'unit': 'C',
        'typical_range': (30, 150),
        'warning_low': 25,
        'weight': 0.9,
    },
    'ntu': {
        'description': 'Number of transfer units',
        'unit': 'dimensionless',
        'typical_range': (1.0, 4.0),
        'weight': 0.8,
    },

    # Efficiency metrics
    'boiler_efficiency_gain_pct': {
        'description': 'Efficiency gain from air preheating',
        'unit': '%',
        'typical_range': (1.0, 5.0),
        'warning_low': 1.5,
        'weight': 1.0,
    },
    'fuel_savings_pct': {
        'description': 'Fuel savings from heat recovery',
        'unit': '%',
        'typical_range': (1.0, 5.0),
        'weight': 1.0,
    },

    # Corrosion indicators
    'corrosion_rate_mm_yr': {
        'description': 'Estimated cold-end corrosion rate',
        'unit': 'mm/year',
        'typical_range': (0, 0.5),
        'warning_high': 0.2,
        'alarm_high': 0.4,
        'weight': 1.1,
    },
    'metal_temp_cold_end_c': {
        'description': 'Metal temperature at cold end',
        'unit': 'C',
        'typical_range': (90, 180),
        'warning_low': 100,
        'alarm_low': 90,
        'weight': 1.2,
    },
}


# =============================================================================
# LIME EXPLAINER CLASS
# =============================================================================

class LIMEAirPreheaterExplainer:
    """
    LIME-based explainer for GL-024 Air Preheater Agent.

    Provides deterministic, auditable explanations for heat transfer
    performance, leakage analysis, cold-end protection, fouling assessment,
    and optimization decisions using local linear interpretable models.
    """

    def __init__(self, config: ExplainerConfig):
        """
        Initialize the LIME explainer.

        Args:
            config: Explainer configuration parameters
        """
        self.config = config
        self.feature_info = AIR_PREHEATER_FEATURE_INFO
        self._rng = np.random.RandomState(config.random_seed)

    def explain_heat_transfer_performance(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "effectiveness",
    ) -> LIMEExplanation:
        """
        Explain heat transfer performance analysis.

        Analyzes factors driving current effectiveness level including
        temperature profiles, flow rates, and heat transfer coefficients.

        Args:
            features: Input features for heat transfer analysis
            prediction: Current effectiveness value
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

        # Build feature explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'heat_transfer'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_heat_transfer_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_heat_transfer_recommendations(
            features, feature_explanations
        )

        # Generate warnings
        warnings = self._check_heat_transfer_warnings(features)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="heat_transfer_performance",
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

    def explain_leakage_analysis(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "leakage_pct",
    ) -> LIMEExplanation:
        """
        Explain leakage detection and analysis results.

        Analyzes factors driving leakage estimation including O2 differential,
        seal condition, and pressure differentials.

        Args:
            features: Input features for leakage analysis
            prediction: Estimated leakage percentage
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with leakage factor contributions
        """
        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'leakage_analysis'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_leakage_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_leakage_recommendations(
            features, feature_explanations
        )

        # Generate warnings
        warnings = self._check_leakage_warnings(features)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="leakage_analysis",
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

    def explain_cold_end_protection(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "cold_end_temp_margin_c",
    ) -> LIMEExplanation:
        """
        Explain cold-end protection and acid dew point analysis.

        Analyzes factors affecting acid dew point calculation and
        cold-end corrosion risk including fuel sulfur, moisture, and
        gas outlet temperatures.

        Args:
            features: Cold-end related features
            prediction: Temperature margin above dew point
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with cold-end factor contributions
        """
        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'cold_end_protection'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_cold_end_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_cold_end_recommendations(
            features, feature_explanations
        )

        # Generate warnings
        warnings = self._check_cold_end_warnings(features)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="cold_end_protection",
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

    def explain_fouling_assessment(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "cleanliness_factor",
    ) -> LIMEExplanation:
        """
        Explain fouling assessment and cleanliness analysis.

        Analyzes factors driving fouling and cleanliness indicators
        including pressure drops, effectiveness degradation, and
        heat transfer coefficients.

        Args:
            features: Fouling-related features
            prediction: Cleanliness factor value
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with fouling factor contributions
        """
        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'fouling_assessment'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_fouling_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_fouling_recommendations(
            features, feature_explanations
        )

        # Generate warnings
        warnings = self._check_fouling_warnings(features)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="fouling_assessment",
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

    def explain_optimization_decision(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "optimal_exit_gas_temp_c",
    ) -> LIMEExplanation:
        """
        Explain optimization decision and setpoint recommendations.

        Analyzes factors driving recommended setpoints including
        efficiency targets, safety margins, and equipment constraints.

        Args:
            features: Optimization-related features
            prediction: Recommended optimal setpoint
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with optimization factor contributions
        """
        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'optimization'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_optimization_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            features, feature_explanations, prediction
        )

        # Generate warnings
        warnings = self._check_optimization_warnings(features)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="optimization_decision",
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

        Uses feature-specific ranges for realistic perturbations
        based on engineering knowledge of air preheater operation.
        """
        n_features = len(features)
        perturbations = np.zeros((self.config.num_samples, n_features))

        feature_names = list(features.keys())
        feature_values = np.array([features[k] for k in feature_names])

        for i, name in enumerate(feature_names):
            info = self.feature_info.get(name, {})
            typical_range = info.get('typical_range', (0, 100))

            # Calculate perturbation scale based on feature range
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

        # Generate synthetic predictions using linear surrogate
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
        """Build feature explanations from LIME weights."""
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

        if context == 'heat_transfer':
            if name == 'effectiveness':
                if value < 0.55:
                    return f"Low effectiveness of {value:.2f} indicates significant performance degradation"
                elif value > 0.75:
                    return f"Good effectiveness of {value:.2f} indicates healthy heat transfer"
                return f"Moderate effectiveness of {value:.2f}"
            elif name == 'gas_inlet_temp_c':
                return f"Gas inlet at {value:.0f}C provides heat source for recovery"
            elif name == 'air_outlet_temp_c':
                return f"Air outlet at {value:.0f}C reflects heat recovery achievement"
            elif name == 'lmtd_c':
                return f"LMTD of {value:.1f}C drives heat transfer rate"

        elif context == 'leakage_analysis':
            if name == 'o2_rise_pct':
                if value > 1.5:
                    return f"O2 rise of {value:.2f}% indicates significant leakage"
                return f"O2 rise of {value:.2f}% within acceptable range"
            elif name == 'leakage_pct':
                if value > 10:
                    return f"High leakage of {value:.1f}% causing efficiency loss"
                return f"Leakage of {value:.1f}% is acceptable"
            elif name == 'seal_gap_mm':
                return f"Seal gap of {value:.1f}mm affects air bypass"

        elif context == 'cold_end_protection':
            if name == 'cold_end_temp_margin_c':
                if value < 15:
                    return f"CRITICAL: Only {value:.1f}C margin above acid dew point"
                elif value < 25:
                    return f"LOW margin of {value:.1f}C above acid dew point"
                return f"Adequate margin of {value:.1f}C above acid dew point"
            elif name == 'acid_dew_point_c':
                return f"Acid dew point at {value:.0f}C from fuel sulfur content"
            elif name == 'fuel_sulfur_pct':
                return f"Fuel sulfur of {value:.2f}% raises acid dew point"
            elif name == 'gas_outlet_temp_c':
                return f"Gas outlet of {value:.0f}C determines cold-end temperature"

        elif context == 'fouling_assessment':
            if name == 'cleanliness_factor':
                if value < 0.75:
                    return f"Low cleanliness factor of {value:.2f} indicates heavy fouling"
                elif value > 0.90:
                    return f"Good cleanliness factor of {value:.2f} indicates clean surfaces"
                return f"Cleanliness factor of {value:.2f} suggests moderate fouling"
            elif name == 'dp_ratio':
                if value > 1.3:
                    return f"Elevated pressure drop ratio of {value:.2f} suggests fouling"
                return f"Normal pressure drop ratio of {value:.2f}"
            elif name == 'effectiveness_degradation_pct':
                return f"Effectiveness degraded by {value:.1f}% from design"

        elif context == 'optimization':
            if name == 'boiler_efficiency_gain_pct':
                return f"APH contributing {value:.2f}% efficiency gain"
            elif name == 'fuel_savings_pct':
                return f"Fuel savings of {value:.2f}% from heat recovery"
            elif name == 'boiler_load_pct':
                return f"Boiler at {value:.0f}% load affects optimal setpoints"

        return f"{base_desc}: {value:.2f} {unit}"

    # =========================================================================
    # SUMMARY GENERATORS
    # =========================================================================

    def _generate_heat_transfer_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for heat transfer performance explanation."""
        effectiveness = features.get('effectiveness', prediction)
        design_eff = features.get('design_effectiveness', 0.75)
        degradation = (design_eff - effectiveness) / design_eff * 100 if design_eff > 0 else 0

        gas_in = features.get('gas_inlet_temp_c', 0)
        gas_out = features.get('gas_outlet_temp_c', 0)
        air_in = features.get('air_inlet_temp_c', 0)
        air_out = features.get('air_outlet_temp_c', 0)

        if effectiveness >= 0.70:
            status = "Good heat transfer performance"
        elif effectiveness >= 0.55:
            status = "Moderate heat transfer performance"
        else:
            status = "Poor heat transfer performance"

        top_factors = [e.feature_name.replace('_', ' ') for e in explanations[:3]]

        return (
            f"{status} with effectiveness of {effectiveness:.2f} "
            f"({degradation:.1f}% below design). "
            f"Gas cooled from {gas_in:.0f}C to {gas_out:.0f}C, "
            f"air heated from {air_in:.0f}C to {air_out:.0f}C. "
            f"Key factors: {', '.join(top_factors)}."
        )

    def _generate_leakage_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for leakage analysis explanation."""
        leakage = features.get('leakage_pct', prediction)
        o2_rise = features.get('o2_rise_pct', 0)
        heat_loss = features.get('leakage_heat_loss_kw', 0)

        if leakage > 12:
            status = "HIGH leakage requiring attention"
        elif leakage > 8:
            status = "Elevated leakage"
        else:
            status = "Normal leakage levels"

        return (
            f"{status} with {leakage:.1f}% air-to-gas leakage "
            f"(O2 rise: {o2_rise:.2f}%). "
            f"Estimated heat loss from leakage: {heat_loss:.0f} kW. "
            f"Primary indicator: {explanations[0].feature_name.replace('_', ' ') if explanations else 'N/A'}."
        )

    def _generate_cold_end_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for cold-end protection explanation."""
        margin = features.get('cold_end_temp_margin_c', prediction)
        acid_dp = features.get('acid_dew_point_c', 130)
        gas_out = features.get('gas_outlet_temp_c', 150)
        sulfur = features.get('fuel_sulfur_pct', 0)

        if margin < 15:
            status = "CRITICAL: Operating near acid dew point"
        elif margin < 25:
            status = "WARNING: Low margin above acid dew point"
        elif margin > 50:
            status = "Excessive margin (opportunity for more recovery)"
        else:
            status = "Adequate cold-end protection"

        return (
            f"{status}. Gas outlet at {gas_out:.0f}C provides {margin:.1f}C margin "
            f"above acid dew point of {acid_dp:.0f}C. "
            f"Fuel sulfur content: {sulfur:.2f}%."
        )

    def _generate_fouling_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for fouling assessment explanation."""
        cleanliness = features.get('cleanliness_factor', prediction)
        dp_ratio = features.get('dp_ratio', 1.0)
        degradation = features.get('effectiveness_degradation_pct', 0)

        if cleanliness < 0.70:
            status = "HEAVY fouling"
            action = "Priority cleaning required"
        elif cleanliness < 0.80:
            status = "MODERATE fouling"
            action = "Schedule cleaning"
        elif cleanliness < 0.90:
            status = "LIGHT fouling"
            action = "Monitor and plan cleaning"
        else:
            status = "CLEAN"
            action = "Normal operation"

        return (
            f"{status} detected with cleanliness factor of {cleanliness:.2f}. "
            f"Pressure drop ratio: {dp_ratio:.2f}, effectiveness degradation: {degradation:.1f}%. "
            f"Recommended action: {action}."
        )

    def _generate_optimization_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for optimization decision explanation."""
        optimal_exit = prediction
        current_exit = features.get('gas_outlet_temp_c', 0)
        acid_dp = features.get('acid_dew_point_c', 130)
        eff_gain = features.get('boiler_efficiency_gain_pct', 0)

        temp_diff = current_exit - optimal_exit
        if abs(temp_diff) < 5:
            status = "Currently operating near optimal"
        elif temp_diff > 0:
            status = f"Can reduce exit temp by {temp_diff:.0f}C for more recovery"
        else:
            status = f"Should increase exit temp by {abs(temp_diff):.0f}C for protection"

        return (
            f"Optimal exit gas temperature: {optimal_exit:.0f}C (current: {current_exit:.0f}C). "
            f"{status}. "
            f"Acid dew point: {acid_dp:.0f}C. Efficiency gain: {eff_gain:.2f}%. "
            f"Key driver: {explanations[0].feature_name.replace('_', ' ') if explanations else 'N/A'}."
        )

    # =========================================================================
    # RECOMMENDATION GENERATORS
    # =========================================================================

    def _generate_heat_transfer_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate recommendations for heat transfer performance."""
        recommendations = []

        effectiveness = features.get('effectiveness', 0.70)
        design_eff = features.get('design_effectiveness', 0.75)
        degradation = (design_eff - effectiveness) / design_eff * 100 if design_eff > 0 else 0

        if degradation > 20:
            recommendations.append(
                f"Significant performance degradation ({degradation:.1f}%) - inspect for fouling or damage"
            )
        elif degradation > 10:
            recommendations.append(
                f"Moderate degradation ({degradation:.1f}%) - schedule preventive cleaning"
            )

        lmtd = features.get('lmtd_c', 50)
        if lmtd < 30:
            recommendations.append(
                "Low LMTD indicates possible flow imbalance or fouling"
            )

        air_temp_rise = features.get('air_temp_rise_c', 0)
        if air_temp_rise < 100:
            recommendations.append(
                f"Low air temperature rise ({air_temp_rise:.0f}C) - check for bypass or leakage"
            )

        return recommendations

    def _generate_leakage_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate recommendations for leakage analysis."""
        recommendations = []

        leakage = features.get('leakage_pct', 5)
        o2_rise = features.get('o2_rise_pct', 0.5)

        if leakage > 12:
            recommendations.append(
                f"High leakage ({leakage:.1f}%) - prioritize seal inspection and replacement"
            )
        elif leakage > 8:
            recommendations.append(
                f"Elevated leakage ({leakage:.1f}%) - schedule seal inspection"
            )

        if o2_rise > 1.5:
            recommendations.append(
                f"O2 rise of {o2_rise:.2f}% confirms significant air bypass"
            )

        seal_gap = features.get('seal_gap_mm', 5)
        if seal_gap > 8:
            recommendations.append(
                f"Seal gap ({seal_gap:.1f}mm) exceeds typical limits - adjust or replace seals"
            )

        heat_loss = features.get('leakage_heat_loss_kw', 0)
        if heat_loss > 300:
            recommendations.append(
                f"Leakage causing {heat_loss:.0f}kW heat loss - economic case for repair"
            )

        return recommendations

    def _generate_cold_end_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate recommendations for cold-end protection."""
        recommendations = []

        margin = features.get('cold_end_temp_margin_c', 30)
        gas_out = features.get('gas_outlet_temp_c', 150)
        acid_dp = features.get('acid_dew_point_c', 130)

        if margin < 15:
            recommendations.append(
                f"CRITICAL: Increase exit gas temp above {acid_dp + 20:.0f}C to prevent corrosion"
            )
        elif margin < 25:
            recommendations.append(
                f"Increase exit gas temp to maintain minimum 25C margin above dew point"
            )
        elif margin > 50:
            recommendations.append(
                f"Margin of {margin:.0f}C is conservative - consider lowering exit temp to {acid_dp + 30:.0f}C for more recovery"
            )

        sulfur = features.get('fuel_sulfur_pct', 0)
        if sulfur > 2:
            recommendations.append(
                f"High sulfur fuel ({sulfur:.2f}%) requires extra cold-end margin"
            )

        corrosion_rate = features.get('corrosion_rate_mm_yr', 0)
        if corrosion_rate > 0.2:
            recommendations.append(
                f"Elevated corrosion rate ({corrosion_rate:.2f}mm/yr) - inspect cold-end elements"
            )

        return recommendations

    def _generate_fouling_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate recommendations for fouling assessment."""
        recommendations = []

        cleanliness = features.get('cleanliness_factor', 0.85)
        dp_ratio = features.get('dp_ratio', 1.0)

        if cleanliness < 0.70:
            recommendations.append(
                "Priority cleaning required - heavy fouling detected"
            )
        elif cleanliness < 0.80:
            recommendations.append(
                "Schedule soot blowing or water washing at next opportunity"
            )
        elif cleanliness < 0.90:
            recommendations.append(
                "Increase soot blowing frequency to prevent further degradation"
            )

        if dp_ratio > 1.4:
            recommendations.append(
                f"High pressure drop (ratio: {dp_ratio:.2f}) indicates flow restriction"
            )

        gas_side_dp = features.get('gas_side_dp_mbar', 15)
        if gas_side_dp > 25:
            recommendations.append(
                f"High gas-side DP ({gas_side_dp:.0f}mbar) affecting ID fan power"
            )

        return recommendations

    def _generate_optimization_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        optimal_temp: float,
    ) -> List[str]:
        """Generate recommendations for optimization decisions."""
        recommendations = []

        current_exit = features.get('gas_outlet_temp_c', 150)
        temp_diff = current_exit - optimal_temp

        if temp_diff > 10:
            potential_gain = temp_diff * 0.05  # Approx 0.05% per degree
            recommendations.append(
                f"Lower exit gas temp by {temp_diff:.0f}C for ~{potential_gain:.1f}% efficiency gain"
            )
        elif temp_diff < -5:
            recommendations.append(
                f"Raise exit gas temp by {abs(temp_diff):.0f}C to protect against corrosion"
            )

        eff_gain = features.get('boiler_efficiency_gain_pct', 0)
        if eff_gain < 2.0:
            recommendations.append(
                "Low efficiency gain from APH - check for fouling or leakage issues"
            )

        load = features.get('boiler_load_pct', 100)
        if load < 50:
            recommendations.append(
                f"At {load:.0f}% load, consider wider temperature margins for turndown"
            )

        return recommendations

    # =========================================================================
    # WARNING CHECKERS
    # =========================================================================

    def _check_heat_transfer_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for heat transfer performance warnings."""
        warnings = []

        effectiveness = features.get('effectiveness', 0.70)
        if effectiveness < 0.50:
            warnings.append(f"ALARM: Very low effectiveness ({effectiveness:.2f}) - immediate investigation required")
        elif effectiveness < 0.55:
            warnings.append(f"WARNING: Low effectiveness ({effectiveness:.2f})")

        gas_outlet = features.get('gas_outlet_temp_c', 150)
        if gas_outlet > 200:
            warnings.append(f"WARNING: High exit gas temperature ({gas_outlet:.0f}C) - heat recovery opportunity lost")

        return warnings

    def _check_leakage_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for leakage-related warnings."""
        warnings = []

        leakage = features.get('leakage_pct', 5)
        if leakage > 15:
            warnings.append(f"ALARM: Excessive leakage ({leakage:.1f}%) - seal failure likely")
        elif leakage > 10:
            warnings.append(f"WARNING: High leakage ({leakage:.1f}%)")

        o2_rise = features.get('o2_rise_pct', 0.5)
        if o2_rise > 2.5:
            warnings.append(f"ALARM: O2 rise of {o2_rise:.2f}% exceeds limits")
        elif o2_rise > 1.5:
            warnings.append(f"WARNING: Elevated O2 rise ({o2_rise:.2f}%)")

        return warnings

    def _check_cold_end_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for cold-end protection warnings."""
        warnings = []

        margin = features.get('cold_end_temp_margin_c', 30)
        if margin < 10:
            warnings.append(f"ALARM: Operating below acid dew point margin ({margin:.1f}C)")
        elif margin < 20:
            warnings.append(f"WARNING: Low cold-end margin ({margin:.1f}C)")

        metal_temp = features.get('metal_temp_cold_end_c', 120)
        acid_dp = features.get('acid_dew_point_c', 130)
        if metal_temp < acid_dp:
            warnings.append(f"CRITICAL: Metal temp ({metal_temp:.0f}C) below acid dew point ({acid_dp:.0f}C)")

        corrosion_rate = features.get('corrosion_rate_mm_yr', 0)
        if corrosion_rate > 0.3:
            warnings.append(f"ALARM: High corrosion rate ({corrosion_rate:.2f}mm/yr)")

        return warnings

    def _check_fouling_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for fouling-related warnings."""
        warnings = []

        cleanliness = features.get('cleanliness_factor', 0.85)
        if cleanliness < 0.65:
            warnings.append(f"ALARM: Severe fouling (cleanliness: {cleanliness:.2f})")
        elif cleanliness < 0.75:
            warnings.append(f"WARNING: Moderate fouling (cleanliness: {cleanliness:.2f})")

        dp_ratio = features.get('dp_ratio', 1.0)
        if dp_ratio > 1.5:
            warnings.append(f"ALARM: Pressure drop ratio ({dp_ratio:.2f}) indicates blockage")
        elif dp_ratio > 1.3:
            warnings.append(f"WARNING: Elevated pressure drop ratio ({dp_ratio:.2f})")

        return warnings

    def _check_optimization_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for optimization-related warnings."""
        warnings = []

        # Combine relevant warnings from other checks
        warnings.extend(self._check_cold_end_warnings(features))
        warnings.extend(self._check_heat_transfer_warnings(features))

        # Add optimization-specific warnings
        eff_gain = features.get('boiler_efficiency_gain_pct', 2.0)
        if eff_gain < 1.0:
            warnings.append(f"WARNING: Low efficiency gain ({eff_gain:.2f}%) from APH")

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

def create_explainer(config: Optional[ExplainerConfig] = None) -> LIMEAirPreheaterExplainer:
    """
    Factory function to create a LIME explainer for air preheater.

    Args:
        config: Optional explainer configuration. Uses defaults if not provided.

    Returns:
        Configured LIMEAirPreheaterExplainer instance

    Example:
        >>> explainer = create_explainer()
        >>> explanation = explainer.explain_heat_transfer_performance(
        ...     features={'effectiveness': 0.72, 'gas_inlet_temp_c': 350.0},
        ...     prediction=0.72,
        ... )
    """
    if config is None:
        config = ExplainerConfig()
    return LIMEAirPreheaterExplainer(config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "FeatureExplanation",
    "LIMEExplanation",
    "ExplainerConfig",
    # Feature info
    "AIR_PREHEATER_FEATURE_INFO",
    # Explainer class
    "LIMEAirPreheaterExplainer",
    # Factory function
    "create_explainer",
]
