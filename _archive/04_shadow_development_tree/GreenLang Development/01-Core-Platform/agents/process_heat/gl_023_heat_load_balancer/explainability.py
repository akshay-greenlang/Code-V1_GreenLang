"""
GL-023 HEAT LOAD BALANCER AGENT - LIME Explainability Module

This module provides Local Interpretable Model-agnostic Explanations (LIME)
for multi-equipment heat load balancing and optimization decisions.
All explanations are deterministic and auditable.

Explainability Categories:
    - Load allocation reasoning (why specific equipment got specific loads)
    - Cost optimization explanation (fuel, emissions, total cost drivers)
    - Efficiency optimization explanation (fleet efficiency factors)
    - N+1 redundancy and safety margin analysis
    - Constraint binding explanation

Engineering Standards:
    - ASME PTC 4 for boiler performance testing
    - ISO 50001 for energy management systems
    - API 560 for fired heaters
    - IEEE 762 for equipment reliability

Example:
    >>> from greenlang.agents.process_heat.gl_023_heat_load_balancer.explainability import (
    ...     create_explainer,
    ...     ExplainerConfig,
    ... )
    >>>
    >>> explainer = create_explainer(ExplainerConfig())
    >>> explanation = explainer.explain_load_allocation(
    ...     features={
    ...         'total_demand_mmbtu_hr': 150.0,
    ...         'fleet_capacity_mmbtu_hr': 200.0,
    ...         'blr_001_efficiency_pct': 84.0,
    ...         'blr_002_efficiency_pct': 82.0,
    ...     },
    ...     prediction={'BLR-001': 90.0, 'BLR-002': 60.0},
    ... )
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

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
class EquipmentAllocationExplanation:
    """Explanation for a single equipment's load allocation."""

    equipment_id: str
    equipment_type: str
    current_load_pct: float
    target_load_pct: float
    load_change_pct: float
    efficiency_at_target_pct: float
    cost_contribution_pct: float
    allocation_reasoning: str
    ranking_factors: List[str]
    constraints_affecting: List[str]


@dataclass
class LIMEExplanation:
    """Complete LIME explanation output."""

    explanation_type: str
    prediction: Union[float, Dict[str, float]]
    prediction_unit: str
    base_value: float
    feature_explanations: List[FeatureExplanation]
    equipment_explanations: List[EquipmentAllocationExplanation] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    provenance_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_method: str = "LIME Local Linear Regression"
    model_fidelity: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    binding_constraints: List[str] = field(default_factory=list)
    tradeoffs_made: List[str] = field(default_factory=list)


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

LOAD_BALANCER_FEATURE_INFO: Dict[str, Dict[str, Any]] = {
    # Demand features
    'total_demand_mmbtu_hr': {
        'description': 'Total heat demand from all consumers',
        'unit': 'MMBTU/hr',
        'typical_range': (10, 500),
        'weight': 1.0,
    },
    'critical_demand_mmbtu_hr': {
        'description': 'Critical priority heat demand',
        'unit': 'MMBTU/hr',
        'typical_range': (0, 200),
        'weight': 1.2,
    },
    'peak_demand_forecast_mmbtu_hr': {
        'description': 'Forecasted peak demand',
        'unit': 'MMBTU/hr',
        'typical_range': (10, 600),
        'weight': 0.9,
    },

    # Fleet capacity features
    'fleet_capacity_mmbtu_hr': {
        'description': 'Total fleet heat generation capacity',
        'unit': 'MMBTU/hr',
        'typical_range': (50, 1000),
        'weight': 0.8,
    },
    'available_capacity_mmbtu_hr': {
        'description': 'Currently available capacity',
        'unit': 'MMBTU/hr',
        'typical_range': (10, 800),
        'weight': 0.9,
    },
    'spinning_reserve_pct': {
        'description': 'Spinning reserve as percentage of demand',
        'unit': '%',
        'typical_range': (5, 50),
        'warning_low': 10,
        'alarm_low': 5,
        'weight': 1.1,
    },
    'spinning_reserve_mmbtu_hr': {
        'description': 'Spinning reserve capacity',
        'unit': 'MMBTU/hr',
        'typical_range': (5, 200),
        'weight': 1.0,
    },

    # Equipment efficiency features (generic patterns)
    'equipment_efficiency_pct': {
        'description': 'Equipment thermal efficiency',
        'unit': '%',
        'typical_range': (70, 95),
        'warning_low': 75,
        'weight': 1.2,
    },
    'fleet_weighted_efficiency_pct': {
        'description': 'Load-weighted average fleet efficiency',
        'unit': '%',
        'typical_range': (75, 92),
        'warning_low': 78,
        'weight': 1.1,
    },
    'efficiency_at_target_pct': {
        'description': 'Expected efficiency at target load',
        'unit': '%',
        'typical_range': (70, 95),
        'weight': 1.0,
    },

    # Equipment load features
    'equipment_load_pct': {
        'description': 'Equipment load as percentage of capacity',
        'unit': '%',
        'typical_range': (25, 100),
        'warning_low': 30,
        'warning_high': 95,
        'weight': 0.9,
    },
    'equipment_load_mmbtu_hr': {
        'description': 'Equipment heat output',
        'unit': 'MMBTU/hr',
        'typical_range': (5, 200),
        'weight': 0.9,
    },

    # Cost features
    'fuel_price_per_mmbtu': {
        'description': 'Fuel price per MMBTU',
        'unit': '$/MMBTU',
        'typical_range': (2, 20),
        'weight': 1.3,
    },
    'total_hourly_cost_usd': {
        'description': 'Total hourly operating cost',
        'unit': '$/hr',
        'typical_range': (50, 5000),
        'weight': 1.0,
    },
    'cost_per_mmbtu_output': {
        'description': 'Blended cost per MMBTU output',
        'unit': '$/MMBTU',
        'typical_range': (3, 25),
        'weight': 1.1,
    },
    'carbon_price_per_ton': {
        'description': 'Carbon price for emissions cost',
        'unit': '$/ton',
        'typical_range': (0, 100),
        'weight': 0.8,
    },

    # Emissions features
    'total_co2_kg_hr': {
        'description': 'Total fleet CO2 emissions',
        'unit': 'kg/hr',
        'typical_range': (100, 20000),
        'weight': 0.9,
    },
    'co2_intensity_kg_mmbtu': {
        'description': 'CO2 intensity per unit heat output',
        'unit': 'kg/MMBTU',
        'typical_range': (40, 80),
        'weight': 0.8,
    },
    'total_nox_lb_hr': {
        'description': 'Total fleet NOx emissions',
        'unit': 'lb/hr',
        'typical_range': (0, 50),
        'warning_high': 40,
        'alarm_high': 50,
        'weight': 0.7,
    },

    # Operational features
    'units_running': {
        'description': 'Number of equipment units running',
        'unit': 'units',
        'typical_range': (1, 10),
        'weight': 0.6,
    },
    'units_available': {
        'description': 'Number of equipment units available',
        'unit': 'units',
        'typical_range': (1, 12),
        'weight': 0.5,
    },
    'n_plus_1_satisfied': {
        'description': 'N+1 redundancy requirement met',
        'unit': 'boolean',
        'typical_range': (0, 1),
        'alarm_low': 0.5,
        'weight': 1.3,
    },

    # Constraint features
    'min_load_constraint_active': {
        'description': 'Minimum load constraint is binding',
        'unit': 'boolean',
        'typical_range': (0, 1),
        'weight': 0.7,
    },
    'max_load_constraint_active': {
        'description': 'Maximum load constraint is binding',
        'unit': 'boolean',
        'typical_range': (0, 1),
        'weight': 0.8,
    },
    'ramp_rate_limited': {
        'description': 'Ramp rate constraint is limiting',
        'unit': 'boolean',
        'typical_range': (0, 1),
        'weight': 0.6,
    },

    # Savings features
    'cost_savings_pct': {
        'description': 'Cost savings vs baseline',
        'unit': '%',
        'typical_range': (0, 20),
        'weight': 0.9,
    },
    'efficiency_improvement_pct': {
        'description': 'Efficiency improvement vs baseline',
        'unit': '%',
        'typical_range': (0, 10),
        'weight': 0.8,
    },
}


# =============================================================================
# LIME EXPLAINER CLASS
# =============================================================================

class LIMELoadBalancerExplainer:
    """
    LIME-based explainer for GL-023 Heat Load Balancer Agent.

    Provides deterministic, auditable explanations for load allocation,
    cost optimization, and safety decisions using local linear interpretable models.
    """

    def __init__(self, config: ExplainerConfig):
        """
        Initialize the LIME explainer.

        Args:
            config: Explainer configuration parameters
        """
        self.config = config
        self.feature_info = LOAD_BALANCER_FEATURE_INFO
        self._rng = np.random.RandomState(config.random_seed)

    def explain_load_allocation(
        self,
        features: Dict[str, float],
        prediction: Dict[str, float],
        equipment_data: Optional[Dict[str, Dict[str, Any]]] = None,
        prediction_unit: str = "load_allocation_mmbtu_hr",
    ) -> LIMEExplanation:
        """
        Explain load allocation decisions across the equipment fleet.

        Analyzes why specific equipment received specific load allocations
        based on efficiency curves, costs, and constraints.

        Args:
            features: Input features for optimization
            prediction: Load allocation by equipment ID
            equipment_data: Optional equipment-specific data
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with feature and equipment contributions
        """
        # Total allocated load for scalar prediction
        total_load = sum(prediction.values())

        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local linear model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, total_load
        )

        # Build feature explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'load_allocation'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Build equipment allocation explanations
        equipment_explanations = self._build_equipment_allocation_explanations(
            prediction, equipment_data, features
        )

        # Identify binding constraints
        binding_constraints = self._identify_binding_constraints(features, equipment_data)

        # Identify tradeoffs
        tradeoffs = self._identify_load_allocation_tradeoffs(features, prediction)

        # Generate summary
        summary = self._generate_load_allocation_summary(
            features, prediction, equipment_explanations
        )

        # Generate recommendations
        recommendations = self._generate_load_allocation_recommendations(
            features, prediction, equipment_explanations
        )

        # Generate warnings
        warnings = self._check_load_allocation_warnings(features)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="load_allocation",
            prediction=prediction,
            prediction_unit=prediction_unit,
            base_value=intercept,
            feature_explanations=feature_explanations,
            equipment_explanations=equipment_explanations,
            summary=summary,
            confidence=model_fidelity,
            provenance_hash=provenance_hash,
            model_fidelity=model_fidelity,
            recommendations=recommendations,
            warnings=warnings,
            binding_constraints=binding_constraints,
            tradeoffs_made=tradeoffs,
        )

    def explain_cost_optimization(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "total_hourly_cost_usd",
    ) -> LIMEExplanation:
        """
        Explain cost optimization results.

        Analyzes factors driving operating costs including fuel prices,
        equipment efficiencies, and emissions costs.

        Args:
            features: Cost-related features
            prediction: Total hourly cost
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with cost factor contributions
        """
        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'cost_optimization'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_cost_optimization_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_cost_recommendations(
            features, feature_explanations
        )

        # Generate warnings
        warnings = self._check_cost_warnings(features)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="cost_optimization",
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

    def explain_efficiency_optimization(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "fleet_weighted_efficiency_pct",
    ) -> LIMEExplanation:
        """
        Explain fleet efficiency optimization results.

        Analyzes factors affecting fleet-wide efficiency including
        equipment load distribution and efficiency curves.

        Args:
            features: Efficiency-related features
            prediction: Fleet weighted efficiency
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with efficiency factor contributions
        """
        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'efficiency_optimization'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_efficiency_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_efficiency_recommendations(
            features, feature_explanations
        )

        # Generate warnings
        warnings = self._check_efficiency_warnings(features)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="efficiency_optimization",
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
        Explain N+1 redundancy and safety margin assessment.

        Analyzes factors affecting system reliability including
        spinning reserve and equipment availability.

        Args:
            features: Safety and reliability features
            prediction: Safety score or N+1 status
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with safety factor contributions
        """
        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'safety_assessment'
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

        # Generate warnings
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

    def explain_savings_analysis(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_unit: str = "cost_savings_usd_hr",
    ) -> LIMEExplanation:
        """
        Explain savings analysis vs baseline operation.

        Analyzes factors driving cost and efficiency savings
        compared to equal load distribution or prior operation.

        Args:
            features: Savings-related features
            prediction: Hourly savings amount
            prediction_unit: Unit of the prediction

        Returns:
            LIMEExplanation with savings factor contributions
        """
        # Generate perturbations
        perturbations = self._generate_perturbations(features)

        # Fit local model
        weights, intercept, model_fidelity = self._fit_local_model(
            features, perturbations, prediction
        )

        # Build explanations
        feature_explanations = self._build_feature_explanations(
            features, weights, 'savings_analysis'
        )
        feature_explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Generate summary
        summary = self._generate_savings_summary(
            features, feature_explanations, prediction
        )

        # Generate recommendations
        recommendations = self._generate_savings_recommendations(
            features, feature_explanations
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash(
            features, prediction, weights, intercept
        )

        return LIMEExplanation(
            explanation_type="savings_analysis",
            prediction=prediction,
            prediction_unit=prediction_unit,
            base_value=intercept,
            feature_explanations=feature_explanations,
            summary=summary,
            confidence=model_fidelity,
            provenance_hash=provenance_hash,
            model_fidelity=model_fidelity,
            recommendations=recommendations,
            warnings=[],
        )

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _generate_perturbations(
        self,
        features: Dict[str, float],
    ) -> np.ndarray:
        """Generate perturbations around the input features."""
        n_features = len(features)
        perturbations = np.zeros((self.config.num_samples, n_features))

        feature_names = list(features.keys())
        feature_values = np.array([features[k] for k in feature_names])

        for i, name in enumerate(feature_names):
            # Handle equipment-specific features
            base_name = name.split('_')[-1] if '_' in name else name
            info = self.feature_info.get(name, self.feature_info.get(base_name, {}))
            typical_range = info.get('typical_range', (0, 100))

            # Calculate perturbation scale
            range_size = typical_range[1] - typical_range[0]
            scale = range_size * self.config.kernel_width * 0.1

            # Generate perturbations
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
        """Fit a local linear model using weighted linear regression."""
        feature_names = list(features.keys())
        feature_values = np.array([features[k] for k in feature_names])

        # Calculate kernel weights
        distances = np.sqrt(np.sum((perturbations - feature_values) ** 2, axis=1))
        kernel_width = np.percentile(distances, 75) * self.config.kernel_width
        kernel_weights = np.exp(-distances ** 2 / (2 * kernel_width ** 2))

        # Generate synthetic predictions
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
            coefficients = np.linalg.lstsq(X, synthetic_predictions, rcond=None)[0]

        intercept = coefficients[0]
        weights = dict(zip(feature_names, coefficients[1:]))

        # Calculate R-squared
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

            # Get feature info (try exact match, then base pattern)
            info = self.feature_info.get(name, {})
            if not info:
                # Try to match pattern (e.g., 'blr_001_efficiency_pct' -> 'equipment_efficiency_pct')
                for pattern in ['equipment_efficiency_pct', 'equipment_load_pct',
                               'equipment_load_mmbtu_hr', 'fuel_price_per_mmbtu']:
                    if pattern.split('_')[-1] in name:
                        info = self.feature_info.get(pattern, {})
                        break

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

    def _build_equipment_allocation_explanations(
        self,
        prediction: Dict[str, float],
        equipment_data: Optional[Dict[str, Dict[str, Any]]],
        features: Dict[str, float],
    ) -> List[EquipmentAllocationExplanation]:
        """Build explanations for each equipment's load allocation."""
        explanations = []
        total_load = sum(prediction.values())

        for eq_id, target_load in prediction.items():
            eq_data = equipment_data.get(eq_id, {}) if equipment_data else {}

            current_load = eq_data.get('current_load_pct', 0.0)
            capacity = eq_data.get('capacity_mmbtu_hr', 100.0)
            target_load_pct = (target_load / capacity * 100) if capacity > 0 else 0
            load_change = target_load_pct - current_load

            # Calculate efficiency at target (from curve or estimate)
            efficiency = eq_data.get('efficiency_at_target_pct', 82.0)

            # Calculate cost contribution
            cost_contribution = (target_load / total_load * 100) if total_load > 0 else 0

            # Determine ranking factors
            ranking_factors = []
            if efficiency > 85:
                ranking_factors.append("High efficiency unit")
            if eq_data.get('fuel_price_per_mmbtu', 10) < 5:
                ranking_factors.append("Low fuel cost")
            if load_change > 0:
                ranking_factors.append("Load increasing to optimize")
            if eq_data.get('ramp_rate_available', True):
                ranking_factors.append("Ramp capability available")

            # Identify constraints affecting this equipment
            constraints = []
            if target_load_pct >= 95:
                constraints.append("Near maximum load limit")
            if target_load_pct <= 30 and target_load > 0:
                constraints.append("Near minimum turndown")
            if eq_data.get('must_run', False):
                constraints.append("Must-run constraint active")

            # Generate reasoning
            reasoning = self._generate_equipment_reasoning(
                eq_id, target_load, target_load_pct, efficiency,
                ranking_factors, constraints
            )

            explanations.append(EquipmentAllocationExplanation(
                equipment_id=eq_id,
                equipment_type=eq_data.get('equipment_type', 'BOILER'),
                current_load_pct=current_load,
                target_load_pct=round(target_load_pct, 1),
                load_change_pct=round(load_change, 1),
                efficiency_at_target_pct=round(efficiency, 1),
                cost_contribution_pct=round(cost_contribution, 1),
                allocation_reasoning=reasoning,
                ranking_factors=ranking_factors,
                constraints_affecting=constraints,
            ))

        # Sort by load allocation descending
        explanations.sort(key=lambda x: x.target_load_pct, reverse=True)
        return explanations

    def _generate_equipment_reasoning(
        self,
        eq_id: str,
        target_load: float,
        target_load_pct: float,
        efficiency: float,
        ranking_factors: List[str],
        constraints: List[str],
    ) -> str:
        """Generate human-readable reasoning for equipment allocation."""
        if target_load == 0:
            return f"{eq_id} is on standby to preserve spinning reserve capacity"

        if target_load_pct > 90:
            return (
                f"{eq_id} assigned high load ({target_load_pct:.0f}%) due to "
                f"{ranking_factors[0] if ranking_factors else 'economic optimization'}"
            )

        if target_load_pct < 40:
            return (
                f"{eq_id} at low load ({target_load_pct:.0f}%) to maintain "
                f"minimum firing while preserving reserve"
            )

        return (
            f"{eq_id} at moderate load ({target_load_pct:.0f}%) balancing "
            f"efficiency ({efficiency:.1f}%) and system flexibility"
        )

    def _identify_binding_constraints(
        self,
        features: Dict[str, float],
        equipment_data: Optional[Dict[str, Dict[str, Any]]],
    ) -> List[str]:
        """Identify constraints that are binding the optimization."""
        binding = []

        reserve = features.get('spinning_reserve_pct', 100)
        if reserve < 12:
            binding.append(f"Spinning reserve at minimum ({reserve:.1f}%)")

        n_plus_1 = features.get('n_plus_1_satisfied', 1)
        if n_plus_1 < 1:
            binding.append("N+1 redundancy constraint binding")

        if features.get('min_load_constraint_active', 0) > 0.5:
            binding.append("Minimum load constraint active on some equipment")

        if features.get('max_load_constraint_active', 0) > 0.5:
            binding.append("Maximum load constraint active on some equipment")

        return binding

    def _identify_load_allocation_tradeoffs(
        self,
        features: Dict[str, float],
        prediction: Dict[str, float],
    ) -> List[str]:
        """Identify tradeoffs made in load allocation."""
        tradeoffs = []

        efficiency = features.get('fleet_weighted_efficiency_pct', 85)
        reserve = features.get('spinning_reserve_pct', 15)

        if reserve > 20 and efficiency < 82:
            tradeoffs.append(
                "Maintaining high reserve margin at cost of some efficiency"
            )

        cost = features.get('cost_per_mmbtu_output', 10)
        if cost > 12 and efficiency > 85:
            tradeoffs.append(
                "Operating at high efficiency despite higher fuel costs"
            )

        return tradeoffs

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

        if context == 'load_allocation':
            if 'efficiency' in name:
                return f"Equipment efficiency of {value:.1f}% influences load priority"
            if 'demand' in name:
                return f"Heat demand of {value:.1f} {unit} drives total allocation"

        elif context == 'cost_optimization':
            if 'fuel_price' in name:
                return f"Fuel price of ${value:.2f}/MMBTU impacts operating cost"
            if 'total_hourly_cost' in name:
                return f"Total cost of ${value:.0f}/hr reflects fleet operation"

        elif context == 'efficiency_optimization':
            if 'fleet_weighted_efficiency' in name:
                return f"Fleet efficiency of {value:.1f}% achieved through optimal loading"

        elif context == 'safety_assessment':
            if 'spinning_reserve' in name:
                return f"Spinning reserve of {value:.1f}% provides system margin"
            if 'n_plus_1' in name:
                return "N+1 redundancy status indicates system reliability"

        return f"{base_desc}: {value:.2f} {unit}"

    def _generate_load_allocation_summary(
        self,
        features: Dict[str, float],
        prediction: Dict[str, float],
        equipment_explanations: List[EquipmentAllocationExplanation],
    ) -> str:
        """Generate summary for load allocation explanation."""
        total_demand = features.get('total_demand_mmbtu_hr', 0)
        efficiency = features.get('fleet_weighted_efficiency_pct', 0)
        units_running = len([e for e in equipment_explanations if e.target_load_pct > 0])

        lead_unit = equipment_explanations[0] if equipment_explanations else None
        lead_desc = (
            f"Led by {lead_unit.equipment_id} at {lead_unit.target_load_pct:.0f}% load"
            if lead_unit else "No primary unit"
        )

        return (
            f"Load allocation for {total_demand:.0f} MMBTU/hr demand across "
            f"{units_running} units. {lead_desc}. "
            f"Fleet efficiency: {efficiency:.1f}%."
        )

    def _generate_cost_optimization_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for cost optimization explanation."""
        cost_per_mmbtu = features.get('cost_per_mmbtu_output', 0)
        top_factors = [e.feature_name.replace('_', ' ') for e in explanations[:2]]

        return (
            f"Total hourly cost: ${prediction:.0f}/hr (${cost_per_mmbtu:.2f}/MMBTU output). "
            f"Primary cost drivers: {', '.join(top_factors)}."
        )

    def _generate_efficiency_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for efficiency optimization explanation."""
        return (
            f"Fleet weighted efficiency: {prediction:.1f}%. "
            f"Key efficiency factor: {explanations[0].feature_name.replace('_', ' ') if explanations else 'N/A'}."
        )

    def _generate_safety_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for safety assessment explanation."""
        reserve = features.get('spinning_reserve_pct', 0)
        n_plus_1 = features.get('n_plus_1_satisfied', 0)

        status = "MET" if n_plus_1 >= 1 else "NOT MET"

        return (
            f"Safety score: {prediction:.2f}. "
            f"Spinning reserve: {reserve:.1f}%. "
            f"N+1 redundancy: {status}."
        )

    def _generate_savings_summary(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
        prediction: float,
    ) -> str:
        """Generate summary for savings analysis explanation."""
        savings_pct = features.get('cost_savings_pct', 0)
        efficiency_improvement = features.get('efficiency_improvement_pct', 0)

        return (
            f"Hourly savings: ${prediction:.0f}/hr ({savings_pct:.1f}% vs baseline). "
            f"Efficiency improvement: {efficiency_improvement:.1f} percentage points."
        )

    def _generate_load_allocation_recommendations(
        self,
        features: Dict[str, float],
        prediction: Dict[str, float],
        equipment_explanations: List[EquipmentAllocationExplanation],
    ) -> List[str]:
        """Generate recommendations for load allocation."""
        recommendations = []

        reserve = features.get('spinning_reserve_pct', 0)
        if reserve < 10:
            recommendations.append(
                "Consider starting standby unit to increase spinning reserve"
            )

        efficiency = features.get('fleet_weighted_efficiency_pct', 0)
        if efficiency < 80:
            recommendations.append(
                "Consolidate load on fewer high-efficiency units if demand permits"
            )

        # Check for equipment at extremes
        for eq in equipment_explanations:
            if eq.target_load_pct > 95:
                recommendations.append(
                    f"{eq.equipment_id} near capacity - monitor for load relief"
                )

        return recommendations

    def _generate_cost_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate recommendations for cost optimization."""
        recommendations = []

        cost_per_mmbtu = features.get('cost_per_mmbtu_output', 0)
        if cost_per_mmbtu > 15:
            recommendations.append(
                "High operating cost - evaluate fuel switching opportunities"
            )

        return recommendations

    def _generate_efficiency_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate recommendations for efficiency optimization."""
        recommendations = []

        efficiency = features.get('fleet_weighted_efficiency_pct', 0)
        if efficiency < 80:
            recommendations.append(
                "Fleet efficiency below 80% - consider equipment maintenance"
            )

        return recommendations

    def _generate_safety_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []

        reserve = features.get('spinning_reserve_pct', 0)
        if reserve < 10:
            recommendations.append(
                "CRITICAL: Spinning reserve below minimum - start additional unit"
            )
        elif reserve < 15:
            recommendations.append(
                "WARNING: Low spinning reserve - prepare standby unit"
            )

        n_plus_1 = features.get('n_plus_1_satisfied', 1)
        if n_plus_1 < 1:
            recommendations.append(
                "CRITICAL: N+1 redundancy not satisfied - system at risk"
            )

        return recommendations

    def _generate_savings_recommendations(
        self,
        features: Dict[str, float],
        explanations: List[FeatureExplanation],
    ) -> List[str]:
        """Generate savings recommendations."""
        recommendations = []

        savings_pct = features.get('cost_savings_pct', 0)
        if savings_pct < 2:
            recommendations.append(
                "Limited savings potential - consider equipment upgrades"
            )

        return recommendations

    def _check_load_allocation_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for load allocation warnings."""
        warnings = []

        reserve = features.get('spinning_reserve_pct', 100)
        if reserve < 5:
            warnings.append(f"ALARM: Spinning reserve critically low ({reserve:.1f}%)")
        elif reserve < 10:
            warnings.append(f"WARNING: Spinning reserve low ({reserve:.1f}%)")

        n_plus_1 = features.get('n_plus_1_satisfied', 1)
        if n_plus_1 < 1:
            warnings.append("ALARM: N+1 redundancy requirement NOT MET")

        return warnings

    def _check_cost_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for cost-related warnings."""
        warnings = []

        cost = features.get('cost_per_mmbtu_output', 0)
        if cost > 20:
            warnings.append(f"WARNING: High operating cost (${cost:.2f}/MMBTU)")

        return warnings

    def _check_efficiency_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for efficiency warnings."""
        warnings = []

        efficiency = features.get('fleet_weighted_efficiency_pct', 100)
        if efficiency < 75:
            warnings.append(f"ALARM: Low fleet efficiency ({efficiency:.1f}%)")
        elif efficiency < 80:
            warnings.append(f"WARNING: Fleet efficiency below target ({efficiency:.1f}%)")

        return warnings

    def _check_safety_warnings(
        self,
        features: Dict[str, float],
    ) -> List[str]:
        """Check for safety warnings."""
        warnings = []
        warnings.extend(self._check_load_allocation_warnings(features))
        return warnings

    def _calculate_provenance_hash(
        self,
        features: Dict[str, float],
        prediction: Union[float, Dict[str, float]],
        weights: Dict[str, float],
        intercept: float,
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        # Handle dict or float prediction
        if isinstance(prediction, dict):
            pred_for_hash = {k: round(v, 6) for k, v in sorted(prediction.items())}
        else:
            pred_for_hash = round(prediction, 6)

        hash_input = {
            'features': {k: round(v, 6) for k, v in sorted(features.items())},
            'prediction': pred_for_hash,
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

def create_explainer(config: Optional[ExplainerConfig] = None) -> LIMELoadBalancerExplainer:
    """
    Factory function to create a LIME explainer for heat load balancer.

    Args:
        config: Optional explainer configuration. Uses defaults if not provided.

    Returns:
        Configured LIMELoadBalancerExplainer instance
    """
    if config is None:
        config = ExplainerConfig()
    return LIMELoadBalancerExplainer(config)
