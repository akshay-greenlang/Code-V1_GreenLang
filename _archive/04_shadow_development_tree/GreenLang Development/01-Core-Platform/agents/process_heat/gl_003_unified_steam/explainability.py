"""
GL-003 UNIFIED STEAM SYSTEM OPTIMIZER - SHAP Explainability Module

This module provides SHAP (SHapley Additive exPlanations) based explainability
for steam system optimization decisions. It helps operators understand:
- Why specific recommendations were made
- Which factors most influenced the optimization
- How changing inputs would affect outcomes

Features:
    - SHAP value calculation for optimization features
    - Feature importance ranking
    - What-if analysis
    - Visualization support
    - Audit trail for regulatory compliance

Example:
    >>> from greenlang.agents.process_heat.gl_003_unified_steam.explainability import (
    ...     SHAPSteamAnalyzer,
    ... )
    >>>
    >>> analyzer = SHAPSteamAnalyzer()
    >>> explanation = analyzer.explain_optimization(input_data, output_data)
    >>> print(explanation.summary)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class FeatureCategory(str, Enum):
    """Categories of features for explainability."""
    HEADER_PRESSURE = "header_pressure"
    HEADER_FLOW = "header_flow"
    STEAM_QUALITY = "steam_quality"
    CONDENSATE = "condensate"
    PRV_OPERATION = "prv_operation"
    FLASH_RECOVERY = "flash_recovery"
    ECONOMIC = "economic"
    ENVIRONMENTAL = "environmental"


class ExplanationType(str, Enum):
    """Types of explanations."""
    FEATURE_IMPORTANCE = "feature_importance"
    CONTRIBUTION_ANALYSIS = "contribution_analysis"
    COUNTERFACTUAL = "counterfactual"
    SENSITIVITY = "sensitivity"


# =============================================================================
# DATA MODELS
# =============================================================================

class FeatureContribution(BaseModel):
    """SHAP-style feature contribution to outcome."""

    feature_name: str = Field(..., description="Name of the feature")
    feature_value: float = Field(..., description="Current value of the feature")
    shap_value: float = Field(..., description="SHAP contribution value")
    baseline_value: float = Field(..., description="Baseline/expected value")
    contribution_pct: float = Field(0.0, description="Percentage contribution")
    category: FeatureCategory = Field(..., description="Feature category")
    direction: str = Field("neutral", description="positive/negative/neutral")
    unit: str = Field("", description="Unit of measurement")
    description: str = Field("", description="Human-readable description")


class FeatureImportance(BaseModel):
    """Feature importance ranking."""

    feature_name: str = Field(..., description="Name of the feature")
    importance_score: float = Field(..., description="Importance score (0-1)")
    rank: int = Field(..., description="Importance rank")
    category: FeatureCategory = Field(..., description="Feature category")
    avg_impact: float = Field(0.0, description="Average impact on outcome")
    direction_bias: str = Field("neutral", description="Typical impact direction")


class SensitivityResult(BaseModel):
    """Result of sensitivity analysis."""

    feature_name: str = Field(..., description="Feature analyzed")
    baseline_value: float = Field(..., description="Baseline feature value")
    baseline_outcome: float = Field(..., description="Baseline outcome")
    test_values: List[float] = Field(default_factory=list)
    outcome_values: List[float] = Field(default_factory=list)
    sensitivity: float = Field(0.0, description="dOutcome/dFeature")
    elasticity: float = Field(0.0, description="% change outcome / % change feature")


class CounterfactualExplanation(BaseModel):
    """What-if counterfactual explanation."""

    scenario_name: str = Field(..., description="Name of scenario")
    changes: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Feature changes (from, to)"
    )
    original_outcome: float = Field(..., description="Original outcome value")
    counterfactual_outcome: float = Field(..., description="Counterfactual outcome")
    outcome_change_pct: float = Field(0.0, description="Percentage change in outcome")
    feasibility_score: float = Field(1.0, description="How feasible is this scenario (0-1)")
    recommendation: str = Field("", description="Actionable recommendation")


class OptimizationExplanation(BaseModel):
    """Complete explanation of optimization results."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    explanation_type: ExplanationType = Field(ExplanationType.CONTRIBUTION_ANALYSIS)

    # Main outcome
    outcome_name: str = Field("system_efficiency", description="What was optimized")
    outcome_value: float = Field(..., description="Optimization outcome value")
    baseline_outcome: float = Field(50.0, description="Baseline/expected outcome")

    # Feature contributions
    contributions: List[FeatureContribution] = Field(default_factory=list)
    feature_importance: List[FeatureImportance] = Field(default_factory=list)

    # Analysis results
    sensitivity_analysis: List[SensitivityResult] = Field(default_factory=list)
    counterfactuals: List[CounterfactualExplanation] = Field(default_factory=list)

    # Summary
    summary: str = Field("", description="Natural language summary")
    key_drivers: List[str] = Field(default_factory=list, description="Top 3 drivers")
    improvement_opportunities: List[str] = Field(default_factory=list)

    # Provenance
    provenance_hash: str = Field("", description="SHA-256 hash for audit")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# STEAM SYSTEM FEATURE EXTRACTORS
# =============================================================================

class SteamFeatureExtractor:
    """
    Extract features from steam system data for SHAP analysis.

    Converts raw steam system readings into normalized features
    suitable for explainability analysis.
    """

    def __init__(self) -> None:
        """Initialize feature extractor."""
        self.feature_definitions = self._define_features()
        logger.debug("SteamFeatureExtractor initialized")

    def _define_features(self) -> Dict[str, Dict[str, Any]]:
        """Define steam system features for analysis."""
        return {
            # Header pressure features
            "hp_pressure_deviation": {
                "category": FeatureCategory.HEADER_PRESSURE,
                "unit": "psi",
                "baseline": 0.0,
                "weight": 0.15,
                "description": "High-pressure header deviation from setpoint",
            },
            "mp_pressure_deviation": {
                "category": FeatureCategory.HEADER_PRESSURE,
                "unit": "psi",
                "baseline": 0.0,
                "weight": 0.12,
                "description": "Medium-pressure header deviation from setpoint",
            },
            "lp_pressure_deviation": {
                "category": FeatureCategory.HEADER_PRESSURE,
                "unit": "psi",
                "baseline": 0.0,
                "weight": 0.10,
                "description": "Low-pressure header deviation from setpoint",
            },

            # Flow balance features
            "header_imbalance_pct": {
                "category": FeatureCategory.HEADER_FLOW,
                "unit": "%",
                "baseline": 0.0,
                "weight": 0.12,
                "description": "Supply-demand imbalance percentage",
            },
            "total_steam_flow": {
                "category": FeatureCategory.HEADER_FLOW,
                "unit": "klb/hr",
                "baseline": 100.0,
                "weight": 0.08,
                "description": "Total steam production rate",
            },

            # Quality features
            "dryness_fraction": {
                "category": FeatureCategory.STEAM_QUALITY,
                "unit": "fraction",
                "baseline": 0.98,
                "weight": 0.10,
                "description": "Steam dryness fraction",
            },
            "tds_deviation": {
                "category": FeatureCategory.STEAM_QUALITY,
                "unit": "ppm",
                "baseline": 0.0,
                "weight": 0.08,
                "description": "TDS deviation from limit",
            },
            "conductivity": {
                "category": FeatureCategory.STEAM_QUALITY,
                "unit": "uS/cm",
                "baseline": 0.15,
                "weight": 0.05,
                "description": "Cation conductivity",
            },

            # Condensate features
            "condensate_return_rate": {
                "category": FeatureCategory.CONDENSATE,
                "unit": "%",
                "baseline": 85.0,
                "weight": 0.08,
                "description": "Condensate return percentage",
            },
            "condensate_temperature": {
                "category": FeatureCategory.CONDENSATE,
                "unit": "F",
                "baseline": 180.0,
                "weight": 0.05,
                "description": "Condensate return temperature",
            },

            # PRV features
            "prv_opening_deviation": {
                "category": FeatureCategory.PRV_OPERATION,
                "unit": "%",
                "baseline": 0.0,
                "weight": 0.07,
                "description": "PRV opening deviation from target range",
            },
        }

    def extract_features(
        self,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Extract features from input/output data.

        Args:
            input_data: Raw input data dictionary
            output_data: Optional output data dictionary

        Returns:
            Dictionary of feature name -> feature value
        """
        features = {}

        # Extract header pressure features
        if "header_readings" in input_data:
            for reading in input_data.get("header_readings", []):
                header_id = reading.get("header_id", "").lower()
                deviation = reading.get("current_pressure_psig", 0) - reading.get(
                    "pressure_setpoint_psig", 0
                )

                if "hp" in header_id:
                    features["hp_pressure_deviation"] = deviation
                elif "mp" in header_id:
                    features["mp_pressure_deviation"] = deviation
                elif "lp" in header_id:
                    features["lp_pressure_deviation"] = deviation

        # Extract flow features
        features["total_steam_flow"] = input_data.get("total_steam_flow_lb_hr", 0) / 1000

        # Extract quality features
        if "quality_readings" in input_data:
            readings = input_data.get("quality_readings", [])
            if readings:
                avg_dryness = sum(r.get("dryness_fraction", 0.98) for r in readings) / len(readings)
                features["dryness_fraction"] = avg_dryness

        # Extract condensate features
        if "condensate_readings" in input_data:
            readings = input_data.get("condensate_readings", [])
            if readings:
                total_return = sum(r.get("flow_rate_lb_hr", 0) for r in readings)
                total_steam = input_data.get("total_steam_flow_lb_hr", 1)
                features["condensate_return_rate"] = (total_return / total_steam * 100) if total_steam > 0 else 0

                avg_temp = sum(r.get("temperature_f", 180) for r in readings) / len(readings)
                features["condensate_temperature"] = avg_temp

        # Fill defaults for missing features
        for feature_name, definition in self.feature_definitions.items():
            if feature_name not in features:
                features[feature_name] = definition["baseline"]

        return features


# =============================================================================
# SHAP-STYLE CONTRIBUTION CALCULATOR
# =============================================================================

class SHAPContributionCalculator:
    """
    Calculate SHAP-style contributions for steam system features.

    Uses a simplified Shapley value approximation based on
    marginal contributions and feature weights.
    """

    def __init__(
        self,
        feature_definitions: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Initialize SHAP calculator.

        Args:
            feature_definitions: Feature definitions with weights
        """
        self.feature_defs = feature_definitions
        logger.debug("SHAPContributionCalculator initialized")

    def calculate_contributions(
        self,
        features: Dict[str, float],
        outcome_value: float,
        baseline_outcome: float = 85.0,
    ) -> List[FeatureContribution]:
        """
        Calculate SHAP-style contributions for each feature.

        Uses marginal contribution method:
        SHAP_i = weight_i * (value_i - baseline_i) / baseline_i * scale_factor

        Args:
            features: Feature values
            outcome_value: Actual outcome value
            baseline_outcome: Expected baseline outcome

        Returns:
            List of FeatureContribution objects
        """
        contributions = []
        total_deviation = outcome_value - baseline_outcome

        # Calculate raw contributions
        raw_contributions = {}
        for feature_name, value in features.items():
            if feature_name in self.feature_defs:
                definition = self.feature_defs[feature_name]
                baseline = definition["baseline"]
                weight = definition["weight"]

                # Calculate normalized deviation
                if baseline != 0:
                    normalized_dev = (value - baseline) / abs(baseline)
                else:
                    normalized_dev = value

                # Apply weight
                raw_contributions[feature_name] = weight * normalized_dev

        # Normalize to sum to total deviation
        total_raw = sum(abs(v) for v in raw_contributions.values())
        if total_raw > 0:
            scale_factor = abs(total_deviation) / total_raw
        else:
            scale_factor = 1.0

        # Create contribution objects
        for feature_name, raw_value in raw_contributions.items():
            definition = self.feature_defs[feature_name]
            shap_value = raw_value * scale_factor

            # Determine direction
            if shap_value > 0.1:
                direction = "positive"
            elif shap_value < -0.1:
                direction = "negative"
            else:
                direction = "neutral"

            # Calculate contribution percentage
            if total_deviation != 0:
                contribution_pct = (shap_value / abs(total_deviation)) * 100
            else:
                contribution_pct = 0.0

            contribution = FeatureContribution(
                feature_name=feature_name,
                feature_value=features.get(feature_name, 0),
                shap_value=shap_value,
                baseline_value=definition["baseline"],
                contribution_pct=contribution_pct,
                category=definition["category"],
                direction=direction,
                unit=definition["unit"],
                description=definition["description"],
            )
            contributions.append(contribution)

        # Sort by absolute SHAP value
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

        return contributions

    def calculate_feature_importance(
        self,
        contributions: List[FeatureContribution],
    ) -> List[FeatureImportance]:
        """
        Calculate feature importance from contributions.

        Args:
            contributions: List of feature contributions

        Returns:
            List of FeatureImportance objects
        """
        # Calculate importance as absolute SHAP value normalized
        total_shap = sum(abs(c.shap_value) for c in contributions)

        importance_list = []
        for rank, contribution in enumerate(contributions, 1):
            if total_shap > 0:
                importance_score = abs(contribution.shap_value) / total_shap
            else:
                importance_score = 0.0

            importance = FeatureImportance(
                feature_name=contribution.feature_name,
                importance_score=importance_score,
                rank=rank,
                category=contribution.category,
                avg_impact=contribution.shap_value,
                direction_bias=contribution.direction,
            )
            importance_list.append(importance)

        return importance_list


# =============================================================================
# SENSITIVITY ANALYZER
# =============================================================================

class SensitivityAnalyzer:
    """
    Analyze sensitivity of outcomes to feature changes.

    Performs what-if analysis to understand how changes in
    individual features affect optimization outcomes.
    """

    def __init__(
        self,
        feature_definitions: Dict[str, Dict[str, Any]],
    ) -> None:
        """Initialize sensitivity analyzer."""
        self.feature_defs = feature_definitions
        logger.debug("SensitivityAnalyzer initialized")

    def analyze_sensitivity(
        self,
        features: Dict[str, float],
        outcome_value: float,
        feature_to_analyze: str,
        perturbation_range: float = 0.2,
        num_points: int = 5,
    ) -> SensitivityResult:
        """
        Analyze sensitivity for a specific feature.

        Args:
            features: Current feature values
            outcome_value: Current outcome value
            feature_to_analyze: Feature to analyze
            perturbation_range: Range of perturbation (+/- %)
            num_points: Number of test points

        Returns:
            SensitivityResult object
        """
        if feature_to_analyze not in features:
            raise ValueError(f"Unknown feature: {feature_to_analyze}")

        baseline_value = features[feature_to_analyze]
        definition = self.feature_defs.get(feature_to_analyze, {})
        weight = definition.get("weight", 0.1)

        # Generate test values
        if baseline_value != 0:
            min_val = baseline_value * (1 - perturbation_range)
            max_val = baseline_value * (1 + perturbation_range)
        else:
            min_val = -perturbation_range
            max_val = perturbation_range

        step = (max_val - min_val) / (num_points - 1) if num_points > 1 else 0
        test_values = [min_val + i * step for i in range(num_points)]

        # Estimate outcome for each test value
        # Simplified linear model: dOutcome = weight * dFeature / baseline_feature
        outcome_values = []
        definition_baseline = definition.get("baseline", baseline_value)

        for test_val in test_values:
            if definition_baseline != 0:
                delta_feature = (test_val - baseline_value) / abs(definition_baseline)
            else:
                delta_feature = test_val - baseline_value

            delta_outcome = weight * delta_feature * 10  # Scale factor
            estimated_outcome = outcome_value + delta_outcome
            outcome_values.append(estimated_outcome)

        # Calculate sensitivity (slope)
        if len(test_values) >= 2:
            dx = test_values[-1] - test_values[0]
            dy = outcome_values[-1] - outcome_values[0]
            sensitivity = dy / dx if dx != 0 else 0
        else:
            sensitivity = 0

        # Calculate elasticity
        if baseline_value != 0 and outcome_value != 0:
            elasticity = sensitivity * baseline_value / outcome_value
        else:
            elasticity = 0

        return SensitivityResult(
            feature_name=feature_to_analyze,
            baseline_value=baseline_value,
            baseline_outcome=outcome_value,
            test_values=test_values,
            outcome_values=outcome_values,
            sensitivity=sensitivity,
            elasticity=elasticity,
        )


# =============================================================================
# COUNTERFACTUAL GENERATOR
# =============================================================================

class CounterfactualGenerator:
    """
    Generate counterfactual explanations for optimization.

    Creates "what-if" scenarios to explain how different
    operating conditions would affect optimization outcomes.
    """

    def __init__(
        self,
        feature_definitions: Dict[str, Dict[str, Any]],
    ) -> None:
        """Initialize counterfactual generator."""
        self.feature_defs = feature_definitions
        logger.debug("CounterfactualGenerator initialized")

    def generate_improvement_counterfactual(
        self,
        features: Dict[str, float],
        outcome_value: float,
        target_improvement_pct: float = 5.0,
    ) -> CounterfactualExplanation:
        """
        Generate counterfactual for target improvement.

        Args:
            features: Current feature values
            outcome_value: Current outcome
            target_improvement_pct: Target improvement percentage

        Returns:
            CounterfactualExplanation object
        """
        target_outcome = outcome_value * (1 + target_improvement_pct / 100)

        # Identify features with room for improvement
        changes = {}
        total_impact = 0

        for feature_name, value in features.items():
            if feature_name not in self.feature_defs:
                continue

            definition = self.feature_defs[feature_name]
            baseline = definition["baseline"]
            weight = definition["weight"]

            # Check if feature is suboptimal
            deviation = value - baseline
            if abs(deviation) > 0.01 * abs(baseline if baseline != 0 else 1):
                # Calculate improvement from moving toward baseline
                improvement_potential = weight * abs(deviation) / (abs(baseline) if baseline != 0 else 1)
                total_impact += improvement_potential

                # Suggest moving 50% toward baseline
                new_value = value - 0.5 * deviation
                changes[feature_name] = (value, new_value)

        # Estimate counterfactual outcome
        counterfactual_outcome = outcome_value + total_impact * 10

        # Generate recommendation
        if changes:
            top_change = max(changes.items(), key=lambda x: abs(x[1][0] - x[1][1]))
            recommendation = (
                f"Adjust {top_change[0]} from {top_change[1][0]:.2f} to "
                f"{top_change[1][1]:.2f} for primary improvement"
            )
        else:
            recommendation = "System is near optimal - no major adjustments recommended"

        return CounterfactualExplanation(
            scenario_name=f"Improvement Target: +{target_improvement_pct}%",
            changes=changes,
            original_outcome=outcome_value,
            counterfactual_outcome=counterfactual_outcome,
            outcome_change_pct=((counterfactual_outcome - outcome_value) / outcome_value * 100)
            if outcome_value != 0 else 0,
            feasibility_score=min(1.0, 0.9 - len(changes) * 0.1),
            recommendation=recommendation,
        )

    def generate_scenario_counterfactual(
        self,
        features: Dict[str, float],
        outcome_value: float,
        scenario: Dict[str, float],
        scenario_name: str,
    ) -> CounterfactualExplanation:
        """
        Generate counterfactual for a specific scenario.

        Args:
            features: Current feature values
            outcome_value: Current outcome
            scenario: Dictionary of feature changes
            scenario_name: Name for the scenario

        Returns:
            CounterfactualExplanation object
        """
        changes = {}
        total_impact = 0

        for feature_name, new_value in scenario.items():
            if feature_name not in features:
                continue

            old_value = features[feature_name]
            changes[feature_name] = (old_value, new_value)

            if feature_name in self.feature_defs:
                definition = self.feature_defs[feature_name]
                weight = definition["weight"]
                baseline = definition["baseline"]

                # Calculate impact of change
                if baseline != 0:
                    old_deviation = (old_value - baseline) / abs(baseline)
                    new_deviation = (new_value - baseline) / abs(baseline)
                    impact_change = weight * (new_deviation - old_deviation)
                else:
                    impact_change = weight * (new_value - old_value)

                total_impact += impact_change

        counterfactual_outcome = outcome_value + total_impact * 10

        return CounterfactualExplanation(
            scenario_name=scenario_name,
            changes=changes,
            original_outcome=outcome_value,
            counterfactual_outcome=counterfactual_outcome,
            outcome_change_pct=((counterfactual_outcome - outcome_value) / outcome_value * 100)
            if outcome_value != 0 else 0,
            feasibility_score=0.8,
            recommendation=f"Scenario '{scenario_name}' would change outcome by "
                          f"{total_impact * 10:.1f} points",
        )


# =============================================================================
# MAIN SHAP STEAM ANALYZER
# =============================================================================

class SHAPSteamAnalyzer:
    """
    SHAP-based explainability for steam system optimization.

    Provides comprehensive explanations of optimization decisions
    including feature importance, contributions, sensitivity analysis,
    and counterfactual explanations.

    Example:
        >>> analyzer = SHAPSteamAnalyzer()
        >>> explanation = analyzer.explain_optimization(
        ...     input_data={"header_readings": [...], "total_steam_flow_lb_hr": 100000},
        ...     outcome_value=92.5,
        ...     outcome_name="system_efficiency"
        ... )
        >>> print(explanation.summary)
        >>> for driver in explanation.key_drivers:
        ...     print(f"  - {driver}")
    """

    def __init__(self) -> None:
        """Initialize SHAP Steam Analyzer."""
        self.feature_extractor = SteamFeatureExtractor()
        self.contribution_calc = SHAPContributionCalculator(
            self.feature_extractor.feature_definitions
        )
        self.sensitivity_analyzer = SensitivityAnalyzer(
            self.feature_extractor.feature_definitions
        )
        self.counterfactual_gen = CounterfactualGenerator(
            self.feature_extractor.feature_definitions
        )

        logger.info("SHAPSteamAnalyzer initialized")

    def explain_optimization(
        self,
        input_data: Dict[str, Any],
        outcome_value: float,
        outcome_name: str = "system_efficiency",
        baseline_outcome: float = 85.0,
        include_sensitivity: bool = True,
        include_counterfactuals: bool = True,
        top_n_features: int = 5,
    ) -> OptimizationExplanation:
        """
        Generate comprehensive explanation of optimization results.

        Args:
            input_data: Input data dictionary
            outcome_value: Optimization outcome value
            outcome_name: Name of outcome being explained
            baseline_outcome: Baseline/expected outcome
            include_sensitivity: Include sensitivity analysis
            include_counterfactuals: Include counterfactual explanations
            top_n_features: Number of top features for sensitivity

        Returns:
            OptimizationExplanation object
        """
        # Extract features
        features = self.feature_extractor.extract_features(input_data)

        # Calculate contributions
        contributions = self.contribution_calc.calculate_contributions(
            features,
            outcome_value,
            baseline_outcome,
        )

        # Calculate feature importance
        importance = self.contribution_calc.calculate_feature_importance(contributions)

        # Sensitivity analysis for top features
        sensitivity_results = []
        if include_sensitivity:
            top_features = [c.feature_name for c in contributions[:top_n_features]]
            for feature_name in top_features:
                try:
                    result = self.sensitivity_analyzer.analyze_sensitivity(
                        features,
                        outcome_value,
                        feature_name,
                    )
                    sensitivity_results.append(result)
                except Exception as e:
                    logger.warning(f"Sensitivity analysis failed for {feature_name}: {e}")

        # Counterfactual explanations
        counterfactuals = []
        if include_counterfactuals:
            # Generate improvement counterfactual
            improvement = self.counterfactual_gen.generate_improvement_counterfactual(
                features,
                outcome_value,
            )
            counterfactuals.append(improvement)

            # Generate a specific scenario
            scenario = self.counterfactual_gen.generate_scenario_counterfactual(
                features,
                outcome_value,
                {"condensate_return_rate": 90.0, "dryness_fraction": 0.99},
                "Optimal Quality & Recovery",
            )
            counterfactuals.append(scenario)

        # Generate summary
        summary = self._generate_summary(
            contributions,
            outcome_value,
            baseline_outcome,
            outcome_name,
        )

        # Extract key drivers
        key_drivers = [
            f"{c.feature_name}: {c.shap_value:+.2f} ({c.direction})"
            for c in contributions[:3]
        ]

        # Identify improvement opportunities
        improvement_opportunities = self._identify_improvements(contributions)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data,
            outcome_value,
            contributions,
        )

        return OptimizationExplanation(
            explanation_type=ExplanationType.CONTRIBUTION_ANALYSIS,
            outcome_name=outcome_name,
            outcome_value=outcome_value,
            baseline_outcome=baseline_outcome,
            contributions=contributions,
            feature_importance=importance,
            sensitivity_analysis=sensitivity_results,
            counterfactuals=counterfactuals,
            summary=summary,
            key_drivers=key_drivers,
            improvement_opportunities=improvement_opportunities,
            provenance_hash=provenance_hash,
        )

    def explain_recommendation(
        self,
        recommendation: str,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """
        Explain why a specific recommendation was made.

        Args:
            recommendation: The recommendation text
            input_data: Input data that led to recommendation
            context: Additional context

        Returns:
            Natural language explanation
        """
        features = self.feature_extractor.extract_features(input_data)

        # Identify which features are out of spec
        deviations = []
        for feature_name, value in features.items():
            if feature_name in self.feature_extractor.feature_definitions:
                definition = self.feature_extractor.feature_definitions[feature_name]
                baseline = definition["baseline"]
                threshold = abs(baseline * 0.1) if baseline != 0 else 0.1

                if abs(value - baseline) > threshold:
                    deviations.append({
                        "feature": feature_name,
                        "value": value,
                        "expected": baseline,
                        "description": definition["description"],
                    })

        if deviations:
            deviation_text = "; ".join(
                f"{d['feature']} is {d['value']:.2f} (expected {d['expected']:.2f})"
                for d in deviations[:3]
            )
            explanation = (
                f"This recommendation is based on the following observations: "
                f"{deviation_text}. {recommendation}"
            )
        else:
            explanation = (
                f"This recommendation is based on system optimization analysis. "
                f"{recommendation}"
            )

        return explanation

    def _generate_summary(
        self,
        contributions: List[FeatureContribution],
        outcome_value: float,
        baseline_outcome: float,
        outcome_name: str,
    ) -> str:
        """Generate natural language summary."""
        deviation = outcome_value - baseline_outcome
        deviation_pct = (deviation / baseline_outcome * 100) if baseline_outcome != 0 else 0

        if deviation > 0:
            performance = "above"
        elif deviation < 0:
            performance = "below"
        else:
            performance = "at"

        # Top contributors
        positive_contributors = [c for c in contributions if c.direction == "positive"]
        negative_contributors = [c for c in contributions if c.direction == "negative"]

        summary = (
            f"The {outcome_name} of {outcome_value:.1f}% is {abs(deviation_pct):.1f}% "
            f"{performance} the baseline of {baseline_outcome:.1f}%. "
        )

        if positive_contributors:
            top_positive = positive_contributors[0]
            summary += (
                f"The primary positive driver is {top_positive.feature_name} "
                f"(contributing +{top_positive.shap_value:.2f}). "
            )

        if negative_contributors:
            top_negative = negative_contributors[0]
            summary += (
                f"The primary negative impact is from {top_negative.feature_name} "
                f"(contributing {top_negative.shap_value:.2f})."
            )

        return summary

    def _identify_improvements(
        self,
        contributions: List[FeatureContribution],
    ) -> List[str]:
        """Identify improvement opportunities from contributions."""
        opportunities = []

        for contribution in contributions:
            if contribution.direction == "negative":
                opportunities.append(
                    f"Improve {contribution.feature_name}: currently {contribution.feature_value:.2f} "
                    f"{contribution.unit}, target is {contribution.baseline_value:.2f} {contribution.unit}"
                )

        return opportunities[:5]  # Top 5 opportunities

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        outcome_value: float,
        contributions: List[FeatureContribution],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outcome_value": outcome_value,
            "num_contributions": len(contributions),
            "top_contributors": [c.feature_name for c in contributions[:3]],
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

class SHAPVisualizationHelper:
    """
    Helper class for generating SHAP visualization data.

    Prepares data in formats suitable for plotting libraries
    like matplotlib, plotly, or web-based visualizations.
    """

    @staticmethod
    def prepare_waterfall_data(
        explanation: OptimizationExplanation,
    ) -> Dict[str, Any]:
        """
        Prepare data for waterfall chart.

        Args:
            explanation: Optimization explanation

        Returns:
            Dictionary with waterfall chart data
        """
        labels = ["Baseline"]
        values = [explanation.baseline_outcome]
        colors = ["baseline"]

        for contribution in explanation.contributions:
            labels.append(contribution.feature_name)
            values.append(contribution.shap_value)
            if contribution.direction == "positive":
                colors.append("positive")
            elif contribution.direction == "negative":
                colors.append("negative")
            else:
                colors.append("neutral")

        labels.append("Final")
        values.append(explanation.outcome_value)
        colors.append("final")

        return {
            "labels": labels,
            "values": values,
            "colors": colors,
            "chart_type": "waterfall",
        }

    @staticmethod
    def prepare_importance_bar_data(
        explanation: OptimizationExplanation,
    ) -> Dict[str, Any]:
        """
        Prepare data for feature importance bar chart.

        Args:
            explanation: Optimization explanation

        Returns:
            Dictionary with bar chart data
        """
        return {
            "features": [f.feature_name for f in explanation.feature_importance],
            "importance": [f.importance_score for f in explanation.feature_importance],
            "categories": [f.category.value for f in explanation.feature_importance],
            "chart_type": "horizontal_bar",
        }

    @staticmethod
    def prepare_sensitivity_line_data(
        sensitivity_result: SensitivityResult,
    ) -> Dict[str, Any]:
        """
        Prepare data for sensitivity line chart.

        Args:
            sensitivity_result: Sensitivity analysis result

        Returns:
            Dictionary with line chart data
        """
        return {
            "x_values": sensitivity_result.test_values,
            "y_values": sensitivity_result.outcome_values,
            "x_label": sensitivity_result.feature_name,
            "y_label": "Outcome",
            "baseline_x": sensitivity_result.baseline_value,
            "baseline_y": sensitivity_result.baseline_outcome,
            "sensitivity": sensitivity_result.sensitivity,
            "elasticity": sensitivity_result.elasticity,
            "chart_type": "line",
        }
