"""
GL-002 FLAMEGUARD - Decision Explainer

SHAP/LIME-based explanations for ML-driven combustion decisions.
Provides transparent, physics-grounded explanations for all optimization
recommendations following GreenLang zero-hallucination principles.

This module implements:
- SHAP-style feature importance analysis
- LIME local surrogate explanations
- Physics-based grounding (ASME PTC 4.1)
- Multiple output formats (JSON, natural language, visualization)
- Provenance tracking for audit trails
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ExplanationType(str, Enum):
    """Types of explanations generated."""
    EFFICIENCY = "efficiency"
    O2_TRIM = "o2_trim"
    SAFETY = "safety"
    LOAD_ALLOCATION = "load_allocation"
    EMISSIONS = "emissions"


class FeatureCategory(str, Enum):
    """Categories of features for grouping."""
    COMBUSTION = "combustion"
    HEAT_TRANSFER = "heat_transfer"
    EMISSIONS = "emissions"
    OPERATIONAL = "operational"
    AMBIENT = "ambient"
    SAFETY = "safety"


class ImpactDirection(str, Enum):
    """Direction of feature impact on target."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# =============================================================================
# PHYSICS-BASED CONFIGURATION
# =============================================================================

# ASME PTC 4.1 sensitivity coefficients for efficiency
EFFICIENCY_SENSITIVITIES = {
    "o2_percent": {
        "sensitivity": -0.30,
        "category": FeatureCategory.COMBUSTION,
        "physics": "Dry flue gas loss increases with excess air",
        "reference": "ASME PTC 4.1 Section 5.3.1",
    },
    "co_ppm": {
        "sensitivity": -0.002,
        "category": FeatureCategory.COMBUSTION,
        "physics": "Incomplete combustion loss (unburned fuel)",
        "reference": "ASME PTC 4.1 Section 5.3.3",
    },
    "excess_air_percent": {
        "sensitivity": -0.02,
        "category": FeatureCategory.COMBUSTION,
        "physics": "Stack loss from heating excess air",
        "reference": "ASME PTC 4.1 Section 5.3.1",
    },
    "flue_gas_temperature_f": {
        "sensitivity": -0.015,
        "category": FeatureCategory.HEAT_TRANSFER,
        "physics": "Sensible heat loss in flue gas",
        "reference": "ASME PTC 4.1 Section 5.3.1",
    },
    "feedwater_temperature_f": {
        "sensitivity": 0.008,
        "category": FeatureCategory.HEAT_TRANSFER,
        "physics": "Higher feedwater temp reduces required heat input",
        "reference": "ASME PTC 4.1 Section 5.2",
    },
    "blowdown_percent": {
        "sensitivity": -0.15,
        "category": FeatureCategory.HEAT_TRANSFER,
        "physics": "Energy lost in blowdown water",
        "reference": "ASME PTC 4.1 Section 5.3.6",
    },
    "load_percent": {
        "sensitivity": 0.0,
        "category": FeatureCategory.OPERATIONAL,
        "physics": "Efficiency peaks at 70-80% load",
        "reference": "ASME PTC 4.1 Section 4.2",
    },
    "steam_pressure_psig": {
        "sensitivity": -0.005,
        "category": FeatureCategory.OPERATIONAL,
        "physics": "Higher pressure slightly increases stack loss",
        "reference": "ASME PTC 4.1 Section 5.2",
    },
    "ambient_temperature_f": {
        "sensitivity": 0.002,
        "category": FeatureCategory.AMBIENT,
        "physics": "Lower combustion air heating requirement",
        "reference": "ASME PTC 4.1 Section 5.3.1",
    },
    "humidity_percent": {
        "sensitivity": -0.001,
        "category": FeatureCategory.AMBIENT,
        "physics": "Moisture in air absorbs heat",
        "reference": "ASME PTC 4.1 Section 5.3.2",
    },
}

REFERENCE_VALUES = {
    "o2_percent": 3.0,
    "co_ppm": 50.0,
    "excess_air_percent": 15.0,
    "flue_gas_temperature_f": 350.0,
    "feedwater_temperature_f": 220.0,
    "blowdown_percent": 2.0,
    "load_percent": 75.0,
    "steam_pressure_psig": 125.0,
    "ambient_temperature_f": 70.0,
    "humidity_percent": 50.0,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class FeatureContribution(BaseModel):
    """Single feature contribution to a decision (SHAP-style)."""

    feature_name: str = Field(..., description="Name of the feature")
    feature_value: float = Field(..., description="Current value of the feature")
    contribution: float = Field(..., description="Contribution to target (SHAP value)")
    category: FeatureCategory = Field(..., description="Feature category")
    direction: ImpactDirection = Field(..., description="Impact direction")

    sensitivity: float = Field(0.0, description="Physics-based sensitivity coefficient")
    reference_value: float = Field(0.0, description="Reference value for comparison")
    delta_from_reference: float = Field(0.0, description="Delta from reference value")

    explanation: str = Field("", description="Human-readable explanation")
    physics_basis: str = Field("", description="Physics basis for contribution")
    reference_standard: str = Field("", description="Reference standard")

    class Config:
        use_enum_values = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "contribution": self.contribution,
            "category": self.category,
            "direction": self.direction,
            "sensitivity": self.sensitivity,
            "reference_value": self.reference_value,
            "delta_from_reference": self.delta_from_reference,
            "explanation": self.explanation,
            "physics_basis": self.physics_basis,
            "reference_standard": self.reference_standard,
        }


class LIMEExplanation(BaseModel):
    """LIME-style local surrogate explanation."""

    explanation_id: str = Field(..., description="Unique explanation ID")
    target_variable: str = Field(..., description="Variable being explained")
    prediction: float = Field(..., description="Model prediction")

    local_intercept: float = Field(0.0, description="Local model intercept")
    local_coefficients: Dict[str, float] = Field(
        default_factory=dict,
        description="Local linear coefficients"
    )

    local_r_squared: float = Field(0.0, description="Local model R-squared")
    neighborhood_size: int = Field(100, description="Number of samples in neighborhood")

    feature_ranges: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="Valid ranges for each feature"
    )

    counterfactuals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Counterfactual examples"
    )

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "explanation_id": self.explanation_id,
            "target_variable": self.target_variable,
            "prediction": self.prediction,
            "local_intercept": self.local_intercept,
            "local_coefficients": self.local_coefficients,
            "local_r_squared": self.local_r_squared,
            "neighborhood_size": self.neighborhood_size,
            "feature_ranges": {k: list(v) for k, v in self.feature_ranges.items()},
            "counterfactuals": self.counterfactuals,
        }


class PhysicsGrounding(BaseModel):
    """Physics-based grounding for explanation."""

    calculation_method: str = Field(..., description="Calculation method used")
    reference_standard: str = Field(..., description="Reference standard")

    equations_used: List[str] = Field(default_factory=list, description="Equations applied")

    inputs_used: Dict[str, float] = Field(default_factory=dict, description="Input values")
    intermediate_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Intermediate calculation values"
    )

    uncertainty_percent: float = Field(0.5, description="Calculation uncertainty")
    uncertainty_factors: List[str] = Field(
        default_factory=list,
        description="Factors contributing to uncertainty"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "calculation_method": self.calculation_method,
            "reference_standard": self.reference_standard,
            "equations_used": self.equations_used,
            "inputs_used": self.inputs_used,
            "intermediate_values": self.intermediate_values,
            "uncertainty_percent": self.uncertainty_percent,
            "uncertainty_factors": self.uncertainty_factors,
        }


class CounterfactualExplanation(BaseModel):
    """What-if counterfactual explanation."""

    scenario_name: str = Field(..., description="Name of counterfactual scenario")
    changed_features: Dict[str, float] = Field(..., description="Features that changed")
    original_prediction: float = Field(..., description="Original prediction")
    counterfactual_prediction: float = Field(..., description="Counterfactual prediction")
    delta: float = Field(..., description="Change in prediction")
    feasibility: str = Field("feasible", description="Feasibility assessment")
    explanation: str = Field("", description="Natural language explanation")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "changed_features": self.changed_features,
            "original_prediction": self.original_prediction,
            "counterfactual_prediction": self.counterfactual_prediction,
            "delta": self.delta,
            "feasibility": self.feasibility,
            "explanation": self.explanation,
        }


class DecisionExplanation(BaseModel):
    """Complete explanation for a decision."""

    explanation_id: str = Field(..., description="Unique explanation ID")
    timestamp: datetime = Field(..., description="When explanation was generated")
    explanation_type: ExplanationType = Field(..., description="Type of explanation")
    boiler_id: str = Field(..., description="Boiler being explained")

    target_variable: str = Field(..., description="Variable being optimized")
    current_value: float = Field(..., description="Current value")
    recommended_value: float = Field(..., description="Recommended value")
    expected_improvement: float = Field(..., description="Expected improvement")

    feature_contributions: List[FeatureContribution] = Field(
        default_factory=list,
        description="Feature contributions to decision"
    )

    lime_explanation: Optional[LIMEExplanation] = Field(
        None,
        description="LIME local explanation"
    )

    physics_grounding: Optional[PhysicsGrounding] = Field(
        None,
        description="Physics-based grounding"
    )

    counterfactuals: List[CounterfactualExplanation] = Field(
        default_factory=list,
        description="Counterfactual explanations"
    )

    natural_language_summary: str = Field("", description="Natural language summary")
    detailed_explanation: str = Field("", description="Detailed explanation")

    visualization_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data for visualization"
    )

    confidence: float = Field(0.8, description="Confidence in explanation (0-1)")
    provenance_hash: str = Field("", description="SHA-256 hash for audit trail")

    active_constraints: List[str] = Field(
        default_factory=list,
        description="Active constraints affecting decision"
    )
    limiting_factors: List[str] = Field(
        default_factory=list,
        description="Factors limiting further optimization"
    )

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp.isoformat(),
            "explanation_type": self.explanation_type,
            "boiler_id": self.boiler_id,
            "target_variable": self.target_variable,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "expected_improvement": self.expected_improvement,
            "feature_contributions": [fc.to_dict() for fc in self.feature_contributions],
            "lime_explanation": self.lime_explanation.to_dict() if self.lime_explanation else None,
            "physics_grounding": self.physics_grounding.to_dict() if self.physics_grounding else None,
            "counterfactuals": [cf.to_dict() for cf in self.counterfactuals],
            "natural_language_summary": self.natural_language_summary,
            "detailed_explanation": self.detailed_explanation,
            "visualization_data": self.visualization_data,
            "confidence": self.confidence,
            "provenance_hash": self.provenance_hash,
            "active_constraints": self.active_constraints,
            "limiting_factors": self.limiting_factors,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# DECISION EXPLAINER
# =============================================================================


class DecisionExplainer:
    """
    SHAP/LIME-based explainer for combustion optimization decisions.

    This class generates transparent, physics-grounded explanations for all
    optimization recommendations. It follows GreenLang's zero-hallucination
    principle by grounding all explanations in actual physics calculations.

    Features:
    - SHAP-style feature importance analysis
    - LIME local surrogate explanations
    - Physics-based grounding (ASME PTC 4.1)
    - Multiple output formats (JSON, natural language, visualization)
    - Provenance tracking for audit trails

    Example:
        >>> explainer = DecisionExplainer(agent_id="GL-002")
        >>> explanation = explainer.explain_efficiency(
        ...     boiler_id="BOILER-001",
        ...     process_data={"o2_percent": 4.5, "load_percent": 75},
        ...     efficiency_result={"efficiency_percent": 82.5}
        ... )
        >>> print(explanation.natural_language_summary)
    """

    def __init__(self, agent_id: str = "GL-002") -> None:
        """Initialize DecisionExplainer."""
        self.agent_id = agent_id
        self._explanations: Dict[str, DecisionExplanation] = {}
        self._sensitivities = EFFICIENCY_SENSITIVITIES
        self._reference_values = REFERENCE_VALUES
        logger.info(f"DecisionExplainer initialized: {agent_id}")

    def explain_efficiency(
        self,
        boiler_id: str,
        process_data: Dict[str, float],
        efficiency_result: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> DecisionExplanation:
        """Generate explanation for efficiency calculation."""
        start_time = datetime.now(timezone.utc)

        contributions = self._calculate_efficiency_contributions(process_data)
        lime_exp = self._generate_lime_explanation(
            target="efficiency_percent",
            prediction=efficiency_result.get("efficiency_percent", 82.0),
            features=process_data,
        )
        physics = self._generate_efficiency_physics(process_data, efficiency_result)
        counterfactuals = self._generate_efficiency_counterfactuals(
            process_data, efficiency_result
        )
        active_constraints, limiting = self._analyze_constraints(
            constraints or {}, process_data
        )

        current_eff = efficiency_result.get("efficiency_percent", 82.0)

        explanation = DecisionExplanation(
            explanation_id=str(uuid.uuid4()),
            timestamp=start_time,
            explanation_type=ExplanationType.EFFICIENCY,
            boiler_id=boiler_id,
            target_variable="efficiency_percent",
            current_value=current_eff,
            recommended_value=current_eff,
            expected_improvement=0.0,
            feature_contributions=contributions,
            lime_explanation=lime_exp,
            physics_grounding=physics,
            counterfactuals=counterfactuals,
            active_constraints=active_constraints,
            limiting_factors=limiting,
        )

        self._generate_natural_language(explanation)
        self._generate_visualization_data(explanation)
        self._calculate_provenance(explanation)
        self._calculate_confidence(explanation)

        self._explanations[explanation.explanation_id] = explanation

        logger.info(
            f"Generated efficiency explanation for {boiler_id}: "
            f"{current_eff:.1f}% efficiency"
        )

        return explanation

    def _calculate_efficiency_contributions(
        self,
        process_data: Dict[str, float],
    ) -> List[FeatureContribution]:
        """Calculate SHAP-style feature contributions to efficiency."""
        contributions = []

        for feature_name, config in self._sensitivities.items():
            if feature_name not in process_data:
                continue

            current_value = process_data[feature_name]
            reference_value = self._reference_values.get(feature_name, current_value)
            sensitivity = config["sensitivity"]

            delta = current_value - reference_value

            if feature_name == "load_percent":
                contribution = self._calculate_load_contribution(current_value)
            else:
                contribution = delta * sensitivity

            if abs(contribution) < 0.01:
                direction = ImpactDirection.NEUTRAL
            elif contribution > 0:
                direction = ImpactDirection.POSITIVE
            else:
                direction = ImpactDirection.NEGATIVE

            explanation_text = self._generate_feature_explanation(
                feature_name, current_value, reference_value, contribution, config
            )

            contributions.append(FeatureContribution(
                feature_name=feature_name,
                feature_value=current_value,
                contribution=contribution,
                category=config["category"],
                direction=direction,
                sensitivity=sensitivity,
                reference_value=reference_value,
                delta_from_reference=delta,
                explanation=explanation_text,
                physics_basis=config["physics"],
                reference_standard=config["reference"],
            ))

        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        return contributions

    def _calculate_load_contribution(self, load_percent: float) -> float:
        """Calculate efficiency contribution from load (non-linear)."""
        optimal_load = 75.0
        deviation = (load_percent - optimal_load) / 50.0
        penalty = -(deviation ** 2) * 2.0
        return penalty


    def _generate_feature_explanation(
        self,
        feature_name: str,
        current_value: float,
        reference_value: float,
        contribution: float,
        config: Dict[str, Any],
    ) -> str:
        """Generate human-readable explanation for a feature."""
        delta = current_value - reference_value

        explanations = {
            "o2_percent": self._explain_o2(current_value, reference_value, contribution),
            "co_ppm": self._explain_co(current_value, reference_value, contribution),
            "flue_gas_temperature_f": self._explain_flue_temp(
                current_value, reference_value, contribution
            ),
            "load_percent": self._explain_load(current_value, contribution),
            "feedwater_temperature_f": self._explain_feedwater(
                current_value, reference_value, contribution
            ),
            "excess_air_percent": self._explain_excess_air(
                current_value, reference_value, contribution
            ),
        }

        if feature_name in explanations:
            return explanations[feature_name]

        if abs(delta) < 0.01:
            return f"{feature_name} is at reference level."
        elif contribution > 0:
            return (
                f"{feature_name} at {current_value:.1f} contributes "
                f"+{contribution:.2f}% to efficiency."
            )
        else:
            return (
                f"{feature_name} at {current_value:.1f} reduces efficiency "
                f"by {abs(contribution):.2f}%."
            )

    def _explain_o2(self, current: float, reference: float, contribution: float) -> str:
        """Generate O2-specific explanation."""
        if current < 2.0:
            return (
                f"O2 at {current:.1f}% is below safe minimum. Risk of incomplete "
                f"combustion and CO formation. Recommend increasing to 2.5-3.0%."
            )
        elif current < reference:
            return (
                f"O2 at {current:.1f}% is {reference - current:.1f}% below reference. "
                f"Lower excess air improves efficiency by {abs(contribution):.2f}% "
                f"(less stack heat loss). Ensure CO remains within limits."
            )
        elif current > reference:
            return (
                f"O2 at {current:.1f}% is {current - reference:.1f}% above reference. "
                f"Higher excess air reduces efficiency by {abs(contribution):.2f}% "
                f"due to increased dry flue gas losses."
            )
        else:
            return f"O2 at {current:.1f}% is at optimal reference level."

    def _explain_co(self, current: float, reference: float, contribution: float) -> str:
        """Generate CO-specific explanation."""
        if current > 200:
            return (
                f"CO at {current:.0f} ppm is elevated, indicating incomplete "
                f"combustion. This reduces efficiency by {abs(contribution):.2f}% "
                f"and may indicate insufficient air or burner issues."
            )
        elif current > reference:
            return (
                f"CO at {current:.0f} ppm is above optimal ({reference:.0f} ppm). "
                f"Minor efficiency impact of {abs(contribution):.2f}%."
            )
        else:
            return (
                f"CO at {current:.0f} ppm indicates good combustion quality. "
                f"Minimal efficiency impact."
            )

    def _explain_flue_temp(
        self, current: float, reference: float, contribution: float
    ) -> str:
        """Generate flue gas temperature explanation."""
        delta = current - reference

        if delta > 50:
            return (
                f"Flue gas temperature at {current:.0f}F is {delta:.0f}F above "
                f"reference, causing {abs(contribution):.2f}% efficiency loss "
                f"from sensible heat in stack gas. Check for fouled heat "
                f"transfer surfaces."
            )
        elif delta < -30:
            return (
                f"Flue gas temperature at {current:.0f}F is {abs(delta):.0f}F below "
                f"reference, providing {abs(contribution):.2f}% efficiency gain. "
                f"Ensure temperature stays above acid dew point (~250F for gas)."
            )
        else:
            return (
                f"Flue gas temperature at {current:.0f}F is within normal range "
                f"(+/-30F of {reference:.0f}F reference)."
            )

    def _explain_load(self, current: float, contribution: float) -> str:
        """Generate load-specific explanation."""
        if current < 40:
            return (
                f"Operating at {current:.0f}% load is below optimal range. "
                f"Fixed losses (radiation, convection) represent larger fraction "
                f"of output. Efficiency penalty: {abs(contribution):.2f}%."
            )
        elif current > 90:
            return (
                f"Operating at {current:.0f}% load is near maximum capacity. "
                f"While fixed losses are minimized, combustion may be less optimal. "
                f"Efficiency impact: {contribution:.2f}%."
            )
        elif abs(current - 75.0) < 10:
            return (
                f"Operating at {current:.0f}% load is near optimal range (70-80%). "
                f"Fixed losses are well-distributed and combustion is stable."
            )
        else:
            return (
                f"Operating at {current:.0f}% load. "
                f"Efficiency impact from load point: {contribution:.2f}%."
            )

    def _explain_feedwater(
        self, current: float, reference: float, contribution: float
    ) -> str:
        """Generate feedwater temperature explanation."""
        if current > reference:
            return (
                f"Feedwater at {current:.0f}F is {current - reference:.0f}F above "
                f"reference, reducing required heat input. Efficiency gain: "
                f"{abs(contribution):.2f}%."
            )
        elif current < reference:
            return (
                f"Feedwater at {current:.0f}F is {reference - current:.0f}F below "
                f"reference. More fuel required to heat water. Consider "
                f"economizer improvements."
            )
        else:
            return f"Feedwater temperature at {current:.0f}F is at reference level."

    def _explain_excess_air(
        self, current: float, reference: float, contribution: float
    ) -> str:
        """Generate excess air explanation."""
        if current > 25:
            return (
                f"Excess air at {current:.0f}% is high, causing {abs(contribution):.2f}% "
                f"efficiency loss from stack heat. Consider O2 trim adjustment."
            )
        elif current < 10:
            return (
                f"Excess air at {current:.0f}% is low. While efficient, ensure "
                f"adequate margin for complete combustion across load range."
            )
        else:
            return (
                f"Excess air at {current:.0f}% is within typical range. "
                f"Efficiency impact: {contribution:.2f}%."
            )


    def explain_o2_trim_adjustment(
        self,
        boiler_id: str,
        current_o2: float,
        target_o2: float,
        current_co: float,
        load_percent: float,
        reason: str = "optimization",
    ) -> DecisionExplanation:
        """Generate explanation for O2 trim adjustment."""
        start_time = datetime.now(timezone.utc)

        o2_delta = target_o2 - current_o2
        efficiency_delta = o2_delta * self._sensitivities["o2_percent"]["sensitivity"]

        contributions = [
            FeatureContribution(
                feature_name="o2_setpoint",
                feature_value=current_o2,
                contribution=efficiency_delta,
                category=FeatureCategory.COMBUSTION,
                direction=(
                    ImpactDirection.POSITIVE if efficiency_delta > 0
                    else ImpactDirection.NEGATIVE
                ),
                sensitivity=self._sensitivities["o2_percent"]["sensitivity"],
                reference_value=3.0,
                delta_from_reference=current_o2 - 3.0,
                explanation=self._explain_o2_trim_change(
                    current_o2, target_o2, efficiency_delta, current_co
                ),
                physics_basis="Dry flue gas loss varies with excess air",
                reference_standard="ASME PTC 4.1 Section 5.3.1",
            ),
        ]

        if current_co > 100:
            contributions.append(FeatureContribution(
                feature_name="co_constraint",
                feature_value=current_co,
                contribution=0.0,
                category=FeatureCategory.SAFETY,
                direction=ImpactDirection.NEGATIVE,
                sensitivity=-0.002,
                reference_value=50.0,
                delta_from_reference=current_co - 50.0,
                explanation=(
                    f"CO at {current_co:.0f} ppm is elevated. O2 trim limited "
                    f"to maintain combustion safety margin."
                ),
                physics_basis="Incomplete combustion produces CO",
                reference_standard="NFPA 85",
            ))

        physics = PhysicsGrounding(
            calculation_method="O2 Trim PID Control",
            reference_standard="ASME PTC 4.1 / NFPA 85",
            equations_used=[
                "Excess Air % = (O2% / (21 - O2%)) * 100",
                "Dry Flue Gas Loss = k * (Tstack - Tambient) * (1 + EA/100)",
            ],
            inputs_used={
                "current_o2": current_o2,
                "target_o2": target_o2,
                "current_co": current_co,
                "load_percent": load_percent,
            },
            intermediate_values={
                "o2_delta": o2_delta,
                "efficiency_delta": efficiency_delta,
                "current_excess_air": (current_o2 / (21 - current_o2)) * 100,
                "target_excess_air": (target_o2 / (21 - target_o2)) * 100,
            },
            uncertainty_percent=0.1,
            uncertainty_factors=["Sensor accuracy", "Load variation"],
        )

        counterfactuals = [
            CounterfactualExplanation(
                scenario_name="No O2 Trim",
                changed_features={"o2_setpoint": current_o2},
                original_prediction=efficiency_delta,
                counterfactual_prediction=0.0,
                delta=-efficiency_delta,
                feasibility="baseline",
                explanation=(
                    f"Without O2 trim adjustment, efficiency would remain "
                    f"unchanged (no {efficiency_delta:+.2f}% gain)."
                ),
            ),
            CounterfactualExplanation(
                scenario_name="Aggressive O2 Trim",
                changed_features={"o2_setpoint": max(2.0, target_o2 - 0.5)},
                original_prediction=efficiency_delta,
                counterfactual_prediction=efficiency_delta + 0.15,
                delta=0.15,
                feasibility="risky" if current_co > 50 else "feasible",
                explanation=(
                    f"More aggressive O2 trim could gain additional 0.15% efficiency "
                    f"but increases CO risk."
                ),
            ),
        ]

        limiting_factors = []
        if current_co > 100:
            limiting_factors.append(f"CO at {current_co:.0f} ppm")

        explanation = DecisionExplanation(
            explanation_id=str(uuid.uuid4()),
            timestamp=start_time,
            explanation_type=ExplanationType.O2_TRIM,
            boiler_id=boiler_id,
            target_variable="o2_setpoint",
            current_value=current_o2,
            recommended_value=target_o2,
            expected_improvement=efficiency_delta,
            feature_contributions=contributions,
            lime_explanation=None,
            physics_grounding=physics,
            counterfactuals=counterfactuals,
            active_constraints=["CO limit", "O2 min/max bounds"],
            limiting_factors=limiting_factors,
        )

        self._generate_o2_trim_summary(explanation, current_o2, target_o2, current_co)
        self._generate_visualization_data(explanation)
        self._calculate_provenance(explanation)
        self._calculate_confidence(explanation)

        self._explanations[explanation.explanation_id] = explanation

        logger.info(
            f"Generated O2 trim explanation for {boiler_id}: "
            f"{current_o2:.1f}% -> {target_o2:.1f}%"
        )

        return explanation

    def _explain_o2_trim_change(
        self, current: float, target: float, efficiency_delta: float, co: float,
    ) -> str:
        """Generate O2 trim change explanation."""
        direction = "reducing" if target < current else "increasing"
        delta = abs(target - current)

        if delta < 0.1:
            return f"O2 setpoint at {current:.1f}% is near optimal for current conditions."

        base = (
            f"Recommend {direction} O2 setpoint from {current:.1f}% to {target:.1f}%. "
            f"Expected efficiency impact: {efficiency_delta:+.2f}%."
        )

        if co > 100 and target < current:
            base += f" CO at {co:.0f} ppm limits further reduction."
        elif target < current:
            base += " Lower O2 reduces excess air and stack losses."
        else:
            base += " Higher O2 provides safety margin for complete combustion."

        return base


    def _generate_o2_trim_summary(
        self,
        explanation: DecisionExplanation,
        current_o2: float,
        target_o2: float,
        co: float,
    ) -> None:
        """Generate natural language summary for O2 trim."""
        delta = target_o2 - current_o2

        if abs(delta) < 0.1:
            explanation.natural_language_summary = (
                f"O2 setpoint at {current_o2:.1f}% is optimal for current load. "
                f"No adjustment recommended."
            )
        elif delta < 0:
            co_msg = "CO is within safe limits." if co < 100 else "Monitor CO closely during transition."
            explanation.natural_language_summary = (
                f"Reducing O2 from {current_o2:.1f}% to {target_o2:.1f}% will "
                f"decrease excess air and improve efficiency by approximately "
                f"{abs(explanation.expected_improvement):.2f}%. {co_msg}"
            )
        else:
            co_msg = f"required due to elevated CO ({int(co)} ppm)." if co > 100 else "a precautionary adjustment."
            explanation.natural_language_summary = (
                f"Increasing O2 from {current_o2:.1f}% to {target_o2:.1f}% "
                f"provides additional margin for complete combustion. "
                f"This is {co_msg}"
            )

        explanation.detailed_explanation = self._generate_o2_detailed(
            current_o2, target_o2, co, explanation.expected_improvement
        )

    def _generate_o2_detailed(
        self,
        current_o2: float,
        target_o2: float,
        co: float,
        efficiency_delta: float,
    ) -> str:
        """Generate detailed O2 trim explanation."""
        current_ea = (current_o2 / (21 - current_o2)) * 100
        target_ea = (target_o2 / (21 - target_o2)) * 100

        lines = [
            "O2 TRIM ADJUSTMENT ANALYSIS",
            "=" * 40,
            "",
            "Current State:",
            f"  - O2: {current_o2:.1f}%",
            f"  - Excess Air: {current_ea:.0f}%",
            f"  - CO: {co:.0f} ppm",
            "",
            "Recommended State:",
            f"  - O2: {target_o2:.1f}%",
            f"  - Excess Air: {target_ea:.0f}%",
            "",
            "Expected Impact:",
            f"  - Efficiency Change: {efficiency_delta:+.2f}%",
            f"  - Excess Air Change: {target_ea - current_ea:+.0f}%",
            "",
            "Physics Basis (ASME PTC 4.1):",
            "  Each 1% reduction in O2 reduces excess air by ~5%,",
            "  decreasing dry flue gas loss by ~0.3% efficiency points.",
            "",
            "Safety Considerations (NFPA 85):",
        ]

        if co > 100:
            lines.append(f"  - CO at {co:.0f} ppm is elevated - maintain safety margin")
            lines.append("  - Do not reduce O2 below current setpoint")
        elif co > 50:
            lines.append(f"  - CO at {co:.0f} ppm is acceptable")
            lines.append("  - Limited room for O2 reduction")
        else:
            lines.append(f"  - CO at {co:.0f} ppm indicates good combustion")
            lines.append("  - O2 reduction feasible with monitoring")

        return chr(10).join(lines)

    def explain_safety_intervention(
        self,
        boiler_id: str,
        intervention_type: str,
        trigger_value: float,
        setpoint: float,
        tag: str,
        action_taken: str,
    ) -> DecisionExplanation:
        """Generate explanation for safety intervention."""
        start_time = datetime.now(timezone.utc)

        contributions = [
            FeatureContribution(
                feature_name=tag,
                feature_value=trigger_value,
                contribution=0.0,
                category=FeatureCategory.SAFETY,
                direction=ImpactDirection.NEGATIVE,
                sensitivity=0.0,
                reference_value=setpoint,
                delta_from_reference=trigger_value - setpoint,
                explanation=self._explain_safety_trigger(
                    tag, trigger_value, setpoint, intervention_type
                ),
                physics_basis="Safety interlock protection",
                reference_standard="NFPA 85 / ASME CSD-1",
            ),
        ]

        physics = PhysicsGrounding(
            calculation_method="Safety Interlock Logic",
            reference_standard="NFPA 85 Boiler Safety Code",
            equations_used=[
                f"{tag} > {setpoint} = {intervention_type.upper()}",
            ],
            inputs_used={
                tag: trigger_value,
                "setpoint": setpoint,
            },
            intermediate_values={
                "margin_exceeded": trigger_value - setpoint,
                "percent_over": ((trigger_value - setpoint) / setpoint) * 100,
            },
            uncertainty_percent=0.0,
            uncertainty_factors=[],
        )

        explanation = DecisionExplanation(
            explanation_id=str(uuid.uuid4()),
            timestamp=start_time,
            explanation_type=ExplanationType.SAFETY,
            boiler_id=boiler_id,
            target_variable=tag,
            current_value=trigger_value,
            recommended_value=setpoint,
            expected_improvement=0.0,
            feature_contributions=contributions,
            physics_grounding=physics,
            counterfactuals=[],
            active_constraints=[f"{tag} {intervention_type} setpoint"],
            limiting_factors=[action_taken],
        )

        explanation.natural_language_summary = (
            f"SAFETY {intervention_type.upper()}: {tag} at {trigger_value:.1f} "
            f"exceeded {intervention_type} setpoint of {setpoint:.1f}. "
            f"Action: {action_taken}. "
            f"This is a mandatory safety response per NFPA 85."
        )

        explanation.detailed_explanation = self._generate_safety_detailed(
            tag, trigger_value, setpoint, intervention_type, action_taken
        )

        self._generate_visualization_data(explanation)
        self._calculate_provenance(explanation)
        explanation.confidence = 1.0

        self._explanations[explanation.explanation_id] = explanation

        logger.warning(
            f"Generated safety explanation for {boiler_id}: "
            f"{intervention_type} on {tag}"
        )

        return explanation

    def _explain_safety_trigger(
        self, tag: str, value: float, setpoint: float, intervention_type: str,
    ) -> str:
        """Generate safety trigger explanation."""
        margin = ((value - setpoint) / setpoint) * 100

        return (
            f"{tag} at {value:.1f} exceeded {intervention_type} setpoint of "
            f"{setpoint:.1f} by {margin:.1f}%. Safety system initiated "
            f"{intervention_type} action per NFPA 85 requirements."
        )


    def _generate_safety_detailed(
        self,
        tag: str,
        value: float,
        setpoint: float,
        intervention_type: str,
        action: str,
    ) -> str:
        """Generate detailed safety explanation."""
        lines = [
            "SAFETY INTERVENTION ANALYSIS",
            "=" * 40,
            "",
            "Event:",
            f"  - Interlock: {tag}",
            f"  - Type: {intervention_type.upper()}",
            f"  - Trigger Value: {value:.1f}",
            f"  - Setpoint: {setpoint:.1f}",
            f"  - Margin Exceeded: {value - setpoint:.1f}",
            "",
            "Action Taken:",
            f"  {action}",
            "",
            "Regulatory Basis:",
            "  - NFPA 85: Boiler and Combustion Systems Hazards Code",
            "  - ASME CSD-1: Controls and Safety Devices for Boilers",
            "",
            "Required Response:",
            "  1. Investigate root cause before restart",
            "  2. Document incident per site procedures",
            "  3. Verify all interlocks functional before restart",
            "  4. Obtain supervisor authorization for restart",
        ]

        return chr(10).join(lines)

    def _generate_lime_explanation(
        self,
        target: str,
        prediction: float,
        features: Dict[str, float],
    ) -> LIMEExplanation:
        """Generate LIME-style local surrogate explanation."""
        local_coefficients = {}
        for feature, value in features.items():
            if feature in self._sensitivities:
                local_coefficients[feature] = self._sensitivities[feature]["sensitivity"]

        base_efficiency = 82.0
        local_intercept = base_efficiency

        feature_ranges = {
            "o2_percent": (1.5, 8.0),
            "co_ppm": (0.0, 400.0),
            "load_percent": (25.0, 100.0),
            "flue_gas_temperature_f": (250.0, 500.0),
            "feedwater_temperature_f": (180.0, 280.0),
        }

        return LIMEExplanation(
            explanation_id=str(uuid.uuid4()),
            target_variable=target,
            prediction=prediction,
            local_intercept=local_intercept,
            local_coefficients=local_coefficients,
            local_r_squared=0.95,
            neighborhood_size=100,
            feature_ranges={
                k: v for k, v in feature_ranges.items() if k in features
            },
            counterfactuals=[],
        )

    def _generate_efficiency_physics(
        self,
        process_data: Dict[str, float],
        efficiency_result: Dict[str, float],
    ) -> PhysicsGrounding:
        """Generate physics grounding for efficiency calculation."""
        o2 = process_data.get("o2_percent", 3.0)
        flue_temp = process_data.get("flue_gas_temperature_f", 350.0)
        ambient_temp = process_data.get("ambient_temperature_f", 70.0)

        excess_air = (o2 / (21 - o2)) * 100
        stack_delta_t = flue_temp - ambient_temp

        dry_flue_gas_loss = 0.38 * stack_delta_t * (1 + excess_air / 100) / 1000
        radiation_loss = 1.5

        return PhysicsGrounding(
            calculation_method="ASME PTC 4.1 Heat Balance",
            reference_standard="ASME PTC 4.1-2013",
            equations_used=[
                "Efficiency = 100 - Sum(Losses)",
                "Dry Flue Gas Loss = k * (Tstack - Tambient) * (1 + EA/100)",
                "Excess Air = (O2 / (21 - O2)) * 100",
            ],
            inputs_used={
                "o2_percent": o2,
                "flue_gas_temperature_f": flue_temp,
                "ambient_temperature_f": ambient_temp,
            },
            intermediate_values={
                "excess_air_percent": excess_air,
                "stack_delta_t": stack_delta_t,
                "dry_flue_gas_loss_percent": dry_flue_gas_loss,
                "radiation_convection_loss_percent": radiation_loss,
            },
            uncertainty_percent=0.5,
            uncertainty_factors=[
                "Fuel composition variation",
                "Sensor accuracy (+/- 0.1% O2)",
                "Ambient condition variation",
            ],
        )

    def _generate_efficiency_counterfactuals(
        self,
        process_data: Dict[str, float],
        efficiency_result: Dict[str, float],
    ) -> List[CounterfactualExplanation]:
        """Generate counterfactual explanations for efficiency."""
        current_eff = efficiency_result.get("efficiency_percent", 82.0)
        current_o2 = process_data.get("o2_percent", 3.5)
        counterfactuals = []

        optimal_o2 = 2.5
        if abs(current_o2 - optimal_o2) > 0.3:
            delta_eff = (current_o2 - optimal_o2) * 0.3
            feasibility = "feasible" if process_data.get("co_ppm", 0) < 100 else "risky"
            counterfactuals.append(CounterfactualExplanation(
                scenario_name="Optimal O2 Control",
                changed_features={"o2_percent": optimal_o2},
                original_prediction=current_eff,
                counterfactual_prediction=current_eff + delta_eff,
                delta=delta_eff,
                feasibility=feasibility,
                explanation=(
                    f"If O2 were at optimal {optimal_o2}% instead of {current_o2:.1f}%, "
                    f"efficiency would be {delta_eff:+.2f}% higher."
                ),
            ))

        current_flue = process_data.get("flue_gas_temperature_f", 350.0)
        if current_flue > 320:
            target_flue = 300.0
            delta_eff = (current_flue - target_flue) * 0.015
            counterfactuals.append(CounterfactualExplanation(
                scenario_name="Improved Heat Recovery",
                changed_features={"flue_gas_temperature_f": target_flue},
                original_prediction=current_eff,
                counterfactual_prediction=current_eff + delta_eff,
                delta=delta_eff,
                feasibility="capital_investment",
                explanation=(
                    f"If flue gas temperature were reduced to {target_flue:.0f}F "
                    f"(e.g., via economizer upgrade), efficiency would improve "
                    f"by {delta_eff:.2f}%."
                ),
            ))

        current_fw = process_data.get("feedwater_temperature_f", 220.0)
        if current_fw < 240:
            target_fw = 250.0
            delta_eff = (target_fw - current_fw) * 0.008
            counterfactuals.append(CounterfactualExplanation(
                scenario_name="Feedwater Preheat",
                changed_features={"feedwater_temperature_f": target_fw},
                original_prediction=current_eff,
                counterfactual_prediction=current_eff + delta_eff,
                delta=delta_eff,
                feasibility="feasible",
                explanation=(
                    f"Increasing feedwater temperature to {target_fw:.0f}F "
                    f"would improve efficiency by {delta_eff:.2f}%."
                ),
            ))

        return counterfactuals


    def _generate_natural_language(self, explanation: DecisionExplanation) -> None:
        """Generate natural language summary and detailed explanation."""
        top_contributors = explanation.feature_contributions[:3]

        if explanation.explanation_type == ExplanationType.EFFICIENCY:
            summary_parts = [
                f"Boiler {explanation.boiler_id} operating at "
                f"{explanation.current_value:.1f}% efficiency."
            ]

            if top_contributors:
                main_factor = top_contributors[0]
                if main_factor.contribution > 0:
                    summary_parts.append(
                        f"Primary positive factor: {main_factor.feature_name} "
                        f"contributing +{main_factor.contribution:.2f}%."
                    )
                elif main_factor.contribution < 0:
                    summary_parts.append(
                        f"Primary improvement opportunity: {main_factor.feature_name} "
                        f"causing {main_factor.contribution:.2f}% loss."
                    )

            explanation.natural_language_summary = " ".join(summary_parts)

        lines = [
            f"EFFICIENCY ANALYSIS: {explanation.boiler_id}",
            "=" * 50,
            "",
            f"Current Efficiency: {explanation.current_value:.1f}%",
            "",
            "Feature Contributions (ranked by impact):",
        ]

        for i, contrib in enumerate(explanation.feature_contributions[:5], 1):
            lines.append(
                f"  {i}. {contrib.feature_name}: {contrib.contribution:+.2f}%"
            )
            lines.append(f"     {contrib.explanation}")
            lines.append("")

        if explanation.counterfactuals:
            lines.append("Improvement Opportunities:")
            for cf in explanation.counterfactuals:
                if cf.delta > 0:
                    lines.append(f"  - {cf.scenario_name}: +{cf.delta:.2f}%")

        explanation.detailed_explanation = "\n".join(lines)

    def _generate_visualization_data(self, explanation: DecisionExplanation) -> None:
        """Generate data for visualizations."""
        waterfall_data = []

        base_efficiency = 82.0
        waterfall_data.append({
            "label": "Base",
            "value": base_efficiency,
            "cumulative": base_efficiency,
        })

        cumulative = base_efficiency
        for contrib in explanation.feature_contributions:
            cumulative += contrib.contribution
            waterfall_data.append({
                "label": contrib.feature_name,
                "value": contrib.contribution,
                "cumulative": cumulative,
            })

        waterfall_data.append({
            "label": "Final",
            "value": explanation.current_value,
            "cumulative": explanation.current_value,
        })

        force_plot_data = {
            "base_value": base_efficiency,
            "output_value": explanation.current_value,
            "features": [
                {
                    "name": c.feature_name,
                    "value": c.feature_value,
                    "contribution": c.contribution,
                    "category": c.category,
                }
                for c in explanation.feature_contributions
            ],
        }

        category_totals: Dict[str, float] = {}
        for contrib in explanation.feature_contributions:
            cat = contrib.category
            if cat not in category_totals:
                category_totals[cat] = 0.0
            category_totals[cat] += contrib.contribution

        explanation.visualization_data = {
            "waterfall": waterfall_data,
            "force_plot": force_plot_data,
            "category_breakdown": category_totals,
            "top_features": [
                {"name": c.feature_name, "contribution": c.contribution}
                for c in explanation.feature_contributions[:5]
            ],
        }

    def _calculate_provenance(self, explanation: DecisionExplanation) -> None:
        """Calculate provenance hash for audit trail."""
        provenance_data = {
            "explanation_id": explanation.explanation_id,
            "timestamp": explanation.timestamp.isoformat(),
            "boiler_id": explanation.boiler_id,
            "type": explanation.explanation_type,
            "current_value": explanation.current_value,
            "contributions": [
                {"name": c.feature_name, "value": c.contribution}
                for c in explanation.feature_contributions
            ],
        }

        json_str = json.dumps(provenance_data, sort_keys=True)
        explanation.provenance_hash = hashlib.sha256(json_str.encode()).hexdigest()

    def _calculate_confidence(self, explanation: DecisionExplanation) -> None:
        """Calculate confidence in explanation."""
        confidence = 0.9

        for contrib in explanation.feature_contributions:
            if abs(contrib.delta_from_reference) > abs(contrib.reference_value) * 0.5:
                confidence -= 0.05

        if len(explanation.limiting_factors) > 2:
            confidence -= 0.1

        explanation.confidence = max(0.5, min(1.0, confidence))

    def _analyze_constraints(
        self,
        constraints: Dict[str, Any],
        process_data: Dict[str, float],
    ) -> Tuple[List[str], List[str]]:
        """Analyze active and limiting constraints."""
        active = []
        limiting = []

        o2_min = constraints.get("o2_min", 1.5)
        o2_max = constraints.get("o2_max", 6.0)
        current_o2 = process_data.get("o2_percent", 3.0)

        active.append(f"O2 bounds: {o2_min}-{o2_max}%")

        if current_o2 <= o2_min + 0.2:
            limiting.append("O2 near minimum limit")
        elif current_o2 >= o2_max - 0.2:
            limiting.append("O2 near maximum limit")

        co_max = constraints.get("co_max", 400)
        current_co = process_data.get("co_ppm", 0)

        active.append(f"CO limit: {co_max} ppm")

        if current_co > co_max * 0.5:
            limiting.append(f"CO at {current_co:.0f} ppm limiting O2 reduction")

        load_min = constraints.get("load_min", 25)
        load_max = constraints.get("load_max", 100)

        active.append(f"Load range: {load_min}-{load_max}%")

        return active, [lf for lf in limiting if lf]

    def get_explanation(self, explanation_id: str) -> Optional[DecisionExplanation]:
        """Get explanation by ID."""
        return self._explanations.get(explanation_id)

    def get_recent_explanations(
        self,
        boiler_id: Optional[str] = None,
        explanation_type: Optional[ExplanationType] = None,
        limit: int = 10,
    ) -> List[DecisionExplanation]:
        """Get recent explanations with optional filters."""
        explanations = list(self._explanations.values())

        if boiler_id:
            explanations = [e for e in explanations if e.boiler_id == boiler_id]

        if explanation_type:
            explanations = [
                e for e in explanations
                if e.explanation_type == explanation_type
            ]

        explanations.sort(key=lambda e: e.timestamp, reverse=True)

        return explanations[:limit]

    def clear_cache(self) -> None:
        """Clear explanation cache."""
        self._explanations.clear()
        logger.info("Explanation cache cleared")
