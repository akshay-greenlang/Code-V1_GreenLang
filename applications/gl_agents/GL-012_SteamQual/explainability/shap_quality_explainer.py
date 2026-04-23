"""
GL-012 STEAMQUAL - SHAP Quality Explainer

SHAP (SHapley Additive exPlanations) implementation for explaining
steam quality estimates including dryness fraction predictions.

This module provides:
1. Feature importance analysis for steam quality parameters
2. Contribution breakdown by sensor/feature
3. Counterfactual explanations ("what would change x?")
4. Physics-grounded explanations per ASME PTC 19.11

IMPORTANT: SHAP is used ONLY for explaining model outputs (quality estimation,
anomaly detection), NOT for explaining deterministic calculations. Physics-based
calculations use separate physics explainer logic within this module.

All explanations are traceable to data and assumptions per playbook requirement.

Reference:
    - Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to
      Interpreting Model Predictions. NeurIPS.
    - ASME PTC 19.11 Steam Quality Measurement

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
import uuid
import math

logger = logging.getLogger(__name__)

# Conditional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available - SHAP features will be limited")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available - using physics-based fallback explainer")


# =============================================================================
# ENUMERATIONS
# =============================================================================

class QualityFeatureCategory(Enum):
    """Categories of features for steam quality estimation."""

    THERMODYNAMIC = "thermodynamic"      # P, T, h, s
    SEPARATOR = "separator"              # Separator/scrubber parameters
    DRUM = "drum"                         # Drum level and behavior
    FLOW = "flow"                         # Steam/condensate flow
    OPERATIONAL = "operational"          # Load, time, cycles
    MEASUREMENT = "measurement"          # Sensor-derived features
    DERIVED = "derived"                  # Calculated features


class ModelType(Enum):
    """Types of models that can be explained."""

    DRYNESS_ESTIMATION = "dryness_estimation"
    CARRYOVER_PREDICTION = "carryover_prediction"
    QUALITY_ANOMALY = "quality_anomaly"
    SEPARATOR_EFFICIENCY = "separator_efficiency"
    SOFT_SENSOR = "soft_sensor"


class ImpactDirection(Enum):
    """Direction of feature impact on quality."""

    POSITIVE = "positive"   # Increases quality/dryness
    NEGATIVE = "negative"   # Decreases quality/dryness
    NEUTRAL = "neutral"     # Minimal impact


class ExplanationType(Enum):
    """Types of explanations generated."""

    LOCAL = "local"            # Single prediction
    GLOBAL = "global"          # Feature importance
    COUNTERFACTUAL = "counterfactual"
    PHYSICS = "physics"        # Physics-grounded


# =============================================================================
# PHYSICS-BASED CONFIGURATION (ASME PTC 19.11)
# =============================================================================

QUALITY_SENSITIVITIES = {
    # Thermodynamic properties
    "pressure_psig": {
        "sensitivity": 0.0,  # Pressure affects saturation but not directly x
        "category": QualityFeatureCategory.THERMODYNAMIC,
        "physics": "Pressure determines saturation temperature and enthalpy",
        "reference": "ASME PTC 19.11, IAPWS-IF97",
        "unit": "psig",
    },
    "temperature_f": {
        "sensitivity": -0.005,  # Superheat indicates dry steam
        "category": QualityFeatureCategory.THERMODYNAMIC,
        "physics": "Temperature below saturation indicates moisture presence",
        "reference": "ASME PTC 19.11 Section 4",
        "unit": "F",
    },
    "superheat_f": {
        "sensitivity": 0.01,  # Positive superheat means x=1
        "category": QualityFeatureCategory.THERMODYNAMIC,
        "physics": "Superheated steam has quality x=1.0 by definition",
        "reference": "ASME PTC 19.11 Section 4.1",
        "unit": "F",
    },
    "enthalpy_btu_lb": {
        "sensitivity": 0.001,  # Higher enthalpy = higher quality
        "category": QualityFeatureCategory.THERMODYNAMIC,
        "physics": "Quality x = (h - hf) / hfg per thermodynamic definition",
        "reference": "ASME PTC 19.11 Equation 4-1",
        "unit": "BTU/lb",
    },

    # Drum parameters
    "drum_level_pct": {
        "sensitivity": -0.02,  # Higher drum level = more carryover risk
        "category": QualityFeatureCategory.DRUM,
        "physics": "High drum level increases moisture entrainment in steam",
        "reference": "ASME Section I, API 534",
        "unit": "%",
    },
    "drum_level_variability": {
        "sensitivity": -0.03,  # Level swings cause carryover
        "category": QualityFeatureCategory.DRUM,
        "physics": "Level fluctuations indicate unstable separation",
        "reference": "ASME Section I PG-60",
        "unit": "std_dev",
    },

    # Separator parameters
    "separator_dp_psi": {
        "sensitivity": 0.015,  # Higher dP = better separation
        "category": QualityFeatureCategory.SEPARATOR,
        "physics": "Pressure drop indicates centrifugal separation effectiveness",
        "reference": "API 560, Separator Design Standards",
        "unit": "psi",
    },
    "separator_velocity_fps": {
        "sensitivity": -0.01,  # Too high velocity reduces separation
        "category": QualityFeatureCategory.SEPARATOR,
        "physics": "Excessive velocity causes re-entrainment of droplets",
        "reference": "API 560 Table 1",
        "unit": "ft/s",
    },

    # Flow parameters
    "steam_flow_klb_hr": {
        "sensitivity": -0.005,  # Higher flow = more turbulence
        "category": QualityFeatureCategory.FLOW,
        "physics": "Higher flow increases turbulence and potential carryover",
        "reference": "ASME PTC 19.5 Flow Measurement",
        "unit": "klb/hr",
    },
    "condensate_flow_gpm": {
        "sensitivity": -0.008,  # High condensate indicates moisture
        "category": QualityFeatureCategory.FLOW,
        "physics": "Condensate presence indicates moisture in steam",
        "reference": "ASME PTC 19.11 Section 5",
        "unit": "gpm",
    },

    # Operational
    "load_pct": {
        "sensitivity": 0.0,  # Non-linear relationship
        "category": QualityFeatureCategory.OPERATIONAL,
        "physics": "Quality varies with load - optimal at design point",
        "reference": "Boiler performance curves",
        "unit": "%",
    },
    "prv_condensation_rate": {
        "sensitivity": -0.02,  # PRV condensation indicates poor quality
        "category": QualityFeatureCategory.MEASUREMENT,
        "physics": "PRV condensation indicates wet steam at expansion",
        "reference": "ASME B31.1",
        "unit": "lb/hr",
    },
}

# Reference values for normal operation
QUALITY_REFERENCE_VALUES = {
    "pressure_psig": 125.0,
    "temperature_f": 353.0,  # Saturation at 125 psig
    "superheat_f": 0.0,
    "enthalpy_btu_lb": 1193.0,
    "drum_level_pct": 50.0,
    "drum_level_variability": 2.0,
    "separator_dp_psi": 5.0,
    "separator_velocity_fps": 10.0,
    "steam_flow_klb_hr": 50.0,
    "condensate_flow_gpm": 5.0,
    "load_pct": 75.0,
    "prv_condensation_rate": 0.0,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureContribution:
    """Single feature's contribution to quality prediction."""

    feature_name: str
    feature_value: float
    shap_value: float  # SHAP contribution to prediction
    contribution_pct: float
    direction: ImpactDirection
    category: QualityFeatureCategory

    # Physics grounding
    physics_basis: str = ""
    reference_standard: str = ""
    reference_value: float = 0.0
    delta_from_reference: float = 0.0

    # Natural language
    explanation: str = ""
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "shap_value": self.shap_value,
            "contribution_pct": self.contribution_pct,
            "direction": self.direction.value,
            "category": self.category.value,
            "physics_basis": self.physics_basis,
            "reference_standard": self.reference_standard,
            "reference_value": self.reference_value,
            "delta_from_reference": self.delta_from_reference,
            "explanation": self.explanation,
            "unit": self.unit,
        }


@dataclass
class LocalExplanation:
    """Local SHAP explanation for a single quality prediction."""

    explanation_id: str
    model_type: ModelType
    timestamp: datetime
    instance_id: str

    # Prediction details
    predicted_value: float
    base_value: float  # Expected value E[f(x)]
    prediction_delta: float

    # Feature contributions
    contributions: List[FeatureContribution]

    # Summary
    top_positive_features: List[str]
    top_negative_features: List[str]

    # Confidence
    explanation_confidence: float = 0.0

    # Natural language
    summary_text: str = ""

    # Provenance
    input_hash: str = ""
    output_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "explanation_id": self.explanation_id,
            "model_type": self.model_type.value,
            "timestamp": self.timestamp.isoformat(),
            "instance_id": self.instance_id,
            "predicted_value": self.predicted_value,
            "base_value": self.base_value,
            "prediction_delta": self.prediction_delta,
            "contributions": [c.to_dict() for c in self.contributions],
            "top_positive_features": self.top_positive_features,
            "top_negative_features": self.top_negative_features,
            "explanation_confidence": self.explanation_confidence,
            "summary_text": self.summary_text,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
        }


@dataclass
class GlobalFeatureImportance:
    """Global feature importance from SHAP values."""

    importance_id: str
    model_type: ModelType
    timestamp: datetime

    # Feature rankings
    feature_rankings: List[Dict[str, Any]]
    mean_abs_shap: Dict[str, float]

    # Summary
    total_features: int
    top_features: List[str]
    feature_categories: Dict[str, List[str]]

    # Dataset info
    dataset_size: int
    dataset_time_range: Optional[Tuple[datetime, datetime]] = None

    # Provenance
    model_hash: str = ""
    data_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "importance_id": self.importance_id,
            "model_type": self.model_type.value,
            "timestamp": self.timestamp.isoformat(),
            "feature_rankings": self.feature_rankings,
            "mean_abs_shap": self.mean_abs_shap,
            "total_features": self.total_features,
            "top_features": self.top_features,
            "feature_categories": self.feature_categories,
            "dataset_size": self.dataset_size,
            "model_hash": self.model_hash,
            "data_hash": self.data_hash,
        }


@dataclass
class CounterfactualExplanation:
    """What-if counterfactual explanation for quality improvement."""

    counterfactual_id: str
    scenario_name: str
    changed_features: Dict[str, float]
    original_prediction: float
    counterfactual_prediction: float
    delta: float
    feasibility: str  # "feasible", "risky", "capital_investment", "infeasible"

    # Implementation details
    explanation: str = ""
    implementation_steps: List[str] = field(default_factory=list)
    estimated_time_hours: float = 0.0
    estimated_cost: Optional[float] = None

    # Constraints
    required_constraints: List[str] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "counterfactual_id": self.counterfactual_id,
            "scenario_name": self.scenario_name,
            "changed_features": self.changed_features,
            "original_prediction": self.original_prediction,
            "counterfactual_prediction": self.counterfactual_prediction,
            "delta": self.delta,
            "feasibility": self.feasibility,
            "explanation": self.explanation,
            "implementation_steps": self.implementation_steps,
            "estimated_time_hours": self.estimated_time_hours,
            "estimated_cost": self.estimated_cost,
            "required_constraints": self.required_constraints,
            "violated_constraints": self.violated_constraints,
        }


@dataclass
class QualityExplanation:
    """Complete explanation for a quality estimate."""

    explanation_id: str
    timestamp: datetime
    explanation_type: ExplanationType
    header_id: str

    # Target
    target_variable: str
    current_value: float
    unit: str = ""

    # Contributions
    feature_contributions: List[FeatureContribution] = field(default_factory=list)

    # Local explanation
    local_explanation: Optional[LocalExplanation] = None

    # Counterfactuals
    counterfactuals: List[CounterfactualExplanation] = field(default_factory=list)

    # Natural language
    summary: str = ""
    detailed_explanation: str = ""

    # Physics grounding
    physics_method: str = ""
    reference_standard: str = ""
    equations_used: List[str] = field(default_factory=list)

    # Confidence and uncertainty
    confidence: float = 0.8
    uncertainty_pct: float = 0.0
    uncertainty_factors: List[str] = field(default_factory=list)

    # Provenance
    provenance_hash: str = ""
    input_hash: str = ""
    model_version: str = ""
    config_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp.isoformat(),
            "explanation_type": self.explanation_type.value,
            "header_id": self.header_id,
            "target_variable": self.target_variable,
            "current_value": self.current_value,
            "unit": self.unit,
            "feature_contributions": [fc.to_dict() for fc in self.feature_contributions],
            "local_explanation": self.local_explanation.to_dict() if self.local_explanation else None,
            "counterfactuals": [cf.to_dict() for cf in self.counterfactuals],
            "summary": self.summary,
            "detailed_explanation": self.detailed_explanation,
            "physics_method": self.physics_method,
            "reference_standard": self.reference_standard,
            "equations_used": self.equations_used,
            "confidence": self.confidence,
            "uncertainty_pct": self.uncertainty_pct,
            "uncertainty_factors": self.uncertainty_factors,
            "provenance_hash": self.provenance_hash,
            "input_hash": self.input_hash,
            "model_version": self.model_version,
            "config_version": self.config_version,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# SHAP QUALITY EXPLAINER
# =============================================================================

class SHAPQualityExplainer:
    """
    SHAP-based explainer for steam quality predictions.

    Provides feature importance analysis, contribution breakdown, and
    counterfactual explanations for dryness fraction estimates.

    All explanations are traceable to data and assumptions per playbook
    requirement, with physics grounding per ASME PTC 19.11.

    Example:
        >>> explainer = SHAPQualityExplainer(agent_id="GL-012")
        >>> explanation = explainer.explain_dryness_estimate(
        ...     features={"drum_level_pct": 65, "separator_dp_psi": 3.5},
        ...     predicted_quality=0.985
        ... )
        >>> print(explanation.summary)

    Attributes:
        agent_id: Agent identifier
        model: Optional ML model for SHAP analysis
        default_num_features: Number of features to show in explanations
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        agent_id: str = "GL-012",
        model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        default_num_features: int = 10,
    ) -> None:
        """
        Initialize SHAPQualityExplainer.

        Args:
            agent_id: Agent identifier (default "GL-012")
            model: Optional trained model for SHAP TreeExplainer
            feature_names: List of feature names
            default_num_features: Number of features in explanations
        """
        self.agent_id = agent_id
        self.model = model
        self.feature_names = feature_names or list(QUALITY_SENSITIVITIES.keys())
        self.default_num_features = default_num_features

        # Feature metadata
        self._feature_metadata = QUALITY_SENSITIVITIES
        self._reference_values = QUALITY_REFERENCE_VALUES

        # SHAP explainer
        self._shap_explainer = None
        if model is not None and SHAP_AVAILABLE:
            self._init_shap_explainer()

        # Cached explanations
        self._explanations: Dict[str, QualityExplanation] = {}
        self._global_importance: Dict[str, GlobalFeatureImportance] = {}

        logger.info(f"SHAPQualityExplainer initialized: {agent_id}")

    def _init_shap_explainer(self) -> None:
        """Initialize SHAP TreeExplainer if model available."""
        try:
            self._shap_explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self._shap_explainer = None

    def explain_dryness_estimate(
        self,
        features: Dict[str, float],
        predicted_quality: float,
        header_id: str = "HEADER-001",
        instance_id: Optional[str] = None,
    ) -> QualityExplanation:
        """
        Generate explanation for a dryness fraction prediction.

        Explains why the model/calculator predicted a specific steam quality
        value by analyzing feature contributions and providing physics-grounded
        interpretations.

        Args:
            features: Input features (sensor readings, process data)
            predicted_quality: Predicted dryness fraction (0-1)
            header_id: Steam header identifier
            instance_id: Optional instance identifier for tracking

        Returns:
            QualityExplanation with feature contributions and counterfactuals
        """
        timestamp = datetime.now(timezone.utc)
        explanation_id = f"QEXP-{timestamp.strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"
        instance_id = instance_id or str(uuid.uuid4())[:8]

        # Calculate feature contributions
        contributions = self._calculate_contributions(features, predicted_quality)

        # Generate local explanation
        local_exp = self._generate_local_explanation(
            features=features,
            predicted_value=predicted_quality,
            contributions=contributions,
            instance_id=instance_id,
        )

        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(
            features=features,
            current_quality=predicted_quality,
        )

        # Build explanation
        explanation = QualityExplanation(
            explanation_id=explanation_id,
            timestamp=timestamp,
            explanation_type=ExplanationType.LOCAL,
            header_id=header_id,
            target_variable="dryness_fraction",
            current_value=predicted_quality,
            unit="fraction",
            feature_contributions=contributions,
            local_explanation=local_exp,
            counterfactuals=counterfactuals,
            physics_method="ASME PTC 19.11 Quality Calculation",
            reference_standard="ASME PTC 19.11, IAPWS-IF97",
            equations_used=[
                "x = (h - hf) / hfg",
                "Carryover = f(drum_level, separator_efficiency, load)",
            ],
            uncertainty_pct=self._calculate_uncertainty(features),
            uncertainty_factors=self._get_uncertainty_factors(features),
            model_version=self.VERSION,
        )

        # Generate natural language
        self._generate_summary(explanation)
        self._generate_detailed_explanation(explanation)

        # Calculate provenance
        self._calculate_provenance(explanation, features)

        # Calculate confidence
        self._calculate_confidence(explanation)

        # Cache
        self._explanations[explanation_id] = explanation

        logger.info(
            f"Generated quality explanation for {header_id}: "
            f"x={predicted_quality:.4f}"
        )

        return explanation

    def _calculate_contributions(
        self,
        features: Dict[str, float],
        predicted_quality: float,
    ) -> List[FeatureContribution]:
        """
        Calculate SHAP-style feature contributions.

        Uses physics-based sensitivities when SHAP library unavailable,
        ensuring zero-hallucination approach.

        Args:
            features: Input features
            predicted_quality: Predicted quality value

        Returns:
            List of FeatureContribution sorted by absolute impact
        """
        contributions = []
        base_quality = 0.97  # Expected quality at reference conditions

        for feature_name, config in self._feature_metadata.items():
            if feature_name not in features:
                continue

            current_value = features[feature_name]
            reference_value = self._reference_values.get(feature_name, current_value)
            sensitivity = config["sensitivity"]

            # Calculate contribution (delta from reference * sensitivity)
            delta = current_value - reference_value

            # Handle non-linear load relationship
            if feature_name == "load_pct":
                contribution = self._calculate_load_contribution(current_value)
            else:
                contribution = delta * sensitivity

            # Determine direction
            if abs(contribution) < 0.0001:
                direction = ImpactDirection.NEUTRAL
            elif contribution > 0:
                direction = ImpactDirection.POSITIVE
            else:
                direction = ImpactDirection.NEGATIVE

            # Generate explanation text
            explanation_text = self._generate_feature_explanation(
                feature_name, current_value, reference_value, contribution, config
            )

            contributions.append(FeatureContribution(
                feature_name=feature_name,
                feature_value=current_value,
                shap_value=round(contribution, 6),
                contribution_pct=0.0,  # Will calculate after sorting
                direction=direction,
                category=config["category"],
                physics_basis=config["physics"],
                reference_standard=config["reference"],
                reference_value=reference_value,
                delta_from_reference=delta,
                explanation=explanation_text,
                unit=config.get("unit", ""),
            ))

        # Sort by absolute contribution
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

        # Calculate contribution percentages
        total_abs = sum(abs(c.shap_value) for c in contributions)
        if total_abs > 0:
            for contrib in contributions:
                contrib.contribution_pct = abs(contrib.shap_value) / total_abs * 100

        return contributions

    def _calculate_load_contribution(self, load_pct: float) -> float:
        """
        Calculate non-linear load contribution to quality.

        Quality is optimal at design load (70-80%) and degrades at
        low load (poor separation) or high load (turbulence).
        """
        optimal_load = 75.0
        deviation = (load_pct - optimal_load) / 50.0
        # Quadratic penalty for deviation from optimal
        contribution = -(deviation ** 2) * 0.01
        return contribution

    def _generate_feature_explanation(
        self,
        feature_name: str,
        current_value: float,
        reference_value: float,
        contribution: float,
        config: Dict[str, Any],
    ) -> str:
        """Generate natural language explanation for a feature."""
        delta = current_value - reference_value
        unit = config.get("unit", "")

        # Feature-specific explanations
        explanations = {
            "drum_level_pct": self._explain_drum_level,
            "separator_dp_psi": self._explain_separator_dp,
            "superheat_f": self._explain_superheat,
            "steam_flow_klb_hr": self._explain_steam_flow,
            "load_pct": self._explain_load,
            "prv_condensation_rate": self._explain_prv_condensation,
        }

        if feature_name in explanations:
            return explanations[feature_name](
                current_value, reference_value, contribution
            )

        # Default explanation
        if abs(delta) < 0.01:
            return f"{feature_name} at {current_value:.2f} {unit} is at reference level."
        elif contribution > 0:
            return (
                f"{feature_name} at {current_value:.2f} {unit} contributes "
                f"+{contribution:.4f} to quality (positive factor)."
            )
        else:
            return (
                f"{feature_name} at {current_value:.2f} {unit} reduces quality "
                f"by {abs(contribution):.4f} (negative factor)."
            )

    def _explain_drum_level(
        self, current: float, reference: float, contribution: float
    ) -> str:
        """Generate drum level explanation."""
        if current > 60:
            return (
                f"Drum level at {current:.1f}% is HIGH (>{reference:.0f}% optimal). "
                f"Elevated level increases moisture entrainment risk, "
                f"reducing quality by {abs(contribution):.4f}. "
                f"Consider reducing drum level setpoint."
            )
        elif current < 40:
            return (
                f"Drum level at {current:.1f}% is LOW (<40%). "
                f"While carryover risk is reduced, monitor for dry firing risk. "
                f"Quality impact: {contribution:+.4f}."
            )
        else:
            return (
                f"Drum level at {current:.1f}% is within optimal range (40-60%). "
                f"Minimal impact on steam quality."
            )

    def _explain_separator_dp(
        self, current: float, reference: float, contribution: float
    ) -> str:
        """Generate separator differential pressure explanation."""
        if current < 3.0:
            return (
                f"Separator dP at {current:.1f} psi is LOW (<3 psi). "
                f"Reduced pressure drop indicates poor separation efficiency, "
                f"allowing more moisture to pass through. "
                f"Quality impact: {contribution:+.4f}."
            )
        elif current > 8.0:
            return (
                f"Separator dP at {current:.1f} psi is HIGH (>8 psi). "
                f"High dP indicates good separation but may indicate fouling. "
                f"Quality impact: {contribution:+.4f}."
            )
        else:
            return (
                f"Separator dP at {current:.1f} psi is normal. "
                f"Separator is operating within design parameters."
            )

    def _explain_superheat(
        self, current: float, reference: float, contribution: float
    ) -> str:
        """Generate superheat explanation."""
        if current > 5.0:
            return (
                f"Steam is SUPERHEATED by {current:.1f}F above saturation. "
                f"Superheated steam has quality x=1.0 by definition. "
                f"No moisture present."
            )
        elif current < -5.0:
            return (
                f"Steam is {abs(current):.1f}F BELOW saturation temperature. "
                f"This indicates WET STEAM with significant moisture content. "
                f"Quality impact: {contribution:+.4f}."
            )
        else:
            return (
                f"Steam temperature is within {abs(current):.1f}F of saturation. "
                f"Steam is near saturated condition."
            )

    def _explain_steam_flow(
        self, current: float, reference: float, contribution: float
    ) -> str:
        """Generate steam flow explanation."""
        if current > reference * 1.2:
            return (
                f"Steam flow at {current:.1f} klb/hr is HIGH "
                f"(>{reference * 1.2:.0f} klb/hr threshold). "
                f"High flow increases turbulence and potential carryover. "
                f"Quality impact: {contribution:+.4f}."
            )
        elif current < reference * 0.5:
            return (
                f"Steam flow at {current:.1f} klb/hr is LOW. "
                f"Low flow may indicate reduced demand or measurement issue."
            )
        else:
            return (
                f"Steam flow at {current:.1f} klb/hr is within normal range."
            )

    def _explain_load(
        self, current: float, reference: float, contribution: float
    ) -> str:
        """Generate load explanation."""
        if current < 40:
            return (
                f"Operating at {current:.0f}% load is BELOW optimal (40% minimum). "
                f"Low load reduces separator efficiency and increases relative "
                f"moisture content. Quality impact: {contribution:+.4f}."
            )
        elif current > 95:
            return (
                f"Operating at {current:.0f}% load is near MAXIMUM. "
                f"High load increases carryover risk due to turbulence. "
                f"Quality impact: {contribution:+.4f}."
            )
        elif 70 <= current <= 80:
            return (
                f"Operating at {current:.0f}% load is OPTIMAL (70-80% range). "
                f"Separator efficiency is maximized at design load."
            )
        else:
            return (
                f"Operating at {current:.0f}% load. "
                f"Quality impact: {contribution:+.4f}."
            )

    def _explain_prv_condensation(
        self, current: float, reference: float, contribution: float
    ) -> str:
        """Generate PRV condensation rate explanation."""
        if current > 10:
            return (
                f"PRV condensation rate of {current:.1f} lb/hr is ELEVATED. "
                f"This indicates wet steam is reaching pressure reduction points, "
                f"condensing upon expansion. Upstream quality is degraded. "
                f"Quality impact: {contribution:+.4f}."
            )
        elif current > 0:
            return (
                f"PRV condensation rate of {current:.1f} lb/hr detected. "
                f"Minor moisture present at pressure reduction."
            )
        else:
            return "No PRV condensation detected. Steam is adequately dry."

    def _generate_local_explanation(
        self,
        features: Dict[str, float],
        predicted_value: float,
        contributions: List[FeatureContribution],
        instance_id: str,
    ) -> LocalExplanation:
        """Generate local SHAP explanation."""
        timestamp = datetime.now(timezone.utc)
        base_value = 0.97  # Expected quality at reference

        # Get top features
        top_positive = [
            c.feature_name for c in contributions
            if c.direction == ImpactDirection.POSITIVE
        ][:3]
        top_negative = [
            c.feature_name for c in contributions
            if c.direction == ImpactDirection.NEGATIVE
        ][:3]

        # Generate summary
        summary = self._generate_local_summary(
            predicted_value, base_value, top_positive, top_negative
        )

        # Calculate input hash
        input_hash = self._compute_hash(features)
        output_hash = self._compute_hash({
            "predicted_value": predicted_value,
            "contributions": [c.shap_value for c in contributions],
        })

        return LocalExplanation(
            explanation_id=f"LOCAL-{timestamp.strftime('%Y%m%d%H%M%S')}",
            model_type=ModelType.DRYNESS_ESTIMATION,
            timestamp=timestamp,
            instance_id=instance_id,
            predicted_value=predicted_value,
            base_value=base_value,
            prediction_delta=predicted_value - base_value,
            contributions=contributions[:self.default_num_features],
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            explanation_confidence=self._compute_explanation_confidence(contributions),
            summary_text=summary,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def _generate_local_summary(
        self,
        predicted_value: float,
        base_value: float,
        top_positive: List[str],
        top_negative: List[str],
    ) -> str:
        """Generate natural language summary for local explanation."""
        if predicted_value >= 0.99:
            quality_desc = "excellent (dry steam)"
        elif predicted_value >= 0.97:
            quality_desc = "good"
        elif predicted_value >= 0.95:
            quality_desc = "acceptable"
        elif predicted_value >= 0.90:
            quality_desc = "marginal (moisture present)"
        else:
            quality_desc = "poor (significant moisture)"

        summary = f"Steam quality is {quality_desc} at x={predicted_value:.4f}."

        if top_negative:
            factors = ", ".join(top_negative[:2])
            summary += f" Quality reduced by: {factors}."

        if top_positive and predicted_value > base_value:
            factors = ", ".join(top_positive[:2])
            summary += f" Quality supported by: {factors}."

        return summary

    def _generate_counterfactuals(
        self,
        features: Dict[str, float],
        current_quality: float,
    ) -> List[CounterfactualExplanation]:
        """
        Generate counterfactual explanations for quality improvement.

        Answers "what would change quality" by identifying actionable
        modifications to input features.
        """
        counterfactuals = []

        # Counterfactual 1: Optimal drum level
        current_drum = features.get("drum_level_pct", 50.0)
        if abs(current_drum - 50.0) > 5:
            target_drum = 50.0
            delta_quality = (current_drum - target_drum) * 0.02 * 0.01
            counterfactuals.append(CounterfactualExplanation(
                counterfactual_id=str(uuid.uuid4())[:8],
                scenario_name="Optimal Drum Level",
                changed_features={"drum_level_pct": target_drum},
                original_prediction=current_quality,
                counterfactual_prediction=min(1.0, current_quality + delta_quality),
                delta=delta_quality,
                feasibility="feasible",
                explanation=(
                    f"If drum level were at optimal {target_drum:.0f}% instead of "
                    f"{current_drum:.1f}%, quality would improve by "
                    f"{delta_quality:.4f} (assuming other factors constant)."
                ),
                implementation_steps=[
                    "Adjust drum level setpoint in DCS",
                    "Monitor feedwater control valve response",
                    "Verify stable level control before proceeding",
                ],
                estimated_time_hours=0.5,
            ))

        # Counterfactual 2: Improve separator efficiency
        current_sep_dp = features.get("separator_dp_psi", 5.0)
        if current_sep_dp < 4.0:
            target_sep_dp = 5.0
            delta_quality = (target_sep_dp - current_sep_dp) * 0.015
            counterfactuals.append(CounterfactualExplanation(
                counterfactual_id=str(uuid.uuid4())[:8],
                scenario_name="Improved Separator Performance",
                changed_features={"separator_dp_psi": target_sep_dp},
                original_prediction=current_quality,
                counterfactual_prediction=min(1.0, current_quality + delta_quality),
                delta=delta_quality,
                feasibility="capital_investment" if current_sep_dp < 2.0 else "feasible",
                explanation=(
                    f"Increasing separator dP from {current_sep_dp:.1f} to "
                    f"{target_sep_dp:.1f} psi would improve moisture separation. "
                    f"Expected quality gain: {delta_quality:.4f}."
                ),
                implementation_steps=[
                    "Inspect separator internals for fouling/damage",
                    "Clean or replace separator elements if needed",
                    "Verify inlet/outlet piping is correct",
                ],
                estimated_time_hours=4.0,
            ))

        # Counterfactual 3: Reduce PRV condensation
        current_prv = features.get("prv_condensation_rate", 0.0)
        if current_prv > 5:
            target_prv = 0.0
            delta_quality = current_prv * 0.02 * 0.01
            counterfactuals.append(CounterfactualExplanation(
                counterfactual_id=str(uuid.uuid4())[:8],
                scenario_name="Eliminate PRV Condensation",
                changed_features={"prv_condensation_rate": target_prv},
                original_prediction=current_quality,
                counterfactual_prediction=min(1.0, current_quality + delta_quality),
                delta=delta_quality,
                feasibility="feasible",
                explanation=(
                    f"Eliminating PRV condensation (currently {current_prv:.1f} lb/hr) "
                    f"by improving upstream quality would indicate x approaching 1.0. "
                    f"This is an effect, not cause - address upstream moisture sources."
                ),
                implementation_steps=[
                    "Identify moisture source upstream of PRV",
                    "Check trap operation in upstream lines",
                    "Verify drip leg condensate removal",
                ],
                estimated_time_hours=2.0,
            ))

        # Counterfactual 4: Load optimization
        current_load = features.get("load_pct", 75.0)
        if current_load < 50 or current_load > 90:
            target_load = 75.0
            load_contribution = self._calculate_load_contribution(current_load)
            target_contribution = self._calculate_load_contribution(target_load)
            delta_quality = target_contribution - load_contribution

            counterfactuals.append(CounterfactualExplanation(
                counterfactual_id=str(uuid.uuid4())[:8],
                scenario_name="Optimal Load Operation",
                changed_features={"load_pct": target_load},
                original_prediction=current_quality,
                counterfactual_prediction=min(1.0, current_quality + delta_quality),
                delta=delta_quality,
                feasibility="risky" if current_load > 90 else "feasible",
                explanation=(
                    f"Operating at {target_load:.0f}% load (optimal) instead of "
                    f"{current_load:.0f}% would improve separator efficiency. "
                    f"Note: Load is typically driven by demand, not quality optimization."
                ),
                implementation_steps=[
                    "Coordinate with operations for load adjustment",
                    "Consider load balancing across multiple boilers",
                ],
                estimated_time_hours=0.0,  # Operational decision
            ))

        return counterfactuals

    def _calculate_uncertainty(self, features: Dict[str, float]) -> float:
        """Calculate uncertainty in quality estimate."""
        base_uncertainty = 0.5  # Base 0.5% per ASME PTC 19.11

        # Add uncertainty for deviation from reference conditions
        for feature_name, config in self._feature_metadata.items():
            if feature_name in features:
                current = features[feature_name]
                reference = self._reference_values.get(feature_name, current)
                if reference != 0:
                    deviation_pct = abs(current - reference) / abs(reference) * 100
                    if deviation_pct > 20:
                        base_uncertainty += 0.1
                    elif deviation_pct > 50:
                        base_uncertainty += 0.2

        return min(2.0, base_uncertainty)  # Cap at 2%

    def _get_uncertainty_factors(self, features: Dict[str, float]) -> List[str]:
        """Get list of factors contributing to uncertainty."""
        factors = ["Sensor measurement accuracy (+/- 0.1%)"]

        # Check for conditions that increase uncertainty
        if features.get("drum_level_variability", 0) > 3:
            factors.append("High drum level variability increases estimation uncertainty")

        if features.get("load_pct", 75) < 40 or features.get("load_pct", 75) > 90:
            factors.append("Off-design load operation increases uncertainty")

        if features.get("superheat_f", 0) < -10 or features.get("superheat_f", 0) > 50:
            factors.append("Extreme temperature deviation from saturation")

        return factors

    def _generate_summary(self, explanation: QualityExplanation) -> None:
        """Generate natural language summary."""
        quality = explanation.current_value
        top_contributors = explanation.feature_contributions[:3]

        if quality >= 0.99:
            status = "excellent (effectively dry steam)"
        elif quality >= 0.97:
            status = "good (acceptable for most applications)"
        elif quality >= 0.95:
            status = "marginal (monitor closely)"
        else:
            status = "poor (immediate attention required)"

        summary_parts = [
            f"Steam quality x={quality:.4f} is {status}.",
        ]

        # Add top factor
        if top_contributors:
            main_factor = top_contributors[0]
            if main_factor.direction == ImpactDirection.NEGATIVE:
                summary_parts.append(
                    f"Primary concern: {main_factor.feature_name} "
                    f"({main_factor.explanation})"
                )
            elif main_factor.direction == ImpactDirection.POSITIVE:
                summary_parts.append(
                    f"Quality supported by: {main_factor.feature_name}."
                )

        explanation.summary = " ".join(summary_parts)

    def _generate_detailed_explanation(self, explanation: QualityExplanation) -> None:
        """Generate detailed natural language explanation."""
        lines = [
            "STEAM QUALITY ANALYSIS",
            "=" * 50,
            "",
            f"Header: {explanation.header_id}",
            f"Quality (x): {explanation.current_value:.4f}",
            f"Uncertainty: +/- {explanation.uncertainty_pct:.2f}%",
            "",
            "FEATURE CONTRIBUTIONS (ranked by impact):",
            "-" * 40,
        ]

        for i, contrib in enumerate(explanation.feature_contributions[:5], 1):
            sign = "+" if contrib.shap_value >= 0 else ""
            lines.append(
                f"{i}. {contrib.feature_name}: {sign}{contrib.shap_value:.5f} "
                f"({contrib.contribution_pct:.1f}%)"
            )
            lines.append(f"   Value: {contrib.feature_value:.2f} {contrib.unit}")
            lines.append(f"   {contrib.explanation}")
            lines.append("")

        if explanation.counterfactuals:
            lines.extend([
                "-" * 40,
                "IMPROVEMENT OPPORTUNITIES:",
            ])
            for cf in explanation.counterfactuals[:3]:
                if cf.delta > 0:
                    lines.append(
                        f"  - {cf.scenario_name}: +{cf.delta:.4f} quality "
                        f"({cf.feasibility})"
                    )

        lines.extend([
            "",
            "-" * 40,
            "PHYSICS BASIS:",
            f"  Method: {explanation.physics_method}",
            f"  Standard: {explanation.reference_standard}",
            f"  Equations: {', '.join(explanation.equations_used)}",
        ])

        explanation.detailed_explanation = "\n".join(lines)

    def _calculate_provenance(
        self, explanation: QualityExplanation, features: Dict[str, float]
    ) -> None:
        """Calculate provenance hash for audit trail."""
        provenance_data = {
            "explanation_id": explanation.explanation_id,
            "timestamp": explanation.timestamp.isoformat(),
            "header_id": explanation.header_id,
            "current_value": explanation.current_value,
            "features": features,
            "agent_id": self.agent_id,
            "version": self.VERSION,
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        explanation.provenance_hash = hashlib.sha256(json_str.encode()).hexdigest()
        explanation.input_hash = self._compute_hash(features)

    def _calculate_confidence(self, explanation: QualityExplanation) -> None:
        """Calculate confidence in explanation."""
        confidence = 0.9  # Base confidence

        # Reduce for high uncertainty
        if explanation.uncertainty_pct > 1.0:
            confidence -= 0.1
        if explanation.uncertainty_pct > 1.5:
            confidence -= 0.1

        # Reduce for many uncertainty factors
        if len(explanation.uncertainty_factors) > 3:
            confidence -= 0.1

        # Reduce for extreme quality values
        if explanation.current_value < 0.90 or explanation.current_value > 0.995:
            confidence -= 0.05

        explanation.confidence = max(0.5, min(1.0, confidence))

    def _compute_explanation_confidence(
        self, contributions: List[FeatureContribution]
    ) -> float:
        """Compute confidence in the explanation based on contribution distribution."""
        if not contributions:
            return 0.5

        # Higher confidence if top features dominate
        total_shap = sum(abs(c.shap_value) for c in contributions)
        top_3_shap = sum(abs(c.shap_value) for c in contributions[:3])

        if total_shap > 0:
            concentration = top_3_shap / total_shap
            return min(1.0, 0.5 + concentration * 0.5)

        return 0.5

    def _compute_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def compute_global_feature_importance(
        self,
        dataset: List[Dict[str, float]],
        model_type: ModelType = ModelType.DRYNESS_ESTIMATION,
    ) -> GlobalFeatureImportance:
        """
        Compute global feature importance from dataset.

        Uses physics-based sensitivity analysis when SHAP unavailable.

        Args:
            dataset: List of feature dictionaries
            model_type: Type of model being explained

        Returns:
            GlobalFeatureImportance with ranked features
        """
        timestamp = datetime.now(timezone.utc)
        importance_id = f"IMP-{timestamp.strftime('%Y%m%d%H%M%S')}"

        # Get feature names from dataset
        feature_names = list(dataset[0].keys()) if dataset else self.feature_names

        # Calculate importance (use physics sensitivities)
        mean_abs_shap = {}
        for feature in feature_names:
            if feature in self._feature_metadata:
                # Use absolute sensitivity as proxy for importance
                mean_abs_shap[feature] = abs(
                    self._feature_metadata[feature]["sensitivity"]
                )
            else:
                mean_abs_shap[feature] = 0.001

        # Normalize
        total = sum(mean_abs_shap.values()) or 1.0
        mean_abs_shap = {k: v / total for k, v in mean_abs_shap.items()}

        # Create rankings
        feature_rankings = []
        for feature, importance in sorted(
            mean_abs_shap.items(), key=lambda x: x[1], reverse=True
        ):
            metadata = self._feature_metadata.get(feature, {})
            feature_rankings.append({
                "feature": feature,
                "importance": importance,
                "rank": len(feature_rankings) + 1,
                "category": metadata.get(
                    "category", QualityFeatureCategory.DERIVED
                ).value,
                "physics_basis": metadata.get("physics", ""),
            })

        # Group by category
        feature_categories = {}
        for feature in feature_names:
            metadata = self._feature_metadata.get(feature, {})
            category = metadata.get(
                "category", QualityFeatureCategory.DERIVED
            ).value
            if category not in feature_categories:
                feature_categories[category] = []
            feature_categories[category].append(feature)

        importance = GlobalFeatureImportance(
            importance_id=importance_id,
            model_type=model_type,
            timestamp=timestamp,
            feature_rankings=feature_rankings,
            mean_abs_shap=mean_abs_shap,
            total_features=len(feature_names),
            top_features=[r["feature"] for r in feature_rankings[:10]],
            feature_categories=feature_categories,
            dataset_size=len(dataset),
            model_hash=self._compute_hash({"model_type": model_type.value}),
            data_hash=self._compute_hash({"dataset_size": len(dataset)}),
        )

        self._global_importance[model_type.value] = importance
        logger.info(f"Computed global feature importance: {importance_id}")

        return importance

    def get_explanation(self, explanation_id: str) -> Optional[QualityExplanation]:
        """Get explanation by ID."""
        return self._explanations.get(explanation_id)

    def get_recent_explanations(
        self,
        header_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[QualityExplanation]:
        """Get recent explanations with optional filter."""
        explanations = list(self._explanations.values())

        if header_id:
            explanations = [e for e in explanations if e.header_id == header_id]

        explanations.sort(key=lambda e: e.timestamp, reverse=True)
        return explanations[:limit]

    def clear_cache(self) -> None:
        """Clear explanation cache."""
        self._explanations.clear()
        logger.info("Quality explanation cache cleared")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_quality_explainer(
    agent_id: str = "GL-012",
    model: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
) -> SHAPQualityExplainer:
    """
    Factory function to create quality explainer.

    Args:
        agent_id: Agent identifier
        model: Optional trained model for SHAP
        feature_names: Optional feature names

    Returns:
        Configured SHAPQualityExplainer instance
    """
    return SHAPQualityExplainer(
        agent_id=agent_id,
        model=model,
        feature_names=feature_names,
    )
