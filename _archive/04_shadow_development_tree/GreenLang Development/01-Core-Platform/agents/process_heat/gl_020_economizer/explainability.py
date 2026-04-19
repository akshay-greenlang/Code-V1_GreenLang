"""
GL-020 ECONOPULSE - SHAP Explainability Module

Provides SHAP (SHapley Additive exPlanations) feature analysis for
economizer optimization decisions and predictions.

Features:
    - Feature importance analysis for fouling predictions
    - SHAP value calculation for individual predictions
    - Feature interaction analysis
    - NFPA 86 compliance explanations
    - Audit-ready provenance tracking

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units
    - NFPA 86 Standard for Ovens and Furnaces

Zero-Hallucination: All explanations are derived from deterministic calculations.
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Feature categories for grouping
FEATURE_CATEGORIES = {
    "temperature": [
        "gas_inlet_temp_f",
        "gas_outlet_temp_f",
        "water_inlet_temp_f",
        "water_outlet_temp_f",
        "cold_end_metal_temp_f",
        "ambient_temp_f",
    ],
    "flow": [
        "gas_flow_lb_hr",
        "water_flow_lb_hr",
        "load_pct",
    ],
    "pressure": [
        "gas_side_dp_in_wc",
        "water_side_dp_psi",
        "drum_pressure_psig",
    ],
    "chemistry": [
        "flue_gas_o2_pct",
        "flue_gas_so2_ppm",
        "flue_gas_moisture_pct",
        "fuel_sulfur_pct",
        "feedwater_ph",
        "feedwater_hardness_ppm",
    ],
    "heat_transfer": [
        "effectiveness_ratio",
        "ua_degradation_pct",
        "lmtd_f",
        "ntu",
    ],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

class ExplanationType(str, Enum):
    """Types of explanations."""
    FOULING_PREDICTION = "fouling_prediction"
    SOOT_BLOWING = "soot_blowing"
    ACID_DEW_POINT = "acid_dew_point"
    EFFECTIVENESS = "effectiveness"
    STEAMING_RISK = "steaming_risk"
    CLEANING_RECOMMENDATION = "cleaning_recommendation"


@dataclass
class FeatureContribution:
    """Individual feature contribution to a prediction."""
    feature_name: str
    feature_value: float
    shap_value: float
    contribution_pct: float
    direction: str  # "positive", "negative", "neutral"
    category: str
    description: str


@dataclass
class SHAPExplanation:
    """Complete SHAP explanation for a prediction."""
    explanation_id: str
    explanation_type: ExplanationType
    timestamp: datetime

    # Prediction info
    predicted_value: float
    predicted_class: Optional[str]
    base_value: float

    # Feature contributions
    feature_contributions: List[FeatureContribution]
    top_positive_features: List[str]
    top_negative_features: List[str]

    # Summary
    summary_text: str
    confidence_score: float

    # Compliance
    nfpa_86_relevant: bool
    nfpa_86_compliance_notes: List[str]

    # Provenance
    calculation_method: str
    provenance_hash: str


@dataclass
class FeatureInteraction:
    """Feature interaction effect."""
    feature_1: str
    feature_2: str
    interaction_value: float
    interaction_type: str  # "synergistic", "antagonistic", "additive"
    description: str


@dataclass
class GlobalFeatureImportance:
    """Global feature importance summary."""
    feature_name: str
    mean_abs_shap: float
    std_shap: float
    importance_rank: int
    category: str
    description: str


# =============================================================================
# SHAP FEATURE ANALYZER
# =============================================================================

class SHAPFeatureAnalyzer:
    """
    SHAP Feature Analyzer for GL-020 ECONOPULSE.

    Provides explainability for economizer optimization decisions using
    SHAP (SHapley Additive exPlanations) methodology adapted for
    deterministic industrial calculations.

    Unlike ML-based SHAP, this analyzer computes feature contributions
    based on the physical relationships in economizer performance:
    - Temperature differentials affect heat transfer
    - Pressure drops indicate fouling
    - Flow rates impact effectiveness
    - Chemistry affects corrosion risk

    NFPA 86 Compliance:
        The analyzer tracks features relevant to NFPA 86 safety requirements
        for industrial heating equipment, including:
        - Temperature limits
        - Flow safety margins
        - Steaming prevention
        - Corrosion protection
    """

    # Physical sensitivity coefficients for SHAP-like calculations
    # These represent typical partial derivatives of outputs with respect to inputs
    SENSITIVITY_COEFFICIENTS = {
        # Temperature sensitivities (effect on effectiveness)
        "gas_inlet_temp_f": 0.002,  # Higher inlet = more heat available
        "gas_outlet_temp_f": -0.003,  # Higher outlet = less heat transferred
        "water_inlet_temp_f": -0.001,  # Higher inlet = smaller LMTD
        "water_outlet_temp_f": 0.001,  # Higher outlet = more heat absorbed

        # Flow sensitivities
        "gas_flow_lb_hr": 0.00001,  # More flow = more heat capacity
        "water_flow_lb_hr": 0.00001,  # More flow = more heat capacity
        "load_pct": 0.005,  # Higher load = more heat transfer

        # Pressure drop sensitivities (effect on fouling score)
        "gas_side_dp_in_wc": 0.2,  # Higher DP = more fouling
        "water_side_dp_psi": 0.15,  # Higher DP = more scaling

        # Chemistry sensitivities
        "flue_gas_so2_ppm": 0.01,  # More SO2 = higher acid dew point
        "flue_gas_moisture_pct": 0.05,  # More moisture = higher dew point
        "fuel_sulfur_pct": 0.5,  # More sulfur = higher corrosion risk

        # Effectiveness sensitivities
        "effectiveness_ratio": 1.0,  # Direct indicator
        "ua_degradation_pct": -0.01,  # Degradation reduces performance
    }

    # NFPA 86 relevant features
    NFPA_86_FEATURES = [
        "gas_inlet_temp_f",
        "gas_outlet_temp_f",
        "water_outlet_temp_f",
        "water_flow_lb_hr",
        "gas_flow_lb_hr",
        "drum_pressure_psig",
        "load_pct",
    ]

    def __init__(
        self,
        baseline_values: Optional[Dict[str, float]] = None,
        track_history: bool = True,
        max_history_size: int = 1000,
    ):
        """
        Initialize SHAP Feature Analyzer.

        Args:
            baseline_values: Baseline feature values for SHAP calculation
            track_history: Whether to track explanation history
            max_history_size: Maximum history size
        """
        self.baseline_values = baseline_values or self._default_baseline()
        self.track_history = track_history
        self.max_history_size = max_history_size
        self._history: List[SHAPExplanation] = []
        self._feature_importance_cache: Dict[str, float] = {}

        logger.info("SHAPFeatureAnalyzer initialized")

    def _default_baseline(self) -> Dict[str, float]:
        """Get default baseline values representing normal operation."""
        return {
            "gas_inlet_temp_f": 600.0,
            "gas_outlet_temp_f": 350.0,
            "water_inlet_temp_f": 250.0,
            "water_outlet_temp_f": 350.0,
            "cold_end_metal_temp_f": 300.0,
            "gas_flow_lb_hr": 100000.0,
            "water_flow_lb_hr": 80000.0,
            "load_pct": 75.0,
            "gas_side_dp_in_wc": 2.0,
            "water_side_dp_psi": 5.0,
            "drum_pressure_psig": 500.0,
            "flue_gas_o2_pct": 3.0,
            "flue_gas_so2_ppm": 10.0,
            "flue_gas_moisture_pct": 10.0,
            "fuel_sulfur_pct": 0.01,
            "effectiveness_ratio": 1.0,
            "ua_degradation_pct": 0.0,
        }

    def _get_feature_category(self, feature_name: str) -> str:
        """Get category for a feature."""
        for category, features in FEATURE_CATEGORIES.items():
            if feature_name in features:
                return category
        return "other"

    def _calculate_shap_value(
        self,
        feature_name: str,
        current_value: float,
        baseline_value: float,
    ) -> float:
        """
        Calculate SHAP-like value for a feature.

        Uses physical sensitivity coefficients to compute the contribution
        of a feature deviation from baseline to the output.

        Args:
            feature_name: Feature name
            current_value: Current feature value
            baseline_value: Baseline feature value

        Returns:
            SHAP value (contribution to output deviation)
        """
        sensitivity = self.SENSITIVITY_COEFFICIENTS.get(feature_name, 0.0)
        deviation = current_value - baseline_value

        # Apply sensitivity coefficient
        shap_value = sensitivity * deviation

        return shap_value

    def _generate_feature_description(
        self,
        feature_name: str,
        current_value: float,
        baseline_value: float,
        shap_value: float,
    ) -> str:
        """Generate human-readable description of feature contribution."""
        deviation = current_value - baseline_value
        deviation_pct = abs(deviation / baseline_value * 100) if baseline_value != 0 else 0

        direction = "above" if deviation > 0 else "below"
        impact = "increases" if shap_value > 0 else "decreases"

        descriptions = {
            "gas_inlet_temp_f": f"Gas inlet temperature is {deviation_pct:.1f}% {direction} baseline, which {impact} heat availability",
            "gas_outlet_temp_f": f"Gas outlet temperature is {deviation_pct:.1f}% {direction} baseline, indicating {'reduced' if deviation > 0 else 'improved'} heat transfer",
            "water_inlet_temp_f": f"Water inlet temperature is {deviation_pct:.1f}% {direction} baseline, affecting LMTD",
            "water_outlet_temp_f": f"Water outlet temperature is {deviation_pct:.1f}% {direction} baseline, indicating heat absorption",
            "gas_flow_lb_hr": f"Gas flow is {deviation_pct:.1f}% {direction} design, affecting heat capacity rate",
            "water_flow_lb_hr": f"Water flow is {deviation_pct:.1f}% {direction} design, affecting heat removal",
            "load_pct": f"Operating load is {deviation_pct:.1f}% {direction} baseline",
            "gas_side_dp_in_wc": f"Gas-side pressure drop is {deviation_pct:.1f}% {direction} design, indicating {'fouling' if deviation > 0 else 'clean tubes'}",
            "water_side_dp_psi": f"Water-side pressure drop is {deviation_pct:.1f}% {direction} design, indicating {'scaling' if deviation > 0 else 'clean tubes'}",
            "flue_gas_so2_ppm": f"SO2 concentration is {deviation_pct:.1f}% {direction} baseline, affecting acid dew point",
            "fuel_sulfur_pct": f"Fuel sulfur is {deviation_pct:.1f}% {direction} baseline, affecting corrosion risk",
            "effectiveness_ratio": f"Heat transfer effectiveness is {deviation_pct:.1f}% {direction} design",
        }

        return descriptions.get(
            feature_name,
            f"{feature_name} is {deviation_pct:.1f}% {direction} baseline"
        )

    def explain_prediction(
        self,
        features: Dict[str, float],
        prediction: float,
        prediction_type: ExplanationType,
        predicted_class: Optional[str] = None,
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for a prediction.

        Args:
            features: Input feature values
            prediction: Model/calculation prediction value
            prediction_type: Type of prediction being explained
            predicted_class: Optional class label for classification

        Returns:
            SHAPExplanation with feature contributions
        """
        explanation_id = hashlib.sha256(
            json.dumps({
                "features": features,
                "prediction": prediction,
                "type": prediction_type.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, sort_keys=True).encode()
        ).hexdigest()[:12]

        # Calculate SHAP values for each feature
        feature_contributions = []
        total_abs_shap = 0.0

        for feature_name, current_value in features.items():
            baseline_value = self.baseline_values.get(feature_name, current_value)
            shap_value = self._calculate_shap_value(
                feature_name, current_value, baseline_value
            )

            total_abs_shap += abs(shap_value)

            feature_contributions.append({
                "feature_name": feature_name,
                "current_value": current_value,
                "baseline_value": baseline_value,
                "shap_value": shap_value,
            })

        # Calculate contribution percentages and create FeatureContribution objects
        contributions = []
        for fc in feature_contributions:
            contribution_pct = (
                abs(fc["shap_value"]) / total_abs_shap * 100
                if total_abs_shap > 0 else 0.0
            )

            if fc["shap_value"] > 0.001:
                direction = "positive"
            elif fc["shap_value"] < -0.001:
                direction = "negative"
            else:
                direction = "neutral"

            contributions.append(FeatureContribution(
                feature_name=fc["feature_name"],
                feature_value=fc["current_value"],
                shap_value=round(fc["shap_value"], 6),
                contribution_pct=round(contribution_pct, 2),
                direction=direction,
                category=self._get_feature_category(fc["feature_name"]),
                description=self._generate_feature_description(
                    fc["feature_name"],
                    fc["current_value"],
                    fc["baseline_value"],
                    fc["shap_value"],
                ),
            ))

        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)

        # Get top positive and negative features
        top_positive = [c.feature_name for c in contributions if c.direction == "positive"][:3]
        top_negative = [c.feature_name for c in contributions if c.direction == "negative"][:3]

        # Calculate base value (expected prediction at baseline)
        base_value = self._calculate_base_value(prediction_type)

        # Generate summary text
        summary_text = self._generate_summary(
            prediction_type, prediction, contributions[:5]
        )

        # Check NFPA 86 relevance
        nfpa_features = [c for c in contributions if c.feature_name in self.NFPA_86_FEATURES]
        nfpa_relevant = len(nfpa_features) > 0
        nfpa_notes = self._generate_nfpa_notes(nfpa_features, prediction_type)

        # Calculate confidence score based on feature coverage
        confidence_score = min(1.0, len(features) / 10.0)

        # Create provenance hash
        provenance_hash = hashlib.sha256(
            json.dumps({
                "features": features,
                "shap_values": [c.shap_value for c in contributions],
                "prediction": prediction,
            }, sort_keys=True).encode()
        ).hexdigest()[:16]

        explanation = SHAPExplanation(
            explanation_id=explanation_id,
            explanation_type=prediction_type,
            timestamp=datetime.now(timezone.utc),
            predicted_value=prediction,
            predicted_class=predicted_class,
            base_value=base_value,
            feature_contributions=contributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            summary_text=summary_text,
            confidence_score=confidence_score,
            nfpa_86_relevant=nfpa_relevant,
            nfpa_86_compliance_notes=nfpa_notes,
            calculation_method="SHAP_PHYSICAL_SENSITIVITY",
            provenance_hash=provenance_hash,
        )

        # Track history
        if self.track_history:
            self._history.append(explanation)
            if len(self._history) > self.max_history_size:
                self._history = self._history[-self.max_history_size:]

        return explanation

    def _calculate_base_value(self, prediction_type: ExplanationType) -> float:
        """Calculate expected value at baseline for prediction type."""
        base_values = {
            ExplanationType.FOULING_PREDICTION: 0.0,  # No fouling at baseline
            ExplanationType.SOOT_BLOWING: 0.5,  # Neutral recommendation
            ExplanationType.ACID_DEW_POINT: 270.0,  # Typical acid dew point
            ExplanationType.EFFECTIVENESS: 1.0,  # Design effectiveness
            ExplanationType.STEAMING_RISK: 0.0,  # No risk at design conditions
            ExplanationType.CLEANING_RECOMMENDATION: 0.0,  # No cleaning needed
        }
        return base_values.get(prediction_type, 0.5)

    def _generate_summary(
        self,
        prediction_type: ExplanationType,
        prediction: float,
        top_contributions: List[FeatureContribution],
    ) -> str:
        """Generate summary text for explanation."""
        if not top_contributions:
            return f"{prediction_type.value} prediction: {prediction:.3f}"

        top_feature = top_contributions[0]

        summaries = {
            ExplanationType.FOULING_PREDICTION: (
                f"Fouling score of {prediction:.2f} is primarily driven by "
                f"{top_feature.feature_name} ({top_feature.direction} contribution of "
                f"{top_feature.contribution_pct:.1f}%). {top_feature.description}"
            ),
            ExplanationType.SOOT_BLOWING: (
                f"Soot blowing recommendation score: {prediction:.2f}. "
                f"Main factor: {top_feature.feature_name} "
                f"(contributing {top_feature.contribution_pct:.1f}%)"
            ),
            ExplanationType.ACID_DEW_POINT: (
                f"Acid dew point of {prediction:.1f}F is influenced by "
                f"{top_feature.feature_name} ({top_feature.contribution_pct:.1f}% contribution)"
            ),
            ExplanationType.EFFECTIVENESS: (
                f"Heat transfer effectiveness of {prediction:.2%} is primarily "
                f"affected by {top_feature.feature_name}"
            ),
            ExplanationType.STEAMING_RISK: (
                f"Steaming risk score: {prediction:.1f}. Key factor: {top_feature.feature_name}"
            ),
            ExplanationType.CLEANING_RECOMMENDATION: (
                f"Cleaning recommendation score: {prediction:.2f}. "
                f"Primary driver: {top_feature.feature_name}"
            ),
        }

        return summaries.get(
            prediction_type,
            f"Prediction: {prediction:.3f}. Top factor: {top_feature.feature_name}"
        )

    def _generate_nfpa_notes(
        self,
        nfpa_features: List[FeatureContribution],
        prediction_type: ExplanationType,
    ) -> List[str]:
        """Generate NFPA 86 compliance notes."""
        notes = []

        for feature in nfpa_features:
            if feature.feature_name == "water_flow_lb_hr" and feature.direction == "negative":
                notes.append(
                    "NFPA 86 4.3.1: Low water flow may indicate insufficient cooling. "
                    "Verify minimum flow requirements are met."
                )
            elif feature.feature_name == "gas_inlet_temp_f" and feature.direction == "positive":
                notes.append(
                    "NFPA 86 4.2.2: Elevated gas temperatures detected. "
                    "Verify temperature limits per equipment rating."
                )
            elif feature.feature_name == "water_outlet_temp_f" and feature.direction == "positive":
                notes.append(
                    "NFPA 86 4.3.2: High water outlet temperature may indicate "
                    "approach to steaming conditions. Monitor closely."
                )
            elif feature.feature_name == "drum_pressure_psig" and feature.direction == "negative":
                notes.append(
                    "NFPA 86 4.4.1: Low drum pressure may affect saturation margin. "
                    "Verify operating pressure is within design limits."
                )

        if prediction_type == ExplanationType.STEAMING_RISK and nfpa_features:
            notes.append(
                "NFPA 86 Chapter 9: Steaming economizers pose water hammer risk. "
                "Ensure adequate water flow and temperature margins."
            )

        return notes

    def analyze_feature_interactions(
        self,
        features: Dict[str, float],
        interaction_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> List[FeatureInteraction]:
        """
        Analyze feature interactions.

        Args:
            features: Feature values
            interaction_pairs: Optional specific pairs to analyze

        Returns:
            List of FeatureInteraction objects
        """
        interactions = []

        # Default interaction pairs based on physical relationships
        if interaction_pairs is None:
            interaction_pairs = [
                ("gas_inlet_temp_f", "gas_outlet_temp_f"),  # Temperature drop
                ("water_inlet_temp_f", "water_outlet_temp_f"),  # Temperature rise
                ("gas_flow_lb_hr", "gas_side_dp_in_wc"),  # Flow vs DP
                ("water_flow_lb_hr", "water_side_dp_psi"),  # Flow vs DP
                ("flue_gas_so2_ppm", "cold_end_metal_temp_f"),  # Corrosion interaction
                ("load_pct", "effectiveness_ratio"),  # Load vs performance
            ]

        for feature_1, feature_2 in interaction_pairs:
            if feature_1 in features and feature_2 in features:
                interaction = self._calculate_interaction(
                    feature_1, features[feature_1],
                    feature_2, features[feature_2],
                )
                if interaction is not None:
                    interactions.append(interaction)

        return interactions

    def _calculate_interaction(
        self,
        feature_1: str,
        value_1: float,
        feature_2: str,
        value_2: float,
    ) -> Optional[FeatureInteraction]:
        """Calculate interaction between two features."""
        baseline_1 = self.baseline_values.get(feature_1, value_1)
        baseline_2 = self.baseline_values.get(feature_2, value_2)

        # Normalized deviations
        dev_1 = (value_1 - baseline_1) / baseline_1 if baseline_1 != 0 else 0
        dev_2 = (value_2 - baseline_2) / baseline_2 if baseline_2 != 0 else 0

        # Simple interaction model: product of deviations
        interaction_value = dev_1 * dev_2

        # Determine interaction type
        if dev_1 * dev_2 > 0.01:
            interaction_type = "synergistic"
        elif dev_1 * dev_2 < -0.01:
            interaction_type = "antagonistic"
        else:
            interaction_type = "additive"

        # Generate description
        descriptions = {
            ("gas_inlet_temp_f", "gas_outlet_temp_f"):
                "Temperature drop across economizer indicates heat transfer effectiveness",
            ("water_inlet_temp_f", "water_outlet_temp_f"):
                "Water temperature rise indicates heat absorption rate",
            ("gas_flow_lb_hr", "gas_side_dp_in_wc"):
                "Flow-DP relationship indicates gas-side fouling or restriction",
            ("water_flow_lb_hr", "water_side_dp_psi"):
                "Flow-DP relationship indicates water-side scaling or restriction",
            ("flue_gas_so2_ppm", "cold_end_metal_temp_f"):
                "SO2-temperature interaction determines acid corrosion risk",
            ("load_pct", "effectiveness_ratio"):
                "Load-effectiveness relationship indicates off-design performance",
        }

        description = descriptions.get(
            (feature_1, feature_2),
            f"Interaction between {feature_1} and {feature_2}"
        )

        return FeatureInteraction(
            feature_1=feature_1,
            feature_2=feature_2,
            interaction_value=round(interaction_value, 6),
            interaction_type=interaction_type,
            description=description,
        )

    def get_global_feature_importance(
        self,
        history_limit: Optional[int] = None,
    ) -> List[GlobalFeatureImportance]:
        """
        Calculate global feature importance from explanation history.

        Args:
            history_limit: Optional limit on history to analyze

        Returns:
            List of GlobalFeatureImportance sorted by importance
        """
        if not self._history:
            # Return default importance based on sensitivity coefficients
            return self._default_global_importance()

        history = self._history
        if history_limit:
            history = history[-history_limit:]

        # Aggregate SHAP values by feature
        feature_shaps: Dict[str, List[float]] = {}

        for explanation in history:
            for contribution in explanation.feature_contributions:
                if contribution.feature_name not in feature_shaps:
                    feature_shaps[contribution.feature_name] = []
                feature_shaps[contribution.feature_name].append(
                    abs(contribution.shap_value)
                )

        # Calculate statistics
        importance_list = []
        for feature_name, shap_values in feature_shaps.items():
            mean_abs = sum(shap_values) / len(shap_values)
            std = (
                math.sqrt(sum((x - mean_abs) ** 2 for x in shap_values) / len(shap_values))
                if len(shap_values) > 1 else 0.0
            )

            importance_list.append({
                "feature_name": feature_name,
                "mean_abs_shap": mean_abs,
                "std_shap": std,
            })

        # Sort by importance
        importance_list.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

        # Create GlobalFeatureImportance objects
        result = []
        for rank, item in enumerate(importance_list, 1):
            result.append(GlobalFeatureImportance(
                feature_name=item["feature_name"],
                mean_abs_shap=round(item["mean_abs_shap"], 6),
                std_shap=round(item["std_shap"], 6),
                importance_rank=rank,
                category=self._get_feature_category(item["feature_name"]),
                description=self._get_feature_importance_description(item["feature_name"]),
            ))

        return result

    def _default_global_importance(self) -> List[GlobalFeatureImportance]:
        """Get default global importance based on sensitivity coefficients."""
        sorted_features = sorted(
            self.SENSITIVITY_COEFFICIENTS.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return [
            GlobalFeatureImportance(
                feature_name=name,
                mean_abs_shap=abs(sensitivity),
                std_shap=0.0,
                importance_rank=rank,
                category=self._get_feature_category(name),
                description=self._get_feature_importance_description(name),
            )
            for rank, (name, sensitivity) in enumerate(sorted_features, 1)
        ]

    def _get_feature_importance_description(self, feature_name: str) -> str:
        """Get description for feature importance."""
        descriptions = {
            "gas_inlet_temp_f": "Gas inlet temperature determines available heat",
            "gas_outlet_temp_f": "Gas outlet temperature indicates heat recovery",
            "water_inlet_temp_f": "Feedwater temperature affects LMTD",
            "water_outlet_temp_f": "Water outlet temperature shows heat absorption",
            "gas_side_dp_in_wc": "Gas-side DP is primary fouling indicator",
            "water_side_dp_psi": "Water-side DP indicates scaling",
            "gas_flow_lb_hr": "Gas flow rate determines heat capacity",
            "water_flow_lb_hr": "Water flow rate affects heat removal",
            "load_pct": "Operating load affects all heat transfer",
            "flue_gas_so2_ppm": "SO2 determines acid dew point risk",
            "fuel_sulfur_pct": "Fuel sulfur drives corrosion potential",
            "effectiveness_ratio": "Direct measure of heat exchanger performance",
        }
        return descriptions.get(feature_name, f"Importance of {feature_name}")

    def explain_cleaning_recommendation(
        self,
        dp_ratio: float,
        u_degradation_pct: float,
        hours_since_cleaning: float,
        features: Dict[str, float],
    ) -> SHAPExplanation:
        """
        Generate explanation for cleaning recommendation.

        Args:
            dp_ratio: Current DP ratio (actual/design)
            u_degradation_pct: UA degradation percentage
            hours_since_cleaning: Hours since last cleaning
            features: Input features

        Returns:
            SHAP explanation for cleaning recommendation
        """
        # Calculate cleaning score (0-1)
        cleaning_score = min(1.0, (
            (dp_ratio - 1.0) * 0.5 +
            u_degradation_pct / 100 * 0.3 +
            min(hours_since_cleaning / 720, 1.0) * 0.2  # 720 hours = 30 days
        ))

        # Determine cleaning class
        if cleaning_score < 0.2:
            predicted_class = "not_required"
        elif cleaning_score < 0.4:
            predicted_class = "monitor"
        elif cleaning_score < 0.6:
            predicted_class = "recommended"
        elif cleaning_score < 0.8:
            predicted_class = "required"
        else:
            predicted_class = "urgent"

        # Add cleaning-specific features
        cleaning_features = dict(features)
        cleaning_features["dp_ratio"] = dp_ratio
        cleaning_features["u_degradation_pct"] = u_degradation_pct
        cleaning_features["hours_since_cleaning"] = hours_since_cleaning

        return self.explain_prediction(
            features=cleaning_features,
            prediction=cleaning_score,
            prediction_type=ExplanationType.CLEANING_RECOMMENDATION,
            predicted_class=predicted_class,
        )

    def get_history(
        self,
        explanation_type: Optional[ExplanationType] = None,
        limit: Optional[int] = None,
    ) -> List[SHAPExplanation]:
        """
        Get explanation history.

        Args:
            explanation_type: Optional filter by type
            limit: Optional limit on results

        Returns:
            List of historical explanations
        """
        history = self._history

        if explanation_type:
            history = [e for e in history if e.explanation_type == explanation_type]

        if limit:
            history = history[-limit:]

        return history

    def clear_history(self) -> None:
        """Clear explanation history."""
        self._history = []
        logger.info("SHAP explanation history cleared")


def create_shap_analyzer(
    baseline_values: Optional[Dict[str, float]] = None,
    track_history: bool = True,
) -> SHAPFeatureAnalyzer:
    """Factory function to create SHAPFeatureAnalyzer."""
    return SHAPFeatureAnalyzer(
        baseline_values=baseline_values,
        track_history=track_history,
    )
