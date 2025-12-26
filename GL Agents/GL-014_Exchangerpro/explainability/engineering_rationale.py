# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Engineering Rationale Generator

Generates deterministic, rule-based explanations using thermal engineering
terminology. Maps ML features to physical engineering concepts for
operator-friendly narratives.

Zero-Hallucination Principle:
- All rationales are derived from deterministic rules and calculations
- Engineering formulas are explicit and traceable
- No LLM is used for numeric calculations
- Provenance tracking via SHA-256 hashes

Author: GreenLang AI Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib
import json
import logging
import uuid

import numpy as np

from .explanation_schemas import (
    ConfidenceLevel,
    EngineeringRationale,
    FeatureCategory,
    FoulingMechanism,
    PredictionType,
)

logger = logging.getLogger(__name__)


class RationaleCategory(Enum):
    """Category of engineering rationale."""
    THERMAL_PERFORMANCE = "thermal_performance"
    HYDRAULIC_PERFORMANCE = "hydraulic_performance"
    FOULING_MECHANISM = "fouling_mechanism"
    OPERATING_CONDITION = "operating_condition"
    DESIGN_LIMIT = "design_limit"
    MAINTENANCE = "maintenance"
    SAFETY = "safety"


class SeverityLevel(Enum):
    """Severity level for observations."""
    CRITICAL = "critical"
    WARNING = "warning"
    NORMAL = "normal"
    OPTIMAL = "optimal"


@dataclass
class EngineeringObservation:
    """Single engineering observation with rationale."""
    category: RationaleCategory
    observation: str
    engineering_explanation: str
    severity: SeverityLevel
    metric_name: str
    metric_value: float
    threshold_value: Optional[float]
    unit: str
    recommendation: Optional[str]
    references: List[str] = field(default_factory=list)


@dataclass
class ThermalPerformanceMetrics:
    """Thermal performance indicators for heat exchanger."""
    U_actual: float  # Actual heat transfer coefficient (W/m2K)
    U_clean: float   # Clean heat transfer coefficient (W/m2K)
    fouling_factor: float  # Calculated fouling factor (m2K/W)
    effectiveness: float   # Heat exchanger effectiveness (-)
    NTU: float            # Number of Transfer Units (-)
    LMTD: float           # Log Mean Temperature Difference (K)
    heat_duty: float      # Actual heat duty (kW)
    design_duty: float    # Design heat duty (kW)


@dataclass
class HydraulicPerformanceMetrics:
    """Hydraulic performance indicators."""
    delta_P_hot: float    # Pressure drop hot side (kPa)
    delta_P_cold: float   # Pressure drop cold side (kPa)
    delta_P_ratio_hot: float   # Ratio to clean pressure drop
    delta_P_ratio_cold: float  # Ratio to clean pressure drop
    velocity_hot: float   # Velocity hot side (m/s)
    velocity_cold: float  # Velocity cold side (m/s)
    reynolds_hot: float   # Reynolds number hot side
    reynolds_cold: float  # Reynolds number cold side


class EngineeringRationaleGenerator:
    """
    Generator for engineering-based explanations.

    Maps ML features to physical engineering concepts and generates
    operator-friendly narratives using thermal engineering terminology.

    Features:
    - Map ML features to engineering concepts
    - Generate operator-friendly narratives
    - Include thermal engineering terminology
    - Reference calculation methodology and versions

    Example narratives:
    - "High delta-P normalized to flow suggests tube-side deposition"
    - "Reduced U relative to clean conditions indicates thermal resistance buildup"
    - "Asymmetric fouling pattern suggests uneven flow distribution"

    Example:
        >>> generator = EngineeringRationaleGenerator()
        >>> rationale = generator.generate_rationale(
        ...     observations=data,
        ...     exchanger_id="HX-001",
        ...     prediction_type=PredictionType.FOULING_FACTOR
        ... )
        >>> print(rationale.summary)
    """

    VERSION = "1.0.0"

    # Engineering thresholds based on industry standards
    THRESHOLDS = {
        # Fouling factor thresholds (m2K/W)
        "fouling_factor_normal": 0.0002,
        "fouling_factor_warning": 0.0005,
        "fouling_factor_critical": 0.001,

        # Heat transfer coefficient degradation (%)
        "U_degradation_normal": 10,
        "U_degradation_warning": 25,
        "U_degradation_critical": 40,

        # Pressure drop ratio thresholds
        "delta_P_ratio_normal": 1.2,
        "delta_P_ratio_warning": 1.5,
        "delta_P_ratio_critical": 2.0,

        # Velocity thresholds (m/s)
        "velocity_min_particulate": 0.5,
        "velocity_max_erosion": 3.0,

        # Temperature thresholds (C)
        "wall_temp_crystallization": 60,
        "wall_temp_reaction": 80,

        # Operating time thresholds (days)
        "days_cleaning_warning": 180,
        "days_cleaning_critical": 365,
    }

    # Feature to engineering term mapping
    FEATURE_MAPPING = {
        "delta_P": "Pressure Drop",
        "delta_P_normalized": "Normalized Pressure Drop",
        "U_actual": "Heat Transfer Coefficient (U)",
        "U_clean": "Clean Heat Transfer Coefficient",
        "fouling_factor": "Fouling Factor (Rf)",
        "effectiveness": "Thermal Effectiveness",
        "NTU": "Number of Transfer Units",
        "LMTD": "Log Mean Temperature Difference",
        "heat_duty": "Heat Duty (Q)",
        "velocity_hot": "Hot-Side Velocity",
        "velocity_cold": "Cold-Side Velocity",
        "T_hot_in": "Hot Inlet Temperature",
        "T_hot_out": "Hot Outlet Temperature",
        "T_cold_in": "Cold Inlet Temperature",
        "T_cold_out": "Cold Outlet Temperature",
        "wall_temperature": "Wall Temperature",
        "reynolds_hot": "Hot-Side Reynolds Number",
        "reynolds_cold": "Cold-Side Reynolds Number",
        "days_since_cleaning": "Days Since Cleaning",
        "operating_hours": "Operating Hours",
    }

    def __init__(self) -> None:
        """Initialize engineering rationale generator."""
        logger.info(f"EngineeringRationaleGenerator initialized (version {self.VERSION})")

    def generate_rationale(
        self,
        observations: Dict[str, float],
        exchanger_id: str,
        prediction_type: PredictionType = PredictionType.FOULING_FACTOR,
        prediction_value: Optional[float] = None,
        design_data: Optional[Dict[str, float]] = None,
    ) -> EngineeringRationale:
        """
        Generate complete engineering rationale for fouling analysis.

        Args:
            observations: Dictionary of feature observations
            exchanger_id: Heat exchanger identifier
            prediction_type: Type of prediction being explained
            prediction_value: The predicted value (optional)
            design_data: Original design specifications (optional)

        Returns:
            EngineeringRationale with complete engineering explanation
        """
        rationale_id = str(uuid.uuid4())

        # Analyze thermal performance
        thermal_indicators = self._analyze_thermal_performance(observations, design_data)

        # Analyze hydraulic performance
        hydraulic_indicators = self._analyze_hydraulic_performance(observations, design_data)

        # Identify fouling mechanism
        fouling_mechanism, mechanism_evidence = self._identify_fouling_mechanism(
            observations, thermal_indicators, hydraulic_indicators
        )

        # Generate key observations
        key_observations = self._generate_key_observations(
            observations, thermal_indicators, hydraulic_indicators
        )

        # Generate summary
        summary = self._generate_summary(
            exchanger_id, observations, thermal_indicators,
            fouling_mechanism, prediction_value
        )

        # Generate detailed rationale
        detailed_rationale = self._generate_detailed_rationale(
            observations, thermal_indicators, hydraulic_indicators,
            fouling_mechanism, key_observations
        )

        # Generate recommendations
        operational_recs = self._generate_operational_recommendations(
            observations, thermal_indicators, hydraulic_indicators, fouling_mechanism
        )
        maintenance_recs = self._generate_maintenance_recommendations(
            observations, thermal_indicators, fouling_mechanism
        )

        # Determine confidence
        confidence = self._determine_confidence(observations, thermal_indicators)

        # Compute provenance hash
        provenance_data = {
            "rationale_id": rationale_id,
            "exchanger_id": exchanger_id,
            "observations": observations,
            "summary": summary[:100],
            "version": self.VERSION,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return EngineeringRationale(
            rationale_id=rationale_id,
            exchanger_id=exchanger_id,
            summary=summary,
            detailed_rationale=detailed_rationale,
            key_observations=key_observations,
            thermal_indicators=thermal_indicators,
            hydraulic_indicators=hydraulic_indicators,
            fouling_mechanism=fouling_mechanism,
            mechanism_evidence=mechanism_evidence,
            operational_recommendations=operational_recs,
            maintenance_recommendations=maintenance_recs,
            confidence=confidence,
            methodology_version=self.VERSION,
            calculation_references=self._get_calculation_references(),
            provenance_hash=provenance_hash,
        )

    def explain_feature(
        self,
        feature_name: str,
        feature_value: float,
        context: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Generate engineering explanation for a single feature.

        Args:
            feature_name: Name of the feature
            feature_value: Value of the feature
            context: Optional context (other feature values)

        Returns:
            Engineering explanation string
        """
        engineering_name = self.FEATURE_MAPPING.get(feature_name, feature_name)

        # Generate context-aware explanation
        if feature_name == "delta_P_normalized":
            if feature_value > 1.5:
                return (
                    f"{engineering_name} = {feature_value:.2f} (>1.5x clean value). "
                    f"Elevated normalized pressure drop indicates significant flow area "
                    f"reduction due to fouling deposits on tube surfaces."
                )
            elif feature_value > 1.2:
                return (
                    f"{engineering_name} = {feature_value:.2f} (1.2-1.5x clean). "
                    f"Moderate increase in pressure drop suggests early-stage fouling "
                    f"accumulation. Monitor trend for acceleration."
                )
            else:
                return (
                    f"{engineering_name} = {feature_value:.2f} (near clean value). "
                    f"Pressure drop within normal operating range."
                )

        elif feature_name == "fouling_factor":
            if feature_value > self.THRESHOLDS["fouling_factor_critical"]:
                return (
                    f"Fouling Factor Rf = {feature_value:.6f} m2K/W (CRITICAL). "
                    f"Exceeds critical threshold of {self.THRESHOLDS['fouling_factor_critical']:.4f} m2K/W. "
                    f"Significant thermal resistance buildup. Cleaning required."
                )
            elif feature_value > self.THRESHOLDS["fouling_factor_warning"]:
                return (
                    f"Fouling Factor Rf = {feature_value:.6f} m2K/W (WARNING). "
                    f"Exceeds warning threshold. Thermal performance degraded. "
                    f"Plan cleaning in near term."
                )
            else:
                return (
                    f"Fouling Factor Rf = {feature_value:.6f} m2K/W (NORMAL). "
                    f"Within acceptable operating limits."
                )

        elif feature_name in ["velocity_hot", "velocity_cold"]:
            side = "hot" if "hot" in feature_name else "cold"
            if feature_value < self.THRESHOLDS["velocity_min_particulate"]:
                return (
                    f"{engineering_name} = {feature_value:.2f} m/s (LOW). "
                    f"Velocity below {self.THRESHOLDS['velocity_min_particulate']:.1f} m/s "
                    f"promotes particulate settling and deposition on {side}-side surfaces."
                )
            elif feature_value > self.THRESHOLDS["velocity_max_erosion"]:
                return (
                    f"{engineering_name} = {feature_value:.2f} m/s (HIGH). "
                    f"Velocity above {self.THRESHOLDS['velocity_max_erosion']:.1f} m/s "
                    f"may cause erosion of protective films on {side} side."
                )
            else:
                return (
                    f"{engineering_name} = {feature_value:.2f} m/s. "
                    f"Velocity within optimal range for {side}-side heat transfer "
                    f"and fouling mitigation."
                )

        elif feature_name == "U_actual":
            if context and "U_clean" in context:
                U_clean = context["U_clean"]
                degradation = (1 - feature_value / U_clean) * 100
                return (
                    f"Heat Transfer Coefficient U = {feature_value:.1f} W/m2K. "
                    f"Degraded by {degradation:.1f}% from clean value ({U_clean:.1f} W/m2K). "
                    f"{'Cleaning recommended.' if degradation > 25 else 'Within acceptable range.'}"
                )
            return f"Heat Transfer Coefficient U = {feature_value:.1f} W/m2K."

        elif feature_name == "days_since_cleaning":
            if feature_value > self.THRESHOLDS["days_cleaning_critical"]:
                return (
                    f"Days Since Cleaning = {feature_value:.0f} days (CRITICAL). "
                    f"Exceeds {self.THRESHOLDS['days_cleaning_critical']:.0f} day threshold. "
                    f"Extended operation without cleaning leads to compacted deposits."
                )
            elif feature_value > self.THRESHOLDS["days_cleaning_warning"]:
                return (
                    f"Days Since Cleaning = {feature_value:.0f} days (WARNING). "
                    f"Approaching maintenance interval. Schedule cleaning."
                )
            else:
                return (
                    f"Days Since Cleaning = {feature_value:.0f} days. "
                    f"Within normal maintenance interval."
                )

        else:
            return f"{engineering_name} = {feature_value:.4f}"

    def generate_narrative(
        self,
        observations: Dict[str, float],
        top_features: List[str],
        prediction_value: float,
        prediction_type: PredictionType = PredictionType.FOULING_FACTOR,
    ) -> str:
        """
        Generate operator-friendly narrative explanation.

        Args:
            observations: Feature observations
            top_features: Top contributing features
            prediction_value: The predicted value
            prediction_type: Type of prediction

        Returns:
            Narrative explanation string
        """
        narrative_parts = []

        # Opening statement
        if prediction_type == PredictionType.FOULING_FACTOR:
            if prediction_value > self.THRESHOLDS["fouling_factor_critical"]:
                narrative_parts.append(
                    f"Analysis indicates severe fouling condition with Rf = {prediction_value:.6f} m2K/W. "
                    f"This exceeds the critical threshold and requires immediate attention."
                )
            elif prediction_value > self.THRESHOLDS["fouling_factor_warning"]:
                narrative_parts.append(
                    f"Moderate fouling detected with Rf = {prediction_value:.6f} m2K/W. "
                    f"Performance degradation is measurable but still within operational limits."
                )
            else:
                narrative_parts.append(
                    f"Fouling factor Rf = {prediction_value:.6f} m2K/W is within normal operating range. "
                    f"Heat exchanger performance is satisfactory."
                )

        # Key driver explanations
        narrative_parts.append("\nKey contributing factors:")

        for feature in top_features[:3]:
            value = observations.get(feature, 0)
            explanation = self.explain_feature(feature, value, observations)
            narrative_parts.append(f"  - {explanation}")

        # Engineering insight
        narrative_parts.append("\nEngineering Assessment:")

        # Check for specific patterns
        if observations.get("delta_P_normalized", 1) > 1.5:
            narrative_parts.append(
                "  High normalized pressure drop relative to clean conditions suggests "
                "significant deposition on heat transfer surfaces, reducing flow cross-section."
            )

        if observations.get("velocity_hot", 1) < 0.5 or observations.get("velocity_cold", 1) < 0.5:
            narrative_parts.append(
                "  Low fluid velocity promotes particle settling and deposition. "
                "Consider increasing flow rate to enhance wall shear stress."
            )

        if observations.get("wall_temperature", 0) > 80:
            narrative_parts.append(
                "  Elevated wall temperature accelerates chemical reaction fouling. "
                "Consider process modifications to reduce thermal stress."
            )

        return "\n".join(narrative_parts)

    def _analyze_thermal_performance(
        self,
        observations: Dict[str, float],
        design_data: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Analyze thermal performance indicators."""
        indicators = {}

        # Current heat transfer coefficient
        U_actual = observations.get("U_actual", observations.get("heat_transfer_coefficient", 0))
        indicators["U_actual"] = U_actual
        indicators["U_actual_unit"] = "W/m2K"

        # Clean/design heat transfer coefficient
        U_clean = observations.get("U_clean", design_data.get("U_design", U_actual * 1.2) if design_data else U_actual * 1.2)
        indicators["U_clean"] = U_clean

        # U degradation percentage
        if U_clean > 0:
            U_degradation = (1 - U_actual / U_clean) * 100
            indicators["U_degradation_pct"] = round(U_degradation, 2)
            indicators["U_degradation_severity"] = self._get_severity(
                U_degradation,
                self.THRESHOLDS["U_degradation_normal"],
                self.THRESHOLDS["U_degradation_warning"],
                self.THRESHOLDS["U_degradation_critical"],
            )

        # Fouling factor
        fouling_factor = observations.get("fouling_factor", 0)
        indicators["fouling_factor"] = fouling_factor
        indicators["fouling_factor_unit"] = "m2K/W"
        indicators["fouling_factor_severity"] = self._get_severity(
            fouling_factor,
            self.THRESHOLDS["fouling_factor_normal"],
            self.THRESHOLDS["fouling_factor_warning"],
            self.THRESHOLDS["fouling_factor_critical"],
        )

        # Effectiveness
        effectiveness = observations.get("effectiveness", 0)
        indicators["effectiveness"] = effectiveness

        # Heat duty
        heat_duty = observations.get("heat_duty", 0)
        design_duty = observations.get("design_duty", design_data.get("design_duty", heat_duty) if design_data else heat_duty)
        indicators["heat_duty"] = heat_duty
        indicators["design_duty"] = design_duty
        if design_duty > 0:
            indicators["duty_ratio"] = round(heat_duty / design_duty * 100, 2)

        # LMTD
        indicators["LMTD"] = observations.get("LMTD", 0)

        return indicators

    def _analyze_hydraulic_performance(
        self,
        observations: Dict[str, float],
        design_data: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Analyze hydraulic performance indicators."""
        indicators = {}

        # Pressure drops
        delta_P_hot = observations.get("delta_P_hot", observations.get("delta_P", 0))
        delta_P_cold = observations.get("delta_P_cold", 0)
        indicators["delta_P_hot"] = delta_P_hot
        indicators["delta_P_cold"] = delta_P_cold
        indicators["delta_P_unit"] = "kPa"

        # Normalized pressure drops
        delta_P_normalized = observations.get("delta_P_normalized", 1)
        indicators["delta_P_normalized"] = delta_P_normalized
        indicators["delta_P_severity"] = self._get_severity(
            delta_P_normalized,
            self.THRESHOLDS["delta_P_ratio_normal"],
            self.THRESHOLDS["delta_P_ratio_warning"],
            self.THRESHOLDS["delta_P_ratio_critical"],
        )

        # Velocities
        velocity_hot = observations.get("velocity_hot", 1)
        velocity_cold = observations.get("velocity_cold", 1)
        indicators["velocity_hot"] = velocity_hot
        indicators["velocity_cold"] = velocity_cold
        indicators["velocity_unit"] = "m/s"

        # Velocity assessments
        indicators["velocity_hot_assessment"] = self._assess_velocity(velocity_hot)
        indicators["velocity_cold_assessment"] = self._assess_velocity(velocity_cold)

        # Reynolds numbers
        indicators["reynolds_hot"] = observations.get("reynolds_hot", 0)
        indicators["reynolds_cold"] = observations.get("reynolds_cold", 0)

        # Flow regime
        indicators["flow_regime_hot"] = self._get_flow_regime(indicators["reynolds_hot"])
        indicators["flow_regime_cold"] = self._get_flow_regime(indicators["reynolds_cold"])

        return indicators

    def _identify_fouling_mechanism(
        self,
        observations: Dict[str, float],
        thermal_indicators: Dict[str, Any],
        hydraulic_indicators: Dict[str, Any],
    ) -> Tuple[FoulingMechanism, List[str]]:
        """Identify fouling mechanism with evidence."""
        evidence = []
        scores = {
            FoulingMechanism.PARTICULATE: 0.0,
            FoulingMechanism.CRYSTALLIZATION: 0.0,
            FoulingMechanism.BIOLOGICAL: 0.0,
            FoulingMechanism.CORROSION: 0.0,
            FoulingMechanism.CHEMICAL_REACTION: 0.0,
        }

        # Low velocity indicates particulate fouling
        if observations.get("velocity_hot", 1) < self.THRESHOLDS["velocity_min_particulate"]:
            scores[FoulingMechanism.PARTICULATE] += 0.4
            evidence.append("Low hot-side velocity promotes particle settling")

        if observations.get("velocity_cold", 1) < self.THRESHOLDS["velocity_min_particulate"]:
            scores[FoulingMechanism.PARTICULATE] += 0.3
            evidence.append("Low cold-side velocity promotes particle settling")

        # High wall temperature indicates chemical reaction
        wall_temp = observations.get("wall_temperature", observations.get("T_hot_in", 50))
        if wall_temp > self.THRESHOLDS["wall_temp_reaction"]:
            scores[FoulingMechanism.CHEMICAL_REACTION] += 0.5
            evidence.append(f"High wall temperature ({wall_temp:.1f}C) accelerates chemical reactions")
        elif wall_temp > self.THRESHOLDS["wall_temp_crystallization"]:
            scores[FoulingMechanism.CRYSTALLIZATION] += 0.4
            evidence.append(f"Elevated wall temperature ({wall_temp:.1f}C) promotes crystallization")

        # Supersaturation indicates crystallization
        if observations.get("supersaturation", 0) > 1.0:
            scores[FoulingMechanism.CRYSTALLIZATION] += 0.5
            evidence.append("Supersaturated conditions drive crystalline deposit formation")

        # pH extremes indicate corrosion
        ph = observations.get("ph_level", 7)
        if ph < 5:
            scores[FoulingMechanism.CORROSION] += 0.5
            evidence.append(f"Low pH ({ph:.1f}) promotes acidic corrosion")
        elif ph > 9:
            scores[FoulingMechanism.CORROSION] += 0.4
            evidence.append(f"High pH ({ph:.1f}) promotes caustic corrosion")

        # Low temperature + organic content indicates biological
        if wall_temp < 40 and observations.get("organic_content", 0) > 0:
            scores[FoulingMechanism.BIOLOGICAL] += 0.4
            evidence.append("Low temperature with organic content supports biofilm growth")

        # High pressure drop ratio relative to thermal degradation
        delta_P_ratio = hydraulic_indicators.get("delta_P_normalized", 1)
        U_degradation = thermal_indicators.get("U_degradation_pct", 0)

        if delta_P_ratio > 1.5 and U_degradation < 15:
            scores[FoulingMechanism.PARTICULATE] += 0.3
            evidence.append("High pressure drop with moderate thermal degradation suggests porous deposits")

        # Select mechanism with highest score
        best_mechanism = max(scores, key=scores.get)

        if scores[best_mechanism] < 0.3:
            best_mechanism = FoulingMechanism.COMBINED
            evidence.append("Multiple mechanisms may contribute to observed fouling")

        return best_mechanism, evidence

    def _generate_key_observations(
        self,
        observations: Dict[str, float],
        thermal_indicators: Dict[str, Any],
        hydraulic_indicators: Dict[str, Any],
    ) -> List[str]:
        """Generate list of key engineering observations."""
        key_obs = []

        # Thermal observations
        if thermal_indicators.get("U_degradation_severity") == SeverityLevel.CRITICAL:
            key_obs.append(
                f"CRITICAL: Heat transfer coefficient degraded by "
                f"{thermal_indicators.get('U_degradation_pct', 0):.1f}% from clean conditions"
            )
        elif thermal_indicators.get("U_degradation_severity") == SeverityLevel.WARNING:
            key_obs.append(
                f"WARNING: Moderate U degradation ({thermal_indicators.get('U_degradation_pct', 0):.1f}%)"
            )

        # Fouling factor
        ff = thermal_indicators.get("fouling_factor", 0)
        if ff > self.THRESHOLDS["fouling_factor_warning"]:
            key_obs.append(f"Fouling factor Rf = {ff:.6f} m2K/W exceeds acceptable limits")

        # Hydraulic observations
        if hydraulic_indicators.get("delta_P_severity") == SeverityLevel.CRITICAL:
            key_obs.append(
                f"CRITICAL: Pressure drop {hydraulic_indicators.get('delta_P_normalized', 1):.2f}x clean value"
            )
        elif hydraulic_indicators.get("delta_P_severity") == SeverityLevel.WARNING:
            key_obs.append(
                f"WARNING: Elevated pressure drop ({hydraulic_indicators.get('delta_P_normalized', 1):.2f}x)"
            )

        # Velocity observations
        if hydraulic_indicators.get("velocity_hot_assessment") == "low":
            key_obs.append(
                f"Low hot-side velocity ({hydraulic_indicators.get('velocity_hot', 0):.2f} m/s) "
                f"promotes deposition"
            )

        # Flow regime
        if hydraulic_indicators.get("flow_regime_hot") == "laminar":
            key_obs.append("Laminar flow on hot side reduces heat transfer and self-cleaning")

        # Operating time
        days = observations.get("days_since_cleaning", 0)
        if days > self.THRESHOLDS["days_cleaning_critical"]:
            key_obs.append(f"Extended operation ({days:.0f} days) without cleaning")

        return key_obs

    def _generate_summary(
        self,
        exchanger_id: str,
        observations: Dict[str, float],
        thermal_indicators: Dict[str, Any],
        fouling_mechanism: FoulingMechanism,
        prediction_value: Optional[float],
    ) -> str:
        """Generate executive summary."""
        # Determine overall status
        ff_severity = thermal_indicators.get("fouling_factor_severity", SeverityLevel.NORMAL)

        if ff_severity == SeverityLevel.CRITICAL:
            status = "CRITICAL FOULING CONDITION"
            action = "Immediate cleaning required."
        elif ff_severity == SeverityLevel.WARNING:
            status = "ELEVATED FOULING DETECTED"
            action = "Schedule cleaning within maintenance window."
        else:
            status = "NORMAL OPERATION"
            action = "Continue monitoring."

        summary = (
            f"Heat Exchanger {exchanger_id}: {status}\n\n"
            f"Fouling Factor: {thermal_indicators.get('fouling_factor', 0):.6f} m2K/W\n"
            f"U Degradation: {thermal_indicators.get('U_degradation_pct', 0):.1f}%\n"
            f"Dominant Mechanism: {fouling_mechanism.value.replace('_', ' ').title()}\n\n"
            f"Recommendation: {action}"
        )

        return summary

    def _generate_detailed_rationale(
        self,
        observations: Dict[str, float],
        thermal_indicators: Dict[str, Any],
        hydraulic_indicators: Dict[str, Any],
        fouling_mechanism: FoulingMechanism,
        key_observations: List[str],
    ) -> str:
        """Generate detailed engineering rationale."""
        sections = []

        # Thermal Performance Section
        sections.append("=== THERMAL PERFORMANCE ANALYSIS ===")
        sections.append(
            f"Current Heat Transfer Coefficient (U): {thermal_indicators.get('U_actual', 0):.1f} W/m2K"
        )
        sections.append(
            f"Clean/Design U: {thermal_indicators.get('U_clean', 0):.1f} W/m2K"
        )
        sections.append(
            f"Performance Degradation: {thermal_indicators.get('U_degradation_pct', 0):.1f}%"
        )
        sections.append(
            f"Calculated Fouling Factor (Rf): {thermal_indicators.get('fouling_factor', 0):.6f} m2K/W"
        )
        sections.append("")

        # Hydraulic Performance Section
        sections.append("=== HYDRAULIC PERFORMANCE ANALYSIS ===")
        sections.append(
            f"Pressure Drop Ratio (to clean): {hydraulic_indicators.get('delta_P_normalized', 1):.2f}x"
        )
        sections.append(
            f"Hot-Side Velocity: {hydraulic_indicators.get('velocity_hot', 0):.2f} m/s "
            f"({hydraulic_indicators.get('velocity_hot_assessment', 'unknown')})"
        )
        sections.append(
            f"Cold-Side Velocity: {hydraulic_indicators.get('velocity_cold', 0):.2f} m/s "
            f"({hydraulic_indicators.get('velocity_cold_assessment', 'unknown')})"
        )
        sections.append(
            f"Flow Regime (Hot/Cold): {hydraulic_indicators.get('flow_regime_hot', 'unknown')}/"
            f"{hydraulic_indicators.get('flow_regime_cold', 'unknown')}"
        )
        sections.append("")

        # Fouling Mechanism Section
        sections.append("=== FOULING MECHANISM ASSESSMENT ===")
        sections.append(f"Dominant Mechanism: {fouling_mechanism.value.replace('_', ' ').title()}")
        sections.append("")

        # Key Observations
        sections.append("=== KEY OBSERVATIONS ===")
        for obs in key_observations:
            sections.append(f"  - {obs}")
        sections.append("")

        # Engineering Basis
        sections.append("=== ENGINEERING BASIS ===")
        sections.append(
            "Fouling factor calculated from: Rf = 1/U_actual - 1/U_clean"
        )
        sections.append(
            "Pressure drop correlation: delta_P proportional to (D_clean/D_fouled)^5"
        )
        sections.append(
            f"Analysis methodology: {self.VERSION}"
        )

        return "\n".join(sections)

    def _generate_operational_recommendations(
        self,
        observations: Dict[str, float],
        thermal_indicators: Dict[str, Any],
        hydraulic_indicators: Dict[str, Any],
        fouling_mechanism: FoulingMechanism,
    ) -> List[str]:
        """Generate operational recommendations."""
        recommendations = []

        # Velocity-based recommendations
        if hydraulic_indicators.get("velocity_hot_assessment") == "low":
            recommendations.append(
                "Increase hot-side flow rate to achieve velocity > 0.5 m/s to reduce deposition"
            )

        if hydraulic_indicators.get("velocity_cold_assessment") == "low":
            recommendations.append(
                "Increase cold-side flow rate to enhance wall shear stress"
            )

        # Temperature-based recommendations
        if fouling_mechanism == FoulingMechanism.CHEMICAL_REACTION:
            recommendations.append(
                "Consider reducing inlet temperature to slow reaction fouling kinetics"
            )

        if fouling_mechanism == FoulingMechanism.CRYSTALLIZATION:
            recommendations.append(
                "Monitor saturation levels and consider anti-scalant treatment"
            )

        # Flow regime recommendations
        if hydraulic_indicators.get("flow_regime_hot") == "laminar":
            recommendations.append(
                "Increase hot-side flow to achieve turbulent regime (Re > 4000) for improved heat transfer"
            )

        # Pressure drop recommendations
        if hydraulic_indicators.get("delta_P_severity") == SeverityLevel.CRITICAL:
            recommendations.append(
                "High pressure drop may cause pump cavitation - verify pump operating point"
            )

        return recommendations[:5]

    def _generate_maintenance_recommendations(
        self,
        observations: Dict[str, float],
        thermal_indicators: Dict[str, Any],
        fouling_mechanism: FoulingMechanism,
    ) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []

        ff_severity = thermal_indicators.get("fouling_factor_severity", SeverityLevel.NORMAL)

        if ff_severity == SeverityLevel.CRITICAL:
            recommendations.append("IMMEDIATE: Schedule emergency cleaning within 7 days")
            recommendations.append("Implement enhanced monitoring until cleaning completed")

        elif ff_severity == SeverityLevel.WARNING:
            recommendations.append("Schedule cleaning during next maintenance window")
            recommendations.append("Increase monitoring frequency to weekly")

        # Mechanism-specific recommendations
        if fouling_mechanism == FoulingMechanism.CRYSTALLIZATION:
            recommendations.append("Use chemical cleaning with appropriate descaler")
            recommendations.append("Consider installing water softening upstream")

        elif fouling_mechanism == FoulingMechanism.BIOLOGICAL:
            recommendations.append("Use biocide treatment during cleaning")
            recommendations.append("Consider continuous biocide dosing")

        elif fouling_mechanism == FoulingMechanism.PARTICULATE:
            recommendations.append("High-pressure water jetting recommended for cleaning")
            recommendations.append("Install strainer upstream to capture particles")

        elif fouling_mechanism == FoulingMechanism.CHEMICAL_REACTION:
            recommendations.append("Mechanical cleaning may be required for polymerized deposits")

        # Operating time recommendations
        days = observations.get("days_since_cleaning", 0)
        if days > self.THRESHOLDS["days_cleaning_warning"]:
            recommendations.append(
                f"Establish regular cleaning interval (current: {days:.0f} days)"
            )

        return recommendations[:5]

    def _determine_confidence(
        self,
        observations: Dict[str, float],
        thermal_indicators: Dict[str, Any],
    ) -> ConfidenceLevel:
        """Determine confidence level of the rationale."""
        # Check data completeness
        required_features = [
            "U_actual", "delta_P_normalized", "velocity_hot",
            "days_since_cleaning", "fouling_factor"
        ]

        available = sum(1 for f in required_features if f in observations)
        completeness = available / len(required_features)

        if completeness >= 0.8 and "U_clean" in observations:
            return ConfidenceLevel.HIGH
        elif completeness >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _get_severity(
        self,
        value: float,
        normal_threshold: float,
        warning_threshold: float,
        critical_threshold: float,
    ) -> SeverityLevel:
        """Determine severity level based on thresholds."""
        if value >= critical_threshold:
            return SeverityLevel.CRITICAL
        elif value >= warning_threshold:
            return SeverityLevel.WARNING
        elif value >= normal_threshold:
            return SeverityLevel.NORMAL
        else:
            return SeverityLevel.OPTIMAL

    def _assess_velocity(self, velocity: float) -> str:
        """Assess velocity level."""
        if velocity < self.THRESHOLDS["velocity_min_particulate"]:
            return "low"
        elif velocity > self.THRESHOLDS["velocity_max_erosion"]:
            return "high"
        else:
            return "optimal"

    def _get_flow_regime(self, reynolds: float) -> str:
        """Determine flow regime from Reynolds number."""
        if reynolds < 2300:
            return "laminar"
        elif reynolds < 4000:
            return "transitional"
        else:
            return "turbulent"

    def _get_calculation_references(self) -> List[str]:
        """Get list of calculation methodology references."""
        return [
            "TEMA Standards for Heat Exchanger Design",
            "Kern, D.Q. - Process Heat Transfer",
            "Epstein, N. - Fouling Science and Technology (1983)",
            "Taborek, J. et al. - Fouling in Heat Exchangers (1972)",
            "HTRI Guidelines for Fouling Assessment",
        ]


# Convenience functions
def generate_engineering_explanation(
    observations: Dict[str, float],
    exchanger_id: str,
    prediction_type: PredictionType = PredictionType.FOULING_FACTOR,
) -> EngineeringRationale:
    """
    Convenience function to generate engineering rationale.

    Args:
        observations: Feature observations
        exchanger_id: Heat exchanger identifier
        prediction_type: Type of prediction

    Returns:
        EngineeringRationale with complete explanation
    """
    generator = EngineeringRationaleGenerator()
    return generator.generate_rationale(observations, exchanger_id, prediction_type)


def explain_feature_engineering(
    feature_name: str,
    feature_value: float,
    context: Optional[Dict[str, float]] = None,
) -> str:
    """
    Convenience function to explain a single feature.

    Args:
        feature_name: Name of the feature
        feature_value: Value of the feature
        context: Optional context

    Returns:
        Engineering explanation string
    """
    generator = EngineeringRationaleGenerator()
    return generator.explain_feature(feature_name, feature_value, context)
