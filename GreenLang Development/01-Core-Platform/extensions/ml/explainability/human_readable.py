# -*- coding: utf-8 -*-
"""
Human-Readable Explanation Generator for Process Heat Domain.

This module converts technical ML explanations (SHAP values, LIME weights)
into clear, actionable natural language explanations tailored for
different audiences and Process Heat domain contexts.

The generator follows zero-hallucination principles:
- All explanations are derived from actual model outputs
- No fabricated insights or recommendations
- Complete provenance tracking for audit trails

Example:
    >>> from greenlang.ml.explainability import HumanReadableExplainer
    >>> explainer = HumanReadableExplainer(
    ...     domain="process_heat",
    ...     equipment_type="boiler"
    ... )
    >>> explanation = explainer.generate_explanation(
    ...     prediction=0.85,
    ...     feature_contributions={"flue_gas_temperature": 0.35, "days_since_cleaning": 0.28},
    ...     top_features=[("flue_gas_temperature", 0.35), ("days_since_cleaning", 0.28)]
    ... )
    >>> print(explanation)
    "The model predicts HIGH fouling risk (85%) because:
     1) Flue gas temperature increased by 15F (contributing 35%)
     2) Days since last cleaning exceeded 45 (contributing 28%)..."

Author: GreenLang Team
Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import hashlib
import logging

from .schemas import (
    ExplanationResult,
    ExplainerType,
    ExplanationLevel,
    ProcessHeatContext,
    compute_provenance_hash,
)

logger = logging.getLogger(__name__)


class AudienceType(str, Enum):
    """Target audience for explanation."""

    OPERATOR = "operator"  # Equipment operators
    ENGINEER = "engineer"  # Process engineers
    MANAGER = "manager"  # Plant managers
    EXECUTIVE = "executive"  # C-suite executives
    AUDITOR = "auditor"  # Regulatory auditors
    TECHNICAL = "technical"  # Technical specialists


class RiskLevel(str, Enum):
    """Risk level classifications."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class HumanReadableExplainer:
    """
    Human-Readable Explanation Generator for Process Heat Domain.

    Transforms technical ML explanations into clear, actionable narratives
    tailored for different stakeholders in process heat operations.

    This class provides:
    - Domain-specific explanation templates
    - Audience-appropriate language levels
    - Actionable recommendations
    - Risk contextualization
    - Complete provenance tracking

    Attributes:
        domain: Domain context (process_heat, emissions, energy)
        equipment_type: Type of equipment being monitored
        audience: Target audience for explanations
        include_recommendations: Whether to include action items
        templates: Domain-specific language templates

    Example:
        >>> explainer = HumanReadableExplainer(
        ...     domain="process_heat",
        ...     equipment_type="boiler",
        ...     audience=AudienceType.OPERATOR
        ... )
        >>> text = explainer.generate_explanation(
        ...     prediction=0.85,
        ...     feature_contributions={"temperature": 0.4, "pressure": 0.3}
        ... )
    """

    # Process Heat domain templates
    PROCESS_HEAT_TEMPLATES = {
        "fouling_risk": {
            "prediction_templates": {
                "high": "The model predicts {risk_level} fouling risk ({prediction:.0%}) for {equipment_type}.",
                "medium": "Fouling risk is at a {risk_level} level ({prediction:.0%}) for {equipment_type}.",
                "low": "Fouling risk is {risk_level} ({prediction:.0%}) for {equipment_type}."
            },
            "feature_descriptions": {
                "flue_gas_temperature": "Flue gas temperature",
                "stack_temperature": "Stack temperature",
                "days_since_cleaning": "Days since last cleaning",
                "operating_hours": "Cumulative operating hours",
                "cycles": "Start/stop cycles",
                "excess_air": "Excess air percentage",
                "fouling_factor": "Current fouling factor",
                "inlet_temperature": "Inlet temperature",
                "outlet_temperature": "Outlet temperature",
                "differential_pressure": "Differential pressure drop",
            },
            "impact_templates": {
                "positive": "{feature} is elevated ({direction_word}), contributing {contribution:.0%} to the risk",
                "negative": "{feature} is within normal range, reducing risk by {contribution:.0%}"
            },
            "recommendations": {
                "flue_gas_temperature": "Consider reducing firing rate or checking air-fuel ratio",
                "days_since_cleaning": "Schedule cleaning/maintenance based on current fouling accumulation",
                "operating_hours": "Plan preventive maintenance based on run-time",
                "excess_air": "Optimize combustion by adjusting air-fuel mixture",
                "differential_pressure": "Inspect heat transfer surfaces for deposit buildup",
                "default": "Monitor this parameter and consider optimization"
            }
        },
        "efficiency": {
            "prediction_templates": {
                "high": "{equipment_type} is operating at {risk_level} efficiency ({prediction:.0%}).",
                "medium": "{equipment_type} efficiency is moderate ({prediction:.0%}).",
                "low": "{equipment_type} efficiency is below optimal ({prediction:.0%})."
            },
            "feature_descriptions": {
                "stack_temperature": "Stack temperature (heat loss indicator)",
                "excess_air": "Excess air percentage",
                "flue_gas_oxygen": "Flue gas oxygen content",
                "combustion_efficiency": "Combustion efficiency",
                "ambient_temperature": "Ambient conditions",
                "fuel_quality": "Fuel quality index",
                "load_factor": "Current load factor",
            },
            "impact_templates": {
                "positive": "{feature} is contributing positively ({contribution:.0%})",
                "negative": "{feature} is reducing efficiency ({contribution:.0%})"
            },
            "recommendations": {
                "stack_temperature": "High stack temperature indicates heat loss - consider economizer optimization",
                "excess_air": "Optimize excess air to balance combustion efficiency and emissions",
                "load_factor": "Consider load optimization to improve part-load efficiency",
                "default": "Review operating parameters for optimization opportunities"
            }
        },
        "maintenance": {
            "prediction_templates": {
                "high": "Maintenance is {risk_level} recommended for {equipment_type}.",
                "medium": "Routine maintenance check suggested for {equipment_type}.",
                "low": "{equipment_type} is in good operating condition."
            },
            "feature_descriptions": {
                "vibration": "Equipment vibration levels",
                "temperature_differential": "Abnormal temperature patterns",
                "pressure_drop": "Pressure drop across system",
                "runtime_hours": "Total runtime since last maintenance",
                "start_cycles": "Number of start/stop cycles",
                "anomaly_score": "Anomaly detection score",
            },
            "impact_templates": {
                "positive": "{feature} indicates wear or degradation ({contribution:.0%})",
                "negative": "{feature} is within acceptable limits"
            },
            "recommendations": {
                "vibration": "Schedule vibration analysis and bearing inspection",
                "temperature_differential": "Inspect for blockages or heat exchanger fouling",
                "pressure_drop": "Check for leaks, blockages, or filter condition",
                "runtime_hours": "Follow manufacturer's recommended maintenance intervals",
                "default": "Include in next scheduled maintenance inspection"
            }
        }
    }

    # Audience-specific language adjustments
    AUDIENCE_STYLES = {
        AudienceType.OPERATOR: {
            "complexity": "simple",
            "include_technical": False,
            "action_oriented": True,
            "include_numbers": True,
            "max_features": 3,
        },
        AudienceType.ENGINEER: {
            "complexity": "technical",
            "include_technical": True,
            "action_oriented": True,
            "include_numbers": True,
            "max_features": 5,
        },
        AudienceType.MANAGER: {
            "complexity": "moderate",
            "include_technical": False,
            "action_oriented": True,
            "include_numbers": True,
            "max_features": 3,
        },
        AudienceType.EXECUTIVE: {
            "complexity": "simple",
            "include_technical": False,
            "action_oriented": False,
            "include_numbers": True,
            "max_features": 2,
        },
        AudienceType.AUDITOR: {
            "complexity": "technical",
            "include_technical": True,
            "action_oriented": False,
            "include_numbers": True,
            "max_features": 10,
        },
        AudienceType.TECHNICAL: {
            "complexity": "technical",
            "include_technical": True,
            "action_oriented": True,
            "include_numbers": True,
            "max_features": 10,
        },
    }

    def __init__(
        self,
        domain: str = "process_heat",
        equipment_type: str = "boiler",
        model_type: str = "fouling_risk",
        audience: AudienceType = AudienceType.ENGINEER,
        include_recommendations: bool = True,
        include_provenance: bool = True,
        process_heat_context: Optional[ProcessHeatContext] = None,
    ):
        """
        Initialize human-readable explainer.

        Args:
            domain: Domain context (process_heat, emissions, energy)
            equipment_type: Type of equipment (boiler, furnace, etc.)
            model_type: Type of model (fouling_risk, efficiency, maintenance)
            audience: Target audience for explanations
            include_recommendations: Include actionable recommendations
            include_provenance: Include provenance hash in output
            process_heat_context: Additional domain context
        """
        self.domain = domain
        self.equipment_type = equipment_type
        self.model_type = model_type
        self.audience = audience
        self.include_recommendations = include_recommendations
        self.include_provenance = include_provenance
        self.process_heat_context = process_heat_context

        # Load templates
        self.templates = self.PROCESS_HEAT_TEMPLATES.get(
            model_type,
            self.PROCESS_HEAT_TEMPLATES["fouling_risk"]
        )
        self.style = self.AUDIENCE_STYLES.get(audience, self.AUDIENCE_STYLES[AudienceType.ENGINEER])

        logger.info(
            f"HumanReadableExplainer initialized: domain={domain}, "
            f"equipment={equipment_type}, audience={audience.value}"
        )

    def generate_explanation(
        self,
        prediction: float,
        feature_contributions: Dict[str, float],
        top_features: Optional[List[Tuple[str, float]]] = None,
        feature_values: Optional[Dict[str, float]] = None,
        confidence: float = 0.85,
    ) -> str:
        """
        Generate human-readable explanation from model outputs.

        Args:
            prediction: Model prediction value (0-1 for probabilities)
            feature_contributions: SHAP/LIME feature contributions
            top_features: Pre-sorted top features (optional)
            feature_values: Actual feature values (optional)
            confidence: Explanation confidence score

        Returns:
            Human-readable explanation text

        Example:
            >>> text = explainer.generate_explanation(
            ...     prediction=0.85,
            ...     feature_contributions={"temperature": 0.35, "pressure": 0.28},
            ...     feature_values={"temperature": 450, "pressure": 80}
            ... )
        """
        start_time = datetime.now()

        # Get top features if not provided
        if top_features is None:
            top_features = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:self.style["max_features"]]

        # Determine risk level
        risk_level = self._get_risk_level(prediction)

        # Build explanation sections
        sections = []

        # 1. Summary statement
        summary = self._generate_summary(prediction, risk_level)
        sections.append(summary)

        # 2. Contributing factors
        factors = self._generate_factors_explanation(
            top_features, feature_values, prediction
        )
        sections.append(factors)

        # 3. Confidence statement
        if self.style["include_technical"]:
            confidence_text = self._generate_confidence_statement(confidence)
            sections.append(confidence_text)

        # 4. Recommendations
        if self.include_recommendations and self.style["action_oriented"]:
            recommendations = self._generate_recommendations(top_features)
            if recommendations:
                sections.append(recommendations)

        # 5. Provenance (for auditors)
        if self.include_provenance and self.audience == AudienceType.AUDITOR:
            provenance = self._generate_provenance_section(
                prediction, feature_contributions
            )
            sections.append(provenance)

        explanation = "\n\n".join(sections)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug(f"Generated explanation in {elapsed_ms:.2f}ms")

        return explanation

    def _get_risk_level(self, prediction: float) -> RiskLevel:
        """
        Convert prediction value to risk level.

        Args:
            prediction: Model prediction (0-1)

        Returns:
            Corresponding risk level
        """
        if prediction >= 0.9:
            return RiskLevel.CRITICAL
        elif prediction >= 0.7:
            return RiskLevel.HIGH
        elif prediction >= 0.4:
            return RiskLevel.MEDIUM
        elif prediction >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _generate_summary(self, prediction: float, risk_level: RiskLevel) -> str:
        """
        Generate summary statement.

        Args:
            prediction: Model prediction
            risk_level: Risk level classification

        Returns:
            Summary text
        """
        # Get template based on risk level
        if risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            template_key = "high"
        elif risk_level == RiskLevel.MEDIUM:
            template_key = "medium"
        else:
            template_key = "low"

        template = self.templates["prediction_templates"].get(
            template_key,
            "Model prediction: {prediction:.0%}"
        )

        return template.format(
            prediction=prediction,
            risk_level=risk_level.value.upper(),
            equipment_type=self.equipment_type.title()
        )

    def _generate_factors_explanation(
        self,
        top_features: List[Tuple[str, float]],
        feature_values: Optional[Dict[str, float]],
        prediction: float
    ) -> str:
        """
        Generate explanation of contributing factors.

        Args:
            top_features: List of (feature_name, contribution)
            feature_values: Actual feature values
            prediction: Model prediction

        Returns:
            Factors explanation text
        """
        lines = ["Key factors influencing this prediction:"]

        for i, (feature_name, contribution) in enumerate(top_features, 1):
            # Get readable feature name
            readable_name = self.templates["feature_descriptions"].get(
                feature_name,
                feature_name.replace("_", " ").title()
            )

            # Determine direction
            is_positive = contribution > 0
            direction_word = "increasing" if is_positive else "decreasing"
            impact_type = "positive" if is_positive else "negative"

            # Get template
            template = self.templates["impact_templates"].get(
                impact_type,
                "{feature} contributes {contribution:.0%}"
            )

            # Format contribution
            contribution_pct = abs(contribution)
            if contribution_pct > 1:
                # If contribution is already a percentage
                contribution_pct = contribution_pct / 100

            # Build line
            if self.style["include_numbers"] and feature_values:
                value = feature_values.get(feature_name)
                if value is not None:
                    line = f"{i}. {readable_name}: {value:.1f} - contributing {contribution_pct:.0%} to risk"
                else:
                    line = template.format(
                        feature=readable_name,
                        contribution=contribution_pct,
                        direction_word=direction_word
                    )
            else:
                line = template.format(
                    feature=readable_name,
                    contribution=contribution_pct,
                    direction_word=direction_word
                )

            lines.append(f"  {line}")

        return "\n".join(lines)

    def _generate_confidence_statement(self, confidence: float) -> str:
        """
        Generate confidence statement.

        Args:
            confidence: Confidence score (0-1)

        Returns:
            Confidence statement text
        """
        if confidence >= 0.9:
            level = "high"
        elif confidence >= 0.7:
            level = "moderate"
        else:
            level = "limited"

        return f"Explanation confidence: {level} ({confidence:.0%})"

    def _generate_recommendations(
        self,
        top_features: List[Tuple[str, float]]
    ) -> str:
        """
        Generate actionable recommendations.

        Args:
            top_features: Top contributing features

        Returns:
            Recommendations text
        """
        lines = ["Recommended actions:"]
        seen_recommendations = set()

        for feature_name, contribution in top_features:
            # Only recommend for significant positive contributions
            if contribution > 0.1:  # More than 10% contribution
                recommendation = self.templates["recommendations"].get(
                    feature_name,
                    self.templates["recommendations"].get("default")
                )

                if recommendation and recommendation not in seen_recommendations:
                    lines.append(f"  - {recommendation}")
                    seen_recommendations.add(recommendation)

        if len(lines) == 1:
            return ""  # No recommendations generated

        return "\n".join(lines)

    def _generate_provenance_section(
        self,
        prediction: float,
        feature_contributions: Dict[str, float]
    ) -> str:
        """
        Generate provenance section for audit purposes.

        Args:
            prediction: Model prediction
            feature_contributions: Feature contributions

        Returns:
            Provenance section text
        """
        provenance_data = {
            "prediction": prediction,
            "contributions": feature_contributions,
            "timestamp": datetime.utcnow().isoformat(),
            "model_type": self.model_type,
            "equipment_type": self.equipment_type,
        }

        provenance_hash = compute_provenance_hash(provenance_data)

        return (
            f"Provenance (for audit trail):\n"
            f"  Hash: {provenance_hash}\n"
            f"  Generated: {datetime.utcnow().isoformat()}"
        )

    def convert_explanation_result(
        self,
        result: ExplanationResult
    ) -> str:
        """
        Convert ExplanationResult to human-readable text.

        Args:
            result: ExplanationResult from SHAP/LIME explainer

        Returns:
            Human-readable explanation text
        """
        return self.generate_explanation(
            prediction=result.prediction,
            feature_contributions=result.feature_contributions,
            top_features=result.top_features,
            confidence=result.confidence,
        )

    def set_audience(self, audience: AudienceType) -> None:
        """
        Update target audience.

        Args:
            audience: New target audience
        """
        self.audience = audience
        self.style = self.AUDIENCE_STYLES.get(audience, self.AUDIENCE_STYLES[AudienceType.ENGINEER])
        logger.debug(f"Audience updated to {audience.value}")

    def set_equipment_type(self, equipment_type: str) -> None:
        """
        Update equipment type.

        Args:
            equipment_type: New equipment type
        """
        self.equipment_type = equipment_type
        logger.debug(f"Equipment type updated to {equipment_type}")


class ProcessHeatExplanationTemplates:
    """
    Pre-built explanation templates for common Process Heat scenarios.

    Provides ready-to-use templates for various equipment types and
    model outputs in the Process Heat domain.
    """

    @staticmethod
    def get_boiler_fouling_template() -> Dict[str, Any]:
        """Get template for boiler fouling risk explanations."""
        return {
            "summary_high": (
                "ALERT: Boiler fouling risk is HIGH ({prediction:.0%}). "
                "Immediate attention recommended to prevent efficiency loss and potential shutdown."
            ),
            "summary_medium": (
                "Boiler fouling risk is MODERATE ({prediction:.0%}). "
                "Schedule inspection and consider cleaning within the next maintenance window."
            ),
            "summary_low": (
                "Boiler fouling risk is LOW ({prediction:.0%}). "
                "Current operation is within acceptable parameters."
            ),
            "factors": {
                "flue_gas_temperature": (
                    "Elevated flue gas temperature ({value:.0f}F) indicates heat transfer degradation, "
                    "likely due to fouling deposits on heat exchange surfaces."
                ),
                "days_since_cleaning": (
                    "It has been {value:.0f} days since the last cleaning. "
                    "Fouling accumulates over time and should be addressed."
                ),
                "stack_temperature": (
                    "Stack temperature of {value:.0f}F suggests heat not being "
                    "effectively transferred to the process."
                ),
                "differential_pressure": (
                    "Pressure drop of {value:.1f} PSI across the boiler indicates "
                    "possible flow restriction from deposits."
                ),
            },
            "recommendations": [
                "Inspect water-side surfaces for scale buildup",
                "Check fire-side surfaces for soot and combustion deposits",
                "Review water treatment program effectiveness",
                "Consider chemical or mechanical cleaning",
            ]
        }

    @staticmethod
    def get_furnace_efficiency_template() -> Dict[str, Any]:
        """Get template for furnace efficiency explanations."""
        return {
            "summary_optimal": (
                "Furnace operating at OPTIMAL efficiency ({prediction:.0%}). "
                "Current combustion parameters are well-tuned."
            ),
            "summary_suboptimal": (
                "Furnace efficiency is BELOW OPTIMAL ({prediction:.0%}). "
                "Combustion tuning recommended to reduce fuel consumption and emissions."
            ),
            "factors": {
                "excess_air": (
                    "Excess air at {value:.0f}% is {assessment}. "
                    "Optimal range is typically 10-20% for most fuels."
                ),
                "flue_gas_oxygen": (
                    "Flue gas O2 at {value:.1f}% indicates {assessment} air-fuel ratio."
                ),
                "stack_temperature": (
                    "Stack temperature of {value:.0f}F represents {assessment} heat loss."
                ),
            },
            "recommendations": [
                "Perform combustion analysis and tune burners",
                "Verify air damper positions and linkages",
                "Check fuel supply pressure and quality",
                "Inspect burner components for wear",
            ]
        }

    @staticmethod
    def get_heat_exchanger_template() -> Dict[str, Any]:
        """Get template for heat exchanger performance explanations."""
        return {
            "summary": (
                "Heat exchanger effectiveness is {prediction:.0%}. "
                "{assessment}"
            ),
            "factors": {
                "approach_temperature": (
                    "Approach temperature of {value:.0f}F is {assessment}. "
                    "Higher approach indicates reduced heat transfer."
                ),
                "fouling_resistance": (
                    "Calculated fouling resistance is {value:.6f} m2-K/W, "
                    "which is {assessment} the design value."
                ),
                "pressure_drop": (
                    "Shell/tube pressure drop of {value:.1f} PSI is {assessment}. "
                    "Elevated drop indicates flow restrictions."
                ),
            },
            "recommendations": [
                "Compare current performance to design specifications",
                "Check for tube fouling or blockage",
                "Verify flow rates on both shell and tube sides",
                "Consider chemical cleaning if fouling is suspected",
            ]
        }


def create_process_heat_explainer(
    equipment_type: str,
    model_type: str = "fouling_risk",
    audience: str = "engineer"
) -> HumanReadableExplainer:
    """
    Factory function to create Process Heat explainer.

    Args:
        equipment_type: Type of equipment (boiler, furnace, heat_exchanger)
        model_type: Type of model (fouling_risk, efficiency, maintenance)
        audience: Target audience (operator, engineer, manager, executive)

    Returns:
        Configured HumanReadableExplainer

    Example:
        >>> explainer = create_process_heat_explainer(
        ...     equipment_type="boiler",
        ...     model_type="fouling_risk",
        ...     audience="operator"
        ... )
    """
    audience_map = {
        "operator": AudienceType.OPERATOR,
        "engineer": AudienceType.ENGINEER,
        "manager": AudienceType.MANAGER,
        "executive": AudienceType.EXECUTIVE,
        "auditor": AudienceType.AUDITOR,
        "technical": AudienceType.TECHNICAL,
    }

    return HumanReadableExplainer(
        domain="process_heat",
        equipment_type=equipment_type,
        model_type=model_type,
        audience=audience_map.get(audience, AudienceType.ENGINEER),
        include_recommendations=True,
        include_provenance=True,
    )
