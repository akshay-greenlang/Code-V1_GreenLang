# -*- coding: utf-8 -*-
"""
NaturalLanguageExplainer - Human-readable explanation generator for Process Heat agents.

This module provides the NaturalLanguageExplainer class that transforms technical ML
explanations (SHAP values, feature contributions) into clear, actionable natural language
explanations tailored for different audiences (operator, engineer, executive, auditor).

Features:
    - Multi-audience explanation generation (4 audience levels)
    - Domain-specific templates for emissions, efficiency, maintenance
    - Multiple output formats (text, markdown, HTML)
    - Multi-language support (extensible)
    - Zero-hallucination principle (no fabricated insights)
    - Complete provenance tracking for audit trails

Example:
    >>> from greenlang.ml.explainability import NaturalLanguageExplainer
    >>> explainer = NaturalLanguageExplainer()
    >>> explanation = explainer.explain_prediction(
    ...     prediction=0.85,
    ...     shap_values={"temp": 0.35, "days_clean": 0.28},
    ...     feature_names={"temp": "Flue Gas Temp (F)", "days_clean": "Days Since Cleaning"}
    ... )
    >>> print(explanation.text_summary)

Author: GreenLang ML Team
Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class Audience(str, Enum):
    """Target audience types for explanations."""
    OPERATOR = "operator"      # Equipment operators - simple, actionable
    ENGINEER = "engineer"      # Process engineers - technical details
    EXECUTIVE = "executive"    # C-suite - high-level summary
    AUDITOR = "auditor"        # Regulatory auditors - compliance-focused


class OutputFormat(str, Enum):
    """Supported output formats for explanations."""
    PLAIN_TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"


class DecisionType(str, Enum):
    """Types of decisions explained."""
    FOULING_RISK = "fouling_risk"
    EFFICIENCY_DEGRADATION = "efficiency_degradation"
    MAINTENANCE_NEEDED = "maintenance_needed"
    EMISSIONS_HIGH = "emissions_high"
    ENERGY_WASTE = "energy_waste"


class ExplanationOutput(BaseModel):
    """Output model for explanations."""
    text_summary: str = Field(..., description="Plain text explanation")
    markdown_summary: str = Field(..., description="Markdown formatted explanation")
    html_summary: str = Field(..., description="HTML formatted explanation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Explanation confidence")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    top_factors: List[Tuple[str, float]] = Field(..., description="Top contributing factors")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    audience: str = Field(..., description="Target audience")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class NaturalLanguageExplainer:
    """
    Human-readable explanation generator for Process Heat agents.

    Transforms technical ML explanations into clear, audience-appropriate narratives.
    Supports multiple audiences, output formats, and languages.

    Attributes:
        default_audience: Default target audience
        default_format: Default output format
        templates: Domain-specific explanation templates

    Example:
        >>> explainer = NaturalLanguageExplainer(default_audience=Audience.OPERATOR)
        >>> result = explainer.explain_prediction(
        ...     prediction=0.85,
        ...     shap_values={"temp": 0.35, "pressure": 0.28},
        ...     feature_names={"temp": "Flue Gas Temperature", "pressure": "System Pressure"}
        ... )
        >>> print(result.text_summary)
    """

    # Explanation templates for different decision types
    TEMPLATES = {
        DecisionType.FOULING_RISK: {
            "operator": {
                "title": "Boiler Fouling Alert",
                "summary": "The system detected {risk_level} fouling risk ({prediction:.0%}). "
                          "{action}",
                "factors": "Main causes: {factors}",
                "action_high": "Cleaning or maintenance needed soon.",
                "action_medium": "Monitor closely and plan maintenance.",
                "action_low": "Equipment operating normally."
            },
            "engineer": {
                "title": "Boiler Fouling Risk Analysis",
                "summary": "ML model predicts {risk_level} fouling risk ({prediction:.0%}) with "
                          "{confidence:.0%} confidence. Analysis based on {feature_count} features.",
                "factors": "Contributing factors (in order of impact): {factors}",
                "technical": "SHAP-based feature attribution indicates {top_factor} has {top_contribution:.0%} "
                           "impact on prediction."
            },
            "executive": {
                "title": "Boiler Performance Summary",
                "summary": "Equipment status: {risk_level_short}. "
                          "Risk level is at {prediction:.0%}.",
                "factors": "Key drivers: {factors_short}",
                "impact": "Potential cost impact: {cost_impact}"
            },
            "auditor": {
                "title": "Fouling Risk Assessment - Audit Report",
                "summary": "Prediction: {prediction:.4f}, Risk Level: {risk_level}, "
                          "Confidence: {confidence:.0%}",
                "factors": "All contributing features: {all_factors}",
                "compliance": "Assessment meets regulatory standards: ISO 8854, ASME"
            }
        },
        DecisionType.EFFICIENCY_DEGRADATION: {
            "operator": {
                "title": "Energy Efficiency Notice",
                "summary": "Efficiency has {direction} to {prediction:.0%}. "
                          "{action}",
                "factors": "Main issues: {factors}",
                "action_good": "Equipment is running well.",
                "action_fair": "Some efficiency improvements possible.",
                "action_poor": "Significant efficiency loss - action needed."
            },
            "engineer": {
                "title": "Efficiency Analysis Report",
                "summary": "Current efficiency: {prediction:.0%}. Expected baseline: {baseline:.0%}. "
                          "Loss: {loss:.1f}%",
                "factors": "Efficiency drivers: {factors}",
                "calc": "Calculation method: ASME PTC-4 with stack temperature correction"
            },
            "executive": {
                "title": "Operational Efficiency Dashboard",
                "summary": "Efficiency Status: {status}. Current: {prediction:.0%}.",
                "factors": "Key metrics: {factors_short}",
                "financial": "Annual fuel cost impact: {financial_impact}"
            },
            "auditor": {
                "title": "Energy Efficiency Audit Results",
                "summary": "Efficiency Score: {prediction:.4f}, Assessment Date: {date}",
                "factors": "Analyzed parameters: {all_factors}",
                "standard": "Assessment per: ASME PTC-4, ANSI Z535"
            }
        },
        DecisionType.MAINTENANCE_NEEDED: {
            "operator": {
                "title": "Maintenance Alert",
                "summary": "Equipment needs {urgency} maintenance. Priority: {priority}.",
                "factors": "Issues detected: {factors}",
                "action": "{next_steps}"
            },
            "engineer": {
                "title": "Predictive Maintenance Analysis",
                "summary": "Failure probability: {prediction:.0%}. Recommended action: {recommendation}.",
                "factors": "Degradation indicators: {factors}",
                "technical": "Weibull analysis indicates {weibull_assessment}"
            },
            "executive": {
                "title": "Equipment Health Status",
                "summary": "Status: {health_status}. Maintenance urgency: {urgency}.",
                "factors": "Condition summary: {factors_short}",
                "financial": "Estimated repair cost: {repair_cost}"
            },
            "auditor": {
                "title": "Maintenance Compliance Report",
                "summary": "Assessment: {prediction:.0%}, Compliance: {compliance_status}",
                "factors": "All assessed parameters: {all_factors}",
                "standard": "Maintenance per: ISO 13373-1, IEC 60571"
            }
        }
    }

    # Risk level thresholds and descriptions
    RISK_LEVELS = {
        (0.9, 1.0): ("CRITICAL", "urgent action required immediately"),
        (0.7, 0.9): ("HIGH", "prompt attention needed"),
        (0.4, 0.7): ("MEDIUM", "monitor and plan action"),
        (0.2, 0.4): ("LOW", "keep under observation"),
        (0.0, 0.2): ("MINIMAL", "no action needed")
    }

    # Audience-specific complexity levels
    AUDIENCE_CONFIG = {
        Audience.OPERATOR: {
            "complexity": "simple",
            "max_factors": 3,
            "include_technical": False,
            "include_numbers": True,
            "include_recommendations": True,
            "language_style": "action-oriented"
        },
        Audience.ENGINEER: {
            "complexity": "technical",
            "max_factors": 5,
            "include_technical": True,
            "include_numbers": True,
            "include_recommendations": True,
            "language_style": "precise"
        },
        Audience.EXECUTIVE: {
            "complexity": "minimal",
            "max_factors": 2,
            "include_technical": False,
            "include_numbers": True,
            "include_recommendations": False,
            "language_style": "business-oriented"
        },
        Audience.AUDITOR: {
            "complexity": "comprehensive",
            "max_factors": 10,
            "include_technical": True,
            "include_numbers": True,
            "include_recommendations": False,
            "language_style": "compliance-focused"
        }
    }

    def __init__(
        self,
        default_audience: Audience = Audience.ENGINEER,
        default_format: OutputFormat = OutputFormat.PLAIN_TEXT,
        include_provenance: bool = True
    ):
        """
        Initialize NaturalLanguageExplainer.

        Args:
            default_audience: Default target audience for explanations
            default_format: Default output format (text, markdown, html)
            include_provenance: Include SHA-256 provenance hashes
        """
        self.default_audience = default_audience
        self.default_format = default_format
        self.include_provenance = include_provenance
        logger.info(f"NaturalLanguageExplainer initialized for {default_audience.value}")

    def explain_prediction(
        self,
        prediction: float,
        shap_values: Dict[str, float],
        feature_names: Dict[str, str],
        decision_type: DecisionType = DecisionType.FOULING_RISK,
        audience: Optional[Audience] = None,
        confidence: float = 0.85,
        feature_values: Optional[Dict[str, float]] = None,
        baseline: Optional[float] = None
    ) -> ExplanationOutput:
        """
        Generate explanation for a model prediction.

        Args:
            prediction: Model's predicted value (0-1 for probabilities)
            shap_values: Feature contributions (SHAP values) {feature: contribution}
            feature_names: Readable names for features {feature_id: readable_name}
            decision_type: Type of decision being explained
            audience: Target audience (uses default if None)
            confidence: Confidence score for the explanation (0-1)
            feature_values: Actual feature values {feature: value}
            baseline: Baseline value for comparison

        Returns:
            ExplanationOutput with text, markdown, and HTML summaries

        Example:
            >>> explainer = NaturalLanguageExplainer()
            >>> result = explainer.explain_prediction(
            ...     prediction=0.85,
            ...     shap_values={"flue_temp": 0.35, "days_clean": 0.28},
            ...     feature_names={"flue_temp": "Flue Gas Temperature", "days_clean": "Days Since Cleaning"},
            ...     decision_type=DecisionType.FOULING_RISK,
            ...     audience=Audience.OPERATOR
            ... )
        """
        audience = audience or self.default_audience
        config = self.AUDIENCE_CONFIG[audience]

        # Get top factors
        top_factors = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:config["max_factors"]]

        # Get risk level
        risk_level, risk_description = self._get_risk_level(prediction)

        # Generate text explanation
        text_summary = self._generate_text_explanation(
            prediction=prediction,
            decision_type=decision_type,
            audience=audience,
            top_factors=top_factors,
            feature_names=feature_names,
            feature_values=feature_values,
            risk_level=risk_level,
            confidence=confidence,
            baseline=baseline
        )

        # Convert to other formats
        markdown_summary = self._to_markdown(text_summary)
        html_summary = self._to_html(text_summary)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            prediction=prediction,
            decision_type=decision_type,
            audience=audience,
            top_factors=top_factors
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            prediction, shap_values, audience.value, decision_type.value
        )

        return ExplanationOutput(
            text_summary=text_summary,
            markdown_summary=markdown_summary,
            html_summary=html_summary,
            confidence=confidence,
            provenance_hash=provenance_hash,
            top_factors=top_factors,
            recommendations=recommendations,
            audience=audience.value,
            metadata={
                "decision_type": decision_type.value,
                "risk_level": risk_level,
                "feature_count": len(shap_values),
                "top_factor": top_factors[0][0] if top_factors else None,
                "top_contribution": top_factors[0][1] if top_factors else None
            }
        )

    def explain_decision(
        self,
        decision_type: DecisionType,
        factors: Dict[str, Any],
        confidence: float,
        audience: Optional[Audience] = None
    ) -> str:
        """
        Generate explanation for a decision with structured factors.

        Args:
            decision_type: Type of decision being explained
            factors: Structured factors dict {name: value/description}
            confidence: Confidence score for the decision
            audience: Target audience (uses default if None)

        Returns:
            Human-readable decision explanation

        Example:
            >>> explainer = NaturalLanguageExplainer()
            >>> explanation = explainer.explain_decision(
            ...     decision_type=DecisionType.MAINTENANCE_NEEDED,
            ...     factors={
            ...         "vibration": "High - 7.2 mm/s",
            ...         "temperature": "Elevated - 85C",
            ...         "runtime": "4200 hours"
            ...     },
            ...     confidence=0.92,
            ...     audience=Audience.OPERATOR
            ... )
        """
        audience = audience or self.default_audience

        factors_text = self._format_factors_list(factors)

        if confidence >= 0.9:
            conf_text = "very high confidence"
        elif confidence >= 0.7:
            conf_text = "good confidence"
        else:
            conf_text = "moderate confidence"

        explanation = (
            f"Decision: {decision_type.value.replace('_', ' ').title()}\n"
            f"Confidence: {conf_text} ({confidence:.0%})\n\n"
            f"Contributing Factors:\n{factors_text}"
        )

        return explanation

    def generate_summary(
        self,
        explanations: List[ExplanationOutput],
        audience: Optional[Audience] = None,
        output_format: OutputFormat = OutputFormat.PLAIN_TEXT
    ) -> str:
        """
        Generate high-level summary from multiple explanations.

        Useful for combining multiple model outputs into a single coherent narrative.

        Args:
            explanations: List of ExplanationOutput objects
            audience: Target audience (uses default if None)
            output_format: Output format (text, markdown, html)

        Returns:
            Combined summary tailored to audience

        Example:
            >>> explainer = NaturalLanguageExplainer()
            >>> expl1 = explainer.explain_prediction(...)  # Fouling risk
            >>> expl2 = explainer.explain_prediction(...)  # Efficiency
            >>> summary = explainer.generate_summary(
            ...     [expl1, expl2],
            ...     audience=Audience.OPERATOR
            ... )
        """
        audience = audience or self.default_audience
        config = self.AUDIENCE_CONFIG[audience]

        # Aggregate key findings
        all_recommendations = []
        avg_confidence = 0.0
        decision_summary = {}

        for expl in explanations:
            all_recommendations.extend(expl.recommendations)
            avg_confidence += expl.confidence
            if expl.metadata.get("decision_type"):
                decision_summary[expl.metadata["decision_type"]] = expl.metadata.get("risk_level")

        avg_confidence /= len(explanations) if explanations else 1

        # Build summary based on audience
        summary_lines = []

        if audience == Audience.EXECUTIVE:
            summary_lines.append("EQUIPMENT STATUS SUMMARY")
            summary_lines.append("-" * 40)
            for decision, risk_level in decision_summary.items():
                summary_lines.append(f"  {decision.title()}: {risk_level}")
            summary_lines.append(f"\nOverall Confidence: {avg_confidence:.0%}")

        elif audience == Audience.OPERATOR:
            summary_lines.append("ACTIONS NEEDED")
            summary_lines.append("-" * 40)
            if all_recommendations:
                for i, rec in enumerate(all_recommendations[:5], 1):
                    summary_lines.append(f"  {i}. {rec}")
            else:
                summary_lines.append("  No immediate actions required.")

        elif audience == Audience.ENGINEER:
            summary_lines.append("ANALYSIS SUMMARY")
            summary_lines.append("-" * 40)
            for expl in explanations:
                summary_lines.append(f"\n{expl.metadata.get('decision_type', 'Analysis')}:")
                summary_lines.append(f"  Confidence: {expl.confidence:.0%}")
                if expl.top_factors:
                    summary_lines.append(f"  Top Factor: {expl.top_factors[0][0]}")

        elif audience == Audience.AUDITOR:
            summary_lines.append("AUDIT REPORT SUMMARY")
            summary_lines.append("-" * 40)
            for expl in explanations:
                summary_lines.append(f"\nHash: {expl.provenance_hash[:16]}...")
                summary_lines.append(f"Timestamp: {expl.timestamp.isoformat()}")

        summary_text = "\n".join(summary_lines)

        # Convert to requested format
        if output_format == OutputFormat.MARKDOWN:
            return self._to_markdown(summary_text)
        elif output_format == OutputFormat.HTML:
            return self._to_html(summary_text)
        else:
            return summary_text

    def translate_to_language(
        self,
        explanation: str,
        language: str = "en"
    ) -> str:
        """
        Translate explanation to different language.

        Currently supports English. Extensible for other languages.

        Args:
            explanation: Explanation text to translate
            language: Target language code (e.g., 'en', 'es', 'fr', 'de')

        Returns:
            Translated explanation

        Note:
            Current implementation supports 'en' (English).
            Other languages would require translation service integration.
        """
        if language == "en":
            return explanation

        # This is where language translation would be integrated
        logger.warning(f"Language '{language}' not yet supported. Returning English.")
        return explanation

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _get_risk_level(self, prediction: float) -> Tuple[str, str]:
        """Get risk level name and description from prediction."""
        for (min_val, max_val), (level, desc) in self.RISK_LEVELS.items():
            if min_val <= prediction < max_val:
                return level, desc
        return "UNKNOWN", "unable to determine risk level"

    def _generate_text_explanation(
        self,
        prediction: float,
        decision_type: DecisionType,
        audience: Audience,
        top_factors: List[Tuple[str, float]],
        feature_names: Dict[str, str],
        feature_values: Optional[Dict[str, float]],
        risk_level: str,
        confidence: float,
        baseline: Optional[float]
    ) -> str:
        """Generate plain text explanation."""
        lines = []

        # Title
        if decision_type in self.TEMPLATES:
            templates = self.TEMPLATES[decision_type].get(audience.value, {})
            title = templates.get("title", decision_type.value.title())
            lines.append(title)
            lines.append("=" * len(title))
            lines.append("")

        # Risk level and prediction
        config = self.AUDIENCE_CONFIG[audience]
        if config["include_numbers"]:
            lines.append(f"Prediction: {prediction:.1%}")
            if baseline is not None:
                lines.append(f"Baseline: {baseline:.1%}")
                lines.append(f"Difference: {(prediction - baseline):.1%}")
            lines.append(f"Confidence: {confidence:.0%}")
            lines.append("")

        # Main factors
        lines.append("Key Factors:")
        for i, (feature_id, contribution) in enumerate(top_factors[:config["max_factors"]], 1):
            feature_name = feature_names.get(feature_id, feature_id)
            feature_value = feature_values.get(feature_id) if feature_values else None

            contribution_str = f"{abs(contribution):.1%}"
            if feature_value is not None:
                lines.append(f"  {i}. {feature_name}: {feature_value:.2f} (impact: {contribution_str})")
            else:
                lines.append(f"  {i}. {feature_name} (impact: {contribution_str})")

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        prediction: float,
        decision_type: DecisionType,
        audience: Audience,
        top_factors: List[Tuple[str, float]]
    ) -> List[str]:
        """Generate actionable recommendations."""
        config = self.AUDIENCE_CONFIG[audience]
        if not config["include_recommendations"]:
            return []

        recommendations = []

        if decision_type == DecisionType.FOULING_RISK:
            if prediction > 0.7:
                recommendations.append("Schedule equipment cleaning or maintenance")
                recommendations.append("Inspect for deposit buildup")
            if prediction > 0.5:
                recommendations.append("Monitor fouling indicators closely")

        elif decision_type == DecisionType.EFFICIENCY_DEGRADATION:
            if prediction < 0.8:
                recommendations.append("Perform combustion analysis")
                recommendations.append("Check heat transfer surfaces")
            if prediction < 0.7:
                recommendations.append("Consider immediate optimization actions")

        elif decision_type == DecisionType.MAINTENANCE_NEEDED:
            if prediction > 0.7:
                recommendations.append("Schedule maintenance within 1 week")
                recommendations.append("Allocate resources for repairs")
            if prediction > 0.5:
                recommendations.append("Add to next preventive maintenance window")

        return recommendations

    def _format_factors_list(self, factors: Dict[str, Any]) -> str:
        """Format factors dictionary into readable list."""
        lines = []
        for name, value in factors.items():
            if isinstance(value, (int, float)):
                lines.append(f"  • {name}: {value:.2f}")
            else:
                lines.append(f"  • {name}: {value}")
        return "\n".join(lines) if lines else "  (no factors)"

    def _to_markdown(self, text: str) -> str:
        """Convert text explanation to markdown format."""
        lines = text.split("\n")
        markdown_lines = []

        for i, line in enumerate(lines):
            # First line becomes h1
            if i == 0:
                markdown_lines.append(f"# {line}")
            # Lines with all caps and specific length are headers
            elif line.isupper() and len(line) > 5 and not line.startswith("  "):
                markdown_lines.append(f"## {line}")
            # Equals signs become horizontal rules
            elif all(c == "=" for c in line.strip()) and line.strip():
                markdown_lines.append("---")
            # List items
            elif line.startswith("  •"):
                markdown_lines.append(line.replace("•", "-"))
            else:
                markdown_lines.append(line)

        return "\n".join(markdown_lines)

    def _to_html(self, text: str) -> str:
        """Convert text explanation to HTML format."""
        html = "<div class='explanation'>\n"

        lines = text.split("\n")
        in_list = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                if in_list:
                    html += "</ul>\n"
                    in_list = False
                html += "<br/>\n"
            elif line.isupper() and len(line) > 5 and not line.startswith("  "):
                if in_list:
                    html += "</ul>\n"
                    in_list = False
                html += f"<h2>{stripped}</h2>\n"
            elif line.startswith("  "):
                if not in_list:
                    html += "<ul>\n"
                    in_list = True
                item = stripped.lstrip("•- ")
                html += f"  <li>{item}</li>\n"
            else:
                if in_list:
                    html += "</ul>\n"
                    in_list = False
                if stripped:
                    html += f"<p>{stripped}</p>\n"

        if in_list:
            html += "</ul>\n"

        html += "</div>"
        return html

    def _calculate_provenance_hash(
        self,
        prediction: float,
        shap_values: Dict[str, float],
        audience: str,
        decision_type: str
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        data = {
            "prediction": prediction,
            "shap_values": shap_values,
            "audience": audience,
            "decision_type": decision_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def create_natural_language_explainer(
    audience: str = "engineer",
    output_format: str = "text"
) -> NaturalLanguageExplainer:
    """
    Factory function to create configured NaturalLanguageExplainer.

    Args:
        audience: Target audience ('operator', 'engineer', 'executive', 'auditor')
        output_format: Output format ('text', 'markdown', 'html')

    Returns:
        Configured NaturalLanguageExplainer instance

    Example:
        >>> explainer = create_natural_language_explainer(
        ...     audience='operator',
        ...     output_format='markdown'
        ... )
    """
    audience_map = {
        "operator": Audience.OPERATOR,
        "engineer": Audience.ENGINEER,
        "executive": Audience.EXECUTIVE,
        "auditor": Audience.AUDITOR,
    }

    return NaturalLanguageExplainer(
        default_audience=audience_map.get(audience, Audience.ENGINEER),
        default_format=OutputFormat(output_format) if output_format in [e.value for e in OutputFormat] else OutputFormat.PLAIN_TEXT
    )
