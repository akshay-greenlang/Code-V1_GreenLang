"""
Template-based Narrative Generator for GL-016 Waterguard

This module generates human-readable operator-facing narratives from
structured explanations. Uses ONLY template-based generation - NO LLM.

All narratives are deterministic and derived from structured data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from string import Template
from typing import Any, Dict, List, Optional, Union

from .explanation_schemas import (
    ExplanationPayload,
    FeatureContribution,
    FeatureDirection,
    LocalExplanation,
    RecommendationType,
)

logger = logging.getLogger(__name__)


class NarrativeStyle(str, Enum):
    """Narrative output styles."""
    BRIEF = "brief"  # Short, action-focused
    STANDARD = "standard"  # Balanced detail
    DETAILED = "detailed"  # Full technical detail
    OPERATOR = "operator"  # Non-technical for operators


@dataclass
class NarrativeTemplate:
    """Template for generating narrative text."""
    name: str
    template: str
    style: NarrativeStyle = NarrativeStyle.STANDARD
    applicable_to: List[RecommendationType] = field(default_factory=list)


@dataclass
class NarrativeConfig:
    """Configuration for narrative generation."""
    # Default style
    default_style: NarrativeStyle = NarrativeStyle.OPERATOR

    # Number of features to include
    max_features: int = 3

    # Site-specific terminology mapping
    terminology: Dict[str, str] = field(default_factory=dict)

    # Units display preference
    show_units: bool = True

    # Confidence display
    show_confidence: bool = True
    confidence_threshold_display: float = 0.8

    # Locale for number formatting
    decimal_places: int = 1


class NarrativeGenerator:
    """
    Template-based narrative generator for water treatment explanations.

    Converts structured explanations into human-readable text using
    pre-defined templates. NO generative AI is used.

    Features:
    - Multiple narrative styles (brief, standard, detailed, operator)
    - Site-configurable terminology
    - Multi-language ready structure
    - Consistent, auditable output
    """

    # Default templates for different recommendation types
    DEFAULT_TEMPLATES = {
        RecommendationType.BLOWDOWN_INCREASE: {
            NarrativeStyle.BRIEF: NarrativeTemplate(
                name="blowdown_increase_brief",
                template="Increase blowdown by ${change_percent}% due to ${primary_factor}.",
                style=NarrativeStyle.BRIEF,
                applicable_to=[RecommendationType.BLOWDOWN_INCREASE]
            ),
            NarrativeStyle.STANDARD: NarrativeTemplate(
                name="blowdown_increase_standard",
                template=(
                    "Blowdown increased ${change_percent}% because ${primary_factor_name} "
                    "${direction_verb} ${change_value} ${unit} ${comparison}. "
                    "${secondary_factors}"
                ),
                style=NarrativeStyle.STANDARD,
                applicable_to=[RecommendationType.BLOWDOWN_INCREASE]
            ),
            NarrativeStyle.OPERATOR: NarrativeTemplate(
                name="blowdown_increase_operator",
                template=(
                    "ACTION: Increase blowdown rate by ${change_percent}%.\n"
                    "REASON: ${primary_factor_name} is ${status} at ${value} ${unit}.\n"
                    "IMPACT: This will help ${impact_description}."
                ),
                style=NarrativeStyle.OPERATOR,
                applicable_to=[RecommendationType.BLOWDOWN_INCREASE]
            ),
        },
        RecommendationType.BLOWDOWN_DECREASE: {
            NarrativeStyle.BRIEF: NarrativeTemplate(
                name="blowdown_decrease_brief",
                template="Decrease blowdown by ${change_percent}% due to ${primary_factor}.",
                style=NarrativeStyle.BRIEF,
                applicable_to=[RecommendationType.BLOWDOWN_DECREASE]
            ),
            NarrativeStyle.STANDARD: NarrativeTemplate(
                name="blowdown_decrease_standard",
                template=(
                    "Blowdown decreased ${change_percent}% because ${primary_factor_name} "
                    "is ${status}. Current value: ${value} ${unit}. "
                    "${secondary_factors}"
                ),
                style=NarrativeStyle.STANDARD,
                applicable_to=[RecommendationType.BLOWDOWN_DECREASE]
            ),
            NarrativeStyle.OPERATOR: NarrativeTemplate(
                name="blowdown_decrease_operator",
                template=(
                    "ACTION: Decrease blowdown rate by ${change_percent}%.\n"
                    "REASON: ${primary_factor_name} is ${status} at ${value} ${unit}.\n"
                    "IMPACT: This will conserve water while maintaining quality."
                ),
                style=NarrativeStyle.OPERATOR,
                applicable_to=[RecommendationType.BLOWDOWN_DECREASE]
            ),
        },
        RecommendationType.DOSING_INCREASE: {
            NarrativeStyle.BRIEF: NarrativeTemplate(
                name="dosing_increase_brief",
                template="Increase chemical dosing by ${change_percent}% due to ${primary_factor}.",
                style=NarrativeStyle.BRIEF,
                applicable_to=[RecommendationType.DOSING_INCREASE]
            ),
            NarrativeStyle.OPERATOR: NarrativeTemplate(
                name="dosing_increase_operator",
                template=(
                    "ACTION: Increase treatment chemical dosing by ${change_percent}%.\n"
                    "REASON: ${primary_factor_name} requires attention at ${value} ${unit}.\n"
                    "IMPACT: This will improve water treatment effectiveness."
                ),
                style=NarrativeStyle.OPERATOR,
                applicable_to=[RecommendationType.DOSING_INCREASE]
            ),
        },
        RecommendationType.DOSING_DECREASE: {
            NarrativeStyle.BRIEF: NarrativeTemplate(
                name="dosing_decrease_brief",
                template="Decrease chemical dosing by ${change_percent}% due to ${primary_factor}.",
                style=NarrativeStyle.BRIEF,
                applicable_to=[RecommendationType.DOSING_DECREASE]
            ),
            NarrativeStyle.OPERATOR: NarrativeTemplate(
                name="dosing_decrease_operator",
                template=(
                    "ACTION: Decrease treatment chemical dosing by ${change_percent}%.\n"
                    "REASON: ${primary_factor_name} is at optimal level (${value} ${unit}).\n"
                    "IMPACT: This will reduce chemical costs while maintaining quality."
                ),
                style=NarrativeStyle.OPERATOR,
                applicable_to=[RecommendationType.DOSING_DECREASE]
            ),
        },
        RecommendationType.MAINTAIN_CURRENT: {
            NarrativeStyle.BRIEF: NarrativeTemplate(
                name="maintain_brief",
                template="Maintain current settings. All parameters within target.",
                style=NarrativeStyle.BRIEF,
                applicable_to=[RecommendationType.MAINTAIN_CURRENT]
            ),
            NarrativeStyle.OPERATOR: NarrativeTemplate(
                name="maintain_operator",
                template=(
                    "ACTION: No changes required.\n"
                    "STATUS: All water quality parameters are within acceptable ranges.\n"
                    "KEY METRICS: ${primary_factor_name} at ${value} ${unit}."
                ),
                style=NarrativeStyle.OPERATOR,
                applicable_to=[RecommendationType.MAINTAIN_CURRENT]
            ),
        },
        RecommendationType.EMERGENCY_ACTION: {
            NarrativeStyle.OPERATOR: NarrativeTemplate(
                name="emergency_operator",
                template=(
                    "URGENT ACTION REQUIRED!\n"
                    "ISSUE: ${primary_factor_name} is critically ${status} at ${value} ${unit}.\n"
                    "IMMEDIATE ACTION: ${action_description}\n"
                    "Contact supervisor if condition persists."
                ),
                style=NarrativeStyle.OPERATOR,
                applicable_to=[RecommendationType.EMERGENCY_ACTION]
            ),
        },
    }

    # Feature-specific terminology
    FEATURE_TERMINOLOGY = {
        'conductivity': {
            'name': 'Conductivity',
            'high_status': 'elevated',
            'low_status': 'below target',
            'optimal_status': 'within range',
            'impact_high': 'reduce dissolved solids concentration',
            'impact_low': 'optimize water usage',
        },
        'ph': {
            'name': 'pH',
            'high_status': 'too alkaline',
            'low_status': 'too acidic',
            'optimal_status': 'balanced',
            'impact_high': 'prevent scale formation',
            'impact_low': 'prevent corrosion',
        },
        'temperature': {
            'name': 'Temperature',
            'high_status': 'elevated',
            'low_status': 'below normal',
            'optimal_status': 'normal',
            'impact_high': 'manage thermal load',
            'impact_low': 'optimize heat transfer',
        },
        'tds': {
            'name': 'Total Dissolved Solids',
            'high_status': 'high',
            'low_status': 'low',
            'optimal_status': 'acceptable',
            'impact_high': 'prevent scaling and deposits',
            'impact_low': 'optimize water chemistry',
        },
        'hardness': {
            'name': 'Hardness',
            'high_status': 'elevated',
            'low_status': 'soft',
            'optimal_status': 'balanced',
            'impact_high': 'prevent calcium scale',
            'impact_low': 'maintain protective film',
        },
        'cycles_of_concentration': {
            'name': 'Cycles of Concentration',
            'high_status': 'too high',
            'low_status': 'too low',
            'optimal_status': 'optimal',
            'impact_high': 'reduce mineral concentration',
            'impact_low': 'improve water efficiency',
        },
    }

    def __init__(
        self,
        config: Optional[NarrativeConfig] = None,
        custom_templates: Optional[Dict[str, NarrativeTemplate]] = None
    ):
        """
        Initialize the narrative generator.

        Args:
            config: Optional configuration
            custom_templates: Optional custom templates to add/override
        """
        self.config = config or NarrativeConfig()
        self._templates = self._build_template_registry(custom_templates)
        self._terminology = {**self.FEATURE_TERMINOLOGY}

        # Apply site-specific terminology
        if self.config.terminology:
            for feature, terms in self.config.terminology.items():
                if feature in self._terminology:
                    self._terminology[feature].update(terms)
                else:
                    self._terminology[feature] = terms

    def _build_template_registry(
        self,
        custom_templates: Optional[Dict[str, NarrativeTemplate]]
    ) -> Dict[str, NarrativeTemplate]:
        """Build template registry from defaults and custom templates."""
        registry = {}

        # Add default templates
        for rec_type, style_templates in self.DEFAULT_TEMPLATES.items():
            for style, template in style_templates.items():
                key = f"{rec_type.value}:{style.value}"
                registry[key] = template

        # Override with custom templates
        if custom_templates:
            registry.update(custom_templates)

        return registry

    def _get_template(
        self,
        recommendation_type: RecommendationType,
        style: NarrativeStyle
    ) -> Optional[NarrativeTemplate]:
        """Get template for recommendation type and style."""
        key = f"{recommendation_type.value}:{style.value}"
        template = self._templates.get(key)

        if template is None:
            # Fall back to standard style
            key = f"{recommendation_type.value}:{NarrativeStyle.STANDARD.value}"
            template = self._templates.get(key)

        return template

    def _get_feature_terminology(
        self,
        feature_name: str,
        direction: FeatureDirection
    ) -> Dict[str, str]:
        """Get terminology for a feature based on its direction."""
        base_terms = self._terminology.get(
            feature_name,
            {
                'name': feature_name.replace('_', ' ').title(),
                'high_status': 'high',
                'low_status': 'low',
                'optimal_status': 'normal',
                'impact_high': 'optimize this parameter',
                'impact_low': 'optimize this parameter',
            }
        )

        if direction == FeatureDirection.INCREASING:
            status = base_terms.get('high_status', 'elevated')
            impact = base_terms.get('impact_high', 'address this condition')
        elif direction == FeatureDirection.DECREASING:
            status = base_terms.get('low_status', 'low')
            impact = base_terms.get('impact_low', 'address this condition')
        else:
            status = base_terms.get('optimal_status', 'stable')
            impact = 'maintain current conditions'

        return {
            'name': base_terms.get('name', feature_name),
            'status': status,
            'impact_description': impact,
        }

    def _format_value(self, value: float, unit: Optional[str] = None) -> str:
        """Format a numeric value with unit."""
        formatted = f"{value:.{self.config.decimal_places}f}"
        if unit and self.config.show_units:
            formatted += f" {unit}"
        return formatted

    def generate_narrative(
        self,
        explanation: Union[LocalExplanation, ExplanationPayload],
        recommendation_type: Optional[RecommendationType] = None,
        recommendation_value: Optional[float] = None,
        style: Optional[NarrativeStyle] = None
    ) -> str:
        """
        Generate human-readable narrative from explanation.

        Args:
            explanation: Explanation to narrate
            recommendation_type: Type of recommendation (inferred if not provided)
            recommendation_value: Numeric recommendation value
            style: Narrative style to use

        Returns:
            Human-readable narrative string
        """
        style = style or self.config.default_style

        # Handle ExplanationPayload
        if isinstance(explanation, ExplanationPayload):
            recommendation_type = explanation.recommendation_type
            recommendation_value = explanation.recommendation_value
            features = [
                FeatureContribution(
                    feature_name=f['name'],
                    value=f['value'],
                    contribution=f['contribution'],
                    direction=FeatureDirection(f['direction']),
                    unit=f.get('unit')
                )
                for f in explanation.top_factors[:self.config.max_features]
            ]
        else:
            features = explanation.get_top_features(self.config.max_features)

        if not features:
            return "No significant factors identified for this recommendation."

        # Default recommendation type
        if recommendation_type is None:
            recommendation_type = RecommendationType.MAINTAIN_CURRENT

        # Get template
        template = self._get_template(recommendation_type, style)
        if template is None:
            return self._generate_fallback_narrative(
                features, recommendation_type, recommendation_value
            )

        # Prepare template variables
        primary = features[0]
        primary_terms = self._get_feature_terminology(
            primary.feature_name, primary.direction
        )

        template_vars = {
            'primary_factor': primary.feature_name,
            'primary_factor_name': primary_terms['name'],
            'value': self._format_value(primary.value),
            'unit': primary.unit or '',
            'status': primary_terms['status'],
            'impact_description': primary_terms['impact_description'],
            'change_percent': abs(recommendation_value or 0),
            'change_value': abs(primary.contribution),
            'direction_verb': 'rose' if primary.direction == FeatureDirection.INCREASING else 'dropped',
            'comparison': 'above target' if primary.direction == FeatureDirection.INCREASING else 'below target',
            'action_description': self._get_action_description(recommendation_type),
        }

        # Add secondary factors
        if len(features) > 1:
            secondary = [
                f"{self._get_feature_terminology(f.feature_name, f.direction)['name']} "
                f"({self._format_value(f.value, f.unit)})"
                for f in features[1:3]
            ]
            template_vars['secondary_factors'] = (
                f"Additional factors: {', '.join(secondary)}."
            )
        else:
            template_vars['secondary_factors'] = ""

        # Apply template
        try:
            result = Template(template.template).safe_substitute(template_vars)
        except Exception as e:
            logger.warning(f"Template substitution failed: {e}")
            result = self._generate_fallback_narrative(
                features, recommendation_type, recommendation_value
            )

        return result.strip()

    def _get_action_description(
        self,
        recommendation_type: RecommendationType
    ) -> str:
        """Get action description for recommendation type."""
        actions = {
            RecommendationType.BLOWDOWN_INCREASE: "Increase blowdown rate immediately",
            RecommendationType.BLOWDOWN_DECREASE: "Reduce blowdown rate",
            RecommendationType.DOSING_INCREASE: "Increase chemical dosing",
            RecommendationType.DOSING_DECREASE: "Reduce chemical dosing",
            RecommendationType.MAINTAIN_CURRENT: "Maintain current settings",
            RecommendationType.EMERGENCY_ACTION: "Take immediate corrective action",
        }
        return actions.get(recommendation_type, "Review and adjust as needed")

    def _generate_fallback_narrative(
        self,
        features: List[FeatureContribution],
        recommendation_type: RecommendationType,
        recommendation_value: Optional[float]
    ) -> str:
        """Generate simple fallback narrative when template fails."""
        if not features:
            return "Recommendation based on current water quality parameters."

        primary = features[0]
        action = recommendation_type.value.replace('_', ' ').title()
        value_str = f" by {recommendation_value}%" if recommendation_value else ""

        return (
            f"{action}{value_str}. "
            f"Primary factor: {primary.feature_name} at "
            f"{self._format_value(primary.value, primary.unit)}."
        )

    def generate_summary(
        self,
        explanations: List[Union[LocalExplanation, ExplanationPayload]],
        style: Optional[NarrativeStyle] = None
    ) -> str:
        """
        Generate summary narrative from multiple explanations.

        Args:
            explanations: List of explanations to summarize
            style: Narrative style

        Returns:
            Summary narrative string
        """
        if not explanations:
            return "No recommendations to summarize."

        style = style or self.config.default_style

        summaries = []
        for i, exp in enumerate(explanations, 1):
            narrative = self.generate_narrative(exp, style=NarrativeStyle.BRIEF)
            summaries.append(f"{i}. {narrative}")

        header = f"Summary of {len(explanations)} recommendations:\n"
        return header + "\n".join(summaries)

    def add_custom_template(
        self,
        template: NarrativeTemplate,
        recommendation_type: RecommendationType,
        style: NarrativeStyle
    ) -> None:
        """Add or override a template."""
        key = f"{recommendation_type.value}:{style.value}"
        self._templates[key] = template
        logger.info(f"Added custom template: {key}")

    def set_terminology(
        self,
        feature_name: str,
        terminology: Dict[str, str]
    ) -> None:
        """Set custom terminology for a feature."""
        if feature_name in self._terminology:
            self._terminology[feature_name].update(terminology)
        else:
            self._terminology[feature_name] = terminology
        logger.info(f"Updated terminology for: {feature_name}")
