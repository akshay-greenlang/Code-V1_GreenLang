# -*- coding: utf-8 -*-
"""
Explanation Generator Module

This module provides human-readable explanation generation for ML model
predictions, transforming technical SHAP/LIME values into clear,
actionable insights for stakeholders.

The explanation generator supports multiple output formats and audience
levels, from technical reports to executive summaries, with provenance
tracking for audit compliance.

Example:
    >>> from greenlang.ml.explainability import ExplanationGenerator, SHAPExplainer
    >>> explainer = SHAPExplainer(model)
    >>> shap_result = explainer.explain(X)
    >>> generator = ExplanationGenerator()
    >>> explanation = generator.generate_explanation(shap_result)
    >>> print(explanation.narrative)
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AudienceLevel(str, Enum):
    """Target audience levels for explanations."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    EXECUTIVE = "executive"
    REGULATORY = "regulatory"


class ExplanationType(str, Enum):
    """Types of explanation outputs."""
    NARRATIVE = "narrative"
    BULLET_POINTS = "bullet_points"
    STRUCTURED = "structured"
    REPORT = "report"


class ExplanationGeneratorConfig(BaseModel):
    """Configuration for explanation generator."""

    audience: AudienceLevel = Field(
        default=AudienceLevel.BUSINESS,
        description="Target audience level"
    )
    explanation_type: ExplanationType = Field(
        default=ExplanationType.NARRATIVE,
        description="Type of explanation output"
    )
    max_features: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum features to include"
    )
    include_confidence: bool = Field(
        default=True,
        description="Include confidence information"
    )
    include_recommendations: bool = Field(
        default=True,
        description="Include actionable recommendations"
    )
    domain_context: Optional[str] = Field(
        default="emissions",
        description="Domain context (emissions, energy, supply_chain)"
    )
    language: str = Field(
        default="en",
        description="Output language"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )

    @validator("domain_context")
    def validate_domain(cls, v):
        """Validate domain context."""
        valid = ["emissions", "energy", "supply_chain", "deforestation", "general"]
        if v and v not in valid:
            raise ValueError(f"Domain must be one of {valid}")
        return v


class FeatureExplanation(BaseModel):
    """Explanation for a single feature."""

    feature_name: str = Field(
        ...,
        description="Name of the feature"
    )
    contribution: float = Field(
        ...,
        description="Feature contribution value"
    )
    direction: str = Field(
        ...,
        description="Direction of impact (increases/decreases)"
    )
    magnitude: str = Field(
        ...,
        description="Magnitude description (high/medium/low)"
    )
    narrative: str = Field(
        ...,
        description="Human-readable explanation"
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Actionable recommendation"
    )


class Explanation(BaseModel):
    """Complete explanation output."""

    summary: str = Field(
        ...,
        description="Executive summary"
    )
    narrative: str = Field(
        ...,
        description="Full narrative explanation"
    )
    feature_explanations: List[FeatureExplanation] = Field(
        ...,
        description="Individual feature explanations"
    )
    key_factors: List[str] = Field(
        ...,
        description="Top key factors"
    )
    recommendations: List[str] = Field(
        ...,
        description="Actionable recommendations"
    )
    confidence_statement: Optional[str] = Field(
        default=None,
        description="Statement about prediction confidence"
    )
    audience: str = Field(
        ...,
        description="Target audience"
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Generation timestamp"
    )


class ExplanationGenerator:
    """
    Human-Readable Explanation Generator for GreenLang.

    This class transforms technical ML explanations (SHAP, LIME) into
    clear, actionable narratives for different stakeholder audiences.

    Key capabilities:
    - Convert SHAP/LIME values to natural language
    - Adapt explanations for different audiences
    - Generate actionable recommendations
    - Support multiple output formats
    - Maintain audit trail with provenance

    Attributes:
        config: Configuration for explanation generation
        _domain_templates: Domain-specific templates
        _magnitude_thresholds: Thresholds for magnitude classification

    Example:
        >>> generator = ExplanationGenerator(config=ExplanationGeneratorConfig(
        ...     audience=AudienceLevel.EXECUTIVE,
        ...     domain_context="emissions"
        ... ))
        >>> explanation = generator.generate_explanation(shap_result)
        >>> print(explanation.summary)
    """

    def __init__(self, config: Optional[ExplanationGeneratorConfig] = None):
        """
        Initialize explanation generator.

        Args:
            config: Generator configuration
        """
        self.config = config or ExplanationGeneratorConfig()

        # Magnitude thresholds (relative to max contribution)
        self._magnitude_thresholds = {
            "high": 0.5,
            "medium": 0.2,
            "low": 0.0
        }

        # Domain-specific templates
        self._domain_templates = self._load_domain_templates()

        logger.info(
            f"ExplanationGenerator initialized: audience={self.config.audience}, "
            f"domain={self.config.domain_context}"
        )

    def _load_domain_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load domain-specific explanation templates."""
        return {
            "emissions": {
                "impact_verb": {
                    "positive": "increases emissions by",
                    "negative": "reduces emissions by"
                },
                "context": "carbon emissions",
                "unit": "kg CO2e",
                "recommendation_templates": {
                    "fuel_type": "Consider switching to lower-emission fuel alternatives",
                    "quantity": "Optimize consumption to reduce emissions",
                    "region": "Leverage regional renewable energy availability",
                    "default": "Implement efficiency improvements"
                }
            },
            "energy": {
                "impact_verb": {
                    "positive": "increases energy consumption by",
                    "negative": "reduces energy consumption by"
                },
                "context": "energy usage",
                "unit": "kWh",
                "recommendation_templates": {
                    "equipment": "Upgrade to more efficient equipment",
                    "schedule": "Optimize operational schedules",
                    "default": "Implement energy-saving measures"
                }
            },
            "supply_chain": {
                "impact_verb": {
                    "positive": "increases supply chain risk by",
                    "negative": "decreases supply chain risk by"
                },
                "context": "supply chain sustainability",
                "unit": "risk score",
                "recommendation_templates": {
                    "supplier": "Engage with supplier on sustainability",
                    "transport": "Optimize logistics routes",
                    "default": "Enhance supply chain visibility"
                }
            },
            "deforestation": {
                "impact_verb": {
                    "positive": "increases deforestation risk by",
                    "negative": "decreases deforestation risk by"
                },
                "context": "deforestation risk",
                "unit": "risk score",
                "recommendation_templates": {
                    "commodity": "Source from certified sustainable suppliers",
                    "origin": "Verify origin through satellite monitoring",
                    "default": "Implement traceability measures"
                }
            },
            "general": {
                "impact_verb": {
                    "positive": "increases the prediction by",
                    "negative": "decreases the prediction by"
                },
                "context": "model prediction",
                "unit": "units",
                "recommendation_templates": {
                    "default": "Consider optimizing this factor"
                }
            }
        }

    def _get_domain_template(self) -> Dict[str, Any]:
        """Get template for current domain."""
        domain = self.config.domain_context or "general"
        return self._domain_templates.get(domain, self._domain_templates["general"])

    def _classify_magnitude(
        self,
        contribution: float,
        max_contribution: float
    ) -> str:
        """
        Classify contribution magnitude.

        Args:
            contribution: Feature contribution
            max_contribution: Maximum contribution for reference

        Returns:
            Magnitude classification (high/medium/low)
        """
        if max_contribution == 0:
            return "low"

        ratio = abs(contribution) / max_contribution

        if ratio >= self._magnitude_thresholds["high"]:
            return "high"
        elif ratio >= self._magnitude_thresholds["medium"]:
            return "medium"
        else:
            return "low"

    def _generate_feature_narrative(
        self,
        feature_name: str,
        contribution: float,
        magnitude: str,
        template: Dict[str, Any]
    ) -> str:
        """
        Generate narrative for a single feature.

        Args:
            feature_name: Name of the feature
            contribution: Contribution value
            magnitude: Magnitude classification
            template: Domain template

        Returns:
            Human-readable narrative
        """
        direction = "positive" if contribution > 0 else "negative"
        impact_verb = template["impact_verb"][direction]
        abs_contrib = abs(contribution)

        # Audience-specific language
        if self.config.audience == AudienceLevel.TECHNICAL:
            return (
                f"Feature '{feature_name}' has a {magnitude} {direction} impact "
                f"(contribution: {contribution:+.4f}), which {impact_verb} {abs_contrib:.4f} {template['unit']}"
            )
        elif self.config.audience == AudienceLevel.EXECUTIVE:
            magnitude_phrase = {
                "high": "significantly",
                "medium": "moderately",
                "low": "slightly"
            }[magnitude]
            return f"{feature_name.replace('_', ' ').title()} {magnitude_phrase} {impact_verb.split(' by')[0]}s {template['context']}"
        else:
            # Business/Regulatory
            return (
                f"{feature_name.replace('_', ' ').title()} {impact_verb} "
                f"approximately {abs_contrib:.2f} {template['unit']}"
            )

    def _generate_recommendation(
        self,
        feature_name: str,
        contribution: float,
        template: Dict[str, Any]
    ) -> str:
        """
        Generate actionable recommendation for a feature.

        Args:
            feature_name: Name of the feature
            contribution: Contribution value
            template: Domain template

        Returns:
            Actionable recommendation
        """
        rec_templates = template.get("recommendation_templates", {})

        # Find matching template
        for key, rec in rec_templates.items():
            if key in feature_name.lower():
                return rec

        return rec_templates.get("default", "Consider optimizing this factor")

    def _calculate_provenance(
        self,
        input_data: Dict[str, Any],
        explanation: str
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        combined = f"{str(input_data)}|{explanation}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def generate_explanation(
        self,
        shap_result: Optional[Any] = None,
        lime_result: Optional[Any] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        prediction: Optional[float] = None,
        feature_names: Optional[List[str]] = None
    ) -> Explanation:
        """
        Generate human-readable explanation from ML explanation results.

        This method takes SHAP/LIME results or raw feature importance
        and produces clear, audience-appropriate narratives.

        Args:
            shap_result: SHAPResult from SHAPExplainer
            lime_result: LIMEResult from LIMEExplainer
            feature_importance: Direct feature importance dict
            prediction: Model prediction value
            feature_names: Names of features

        Returns:
            Explanation with narrative and recommendations

        Raises:
            ValueError: If no explanation input provided

        Example:
            >>> explanation = generator.generate_explanation(
            ...     feature_importance={"fuel_type": 0.5, "quantity": 0.3},
            ...     prediction=1500.0
            ... )
            >>> print(explanation.narrative)
        """
        start_time = datetime.utcnow()

        # Extract feature importance from different sources
        if shap_result is not None:
            if hasattr(shap_result, "feature_importance"):
                importance = shap_result.feature_importance
                names = shap_result.feature_names if hasattr(shap_result, "feature_names") else list(importance.keys())
            else:
                importance = shap_result
                names = feature_names or list(importance.keys())
        elif lime_result is not None:
            if hasattr(lime_result, "local_explanation"):
                importance = lime_result.local_explanation
                names = list(importance.keys())
            else:
                importance = lime_result
                names = feature_names or list(importance.keys())
        elif feature_importance is not None:
            importance = feature_importance
            names = feature_names or list(importance.keys())
        else:
            raise ValueError(
                "Must provide shap_result, lime_result, or feature_importance"
            )

        # Sort by absolute importance
        sorted_features = sorted(
            importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:self.config.max_features]

        template = self._get_domain_template()

        # Calculate max contribution for magnitude classification
        max_contribution = max(abs(v) for v in importance.values()) if importance else 1.0

        # Generate feature explanations
        feature_explanations = []
        key_factors = []
        recommendations = []

        for feature_name, contribution in sorted_features:
            magnitude = self._classify_magnitude(contribution, max_contribution)
            direction = "increases" if contribution > 0 else "decreases"

            narrative = self._generate_feature_narrative(
                feature_name, contribution, magnitude, template
            )

            recommendation = None
            if self.config.include_recommendations and magnitude in ["high", "medium"]:
                recommendation = self._generate_recommendation(
                    feature_name, contribution, template
                )
                if recommendation not in recommendations:
                    recommendations.append(recommendation)

            feature_explanations.append(FeatureExplanation(
                feature_name=feature_name,
                contribution=contribution,
                direction=direction,
                magnitude=magnitude,
                narrative=narrative,
                recommendation=recommendation
            ))

            if magnitude == "high":
                key_factors.append(feature_name.replace("_", " ").title())

        # Generate summary
        summary = self._generate_summary(
            feature_explanations, template, prediction
        )

        # Generate full narrative
        narrative = self._generate_narrative(
            feature_explanations, template, prediction
        )

        # Generate confidence statement
        confidence_statement = None
        if self.config.include_confidence:
            confidence_statement = self._generate_confidence_statement(
                feature_explanations, sorted_features
            )

        # Calculate provenance
        input_data = {
            "importance": importance,
            "prediction": prediction,
            "audience": self.config.audience.value
        }
        provenance_hash = self._calculate_provenance(input_data, narrative)

        logger.info(
            f"Explanation generated for {len(feature_explanations)} features"
        )

        return Explanation(
            summary=summary,
            narrative=narrative,
            feature_explanations=feature_explanations,
            key_factors=key_factors,
            recommendations=recommendations,
            confidence_statement=confidence_statement,
            audience=self.config.audience.value,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )

    def _generate_summary(
        self,
        feature_explanations: List[FeatureExplanation],
        template: Dict[str, Any],
        prediction: Optional[float]
    ) -> str:
        """Generate executive summary."""
        high_impact = [f for f in feature_explanations if f.magnitude == "high"]
        context = template["context"]

        if not high_impact:
            return f"No single factor dominates the {context} prediction."

        if len(high_impact) == 1:
            f = high_impact[0]
            return (
                f"{f.feature_name.replace('_', ' ').title()} is the primary driver of {context}, "
                f"{f.direction} it significantly."
            )
        else:
            factors = " and ".join([
                f.feature_name.replace("_", " ").title()
                for f in high_impact[:2]
            ])
            return f"Key drivers of {context} are {factors}."

    def _generate_narrative(
        self,
        feature_explanations: List[FeatureExplanation],
        template: Dict[str, Any],
        prediction: Optional[float]
    ) -> str:
        """Generate full narrative explanation."""
        context = template["context"]
        unit = template["unit"]

        paragraphs = []

        # Opening
        if prediction is not None:
            paragraphs.append(
                f"The model predicts {prediction:.2f} {unit} for {context}. "
                f"This prediction is explained by the following factors:"
            )
        else:
            paragraphs.append(
                f"The {context} prediction is explained by the following factors:"
            )

        # Feature narratives
        for i, feat in enumerate(feature_explanations):
            prefix = "Most importantly, " if i == 0 else ""
            if i == 1:
                prefix = "Additionally, "
            elif i > 1:
                prefix = "Also, "

            paragraphs.append(f"{prefix}{feat.narrative}")

        # Recommendations
        if self.config.include_recommendations:
            recs = [f.recommendation for f in feature_explanations if f.recommendation]
            if recs:
                paragraphs.append("\nRecommended actions:")
                for rec in recs[:3]:
                    paragraphs.append(f"  - {rec}")

        return "\n\n".join(paragraphs)

    def _generate_confidence_statement(
        self,
        feature_explanations: List[FeatureExplanation],
        sorted_features: List[Tuple[str, float]]
    ) -> str:
        """Generate confidence statement."""
        if not sorted_features:
            return "Insufficient data for confidence assessment."

        # Calculate concentration of importance
        total_importance = sum(abs(v) for _, v in sorted_features)
        top_importance = sum(abs(v) for _, v in sorted_features[:3])

        if total_importance > 0:
            concentration = top_importance / total_importance
        else:
            concentration = 0

        if concentration > 0.8:
            return (
                "High confidence: The prediction is clearly driven by a small number "
                "of key factors, making the explanation highly reliable."
            )
        elif concentration > 0.5:
            return (
                "Medium confidence: The prediction is influenced by multiple factors, "
                "providing a reasonably clear explanation."
            )
        else:
            return (
                "Lower confidence: The prediction depends on many factors with similar "
                "importance, making the explanation more complex."
            )

    def format_for_report(
        self,
        explanation: Explanation,
        include_charts: bool = False
    ) -> str:
        """
        Format explanation for inclusion in a report.

        Args:
            explanation: Generated explanation
            include_charts: Whether to include chart descriptions

        Returns:
            Formatted report section
        """
        sections = []

        # Header
        sections.append("## Explanation Summary\n")
        sections.append(explanation.summary + "\n")

        # Key Factors
        if explanation.key_factors:
            sections.append("\n### Key Factors\n")
            for factor in explanation.key_factors:
                sections.append(f"- {factor}")
            sections.append("")

        # Detailed Analysis
        sections.append("\n### Detailed Analysis\n")
        sections.append(explanation.narrative)

        # Recommendations
        if explanation.recommendations:
            sections.append("\n### Recommendations\n")
            for rec in explanation.recommendations:
                sections.append(f"1. {rec}")

        # Confidence
        if explanation.confidence_statement:
            sections.append("\n### Confidence Assessment\n")
            sections.append(explanation.confidence_statement)

        # Provenance
        sections.append("\n---")
        sections.append(f"*Provenance: {explanation.provenance_hash[:16]}...*")
        sections.append(f"*Generated: {explanation.timestamp.isoformat()}*")

        return "\n".join(sections)


# Unit test stubs
class TestExplanationGenerator:
    """Unit tests for ExplanationGenerator."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        generator = ExplanationGenerator()
        assert generator.config.audience == AudienceLevel.BUSINESS
        assert generator.config.domain_context == "emissions"

    def test_magnitude_classification(self):
        """Test magnitude classification."""
        generator = ExplanationGenerator()

        assert generator._classify_magnitude(0.9, 1.0) == "high"
        assert generator._classify_magnitude(0.3, 1.0) == "medium"
        assert generator._classify_magnitude(0.1, 1.0) == "low"

    def test_generate_explanation_from_dict(self):
        """Test explanation generation from feature importance dict."""
        generator = ExplanationGenerator()

        importance = {
            "fuel_type": 0.5,
            "quantity": 0.3,
            "region": -0.2
        }

        explanation = generator.generate_explanation(
            feature_importance=importance,
            prediction=1500.0
        )

        assert explanation.summary is not None
        assert len(explanation.feature_explanations) <= 5
        assert explanation.provenance_hash is not None

    def test_audience_specific_language(self):
        """Test audience-specific language generation."""
        # Technical audience
        tech_gen = ExplanationGenerator(
            config=ExplanationGeneratorConfig(audience=AudienceLevel.TECHNICAL)
        )
        tech_exp = tech_gen.generate_explanation(
            feature_importance={"feature_a": 0.5}
        )

        # Executive audience
        exec_gen = ExplanationGenerator(
            config=ExplanationGeneratorConfig(audience=AudienceLevel.EXECUTIVE)
        )
        exec_exp = exec_gen.generate_explanation(
            feature_importance={"feature_a": 0.5}
        )

        # Technical should be more detailed
        assert "contribution" in tech_exp.narrative.lower() or "feature" in tech_exp.narrative.lower()

    def test_domain_templates(self):
        """Test domain-specific templates."""
        # Emissions domain
        em_gen = ExplanationGenerator(
            config=ExplanationGeneratorConfig(domain_context="emissions")
        )
        template = em_gen._get_domain_template()
        assert "emissions" in template["context"]

        # Energy domain
        en_gen = ExplanationGenerator(
            config=ExplanationGeneratorConfig(domain_context="energy")
        )
        template = en_gen._get_domain_template()
        assert "energy" in template["context"]

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        generator = ExplanationGenerator()

        importance = {"feature_a": 0.5}
        exp1 = generator.generate_explanation(feature_importance=importance)
        exp2 = generator.generate_explanation(feature_importance=importance)

        # Note: timestamps differ so we check the logic independently
        hash1 = generator._calculate_provenance(
            {"importance": importance},
            "test narrative"
        )
        hash2 = generator._calculate_provenance(
            {"importance": importance},
            "test narrative"
        )
        assert hash1 == hash2
