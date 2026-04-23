"""
ExplanationGenerator - Generate comprehensive explanations for combustion optimization.

This module orchestrates all explainers to generate comprehensive explanations
suitable for different audiences (operators, engineers, executives).

Example:
    >>> generator = ExplanationGenerator(config)
    >>> explanation = generator.generate_full_explanation(context)
    >>> operator_view = generator.format_for_operator(explanation)
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from .explainability_payload import (
    ExplanationContext,
    ComprehensiveExplanation,
    OperatorExplanation,
    EngineerExplanation,
    PhysicsExplanation,
    SHAPExplanation,
    LIMEExplanation,
    ConstraintExplanation,
    RecommendationExplanation,
    ExplanationType,
    ConfidenceLevel,
    AudienceLevel,
)
from .physics_explainer import PhysicsExplainer, PhysicsExplainerConfig
from .shap_explainer import SHAPExplainer, SHAPExplainerConfig
from .lime_explainer import LIMEExplainer, LIMEExplainerConfig
from .constraint_explainer import ConstraintExplainer, ConstraintExplainerConfig
from .recommendation_explainer import RecommendationExplainer, RecommendationExplainerConfig

logger = logging.getLogger(__name__)


class BaseModel(Protocol):
    """Protocol for models that can be explained."""

    def predict(self, X: Any) -> Any:
        """Make predictions on input features."""
        ...


class ExplanationGeneratorConfig:
    """Configuration for ExplanationGenerator."""

    def __init__(
        self,
        include_physics_by_default: bool = True,
        include_shap_by_default: bool = True,
        include_lime_by_default: bool = False,
        include_constraints_by_default: bool = True,
        include_recommendations_by_default: bool = True,
        default_audience: AudienceLevel = AudienceLevel.OPERATOR,
        max_key_insights: int = 5,
        max_action_items: int = 5,
    ):
        """
        Initialize ExplanationGenerator configuration.

        Args:
            include_physics_by_default: Include physics explanations by default
            include_shap_by_default: Include SHAP explanations by default
            include_lime_by_default: Include LIME explanations by default
            include_constraints_by_default: Include constraint explanations by default
            include_recommendations_by_default: Include recommendation explanations
            default_audience: Default target audience
            max_key_insights: Maximum number of key insights to include
            max_action_items: Maximum number of action items to include
        """
        self.include_physics_by_default = include_physics_by_default
        self.include_shap_by_default = include_shap_by_default
        self.include_lime_by_default = include_lime_by_default
        self.include_constraints_by_default = include_constraints_by_default
        self.include_recommendations_by_default = include_recommendations_by_default
        self.default_audience = default_audience
        self.max_key_insights = max_key_insights
        self.max_action_items = max_action_items


class ExplanationGenerator:
    """
    Generator for comprehensive combustion optimization explanations.

    This class orchestrates all individual explainers to create unified
    explanations suitable for different audiences.

    Attributes:
        config: Configuration parameters
        physics_explainer: Physics-based explanation generator
        shap_explainer: SHAP explanation generator
        lime_explainer: LIME explanation generator
        constraint_explainer: Constraint explanation generator
        recommendation_explainer: Recommendation explanation generator

    Example:
        >>> config = ExplanationGeneratorConfig()
        >>> generator = ExplanationGenerator(config)
        >>> explanation = generator.generate_full_explanation(context)
    """

    def __init__(
        self,
        config: Optional[ExplanationGeneratorConfig] = None,
        physics_config: Optional[PhysicsExplainerConfig] = None,
        shap_config: Optional[SHAPExplainerConfig] = None,
        lime_config: Optional[LIMEExplainerConfig] = None,
        constraint_config: Optional[ConstraintExplainerConfig] = None,
        recommendation_config: Optional[RecommendationExplainerConfig] = None,
        model: Optional[BaseModel] = None,
    ):
        """
        Initialize ExplanationGenerator.

        Args:
            config: Main configuration. Uses defaults if not provided.
            physics_config: Configuration for physics explainer
            shap_config: Configuration for SHAP explainer
            lime_config: Configuration for LIME explainer
            constraint_config: Configuration for constraint explainer
            recommendation_config: Configuration for recommendation explainer
            model: ML model for SHAP/LIME explanations
        """
        self.config = config or ExplanationGeneratorConfig()
        self.model = model

        # Initialize individual explainers
        self.physics_explainer = PhysicsExplainer(physics_config)
        self.shap_explainer = SHAPExplainer(shap_config)
        self.lime_explainer = LIMEExplainer(lime_config)
        self.constraint_explainer = ConstraintExplainer(constraint_config)
        self.recommendation_explainer = RecommendationExplainer(recommendation_config)

        logger.info("ExplanationGenerator initialized")

    def set_model(self, model: BaseModel) -> None:
        """
        Set the ML model for SHAP/LIME explanations.

        Args:
            model: Model with predict method
        """
        self.model = model
        logger.info(f"Model set: {model.__class__.__name__}")

    def generate_full_explanation(
        self,
        context: ExplanationContext,
    ) -> ComprehensiveExplanation:
        """
        Generate a comprehensive explanation from context.

        Orchestrates all explainers based on context settings to create
        a unified explanation.

        Args:
            context: ExplanationContext with all required data

        Returns:
            ComprehensiveExplanation combining all explanation types

        Example:
            >>> context = ExplanationContext(
            ...     context_id="ctx-001",
            ...     boiler_id="boiler-1",
            ...     current_state={"o2_percent": 3.5, "efficiency": 0.85},
            ...     optimization_result={"status": "optimal"}
            ... )
            >>> explanation = generator.generate_full_explanation(context)
        """
        start_time = datetime.now()
        logger.info(f"Generating full explanation for context {context.context_id}")

        # Initialize explanation components
        physics_explanation = None
        shap_explanation = None
        lime_explanation = None
        constraint_explanations = []
        recommendation_explanations = []

        # Generate physics explanation if requested
        if context.include_physics:
            physics_explanation = self._generate_physics_explanation(context)

        # Generate SHAP explanation if requested and model available
        if context.include_shap and self.model is not None:
            shap_explanation = self._generate_shap_explanation(context)

        # Generate LIME explanation if requested and model available
        if context.include_lime and self.model is not None:
            lime_explanation = self._generate_lime_explanation(context)

        # Generate constraint explanations if requested
        if context.include_constraints and context.optimization_result:
            constraint_explanations = self._generate_constraint_explanations(context)

        # Generate recommendation explanations
        if context.recommendations:
            recommendation_explanations = self._generate_recommendation_explanations(context)

        # Extract key insights
        key_insights = self._extract_key_insights(
            physics_explanation,
            shap_explanation,
            constraint_explanations,
            recommendation_explanations,
        )

        # Generate action items
        action_items = self._generate_action_items(
            constraint_explanations,
            recommendation_explanations,
        )

        # Generate summaries for different audiences
        executive_summary = self._generate_executive_summary(context, key_insights)
        operator_summary = self._generate_operator_summary(context, action_items)
        engineering_summary = self._generate_engineering_summary(
            context, physics_explanation, constraint_explanations
        )

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            physics_explanation, shap_explanation, recommendation_explanations
        )

        # Generate overall summary
        summary = self._generate_overall_summary(context, key_insights, confidence)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Full explanation generated in {processing_time:.1f}ms")

        return ComprehensiveExplanation(
            explanation_id=f"comp-{uuid.uuid4().hex[:12]}",
            explanation_type=ExplanationType.COMPREHENSIVE,
            summary=summary,
            confidence=confidence,
            confidence_level=self._get_confidence_level(confidence),
            provenance_hash=self._calculate_provenance_hash(context),
            context=context,
            physics_explanation=physics_explanation,
            shap_explanation=shap_explanation,
            lime_explanation=lime_explanation,
            constraint_explanations=constraint_explanations,
            recommendation_explanations=recommendation_explanations,
            executive_summary=executive_summary,
            operator_summary=operator_summary,
            engineering_summary=engineering_summary,
            key_insights=key_insights,
            action_items=action_items,
        )

    def format_for_operator(
        self,
        explanation: ComprehensiveExplanation,
    ) -> OperatorExplanation:
        """
        Format comprehensive explanation for operators.

        Creates a plain language view focused on actions and outcomes.

        Args:
            explanation: ComprehensiveExplanation to format

        Returns:
            OperatorExplanation with operator-friendly content

        Example:
            >>> operator_view = generator.format_for_operator(explanation)
            >>> print(operator_view.summary)
        """
        logger.debug("Formatting explanation for operator")

        # Extract what changed
        what_changed = self._extract_what_changed(explanation)

        # Determine why it matters
        why_matters = self._determine_why_matters(explanation)

        # Extract what to do
        what_to_do = self._extract_what_to_do(explanation)

        # Determine expected results
        expected_results = self._determine_expected_results(explanation)

        # Extract cautions
        cautions = self._extract_cautions(explanation)

        # Generate confidence statement
        confidence_statement = self._generate_confidence_statement(explanation.confidence)

        return OperatorExplanation(
            explanation_id=explanation.explanation_id,
            title="Combustion Optimization Recommendation",
            summary=explanation.operator_summary,
            what_changed=what_changed,
            why_matters=why_matters,
            what_to_do=what_to_do,
            expected_results=expected_results,
            cautions=cautions,
            confidence_statement=confidence_statement,
        )

    def format_for_engineer(
        self,
        explanation: ComprehensiveExplanation,
    ) -> EngineerExplanation:
        """
        Format comprehensive explanation for engineers.

        Creates a technical view with detailed analysis.

        Args:
            explanation: ComprehensiveExplanation to format

        Returns:
            EngineerExplanation with technical content

        Example:
            >>> engineer_view = generator.format_for_engineer(explanation)
            >>> print(engineer_view.physics_basis)
        """
        logger.debug("Formatting explanation for engineer")

        # Extract physics basis
        physics_basis = ""
        if explanation.physics_explanation:
            physics_basis = explanation.physics_explanation.engineering_narrative

        # Determine mathematical model
        mathematical_model = self._describe_mathematical_model(explanation)

        # Extract feature importance
        feature_importance = {}
        if explanation.shap_explanation:
            feature_importance = explanation.shap_explanation.feature_importance

        # Extract sensitivity analysis
        sensitivity_analysis = self._extract_sensitivity_analysis(explanation)

        # Extract uncertainty quantification
        uncertainty_quantification = self._extract_uncertainty_quantification(explanation)

        # Extract model diagnostics
        model_diagnostics = self._extract_model_diagnostics(explanation)

        # Generate references
        references = self._generate_references()

        return EngineerExplanation(
            explanation_id=explanation.explanation_id,
            title="Combustion Optimization Technical Analysis",
            technical_summary=explanation.engineering_summary,
            physics_basis=physics_basis,
            mathematical_model=mathematical_model,
            feature_importance=feature_importance,
            constraint_analysis=explanation.constraint_explanations,
            sensitivity_analysis=sensitivity_analysis,
            uncertainty_quantification=uncertainty_quantification,
            model_diagnostics=model_diagnostics,
            references=references,
        )

    def export_explanation(
        self,
        explanation: ComprehensiveExplanation,
        format: str = "json",
    ) -> bytes:
        """
        Export explanation to specified format.

        Args:
            explanation: ComprehensiveExplanation to export
            format: Export format ("json", "html", "pdf")

        Returns:
            Bytes containing exported explanation

        Example:
            >>> data = generator.export_explanation(explanation, format="json")
            >>> with open("explanation.json", "wb") as f:
            ...     f.write(data)
        """
        logger.info(f"Exporting explanation in {format} format")

        if format.lower() == "json":
            return self._export_json(explanation)
        elif format.lower() == "html":
            return self._export_html(explanation)
        elif format.lower() == "markdown":
            return self._export_markdown(explanation)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _generate_physics_explanation(
        self,
        context: ExplanationContext,
    ) -> Optional[PhysicsExplanation]:
        """Generate physics-based explanation from context."""
        try:
            state = context.current_state

            # Calculate lambda and excess air
            o2_percent = state.get("o2_percent", 3.0)
            lambda_val = 1 + o2_percent / 21  # Approximate
            excess_air = (lambda_val - 1) * 100

            # Generate stoichiometry explanation
            stoich = self.physics_explainer.explain_stoichiometry(lambda_val, excess_air)

            # Generate efficiency explanation if we have before/after data
            efficiency = None
            if "efficiency" in state and "target_efficiency" in context.optimization_result:
                before = {"efficiency": state["efficiency"], "o2_percent": o2_percent}
                after = {
                    "efficiency": context.optimization_result.get("target_efficiency", state["efficiency"]),
                    "o2_percent": context.optimization_result.get("target_o2", o2_percent)
                }
                efficiency = self.physics_explainer.explain_efficiency_change(before, after)

            # Generate emission explanations
            emissions = []
            for emission_type in ["NOx", "CO"]:
                key = f"{emission_type.lower()}_ppm"
                if key in state:
                    conditions = {
                        "emission_type": emission_type,
                        "current_level_ppm": state[key],
                        "o2_percent": o2_percent,
                        "t_flame": state.get("t_flame", 1500),
                        "regulatory_limit_ppm": state.get(f"{key}_limit", 100),
                    }
                    emissions.append(
                        self.physics_explainer.explain_emission_formation(conditions)
                    )

            # Generate stability explanation
            stability = None
            stability_factors = {
                "air_fuel_ratio": lambda_val,
                "firing_rate_percent": state.get("load_percent", 70),
                "flame_signal": state.get("flame_signal", 0.9),
                "pressure_fluctuation": state.get("pressure_fluct", 2.0),
            }
            stability = self.physics_explainer.explain_stability_risk(stability_factors)

            # Generate engineering narrative
            narrative = self.physics_explainer.generate_engineering_narrative({
                "stoichiometry": stoich,
                "efficiency": efficiency,
                "emissions": emissions,
                "stability": stability,
            })

            return PhysicsExplanation(
                explanation_id=f"physics-{uuid.uuid4().hex[:12]}",
                explanation_type=ExplanationType.PHYSICS,
                summary=stoich.summary,
                confidence=0.9,
                confidence_level=ConfidenceLevel.HIGH,
                stoichiometry=stoich,
                efficiency=efficiency,
                emissions=emissions,
                stability=stability,
                engineering_narrative=narrative,
            )

        except Exception as e:
            logger.warning(f"Failed to generate physics explanation: {e}")
            return None

    def _generate_shap_explanation(
        self,
        context: ExplanationContext,
    ) -> Optional[SHAPExplanation]:
        """Generate SHAP explanation from context."""
        if self.model is None:
            return None

        try:
            return self.shap_explainer.explain_prediction(
                self.model, context.current_state
            )
        except Exception as e:
            logger.warning(f"Failed to generate SHAP explanation: {e}")
            return None

    def _generate_lime_explanation(
        self,
        context: ExplanationContext,
    ) -> Optional[LIMEExplanation]:
        """Generate LIME explanation from context."""
        if self.model is None:
            return None

        try:
            return self.lime_explainer.explain_instance(
                self.model, context.current_state
            )
        except Exception as e:
            logger.warning(f"Failed to generate LIME explanation: {e}")
            return None

    def _generate_constraint_explanations(
        self,
        context: ExplanationContext,
    ) -> List[ConstraintExplanation]:
        """Generate constraint explanations from context."""
        try:
            return self.constraint_explainer.explain_binding_constraints(
                context.optimization_result
            )
        except Exception as e:
            logger.warning(f"Failed to generate constraint explanations: {e}")
            return []

    def _generate_recommendation_explanations(
        self,
        context: ExplanationContext,
    ) -> List[RecommendationExplanation]:
        """Generate recommendation explanations from context."""
        explanations = []
        try:
            for rec in context.recommendations:
                explanations.append(
                    self.recommendation_explainer.explain_recommendation(rec)
                )
        except Exception as e:
            logger.warning(f"Failed to generate recommendation explanations: {e}")
        return explanations

    def _extract_key_insights(
        self,
        physics: Optional[PhysicsExplanation],
        shap: Optional[SHAPExplanation],
        constraints: List[ConstraintExplanation],
        recommendations: List[RecommendationExplanation],
    ) -> List[str]:
        """Extract key insights from all explanations."""
        insights = []

        # Physics insights
        if physics and physics.stoichiometry:
            if physics.stoichiometry.excess_air_percent > 25:
                insights.append("High excess air is reducing efficiency")
            elif physics.stoichiometry.excess_air_percent < 10:
                insights.append("Low excess air may risk incomplete combustion")

        if physics and physics.stability:
            if physics.stability.stability_index < 0.7:
                insights.append("Combustion stability margin is below optimal")

        # Constraint insights
        binding = [c for c in constraints if c.status.value == "binding"]
        if binding:
            insights.append(f"{len(binding)} constraint(s) are limiting optimization")

        # Recommendation insights
        if recommendations:
            total_efficiency = sum(
                r.impact_prediction.efficiency_change for r in recommendations
            )
            if total_efficiency > 0:
                insights.append(
                    f"Recommendations could improve efficiency by {total_efficiency:.2f}%"
                )

        # SHAP insights
        if shap and shap.top_features:
            top = shap.top_features[0]
            insights.append(
                f"{top.feature_name} is the most influential factor on predictions"
            )

        return insights[:self.config.max_key_insights]

    def _generate_action_items(
        self,
        constraints: List[ConstraintExplanation],
        recommendations: List[RecommendationExplanation],
    ) -> List[str]:
        """Generate prioritized action items."""
        actions = []

        # Priority 1: Address violations
        violated = [c for c in constraints if c.status.value == "violated"]
        for v in violated:
            actions.append(f"URGENT: Address {v.constraint_name} violation")

        # Priority 2: Implement high-priority recommendations
        high_priority = [r for r in recommendations if r.recommendation.priority <= 2]
        for r in high_priority:
            actions.append(
                f"Implement: {r.recommendation.parameter_name} adjustment "
                f"(Priority {r.recommendation.priority})"
            )

        # Priority 3: Monitor binding constraints
        binding = [c for c in constraints if c.status.value == "binding"]
        if binding:
            actions.append(f"Monitor {len(binding)} binding constraint(s)")

        return actions[:self.config.max_action_items]

    def _generate_executive_summary(
        self,
        context: ExplanationContext,
        insights: List[str],
    ) -> str:
        """Generate executive summary."""
        summary_parts = []

        # Overall status
        summary_parts.append(
            f"Combustion optimization analysis for {context.boiler_id}. "
        )

        # Key findings
        if insights:
            summary_parts.append(f"Key findings: {'; '.join(insights[:3])}. ")

        # Recommendations summary
        if context.recommendations:
            n_recs = len(context.recommendations)
            summary_parts.append(f"{n_recs} optimization recommendation(s) identified. ")

        return "".join(summary_parts)

    def _generate_operator_summary(
        self,
        context: ExplanationContext,
        actions: List[str],
    ) -> str:
        """Generate operator summary."""
        summary_parts = []

        # Current status
        state = context.current_state
        o2 = state.get("o2_percent", "N/A")
        efficiency = state.get("efficiency", "N/A")
        if isinstance(efficiency, float):
            efficiency = f"{efficiency*100:.1f}%"

        summary_parts.append(
            f"Current operation: O2 at {o2}%, efficiency at {efficiency}. "
        )

        # Actions needed
        if actions:
            summary_parts.append(f"Actions needed: {actions[0]}. ")
            if len(actions) > 1:
                summary_parts.append(f"Plus {len(actions)-1} additional items. ")
        else:
            summary_parts.append("No immediate actions required. ")

        return "".join(summary_parts)

    def _generate_engineering_summary(
        self,
        context: ExplanationContext,
        physics: Optional[PhysicsExplanation],
        constraints: List[ConstraintExplanation],
    ) -> str:
        """Generate engineering summary."""
        summary_parts = []

        # Technical overview
        state = context.current_state
        summary_parts.append(
            f"Technical Analysis for {context.boiler_id}\n"
            f"Operating state: {json.dumps(state, indent=2)}\n"
        )

        # Physics summary
        if physics and physics.stoichiometry:
            s = physics.stoichiometry
            summary_parts.append(
                f"Stoichiometry: lambda={s.lambda_value:.2f}, "
                f"excess air={s.excess_air_percent:.1f}%, "
                f"predicted O2={s.oxygen_percent:.1f}%\n"
            )

        # Constraint summary
        if constraints:
            summary_parts.append(f"Active constraints: {len(constraints)}\n")
            for c in constraints[:3]:
                summary_parts.append(
                    f"  - {c.constraint_name}: {c.current_value:.2f} "
                    f"(limit: {c.limit_value:.2f})\n"
                )

        return "".join(summary_parts)

    def _calculate_overall_confidence(
        self,
        physics: Optional[PhysicsExplanation],
        shap: Optional[SHAPExplanation],
        recommendations: List[RecommendationExplanation],
    ) -> float:
        """Calculate overall explanation confidence."""
        confidences = []

        if physics:
            confidences.append(physics.confidence)
        if shap:
            confidences.append(shap.confidence)
        for r in recommendations:
            confidences.append(r.confidence)

        if confidences:
            return sum(confidences) / len(confidences)
        return 0.75  # Default moderate confidence

    def _generate_overall_summary(
        self,
        context: ExplanationContext,
        insights: List[str],
        confidence: float,
    ) -> str:
        """Generate overall explanation summary."""
        return (
            f"Comprehensive analysis for {context.boiler_id} completed. "
            f"{len(insights)} key insight(s) identified. "
            f"Overall confidence: {confidence*100:.0f}%. "
            f"Target audience: {context.target_audience.value}."
        )

    def _extract_what_changed(
        self,
        explanation: ComprehensiveExplanation,
    ) -> List[str]:
        """Extract what changed for operator view."""
        changes = []

        if explanation.recommendation_explanations:
            for rec_exp in explanation.recommendation_explanations[:3]:
                rec = rec_exp.recommendation
                changes.append(
                    f"{rec.parameter_name}: recommend changing from "
                    f"{rec.current_value:.1f} to {rec.recommended_value:.1f} {rec.unit}"
                )

        return changes

    def _determine_why_matters(
        self,
        explanation: ComprehensiveExplanation,
    ) -> str:
        """Determine why the explanation matters."""
        if explanation.recommendation_explanations:
            total_savings = sum(
                r.impact_prediction.fuel_savings_percent
                for r in explanation.recommendation_explanations
            )
            if total_savings > 0:
                return f"These changes could save approximately {total_savings:.1f}% fuel"

        return "Optimizing combustion improves efficiency and reduces emissions"

    def _extract_what_to_do(
        self,
        explanation: ComprehensiveExplanation,
    ) -> List[str]:
        """Extract action items for operator view."""
        return explanation.action_items

    def _determine_expected_results(
        self,
        explanation: ComprehensiveExplanation,
    ) -> str:
        """Determine expected results."""
        if explanation.recommendation_explanations:
            rec = explanation.recommendation_explanations[0]
            impact = rec.impact_prediction
            return (
                f"Expected efficiency improvement: {impact.efficiency_change:.2f}%, "
                f"fuel savings: {impact.fuel_savings_percent:.1f}%"
            )
        return "Improved combustion performance expected"

    def _extract_cautions(
        self,
        explanation: ComprehensiveExplanation,
    ) -> List[str]:
        """Extract cautions for operator view."""
        cautions = []

        if explanation.physics_explanation and explanation.physics_explanation.stability:
            stab = explanation.physics_explanation.stability
            cautions.extend(stab.warnings)

        if explanation.recommendation_explanations:
            for rec_exp in explanation.recommendation_explanations:
                cautions.extend(rec_exp.impact_prediction.risk_factors)

        return list(set(cautions))[:5]

    def _generate_confidence_statement(
        self,
        confidence: float,
    ) -> str:
        """Generate confidence statement for operator."""
        if confidence > 0.9:
            return "We are very confident in these recommendations"
        elif confidence > 0.75:
            return "We are confident in these recommendations"
        elif confidence > 0.5:
            return "These recommendations are based on available data but have some uncertainty"
        else:
            return "These recommendations have significant uncertainty - proceed with caution"

    def _describe_mathematical_model(
        self,
        explanation: ComprehensiveExplanation,
    ) -> str:
        """Describe mathematical model used."""
        models = []

        if explanation.physics_explanation:
            models.append("Combustion physics (stoichiometry, heat balance)")

        if explanation.shap_explanation:
            models.append(f"SHAP (model: {explanation.shap_explanation.model_name})")

        if explanation.lime_explanation:
            models.append(f"LIME (model: {explanation.lime_explanation.model_name})")

        return ", ".join(models) if models else "No models applied"

    def _extract_sensitivity_analysis(
        self,
        explanation: ComprehensiveExplanation,
    ) -> Dict[str, float]:
        """Extract sensitivity analysis from explanations."""
        sensitivities = {}

        if explanation.physics_explanation:
            for emission in explanation.physics_explanation.emissions:
                sensitivities[f"{emission.emission_type}_temp_sensitivity"] = emission.temperature_sensitivity
                sensitivities[f"{emission.emission_type}_o2_sensitivity"] = emission.o2_sensitivity

        return sensitivities

    def _extract_uncertainty_quantification(
        self,
        explanation: ComprehensiveExplanation,
    ) -> Dict[str, Any]:
        """Extract uncertainty quantification."""
        uq = {}

        for rec_exp in explanation.recommendation_explanations:
            bounds = rec_exp.impact_prediction.efficiency_change_bounds
            uq[rec_exp.recommendation.parameter_name] = bounds

        return uq

    def _extract_model_diagnostics(
        self,
        explanation: ComprehensiveExplanation,
    ) -> Dict[str, Any]:
        """Extract model diagnostics."""
        diagnostics = {}

        if explanation.shap_explanation:
            diagnostics["shap_base_value"] = explanation.shap_explanation.shap_values.base_value
            diagnostics["shap_prediction"] = explanation.shap_explanation.shap_values.prediction

        if explanation.lime_explanation:
            diagnostics["lime_r_squared"] = explanation.lime_explanation.local_r_squared
            diagnostics["lime_sample_size"] = explanation.lime_explanation.sample_size

        return diagnostics

    def _generate_references(self) -> List[str]:
        """Generate technical references."""
        return [
            "API 560 - Fired Heaters for General Refinery Service",
            "ASME PTC 4 - Fired Steam Generators Performance Test Codes",
            "EPA AP-42 Compilation of Air Emission Factors",
            "NFPA 85 - Boiler and Combustion Systems Hazards Code",
        ]

    def _export_json(
        self,
        explanation: ComprehensiveExplanation,
    ) -> bytes:
        """Export explanation as JSON."""
        return json.dumps(explanation.dict(), default=str, indent=2).encode("utf-8")

    def _export_html(
        self,
        explanation: ComprehensiveExplanation,
    ) -> bytes:
        """Export explanation as HTML."""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html><head><title>Combustion Optimization Explanation</title>")
        html.append("<style>body{font-family:Arial,sans-serif;margin:20px;}</style>")
        html.append("</head><body>")
        html.append(f"<h1>Explanation: {explanation.explanation_id}</h1>")
        html.append(f"<p><strong>Summary:</strong> {explanation.summary}</p>")
        html.append(f"<p><strong>Confidence:</strong> {explanation.confidence*100:.0f}%</p>")

        if explanation.key_insights:
            html.append("<h2>Key Insights</h2><ul>")
            for insight in explanation.key_insights:
                html.append(f"<li>{insight}</li>")
            html.append("</ul>")

        if explanation.action_items:
            html.append("<h2>Action Items</h2><ol>")
            for action in explanation.action_items:
                html.append(f"<li>{action}</li>")
            html.append("</ol>")

        html.append("</body></html>")
        return "\n".join(html).encode("utf-8")

    def _export_markdown(
        self,
        explanation: ComprehensiveExplanation,
    ) -> bytes:
        """Export explanation as Markdown."""
        md = []
        md.append(f"# Combustion Optimization Explanation\n")
        md.append(f"**ID:** {explanation.explanation_id}\n")
        md.append(f"**Timestamp:** {explanation.timestamp}\n\n")
        md.append(f"## Summary\n{explanation.summary}\n\n")
        md.append(f"**Confidence:** {explanation.confidence*100:.0f}%\n\n")

        if explanation.key_insights:
            md.append("## Key Insights\n")
            for insight in explanation.key_insights:
                md.append(f"- {insight}\n")
            md.append("\n")

        if explanation.action_items:
            md.append("## Action Items\n")
            for i, action in enumerate(explanation.action_items, 1):
                md.append(f"{i}. {action}\n")
            md.append("\n")

        return "\n".join(md).encode("utf-8")

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map confidence score to confidence level."""
        if confidence > 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence > 0.85:
            return ConfidenceLevel.HIGH
        elif confidence > 0.70:
            return ConfidenceLevel.MEDIUM
        elif confidence > 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _calculate_provenance_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        data_str = str(data.dict() if hasattr(data, 'dict') else data)
        return hashlib.sha256(data_str.encode()).hexdigest()
