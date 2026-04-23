"""
GL-006 HEATRECLAIM - Explainability Report Generator

Combines all explainability components (SHAP, LIME, Causal,
Engineering Rationale) into comprehensive, actionable reports
for different stakeholders.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum
import hashlib
import json
import logging

from ..core.schemas import (
    HeatStream,
    HENDesign,
    OptimizationResult,
    ExplainabilityReport,
    FeatureImportance,
)
from .shap_explainer import SHAPExplainer, SHAPResult
from .lime_explainer import LIMEExplainer, LIMEResult
from .causal_analyzer import CausalAnalyzer, CausalAnalysisResult
from .engineering_rationale import (
    EngineeringRationaleGenerator,
    EngineeringRationale,
)

logger = logging.getLogger(__name__)


class AudienceType(Enum):
    """Target audience for report."""
    TECHNICAL = "technical"  # Engineers, domain experts
    EXECUTIVE = "executive"  # Management, decision makers
    REGULATORY = "regulatory"  # Auditors, compliance
    DEVELOPER = "developer"  # System developers


class ReportFormat(Enum):
    """Output format for report."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    STRUCTURED = "structured"


@dataclass
class ReportSection:
    """Section of explainability report."""
    title: str
    content: str
    subsections: List["ReportSection"] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None
    confidence: float = 1.0


@dataclass
class ExplainabilityReportFull:
    """Complete explainability report."""
    title: str
    generated_at: datetime
    audience: AudienceType
    executive_summary: str
    sections: List[ReportSection]
    key_findings: List[str]
    recommendations: List[str]
    provenance: Dict[str, str]
    computation_hash: str


class ExplainabilityReportGenerator:
    """
    Generates comprehensive explainability reports.

    Integrates outputs from SHAP, LIME, causal analysis,
    and engineering rationale into cohesive reports
    tailored for different audiences.

    Example:
        >>> generator = ExplainabilityReportGenerator()
        >>> report = generator.generate_report(
        ...     design, hot_streams, cold_streams,
        ...     optimization_result,
        ...     audience=AudienceType.TECHNICAL
        ... )
        >>> print(report.executive_summary)
    """

    VERSION = "1.0.0"

    def __init__(self) -> None:
        """Initialize report generator with component analyzers."""
        self.shap_explainer = SHAPExplainer()
        self.lime_explainer = LIMEExplainer()
        self.causal_analyzer = CausalAnalyzer()
        self.rationale_generator = EngineeringRationaleGenerator()

    def generate_report(
        self,
        design: HENDesign,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        optimization_result: Optional[OptimizationResult] = None,
        pinch_result: Optional[Any] = None,
        audience: AudienceType = AudienceType.TECHNICAL,
        format: ReportFormat = ReportFormat.STRUCTURED,
        delta_t_min: float = 10.0,
    ) -> ExplainabilityReportFull:
        """
        Generate complete explainability report.

        Args:
            design: The HEN design to explain
            hot_streams: Hot process streams
            cold_streams: Cold process streams
            optimization_result: Full optimization result
            pinch_result: Pinch analysis result
            audience: Target audience for report
            format: Output format
            delta_t_min: Minimum approach temperature

        Returns:
            Complete ExplainabilityReportFull
        """
        generated_at = datetime.now(timezone.utc)

        # Run all analyses
        shap_result = self.shap_explainer.explain_design(
            design, hot_streams, cold_streams, optimization_result
        )

        lime_result = self.lime_explainer.explain_design(
            design, hot_streams, cold_streams
        )

        causal_result = self.causal_analyzer.analyze_design(
            design, hot_streams, cold_streams,
            pinch_result.pinch_temperature_C if pinch_result else None,
            delta_t_min
        )

        engineering_rationale = self.rationale_generator.generate_rationale(
            design, hot_streams, cold_streams, pinch_result, delta_t_min
        )

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            design, shap_result, causal_result, engineering_rationale, audience
        )

        # Generate sections based on audience
        sections = self._generate_sections(
            design,
            shap_result,
            lime_result,
            causal_result,
            engineering_rationale,
            audience,
        )

        # Extract key findings
        key_findings = self._extract_key_findings(
            shap_result, causal_result, engineering_rationale
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            causal_result, engineering_rationale
        )

        # Compile provenance
        provenance = {
            "shap_hash": shap_result.computation_hash,
            "lime_hash": lime_result.computation_hash,
            "causal_hash": causal_result.computation_hash,
            "rationale_hash": engineering_rationale.computation_hash,
            "generator_version": self.VERSION,
        }

        # Compute overall hash
        computation_hash = self._compute_hash(provenance, key_findings)

        return ExplainabilityReportFull(
            title=f"Heat Recovery Optimization Explainability Report - {design.design_name}",
            generated_at=generated_at,
            audience=audience,
            executive_summary=executive_summary,
            sections=sections,
            key_findings=key_findings,
            recommendations=recommendations,
            provenance=provenance,
            computation_hash=computation_hash,
        )

    def generate_quick_explanation(
        self,
        design: HENDesign,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
    ) -> Dict[str, Any]:
        """
        Generate quick, lightweight explanation.

        Args:
            design: The HEN design
            hot_streams: Hot streams
            cold_streams: Cold streams

        Returns:
            Dictionary with key explanations
        """
        # Quick SHAP analysis
        shap_result = self.shap_explainer.explain_design(
            design, hot_streams, cold_streams
        )

        # Top 5 feature importances
        top_features = list(shap_result.feature_importance.items())[:5]

        # Quick engineering summary
        total_hot = sum(s.duty_kW for s in hot_streams)
        total_cold = sum(s.duty_kW for s in cold_streams)
        recovery_pct = design.total_heat_recovered_kW / min(total_hot, total_cold) * 100 \
            if min(total_hot, total_cold) > 0 else 0

        return {
            "design_name": design.design_name,
            "heat_recovered_kW": design.total_heat_recovered_kW,
            "recovery_percentage": round(recovery_pct, 1),
            "exchanger_count": design.exchanger_count,
            "top_drivers": [
                {"feature": name, "importance": round(value, 4)}
                for name, value in top_features
            ],
            "summary": (
                f"Design recovers {design.total_heat_recovered_kW:.0f} kW "
                f"({recovery_pct:.0f}% of potential) using "
                f"{design.exchanger_count} exchangers. "
                f"Key driver: {top_features[0][0] if top_features else 'N/A'}"
            ),
        }

    def to_markdown(
        self,
        report: ExplainabilityReportFull,
    ) -> str:
        """Convert report to markdown format."""
        md = f"# {report.title}\n\n"
        md += f"*Generated: {report.generated_at.isoformat()}*\n\n"
        md += f"*Audience: {report.audience.value}*\n\n"

        md += "## Executive Summary\n\n"
        md += f"{report.executive_summary}\n\n"

        md += "## Key Findings\n\n"
        for finding in report.key_findings:
            md += f"- {finding}\n"
        md += "\n"

        for section in report.sections:
            md += f"## {section.title}\n\n"
            md += f"{section.content}\n\n"

            for subsection in section.subsections:
                md += f"### {subsection.title}\n\n"
                md += f"{subsection.content}\n\n"

        md += "## Recommendations\n\n"
        for i, rec in enumerate(report.recommendations, 1):
            md += f"{i}. {rec}\n"
        md += "\n"

        md += "---\n"
        md += f"*Computation Hash: {report.computation_hash[:16]}...*\n"

        return md

    def to_schema_report(
        self,
        report: ExplainabilityReportFull,
        shap_result: SHAPResult,
    ) -> ExplainabilityReport:
        """Convert to standard ExplainabilityReport schema."""
        feature_importance = [
            FeatureImportance(
                feature_name=name,
                importance_value=value,
                rank=i + 1,
                direction="positive" if value > 0 else "negative",
                category="input",
            )
            for i, (name, value) in enumerate(
                list(shap_result.feature_importance.items())[:10]
            )
        ]

        return ExplainabilityReport(
            method="hybrid",
            feature_importance=feature_importance,
            local_explanations={
                "summary": report.executive_summary[:500],
            },
            global_explanations={
                "key_findings": report.key_findings,
            },
            confidence_score=0.9,
            computation_hash=report.computation_hash,
        )

    def _generate_executive_summary(
        self,
        design: HENDesign,
        shap_result: SHAPResult,
        causal_result: CausalAnalysisResult,
        engineering_rationale: EngineeringRationale,
        audience: AudienceType,
    ) -> str:
        """Generate executive summary tailored to audience."""
        if audience == AudienceType.EXECUTIVE:
            summary = (
                f"The heat recovery system design '{design.design_name}' "
                f"achieves {design.total_heat_recovered_kW:.0f} kW of heat recovery "
                f"using {design.exchanger_count} heat exchangers. "
                f"Hot utility requirement: {design.hot_utility_required_kW:.0f} kW, "
                f"Cold utility requirement: {design.cold_utility_required_kW:.0f} kW. "
            )

            if causal_result.recommendations:
                summary += f"Key opportunity: {causal_result.recommendations[0]}"

        elif audience == AudienceType.REGULATORY:
            summary = (
                f"Design '{design.design_name}' has been analyzed for transparency "
                f"and auditability. Heat recovery: {design.total_heat_recovered_kW:.0f} kW. "
                f"All calculations are deterministic and traceable. "
                f"Engineering rationale confidence: "
                f"{engineering_rationale.match_rationales[0].overall_confidence:.0%} "
                if engineering_rationale.match_rationales else ""
                f"Provenance hash: {shap_result.computation_hash[:16]}..."
            )

        else:  # TECHNICAL or DEVELOPER
            top_driver = list(shap_result.feature_importance.keys())[0] \
                if shap_result.feature_importance else "N/A"

            summary = (
                f"Heat Exchanger Network '{design.design_name}' analysis:\n"
                f"• Heat recovered: {design.total_heat_recovered_kW:.1f} kW\n"
                f"• Exchangers: {design.exchanger_count}\n"
                f"• Hot utility: {design.hot_utility_required_kW:.1f} kW\n"
                f"• Cold utility: {design.cold_utility_required_kW:.1f} kW\n"
                f"• Primary optimization driver: {top_driver}\n"
                f"• Pinch interpretation: {engineering_rationale.pinch_interpretation[:200]}..."
            )

        return summary

    def _generate_sections(
        self,
        design: HENDesign,
        shap_result: SHAPResult,
        lime_result: LIMEResult,
        causal_result: CausalAnalysisResult,
        engineering_rationale: EngineeringRationale,
        audience: AudienceType,
    ) -> List[ReportSection]:
        """Generate report sections based on audience."""
        sections = []

        # Engineering Rationale Section
        if audience in [AudienceType.TECHNICAL, AudienceType.REGULATORY]:
            sections.append(ReportSection(
                title="Engineering Rationale",
                content=engineering_rationale.pinch_interpretation,
                subsections=[
                    ReportSection(
                        title="Utility Justification",
                        content=engineering_rationale.utility_justification,
                    ),
                    ReportSection(
                        title="Design Trade-offs",
                        content="\n".join(
                            f"• {t}" for t in engineering_rationale.design_trade_offs
                        ),
                    ),
                ],
                confidence=0.99,
            ))

        # Feature Attribution Section
        sections.append(ReportSection(
            title="Feature Attribution Analysis",
            content=(
                "SHAP and LIME analyses identify which input features "
                "most strongly influence the optimization outcome."
            ),
            subsections=[
                ReportSection(
                    title="Top Feature Importances (SHAP)",
                    content="\n".join(
                        f"• {name}: {value:.4f}"
                        for name, value in list(shap_result.feature_importance.items())[:10]
                    ),
                    data={"shap_values": shap_result.feature_importance},
                ),
                ReportSection(
                    title="Local Explanation (LIME)",
                    content="\n".join(lime_result.explanation_text[:5]),
                    data={"lime_weights": lime_result.feature_weights},
                ),
            ],
            confidence=shap_result.expected_value if hasattr(shap_result, 'expected_value') else 0.9,
        ))

        # Causal Analysis Section
        if audience in [AudienceType.TECHNICAL, AudienceType.DEVELOPER]:
            sections.append(ReportSection(
                title="Causal Analysis",
                content=(
                    "Causal inference identifies cause-effect relationships "
                    "and enables counterfactual reasoning."
                ),
                subsections=[
                    ReportSection(
                        title="Key Causal Drivers",
                        content="\n".join(
                            f"• {d}" for d in causal_result.key_drivers
                        ),
                    ),
                    ReportSection(
                        title="What-If Scenarios",
                        content="\n".join(
                            f"• {cf.scenario}: {cf.explanation}"
                            for cf in causal_result.counterfactuals
                        ),
                    ),
                ],
                confidence=0.85,
            ))

        # Match-level Explanations
        if audience == AudienceType.TECHNICAL:
            match_content = []
            for match in engineering_rationale.match_rationales[:5]:
                match_content.append(
                    f"**{match.exchanger_id}** ({match.hot_stream_id} → {match.cold_stream_id}): "
                    f"Confidence {match.overall_confidence:.0%}"
                )

            sections.append(ReportSection(
                title="Match-Level Explanations",
                content="\n".join(match_content) if match_content else "No matches to explain",
            ))

        return sections

    def _extract_key_findings(
        self,
        shap_result: SHAPResult,
        causal_result: CausalAnalysisResult,
        engineering_rationale: EngineeringRationale,
    ) -> List[str]:
        """Extract key findings from all analyses."""
        findings = []

        # SHAP findings
        if shap_result.feature_importance:
            top_feature = list(shap_result.feature_importance.keys())[0]
            top_value = list(shap_result.feature_importance.values())[0]
            findings.append(
                f"Primary driver: {top_feature} (importance: {top_value:.4f})"
            )

        # Causal findings
        if causal_result.key_drivers:
            findings.append(f"Causal pathway: {causal_result.key_drivers[0]}")

        # Engineering findings
        for rationale in engineering_rationale.design_rationales:
            if rationale.applied:
                findings.append(f"{rationale.rule_name}: {rationale.impact}")

        # Constraint findings
        findings.extend(engineering_rationale.key_constraints[:2])

        return findings[:10]  # Top 10 findings

    def _generate_recommendations(
        self,
        causal_result: CausalAnalysisResult,
        engineering_rationale: EngineeringRationale,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Causal recommendations
        recommendations.extend(causal_result.recommendations)

        # Engineering-based recommendations
        for rationale in engineering_rationale.design_rationales:
            if not rationale.applied:
                recommendations.append(
                    f"Consider addressing: {rationale.rule_name} - {rationale.description}"
                )

        # Deduplicate and limit
        unique_recs = list(dict.fromkeys(recommendations))
        return unique_recs[:5]

    def _compute_hash(
        self,
        provenance: Dict[str, str],
        key_findings: List[str],
    ) -> str:
        """Compute SHA-256 hash for report provenance."""
        data = {
            "provenance": provenance,
            "key_findings": key_findings,
            "version": self.VERSION,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
