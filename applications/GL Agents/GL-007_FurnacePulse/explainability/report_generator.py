"""
GL-007 FurnacePulse - Explainability Report Generator

Combines SHAP, LIME, engineering rationale, and model cards into
comprehensive, audit-ready reports for different stakeholders.

This module provides:
- Executive summaries tailored to audience (operators, engineers, safety)
- Combined SHAP/LIME/engineering explanations
- Model confidence and uncertainty bounds
- Audit-ready documentation with provenance tracking

Zero-Hallucination: All report content is derived from deterministic
calculations and rule-based transformations, not LLM generation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import hashlib
import json
import logging

import numpy as np

from .shap_explainer import SHAPExplainer, SHAPResult, GlobalSHAPSummary
from .lime_explainer import LIMEExplainer, LIMEResult, ConfidenceLevel
from .engineering_rationale import EngineeringRationale as EngineeringRationaleGenerator
from .model_cards import ModelCardGenerator, ModelCard, ModelType

logger = logging.getLogger(__name__)


class AudienceType(Enum):
    """Target audience for report."""
    OPERATOR = "operator"          # Control room operators
    PROCESS_ENGINEER = "process_engineer"  # Process engineers
    RELIABILITY_ENGINEER = "reliability_engineer"  # Reliability/maintenance
    SAFETY = "safety"              # Safety personnel
    MANAGEMENT = "management"      # Plant management
    AUDIT = "audit"               # External auditors


class ReportFormat(Enum):
    """Output format for report."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    STRUCTURED = "structured"


class UrgencyLevel(Enum):
    """Urgency level for report."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class ReportSection:
    """Section of explainability report."""

    section_id: str
    title: str
    content: str
    subsections: List["ReportSection"] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None
    visualization_type: Optional[str] = None  # "chart", "table", "heatmap"
    audience_relevance: List[AudienceType] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class UncertaintyBounds:
    """Uncertainty bounds for prediction."""

    prediction_value: float
    lower_bound_95: float
    upper_bound_95: float
    lower_bound_80: float
    upper_bound_80: float
    confidence_method: str
    model_uncertainty: float
    data_uncertainty: float


@dataclass
class AuditTrail:
    """Audit trail information."""

    report_id: str
    generated_at: datetime
    model_versions: Dict[str, str]
    data_hash: str
    computation_hashes: Dict[str, str]
    input_parameters: Dict[str, Any]
    provenance_chain: List[Dict[str, str]]


@dataclass
class ExplainabilityReport:
    """Complete explainability report."""

    # Identification
    report_id: str
    report_type: str
    title: str
    generated_at: datetime

    # Target
    prediction_type: str
    prediction_id: str
    prediction_value: float
    uncertainty_bounds: UncertaintyBounds

    # Audience
    primary_audience: AudienceType
    urgency: UrgencyLevel

    # Content
    executive_summary: str
    sections: List[ReportSection]
    key_findings: List[str]
    recommendations: List[str]

    # Supporting analysis
    shap_summary: Dict[str, Any]
    lime_summary: Dict[str, Any]
    engineering_summary: Dict[str, Any]

    # Audit
    audit_trail: AuditTrail
    model_card_reference: Optional[str]
    computation_hash: str


class ExplainabilityReportGenerator:
    """
    Generator for comprehensive explainability reports.

    Integrates outputs from SHAP, LIME, engineering rationale,
    and model cards into cohesive reports tailored for different
    audiences (operators, engineers, safety, management).

    Example:
        >>> generator = ExplainabilityReportGenerator()
        >>> report = generator.generate_report(
        ...     prediction_type="hotspot",
        ...     prediction_value=75.0,
        ...     sensor_readings=readings,
        ...     audience=AudienceType.OPERATOR
        ... )
        >>> print(report.executive_summary)
        >>> markdown = generator.to_markdown(report)
    """

    VERSION = "1.0.0"

    # Audience-specific content configuration
    AUDIENCE_CONFIG = {
        AudienceType.OPERATOR: {
            "detail_level": "low",
            "technical_depth": "basic",
            "focus": ["immediate_actions", "monitoring_guidance"],
            "include_stats": False,
            "include_model_details": False,
        },
        AudienceType.PROCESS_ENGINEER: {
            "detail_level": "high",
            "technical_depth": "advanced",
            "focus": ["root_cause", "feature_importance", "engineering_rationale"],
            "include_stats": True,
            "include_model_details": True,
        },
        AudienceType.RELIABILITY_ENGINEER: {
            "detail_level": "high",
            "technical_depth": "advanced",
            "focus": ["degradation_drivers", "rul_factors", "maintenance_recommendations"],
            "include_stats": True,
            "include_model_details": True,
        },
        AudienceType.SAFETY: {
            "detail_level": "medium",
            "technical_depth": "intermediate",
            "focus": ["safety_implications", "risk_assessment", "immediate_actions"],
            "include_stats": False,
            "include_model_details": False,
        },
        AudienceType.MANAGEMENT: {
            "detail_level": "low",
            "technical_depth": "basic",
            "focus": ["business_impact", "risk_summary", "resource_needs"],
            "include_stats": True,
            "include_model_details": False,
        },
        AudienceType.AUDIT: {
            "detail_level": "high",
            "technical_depth": "advanced",
            "focus": ["provenance", "model_validation", "uncertainty", "limitations"],
            "include_stats": True,
            "include_model_details": True,
        },
    }

    def __init__(
        self,
        model_version: str = "1.0.0",
    ) -> None:
        """
        Initialize report generator.

        Args:
            model_version: Version of the prediction model
        """
        self.model_version = model_version
        self.shap_explainer = SHAPExplainer(model_version=model_version)
        self.lime_explainer = LIMEExplainer(model_version=model_version)
        self.rationale_generator = EngineeringRationaleGenerator()
        self.model_card_generator = ModelCardGenerator()
        self._report_counter = 0

    def generate_report(
        self,
        prediction_type: str,
        prediction_value: float,
        sensor_readings: Dict[str, float],
        feature_importance: Optional[Dict[str, float]] = None,
        audience: AudienceType = AudienceType.PROCESS_ENGINEER,
        prediction_id: Optional[str] = None,
        model_card: Optional[ModelCard] = None,
        include_shap: bool = True,
        include_lime: bool = True,
        include_engineering: bool = True,
    ) -> ExplainabilityReport:
        """
        Generate comprehensive explainability report.

        Args:
            prediction_type: Type of prediction (hotspot, efficiency, rul)
            prediction_value: The predicted value
            sensor_readings: Current sensor values
            feature_importance: Pre-computed feature importance (optional)
            audience: Target audience for report
            prediction_id: Optional prediction identifier
            model_card: Optional model card for reference
            include_shap: Include SHAP analysis
            include_lime: Include LIME analysis
            include_engineering: Include engineering rationale

        Returns:
            Complete ExplainabilityReport
        """
        self._report_counter += 1
        generated_at = datetime.now(timezone.utc)
        report_id = f"RPT-{generated_at.strftime('%Y%m%d%H%M%S')}-{self._report_counter:04d}"

        if prediction_id is None:
            prediction_id = f"{prediction_type}_{self._report_counter}"

        # Run analyses
        shap_result = None
        lime_result = None
        engineering_result = None

        if include_shap:
            shap_result = self._run_shap_analysis(
                prediction_type, sensor_readings, prediction_value
            )
            if feature_importance is None and shap_result:
                feature_importance = shap_result.feature_importance

        if include_lime:
            lime_result = self._run_lime_analysis(
                prediction_type, sensor_readings, prediction_value
            )

        if include_engineering and feature_importance:
            engineering_result = self.rationale_generator.generate_rationale(
                prediction_type=prediction_type,
                prediction_value=prediction_value,
                sensor_readings=sensor_readings,
                feature_importance=feature_importance,
            )

        # Determine urgency
        urgency = self._determine_urgency(prediction_type, prediction_value)

        # Calculate uncertainty bounds
        uncertainty_bounds = self._calculate_uncertainty_bounds(
            prediction_value, prediction_type, shap_result, lime_result
        )

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            prediction_type, prediction_value, urgency, audience,
            shap_result, lime_result, engineering_result
        )

        # Generate sections
        sections = self._generate_sections(
            prediction_type, prediction_value, sensor_readings,
            shap_result, lime_result, engineering_result,
            audience, model_card
        )

        # Extract key findings
        key_findings = self._extract_key_findings(
            shap_result, lime_result, engineering_result, audience
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            prediction_type, prediction_value, urgency,
            engineering_result, audience
        )

        # Build summaries
        shap_summary = self._build_shap_summary(shap_result) if shap_result else {}
        lime_summary = self._build_lime_summary(lime_result) if lime_result else {}
        engineering_summary = self._build_engineering_summary(engineering_result) if engineering_result else {}

        # Build audit trail
        audit_trail = self._build_audit_trail(
            report_id, generated_at, sensor_readings,
            shap_result, lime_result, engineering_result
        )

        # Compute hash
        computation_hash = self._compute_hash(
            report_id, prediction_type, prediction_value, executive_summary
        )

        return ExplainabilityReport(
            report_id=report_id,
            report_type=f"{prediction_type}_explainability",
            title=self._generate_title(prediction_type, prediction_value, urgency),
            generated_at=generated_at,
            prediction_type=prediction_type,
            prediction_id=prediction_id,
            prediction_value=prediction_value,
            uncertainty_bounds=uncertainty_bounds,
            primary_audience=audience,
            urgency=urgency,
            executive_summary=executive_summary,
            sections=sections,
            key_findings=key_findings,
            recommendations=recommendations,
            shap_summary=shap_summary,
            lime_summary=lime_summary,
            engineering_summary=engineering_summary,
            audit_trail=audit_trail,
            model_card_reference=model_card.model_id if model_card else None,
            computation_hash=computation_hash,
        )

    def generate_comparison_report(
        self,
        reports: List[ExplainabilityReport],
        comparison_type: str = "temporal",
    ) -> Dict[str, Any]:
        """
        Generate comparison report across multiple predictions.

        Args:
            reports: List of explainability reports
            comparison_type: Type of comparison (temporal, spatial, scenario)

        Returns:
            Comparison analysis dictionary
        """
        if len(reports) < 2:
            return {"error": "Need at least 2 reports for comparison"}

        # Sort by timestamp
        sorted_reports = sorted(reports, key=lambda r: r.generated_at)

        comparison = {
            "comparison_type": comparison_type,
            "num_reports": len(reports),
            "time_range": {
                "start": sorted_reports[0].generated_at.isoformat(),
                "end": sorted_reports[-1].generated_at.isoformat(),
            },
            "prediction_trend": [],
            "feature_importance_evolution": {},
            "urgency_distribution": {},
            "key_changes": [],
        }

        # Track prediction values
        for report in sorted_reports:
            comparison["prediction_trend"].append({
                "timestamp": report.generated_at.isoformat(),
                "value": report.prediction_value,
                "urgency": report.urgency.value,
            })

        # Urgency distribution
        for report in reports:
            urgency = report.urgency.value
            comparison["urgency_distribution"][urgency] = \
                comparison["urgency_distribution"].get(urgency, 0) + 1

        # Feature importance evolution
        if reports[0].shap_summary and "top_features" in reports[0].shap_summary:
            all_features = set()
            for report in reports:
                if report.shap_summary and "top_features" in report.shap_summary:
                    all_features.update(report.shap_summary["top_features"].keys())

            for feature in all_features:
                comparison["feature_importance_evolution"][feature] = []
                for report in sorted_reports:
                    importance = 0.0
                    if report.shap_summary and "top_features" in report.shap_summary:
                        importance = report.shap_summary["top_features"].get(feature, 0.0)
                    comparison["feature_importance_evolution"][feature].append({
                        "timestamp": report.generated_at.isoformat(),
                        "importance": importance,
                    })

        # Identify key changes
        for i in range(1, len(sorted_reports)):
            prev = sorted_reports[i - 1]
            curr = sorted_reports[i]

            value_change = curr.prediction_value - prev.prediction_value
            if abs(value_change) > 10:  # Significant change threshold
                comparison["key_changes"].append({
                    "timestamp": curr.generated_at.isoformat(),
                    "change": f"Prediction changed by {value_change:+.1f}",
                    "previous_value": prev.prediction_value,
                    "current_value": curr.prediction_value,
                })

            if prev.urgency != curr.urgency:
                comparison["key_changes"].append({
                    "timestamp": curr.generated_at.isoformat(),
                    "change": f"Urgency changed from {prev.urgency.value} to {curr.urgency.value}",
                })

        return comparison

    def to_markdown(self, report: ExplainabilityReport) -> str:
        """
        Convert report to markdown format.

        Args:
            report: ExplainabilityReport to convert

        Returns:
            Markdown string
        """
        md = f"# {report.title}\n\n"

        # Metadata
        md += f"**Report ID:** {report.report_id}  \n"
        md += f"**Generated:** {report.generated_at.isoformat()}  \n"
        md += f"**Audience:** {report.primary_audience.value}  \n"
        md += f"**Urgency:** {report.urgency.value.upper()}  \n\n"

        # Prediction summary
        md += "## Prediction Summary\n\n"
        md += f"- **Type:** {report.prediction_type}\n"
        md += f"- **Value:** {report.prediction_value:.2f}\n"
        md += f"- **95% Confidence Interval:** [{report.uncertainty_bounds.lower_bound_95:.2f}, {report.uncertainty_bounds.upper_bound_95:.2f}]\n\n"

        # Executive summary
        md += "## Executive Summary\n\n"
        md += f"{report.executive_summary}\n\n"

        # Key findings
        md += "## Key Findings\n\n"
        for i, finding in enumerate(report.key_findings, 1):
            md += f"{i}. {finding}\n"
        md += "\n"

        # Sections
        for section in report.sections:
            md += f"## {section.title}\n\n"
            md += f"{section.content}\n\n"

            for subsection in section.subsections:
                md += f"### {subsection.title}\n\n"
                md += f"{subsection.content}\n\n"

        # Recommendations
        md += "## Recommendations\n\n"
        for i, rec in enumerate(report.recommendations, 1):
            md += f"{i}. {rec}\n"
        md += "\n"

        # SHAP summary (if available)
        if report.shap_summary:
            md += "## Feature Attribution (SHAP)\n\n"
            if "top_features" in report.shap_summary:
                md += "| Feature | Importance |\n|---------|------------|\n"
                for name, value in list(report.shap_summary["top_features"].items())[:5]:
                    md += f"| {name} | {value:.4f} |\n"
            md += "\n"

        # Audit information
        md += "## Audit Trail\n\n"
        md += f"- **Data Hash:** {report.audit_trail.data_hash[:16]}...\n"
        md += f"- **Computation Hash:** {report.computation_hash[:16]}...\n"
        md += f"- **Model Version:** {report.audit_trail.model_versions.get('prediction_model', 'N/A')}\n\n"

        md += "---\n"
        md += f"*This report was generated automatically by FurnacePulse Explainability Module v{self.VERSION}*\n"

        return md

    def to_json(self, report: ExplainabilityReport) -> str:
        """
        Convert report to JSON format.

        Args:
            report: ExplainabilityReport to convert

        Returns:
            JSON string
        """
        def serialize(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, "__dataclass_fields__"):
                return {k: serialize(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            return obj

        return json.dumps(serialize(report), indent=2)

    def to_html(self, report: ExplainabilityReport) -> str:
        """
        Convert report to HTML format.

        Args:
            report: ExplainabilityReport to convert

        Returns:
            HTML string
        """
        # Convert markdown to basic HTML
        md = self.to_markdown(report)

        # Simple markdown to HTML conversion
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; border-bottom: 2px solid #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #777; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        .urgency-critical {{ color: #d32f2f; font-weight: bold; }}
        .urgency-high {{ color: #f57c00; font-weight: bold; }}
        .urgency-medium {{ color: #fbc02d; }}
        .urgency-low {{ color: #388e3c; }}
        .metadata {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        .recommendation {{ background-color: #e3f2fd; padding: 10px; margin: 5px 0; border-left: 3px solid #2196f3; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>

    <div class="metadata">
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Generated:</strong> {report.generated_at.isoformat()}</p>
        <p><strong>Audience:</strong> {report.primary_audience.value}</p>
        <p><strong>Urgency:</strong> <span class="urgency-{report.urgency.value}">{report.urgency.value.upper()}</span></p>
    </div>

    <h2>Executive Summary</h2>
    <p>{report.executive_summary}</p>

    <h2>Key Findings</h2>
    <ol>
"""

        for finding in report.key_findings:
            html += f"        <li>{finding}</li>\n"

        html += """    </ol>

    <h2>Recommendations</h2>
"""

        for rec in report.recommendations:
            html += f'    <div class="recommendation">{rec}</div>\n'

        html += f"""
    <hr>
    <p><em>Computation Hash: {report.computation_hash[:16]}...</em></p>
</body>
</html>"""

        return html

    def _run_shap_analysis(
        self,
        prediction_type: str,
        sensor_readings: Dict[str, float],
        prediction_value: float,
    ) -> Optional[SHAPResult]:
        """Run SHAP analysis for prediction."""
        try:
            if prediction_type == "hotspot":
                return self.shap_explainer.explain_hotspot_prediction(
                    sensor_readings=sensor_readings,
                    hotspot_prediction={"severity_score": prediction_value},
                )
            elif prediction_type == "efficiency":
                return self.shap_explainer.explain_efficiency_prediction(
                    sensor_readings=sensor_readings,
                    efficiency_prediction={"efficiency_percent": prediction_value},
                )
            elif prediction_type == "rul":
                return self.shap_explainer.explain_rul_prediction(
                    sensor_readings=sensor_readings,
                    rul_prediction={"rul_hours": prediction_value},
                )
            else:
                return None
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return None

    def _run_lime_analysis(
        self,
        prediction_type: str,
        sensor_readings: Dict[str, float],
        prediction_value: float,
    ) -> Optional[LIMEResult]:
        """Run LIME analysis for prediction."""
        try:
            if prediction_type == "hotspot":
                return self.lime_explainer.explain_hotspot(
                    sensor_readings=sensor_readings,
                    hotspot_severity=prediction_value,
                )
            elif prediction_type == "efficiency":
                return self.lime_explainer.explain_efficiency(
                    sensor_readings=sensor_readings,
                    efficiency_value=prediction_value,
                )
            elif prediction_type == "rul":
                return self.lime_explainer.explain_rul(
                    sensor_readings=sensor_readings,
                    rul_hours=prediction_value,
                )
            else:
                return None
        except Exception as e:
            logger.error(f"LIME analysis failed: {e}")
            return None

    def _determine_urgency(
        self,
        prediction_type: str,
        prediction_value: float,
    ) -> UrgencyLevel:
        """Determine urgency level based on prediction."""
        if prediction_type == "hotspot":
            if prediction_value >= 85:
                return UrgencyLevel.CRITICAL
            elif prediction_value >= 70:
                return UrgencyLevel.HIGH
            elif prediction_value >= 50:
                return UrgencyLevel.MEDIUM
            elif prediction_value >= 30:
                return UrgencyLevel.LOW
            else:
                return UrgencyLevel.INFORMATIONAL

        elif prediction_type == "efficiency":
            if prediction_value < 75:
                return UrgencyLevel.HIGH
            elif prediction_value < 82:
                return UrgencyLevel.MEDIUM
            elif prediction_value < 88:
                return UrgencyLevel.LOW
            else:
                return UrgencyLevel.INFORMATIONAL

        elif prediction_type == "rul":
            if prediction_value < 500:
                return UrgencyLevel.CRITICAL
            elif prediction_value < 2000:
                return UrgencyLevel.HIGH
            elif prediction_value < 5000:
                return UrgencyLevel.MEDIUM
            else:
                return UrgencyLevel.LOW

        return UrgencyLevel.INFORMATIONAL

    def _calculate_uncertainty_bounds(
        self,
        prediction_value: float,
        prediction_type: str,
        shap_result: Optional[SHAPResult],
        lime_result: Optional[LIMEResult],
    ) -> UncertaintyBounds:
        """Calculate uncertainty bounds for prediction."""
        # Estimate uncertainty from LIME score if available
        model_uncertainty = 0.1  # Default 10%
        if lime_result and lime_result.score > 0:
            model_uncertainty = max(0.05, 1 - lime_result.score) * 0.5

        # Data uncertainty based on prediction type
        data_uncertainty_rates = {
            "hotspot": 0.08,
            "efficiency": 0.03,
            "rul": 0.15,
        }
        data_uncertainty = data_uncertainty_rates.get(prediction_type, 0.1)

        # Combined uncertainty
        total_uncertainty = np.sqrt(model_uncertainty**2 + data_uncertainty**2)

        # Calculate bounds
        std_95 = prediction_value * total_uncertainty * 1.96
        std_80 = prediction_value * total_uncertainty * 1.28

        return UncertaintyBounds(
            prediction_value=prediction_value,
            lower_bound_95=max(0, prediction_value - std_95),
            upper_bound_95=prediction_value + std_95,
            lower_bound_80=max(0, prediction_value - std_80),
            upper_bound_80=prediction_value + std_80,
            confidence_method="combined_shap_lime",
            model_uncertainty=round(model_uncertainty, 4),
            data_uncertainty=round(data_uncertainty, 4),
        )

    def _generate_title(
        self,
        prediction_type: str,
        prediction_value: float,
        urgency: UrgencyLevel,
    ) -> str:
        """Generate report title."""
        type_names = {
            "hotspot": "Hotspot Detection",
            "efficiency": "Efficiency Assessment",
            "rul": "Remaining Useful Life",
        }
        type_name = type_names.get(prediction_type, prediction_type.title())

        if urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]:
            return f"[{urgency.value.upper()}] {type_name} Explainability Report"
        else:
            return f"{type_name} Explainability Report"

    def _generate_executive_summary(
        self,
        prediction_type: str,
        prediction_value: float,
        urgency: UrgencyLevel,
        audience: AudienceType,
        shap_result: Optional[SHAPResult],
        lime_result: Optional[LIMEResult],
        engineering_result: Optional[Any],
    ) -> str:
        """Generate executive summary tailored to audience."""
        config = self.AUDIENCE_CONFIG[audience]

        # Get top driver
        top_driver = "Unknown"
        if shap_result and shap_result.top_drivers:
            top_driver = shap_result.top_drivers[0].feature_name

        # Build summary based on audience
        if audience == AudienceType.OPERATOR:
            if prediction_type == "hotspot":
                if urgency == UrgencyLevel.CRITICAL:
                    summary = (
                        f"CRITICAL: Hotspot severity at {prediction_value:.0f}%. "
                        f"Primary cause: {top_driver}. "
                        "Immediate action required. Reduce firing rate and alert supervisor."
                    )
                elif urgency == UrgencyLevel.HIGH:
                    summary = (
                        f"HIGH ALERT: Elevated hotspot severity ({prediction_value:.0f}%). "
                        f"Key factor: {top_driver}. "
                        "Increase monitoring frequency and prepare for intervention."
                    )
                else:
                    summary = (
                        f"Hotspot monitoring: Severity at {prediction_value:.0f}% (within limits). "
                        "Continue normal operations with standard monitoring."
                    )
            elif prediction_type == "efficiency":
                summary = (
                    f"Furnace efficiency: {prediction_value:.1f}%. "
                    f"Primary factor: {top_driver}. "
                    f"{'Action needed to improve.' if prediction_value < 85 else 'Operating within target range.'}"
                )
            else:  # RUL
                summary = (
                    f"Estimated remaining life: {prediction_value:.0f} hours "
                    f"({prediction_value/24:.0f} days). "
                    f"Key degradation factor: {top_driver}."
                )

        elif audience == AudienceType.PROCESS_ENGINEER:
            summary = (
                f"Analysis of {prediction_type} prediction ({prediction_value:.2f}). "
                f"SHAP analysis identifies {top_driver} as primary driver "
                f"(importance: {shap_result.top_drivers[0].shap_value:.4f if shap_result and shap_result.top_drivers else 0:.4f}). "
            )
            if lime_result:
                summary += f"LIME local model R2: {lime_result.score:.3f}. "
            if engineering_result:
                summary += f"Engineering assessment: {engineering_result.engineering_summary[:100]}..."

        elif audience == AudienceType.SAFETY:
            safety_status = "requires immediate attention" if urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH] else "within acceptable limits"
            summary = (
                f"Safety assessment: {prediction_type.title()} {safety_status}. "
                f"Current value: {prediction_value:.1f}. "
                f"Urgency level: {urgency.value}. "
                "See safety implications section for required actions."
            )

        elif audience == AudienceType.MANAGEMENT:
            summary = (
                f"{prediction_type.title()} status: {urgency.value.upper()}. "
                f"Value: {prediction_value:.1f}. "
                f"Primary factor: {top_driver}. "
            )
            if urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]:
                summary += "Immediate resource allocation may be required."
            else:
                summary += "Situation is under control."

        elif audience == AudienceType.AUDIT:
            summary = (
                f"Explainability analysis for {prediction_type} prediction. "
                f"Predicted value: {prediction_value:.4f}. "
                f"Analysis methods: SHAP (global/local), LIME (local surrogate), "
                "engineering rationale (deterministic rules). "
                "All computations are reproducible with hash-verified provenance."
            )

        else:  # RELIABILITY_ENGINEER
            summary = (
                f"Component health assessment: {prediction_type} at {prediction_value:.2f}. "
                f"Degradation indicator: {top_driver}. "
            )
            if prediction_type == "rul":
                summary += f"Maintenance window: {prediction_value * 0.8 / 24:.0f} days recommended."

        return summary

    def _generate_sections(
        self,
        prediction_type: str,
        prediction_value: float,
        sensor_readings: Dict[str, float],
        shap_result: Optional[SHAPResult],
        lime_result: Optional[LIMEResult],
        engineering_result: Optional[Any],
        audience: AudienceType,
        model_card: Optional[ModelCard],
    ) -> List[ReportSection]:
        """Generate report sections based on audience."""
        sections = []
        config = self.AUDIENCE_CONFIG[audience]

        # Prediction details section
        sections.append(ReportSection(
            section_id="pred_details",
            title="Prediction Details",
            content=self._generate_prediction_details(prediction_type, prediction_value, sensor_readings),
            audience_relevance=[AudienceType.PROCESS_ENGINEER, AudienceType.RELIABILITY_ENGINEER, AudienceType.AUDIT],
        ))

        # Feature attribution section
        if shap_result and config["include_stats"]:
            sections.append(ReportSection(
                section_id="feature_attribution",
                title="Feature Attribution Analysis",
                content=self._generate_feature_attribution_content(shap_result, lime_result),
                data={"shap_values": shap_result.feature_importance},
                visualization_type="chart",
                audience_relevance=[AudienceType.PROCESS_ENGINEER, AudienceType.RELIABILITY_ENGINEER],
            ))

        # Engineering rationale section
        if engineering_result and "engineering_rationale" in config["focus"]:
            sections.append(ReportSection(
                section_id="engineering_rationale",
                title="Engineering Rationale",
                content=engineering_result.engineering_summary,
                subsections=[
                    ReportSection(
                        section_id="physical_phenomenon",
                        title="Physical Phenomenon",
                        content=f"{engineering_result.primary_phenomenon.name}: {engineering_result.primary_phenomenon.description}",
                    ),
                ],
                audience_relevance=[AudienceType.PROCESS_ENGINEER, AudienceType.RELIABILITY_ENGINEER],
            ))

        # Operator guidance section
        if audience == AudienceType.OPERATOR:
            sections.append(ReportSection(
                section_id="operator_guidance",
                title="Operator Guidance",
                content=engineering_result.operator_guidance if engineering_result else "Continue standard monitoring.",
                audience_relevance=[AudienceType.OPERATOR],
            ))

        # Safety implications section
        if audience == AudienceType.SAFETY or config["focus"] == ["safety_implications"]:
            safety_content = "\n".join(engineering_result.safety_implications) if engineering_result else "No specific safety concerns identified."
            sections.append(ReportSection(
                section_id="safety",
                title="Safety Implications",
                content=safety_content,
                audience_relevance=[AudienceType.SAFETY, AudienceType.OPERATOR],
            ))

        # Model information section (for audit)
        if config["include_model_details"] and model_card:
            sections.append(ReportSection(
                section_id="model_info",
                title="Model Information",
                content=(
                    f"Model: {model_card.model_name} v{model_card.model_version}\n"
                    f"Type: {model_card.model_type.value}\n"
                    f"Status: {model_card.status.value}\n"
                    f"Training samples: {model_card.training_dataset_size:,}"
                ),
                audience_relevance=[AudienceType.AUDIT, AudienceType.PROCESS_ENGINEER],
            ))

        return sections

    def _generate_prediction_details(
        self,
        prediction_type: str,
        prediction_value: float,
        sensor_readings: Dict[str, float],
    ) -> str:
        """Generate prediction details content."""
        content = f"Prediction type: {prediction_type}\n"
        content += f"Predicted value: {prediction_value:.4f}\n\n"
        content += "Key sensor readings:\n"

        # Show top 5 sensor readings
        for i, (name, value) in enumerate(sorted(sensor_readings.items())[:5]):
            content += f"  - {name}: {value:.2f}\n"

        return content

    def _generate_feature_attribution_content(
        self,
        shap_result: SHAPResult,
        lime_result: Optional[LIMEResult],
    ) -> str:
        """Generate feature attribution content."""
        content = "SHAP Analysis:\n"
        content += f"Base value: {shap_result.base_value:.4f}\n"
        content += f"Top contributing features:\n"

        for driver in shap_result.top_drivers[:5]:
            content += (
                f"  - {driver.feature_name}: {driver.direction} prediction by "
                f"{abs(driver.shap_value):.4f} ({driver.contribution_percent:.1f}%)\n"
            )

        if lime_result:
            content += f"\nLIME Analysis:\n"
            content += f"Local model R2: {lime_result.score:.4f}\n"
            content += f"Confidence: {lime_result.confidence_level.value}\n"

        return content

    def _extract_key_findings(
        self,
        shap_result: Optional[SHAPResult],
        lime_result: Optional[LIMEResult],
        engineering_result: Optional[Any],
        audience: AudienceType,
    ) -> List[str]:
        """Extract key findings from analyses."""
        findings = []

        if shap_result and shap_result.top_drivers:
            top = shap_result.top_drivers[0]
            findings.append(
                f"Primary driver: {top.feature_name} ({top.direction} prediction by {top.contribution_percent:.1f}%)"
            )

            if len(shap_result.top_drivers) > 1:
                second = shap_result.top_drivers[1]
                findings.append(
                    f"Secondary driver: {second.feature_name} ({second.contribution_percent:.1f}% contribution)"
                )

        if lime_result:
            if lime_result.confidence_level == ConfidenceLevel.HIGH:
                findings.append("Explanation confidence is HIGH - local model explains prediction well")
            elif lime_result.confidence_level == ConfidenceLevel.LOW:
                findings.append("Explanation confidence is LOW - prediction involves complex interactions")

        if engineering_result:
            findings.append(f"Physical phenomenon: {engineering_result.primary_phenomenon.name}")

        # Limit findings based on audience
        config = self.AUDIENCE_CONFIG[audience]
        if config["detail_level"] == "low":
            return findings[:3]
        elif config["detail_level"] == "medium":
            return findings[:5]
        else:
            return findings[:10]

    def _generate_recommendations(
        self,
        prediction_type: str,
        prediction_value: float,
        urgency: UrgencyLevel,
        engineering_result: Optional[Any],
        audience: AudienceType,
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Engineering-based recommendations
        if engineering_result and engineering_result.corrective_actions:
            for action in engineering_result.corrective_actions[:3]:
                recommendations.append(
                    f"[{action.priority.value.upper()}] {action.action_description}"
                )

        # Add urgency-based recommendations
        if urgency == UrgencyLevel.CRITICAL:
            recommendations.insert(0, "IMMEDIATE: Escalate to shift supervisor and safety personnel")

        elif urgency == UrgencyLevel.HIGH:
            recommendations.append("Increase monitoring frequency to 5-minute intervals")

        # Audience-specific recommendations
        if audience == AudienceType.OPERATOR:
            recommendations.append("Document observations in shift log")

        elif audience == AudienceType.MANAGEMENT:
            if urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]:
                recommendations.append("Allocate resources for immediate investigation")
                recommendations.append("Prepare communication for stakeholders")

        elif audience == AudienceType.RELIABILITY_ENGINEER:
            recommendations.append("Update maintenance schedule based on findings")
            recommendations.append("Review historical trend for pattern analysis")

        return recommendations[:6]  # Limit to 6 recommendations

    def _build_shap_summary(self, shap_result: SHAPResult) -> Dict[str, Any]:
        """Build SHAP summary dictionary."""
        return {
            "base_value": shap_result.base_value,
            "predicted_value": shap_result.predicted_value,
            "top_features": shap_result.feature_importance,
            "top_drivers": [
                {
                    "feature": d.feature_name,
                    "shap_value": d.shap_value,
                    "direction": d.direction,
                    "contribution_percent": d.contribution_percent,
                }
                for d in shap_result.top_drivers[:5]
            ],
            "computation_hash": shap_result.computation_hash,
        }

    def _build_lime_summary(self, lime_result: LIMEResult) -> Dict[str, Any]:
        """Build LIME summary dictionary."""
        return {
            "local_prediction": lime_result.local_prediction,
            "intercept": lime_result.intercept,
            "r2_score": lime_result.score,
            "confidence_level": lime_result.confidence_level.value,
            "feature_weights": lime_result.feature_weights,
            "computation_hash": lime_result.computation_hash,
        }

    def _build_engineering_summary(self, engineering_result: Any) -> Dict[str, Any]:
        """Build engineering summary dictionary."""
        return {
            "primary_phenomenon": engineering_result.primary_phenomenon.name,
            "phenomenon_category": engineering_result.primary_phenomenon.category.value,
            "root_cause": engineering_result.root_cause_analysis.primary_cause,
            "root_cause_confidence": engineering_result.root_cause_analysis.confidence,
            "num_corrective_actions": len(engineering_result.corrective_actions),
            "computation_hash": engineering_result.computation_hash,
        }

    def _build_audit_trail(
        self,
        report_id: str,
        generated_at: datetime,
        sensor_readings: Dict[str, float],
        shap_result: Optional[SHAPResult],
        lime_result: Optional[LIMEResult],
        engineering_result: Optional[Any],
    ) -> AuditTrail:
        """Build audit trail for report."""
        # Compute data hash
        data_str = json.dumps(sensor_readings, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()

        # Collect computation hashes
        computation_hashes = {}
        if shap_result:
            computation_hashes["shap"] = shap_result.computation_hash
        if lime_result:
            computation_hashes["lime"] = lime_result.computation_hash
        if engineering_result:
            computation_hashes["engineering"] = engineering_result.computation_hash

        # Build provenance chain
        provenance_chain = [
            {"step": "data_collection", "hash": data_hash[:16]},
        ]
        if shap_result:
            provenance_chain.append({"step": "shap_analysis", "hash": shap_result.computation_hash[:16]})
        if lime_result:
            provenance_chain.append({"step": "lime_analysis", "hash": lime_result.computation_hash[:16]})
        if engineering_result:
            provenance_chain.append({"step": "engineering_rationale", "hash": engineering_result.computation_hash[:16]})
        provenance_chain.append({"step": "report_generation", "hash": report_id})

        return AuditTrail(
            report_id=report_id,
            generated_at=generated_at,
            model_versions={
                "prediction_model": self.model_version,
                "shap_explainer": SHAPExplainer.VERSION,
                "lime_explainer": LIMEExplainer.VERSION,
                "engineering_rationale": EngineeringRationaleGenerator.VERSION,
                "report_generator": self.VERSION,
            },
            data_hash=data_hash,
            computation_hashes=computation_hashes,
            input_parameters={
                "num_sensors": len(sensor_readings),
                "include_shap": shap_result is not None,
                "include_lime": lime_result is not None,
                "include_engineering": engineering_result is not None,
            },
            provenance_chain=provenance_chain,
        )

    def _compute_hash(
        self,
        report_id: str,
        prediction_type: str,
        prediction_value: float,
        executive_summary: str,
    ) -> str:
        """Compute SHA-256 hash for report."""
        data = {
            "report_id": report_id,
            "prediction_type": prediction_type,
            "prediction_value": prediction_value,
            "summary_hash": hashlib.md5(executive_summary.encode()).hexdigest(),
            "generator_version": self.VERSION,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
