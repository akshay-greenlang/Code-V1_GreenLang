"""
ThermalIQ Explainability Report Generator

Generates comprehensive explainability reports combining SHAP, LIME,
and engineering rationale for thermal system analyses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
from pathlib import Path
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .shap_explainer import SHAPExplanation, ThermalSHAPExplainer
from .lime_explainer import LIMEExplanation, ThermalLIMEExplainer
from .engineering_rationale import EngineeringRationale, EngineeringRationaleGenerator


class ReportFormat(Enum):
    """Output formats for explainability reports."""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Recommendation:
    """A recommendation from the analysis."""
    title: str
    description: str
    priority: RecommendationPriority
    category: str
    expected_impact: str
    implementation_effort: str  # "low", "medium", "high"
    estimated_savings: Optional[float] = None
    savings_unit: str = "kW"
    related_features: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "category": self.category,
            "expected_impact": self.expected_impact,
            "implementation_effort": self.implementation_effort,
            "estimated_savings": self.estimated_savings,
            "savings_unit": self.savings_unit,
            "related_features": self.related_features,
            "citations": self.citations
        }


@dataclass
class ReportSection:
    """A section of the explainability report."""
    title: str
    content: str
    figures: List[go.Figure] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    subsections: List["ReportSection"] = field(default_factory=list)


@dataclass
class ExplainabilityReport:
    """Complete explainability report for thermal analysis."""
    title: str
    analysis_type: str
    timestamp: str
    executive_summary: str
    sections: List[ReportSection]
    recommendations: List[Recommendation]
    shap_explanation: Optional[SHAPExplanation] = None
    lime_explanation: Optional[LIMEExplanation] = None
    engineering_rationale: Optional[EngineeringRationale] = None
    figures: Dict[str, go.Figure] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary (without figures)."""
        return {
            "title": self.title,
            "analysis_type": self.analysis_type,
            "timestamp": self.timestamp,
            "executive_summary": self.executive_summary,
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "tables": s.tables
                }
                for s in self.sections
            ],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "shap_explanation": self.shap_explanation.to_dict() if self.shap_explanation else None,
            "lime_explanation": self.lime_explanation.to_dict() if self.lime_explanation else None,
            "engineering_rationale": self.engineering_rationale.to_dict() if self.engineering_rationale else None,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ExplainabilityReportGenerator:
    """
    Generates comprehensive explainability reports for thermal analyses.

    Combines SHAP explanations, LIME explanations, and engineering
    rationale into cohesive, actionable reports.
    """

    def __init__(
        self,
        include_visualizations: bool = True,
        include_technical_details: bool = True,
        audience: str = "engineer"  # "engineer", "operator", "manager"
    ):
        """
        Initialize the report generator.

        Args:
            include_visualizations: Whether to include Plotly figures
            include_technical_details: Whether to include detailed explanations
            audience: Target audience for report language
        """
        self.include_visualizations = include_visualizations
        self.include_technical_details = include_technical_details
        self.audience = audience
        self.rationale_generator = EngineeringRationaleGenerator(
            include_equations=(audience == "engineer"),
            include_standards=True,
            detail_level="detailed" if include_technical_details else "standard"
        )

    def generate_full_report(
        self,
        analysis_result: Dict[str, Any],
        shap_explanation: Optional[SHAPExplanation] = None,
        lime_explanation: Optional[LIMEExplanation] = None,
        model: Optional[Any] = None,
        title: Optional[str] = None
    ) -> ExplainabilityReport:
        """
        Generate a comprehensive explainability report.

        Args:
            analysis_result: Results from thermal analysis
            shap_explanation: Optional SHAP explanation
            lime_explanation: Optional LIME explanation
            model: Optional model for generating explanations
            title: Optional report title

        Returns:
            ExplainabilityReport with all components
        """
        # Determine analysis type
        analysis_type = self._determine_analysis_type(analysis_result)

        # Generate engineering rationale
        if analysis_type == "efficiency":
            rationale = self.rationale_generator.generate_efficiency_rationale(analysis_result)
        elif analysis_type == "exergy":
            rationale = self.rationale_generator.generate_exergy_rationale(analysis_result)
        elif analysis_type == "fluid_selection":
            rationale = self.rationale_generator.generate_fluid_recommendation_rationale(
                analysis_result.get("recommended_fluid", {}),
                analysis_result.get("alternatives", []),
                analysis_result.get("operating_conditions")
            )
        else:
            rationale = self.rationale_generator.generate_efficiency_rationale(analysis_result)

        # Generate executive summary
        summary = self.generate_summary(analysis_result, rationale)

        # Generate recommendations
        recommendations = self.generate_recommendations(
            analysis_result,
            shap_explanation,
            lime_explanation,
            rationale
        )

        # Build report sections
        sections = self._build_report_sections(
            analysis_result,
            shap_explanation,
            lime_explanation,
            rationale
        )

        # Generate figures
        figures = {}
        if self.include_visualizations and PLOTLY_AVAILABLE:
            figures = self._generate_figures(
                analysis_result,
                shap_explanation,
                lime_explanation
            )

        return ExplainabilityReport(
            title=title or f"Thermal Analysis Report - {analysis_type.title()}",
            analysis_type=analysis_type,
            timestamp=datetime.now().isoformat(),
            executive_summary=summary,
            sections=sections,
            recommendations=recommendations,
            shap_explanation=shap_explanation,
            lime_explanation=lime_explanation,
            engineering_rationale=rationale,
            figures=figures,
            metadata={
                "audience": self.audience,
                "include_technical_details": self.include_technical_details,
                "analysis_result_keys": list(analysis_result.keys())
            }
        )

    def generate_summary(
        self,
        result: Dict[str, Any],
        rationale: Optional[EngineeringRationale] = None
    ) -> str:
        """
        Generate executive summary for the analysis.

        Args:
            result: Analysis result
            rationale: Optional engineering rationale

        Returns:
            Executive summary text
        """
        analysis_type = self._determine_analysis_type(result)

        if analysis_type == "efficiency":
            return self._generate_efficiency_summary(result, rationale)
        elif analysis_type == "exergy":
            return self._generate_exergy_summary(result, rationale)
        elif analysis_type == "fluid_selection":
            return self._generate_fluid_summary(result, rationale)
        else:
            return self._generate_generic_summary(result, rationale)

    def generate_recommendations(
        self,
        result: Dict[str, Any],
        shap_explanation: Optional[SHAPExplanation] = None,
        lime_explanation: Optional[LIMEExplanation] = None,
        rationale: Optional[EngineeringRationale] = None
    ) -> List[Recommendation]:
        """
        Generate actionable recommendations from the analysis.

        Args:
            result: Analysis result
            shap_explanation: Optional SHAP explanation
            lime_explanation: Optional LIME explanation
            rationale: Optional engineering rationale

        Returns:
            List of prioritized recommendations
        """
        recommendations = []

        # Extract from rationale
        if rationale:
            for section in rationale.sections:
                for rec_text in section.recommendations:
                    recommendations.append(Recommendation(
                        title=rec_text[:50] + "..." if len(rec_text) > 50 else rec_text,
                        description=rec_text,
                        priority=RecommendationPriority.MEDIUM,
                        category=section.category.value,
                        expected_impact="Moderate improvement expected",
                        implementation_effort="medium"
                    ))

        # Extract from SHAP explanation
        if shap_explanation:
            for contrib in shap_explanation.top_negative_features[:3]:
                if contrib.shap_value < -0.1:
                    recommendations.append(Recommendation(
                        title=f"Address {contrib.feature_name}",
                        description=(
                            f"The feature '{contrib.feature_name}' (current value: "
                            f"{contrib.feature_value:.2f} {contrib.unit}) has a significant "
                            f"negative impact on the prediction (SHAP: {contrib.shap_value:.3f}). "
                            f"Adjusting this parameter could improve performance."
                        ),
                        priority=RecommendationPriority.HIGH,
                        category="optimization",
                        expected_impact=f"Potential improvement of {abs(contrib.shap_value):.2f} units",
                        implementation_effort="medium",
                        related_features=[contrib.feature_name]
                    ))

        # Add analysis-specific recommendations
        analysis_type = self._determine_analysis_type(result)

        if analysis_type == "efficiency":
            recommendations.extend(self._generate_efficiency_recommendations(result))
        elif analysis_type == "exergy":
            recommendations.extend(self._generate_exergy_recommendations(result))
        elif analysis_type == "fluid_selection":
            recommendations.extend(self._generate_fluid_recommendations(result))

        # Sort by priority
        priority_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3
        }
        recommendations.sort(key=lambda r: priority_order.get(r.priority, 4))

        # Remove duplicates (by title)
        seen_titles = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique_recommendations.append(rec)

        return unique_recommendations[:10]  # Top 10 recommendations

    def export_pdf(
        self,
        report: ExplainabilityReport,
        path: Union[str, Path]
    ) -> None:
        """
        Export report to PDF format.

        Args:
            report: ExplainabilityReport to export
            path: Output file path
        """
        path = Path(path)

        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                Image, PageBreak
            )
            from reportlab.lib import colors
            from io import BytesIO
        except ImportError:
            raise ImportError(
                "ReportLab is required for PDF export. Install with: pip install reportlab"
            )

        # Create PDF document
        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        heading_style = styles['Heading2']
        body_style = styles['Normal']

        # Build content
        story = []

        # Title
        story.append(Paragraph(report.title, title_style))
        story.append(Spacer(1, 12))

        # Metadata
        story.append(Paragraph(f"Generated: {report.timestamp}", body_style))
        story.append(Paragraph(f"Analysis Type: {report.analysis_type.title()}", body_style))
        story.append(Spacer(1, 24))

        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Paragraph(report.executive_summary, body_style))
        story.append(Spacer(1, 24))

        # Sections
        for section in report.sections:
            story.append(Paragraph(section.title, heading_style))
            story.append(Paragraph(section.content, body_style))

            # Add tables if present
            for table_data in section.tables:
                if 'data' in table_data:
                    t = Table(table_data['data'])
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(t)

            story.append(Spacer(1, 12))

        # Recommendations
        story.append(PageBreak())
        story.append(Paragraph("Recommendations", heading_style))

        for i, rec in enumerate(report.recommendations, 1):
            rec_text = f"{i}. [{rec.priority.value.upper()}] {rec.title}"
            story.append(Paragraph(rec_text, styles['Heading3']))
            story.append(Paragraph(rec.description, body_style))
            story.append(Paragraph(
                f"Expected Impact: {rec.expected_impact} | Effort: {rec.implementation_effort}",
                body_style
            ))
            story.append(Spacer(1, 12))

        # Build PDF
        doc.build(story)

    def export_html(
        self,
        report: ExplainabilityReport,
        path: Union[str, Path],
        include_interactive_plots: bool = True
    ) -> None:
        """
        Export report to HTML format.

        Args:
            report: ExplainabilityReport to export
            path: Output file path
            include_interactive_plots: Whether to embed Plotly plots
        """
        path = Path(path)

        html_content = self._generate_html_content(report, include_interactive_plots)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_html_content(
        self,
        report: ExplainabilityReport,
        include_plots: bool
    ) -> str:
        """Generate HTML content for the report."""
        # CSS styles
        css = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .report-container {
                background-color: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }
            h3 {
                color: #7f8c8d;
            }
            .metadata {
                color: #7f8c8d;
                font-size: 0.9em;
                margin-bottom: 30px;
            }
            .executive-summary {
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }
            .recommendation {
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 15px 0;
                background-color: #fafafa;
            }
            .recommendation.critical {
                border-color: #e74c3c;
                background-color: #fdf2f2;
            }
            .recommendation.high {
                border-color: #f39c12;
                background-color: #fef9e7;
            }
            .recommendation.medium {
                border-color: #3498db;
            }
            .recommendation.low {
                border-color: #27ae60;
                background-color: #f0f9f4;
            }
            .priority-badge {
                display: inline-block;
                padding: 3px 10px;
                border-radius: 3px;
                font-size: 0.8em;
                font-weight: bold;
                text-transform: uppercase;
            }
            .priority-critical { background-color: #e74c3c; color: white; }
            .priority-high { background-color: #f39c12; color: white; }
            .priority-medium { background-color: #3498db; color: white; }
            .priority-low { background-color: #27ae60; color: white; }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #3498db;
                color: white;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .section {
                margin: 30px 0;
            }
            .figure-container {
                margin: 20px 0;
                text-align: center;
            }
            .key-values {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .key-value-card {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                border-left: 3px solid #3498db;
            }
            .key-value-card .label {
                font-size: 0.85em;
                color: #7f8c8d;
            }
            .key-value-card .value {
                font-size: 1.2em;
                font-weight: bold;
                color: #2c3e50;
            }
        </style>
        """

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{report.title}</title>",
            css,
            "</head>",
            "<body>",
            "<div class='report-container'>",
            f"<h1>{report.title}</h1>",
            f"<div class='metadata'>",
            f"<p>Generated: {report.timestamp} | Analysis Type: {report.analysis_type.title()}</p>",
            "</div>",
            "<h2>Executive Summary</h2>",
            f"<div class='executive-summary'>{report.executive_summary}</div>"
        ]

        # Add sections
        for section in report.sections:
            html_parts.append(f"<div class='section'>")
            html_parts.append(f"<h2>{section.title}</h2>")
            html_parts.append(f"<p>{section.content}</p>")

            # Add tables
            for table in section.tables:
                html_parts.append(self._generate_html_table(table))

            html_parts.append("</div>")

        # Add figures
        if include_plots and report.figures:
            html_parts.append("<h2>Visualizations</h2>")
            for name, fig in report.figures.items():
                html_parts.append(f"<div class='figure-container'>")
                html_parts.append(f"<h3>{name}</h3>")
                html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                html_parts.append("</div>")

        # Add recommendations
        html_parts.append("<h2>Recommendations</h2>")
        for rec in report.recommendations:
            priority_class = rec.priority.value
            html_parts.append(f"<div class='recommendation {priority_class}'>")
            html_parts.append(
                f"<h3>{rec.title} "
                f"<span class='priority-badge priority-{priority_class}'>{rec.priority.value}</span>"
                f"</h3>"
            )
            html_parts.append(f"<p>{rec.description}</p>")
            html_parts.append(
                f"<p><strong>Expected Impact:</strong> {rec.expected_impact} | "
                f"<strong>Effort:</strong> {rec.implementation_effort}</p>"
            )
            if rec.estimated_savings:
                html_parts.append(
                    f"<p><strong>Estimated Savings:</strong> "
                    f"{rec.estimated_savings:.2f} {rec.savings_unit}</p>"
                )
            html_parts.append("</div>")

        # Close HTML
        html_parts.extend([
            "</div>",  # report-container
            "</body>",
            "</html>"
        ])

        return "\n".join(html_parts)

    def _generate_html_table(self, table_data: Dict[str, Any]) -> str:
        """Generate HTML table from table data."""
        if 'data' not in table_data:
            return ""

        data = table_data['data']
        if not data:
            return ""

        html = ["<table>"]

        # Header row
        if data:
            html.append("<thead><tr>")
            for cell in data[0]:
                html.append(f"<th>{cell}</th>")
            html.append("</tr></thead>")

        # Body rows
        html.append("<tbody>")
        for row in data[1:]:
            html.append("<tr>")
            for cell in row:
                html.append(f"<td>{cell}</td>")
            html.append("</tr>")
        html.append("</tbody>")

        html.append("</table>")
        return "\n".join(html)

    def _determine_analysis_type(self, result: Dict[str, Any]) -> str:
        """Determine the type of analysis from result keys."""
        if 'exergy_efficiency' in result or 'exergy_destruction' in result:
            return "exergy"
        elif 'recommended_fluid' in result or 'fluid_scores' in result:
            return "fluid_selection"
        elif 'efficiency' in result or 'heat_input' in result:
            return "efficiency"
        else:
            return "general"

    def _build_report_sections(
        self,
        result: Dict[str, Any],
        shap_explanation: Optional[SHAPExplanation],
        lime_explanation: Optional[LIMEExplanation],
        rationale: Optional[EngineeringRationale]
    ) -> List[ReportSection]:
        """Build report sections from analysis components."""
        sections = []

        # Analysis results section
        results_section = self._build_results_section(result)
        sections.append(results_section)

        # Engineering rationale sections
        if rationale:
            for rat_section in rationale.sections:
                sections.append(ReportSection(
                    title=rat_section.title,
                    content=rat_section.content,
                    tables=self._build_key_values_table(rat_section.key_values)
                ))

        # SHAP explanation section
        if shap_explanation and self.include_technical_details:
            sections.append(self._build_shap_section(shap_explanation))

        # LIME explanation section
        if lime_explanation and self.include_technical_details:
            sections.append(self._build_lime_section(lime_explanation))

        return sections

    def _build_results_section(self, result: Dict[str, Any]) -> ReportSection:
        """Build the analysis results section."""
        content_lines = ["The thermal analysis produced the following key results:"]

        # Format key results
        key_metrics = {}
        for key, value in result.items():
            if isinstance(value, (int, float)):
                key_metrics[key.replace('_', ' ').title()] = f"{value:.2f}"
            elif isinstance(value, str):
                key_metrics[key.replace('_', ' ').title()] = value

        content = "\n".join(content_lines)

        return ReportSection(
            title="Analysis Results",
            content=content,
            tables=self._build_key_values_table(key_metrics)
        )

    def _build_key_values_table(self, key_values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build table data from key-value pairs."""
        if not key_values:
            return []

        data = [["Parameter", "Value"]]
        for key, value in key_values.items():
            data.append([key, str(value)])

        return [{"data": data}]

    def _build_shap_section(self, explanation: SHAPExplanation) -> ReportSection:
        """Build SHAP explanation section."""
        content = (
            f"SHAP (SHapley Additive exPlanations) analysis reveals how each input "
            f"feature contributes to the prediction. The base prediction is "
            f"{explanation.base_value:.4f}, and the final prediction is "
            f"{explanation.predicted_value:.4f}.\n\n"
            f"Top contributing features:\n"
        )

        for contrib in explanation.feature_contributions[:5]:
            direction = "increases" if contrib.shap_value > 0 else "decreases"
            content += (
                f"- {contrib.feature_name} ({contrib.feature_value:.2f} {contrib.unit}): "
                f"{direction} prediction by {abs(contrib.shap_value):.4f}\n"
            )

        # Build feature importance table
        table_data = [["Feature", "Value", "SHAP Value", "Impact"]]
        for contrib in sorted(
            explanation.feature_contributions,
            key=lambda x: abs(x.shap_value),
            reverse=True
        )[:10]:
            impact = "+" if contrib.shap_value > 0 else "-"
            table_data.append([
                contrib.feature_name,
                f"{contrib.feature_value:.2f}",
                f"{contrib.shap_value:.4f}",
                impact * min(5, int(abs(contrib.shap_value) * 10) + 1)
            ])

        return ReportSection(
            title="SHAP Feature Importance",
            content=content,
            tables=[{"data": table_data}]
        )

    def _build_lime_section(self, explanation: LIMEExplanation) -> ReportSection:
        """Build LIME explanation section."""
        content = (
            f"LIME (Local Interpretable Model-agnostic Explanations) provides a "
            f"local linear approximation of the model's behavior around this instance.\n\n"
            f"Predicted value: {explanation.predicted_value:.4f}\n"
            f"Local model R-squared: {explanation.local_model.score:.3f}\n\n"
            f"Key factors affecting this prediction:\n"
        )

        for fw in explanation.feature_weights[:5]:
            direction = "increases" if fw.weight > 0 else "decreases"
            content += f"- {fw.condition}: {direction} prediction\n"

        return ReportSection(
            title="LIME Local Explanation",
            content=content,
            tables=[]
        )

    def _generate_figures(
        self,
        result: Dict[str, Any],
        shap_explanation: Optional[SHAPExplanation],
        lime_explanation: Optional[LIMEExplanation]
    ) -> Dict[str, go.Figure]:
        """Generate visualization figures for the report."""
        figures = {}

        if not PLOTLY_AVAILABLE:
            return figures

        # SHAP feature importance plot
        if shap_explanation:
            try:
                shap_explainer = ThermalSHAPExplainer.__new__(ThermalSHAPExplainer)
                shap_explainer.THERMAL_FEATURE_METADATA = ThermalSHAPExplainer.THERMAL_FEATURE_METADATA
                figures["SHAP Feature Importance"] = shap_explainer.feature_importance_plot(
                    shap_explanation
                )
            except Exception:
                pass

        # LIME explanation plot
        if lime_explanation:
            try:
                lime_explainer = ThermalLIMEExplainer.__new__(ThermalLIMEExplainer)
                figures["LIME Explanation"] = lime_explainer.plot_explanation(
                    lime_explanation
                )
            except Exception:
                pass

        return figures

    def _generate_efficiency_summary(
        self,
        result: Dict[str, Any],
        rationale: Optional[EngineeringRationale]
    ) -> str:
        """Generate efficiency analysis summary."""
        efficiency = result.get('efficiency', 0)
        heat_input = result.get('heat_input', 0)
        losses = result.get('losses', {})

        summary = (
            f"This thermal efficiency analysis evaluates equipment performance "
            f"based on First Law of Thermodynamics principles. "
        )

        if efficiency:
            summary += f"The calculated efficiency is {efficiency:.1f}%. "

        if heat_input and losses:
            total_losses = sum(losses.values())
            summary += (
                f"Of the {heat_input:.1f} kW heat input, {total_losses:.1f} kW "
                f"({total_losses/heat_input*100:.1f}%) is lost to the environment. "
            )

        if rationale:
            summary += rationale.overall_assessment

        return summary

    def _generate_exergy_summary(
        self,
        result: Dict[str, Any],
        rationale: Optional[EngineeringRationale]
    ) -> str:
        """Generate exergy analysis summary."""
        exergy_eff = result.get('exergy_efficiency', 0)
        destruction = result.get('exergy_destruction', 0)

        summary = (
            f"This Second Law analysis quantifies thermodynamic irreversibilities "
            f"and improvement potential beyond first-law efficiency. "
        )

        if exergy_eff:
            summary += f"The exergy efficiency is {exergy_eff:.1f}%, "

        if destruction:
            summary += f"with {destruction:.1f} kW of exergy destruction. "

        if rationale:
            summary += rationale.overall_assessment

        return summary

    def _generate_fluid_summary(
        self,
        result: Dict[str, Any],
        rationale: Optional[EngineeringRationale]
    ) -> str:
        """Generate fluid selection summary."""
        fluid = result.get('recommended_fluid', {})
        fluid_name = fluid.get('name', 'The recommended fluid')

        summary = (
            f"This analysis evaluates heat transfer fluids based on thermophysical "
            f"properties and operating conditions suitability. "
            f"{fluid_name} is recommended as the optimal choice. "
        )

        if rationale:
            summary += rationale.overall_assessment

        return summary

    def _generate_generic_summary(
        self,
        result: Dict[str, Any],
        rationale: Optional[EngineeringRationale]
    ) -> str:
        """Generate generic analysis summary."""
        summary = (
            "This thermal analysis evaluates system performance based on "
            "thermodynamic principles. Key findings are presented with "
            "engineering rationale and actionable recommendations."
        )

        if rationale:
            summary += " " + rationale.summary

        return summary

    def _generate_efficiency_recommendations(
        self,
        result: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate efficiency-specific recommendations."""
        recommendations = []
        efficiency = result.get('efficiency', 100)
        losses = result.get('losses', {})

        if efficiency < 80:
            recommendations.append(Recommendation(
                title="Conduct Comprehensive Energy Audit",
                description=(
                    "With efficiency below 80%, a detailed energy audit is recommended "
                    "to identify all improvement opportunities and prioritize investments."
                ),
                priority=RecommendationPriority.HIGH,
                category="audit",
                expected_impact="Identify 10-15% efficiency improvement potential",
                implementation_effort="medium"
            ))

        # Loss-specific recommendations
        for loss_name, loss_value in sorted(losses.items(), key=lambda x: x[1], reverse=True)[:2]:
            if "stack" in loss_name.lower() and loss_value > 50:
                recommendations.append(Recommendation(
                    title="Install Economizer for Stack Heat Recovery",
                    description=(
                        f"Stack losses of {loss_value:.1f} kW represent significant "
                        f"recoverable energy. An economizer can capture this heat "
                        f"to preheat feedwater or combustion air."
                    ),
                    priority=RecommendationPriority.HIGH,
                    category="heat_recovery",
                    expected_impact=f"Recover up to {loss_value*0.5:.1f} kW",
                    implementation_effort="high",
                    estimated_savings=loss_value * 0.5,
                    savings_unit="kW"
                ))

        return recommendations

    def _generate_exergy_recommendations(
        self,
        result: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate exergy-specific recommendations."""
        recommendations = []
        irreversibilities = result.get('irreversibilities', {})

        for source, value in sorted(irreversibilities.items(), key=lambda x: x[1], reverse=True)[:2]:
            if "combustion" in source.lower():
                recommendations.append(Recommendation(
                    title="Reduce Combustion Irreversibility",
                    description=(
                        f"Combustion irreversibility of {value:.1f} kW can be reduced "
                        f"through air preheating, oxygen enrichment, or staged combustion."
                    ),
                    priority=RecommendationPriority.MEDIUM,
                    category="combustion",
                    expected_impact="10-20% reduction in combustion exergy destruction",
                    implementation_effort="high"
                ))
            elif "heat_transfer" in source.lower():
                recommendations.append(Recommendation(
                    title="Optimize Heat Transfer Surfaces",
                    description=(
                        f"Heat transfer irreversibility of {value:.1f} kW indicates "
                        f"large temperature differences. Consider increasing heat "
                        f"transfer area or using counterflow arrangements."
                    ),
                    priority=RecommendationPriority.MEDIUM,
                    category="heat_transfer",
                    expected_impact="Reduce temperature approach, improve exergy efficiency",
                    implementation_effort="medium"
                ))

        return recommendations

    def _generate_fluid_recommendations(
        self,
        result: Dict[str, Any]
    ) -> List[Recommendation]:
        """Generate fluid selection recommendations."""
        recommendations = []
        fluid = result.get('recommended_fluid', {})
        conditions = result.get('operating_conditions', {})

        recommendations.append(Recommendation(
            title="Verify Material Compatibility",
            description=(
                f"Before implementing {fluid.get('name', 'the selected fluid')}, "
                f"verify compatibility with all system materials including seals, "
                f"gaskets, and heat exchanger surfaces."
            ),
            priority=RecommendationPriority.HIGH,
            category="implementation",
            expected_impact="Prevent premature system failures",
            implementation_effort="low"
        ))

        if conditions.get('max_temperature', 0) > 200:
            recommendations.append(Recommendation(
                title="Implement Nitrogen Blanketing",
                description=(
                    "For high-temperature operation, nitrogen blanketing of the "
                    "expansion tank prevents oxidation and extends fluid life."
                ),
                priority=RecommendationPriority.MEDIUM,
                category="implementation",
                expected_impact="Extend fluid service life by 2-3x",
                implementation_effort="medium"
            ))

        return recommendations
