"""
Supplier Benchmark Report Template - PACK-007 EUDR Professional Pack

This module generates supplier benchmarking reports with industry comparisons, peer group
rankings, percentile analysis, scoring dimension breakdowns, improvement trends, best practices,
and recommendations for EUDR compliance.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from supplier_benchmark_report import SupplierBenchmarkReportTemplate, ReportData
    >>> data = ReportData(
    ...     operator_name="Global Supply Co",
    ...     report_date="2026-03-15",
    ...     supplier_name="Acme Forest Products",
    ...     supplier_id="SUP-12345"
    ... )
    >>> template = SupplierBenchmarkReportTemplate()
    >>> report = template.render(data, format="markdown")
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ReportConfig(BaseModel):
    """Configuration for Supplier Benchmark Report generation."""

    include_peer_rankings: bool = Field(
        default=True,
        description="Include peer group rankings"
    )
    include_percentile_charts: bool = Field(
        default=True,
        description="Include percentile charts (text-based)"
    )
    include_trends: bool = Field(
        default=True,
        description="Include historical improvement trends"
    )
    include_best_practices: bool = Field(
        default=True,
        description="Include best practices from top performers"
    )
    peer_group_size: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of peers in comparison group"
    )


class SupplierScore(BaseModel):
    """Supplier scoring information."""

    supplier_id: str = Field(..., description="Supplier identifier")
    supplier_name: str = Field(..., description="Supplier name")
    overall_score: float = Field(..., ge=0, le=100, description="Overall compliance score")
    traceability_score: float = Field(..., ge=0, le=100, description="Traceability score")
    due_diligence_score: float = Field(..., ge=0, le=100, description="Due diligence score")
    risk_management_score: float = Field(..., ge=0, le=100, description="Risk management score")
    documentation_score: float = Field(..., ge=0, le=100, description="Documentation score")
    certification_score: float = Field(..., ge=0, le=100, description="Certification score")


class PeerRanking(BaseModel):
    """Peer group ranking information."""

    rank: int = Field(..., ge=1, description="Rank in peer group")
    supplier_name: str = Field(..., description="Supplier name (anonymized for peers)")
    overall_score: float = Field(..., ge=0, le=100, description="Overall score")
    industry_segment: str = Field(..., description="Industry segment")
    country: str = Field(..., description="Country of operation")
    is_subject: bool = Field(
        default=False,
        description="True if this is the subject supplier being benchmarked"
    )


class PercentileScore(BaseModel):
    """Percentile score information."""

    dimension: str = Field(..., description="Scoring dimension")
    score: float = Field(..., ge=0, le=100, description="Actual score")
    percentile: float = Field(..., ge=0, le=100, description="Percentile rank (0-100)")
    peer_average: float = Field(..., ge=0, le=100, description="Peer group average")
    top_quartile_threshold: float = Field(..., ge=0, le=100, description="Top 25% threshold")


class TrendDataPoint(BaseModel):
    """Historical trend data point."""

    period: str = Field(..., description="Time period (e.g., 2025-Q4)")
    score: float = Field(..., ge=0, le=100, description="Score for this period")
    peer_average: float = Field(..., ge=0, le=100, description="Peer average for this period")


class BestPractice(BaseModel):
    """Best practice from top performers."""

    practice_area: str = Field(..., description="Area of practice")
    description: str = Field(..., description="Description of the practice")
    adopted_by_top_performers: int = Field(
        ...,
        ge=0,
        description="Number of top performers using this practice"
    )
    estimated_impact: str = Field(..., description="Estimated impact (HIGH, MEDIUM, LOW)")


class ReportData(BaseModel):
    """Data model for Supplier Benchmark Report."""

    operator_name: str = Field(..., description="Operator name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    supplier_id: str = Field(..., description="Subject supplier identifier")
    supplier_name: str = Field(..., description="Subject supplier name")
    supplier_score: SupplierScore = Field(..., description="Subject supplier scores")
    peer_rankings: List[PeerRanking] = Field(
        default_factory=list,
        description="Peer group rankings"
    )
    percentile_scores: List[PercentileScore] = Field(
        default_factory=list,
        description="Percentile scores by dimension"
    )
    trends: Dict[str, List[TrendDataPoint]] = Field(
        default_factory=dict,
        description="Historical trends by dimension"
    )
    best_practices: List[BestPractice] = Field(
        default_factory=list,
        description="Best practices from top performers"
    )
    improvement_recommendations: List[str] = Field(
        default_factory=list,
        description="Specific recommendations for improvement"
    )
    industry_segment: str = Field(..., description="Industry segment")
    peer_group_description: str = Field(..., description="Description of peer group")

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class SupplierBenchmarkReportTemplate:
    """
    Supplier Benchmark Report Template for EUDR Professional Pack.

    Generates comprehensive supplier benchmarking reports with industry comparisons,
    peer group rankings, percentile analysis, scoring dimension breakdowns, improvement
    trends, and best practices.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = SupplierBenchmarkReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Peer Group Rankings" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Supplier Benchmark Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the supplier benchmark report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Supplier Benchmark Report for {data.supplier_name} in {format} format"
        )

        if format == "markdown":
            content = self._render_markdown(data)
        elif format == "html":
            content = self._render_html(data)
        elif format == "json":
            content = self._render_json(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Add provenance hash
        content_hash = self._calculate_hash(content)
        logger.info(f"Report generated with hash: {content_hash}")

        return content

    def _render_markdown(self, data: ReportData) -> str:
        """Render report in Markdown format."""
        sections = []

        # Header
        sections.append(f"# Supplier Benchmark Report")
        sections.append(f"")
        sections.append(f"**Operator:** {data.operator_name}")
        sections.append(f"**Supplier:** {data.supplier_name} ({data.supplier_id})")
        sections.append(f"**Industry Segment:** {data.industry_segment}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        score = data.supplier_score
        sections.append(
            f"This benchmark report compares **{data.supplier_name}** against {data.peer_group_description}. "
            f"The supplier achieved an overall compliance score of **{score.overall_score:.1f}/100**, "
            f"which places them in the "
        )

        # Calculate rank from peer rankings
        subject_ranking = next(
            (r for r in data.peer_rankings if r.is_subject),
            None
        )
        if subject_ranking:
            total_peers = len(data.peer_rankings)
            percentile = ((total_peers - subject_ranking.rank + 1) / total_peers) * 100
            sections.append(
                f"**{self._get_percentile_description(percentile)}** "
                f"(rank {subject_ranking.rank} of {total_peers})."
            )
        else:
            sections.append("peer group.")

        sections.append(f"")

        # Overall Performance Summary
        sections.append(f"## Overall Performance Summary")
        sections.append(f"")
        sections.append(f"### Scoring Dimensions")
        sections.append(f"")
        sections.append(f"| Dimension | Score | Status |")
        sections.append(f"|-----------|-------|--------|")

        dimensions = [
            ("Overall Score", score.overall_score),
            ("Traceability", score.traceability_score),
            ("Due Diligence", score.due_diligence_score),
            ("Risk Management", score.risk_management_score),
            ("Documentation", score.documentation_score),
            ("Certification", score.certification_score),
        ]

        for dim_name, dim_score in dimensions:
            status = self._get_score_status(dim_score)
            sections.append(f"| {dim_name} | {dim_score:.1f}/100 | {status} |")
        sections.append(f"")

        # Peer Group Rankings
        if self.config.include_peer_rankings and data.peer_rankings:
            sections.append(f"## Peer Group Rankings")
            sections.append(f"")
            sections.append(
                f"Comparison against {len(data.peer_rankings)} suppliers in the {data.industry_segment} segment:"
            )
            sections.append(f"")

            sections.append(f"| Rank | Supplier | Score | Country | Segment |")
            sections.append(f"|------|----------|-------|---------|---------|")

            for ranking in data.peer_rankings[:20]:
                supplier_display = f"**{ranking.supplier_name}**" if ranking.is_subject else ranking.supplier_name
                sections.append(
                    f"| {ranking.rank} | {supplier_display} | {ranking.overall_score:.1f} | "
                    f"{ranking.country} | {ranking.industry_segment} |"
                )
            sections.append(f"")

        # Percentile Analysis
        if self.config.include_percentile_charts and data.percentile_scores:
            sections.append(f"## Percentile Analysis")
            sections.append(f"")
            sections.append(
                f"Detailed percentile rankings across scoring dimensions:"
            )
            sections.append(f"")

            sections.append(
                f"| Dimension | Score | Percentile | Peer Avg | Top 25% |"
            )
            sections.append(
                f"|-----------|-------|------------|----------|---------|"
            )

            for perc in sorted(data.percentile_scores, key=lambda x: x.percentile, reverse=True):
                sections.append(
                    f"| {perc.dimension} | {perc.score:.1f} | {perc.percentile:.0f}th | "
                    f"{perc.peer_average:.1f} | {perc.top_quartile_threshold:.1f} |"
                )
            sections.append(f"")

            # Percentile chart
            sections.append(f"### Percentile Visualization")
            sections.append(f"")
            sections.append(f"```")
            sections.append(self._create_percentile_chart(data.percentile_scores))
            sections.append(f"```")
            sections.append(f"")

        # Improvement Trends
        if self.config.include_trends and data.trends:
            sections.append(f"## Improvement Trends")
            sections.append(f"")
            sections.append(
                f"Historical performance trends showing progress over time:"
            )
            sections.append(f"")

            for dimension, trend_data in data.trends.items():
                if not trend_data:
                    continue

                sections.append(f"### {dimension}")
                sections.append(f"")
                sections.append(f"| Period | Score | Peer Avg | Delta |")
                sections.append(f"|--------|-------|----------|-------|")

                for point in trend_data:
                    delta = point.score - point.peer_average
                    delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
                    sections.append(
                        f"| {point.period} | {point.score:.1f} | {point.peer_average:.1f} | {delta_str} |"
                    )
                sections.append(f"")

                # Trend analysis
                if len(trend_data) >= 2:
                    first = trend_data[0]
                    last = trend_data[-1]
                    change = last.score - first.score
                    trend_direction = "improved" if change > 0 else "declined"
                    sections.append(
                        f"**Trend:** Score {trend_direction} by {abs(change):.1f} points "
                        f"from {first.period} to {last.period}."
                    )
                    sections.append(f"")

        # Best Practices
        if self.config.include_best_practices and data.best_practices:
            sections.append(f"## Best Practices from Top Performers")
            sections.append(f"")
            sections.append(
                f"The following practices are commonly adopted by top-performing suppliers:"
            )
            sections.append(f"")

            sections.append(f"| Practice Area | Description | Top Performers | Impact |")
            sections.append(f"|---------------|-------------|----------------|--------|")

            for practice in data.best_practices:
                sections.append(
                    f"| {practice.practice_area} | {practice.description} | "
                    f"{practice.adopted_by_top_performers} | {practice.estimated_impact} |"
                )
            sections.append(f"")

        # Improvement Recommendations
        if data.improvement_recommendations:
            sections.append(f"## Improvement Recommendations")
            sections.append(f"")
            sections.append(
                f"Based on the benchmarking analysis, the following improvements are recommended "
                f"in priority order:"
            )
            sections.append(f"")
            for idx, rec in enumerate(data.improvement_recommendations, 1):
                sections.append(f"{idx}. {rec}")
            sections.append(f"")

        # Gap Analysis
        sections.append(f"## Gap Analysis")
        sections.append(f"")
        sections.append(f"### Gaps vs. Top Quartile")
        sections.append(f"")

        for perc in data.percentile_scores:
            gap = perc.top_quartile_threshold - perc.score
            if gap > 0:
                sections.append(
                    f"- **{perc.dimension}**: {gap:.1f} points below top quartile threshold "
                    f"({perc.top_quartile_threshold:.1f})"
                )
        sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(f"**Peer Group Definition:** {data.peer_group_description}")
        sections.append(f"")
        sections.append(
            f"*Report generated on {data.report_date} using GreenLang EUDR Professional Pack*"
        )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render report in HTML format."""
        score = data.supplier_score

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Supplier Benchmark Report - {data.supplier_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #9b59b6; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #9b59b6; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #8e44ad; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .excellent {{ color: #27ae60; font-weight: bold; }}
        .good {{ color: #f39c12; }}
        .needs-improvement {{ color: #e74c3c; }}
        pre {{ background-color: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Supplier Benchmark Report</h1>

    <div class="summary">
        <p><strong>Operator:</strong> {data.operator_name}</p>
        <p><strong>Supplier:</strong> {data.supplier_name} ({data.supplier_id})</p>
        <p><strong>Industry Segment:</strong> {data.industry_segment}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Overall Score:</strong> <span class="metric">{score.overall_score:.1f}/100</span></p>
    </div>

    <h2>Overall Performance Summary</h2>
    <h3>Scoring Dimensions</h3>
    <table>
        <tr><th>Dimension</th><th>Score</th><th>Status</th></tr>
        <tr><td>Overall Score</td><td>{score.overall_score:.1f}/100</td><td>{self._get_score_status_html(score.overall_score)}</td></tr>
        <tr><td>Traceability</td><td>{score.traceability_score:.1f}/100</td><td>{self._get_score_status_html(score.traceability_score)}</td></tr>
        <tr><td>Due Diligence</td><td>{score.due_diligence_score:.1f}/100</td><td>{self._get_score_status_html(score.due_diligence_score)}</td></tr>
        <tr><td>Risk Management</td><td>{score.risk_management_score:.1f}/100</td><td>{self._get_score_status_html(score.risk_management_score)}</td></tr>
        <tr><td>Documentation</td><td>{score.documentation_score:.1f}/100</td><td>{self._get_score_status_html(score.documentation_score)}</td></tr>
        <tr><td>Certification</td><td>{score.certification_score:.1f}/100</td><td>{self._get_score_status_html(score.certification_score)}</td></tr>
    </table>
"""

        if self.config.include_peer_rankings and data.peer_rankings:
            html += f"""
    <h2>Peer Group Rankings</h2>
    <table>
        <tr><th>Rank</th><th>Supplier</th><th>Score</th><th>Country</th><th>Segment</th></tr>
"""
            for ranking in data.peer_rankings[:20]:
                row_style = ' style="background-color: #fff3cd;"' if ranking.is_subject else ''
                html += f"""        <tr{row_style}>
            <td>{ranking.rank}</td>
            <td>{ranking.supplier_name}</td>
            <td>{ranking.overall_score:.1f}</td>
            <td>{ranking.country}</td>
            <td>{ranking.industry_segment}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if self.config.include_percentile_charts and data.percentile_scores:
            html += f"""
    <h2>Percentile Analysis</h2>
    <table>
        <tr><th>Dimension</th><th>Score</th><th>Percentile</th><th>Peer Avg</th><th>Top 25%</th></tr>
"""
            for perc in sorted(data.percentile_scores, key=lambda x: x.percentile, reverse=True):
                html += f"""        <tr>
            <td>{perc.dimension}</td>
            <td>{perc.score:.1f}</td>
            <td>{perc.percentile:.0f}th</td>
            <td>{perc.peer_average:.1f}</td>
            <td>{perc.top_quartile_threshold:.1f}</td>
        </tr>
"""
            html += f"""    </table>
    <pre>{self._create_percentile_chart(data.percentile_scores)}</pre>
"""

        if self.config.include_best_practices and data.best_practices:
            html += f"""
    <h2>Best Practices from Top Performers</h2>
    <table>
        <tr><th>Practice Area</th><th>Description</th><th>Top Performers</th><th>Impact</th></tr>
"""
            for practice in data.best_practices:
                html += f"""        <tr>
            <td>{practice.practice_area}</td>
            <td>{practice.description}</td>
            <td>{practice.adopted_by_top_performers}</td>
            <td>{practice.estimated_impact}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if data.improvement_recommendations:
            html += f"""
    <h2>Improvement Recommendations</h2>
    <ol>
"""
            for rec in data.improvement_recommendations:
                html += f"        <li>{rec}</li>\n"
            html += f"""    </ol>
"""

        html += f"""
    <div class="footer">
        <p><strong>Peer Group:</strong> {data.peer_group_description}</p>
        <p><em>Report generated on {data.report_date} using GreenLang EUDR Professional Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        report_dict = {
            "report_type": "supplier_benchmark",
            "operator_name": data.operator_name,
            "report_date": data.report_date,
            "supplier": {
                "id": data.supplier_id,
                "name": data.supplier_name,
                "industry_segment": data.industry_segment,
            },
            "scores": data.supplier_score.dict(),
            "peer_rankings": [ranking.dict() for ranking in data.peer_rankings],
            "percentile_scores": [perc.dict() for perc in data.percentile_scores],
            "trends": {
                dimension: [point.dict() for point in points]
                for dimension, points in data.trends.items()
            },
            "best_practices": [practice.dict() for practice in data.best_practices],
            "improvement_recommendations": data.improvement_recommendations,
            "peer_group_description": data.peer_group_description,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "config": self.config.dict(),
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_score_status(self, score: float) -> str:
        """Get status label for score."""
        if score >= 80:
            return "Excellent ✓"
        elif score >= 60:
            return "Good"
        else:
            return "Needs Improvement"

    def _get_score_status_html(self, score: float) -> str:
        """Get HTML status label for score."""
        if score >= 80:
            return '<span class="excellent">Excellent</span>'
        elif score >= 60:
            return '<span class="good">Good</span>'
        else:
            return '<span class="needs-improvement">Needs Improvement</span>'

    def _get_percentile_description(self, percentile: float) -> str:
        """Get descriptive text for percentile."""
        if percentile >= 90:
            return "top 10%"
        elif percentile >= 75:
            return "top quartile"
        elif percentile >= 50:
            return "upper half"
        elif percentile >= 25:
            return "lower half"
        else:
            return "bottom quartile"

    def _create_percentile_chart(self, percentiles: List[PercentileScore]) -> str:
        """Create text-based percentile chart."""
        chart_lines = []
        chart_lines.append("Percentile Rankings by Dimension")
        chart_lines.append("")

        sorted_percs = sorted(percentiles, key=lambda x: x.percentile, reverse=True)
        scale = 50 / 100  # Scale to 50 characters wide

        for perc in sorted_percs:
            bar_length = int(perc.percentile * scale)
            bar = "█" * bar_length
            chart_lines.append(
                f"{perc.dimension[:25]:25} |{bar} {perc.percentile:.0f}th"
            )

        return "\n".join(chart_lines)
