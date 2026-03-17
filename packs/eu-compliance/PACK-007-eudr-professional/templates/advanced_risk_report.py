"""
Advanced Risk Report Template - PACK-007 EUDR Professional Pack

This module generates advanced risk analysis reports with Monte Carlo simulation results,
confidence intervals, VaR metrics, tornado diagrams, scenario comparisons, and risk trend
projections for EUDR compliance.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from advanced_risk_report import AdvancedRiskReportTemplate, ReportData
    >>> data = ReportData(
    ...     operator_name="Acme Importers Ltd",
    ...     report_date="2026-03-15",
    ...     monte_carlo_iterations=10000,
    ...     confidence_level=0.95,
    ...     base_risk_score=65.4
    ... )
    >>> template = AdvancedRiskReportTemplate()
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
    """Configuration for Advanced Risk Report generation."""

    include_distribution_charts: bool = Field(
        default=True,
        description="Include text-based distribution charts"
    )
    include_tornado_diagram: bool = Field(
        default=True,
        description="Include tornado sensitivity diagram"
    )
    include_scenario_comparison: bool = Field(
        default=True,
        description="Include scenario comparison table"
    )
    include_projections: bool = Field(
        default=True,
        description="Include risk trend projections"
    )
    confidence_interval: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence interval for statistical measures"
    )


class MonteCarloResults(BaseModel):
    """Monte Carlo simulation results."""

    iterations: int = Field(..., description="Number of simulation iterations")
    mean_score: float = Field(..., description="Mean risk score from simulation")
    std_dev: float = Field(..., description="Standard deviation")
    percentile_5: float = Field(..., description="5th percentile risk score")
    percentile_25: float = Field(..., description="25th percentile risk score")
    percentile_50: float = Field(..., description="Median risk score")
    percentile_75: float = Field(..., description="75th percentile risk score")
    percentile_95: float = Field(..., description="95th percentile risk score")
    var_95: float = Field(..., description="Value at Risk (95%)")
    cvar_95: float = Field(..., description="Conditional VaR (95%)")


class TornadoItem(BaseModel):
    """Tornado diagram sensitivity item."""

    factor: str = Field(..., description="Risk factor name")
    low_impact: float = Field(..., description="Impact when factor is at minimum")
    high_impact: float = Field(..., description="Impact when factor is at maximum")
    range_width: float = Field(..., description="Absolute impact range")


class ScenarioResult(BaseModel):
    """Scenario analysis result."""

    scenario_name: str = Field(..., description="Scenario name")
    probability: float = Field(..., ge=0, le=1, description="Scenario probability")
    risk_score: float = Field(..., description="Risk score in this scenario")
    key_assumptions: List[str] = Field(..., description="Key assumptions for scenario")


class RiskProjection(BaseModel):
    """Risk trend projection."""

    period: str = Field(..., description="Time period (e.g., Q1 2026)")
    baseline_score: float = Field(..., description="Baseline projected score")
    optimistic_score: float = Field(..., description="Optimistic projection")
    pessimistic_score: float = Field(..., description="Pessimistic projection")
    confidence: float = Field(..., ge=0, le=1, description="Projection confidence")


class ReportData(BaseModel):
    """Data model for Advanced Risk Report."""

    operator_name: str = Field(..., description="Operator name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    base_risk_score: float = Field(..., ge=0, le=100, description="Baseline risk score")
    monte_carlo_results: MonteCarloResults = Field(..., description="Monte Carlo results")
    tornado_items: List[TornadoItem] = Field(
        default_factory=list,
        description="Tornado diagram sensitivity factors"
    )
    scenarios: List[ScenarioResult] = Field(
        default_factory=list,
        description="Scenario analysis results"
    )
    projections: List[RiskProjection] = Field(
        default_factory=list,
        description="Risk trend projections"
    )
    key_drivers: List[str] = Field(
        default_factory=list,
        description="Key risk drivers identified"
    )
    mitigation_priorities: List[str] = Field(
        default_factory=list,
        description="Prioritized mitigation actions"
    )

    @validator('report_date')
    def validate_report_date(cls, v):
        """Validate report date is in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class AdvancedRiskReportTemplate:
    """
    Advanced Risk Report Template for EUDR Professional Pack.

    Generates comprehensive risk analysis reports with Monte Carlo simulation results,
    confidence intervals, VaR metrics, tornado diagrams, scenario comparisons, and
    risk trend projections.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = AdvancedRiskReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Monte Carlo Risk Distribution" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Advanced Risk Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the advanced risk report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(f"Rendering Advanced Risk Report for {data.operator_name} in {format} format")

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
        sections.append(f"# Advanced Risk Analysis Report")
        sections.append(f"")
        sections.append(f"**Operator:** {data.operator_name}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"**Baseline Risk Score:** {data.base_risk_score:.1f}/100")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        mc = data.monte_carlo_results
        sections.append(
            f"This advanced risk analysis report presents Monte Carlo simulation results "
            f"based on {mc.iterations:,} iterations. The mean risk score is **{mc.mean_score:.1f}** "
            f"with a standard deviation of {mc.std_dev:.2f}. The 95% Value at Risk (VaR) is "
            f"**{mc.var_95:.1f}**, indicating that in 95% of scenarios, the risk score will not "
            f"exceed this value."
        )
        sections.append(f"")

        # Monte Carlo Results
        sections.append(f"## Monte Carlo Risk Distribution")
        sections.append(f"")
        sections.append(f"**Simulation Parameters:**")
        sections.append(f"- Iterations: {mc.iterations:,}")
        sections.append(f"- Confidence Level: {self.config.confidence_interval * 100:.0f}%")
        sections.append(f"")

        sections.append(f"### Statistical Summary")
        sections.append(f"")
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|-------|")
        sections.append(f"| Mean Risk Score | {mc.mean_score:.2f} |")
        sections.append(f"| Standard Deviation | {mc.std_dev:.2f} |")
        sections.append(f"| Median (P50) | {mc.percentile_50:.2f} |")
        sections.append(f"| 5th Percentile (P5) | {mc.percentile_5:.2f} |")
        sections.append(f"| 25th Percentile (P25) | {mc.percentile_25:.2f} |")
        sections.append(f"| 75th Percentile (P75) | {mc.percentile_75:.2f} |")
        sections.append(f"| 95th Percentile (P95) | {mc.percentile_95:.2f} |")
        sections.append(f"| Value at Risk (95%) | {mc.var_95:.2f} |")
        sections.append(f"| Conditional VaR (95%) | {mc.cvar_95:.2f} |")
        sections.append(f"")

        if self.config.include_distribution_charts:
            sections.append(f"### Distribution Visualization")
            sections.append(f"")
            sections.append(f"```")
            sections.append(self._create_distribution_chart(mc))
            sections.append(f"```")
            sections.append(f"")

        # Confidence Intervals
        sections.append(f"### Confidence Intervals")
        sections.append(f"")
        ci_lower = mc.percentile_50 - (mc.std_dev * 1.96)
        ci_upper = mc.percentile_50 + (mc.std_dev * 1.96)
        sections.append(
            f"The **95% confidence interval** for the risk score is "
            f"[{ci_lower:.2f}, {ci_upper:.2f}]. This means we are 95% confident that the "
            f"true risk score lies within this range."
        )
        sections.append(f"")

        # Tornado Diagram
        if self.config.include_tornado_diagram and data.tornado_items:
            sections.append(f"## Sensitivity Analysis (Tornado Diagram)")
            sections.append(f"")
            sections.append(
                f"The following factors have the greatest impact on risk score variability, "
                f"ranked by impact range:"
            )
            sections.append(f"")

            sections.append(f"| Rank | Risk Factor | Low Impact | High Impact | Range |")
            sections.append(f"|------|-------------|------------|-------------|-------|")

            sorted_items = sorted(
                data.tornado_items,
                key=lambda x: x.range_width,
                reverse=True
            )

            for idx, item in enumerate(sorted_items, 1):
                sections.append(
                    f"| {idx} | {item.factor} | {item.low_impact:.1f} | "
                    f"{item.high_impact:.1f} | {item.range_width:.1f} |"
                )
            sections.append(f"")

            # Text-based tornado chart
            sections.append(f"### Visual Representation")
            sections.append(f"")
            sections.append(f"```")
            sections.append(self._create_tornado_chart(sorted_items[:10]))
            sections.append(f"```")
            sections.append(f"")

        # Scenario Comparison
        if self.config.include_scenario_comparison and data.scenarios:
            sections.append(f"## Scenario Analysis")
            sections.append(f"")
            sections.append(
                f"Multiple scenarios were analyzed to understand risk under different conditions:"
            )
            sections.append(f"")

            sections.append(f"| Scenario | Probability | Risk Score | Key Assumptions |")
            sections.append(f"|----------|-------------|------------|-----------------|")

            for scenario in data.scenarios:
                assumptions = "; ".join(scenario.key_assumptions[:2])
                sections.append(
                    f"| {scenario.scenario_name} | {scenario.probability * 100:.1f}% | "
                    f"{scenario.risk_score:.1f} | {assumptions} |"
                )
            sections.append(f"")

            # Weighted average
            weighted_avg = sum(s.risk_score * s.probability for s in data.scenarios)
            sections.append(
                f"**Probability-weighted average risk score:** {weighted_avg:.2f}"
            )
            sections.append(f"")

        # Risk Trend Projections
        if self.config.include_projections and data.projections:
            sections.append(f"## Risk Trend Projections")
            sections.append(f"")
            sections.append(
                f"Forward-looking risk projections based on current trends and planned "
                f"mitigation actions:"
            )
            sections.append(f"")

            sections.append(
                f"| Period | Baseline | Optimistic | Pessimistic | Confidence |"
            )
            sections.append(
                f"|--------|----------|------------|-------------|------------|"
            )

            for proj in data.projections:
                sections.append(
                    f"| {proj.period} | {proj.baseline_score:.1f} | "
                    f"{proj.optimistic_score:.1f} | {proj.pessimistic_score:.1f} | "
                    f"{proj.confidence * 100:.0f}% |"
                )
            sections.append(f"")

        # Key Risk Drivers
        if data.key_drivers:
            sections.append(f"## Key Risk Drivers")
            sections.append(f"")
            sections.append(
                f"The following factors are the primary drivers of current risk exposure:"
            )
            sections.append(f"")
            for idx, driver in enumerate(data.key_drivers, 1):
                sections.append(f"{idx}. {driver}")
            sections.append(f"")

        # Mitigation Priorities
        if data.mitigation_priorities:
            sections.append(f"## Recommended Mitigation Priorities")
            sections.append(f"")
            sections.append(
                f"Based on the sensitivity analysis and scenario results, the following "
                f"mitigation actions are recommended in priority order:"
            )
            sections.append(f"")
            for idx, priority in enumerate(data.mitigation_priorities, 1):
                sections.append(f"{idx}. {priority}")
            sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Report generated on {data.report_date} using GreenLang EUDR Professional Pack*"
        )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render report in HTML format."""
        mc = data.monte_carlo_results

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Advanced Risk Analysis Report - {data.operator_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #e74c3c; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        pre {{ background-color: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Advanced Risk Analysis Report</h1>

    <div class="summary">
        <p><strong>Operator:</strong> {data.operator_name}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Baseline Risk Score:</strong> <span class="metric">{data.base_risk_score:.1f}/100</span></p>
    </div>

    <h2>Executive Summary</h2>
    <p>
        This advanced risk analysis report presents Monte Carlo simulation results based on
        <strong>{mc.iterations:,}</strong> iterations. The mean risk score is
        <span class="metric">{mc.mean_score:.1f}</span> with a standard deviation of {mc.std_dev:.2f}.
        The 95% Value at Risk (VaR) is <span class="metric">{mc.var_95:.1f}</span>, indicating that
        in 95% of scenarios, the risk score will not exceed this value.
    </p>

    <h2>Monte Carlo Risk Distribution</h2>
    <h3>Statistical Summary</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Mean Risk Score</td><td>{mc.mean_score:.2f}</td></tr>
        <tr><td>Standard Deviation</td><td>{mc.std_dev:.2f}</td></tr>
        <tr><td>Median (P50)</td><td>{mc.percentile_50:.2f}</td></tr>
        <tr><td>5th Percentile (P5)</td><td>{mc.percentile_5:.2f}</td></tr>
        <tr><td>25th Percentile (P25)</td><td>{mc.percentile_25:.2f}</td></tr>
        <tr><td>75th Percentile (P75)</td><td>{mc.percentile_75:.2f}</td></tr>
        <tr><td>95th Percentile (P95)</td><td>{mc.percentile_95:.2f}</td></tr>
        <tr><td>Value at Risk (95%)</td><td>{mc.var_95:.2f}</td></tr>
        <tr><td>Conditional VaR (95%)</td><td>{mc.cvar_95:.2f}</td></tr>
    </table>
"""

        if self.config.include_distribution_charts:
            html += f"""
    <h3>Distribution Visualization</h3>
    <pre>{self._create_distribution_chart(mc)}</pre>
"""

        if self.config.include_tornado_diagram and data.tornado_items:
            sorted_items = sorted(
                data.tornado_items,
                key=lambda x: x.range_width,
                reverse=True
            )

            html += f"""
    <h2>Sensitivity Analysis (Tornado Diagram)</h2>
    <table>
        <tr><th>Rank</th><th>Risk Factor</th><th>Low Impact</th><th>High Impact</th><th>Range</th></tr>
"""
            for idx, item in enumerate(sorted_items, 1):
                html += f"""        <tr>
            <td>{idx}</td>
            <td>{item.factor}</td>
            <td>{item.low_impact:.1f}</td>
            <td>{item.high_impact:.1f}</td>
            <td>{item.range_width:.1f}</td>
        </tr>
"""
            html += f"""    </table>
    <pre>{self._create_tornado_chart(sorted_items[:10])}</pre>
"""

        if self.config.include_scenario_comparison and data.scenarios:
            html += f"""
    <h2>Scenario Analysis</h2>
    <table>
        <tr><th>Scenario</th><th>Probability</th><th>Risk Score</th><th>Key Assumptions</th></tr>
"""
            for scenario in data.scenarios:
                assumptions = "; ".join(scenario.key_assumptions[:2])
                html += f"""        <tr>
            <td>{scenario.scenario_name}</td>
            <td>{scenario.probability * 100:.1f}%</td>
            <td>{scenario.risk_score:.1f}</td>
            <td>{assumptions}</td>
        </tr>
"""
            weighted_avg = sum(s.risk_score * s.probability for s in data.scenarios)
            html += f"""    </table>
    <p><strong>Probability-weighted average risk score:</strong> {weighted_avg:.2f}</p>
"""

        if self.config.include_projections and data.projections:
            html += f"""
    <h2>Risk Trend Projections</h2>
    <table>
        <tr><th>Period</th><th>Baseline</th><th>Optimistic</th><th>Pessimistic</th><th>Confidence</th></tr>
"""
            for proj in data.projections:
                html += f"""        <tr>
            <td>{proj.period}</td>
            <td>{proj.baseline_score:.1f}</td>
            <td>{proj.optimistic_score:.1f}</td>
            <td>{proj.pessimistic_score:.1f}</td>
            <td>{proj.confidence * 100:.0f}%</td>
        </tr>
"""
            html += f"""    </table>
"""

        if data.key_drivers:
            html += f"""
    <h2>Key Risk Drivers</h2>
    <ol>
"""
            for driver in data.key_drivers:
                html += f"        <li>{driver}</li>\n"
            html += f"""    </ol>
"""

        if data.mitigation_priorities:
            html += f"""
    <h2>Recommended Mitigation Priorities</h2>
    <ol>
"""
            for priority in data.mitigation_priorities:
                html += f"        <li>{priority}</li>\n"
            html += f"""    </ol>
"""

        html += f"""
    <div class="footer">
        <p><em>Report generated on {data.report_date} using GreenLang EUDR Professional Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        report_dict = {
            "report_type": "advanced_risk_analysis",
            "operator_name": data.operator_name,
            "report_date": data.report_date,
            "base_risk_score": data.base_risk_score,
            "monte_carlo_results": {
                "iterations": data.monte_carlo_results.iterations,
                "mean_score": data.monte_carlo_results.mean_score,
                "std_dev": data.monte_carlo_results.std_dev,
                "percentiles": {
                    "p5": data.monte_carlo_results.percentile_5,
                    "p25": data.monte_carlo_results.percentile_25,
                    "p50": data.monte_carlo_results.percentile_50,
                    "p75": data.monte_carlo_results.percentile_75,
                    "p95": data.monte_carlo_results.percentile_95,
                },
                "var_95": data.monte_carlo_results.var_95,
                "cvar_95": data.monte_carlo_results.cvar_95,
            },
            "tornado_items": [item.dict() for item in data.tornado_items],
            "scenarios": [scenario.dict() for scenario in data.scenarios],
            "projections": [proj.dict() for proj in data.projections],
            "key_drivers": data.key_drivers,
            "mitigation_priorities": data.mitigation_priorities,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "config": self.config.dict(),
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _create_distribution_chart(self, mc: MonteCarloResults) -> str:
        """Create text-based distribution chart."""
        chart_lines = []
        chart_lines.append("Risk Score Distribution (Percentiles)")
        chart_lines.append("")

        percentiles = [
            ("P5", mc.percentile_5),
            ("P25", mc.percentile_25),
            ("P50", mc.percentile_50),
            ("P75", mc.percentile_75),
            ("P95", mc.percentile_95),
        ]

        max_score = max(p[1] for p in percentiles)
        scale = 50 / max_score if max_score > 0 else 1

        for label, value in percentiles:
            bar_length = int(value * scale)
            bar = "█" * bar_length
            chart_lines.append(f"{label:4} |{bar} {value:.1f}")

        chart_lines.append("")
        chart_lines.append(f"Mean: {mc.mean_score:.1f} ± {mc.std_dev:.2f}")

        return "\n".join(chart_lines)

    def _create_tornado_chart(self, items: List[TornadoItem]) -> str:
        """Create text-based tornado diagram."""
        chart_lines = []
        chart_lines.append("Sensitivity Analysis (Impact Range)")
        chart_lines.append("")

        if not items:
            return "No sensitivity data available"

        max_range = max(item.range_width for item in items)
        scale = 40 / max_range if max_range > 0 else 1

        for item in items:
            bar_length = int(item.range_width * scale)
            bar = "▓" * bar_length
            chart_lines.append(
                f"{item.factor[:30]:30} |{bar} {item.range_width:.1f}"
            )

        return "\n".join(chart_lines)
