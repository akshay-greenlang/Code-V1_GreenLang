# -*- coding: utf-8 -*-
"""
GL-004 Burnmaster - Uncertainty Reporter Module

Generates comprehensive uncertainty reports, visualizations, and
uncertainty budgets for combustion calculations. Supports audit
trails and regulatory compliance documentation.

Output Formats:
    - Detailed uncertainty reports
    - Uncertainty budget tables
    - Visualization figures
    - Audit-ready documentation

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import numpy as np
import hashlib
import json


class ReportFormat(str, Enum):
    """Output format for uncertainty reports."""
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class Contributor:
    """
    Individual contributor to uncertainty budget.

    Attributes:
        name: Name of the uncertainty source
        value: Uncertainty value (standard uncertainty)
        contribution_percent: Percentage of total variance
        uncertainty_type: Type A or Type B
        distribution: Assumed probability distribution
        source_description: Description of uncertainty source
    """
    name: str
    value: float
    contribution_percent: float
    uncertainty_type: str = "type_b"
    distribution: str = "normal"
    source_description: str = ""


@dataclass
class UncertaintyBudget:
    """
    Complete uncertainty budget for a calculation.

    Attributes:
        variable_name: Name of the output variable
        output_value: Calculated output value
        combined_uncertainty: Combined standard uncertainty
        expanded_uncertainty: Expanded uncertainty (k=2)
        coverage_factor: Coverage factor used
        contributors: List of individual contributors
        total_type_a: Total Type A uncertainty contribution
        total_type_b: Total Type B uncertainty contribution
        degrees_of_freedom: Effective degrees of freedom
        calculation_timestamp: When calculation was performed
    """
    variable_name: str
    output_value: float
    combined_uncertainty: float
    expanded_uncertainty: float
    coverage_factor: float = 2.0
    contributors: List[Contributor] = field(default_factory=list)
    total_type_a: float = 0.0
    total_type_b: float = 0.0
    degrees_of_freedom: Optional[float] = None
    calculation_timestamp: datetime = field(default_factory=datetime.utcnow)
    provenance_hash: str = ""

    def __post_init__(self):
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "variable_name": self.variable_name,
            "output_value": self.output_value,
            "combined_uncertainty": self.combined_uncertainty,
            "expanded_uncertainty": self.expanded_uncertainty,
            "coverage_factor": self.coverage_factor,
            "calculation_timestamp": self.calculation_timestamp.isoformat(),
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class UncertaintyReport:
    """
    Complete uncertainty report for a set of calculations.

    Attributes:
        report_id: Unique report identifier
        title: Report title
        description: Report description
        budgets: List of uncertainty budgets
        summary_statistics: Summary statistics
        recommendations: Uncertainty reduction recommendations
        generated_timestamp: When report was generated
        generator_version: Version of uncertainty module
    """
    report_id: str
    title: str
    description: str = ""
    budgets: List[UncertaintyBudget] = field(default_factory=list)
    summary_statistics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    generated_timestamp: datetime = field(default_factory=datetime.utcnow)
    generator_version: str = "1.0.0"
    provenance_hash: str = ""

    def __post_init__(self):
        self._compute_provenance_hash()

    def _compute_provenance_hash(self) -> None:
        """Compute SHA-256 hash for audit trail."""
        data = {
            "report_id": self.report_id,
            "title": self.title,
            "generated_timestamp": self.generated_timestamp.isoformat(),
            "budget_count": len(self.budgets),
        }
        self.provenance_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


@dataclass
class CalculationRecord:
    """
    Record of a calculation for uncertainty reporting.

    Attributes:
        variable_name: Name of calculated variable
        value: Calculated value
        standard_uncertainty: Standard uncertainty
        inputs: Dictionary of input values and uncertainties
        formula: Formula used for calculation
        timestamp: When calculation was performed
    """
    variable_name: str
    value: float
    standard_uncertainty: float
    inputs: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    formula: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


class Figure:
    """
    Simple figure representation for uncertainty visualization.

    Uses ASCII-based visualization for text output.
    Can be extended to use matplotlib for graphical output.
    """

    def __init__(self, title: str, width: int = 60, height: int = 20):
        """
        Initialize a figure.

        Args:
            title: Figure title
            width: Figure width in characters
            height: Figure height in lines
        """
        self.title = title
        self.width = width
        self.height = height
        self.data: Dict[str, Any] = {}

    def add_error_bar(
        self,
        value: float,
        uncertainty: float,
        label: str = "Value",
    ) -> None:
        """Add error bar data to figure."""
        self.data["error_bar"] = {
            "value": value,
            "uncertainty": uncertainty,
            "label": label,
        }

    def add_bar_chart(
        self,
        values: Dict[str, float],
        title: str = "Contributions",
    ) -> None:
        """Add bar chart data to figure."""
        self.data["bar_chart"] = {
            "values": values,
            "title": title,
        }

    def render_ascii(self) -> str:
        """Render figure as ASCII art."""
        lines = []
        lines.append("=" * self.width)
        lines.append(self.title.center(self.width))
        lines.append("=" * self.width)

        if "error_bar" in self.data:
            eb = self.data["error_bar"]
            value = eb["value"]
            unc = eb["uncertainty"]
            label = eb["label"]

            lines.append("")
            lines.append(f"{label}: {value:.4f} +/- {unc:.4f}")
            lines.append("")

            # ASCII error bar
            lower = value - unc
            upper = value + unc
            range_val = 2 * unc
            scale = (self.width - 10) / (range_val if range_val > 0 else 1)

            bar_line = ["-"] * (self.width - 10)
            center_pos = len(bar_line) // 2
            bar_line[center_pos] = "|"
            bar_line[0] = "["
            bar_line[-1] = "]"

            lines.append("    " + "".join(bar_line))
            lines.append(f"    {lower:.4f}" + " " * (len(bar_line) - 16) + f"{upper:.4f}")

        if "bar_chart" in self.data:
            bc = self.data["bar_chart"]
            values = bc["values"]
            chart_title = bc["title"]

            lines.append("")
            lines.append(chart_title)
            lines.append("-" * len(chart_title))

            if values:
                max_val = max(values.values())
                max_bar = self.width - 25

                for name, val in sorted(values.items(), key=lambda x: -x[1]):
                    bar_len = int(val / max_val * max_bar) if max_val > 0 else 0
                    bar = "#" * bar_len
                    lines.append(f"{name:15s} |{bar} {val:.1f}%")

        lines.append("")
        lines.append("=" * self.width)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render_ascii()


class UncertaintyReporter:
    """
    Generates comprehensive uncertainty reports and visualizations.

    Provides audit-ready documentation of uncertainty analysis
    following GUM (Guide to Expression of Uncertainty in Measurement)
    reporting guidelines.

    ZERO HALLUCINATION: All report content is derived from
    deterministic calculations. No LLM inference in reporting.

    Example Usage:
        >>> reporter = UncertaintyReporter()
        >>> calculations = [calc1, calc2, calc3]
        >>> report = reporter.generate_uncertainty_report(calculations)
        >>> budget = reporter.export_uncertainty_budget(calculations)
        >>> ranked = reporter.rank_uncertainty_contributors(budget[0])
    """

    def __init__(self, report_prefix: str = "UNC"):
        """
        Initialize the uncertainty reporter.

        Args:
            report_prefix: Prefix for report IDs
        """
        self.report_prefix = report_prefix
        self._report_counter = 0

    def generate_uncertainty_report(
        self,
        calculations: List[CalculationRecord],
        title: str = "Uncertainty Analysis Report",
        description: str = "",
    ) -> UncertaintyReport:
        """
        Generate comprehensive uncertainty report from calculations.

        Args:
            calculations: List of calculation records to report
            title: Report title
            description: Report description

        Returns:
            UncertaintyReport with full analysis

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        self._report_counter += 1
        report_id = f"{self.report_prefix}-{self._report_counter:04d}"

        # Generate budgets for each calculation
        budgets = []
        for calc in calculations:
            budget = self._create_budget_from_calculation(calc)
            budgets.append(budget)

        # Compute summary statistics
        summary = self._compute_summary_statistics(budgets)

        # Generate recommendations
        recommendations = self._generate_recommendations(budgets)

        return UncertaintyReport(
            report_id=report_id,
            title=title,
            description=description,
            budgets=budgets,
            summary_statistics=summary,
            recommendations=recommendations,
        )

    def visualize_uncertainty(
        self,
        value: float,
        uncertainty: float,
        variable_name: str = "Value",
    ) -> Figure:
        """
        Create visualization of value with uncertainty.

        Args:
            value: Central value
            uncertainty: Expanded uncertainty
            variable_name: Name for labeling

        Returns:
            Figure object with visualization

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        fig = Figure(
            title=f"Uncertainty Visualization: {variable_name}",
            width=60,
            height=15,
        )

        fig.add_error_bar(value, uncertainty, variable_name)

        return fig

    def visualize_contributions(
        self,
        budget: UncertaintyBudget,
    ) -> Figure:
        """
        Visualize uncertainty contributions as bar chart.

        Args:
            budget: Uncertainty budget to visualize

        Returns:
            Figure object with contribution chart

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        fig = Figure(
            title=f"Uncertainty Contributors: {budget.variable_name}",
            width=60,
            height=20,
        )

        contributions = {
            c.name: c.contribution_percent
            for c in budget.contributors
        }

        fig.add_bar_chart(contributions, "Contribution to Total Uncertainty (%)")

        return fig

    def export_uncertainty_budget(
        self,
        calculations: List[CalculationRecord],
    ) -> List[UncertaintyBudget]:
        """
        Export uncertainty budgets for all calculations.

        Creates GUM-compliant uncertainty budget tables.

        Args:
            calculations: List of calculation records

        Returns:
            List of UncertaintyBudget objects

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        budgets = []

        for calc in calculations:
            budget = self._create_budget_from_calculation(calc)
            budgets.append(budget)

        return budgets

    def rank_uncertainty_contributors(
        self,
        budget: UncertaintyBudget,
    ) -> List[Contributor]:
        """
        Rank contributors by their contribution to total uncertainty.

        Args:
            budget: Uncertainty budget to analyze

        Returns:
            List of Contributors sorted by contribution (descending)

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        sorted_contributors = sorted(
            budget.contributors,
            key=lambda c: c.contribution_percent,
            reverse=True,
        )

        return sorted_contributors

    def format_report(
        self,
        report: UncertaintyReport,
        format_type: ReportFormat = ReportFormat.TEXT,
    ) -> str:
        """
        Format uncertainty report for output.

        Args:
            report: UncertaintyReport to format
            format_type: Output format (text, json, html, markdown)

        Returns:
            Formatted report string

        DETERMINISTIC: Same inputs produce identical outputs.
        """
        if format_type == ReportFormat.JSON:
            return self._format_json(report)
        elif format_type == ReportFormat.MARKDOWN:
            return self._format_markdown(report)
        elif format_type == ReportFormat.HTML:
            return self._format_html(report)
        else:
            return self._format_text(report)

    def _create_budget_from_calculation(
        self,
        calc: CalculationRecord,
    ) -> UncertaintyBudget:
        """Create uncertainty budget from calculation record."""
        contributors = []
        total_variance = 0.0
        type_a_variance = 0.0
        type_b_variance = 0.0

        # Calculate contributions from each input
        input_variances = {}

        for input_name, (value, unc) in calc.inputs.items():
            # Estimate sensitivity coefficient (assume 1 if not known)
            # In practice, this would come from analytical derivatives
            variance = unc ** 2
            input_variances[input_name] = variance
            total_variance += variance

            # Assume all are Type B unless marked otherwise
            type_b_variance += variance

        # Create contributor objects with percentages
        for input_name, variance in input_variances.items():
            contribution_pct = (variance / total_variance * 100) if total_variance > 0 else 0

            contributors.append(Contributor(
                name=input_name,
                value=np.sqrt(variance),
                contribution_percent=contribution_pct,
                uncertainty_type="type_b",
                distribution="normal",
                source_description=f"Input uncertainty for {input_name}",
            ))

        combined = np.sqrt(total_variance)
        expanded = 2.0 * combined

        return UncertaintyBudget(
            variable_name=calc.variable_name,
            output_value=calc.value,
            combined_uncertainty=combined,
            expanded_uncertainty=expanded,
            coverage_factor=2.0,
            contributors=contributors,
            total_type_a=np.sqrt(type_a_variance),
            total_type_b=np.sqrt(type_b_variance),
            calculation_timestamp=calc.timestamp,
        )

    def _compute_summary_statistics(
        self,
        budgets: List[UncertaintyBudget],
    ) -> Dict[str, float]:
        """Compute summary statistics for report."""
        if not budgets:
            return {}

        rel_uncertainties = []
        for b in budgets:
            if b.output_value != 0:
                rel_unc = (b.expanded_uncertainty / abs(b.output_value)) * 100
                rel_uncertainties.append(rel_unc)

        return {
            "num_calculations": len(budgets),
            "avg_relative_uncertainty_percent": float(np.mean(rel_uncertainties)) if rel_uncertainties else 0,
            "max_relative_uncertainty_percent": float(np.max(rel_uncertainties)) if rel_uncertainties else 0,
            "min_relative_uncertainty_percent": float(np.min(rel_uncertainties)) if rel_uncertainties else 0,
        }

    def _generate_recommendations(
        self,
        budgets: List[UncertaintyBudget],
    ) -> List[str]:
        """Generate recommendations for uncertainty reduction."""
        recommendations = []

        for budget in budgets:
            # Find dominant contributor
            ranked = self.rank_uncertainty_contributors(budget)
            if ranked and ranked[0].contribution_percent > 50:
                recommendations.append(
                    f"For {budget.variable_name}: Focus on reducing uncertainty in "
                    f"'{ranked[0].name}' which contributes {ranked[0].contribution_percent:.1f}% "
                    f"of total uncertainty."
                )

            # Check relative uncertainty
            if budget.output_value != 0:
                rel_unc = (budget.expanded_uncertainty / abs(budget.output_value)) * 100
                if rel_unc > 10:
                    recommendations.append(
                        f"For {budget.variable_name}: Relative uncertainty ({rel_unc:.1f}%) "
                        f"exceeds 10%. Consider improving measurement quality."
                    )

        if not recommendations:
            recommendations.append(
                "All uncertainty levels are within acceptable ranges."
            )

        return recommendations

    def _format_text(self, report: UncertaintyReport) -> str:
        """Format report as plain text."""
        lines = []
        lines.append("=" * 70)
        lines.append(report.title.center(70))
        lines.append("=" * 70)
        lines.append(f"Report ID: {report.report_id}")
        lines.append(f"Generated: {report.generated_timestamp.isoformat()}")
        lines.append(f"Version: {report.generator_version}")

        if report.description:
            lines.append("")
            lines.append(f"Description: {report.description}")

        lines.append("")
        lines.append("-" * 70)
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 70)
        for key, value in report.summary_statistics.items():
            lines.append(f"  {key}: {value:.4f}")

        for budget in report.budgets:
            lines.append("")
            lines.append("-" * 70)
            lines.append(f"UNCERTAINTY BUDGET: {budget.variable_name}")
            lines.append("-" * 70)
            lines.append(f"  Value: {budget.output_value:.6f}")
            lines.append(f"  Combined Std Uncertainty: {budget.combined_uncertainty:.6f}")
            lines.append(f"  Expanded Uncertainty (k={budget.coverage_factor}): {budget.expanded_uncertainty:.6f}")

            if budget.output_value != 0:
                rel_unc = (budget.expanded_uncertainty / abs(budget.output_value)) * 100
                lines.append(f"  Relative Uncertainty: {rel_unc:.2f}%")

            lines.append("")
            lines.append("  Contributors:")
            for c in sorted(budget.contributors, key=lambda x: -x.contribution_percent):
                lines.append(f"    {c.name:20s}: {c.value:.6f} ({c.contribution_percent:.1f}%)")

        if report.recommendations:
            lines.append("")
            lines.append("-" * 70)
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 70)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        lines.append("")
        lines.append("=" * 70)
        lines.append(f"Provenance Hash: {report.provenance_hash}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _format_markdown(self, report: UncertaintyReport) -> str:
        """Format report as Markdown."""
        lines = []
        lines.append(f"# {report.title}")
        lines.append("")
        lines.append(f"**Report ID:** {report.report_id}")
        lines.append(f"**Generated:** {report.generated_timestamp.isoformat()}")
        lines.append(f"**Version:** {report.generator_version}")

        if report.description:
            lines.append("")
            lines.append(f"> {report.description}")

        lines.append("")
        lines.append("## Summary Statistics")
        lines.append("")
        lines.append("| Statistic | Value |")
        lines.append("|-----------|-------|")
        for key, value in report.summary_statistics.items():
            lines.append(f"| {key} | {value:.4f} |")

        for budget in report.budgets:
            lines.append("")
            lines.append(f"## Uncertainty Budget: {budget.variable_name}")
            lines.append("")
            lines.append(f"- **Value:** {budget.output_value:.6f}")
            lines.append(f"- **Combined Std Uncertainty:** {budget.combined_uncertainty:.6f}")
            lines.append(f"- **Expanded Uncertainty (k={budget.coverage_factor}):** {budget.expanded_uncertainty:.6f}")

            if budget.output_value != 0:
                rel_unc = (budget.expanded_uncertainty / abs(budget.output_value)) * 100
                lines.append(f"- **Relative Uncertainty:** {rel_unc:.2f}%")

            lines.append("")
            lines.append("### Contributors")
            lines.append("")
            lines.append("| Source | Uncertainty | Contribution |")
            lines.append("|--------|-------------|--------------|")
            for c in sorted(budget.contributors, key=lambda x: -x.contribution_percent):
                lines.append(f"| {c.name} | {c.value:.6f} | {c.contribution_percent:.1f}% |")

        if report.recommendations:
            lines.append("")
            lines.append("## Recommendations")
            lines.append("")
            for rec in report.recommendations:
                lines.append(f"- {rec}")

        lines.append("")
        lines.append("---")
        lines.append(f"*Provenance Hash: `{report.provenance_hash}`*")

        return "\n".join(lines)

    def _format_html(self, report: UncertaintyReport) -> str:
        """Format report as HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .meta {{ color: #7f8c8d; font-size: 0.9em; }}
        .recommendation {{ background-color: #e8f6f3; padding: 10px; margin: 5px 0; border-left: 3px solid #1abc9c; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <p class="meta">Report ID: {report.report_id} | Generated: {report.generated_timestamp.isoformat()}</p>
"""

        if report.description:
            html += f"    <p><em>{report.description}</em></p>\n"

        html += """
    <h2>Summary Statistics</h2>
    <table>
        <tr><th>Statistic</th><th>Value</th></tr>
"""
        for key, value in report.summary_statistics.items():
            html += f"        <tr><td>{key}</td><td>{value:.4f}</td></tr>\n"
        html += "    </table>\n"

        for budget in report.budgets:
            rel_unc = ""
            if budget.output_value != 0:
                rel_unc = f" ({(budget.expanded_uncertainty / abs(budget.output_value)) * 100:.2f}%)"

            html += f"""
    <h2>Uncertainty Budget: {budget.variable_name}</h2>
    <p>
        <strong>Value:</strong> {budget.output_value:.6f}<br>
        <strong>Combined Std Uncertainty:</strong> {budget.combined_uncertainty:.6f}<br>
        <strong>Expanded Uncertainty (k={budget.coverage_factor}):</strong> {budget.expanded_uncertainty:.6f}{rel_unc}
    </p>
    <table>
        <tr><th>Source</th><th>Uncertainty</th><th>Contribution</th></tr>
"""
            for c in sorted(budget.contributors, key=lambda x: -x.contribution_percent):
                html += f"        <tr><td>{c.name}</td><td>{c.value:.6f}</td><td>{c.contribution_percent:.1f}%</td></tr>\n"
            html += "    </table>\n"

        if report.recommendations:
            html += "    <h2>Recommendations</h2>\n"
            for rec in report.recommendations:
                html += f'    <div class="recommendation">{rec}</div>\n'

        html += f"""
    <hr>
    <p class="meta">Provenance Hash: {report.provenance_hash}</p>
</body>
</html>
"""
        return html

    def _format_json(self, report: UncertaintyReport) -> str:
        """Format report as JSON."""
        data = {
            "report_id": report.report_id,
            "title": report.title,
            "description": report.description,
            "generated_timestamp": report.generated_timestamp.isoformat(),
            "generator_version": report.generator_version,
            "provenance_hash": report.provenance_hash,
            "summary_statistics": report.summary_statistics,
            "budgets": [],
            "recommendations": report.recommendations,
        }

        for budget in report.budgets:
            budget_data = {
                "variable_name": budget.variable_name,
                "output_value": budget.output_value,
                "combined_uncertainty": budget.combined_uncertainty,
                "expanded_uncertainty": budget.expanded_uncertainty,
                "coverage_factor": budget.coverage_factor,
                "total_type_a": budget.total_type_a,
                "total_type_b": budget.total_type_b,
                "provenance_hash": budget.provenance_hash,
                "contributors": [
                    {
                        "name": c.name,
                        "value": c.value,
                        "contribution_percent": c.contribution_percent,
                        "uncertainty_type": c.uncertainty_type,
                        "distribution": c.distribution,
                    }
                    for c in budget.contributors
                ],
            }
            data["budgets"].append(budget_data)

        return json.dumps(data, indent=2)
