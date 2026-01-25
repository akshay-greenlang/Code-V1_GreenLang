"""
Certification Report Generator

This module provides report generation capabilities for certification results,
including PDF, HTML, JSON, and Markdown formats.

Example:
    >>> from backend.certification import CertificationEngine
    >>> from backend.certification.report import ReportGenerator
    >>> engine = CertificationEngine()
    >>> result = engine.evaluate_agent(Path("path/to/agent"))
    >>> generator = ReportGenerator()
    >>> generator.generate_pdf(result, Path("report.pdf"))
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .engine import CertificationResult, CertificationLevel
from .dimensions.base import DimensionResult, DimensionStatus

logger = logging.getLogger(__name__)


class CertificationReport:
    """
    Certification report data model.

    Wraps CertificationResult with additional reporting metadata.
    """

    def __init__(
        self,
        result: CertificationResult,
        title: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        """
        Initialize certification report.

        Args:
            result: Certification result to report
            title: Optional report title
            organization: Optional organization name
        """
        self.result = result
        self.title = title or f"Agent Certification Report: {result.agent_id}"
        self.organization = organization or "GreenLang"
        self.generated_at = datetime.utcnow()

    @property
    def summary(self) -> Dict[str, Any]:
        """Get report summary."""
        return {
            "agent_id": self.result.agent_id,
            "agent_version": self.result.agent_version,
            "certified": self.result.certified,
            "level": self.result.level.value,
            "overall_score": self.result.overall_score,
            "dimensions_passed": self.result.dimensions_passed,
            "dimensions_total": self.result.dimensions_total,
        }


class ReportGenerator:
    """
    Report generator for certification results.

    Generates reports in multiple formats:
    - PDF (requires reportlab)
    - HTML
    - JSON
    - Markdown

    Example:
        >>> generator = ReportGenerator()
        >>> generator.generate_markdown(result, Path("report.md"))
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize report generator.

        Args:
            config: Optional configuration
        """
        self.config = config or {}

    def generate_pdf(
        self,
        result: CertificationResult,
        output_path: Path,
        title: Optional[str] = None,
    ) -> Path:
        """
        Generate PDF report.

        Args:
            result: Certification result
            output_path: Output file path
            title: Optional report title

        Returns:
            Path to generated PDF
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
                PageBreak,
            )
        except ImportError:
            logger.error("reportlab not installed. Install with: pip install reportlab")
            # Fall back to text report
            return self._generate_text_report(result, output_path.with_suffix(".txt"))

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
        )
        story.append(Paragraph(
            title or f"Agent Certification Report",
            title_style,
        ))

        # Agent info
        story.append(Paragraph(f"<b>Agent ID:</b> {result.agent_id}", styles["Normal"]))
        story.append(Paragraph(f"<b>Version:</b> {result.agent_version}", styles["Normal"]))
        story.append(Paragraph(f"<b>Certification ID:</b> {result.certification_id}", styles["Normal"]))
        story.append(Paragraph(f"<b>Date:</b> {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}", styles["Normal"]))
        story.append(Spacer(1, 20))

        # Summary box
        cert_status = "CERTIFIED" if result.certified else "NOT CERTIFIED"
        cert_color = colors.green if result.certified else colors.red

        summary_data = [
            ["Certification Status", cert_status],
            ["Level", result.level.value],
            ["Overall Score", f"{result.overall_score:.1f}/100"],
            ["Weighted Score", f"{result.weighted_score:.1f}/100"],
            ["Dimensions Passed", f"{result.dimensions_passed}/{result.dimensions_total}"],
        ]

        summary_table = Table(summary_data, colWidths=[2.5 * inch, 2.5 * inch])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("TEXTCOLOR", (1, 0), (1, 0), cert_color),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 12),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 30))

        # Dimension results
        story.append(Paragraph("Dimension Results", styles["Heading2"]))
        story.append(Spacer(1, 10))

        dim_data = [["Dimension", "Status", "Score", "Checks"]]
        for dim in result.dimension_results:
            status = "PASS" if dim.passed else "FAIL"
            dim_data.append([
                f"{dim.dimension_id}: {dim.dimension_name}",
                status,
                f"{dim.score:.1f}",
                f"{dim.checks_passed}/{dim.checks_total}",
            ])

        dim_table = Table(dim_data, colWidths=[3 * inch, 1 * inch, 1 * inch, 1 * inch])
        dim_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))

        # Color code rows
        for i, dim in enumerate(result.dimension_results, 1):
            if dim.passed:
                dim_table.setStyle(TableStyle([
                    ("BACKGROUND", (1, i), (1, i), colors.lightgreen),
                ]))
            else:
                dim_table.setStyle(TableStyle([
                    ("BACKGROUND", (1, i), (1, i), colors.lightcoral),
                ]))

        story.append(dim_table)
        story.append(Spacer(1, 30))

        # Remediation (if needed)
        if result.remediation_summary:
            story.append(PageBreak())
            story.append(Paragraph("Remediation Required", styles["Heading2"]))
            story.append(Spacer(1, 10))

            for dimension, suggestions in result.remediation_summary.items():
                story.append(Paragraph(f"<b>{dimension}</b>", styles["Normal"]))
                for suggestion in suggestions[:3]:
                    # Truncate long suggestions
                    short_suggestion = suggestion[:200] + "..." if len(suggestion) > 200 else suggestion
                    story.append(Paragraph(f"  - {short_suggestion}", styles["Normal"]))
                story.append(Spacer(1, 10))

        # Build PDF
        doc.build(story)

        logger.info(f"PDF report generated: {output_path}")
        return output_path

    def generate_html(
        self,
        result: CertificationResult,
        output_path: Path,
        title: Optional[str] = None,
    ) -> Path:
        """
        Generate HTML report.

        Args:
            result: Certification result
            output_path: Output file path
            title: Optional report title

        Returns:
            Path to generated HTML
        """
        title = title or f"Agent Certification Report: {result.agent_id}"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2e7d32;
            border-bottom: 3px solid #2e7d32;
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card .label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .certified {{
            color: #2e7d32;
        }}
        .not-certified {{
            color: #c62828;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #2e7d32;
            color: white;
        }}
        .status-pass {{
            background: #c8e6c9;
            color: #1b5e20;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .status-fail {{
            background: #ffcdd2;
            color: #b71c1c;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .remediation {{
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .remediation h3 {{
            margin-top: 0;
            color: #e65100;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Agent Certification Report</h1>

        <div class="summary">
            <div class="summary-card">
                <div class="label">Agent ID</div>
                <div class="value">{result.agent_id}</div>
            </div>
            <div class="summary-card">
                <div class="label">Version</div>
                <div class="value">{result.agent_version}</div>
            </div>
            <div class="summary-card">
                <div class="label">Status</div>
                <div class="value {'certified' if result.certified else 'not-certified'}">
                    {'CERTIFIED' if result.certified else 'NOT CERTIFIED'}
                </div>
            </div>
            <div class="summary-card">
                <div class="label">Level</div>
                <div class="value">{result.level.value}</div>
            </div>
            <div class="summary-card">
                <div class="label">Overall Score</div>
                <div class="value">{result.overall_score:.1f}%</div>
            </div>
            <div class="summary-card">
                <div class="label">Dimensions</div>
                <div class="value">{result.dimensions_passed}/{result.dimensions_total}</div>
            </div>
        </div>

        <h2>Dimension Results</h2>
        <table>
            <tr>
                <th>ID</th>
                <th>Dimension</th>
                <th>Status</th>
                <th>Score</th>
                <th>Checks</th>
            </tr>
"""

        for dim in result.dimension_results:
            status_class = "status-pass" if dim.passed else "status-fail"
            status_text = "PASS" if dim.passed else "FAIL"
            html += f"""
            <tr>
                <td>{dim.dimension_id}</td>
                <td>{dim.dimension_name}</td>
                <td><span class="{status_class}">{status_text}</span></td>
                <td>{dim.score:.1f}</td>
                <td>{dim.checks_passed}/{dim.checks_total}</td>
            </tr>
"""

        html += """
        </table>
"""

        if result.remediation_summary:
            html += """
        <h2>Remediation Required</h2>
"""
            for dimension, suggestions in result.remediation_summary.items():
                html += f"""
        <div class="remediation">
            <h3>{dimension}</h3>
            <ul>
"""
                for suggestion in suggestions[:3]:
                    html += f"                <li>{suggestion[:300]}</li>\n"
                html += """
            </ul>
        </div>
"""

        html += f"""
        <div class="footer">
            <p>Certification ID: {result.certification_id}</p>
            <p>Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p>Execution Time: {result.execution_time_ms:.2f}ms</p>
        </div>
    </div>
</body>
</html>
"""

        output_path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report generated: {output_path}")
        return output_path

    def generate_json(
        self,
        result: CertificationResult,
        output_path: Path,
    ) -> Path:
        """
        Generate JSON report.

        Args:
            result: Certification result
            output_path: Output file path

        Returns:
            Path to generated JSON
        """
        output_path.write_text(
            json.dumps(result.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(f"JSON report generated: {output_path}")
        return output_path

    def generate_markdown(
        self,
        result: CertificationResult,
        output_path: Path,
        title: Optional[str] = None,
    ) -> Path:
        """
        Generate Markdown report.

        Args:
            result: Certification result
            output_path: Output file path
            title: Optional report title

        Returns:
            Path to generated Markdown
        """
        title = title or f"Agent Certification Report: {result.agent_id}"

        md = f"""# {title}

## Summary

| Property | Value |
|----------|-------|
| Agent ID | `{result.agent_id}` |
| Version | `{result.agent_version}` |
| Certification ID | `{result.certification_id}` |
| Status | **{'CERTIFIED' if result.certified else 'NOT CERTIFIED'}** |
| Level | **{result.level.value}** |
| Overall Score | {result.overall_score:.1f}/100 |
| Weighted Score | {result.weighted_score:.1f}/100 |
| Dimensions | {result.dimensions_passed}/{result.dimensions_total} passed |
| Execution Time | {result.execution_time_ms:.2f}ms |
| Date | {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} |

## Dimension Results

| ID | Dimension | Status | Score | Checks |
|----|-----------|--------|-------|--------|
"""

        for dim in result.dimension_results:
            status = "PASS" if dim.passed else "FAIL"
            md += f"| {dim.dimension_id} | {dim.dimension_name} | {status} | {dim.score:.1f} | {dim.checks_passed}/{dim.checks_total} |\n"

        if result.remediation_summary:
            md += "\n## Remediation Required\n\n"
            for dimension, suggestions in result.remediation_summary.items():
                md += f"### {dimension}\n\n"
                for suggestion in suggestions[:3]:
                    md += f"- {suggestion[:300]}\n"
                md += "\n"

        if result.required_failures:
            md += "\n## Required Dimensions Failed\n\n"
            for failure in result.required_failures:
                md += f"- {failure}\n"

        md += f"\n---\n\n*Generated by GreenLang Agent Certification Framework*\n"

        output_path.write_text(md, encoding="utf-8")
        logger.info(f"Markdown report generated: {output_path}")
        return output_path

    def _generate_text_report(
        self,
        result: CertificationResult,
        output_path: Path,
    ) -> Path:
        """
        Generate plain text report (fallback).

        Args:
            result: Certification result
            output_path: Output file path

        Returns:
            Path to generated text file
        """
        text = f"""
================================================================================
GREENLANG AGENT CERTIFICATION REPORT
================================================================================

Agent ID:          {result.agent_id}
Version:           {result.agent_version}
Certification ID:  {result.certification_id}

--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------
Status:            {'CERTIFIED' if result.certified else 'NOT CERTIFIED'}
Level:             {result.level.value}
Overall Score:     {result.overall_score:.1f}/100
Weighted Score:    {result.weighted_score:.1f}/100
Dimensions:        {result.dimensions_passed}/{result.dimensions_total} passed
Execution Time:    {result.execution_time_ms:.2f}ms

--------------------------------------------------------------------------------
DIMENSION RESULTS
--------------------------------------------------------------------------------
"""

        for dim in result.dimension_results:
            status = "PASS" if dim.passed else "FAIL"
            text += f"{dim.dimension_id} {dim.dimension_name:30} [{status}] {dim.score:.1f}/100\n"

        if result.remediation_summary:
            text += """
--------------------------------------------------------------------------------
REMEDIATION REQUIRED
--------------------------------------------------------------------------------
"""
            for dimension, suggestions in result.remediation_summary.items():
                text += f"\n{dimension}:\n"
                for suggestion in suggestions[:2]:
                    text += f"  - {suggestion[:100]}...\n"

        text += f"""
--------------------------------------------------------------------------------
Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
================================================================================
"""

        output_path.write_text(text, encoding="utf-8")
        logger.info(f"Text report generated: {output_path}")
        return output_path
