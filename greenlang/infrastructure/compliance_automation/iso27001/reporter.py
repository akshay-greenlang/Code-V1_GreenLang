# -*- coding: utf-8 -*-
"""
ISO 27001:2022 Compliance Reporter - SEC-010 Phase 5

Generates comprehensive ISO 27001:2022 compliance reports including:
- Statement of Applicability (SoA)
- Control Assessment Reports
- Gap Analysis Reports
- Executive Summaries
- Audit-ready Documentation

Classes:
    - ISO27001Reporter: Report generator for ISO 27001 compliance.
    - ReportFormat: Enum of supported report formats.

Example:
    >>> from greenlang.infrastructure.compliance_automation.iso27001 import (
    ...     ISO27001Reporter, ISO27001Mapper,
    ... )
    >>> mapper = ISO27001Mapper()
    >>> soa = await mapper.generate_soa()
    >>> reporter = ISO27001Reporter()
    >>> report = reporter.generate_soa_report(soa)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report Models
# ---------------------------------------------------------------------------


class ReportFormat(str, Enum):
    """Supported report output formats."""

    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


class ReportMetadata(BaseModel):
    """Metadata for a compliance report.

    Attributes:
        id: Unique report identifier.
        title: Report title.
        version: Report version.
        generated_at: When the report was generated.
        generated_by: Who/what generated the report.
        period_start: Assessment period start.
        period_end: Assessment period end.
        classification: Document classification.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(default="ISO 27001:2022 Compliance Report")
    version: str = Field(default="1.0")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    generated_by: str = Field(default="GreenLang Compliance Automation")
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    classification: str = Field(default="CONFIDENTIAL")


class ControlSummary(BaseModel):
    """Summary statistics for a control domain.

    Attributes:
        domain: The domain identifier (e.g., "A.5").
        domain_name: Human-readable domain name.
        total_controls: Total controls in domain.
        implemented: Number of implemented controls.
        partially_implemented: Number of partially implemented controls.
        not_implemented: Number of not implemented controls.
        not_applicable: Number of not applicable controls.
        compliance_percentage: Percentage of compliance.
    """

    domain: str
    domain_name: str
    total_controls: int = 0
    implemented: int = 0
    partially_implemented: int = 0
    not_implemented: int = 0
    not_applicable: int = 0
    compliance_percentage: float = 0.0


# ---------------------------------------------------------------------------
# ISO 27001 Reporter
# ---------------------------------------------------------------------------


class ISO27001Reporter:
    """Report generator for ISO 27001:2022 compliance.

    Generates various compliance reports including Statement of Applicability,
    control assessment reports, and gap analysis reports.

    Example:
        >>> reporter = ISO27001Reporter()
        >>> soa_report = reporter.generate_soa_report(soa_data)
        >>> gap_report = reporter.generate_gap_analysis(assessment_results)
    """

    # Domain names for ISO 27001:2022 Annex A
    DOMAIN_NAMES = {
        "A.5": "Organizational Controls",
        "A.6": "People Controls",
        "A.7": "Physical Controls",
        "A.8": "Technological Controls",
    }

    # Domain control counts
    DOMAIN_CONTROL_COUNTS = {
        "A.5": 37,
        "A.6": 8,
        "A.7": 14,
        "A.8": 34,
    }

    def __init__(self, organization_name: str = "GreenLang") -> None:
        """Initialize the reporter.

        Args:
            organization_name: Name of the organization for reports.
        """
        self.organization_name = organization_name
        logger.info("Initialized ISO27001Reporter for %s", organization_name)

    def generate_soa_report(
        self,
        soa_data: Dict[str, Any],
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate a Statement of Applicability report.

        Args:
            soa_data: SoA data from ISO27001Mapper.generate_soa().
            format: Output format (json, markdown, html).

        Returns:
            Formatted report string.
        """
        logger.info("Generating SoA report in %s format", format.value)

        if format == ReportFormat.JSON:
            return self._soa_to_json(soa_data)
        elif format == ReportFormat.MARKDOWN:
            return self._soa_to_markdown(soa_data)
        elif format == ReportFormat.HTML:
            return self._soa_to_html(soa_data)
        else:
            return self._soa_to_json(soa_data)

    def generate_gap_analysis(
        self,
        assessment_results: Dict[str, Any],
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate a gap analysis report.

        Args:
            assessment_results: Results from compliance assessment.
            format: Output format.

        Returns:
            Formatted gap analysis report.
        """
        logger.info("Generating gap analysis report in %s format", format.value)

        gaps = self._extract_gaps(assessment_results)

        if format == ReportFormat.JSON:
            return json.dumps(gaps, indent=2, default=str)
        elif format == ReportFormat.MARKDOWN:
            return self._gaps_to_markdown(gaps)
        elif format == ReportFormat.HTML:
            return self._gaps_to_html(gaps)
        else:
            return json.dumps(gaps, indent=2, default=str)

    def generate_executive_summary(
        self,
        soa_data: Dict[str, Any],
    ) -> str:
        """Generate an executive summary report.

        Args:
            soa_data: SoA data from assessment.

        Returns:
            Executive summary in markdown format.
        """
        logger.info("Generating executive summary")

        summary = soa_data.get("summary", {})
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        report = f"""# ISO 27001:2022 Executive Summary

**Organization:** {self.organization_name}
**Report Date:** {now}
**Framework:** ISO/IEC 27001:2022 Information Security Management System

---

## Overall Compliance Status

| Metric | Value |
|--------|-------|
| **Compliance Score** | {summary.get('compliance_percentage', 0):.1f}% |
| **Total Controls** | {summary.get('total_controls', 93)} |
| **Implemented** | {summary.get('implemented', 0)} |
| **Not Implemented** | {summary.get('not_implemented', 0)} |
| **Not Applicable** | {summary.get('not_applicable', 0)} |

---

## Domain Summary

"""
        # Add domain summaries
        domain_summaries = self._calculate_domain_summaries(soa_data)
        for domain_summary in domain_summaries:
            status_emoji = self._get_status_indicator(domain_summary.compliance_percentage)
            report += f"""### {domain_summary.domain} - {domain_summary.domain_name}

| Status | Count |
|--------|-------|
| Implemented | {domain_summary.implemented} |
| Partial | {domain_summary.partially_implemented} |
| Not Implemented | {domain_summary.not_implemented} |
| Not Applicable | {domain_summary.not_applicable} |
| **Compliance** | **{domain_summary.compliance_percentage:.1f}%** {status_emoji} |

"""

        # Add key findings
        report += self._generate_key_findings(soa_data)

        # Add recommendations
        report += self._generate_recommendations(soa_data)

        return report

    def generate_audit_package(
        self,
        soa_data: Dict[str, Any],
        evidence_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Generate a complete audit documentation package.

        Args:
            soa_data: Statement of Applicability data.
            evidence_data: Optional evidence collection results.

        Returns:
            Dictionary of report names to report content.
        """
        logger.info("Generating audit documentation package")

        package = {
            "01_executive_summary.md": self.generate_executive_summary(soa_data),
            "02_statement_of_applicability.md": self.generate_soa_report(
                soa_data, ReportFormat.MARKDOWN
            ),
            "03_statement_of_applicability.json": self.generate_soa_report(
                soa_data, ReportFormat.JSON
            ),
            "04_gap_analysis.md": self.generate_gap_analysis(
                soa_data, ReportFormat.MARKDOWN
            ),
            "05_control_details.md": self._generate_control_details(soa_data),
        }

        if evidence_data:
            package["06_evidence_index.md"] = self._generate_evidence_index(evidence_data)

        return package

    # -------------------------------------------------------------------------
    # Private Methods - Format Converters
    # -------------------------------------------------------------------------

    def _soa_to_json(self, soa_data: Dict[str, Any]) -> str:
        """Convert SoA to JSON format."""
        return json.dumps(soa_data, indent=2, default=str)

    def _soa_to_markdown(self, soa_data: Dict[str, Any]) -> str:
        """Convert SoA to Markdown format."""
        summary = soa_data.get("summary", {})
        controls = soa_data.get("controls", [])
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        report = f"""# ISO 27001:2022 Statement of Applicability

**Organization:** {self.organization_name}
**Framework Version:** {soa_data.get('version', '2022')}
**Generated:** {now}
**Document Classification:** CONFIDENTIAL

---

## Summary

| Metric | Value |
|--------|-------|
| Total Controls | {summary.get('total_controls', 93)} |
| Implemented | {summary.get('implemented', 0)} |
| Not Implemented | {summary.get('not_implemented', 0)} |
| Not Applicable | {summary.get('not_applicable', 0)} |
| **Compliance** | **{summary.get('compliance_percentage', 0):.1f}%** |

---

## Control Status by Domain

"""
        # Group controls by domain
        domains = {"A.5": [], "A.6": [], "A.7": [], "A.8": []}
        for control in controls:
            control_id = control.get("control_id", "")
            domain = control_id[:3] if len(control_id) >= 3 else "A.5"
            if domain in domains:
                domains[domain].append(control)

        for domain, domain_controls in domains.items():
            domain_name = self.DOMAIN_NAMES.get(domain, "Unknown")
            report += f"\n### {domain} - {domain_name}\n\n"
            report += "| Control | Name | Status | Score |\n"
            report += "|---------|------|--------|-------|\n"

            for control in sorted(domain_controls, key=lambda c: c.get("control_id", "")):
                control_id = control.get("control_id", "")
                control_name = control.get("control_name", "")[:40]
                status = control.get("status", "not_implemented")
                score = control.get("score", 0)
                status_display = self._format_status(status)
                report += f"| {control_id} | {control_name} | {status_display} | {score:.0f}% |\n"

        report += """
---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | """ + now[:10] + """ | Compliance Automation | Initial generation |

---

*This document was automatically generated by GreenLang Compliance Automation.*
"""
        return report

    def _soa_to_html(self, soa_data: Dict[str, Any]) -> str:
        """Convert SoA to HTML format."""
        summary = soa_data.get("summary", {})
        controls = soa_data.get("controls", [])
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISO 27001:2022 Statement of Applicability - {self.organization_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .status-verified {{ color: #27ae60; font-weight: bold; }}
        .status-implemented {{ color: #2ecc71; }}
        .status-partial {{ color: #f39c12; }}
        .status-not-implemented {{ color: #e74c3c; }}
        .summary-box {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 0.9em; color: #7f8c8d; }}
    </style>
</head>
<body>
    <h1>ISO 27001:2022 Statement of Applicability</h1>
    <p><strong>Organization:</strong> {self.organization_name}</p>
    <p><strong>Generated:</strong> {now}</p>

    <div class="summary-box">
        <div class="metric">
            <div class="metric-value">{summary.get('compliance_percentage', 0):.1f}%</div>
            <div class="metric-label">Compliance Score</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('implemented', 0)}</div>
            <div class="metric-label">Implemented</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('not_implemented', 0)}</div>
            <div class="metric-label">Not Implemented</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('total_controls', 93)}</div>
            <div class="metric-label">Total Controls</div>
        </div>
    </div>

    <h2>Control Details</h2>
    <table>
        <thead>
            <tr>
                <th>Control ID</th>
                <th>Control Name</th>
                <th>Status</th>
                <th>Score</th>
                <th>Evidence</th>
            </tr>
        </thead>
        <tbody>
"""
        for control in sorted(controls, key=lambda c: c.get("control_id", "")):
            control_id = control.get("control_id", "")
            control_name = control.get("control_name", "")
            status = control.get("status", "not_implemented")
            score = control.get("score", 0)
            status_class = f"status-{status.replace('_', '-')}"
            evidence_count = len(control.get("evidence_sources", []))

            html += f"""            <tr>
                <td>{control_id}</td>
                <td>{control_name}</td>
                <td class="{status_class}">{self._format_status(status)}</td>
                <td>{score:.0f}%</td>
                <td>{evidence_count} sources</td>
            </tr>
"""

        html += """        </tbody>
    </table>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 0.9em;">
        <p>This document was automatically generated by GreenLang Compliance Automation.</p>
        <p>Classification: CONFIDENTIAL</p>
    </footer>
</body>
</html>
"""
        return html

    def _gaps_to_markdown(self, gaps: List[Dict[str, Any]]) -> str:
        """Convert gaps to Markdown format."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        report = f"""# ISO 27001:2022 Gap Analysis Report

**Organization:** {self.organization_name}
**Report Date:** {now}
**Total Gaps Identified:** {len(gaps)}

---

## Summary by Severity

"""
        # Count by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for gap in gaps:
            severity = gap.get("severity", "medium")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        report += "| Severity | Count |\n"
        report += "|----------|-------|\n"
        for severity, count in severity_counts.items():
            indicator = self._get_severity_indicator(severity)
            report += f"| {indicator} {severity.capitalize()} | {count} |\n"

        report += "\n---\n\n## Gap Details\n\n"

        # Group by severity
        for severity in ["critical", "high", "medium", "low"]:
            severity_gaps = [g for g in gaps if g.get("severity") == severity]
            if severity_gaps:
                indicator = self._get_severity_indicator(severity)
                report += f"\n### {indicator} {severity.capitalize()} Severity Gaps\n\n"

                for gap in severity_gaps:
                    control_id = gap.get("control_id", "Unknown")
                    title = gap.get("title", "Untitled")
                    description = gap.get("description", "No description")
                    remediation = gap.get("remediation_plan", "Not defined")

                    report += f"""#### {control_id}: {title}

**Description:** {description}

**Remediation Plan:** {remediation}

---

"""
        return report

    def _gaps_to_html(self, gaps: List[Dict[str, Any]]) -> str:
        """Convert gaps to HTML format."""
        # Simplified HTML for gaps
        return f"<html><body><pre>{json.dumps(gaps, indent=2, default=str)}</pre></body></html>"

    # -------------------------------------------------------------------------
    # Private Methods - Helpers
    # -------------------------------------------------------------------------

    def _calculate_domain_summaries(
        self,
        soa_data: Dict[str, Any],
    ) -> List[ControlSummary]:
        """Calculate summary statistics for each domain."""
        controls = soa_data.get("controls", [])
        summaries: Dict[str, ControlSummary] = {}

        # Initialize summaries
        for domain, name in self.DOMAIN_NAMES.items():
            summaries[domain] = ControlSummary(
                domain=domain,
                domain_name=name,
            )

        # Count controls
        for control in controls:
            control_id = control.get("control_id", "")
            domain = control_id[:3] if len(control_id) >= 3 else "A.5"

            if domain not in summaries:
                continue

            summary = summaries[domain]
            summary.total_controls += 1

            status = control.get("status", "not_implemented")
            if status in ("verified", "implemented"):
                summary.implemented += 1
            elif status == "partially_implemented":
                summary.partially_implemented += 1
            elif status == "not_applicable":
                summary.not_applicable += 1
            else:
                summary.not_implemented += 1

        # Calculate percentages
        for summary in summaries.values():
            applicable = summary.total_controls - summary.not_applicable
            if applicable > 0:
                summary.compliance_percentage = (
                    (summary.implemented + summary.partially_implemented * 0.5) / applicable
                ) * 100

        return list(summaries.values())

    def _extract_gaps(
        self,
        assessment_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Extract gaps from assessment results."""
        gaps: List[Dict[str, Any]] = []

        controls = assessment_results.get("controls", [])
        for control in controls:
            status = control.get("status", "not_implemented")
            if status in ("not_implemented", "partially_implemented", "non_compliant"):
                gaps.append({
                    "control_id": control.get("control_id", ""),
                    "control_name": control.get("control_name", ""),
                    "title": f"Gap in {control.get('control_name', 'Unknown Control')}",
                    "description": f"Control is {status.replace('_', ' ')}",
                    "severity": self._determine_gap_severity(control),
                    "status": "open",
                    "remediation_plan": self._suggest_remediation(control),
                })

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda g: severity_order.get(g.get("severity", "medium"), 2))

        return gaps

    def _determine_gap_severity(self, control: Dict[str, Any]) -> str:
        """Determine the severity of a gap based on control type."""
        control_id = control.get("control_id", "")
        status = control.get("status", "")

        # Critical controls
        critical_controls = ["A.5.15", "A.5.16", "A.5.17", "A.8.5", "A.8.24", "A.8.15"]
        if control_id in critical_controls and status == "not_implemented":
            return "critical"

        # High severity for not implemented
        if status == "not_implemented":
            return "high"

        # Medium for partial
        return "medium"

    def _suggest_remediation(self, control: Dict[str, Any]) -> str:
        """Suggest remediation steps for a gap."""
        control_id = control.get("control_id", "")
        status = control.get("status", "")

        if status == "not_implemented":
            return f"Implement control {control_id} according to ISO 27001:2022 requirements"
        elif status == "partially_implemented":
            return f"Complete implementation of control {control_id}"
        else:
            return "Review and verify control implementation"

    def _generate_key_findings(self, soa_data: Dict[str, Any]) -> str:
        """Generate key findings section."""
        summary = soa_data.get("summary", {})
        compliance = summary.get("compliance_percentage", 0)

        findings = "\n## Key Findings\n\n"

        if compliance >= 95:
            findings += "1. **Strong Compliance Posture**: Organization demonstrates excellent compliance with ISO 27001:2022 requirements.\n"
        elif compliance >= 80:
            findings += "1. **Good Compliance Progress**: Organization shows solid progress toward full compliance.\n"
        elif compliance >= 60:
            findings += "1. **Moderate Compliance**: Significant work remains to achieve full compliance.\n"
        else:
            findings += "1. **Compliance Concerns**: Urgent attention required to address compliance gaps.\n"

        not_implemented = summary.get("not_implemented", 0)
        if not_implemented > 0:
            findings += f"2. **Outstanding Controls**: {not_implemented} controls require implementation.\n"

        return findings

    def _generate_recommendations(self, soa_data: Dict[str, Any]) -> str:
        """Generate recommendations section."""
        summary = soa_data.get("summary", {})
        not_implemented = summary.get("not_implemented", 0)

        recommendations = "\n## Recommendations\n\n"

        if not_implemented > 10:
            recommendations += "1. **Priority Focus**: Develop a 90-day implementation plan for critical controls.\n"
        elif not_implemented > 0:
            recommendations += "1. **Complete Implementation**: Address remaining control gaps within 60 days.\n"
        else:
            recommendations += "1. **Maintain Compliance**: Continue monitoring and regular assessments.\n"

        recommendations += "2. **Evidence Collection**: Ensure automated evidence collection is enabled for all controls.\n"
        recommendations += "3. **Regular Review**: Conduct quarterly compliance reviews.\n"

        return recommendations

    def _generate_control_details(self, soa_data: Dict[str, Any]) -> str:
        """Generate detailed control documentation."""
        controls = soa_data.get("controls", [])
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        report = f"""# ISO 27001:2022 Control Details

**Organization:** {self.organization_name}
**Report Date:** {now}

---

"""
        for control in sorted(controls, key=lambda c: c.get("control_id", "")):
            control_id = control.get("control_id", "")
            control_name = control.get("control_name", "")
            status = control.get("status", "not_implemented")
            score = control.get("score", 0)
            evidence_sources = control.get("evidence_sources", [])
            technical_controls = control.get("technical_controls", [])

            report += f"""## {control_id}: {control_name}

| Attribute | Value |
|-----------|-------|
| Status | {self._format_status(status)} |
| Score | {score:.0f}% |
| Technical Controls | {', '.join(technical_controls) or 'N/A'} |
| Evidence Sources | {', '.join(evidence_sources) or 'N/A'} |

---

"""
        return report

    def _generate_evidence_index(self, evidence_data: Dict[str, Any]) -> str:
        """Generate an evidence index document."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        report = f"""# ISO 27001:2022 Evidence Index

**Organization:** {self.organization_name}
**Report Date:** {now}

---

## Evidence Summary

This document indexes all compliance evidence collected for ISO 27001:2022 controls.

"""
        for control_id, result in evidence_data.items():
            items = result.get("items", []) if isinstance(result, dict) else []
            report += f"### {control_id}\n\n"
            report += f"Evidence items collected: {len(items)}\n\n"

            if items:
                report += "| Source | Type | Collected |\n"
                report += "|--------|------|----------|\n"
                for item in items:
                    source = item.get("source", "Unknown")
                    etype = item.get("evidence_type", "Unknown")
                    collected = item.get("collected_at", "Unknown")[:10]
                    report += f"| {source} | {etype} | {collected} |\n"
                report += "\n"

        return report

    def _format_status(self, status: str) -> str:
        """Format status for display."""
        status_map = {
            "verified": "Verified",
            "implemented": "Implemented",
            "partially_implemented": "Partial",
            "not_implemented": "Not Implemented",
            "not_applicable": "N/A",
            "non_compliant": "Non-Compliant",
        }
        return status_map.get(status, status.replace("_", " ").title())

    def _get_status_indicator(self, percentage: float) -> str:
        """Get a text indicator for compliance percentage."""
        if percentage >= 95:
            return "[COMPLIANT]"
        elif percentage >= 70:
            return "[PARTIAL]"
        else:
            return "[ATTENTION NEEDED]"

    def _get_severity_indicator(self, severity: str) -> str:
        """Get a text indicator for severity."""
        indicators = {
            "critical": "[CRITICAL]",
            "high": "[HIGH]",
            "medium": "[MEDIUM]",
            "low": "[LOW]",
        }
        return indicators.get(severity, "[UNKNOWN]")


__all__ = [
    "ISO27001Reporter",
    "ReportFormat",
    "ReportMetadata",
    "ControlSummary",
]
