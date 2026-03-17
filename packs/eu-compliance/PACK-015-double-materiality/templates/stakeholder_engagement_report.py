# -*- coding: utf-8 -*-
"""
StakeholderEngagementReportTemplate - Stakeholder consultation report for PACK-015.

Sections:
    1. Engagement Overview
    2. Stakeholder Coverage Analysis
    3. Priority Matrix Summary
    4. Consultation Summary
    5. Synthesized Findings
    6. Validation Results

Author: GreenLang Team
Version: 15.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StakeholderEngagementReportTemplate:
    """
    Stakeholder engagement consultation report template.

    Renders stakeholder coverage analysis, influence-impact priority
    matrix summaries, consultation details, synthesized findings, and
    ESRS 1 sections 22-23 validation results.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize StakeholderEngagementReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render stakeholder engagement report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_overview(data),
            self._md_coverage(data),
            self._md_priority_matrix(data),
            self._md_consultations(data),
            self._md_findings(data),
            self._md_validation(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render stakeholder engagement report as HTML."""
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overview(data),
            self._html_coverage(data),
            self._html_findings(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Stakeholder Engagement Report</title>\n<style>\n{css}\n</style>\n'
            f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render stakeholder engagement report as JSON."""
        self.generated_at = datetime.utcnow()
        result = {
            "template": "stakeholder_engagement_report",
            "version": "15.0.0",
            "generated_at": self.generated_at.isoformat(),
            "stakeholders_identified": data.get("stakeholders_identified", 0),
            "consultations_recorded": data.get("consultations_recorded", 0),
            "total_participants": data.get("total_participants", 0),
            "synthesized_findings": data.get("synthesized_findings", []),
            "validation_checks": data.get("validation_checks", []),
            "validation_passed": data.get("validation_passed", False),
            "category_coverage": data.get("category_coverage", {}),
            "priority_distribution": data.get("priority_distribution", {}),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # -- Markdown sections --

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"# Stakeholder Engagement Report\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render engagement overview."""
        sh_count = data.get("stakeholders_identified", 0)
        con_count = data.get("consultations_recorded", 0)
        participants = data.get("total_participants", 0)
        validated = data.get("validation_passed", False)
        return (
            "## Engagement Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Stakeholders Identified | {sh_count} |\n"
            f"| Consultations Recorded | {con_count} |\n"
            f"| Total Participants | {participants} |\n"
            f"| ESRS 1 Validation | {'PASS' if validated else 'FAIL'} |"
        )

    def _md_coverage(self, data: Dict[str, Any]) -> str:
        """Render stakeholder category coverage."""
        coverage = data.get("category_coverage", {})
        if not coverage:
            return "## Stakeholder Coverage\n\n_No coverage data available._"
        lines = [
            "## Stakeholder Category Coverage", "",
            "| Category | Count |",
            "|----------|-------|",
        ]
        for cat, count in sorted(coverage.items()):
            label = cat.replace("_", " ").title()
            lines.append(f"| {label} | {count} |")
        return "\n".join(lines)

    def _md_priority_matrix(self, data: Dict[str, Any]) -> str:
        """Render priority matrix distribution."""
        dist = data.get("priority_distribution", {})
        if not dist:
            return "## Priority Matrix\n\n_No priority data available._"
        lines = [
            "## Influence-Impact Priority Matrix", "",
            "| Priority Level | Stakeholder Count |",
            "|---------------|------------------|",
        ]
        for level, count in sorted(dist.items()):
            label = level.replace("_", " ").title()
            lines.append(f"| {label} | {count} |")
        return "\n".join(lines)

    def _md_consultations(self, data: Dict[str, Any]) -> str:
        """Render consultation summary."""
        findings = data.get("synthesized_findings", [])
        if not findings:
            return "## Consultation Summary\n\n_No consultation data available._"
        return (
            "## Consultation Summary\n\n"
            f"A total of {data.get('consultations_recorded', 0)} consultations were "
            f"conducted with {data.get('total_participants', 0)} participants across "
            f"{len(data.get('category_coverage', {}))} stakeholder categories."
        )

    def _md_findings(self, data: Dict[str, Any]) -> str:
        """Render synthesized findings."""
        findings = data.get("synthesized_findings", [])
        if not findings:
            return "## Synthesized Findings\n\n_No findings available._"
        lines = [
            "## Synthesized Findings", "",
            "| Finding | Groups | Frequency | Priority |",
            "|---------|--------|-----------|----------|",
        ]
        for f in findings[:15]:
            lines.append(
                f"| {f.get('topic', '-')} | "
                f"{f.get('stakeholder_groups_count', 0)} | "
                f"{f.get('frequency', 0)} | "
                f"**{self._fmt(f.get('priority_score', 0))}** |"
            )
        return "\n".join(lines)

    def _md_validation(self, data: Dict[str, Any]) -> str:
        """Render ESRS 1 validation results."""
        checks = data.get("validation_checks", [])
        if not checks:
            return "## ESRS 1 Validation\n\n_No validation data available._"
        lines = [
            "## ESRS 1 Validation Results", "",
            "| Check | ESRS Reference | Status | Details |",
            "|-------|---------------|--------|---------|",
        ]
        for vc in checks:
            status = "PASS" if vc.get("passed", False) else "FAIL"
            lines.append(
                f"| {vc.get('check_name', '-')} | "
                f"{vc.get('esrs_reference', '-')} | "
                f"**{status}** | {vc.get('details', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render footer."""
        return "---\n\n*Generated by GreenLang PACK-015 Double Materiality Pack*"

    # -- HTML sections --

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return '<h1>Stakeholder Engagement Report</h1>'

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML overview."""
        validated = data.get("validation_passed", False)
        return (
            f'<h2>Overview</h2>\n'
            f'<p>Validation: {"PASS" if validated else "FAIL"}</p>'
        )

    def _html_coverage(self, data: Dict[str, Any]) -> str:
        """Render HTML coverage."""
        return '<h2>Stakeholder Category Coverage</h2>'

    def _html_findings(self, data: Dict[str, Any]) -> str:
        """Render HTML findings."""
        findings = data.get("synthesized_findings", [])
        rows = ""
        for f in findings[:10]:
            rows += (
                f'<tr><td>{f.get("topic", "-")}</td>'
                f'<td>{self._fmt(f.get("priority_score", 0))}</td></tr>\n'
            )
        return (
            f'<h2>Synthesized Findings</h2>\n'
            f'<table><tr><th>Finding</th><th>Priority</th></tr>\n'
            f'{rows}</table>'
        )

    # -- Helpers --

    def _css(self) -> str:
        """Build CSS."""
        return (
            "body{font-family:system-ui,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format numeric value."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
