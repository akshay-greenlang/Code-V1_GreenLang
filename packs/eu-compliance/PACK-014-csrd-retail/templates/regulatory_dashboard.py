# -*- coding: utf-8 -*-
"""
RegulatoryDashboardTemplate - Multi-regulation compliance dashboard for PACK-014.

Renders compliance status across CSRD, PPWR, EUDR, CSDDD, ESPR, ECGT,
EU Taxonomy, and national regulations, with gap analysis, action items,
timeline tracking, and overall readiness scoring.

Example:
    >>> template = RegulatoryDashboardTemplate()
    >>> data = {"regulations": [...], "gaps": {...}}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegulatoryDashboardTemplate:
    """
    Multi-regulation compliance dashboard template for retail.

    Renders compliance status cards for 8+ EU regulations, gap analysis,
    action items with priority and deadline, compliance timeline,
    sub-sector applicability map, and overall readiness score.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    RETAIL_REGULATIONS: List[str] = [
        "CSRD", "PPWR", "EUDR", "CSDDD", "ESPR",
        "ECGT", "EU Taxonomy", "F-Gas Regulation",
    ]

    COMPLIANCE_COLORS: Dict[str, str] = {
        "compliant": "#059669",
        "partial": "#d97706",
        "non_compliant": "#dc2626",
        "not_applicable": "#6b7280",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RegulatoryDashboardTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render regulatory dashboard as Markdown.

        Args:
            data: Report data with regulations, gaps, actions,
                  timeline, readiness_score.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._md_header(data))
        sections.append(self._md_executive_summary(data))
        sections.append(self._md_regulation_status(data))
        sections.append(self._md_gap_analysis(data))
        sections.append(self._md_action_items(data))
        sections.append(self._md_compliance_timeline(data))
        sections.append(self._md_sub_sector_applicability(data))
        sections.append(self._md_readiness_score(data))
        sections.append(self._md_provenance(data))

        return "\n\n".join(sections)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render regulatory dashboard as HTML.

        Args:
            data: Report data dict.

        Returns:
            Complete HTML string.
        """
        self.generated_at = datetime.utcnow()
        md = self.render_markdown(data)
        company = data.get("company_name", "Retail Company")
        period = data.get("period", "FY2025")

        html_parts: List[str] = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            f"<title>Regulatory Compliance Dashboard - {company} - {period}</title>",
            '<meta charset="utf-8">',
            "<style>",
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #1f2937; }",
            "h1 { color: #065f46; border-bottom: 3px solid #065f46; padding-bottom: 8px; }",
            "h2 { color: #047857; margin-top: 32px; }",
            "h3 { color: #059669; }",
            "table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
            "th, td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }",
            "th { background: #f0fdf4; font-weight: 600; }",
            "tr:nth-child(even) { background: #f9fafb; }",
            ".compliant { color: #059669; font-weight: 700; }",
            ".partial { color: #d97706; font-weight: 600; }",
            ".non-compliant { color: #dc2626; font-weight: 700; }",
            ".dashboard-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 16px 0; }",
            ".reg-card { border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; text-align: center; }",
            ".provenance { background: #f3f4f6; padding: 12px; border-radius: 6px; font-size: 11px; }",
            "</style>",
            "</head>",
            "<body>",
        ]
        for line in md.split("\n"):
            if line.startswith("# "):
                html_parts.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_parts.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                html_parts.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("| "):
                html_parts.append(self._md_table_row_to_html(line))
            elif line.startswith("- "):
                html_parts.append(f"<li>{line[2:]}</li>")
            elif line.strip():
                html_parts.append(f"<p>{line}</p>")
        html_parts.extend(["</body>", "</html>"])
        return "\n".join(html_parts)

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render regulatory dashboard as JSON.

        Args:
            data: Report data dict.

        Returns:
            Pretty-printed JSON string.
        """
        self.generated_at = datetime.utcnow()
        output = {
            "template": "regulatory_dashboard",
            "version": "14.0.0",
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_hash(data),
            "regulations_tracked": self.RETAIL_REGULATIONS,
            "data": data,
        }
        return json.dumps(output, indent=2, default=str)

    # ------------------------------------------------------------------
    # Private section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        company = data.get("company_name", "Retail Company")
        period = data.get("period", "FY2025")
        return (
            f"# Regulatory Compliance Dashboard\n\n"
            f"**Company:** {company}  \n"
            f"**Period:** {period}  \n"
            f"**Sub-Sector:** {data.get('sub_sector', 'General Retail')}  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 Regulatory Dashboard v14.0.0"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        regs = data.get("regulations", [])
        compliant = sum(1 for r in regs if r.get("status") == "compliant")
        partial = sum(1 for r in regs if r.get("status") == "partial")
        non_compliant = sum(1 for r in regs if r.get("status") == "non_compliant")

        return (
            f"## Executive Summary\n\n"
            f"- **Regulations Tracked:** {len(regs)}\n"
            f"- **Fully Compliant:** {compliant}\n"
            f"- **Partially Compliant:** {partial}\n"
            f"- **Non-Compliant:** {non_compliant}\n"
            f"- **Overall Readiness:** {data.get('readiness_score', 0):.0f}/100\n"
            f"- **Open Action Items:** {data.get('open_actions', 0)}"
        )

    def _md_regulation_status(self, data: Dict[str, Any]) -> str:
        regs = data.get("regulations", [])
        if not regs:
            return "## Regulation Status\n\nNo regulation data available."
        lines = [
            "## Regulation Status\n",
            "| Regulation | Status | Compliance (%) | Deadline | Priority | Gap Count |",
            "|------------|--------|----------------|----------|----------|-----------|",
        ]
        for r in regs:
            lines.append(
                f"| {r.get('name', 'N/A')} | {r.get('status', 'N/A')} "
                f"| {r.get('compliance_pct', 0):.0f}% | {r.get('deadline', 'N/A')} "
                f"| {r.get('priority', 'N/A')} | {r.get('gap_count', 0)} |"
            )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        gaps = data.get("gaps", [])
        if not gaps:
            return "## Gap Analysis\n\nNo compliance gaps identified."
        lines = [
            "## Compliance Gap Analysis\n",
            "| Regulation | Requirement | Current State | Gap | Effort (weeks) |",
            "|------------|-------------|---------------|-----|----------------|",
        ]
        for g in gaps:
            lines.append(
                f"| {g.get('regulation', 'N/A')} | {g.get('requirement', 'N/A')} "
                f"| {g.get('current_state', 'N/A')} | {g.get('gap', 'N/A')} "
                f"| {g.get('effort_weeks', 0):.0f} |"
            )
        return "\n".join(lines)

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        actions = data.get("actions", [])
        if not actions:
            return "## Action Items\n\nNo action items pending."
        lines = [
            "## Action Items\n",
            "| # | Action | Regulation | Priority | Owner | Deadline | Status |",
            "|---|--------|------------|----------|-------|----------|--------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', 'N/A')} | {a.get('regulation', 'N/A')} "
                f"| {a.get('priority', 'N/A')} | {a.get('owner', 'N/A')} "
                f"| {a.get('deadline', 'N/A')} | {a.get('status', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_compliance_timeline(self, data: Dict[str, Any]) -> str:
        timeline = data.get("timeline", [])
        if not timeline:
            return "## Compliance Timeline\n\nNo timeline data available."
        lines = [
            "## Compliance Timeline\n",
            "| Date | Regulation | Milestone | Status |",
            "|------|------------|-----------|--------|",
        ]
        for t in timeline:
            lines.append(
                f"| {t.get('date', 'N/A')} | {t.get('regulation', 'N/A')} "
                f"| {t.get('milestone', 'N/A')} | {t.get('status', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_sub_sector_applicability(self, data: Dict[str, Any]) -> str:
        applicability = data.get("sub_sector_applicability", [])
        if not applicability:
            return "## Sub-Sector Applicability\n\nNo applicability data available."
        lines = [
            "## Regulation Applicability by Sub-Sector\n",
            "| Regulation | Grocery | Apparel | Electronics | Online | DIY |",
            "|------------|---------|---------|-------------|--------|-----|",
        ]
        for a in applicability:
            lines.append(
                f"| {a.get('regulation', 'N/A')} "
                f"| {'Yes' if a.get('grocery') else 'No'} "
                f"| {'Yes' if a.get('apparel') else 'No'} "
                f"| {'Yes' if a.get('electronics') else 'No'} "
                f"| {'Yes' if a.get('online') else 'No'} "
                f"| {'Yes' if a.get('diy') else 'No'} |"
            )
        return "\n".join(lines)

    def _md_readiness_score(self, data: Dict[str, Any]) -> str:
        readiness = data.get("readiness_breakdown", {})
        if not readiness:
            return "## Readiness Score\n\nNo readiness data available."
        lines = ["## Readiness Score Breakdown\n"]
        for component, score in readiness.items():
            bar_len = min(int(score / 2), 50)
            bar = "#" * bar_len
            label = component.replace("_", " ").title()
            lines.append(f"- **{label}:** {score:.0f}/100 {bar}")
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        h = self._compute_hash(data)
        return (
            f"---\n\n"
            f"**Provenance:** SHA-256 `{h}`  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 RegulatoryDashboardTemplate v14.0.0"
        )

    @staticmethod
    def _compute_hash(data: Dict[str, Any]) -> str:
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _md_table_row_to_html(line: str) -> str:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if all(c.startswith("-") for c in cells):
            return ""
        row = "".join(f"<td>{c}</td>" for c in cells)
        return f"<tr>{row}</tr>"
