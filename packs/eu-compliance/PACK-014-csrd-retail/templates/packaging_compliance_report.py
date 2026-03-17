# -*- coding: utf-8 -*-
"""
PackagingComplianceReportTemplate - PPWR packaging compliance report for PACK-014.

Renders recycled content gap analysis, EPR eco-modulation grading,
material composition breakdown, labeling compliance status, PPWR
target tracking, and packaging optimization recommendations.

Example:
    >>> template = PackagingComplianceReportTemplate()
    >>> data = {"packaging_items": [...], "targets": {...}}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PackagingComplianceReportTemplate:
    """
    PPWR packaging compliance report template for retail.

    Renders recycled content analysis vs PPWR targets (2030/2035/2040),
    material composition, EPR eco-modulation grades, labeling status,
    and packaging optimization roadmap.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    PPWR_TARGETS: Dict[str, Dict[str, float]] = {
        "plastic_contact_sensitive": {"2030": 10.0, "2035": 25.0, "2040": 50.0},
        "plastic_bottles": {"2030": 30.0, "2035": 35.0, "2040": 65.0},
        "plastic_other": {"2030": 35.0, "2035": 50.0, "2040": 65.0},
    }

    GRADE_COLORS: Dict[str, str] = {
        "A": "#059669",
        "B": "#10b981",
        "C": "#d97706",
        "D": "#ea580c",
        "F": "#dc2626",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PackagingComplianceReportTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render packaging compliance report as Markdown.

        Args:
            data: Report data with packaging_items, targets,
                  epr_grades, labeling, recommendations.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._md_header(data))
        sections.append(self._md_executive_summary(data))
        sections.append(self._md_material_composition(data))
        sections.append(self._md_recycled_content_gaps(data))
        sections.append(self._md_epr_eco_modulation(data))
        sections.append(self._md_labeling_compliance(data))
        sections.append(self._md_ppwr_target_tracking(data))
        sections.append(self._md_optimization_roadmap(data))
        sections.append(self._md_provenance(data))

        return "\n\n".join(sections)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render packaging compliance report as HTML with inline CSS.

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
            f"<title>Packaging Compliance Report - {company} - {period}</title>",
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
            ".grade-a { color: #059669; font-weight: 700; }",
            ".grade-b { color: #10b981; font-weight: 600; }",
            ".grade-c { color: #d97706; }",
            ".grade-d { color: #ea580c; }",
            ".grade-f { color: #dc2626; font-weight: 700; }",
            ".provenance { background: #f3f4f6; padding: 12px; border-radius: 6px; font-size: 11px; margin-top: 40px; }",
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
        """Render packaging compliance report as JSON.

        Args:
            data: Report data dict.

        Returns:
            Pretty-printed JSON string.
        """
        self.generated_at = datetime.utcnow()
        provenance_hash = self._compute_hash(data)

        output = {
            "template": "packaging_compliance_report",
            "version": "14.0.0",
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance_hash,
            "ppwr_targets": self.PPWR_TARGETS,
            "data": data,
        }
        return json.dumps(output, indent=2, default=str)

    # ------------------------------------------------------------------
    # Private markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render report header."""
        company = data.get("company_name", "Retail Company")
        period = data.get("period", "FY2025")
        return (
            f"# Packaging Compliance Report (PPWR)\n\n"
            f"**Company:** {company}  \n"
            f"**Period:** {period}  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 Packaging Compliance v14.0.0"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary."""
        items = data.get("packaging_items", [])
        total_tonnage = sum(i.get("weight_tonnes", 0) for i in items)
        compliant = sum(1 for i in items if i.get("compliant", False))
        pct = (compliant / len(items) * 100) if items else 0

        return (
            f"## Executive Summary\n\n"
            f"- **Total Packaging Items:** {len(items)}\n"
            f"- **Total Tonnage:** {total_tonnage:,.1f} tonnes\n"
            f"- **PPWR Compliant Items:** {compliant} ({pct:.0f}%)\n"
            f"- **Avg Recycled Content:** {data.get('avg_recycled_content_pct', 0):.1f}%\n"
            f"- **EPR Fee Exposure:** EUR {data.get('total_epr_fee_eur', 0):,.0f}"
        )

    def _md_material_composition(self, data: Dict[str, Any]) -> str:
        """Render material composition breakdown."""
        materials = data.get("material_breakdown", [])
        if not materials:
            return "## Material Composition\n\nNo material data available."

        lines = [
            "## Material Composition\n",
            "| Material | Tonnage | % of Total | Recyclable | Recycled Content (%) |",
            "|----------|---------|------------|------------|----------------------|",
        ]
        total = sum(m.get("tonnage", 0) for m in materials) or 1
        for m in materials:
            pct = (m.get("tonnage", 0) / total) * 100
            recyclable = "Yes" if m.get("recyclable", False) else "No"
            lines.append(
                f"| {m.get('material', 'N/A')} | {m.get('tonnage', 0):,.1f} "
                f"| {pct:.1f}% | {recyclable} | {m.get('recycled_content_pct', 0):.1f}% |"
            )
        return "\n".join(lines)

    def _md_recycled_content_gaps(self, data: Dict[str, Any]) -> str:
        """Render recycled content gap analysis vs PPWR targets."""
        gaps = data.get("recycled_content_gaps", [])
        if not gaps:
            return "## Recycled Content Gaps\n\nAll items meet current targets."

        lines = [
            "## Recycled Content Gap Analysis\n",
            "| Category | Current (%) | 2030 Target (%) | Gap (pp) | 2040 Target (%) | Gap (pp) |",
            "|----------|-------------|-----------------|----------|-----------------|----------|",
        ]
        for g in gaps:
            gap_2030 = g.get("target_2030", 0) - g.get("current_pct", 0)
            gap_2040 = g.get("target_2040", 0) - g.get("current_pct", 0)
            lines.append(
                f"| {g.get('category', 'N/A')} | {g.get('current_pct', 0):.1f} "
                f"| {g.get('target_2030', 0):.0f} | {max(0, gap_2030):.1f} "
                f"| {g.get('target_2040', 0):.0f} | {max(0, gap_2040):.1f} |"
            )
        return "\n".join(lines)

    def _md_epr_eco_modulation(self, data: Dict[str, Any]) -> str:
        """Render EPR eco-modulation grades."""
        epr = data.get("epr_grades", [])
        if not epr:
            return "## EPR Eco-Modulation\n\nNo EPR data available."

        lines = [
            "## EPR Eco-Modulation Grades\n",
            "| Item | Material | Grade | Fee (EUR/t) | Modulated Fee | Saving/Penalty |",
            "|------|----------|-------|-------------|---------------|----------------|",
        ]
        for e in epr:
            diff = e.get("modulated_fee", 0) - e.get("base_fee", 0)
            sign = "+" if diff >= 0 else ""
            lines.append(
                f"| {e.get('item', 'N/A')} | {e.get('material', 'N/A')} "
                f"| {e.get('grade', 'N/A')} | {e.get('base_fee', 0):,.0f} "
                f"| {e.get('modulated_fee', 0):,.0f} | {sign}{diff:,.0f} |"
            )
        return "\n".join(lines)

    def _md_labeling_compliance(self, data: Dict[str, Any]) -> str:
        """Render labeling compliance status."""
        labels = data.get("labeling", [])
        if not labels:
            return "## Labeling Compliance\n\nNo labeling data available."

        lines = [
            "## Labeling Compliance\n",
            "| Item | Sorting Label | Material ID | Recycled Content | Digital Passport | Status |",
            "|------|--------------|-------------|------------------|------------------|--------|",
        ]
        for lb in labels:
            lines.append(
                f"| {lb.get('item', 'N/A')} "
                f"| {'Pass' if lb.get('sorting_label') else 'Fail'} "
                f"| {'Pass' if lb.get('material_id') else 'Fail'} "
                f"| {'Pass' if lb.get('recycled_content_label') else 'Fail'} "
                f"| {'Pass' if lb.get('digital_passport') else 'Fail'} "
                f"| {'Compliant' if lb.get('compliant') else 'Non-Compliant'} |"
            )
        return "\n".join(lines)

    def _md_ppwr_target_tracking(self, data: Dict[str, Any]) -> str:
        """Render PPWR target tracking timeline."""
        tracking = data.get("ppwr_tracking", {})
        if not tracking:
            return "## PPWR Target Tracking\n\nNo tracking data available."

        lines = ["## PPWR Target Tracking\n"]
        for category, targets in self.PPWR_TARGETS.items():
            current = tracking.get(category, {}).get("current_pct", 0)
            label = category.replace("_", " ").title()
            lines.append(f"### {label}")
            lines.append(f"- Current: {current:.1f}%")
            for year, target in targets.items():
                gap = max(0, target - current)
                status = "On Track" if current >= target else f"Gap: {gap:.1f}pp"
                lines.append(f"- {year}: {target:.0f}% target -- {status}")
            lines.append("")
        return "\n".join(lines)

    def _md_optimization_roadmap(self, data: Dict[str, Any]) -> str:
        """Render packaging optimization roadmap."""
        recommendations = data.get("recommendations", [])
        if not recommendations:
            return "## Optimization Roadmap\n\nNo recommendations available."

        lines = [
            "## Packaging Optimization Roadmap\n",
            "| Priority | Action | Impact | Timeline | Investment (EUR) |",
            "|----------|--------|--------|----------|------------------|",
        ]
        for r in recommendations:
            lines.append(
                f"| {r.get('priority', 'N/A')} | {r.get('action', 'N/A')} "
                f"| {r.get('impact', 'N/A')} | {r.get('timeline', 'N/A')} "
                f"| {r.get('investment_eur', 0):,.0f} |"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        """Render provenance footer."""
        h = self._compute_hash(data)
        return (
            f"---\n\n"
            f"**Provenance:** SHA-256 `{h}`  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 PackagingComplianceReportTemplate v14.0.0"
        )

    @staticmethod
    def _compute_hash(data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of data for provenance."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _md_table_row_to_html(line: str) -> str:
        """Convert a Markdown table row to HTML."""
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if all(c.startswith("-") for c in cells):
            return ""
        row = "".join(f"<td>{c}</td>" for c in cells)
        return f"<tr>{row}</tr>"
