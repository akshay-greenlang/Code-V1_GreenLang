# -*- coding: utf-8 -*-
"""
FoodWasteReportTemplate - Food waste management report for PACK-014.

Renders food waste baseline data, category breakdowns by destination,
reduction target tracking vs EU 30% goal, waste emissions calculations,
and recommendations for waste reduction strategies.

Example:
    >>> template = FoodWasteReportTemplate()
    >>> data = {"waste_records": [...], "targets": {...}}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FoodWasteReportTemplate:
    """
    Food waste management report template for retail.

    Renders waste baselines, category-level analysis (produce, bakery,
    dairy, meat, prepared foods), destination tracking (redistribution,
    animal feed, composting, anaerobic digestion, landfill), reduction
    progress vs EU Farm to Fork 30% target, and waste-related emissions.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    EU_REDUCTION_TARGET_PCT: float = 30.0
    EU_TARGET_YEAR: int = 2030

    DESTINATION_HIERARCHY: List[str] = [
        "Redistribution", "Animal Feed", "Composting",
        "Anaerobic Digestion", "Incineration", "Landfill",
    ]

    DESTINATION_COLORS: Dict[str, str] = {
        "Redistribution": "#059669",
        "Animal Feed": "#10b981",
        "Composting": "#34d399",
        "Anaerobic Digestion": "#d97706",
        "Incineration": "#ea580c",
        "Landfill": "#dc2626",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FoodWasteReportTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render food waste report as Markdown.

        Args:
            data: Report data with waste_records, baseline, targets,
                  category_breakdown, emissions, recommendations.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._md_header(data))
        sections.append(self._md_executive_summary(data))
        sections.append(self._md_baseline(data))
        sections.append(self._md_category_breakdown(data))
        sections.append(self._md_destination_analysis(data))
        sections.append(self._md_reduction_tracking(data))
        sections.append(self._md_waste_emissions(data))
        sections.append(self._md_recommendations(data))
        sections.append(self._md_provenance(data))

        return "\n\n".join(sections)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render food waste report as HTML with inline CSS.

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
            f"<title>Food Waste Report - {company} - {period}</title>",
            '<meta charset="utf-8">',
            "<style>",
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #1f2937; }",
            "h1 { color: #065f46; border-bottom: 3px solid #065f46; padding-bottom: 8px; }",
            "h2 { color: #047857; margin-top: 32px; }",
            "table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
            "th, td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }",
            "th { background: #f0fdf4; font-weight: 600; }",
            "tr:nth-child(even) { background: #f9fafb; }",
            ".on-track { color: #059669; font-weight: 600; }",
            ".behind { color: #dc2626; font-weight: 600; }",
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
        """Render food waste report as JSON.

        Args:
            data: Report data dict.

        Returns:
            Pretty-printed JSON string.
        """
        self.generated_at = datetime.utcnow()
        output = {
            "template": "food_waste_report",
            "version": "14.0.0",
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_hash(data),
            "eu_target": {"reduction_pct": self.EU_REDUCTION_TARGET_PCT, "year": self.EU_TARGET_YEAR},
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
            f"# Food Waste Management Report\n\n"
            f"**Company:** {company}  \n"
            f"**Period:** {period}  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 Food Waste Report v14.0.0"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        total = data.get("total_waste_tonnes", 0)
        baseline = data.get("baseline_tonnes", 0)
        reduction = ((baseline - total) / baseline * 100) if baseline > 0 else 0
        target = self.EU_REDUCTION_TARGET_PCT
        status = "On Track" if reduction >= (target * 0.6) else "Behind Target"

        return (
            f"## Executive Summary\n\n"
            f"- **Total Food Waste:** {total:,.1f} tonnes\n"
            f"- **Baseline (reference year):** {baseline:,.1f} tonnes\n"
            f"- **Reduction Achieved:** {reduction:.1f}%\n"
            f"- **EU 2030 Target:** {target:.0f}% reduction\n"
            f"- **Status:** {status}\n"
            f"- **Waste Emissions:** {data.get('waste_tco2e', 0):,.1f} tCO2e"
        )

    def _md_baseline(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline", {})
        if not baseline:
            return "## Baseline\n\nNo baseline data available."
        return (
            f"## Waste Baseline\n\n"
            f"- **Reference Year:** {baseline.get('year', 'N/A')}\n"
            f"- **Baseline Volume:** {baseline.get('tonnes', 0):,.1f} tonnes\n"
            f"- **Measurement Method:** {baseline.get('method', 'N/A')}\n"
            f"- **Boundary:** {baseline.get('boundary', 'All retail operations')}"
        )

    def _md_category_breakdown(self, data: Dict[str, Any]) -> str:
        categories = data.get("category_breakdown", [])
        if not categories:
            return "## Category Breakdown\n\nNo category data available."
        lines = [
            "## Waste by Food Category\n",
            "| Category | Tonnes | % of Total | Waste Rate (%) | Trend |",
            "|----------|--------|------------|----------------|-------|",
        ]
        total = data.get("total_waste_tonnes", 1)
        for c in categories:
            pct = (c.get("tonnes", 0) / total) * 100
            lines.append(
                f"| {c.get('category', 'N/A')} | {c.get('tonnes', 0):,.1f} "
                f"| {pct:.1f}% | {c.get('waste_rate_pct', 0):.1f}% "
                f"| {c.get('trend', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_destination_analysis(self, data: Dict[str, Any]) -> str:
        destinations = data.get("destinations", [])
        if not destinations:
            return "## Destination Analysis\n\nNo destination data available."
        lines = [
            "## Waste Destination Analysis\n",
            "| Destination | Tonnes | % of Total | Emissions Factor |",
            "|-------------|--------|------------|------------------|",
        ]
        total = data.get("total_waste_tonnes", 1)
        for d in destinations:
            pct = (d.get("tonnes", 0) / total) * 100
            lines.append(
                f"| {d.get('destination', 'N/A')} | {d.get('tonnes', 0):,.1f} "
                f"| {pct:.1f}% | {d.get('ef_kgco2e_per_tonne', 0):.0f} kgCO2e/t |"
            )
        return "\n".join(lines)

    def _md_reduction_tracking(self, data: Dict[str, Any]) -> str:
        tracking = data.get("reduction_tracking", [])
        if not tracking:
            return "## Reduction Tracking\n\nNo tracking data available."
        lines = [
            "## Reduction Progress vs EU 2030 Target\n",
            "| Year | Waste (t) | Reduction (%) | On Track | Required Pace |",
            "|------|-----------|---------------|----------|---------------|",
        ]
        for t in tracking:
            on_track = "Yes" if t.get("on_track", False) else "No"
            lines.append(
                f"| {t.get('year', 'N/A')} | {t.get('tonnes', 0):,.1f} "
                f"| {t.get('reduction_pct', 0):.1f}% | {on_track} "
                f"| {t.get('required_pace_pct', 0):.1f}% |"
            )
        return "\n".join(lines)

    def _md_waste_emissions(self, data: Dict[str, Any]) -> str:
        emissions = data.get("waste_emissions", {})
        if not emissions:
            return "## Waste Emissions\n\nNo emissions data available."
        return (
            f"## Waste-Related Emissions\n\n"
            f"- **Total Waste Emissions:** {emissions.get('total_tco2e', 0):,.1f} tCO2e\n"
            f"- **Landfill (CH4):** {emissions.get('landfill_tco2e', 0):,.1f} tCO2e\n"
            f"- **Incineration:** {emissions.get('incineration_tco2e', 0):,.1f} tCO2e\n"
            f"- **Composting:** {emissions.get('composting_tco2e', 0):,.1f} tCO2e\n"
            f"- **Avoided (redistribution):** -{emissions.get('avoided_tco2e', 0):,.1f} tCO2e"
        )

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        if not recs:
            return "## Recommendations\n\nNo recommendations available."
        lines = [
            "## Recommendations\n",
            "| Priority | Action | Expected Reduction (t) | Investment | Payback |",
            "|----------|--------|------------------------|------------|---------|",
        ]
        for r in recs:
            lines.append(
                f"| {r.get('priority', 'N/A')} | {r.get('action', 'N/A')} "
                f"| {r.get('reduction_tonnes', 0):,.0f} | EUR {r.get('investment_eur', 0):,.0f} "
                f"| {r.get('payback_months', 0):.0f} months |"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        h = self._compute_hash(data)
        return (
            f"---\n\n"
            f"**Provenance:** SHA-256 `{h}`  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 FoodWasteReportTemplate v14.0.0"
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
