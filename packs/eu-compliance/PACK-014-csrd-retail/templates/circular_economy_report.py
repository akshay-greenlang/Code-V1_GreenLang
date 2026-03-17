# -*- coding: utf-8 -*-
"""
CircularEconomyReportTemplate - Circular economy report for PACK-014.

Renders take-back program metrics, material recovery rates, EPR scheme
compliance, Material Circularity Index (MCI), waste diversion rates,
and recycled content tracking.

Example:
    >>> template = CircularEconomyReportTemplate()
    >>> data = {"take_back": [...], "epr": {...}}
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CircularEconomyReportTemplate:
    """
    Circular economy report template for retail.

    Renders take-back collection volumes, material recovery and
    recycling rates, EPR scheme compliance status (WEEE, packaging,
    textiles, batteries), Material Circularity Index, waste diversion
    metrics, and circular economy improvement recommendations.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    EPR_SCHEMES: List[str] = ["WEEE", "Packaging", "Textiles", "Batteries"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CircularEconomyReportTemplate.

        Args:
            config: Optional configuration dict.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render circular economy report as Markdown.

        Args:
            data: Report data with take_back, material_recovery, epr,
                  mci, waste_diversion, recommendations.

        Returns:
            Complete Markdown string.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._md_header(data))
        sections.append(self._md_executive_summary(data))
        sections.append(self._md_take_back_programs(data))
        sections.append(self._md_material_recovery(data))
        sections.append(self._md_epr_compliance(data))
        sections.append(self._md_circularity_index(data))
        sections.append(self._md_waste_diversion(data))
        sections.append(self._md_recommendations(data))
        sections.append(self._md_provenance(data))

        return "\n\n".join(sections)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render circular economy report as HTML.

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
            f"<title>Circular Economy Report - {company} - {period}</title>",
            '<meta charset="utf-8">',
            "<style>",
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #1f2937; }",
            "h1 { color: #065f46; border-bottom: 3px solid #065f46; padding-bottom: 8px; }",
            "h2 { color: #047857; margin-top: 32px; }",
            "table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
            "th, td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }",
            "th { background: #f0fdf4; font-weight: 600; }",
            "tr:nth-child(even) { background: #f9fafb; }",
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
        """Render circular economy report as JSON.

        Args:
            data: Report data dict.

        Returns:
            Pretty-printed JSON string.
        """
        self.generated_at = datetime.utcnow()
        output = {
            "template": "circular_economy_report",
            "version": "14.0.0",
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_hash(data),
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
            f"# Circular Economy Report\n\n"
            f"**Company:** {company}  \n"
            f"**Period:** {period}  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 Circular Economy v14.0.0"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        return (
            f"## Executive Summary\n\n"
            f"- **Material Circularity Index:** {data.get('mci', 0):.2f}\n"
            f"- **Waste Diversion Rate:** {data.get('diversion_rate_pct', 0):.1f}%\n"
            f"- **Take-Back Volume:** {data.get('take_back_tonnes', 0):,.1f} tonnes\n"
            f"- **Recycled Content (avg):** {data.get('avg_recycled_content_pct', 0):.1f}%\n"
            f"- **EPR Compliance:** {data.get('epr_compliant_count', 0)}/{data.get('epr_total_schemes', 0)} schemes"
        )

    def _md_take_back_programs(self, data: Dict[str, Any]) -> str:
        programs = data.get("take_back", [])
        if not programs:
            return "## Take-Back Programs\n\nNo take-back data available."
        lines = [
            "## Take-Back Programs\n",
            "| Program | Stream | Collected (t) | Recovered (t) | Recovery Rate (%) |",
            "|---------|--------|---------------|----------------|-------------------|",
        ]
        for p in programs:
            rate = (p.get("recovered_t", 0) / p.get("collected_t", 1)) * 100
            lines.append(
                f"| {p.get('name', 'N/A')} | {p.get('stream', 'N/A')} "
                f"| {p.get('collected_t', 0):,.1f} | {p.get('recovered_t', 0):,.1f} "
                f"| {rate:.1f}% |"
            )
        return "\n".join(lines)

    def _md_material_recovery(self, data: Dict[str, Any]) -> str:
        recovery = data.get("material_recovery", [])
        if not recovery:
            return "## Material Recovery\n\nNo recovery data available."
        lines = [
            "## Material Recovery\n",
            "| Material | Input (t) | Recycled (t) | Downcycled (t) | Loss (t) | Rate (%) |",
            "|----------|-----------|--------------|----------------|----------|----------|",
        ]
        for r in recovery:
            total_in = r.get("input_t", 1)
            rate = (r.get("recycled_t", 0) / total_in) * 100
            lines.append(
                f"| {r.get('material', 'N/A')} | {r.get('input_t', 0):,.1f} "
                f"| {r.get('recycled_t', 0):,.1f} | {r.get('downcycled_t', 0):,.1f} "
                f"| {r.get('loss_t', 0):,.1f} | {rate:.1f}% |"
            )
        return "\n".join(lines)

    def _md_epr_compliance(self, data: Dict[str, Any]) -> str:
        epr = data.get("epr", [])
        if not epr:
            return "## EPR Compliance\n\nNo EPR data available."
        lines = [
            "## EPR Scheme Compliance\n",
            "| Scheme | Registered | Fee Paid (EUR) | Target Rate (%) | Actual (%) | Compliant |",
            "|--------|------------|----------------|-----------------|------------|-----------|",
        ]
        for e in epr:
            compliant = "Yes" if e.get("compliant", False) else "No"
            lines.append(
                f"| {e.get('scheme', 'N/A')} "
                f"| {'Yes' if e.get('registered') else 'No'} "
                f"| {e.get('fee_eur', 0):,.0f} "
                f"| {e.get('target_rate_pct', 0):.0f}% "
                f"| {e.get('actual_rate_pct', 0):.1f}% "
                f"| {compliant} |"
            )
        return "\n".join(lines)

    def _md_circularity_index(self, data: Dict[str, Any]) -> str:
        mci = data.get("mci_details", {})
        if not mci:
            return "## Circularity Index\n\nNo MCI data available."
        return (
            f"## Material Circularity Index (MCI)\n\n"
            f"- **MCI Score:** {mci.get('score', 0):.2f} (0=linear, 1=fully circular)\n"
            f"- **Virgin Material Input:** {mci.get('virgin_input_pct', 0):.1f}%\n"
            f"- **Recycled Input:** {mci.get('recycled_input_pct', 0):.1f}%\n"
            f"- **End-of-Life Recovery:** {mci.get('eol_recovery_pct', 0):.1f}%\n"
            f"- **Utility Factor:** {mci.get('utility_factor', 1):.2f}\n"
            f"- **Sector Benchmark:** {mci.get('benchmark', 0):.2f}"
        )

    def _md_waste_diversion(self, data: Dict[str, Any]) -> str:
        diversion = data.get("waste_diversion", {})
        if not diversion:
            return "## Waste Diversion\n\nNo diversion data available."
        return (
            f"## Waste Diversion\n\n"
            f"- **Total Waste Generated:** {diversion.get('total_t', 0):,.1f} tonnes\n"
            f"- **Diverted from Landfill:** {diversion.get('diverted_t', 0):,.1f} tonnes\n"
            f"- **Diversion Rate:** {diversion.get('rate_pct', 0):.1f}%\n"
            f"- **Recycling Rate:** {diversion.get('recycling_rate_pct', 0):.1f}%\n"
            f"- **Energy Recovery:** {diversion.get('energy_recovery_t', 0):,.1f} tonnes"
        )

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        if not recs:
            return "## Recommendations\n\nNo recommendations available."
        lines = ["## Circular Economy Recommendations\n"]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"{i}. **{r.get('title', 'N/A')}** -- {r.get('description', 'N/A')} "
                f"(Impact: {r.get('impact', 'N/A')}, Timeline: {r.get('timeline', 'N/A')})"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        h = self._compute_hash(data)
        return (
            f"---\n\n"
            f"**Provenance:** SHA-256 `{h}`  \n"
            f"**Generated:** {self.generated_at.isoformat() if self.generated_at else 'N/A'}  \n"
            f"**Template:** PACK-014 CircularEconomyReportTemplate v14.0.0"
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
