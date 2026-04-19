# -*- coding: utf-8 -*-
"""
MergerAcquisitionReport - M&A Entity Details and Emission Impact for PACK-045.

Generates a merger and acquisition report covering entity details,
emission impact of M&A events, pro-rata adjustment calculations,
structural change timelines, and base year recalculation recommendations.

Sections:
    1. M&A Event Summary
    2. Entity Details
    3. Emission Impact Analysis
    4. Pro-Rata Adjustments
    5. Recalculation Recommendation

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 45.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "45.0.0"


def _event_type_label(event_type: str) -> str:
    """Return formatted M&A event type."""
    mapping = {
        "acquisition": "Acquisition",
        "merger": "Merger",
        "divestiture": "Divestiture",
        "outsourcing": "Outsourcing",
        "insourcing": "Insourcing",
    }
    return mapping.get(event_type.lower(), event_type.title())


class MergerAcquisitionReport:
    """
    Merger and acquisition report template.

    Renders M&A event details, entity emissions, pro-rata adjustment
    calculations, structural change timelines, and base year recalculation
    recommendations. All outputs include SHA-256 provenance hashing for
    audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = MergerAcquisitionReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MergerAcquisitionReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render merger acquisition report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_event_summary(data),
            self._md_entity_details(data),
            self._md_emission_impact(data),
            self._md_prorata_adjustments(data),
            self._md_recommendation(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render merger acquisition report as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_event_summary(data),
            self._html_entity_details(data),
            self._html_emission_impact(data),
            self._html_prorata_adjustments(data),
            self._html_recommendation(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render merger acquisition report as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "merger_acquisition_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "base_year": self._get_val(data, "base_year", ""),
            "events": data.get("events", []),
            "entities": data.get("entities", []),
            "emission_impact": data.get("emission_impact", {}),
            "prorata_adjustments": data.get("prorata_adjustments", []),
            "recommendation": data.get("recommendation", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        return (
            f"# Merger & Acquisition Impact Report - {company}\n\n"
            f"**Base Year:** {base_year} | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_event_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown M&A event summary."""
        events = data.get("events", [])
        if not events:
            return "## 1. M&A Event Summary\n\nNo M&A events recorded."
        lines = [
            "## 1. M&A Event Summary",
            "",
            "| # | Event Type | Entity | Date | Ownership % | Status |",
            "|---|-----------|--------|------|------------|--------|",
        ]
        for i, e in enumerate(events, 1):
            etype = _event_type_label(e.get("event_type", ""))
            entity = e.get("entity_name", "")
            date = e.get("effective_date", "")
            ownership = e.get("ownership_pct", 0)
            status = e.get("status", "pending")
            lines.append(
                f"| {i} | {etype} | {entity} | {date} | "
                f"{ownership:.0f}% | **{status.upper()}** |"
            )
        return "\n".join(lines)

    def _md_entity_details(self, data: Dict[str, Any]) -> str:
        """Render Markdown entity details."""
        entities = data.get("entities", [])
        if not entities:
            return ""
        lines = [
            "## 2. Entity Details",
            "",
            "| Entity | Sector | Country | Employees | Revenue ($M) | Annual Emissions (tCO2e) |",
            "|--------|--------|---------|----------|-------------|------------------------|",
        ]
        for e in entities:
            name = e.get("name", "")
            sector = e.get("sector", "")
            country = e.get("country", "")
            employees = e.get("employees", 0)
            revenue = e.get("revenue_million", 0)
            emissions = e.get("annual_emissions_tco2e", 0)
            lines.append(
                f"| {name} | {sector} | {country} | {employees:,} | "
                f"{revenue:,.1f} | {emissions:,.1f} |"
            )
        return "\n".join(lines)

    def _md_emission_impact(self, data: Dict[str, Any]) -> str:
        """Render Markdown emission impact analysis."""
        impact = data.get("emission_impact", {})
        if not impact:
            return ""
        original = impact.get("original_base_year_tco2e", 0)
        added = impact.get("added_tco2e", 0)
        removed = impact.get("removed_tco2e", 0)
        adjusted = impact.get("adjusted_base_year_tco2e", 0)
        pct_change = impact.get("change_pct", 0)
        lines = [
            "## 3. Emission Impact Analysis",
            "",
            "| Metric | Value (tCO2e) |",
            "|--------|--------------|",
            f"| Original Base Year | {original:,.1f} |",
            f"| Emissions Added (M&A) | {added:+,.1f} |",
            f"| Emissions Removed (Divestitures) | {removed:+,.1f} |",
            f"| **Adjusted Base Year** | **{adjusted:,.1f}** |",
            f"| **Change** | **{pct_change:+.1f}%** |",
        ]
        scopes = impact.get("by_scope", [])
        if scopes:
            lines.extend([
                "",
                "**Impact by Scope:**",
                "",
                "| Scope | Original | Adjustment | Adjusted |",
                "|-------|---------|-----------|---------|",
            ])
            for s in scopes:
                name = s.get("scope", "")
                orig = s.get("original_tco2e", 0)
                adj = s.get("adjustment_tco2e", 0)
                final = s.get("adjusted_tco2e", 0)
                lines.append(f"| {name} | {orig:,.1f} | {adj:+,.1f} | {final:,.1f} |")
        return "\n".join(lines)

    def _md_prorata_adjustments(self, data: Dict[str, Any]) -> str:
        """Render Markdown pro-rata adjustments."""
        adjustments = data.get("prorata_adjustments", [])
        if not adjustments:
            return ""
        lines = [
            "## 4. Pro-Rata Adjustments",
            "",
            "| Entity | Period | Full Year (tCO2e) | Pro-Rata Factor | Adjusted (tCO2e) |",
            "|--------|--------|------------------|----------------|-----------------|",
        ]
        for a in adjustments:
            entity = a.get("entity_name", "")
            period = a.get("period", "")
            full_year = a.get("full_year_tco2e", 0)
            factor = a.get("prorata_factor", 0)
            adjusted = a.get("adjusted_tco2e", 0)
            lines.append(
                f"| {entity} | {period} | {full_year:,.1f} | "
                f"{factor:.4f} | {adjusted:,.1f} |"
            )
        return "\n".join(lines)

    def _md_recommendation(self, data: Dict[str, Any]) -> str:
        """Render Markdown recalculation recommendation."""
        rec = data.get("recommendation", {})
        if not rec:
            return ""
        action = rec.get("action", "")
        rationale = rec.get("rationale", "")
        significance = rec.get("significance_test_result", "")
        return (
            "## 5. Recalculation Recommendation\n\n"
            f"- **Action:** {action}\n"
            f"- **Rationale:** {rationale}\n"
            f"- **Significance Test:** {significance}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-045 Base Year Management v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>M&amp;A Impact - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #e9c46a;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".impact-add{color:#e76f51;}\n"
            ".impact-remove{color:#2a9d8f;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        return (
            '<div class="section">\n'
            f"<h1>Merger &amp; Acquisition Impact &mdash; {company}</h1>\n"
            f"<p><strong>Base Year:</strong> {base_year}</p>\n<hr>\n</div>"
        )

    def _html_event_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML event summary table."""
        events = data.get("events", [])
        if not events:
            return ""
        rows = ""
        for i, e in enumerate(events, 1):
            etype = _event_type_label(e.get("event_type", ""))
            entity = e.get("entity_name", "")
            date = e.get("effective_date", "")
            ownership = e.get("ownership_pct", 0)
            status = e.get("status", "pending").upper()
            rows += f"<tr><td>{i}</td><td>{etype}</td><td>{entity}</td><td>{date}</td><td>{ownership:.0f}%</td><td><strong>{status}</strong></td></tr>\n"
        return (
            '<div class="section">\n<h2>1. M&amp;A Events</h2>\n'
            "<table><thead><tr><th>#</th><th>Type</th><th>Entity</th>"
            "<th>Date</th><th>Ownership</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_entity_details(self, data: Dict[str, Any]) -> str:
        """Render HTML entity details table."""
        entities = data.get("entities", [])
        if not entities:
            return ""
        rows = ""
        for e in entities:
            name = e.get("name", "")
            sector = e.get("sector", "")
            country = e.get("country", "")
            emissions = e.get("annual_emissions_tco2e", 0)
            rows += f"<tr><td>{name}</td><td>{sector}</td><td>{country}</td><td>{emissions:,.1f}</td></tr>\n"
        return (
            '<div class="section">\n<h2>2. Entity Details</h2>\n'
            "<table><thead><tr><th>Entity</th><th>Sector</th>"
            "<th>Country</th><th>Emissions tCO2e</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_emission_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML emission impact analysis."""
        impact = data.get("emission_impact", {})
        if not impact:
            return ""
        original = impact.get("original_base_year_tco2e", 0)
        adjusted = impact.get("adjusted_base_year_tco2e", 0)
        pct = impact.get("change_pct", 0)
        return (
            '<div class="section">\n<h2>3. Emission Impact</h2>\n'
            f"<p><strong>Original:</strong> {original:,.1f} tCO2e | "
            f"<strong>Adjusted:</strong> {adjusted:,.1f} tCO2e | "
            f"<strong>Change:</strong> {pct:+.1f}%</p>\n</div>"
        )

    def _html_prorata_adjustments(self, data: Dict[str, Any]) -> str:
        """Render HTML pro-rata adjustments table."""
        adjustments = data.get("prorata_adjustments", [])
        if not adjustments:
            return ""
        rows = ""
        for a in adjustments:
            entity = a.get("entity_name", "")
            full_year = a.get("full_year_tco2e", 0)
            factor = a.get("prorata_factor", 0)
            adjusted = a.get("adjusted_tco2e", 0)
            rows += f"<tr><td>{entity}</td><td>{full_year:,.1f}</td><td>{factor:.4f}</td><td>{adjusted:,.1f}</td></tr>\n"
        return (
            '<div class="section">\n<h2>4. Pro-Rata Adjustments</h2>\n'
            "<table><thead><tr><th>Entity</th><th>Full Year tCO2e</th>"
            "<th>Factor</th><th>Adjusted tCO2e</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_recommendation(self, data: Dict[str, Any]) -> str:
        """Render HTML recalculation recommendation."""
        rec = data.get("recommendation", {})
        if not rec:
            return ""
        action = rec.get("action", "")
        rationale = rec.get("rationale", "")
        return (
            '<div class="section">\n<h2>5. Recommendation</h2>\n'
            f"<p><strong>Action:</strong> {action}</p>\n"
            f"<p><strong>Rationale:</strong> {rationale}</p>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-045 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
