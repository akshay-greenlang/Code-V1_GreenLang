# -*- coding: utf-8 -*-
"""
GHGEmissionsReportTemplate - ESRS E1-6 GHG Emissions Disclosure Report

Renders Scope 1/2/3 GHG emission summaries, gas disaggregation, intensity
metrics, base-year comparison, and methodology notes per ESRS E1-6
(Gross Scopes 1, 2 and 3 and Total GHG Emissions).

Sections:
    1. Scope 1 Summary
    2. Scope 2 Summary
    3. Scope 3 Summary
    4. Total Emissions
    5. Gas Disaggregation
    6. Intensity Metrics
    7. Base Year Comparison
    8. Methodology Notes

Author: GreenLang Team
Version: 16.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SECTIONS: List[str] = [
    "scope1_summary",
    "scope2_summary",
    "scope3_summary",
    "total_emissions",
    "gas_disaggregation",
    "intensity_metrics",
    "base_year_comparison",
    "methodology_notes",
]


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class GHGEmissionsReportTemplate:
    """
    GHG Emissions disclosure report template per ESRS E1-6.

    Renders Scope 1/2/3 breakdowns, gas-level disaggregation (CO2,
    CH4, N2O, HFCs, PFCs, SF6, NF3), intensity metrics (revenue and
    headcount based), and base-year comparison with methodology notes.

    Example:
        >>> tpl = GHGEmissionsReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GHGEmissionsReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {}
        for section in _SECTIONS:
            result[section] = self.render_section(section, data)
        result["provenance_hash"] = _compute_hash(result)
        result["generated_at"] = self.generated_at.isoformat()
        return result

    def render_section(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render a single section by name."""
        handler = getattr(self, f"_section_{name}", None)
        if handler is None:
            raise ValueError(f"Unknown section: {name}")
        return handler(data)

    def get_sections(self) -> List[str]:
        """Return list of available section names."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if "scope1_emissions" not in data:
            warnings.append("scope1_emissions missing; will default to empty")
        if "scope2_location_tco2e" not in data:
            warnings.append("scope2_location_tco2e missing; will default to 0")
        if "scope3_categories" not in data:
            warnings.append("scope3_categories missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render GHG emissions report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_scope1(data),
            self._md_scope2(data),
            self._md_scope3(data),
            self._md_total(data),
            self._md_gas_disaggregation(data),
            self._md_intensity(data),
            self._md_base_year(data),
            self._md_methodology(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render GHG emissions report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_scope1(data),
            self._html_scope2(data),
            self._html_total(data),
            self._html_gas_disaggregation(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>GHG Emissions Report - ESRS E1-6</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render GHG emissions report as JSON."""
        self.generated_at = _utcnow()
        result = {
            "template": "ghg_emissions_report",
            "esrs_reference": "E1-6",
            "version": "16.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "scope1_total_tco2e": data.get("scope1_total_tco2e", 0.0),
            "scope2_location_tco2e": data.get("scope2_location_tco2e", 0.0),
            "scope2_market_tco2e": data.get("scope2_market_tco2e", 0.0),
            "scope3_total_tco2e": data.get("scope3_total_tco2e", 0.0),
            "total_ghg_tco2e": data.get("total_ghg_tco2e", 0.0),
            "gas_disaggregation": data.get("gas_disaggregation", {}),
            "intensity_metrics": data.get("intensity_metrics", []),
            "base_year": data.get("base_year", {}),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_scope1_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build Scope 1 summary section."""
        emissions = data.get("scope1_emissions", [])
        total = sum(e.get("tco2e", 0.0) for e in emissions)
        return {
            "title": "Scope 1 - Direct GHG Emissions",
            "total_tco2e": round(total, 2),
            "source_count": len(emissions),
            "sources": [
                {
                    "category": e.get("category", ""),
                    "tco2e": round(e.get("tco2e", 0.0), 2),
                    "percentage": (
                        round(e.get("tco2e", 0.0) / total * 100, 1)
                        if total > 0 else 0.0
                    ),
                }
                for e in emissions
            ],
        }

    def _section_scope2_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build Scope 2 summary section."""
        return {
            "title": "Scope 2 - Indirect GHG Emissions from Energy",
            "location_based_tco2e": round(data.get("scope2_location_tco2e", 0.0), 2),
            "market_based_tco2e": round(data.get("scope2_market_tco2e", 0.0), 2),
            "renewable_percentage": round(data.get("renewable_electricity_pct", 0.0), 1),
            "dual_reporting_compliant": data.get("dual_reporting_compliant", False),
        }

    def _section_scope3_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build Scope 3 summary section."""
        categories = data.get("scope3_categories", [])
        total = sum(c.get("tco2e", 0.0) for c in categories)
        return {
            "title": "Scope 3 - Value Chain GHG Emissions",
            "total_tco2e": round(total, 2),
            "categories_reported": len(categories),
            "categories": [
                {
                    "category_number": c.get("category_number", 0),
                    "name": c.get("name", ""),
                    "tco2e": round(c.get("tco2e", 0.0), 2),
                    "methodology": c.get("methodology", ""),
                }
                for c in categories
            ],
        }

    def _section_total_emissions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build total emissions section."""
        s1 = data.get("scope1_total_tco2e", 0.0)
        s2_loc = data.get("scope2_location_tco2e", 0.0)
        s2_mkt = data.get("scope2_market_tco2e", 0.0)
        s3 = data.get("scope3_total_tco2e", 0.0)
        return {
            "title": "Total GHG Emissions",
            "scope1_tco2e": round(s1, 2),
            "scope2_location_tco2e": round(s2_loc, 2),
            "scope2_market_tco2e": round(s2_mkt, 2),
            "scope3_tco2e": round(s3, 2),
            "total_location_tco2e": round(s1 + s2_loc + s3, 2),
            "total_market_tco2e": round(s1 + s2_mkt + s3, 2),
        }

    def _section_gas_disaggregation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build gas disaggregation section."""
        gases = data.get("gas_disaggregation", {})
        gas_list = ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"]
        return {
            "title": "GHG Gas Disaggregation",
            "gases": {g: round(gases.get(g, 0.0), 2) for g in gas_list},
            "biogenic_co2": round(data.get("biogenic_co2_tco2e", 0.0), 2),
        }

    def _section_intensity_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build intensity metrics section."""
        metrics = data.get("intensity_metrics", [])
        return {
            "title": "GHG Intensity Metrics",
            "metrics": [
                {
                    "name": m.get("name", ""),
                    "value": round(m.get("value", 0.0), 4),
                    "unit": m.get("unit", ""),
                    "denominator": m.get("denominator", ""),
                }
                for m in metrics
            ],
        }

    def _section_base_year_comparison(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build base year comparison section."""
        base = data.get("base_year", {})
        base_total = base.get("total_tco2e", 0.0)
        current_total = data.get("total_ghg_tco2e", 0.0)
        change_pct = (
            round((current_total - base_total) / base_total * 100, 1)
            if base_total > 0
            else 0.0
        )
        return {
            "title": "Base Year Comparison",
            "base_year": base.get("year", ""),
            "base_year_tco2e": round(base_total, 2),
            "current_year_tco2e": round(current_total, 2),
            "change_percentage": change_pct,
            "recalculation_trigger": base.get("recalculation_trigger", ""),
            "recalculation_policy": base.get("recalculation_policy", ""),
        }

    def _section_methodology_notes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build methodology notes section."""
        return {
            "title": "Methodology Notes",
            "consolidation_approach": data.get(
                "consolidation_approach", "operational_control"
            ),
            "gwp_source": data.get("gwp_source", "IPCC AR6"),
            "scope3_screening_method": data.get("scope3_screening_method", ""),
            "exclusions": data.get("exclusions", []),
            "data_quality_notes": data.get("data_quality_notes", ""),
            "assurance_level": data.get("assurance_level", "limited"),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"# GHG Emissions Report - ESRS E1-6\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E1-6 Gross Scopes 1, 2 and 3 and Total GHG Emissions"
        )

    def _md_scope1(self, data: Dict[str, Any]) -> str:
        """Render Scope 1 markdown."""
        sec = self._section_scope1_summary(data)
        lines = [f"## {sec['title']}\n", f"**Total:** {sec['total_tco2e']:,.2f} tCO2e\n"]
        if sec["sources"]:
            lines.append("| Category | tCO2e | % |")
            lines.append("|----------|------:|--:|")
            for s in sec["sources"]:
                lines.append(
                    f"| {s['category']} | {s['tco2e']:,.2f} | {s['percentage']:.1f}% |"
                )
        return "\n".join(lines)

    def _md_scope2(self, data: Dict[str, Any]) -> str:
        """Render Scope 2 markdown."""
        sec = self._section_scope2_summary(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Approach | tCO2e |\n|----------|------:|\n"
            f"| Location-based | {sec['location_based_tco2e']:,.2f} |\n"
            f"| Market-based | {sec['market_based_tco2e']:,.2f} |\n\n"
            f"Renewable electricity: {sec['renewable_percentage']:.1f}%"
        )

    def _md_scope3(self, data: Dict[str, Any]) -> str:
        """Render Scope 3 markdown."""
        sec = self._section_scope3_summary(data)
        lines = [f"## {sec['title']}\n", f"**Total:** {sec['total_tco2e']:,.2f} tCO2e\n"]
        if sec["categories"]:
            lines.append("| Cat | Name | tCO2e | Method |")
            lines.append("|----:|------|------:|--------|")
            for c in sec["categories"]:
                lines.append(
                    f"| {c['category_number']} | {c['name']} "
                    f"| {c['tco2e']:,.2f} | {c['methodology']} |"
                )
        return "\n".join(lines)

    def _md_total(self, data: Dict[str, Any]) -> str:
        """Render total emissions markdown."""
        sec = self._section_total_emissions(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Scope | tCO2e |\n|-------|------:|\n"
            f"| Scope 1 | {sec['scope1_tco2e']:,.2f} |\n"
            f"| Scope 2 (Location) | {sec['scope2_location_tco2e']:,.2f} |\n"
            f"| Scope 2 (Market) | {sec['scope2_market_tco2e']:,.2f} |\n"
            f"| Scope 3 | {sec['scope3_tco2e']:,.2f} |\n"
            f"| **Total (Location)** | **{sec['total_location_tco2e']:,.2f}** |\n"
            f"| **Total (Market)** | **{sec['total_market_tco2e']:,.2f}** |"
        )

    def _md_gas_disaggregation(self, data: Dict[str, Any]) -> str:
        """Render gas disaggregation markdown."""
        sec = self._section_gas_disaggregation(data)
        lines = ["## GHG Gas Disaggregation\n", "| Gas | tCO2e |", "|-----|------:|"]
        for gas, val in sec["gases"].items():
            lines.append(f"| {gas} | {val:,.2f} |")
        lines.append(f"\nBiogenic CO2: {sec['biogenic_co2']:,.2f} tCO2e")
        return "\n".join(lines)

    def _md_intensity(self, data: Dict[str, Any]) -> str:
        """Render intensity metrics markdown."""
        sec = self._section_intensity_metrics(data)
        lines = [
            "## GHG Intensity Metrics\n",
            "| Metric | Value | Unit |",
            "|--------|------:|------|",
        ]
        for m in sec["metrics"]:
            lines.append(f"| {m['name']} | {m['value']:,.4f} | {m['unit']} |")
        return "\n".join(lines)

    def _md_base_year(self, data: Dict[str, Any]) -> str:
        """Render base year comparison markdown."""
        sec = self._section_base_year_comparison(data)
        direction = "increase" if sec["change_percentage"] > 0 else "decrease"
        return (
            f"## Base Year Comparison\n\n"
            f"**Base Year:** {sec['base_year']}  \n"
            f"**Base Year Emissions:** {sec['base_year_tco2e']:,.2f} tCO2e  \n"
            f"**Current Year Emissions:** {sec['current_year_tco2e']:,.2f} tCO2e  \n"
            f"**Change:** {abs(sec['change_percentage']):.1f}% {direction}"
        )

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render methodology notes markdown."""
        sec = self._section_methodology_notes(data)
        return (
            f"## Methodology Notes\n\n"
            f"- **Consolidation:** {sec['consolidation_approach']}\n"
            f"- **GWP Source:** {sec['gwp_source']}\n"
            f"- **Assurance Level:** {sec['assurance_level']}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-016 ESRS E1 Climate Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:900px;margin:auto}"
            "h1{color:#1a5632;border-bottom:2px solid #1a5632;padding-bottom:.3em}"
            "h2{color:#2d7a4f;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#f0f7f3}"
            ".total{font-weight:bold;background:#e8f5e9}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"<h1>GHG Emissions Report - ESRS E1-6</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_scope1(self, data: Dict[str, Any]) -> str:
        """Render Scope 1 HTML."""
        sec = self._section_scope1_summary(data)
        rows = "".join(
            f"<tr><td>{s['category']}</td><td>{s['tco2e']:,.2f}</td></tr>"
            for s in sec["sources"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_tco2e']:,.2f} tCO2e</p>\n"
            f"<table><tr><th>Category</th><th>tCO2e</th></tr>{rows}</table>"
        )

    def _html_scope2(self, data: Dict[str, Any]) -> str:
        """Render Scope 2 HTML."""
        sec = self._section_scope2_summary(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Approach</th><th>tCO2e</th></tr>"
            f"<tr><td>Location-based</td><td>{sec['location_based_tco2e']:,.2f}</td></tr>"
            f"<tr><td>Market-based</td><td>{sec['market_based_tco2e']:,.2f}</td></tr></table>"
        )

    def _html_total(self, data: Dict[str, Any]) -> str:
        """Render total emissions HTML."""
        sec = self._section_total_emissions(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Scope</th><th>tCO2e</th></tr>"
            f"<tr><td>Scope 1</td><td>{sec['scope1_tco2e']:,.2f}</td></tr>"
            f"<tr><td>Scope 2 (Location)</td>"
            f"<td>{sec['scope2_location_tco2e']:,.2f}</td></tr>"
            f"<tr><td>Scope 3</td><td>{sec['scope3_tco2e']:,.2f}</td></tr>"
            f"<tr class='total'><td>Total (Location)</td>"
            f"<td>{sec['total_location_tco2e']:,.2f}</td></tr></table>"
        )

    def _html_gas_disaggregation(self, data: Dict[str, Any]) -> str:
        """Render gas disaggregation HTML."""
        sec = self._section_gas_disaggregation(data)
        rows = "".join(
            f"<tr><td>{gas}</td><td>{val:,.2f}</td></tr>"
            for gas, val in sec["gases"].items()
        )
        return (
            f"<h2>Gas Disaggregation</h2>\n"
            f"<table><tr><th>Gas</th><th>tCO2e</th></tr>{rows}</table>"
        )
