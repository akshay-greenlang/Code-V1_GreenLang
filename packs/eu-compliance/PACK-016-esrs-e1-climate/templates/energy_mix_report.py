# -*- coding: utf-8 -*-
"""
EnergyMixReportTemplate - ESRS E1-5 Energy Consumption and Mix Report

Renders total energy consumption, fossil and renewable breakdowns,
energy intensity metrics, and renewable share progress tracking
per ESRS E1-5.

Sections:
    1. Total Consumption
    2. Fossil Breakdown
    3. Renewable Breakdown
    4. Energy Intensity
    5. Renewable Share Progress

Author: GreenLang Team
Version: 16.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_SECTIONS: List[str] = [
    "total_consumption",
    "fossil_breakdown",
    "renewable_breakdown",
    "energy_intensity",
    "renewable_share_progress",
]

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

class EnergyMixReportTemplate:
    """
    Energy consumption and mix report template per ESRS E1-5.

    Renders total energy consumption from fossil and renewable sources,
    self-generated and purchased energy, intensity ratios, and progress
    toward renewable energy targets as required by ESRS E1-5.

    Example:
        >>> tpl = EnergyMixReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnergyMixReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = utcnow()
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
        if "total_energy_mwh" not in data:
            warnings.append("total_energy_mwh missing; will default to 0")
        if "fossil_sources" not in data:
            warnings.append("fossil_sources missing; will default to empty")
        if "renewable_sources" not in data:
            warnings.append("renewable_sources missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render energy mix report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_total_consumption(data),
            self._md_fossil(data),
            self._md_renewable(data),
            self._md_intensity(data),
            self._md_renewable_progress(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render energy mix report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_total_consumption(data),
            self._html_fossil(data),
            self._html_renewable(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Energy Mix Report - ESRS E1-5</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render energy mix report as JSON."""
        self.generated_at = utcnow()
        result = {
            "template": "energy_mix_report",
            "esrs_reference": "E1-5",
            "version": "16.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "total_energy_mwh": data.get("total_energy_mwh", 0.0),
            "fossil_energy_mwh": data.get("fossil_energy_mwh", 0.0),
            "renewable_energy_mwh": data.get("renewable_energy_mwh", 0.0),
            "renewable_share_pct": data.get("renewable_share_pct", 0.0),
            "energy_intensity": data.get("energy_intensity", []),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_total_consumption(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build total consumption section."""
        total = data.get("total_energy_mwh", 0.0)
        fossil = data.get("fossil_energy_mwh", 0.0)
        renewable = data.get("renewable_energy_mwh", 0.0)
        nuclear = data.get("nuclear_energy_mwh", 0.0)
        return {
            "title": "Total Energy Consumption",
            "total_mwh": round(total, 2),
            "fossil_mwh": round(fossil, 2),
            "renewable_mwh": round(renewable, 2),
            "nuclear_mwh": round(nuclear, 2),
            "fossil_share_pct": round(fossil / total * 100, 1) if total > 0 else 0.0,
            "renewable_share_pct": round(renewable / total * 100, 1) if total > 0 else 0.0,
            "self_generated_mwh": round(data.get("self_generated_mwh", 0.0), 2),
            "purchased_mwh": round(data.get("purchased_mwh", 0.0), 2),
        }

    def _section_fossil_breakdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build fossil energy breakdown section."""
        sources = data.get("fossil_sources", [])
        total_fossil = sum(s.get("mwh", 0.0) for s in sources)
        return {
            "title": "Fossil Energy Breakdown",
            "total_fossil_mwh": round(total_fossil, 2),
            "sources": [
                {
                    "fuel_type": s.get("fuel_type", ""),
                    "mwh": round(s.get("mwh", 0.0), 2),
                    "percentage": (
                        round(s.get("mwh", 0.0) / total_fossil * 100, 1)
                        if total_fossil > 0 else 0.0
                    ),
                    "scope": s.get("scope", ""),
                }
                for s in sources
            ],
            "coal_mwh": round(data.get("coal_mwh", 0.0), 2),
            "natural_gas_mwh": round(data.get("natural_gas_mwh", 0.0), 2),
            "oil_mwh": round(data.get("oil_mwh", 0.0), 2),
        }

    def _section_renewable_breakdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build renewable energy breakdown section."""
        sources = data.get("renewable_sources", [])
        total_renewable = sum(s.get("mwh", 0.0) for s in sources)
        return {
            "title": "Renewable Energy Breakdown",
            "total_renewable_mwh": round(total_renewable, 2),
            "sources": [
                {
                    "source_type": s.get("source_type", ""),
                    "mwh": round(s.get("mwh", 0.0), 2),
                    "percentage": (
                        round(s.get("mwh", 0.0) / total_renewable * 100, 1)
                        if total_renewable > 0 else 0.0
                    ),
                    "procurement_method": s.get("procurement_method", ""),
                }
                for s in sources
            ],
            "solar_mwh": round(data.get("solar_mwh", 0.0), 2),
            "wind_mwh": round(data.get("wind_mwh", 0.0), 2),
            "hydro_mwh": round(data.get("hydro_mwh", 0.0), 2),
            "biomass_mwh": round(data.get("biomass_mwh", 0.0), 2),
            "geothermal_mwh": round(data.get("geothermal_mwh", 0.0), 2),
        }

    def _section_energy_intensity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build energy intensity section."""
        metrics = data.get("energy_intensity", [])
        return {
            "title": "Energy Intensity Metrics",
            "metrics": [
                {
                    "name": m.get("name", ""),
                    "value": round(m.get("value", 0.0), 4),
                    "unit": m.get("unit", ""),
                    "denominator": m.get("denominator", ""),
                    "trend": m.get("trend", ""),
                }
                for m in metrics
            ],
        }

    def _section_renewable_share_progress(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build renewable share progress section."""
        target = data.get("renewable_target_pct", 0.0)
        current = data.get("renewable_share_pct", 0.0)
        gap = round(target - current, 1) if target > 0 else 0.0
        return {
            "title": "Renewable Share Progress",
            "current_share_pct": round(current, 1),
            "target_share_pct": round(target, 1),
            "gap_pct": gap,
            "target_year": data.get("renewable_target_year", ""),
            "on_track": current >= target if target > 0 else True,
            "annual_milestones": data.get("renewable_milestones", []),
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
            f"# Energy Consumption & Mix Report - ESRS E1-5\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E1-5 Energy Consumption and Mix"
        )

    def _md_total_consumption(self, data: Dict[str, Any]) -> str:
        """Render total consumption markdown."""
        sec = self._section_total_consumption(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Category | MWh | Share |\n|----------|----:|------:|\n"
            f"| Fossil | {sec['fossil_mwh']:,.2f} | {sec['fossil_share_pct']:.1f}% |\n"
            f"| Renewable | {sec['renewable_mwh']:,.2f} | {sec['renewable_share_pct']:.1f}% |\n"
            f"| Nuclear | {sec['nuclear_mwh']:,.2f} | - |\n"
            f"| **Total** | **{sec['total_mwh']:,.2f}** | **100%** |"
        )

    def _md_fossil(self, data: Dict[str, Any]) -> str:
        """Render fossil breakdown markdown."""
        sec = self._section_fossil_breakdown(data)
        lines = [f"## {sec['title']}\n", "| Fuel Type | MWh | % |", "|-----------|----:|--:|"]
        for s in sec["sources"]:
            lines.append(f"| {s['fuel_type']} | {s['mwh']:,.2f} | {s['percentage']:.1f}% |")
        return "\n".join(lines)

    def _md_renewable(self, data: Dict[str, Any]) -> str:
        """Render renewable breakdown markdown."""
        sec = self._section_renewable_breakdown(data)
        lines = [f"## {sec['title']}\n", "| Source | MWh | % |", "|--------|----:|--:|"]
        for s in sec["sources"]:
            lines.append(
                f"| {s['source_type']} | {s['mwh']:,.2f} | {s['percentage']:.1f}% |"
            )
        return "\n".join(lines)

    def _md_intensity(self, data: Dict[str, Any]) -> str:
        """Render energy intensity markdown."""
        sec = self._section_energy_intensity(data)
        lines = [
            "## Energy Intensity Metrics\n",
            "| Metric | Value | Unit |",
            "|--------|------:|------|",
        ]
        for m in sec["metrics"]:
            lines.append(f"| {m['name']} | {m['value']:,.4f} | {m['unit']} |")
        return "\n".join(lines)

    def _md_renewable_progress(self, data: Dict[str, Any]) -> str:
        """Render renewable share progress markdown."""
        sec = self._section_renewable_share_progress(data)
        status = "On Track" if sec["on_track"] else "Behind Target"
        return (
            f"## Renewable Share Progress\n\n"
            f"**Current Share:** {sec['current_share_pct']:.1f}%  \n"
            f"**Target:** {sec['target_share_pct']:.1f}% by {sec['target_year']}  \n"
            f"**Gap:** {sec['gap_pct']:.1f}pp  \n"
            f"**Status:** {status}"
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
            f"<h1>Energy Consumption & Mix - ESRS E1-5</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_total_consumption(self, data: Dict[str, Any]) -> str:
        """Render total consumption HTML."""
        sec = self._section_total_consumption(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Category</th><th>MWh</th><th>Share</th></tr>"
            f"<tr><td>Fossil</td><td>{sec['fossil_mwh']:,.2f}</td>"
            f"<td>{sec['fossil_share_pct']:.1f}%</td></tr>"
            f"<tr><td>Renewable</td><td>{sec['renewable_mwh']:,.2f}</td>"
            f"<td>{sec['renewable_share_pct']:.1f}%</td></tr>"
            f"<tr class='total'><td>Total</td><td>{sec['total_mwh']:,.2f}</td>"
            f"<td>100%</td></tr></table>"
        )

    def _html_fossil(self, data: Dict[str, Any]) -> str:
        """Render fossil breakdown HTML."""
        sec = self._section_fossil_breakdown(data)
        rows = "".join(
            f"<tr><td>{s['fuel_type']}</td><td>{s['mwh']:,.2f}</td></tr>"
            for s in sec["sources"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Fuel Type</th><th>MWh</th></tr>{rows}</table>"
        )

    def _html_renewable(self, data: Dict[str, Any]) -> str:
        """Render renewable breakdown HTML."""
        sec = self._section_renewable_breakdown(data)
        rows = "".join(
            f"<tr><td>{s['source_type']}</td><td>{s['mwh']:,.2f}</td></tr>"
            for s in sec["sources"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Source</th><th>MWh</th></tr>{rows}</table>"
        )
