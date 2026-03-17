# -*- coding: utf-8 -*-
"""
CarbonPricingReportTemplate - ESRS E1-8 Carbon Pricing Disclosure Report

Renders carbon pricing mechanism overview, coverage summary, internal
shadow pricing, and scenario analysis per ESRS E1-8.

Sections:
    1. Mechanism Overview
    2. Coverage Summary
    3. Shadow Pricing
    4. Scenario Analysis

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
    "mechanism_overview",
    "coverage_summary",
    "shadow_pricing",
    "scenario_analysis",
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


class CarbonPricingReportTemplate:
    """
    Carbon pricing disclosure report template per ESRS E1-8.

    Renders exposure to carbon pricing mechanisms (ETS, carbon taxes),
    coverage of emissions under pricing schemes, internal shadow pricing
    for investment decisions, and sensitivity/scenario analysis.

    Example:
        >>> tpl = CarbonPricingReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CarbonPricingReportTemplate."""
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
        if "carbon_pricing_mechanisms" not in data:
            warnings.append("carbon_pricing_mechanisms missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render carbon pricing report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_mechanism_overview(data),
            self._md_coverage(data),
            self._md_shadow_pricing(data),
            self._md_scenario(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render carbon pricing report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_mechanism_overview(data),
            self._html_coverage(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Carbon Pricing Report - ESRS E1-8</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render carbon pricing report as JSON."""
        self.generated_at = _utcnow()
        mechanisms = data.get("carbon_pricing_mechanisms", [])
        result = {
            "template": "carbon_pricing_report",
            "esrs_reference": "E1-8",
            "version": "16.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "mechanism_count": len(mechanisms),
            "total_cost_eur": round(
                sum(m.get("annual_cost_eur", 0.0) for m in mechanisms), 2
            ),
            "emissions_covered_tco2e": round(
                sum(m.get("covered_tco2e", 0.0) for m in mechanisms), 2
            ),
            "uses_shadow_pricing": data.get("uses_shadow_pricing", False),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_mechanism_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build mechanism overview section."""
        mechanisms = data.get("carbon_pricing_mechanisms", [])
        total_cost = sum(m.get("annual_cost_eur", 0.0) for m in mechanisms)
        return {
            "title": "Carbon Pricing Mechanism Overview",
            "mechanism_count": len(mechanisms),
            "total_annual_cost_eur": round(total_cost, 2),
            "mechanisms": [
                {
                    "name": m.get("name", ""),
                    "type": m.get("type", ""),
                    "jurisdiction": m.get("jurisdiction", ""),
                    "price_per_tco2e_eur": round(m.get("price_per_tco2e_eur", 0.0), 2),
                    "annual_cost_eur": round(m.get("annual_cost_eur", 0.0), 2),
                    "free_allowances_tco2e": round(
                        m.get("free_allowances_tco2e", 0.0), 2
                    ),
                }
                for m in mechanisms
            ],
        }

    def _section_coverage_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build coverage summary section."""
        mechanisms = data.get("carbon_pricing_mechanisms", [])
        total_covered = sum(m.get("covered_tco2e", 0.0) for m in mechanisms)
        total_emissions = data.get("total_ghg_tco2e", 0.0)
        return {
            "title": "Emissions Coverage Summary",
            "total_covered_tco2e": round(total_covered, 2),
            "total_emissions_tco2e": round(total_emissions, 2),
            "coverage_percentage": (
                round(total_covered / total_emissions * 100, 1)
                if total_emissions > 0 else 0.0
            ),
            "by_mechanism": [
                {
                    "name": m.get("name", ""),
                    "covered_tco2e": round(m.get("covered_tco2e", 0.0), 2),
                    "scope_coverage": m.get("scope_coverage", ""),
                }
                for m in mechanisms
            ],
        }

    def _section_shadow_pricing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build shadow pricing section."""
        shadow = data.get("shadow_pricing", {})
        return {
            "title": "Internal Carbon Shadow Pricing",
            "uses_shadow_pricing": shadow.get("enabled", False),
            "price_per_tco2e_eur": round(shadow.get("price_per_tco2e_eur", 0.0), 2),
            "application_scope": shadow.get("application_scope", ""),
            "decision_types": shadow.get("decision_types", []),
            "price_trajectory": shadow.get("price_trajectory", []),
            "methodology": shadow.get("methodology", ""),
            "last_review_date": shadow.get("last_review_date", ""),
        }

    def _section_scenario_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build scenario analysis section."""
        scenarios = data.get("pricing_scenarios", [])
        return {
            "title": "Carbon Price Scenario Analysis",
            "scenarios": [
                {
                    "name": s.get("name", ""),
                    "price_per_tco2e_eur": round(s.get("price_per_tco2e_eur", 0.0), 2),
                    "year": s.get("year", ""),
                    "financial_impact_eur": round(s.get("financial_impact_eur", 0.0), 2),
                    "impact_on_ebitda_pct": round(
                        s.get("impact_on_ebitda_pct", 0.0), 2
                    ),
                }
                for s in scenarios
            ],
            "scenario_count": len(scenarios),
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
            f"# Carbon Pricing Report - ESRS E1-8\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E1-8 Internal Carbon Pricing"
        )

    def _md_mechanism_overview(self, data: Dict[str, Any]) -> str:
        """Render mechanism overview markdown."""
        sec = self._section_mechanism_overview(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Annual Cost:** EUR {sec['total_annual_cost_eur']:,.2f}\n",
            "| Mechanism | Type | Jurisdiction | Price/tCO2e | Annual Cost |",
            "|-----------|------|-------------|------------:|------------|",
        ]
        for m in sec["mechanisms"]:
            lines.append(
                f"| {m['name']} | {m['type']} | {m['jurisdiction']} "
                f"| EUR {m['price_per_tco2e_eur']:,.2f} | EUR {m['annual_cost_eur']:,.2f} |"
            )
        return "\n".join(lines)

    def _md_coverage(self, data: Dict[str, Any]) -> str:
        """Render coverage summary markdown."""
        sec = self._section_coverage_summary(data)
        return (
            f"## {sec['title']}\n\n"
            f"**Covered Emissions:** {sec['total_covered_tco2e']:,.2f} tCO2e  \n"
            f"**Total Emissions:** {sec['total_emissions_tco2e']:,.2f} tCO2e  \n"
            f"**Coverage:** {sec['coverage_percentage']:.1f}%"
        )

    def _md_shadow_pricing(self, data: Dict[str, Any]) -> str:
        """Render shadow pricing markdown."""
        sec = self._section_shadow_pricing(data)
        enabled = "Yes" if sec["uses_shadow_pricing"] else "No"
        decisions = ", ".join(sec["decision_types"]) if sec["decision_types"] else "N/A"
        return (
            f"## {sec['title']}\n\n"
            f"**Active:** {enabled}  \n"
            f"**Price:** EUR {sec['price_per_tco2e_eur']:,.2f}/tCO2e  \n"
            f"**Scope:** {sec['application_scope']}  \n"
            f"**Decision Types:** {decisions}"
        )

    def _md_scenario(self, data: Dict[str, Any]) -> str:
        """Render scenario analysis markdown."""
        sec = self._section_scenario_analysis(data)
        lines = [
            f"## {sec['title']}\n",
            "| Scenario | Year | Price/tCO2e | Financial Impact | EBITDA Impact |",
            "|----------|-----:|------------|----------------:|--------------:|",
        ]
        for s in sec["scenarios"]:
            lines.append(
                f"| {s['name']} | {s['year']} "
                f"| EUR {s['price_per_tco2e_eur']:,.2f} "
                f"| EUR {s['financial_impact_eur']:,.2f} "
                f"| {s['impact_on_ebitda_pct']:.2f}% |"
            )
        return "\n".join(lines)

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
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"<h1>Carbon Pricing Report - ESRS E1-8</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_mechanism_overview(self, data: Dict[str, Any]) -> str:
        """Render mechanism overview HTML."""
        sec = self._section_mechanism_overview(data)
        rows = "".join(
            f"<tr><td>{m['name']}</td><td>{m['type']}</td>"
            f"<td>EUR {m['price_per_tco2e_eur']:,.2f}</td></tr>"
            for m in sec["mechanisms"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Mechanism</th><th>Type</th><th>Price</th></tr>"
            f"{rows}</table>"
        )

    def _html_coverage(self, data: Dict[str, Any]) -> str:
        """Render coverage summary HTML."""
        sec = self._section_coverage_summary(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Coverage: {sec['coverage_percentage']:.1f}% "
            f"({sec['total_covered_tco2e']:,.2f} tCO2e)</p>"
        )
