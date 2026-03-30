# -*- coding: utf-8 -*-
"""
E3WaterReportTemplate - ESRS E3 Water and Marine Resources Report

Renders water policies, actions, targets, withdrawal, discharge,
consumption, water stress exposure, recycling metrics, marine impacts,
and financial effects per ESRS E3.

Sections:
    1. Water Policies
    2. Water Actions
    3. Water Targets
    4. Water Withdrawal
    5. Water Discharge
    6. Water Consumption
    7. Water Stress Exposure
    8. Recycling Metrics
    9. Marine Impacts
    10. Financial Effects

Author: GreenLang Team
Version: 17.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_SECTIONS: List[str] = [
    "water_policies",
    "water_actions",
    "water_targets",
    "water_withdrawal",
    "water_discharge",
    "water_consumption",
    "water_stress_exposure",
    "recycling_metrics",
    "marine_impacts",
    "financial_effects",
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

class E3WaterReportTemplate:
    """
    ESRS E3 Water and Marine Resources report template.

    Renders water management policies, withdrawal/discharge/consumption
    volumes by source, water stress area exposure, recycling and reuse
    metrics, marine resource impacts, and financial effects per ESRS E3.

    Example:
        >>> tpl = E3WaterReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize E3WaterReportTemplate."""
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
        if "water_withdrawal" not in data:
            warnings.append("water_withdrawal missing; will default to empty")
        if "water_discharge" not in data:
            warnings.append("water_discharge missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render E3 Water report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_water_policies(data),
            self._md_water_actions(data),
            self._md_water_targets(data),
            self._md_water_withdrawal(data),
            self._md_water_discharge(data),
            self._md_water_consumption(data),
            self._md_water_stress_exposure(data),
            self._md_recycling_metrics(data),
            self._md_marine_impacts(data),
            self._md_financial_effects(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render E3 Water report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_water_withdrawal(data),
            self._html_water_discharge(data),
            self._html_water_consumption(data),
            self._html_water_stress(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>ESRS E3 Water and Marine Resources Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render E3 Water report as JSON string."""
        self.generated_at = utcnow()
        result = {
            "template": "e3_water_report",
            "esrs_reference": "ESRS E3",
            "version": "17.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "total_withdrawal_m3": data.get("total_withdrawal_m3", 0.0),
            "total_discharge_m3": data.get("total_discharge_m3", 0.0),
            "total_consumption_m3": data.get("total_consumption_m3", 0.0),
            "water_stress_sites": len(data.get("water_stress_sites", [])),
            "recycling_rate_pct": data.get("recycling_rate_pct", 0.0),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_water_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build water policies section."""
        policies = data.get("water_policies", [])
        return {
            "title": "Water and Marine Resource Policies",
            "has_policy": len(policies) > 0,
            "policy_count": len(policies),
            "policies": [
                {
                    "name": p.get("name", ""),
                    "scope": p.get("scope", ""),
                    "water_stewardship_commitment": p.get(
                        "water_stewardship_commitment", False
                    ),
                }
                for p in policies
            ],
        }

    def _section_water_actions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build water actions section."""
        actions = data.get("water_actions", [])
        return {
            "title": "Water Management Actions",
            "action_count": len(actions),
            "actions": [
                {
                    "description": a.get("description", ""),
                    "type": a.get("type", ""),
                    "status": a.get("status", ""),
                    "expected_savings_m3": a.get("expected_savings_m3", 0.0),
                    "investment_eur": a.get("investment_eur", 0.0),
                }
                for a in actions
            ],
        }

    def _section_water_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build water targets section."""
        targets = data.get("water_targets", [])
        return {
            "title": "Water Targets",
            "target_count": len(targets),
            "targets": [
                {
                    "name": t.get("name", ""),
                    "metric": t.get("metric", ""),
                    "base_year": t.get("base_year", ""),
                    "target_year": t.get("target_year", ""),
                    "baseline_value": t.get("baseline_value", 0.0),
                    "target_value": t.get("target_value", 0.0),
                    "current_value": t.get("current_value", 0.0),
                    "unit": t.get("unit", "m3"),
                }
                for t in targets
            ],
        }

    def _section_water_withdrawal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build water withdrawal section."""
        sources = data.get("water_withdrawal", [])
        total = sum(s.get("volume_m3", 0.0) for s in sources)
        return {
            "title": "Water Withdrawal",
            "total_m3": round(total, 2),
            "source_count": len(sources),
            "sources": [
                {
                    "source_type": s.get("source_type", ""),
                    "volume_m3": round(s.get("volume_m3", 0.0), 2),
                    "freshwater_m3": round(s.get("freshwater_m3", 0.0), 2),
                    "other_water_m3": round(s.get("other_water_m3", 0.0), 2),
                    "percentage": (
                        round(s.get("volume_m3", 0.0) / total * 100, 1)
                        if total > 0 else 0.0
                    ),
                }
                for s in sources
            ],
        }

    def _section_water_discharge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build water discharge section."""
        destinations = data.get("water_discharge", [])
        total = sum(d.get("volume_m3", 0.0) for d in destinations)
        return {
            "title": "Water Discharge",
            "total_m3": round(total, 2),
            "destination_count": len(destinations),
            "destinations": [
                {
                    "destination_type": d.get("destination_type", ""),
                    "volume_m3": round(d.get("volume_m3", 0.0), 2),
                    "treatment_level": d.get("treatment_level", ""),
                    "freshwater_m3": round(d.get("freshwater_m3", 0.0), 2),
                }
                for d in destinations
            ],
        }

    def _section_water_consumption(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build water consumption section."""
        total_withdrawal = data.get("total_withdrawal_m3", 0.0)
        total_discharge = data.get("total_discharge_m3", 0.0)
        consumption = total_withdrawal - total_discharge
        return {
            "title": "Water Consumption",
            "total_withdrawal_m3": round(total_withdrawal, 2),
            "total_discharge_m3": round(total_discharge, 2),
            "total_consumption_m3": round(consumption, 2),
            "consumption_in_stress_areas_m3": round(
                data.get("consumption_in_stress_areas_m3", 0.0), 2
            ),
            "year_over_year_change_pct": data.get("consumption_yoy_change_pct", 0.0),
        }

    def _section_water_stress_exposure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build water stress exposure section."""
        sites = data.get("water_stress_sites", [])
        return {
            "title": "Water Stress Area Exposure",
            "site_count": len(sites),
            "total_withdrawal_stress_m3": round(
                sum(s.get("withdrawal_m3", 0.0) for s in sites), 2
            ),
            "sites": [
                {
                    "site_name": s.get("site_name", ""),
                    "location": s.get("location", ""),
                    "stress_level": s.get("stress_level", ""),
                    "withdrawal_m3": round(s.get("withdrawal_m3", 0.0), 2),
                    "tool_used": s.get("tool_used", "WRI Aqueduct"),
                }
                for s in sites
            ],
        }

    def _section_recycling_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build water recycling metrics section."""
        return {
            "title": "Water Recycling and Reuse",
            "total_recycled_m3": round(data.get("total_recycled_m3", 0.0), 2),
            "recycling_rate_pct": round(data.get("recycling_rate_pct", 0.0), 1),
            "reuse_volume_m3": round(data.get("reuse_volume_m3", 0.0), 2),
            "rainwater_harvested_m3": round(data.get("rainwater_harvested_m3", 0.0), 2),
            "greywater_reused_m3": round(data.get("greywater_reused_m3", 0.0), 2),
        }

    def _section_marine_impacts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build marine impacts section."""
        impacts = data.get("marine_impacts", [])
        return {
            "title": "Marine Resource Impacts",
            "impact_count": len(impacts),
            "coastal_operations": data.get("coastal_operations", False),
            "impacts": [
                {
                    "description": i.get("description", ""),
                    "type": i.get("type", ""),
                    "severity": i.get("severity", ""),
                    "mitigation": i.get("mitigation", ""),
                }
                for i in impacts
            ],
        }

    def _section_financial_effects(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build financial effects section."""
        return {
            "title": "Financial Effects of Water and Marine Impacts",
            "water_costs_eur": data.get("water_costs_eur", 0.0),
            "treatment_costs_eur": data.get("treatment_costs_eur", 0.0),
            "water_efficiency_capex_eur": data.get("water_efficiency_capex_eur", 0.0),
            "water_risk_exposure_eur": data.get("water_risk_exposure_eur", 0.0),
            "fines_eur": data.get("water_fines_eur", 0.0),
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
            f"# ESRS E3 Water and Marine Resources Report\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E3 Water and Marine Resources"
        )

    def _md_water_policies(self, data: Dict[str, Any]) -> str:
        """Render water policies markdown."""
        sec = self._section_water_policies(data)
        lines = [f"## {sec['title']}\n", f"**Policies:** {sec['policy_count']}\n"]
        for p in sec["policies"]:
            steward = "Yes" if p["water_stewardship_commitment"] else "No"
            lines.append(f"- **{p['name']}** (Scope: {p['scope']}, Stewardship: {steward})")
        return "\n".join(lines)

    def _md_water_actions(self, data: Dict[str, Any]) -> str:
        """Render water actions markdown."""
        sec = self._section_water_actions(data)
        lines = [f"## {sec['title']}\n"]
        if sec["actions"]:
            lines.append("| Action | Type | Status | Savings (m3) | Investment (EUR) |")
            lines.append("|--------|------|--------|------------:|-----------------:|")
            for a in sec["actions"]:
                lines.append(
                    f"| {a['description']} | {a['type']} | {a['status']} "
                    f"| {a['expected_savings_m3']:,.2f} | {a['investment_eur']:,.2f} |"
                )
        return "\n".join(lines)

    def _md_water_targets(self, data: Dict[str, Any]) -> str:
        """Render water targets markdown."""
        sec = self._section_water_targets(data)
        lines = [f"## {sec['title']}\n"]
        if sec["targets"]:
            lines.append("| Target | Metric | Baseline | Current | Goal | Unit |")
            lines.append("|--------|--------|--------:|--------:|-----:|------|")
            for t in sec["targets"]:
                lines.append(
                    f"| {t['name']} | {t['metric']} | {t['baseline_value']:,.2f} "
                    f"| {t['current_value']:,.2f} | {t['target_value']:,.2f} | {t['unit']} |"
                )
        return "\n".join(lines)

    def _md_water_withdrawal(self, data: Dict[str, Any]) -> str:
        """Render water withdrawal markdown."""
        sec = self._section_water_withdrawal(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Withdrawal:** {sec['total_m3']:,.2f} m3\n",
        ]
        if sec["sources"]:
            lines.append("| Source | Volume (m3) | Freshwater (m3) | % |")
            lines.append("|--------|----------:|--------------:|--:|")
            for s in sec["sources"]:
                lines.append(
                    f"| {s['source_type']} | {s['volume_m3']:,.2f} "
                    f"| {s['freshwater_m3']:,.2f} | {s['percentage']:.1f}% |"
                )
        return "\n".join(lines)

    def _md_water_discharge(self, data: Dict[str, Any]) -> str:
        """Render water discharge markdown."""
        sec = self._section_water_discharge(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Discharge:** {sec['total_m3']:,.2f} m3\n",
        ]
        if sec["destinations"]:
            lines.append("| Destination | Volume (m3) | Treatment |")
            lines.append("|-------------|----------:|-----------|")
            for d in sec["destinations"]:
                lines.append(
                    f"| {d['destination_type']} | {d['volume_m3']:,.2f} "
                    f"| {d['treatment_level']} |"
                )
        return "\n".join(lines)

    def _md_water_consumption(self, data: Dict[str, Any]) -> str:
        """Render water consumption markdown."""
        sec = self._section_water_consumption(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Metric | m3 |\n|--------|---:|\n"
            f"| Total Withdrawal | {sec['total_withdrawal_m3']:,.2f} |\n"
            f"| Total Discharge | {sec['total_discharge_m3']:,.2f} |\n"
            f"| **Net Consumption** | **{sec['total_consumption_m3']:,.2f}** |\n"
            f"| In Stress Areas | {sec['consumption_in_stress_areas_m3']:,.2f} |"
        )

    def _md_water_stress_exposure(self, data: Dict[str, Any]) -> str:
        """Render water stress exposure markdown."""
        sec = self._section_water_stress_exposure(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Sites in Stress Areas:** {sec['site_count']}  ",
            f"**Total Withdrawal from Stress Areas:** "
            f"{sec['total_withdrawal_stress_m3']:,.2f} m3\n",
        ]
        if sec["sites"]:
            lines.append("| Site | Location | Stress Level | Withdrawal (m3) |")
            lines.append("|------|----------|-------------|---------------:|")
            for s in sec["sites"]:
                lines.append(
                    f"| {s['site_name']} | {s['location']} "
                    f"| {s['stress_level']} | {s['withdrawal_m3']:,.2f} |"
                )
        return "\n".join(lines)

    def _md_recycling_metrics(self, data: Dict[str, Any]) -> str:
        """Render recycling metrics markdown."""
        sec = self._section_recycling_metrics(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Metric | Value |\n|--------|------:|\n"
            f"| Recycled Volume | {sec['total_recycled_m3']:,.2f} m3 |\n"
            f"| Recycling Rate | {sec['recycling_rate_pct']:.1f}% |\n"
            f"| Reuse Volume | {sec['reuse_volume_m3']:,.2f} m3 |\n"
            f"| Rainwater Harvested | {sec['rainwater_harvested_m3']:,.2f} m3 |\n"
            f"| Greywater Reused | {sec['greywater_reused_m3']:,.2f} m3 |"
        )

    def _md_marine_impacts(self, data: Dict[str, Any]) -> str:
        """Render marine impacts markdown."""
        sec = self._section_marine_impacts(data)
        coastal = "Yes" if sec["coastal_operations"] else "No"
        lines = [
            f"## {sec['title']}\n",
            f"**Coastal Operations:** {coastal}  ",
            f"**Impacts Identified:** {sec['impact_count']}\n",
        ]
        for i in sec["impacts"]:
            lines.append(
                f"- **{i['type']}** ({i['severity']}): {i['description']} "
                f"| Mitigation: {i['mitigation']}"
            )
        return "\n".join(lines)

    def _md_financial_effects(self, data: Dict[str, Any]) -> str:
        """Render financial effects markdown."""
        sec = self._section_financial_effects(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Item | EUR |\n|------|----:|\n"
            f"| Water Costs | {sec['water_costs_eur']:,.2f} |\n"
            f"| Treatment Costs | {sec['treatment_costs_eur']:,.2f} |\n"
            f"| Efficiency CapEx | {sec['water_efficiency_capex_eur']:,.2f} |\n"
            f"| Risk Exposure | {sec['water_risk_exposure_eur']:,.2f} |\n"
            f"| Fines | {sec['fines_eur']:,.2f} |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-017 ESRS Full Coverage Pack on {ts}*"

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
            f"<h1>ESRS E3 Water and Marine Resources Report</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_water_withdrawal(self, data: Dict[str, Any]) -> str:
        """Render water withdrawal HTML."""
        sec = self._section_water_withdrawal(data)
        rows = "".join(
            f"<tr><td>{s['source_type']}</td><td>{s['volume_m3']:,.2f}</td></tr>"
            for s in sec["sources"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_m3']:,.2f} m3</p>\n"
            f"<table><tr><th>Source</th><th>Volume (m3)</th></tr>{rows}</table>"
        )

    def _html_water_discharge(self, data: Dict[str, Any]) -> str:
        """Render water discharge HTML."""
        sec = self._section_water_discharge(data)
        rows = "".join(
            f"<tr><td>{d['destination_type']}</td><td>{d['volume_m3']:,.2f}</td>"
            f"<td>{d['treatment_level']}</td></tr>"
            for d in sec["destinations"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_m3']:,.2f} m3</p>\n"
            f"<table><tr><th>Destination</th><th>Volume (m3)</th><th>Treatment</th></tr>"
            f"{rows}</table>"
        )

    def _html_water_consumption(self, data: Dict[str, Any]) -> str:
        """Render water consumption HTML."""
        sec = self._section_water_consumption(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Metric</th><th>m3</th></tr>"
            f"<tr><td>Withdrawal</td><td>{sec['total_withdrawal_m3']:,.2f}</td></tr>"
            f"<tr><td>Discharge</td><td>{sec['total_discharge_m3']:,.2f}</td></tr>"
            f"<tr class='total'><td>Consumption</td>"
            f"<td>{sec['total_consumption_m3']:,.2f}</td></tr></table>"
        )

    def _html_water_stress(self, data: Dict[str, Any]) -> str:
        """Render water stress HTML."""
        sec = self._section_water_stress_exposure(data)
        rows = "".join(
            f"<tr><td>{s['site_name']}</td><td>{s['stress_level']}</td>"
            f"<td>{s['withdrawal_m3']:,.2f}</td></tr>"
            for s in sec["sites"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Sites: {sec['site_count']}</p>\n"
            f"<table><tr><th>Site</th><th>Stress</th><th>Withdrawal (m3)</th></tr>"
            f"{rows}</table>"
        )
