# -*- coding: utf-8 -*-
"""
E2PollutionReportTemplate - ESRS E2 Pollution Disclosure Report

Renders pollution policies, actions, targets, air emissions, water
discharges, soil contamination, substances of concern, SVHC assessment,
financial effects, and methodology notes per ESRS E2.

Sections:
    1. Pollution Policies
    2. Pollution Actions
    3. Pollution Targets
    4. Air Emissions
    5. Water Discharges
    6. Soil Contamination
    7. Substances of Concern
    8. SVHC Assessment
    9. Financial Effects
    10. Methodology Notes

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
    "pollution_policies",
    "pollution_actions",
    "pollution_targets",
    "air_emissions",
    "water_discharges",
    "soil_contamination",
    "substances_of_concern",
    "svhc_assessment",
    "financial_effects",
    "methodology_notes",
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

class E2PollutionReportTemplate:
    """
    ESRS E2 Pollution disclosure report template.

    Renders pollution-related policies, mitigation actions, targets,
    air/water/soil pollution metrics, substances of concern, SVHC
    assessment, and financial effects per ESRS E2.

    Example:
        >>> tpl = E2PollutionReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize E2PollutionReportTemplate."""
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
        if "air_pollutants" not in data:
            warnings.append("air_pollutants missing; will default to empty")
        if "water_pollutants" not in data:
            warnings.append("water_pollutants missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render E2 Pollution report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_pollution_policies(data),
            self._md_pollution_actions(data),
            self._md_pollution_targets(data),
            self._md_air_emissions(data),
            self._md_water_discharges(data),
            self._md_soil_contamination(data),
            self._md_substances_of_concern(data),
            self._md_svhc_assessment(data),
            self._md_financial_effects(data),
            self._md_methodology_notes(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render E2 Pollution report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_air_emissions(data),
            self._html_water_discharges(data),
            self._html_substances_of_concern(data),
            self._html_financial_effects(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>ESRS E2 Pollution Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render E2 Pollution report as JSON string."""
        self.generated_at = utcnow()
        result = {
            "template": "e2_pollution_report",
            "esrs_reference": "ESRS E2",
            "version": "17.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "air_pollutants": data.get("air_pollutants", []),
            "water_pollutants": data.get("water_pollutants", []),
            "soil_pollutants": data.get("soil_pollutants", []),
            "substances_of_concern_count": len(data.get("substances_of_concern", [])),
            "svhc_count": len(data.get("svhc_substances", [])),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_pollution_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build pollution policies section."""
        policies = data.get("pollution_policies", [])
        return {
            "title": "Pollution Policies",
            "has_policy": len(policies) > 0,
            "policy_count": len(policies),
            "policies": [
                {
                    "name": p.get("name", ""),
                    "scope": p.get("scope", ""),
                    "alignment": p.get("alignment", ""),
                    "approval_date": p.get("approval_date", ""),
                }
                for p in policies
            ],
        }

    def _section_pollution_actions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build pollution actions section."""
        actions = data.get("pollution_actions", [])
        total_investment = sum(a.get("investment_eur", 0.0) for a in actions)
        return {
            "title": "Pollution Actions",
            "action_count": len(actions),
            "total_investment_eur": round(total_investment, 2),
            "actions": [
                {
                    "description": a.get("description", ""),
                    "type": a.get("type", ""),
                    "status": a.get("status", ""),
                    "investment_eur": a.get("investment_eur", 0.0),
                    "expected_outcome": a.get("expected_outcome", ""),
                }
                for a in actions
            ],
        }

    def _section_pollution_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build pollution targets section."""
        targets = data.get("pollution_targets", [])
        return {
            "title": "Pollution Targets",
            "target_count": len(targets),
            "targets": [
                {
                    "name": t.get("name", ""),
                    "pollutant": t.get("pollutant", ""),
                    "base_year": t.get("base_year", ""),
                    "target_year": t.get("target_year", ""),
                    "baseline_value": t.get("baseline_value", 0.0),
                    "target_value": t.get("target_value", 0.0),
                    "current_value": t.get("current_value", 0.0),
                    "unit": t.get("unit", ""),
                }
                for t in targets
            ],
        }

    def _section_air_emissions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build air emissions section."""
        pollutants = data.get("air_pollutants", [])
        total = sum(p.get("quantity_tonnes", 0.0) for p in pollutants)
        return {
            "title": "Air Emissions",
            "total_tonnes": round(total, 4),
            "pollutant_count": len(pollutants),
            "pollutants": [
                {
                    "name": p.get("name", ""),
                    "quantity_tonnes": round(p.get("quantity_tonnes", 0.0), 4),
                    "source": p.get("source", ""),
                    "regulatory_limit": p.get("regulatory_limit", None),
                    "measurement_method": p.get("measurement_method", ""),
                }
                for p in pollutants
            ],
        }

    def _section_water_discharges(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build water discharges section."""
        pollutants = data.get("water_pollutants", [])
        return {
            "title": "Water Discharges",
            "pollutant_count": len(pollutants),
            "total_volume_m3": round(data.get("water_discharge_volume_m3", 0.0), 2),
            "receiving_bodies": data.get("receiving_water_bodies", []),
            "pollutants": [
                {
                    "name": p.get("name", ""),
                    "concentration_mg_l": p.get("concentration_mg_l", 0.0),
                    "load_kg": round(p.get("load_kg", 0.0), 4),
                    "limit_mg_l": p.get("limit_mg_l", None),
                }
                for p in pollutants
            ],
        }

    def _section_soil_contamination(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build soil contamination section."""
        sites = data.get("contaminated_sites", [])
        return {
            "title": "Soil Contamination",
            "contaminated_site_count": len(sites),
            "total_area_hectares": round(
                sum(s.get("area_hectares", 0.0) for s in sites), 2
            ),
            "remediation_in_progress": sum(
                1 for s in sites if s.get("remediation_status") == "in_progress"
            ),
            "sites": [
                {
                    "site_name": s.get("site_name", ""),
                    "contaminant": s.get("contaminant", ""),
                    "area_hectares": s.get("area_hectares", 0.0),
                    "remediation_status": s.get("remediation_status", ""),
                }
                for s in sites
            ],
        }

    def _section_substances_of_concern(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build substances of concern section."""
        substances = data.get("substances_of_concern", [])
        total_tonnes = sum(s.get("quantity_tonnes", 0.0) for s in substances)
        return {
            "title": "Substances of Concern",
            "substance_count": len(substances),
            "total_tonnes": round(total_tonnes, 4),
            "substances": [
                {
                    "name": s.get("name", ""),
                    "cas_number": s.get("cas_number", ""),
                    "quantity_tonnes": round(s.get("quantity_tonnes", 0.0), 4),
                    "use_category": s.get("use_category", ""),
                    "hazard_classification": s.get("hazard_classification", ""),
                }
                for s in substances
            ],
        }

    def _section_svhc_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build SVHC (Substances of Very High Concern) assessment section."""
        svhcs = data.get("svhc_substances", [])
        return {
            "title": "SVHC Assessment",
            "svhc_count": len(svhcs),
            "substitution_plans": sum(
                1 for s in svhcs if s.get("substitution_planned", False)
            ),
            "svhcs": [
                {
                    "name": s.get("name", ""),
                    "cas_number": s.get("cas_number", ""),
                    "quantity_tonnes": round(s.get("quantity_tonnes", 0.0), 4),
                    "authorization_status": s.get("authorization_status", ""),
                    "substitution_planned": s.get("substitution_planned", False),
                    "substitution_timeline": s.get("substitution_timeline", ""),
                }
                for s in svhcs
            ],
        }

    def _section_financial_effects(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build financial effects section."""
        return {
            "title": "Financial Effects of Pollution",
            "remediation_costs_eur": data.get("remediation_costs_eur", 0.0),
            "fines_and_penalties_eur": data.get("fines_and_penalties_eur", 0.0),
            "pollution_prevention_capex_eur": data.get("pollution_prevention_capex_eur", 0.0),
            "pollution_prevention_opex_eur": data.get("pollution_prevention_opex_eur", 0.0),
            "potential_liability_eur": data.get("potential_liability_eur", 0.0),
            "insurance_coverage_eur": data.get("insurance_coverage_eur", 0.0),
        }

    def _section_methodology_notes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build methodology notes section."""
        return {
            "title": "Methodology Notes",
            "measurement_standards": data.get("measurement_standards", []),
            "estimation_methods": data.get("estimation_methods", []),
            "reporting_boundary": data.get("reporting_boundary", "operational_control"),
            "data_quality_notes": data.get("data_quality_notes", ""),
            "third_party_verification": data.get("third_party_verification", False),
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
            f"# ESRS E2 Pollution Report\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E2 Pollution"
        )

    def _md_pollution_policies(self, data: Dict[str, Any]) -> str:
        """Render pollution policies markdown."""
        sec = self._section_pollution_policies(data)
        lines = [f"## {sec['title']}\n", f"**Policies in place:** {sec['policy_count']}\n"]
        for p in sec["policies"]:
            lines.append(f"- **{p['name']}** (Scope: {p['scope']}, Alignment: {p['alignment']})")
        return "\n".join(lines)

    def _md_pollution_actions(self, data: Dict[str, Any]) -> str:
        """Render pollution actions markdown."""
        sec = self._section_pollution_actions(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Investment:** EUR {sec['total_investment_eur']:,.2f}\n",
        ]
        if sec["actions"]:
            lines.append("| Action | Type | Status | Investment (EUR) |")
            lines.append("|--------|------|--------|----------------:|")
            for a in sec["actions"]:
                lines.append(
                    f"| {a['description']} | {a['type']} "
                    f"| {a['status']} | {a['investment_eur']:,.2f} |"
                )
        return "\n".join(lines)

    def _md_pollution_targets(self, data: Dict[str, Any]) -> str:
        """Render pollution targets markdown."""
        sec = self._section_pollution_targets(data)
        lines = [f"## {sec['title']}\n"]
        if sec["targets"]:
            lines.append("| Target | Pollutant | Baseline | Current | Target | Unit |")
            lines.append("|--------|-----------|--------:|--------:|-------:|------|")
            for t in sec["targets"]:
                lines.append(
                    f"| {t['name']} | {t['pollutant']} | {t['baseline_value']:.2f} "
                    f"| {t['current_value']:.2f} | {t['target_value']:.2f} | {t['unit']} |"
                )
        return "\n".join(lines)

    def _md_air_emissions(self, data: Dict[str, Any]) -> str:
        """Render air emissions markdown."""
        sec = self._section_air_emissions(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Air Emissions:** {sec['total_tonnes']:.4f} tonnes\n",
        ]
        if sec["pollutants"]:
            lines.append("| Pollutant | Quantity (t) | Source | Method |")
            lines.append("|-----------|------------:|--------|--------|")
            for p in sec["pollutants"]:
                lines.append(
                    f"| {p['name']} | {p['quantity_tonnes']:.4f} "
                    f"| {p['source']} | {p['measurement_method']} |"
                )
        return "\n".join(lines)

    def _md_water_discharges(self, data: Dict[str, Any]) -> str:
        """Render water discharges markdown."""
        sec = self._section_water_discharges(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Discharge Volume:** {sec['total_volume_m3']:,.2f} m3\n",
        ]
        if sec["pollutants"]:
            lines.append("| Pollutant | Concentration (mg/L) | Load (kg) |")
            lines.append("|-----------|--------------------:|----------:|")
            for p in sec["pollutants"]:
                lines.append(
                    f"| {p['name']} | {p['concentration_mg_l']:.2f} | {p['load_kg']:.4f} |"
                )
        return "\n".join(lines)

    def _md_soil_contamination(self, data: Dict[str, Any]) -> str:
        """Render soil contamination markdown."""
        sec = self._section_soil_contamination(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Sites:** {sec['contaminated_site_count']}  ",
            f"**Total Area:** {sec['total_area_hectares']:.2f} ha  ",
            f"**Remediation In Progress:** {sec['remediation_in_progress']}\n",
        ]
        if sec["sites"]:
            lines.append("| Site | Contaminant | Area (ha) | Status |")
            lines.append("|------|-------------|----------:|--------|")
            for s in sec["sites"]:
                lines.append(
                    f"| {s['site_name']} | {s['contaminant']} "
                    f"| {s['area_hectares']:.2f} | {s['remediation_status']} |"
                )
        return "\n".join(lines)

    def _md_substances_of_concern(self, data: Dict[str, Any]) -> str:
        """Render substances of concern markdown."""
        sec = self._section_substances_of_concern(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total:** {sec['total_tonnes']:.4f} tonnes across "
            f"{sec['substance_count']} substances\n",
        ]
        if sec["substances"]:
            lines.append("| Substance | CAS | Quantity (t) | Category | Hazard |")
            lines.append("|-----------|-----|------------:|----------|--------|")
            for s in sec["substances"]:
                lines.append(
                    f"| {s['name']} | {s['cas_number']} | {s['quantity_tonnes']:.4f} "
                    f"| {s['use_category']} | {s['hazard_classification']} |"
                )
        return "\n".join(lines)

    def _md_svhc_assessment(self, data: Dict[str, Any]) -> str:
        """Render SVHC assessment markdown."""
        sec = self._section_svhc_assessment(data)
        lines = [
            f"## {sec['title']}\n",
            f"**SVHCs Identified:** {sec['svhc_count']}  ",
            f"**Substitution Plans:** {sec['substitution_plans']}\n",
        ]
        for s in sec["svhcs"]:
            sub = "Planned" if s["substitution_planned"] else "None"
            lines.append(
                f"- **{s['name']}** (CAS: {s['cas_number']}) - "
                f"{s['quantity_tonnes']:.4f} t - Substitution: {sub}"
            )
        return "\n".join(lines)

    def _md_financial_effects(self, data: Dict[str, Any]) -> str:
        """Render financial effects markdown."""
        sec = self._section_financial_effects(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Item | EUR |\n|------|----:|\n"
            f"| Remediation Costs | {sec['remediation_costs_eur']:,.2f} |\n"
            f"| Fines & Penalties | {sec['fines_and_penalties_eur']:,.2f} |\n"
            f"| Prevention CapEx | {sec['pollution_prevention_capex_eur']:,.2f} |\n"
            f"| Prevention OpEx | {sec['pollution_prevention_opex_eur']:,.2f} |\n"
            f"| Potential Liability | {sec['potential_liability_eur']:,.2f} |\n"
            f"| Insurance Coverage | {sec['insurance_coverage_eur']:,.2f} |"
        )

    def _md_methodology_notes(self, data: Dict[str, Any]) -> str:
        """Render methodology notes markdown."""
        sec = self._section_methodology_notes(data)
        verified = "Yes" if sec["third_party_verification"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"- **Reporting Boundary:** {sec['reporting_boundary']}\n"
            f"- **Third-Party Verification:** {verified}\n"
            f"- **Data Quality:** {sec['data_quality_notes']}"
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
            f"<h1>ESRS E2 Pollution Report</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_air_emissions(self, data: Dict[str, Any]) -> str:
        """Render air emissions HTML."""
        sec = self._section_air_emissions(data)
        rows = "".join(
            f"<tr><td>{p['name']}</td><td>{p['quantity_tonnes']:.4f}</td>"
            f"<td>{p['source']}</td></tr>"
            for p in sec["pollutants"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_tonnes']:.4f} tonnes</p>\n"
            f"<table><tr><th>Pollutant</th><th>Quantity (t)</th><th>Source</th></tr>"
            f"{rows}</table>"
        )

    def _html_water_discharges(self, data: Dict[str, Any]) -> str:
        """Render water discharges HTML."""
        sec = self._section_water_discharges(data)
        rows = "".join(
            f"<tr><td>{p['name']}</td><td>{p['concentration_mg_l']:.2f}</td>"
            f"<td>{p['load_kg']:.4f}</td></tr>"
            for p in sec["pollutants"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Volume: {sec['total_volume_m3']:,.2f} m3</p>\n"
            f"<table><tr><th>Pollutant</th><th>mg/L</th><th>Load (kg)</th></tr>"
            f"{rows}</table>"
        )

    def _html_substances_of_concern(self, data: Dict[str, Any]) -> str:
        """Render substances of concern HTML."""
        sec = self._section_substances_of_concern(data)
        rows = "".join(
            f"<tr><td>{s['name']}</td><td>{s['cas_number']}</td>"
            f"<td>{s['quantity_tonnes']:.4f}</td></tr>"
            for s in sec["substances"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_tonnes']:.4f} tonnes</p>\n"
            f"<table><tr><th>Substance</th><th>CAS</th><th>Quantity (t)</th></tr>"
            f"{rows}</table>"
        )

    def _html_financial_effects(self, data: Dict[str, Any]) -> str:
        """Render financial effects HTML."""
        sec = self._section_financial_effects(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Item</th><th>EUR</th></tr>"
            f"<tr><td>Remediation</td><td>{sec['remediation_costs_eur']:,.2f}</td></tr>"
            f"<tr><td>Fines</td><td>{sec['fines_and_penalties_eur']:,.2f}</td></tr>"
            f"<tr><td>Prevention CapEx</td>"
            f"<td>{sec['pollution_prevention_capex_eur']:,.2f}</td></tr>"
            f"<tr><td>Prevention OpEx</td>"
            f"<td>{sec['pollution_prevention_opex_eur']:,.2f}</td></tr></table>"
        )
