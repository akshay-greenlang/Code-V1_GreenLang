# -*- coding: utf-8 -*-
"""
E4BiodiversityReportTemplate - ESRS E4 Biodiversity and Ecosystems Report

Renders transition plan, biodiversity policies, actions, targets, site
assessments, land use metrics, species impacts, ecosystem dependencies,
deforestation status, and financial effects per ESRS E4.

Sections:
    1. Transition Plan
    2. Biodiversity Policies
    3. Biodiversity Actions
    4. Biodiversity Targets
    5. Site Assessments
    6. Land Use Metrics
    7. Species Impacts
    8. Ecosystem Dependencies
    9. Deforestation Status
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
    "transition_plan",
    "biodiversity_policies",
    "biodiversity_actions",
    "biodiversity_targets",
    "site_assessments",
    "land_use_metrics",
    "species_impacts",
    "ecosystem_dependencies",
    "deforestation_status",
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

class E4BiodiversityReportTemplate:
    """
    ESRS E4 Biodiversity and Ecosystems report template.

    Renders biodiversity transition plans, policies, action plans,
    measurable targets, site-level assessments, land use change metrics,
    species impact analysis, ecosystem service dependencies, deforestation
    status, and financial effects per ESRS E4.

    Example:
        >>> tpl = E4BiodiversityReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize E4BiodiversityReportTemplate."""
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
        if "site_assessments" not in data:
            warnings.append("site_assessments missing; will default to empty")
        if "species_impacts" not in data:
            warnings.append("species_impacts missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render E4 Biodiversity report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_transition_plan(data),
            self._md_biodiversity_policies(data),
            self._md_biodiversity_actions(data),
            self._md_biodiversity_targets(data),
            self._md_site_assessments(data),
            self._md_land_use_metrics(data),
            self._md_species_impacts(data),
            self._md_ecosystem_dependencies(data),
            self._md_deforestation_status(data),
            self._md_financial_effects(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render E4 Biodiversity report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_site_assessments(data),
            self._html_land_use(data),
            self._html_species_impacts(data),
            self._html_deforestation(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>ESRS E4 Biodiversity Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render E4 Biodiversity report as JSON string."""
        self.generated_at = utcnow()
        result = {
            "template": "e4_biodiversity_report",
            "esrs_reference": "ESRS E4",
            "version": "17.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "site_count": len(data.get("site_assessments", [])),
            "species_impacted": len(data.get("species_impacts", [])),
            "deforestation_free": data.get("deforestation_free", False),
            "total_land_use_hectares": data.get("total_land_use_hectares", 0.0),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_transition_plan(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build transition plan section."""
        return {
            "title": "Biodiversity Transition Plan",
            "has_transition_plan": data.get("has_biodiversity_transition_plan", False),
            "plan_description": data.get("transition_plan_description", ""),
            "aligned_with_gbf": data.get("aligned_with_gbf", False),
            "key_milestones": data.get("transition_milestones", []),
            "investment_planned_eur": data.get("transition_investment_eur", 0.0),
        }

    def _section_biodiversity_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build biodiversity policies section."""
        policies = data.get("biodiversity_policies", [])
        return {
            "title": "Biodiversity Policies",
            "has_policy": len(policies) > 0,
            "policy_count": len(policies),
            "policies": [
                {
                    "name": p.get("name", ""),
                    "scope": p.get("scope", ""),
                    "no_net_loss_commitment": p.get("no_net_loss_commitment", False),
                    "protected_areas_policy": p.get("protected_areas_policy", False),
                }
                for p in policies
            ],
        }

    def _section_biodiversity_actions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build biodiversity actions section."""
        actions = data.get("biodiversity_actions", [])
        return {
            "title": "Biodiversity Actions",
            "action_count": len(actions),
            "total_investment_eur": round(
                sum(a.get("investment_eur", 0.0) for a in actions), 2
            ),
            "actions": [
                {
                    "description": a.get("description", ""),
                    "type": a.get("type", ""),
                    "status": a.get("status", ""),
                    "mitigation_hierarchy": a.get("mitigation_hierarchy", ""),
                    "investment_eur": a.get("investment_eur", 0.0),
                }
                for a in actions
            ],
        }

    def _section_biodiversity_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build biodiversity targets section."""
        targets = data.get("biodiversity_targets", [])
        return {
            "title": "Biodiversity Targets",
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
                    "aligned_with_gbf_target": t.get("aligned_with_gbf_target", ""),
                }
                for t in targets
            ],
        }

    def _section_site_assessments(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build site assessments section."""
        sites = data.get("site_assessments", [])
        sensitive = sum(1 for s in sites if s.get("biodiversity_sensitive", False))
        return {
            "title": "Site-Level Biodiversity Assessments",
            "total_sites": len(sites),
            "sensitive_sites": sensitive,
            "sites": [
                {
                    "site_name": s.get("site_name", ""),
                    "location": s.get("location", ""),
                    "area_hectares": s.get("area_hectares", 0.0),
                    "biodiversity_sensitive": s.get("biodiversity_sensitive", False),
                    "proximity_to_protected_area": s.get(
                        "proximity_to_protected_area", ""
                    ),
                    "key_species": s.get("key_species", []),
                    "assessment_date": s.get("assessment_date", ""),
                }
                for s in sites
            ],
        }

    def _section_land_use_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build land use metrics section."""
        return {
            "title": "Land Use Metrics",
            "total_land_use_hectares": round(
                data.get("total_land_use_hectares", 0.0), 2
            ),
            "sealed_land_hectares": round(data.get("sealed_land_hectares", 0.0), 2),
            "nature_oriented_hectares": round(
                data.get("nature_oriented_hectares", 0.0), 2
            ),
            "land_use_change_hectares": round(
                data.get("land_use_change_hectares", 0.0), 2
            ),
            "restored_hectares": round(data.get("restored_hectares", 0.0), 2),
            "net_land_conversion": round(data.get("net_land_conversion", 0.0), 2),
        }

    def _section_species_impacts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build species impacts section."""
        species = data.get("species_impacts", [])
        return {
            "title": "Species Impacts",
            "species_count": len(species),
            "iucn_red_list_count": sum(
                1 for s in species if s.get("iucn_red_list", False)
            ),
            "species": [
                {
                    "species_name": s.get("species_name", ""),
                    "iucn_status": s.get("iucn_status", ""),
                    "iucn_red_list": s.get("iucn_red_list", False),
                    "impact_type": s.get("impact_type", ""),
                    "impact_severity": s.get("impact_severity", ""),
                    "mitigation_action": s.get("mitigation_action", ""),
                }
                for s in species
            ],
        }

    def _section_ecosystem_dependencies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build ecosystem dependencies section."""
        deps = data.get("ecosystem_dependencies", [])
        return {
            "title": "Ecosystem Service Dependencies",
            "dependency_count": len(deps),
            "dependencies": [
                {
                    "service_type": d.get("service_type", ""),
                    "description": d.get("description", ""),
                    "criticality": d.get("criticality", ""),
                    "value_chain_stage": d.get("value_chain_stage", ""),
                    "risk_if_degraded": d.get("risk_if_degraded", ""),
                }
                for d in deps
            ],
        }

    def _section_deforestation_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build deforestation status section."""
        return {
            "title": "Deforestation Status",
            "deforestation_free": data.get("deforestation_free", False),
            "deforestation_free_commitment_date": data.get(
                "deforestation_free_commitment_date", ""
            ),
            "commodity_risk_areas": data.get("commodity_risk_areas", []),
            "monitoring_system": data.get("deforestation_monitoring_system", ""),
            "satellite_monitoring": data.get("satellite_monitoring", False),
            "forest_loss_hectares": round(data.get("forest_loss_hectares", 0.0), 2),
        }

    def _section_financial_effects(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build financial effects section."""
        return {
            "title": "Financial Effects of Biodiversity Impacts",
            "biodiversity_capex_eur": data.get("biodiversity_capex_eur", 0.0),
            "biodiversity_opex_eur": data.get("biodiversity_opex_eur", 0.0),
            "restoration_costs_eur": data.get("restoration_costs_eur", 0.0),
            "ecosystem_service_value_at_risk_eur": data.get(
                "ecosystem_service_value_at_risk_eur", 0.0
            ),
            "offset_credits_eur": data.get("offset_credits_eur", 0.0),
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
            f"# ESRS E4 Biodiversity and Ecosystems Report\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E4 Biodiversity and Ecosystems"
        )

    def _md_transition_plan(self, data: Dict[str, Any]) -> str:
        """Render transition plan markdown."""
        sec = self._section_transition_plan(data)
        has_plan = "Yes" if sec["has_transition_plan"] else "No"
        gbf = "Yes" if sec["aligned_with_gbf"] else "No"
        lines = [
            f"## {sec['title']}\n",
            f"**Has Plan:** {has_plan}  ",
            f"**Aligned with GBF:** {gbf}  ",
            f"**Investment Planned:** EUR {sec['investment_planned_eur']:,.2f}\n",
            f"{sec['plan_description']}",
        ]
        if sec["key_milestones"]:
            lines.append("\n**Key Milestones:**")
            for m in sec["key_milestones"]:
                lines.append(f"- {m}")
        return "\n".join(lines)

    def _md_biodiversity_policies(self, data: Dict[str, Any]) -> str:
        """Render biodiversity policies markdown."""
        sec = self._section_biodiversity_policies(data)
        lines = [f"## {sec['title']}\n", f"**Policies:** {sec['policy_count']}\n"]
        for p in sec["policies"]:
            nnl = "Yes" if p["no_net_loss_commitment"] else "No"
            lines.append(f"- **{p['name']}** (Scope: {p['scope']}, No-Net-Loss: {nnl})")
        return "\n".join(lines)

    def _md_biodiversity_actions(self, data: Dict[str, Any]) -> str:
        """Render biodiversity actions markdown."""
        sec = self._section_biodiversity_actions(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Investment:** EUR {sec['total_investment_eur']:,.2f}\n",
        ]
        if sec["actions"]:
            lines.append("| Action | Type | Hierarchy | Status | EUR |")
            lines.append("|--------|------|-----------|--------|----:|")
            for a in sec["actions"]:
                lines.append(
                    f"| {a['description']} | {a['type']} | {a['mitigation_hierarchy']} "
                    f"| {a['status']} | {a['investment_eur']:,.2f} |"
                )
        return "\n".join(lines)

    def _md_biodiversity_targets(self, data: Dict[str, Any]) -> str:
        """Render biodiversity targets markdown."""
        sec = self._section_biodiversity_targets(data)
        lines = [f"## {sec['title']}\n"]
        if sec["targets"]:
            lines.append("| Target | Metric | Baseline | Current | Goal | GBF |")
            lines.append("|--------|--------|--------:|--------:|-----:|-----|")
            for t in sec["targets"]:
                lines.append(
                    f"| {t['name']} | {t['metric']} | {t['baseline_value']:.2f} "
                    f"| {t['current_value']:.2f} | {t['target_value']:.2f} "
                    f"| {t['aligned_with_gbf_target']} |"
                )
        return "\n".join(lines)

    def _md_site_assessments(self, data: Dict[str, Any]) -> str:
        """Render site assessments markdown."""
        sec = self._section_site_assessments(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Sites:** {sec['total_sites']}  ",
            f"**Sensitive Sites:** {sec['sensitive_sites']}\n",
        ]
        if sec["sites"]:
            lines.append("| Site | Location | Area (ha) | Sensitive | Protected Area |")
            lines.append("|------|----------|----------:|:---------:|----------------|")
            for s in sec["sites"]:
                sens = "Yes" if s["biodiversity_sensitive"] else "No"
                lines.append(
                    f"| {s['site_name']} | {s['location']} | {s['area_hectares']:.2f} "
                    f"| {sens} | {s['proximity_to_protected_area']} |"
                )
        return "\n".join(lines)

    def _md_land_use_metrics(self, data: Dict[str, Any]) -> str:
        """Render land use metrics markdown."""
        sec = self._section_land_use_metrics(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Metric | Hectares |\n|--------|--------:|\n"
            f"| Total Land Use | {sec['total_land_use_hectares']:,.2f} |\n"
            f"| Sealed Land | {sec['sealed_land_hectares']:,.2f} |\n"
            f"| Nature-Oriented | {sec['nature_oriented_hectares']:,.2f} |\n"
            f"| Land Use Change | {sec['land_use_change_hectares']:,.2f} |\n"
            f"| Restored | {sec['restored_hectares']:,.2f} |\n"
            f"| Net Conversion | {sec['net_land_conversion']:,.2f} |"
        )

    def _md_species_impacts(self, data: Dict[str, Any]) -> str:
        """Render species impacts markdown."""
        sec = self._section_species_impacts(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Species Impacted:** {sec['species_count']}  ",
            f"**IUCN Red List:** {sec['iucn_red_list_count']}\n",
        ]
        if sec["species"]:
            lines.append("| Species | IUCN Status | Impact Type | Severity |")
            lines.append("|---------|-------------|-------------|----------|")
            for s in sec["species"]:
                lines.append(
                    f"| {s['species_name']} | {s['iucn_status']} "
                    f"| {s['impact_type']} | {s['impact_severity']} |"
                )
        return "\n".join(lines)

    def _md_ecosystem_dependencies(self, data: Dict[str, Any]) -> str:
        """Render ecosystem dependencies markdown."""
        sec = self._section_ecosystem_dependencies(data)
        lines = [f"## {sec['title']}\n"]
        for d in sec["dependencies"]:
            lines.append(
                f"- **{d['service_type']}** ({d['criticality']}): {d['description']} "
                f"| Stage: {d['value_chain_stage']}"
            )
        return "\n".join(lines)

    def _md_deforestation_status(self, data: Dict[str, Any]) -> str:
        """Render deforestation status markdown."""
        sec = self._section_deforestation_status(data)
        free = "Yes" if sec["deforestation_free"] else "No"
        sat = "Yes" if sec["satellite_monitoring"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"**Deforestation-Free:** {free}  \n"
            f"**Commitment Date:** {sec['deforestation_free_commitment_date']}  \n"
            f"**Satellite Monitoring:** {sat}  \n"
            f"**Forest Loss:** {sec['forest_loss_hectares']:,.2f} ha"
        )

    def _md_financial_effects(self, data: Dict[str, Any]) -> str:
        """Render financial effects markdown."""
        sec = self._section_financial_effects(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Item | EUR |\n|------|----:|\n"
            f"| Biodiversity CapEx | {sec['biodiversity_capex_eur']:,.2f} |\n"
            f"| Biodiversity OpEx | {sec['biodiversity_opex_eur']:,.2f} |\n"
            f"| Restoration Costs | {sec['restoration_costs_eur']:,.2f} |\n"
            f"| Ecosystem Value at Risk | {sec['ecosystem_service_value_at_risk_eur']:,.2f} |\n"
            f"| Offset Credits | {sec['offset_credits_eur']:,.2f} |"
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
            f"<h1>ESRS E4 Biodiversity and Ecosystems Report</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_site_assessments(self, data: Dict[str, Any]) -> str:
        """Render site assessments HTML."""
        sec = self._section_site_assessments(data)
        rows = "".join(
            f"<tr><td>{s['site_name']}</td><td>{s['location']}</td>"
            f"<td>{s['area_hectares']:.2f}</td>"
            f"<td>{'Yes' if s['biodiversity_sensitive'] else 'No'}</td></tr>"
            for s in sec["sites"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Sites: {sec['total_sites']} | Sensitive: {sec['sensitive_sites']}</p>\n"
            f"<table><tr><th>Site</th><th>Location</th><th>Area (ha)</th>"
            f"<th>Sensitive</th></tr>{rows}</table>"
        )

    def _html_land_use(self, data: Dict[str, Any]) -> str:
        """Render land use HTML."""
        sec = self._section_land_use_metrics(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Metric</th><th>Hectares</th></tr>"
            f"<tr><td>Total Land Use</td><td>{sec['total_land_use_hectares']:,.2f}</td></tr>"
            f"<tr><td>Sealed</td><td>{sec['sealed_land_hectares']:,.2f}</td></tr>"
            f"<tr><td>Nature-Oriented</td><td>{sec['nature_oriented_hectares']:,.2f}</td></tr>"
            f"<tr><td>Restored</td><td>{sec['restored_hectares']:,.2f}</td></tr></table>"
        )

    def _html_species_impacts(self, data: Dict[str, Any]) -> str:
        """Render species impacts HTML."""
        sec = self._section_species_impacts(data)
        rows = "".join(
            f"<tr><td>{s['species_name']}</td><td>{s['iucn_status']}</td>"
            f"<td>{s['impact_type']}</td></tr>"
            for s in sec["species"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Species</th><th>IUCN</th><th>Impact</th></tr>"
            f"{rows}</table>"
        )

    def _html_deforestation(self, data: Dict[str, Any]) -> str:
        """Render deforestation HTML."""
        sec = self._section_deforestation_status(data)
        status = "Deforestation-Free" if sec["deforestation_free"] else "Not Verified"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Status: <strong>{status}</strong></p>\n"
            f"<p>Forest Loss: {sec['forest_loss_hectares']:,.2f} ha</p>"
        )
