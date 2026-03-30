# -*- coding: utf-8 -*-
"""
ClimateRiskReportTemplate - ESRS E1-9 Climate Risk Disclosure Report

Renders physical risk summary, transition risk summary, opportunity
summary, scenario analysis, financial effects, and time-horizon
breakdown per ESRS E1-9.

Sections:
    1. Physical Risk Summary
    2. Transition Risk Summary
    3. Opportunity Summary
    4. Scenario Analysis
    5. Financial Effects
    6. Time Horizon Breakdown

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
    "physical_risk_summary",
    "transition_risk_summary",
    "opportunity_summary",
    "scenario_analysis",
    "financial_effects",
    "time_horizon_breakdown",
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

class ClimateRiskReportTemplate:
    """
    Climate risk and opportunity disclosure report template per ESRS E1-9.

    Renders anticipated financial effects from physical risks (acute
    and chronic), transition risks (policy, technology, market,
    reputation), climate-related opportunities, multi-scenario analysis,
    quantified financial effects, and time-horizon disaggregation.

    Example:
        >>> tpl = ClimateRiskReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ClimateRiskReportTemplate."""
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
        if "physical_risks" not in data:
            warnings.append("physical_risks missing; will default to empty")
        if "transition_risks" not in data:
            warnings.append("transition_risks missing; will default to empty")
        if "climate_opportunities" not in data:
            warnings.append("climate_opportunities missing; will default to empty")
        if "climate_scenarios" not in data:
            warnings.append("climate_scenarios missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render climate risk report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_physical_risks(data),
            self._md_transition_risks(data),
            self._md_opportunities(data),
            self._md_scenarios(data),
            self._md_financial_effects(data),
            self._md_time_horizons(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render climate risk report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_physical_risks(data),
            self._html_transition_risks(data),
            self._html_financial_effects(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Climate Risk Report - ESRS E1-9</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render climate risk report as JSON."""
        self.generated_at = utcnow()
        result = {
            "template": "climate_risk_report",
            "esrs_reference": "E1-9",
            "version": "16.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "physical_risks_count": len(data.get("physical_risks", [])),
            "transition_risks_count": len(data.get("transition_risks", [])),
            "opportunities_count": len(data.get("climate_opportunities", [])),
            "scenarios_analyzed": len(data.get("climate_scenarios", [])),
            "total_financial_impact_eur": data.get("total_financial_impact_eur", 0.0),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_physical_risk_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build physical risk summary section."""
        risks = data.get("physical_risks", [])
        acute = [r for r in risks if r.get("category") == "acute"]
        chronic = [r for r in risks if r.get("category") == "chronic"]
        total_exposure = sum(r.get("exposure_eur", 0.0) for r in risks)
        return {
            "title": "Physical Risk Summary",
            "total_risks": len(risks),
            "acute_count": len(acute),
            "chronic_count": len(chronic),
            "total_exposure_eur": round(total_exposure, 2),
            "risks": [
                {
                    "hazard": r.get("hazard", ""),
                    "category": r.get("category", ""),
                    "likelihood": r.get("likelihood", ""),
                    "severity": r.get("severity", ""),
                    "exposure_eur": round(r.get("exposure_eur", 0.0), 2),
                    "affected_assets": r.get("affected_assets", []),
                    "time_horizon": r.get("time_horizon", ""),
                }
                for r in risks
            ],
        }

    def _section_transition_risk_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build transition risk summary section."""
        risks = data.get("transition_risks", [])
        categories = {}
        for r in risks:
            cat = r.get("category", "other")
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        total_impact = sum(r.get("financial_impact_eur", 0.0) for r in risks)
        return {
            "title": "Transition Risk Summary",
            "total_risks": len(risks),
            "by_category": categories,
            "total_financial_impact_eur": round(total_impact, 2),
            "risks": [
                {
                    "name": r.get("name", ""),
                    "category": r.get("category", ""),
                    "likelihood": r.get("likelihood", ""),
                    "impact_level": r.get("impact_level", ""),
                    "financial_impact_eur": round(
                        r.get("financial_impact_eur", 0.0), 2
                    ),
                    "time_horizon": r.get("time_horizon", ""),
                    "mitigation_status": r.get("mitigation_status", ""),
                }
                for r in risks
            ],
        }

    def _section_opportunity_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build opportunity summary section."""
        opportunities = data.get("climate_opportunities", [])
        total_value = sum(o.get("potential_value_eur", 0.0) for o in opportunities)
        return {
            "title": "Climate-Related Opportunity Summary",
            "total_opportunities": len(opportunities),
            "total_potential_value_eur": round(total_value, 2),
            "opportunities": [
                {
                    "name": o.get("name", ""),
                    "category": o.get("category", ""),
                    "potential_value_eur": round(o.get("potential_value_eur", 0.0), 2),
                    "time_horizon": o.get("time_horizon", ""),
                    "likelihood": o.get("likelihood", ""),
                    "strategy_link": o.get("strategy_link", ""),
                }
                for o in opportunities
            ],
        }

    def _section_scenario_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build scenario analysis section."""
        scenarios = data.get("climate_scenarios", [])
        return {
            "title": "Climate Scenario Analysis",
            "scenarios_count": len(scenarios),
            "scenarios": [
                {
                    "name": s.get("name", ""),
                    "provider": s.get("provider", ""),
                    "temperature_outcome": s.get("temperature_outcome", ""),
                    "time_horizon": s.get("time_horizon", ""),
                    "physical_risk_impact_eur": round(
                        s.get("physical_risk_impact_eur", 0.0), 2
                    ),
                    "transition_risk_impact_eur": round(
                        s.get("transition_risk_impact_eur", 0.0), 2
                    ),
                    "opportunity_value_eur": round(
                        s.get("opportunity_value_eur", 0.0), 2
                    ),
                    "net_impact_eur": round(s.get("net_impact_eur", 0.0), 2),
                }
                for s in scenarios
            ],
        }

    def _section_financial_effects(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build financial effects section."""
        effects = data.get("financial_effects", {})
        return {
            "title": "Anticipated Financial Effects",
            "revenue_at_risk_eur": round(effects.get("revenue_at_risk_eur", 0.0), 2),
            "assets_at_risk_eur": round(effects.get("assets_at_risk_eur", 0.0), 2),
            "stranded_assets_eur": round(effects.get("stranded_assets_eur", 0.0), 2),
            "adaptation_cost_eur": round(effects.get("adaptation_cost_eur", 0.0), 2),
            "insurance_cost_increase_eur": round(
                effects.get("insurance_cost_increase_eur", 0.0), 2
            ),
            "opportunity_revenue_eur": round(
                effects.get("opportunity_revenue_eur", 0.0), 2
            ),
            "net_financial_impact_eur": round(
                effects.get("net_financial_impact_eur", 0.0), 2
            ),
            "quantification_methodology": effects.get(
                "quantification_methodology", ""
            ),
        }

    def _section_time_horizon_breakdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build time horizon breakdown section."""
        horizons = data.get("time_horizon_breakdown", {})
        return {
            "title": "Time Horizon Breakdown",
            "short_term": {
                "label": "Short-term (0-1 years)",
                "physical_risk_eur": round(
                    horizons.get("short_term", {}).get("physical_risk_eur", 0.0), 2
                ),
                "transition_risk_eur": round(
                    horizons.get("short_term", {}).get("transition_risk_eur", 0.0), 2
                ),
                "opportunity_eur": round(
                    horizons.get("short_term", {}).get("opportunity_eur", 0.0), 2
                ),
            },
            "medium_term": {
                "label": "Medium-term (1-5 years)",
                "physical_risk_eur": round(
                    horizons.get("medium_term", {}).get("physical_risk_eur", 0.0), 2
                ),
                "transition_risk_eur": round(
                    horizons.get("medium_term", {}).get("transition_risk_eur", 0.0), 2
                ),
                "opportunity_eur": round(
                    horizons.get("medium_term", {}).get("opportunity_eur", 0.0), 2
                ),
            },
            "long_term": {
                "label": "Long-term (5+ years)",
                "physical_risk_eur": round(
                    horizons.get("long_term", {}).get("physical_risk_eur", 0.0), 2
                ),
                "transition_risk_eur": round(
                    horizons.get("long_term", {}).get("transition_risk_eur", 0.0), 2
                ),
                "opportunity_eur": round(
                    horizons.get("long_term", {}).get("opportunity_eur", 0.0), 2
                ),
            },
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
            f"# Climate Risk & Opportunity Report - ESRS E1-9\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E1-9 Anticipated Financial Effects from Material "
            f"Physical and Transition Risks and Climate-Related Opportunities"
        )

    def _md_physical_risks(self, data: Dict[str, Any]) -> str:
        """Render physical risks markdown."""
        sec = self._section_physical_risk_summary(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Risks:** {sec['total_risks']} "
            f"(Acute: {sec['acute_count']}, Chronic: {sec['chronic_count']})  \n"
            f"**Total Exposure:** EUR {sec['total_exposure_eur']:,.2f}\n",
            "| Hazard | Category | Likelihood | Severity | Exposure (EUR) |",
            "|--------|----------|------------|----------|---------------:|",
        ]
        for r in sec["risks"]:
            lines.append(
                f"| {r['hazard']} | {r['category']} | {r['likelihood']} "
                f"| {r['severity']} | {r['exposure_eur']:,.2f} |"
            )
        return "\n".join(lines)

    def _md_transition_risks(self, data: Dict[str, Any]) -> str:
        """Render transition risks markdown."""
        sec = self._section_transition_risk_summary(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Risks:** {sec['total_risks']}  \n"
            f"**Total Financial Impact:** EUR {sec['total_financial_impact_eur']:,.2f}\n",
            "| Risk | Category | Likelihood | Impact | Financial (EUR) |",
            "|------|----------|------------|--------|----------------:|",
        ]
        for r in sec["risks"]:
            lines.append(
                f"| {r['name']} | {r['category']} | {r['likelihood']} "
                f"| {r['impact_level']} | {r['financial_impact_eur']:,.2f} |"
            )
        return "\n".join(lines)

    def _md_opportunities(self, data: Dict[str, Any]) -> str:
        """Render opportunities markdown."""
        sec = self._section_opportunity_summary(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Opportunities:** {sec['total_opportunities']}  \n"
            f"**Total Potential Value:** EUR {sec['total_potential_value_eur']:,.2f}\n",
            "| Opportunity | Category | Value (EUR) | Horizon |",
            "|-------------|----------|------------|---------|",
        ]
        for o in sec["opportunities"]:
            lines.append(
                f"| {o['name']} | {o['category']} "
                f"| {o['potential_value_eur']:,.2f} | {o['time_horizon']} |"
            )
        return "\n".join(lines)

    def _md_scenarios(self, data: Dict[str, Any]) -> str:
        """Render scenario analysis markdown."""
        sec = self._section_scenario_analysis(data)
        lines = [
            f"## {sec['title']}\n",
            "| Scenario | Temp | Physical (EUR) | Transition (EUR) | Net (EUR) |",
            "|----------|------|---------------:|-----------------:|----------:|",
        ]
        for s in sec["scenarios"]:
            lines.append(
                f"| {s['name']} | {s['temperature_outcome']} "
                f"| {s['physical_risk_impact_eur']:,.2f} "
                f"| {s['transition_risk_impact_eur']:,.2f} "
                f"| {s['net_impact_eur']:,.2f} |"
            )
        return "\n".join(lines)

    def _md_financial_effects(self, data: Dict[str, Any]) -> str:
        """Render financial effects markdown."""
        sec = self._section_financial_effects(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Category | EUR |\n|----------|----:|\n"
            f"| Revenue at Risk | {sec['revenue_at_risk_eur']:,.2f} |\n"
            f"| Assets at Risk | {sec['assets_at_risk_eur']:,.2f} |\n"
            f"| Stranded Assets | {sec['stranded_assets_eur']:,.2f} |\n"
            f"| Adaptation Cost | {sec['adaptation_cost_eur']:,.2f} |\n"
            f"| Insurance Increase | {sec['insurance_cost_increase_eur']:,.2f} |\n"
            f"| Opportunity Revenue | {sec['opportunity_revenue_eur']:,.2f} |\n"
            f"| **Net Impact** | **{sec['net_financial_impact_eur']:,.2f}** |"
        )

    def _md_time_horizons(self, data: Dict[str, Any]) -> str:
        """Render time horizon breakdown markdown."""
        sec = self._section_time_horizon_breakdown(data)
        lines = [
            "## Time Horizon Breakdown\n",
            "| Horizon | Physical Risk (EUR) | Transition Risk (EUR) | Opportunity (EUR) |",
            "|---------|--------------------:|----------------------:|------------------:|",
        ]
        for key in ("short_term", "medium_term", "long_term"):
            h = sec[key]
            lines.append(
                f"| {h['label']} | {h['physical_risk_eur']:,.2f} "
                f"| {h['transition_risk_eur']:,.2f} | {h['opportunity_eur']:,.2f} |"
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
            ".high-risk{background:#ffebee;color:#c62828}"
            ".medium-risk{background:#fff3e0;color:#e65100}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"<h1>Climate Risk & Opportunity Report - ESRS E1-9</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_physical_risks(self, data: Dict[str, Any]) -> str:
        """Render physical risks HTML."""
        sec = self._section_physical_risk_summary(data)
        rows = "".join(
            f"<tr><td>{r['hazard']}</td><td>{r['category']}</td>"
            f"<td>{r['likelihood']}</td><td>{r['exposure_eur']:,.2f}</td></tr>"
            for r in sec["risks"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Hazard</th><th>Category</th>"
            f"<th>Likelihood</th><th>Exposure (EUR)</th></tr>{rows}</table>"
        )

    def _html_transition_risks(self, data: Dict[str, Any]) -> str:
        """Render transition risks HTML."""
        sec = self._section_transition_risk_summary(data)
        rows = "".join(
            f"<tr><td>{r['name']}</td><td>{r['category']}</td>"
            f"<td>{r['financial_impact_eur']:,.2f}</td></tr>"
            for r in sec["risks"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Risk</th><th>Category</th>"
            f"<th>Financial Impact (EUR)</th></tr>{rows}</table>"
        )

    def _html_financial_effects(self, data: Dict[str, Any]) -> str:
        """Render financial effects HTML."""
        sec = self._section_financial_effects(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Category</th><th>EUR</th></tr>"
            f"<tr><td>Revenue at Risk</td><td>{sec['revenue_at_risk_eur']:,.2f}</td></tr>"
            f"<tr><td>Assets at Risk</td><td>{sec['assets_at_risk_eur']:,.2f}</td></tr>"
            f"<tr><td>Net Impact</td><td>{sec['net_financial_impact_eur']:,.2f}</td></tr>"
            f"</table>"
        )
