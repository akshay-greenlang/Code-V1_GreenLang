# -*- coding: utf-8 -*-
"""
CarbonCreditsReportTemplate - ESRS E1-7 Carbon Credits Disclosure Report

Renders carbon credit portfolio summary, removal vs avoidance breakdown,
credit quality assessment, project details, and SBTi compliance
per ESRS E1-7.

Sections:
    1. Portfolio Summary
    2. Removals Summary
    3. Credit Quality
    4. Project Details
    5. SBTi Compliance

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
    "portfolio_summary",
    "removals_summary",
    "credit_quality",
    "project_details",
    "sbti_compliance",
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


class CarbonCreditsReportTemplate:
    """
    Carbon credits disclosure report template per ESRS E1-7.

    Renders portfolio of carbon credits with distinction between
    removals and avoidance/reduction, quality tier assessment based
    on certification standards, project-level details, and alignment
    with SBTi beyond value chain mitigation guidance.

    Example:
        >>> tpl = CarbonCreditsReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CarbonCreditsReportTemplate."""
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
        if "carbon_credits" not in data:
            warnings.append("carbon_credits missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render carbon credits report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_portfolio(data),
            self._md_removals(data),
            self._md_quality(data),
            self._md_projects(data),
            self._md_sbti(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render carbon credits report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_portfolio(data),
            self._html_quality(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Carbon Credits Report - ESRS E1-7</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render carbon credits report as JSON."""
        self.generated_at = _utcnow()
        credits = data.get("carbon_credits", [])
        total_tco2e = sum(c.get("tco2e", 0.0) for c in credits)
        result = {
            "template": "carbon_credits_report",
            "esrs_reference": "E1-7",
            "version": "16.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "total_credits_tco2e": round(total_tco2e, 2),
            "removal_credits_tco2e": round(
                sum(c.get("tco2e", 0.0) for c in credits if c.get("type") == "removal"), 2
            ),
            "avoidance_credits_tco2e": round(
                sum(c.get("tco2e", 0.0) for c in credits if c.get("type") == "avoidance"), 2
            ),
            "credit_count": len(credits),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_portfolio_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build portfolio summary section."""
        credits = data.get("carbon_credits", [])
        total = sum(c.get("tco2e", 0.0) for c in credits)
        total_cost = sum(c.get("cost_eur", 0.0) for c in credits)
        return {
            "title": "Carbon Credit Portfolio Summary",
            "total_credits_tco2e": round(total, 2),
            "total_cost_eur": round(total_cost, 2),
            "average_price_per_tco2e": (
                round(total_cost / total, 2) if total > 0 else 0.0
            ),
            "credit_count": len(credits),
            "retired_count": sum(1 for c in credits if c.get("status") == "retired"),
            "held_count": sum(1 for c in credits if c.get("status") == "held"),
            "offset_share_of_total_emissions_pct": round(
                data.get("offset_share_pct", 0.0), 1
            ),
        }

    def _section_removals_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build removals summary section."""
        credits = data.get("carbon_credits", [])
        removals = [c for c in credits if c.get("type") == "removal"]
        avoidance = [c for c in credits if c.get("type") == "avoidance"]
        total_removals = sum(c.get("tco2e", 0.0) for c in removals)
        total_avoidance = sum(c.get("tco2e", 0.0) for c in avoidance)
        return {
            "title": "Removal vs Avoidance Breakdown",
            "removal_tco2e": round(total_removals, 2),
            "avoidance_tco2e": round(total_avoidance, 2),
            "removal_count": len(removals),
            "avoidance_count": len(avoidance),
            "removal_methods": list({c.get("method", "") for c in removals}),
            "avoidance_methods": list({c.get("method", "") for c in avoidance}),
        }

    def _section_credit_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build credit quality section."""
        credits = data.get("carbon_credits", [])
        quality_tiers = {}
        for c in credits:
            tier = c.get("quality_tier", "unrated")
            if tier not in quality_tiers:
                quality_tiers[tier] = {"count": 0, "tco2e": 0.0}
            quality_tiers[tier]["count"] += 1
            quality_tiers[tier]["tco2e"] += c.get("tco2e", 0.0)
        return {
            "title": "Credit Quality Assessment",
            "tiers": {
                k: {"count": v["count"], "tco2e": round(v["tco2e"], 2)}
                for k, v in quality_tiers.items()
            },
            "certification_standards": list(
                {c.get("standard", "") for c in credits if c.get("standard")}
            ),
            "vintage_range": {
                "earliest": min((c.get("vintage_year", 9999) for c in credits), default=0),
                "latest": max((c.get("vintage_year", 0) for c in credits), default=0),
            },
        }

    def _section_project_details(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build project details section."""
        credits = data.get("carbon_credits", [])
        return {
            "title": "Project Details",
            "projects": [
                {
                    "project_name": c.get("project_name", ""),
                    "type": c.get("type", ""),
                    "method": c.get("method", ""),
                    "standard": c.get("standard", ""),
                    "location": c.get("location", ""),
                    "tco2e": round(c.get("tco2e", 0.0), 2),
                    "vintage_year": c.get("vintage_year", ""),
                    "registry_id": c.get("registry_id", ""),
                }
                for c in credits
            ],
        }

    def _section_sbti_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build SBTi compliance section."""
        sbti = data.get("sbti_credit_compliance", {})
        return {
            "title": "SBTi Carbon Credit Compliance",
            "beyond_value_chain_mitigation": sbti.get(
                "beyond_value_chain_mitigation", False
            ),
            "neutralization_credits_only": sbti.get(
                "neutralization_credits_only", False
            ),
            "removal_share_pct": round(sbti.get("removal_share_pct", 0.0), 1),
            "counted_toward_targets": sbti.get("counted_toward_targets", False),
            "compliance_notes": sbti.get("compliance_notes", ""),
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
            f"# Carbon Credits Report - ESRS E1-7\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E1-7 GHG Removals and GHG Mitigation Projects "
            f"Financed Through Carbon Credits"
        )

    def _md_portfolio(self, data: Dict[str, Any]) -> str:
        """Render portfolio summary markdown."""
        sec = self._section_portfolio_summary(data)
        return (
            f"## {sec['title']}\n\n"
            f"**Total Credits:** {sec['total_credits_tco2e']:,.2f} tCO2e  \n"
            f"**Total Cost:** EUR {sec['total_cost_eur']:,.2f}  \n"
            f"**Avg Price:** EUR {sec['average_price_per_tco2e']:,.2f}/tCO2e  \n"
            f"**Offset Share:** {sec['offset_share_of_total_emissions_pct']:.1f}%"
        )

    def _md_removals(self, data: Dict[str, Any]) -> str:
        """Render removals summary markdown."""
        sec = self._section_removals_summary(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Type | tCO2e | Count |\n|------|------:|------:|\n"
            f"| Removals | {sec['removal_tco2e']:,.2f} | {sec['removal_count']} |\n"
            f"| Avoidance | {sec['avoidance_tco2e']:,.2f} | {sec['avoidance_count']} |"
        )

    def _md_quality(self, data: Dict[str, Any]) -> str:
        """Render credit quality markdown."""
        sec = self._section_credit_quality(data)
        lines = [f"## {sec['title']}\n", "| Tier | Count | tCO2e |", "|------|------:|------:|"]
        for tier, vals in sec["tiers"].items():
            lines.append(f"| {tier} | {vals['count']} | {vals['tco2e']:,.2f} |")
        standards = ", ".join(sec["certification_standards"]) if sec["certification_standards"] else "N/A"
        lines.append(f"\n**Certification Standards:** {standards}")
        return "\n".join(lines)

    def _md_projects(self, data: Dict[str, Any]) -> str:
        """Render project details markdown."""
        sec = self._section_project_details(data)
        lines = [
            f"## {sec['title']}\n",
            "| Project | Type | Standard | Location | tCO2e |",
            "|---------|------|----------|----------|------:|",
        ]
        for p in sec["projects"]:
            lines.append(
                f"| {p['project_name']} | {p['type']} | {p['standard']} "
                f"| {p['location']} | {p['tco2e']:,.2f} |"
            )
        return "\n".join(lines)

    def _md_sbti(self, data: Dict[str, Any]) -> str:
        """Render SBTi compliance markdown."""
        sec = self._section_sbti_compliance(data)
        bvcm = "Yes" if sec["beyond_value_chain_mitigation"] else "No"
        counted = "Yes" if sec["counted_toward_targets"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"**BVCM Aligned:** {bvcm}  \n"
            f"**Removal Share:** {sec['removal_share_pct']:.1f}%  \n"
            f"**Counted Toward Targets:** {counted}"
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
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"<h1>Carbon Credits Report - ESRS E1-7</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_portfolio(self, data: Dict[str, Any]) -> str:
        """Render portfolio summary HTML."""
        sec = self._section_portfolio_summary(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_credits_tco2e']:,.2f} tCO2e | "
            f"Cost: EUR {sec['total_cost_eur']:,.2f}</p>"
        )

    def _html_quality(self, data: Dict[str, Any]) -> str:
        """Render credit quality HTML."""
        sec = self._section_credit_quality(data)
        rows = "".join(
            f"<tr><td>{tier}</td><td>{vals['count']}</td>"
            f"<td>{vals['tco2e']:,.2f}</td></tr>"
            for tier, vals in sec["tiers"].items()
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Tier</th><th>Count</th><th>tCO2e</th></tr>"
            f"{rows}</table>"
        )
