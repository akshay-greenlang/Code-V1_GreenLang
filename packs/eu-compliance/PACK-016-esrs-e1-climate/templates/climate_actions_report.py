# -*- coding: utf-8 -*-
"""
ClimateActionsReportTemplate - ESRS E1-3 Climate Actions Disclosure Report

Renders action summaries, resource allocation, timeline milestones,
EU Taxonomy alignment, and progress tracking per ESRS E1-3.

Sections:
    1. Action Summary
    2. Resource Allocation
    3. Action Timeline
    4. Taxonomy Alignment
    5. Progress Tracking

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
    "action_summary",
    "resource_allocation",
    "action_timeline",
    "taxonomy_alignment",
    "progress_tracking",
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

class ClimateActionsReportTemplate:
    """
    Climate actions disclosure report template per ESRS E1-3.

    Renders climate change mitigation and adaptation actions with
    resource allocation details, phased implementation timeline,
    EU Taxonomy alignment assessment, and KPI-based progress tracking.

    Example:
        >>> tpl = ClimateActionsReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ClimateActionsReportTemplate."""
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
        if "climate_actions" not in data:
            warnings.append("climate_actions missing; will default to empty")
        if "resource_allocation" not in data:
            warnings.append("resource_allocation missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render climate actions report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_action_summary(data),
            self._md_resource_allocation(data),
            self._md_timeline(data),
            self._md_taxonomy(data),
            self._md_progress(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render climate actions report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_action_summary(data),
            self._html_resource_allocation(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Climate Actions Report - ESRS E1-3</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render climate actions report as JSON."""
        self.generated_at = utcnow()
        actions = data.get("climate_actions", [])
        result = {
            "template": "climate_actions_report",
            "esrs_reference": "E1-3",
            "version": "16.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "total_actions": len(actions),
            "mitigation_actions": sum(1 for a in actions if a.get("type") == "mitigation"),
            "adaptation_actions": sum(1 for a in actions if a.get("type") == "adaptation"),
            "total_investment_eur": data.get("total_investment_eur", 0.0),
            "expected_abatement_tco2e": data.get("expected_abatement_tco2e", 0.0),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_action_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build action summary section."""
        actions = data.get("climate_actions", [])
        mitigation = [a for a in actions if a.get("type") == "mitigation"]
        adaptation = [a for a in actions if a.get("type") == "adaptation"]
        total_abatement = sum(a.get("expected_abatement_tco2e", 0.0) for a in actions)
        return {
            "title": "Climate Action Summary",
            "total_actions": len(actions),
            "mitigation_count": len(mitigation),
            "adaptation_count": len(adaptation),
            "total_expected_abatement_tco2e": round(total_abatement, 2),
            "actions": [
                {
                    "name": a.get("name", ""),
                    "type": a.get("type", ""),
                    "status": a.get("status", ""),
                    "expected_abatement_tco2e": round(
                        a.get("expected_abatement_tco2e", 0.0), 2
                    ),
                    "start_year": a.get("start_year", ""),
                    "completion_year": a.get("completion_year", ""),
                }
                for a in actions
            ],
        }

    def _section_resource_allocation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build resource allocation section."""
        allocation = data.get("resource_allocation", {})
        items = allocation.get("items", [])
        total = sum(it.get("amount_eur", 0.0) for it in items)
        return {
            "title": "Resource Allocation",
            "total_investment_eur": round(total, 2),
            "capex_eur": round(allocation.get("capex_eur", 0.0), 2),
            "opex_eur": round(allocation.get("opex_eur", 0.0), 2),
            "items": [
                {
                    "action_name": it.get("action_name", ""),
                    "amount_eur": round(it.get("amount_eur", 0.0), 2),
                    "type": it.get("type", ""),
                    "funding_source": it.get("funding_source", ""),
                }
                for it in items
            ],
        }

    def _section_action_timeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build action timeline section."""
        actions = data.get("climate_actions", [])
        return {
            "title": "Action Implementation Timeline",
            "ongoing": [a for a in actions if a.get("status") == "ongoing"],
            "planned": [a for a in actions if a.get("status") == "planned"],
            "completed": [a for a in actions if a.get("status") == "completed"],
            "total_actions": len(actions),
        }

    def _section_taxonomy_alignment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build EU Taxonomy alignment section."""
        taxonomy = data.get("taxonomy_alignment", {})
        actions = data.get("climate_actions", [])
        aligned = [a for a in actions if a.get("taxonomy_aligned", False)]
        return {
            "title": "EU Taxonomy Alignment",
            "aligned_actions_count": len(aligned),
            "total_actions": len(actions),
            "alignment_percentage": (
                round(len(aligned) / len(actions) * 100, 1) if actions else 0.0
            ),
            "aligned_capex_eur": round(taxonomy.get("aligned_capex_eur", 0.0), 2),
            "substantial_contribution_criteria": taxonomy.get(
                "substantial_contribution_criteria", []
            ),
            "dnsh_assessment": taxonomy.get("dnsh_assessment", ""),
        }

    def _section_progress_tracking(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build progress tracking section."""
        progress = data.get("progress_tracking", {})
        kpis = progress.get("kpis", [])
        return {
            "title": "Progress Tracking",
            "overall_progress_pct": round(progress.get("overall_progress_pct", 0.0), 1),
            "reporting_period": progress.get("reporting_period", ""),
            "kpis": [
                {
                    "name": k.get("name", ""),
                    "target": k.get("target", 0.0),
                    "actual": k.get("actual", 0.0),
                    "unit": k.get("unit", ""),
                    "on_track": k.get("on_track", False),
                }
                for k in kpis
            ],
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
            f"# Climate Actions Report - ESRS E1-3\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E1-3 Actions and Resources in Relation to "
            f"Climate Change Policies"
        )

    def _md_action_summary(self, data: Dict[str, Any]) -> str:
        """Render action summary markdown."""
        sec = self._section_action_summary(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Actions:** {sec['total_actions']} "
            f"(Mitigation: {sec['mitigation_count']}, Adaptation: {sec['adaptation_count']})  \n"
            f"**Expected Abatement:** {sec['total_expected_abatement_tco2e']:,.2f} tCO2e\n",
            "| Action | Type | Status | Abatement (tCO2e) |",
            "|--------|------|--------|------------------:|",
        ]
        for a in sec["actions"]:
            lines.append(
                f"| {a['name']} | {a['type']} | {a['status']} "
                f"| {a['expected_abatement_tco2e']:,.2f} |"
            )
        return "\n".join(lines)

    def _md_resource_allocation(self, data: Dict[str, Any]) -> str:
        """Render resource allocation markdown."""
        sec = self._section_resource_allocation(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Investment:** EUR {sec['total_investment_eur']:,.2f}  \n"
            f"**CapEx:** EUR {sec['capex_eur']:,.2f} | "
            f"**OpEx:** EUR {sec['opex_eur']:,.2f}\n",
            "| Action | Amount (EUR) | Type | Source |",
            "|--------|------------:|------|--------|",
        ]
        for it in sec["items"]:
            lines.append(
                f"| {it['action_name']} | {it['amount_eur']:,.2f} "
                f"| {it['type']} | {it['funding_source']} |"
            )
        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        """Render action timeline markdown."""
        sec = self._section_action_timeline(data)
        lines = [f"## {sec['title']}\n"]
        for status_key, label in [
            ("completed", "Completed"),
            ("ongoing", "Ongoing"),
            ("planned", "Planned"),
        ]:
            items = sec[status_key]
            lines.append(f"### {label} ({len(items)})")
            for a in items:
                lines.append(f"- {a.get('name', '')} ({a.get('start_year', '')}-{a.get('completion_year', '')})")
            lines.append("")
        return "\n".join(lines)

    def _md_taxonomy(self, data: Dict[str, Any]) -> str:
        """Render taxonomy alignment markdown."""
        sec = self._section_taxonomy_alignment(data)
        return (
            f"## {sec['title']}\n\n"
            f"**Aligned Actions:** {sec['aligned_actions_count']}/{sec['total_actions']} "
            f"({sec['alignment_percentage']:.1f}%)  \n"
            f"**Aligned CapEx:** EUR {sec['aligned_capex_eur']:,.2f}"
        )

    def _md_progress(self, data: Dict[str, Any]) -> str:
        """Render progress tracking markdown."""
        sec = self._section_progress_tracking(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Overall Progress:** {sec['overall_progress_pct']:.1f}%\n",
            "| KPI | Target | Actual | Unit | On Track |",
            "|-----|-------:|-------:|------|:--------:|",
        ]
        for k in sec["kpis"]:
            track = "Yes" if k["on_track"] else "No"
            lines.append(
                f"| {k['name']} | {k['target']} | {k['actual']} "
                f"| {k['unit']} | {track} |"
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
            f"<h1>Climate Actions Report - ESRS E1-3</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_action_summary(self, data: Dict[str, Any]) -> str:
        """Render action summary HTML."""
        sec = self._section_action_summary(data)
        rows = "".join(
            f"<tr><td>{a['name']}</td><td>{a['type']}</td>"
            f"<td>{a['status']}</td><td>{a['expected_abatement_tco2e']:,.2f}</td></tr>"
            for a in sec["actions"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Action</th><th>Type</th><th>Status</th>"
            f"<th>Abatement</th></tr>{rows}</table>"
        )

    def _html_resource_allocation(self, data: Dict[str, Any]) -> str:
        """Render resource allocation HTML."""
        sec = self._section_resource_allocation(data)
        rows = "".join(
            f"<tr><td>{it['action_name']}</td><td>{it['amount_eur']:,.2f}</td></tr>"
            for it in sec["items"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: EUR {sec['total_investment_eur']:,.2f}</p>\n"
            f"<table><tr><th>Action</th><th>Amount (EUR)</th></tr>{rows}</table>"
        )
