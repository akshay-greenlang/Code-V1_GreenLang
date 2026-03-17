# -*- coding: utf-8 -*-
"""
ClimatePolicyReportTemplate - ESRS E1-2 Climate Policy Disclosure Report

Renders climate-related policies overview, mitigation policies, adaptation
policies, and value chain scope coverage per ESRS E1-2.

Sections:
    1. Policy Overview
    2. Mitigation Policies
    3. Adaptation Policies
    4. Value Chain Scope

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
    "policy_overview",
    "mitigation_policies",
    "adaptation_policies",
    "value_chain_scope",
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


class ClimatePolicyReportTemplate:
    """
    Climate policy disclosure report template per ESRS E1-2.

    Renders policies adopted to manage climate change mitigation and
    adaptation, including scope of coverage across the value chain,
    alignment with international frameworks, and governance oversight.

    Example:
        >>> tpl = ClimatePolicyReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ClimatePolicyReportTemplate."""
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
        if "policies" not in data:
            warnings.append("policies missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render climate policy report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_policy_overview(data),
            self._md_mitigation(data),
            self._md_adaptation(data),
            self._md_value_chain(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render climate policy report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_policy_overview(data),
            self._html_mitigation(data),
            self._html_adaptation(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Climate Policy Report - ESRS E1-2</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render climate policy report as JSON."""
        self.generated_at = _utcnow()
        policies = data.get("policies", [])
        result = {
            "template": "climate_policy_report",
            "esrs_reference": "E1-2",
            "version": "16.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "total_policies": len(policies),
            "mitigation_count": sum(
                1 for p in policies if p.get("type") == "mitigation"
            ),
            "adaptation_count": sum(
                1 for p in policies if p.get("type") == "adaptation"
            ),
            "value_chain_coverage": data.get("value_chain_coverage", {}),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_policy_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build policy overview section."""
        policies = data.get("policies", [])
        mitigation = [p for p in policies if p.get("type") == "mitigation"]
        adaptation = [p for p in policies if p.get("type") == "adaptation"]
        combined = [p for p in policies if p.get("type") == "combined"]
        return {
            "title": "Climate Policy Overview",
            "total_policies": len(policies),
            "mitigation_count": len(mitigation),
            "adaptation_count": len(adaptation),
            "combined_count": len(combined),
            "governance_body": data.get("policy_governance_body", ""),
            "last_review_date": data.get("policy_review_date", ""),
            "frameworks_referenced": data.get("frameworks_referenced", []),
        }

    def _section_mitigation_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build mitigation policies section."""
        policies = data.get("policies", [])
        mitigation = [
            p for p in policies if p.get("type") in ("mitigation", "combined")
        ]
        return {
            "title": "Climate Change Mitigation Policies",
            "policy_count": len(mitigation),
            "policies": [
                {
                    "name": p.get("name", ""),
                    "scope": p.get("scope", ""),
                    "objectives": p.get("objectives", []),
                    "adoption_date": p.get("adoption_date", ""),
                    "review_cycle": p.get("review_cycle", ""),
                    "responsible_body": p.get("responsible_body", ""),
                    "kpis": p.get("kpis", []),
                }
                for p in mitigation
            ],
        }

    def _section_adaptation_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build adaptation policies section."""
        policies = data.get("policies", [])
        adaptation = [
            p for p in policies if p.get("type") in ("adaptation", "combined")
        ]
        return {
            "title": "Climate Change Adaptation Policies",
            "policy_count": len(adaptation),
            "policies": [
                {
                    "name": p.get("name", ""),
                    "scope": p.get("scope", ""),
                    "objectives": p.get("objectives", []),
                    "adoption_date": p.get("adoption_date", ""),
                    "climate_hazards_addressed": p.get(
                        "climate_hazards_addressed", []
                    ),
                    "responsible_body": p.get("responsible_body", ""),
                }
                for p in adaptation
            ],
        }

    def _section_value_chain_scope(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build value chain scope section."""
        coverage = data.get("value_chain_coverage", {})
        return {
            "title": "Value Chain Scope",
            "own_operations": coverage.get("own_operations", False),
            "upstream": coverage.get("upstream", False),
            "downstream": coverage.get("downstream", False),
            "upstream_details": coverage.get("upstream_details", ""),
            "downstream_details": coverage.get("downstream_details", ""),
            "supplier_engagement": coverage.get("supplier_engagement", ""),
            "customer_engagement": coverage.get("customer_engagement", ""),
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
            f"# Climate Policy Report - ESRS E1-2\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E1-2 Policies Related to Climate Change Mitigation "
            f"and Adaptation"
        )

    def _md_policy_overview(self, data: Dict[str, Any]) -> str:
        """Render policy overview markdown."""
        sec = self._section_policy_overview(data)
        frameworks = ", ".join(sec["frameworks_referenced"]) if sec["frameworks_referenced"] else "N/A"
        return (
            f"## {sec['title']}\n\n"
            f"**Total Policies:** {sec['total_policies']}  \n"
            f"**Mitigation:** {sec['mitigation_count']}  \n"
            f"**Adaptation:** {sec['adaptation_count']}  \n"
            f"**Combined:** {sec['combined_count']}  \n"
            f"**Governance Body:** {sec['governance_body']}  \n"
            f"**Frameworks Referenced:** {frameworks}"
        )

    def _md_mitigation(self, data: Dict[str, Any]) -> str:
        """Render mitigation policies markdown."""
        sec = self._section_mitigation_policies(data)
        lines = [f"## {sec['title']}\n"]
        for p in sec["policies"]:
            lines.append(f"### {p['name']}")
            lines.append(f"- **Scope:** {p['scope']}")
            lines.append(f"- **Adopted:** {p['adoption_date']}")
            lines.append(f"- **Review Cycle:** {p['review_cycle']}")
            if p["objectives"]:
                lines.append("- **Objectives:**")
                for obj in p["objectives"]:
                    lines.append(f"  - {obj}")
            lines.append("")
        return "\n".join(lines)

    def _md_adaptation(self, data: Dict[str, Any]) -> str:
        """Render adaptation policies markdown."""
        sec = self._section_adaptation_policies(data)
        lines = [f"## {sec['title']}\n"]
        for p in sec["policies"]:
            lines.append(f"### {p['name']}")
            lines.append(f"- **Scope:** {p['scope']}")
            lines.append(f"- **Adopted:** {p['adoption_date']}")
            if p["climate_hazards_addressed"]:
                hazards = ", ".join(p["climate_hazards_addressed"])
                lines.append(f"- **Hazards Addressed:** {hazards}")
            lines.append("")
        return "\n".join(lines)

    def _md_value_chain(self, data: Dict[str, Any]) -> str:
        """Render value chain scope markdown."""
        sec = self._section_value_chain_scope(data)
        own = "Covered" if sec["own_operations"] else "Not Covered"
        up = "Covered" if sec["upstream"] else "Not Covered"
        down = "Covered" if sec["downstream"] else "Not Covered"
        return (
            f"## {sec['title']}\n\n"
            f"| Scope | Status |\n|-------|--------|\n"
            f"| Own Operations | {own} |\n"
            f"| Upstream | {up} |\n"
            f"| Downstream | {down} |"
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
            f"<h1>Climate Policy Report - ESRS E1-2</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_policy_overview(self, data: Dict[str, Any]) -> str:
        """Render policy overview HTML."""
        sec = self._section_policy_overview(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Type</th><th>Count</th></tr>"
            f"<tr><td>Mitigation</td><td>{sec['mitigation_count']}</td></tr>"
            f"<tr><td>Adaptation</td><td>{sec['adaptation_count']}</td></tr>"
            f"<tr><td>Combined</td><td>{sec['combined_count']}</td></tr></table>"
        )

    def _html_mitigation(self, data: Dict[str, Any]) -> str:
        """Render mitigation policies HTML."""
        sec = self._section_mitigation_policies(data)
        rows = "".join(
            f"<tr><td>{p['name']}</td><td>{p['scope']}</td>"
            f"<td>{p['adoption_date']}</td></tr>"
            for p in sec["policies"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Policy</th><th>Scope</th><th>Adopted</th></tr>"
            f"{rows}</table>"
        )

    def _html_adaptation(self, data: Dict[str, Any]) -> str:
        """Render adaptation policies HTML."""
        sec = self._section_adaptation_policies(data)
        rows = "".join(
            f"<tr><td>{p['name']}</td><td>{p['scope']}</td></tr>"
            for p in sec["policies"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Policy</th><th>Scope</th></tr>"
            f"{rows}</table>"
        )
