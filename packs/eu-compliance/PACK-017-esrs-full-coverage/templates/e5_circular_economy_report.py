# -*- coding: utf-8 -*-
"""
E5CircularReport - ESRS E5 Resource Use and Circular Economy Disclosure Report

Renders circular economy disclosures covering policies (E5-1), actions and
resources (E5-2), targets (E5-3), resource inflows (E5-4), resource
outflows (E5-5), and anticipated financial effects (E5-6).

Sections:
    1. Policies (E5-1)
    2. Actions & Resources (E5-2)
    3. Targets (E5-3)
    4. Resource Inflows (E5-4)
    5. Resource Outflows (E5-5)
    6. Financial Effects (E5-6)

Author: GreenLang Team
Version: 17.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SECTIONS: List[str] = [
    "policies",
    "actions_resources",
    "targets",
    "resource_inflows",
    "resource_outflows",
    "financial_effects",
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


class E5CircularReport:
    """
    ESRS E5 Resource Use and Circular Economy disclosure report template.

    Covers all six disclosure requirements under ESRS E5: policies related
    to resource use and circular economy (E5-1), actions and resources
    (E5-2), resource use and circular economy targets (E5-3), resource
    inflows including material composition and recycled content (E5-4),
    resource outflows including waste generation and management (E5-5),
    and anticipated financial effects (E5-6).

    ESRS References: ESRS E5-1 through E5-6

    Example:
        >>> tpl = E5CircularReport()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize E5CircularReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> str:
        """Render ESRS E5 circular economy report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_policies(data),
            self._md_actions(data),
            self._md_targets(data),
            self._md_inflows(data),
            self._md_outflows(data),
            self._md_financial(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> str:
        """Render ESRS E5 circular economy report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_inflows(data),
            self._html_outflows(data),
            self._html_targets(data),
            self._html_financial(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>ESRS E5 Circular Economy Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Render ESRS E5 circular economy report as JSON."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "e5_circular_economy_report",
            "esrs_reference": "E5-1 to E5-6",
            "version": "17.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "policies": self._section_policies(data),
            "actions_resources": self._section_actions(data),
            "targets": self._section_targets(data),
            "resource_inflows": self._section_inflows(data),
            "resource_outflows": self._section_outflows(data),
            "financial_effects": self._section_financial(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

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
        if "material_inflows_tonnes" not in data:
            warnings.append("material_inflows_tonnes missing; will default to 0")
        if "waste_generated_tonnes" not in data:
            warnings.append("waste_generated_tonnes missing; will default to 0")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build circular economy policies section (E5-1)."""
        policies = data.get("circular_policies", [])
        return {
            "title": "Policies Related to Resource Use and Circular Economy (E5-1)",
            "policy_count": len(policies),
            "policies": [
                {
                    "name": p.get("name", ""),
                    "scope": p.get("scope", ""),
                    "waste_hierarchy_applied": p.get("waste_hierarchy_applied", False),
                    "design_for_circularity": p.get("design_for_circularity", False),
                }
                for p in policies
            ],
        }

    def _section_actions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build actions and resources section (E5-2)."""
        actions = data.get("circular_actions", [])
        return {
            "title": "Actions and Resources (E5-2)",
            "action_count": len(actions),
            "actions": [
                {
                    "description": a.get("description", ""),
                    "type": a.get("type", ""),
                    "status": a.get("status", ""),
                    "investment_eur": a.get("investment_eur", 0.0),
                    "expected_waste_reduction_pct": a.get("expected_waste_reduction_pct", 0.0),
                }
                for a in actions
            ],
            "total_investment_eur": sum(a.get("investment_eur", 0.0) for a in actions),
        }

    def _section_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build circular economy targets section (E5-3)."""
        targets = data.get("circular_targets", [])
        return {
            "title": "Resource Use and Circular Economy Targets (E5-3)",
            "target_count": len(targets),
            "targets": [
                {
                    "metric": t.get("metric", ""),
                    "base_year": t.get("base_year", ""),
                    "target_year": t.get("target_year", ""),
                    "base_value": t.get("base_value", 0.0),
                    "target_value": t.get("target_value", 0.0),
                    "current_value": t.get("current_value", 0.0),
                    "unit": t.get("unit", ""),
                    "progress_pct": t.get("progress_pct", 0.0),
                }
                for t in targets
            ],
        }

    def _section_inflows(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build resource inflows section (E5-4)."""
        materials = data.get("material_inflows", [])
        total_tonnes = data.get("material_inflows_tonnes", 0.0)
        recycled_tonnes = data.get("recycled_content_tonnes", 0.0)
        return {
            "title": "Resource Inflows (E5-4)",
            "total_material_inflow_tonnes": total_tonnes,
            "recycled_content_tonnes": recycled_tonnes,
            "recycled_content_pct": round(
                recycled_tonnes / total_tonnes * 100, 1
            ) if total_tonnes > 0 else 0.0,
            "biological_material_tonnes": data.get("biological_material_tonnes", 0.0),
            "secondary_raw_material_tonnes": data.get("secondary_raw_material_tonnes", 0.0),
            "materials": [
                {
                    "name": m.get("name", ""),
                    "tonnes": m.get("tonnes", 0.0),
                    "renewable": m.get("renewable", False),
                    "recycled_pct": m.get("recycled_pct", 0.0),
                }
                for m in materials
            ],
        }

    def _section_outflows(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build resource outflows section (E5-5)."""
        waste = data.get("waste_streams", [])
        total_waste = data.get("waste_generated_tonnes", 0.0)
        recycled = data.get("waste_recycled_tonnes", 0.0)
        landfill = data.get("waste_landfill_tonnes", 0.0)
        incinerated = data.get("waste_incinerated_tonnes", 0.0)
        return {
            "title": "Resource Outflows (E5-5)",
            "total_waste_tonnes": total_waste,
            "waste_recycled_tonnes": recycled,
            "waste_landfill_tonnes": landfill,
            "waste_incinerated_tonnes": incinerated,
            "waste_other_tonnes": total_waste - recycled - landfill - incinerated,
            "recycling_rate_pct": round(
                recycled / total_waste * 100, 1
            ) if total_waste > 0 else 0.0,
            "hazardous_waste_tonnes": data.get("hazardous_waste_tonnes", 0.0),
            "radioactive_waste_tonnes": data.get("radioactive_waste_tonnes", 0.0),
            "waste_streams": [
                {
                    "category": w.get("category", ""),
                    "tonnes": w.get("tonnes", 0.0),
                    "treatment": w.get("treatment", ""),
                    "hazardous": w.get("hazardous", False),
                }
                for w in waste
            ],
            "products_designed_for_circularity_pct": data.get(
                "products_designed_for_circularity_pct", 0.0
            ),
        }

    def _section_financial(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build financial effects section (E5-6)."""
        return {
            "title": "Anticipated Financial Effects (E5-6)",
            "financial_risks_eur": data.get("circular_financial_risks_eur", 0.0),
            "financial_opportunities_eur": data.get("circular_financial_opportunities_eur", 0.0),
            "waste_cost_savings_eur": data.get("waste_cost_savings_eur", 0.0),
            "circular_revenue_eur": data.get("circular_revenue_eur", 0.0),
            "time_horizon": data.get("financial_time_horizon", "short/medium/long-term"),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# ESRS E5 Resource Use and Circular Economy Report\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E5-1 through E5-6"
        )

    def _md_policies(self, data: Dict[str, Any]) -> str:
        """Render policies section as markdown."""
        sec = self._section_policies(data)
        lines = [f"## {sec['title']}\n"]
        for p in sec["policies"]:
            flags = []
            if p["waste_hierarchy_applied"]:
                flags.append("waste hierarchy")
            if p["design_for_circularity"]:
                flags.append("design for circularity")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            lines.append(f"- **{p['name']}**{flag_str}: {p['scope']}")
        return "\n".join(lines)

    def _md_actions(self, data: Dict[str, Any]) -> str:
        """Render actions section as markdown."""
        sec = self._section_actions(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Investment:** EUR {sec['total_investment_eur']:,.0f}\n",
            "| Action | Status | Waste Reduction | Investment (EUR) |",
            "|--------|--------|----------------:|-----------------:|",
        ]
        for a in sec["actions"]:
            lines.append(
                f"| {a['description']} | {a['status']} "
                f"| {a['expected_waste_reduction_pct']:.1f}% | {a['investment_eur']:,.0f} |"
            )
        return "\n".join(lines)

    def _md_targets(self, data: Dict[str, Any]) -> str:
        """Render targets section as markdown."""
        sec = self._section_targets(data)
        lines = [
            f"## {sec['title']}\n",
            "| Metric | Target Year | Current | Unit | Progress |",
            "|--------|-------------|--------:|------|--------:|",
        ]
        for t in sec["targets"]:
            lines.append(
                f"| {t['metric']} | {t['target_year']} "
                f"| {t['current_value']:,.1f} | {t['unit']} | {t['progress_pct']:.1f}% |"
            )
        return "\n".join(lines)

    def _md_inflows(self, data: Dict[str, Any]) -> str:
        """Render resource inflows as markdown."""
        sec = self._section_inflows(data)
        lines = [
            f"## {sec['title']}\n",
            f"| Metric | Value |\n|--------|------:|\n"
            f"| Total Material Inflow | {sec['total_material_inflow_tonnes']:,.1f} t |\n"
            f"| Recycled Content | {sec['recycled_content_tonnes']:,.1f} t "
            f"({sec['recycled_content_pct']:.1f}%) |\n"
            f"| Biological Material | {sec['biological_material_tonnes']:,.1f} t |\n"
            f"| Secondary Raw Materials | {sec['secondary_raw_material_tonnes']:,.1f} t |",
        ]
        return "\n".join(lines)

    def _md_outflows(self, data: Dict[str, Any]) -> str:
        """Render resource outflows as markdown."""
        sec = self._section_outflows(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Waste Category | Tonnes |\n|----------------|-------:|\n"
            f"| Total Waste | {sec['total_waste_tonnes']:,.1f} |\n"
            f"| Recycled | {sec['waste_recycled_tonnes']:,.1f} "
            f"({sec['recycling_rate_pct']:.1f}%) |\n"
            f"| Landfill | {sec['waste_landfill_tonnes']:,.1f} |\n"
            f"| Incinerated | {sec['waste_incinerated_tonnes']:,.1f} |\n"
            f"| Hazardous | {sec['hazardous_waste_tonnes']:,.1f} |\n\n"
            f"Products designed for circularity: "
            f"{sec['products_designed_for_circularity_pct']:.1f}%"
        )

    def _md_financial(self, data: Dict[str, Any]) -> str:
        """Render financial effects as markdown."""
        sec = self._section_financial(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Category | EUR |\n|----------|----:|\n"
            f"| Financial Risks | {sec['financial_risks_eur']:,.0f} |\n"
            f"| Financial Opportunities | {sec['financial_opportunities_eur']:,.0f} |\n"
            f"| Waste Cost Savings | {sec['waste_cost_savings_eur']:,.0f} |\n"
            f"| Circular Revenue | {sec['circular_revenue_eur']:,.0f} |"
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
            ".report{max-width:960px;margin:auto}"
            "h1{color:#e65100;border-bottom:2px solid #e65100;padding-bottom:.3em}"
            "h2{color:#ef6c00;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#fff3e0}"
            ".circular{color:#2e7d32;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>ESRS E5 Resource Use and Circular Economy Report</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> "
            f"| {data.get('reporting_year', '')}</p>"
        )

    def _html_inflows(self, data: Dict[str, Any]) -> str:
        """Render inflows HTML."""
        sec = self._section_inflows(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Total Inflow</td><td>{sec['total_material_inflow_tonnes']:,.1f} t</td></tr>"
            f"<tr><td>Recycled Content</td>"
            f"<td>{sec['recycled_content_pct']:.1f}%</td></tr></table>"
        )

    def _html_outflows(self, data: Dict[str, Any]) -> str:
        """Render outflows HTML."""
        sec = self._section_outflows(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Category</th><th>Tonnes</th></tr>"
            f"<tr><td>Total Waste</td><td>{sec['total_waste_tonnes']:,.1f}</td></tr>"
            f"<tr><td>Recycled</td><td>{sec['waste_recycled_tonnes']:,.1f}</td></tr>"
            f"<tr><td>Landfill</td><td>{sec['waste_landfill_tonnes']:,.1f}</td></tr>"
            f"<tr><td>Hazardous</td><td>{sec['hazardous_waste_tonnes']:,.1f}</td></tr></table>"
        )

    def _html_targets(self, data: Dict[str, Any]) -> str:
        """Render targets HTML."""
        sec = self._section_targets(data)
        rows = "".join(
            f"<tr><td>{t['metric']}</td><td>{t['target_year']}</td>"
            f"<td>{t['progress_pct']:.1f}%</td></tr>"
            for t in sec["targets"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Metric</th><th>Target Year</th>"
            f"<th>Progress</th></tr>{rows}</table>"
        )

    def _html_financial(self, data: Dict[str, Any]) -> str:
        """Render financial effects HTML."""
        sec = self._section_financial(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Category</th><th>EUR</th></tr>"
            f"<tr><td>Financial Risks</td><td>{sec['financial_risks_eur']:,.0f}</td></tr>"
            f"<tr><td>Opportunities</td><td>{sec['financial_opportunities_eur']:,.0f}</td></tr>"
            f"<tr><td>Circular Revenue</td><td>{sec['circular_revenue_eur']:,.0f}</td></tr>"
            f"</table>"
        )
