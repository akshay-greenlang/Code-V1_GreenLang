# -*- coding: utf-8 -*-
"""
DecarbonizationRoadmapTemplate - Decarbonization Pathway Report

Generates decarbonization pathway reports covering baseline chart,
technology options, investment timeline, annual milestones,
SBTi gap analysis, and cost-benefit analysis.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _esc(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class DecarbonizationRoadmapData(BaseModel):
    """Data model for decarbonization roadmap report."""
    company_name: str = Field(default="")
    baseline_year: int = Field(default=2020)
    target_year: int = Field(default=2030)
    baseline_emissions: float = Field(default=0.0)
    target_emissions: float = Field(default=0.0)
    target_reduction_pct: float = Field(default=0.0)
    technology_options: List[Dict[str, Any]] = Field(default_factory=list)
    investment_timeline: Dict[str, Any] = Field(default_factory=dict)
    annual_milestones: List[Dict[str, Any]] = Field(default_factory=list)
    sbti_alignment: Dict[str, Any] = Field(default_factory=dict)
    cost_benefit: Dict[str, Any] = Field(default_factory=dict)


class DecarbonizationRoadmapTemplate:
    """
    Decarbonization pathway report template.

    Generates roadmap visualization with baseline/target, technology
    MAC curve, investment timeline, SBTi alignment, and cost-benefit.
    """

    PACK_ID = "PACK-013"
    TEMPLATE_NAME = "decarbonization_roadmap"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        raise ValueError(f"Unsupported format '{fmt}'.")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        sections: List[str] = []
        name = data.get("company_name", "Manufacturing Company")
        by = data.get("baseline_year", 2020)
        ty = data.get("target_year", 2030)
        be = data.get("baseline_emissions", 0.0)
        te = data.get("target_emissions", 0.0)
        rpct = data.get("target_reduction_pct", 0.0)

        sections.append(
            f"# Decarbonization Roadmap\n\n"
            f"**Company:** {name}\n\n"
            f"**Baseline:** {by} ({be:,.0f} tCO2e) -> **Target:** {ty} ({te:,.0f} tCO2e)\n\n"
            f"**Reduction Target:** {rpct:,.1f}%\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        # Emissions pathway chart (text representation)
        milestones = data.get("annual_milestones", [])
        if milestones:
            rows = ["## Emissions Pathway\n",
                     "| Year | Emissions (tCO2e) | Reduction (%) | CAPEX (EUR) |",
                     "|------|-------------------|---------------|-------------|"]
            for m in milestones:
                rows.append(
                    f"| {m.get('year', '')} | {m.get('expected_emissions_tco2e', 0.0):,.0f} | "
                    f"{m.get('reduction_from_baseline_pct', 0.0):,.1f}% | "
                    f"{m.get('annual_capex_eur', 0.0):,.0f} |"
                )
            sections.append("\n".join(rows))

        # Technology options (MAC curve)
        techs = data.get("technology_options", [])
        if techs:
            rows = ["## Technology Options (by MAC)\n",
                     "| Technology | Category | Abatement (tCO2e) | MAC (EUR/tCO2e) | CAPEX (EUR) | Recommended |",
                     "|------------|----------|-------------------|-----------------|-------------|-------------|"]
            for t in techs:
                rec = "Yes" if t.get("recommended", False) else "No"
                rows.append(
                    f"| {t.get('technology_name', '')} | {t.get('category', '')} | "
                    f"{t.get('abatement_tco2e', 0.0):,.0f} | {t.get('mac_eur_per_tco2e', 0.0):,.0f} | "
                    f"{t.get('capex_eur', 0.0):,.0f} | {rec} |"
                )
            sections.append("\n".join(rows))

        # Investment timeline
        inv = data.get("investment_timeline", {})
        if inv:
            sections.append(
                f"## Investment Summary\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Total Investment | EUR {inv.get('total_investment', 0.0):,.0f} |\n"
                f"| Carbon Savings | EUR {inv.get('carbon_savings_eur', 0.0):,.0f} |\n"
                f"| ROI | {inv.get('roi_pct', 0.0):,.1f}% |\n"
                f"| Technologies Deployed | {inv.get('technologies_deployed', 0)} |"
            )

        # SBTi alignment
        sbti = data.get("sbti_alignment", {})
        if sbti:
            aligned = sbti.get("aligned", False)
            sections.append(
                f"## SBTi Alignment\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Pathway | {sbti.get('pathway', '1.5C')} |\n"
                f"| Required Rate | {sbti.get('minimum_annual_rate', 4.2):,.1f}%/yr |\n"
                f"| Actual Rate | {sbti.get('actual_annual_rate', 0.0):,.1f}%/yr |\n"
                f"| **Aligned** | **{'Yes' if aligned else 'No'}** |"
            )

            if not aligned:
                gap = sbti.get("gap_pct", 0.0)
                sections.append(
                    f"\n> **Gap:** {gap:,.1f} percentage points below SBTi minimum. "
                    f"Additional measures required to close the gap."
                )

        # Cost-benefit
        cb = data.get("cost_benefit", {})
        if cb:
            sections.append(
                f"## Cost-Benefit Analysis\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Total CAPEX | EUR {cb.get('total_capex', 0.0):,.0f} |\n"
                f"| Annual OPEX Savings | EUR {cb.get('annual_opex_savings', 0.0):,.0f} |\n"
                f"| Carbon Cost Avoided | EUR {cb.get('carbon_cost_avoided', 0.0):,.0f} |\n"
                f"| NPV (10yr) | EUR {cb.get('npv_10yr', 0.0):,.0f} |\n"
                f"| Payback Period | {cb.get('payback_years', 0.0):,.1f} years |"
            )

        content = "\n\n".join(sections)
        ph = self._provenance(content)
        content += f"\n\n---\n\n*Generated by GreenLang {self.PACK_ID}*\n\n**Provenance:** `{ph}`"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("company_name", "Manufacturing Company"))
        be = data.get("baseline_emissions", 0.0)
        te = data.get("target_emissions", 0.0)
        rpct = data.get("target_reduction_pct", 0.0)
        body = (
            f'<div class="section" style="text-align:center">'
            f'<h2>Decarbonization Target</h2>'
            f'<p>{be:,.0f} tCO2e -> {te:,.0f} tCO2e</p>'
            f'<p style="font-size:2em;color:#1a5276"><strong>-{rpct:,.0f}%</strong></p>'
            f'</div>'
        )
        ph = self._provenance(body)
        return self._wrap_html(f"Decarbonization Roadmap - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        report = {"report_type": self.TEMPLATE_NAME, "pack_id": self.PACK_ID,
                  "version": self.VERSION, "generated_at": self.generated_at, **data}
        report["provenance_hash"] = self._provenance(json.dumps(report, default=str, sort_keys=True))
        return report

    @staticmethod
    def _provenance(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _wrap_html(self, title: str, body: str, ph: str) -> str:
        return (
            f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
            f'<title>{_esc(title)}</title>'
            f'<style>body{{font-family:sans-serif;max-width:1000px;margin:40px auto}}'
            f'.section{{margin:20px 0;padding:15px;background:#fafafa;border-radius:6px}}</style>'
            f'</head><body><h1>{_esc(title)}</h1>{body}'
            f'<div style="margin-top:30px;font-family:monospace;font-size:0.85em">'
            f'Provenance: {ph}</div></body></html>'
        )
