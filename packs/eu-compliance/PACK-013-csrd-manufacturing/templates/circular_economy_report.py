# -*- coding: utf-8 -*-
"""
CircularEconomyReportTemplate - Circular Economy Metrics Report

Generates circular economy metrics reports covering material flows,
MCI score, recycled content, waste hierarchy, EPR compliance,
critical raw materials, and product recyclability.

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


class CircularEconomyReportData(BaseModel):
    """Data model for circular economy report."""
    company_name: str = Field(default="")
    reporting_year: int = Field(default=2024)
    mci_score: float = Field(default=0.0)
    recycled_content_pct: float = Field(default=0.0)
    waste_diversion_pct: float = Field(default=0.0)
    material_flows: Dict[str, Any] = Field(default_factory=dict)
    waste_hierarchy: Dict[str, float] = Field(default_factory=dict)
    epr_compliance: Dict[str, Any] = Field(default_factory=dict)
    crm_usage: List[Dict[str, Any]] = Field(default_factory=list)
    product_recyclability: List[Dict[str, Any]] = Field(default_factory=list)


class CircularEconomyReportTemplate:
    """
    Circular economy metrics report template.

    Covers material circularity indicator, waste hierarchy analysis,
    EPR compliance status, CRM tracking, and ESRS E5 alignment.
    """

    PACK_ID = "PACK-013"
    TEMPLATE_NAME = "circular_economy_report"
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
        year = data.get("reporting_year", 2024)
        mci = data.get("mci_score", 0.0)
        rc = data.get("recycled_content_pct", 0.0)
        wd = data.get("waste_diversion_pct", 0.0)

        sections.append(
            f"# Circular Economy Report\n\n"
            f"**Company:** {name} | **Year:** {year}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        sections.append(
            f"## Key Metrics\n\n"
            f"| Metric | Value | Rating |\n|--------|-------|--------|\n"
            f"| **MCI Score** | **{mci:.2f}** | {self._rating(mci)} |\n"
            f"| Recycled Content | {rc:.1f}% | {'Good' if rc >= 30 else 'Needs Improvement'} |\n"
            f"| Waste Diversion | {wd:.1f}% | {'Good' if wd >= 80 else 'Needs Improvement'} |"
        )

        flows = data.get("material_flows", {})
        if flows:
            sections.append(
                f"## Material Flows\n\n"
                f"| Flow | Value (kg) |\n|------|------------|\n"
                f"| Total Input | {flows.get('total_input_kg', 0.0):,.0f} |\n"
                f"| Virgin Input | {flows.get('virgin_input_kg', 0.0):,.0f} |\n"
                f"| Recycled Input | {flows.get('recycled_input_kg', 0.0):,.0f} |\n"
                f"| Total Output | {flows.get('total_output_kg', 0.0):,.0f} |"
            )

        hierarchy = data.get("waste_hierarchy", {})
        if hierarchy:
            rows = ["## Waste Hierarchy\n",
                     "| Method | Mass (kg) | Share (%) |",
                     "|--------|-----------|-----------|"]
            total_waste = sum(hierarchy.values())
            for method in ["reuse", "recycle", "recovery", "incineration", "landfill"]:
                mass = hierarchy.get(method, 0.0)
                pct = (mass / total_waste * 100) if total_waste > 0 else 0.0
                rows.append(f"| {method.title()} | {mass:,.0f} | {pct:,.1f}% |")
            sections.append("\n".join(rows))

        epr = data.get("epr_compliance", {})
        if epr:
            sections.append(
                f"## EPR Compliance\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Total Schemes | {epr.get('total_schemes', 0)} |\n"
                f"| Compliant | {epr.get('compliant', 0)} |\n"
                f"| Compliance Rate | {epr.get('compliance_rate', 0.0):.1f}% |"
            )

        crm = data.get("crm_usage", [])
        if crm:
            rows = ["## Critical Raw Materials\n",
                     "| Material | Mass (kg) | Recycled (%) | Substitution Plan |",
                     "|----------|-----------|--------------|-------------------|"]
            for c in crm:
                rows.append(
                    f"| {c.get('material', '')} | {c.get('mass_kg', 0.0):,.0f} | "
                    f"{c.get('recycled_pct', 0.0):,.0f}% | {c.get('substitution', 'N/A')} |"
                )
            sections.append("\n".join(rows))

        recyclability = data.get("product_recyclability", [])
        if recyclability:
            rows = ["## Product Recyclability\n",
                     "| Product | Recyclability (%) | Design for Disassembly |",
                     "|---------|-------------------|------------------------|"]
            for p in recyclability:
                rows.append(
                    f"| {p.get('product', '')} | {p.get('recyclability_pct', 0.0):,.0f}% | "
                    f"{p.get('design_for_disassembly', False)} |"
                )
            sections.append("\n".join(rows))

        content = "\n\n".join(sections)
        ph = self._provenance(content)
        content += f"\n\n---\n\n*Generated by GreenLang {self.PACK_ID}*\n\n**Provenance:** `{ph}`"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("company_name", "Manufacturing Company"))
        mci = data.get("mci_score", 0.0)
        body = f'<div class="section"><h2>MCI Score: {mci:.2f}</h2><p>{self._rating(mci)}</p></div>'
        ph = self._provenance(body)
        return self._wrap_html(f"Circular Economy - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        report = {"report_type": self.TEMPLATE_NAME, "pack_id": self.PACK_ID,
                  "version": self.VERSION, "generated_at": self.generated_at, **data}
        report["provenance_hash"] = self._provenance(json.dumps(report, default=str, sort_keys=True))
        return report

    @staticmethod
    def _rating(mci: float) -> str:
        if mci >= 0.8:
            return "EXCELLENT"
        if mci >= 0.6:
            return "GOOD"
        if mci >= 0.4:
            return "MODERATE"
        return "LOW"

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
