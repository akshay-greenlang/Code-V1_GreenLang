# -*- coding: utf-8 -*-
"""
BATComplianceReportTemplate - BAT Compliance Assessment Report

Generates BAT compliance assessment reports covering facility overview,
parameter compliance table, gap analysis, transformation plan timeline,
investment requirements, and penalty risk assessment.

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


class BATComplianceReportData(BaseModel):
    """Data model for BAT compliance report."""
    company_name: str = Field(default="")
    reporting_year: int = Field(default=2024)
    compliance_status: str = Field(default="UNKNOWN")
    parameter_results: List[Dict[str, Any]] = Field(default_factory=list)
    gap_analysis: List[Dict[str, Any]] = Field(default_factory=list)
    transformation_plan: List[Dict[str, Any]] = Field(default_factory=list)
    investment_summary: Dict[str, Any] = Field(default_factory=dict)
    penalty_risk: Dict[str, Any] = Field(default_factory=dict)
    applicable_brefs: List[str] = Field(default_factory=list)


class BATComplianceReportTemplate:
    """
    BAT compliance assessment report template.

    Covers IED/BREF parameter compliance, gap analysis,
    transformation planning, investment requirements, and penalty risk.
    """

    PACK_ID = "PACK-013"
    TEMPLATE_NAME = "bat_compliance_report"
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
        status = data.get("compliance_status", "UNKNOWN")

        status_icon = {"COMPLIANT": "[PASS]", "PARTIAL": "[PARTIAL]", "NON_COMPLIANT": "[FAIL]"}.get(status, "[?]")

        sections.append(
            f"# BAT Compliance Assessment\n\n"
            f"**Company:** {name} | **Year:** {year}\n\n"
            f"**Overall Status:** {status_icon} **{status}**\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        brefs = data.get("applicable_brefs", [])
        if brefs:
            sections.append(
                f"## Applicable BREFs\n\n" + "\n".join(f"- {b}" for b in brefs)
            )

        params = data.get("parameter_results", [])
        if params:
            rows = ["## Parameter Compliance\n",
                     "| Parameter | Measured | BAT-AEL Range | Unit | Status |",
                     "|-----------|----------|---------------|------|--------|"]
            for p in params:
                s = p.get("status", "")
                badge = {"COMPLIANT": "PASS", "BEST_PRACTICE": "BEST", "NON_COMPLIANT": "FAIL", "MARGINAL": "WARN"}.get(s, s)
                rows.append(
                    f"| {p.get('parameter_name', '')} | {p.get('measured_value', 0.0):,.2f} | "
                    f"{p.get('bat_ael_lower', 0.0)}-{p.get('bat_ael_upper', 0.0)} | "
                    f"{p.get('unit', '')} | {badge} |"
                )
            sections.append("\n".join(rows))

        gaps = data.get("gap_analysis", [])
        if gaps:
            rows = ["## Gap Analysis\n",
                     "| Parameter | Exceedance (%) | Severity | Facility |",
                     "|-----------|----------------|----------|----------|"]
            for g in gaps:
                rows.append(
                    f"| {g.get('parameter_name', '')} | {g.get('exceedance_pct', 0.0):,.1f}% | "
                    f"{g.get('severity', '')} | {g.get('facility_id', '')} |"
                )
            sections.append("\n".join(rows))

        plan = data.get("transformation_plan", [])
        if plan:
            rows = ["## Transformation Plan\n",
                     "| Action | Parameter | Priority | Cost (EUR) | Timeline |",
                     "|--------|-----------|----------|------------|----------|"]
            for a in plan:
                rows.append(
                    f"| {a.get('action_id', '')} | {a.get('parameter_name', '')} | "
                    f"{a.get('priority', '')} | {a.get('estimated_cost_eur', 0.0):,.0f} | "
                    f"{a.get('timeline_months', 0)} months |"
                )
            sections.append("\n".join(rows))

        inv = data.get("investment_summary", {})
        if inv:
            sections.append(
                f"## Investment Requirements\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Total Required | EUR {inv.get('total_required', 0.0):,.0f} |\n"
                f"| Budget Available | EUR {inv.get('budget_available', 0.0):,.0f} |\n"
                f"| Budget Gap | EUR {inv.get('budget_gap', 0.0):,.0f} |"
            )

        risk = data.get("penalty_risk", {})
        if risk:
            sections.append(
                f"## Penalty Risk\n\n"
                f"**Risk Level:** {risk.get('risk_level', 'UNKNOWN')}\n\n"
                f"**Estimated Total Penalty:** EUR {risk.get('estimated_total_eur', 0.0):,.0f}"
            )

        content = "\n\n".join(sections)
        ph = self._provenance(content)
        content += f"\n\n---\n\n*Generated by GreenLang {self.PACK_ID}*\n\n**Provenance:** `{ph}`"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("company_name", "Manufacturing Company"))
        status = data.get("compliance_status", "UNKNOWN")
        color = {"COMPLIANT": "#27ae60", "NON_COMPLIANT": "#e74c3c"}.get(status, "#f39c12")
        body = f'<div class="section"><h2 style="color:{color}">Status: {_esc(status)}</h2></div>'
        ph = self._provenance(body)
        return self._wrap_html(f"BAT Compliance - {name}", body, ph)

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
