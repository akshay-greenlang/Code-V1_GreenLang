# -*- coding: utf-8 -*-
"""
WaterPollutionReportTemplate - Water & Pollution Disclosure

Generates water and pollution disclosure reports covering water balance,
water stress assessment, pollutant inventory, IED compliance,
REACH SVHC tracking, and ESRS E2/E3 metrics.

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


class WaterPollutionReportData(BaseModel):
    """Data model for water and pollution report."""
    company_name: str = Field(default="")
    reporting_year: int = Field(default=2024)
    water_withdrawal_m3: float = Field(default=0.0)
    water_discharge_m3: float = Field(default=0.0)
    water_consumption_m3: float = Field(default=0.0)
    water_recycled_pct: float = Field(default=0.0)
    water_stress_sites: List[Dict[str, Any]] = Field(default_factory=list)
    pollutant_inventory: List[Dict[str, Any]] = Field(default_factory=list)
    ied_compliance: Dict[str, Any] = Field(default_factory=dict)
    reach_svhc: List[Dict[str, Any]] = Field(default_factory=list)
    esrs_e2_metrics: Dict[str, Any] = Field(default_factory=dict)
    esrs_e3_metrics: Dict[str, Any] = Field(default_factory=dict)


class WaterPollutionReportTemplate:
    """
    Water and pollution disclosure report template.

    Covers water balance, water stress assessment, pollutant inventory,
    IED compliance, REACH SVHC tracking, and ESRS E2/E3 metrics.
    """

    PACK_ID = "PACK-013"
    TEMPLATE_NAME = "water_pollution_report"
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

        sections.append(
            f"# Water & Pollution Disclosure\n\n"
            f"**Company:** {name} | **Year:** {year}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        # Water balance
        withdrawal = data.get("water_withdrawal_m3", 0.0)
        discharge = data.get("water_discharge_m3", 0.0)
        consumption = data.get("water_consumption_m3", 0.0)
        recycled = data.get("water_recycled_pct", 0.0)

        sections.append(
            f"## Water Balance\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Withdrawal | {withdrawal:,.0f} m3 |\n"
            f"| Total Discharge | {discharge:,.0f} m3 |\n"
            f"| Net Consumption | {consumption:,.0f} m3 |\n"
            f"| Recycled/Reused | {recycled:,.1f}% |"
        )

        # Water stress sites
        stress = data.get("water_stress_sites", [])
        if stress:
            rows = ["## Water Stress Assessment\n",
                     "| Site | Location | Stress Level | Withdrawal (m3) |",
                     "|------|----------|-------------|-----------------|"]
            for s in stress:
                rows.append(
                    f"| {s.get('site', '')} | {s.get('location', '')} | "
                    f"{s.get('stress_level', '')} | {s.get('withdrawal_m3', 0.0):,.0f} |"
                )
            sections.append("\n".join(rows))

        # Pollutant inventory
        pollutants = data.get("pollutant_inventory", [])
        if pollutants:
            rows = ["## Pollutant Inventory\n",
                     "| Pollutant | Medium | Quantity | Unit | Limit | Status |",
                     "|-----------|--------|----------|------|-------|--------|"]
            for p in pollutants:
                status = "PASS" if p.get("within_limit", True) else "EXCEED"
                rows.append(
                    f"| {p.get('pollutant', '')} | {p.get('medium', 'air')} | "
                    f"{p.get('quantity', 0.0):,.2f} | {p.get('unit', '')} | "
                    f"{p.get('limit', 0.0):,.2f} | {status} |"
                )
            sections.append("\n".join(rows))

        # IED compliance
        ied = data.get("ied_compliance", {})
        if ied:
            sections.append(
                f"## IED Compliance\n\n"
                f"| Requirement | Status |\n|-------------|--------|\n"
                f"| Permit Valid | {ied.get('permit_valid', 'N/A')} |\n"
                f"| BAT Conclusions Applied | {ied.get('bat_applied', False)} |\n"
                f"| Monitoring Plan | {ied.get('monitoring_plan', 'N/A')} |\n"
                f"| Baseline Report | {ied.get('baseline_report', False)} |"
            )

        # REACH SVHC
        svhc = data.get("reach_svhc", [])
        if svhc:
            rows = ["## REACH SVHC Tracking\n",
                     "| Substance | CAS Number | Use (tonnes) | Substitution Status |",
                     "|-----------|------------|--------------|---------------------|"]
            for s in svhc:
                rows.append(
                    f"| {s.get('substance', '')} | {s.get('cas_number', '')} | "
                    f"{s.get('use_tonnes', 0.0):,.2f} | {s.get('substitution_status', 'N/A')} |"
                )
            sections.append("\n".join(rows))

        # ESRS E2/E3
        e2 = data.get("esrs_e2_metrics", {})
        if e2:
            sections.append(
                f"## ESRS E2 - Pollution Metrics\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Total Air Pollutants | {e2.get('air_pollutants_tonnes', 0.0):,.2f} tonnes |\n"
                f"| Total Water Pollutants | {e2.get('water_pollutants_tonnes', 0.0):,.2f} tonnes |\n"
                f"| SVHC Count | {e2.get('svhc_count', 0)} |"
            )

        e3 = data.get("esrs_e3_metrics", {})
        if e3:
            sections.append(
                f"## ESRS E3 - Water & Marine Resources\n\n"
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Water Intensity (m3/unit) | {e3.get('water_intensity', 0.0):,.2f} |\n"
                f"| Sites in Stress Areas | {e3.get('stress_area_sites', 0)} |\n"
                f"| Water Policy | {e3.get('has_water_policy', False)} |"
            )

        content = "\n\n".join(sections)
        ph = self._provenance(content)
        content += f"\n\n---\n\n*Generated by GreenLang {self.PACK_ID}*\n\n**Provenance:** `{ph}`"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("company_name", "Manufacturing Company"))
        withdrawal = data.get("water_withdrawal_m3", 0.0)
        body = f'<div class="section"><h2>Water Balance</h2><p>Withdrawal: {withdrawal:,.0f} m3</p></div>'
        ph = self._provenance(body)
        return self._wrap_html(f"Water & Pollution - {name}", body, ph)

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
