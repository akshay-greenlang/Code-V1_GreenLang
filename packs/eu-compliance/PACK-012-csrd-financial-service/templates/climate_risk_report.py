# -*- coding: utf-8 -*-
"""
ClimateRiskReportTemplate - TCFD-aligned Climate Risk Report

Generates TCFD-aligned climate risk reports with physical risk heatmaps,
transition risk sector analysis, NGFS scenario results, and stress test
impact quantification.

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
    """Escape HTML special characters."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )



class ClimateRiskReportData(BaseModel):
    """Data for climate risk report."""
    institution_name: str = Field(default="")
    reporting_date: str = Field(default="")
    total_exposure_eur: float = Field(default=0.0)
    scenarios_tested: int = Field(default=0)
    scenario_impacts: List[Dict[str, Any]] = Field(default_factory=list)
    max_credit_loss_pct: float = Field(default=0.0)
    max_market_value_impact_pct: float = Field(default=0.0)
    high_risk_exposure_pct: float = Field(default=0.0)
    physical_risk_exposure_pct: float = Field(default=0.0)
    transition_risk_exposure_pct: float = Field(default=0.0)
    sector_risk_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    country_risk_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    governance_description: str = Field(default="")
    strategy_description: str = Field(default="")
    risk_management_description: str = Field(default="")


class ClimateRiskReportTemplate:
    """TCFD-aligned climate risk report template."""

    PACK_ID = "PACK-012"
    TEMPLATE_NAME = "climate_risk_report"
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
        name = data.get("institution_name", "Financial Institution")
        date = data.get("reporting_date", "")
        total_exp = data.get("total_exposure_eur", 0.0)
        scenarios = data.get("scenarios_tested", 0)
        max_cl = data.get("max_credit_loss_pct", 0.0)
        max_mv = data.get("max_market_value_impact_pct", 0.0)
        hr_pct = data.get("high_risk_exposure_pct", 0.0)
        phys_pct = data.get("physical_risk_exposure_pct", 0.0)
        trans_pct = data.get("transition_risk_exposure_pct", 0.0)

        sections.append(
            f"# Climate Risk Report (TCFD-aligned)\n\n"
            f"**Institution:** {name}\n\n"
            f"**Reporting Date:** {date}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

        # Risk overview
        sections.append(
            f"## Risk Overview\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Total Exposure | EUR {total_exp:,.0f} |\n"
            f"| Scenarios Tested | {scenarios} |\n"
            f"| Max Credit Loss | {max_cl:.2f}% |\n"
            f"| Max Market Value Impact | {max_mv:.2f}% |\n"
            f"| Climate-Sensitive Exposure | {hr_pct:.1f}% |\n"
            f"| Physical Risk Exposure | {phys_pct:.1f}% |\n"
            f"| Transition Risk Exposure | {trans_pct:.1f}% |"
        )

        # Scenario results
        impacts = data.get("scenario_impacts", [])
        if impacts:
            rows = ["## Scenario Analysis Results\n",
                     "| Scenario | Physical Impact | Transition Impact | Total Impact | Credit Loss (%) |",
                     "|----------|----------------|-------------------|-------------|-----------------|"]
            for si in impacts:
                rows.append(
                    f"| {si.get('scenario', '')} | EUR {si.get('physical_impact_eur', 0.0):,.0f} | "
                    f"EUR {si.get('transition_impact_eur', 0.0):,.0f} | "
                    f"EUR {si.get('total_impact_eur', 0.0):,.0f} | {si.get('credit_loss_pct', 0.0):.2f}% |"
                )
            sections.append("\n".join(rows))

        # Risk heatmap (ASCII)
        sections.append(
            f"## Risk Heatmap\n\n```\n"
            f"Physical Risk:    [{'!' * int(phys_pct):<50s}] {phys_pct:.1f}%\n"
            f"Transition Risk:  [{'!' * int(trans_pct):<50s}] {trans_pct:.1f}%\n"
            f"Overall:          [{'!' * int(hr_pct):<50s}] {hr_pct:.1f}%\n"
            f"```"
        )

        # Sector breakdown
        sec_risk = data.get("sector_risk_breakdown", [])
        if sec_risk:
            rows = ["## Sector Risk Breakdown\n",
                     "| Sector | Exposure | Risk Level |",
                     "|--------|----------|-----------|"]
            for sr in sec_risk[:10]:
                rows.append(
                    f"| {sr.get('sector', '')} | EUR {sr.get('exposure', 0.0):,.0f} | "
                    f"{sr.get('risk_level', 'Medium')} |"
                )
            sections.append("\n".join(rows))

        # TCFD pillars
        for pillar, key in [("Governance", "governance_description"),
                             ("Strategy", "strategy_description"),
                             ("Risk Management", "risk_management_description")]:
            desc = data.get(key, "")
            if desc:
                sections.append(f"## {pillar}\n\n{desc}")

        content = "\n\n".join(s for s in sections if s)
        ph = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(ph)
        content += f"\n\n<!-- provenance_hash: {ph} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("institution_name", "Financial Institution"))
        max_cl = data.get("max_credit_loss_pct", 0.0)
        body = (
            f'<div class="section"><h2>Risk Overview</h2>'
            f'<p>Max Credit Loss: {max_cl:.2f}%</p>'
            f'<p>Scenarios Tested: {data.get("scenarios_tested", 0)}</p></div>'
        )
        ph = self._compute_provenance_hash(body)
        return self._wrap_html(f"Climate Risk Report - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "report_type": "climate_risk_tcfd",
            "pack_id": self.PACK_ID, "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION, "generated_at": self.generated_at,
            "institution_name": data.get("institution_name", ""),
            "reporting_date": data.get("reporting_date", ""),
            "risk_overview": {
                "total_exposure_eur": data.get("total_exposure_eur", 0.0),
                "scenarios_tested": data.get("scenarios_tested", 0),
                "max_credit_loss_pct": data.get("max_credit_loss_pct", 0.0),
                "max_market_value_impact_pct": data.get("max_market_value_impact_pct", 0.0),
                "high_risk_exposure_pct": data.get("high_risk_exposure_pct", 0.0),
            },
            "scenario_impacts": data.get("scenario_impacts", []),
            "sector_risk_breakdown": data.get("sector_risk_breakdown", []),
        }
        cs = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(cs)
        return report

    def _md_footer(self, provenance_hash: str) -> str:
        return (
            "---\n\n"
            f"*Report generated by GreenLang {self.PACK_ID} | "
            f"Template: {self.TEMPLATE_NAME} v{self.VERSION}*\n\n"
            f"**Provenance Hash (SHA-256):** `{provenance_hash}`"
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>body{font-family:Arial,sans-serif;margin:40px auto;max-width:1000px}"
            "h1{color:#c0392b;border-bottom:3px solid #e74c3c}h2{color:#c0392b}"
            ".section{margin:20px 0;padding:15px;background:#fdf2f0;border:1px solid #f5b7b1;border-radius:6px}"
            ".data-table{width:100%;border-collapse:collapse}.data-table td,.data-table th{padding:8px;border:1px solid #ddd}"
            ".data-table th{background:#c0392b;color:#fff}"
            ".provenance{margin-top:30px;padding:10px;background:#eaf2f8;font-family:monospace;font-size:.85em}"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n{body}\n"
            f'<div class="provenance">Provenance: {provenance_hash}</div>\n'
            f"<!-- provenance_hash: {provenance_hash} -->\n</body>\n</html>"
        )

    @staticmethod
    def _compute_provenance_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
