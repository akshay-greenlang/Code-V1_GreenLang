# -*- coding: utf-8 -*-
"""
FinancedEmissionsDashboard - Portfolio Emissions Dashboard

Generates a portfolio emissions dashboard with WACI waterfall, asset class
drill-down, and data quality traffic light visualization.

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



class FinancedEmissionsDashboardData(BaseModel):
    """Data for financed emissions dashboard."""
    institution_name: str = Field(default="")
    reporting_period: str = Field(default="")
    total_financed_emissions_tco2e: float = Field(default=0.0)
    scope1_financed: float = Field(default=0.0)
    scope2_financed: float = Field(default=0.0)
    portfolio_value_eur: float = Field(default=0.0)
    waci: float = Field(default=0.0)
    weighted_data_quality_score: float = Field(default=5.0)
    target_data_quality: float = Field(default=3.0)
    asset_class_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    sector_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    yoy_comparison: Dict[str, Any] = Field(default_factory=dict)
    quality_distribution: Dict[str, int] = Field(default_factory=dict)
    top_emitters: List[Dict[str, Any]] = Field(default_factory=list)


class FinancedEmissionsDashboard:
    """Portfolio financed emissions dashboard template."""

    PACK_ID = "PACK-012"
    TEMPLATE_NAME = "financed_emissions_dashboard"
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
        period = data.get("reporting_period", "")
        total = data.get("total_financed_emissions_tco2e", 0.0)
        s1 = data.get("scope1_financed", 0.0)
        s2 = data.get("scope2_financed", 0.0)
        pv = data.get("portfolio_value_eur", 0.0)
        waci = data.get("waci", 0.0)
        wdq = data.get("weighted_data_quality_score", 5.0)
        tdq = data.get("target_data_quality", 3.0)

        sections.append(
            f"# Financed Emissions Dashboard\n\n"
            f"**{name}** | **Period:** {period}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}"
        )

        # KPI gauges
        dq_status = "GREEN" if wdq <= tdq else ("AMBER" if wdq <= tdq + 1 else "RED")
        sections.append(
            f"## Key Metrics\n\n"
            f"```\n"
            f"  Total Financed Emissions:  {total:>12,.2f} tCO2e\n"
            f"  Scope 1 (financed):        {s1:>12,.2f} tCO2e\n"
            f"  Scope 2 (financed):        {s2:>12,.2f} tCO2e\n"
            f"  Portfolio Value:       EUR {pv:>12,.0f}\n"
            f"  WACI:                      {waci:>12,.4f} tCO2e/EUR M\n"
            f"  Data Quality:              {wdq:>12.2f} / 5.0  [{dq_status}]\n"
            f"```"
        )

        # WACI waterfall
        sections.append(
            f"## WACI Waterfall\n\n```\n"
            f"  Scope 1 contrib:  [{'#' * max(int(s1 / max(total, 1) * 40), 1):<40s}] "
            f"{s1 / max(total, 1) * 100:.1f}%\n"
            f"  Scope 2 contrib:  [{'#' * max(int(s2 / max(total, 1) * 40), 1):<40s}] "
            f"{s2 / max(total, 1) * 100:.1f}%\n"
            f"```"
        )

        # Asset class drill-down
        ac = data.get("asset_class_breakdown", [])
        if ac:
            rows = ["## Asset Class Drill-Down\n",
                     "| Asset Class | Outstanding | Financed Emissions | Share |",
                     "|-------------|-------------|-------------------|-------|"]
            for a in ac:
                fe_val = a.get("financed_emissions", 0.0)
                share = fe_val / max(total, 1) * 100
                rows.append(
                    f"| {a.get('asset_class', '')} | EUR {a.get('outstanding_total', 0.0):,.0f} | "
                    f"{fe_val:,.2f} tCO2e | {share:.1f}% |"
                )
            sections.append("\n".join(rows))

        # Data quality traffic light
        dist = data.get("quality_distribution", {})
        if dist:
            sections.append("## Data Quality Traffic Light\n")
            sections.append("```")
            labels = {1: "Verified  ", 2: "Reported  ", 3: "Physical  ", 4: "Economic  ", 5: "Estimated "}
            colors = {1: "GREEN ", 2: "GREEN ", 3: "AMBER ", 4: "AMBER ", 5: "RED   "}
            for score in range(1, 6):
                count = dist.get(str(score), dist.get(score, 0))
                bar = "#" * min(count, 40)
                sections.append(f"  DQ{score} {labels.get(score, '')} [{colors.get(score, '')}{bar:<40s}] {count}")
            sections.append("```")

        # Top emitters
        top = data.get("top_emitters", [])
        if top:
            rows = ["## Top 10 Emitters\n",
                     "| # | Counterparty | Financed Emissions (tCO2e) |",
                     "|---|-------------|---------------------------|"]
            for i, t in enumerate(top[:10], 1):
                rows.append(f"| {i} | {t.get('name', '')} | {t.get('financed_emissions', 0.0):,.2f} |")
            sections.append("\n".join(rows))

        # YoY
        yoy = data.get("yoy_comparison", {})
        if yoy.get("prior_period_total") is not None:
            change = yoy.get("yoy_change_pct")
            arrow = "^" if change and change > 0 else "v" if change and change < 0 else "="
            sections.append(
                f"## Year-over-Year\n\n"
                f"| | Prior | Current | Change |\n"
                f"|--|-------|---------|--------|\n"
                f"| Emissions | {yoy.get('prior_period_total', 0.0):,.2f} | "
                f"{yoy.get('current_period_total', 0.0):,.2f} | {arrow} "
                f"{'+' if change and change > 0 else ''}{change:.1f}% |" if change is not None else ""
            )

        content = "\n\n".join(s for s in sections if s)
        ph = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(ph)
        content += f"\n\n<!-- provenance_hash: {ph} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        name = _esc(data.get("institution_name", "Financial Institution"))
        total = data.get("total_financed_emissions_tco2e", 0.0)
        body = (
            f'<div class="section"><h2>Dashboard Summary</h2>'
            f'<p><strong>Total Financed Emissions:</strong> {total:,.2f} tCO2e</p>'
            f'<p><strong>WACI:</strong> {data.get("waci", 0.0):,.4f} tCO2e/EUR M</p></div>'
        )
        ph = self._compute_provenance_hash(body)
        return self._wrap_html(f"Financed Emissions Dashboard - {name}", body, ph)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "report_type": "financed_emissions_dashboard",
            "pack_id": self.PACK_ID, "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION, "generated_at": self.generated_at,
            "summary": {
                "total_financed_emissions_tco2e": data.get("total_financed_emissions_tco2e", 0.0),
                "waci": data.get("waci", 0.0),
                "weighted_data_quality_score": data.get("weighted_data_quality_score", 5.0),
                "portfolio_value_eur": data.get("portfolio_value_eur", 0.0),
            },
            "asset_class_breakdown": data.get("asset_class_breakdown", []),
            "sector_breakdown": data.get("sector_breakdown", []),
            "yoy_comparison": data.get("yoy_comparison", {}),
            "quality_distribution": data.get("quality_distribution", {}),
        }
        cs = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(cs)
        return report

    def _md_footer(self, provenance_hash: str) -> str:
        return (
            "---\n\n"
            f"*Report generated by GreenLang {self.PACK_ID} | "
            f"Template: {self.TEMPLATE_NAME} v{self.VERSION}*\n\n"
            f"*Generated: {self.generated_at}*\n\n"
            f"**Provenance Hash (SHA-256):** `{provenance_hash}`"
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        return (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>body{font-family:'Segoe UI',Arial,sans-serif;margin:40px auto;max-width:1000px;"
            "color:#2c3e50;line-height:1.6}"
            "h1{color:#1a5276;border-bottom:3px solid #1abc9c;padding-bottom:10px}"
            "h2{color:#1a5276;margin-top:30px}"
            ".section{margin:20px 0;padding:15px;background:#fafafa;border-radius:6px;border:1px solid #ecf0f1}"
            ".data-table{width:100%;border-collapse:collapse;margin:10px 0}"
            ".data-table td,.data-table th{padding:8px 12px;border:1px solid #ddd}"
            ".data-table th{background:#1a5276;color:white;text-align:left}"
            ".data-table tr:nth-child(even){background:#f2f3f4}"
            ".provenance{margin-top:40px;padding:10px;background:#eaf2f8;border-radius:4px;"
            "font-size:.85em;font-family:monospace}"
            ".footer{margin-top:30px;font-size:.85em;color:#7f8c8d}"
            "</style>\n</head>\n<body>\n"
            f"<h1>{_esc(title)}</h1>\n"
            f'<p>Pack: {self.PACK_ID} | Template: {self.TEMPLATE_NAME} v{self.VERSION} | '
            f"Generated: {self.generated_at}</p>\n"
            f"{body}\n"
            f'<div class="provenance">Provenance Hash (SHA-256): {provenance_hash}</div>\n'
            f'<div class="footer">Generated by GreenLang {self.PACK_ID}</div>\n'
            f"<!-- provenance_hash: {provenance_hash} -->\n"
            "</body>\n</html>"
        )

    @staticmethod
    def _compute_provenance_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
