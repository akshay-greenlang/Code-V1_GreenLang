# -*- coding: utf-8 -*-
"""
PCAFReportTemplate - PCAF Financed Emissions Disclosure

Generates PCAF-compliant financed emissions disclosure reports for financial
institutions. Includes asset class breakdown, data quality scores,
year-over-year comparison, and sector attribution.

Supports markdown, HTML, and JSON output formats.

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



# ---------------------------------------------------------------------------
#  Data Models
# ---------------------------------------------------------------------------

class PCAFAssetClassBreakdown(BaseModel):
    """Asset class breakdown for PCAF report."""
    asset_class: str = Field(..., description="PCAF asset class")
    count: int = Field(default=0)
    outstanding_total: float = Field(default=0.0)
    financed_emissions: float = Field(default=0.0)
    data_quality_score: float = Field(default=5.0)


class PCAFSectorBreakdown(BaseModel):
    """Sector breakdown for PCAF report."""
    sector: str = Field(...)
    count: int = Field(default=0)
    outstanding_total: float = Field(default=0.0)
    financed_emissions: float = Field(default=0.0)


class PCAFYoYComparison(BaseModel):
    """Year-over-year comparison data."""
    prior_period_total: Optional[float] = Field(None)
    current_period_total: float = Field(default=0.0)
    yoy_change_pct: Optional[float] = Field(None)
    prior_waci: Optional[float] = Field(None)
    current_waci: float = Field(default=0.0)
    yoy_waci_change_pct: Optional[float] = Field(None)


class PCAFReportData(BaseModel):
    """Complete data for PCAF report."""
    institution_name: str = Field(default="")
    reporting_period: str = Field(default="")
    pcaf_version: str = Field(default="2022")
    total_financed_emissions_tco2e: float = Field(default=0.0)
    scope1_financed: float = Field(default=0.0)
    scope2_financed: float = Field(default=0.0)
    scope3_financed: float = Field(default=0.0)
    portfolio_value_eur: float = Field(default=0.0)
    waci: float = Field(default=0.0)
    weighted_data_quality_score: float = Field(default=5.0)
    counterparties_covered: int = Field(default=0)
    asset_class_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    sector_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    yoy_comparison: Dict[str, Any] = Field(default_factory=dict)
    quality_distribution: Dict[str, int] = Field(default_factory=dict)
    methodology_notes: str = Field(default="")


# ---------------------------------------------------------------------------
#  Template
# ---------------------------------------------------------------------------

class PCAFReportTemplate:
    """
    PCAF financed emissions disclosure template.

    Generates PCAF Standard compliant financed emissions reports with
    asset class breakdown, data quality scores, YoY comparison, and
    sector attribution analysis.
    """

    PACK_ID = "PACK-012"
    TEMPLATE_NAME = "pcaf_report"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: str = datetime.now(timezone.utc).isoformat()

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """Render PCAF report in specified format."""
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        raise ValueError(f"Unsupported format '{fmt}'. Use 'markdown', 'html', or 'json'.")

    # ---- Markdown ----

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render as GitHub-flavored Markdown."""
        sections: List[str] = []

        name = data.get("institution_name", "Financial Institution")
        period = data.get("reporting_period", "")
        total = data.get("total_financed_emissions_tco2e", 0.0)
        s1 = data.get("scope1_financed", 0.0)
        s2 = data.get("scope2_financed", 0.0)
        s3 = data.get("scope3_financed", 0.0)
        pv = data.get("portfolio_value_eur", 0.0)
        waci = data.get("waci", 0.0)
        wdq = data.get("weighted_data_quality_score", 5.0)
        cpty = data.get("counterparties_covered", 0)
        version = data.get("pcaf_version", "2022")

        sections.append(
            f"# PCAF Financed Emissions Disclosure\n\n"
            f"**Institution:** {name}\n\n"
            f"**Reporting Period:** {period}\n\n"
            f"**PCAF Standard Version:** {version}\n\n"
            f"**Pack:** {self.PACK_ID} | **Template:** {self.TEMPLATE_NAME} v{self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

        # Summary KPIs
        sections.append(
            f"## Summary\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| **Total Financed Emissions** | **{total:,.2f} tCO2e** |\n"
            f"| Scope 1 (financed) | {s1:,.2f} tCO2e |\n"
            f"| Scope 2 (financed) | {s2:,.2f} tCO2e |\n"
            f"| Scope 3 (financed) | {s3:,.2f} tCO2e |\n"
            f"| Portfolio Value | EUR {pv:,.0f} |\n"
            f"| WACI | {waci:,.4f} tCO2e/EUR M |\n"
            f"| Weighted Data Quality | {wdq:.2f} / 5.0 |\n"
            f"| Counterparties Covered | {cpty:,} |"
        )

        # Asset class breakdown
        ac_data = data.get("asset_class_breakdown", [])
        if ac_data:
            rows = ["## Asset Class Breakdown\n",
                     "| Asset Class | Count | Outstanding (EUR) | Financed Emissions (tCO2e) |",
                     "|-------------|-------|-------------------|---------------------------|"]
            for ac in ac_data:
                rows.append(
                    f"| {ac.get('asset_class', '')} | {ac.get('count', 0)} | "
                    f"{ac.get('outstanding_total', 0.0):,.0f} | {ac.get('financed_emissions', 0.0):,.2f} |"
                )
            sections.append("\n".join(rows))

        # Sector breakdown
        sec_data = data.get("sector_breakdown", [])
        if sec_data:
            rows = ["## Sector Attribution\n",
                     "| Sector | Count | Outstanding (EUR) | Financed Emissions (tCO2e) |",
                     "|--------|-------|-------------------|---------------------------|"]
            for s in sec_data[:15]:
                rows.append(
                    f"| {s.get('sector', '')} | {s.get('count', 0)} | "
                    f"{s.get('outstanding_total', 0.0):,.0f} | {s.get('financed_emissions', 0.0):,.2f} |"
                )
            sections.append("\n".join(rows))

        # Data quality distribution
        dist = data.get("quality_distribution", {})
        if dist:
            rows = ["## Data Quality Distribution\n",
                     "| Score | Count | Description |",
                     "|-------|-------|-------------|"]
            labels = {
                "1": "Reported (verified)", "2": "Reported (unverified)",
                "3": "Physical activity", "4": "Economic activity", "5": "Estimated",
            }
            for score in ["1", "2", "3", "4", "5"]:
                count = dist.get(score, dist.get(int(score), 0))
                rows.append(f"| {score} | {count} | {labels.get(score, '')} |")
            sections.append("\n".join(rows))

        # YoY comparison
        yoy = data.get("yoy_comparison", {})
        if yoy.get("prior_period_total") is not None:
            prior = yoy.get("prior_period_total", 0.0)
            current = yoy.get("current_period_total", 0.0)
            change = yoy.get("yoy_change_pct")
            direction = "increase" if (change and change > 0) else "decrease"
            sections.append(
                f"## Year-over-Year Comparison\n\n"
                f"| Metric | Prior Period | Current Period | Change |\n"
                f"|--------|-------------|----------------|--------|\n"
                f"| Financed Emissions | {prior:,.2f} tCO2e | {current:,.2f} tCO2e | "
                f"{'+' if change and change > 0 else ''}{change:.2f}% ({direction}) |" if change is not None
                else f"## Year-over-Year Comparison\n\nNo prior period data available."
            )

        # Methodology
        notes = data.get("methodology_notes", "")
        if notes:
            sections.append(f"## Methodology Notes\n\n{notes}")

        content = "\n\n".join(s for s in sections if s)
        ph = self._compute_provenance_hash(content)
        content += "\n\n" + self._md_footer(ph)
        content += f"\n\n<!-- provenance_hash: {ph} -->"
        return content

    # ---- HTML ----

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render as self-contained HTML."""
        name = _esc(data.get("institution_name", "Financial Institution"))
        period = data.get("reporting_period", "")
        total = data.get("total_financed_emissions_tco2e", 0.0)
        waci = data.get("waci", 0.0)
        wdq = data.get("weighted_data_quality_score", 5.0)

        body_parts: List[str] = []
        body_parts.append(f'<div class="section"><h2>Summary</h2>')
        body_parts.append(f'<table class="data-table">')
        body_parts.append(f'<tr><th>Metric</th><th>Value</th></tr>')
        body_parts.append(f'<tr><td>Total Financed Emissions</td><td>{total:,.2f} tCO2e</td></tr>')
        body_parts.append(f'<tr><td>WACI</td><td>{waci:,.4f} tCO2e/EUR M</td></tr>')
        body_parts.append(f'<tr><td>Data Quality</td><td>{wdq:.2f} / 5.0</td></tr>')
        body_parts.append(f'</table></div>')

        ac_data = data.get("asset_class_breakdown", [])
        if ac_data:
            body_parts.append('<div class="section"><h2>Asset Class Breakdown</h2>')
            body_parts.append('<table class="data-table"><tr><th>Asset Class</th>'
                              '<th>Outstanding</th><th>Financed Emissions</th></tr>')
            for ac in ac_data:
                body_parts.append(
                    f'<tr><td>{_esc(ac.get("asset_class", ""))}</td>'
                    f'<td>EUR {ac.get("outstanding_total", 0.0):,.0f}</td>'
                    f'<td>{ac.get("financed_emissions", 0.0):,.2f} tCO2e</td></tr>'
                )
            body_parts.append('</table></div>')

        body = "\n".join(body_parts)
        ph = self._compute_provenance_hash(body)
        return self._wrap_html(f"PCAF Financed Emissions - {name} ({period})", body, ph)

    # ---- JSON ----

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        report: Dict[str, Any] = {
            "report_type": "pcaf_financed_emissions",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "institution_name": data.get("institution_name", ""),
            "reporting_period": data.get("reporting_period", ""),
            "pcaf_version": data.get("pcaf_version", "2022"),
            "summary": {
                "total_financed_emissions_tco2e": data.get("total_financed_emissions_tco2e", 0.0),
                "scope1_financed": data.get("scope1_financed", 0.0),
                "scope2_financed": data.get("scope2_financed", 0.0),
                "scope3_financed": data.get("scope3_financed", 0.0),
                "portfolio_value_eur": data.get("portfolio_value_eur", 0.0),
                "waci": data.get("waci", 0.0),
                "weighted_data_quality_score": data.get("weighted_data_quality_score", 5.0),
                "counterparties_covered": data.get("counterparties_covered", 0),
            },
            "asset_class_breakdown": data.get("asset_class_breakdown", []),
            "sector_breakdown": data.get("sector_breakdown", []),
            "yoy_comparison": data.get("yoy_comparison", {}),
            "quality_distribution": data.get("quality_distribution", {}),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ---- Helpers ----

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
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            f"<title>{_esc(title)}</title>\n"
            "<style>\n"
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px auto; "
            "color: #2c3e50; line-height: 1.6; max-width: 1000px; }\n"
            "h1 { color: #1a5276; border-bottom: 3px solid #1abc9c; padding-bottom: 10px; }\n"
            "h2 { color: #1a5276; margin-top: 30px; }\n"
            ".section { margin-bottom: 30px; padding: 15px; background: #fafafa; "
            "border-radius: 6px; border: 1px solid #ecf0f1; }\n"
            ".data-table { width: 100%; border-collapse: collapse; margin: 10px 0; }\n"
            ".data-table td, .data-table th { padding: 8px 12px; border: 1px solid #ddd; }\n"
            ".data-table th { background: #1a5276; color: white; text-align: left; }\n"
            ".data-table tr:nth-child(even) { background: #f2f3f4; }\n"
            ".provenance { margin-top: 40px; padding: 10px; background: #eaf2f8; "
            "border-radius: 4px; font-size: 0.85em; font-family: monospace; }\n"
            ".footer { margin-top: 30px; font-size: 0.85em; color: #7f8c8d; }\n"
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
