# -*- coding: utf-8 -*-
"""
RetailESGScorecardTemplate - Board-level ESG scorecard for PACK-014.

Sections:
    1. Executive KPI Dashboard (10 KPIs)
    2. Percentile Rankings
    3. SBTi Alignment Status
    4. Year-over-Year Trends
    5. Peer Comparison
    6. Regulatory Compliance Summary
    7. Board-Ready Highlights

Author: GreenLang Team
Version: 14.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RetailESGScorecardTemplate:
    """
    Board-level retail ESG scorecard template.

    Provides a high-level executive view of ESG performance with
    10 headline KPIs, peer benchmarking, regulatory status, and
    year-over-year trend analysis.
    """

    HEADLINE_KPIS: List[str] = [
        "total_emissions_tco2e",
        "scope1_tco2e",
        "scope2_tco2e",
        "scope3_tco2e",
        "emissions_intensity_per_sqm",
        "renewable_electricity_pct",
        "food_waste_reduction_pct",
        "recycled_content_pct",
        "mci_score",
        "esrs_completeness_pct",
    ]

    KPI_LABELS: Dict[str, str] = {
        "total_emissions_tco2e": "Total GHG Emissions",
        "scope1_tco2e": "Scope 1 Emissions",
        "scope2_tco2e": "Scope 2 Emissions",
        "scope3_tco2e": "Scope 3 Emissions",
        "emissions_intensity_per_sqm": "Emissions Intensity (per sqm)",
        "renewable_electricity_pct": "Renewable Electricity",
        "food_waste_reduction_pct": "Food Waste Reduction",
        "recycled_content_pct": "Recycled Content",
        "mci_score": "Material Circularity Indicator",
        "esrs_completeness_pct": "ESRS Disclosure Completeness",
    }

    KPI_UNITS: Dict[str, str] = {
        "total_emissions_tco2e": "tCO2e",
        "scope1_tco2e": "tCO2e",
        "scope2_tco2e": "tCO2e",
        "scope3_tco2e": "tCO2e",
        "emissions_intensity_per_sqm": "tCO2e/sqm",
        "renewable_electricity_pct": "%",
        "food_waste_reduction_pct": "%",
        "recycled_content_pct": "%",
        "mci_score": "0-1",
        "esrs_completeness_pct": "%",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RetailESGScorecardTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render ESG scorecard as Markdown."""
        self.generated_at = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_kpi_dashboard(data),
            self._md_percentile_rankings(data),
            self._md_sbti_status(data),
            self._md_yoy_trends(data),
            self._md_peer_comparison(data),
            self._md_regulatory_summary(data),
            self._md_board_highlights(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render ESG scorecard as HTML."""
        self.generated_at = datetime.utcnow()
        css = (
            "body{font-family:system-ui,sans-serif;padding:20px;background:#f5f5f5;}"
            ".report{max-width:1400px;margin:auto;background:#fff;padding:30px;border-radius:12px;}"
            "h1{color:#0d6efd;text-align:center;}"
            "h2{color:#198754;margin-top:30px;border-bottom:2px solid #198754;padding-bottom:8px;}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:15px;margin:20px 0;}"
            ".kpi-card{background:#f8f9fa;border-radius:10px;padding:15px;text-align:center;border-left:4px solid #0d6efd;}"
            ".kpi-label{font-size:0.8em;color:#6c757d;text-transform:uppercase;}"
            ".kpi-value{font-size:1.6em;font-weight:700;color:#198754;}"
            ".kpi-unit{font-size:0.75em;color:#adb5bd;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
        )
        kpis = data.get("kpis", {})
        cards = ""
        for kpi_key in self.HEADLINE_KPIS:
            label = self.KPI_LABELS.get(kpi_key, kpi_key)
            unit = self.KPI_UNITS.get(kpi_key, "")
            val = kpis.get(kpi_key, 0)
            formatted = f"{val:,.2f}" if isinstance(val, (int, float)) else str(val)
            cards += f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{formatted}</div><div class="kpi-unit">{unit}</div></div>\n'
        body = f'<h1>Retail ESG Scorecard</h1>\n<div class="kpi-grid">\n{cards}</div>'
        prov = self._provenance(body)
        return f'<!DOCTYPE html>\n<html><head><style>{css}</style></head>\n<body><div class="report">{body}</div>\n<!-- Provenance: {prov} -->\n</body></html>'

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render ESG scorecard as JSON."""
        self.generated_at = datetime.utcnow()
        kpis = data.get("kpis", {})
        result = {
            "template": "retail_esg_scorecard", "version": "14.0.0",
            "generated_at": self.generated_at.isoformat(),
            "headline_kpis": {k: {"label": self.KPI_LABELS.get(k, k), "value": kpis.get(k, 0), "unit": self.KPI_UNITS.get(k, "")} for k in self.HEADLINE_KPIS},
            "percentile_rankings": data.get("percentile_rankings", {}),
            "sbti_status": data.get("sbti_status", {}),
            "yoy_trends": data.get("yoy_trends", {}),
            "peer_comparison": data.get("peer_comparison", []),
            "regulatory_summary": data.get("regulatory_summary", {}),
            "board_highlights": data.get("board_highlights", []),
        }
        result["provenance_hash"] = self._provenance(json.dumps(result, default=str))
        return result

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        entity = data.get("entity_name", "Retail Company")
        year = data.get("reporting_year", "")
        return f"# {entity} - ESG Scorecard {year}\n\n**Generated:** {ts}\n\n---"

    def _md_kpi_dashboard(self, data: Dict[str, Any]) -> str:
        kpis = data.get("kpis", {})
        lines = ["## Executive KPI Dashboard", "", "| KPI | Value | Unit |", "|-----|-------|------|"]
        for k in self.HEADLINE_KPIS:
            label = self.KPI_LABELS.get(k, k)
            val = kpis.get(k, 0)
            unit = self.KPI_UNITS.get(k, "")
            formatted = f"{val:,.2f}" if isinstance(val, (int, float)) else str(val)
            lines.append(f"| {label} | {formatted} | {unit} |")
        return "\n".join(lines)

    def _md_percentile_rankings(self, data: Dict[str, Any]) -> str:
        rankings = data.get("percentile_rankings", {})
        if not rankings:
            return "## Percentile Rankings\n\n_No peer benchmarking data available._"
        lines = ["## Percentile Rankings (vs Retail Peers)", "", "| Metric | Percentile |", "|--------|-----------|"]
        for metric, pctile in rankings.items():
            lines.append(f"| {metric} | P{pctile} |")
        return "\n".join(lines)

    def _md_sbti_status(self, data: Dict[str, Any]) -> str:
        sbti = data.get("sbti_status", {})
        status = sbti.get("status", "Not committed")
        target_year = sbti.get("target_year", "-")
        pathway = sbti.get("pathway", "-")
        return f"## SBTi Alignment\n\n- **Status:** {status}\n- **Target Year:** {target_year}\n- **Pathway:** {pathway}"

    def _md_yoy_trends(self, data: Dict[str, Any]) -> str:
        trends = data.get("yoy_trends", {})
        if not trends:
            return "## Year-over-Year Trends\n\n_No historical data for trend analysis._"
        lines = ["## Year-over-Year Trends", "", "| Metric | Prior Year | Current | Change |", "|--------|-----------|---------|--------|"]
        for metric, vals in trends.items():
            prior = vals.get("prior", 0)
            current = vals.get("current", 0)
            change = current - prior if isinstance(prior, (int, float)) and isinstance(current, (int, float)) else 0
            direction = "+" if change > 0 else ""
            lines.append(f"| {metric} | {prior} | {current} | {direction}{change:.2f} |")
        return "\n".join(lines)

    def _md_peer_comparison(self, data: Dict[str, Any]) -> str:
        peers = data.get("peer_comparison", [])
        if not peers:
            return "## Peer Comparison\n\n_No peer data available._"
        lines = ["## Peer Comparison", "", "| Company | Total Emissions | Intensity | Rank |", "|---------|----------------|-----------|------|"]
        for p in peers:
            lines.append(f"| {p.get('name', '-')} | {p.get('total_emissions', 0):,.0f} | {p.get('intensity', 0):.4f} | {p.get('rank', '-')} |")
        return "\n".join(lines)

    def _md_regulatory_summary(self, data: Dict[str, Any]) -> str:
        reg = data.get("regulatory_summary", {})
        return f"## Regulatory Compliance\n\n- **Regulations Applicable:** {reg.get('applicable', 0)}\n- **Compliant:** {reg.get('compliant', 0)}\n- **Action Items:** {reg.get('action_items', 0)}"

    def _md_board_highlights(self, data: Dict[str, Any]) -> str:
        highlights = data.get("board_highlights", [])
        if not highlights:
            return "## Board-Ready Highlights\n\n_No highlights configured._"
        lines = ["## Board-Ready Highlights", ""]
        for h in highlights:
            lines.append(f"- {h}")
        return "\n".join(lines)

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
