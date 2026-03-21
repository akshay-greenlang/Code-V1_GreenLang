# -*- coding: utf-8 -*-
"""
SupplyChainHeatmapTemplate - Tier 1/2/3 emissions heatmap for PACK-027.

Renders a supply chain emissions heatmap by geography, commodity, and
engagement status. Includes supplier scorecards for top 50 by emissions,
CDP score distribution, SBTi adoption tracking, and year-over-year
improvement trends.

Sections:
    1. Supply Chain Overview (total Scope 3, tier distribution)
    2. Heatmap by Geography
    3. Heatmap by Commodity/Category
    4. Tier Assignment Summary
    5. Top 50 Supplier Scorecards
    6. CDP Score Distribution
    7. SBTi Adoption Tracking
    8. Engagement Program Status
    9. Hotspot Action Plan
   10. Year-over-Year Trends

Output: Markdown, HTML, JSON
Provenance: SHA-256 hash on all outputs

Author: GreenLang Team
Version: 27.0.0
Pack: PACK-027 Enterprise Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "27.0.0"
_PACK_ID = "PACK-027"
_TEMPLATE_ID = "supply_chain_heatmap"

_PRIMARY = "#0d3b2e"
_SECONDARY = "#1a6b4f"
_ACCENT = "#2e8b6e"
_LIGHT = "#e0f2ee"
_LIGHTER = "#f0f9f6"
_CARD_BG = "#b2dfdb"

SUPPLIER_TIERS = [
    {"tier": 1, "label": "Critical", "criteria": "Top 50 suppliers (50-70% of Scope 3)", "engagement": "Collaborate"},
    {"tier": 2, "label": "Strategic", "criteria": "Next 200 suppliers (15-25% of Scope 3)", "engagement": "Require"},
    {"tier": 3, "label": "Managed", "criteria": "Next 1,000 suppliers (5-10% of Scope 3)", "engagement": "Engage"},
    {"tier": 4, "label": "Monitored", "criteria": "Remaining suppliers (long tail)", "engagement": "Inform"},
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec_comma(val: Any, places: int = 0) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        r = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(r).split(".")
        ip = parts[0]
        neg = ip.startswith("-")
        if neg:
            ip = ip[1:]
        f = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0:
                f = "," + f
            f = ch + f
        if neg:
            f = "-" + f
        if len(parts) > 1:
            f += "." + parts[1]
        return f
    except Exception:
        return str(val)


def _pct(val: Any) -> str:
    try:
        return str(round(float(val), 1)) + "%"
    except Exception:
        return str(val)


def _safe_div(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den)
        return float(num) / d if d != 0 else default
    except Exception:
        return default


def _heat_level(pct: float) -> str:
    if pct >= 15:
        return "HIGH"
    elif pct >= 5:
        return "MEDIUM"
    elif pct > 0:
        return "LOW"
    return "NONE"


class SupplyChainHeatmapTemplate:
    """
    Tier 1/2/3 supplier emissions heatmap template.

    Maps supply chain emissions by geography, commodity, and engagement
    status with supplier scorecards and engagement tracking.
    Supports Markdown, HTML, and JSON output.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_overview(data),
            self._md_geo_heatmap(data),
            self._md_commodity_heatmap(data),
            self._md_tier_summary(data),
            self._md_top_suppliers(data),
            self._md_cdp_distribution(data),
            self._md_sbti_tracking(data),
            self._md_engagement_status(data),
            self._md_hotspot_actions(data),
            self._md_yoy_trends(data),
            self._md_citations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(s for s in sections if s)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- SHA-256 Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:24px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.85em;}}"
            f"th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left;}}"
            f"th{{background:{_LIGHT};color:{_PRIMARY};font-weight:600;}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".heat-high{{background:#ffcdd2;color:#b71c1c;font-weight:700;text-align:center;}}"
            f".heat-medium{{background:#fff9c4;color:#f57f17;font-weight:600;text-align:center;}}"
            f".heat-low{{background:#c8e6c9;color:#2e7d32;text-align:center;}}"
            f".heat-none{{background:#f5f5f5;color:#9e9e9e;text-align:center;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
        )
        body = (
            f'<h1>Supply Chain Emissions Heatmap</h1>\n'
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'{data.get("reporting_year", "")} | '
            f'Total Scope 3: {_dec_comma(data.get("scope3_tco2e", 0))} tCO2e</p>\n'
            f'{self._html_geo_table(data)}\n'
            f'{self._html_top_suppliers_table(data)}\n'
            f'<div class="footer">Generated by GreenLang PACK-027 | Supply Chain Heatmap | SHA-256</div>'
        )
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n<title>Supply Chain Heatmap</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- SHA-256 Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "total_scope3_tco2e": data.get("scope3_tco2e", 0),
            "total_suppliers": data.get("total_suppliers", 0),
            "geographic_heatmap": data.get("geo_heatmap", []),
            "commodity_heatmap": data.get("commodity_heatmap", []),
            "tier_summary": [
                {
                    "tier": t["tier"], "label": t["label"],
                    "count": data.get(f"tier{t['tier']}_count", 0),
                    "tco2e": data.get(f"tier{t['tier']}_tco2e", 0),
                    "pct_scope3": data.get(f"tier{t['tier']}_pct", 0),
                }
                for t in SUPPLIER_TIERS
            ],
            "top_50_suppliers": data.get("top_suppliers", [])[:50],
            "cdp_distribution": data.get("cdp_distribution", {}),
            "sbti_adoption": data.get("sbti_adoption", {}),
            "engagement_program": data.get("engagement_program", {}),
            "hotspots": data.get("hotspots", []),
            "yoy_trends": data.get("supply_chain_trends", []),
            "citations": data.get("citations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Markdown sections
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Supply Chain Emissions Heatmap\n\n"
            f"**{data.get('org_name', 'Enterprise')}** | "
            f"{data.get('reporting_year', '')} | Generated: {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        s3 = float(data.get("scope3_tco2e", 0))
        total_suppliers = int(data.get("total_suppliers", 0))
        engaged = int(data.get("engaged_suppliers", 0))
        return (
            f"## Supply Chain Overview\n\n"
            f"| Metric | Value |\n"
            f"|--------|------:|\n"
            f"| Total Scope 3 Emissions | {_dec_comma(s3)} tCO2e |\n"
            f"| Total Suppliers | {_dec_comma(total_suppliers)} |\n"
            f"| Suppliers Engaged | {_dec_comma(engaged)} ({_pct(_safe_div(engaged, total_suppliers) * 100)}) |\n"
            f"| Suppliers with CDP Disclosure | {_dec_comma(data.get('suppliers_cdp', 0))} |\n"
            f"| Suppliers with SBTi Targets | {_dec_comma(data.get('suppliers_sbti', 0))} |\n"
            f"| Average Supplier DQ Level | {data.get('avg_supplier_dq', 'N/A')} |"
        )

    def _md_geo_heatmap(self, data: Dict[str, Any]) -> str:
        geo = data.get("geo_heatmap", [])
        if not geo:
            return "## Geographic Heatmap\n\nGeographic data not available."
        s3 = float(data.get("scope3_tco2e", 1))
        lines = [
            "## Geographic Heatmap\n",
            "| Region/Country | tCO2e | % of Scope 3 | Suppliers | Heat Level |",
            "|----------------|------:|-----------:|----------:|:----------:|",
        ]
        for g in geo:
            val = float(g.get("tco2e", 0))
            p = _safe_div(val, s3) * 100
            lines.append(
                f"| {g.get('region', '')} | {_dec_comma(val)} "
                f"| {_pct(p)} | {_dec_comma(g.get('suppliers', 0))} "
                f"| {_heat_level(p)} |"
            )
        return "\n".join(lines)

    def _md_commodity_heatmap(self, data: Dict[str, Any]) -> str:
        commodities = data.get("commodity_heatmap", [])
        if not commodities:
            return "## Commodity Heatmap\n\nCommodity data not available."
        s3 = float(data.get("scope3_tco2e", 1))
        lines = [
            "## Commodity/Category Heatmap\n",
            "| Commodity/Category | tCO2e | % of Scope 3 | Top Supplier | Heat Level |",
            "|--------------------|------:|-----------:|--------------|:----------:|",
        ]
        for c in commodities:
            val = float(c.get("tco2e", 0))
            p = _safe_div(val, s3) * 100
            lines.append(
                f"| {c.get('commodity', '')} | {_dec_comma(val)} "
                f"| {_pct(p)} | {c.get('top_supplier', 'N/A')} "
                f"| {_heat_level(p)} |"
            )
        return "\n".join(lines)

    def _md_tier_summary(self, data: Dict[str, Any]) -> str:
        lines = [
            "## Tier Assignment Summary\n",
            "| Tier | Label | Criteria | Count | tCO2e | % of S3 | Engagement |",
            "|:----:|-------|----------|------:|------:|:-------:|:----------:|",
        ]
        for t in SUPPLIER_TIERS:
            count = data.get(f"tier{t['tier']}_count", 0)
            tco2e = data.get(f"tier{t['tier']}_tco2e", 0)
            pct = data.get(f"tier{t['tier']}_pct", 0)
            lines.append(
                f"| {t['tier']} | {t['label']} | {t['criteria']} "
                f"| {_dec_comma(count)} | {_dec_comma(tco2e)} "
                f"| {_pct(pct)} | {t['engagement']} |"
            )
        return "\n".join(lines)

    def _md_top_suppliers(self, data: Dict[str, Any]) -> str:
        suppliers = data.get("top_suppliers", [])[:50]
        if not suppliers:
            return "## Top 50 Supplier Scorecards\n\nNo supplier data available."
        s3 = float(data.get("scope3_tco2e", 1))
        lines = [
            "## Top 50 Supplier Scorecards\n",
            "| Rank | Supplier | Country | tCO2e | % S3 | Tier | CDP | SBTi | DQ | YoY |",
            "|:----:|----------|---------|------:|:----:|:----:|:---:|:----:|:--:|:---:|",
        ]
        for i, sup in enumerate(suppliers[:50], 1):
            val = float(sup.get("tco2e", 0))
            lines.append(
                f"| {i} | {sup.get('name', '')} | {sup.get('country', '')} "
                f"| {_dec_comma(val)} | {_pct(_safe_div(val, s3) * 100)} "
                f"| {sup.get('tier', '-')} | {sup.get('cdp_score', '-')} "
                f"| {sup.get('sbti_status', '-')} | {sup.get('dq_level', '-')} "
                f"| {sup.get('yoy_change', '-')} |"
            )
        return "\n".join(lines)

    def _md_cdp_distribution(self, data: Dict[str, Any]) -> str:
        dist = data.get("cdp_distribution", {})
        if not dist:
            return "## CDP Score Distribution\n\nCDP distribution data not available."
        lines = [
            "## CDP Score Distribution\n",
            "| CDP Score | Supplier Count | % of Engaged |",
            "|:---------:|---------------:|-----------:|",
        ]
        for score in ["A", "A-", "B", "B-", "C", "C-", "D", "D-", "Not Disclosed"]:
            count = dist.get(score, 0)
            total = sum(dist.values()) or 1
            lines.append(f"| {score} | {count} | {_pct(_safe_div(count, total) * 100)} |")
        return "\n".join(lines)

    def _md_sbti_tracking(self, data: Dict[str, Any]) -> str:
        sbti = data.get("sbti_adoption", {})
        return (
            f"## SBTi Adoption Tracking\n\n"
            f"| Status | Count | % of Tier 1+2 |\n"
            f"|--------|------:|:-------------:|\n"
            f"| Targets Validated | {_dec_comma(sbti.get('validated', 0))} | {_pct(sbti.get('validated_pct', 0))} |\n"
            f"| Committed | {_dec_comma(sbti.get('committed', 0))} | {_pct(sbti.get('committed_pct', 0))} |\n"
            f"| No Commitment | {_dec_comma(sbti.get('none', 0))} | {_pct(sbti.get('none_pct', 0))} |"
        )

    def _md_engagement_status(self, data: Dict[str, Any]) -> str:
        ep = data.get("engagement_program", {})
        return (
            f"## Engagement Program Status\n\n"
            f"| Stage | Target | Actual | Completion |\n"
            f"|-------|-------:|-------:|:----------:|\n"
            f"| Awareness (letters sent) | {_dec_comma(ep.get('awareness_target', 0))} "
            f"| {_dec_comma(ep.get('awareness_actual', 0))} "
            f"| {_pct(ep.get('awareness_pct', 0))} |\n"
            f"| Measurement (data received) | {_dec_comma(ep.get('measurement_target', 0))} "
            f"| {_dec_comma(ep.get('measurement_actual', 0))} "
            f"| {_pct(ep.get('measurement_pct', 0))} |\n"
            f"| Target Setting (SBTi commit) | {_dec_comma(ep.get('target_setting_target', 0))} "
            f"| {_dec_comma(ep.get('target_setting_actual', 0))} "
            f"| {_pct(ep.get('target_setting_pct', 0))} |\n"
            f"| Reduction (verified impact) | {_dec_comma(ep.get('reduction_target', 0))} "
            f"| {_dec_comma(ep.get('reduction_actual', 0))} "
            f"| {_pct(ep.get('reduction_pct', 0))} |"
        )

    def _md_hotspot_actions(self, data: Dict[str, Any]) -> str:
        hotspots = data.get("hotspots", [])
        if not hotspots:
            return "## Hotspot Action Plan\n\nNo critical hotspots identified."
        lines = [
            "## Hotspot Action Plan\n",
            "| Priority | Hotspot | tCO2e | Action Required | Owner | Deadline |",
            "|:--------:|---------|------:|-----------------|-------|----------|",
        ]
        for i, h in enumerate(hotspots[:10], 1):
            lines.append(
                f"| {i} | {h.get('description', '')} | {_dec_comma(h.get('tco2e', 0))} "
                f"| {h.get('action', '')} | {h.get('owner', '')} | {h.get('deadline', '')} |"
            )
        return "\n".join(lines)

    def _md_yoy_trends(self, data: Dict[str, Any]) -> str:
        trends = data.get("supply_chain_trends", [])
        if not trends:
            return "## Year-over-Year Trends\n\nInsufficient data for trend analysis."
        lines = [
            "## Year-over-Year Trends\n",
            "| Year | Scope 3 tCO2e | Engaged | CDP Response Rate | SBTi Rate | Avg DQ |",
            "|------|-------------:|--------:|-----------------:|---------:|:------:|",
        ]
        for yr in trends:
            lines.append(
                f"| {yr.get('year', '')} | {_dec_comma(yr.get('scope3_tco2e', 0))} "
                f"| {_dec_comma(yr.get('engaged', 0))} "
                f"| {_pct(yr.get('cdp_response_rate', 0))} "
                f"| {_pct(yr.get('sbti_rate', 0))} "
                f"| {yr.get('avg_dq', '-')} |"
            )
        return "\n".join(lines)

    def _md_citations(self, data: Dict[str, Any]) -> str:
        citations = data.get("citations", [
            {"ref": "SC-001", "source": "GHG Protocol Scope 3 Standard", "year": "2011"},
            {"ref": "SC-002", "source": "CDP Supply Chain Program", "year": "2024"},
            {"ref": "SC-003", "source": "WBCSD PACT Framework", "year": "2023"},
        ])
        lines = ["## Citations\n"]
        for c in citations:
            lines.append(f"- [{c.get('ref', '')}] {c.get('source', '')} ({c.get('year', '')})")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-027 Enterprise Net Zero Pack on {ts}*  \n"
            f"*Supply chain emissions heatmap. SHA-256 provenance.*"
        )

    # HTML helpers
    def _html_geo_table(self, data: Dict[str, Any]) -> str:
        geo = data.get("geo_heatmap", [])
        s3 = float(data.get("scope3_tco2e", 1))
        rows = ""
        for g in geo:
            val = float(g.get("tco2e", 0))
            p = _safe_div(val, s3) * 100
            heat = _heat_level(p).lower()
            rows += (
                f'<tr><td>{g.get("region", "")}</td><td>{_dec_comma(val)}</td>'
                f'<td class="heat-{heat}">{_pct(p)}</td>'
                f'<td>{_dec_comma(g.get("suppliers", 0))}</td></tr>\n'
            )
        return (
            f'<h2>Geographic Heatmap</h2>\n'
            f'<table><tr><th>Region</th><th>tCO2e</th><th>% S3</th><th>Suppliers</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_top_suppliers_table(self, data: Dict[str, Any]) -> str:
        suppliers = data.get("top_suppliers", [])[:20]
        rows = ""
        for i, sup in enumerate(suppliers, 1):
            rows += (
                f'<tr><td>{i}</td><td>{sup.get("name", "")}</td>'
                f'<td>{_dec_comma(sup.get("tco2e", 0))}</td>'
                f'<td>{sup.get("tier", "-")}</td>'
                f'<td>{sup.get("cdp_score", "-")}</td>'
                f'<td>{sup.get("sbti_status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>Top Suppliers</h2>\n'
            f'<table><tr><th>#</th><th>Supplier</th><th>tCO2e</th>'
            f'<th>Tier</th><th>CDP</th><th>SBTi</th></tr>\n{rows}</table>'
        )
