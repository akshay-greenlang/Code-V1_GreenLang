# -*- coding: utf-8 -*-
"""
GHGBaselineReportTemplate - Detailed GHG inventory baseline report for PACK-021.

Renders a comprehensive GHG inventory baseline report covering methodology,
organizational boundary, scope-level breakdowns, data quality assessment,
base year statement, and emission factors used. Suitable for internal baseline
documentation and third-party verification.

Sections:
    1. Methodology
    2. Organizational Boundary
    3. Scope 1 Breakdown (by source type)
    4. Scope 2 (location vs market comparison)
    5. Scope 3 (all 15 categories with relevance)
    6. Total Emissions Summary
    7. Data Quality Matrix
    8. Base Year Statement
    9. Emission Factors Used

Author: GreenLang Team
Version: 21.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "21.0.0"

_SCOPE3_CATEGORIES: List[Dict[str, str]] = [
    {"num": "1", "name": "Purchased Goods & Services"},
    {"num": "2", "name": "Capital Goods"},
    {"num": "3", "name": "Fuel & Energy Related Activities"},
    {"num": "4", "name": "Upstream Transportation & Distribution"},
    {"num": "5", "name": "Waste Generated in Operations"},
    {"num": "6", "name": "Business Travel"},
    {"num": "7", "name": "Employee Commuting"},
    {"num": "8", "name": "Upstream Leased Assets"},
    {"num": "9", "name": "Downstream Transportation & Distribution"},
    {"num": "10", "name": "Processing of Sold Products"},
    {"num": "11", "name": "Use of Sold Products"},
    {"num": "12", "name": "End-of-Life Treatment of Sold Products"},
    {"num": "13", "name": "Downstream Leased Assets"},
    {"num": "14", "name": "Franchises"},
    {"num": "15", "name": "Investments"},
]

_DQ_LEVELS: Dict[str, str] = {
    "high": "Primary data from direct measurements",
    "medium": "Activity data with verified emission factors",
    "low": "Estimated or modelled using proxies",
    "very_low": "Spend-based or extrapolated data",
}

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)

def _dec_comma(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        negative = int_part.startswith("-")
        if negative:
            int_part = int_part[1:]
        formatted = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                formatted = "," + formatted
            formatted = ch + formatted
        if negative:
            formatted = "-" + formatted
        if len(parts) > 1:
            formatted += "." + parts[1]
        return formatted
    except Exception:
        return str(val)

def _pct_of(part: Any, total: Any) -> Decimal:
    p = Decimal(str(part))
    t = Decimal(str(total))
    if t == 0:
        return Decimal("0.00")
    return (p / t * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

class GHGBaselineReportTemplate:
    """
    Detailed GHG inventory baseline report template.

    Renders a full GHG inventory baseline covering all three scopes,
    data quality assessment, emission factors, and base year statement
    across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_methodology(data),
            self._md_boundary(data),
            self._md_scope1(data),
            self._md_scope2(data),
            self._md_scope3(data),
            self._md_total_summary(data),
            self._md_data_quality(data),
            self._md_base_year(data),
            self._md_emission_factors(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_methodology(data),
            self._html_boundary(data),
            self._html_scope1(data),
            self._html_scope2(data),
            self._html_scope3(data),
            self._html_total_summary(data),
            self._html_data_quality(data),
            self._html_base_year(data),
            self._html_emission_factors(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>GHG Baseline Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        scope1 = data.get("scope1", {})
        scope2 = data.get("scope2", {})
        scope3_cats = data.get("scope3_categories", {})
        s1_total = Decimal(str(scope1.get("total_tco2e", 0)))
        s2_loc = Decimal(str(scope2.get("location_tco2e", 0)))
        s2_mkt = Decimal(str(scope2.get("market_tco2e", 0)))
        s3_total = sum(
            Decimal(str(scope3_cats.get(c["num"], {}).get("emissions_tco2e", 0)))
            for c in _SCOPE3_CATEGORIES
        )
        grand_total = s1_total + s2_loc + s3_total

        result: Dict[str, Any] = {
            "template": "ghg_baseline_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "base_year": data.get("base_year", ""),
            "methodology": data.get("methodology", {}),
            "boundary": data.get("boundary", {}),
            "scope1": {
                "total_tco2e": str(s1_total),
                "sources": scope1.get("sources", []),
            },
            "scope2": {
                "location_tco2e": str(s2_loc),
                "market_tco2e": str(s2_mkt),
                "details": scope2.get("details", {}),
            },
            "scope3": {
                "total_tco2e": str(s3_total),
                "categories": {
                    c["num"]: {
                        "name": c["name"],
                        "emissions_tco2e": str(
                            Decimal(str(scope3_cats.get(c["num"], {}).get("emissions_tco2e", 0)))
                        ),
                        "relevance": scope3_cats.get(c["num"], {}).get("relevance", "not_assessed"),
                        "method": scope3_cats.get(c["num"], {}).get("method", ""),
                        "data_quality": scope3_cats.get(c["num"], {}).get("data_quality", ""),
                    }
                    for c in _SCOPE3_CATEGORIES
                },
            },
            "total_emissions": {
                "scope1_tco2e": str(s1_total),
                "scope2_location_tco2e": str(s2_loc),
                "scope2_market_tco2e": str(s2_mkt),
                "scope3_tco2e": str(s3_total),
                "grand_total_tco2e": str(grand_total),
            },
            "data_quality": data.get("data_quality", {}),
            "base_year_statement": data.get("base_year_statement", {}),
            "emission_factors": data.get("emission_factors", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# GHG Inventory Baseline Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Base Year:** {data.get('base_year', '')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        meth = data.get("methodology", {})
        return (
            f"## 1. Methodology\n\n"
            f"- **Standard:** {meth.get('standard', 'GHG Protocol Corporate Standard')}\n"
            f"- **Approach:** {meth.get('approach', 'Operational Control')}\n"
            f"- **Global Warming Potentials:** {meth.get('gwp_source', 'IPCC AR6')}\n"
            f"- **Gases Included:** {meth.get('gases', 'CO2, CH4, N2O, HFCs, PFCs, SF6, NF3')}\n"
            f"- **Verification:** {meth.get('verification', 'N/A')}\n"
            f"- **Materiality Threshold:** {meth.get('materiality_threshold', '1% of total emissions')}"
        )

    def _md_boundary(self, data: Dict[str, Any]) -> str:
        boundary = data.get("boundary", {})
        entities = boundary.get("entities", [])
        lines = [
            f"## 2. Organizational Boundary\n",
            f"- **Consolidation Approach:** {boundary.get('consolidation', 'Operational Control')}\n"
            f"- **Reporting Period:** {boundary.get('reporting_period', 'Calendar year')}\n"
            f"- **Entities Included:** {len(entities)}\n",
        ]
        if entities:
            lines.append("| Entity | Country | Scope | Ownership (%) |")
            lines.append("|--------|---------|-------|-------------:|")
            for e in entities:
                lines.append(
                    f"| {e.get('name', '-')} | {e.get('country', '-')} "
                    f"| {e.get('scope', '-')} | {_dec(e.get('ownership_pct', 100))}% |"
                )
        return "\n".join(lines)

    def _md_scope1(self, data: Dict[str, Any]) -> str:
        scope1 = data.get("scope1", {})
        sources = scope1.get("sources", [])
        total = Decimal(str(scope1.get("total_tco2e", 0)))
        lines = [
            f"## 3. Scope 1 - Direct Emissions\n",
            f"**Total Scope 1:** {_dec_comma(total)} tCO2e\n",
            "| Source Type | Emissions (tCO2e) | Share (%) | Data Quality |",
            "|-----------|------------------:|----------:|-------------|",
        ]
        for src in sources:
            emissions = Decimal(str(src.get("emissions_tco2e", 0)))
            lines.append(
                f"| {src.get('type', '-')} | {_dec_comma(emissions)} "
                f"| {_dec(_pct_of(emissions, total))}% "
                f"| {src.get('data_quality', '-')} |"
            )
        return "\n".join(lines)

    def _md_scope2(self, data: Dict[str, Any]) -> str:
        scope2 = data.get("scope2", {})
        loc = Decimal(str(scope2.get("location_tco2e", 0)))
        mkt = Decimal(str(scope2.get("market_tco2e", 0)))
        details = scope2.get("details", {})
        lines = [
            f"## 4. Scope 2 - Indirect Energy Emissions\n",
            "| Method | Emissions (tCO2e) |",
            "|--------|------------------:|",
            f"| Location-Based | {_dec_comma(loc)} |",
            f"| Market-Based | {_dec_comma(mkt)} |",
            f"| **Difference** | **{_dec_comma(loc - mkt)}** |\n",
        ]
        elec_consumed = details.get("electricity_consumed_mwh", 0)
        renewable_pct = details.get("renewable_pct", 0)
        grid_factor = details.get("grid_emission_factor", 0)
        lines.append(
            f"- **Electricity Consumed:** {_dec_comma(elec_consumed)} MWh\n"
            f"- **Renewable Share:** {_dec(renewable_pct)}%\n"
            f"- **Grid Emission Factor:** {_dec(grid_factor, 4)} tCO2e/MWh"
        )
        return "\n".join(lines)

    def _md_scope3(self, data: Dict[str, Any]) -> str:
        scope3_cats = data.get("scope3_categories", {})
        lines = [
            f"## 5. Scope 3 - Value Chain Emissions\n",
            "| Cat | Category | Emissions (tCO2e) | Relevance | Method | Data Quality |",
            "|----:|----------|------------------:|-----------|--------|-------------|",
        ]
        total_s3 = Decimal("0")
        for cat in _SCOPE3_CATEGORIES:
            cat_data = scope3_cats.get(cat["num"], {})
            emissions = Decimal(str(cat_data.get("emissions_tco2e", 0)))
            total_s3 += emissions
            relevance = cat_data.get("relevance", "not_assessed")
            method = cat_data.get("method", "-")
            dq = cat_data.get("data_quality", "-")
            lines.append(
                f"| {cat['num']} | {cat['name']} | {_dec_comma(emissions)} "
                f"| {relevance} | {method} | {dq} |"
            )
        lines.append(
            f"| | **Total Scope 3** | **{_dec_comma(total_s3)}** | | | |"
        )
        return "\n".join(lines)

    def _md_total_summary(self, data: Dict[str, Any]) -> str:
        scope1 = data.get("scope1", {})
        scope2 = data.get("scope2", {})
        scope3_cats = data.get("scope3_categories", {})
        s1 = Decimal(str(scope1.get("total_tco2e", 0)))
        s2_loc = Decimal(str(scope2.get("location_tco2e", 0)))
        s2_mkt = Decimal(str(scope2.get("market_tco2e", 0)))
        s3 = sum(
            Decimal(str(scope3_cats.get(c["num"], {}).get("emissions_tco2e", 0)))
            for c in _SCOPE3_CATEGORIES
        )
        total_loc = s1 + s2_loc + s3
        total_mkt = s1 + s2_mkt + s3
        lines = [
            "## 6. Total Emissions Summary\n",
            "| Scope | Emissions (tCO2e) | Share (Location) |",
            "|-------|------------------:|-----------------:|",
            f"| Scope 1 | {_dec_comma(s1)} | {_dec(_pct_of(s1, total_loc))}% |",
            f"| Scope 2 (Location) | {_dec_comma(s2_loc)} | {_dec(_pct_of(s2_loc, total_loc))}% |",
            f"| Scope 3 | {_dec_comma(s3)} | {_dec(_pct_of(s3, total_loc))}% |",
            f"| **Total (Location)** | **{_dec_comma(total_loc)}** | **100.00%** |",
            f"| **Total (Market)** | **{_dec_comma(total_mkt)}** | - |",
        ]
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality", {})
        matrix = dq.get("matrix", [])
        overall = dq.get("overall_score", "N/A")
        lines = [
            "## 7. Data Quality Matrix\n",
            f"**Overall Data Quality Score:** {overall}\n",
            "| Scope / Source | Quality Level | Score | Description |",
            "|---------------|-------------|------:|------------|",
        ]
        for item in matrix:
            level = item.get("level", "low")
            desc = _DQ_LEVELS.get(level, level)
            lines.append(
                f"| {item.get('source', '-')} | {level.upper()} "
                f"| {_dec(item.get('score', 0))} | {desc} |"
            )
        return "\n".join(lines)

    def _md_base_year(self, data: Dict[str, Any]) -> str:
        stmt = data.get("base_year_statement", {})
        return (
            f"## 8. Base Year Statement\n\n"
            f"- **Base Year:** {stmt.get('base_year', data.get('base_year', 'N/A'))}\n"
            f"- **Rationale:** {stmt.get('rationale', 'Earliest year with reliable data')}\n"
            f"- **Recalculation Policy:** {stmt.get('recalculation_policy', 'Structural changes >5% trigger recalculation')}\n"
            f"- **Significance Threshold:** {stmt.get('significance_threshold', '5%')}\n"
            f"- **Adjustments Made:** {stmt.get('adjustments', 'None')}"
        )

    def _md_emission_factors(self, data: Dict[str, Any]) -> str:
        factors = data.get("emission_factors", [])
        lines = [
            "## 9. Emission Factors Used\n",
            "| Source | Factor | Unit | Database | Year |",
            "|--------|-------:|------|----------|------|",
        ]
        for ef in factors:
            lines.append(
                f"| {ef.get('source', '-')} | {_dec(ef.get('factor', 0), 4)} "
                f"| {ef.get('unit', '-')} | {ef.get('database', '-')} "
                f"| {ef.get('year', '-')} |"
            )
        if not factors:
            lines.append("| _No emission factors specified_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}*  \n"
            f"*GHG Protocol Corporate Standard compliant baseline report.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            "background:#f5f7f5;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;padding-left:12px;}"
            "h3{color:#388e3c;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".dq-high{background:#c8e6c9;color:#1b5e20;font-weight:600;}"
            ".dq-medium{background:#fff9c4;color:#f57f17;font-weight:600;}"
            ".dq-low{background:#ffecb3;color:#e65100;font-weight:600;}"
            ".dq-very_low{background:#ffcdd2;color:#c62828;font-weight:600;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".waterfall-bar{height:20px;border-radius:4px;display:inline-block;}"
            ".s1-bar{background:#43a047;}"
            ".s2-bar{background:#66bb6a;}"
            ".s3-bar{background:#a5d6a7;}"
            ".relevance-relevant{color:#1b5e20;font-weight:600;}"
            ".relevance-not_relevant{color:#9e9e9e;}"
            ".relevance-not_assessed{color:#e65100;font-style:italic;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>GHG Inventory Baseline Report</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Base Year:</strong> {data.get("base_year", "")} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        meth = data.get("methodology", {})
        return (
            f'<h2>1. Methodology</h2>\n'
            f'<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Standard</td><td>{meth.get("standard", "GHG Protocol Corporate Standard")}</td></tr>\n'
            f'<tr><td>Approach</td><td>{meth.get("approach", "Operational Control")}</td></tr>\n'
            f'<tr><td>GWP Source</td><td>{meth.get("gwp_source", "IPCC AR6")}</td></tr>\n'
            f'<tr><td>Gases</td><td>{meth.get("gases", "CO2, CH4, N2O, HFCs, PFCs, SF6, NF3")}</td></tr>\n'
            f'<tr><td>Verification</td><td>{meth.get("verification", "N/A")}</td></tr>\n'
            f'</table>'
        )

    def _html_boundary(self, data: Dict[str, Any]) -> str:
        boundary = data.get("boundary", {})
        entities = boundary.get("entities", [])
        rows = ""
        for e in entities:
            rows += (
                f'<tr><td>{e.get("name", "-")}</td><td>{e.get("country", "-")}</td>'
                f'<td>{e.get("scope", "-")}</td>'
                f'<td>{_dec(e.get("ownership_pct", 100))}%</td></tr>\n'
            )
        return (
            f'<h2>2. Organizational Boundary</h2>\n'
            f'<p><strong>Consolidation:</strong> {boundary.get("consolidation", "Operational Control")} | '
            f'<strong>Entities:</strong> {len(entities)}</p>\n'
            f'<table>\n'
            f'<tr><th>Entity</th><th>Country</th><th>Scope</th><th>Ownership</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_scope1(self, data: Dict[str, Any]) -> str:
        scope1 = data.get("scope1", {})
        sources = scope1.get("sources", [])
        total = Decimal(str(scope1.get("total_tco2e", 0)))
        rows = ""
        for src in sources:
            emissions = Decimal(str(src.get("emissions_tco2e", 0)))
            pct = float(_pct_of(emissions, total))
            rows += (
                f'<tr><td>{src.get("type", "-")}</td>'
                f'<td>{_dec_comma(emissions)}</td>'
                f'<td>{_dec(_pct_of(emissions, total))}%</td>'
                f'<td>{src.get("data_quality", "-")}</td>'
                f'<td><div class="waterfall-bar s1-bar" style="width:{max(pct, 2)}%">&nbsp;</div></td></tr>\n'
            )
        return (
            f'<h2>3. Scope 1 - Direct Emissions</h2>\n'
            f'<p><strong>Total:</strong> {_dec_comma(total)} tCO2e</p>\n'
            f'<table>\n'
            f'<tr><th>Source</th><th>Emissions (tCO2e)</th><th>Share</th>'
            f'<th>Data Quality</th><th>Distribution</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_scope2(self, data: Dict[str, Any]) -> str:
        scope2 = data.get("scope2", {})
        loc = Decimal(str(scope2.get("location_tco2e", 0)))
        mkt = Decimal(str(scope2.get("market_tco2e", 0)))
        diff = loc - mkt
        details = scope2.get("details", {})
        return (
            f'<h2>4. Scope 2 - Indirect Energy Emissions</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Location-Based</div>'
            f'<div class="card-value">{_dec_comma(loc)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Market-Based</div>'
            f'<div class="card-value">{_dec_comma(mkt)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Difference</div>'
            f'<div class="card-value">{_dec_comma(diff)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Electricity Consumed</td><td>{_dec_comma(details.get("electricity_consumed_mwh", 0))} MWh</td></tr>\n'
            f'<tr><td>Renewable Share</td><td>{_dec(details.get("renewable_pct", 0))}%</td></tr>\n'
            f'<tr><td>Grid Emission Factor</td><td>{_dec(details.get("grid_emission_factor", 0), 4)} tCO2e/MWh</td></tr>\n'
            f'</table>'
        )

    def _html_scope3(self, data: Dict[str, Any]) -> str:
        scope3_cats = data.get("scope3_categories", {})
        rows = ""
        total_s3 = Decimal("0")
        for cat in _SCOPE3_CATEGORIES:
            cat_data = scope3_cats.get(cat["num"], {})
            emissions = Decimal(str(cat_data.get("emissions_tco2e", 0)))
            total_s3 += emissions
            relevance = cat_data.get("relevance", "not_assessed")
            rel_cls = f"relevance-{relevance}"
            rows += (
                f'<tr><td>{cat["num"]}</td><td>{cat["name"]}</td>'
                f'<td>{_dec_comma(emissions)}</td>'
                f'<td class="{rel_cls}">{relevance}</td>'
                f'<td>{cat_data.get("method", "-")}</td>'
                f'<td>{cat_data.get("data_quality", "-")}</td></tr>\n'
            )
        rows += (
            f'<tr><th></th><th>Total Scope 3</th>'
            f'<th>{_dec_comma(total_s3)}</th><th></th><th></th><th></th></tr>'
        )
        return (
            f'<h2>5. Scope 3 - Value Chain Emissions</h2>\n'
            f'<table>\n'
            f'<tr><th>Cat</th><th>Category</th><th>Emissions (tCO2e)</th>'
            f'<th>Relevance</th><th>Method</th><th>Data Quality</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_total_summary(self, data: Dict[str, Any]) -> str:
        scope1 = data.get("scope1", {})
        scope2 = data.get("scope2", {})
        scope3_cats = data.get("scope3_categories", {})
        s1 = Decimal(str(scope1.get("total_tco2e", 0)))
        s2_loc = Decimal(str(scope2.get("location_tco2e", 0)))
        s3 = sum(
            Decimal(str(scope3_cats.get(c["num"], {}).get("emissions_tco2e", 0)))
            for c in _SCOPE3_CATEGORIES
        )
        total = s1 + s2_loc + s3
        s1_pct = float(_pct_of(s1, total)) if total > 0 else 0
        s2_pct = float(_pct_of(s2_loc, total)) if total > 0 else 0
        s3_pct = float(_pct_of(s3, total)) if total > 0 else 0
        return (
            f'<h2>6. Total Emissions Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Scope 1</div>'
            f'<div class="card-value">{_dec_comma(s1)}</div><div class="card-unit">tCO2e ({_dec(s1_pct)}%)</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 2</div>'
            f'<div class="card-value">{_dec_comma(s2_loc)}</div><div class="card-unit">tCO2e ({_dec(s2_pct)}%)</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 3</div>'
            f'<div class="card-value">{_dec_comma(s3)}</div><div class="card-unit">tCO2e ({_dec(s3_pct)}%)</div></div>\n'
            f'  <div class="card"><div class="card-label">Grand Total</div>'
            f'<div class="card-value">{_dec_comma(total)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality", {})
        matrix = dq.get("matrix", [])
        overall = dq.get("overall_score", "N/A")
        rows = ""
        for item in matrix:
            level = item.get("level", "low")
            dq_cls = f"dq-{level}"
            rows += (
                f'<tr><td>{item.get("source", "-")}</td>'
                f'<td class="{dq_cls}">{level.upper()}</td>'
                f'<td>{_dec(item.get("score", 0))}</td>'
                f'<td>{_DQ_LEVELS.get(level, level)}</td></tr>\n'
            )
        return (
            f'<h2>7. Data Quality Matrix</h2>\n'
            f'<p><strong>Overall Score:</strong> {overall}</p>\n'
            f'<table>\n'
            f'<tr><th>Source</th><th>Quality Level</th><th>Score</th><th>Description</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_base_year(self, data: Dict[str, Any]) -> str:
        stmt = data.get("base_year_statement", {})
        return (
            f'<h2>8. Base Year Statement</h2>\n'
            f'<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Base Year</td><td>{stmt.get("base_year", data.get("base_year", "N/A"))}</td></tr>\n'
            f'<tr><td>Rationale</td><td>{stmt.get("rationale", "Earliest year with reliable data")}</td></tr>\n'
            f'<tr><td>Recalculation Policy</td><td>{stmt.get("recalculation_policy", "Structural changes >5%")}</td></tr>\n'
            f'<tr><td>Significance Threshold</td><td>{stmt.get("significance_threshold", "5%")}</td></tr>\n'
            f'<tr><td>Adjustments</td><td>{stmt.get("adjustments", "None")}</td></tr>\n'
            f'</table>'
        )

    def _html_emission_factors(self, data: Dict[str, Any]) -> str:
        factors = data.get("emission_factors", [])
        rows = ""
        for ef in factors:
            rows += (
                f'<tr><td>{ef.get("source", "-")}</td>'
                f'<td>{_dec(ef.get("factor", 0), 4)}</td>'
                f'<td>{ef.get("unit", "-")}</td>'
                f'<td>{ef.get("database", "-")}</td>'
                f'<td>{ef.get("year", "-")}</td></tr>\n'
            )
        return (
            f'<h2>9. Emission Factors Used</h2>\n'
            f'<table>\n'
            f'<tr><th>Source</th><th>Factor</th><th>Unit</th><th>Database</th><th>Year</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
