# -*- coding: utf-8 -*-
"""
FootprintReportTemplate - GHG footprint report for PACK-024.

Renders a comprehensive GHG footprint report with scope breakdowns,
Scope 3 category analysis, emission factor sources, data quality
assessment, uncertainty analysis, and year-over-year comparisons.

Sections:
    1. Executive Summary (total emissions, key metrics)
    2. Scope Breakdown (S1, S2 location/market, S3 total)
    3. Scope 3 Category Analysis (15 categories)
    4. Emission Factor Sources
    5. Data Quality Assessment
    6. Uncertainty Analysis
    7. Year-over-Year Comparison
    8. Methodology Notes

Author: GreenLang Team
Version: 24.0.0
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

_MODULE_VERSION = "24.0.0"

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

def _pct(val: Any) -> str:
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)

class FootprintReportTemplate:
    """GHG footprint report template for PACK-024 Carbon Neutral Pack."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_scope_breakdown(data),
            self._md_scope3_analysis(data),
            self._md_ef_sources(data),
            self._md_data_quality(data),
            self._md_uncertainty(data),
            self._md_yoy_comparison(data),
            self._md_methodology(data),
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
            self._html_executive_summary(data),
            self._html_scope_breakdown(data),
            self._html_scope3_analysis(data),
            self._html_data_quality(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>GHG Footprint Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "footprint_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "total_emissions_tco2e": data.get("total_emissions_tco2e", 0),
            "scope_breakdowns": data.get("scope_breakdowns", []),
            "scope3_categories": data.get("scope3_categories", []),
            "data_quality": data.get("data_quality", {}),
            "uncertainty": data.get("uncertainty", {}),
            "methodology": data.get("methodology", {}),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # -- Markdown sections --

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# GHG Footprint Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** PAS 2060:2014 / GHG Protocol\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        total = data.get("total_emissions_tco2e", 0)
        s1 = data.get("scope1_tco2e", 0)
        s2 = data.get("scope2_tco2e", 0)
        s3 = data.get("scope3_tco2e", 0)
        dq = data.get("data_quality", {}).get("overall_score", 0)
        return (
            f"## 1. Executive Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Emissions | {_dec_comma(total, 0)} tCO2e |\n"
            f"| Scope 1 | {_dec_comma(s1, 0)} tCO2e ({_pct(s1 / max(total, 1) * 100)}) |\n"
            f"| Scope 2 | {_dec_comma(s2, 0)} tCO2e ({_pct(s2 / max(total, 1) * 100)}) |\n"
            f"| Scope 3 | {_dec_comma(s3, 0)} tCO2e ({_pct(s3 / max(total, 1) * 100)}) |\n"
            f"| Data Quality Score | {_dec(dq, 1)}/100 |\n"
            f"| Reporting Standard | GHG Protocol Corporate Standard |"
        )

    def _md_scope_breakdown(self, data: Dict[str, Any]) -> str:
        breakdowns = data.get("scope_breakdowns", [])
        lines = [
            "## 2. Scope Breakdown\n",
            "| Scope | Emissions (tCO2e) | % of Total | Sources | Data Quality | Uncertainty |",
            "|-------|------------------:|:----------:|:-------:|:------------:|:-----------:|",
        ]
        for bd in breakdowns:
            lines.append(
                f"| {bd.get('scope', '-')} "
                f"| {_dec_comma(bd.get('total_tco2e', 0), 0)} "
                f"| {_pct(bd.get('pct_of_total', 0))} "
                f"| {bd.get('sources_count', 0)} "
                f"| {bd.get('data_quality_tier', '-')} "
                f"| +/-{_pct(bd.get('uncertainty_pct', 0))} |"
            )
        if not breakdowns:
            lines.append("| - | _No data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_scope3_analysis(self, data: Dict[str, Any]) -> str:
        cats = data.get("scope3_categories", [])
        lines = [
            "## 3. Scope 3 Category Analysis\n",
            "| Cat | Category Name | Emissions (tCO2e) | % of S3 | % of Total | Method | Quality |",
            "|:---:|--------------|------------------:|:-------:|:----------:|--------|:-------:|",
        ]
        for cat in cats:
            lines.append(
                f"| {cat.get('category_id', '-')} "
                f"| {cat.get('category_name', '-')} "
                f"| {_dec_comma(cat.get('emissions_tco2e', 0), 0)} "
                f"| {_pct(cat.get('pct_of_scope3', 0))} "
                f"| {_pct(cat.get('pct_of_total', 0))} "
                f"| {cat.get('methodology', '-')} "
                f"| {cat.get('data_quality_tier', '-')} |"
            )
        if not cats:
            lines.append("| - | _No categories_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_ef_sources(self, data: Dict[str, Any]) -> str:
        sources = data.get("emission_factor_sources", [])
        lines = ["## 4. Emission Factor Sources\n"]
        if sources:
            lines.append("| Source | Version | Scope Coverage | GWP |")
            lines.append("|--------|---------|:-------------:|:---:|")
            for s in sources:
                lines.append(
                    f"| {s.get('name', '-')} | {s.get('version', '-')} "
                    f"| {s.get('scope', '-')} | {s.get('gwp', 'AR6')} |"
                )
        else:
            lines.append("_Emission factor sources documented in methodology section._")
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality", {})
        return (
            f"## 5. Data Quality Assessment\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Overall Score | {_dec(dq.get('overall_score', 0), 1)}/100 |\n"
            f"| Primary Data | {_pct(dq.get('primary_data_pct', 0))} |\n"
            f"| Secondary Data | {_pct(dq.get('secondary_data_pct', 0))} |\n"
            f"| Estimated Data | {_pct(dq.get('estimated_data_pct', 0))} |\n"
            f"| Completeness | {_pct(dq.get('completeness_pct', 0))} |\n"
            f"| PAS 2060 Threshold Met | {dq.get('meets_pas2060_threshold', 'N/A')} |"
        )

    def _md_uncertainty(self, data: Dict[str, Any]) -> str:
        unc = data.get("uncertainty", {})
        return (
            f"## 6. Uncertainty Analysis\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Overall Uncertainty | +/-{_pct(unc.get('overall_pct', 0))} |\n"
            f"| Scope 1 Uncertainty | +/-{_pct(unc.get('scope1_pct', 0))} |\n"
            f"| Scope 2 Uncertainty | +/-{_pct(unc.get('scope2_pct', 0))} |\n"
            f"| Scope 3 Uncertainty | +/-{_pct(unc.get('scope3_pct', 0))} |\n"
            f"| Method | {unc.get('method', 'Root-sum-square (IPCC)')} |"
        )

    def _md_yoy_comparison(self, data: Dict[str, Any]) -> str:
        yoy = data.get("year_over_year", {})
        lines = ["## 7. Year-over-Year Comparison\n"]
        if yoy:
            lines.append(f"| Metric | Value |\n|--------|-------|\n")
            lines.append(f"| Previous Year Emissions | {_dec_comma(yoy.get('previous_tco2e', 0), 0)} tCO2e |")
            lines.append(f"| Current Year Emissions | {_dec_comma(yoy.get('current_tco2e', 0), 0)} tCO2e |")
            lines.append(f"| Change | {_pct(yoy.get('change_pct', 0))} |")
            lines.append(f"| Absolute Change | {_dec_comma(yoy.get('absolute_change_tco2e', 0), 0)} tCO2e |")
        else:
            lines.append("_Year-over-year comparison not available (first reporting year)._")
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        meth = data.get("methodology", {})
        return (
            f"## 8. Methodology Notes\n\n"
            f"- **Standard:** {meth.get('standard', 'GHG Protocol Corporate Standard')}\n"
            f"- **Boundary:** {meth.get('boundary', 'Operational control')}\n"
            f"- **GWP Values:** {meth.get('gwp', 'IPCC AR6 (100-year)')}\n"
            f"- **Base Year:** {meth.get('base_year', 'N/A')}\n"
            f"- **Biogenic Treatment:** {meth.get('biogenic', 'Reported separately')}\n"
            f"- **Exclusions:** {meth.get('exclusions', 'None')}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}*  \n"
            f"*Quantification per GHG Protocol Corporate Standard and PAS 2060:2014.*"
        )

    # -- HTML sections --

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;padding-left:12px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>GHG Footprint Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        total = data.get("total_emissions_tco2e", 0)
        s1 = data.get("scope1_tco2e", 0)
        s2 = data.get("scope2_tco2e", 0)
        s3 = data.get("scope3_tco2e", 0)
        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Emissions</div>'
            f'<div class="card-value">{_dec_comma(total, 0)}</div> tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Scope 1</div>'
            f'<div class="card-value">{_dec_comma(s1, 0)}</div> tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Scope 2</div>'
            f'<div class="card-value">{_dec_comma(s2, 0)}</div> tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Scope 3</div>'
            f'<div class="card-value">{_dec_comma(s3, 0)}</div> tCO2e</div>\n'
            f'</div>'
        )

    def _html_scope_breakdown(self, data: Dict[str, Any]) -> str:
        breakdowns = data.get("scope_breakdowns", [])
        rows = ""
        for bd in breakdowns:
            rows += (
                f'<tr><td>{bd.get("scope", "-")}</td>'
                f'<td>{_dec_comma(bd.get("total_tco2e", 0), 0)}</td>'
                f'<td>{_pct(bd.get("pct_of_total", 0))}</td>'
                f'<td>{bd.get("data_quality_tier", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. Scope Breakdown</h2>\n'
            f'<table><tr><th>Scope</th><th>Emissions (tCO2e)</th>'
            f'<th>% of Total</th><th>Data Quality</th></tr>\n{rows}</table>'
        )

    def _html_scope3_analysis(self, data: Dict[str, Any]) -> str:
        cats = data.get("scope3_categories", [])
        rows = ""
        for cat in cats:
            rows += (
                f'<tr><td>{cat.get("category_id", "-")}</td>'
                f'<td>{cat.get("category_name", "-")}</td>'
                f'<td>{_dec_comma(cat.get("emissions_tco2e", 0), 0)}</td>'
                f'<td>{_pct(cat.get("pct_of_scope3", 0))}</td></tr>\n'
            )
        return (
            f'<h2>3. Scope 3 Categories</h2>\n'
            f'<table><tr><th>Cat</th><th>Category</th>'
            f'<th>Emissions (tCO2e)</th><th>% of S3</th></tr>\n{rows}</table>'
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality", {})
        return (
            f'<h2>5. Data Quality</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Quality Score</div>'
            f'<div class="card-value">{_dec(dq.get("overall_score", 0), 1)}</div>/100</div>\n'
            f'  <div class="card"><div class="card-label">Primary Data</div>'
            f'<div class="card-value">{_pct(dq.get("primary_data_pct", 0))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Completeness</div>'
            f'<div class="card-value">{_pct(dq.get("completeness_pct", 0))}</div></div>\n'
            f'</div>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}'
            f'</div>'
        )
