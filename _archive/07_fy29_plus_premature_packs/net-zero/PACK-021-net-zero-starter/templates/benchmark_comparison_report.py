# -*- coding: utf-8 -*-
"""
BenchmarkComparisonReportTemplate - Peer benchmarking comparison for PACK-021.

Renders a peer benchmarking comparison report with company profile, sector
context, KPI comparison tables, percentile rankings, strengths & gaps,
best practice insights, and improvement opportunities.

Sections:
    1. Company Profile
    2. Sector Context
    3. KPI Comparison Table
    4. Percentile Rankings
    5. Strengths & Gaps
    6. Best Practice Insights
    7. Improvement Opportunities

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

class BenchmarkComparisonReportTemplate:
    """
    Peer benchmarking comparison report template.

    Compares organizational climate performance against sector peers,
    identifies strengths and gaps, highlights best practices, and
    surfaces improvement opportunities.

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
            self._md_company_profile(data),
            self._md_sector_context(data),
            self._md_kpi_comparison(data),
            self._md_percentile_rankings(data),
            self._md_strengths_gaps(data),
            self._md_best_practices(data),
            self._md_improvement_opportunities(data),
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
            self._html_company_profile(data),
            self._html_sector_context(data),
            self._html_kpi_comparison(data),
            self._html_percentile_rankings(data),
            self._html_strengths_gaps(data),
            self._html_best_practices(data),
            self._html_improvement_opportunities(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Benchmark Comparison Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "benchmark_comparison_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "company_profile": data.get("company_profile", {}),
            "sector_context": data.get("sector_context", {}),
            "kpi_comparison": data.get("kpi_comparison", []),
            "percentile_rankings": data.get("percentile_rankings", []),
            "strengths": data.get("strengths", []),
            "gaps": data.get("gaps", []),
            "best_practices": data.get("best_practices", []),
            "improvement_opportunities": data.get("improvement_opportunities", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Peer Benchmark Comparison Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Sector:** {data.get('sector_context', {}).get('sector', '')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_company_profile(self, data: Dict[str, Any]) -> str:
        profile = data.get("company_profile", {})
        return (
            "## 1. Company Profile\n\n"
            f"| Attribute | Value |\n|-----------|-------|\n"
            f"| Organization | {data.get('org_name', '')} |\n"
            f"| Sector | {profile.get('sector', 'N/A')} |\n"
            f"| Sub-Sector | {profile.get('sub_sector', 'N/A')} |\n"
            f"| Revenue | {profile.get('revenue', 'N/A')} |\n"
            f"| Employees | {profile.get('employees', 'N/A')} |\n"
            f"| Total Emissions | {_dec_comma(profile.get('total_emissions_tco2e', 0))} tCO2e |\n"
            f"| SBTi Status | {profile.get('sbti_status', 'N/A')} |\n"
            f"| Net Zero Year | {profile.get('net_zero_year', 'N/A')} |"
        )

    def _md_sector_context(self, data: Dict[str, Any]) -> str:
        ctx = data.get("sector_context", {})
        return (
            "## 2. Sector Context\n\n"
            f"- **Sector:** {ctx.get('sector', 'N/A')}\n"
            f"- **Peer Group Size:** {ctx.get('peer_group_size', 'N/A')}\n"
            f"- **Region:** {ctx.get('region', 'N/A')}\n"
            f"- **Benchmark Source:** {ctx.get('benchmark_source', 'N/A')}\n"
            f"- **Data Year:** {ctx.get('data_year', 'N/A')}\n"
            f"- **Sector Average Emissions:** {_dec_comma(ctx.get('sector_avg_emissions_tco2e', 0))} tCO2e\n"
            f"- **Sector Leader Emissions:** {_dec_comma(ctx.get('sector_leader_emissions_tco2e', 0))} tCO2e"
        )

    def _md_kpi_comparison(self, data: Dict[str, Any]) -> str:
        kpis = data.get("kpi_comparison", [])
        lines = [
            "## 3. KPI Comparison\n",
            "| KPI | Your Value | Unit | Sector Avg | Sector Leader | vs Avg (%) | vs Leader (%) |",
            "|-----|----------:|------|----------:|--------------:|----------:|--------------:|",
        ]
        for kpi in kpis:
            your_val = Decimal(str(kpi.get("your_value", 0)))
            avg_val = Decimal(str(kpi.get("sector_avg", 0)))
            leader_val = Decimal(str(kpi.get("sector_leader", 0)))
            vs_avg = ((your_val - avg_val) / avg_val * Decimal("100")) if avg_val != 0 else Decimal("0")
            vs_leader = ((your_val - leader_val) / leader_val * Decimal("100")) if leader_val != 0 else Decimal("0")
            lines.append(
                f"| {kpi.get('name', '-')} | {_dec_comma(your_val)} "
                f"| {kpi.get('unit', '-')} | {_dec_comma(avg_val)} "
                f"| {_dec_comma(leader_val)} "
                f"| {'+' if vs_avg > 0 else ''}{_dec(vs_avg)}% "
                f"| {'+' if vs_leader > 0 else ''}{_dec(vs_leader)}% |"
            )
        if not kpis:
            lines.append("| _No KPIs defined_ | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_percentile_rankings(self, data: Dict[str, Any]) -> str:
        rankings = data.get("percentile_rankings", [])
        lines = [
            "## 4. Percentile Rankings\n",
            "| KPI | Percentile | Quartile | Position |",
            "|-----|----------:|----------|----------|",
        ]
        for r in rankings:
            pctile = r.get("percentile", 0)
            quartile = "Q1 (Top)" if pctile >= 75 else "Q2" if pctile >= 50 else "Q3" if pctile >= 25 else "Q4 (Bottom)"
            position = "Leader" if pctile >= 90 else "Above Avg" if pctile >= 50 else "Below Avg" if pctile >= 25 else "Laggard"
            lines.append(
                f"| {r.get('kpi', '-')} | P{_dec(pctile, 0)} "
                f"| {quartile} | {position} |"
            )
        if not rankings:
            lines.append("| _No rankings available_ | - | - | - |")
        return "\n".join(lines)

    def _md_strengths_gaps(self, data: Dict[str, Any]) -> str:
        strengths = data.get("strengths", [])
        gaps = data.get("gaps", [])
        lines = ["## 5. Strengths & Gaps\n"]
        lines.append("### Strengths (Above Sector Average)\n")
        if strengths:
            for s in strengths:
                lines.append(f"- **{s.get('kpi', '')}:** {s.get('description', '')}")
        else:
            lines.append("_No areas significantly above sector average._")
        lines.append("\n### Gaps (Below Sector Average)\n")
        if gaps:
            for g in gaps:
                lines.append(f"- **{g.get('kpi', '')}:** {g.get('description', '')}")
        else:
            lines.append("_No significant gaps identified._")
        return "\n".join(lines)

    def _md_best_practices(self, data: Dict[str, Any]) -> str:
        practices = data.get("best_practices", [])
        lines = ["## 6. Best Practice Insights\n"]
        if practices:
            for i, bp in enumerate(practices, 1):
                lines.append(f"### {i}. {bp.get('title', '')}\n")
                lines.append(f"**Source:** {bp.get('source', '-')}")
                lines.append(f"**Sector Relevance:** {bp.get('relevance', '-')}\n")
                lines.append(f"{bp.get('description', '')}\n")
        else:
            lines.append("_No best practice insights available._")
        return "\n".join(lines)

    def _md_improvement_opportunities(self, data: Dict[str, Any]) -> str:
        opportunities = data.get("improvement_opportunities", [])
        lines = [
            "## 7. Improvement Opportunities\n",
            "| # | Opportunity | KPI Impacted | Current Gap | Potential Improvement | Priority |",
            "|---|-----------|-------------|------------|---------------------|----------|",
        ]
        for i, opp in enumerate(opportunities, 1):
            lines.append(
                f"| {i} | {opp.get('opportunity', '-')} "
                f"| {opp.get('kpi', '-')} "
                f"| {opp.get('current_gap', '-')} "
                f"| {opp.get('potential_improvement', '-')} "
                f"| {opp.get('priority', '-')} |"
            )
        if not opportunities:
            lines.append("| - | _No improvement opportunities identified_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}*  \n"
            f"*Peer benchmarking based on publicly available data.*"
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
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".percentile-bar{background:#e0e0e0;border-radius:6px;height:20px;"
            "overflow:hidden;position:relative;}"
            ".percentile-fill{height:100%;border-radius:6px;}"
            ".p-q1{background:linear-gradient(90deg,#43a047,#66bb6a);}"
            ".p-q2{background:linear-gradient(90deg,#66bb6a,#aed581);}"
            ".p-q3{background:linear-gradient(90deg,#ff8f00,#ffb300);}"
            ".p-q4{background:linear-gradient(90deg,#e53935,#ef5350);}"
            ".better{color:#1b5e20;font-weight:600;}"
            ".worse{color:#c62828;font-weight:600;}"
            ".neutral{color:#616161;}"
            ".strength-card{border:1px solid #c8e6c9;border-left:4px solid #43a047;"
            "border-radius:8px;padding:14px;margin:8px 0;}"
            ".gap-card{border:1px solid #ffcdd2;border-left:4px solid #c62828;"
            "border-radius:8px;padding:14px;margin:8px 0;}"
            ".bp-card{border:1px solid #e3f2fd;border-left:4px solid #1565c0;"
            "border-radius:8px;padding:16px;margin:12px 0;}"
            ".priority-high{color:#c62828;font-weight:600;}"
            ".priority-medium{color:#e65100;font-weight:600;}"
            ".priority-low{color:#1b5e20;font-weight:600;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        ctx = data.get("sector_context", {})
        return (
            f'<h1>Peer Benchmark Comparison Report</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Sector:</strong> {ctx.get("sector", "")} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_company_profile(self, data: Dict[str, Any]) -> str:
        profile = data.get("company_profile", {})
        return (
            f'<h2>1. Company Profile</h2>\n'
            f'<table>\n'
            f'<tr><th>Attribute</th><th>Value</th></tr>\n'
            f'<tr><td>Organization</td><td>{data.get("org_name", "")}</td></tr>\n'
            f'<tr><td>Sector</td><td>{profile.get("sector", "N/A")}</td></tr>\n'
            f'<tr><td>Sub-Sector</td><td>{profile.get("sub_sector", "N/A")}</td></tr>\n'
            f'<tr><td>Revenue</td><td>{profile.get("revenue", "N/A")}</td></tr>\n'
            f'<tr><td>Employees</td><td>{profile.get("employees", "N/A")}</td></tr>\n'
            f'<tr><td>Total Emissions</td><td>{_dec_comma(profile.get("total_emissions_tco2e", 0))} tCO2e</td></tr>\n'
            f'<tr><td>SBTi Status</td><td>{profile.get("sbti_status", "N/A")}</td></tr>\n'
            f'<tr><td>Net Zero Year</td><td>{profile.get("net_zero_year", "N/A")}</td></tr>\n'
            f'</table>'
        )

    def _html_sector_context(self, data: Dict[str, Any]) -> str:
        ctx = data.get("sector_context", {})
        return (
            f'<h2>2. Sector Context</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Peer Group</div>'
            f'<div class="card-value">{ctx.get("peer_group_size", 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Sector Avg Emissions</div>'
            f'<div class="card-value">{_dec_comma(ctx.get("sector_avg_emissions_tco2e", 0))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Sector Leader</div>'
            f'<div class="card-value">{_dec_comma(ctx.get("sector_leader_emissions_tco2e", 0))}</div></div>\n'
            f'</div>'
        )

    def _html_kpi_comparison(self, data: Dict[str, Any]) -> str:
        kpis = data.get("kpi_comparison", [])
        rows = ""
        for kpi in kpis:
            your_val = Decimal(str(kpi.get("your_value", 0)))
            avg_val = Decimal(str(kpi.get("sector_avg", 0)))
            leader_val = Decimal(str(kpi.get("sector_leader", 0)))
            lower_better = kpi.get("lower_is_better", True)

            if avg_val != 0:
                vs_avg = ((your_val - avg_val) / avg_val * Decimal("100"))
            else:
                vs_avg = Decimal("0")

            is_better_than_avg = (vs_avg < 0) if lower_better else (vs_avg > 0)
            avg_cls = "better" if is_better_than_avg else "worse" if vs_avg != 0 else "neutral"

            rows += (
                f'<tr><td>{kpi.get("name", "-")}</td>'
                f'<td>{_dec_comma(your_val)}</td>'
                f'<td>{kpi.get("unit", "-")}</td>'
                f'<td>{_dec_comma(avg_val)}</td>'
                f'<td>{_dec_comma(leader_val)}</td>'
                f'<td class="{avg_cls}">{"+" if vs_avg > 0 else ""}{_dec(vs_avg)}%</td></tr>\n'
            )
        return (
            f'<h2>3. KPI Comparison</h2>\n'
            f'<table>\n'
            f'<tr><th>KPI</th><th>Your Value</th><th>Unit</th>'
            f'<th>Sector Avg</th><th>Leader</th><th>vs Avg</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_percentile_rankings(self, data: Dict[str, Any]) -> str:
        rankings = data.get("percentile_rankings", [])
        rows = ""
        for r in rankings:
            pctile = float(Decimal(str(r.get("percentile", 0))))
            q_cls = "p-q1" if pctile >= 75 else "p-q2" if pctile >= 50 else "p-q3" if pctile >= 25 else "p-q4"
            quartile = "Q1 (Top)" if pctile >= 75 else "Q2" if pctile >= 50 else "Q3" if pctile >= 25 else "Q4"
            rows += (
                f'<tr><td>{r.get("kpi", "-")}</td>'
                f'<td><div class="percentile-bar"><div class="percentile-fill {q_cls}" '
                f'style="width:{pctile}%"></div></div></td>'
                f'<td>P{_dec(pctile, 0)}</td>'
                f'<td>{quartile}</td></tr>\n'
            )
        return (
            f'<h2>4. Percentile Rankings</h2>\n'
            f'<table>\n'
            f'<tr><th>KPI</th><th>Percentile Bar</th><th>Rank</th><th>Quartile</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_strengths_gaps(self, data: Dict[str, Any]) -> str:
        strengths = data.get("strengths", [])
        gaps = data.get("gaps", [])
        s_html = ""
        for s in strengths:
            s_html += (
                f'<div class="strength-card"><strong>{s.get("kpi", "")}</strong>: '
                f'{s.get("description", "")}</div>\n'
            )
        g_html = ""
        for g in gaps:
            g_html += (
                f'<div class="gap-card"><strong>{g.get("kpi", "")}</strong>: '
                f'{g.get("description", "")}</div>\n'
            )
        return (
            f'<h2>5. Strengths & Gaps</h2>\n'
            f'<h3>Strengths (Above Sector Average)</h3>\n{s_html}'
            f'<h3>Gaps (Below Sector Average)</h3>\n{g_html}'
        )

    def _html_best_practices(self, data: Dict[str, Any]) -> str:
        practices = data.get("best_practices", [])
        items = ""
        for i, bp in enumerate(practices, 1):
            items += (
                f'<div class="bp-card">'
                f'<strong>{i}. {bp.get("title", "")}</strong><br>'
                f'<small>Source: {bp.get("source", "-")} | '
                f'Relevance: {bp.get("relevance", "-")}</small>'
                f'<p>{bp.get("description", "")}</p></div>\n'
            )
        return f'<h2>6. Best Practice Insights</h2>\n{items}'

    def _html_improvement_opportunities(self, data: Dict[str, Any]) -> str:
        opportunities = data.get("improvement_opportunities", [])
        rows = ""
        for i, opp in enumerate(opportunities, 1):
            priority = opp.get("priority", "MEDIUM").lower()
            pri_cls = f"priority-{priority}"
            rows += (
                f'<tr><td>{i}</td><td>{opp.get("opportunity", "-")}</td>'
                f'<td>{opp.get("kpi", "-")}</td>'
                f'<td>{opp.get("current_gap", "-")}</td>'
                f'<td>{opp.get("potential_improvement", "-")}</td>'
                f'<td class="{pri_cls}">{opp.get("priority", "-")}</td></tr>\n'
            )
        return (
            f'<h2>7. Improvement Opportunities</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Opportunity</th><th>KPI</th>'
            f'<th>Gap</th><th>Potential</th><th>Priority</th></tr>\n'
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
