# -*- coding: utf-8 -*-
"""
SupplierEngagementReportTemplate - Supplier engagement program dashboard for PACK-022.

Renders a supplier engagement program report with tiering, engagement levels,
top emitters, RAG progress tracking, Scope 3 impact estimation, engagement
milestones, resource allocation, and recommendations.

Sections:
    1. Program Overview
    2. Supplier Tiering Summary (4-tier)
    3. Engagement Level Distribution
    4. Top 20 Suppliers by Emissions
    5. Program Progress (RAG dashboard)
    6. Scope 3 Impact Estimation
    7. Engagement Milestones Timeline
    8. Resource Allocation
    9. Recommendations

Author: GreenLang Team
Version: 22.0.0
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

_MODULE_VERSION = "22.0.0"

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
    try:
        p = Decimal(str(part))
        t = Decimal(str(total))
        if t == 0:
            return Decimal("0.00")
        return (p / t * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except Exception:
        return Decimal("0.00")

class SupplierEngagementReportTemplate:
    """
    Supplier engagement program dashboard report template.

    Tracks supplier tiering, engagement levels, emissions impact,
    program milestones, and resource allocation for Scope 3 reduction.

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
            self._md_program_overview(data),
            self._md_supplier_tiering(data),
            self._md_engagement_levels(data),
            self._md_top_suppliers(data),
            self._md_program_progress(data),
            self._md_scope3_impact(data),
            self._md_milestones(data),
            self._md_resource_allocation(data),
            self._md_recommendations(data),
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
            self._html_program_overview(data),
            self._html_supplier_tiering(data),
            self._html_engagement_levels(data),
            self._html_top_suppliers(data),
            self._html_program_progress(data),
            self._html_scope3_impact(data),
            self._html_milestones(data),
            self._html_resource_allocation(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Supplier Engagement Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        tiers = data.get("supplier_tiers", [])
        top_suppliers = data.get("top_suppliers", [])
        progress = data.get("program_progress", [])
        scope3 = data.get("scope3_impact", {})

        total_suppliers = sum(t.get("count", 0) for t in tiers)
        total_emissions = sum(
            Decimal(str(s.get("emissions_tco2e", 0))) for s in top_suppliers
        )

        result: Dict[str, Any] = {
            "template": "supplier_engagement_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "program_overview": data.get("program_overview", {}),
            "summary": {
                "total_suppliers": total_suppliers,
                "total_top20_emissions_tco2e": str(total_emissions),
                "tier_count": len(tiers),
            },
            "supplier_tiers": tiers,
            "engagement_levels": data.get("engagement_levels", []),
            "top_suppliers": top_suppliers,
            "program_progress": progress,
            "scope3_impact": scope3,
            "milestones": data.get("milestones", []),
            "resource_allocation": data.get("resource_allocation", {}),
            "recommendations": data.get("recommendations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Supplier Engagement Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_program_overview(self, data: Dict[str, Any]) -> str:
        overview = data.get("program_overview", {})
        tiers = data.get("supplier_tiers", [])
        total_suppliers = sum(t.get("count", 0) for t in tiers)
        return (
            "## 1. Program Overview\n\n"
            f"- **Program Name:** {overview.get('name', 'Supplier Engagement Program')}\n"
            f"- **Launch Date:** {overview.get('launch_date', 'N/A')}\n"
            f"- **Total Suppliers in Scope:** {_dec_comma(total_suppliers, 0)}\n"
            f"- **Scope 3 Coverage Target:** {_dec(overview.get('scope3_coverage_target_pct', 0))}%\n"
            f"- **SBTi Supplier Engagement Target:** {overview.get('sbti_target', 'N/A')}\n"
            f"- **Program Manager:** {overview.get('program_manager', 'N/A')}\n"
            f"- **Reporting Cadence:** {overview.get('reporting_cadence', 'Quarterly')}"
        )

    def _md_supplier_tiering(self, data: Dict[str, Any]) -> str:
        tiers = data.get("supplier_tiers", [])
        total_suppliers = sum(t.get("count", 0) for t in tiers)
        total_emissions = sum(Decimal(str(t.get("emissions_tco2e", 0))) for t in tiers)
        lines = [
            "## 2. Supplier Tiering Summary\n",
            "| Tier | Description | Supplier Count | Emissions (tCO2e) | Share (%) | Engagement Level |",
            "|:----:|-------------|:--------------:|------------------:|:---------:|:----------------:|",
        ]
        for t in tiers:
            e = Decimal(str(t.get("emissions_tco2e", 0)))
            lines.append(
                f"| {t.get('tier', '-')} | {t.get('description', '-')} "
                f"| {t.get('count', 0)} "
                f"| {_dec_comma(e)} "
                f"| {_dec(_pct_of(e, total_emissions))}% "
                f"| {t.get('engagement_level', '-')} |"
            )
        if tiers:
            lines.append(
                f"| **Total** | | **{total_suppliers}** "
                f"| **{_dec_comma(total_emissions)}** | **100.00%** | |"
            )
        else:
            lines.append("| - | _No tier data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_engagement_levels(self, data: Dict[str, Any]) -> str:
        levels = data.get("engagement_levels", [])
        total = sum(l.get("supplier_count", 0) for l in levels)
        lines = [
            "## 3. Engagement Level Distribution\n",
            "| Level | Description | Suppliers | Share (%) | Avg Response Rate (%) |",
            "|-------|-------------|:---------:|:---------:|:---------------------:|",
        ]
        for lv in levels:
            cnt = lv.get("supplier_count", 0)
            lines.append(
                f"| {lv.get('level', '-')} | {lv.get('description', '-')} "
                f"| {cnt} "
                f"| {_dec(_pct_of(cnt, total))}% "
                f"| {_dec(lv.get('avg_response_rate_pct', 0))}% |"
            )
        if not levels:
            lines.append("| - | _No engagement data_ | - | - | - |")
        return "\n".join(lines)

    def _md_top_suppliers(self, data: Dict[str, Any]) -> str:
        suppliers = data.get("top_suppliers", [])[:20]
        lines = [
            "## 4. Top 20 Suppliers by Emissions\n",
            "| # | Supplier | Category | Emissions (tCO2e) | Share (%) | Tier | Engagement | Has SBTi |",
            "|---|----------|----------|------------------:|:---------:|:----:|:----------:|:--------:|",
        ]
        total_scope3 = Decimal(str(data.get("total_scope3_tco2e", 0)))
        if total_scope3 == 0 and suppliers:
            total_scope3 = sum(Decimal(str(s.get("emissions_tco2e", 0))) for s in suppliers)
        for i, s in enumerate(suppliers, 1):
            e = Decimal(str(s.get("emissions_tco2e", 0)))
            has_sbti = "Yes" if s.get("has_sbti", False) else "No"
            lines.append(
                f"| {i} | {s.get('name', '-')} | {s.get('category', '-')} "
                f"| {_dec_comma(e)} "
                f"| {_dec(_pct_of(e, total_scope3))}% "
                f"| {s.get('tier', '-')} "
                f"| {s.get('engagement', '-')} "
                f"| {has_sbti} |"
            )
        if not suppliers:
            lines.append("| - | _No supplier data_ | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_program_progress(self, data: Dict[str, Any]) -> str:
        progress = data.get("program_progress", [])
        lines = [
            "## 5. Program Progress (RAG Dashboard)\n",
            "| KPI | Target | Actual | Status | Trend |",
            "|-----|--------|--------|:------:|:-----:|",
        ]
        for kpi in progress:
            status = kpi.get("status", "AMBER").upper()
            lines.append(
                f"| {kpi.get('kpi', '-')} | {kpi.get('target', '-')} "
                f"| {kpi.get('actual', '-')} "
                f"| {status} "
                f"| {kpi.get('trend', '-')} |"
            )
        if not progress:
            lines.append("| _No progress data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_scope3_impact(self, data: Dict[str, Any]) -> str:
        impact = data.get("scope3_impact", {})
        categories = impact.get("categories", [])
        lines = [
            "## 6. Scope 3 Impact Estimation\n",
            f"**Total Scope 3 Emissions:** {_dec_comma(impact.get('total_scope3_tco2e', 0))} tCO2e  \n"
            f"**Supplier-Addressable:** {_dec_comma(impact.get('addressable_tco2e', 0))} tCO2e "
            f"({_dec(impact.get('addressable_pct', 0))}%)  \n"
            f"**Estimated Reduction Potential:** {_dec_comma(impact.get('reduction_potential_tco2e', 0))} tCO2e\n",
            "| Scope 3 Category | Emissions (tCO2e) | Addressable (%) | Reduction Target (tCO2e) |",
            "|-----------------|------------------:|:---------------:|-------------------------:|",
        ]
        for cat in categories:
            lines.append(
                f"| {cat.get('category', '-')} "
                f"| {_dec_comma(cat.get('emissions_tco2e', 0))} "
                f"| {_dec(cat.get('addressable_pct', 0))}% "
                f"| {_dec_comma(cat.get('reduction_target_tco2e', 0))} |"
            )
        if not categories:
            lines.append("| _No category data_ | - | - | - |")
        return "\n".join(lines)

    def _md_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        lines = [
            "## 7. Engagement Milestones Timeline\n",
            "| Date | Milestone | Owner | Status | Deliverable |",
            "|------|-----------|-------|:------:|-------------|",
        ]
        for ms in milestones:
            lines.append(
                f"| {ms.get('date', '-')} | {ms.get('milestone', '-')} "
                f"| {ms.get('owner', '-')} "
                f"| {ms.get('status', '-')} "
                f"| {ms.get('deliverable', '-')} |"
            )
        if not milestones:
            lines.append("| - | _No milestones defined_ | - | - | - |")
        return "\n".join(lines)

    def _md_resource_allocation(self, data: Dict[str, Any]) -> str:
        resources = data.get("resource_allocation", {})
        team = resources.get("team", [])
        budget = resources.get("budget", {})
        lines = [
            "## 8. Resource Allocation\n",
            f"**Annual Budget:** EUR {_dec_comma(budget.get('total_eur', 0), 0)}  \n"
            f"**FTE Allocated:** {_dec(resources.get('total_fte', 0), 1)}\n",
            "### Team Allocation\n",
            "| Role | FTE | Focus Area |",
            "|------|:---:|------------|",
        ]
        for member in team:
            lines.append(
                f"| {member.get('role', '-')} "
                f"| {_dec(member.get('fte', 0), 1)} "
                f"| {member.get('focus', '-')} |"
            )
        if not team:
            lines.append("| _No team data_ | - | - |")

        budget_items = budget.get("items", [])
        if budget_items:
            lines.append("")
            lines.append("### Budget Breakdown\n")
            lines.append("| Category | Amount (EUR) | Share (%) |")
            lines.append("|----------|-------------:|:---------:|")
            total_budget = Decimal(str(budget.get("total_eur", 0)))
            for item in budget_items:
                amt = Decimal(str(item.get("amount_eur", 0)))
                lines.append(
                    f"| {item.get('category', '-')} "
                    f"| {_dec_comma(amt, 0)} "
                    f"| {_dec(_pct_of(amt, total_budget))}% |"
                )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        lines = ["## 9. Recommendations\n"]
        if recs:
            for i, rec in enumerate(recs, 1):
                priority = rec.get("priority", "MEDIUM")
                lines.append(f"### {i}. [{priority}] {rec.get('title', 'Recommendation')}\n")
                lines.append(f"{rec.get('description', '')}\n")
                actions = rec.get("actions", [])
                for action in actions:
                    lines.append(f"  - {action}")
                lines.append("")
        else:
            lines.append("_No recommendations at this time._")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-022 Net Zero Acceleration Pack on {ts}*  \n"
            f"*Supplier engagement methodology aligned with SBTi and CDP Supply Chain.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;"
            "font-size:1.8em;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;"
            "padding-left:12px;font-size:1.3em;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".rag-green{color:#1b5e20;font-weight:700;}"
            ".rag-amber{color:#e65100;font-weight:700;}"
            ".rag-red{color:#c62828;font-weight:700;}"
            ".progress-bar{background:#e0e0e0;border-radius:6px;height:18px;overflow:hidden;}"
            ".progress-fill{height:100%;border-radius:6px;}"
            ".fill-green{background:#43a047;}"
            ".fill-amber{background:#ff8f00;}"
            ".fill-red{background:#e53935;}"
            ".tier-1{background:#c8e6c9;}"
            ".tier-2{background:#dcedc8;}"
            ".tier-3{background:#f9fbe7;}"
            ".tier-4{background:#fff9c4;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Supplier Engagement Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_program_overview(self, data: Dict[str, Any]) -> str:
        overview = data.get("program_overview", {})
        tiers = data.get("supplier_tiers", [])
        total_suppliers = sum(t.get("count", 0) for t in tiers)
        return (
            f'<h2>1. Program Overview</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Suppliers</div>'
            f'<div class="card-value">{_dec_comma(total_suppliers, 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Coverage Target</div>'
            f'<div class="card-value">{_dec(overview.get("scope3_coverage_target_pct", 0))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Launch Date</div>'
            f'<div class="card-value">{overview.get("launch_date", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">SBTi Target</div>'
            f'<div class="card-value">{overview.get("sbti_target", "N/A")}</div></div>\n'
            f'</div>'
        )

    def _html_supplier_tiering(self, data: Dict[str, Any]) -> str:
        tiers = data.get("supplier_tiers", [])
        total_emissions = sum(Decimal(str(t.get("emissions_tco2e", 0))) for t in tiers)
        rows = ""
        for idx, t in enumerate(tiers):
            e = Decimal(str(t.get("emissions_tco2e", 0)))
            tier_cls = f" tier-{idx + 1}" if idx < 4 else ""
            rows += (
                f'<tr class="{tier_cls}"><td>{t.get("tier", "-")}</td>'
                f'<td>{t.get("description", "-")}</td>'
                f'<td>{t.get("count", 0)}</td>'
                f'<td>{_dec_comma(e)}</td>'
                f'<td>{_dec(_pct_of(e, total_emissions))}%</td>'
                f'<td>{t.get("engagement_level", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. Supplier Tiering Summary</h2>\n'
            f'<table>\n'
            f'<tr><th>Tier</th><th>Description</th><th>Count</th>'
            f'<th>Emissions (tCO2e)</th><th>Share</th><th>Engagement</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_engagement_levels(self, data: Dict[str, Any]) -> str:
        levels = data.get("engagement_levels", [])
        total = sum(l.get("supplier_count", 0) for l in levels)
        rows = ""
        for lv in levels:
            cnt = lv.get("supplier_count", 0)
            pct = float(_pct_of(cnt, total))
            rows += (
                f'<tr><td>{lv.get("level", "-")}</td>'
                f'<td>{lv.get("description", "-")}</td>'
                f'<td>{cnt}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill fill-green" '
                f'style="width:{min(pct, 100)}%"></div></div> {_dec(pct)}%</td>'
                f'<td>{_dec(lv.get("avg_response_rate_pct", 0))}%</td></tr>\n'
            )
        return (
            f'<h2>3. Engagement Level Distribution</h2>\n'
            f'<table>\n'
            f'<tr><th>Level</th><th>Description</th><th>Suppliers</th>'
            f'<th>Share</th><th>Avg Response Rate</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_top_suppliers(self, data: Dict[str, Any]) -> str:
        suppliers = data.get("top_suppliers", [])[:20]
        total_scope3 = Decimal(str(data.get("total_scope3_tco2e", 0)))
        if total_scope3 == 0 and suppliers:
            total_scope3 = sum(Decimal(str(s.get("emissions_tco2e", 0))) for s in suppliers)
        rows = ""
        for i, s in enumerate(suppliers, 1):
            e = Decimal(str(s.get("emissions_tco2e", 0)))
            has_sbti = "&#10004;" if s.get("has_sbti", False) else "&#10008;"
            sbti_cls = "rag-green" if s.get("has_sbti", False) else "rag-red"
            rows += (
                f'<tr><td>{i}</td><td><strong>{s.get("name", "-")}</strong></td>'
                f'<td>{s.get("category", "-")}</td>'
                f'<td>{_dec_comma(e)}</td>'
                f'<td>{_dec(_pct_of(e, total_scope3))}%</td>'
                f'<td>{s.get("tier", "-")}</td>'
                f'<td>{s.get("engagement", "-")}</td>'
                f'<td class="{sbti_cls}">{has_sbti}</td></tr>\n'
            )
        return (
            f'<h2>4. Top 20 Suppliers by Emissions</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Supplier</th><th>Category</th>'
            f'<th>Emissions (tCO2e)</th><th>Share</th><th>Tier</th>'
            f'<th>Engagement</th><th>SBTi</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_program_progress(self, data: Dict[str, Any]) -> str:
        progress = data.get("program_progress", [])
        rows = ""
        for kpi in progress:
            status = kpi.get("status", "AMBER").upper()
            rag_cls = (
                "rag-green" if status == "GREEN"
                else "rag-red" if status == "RED"
                else "rag-amber"
            )
            rows += (
                f'<tr><td>{kpi.get("kpi", "-")}</td>'
                f'<td>{kpi.get("target", "-")}</td>'
                f'<td>{kpi.get("actual", "-")}</td>'
                f'<td class="{rag_cls}">{status}</td>'
                f'<td>{kpi.get("trend", "-")}</td></tr>\n'
            )
        return (
            f'<h2>5. Program Progress (RAG Dashboard)</h2>\n'
            f'<table>\n'
            f'<tr><th>KPI</th><th>Target</th><th>Actual</th>'
            f'<th>Status</th><th>Trend</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_scope3_impact(self, data: Dict[str, Any]) -> str:
        impact = data.get("scope3_impact", {})
        categories = impact.get("categories", [])
        rows = ""
        for cat in categories:
            rows += (
                f'<tr><td>{cat.get("category", "-")}</td>'
                f'<td>{_dec_comma(cat.get("emissions_tco2e", 0))}</td>'
                f'<td>{_dec(cat.get("addressable_pct", 0))}%</td>'
                f'<td>{_dec_comma(cat.get("reduction_target_tco2e", 0))}</td></tr>\n'
            )
        return (
            f'<h2>6. Scope 3 Impact Estimation</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Scope 3</div>'
            f'<div class="card-value">{_dec_comma(impact.get("total_scope3_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Addressable</div>'
            f'<div class="card-value">{_dec_comma(impact.get("addressable_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Reduction Potential</div>'
            f'<div class="card-value">{_dec_comma(impact.get("reduction_potential_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Category</th><th>Emissions (tCO2e)</th>'
            f'<th>Addressable (%)</th><th>Reduction Target</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        rows = ""
        for ms in milestones:
            status = ms.get("status", "Pending")
            rag_cls = (
                "rag-green" if status.lower() in ("completed", "done", "green")
                else "rag-red" if status.lower() in ("overdue", "red", "blocked")
                else "rag-amber"
            )
            rows += (
                f'<tr><td>{ms.get("date", "-")}</td>'
                f'<td>{ms.get("milestone", "-")}</td>'
                f'<td>{ms.get("owner", "-")}</td>'
                f'<td class="{rag_cls}">{status}</td>'
                f'<td>{ms.get("deliverable", "-")}</td></tr>\n'
            )
        return (
            f'<h2>7. Engagement Milestones Timeline</h2>\n'
            f'<table>\n'
            f'<tr><th>Date</th><th>Milestone</th><th>Owner</th>'
            f'<th>Status</th><th>Deliverable</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_resource_allocation(self, data: Dict[str, Any]) -> str:
        resources = data.get("resource_allocation", {})
        budget = resources.get("budget", {})
        team = resources.get("team", [])
        team_rows = ""
        for member in team:
            team_rows += (
                f'<tr><td>{member.get("role", "-")}</td>'
                f'<td>{_dec(member.get("fte", 0), 1)}</td>'
                f'<td>{member.get("focus", "-")}</td></tr>\n'
            )
        budget_items = budget.get("items", [])
        total_budget = Decimal(str(budget.get("total_eur", 0)))
        budget_rows = ""
        for item in budget_items:
            amt = Decimal(str(item.get("amount_eur", 0)))
            budget_rows += (
                f'<tr><td>{item.get("category", "-")}</td>'
                f'<td>EUR {_dec_comma(amt, 0)}</td>'
                f'<td>{_dec(_pct_of(amt, total_budget))}%</td></tr>\n'
            )
        return (
            f'<h2>8. Resource Allocation</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Annual Budget</div>'
            f'<div class="card-value">EUR {_dec_comma(budget.get("total_eur", 0), 0)}</div></div>\n'
            f'  <div class="card"><div class="card-label">FTE Allocated</div>'
            f'<div class="card-value">{_dec(resources.get("total_fte", 0), 1)}</div></div>\n'
            f'</div>\n'
            f'<h3>Team</h3>\n'
            f'<table>\n'
            f'<tr><th>Role</th><th>FTE</th><th>Focus Area</th></tr>\n'
            f'{team_rows}</table>\n'
            f'<h3>Budget Breakdown</h3>\n'
            f'<table>\n'
            f'<tr><th>Category</th><th>Amount</th><th>Share</th></tr>\n'
            f'{budget_rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        items = ""
        for i, rec in enumerate(recs, 1):
            priority = rec.get("priority", "MEDIUM")
            pri_cls = "rag-red" if priority == "HIGH" else "rag-green" if priority == "LOW" else "rag-amber"
            actions_html = ""
            if rec.get("actions"):
                actions_html = "<ul>" + "".join(f"<li>{a}</li>" for a in rec["actions"]) + "</ul>"
            items += (
                f'<div style="margin:12px 0;padding:12px;border:1px solid #c8e6c9;border-radius:8px;">'
                f'<strong>{i}. <span class="{pri_cls}">[{priority}]</span> '
                f'{rec.get("title", "")}</strong>'
                f'<p>{rec.get("description", "")}</p>'
                f'{actions_html}</div>\n'
            )
        return f'<h2>9. Recommendations</h2>\n{items}'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-022 Net Zero '
            f'Acceleration Pack on {ts}<br>'
            f'Supplier engagement aligned with SBTi and CDP Supply Chain.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
