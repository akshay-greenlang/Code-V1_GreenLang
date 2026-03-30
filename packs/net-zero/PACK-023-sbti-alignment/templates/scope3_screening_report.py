# -*- coding: utf-8 -*-
"""
Scope3ScreeningReportTemplate - 15-category Scope 3 analysis for PACK-023.

Renders a comprehensive Scope 3 screening report covering all 15 GHG Protocol
categories with materiality assessment, coverage tracking against SBTi
67%/90% thresholds, data quality scoring, and target recommendations.

Sections:
    1. Scope 3 Overview (total, % of emissions)
    2. Category Breakdown (15 rows with emissions, %, materiality)
    3. Materiality Heatmap (high/medium/low/negligible)
    4. Coverage Assessment (67%/90% tracking)
    5. Data Quality by Category
    6. Target Recommendations

Author: GreenLang Team
Version: 23.0.0
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

_MODULE_VERSION = "23.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec(val: Any, places: int = 2) -> str:
    """Format a value as a Decimal string with fixed decimal places."""
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)

def _dec_comma(val: Any, places: int = 2) -> str:
    """Format a Decimal value with thousands separator."""
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
    """Format a value as percentage string."""
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)

def _materiality_label(level: str) -> str:
    """Normalize materiality level to display label."""
    s = str(level).upper()
    mapping = {
        "HIGH": "High",
        "MEDIUM": "Medium",
        "LOW": "Low",
        "NEGLIGIBLE": "Negligible",
    }
    return mapping.get(s, level)

class Scope3ScreeningReportTemplate:
    """
    Scope 3 screening report template for SBTi alignment.

    Renders a 15-category Scope 3 materiality analysis with coverage
    tracking against SBTi near-term (67%) and long-term (90%) thresholds,
    data quality assessment, and category-level target recommendations.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Scope3ScreeningReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render Scope 3 screening report as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_scope3_overview(data),
            self._md_category_breakdown(data),
            self._md_materiality_heatmap(data),
            self._md_coverage_assessment(data),
            self._md_data_quality(data),
            self._md_target_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render Scope 3 screening report as self-contained HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_scope3_overview(data),
            self._html_category_breakdown(data),
            self._html_materiality_heatmap(data),
            self._html_coverage_assessment(data),
            self._html_data_quality(data),
            self._html_target_recommendations(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Scope 3 Screening Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render Scope 3 screening report as structured JSON."""
        self.generated_at = utcnow()
        categories = data.get("categories", [])
        overview = data.get("overview", {})
        coverage = data.get("coverage", {})

        result: Dict[str, Any] = {
            "template": "scope3_screening_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "overview": overview,
            "categories": categories,
            "coverage": coverage,
            "data_quality": data.get("data_quality", []),
            "target_recommendations": data.get("target_recommendations", []),
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
            f"# Scope 3 Screening Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_scope3_overview(self, data: Dict[str, Any]) -> str:
        ov = data.get("overview", {})
        return (
            f"## 1. Scope 3 Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total Scope 3 Emissions | {_dec_comma(ov.get('total_scope3_tco2e', 0), 0)} tCO2e |\n"
            f"| Total Scope 1+2 Emissions | {_dec_comma(ov.get('total_s1s2_tco2e', 0), 0)} tCO2e |\n"
            f"| Scope 3 as % of Total | {_pct(ov.get('scope3_pct_of_total', 0))} |\n"
            f"| 40% Materiality Trigger | {ov.get('triggers_40pct', 'N/A')} |\n"
            f"| Categories Assessed | {ov.get('categories_assessed', 0)} / 15 |\n"
            f"| Material Categories | {ov.get('material_categories', 0)} |\n"
            f"| Screening Method | {ov.get('screening_method', 'N/A')} |"
        )

    def _md_category_breakdown(self, data: Dict[str, Any]) -> str:
        categories = data.get("categories", [])
        lines = [
            "## 2. Category Breakdown\n",
            "| Cat # | Category | Emissions (tCO2e) | % of Total S3 "
            "| Materiality | Data Method | Included |",
            "|:-----:|----------|------------------:|:-----------:"
            "|:-----------:|:-----------:|:--------:|",
        ]
        for cat in categories:
            lines.append(
                f"| {cat.get('number', '-')} "
                f"| {cat.get('name', '-')} "
                f"| {_dec_comma(cat.get('emissions_tco2e', 0), 0)} "
                f"| {_pct(cat.get('pct_of_total', 0))} "
                f"| {_materiality_label(cat.get('materiality', 'N/A'))} "
                f"| {cat.get('data_method', '-')} "
                f"| {cat.get('included', '-')} |"
            )
        if not categories:
            lines.append(
                "| - | _No categories assessed_ | - | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_materiality_heatmap(self, data: Dict[str, Any]) -> str:
        categories = data.get("categories", [])
        high = [c for c in categories if str(c.get("materiality", "")).upper() == "HIGH"]
        medium = [c for c in categories if str(c.get("materiality", "")).upper() == "MEDIUM"]
        low = [c for c in categories if str(c.get("materiality", "")).upper() == "LOW"]
        negligible = [c for c in categories if str(c.get("materiality", "")).upper() == "NEGLIGIBLE"]

        lines = [
            "## 3. Materiality Heatmap\n",
            f"| Level | Count | Categories |",
            f"|-------|:-----:|------------|",
        ]
        for level, items in [
            ("High", high), ("Medium", medium),
            ("Low", low), ("Negligible", negligible),
        ]:
            cat_names = ", ".join(
                f"Cat {c.get('number', '?')}" for c in items
            ) if items else "None"
            lines.append(f"| {level} | {len(items)} | {cat_names} |")

        lines.append("")
        lines.append(
            f"**Material categories (High + Medium):** "
            f"{len(high) + len(medium)} of {len(categories)}"
        )
        return "\n".join(lines)

    def _md_coverage_assessment(self, data: Dict[str, Any]) -> str:
        cov = data.get("coverage", {})
        lines = [
            "## 4. Coverage Assessment\n",
            f"| Metric | Value | Threshold | Status |\n"
            f"|--------|-------|:---------:|--------|\n"
            f"| Near-Term Coverage | {_pct(cov.get('near_term_pct', 0))} "
            f"| 67% | {cov.get('near_term_status', 'N/A')} |\n"
            f"| Long-Term Coverage | {_pct(cov.get('long_term_pct', 0))} "
            f"| 90% | {cov.get('long_term_status', 'N/A')} |\n"
            f"| Categories with Targets | {cov.get('categories_with_targets', 0)} "
            f"| - | - |\n"
            f"| Supplier Engagement Coverage | {_pct(cov.get('supplier_engagement_pct', 0))} "
            f"| - | - |",
        ]
        gaps = cov.get("coverage_gaps", [])
        if gaps:
            lines.append("")
            lines.append("### Coverage Gaps\n")
            lines.append("| Category | Emissions (tCO2e) | Gap Impact (%) | Action Needed |")
            lines.append("|----------|------------------:|:--------------:|---------------|")
            for g in gaps:
                lines.append(
                    f"| {g.get('category', '-')} "
                    f"| {_dec_comma(g.get('emissions_tco2e', 0), 0)} "
                    f"| {_pct(g.get('gap_impact_pct', 0))} "
                    f"| {g.get('action', '-')} |"
                )
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality", [])
        lines = [
            "## 5. Data Quality by Category\n",
            "| Cat # | Category | Primary (%) | Secondary (%) "
            "| Proxy (%) | Spend (%) | Overall Score |",
            "|:-----:|----------|:-----------:|:-------------:"
            "|:---------:|:---------:|:-------------:|",
        ]
        for d in dq:
            lines.append(
                f"| {d.get('number', '-')} "
                f"| {d.get('name', '-')} "
                f"| {_pct(d.get('primary_pct', 0))} "
                f"| {_pct(d.get('secondary_pct', 0))} "
                f"| {_pct(d.get('proxy_pct', 0))} "
                f"| {_pct(d.get('spend_pct', 0))} "
                f"| {d.get('overall_score', '-')} |"
            )
        if not dq:
            lines.append(
                "| - | _No data quality assessment_ | - | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_target_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("target_recommendations", [])
        lines = [
            "## 6. Target Recommendations\n",
            "| # | Category | Recommendation | Target Type "
            "| Ambition | Priority |",
            "|---|----------|----------------|:-----------:"
            "|:--------:|:--------:|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('category', '-')} "
                f"| {r.get('recommendation', '-')} "
                f"| {r.get('target_type', '-')} "
                f"| {r.get('ambition', '-')} "
                f"| {r.get('priority', '-')} |"
            )
        if not recs:
            lines.append(
                "| - | _No recommendations_ | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-023 SBTi Alignment Pack on {ts}*  \n"
            f"*Scope 3 screening per GHG Protocol Corporate Value Chain Standard and "
            f"SBTi Corporate Manual V5.3.*"
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
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".mat-high{background:#ffcdd2;color:#c62828;font-weight:700;}"
            ".mat-medium{background:#fff3e0;color:#e65100;font-weight:600;}"
            ".mat-low{background:#e8f5e9;color:#2e7d32;}"
            ".mat-negligible{background:#f5f5f5;color:#757575;}"
            ".progress-bar{width:100%;height:20px;background:#e0e0e0;border-radius:10px;"
            "overflow:hidden;margin:4px 0;}"
            ".progress-fill{height:100%;border-radius:10px;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_mat_class(self, level: str) -> str:
        """Return CSS class for materiality level."""
        s = str(level).upper()
        return {
            "HIGH": "mat-high",
            "MEDIUM": "mat-medium",
            "LOW": "mat-low",
            "NEGLIGIBLE": "mat-negligible",
        }.get(s, "")

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Scope 3 Screening Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_scope3_overview(self, data: Dict[str, Any]) -> str:
        ov = data.get("overview", {})
        trigger = ov.get("triggers_40pct", "N/A")
        return (
            f'<h2>1. Scope 3 Overview</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Scope 3</div>'
            f'<div class="card-value">{_dec_comma(ov.get("total_scope3_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">% of Total</div>'
            f'<div class="card-value">{_pct(ov.get("scope3_pct_of_total", 0))}</div></div>\n'
            f'  <div class="card"><div class="card-label">40% Trigger</div>'
            f'<div class="card-value">{trigger}</div></div>\n'
            f'  <div class="card"><div class="card-label">Categories Assessed</div>'
            f'<div class="card-value">{ov.get("categories_assessed", 0)}/15</div></div>\n'
            f'  <div class="card"><div class="card-label">Material Categories</div>'
            f'<div class="card-value">{ov.get("material_categories", 0)}</div></div>\n'
            f'</div>'
        )

    def _html_category_breakdown(self, data: Dict[str, Any]) -> str:
        categories = data.get("categories", [])
        rows = ""
        for cat in categories:
            mat_cls = self._html_mat_class(cat.get("materiality", ""))
            rows += (
                f'<tr><td>{cat.get("number", "-")}</td>'
                f'<td>{cat.get("name", "-")}</td>'
                f'<td>{_dec_comma(cat.get("emissions_tco2e", 0), 0)}</td>'
                f'<td>{_pct(cat.get("pct_of_total", 0))}</td>'
                f'<td class="{mat_cls}">{_materiality_label(cat.get("materiality", "N/A"))}</td>'
                f'<td>{cat.get("data_method", "-")}</td>'
                f'<td>{cat.get("included", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. Category Breakdown</h2>\n'
            f'<table>\n'
            f'<tr><th>Cat #</th><th>Category</th><th>Emissions (tCO2e)</th>'
            f'<th>% of S3</th><th>Materiality</th><th>Data Method</th>'
            f'<th>Included</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_materiality_heatmap(self, data: Dict[str, Any]) -> str:
        categories = data.get("categories", [])
        levels = ["HIGH", "MEDIUM", "LOW", "NEGLIGIBLE"]
        rows = ""
        for level in levels:
            items = [c for c in categories if str(c.get("materiality", "")).upper() == level]
            cat_names = ", ".join(
                f"Cat {c.get('number', '?')}" for c in items
            ) if items else "None"
            mat_cls = self._html_mat_class(level)
            rows += (
                f'<tr><td class="{mat_cls}">{_materiality_label(level)}</td>'
                f'<td>{len(items)}</td>'
                f'<td>{cat_names}</td></tr>\n'
            )
        return (
            f'<h2>3. Materiality Heatmap</h2>\n'
            f'<table>\n'
            f'<tr><th>Level</th><th>Count</th><th>Categories</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_coverage_assessment(self, data: Dict[str, Any]) -> str:
        cov = data.get("coverage", {})
        nt = float(cov.get("near_term_pct", 0))
        lt = float(cov.get("long_term_pct", 0))
        nt_color = "#43a047" if nt >= 67 else "#ff9800" if nt >= 50 else "#ef5350"
        lt_color = "#43a047" if lt >= 90 else "#ff9800" if lt >= 67 else "#ef5350"

        gaps = cov.get("coverage_gaps", [])
        gap_rows = ""
        for g in gaps:
            gap_rows += (
                f'<tr><td>{g.get("category", "-")}</td>'
                f'<td>{_dec_comma(g.get("emissions_tco2e", 0), 0)}</td>'
                f'<td>{_pct(g.get("gap_impact_pct", 0))}</td>'
                f'<td>{g.get("action", "-")}</td></tr>\n'
            )
        gap_html = ""
        if gaps:
            gap_html = (
                f'<h3>Coverage Gaps</h3>\n'
                f'<table><tr><th>Category</th><th>Emissions (tCO2e)</th>'
                f'<th>Gap Impact</th><th>Action</th></tr>\n'
                f'{gap_rows}</table>\n'
            )

        return (
            f'<h2>4. Coverage Assessment</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Near-Term (req: 67%)</div>'
            f'<div class="card-value">{_pct(nt)}</div>'
            f'<div class="progress-bar"><div class="progress-fill" '
            f'style="width:{min(nt, 100)}%;background:{nt_color};"></div></div></div>\n'
            f'  <div class="card"><div class="card-label">Long-Term (req: 90%)</div>'
            f'<div class="card-value">{_pct(lt)}</div>'
            f'<div class="progress-bar"><div class="progress-fill" '
            f'style="width:{min(lt, 100)}%;background:{lt_color};"></div></div></div>\n'
            f'</div>\n'
            f'{gap_html}'
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality", [])
        rows = ""
        for d in dq:
            rows += (
                f'<tr><td>{d.get("number", "-")}</td>'
                f'<td>{d.get("name", "-")}</td>'
                f'<td>{_pct(d.get("primary_pct", 0))}</td>'
                f'<td>{_pct(d.get("secondary_pct", 0))}</td>'
                f'<td>{_pct(d.get("proxy_pct", 0))}</td>'
                f'<td>{_pct(d.get("spend_pct", 0))}</td>'
                f'<td>{d.get("overall_score", "-")}</td></tr>\n'
            )
        return (
            f'<h2>5. Data Quality by Category</h2>\n'
            f'<table>\n'
            f'<tr><th>Cat #</th><th>Category</th><th>Primary</th>'
            f'<th>Secondary</th><th>Proxy</th><th>Spend</th>'
            f'<th>Overall</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_target_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("target_recommendations", [])
        rows = ""
        for i, r in enumerate(recs, 1):
            rows += (
                f'<tr><td>{i}</td><td>{r.get("category", "-")}</td>'
                f'<td>{r.get("recommendation", "-")}</td>'
                f'<td>{r.get("target_type", "-")}</td>'
                f'<td>{r.get("ambition", "-")}</td>'
                f'<td>{r.get("priority", "-")}</td></tr>\n'
            )
        return (
            f'<h2>6. Target Recommendations</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Category</th><th>Recommendation</th>'
            f'<th>Target Type</th><th>Ambition</th><th>Priority</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-023 SBTi '
            f'Alignment Pack on {ts}<br>'
            f'Scope 3 screening per GHG Protocol Corporate Value Chain '
            f'Standard and SBTi Corporate Manual V5.3.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
