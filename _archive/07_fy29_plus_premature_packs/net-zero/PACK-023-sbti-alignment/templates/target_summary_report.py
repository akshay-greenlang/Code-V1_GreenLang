# -*- coding: utf-8 -*-
"""
TargetSummaryReportTemplate - SBTi target definitions report for PACK-023.

Renders a comprehensive target summary covering near-term, long-term, and
net-zero target definitions, pathway visualization with ACA/SDA milestones,
scope coverage analysis, temperature-aligned ambition assessment, base year
profile, and key assumptions.

Sections:
    1. Target Overview (near-term, long-term, net-zero)
    2. Pathway Visualization (ACA/SDA milestones table)
    3. Scope Coverage (S1+S2, S3 by category)
    4. Ambition Assessment (temperature alignment)
    5. Base Year Profile
    6. Key Assumptions

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

class TargetSummaryReportTemplate:
    """
    SBTi target summary report template.

    Renders near-term, long-term, and net-zero target definitions with
    pathway milestones, scope coverage analysis, temperature-aligned
    ambition assessment, and base year profile.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TargetSummaryReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render target summary report as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_target_overview(data),
            self._md_pathway_visualization(data),
            self._md_scope_coverage(data),
            self._md_ambition_assessment(data),
            self._md_base_year_profile(data),
            self._md_key_assumptions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render target summary report as self-contained HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_target_overview(data),
            self._html_pathway_visualization(data),
            self._html_scope_coverage(data),
            self._html_ambition_assessment(data),
            self._html_base_year_profile(data),
            self._html_key_assumptions(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>SBTi Target Summary Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render target summary report as structured JSON."""
        self.generated_at = utcnow()
        targets = data.get("targets", [])
        milestones = data.get("milestones", [])
        scope_coverage = data.get("scope_coverage", {})
        ambition = data.get("ambition_assessment", {})
        base_year = data.get("base_year", {})

        result: Dict[str, Any] = {
            "template": "target_summary_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "target_overview": {
                "total_targets": len(targets),
                "near_term": [t for t in targets if t.get("type") == "NEAR_TERM"],
                "long_term": [t for t in targets if t.get("type") == "LONG_TERM"],
                "net_zero": [t for t in targets if t.get("type") == "NET_ZERO"],
            },
            "milestones": milestones,
            "scope_coverage": scope_coverage,
            "ambition_assessment": ambition,
            "base_year": base_year,
            "assumptions": data.get("assumptions", []),
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
            f"# SBTi Target Summary Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_target_overview(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        near = [t for t in targets if t.get("type") == "NEAR_TERM"]
        long = [t for t in targets if t.get("type") == "LONG_TERM"]
        nz = [t for t in targets if t.get("type") == "NET_ZERO"]
        lines = [
            "## 1. Target Overview\n",
            f"**Total Targets Defined:** {len(targets)}  \n"
            f"**Near-Term:** {len(near)} | **Long-Term:** {len(long)} | "
            f"**Net-Zero:** {len(nz)}\n",
            "| # | Target Name | Type | Scope | Base Year | Target Year "
            "| Reduction (%) | Pathway | Status |",
            "|---|-------------|------|-------|:---------:|:-----------:"
            "|:-------------:|---------|--------|",
        ]
        for i, t in enumerate(targets, 1):
            lines.append(
                f"| {i} | {t.get('name', '-')} | {t.get('type', '-')} "
                f"| {t.get('scope', '-')} | {t.get('base_year', '-')} "
                f"| {t.get('target_year', '-')} "
                f"| {_pct(t.get('reduction_pct', 0))} "
                f"| {t.get('pathway', '-')} | {t.get('status', '-')} |"
            )
        if not targets:
            lines.append("| - | _No targets defined_ | - | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_pathway_visualization(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        lines = [
            "## 2. Pathway Visualization\n",
            "Year-by-year ACA/SDA milestones against target trajectories.\n",
            "| Year | Pathway | Target Intensity | Target Absolute (tCO2e) "
            "| Cumulative Reduction (%) | Status |",
            "|:----:|---------|:----------------:|:----------------------:"
            "|:-----------------------:|--------|",
        ]
        for ms in milestones:
            lines.append(
                f"| {ms.get('year', '-')} | {ms.get('pathway', '-')} "
                f"| {_dec(ms.get('target_intensity', 0), 4)} "
                f"| {_dec_comma(ms.get('target_absolute_tco2e', 0), 0)} "
                f"| {_pct(ms.get('cumulative_reduction_pct', 0))} "
                f"| {ms.get('status', '-')} |"
            )
        if not milestones:
            lines.append("| - | _No milestones defined_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_scope_coverage(self, data: Dict[str, Any]) -> str:
        sc = data.get("scope_coverage", {})
        s1s2 = sc.get("s1_s2", {})
        s3_cats = sc.get("s3_categories", [])
        lines = [
            "## 3. Scope Coverage\n",
            "### Scope 1 + Scope 2\n",
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total S1+S2 (tCO2e) | {_dec_comma(s1s2.get('total_tco2e', 0), 0)} |\n"
            f"| Coverage (%) | {_pct(s1s2.get('coverage_pct', 0))} |\n"
            f"| Minimum Required (%) | {_pct(s1s2.get('min_required_pct', 95))} |\n"
            f"| Status | {s1s2.get('status', 'N/A')} |\n",
            "### Scope 3 by Category\n",
            "| Cat # | Category | Emissions (tCO2e) | % of Total S3 "
            "| Included in Target | Data Quality |",
            "|:-----:|----------|------------------:|:-----------:"
            "|:------------------:|:------------:|",
        ]
        for cat in s3_cats:
            lines.append(
                f"| {cat.get('category_number', '-')} "
                f"| {cat.get('name', '-')} "
                f"| {_dec_comma(cat.get('emissions_tco2e', 0), 0)} "
                f"| {_pct(cat.get('pct_of_total', 0))} "
                f"| {cat.get('included', '-')} "
                f"| {cat.get('data_quality', '-')} |"
            )
        if not s3_cats:
            lines.append("| - | _No categories assessed_ | - | - | - | - |")

        s3_summary = sc.get("s3_summary", {})
        if s3_summary:
            lines.append("")
            lines.append(
                f"**Total Scope 3:** {_dec_comma(s3_summary.get('total_tco2e', 0), 0)} tCO2e  \n"
                f"**Coverage (near-term):** {_pct(s3_summary.get('near_term_coverage_pct', 0))} "
                f"(required: 67%)  \n"
                f"**Coverage (long-term):** {_pct(s3_summary.get('long_term_coverage_pct', 0))} "
                f"(required: 90%)"
            )
        return "\n".join(lines)

    def _md_ambition_assessment(self, data: Dict[str, Any]) -> str:
        ambition = data.get("ambition_assessment", {})
        lines = [
            "## 4. Ambition Assessment\n",
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Temperature Alignment | {ambition.get('temperature_alignment', 'N/A')} |\n"
            f"| Ambition Level | {ambition.get('ambition_level', 'N/A')} |\n"
            f"| Annual Reduction Rate (S1+S2) | {_pct(ambition.get('arr_s1s2', 0))} |\n"
            f"| Annual Reduction Rate (S3) | {_pct(ambition.get('arr_s3', 0))} |\n"
            f"| 1.5C Threshold (4.2%/yr) | {ambition.get('meets_1_5c', 'N/A')} |\n"
            f"| WB2C Threshold (2.5%/yr) | {ambition.get('meets_wb2c', 'N/A')} |",
        ]
        benchmarks = ambition.get("benchmarks", [])
        if benchmarks:
            lines.append("")
            lines.append("### Benchmark Comparison\n")
            lines.append("| Benchmark | Required ARR | Company ARR | Status |")
            lines.append("|-----------|:------------:|:-----------:|--------|")
            for b in benchmarks:
                lines.append(
                    f"| {b.get('name', '-')} "
                    f"| {_pct(b.get('required_arr', 0))} "
                    f"| {_pct(b.get('company_arr', 0))} "
                    f"| {b.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_base_year_profile(self, data: Dict[str, Any]) -> str:
        by = data.get("base_year", {})
        lines = [
            "## 5. Base Year Profile\n",
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Base Year | {by.get('year', 'N/A')} |\n"
            f"| Scope 1 (tCO2e) | {_dec_comma(by.get('scope1_tco2e', 0), 0)} |\n"
            f"| Scope 2 Location (tCO2e) | {_dec_comma(by.get('scope2_location_tco2e', 0), 0)} |\n"
            f"| Scope 2 Market (tCO2e) | {_dec_comma(by.get('scope2_market_tco2e', 0), 0)} |\n"
            f"| Scope 3 (tCO2e) | {_dec_comma(by.get('scope3_tco2e', 0), 0)} |\n"
            f"| Total (tCO2e) | {_dec_comma(by.get('total_tco2e', 0), 0)} |\n"
            f"| Consolidation Approach | {by.get('consolidation_approach', 'N/A')} |\n"
            f"| Methodology | {by.get('methodology', 'N/A')} |\n"
            f"| Verification Status | {by.get('verification_status', 'N/A')} |",
        ]
        recalculations = by.get("recalculations", [])
        if recalculations:
            lines.append("")
            lines.append("### Base Year Recalculations\n")
            lines.append("| Date | Trigger | Impact (tCO2e) | Significance |")
            lines.append("|------|---------|:--------------:|:------------:|")
            for rc in recalculations:
                lines.append(
                    f"| {rc.get('date', '-')} | {rc.get('trigger', '-')} "
                    f"| {_dec_comma(rc.get('impact_tco2e', 0), 0)} "
                    f"| {_pct(rc.get('significance_pct', 0))} |"
                )
        return "\n".join(lines)

    def _md_key_assumptions(self, data: Dict[str, Any]) -> str:
        assumptions = data.get("assumptions", [])
        lines = ["## 6. Key Assumptions\n"]
        if assumptions:
            lines.append("| # | Assumption | Category | Impact if Wrong |")
            lines.append("|---|------------|----------|----------------|")
            for i, a in enumerate(assumptions, 1):
                lines.append(
                    f"| {i} | {a.get('assumption', '-')} "
                    f"| {a.get('category', '-')} "
                    f"| {a.get('impact_if_wrong', '-')} |"
                )
        else:
            lines.append("_No key assumptions documented._")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-023 SBTi Alignment Pack on {ts}*  \n"
            f"*Targets aligned with SBTi Corporate Manual V5.3 and Net-Zero Standard V1.3.*"
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
            ".badge-pass{display:inline-block;background:#43a047;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-fail{display:inline-block;background:#ef5350;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-warn{display:inline-block;background:#ff9800;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>SBTi Target Summary Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_target_overview(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        near = len([t for t in targets if t.get("type") == "NEAR_TERM"])
        long = len([t for t in targets if t.get("type") == "LONG_TERM"])
        nz = len([t for t in targets if t.get("type") == "NET_ZERO"])
        cards = (
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Targets</div>'
            f'<div class="card-value">{len(targets)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Near-Term</div>'
            f'<div class="card-value">{near}</div></div>\n'
            f'  <div class="card"><div class="card-label">Long-Term</div>'
            f'<div class="card-value">{long}</div></div>\n'
            f'  <div class="card"><div class="card-label">Net-Zero</div>'
            f'<div class="card-value">{nz}</div></div>\n'
            f'</div>\n'
        )
        rows = ""
        for i, t in enumerate(targets, 1):
            rows += (
                f'<tr><td>{i}</td><td><strong>{t.get("name", "-")}</strong></td>'
                f'<td>{t.get("type", "-")}</td><td>{t.get("scope", "-")}</td>'
                f'<td>{t.get("base_year", "-")}</td><td>{t.get("target_year", "-")}</td>'
                f'<td>{_pct(t.get("reduction_pct", 0))}</td>'
                f'<td>{t.get("pathway", "-")}</td>'
                f'<td>{t.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>1. Target Overview</h2>\n{cards}'
            f'<table>\n'
            f'<tr><th>#</th><th>Target Name</th><th>Type</th><th>Scope</th>'
            f'<th>Base Year</th><th>Target Year</th><th>Reduction</th>'
            f'<th>Pathway</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_pathway_visualization(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        rows = ""
        for ms in milestones:
            rows += (
                f'<tr><td>{ms.get("year", "-")}</td>'
                f'<td>{ms.get("pathway", "-")}</td>'
                f'<td>{_dec(ms.get("target_intensity", 0), 4)}</td>'
                f'<td>{_dec_comma(ms.get("target_absolute_tco2e", 0), 0)}</td>'
                f'<td>{_pct(ms.get("cumulative_reduction_pct", 0))}</td>'
                f'<td>{ms.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. Pathway Visualization</h2>\n'
            f'<p>Year-by-year ACA/SDA milestones against target trajectories.</p>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Pathway</th><th>Target Intensity</th>'
            f'<th>Target Absolute (tCO2e)</th><th>Cumulative Reduction</th>'
            f'<th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_scope_coverage(self, data: Dict[str, Any]) -> str:
        sc = data.get("scope_coverage", {})
        s1s2 = sc.get("s1_s2", {})
        s3_cats = sc.get("s3_categories", [])
        s3_summary = sc.get("s3_summary", {})

        s1s2_html = (
            f'<h3>Scope 1 + Scope 2</h3>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total S1+S2</div>'
            f'<div class="card-value">{_dec_comma(s1s2.get("total_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Coverage</div>'
            f'<div class="card-value">{_pct(s1s2.get("coverage_pct", 0))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Status</div>'
            f'<div class="card-value">{s1s2.get("status", "N/A")}</div></div>\n'
            f'</div>\n'
        )

        s3_rows = ""
        for cat in s3_cats:
            s3_rows += (
                f'<tr><td>{cat.get("category_number", "-")}</td>'
                f'<td>{cat.get("name", "-")}</td>'
                f'<td>{_dec_comma(cat.get("emissions_tco2e", 0), 0)}</td>'
                f'<td>{_pct(cat.get("pct_of_total", 0))}</td>'
                f'<td>{cat.get("included", "-")}</td>'
                f'<td>{cat.get("data_quality", "-")}</td></tr>\n'
            )

        s3_summary_html = ""
        if s3_summary:
            s3_summary_html = (
                f'<p><strong>Total Scope 3:</strong> '
                f'{_dec_comma(s3_summary.get("total_tco2e", 0), 0)} tCO2e | '
                f'<strong>Near-term coverage:</strong> '
                f'{_pct(s3_summary.get("near_term_coverage_pct", 0))} (req: 67%) | '
                f'<strong>Long-term coverage:</strong> '
                f'{_pct(s3_summary.get("long_term_coverage_pct", 0))} (req: 90%)</p>\n'
            )

        return (
            f'<h2>3. Scope Coverage</h2>\n'
            f'{s1s2_html}'
            f'<h3>Scope 3 by Category</h3>\n'
            f'<table>\n'
            f'<tr><th>Cat #</th><th>Category</th><th>Emissions (tCO2e)</th>'
            f'<th>% of Total S3</th><th>Included</th><th>Data Quality</th></tr>\n'
            f'{s3_rows}</table>\n'
            f'{s3_summary_html}'
        )

    def _html_ambition_assessment(self, data: Dict[str, Any]) -> str:
        ambition = data.get("ambition_assessment", {})
        temp = ambition.get("temperature_alignment", "N/A")
        benchmarks = ambition.get("benchmarks", [])
        bench_rows = ""
        for b in benchmarks:
            bench_rows += (
                f'<tr><td>{b.get("name", "-")}</td>'
                f'<td>{_pct(b.get("required_arr", 0))}</td>'
                f'<td>{_pct(b.get("company_arr", 0))}</td>'
                f'<td>{b.get("status", "-")}</td></tr>\n'
            )
        bench_html = ""
        if benchmarks:
            bench_html = (
                f'<h3>Benchmark Comparison</h3>\n'
                f'<table><tr><th>Benchmark</th><th>Required ARR</th>'
                f'<th>Company ARR</th><th>Status</th></tr>\n'
                f'{bench_rows}</table>\n'
            )
        return (
            f'<h2>4. Ambition Assessment</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Temperature Alignment</div>'
            f'<div class="card-value">{temp}</div></div>\n'
            f'  <div class="card"><div class="card-label">Ambition Level</div>'
            f'<div class="card-value">{ambition.get("ambition_level", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">ARR S1+S2</div>'
            f'<div class="card-value">{_pct(ambition.get("arr_s1s2", 0))}</div></div>\n'
            f'  <div class="card"><div class="card-label">ARR S3</div>'
            f'<div class="card-value">{_pct(ambition.get("arr_s3", 0))}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>Value</th></tr>\n'
            f'<tr><td>1.5C Threshold (4.2%/yr)</td><td>{ambition.get("meets_1_5c", "N/A")}</td></tr>\n'
            f'<tr><td>WB2C Threshold (2.5%/yr)</td><td>{ambition.get("meets_wb2c", "N/A")}</td></tr>\n'
            f'</table>\n'
            f'{bench_html}'
        )

    def _html_base_year_profile(self, data: Dict[str, Any]) -> str:
        by = data.get("base_year", {})
        recalculations = by.get("recalculations", [])
        recalc_rows = ""
        for rc in recalculations:
            recalc_rows += (
                f'<tr><td>{rc.get("date", "-")}</td>'
                f'<td>{rc.get("trigger", "-")}</td>'
                f'<td>{_dec_comma(rc.get("impact_tco2e", 0), 0)}</td>'
                f'<td>{_pct(rc.get("significance_pct", 0))}</td></tr>\n'
            )
        recalc_html = ""
        if recalculations:
            recalc_html = (
                f'<h3>Base Year Recalculations</h3>\n'
                f'<table><tr><th>Date</th><th>Trigger</th><th>Impact (tCO2e)</th>'
                f'<th>Significance</th></tr>\n{recalc_rows}</table>\n'
            )
        return (
            f'<h2>5. Base Year Profile</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Base Year</div>'
            f'<div class="card-value">{by.get("year", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 1</div>'
            f'<div class="card-value">{_dec_comma(by.get("scope1_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 2 (Location)</div>'
            f'<div class="card-value">{_dec_comma(by.get("scope2_location_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope 3</div>'
            f'<div class="card-value">{_dec_comma(by.get("scope3_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Total</div>'
            f'<div class="card-value">{_dec_comma(by.get("total_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>Value</th></tr>\n'
            f'<tr><td>Consolidation Approach</td><td>{by.get("consolidation_approach", "N/A")}</td></tr>\n'
            f'<tr><td>Methodology</td><td>{by.get("methodology", "N/A")}</td></tr>\n'
            f'<tr><td>Verification Status</td><td>{by.get("verification_status", "N/A")}</td></tr>\n'
            f'</table>\n'
            f'{recalc_html}'
        )

    def _html_key_assumptions(self, data: Dict[str, Any]) -> str:
        assumptions = data.get("assumptions", [])
        rows = ""
        for i, a in enumerate(assumptions, 1):
            rows += (
                f'<tr><td>{i}</td><td>{a.get("assumption", "-")}</td>'
                f'<td>{a.get("category", "-")}</td>'
                f'<td>{a.get("impact_if_wrong", "-")}</td></tr>\n'
            )
        return (
            f'<h2>6. Key Assumptions</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Assumption</th><th>Category</th>'
            f'<th>Impact if Wrong</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-023 SBTi '
            f'Alignment Pack on {ts}<br>'
            f'Targets aligned with SBTi Corporate Manual V5.3 and '
            f'Net-Zero Standard V1.3.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
