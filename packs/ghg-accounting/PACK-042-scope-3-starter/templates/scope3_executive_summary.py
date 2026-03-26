# -*- coding: utf-8 -*-
"""
Scope3ExecutiveSummaryTemplate - C-Suite Executive Summary for PACK-042.

Generates a 2-4 page executive summary designed for board and C-suite
consumption. Includes total Scope 3 headline number, top 5 categories
waterfall chart data, percentage of total footprint (Scope 1+2+3),
year-over-year trend, SBTi alignment status, 3 priority actions, and
data quality summary.

Sections:
    1. Headline Metrics
    2. Top 5 Categories (waterfall chart data)
    3. Full Footprint Context (Scope 1+2+3)
    4. Year-over-Year Trend
    5. SBTi Alignment Status
    6. Priority Actions
    7. Data Quality Summary

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, corporate navy theme)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 42.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "42.0.0"


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    """Format numeric value with thousands separators."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K"
    return f"{value:,.{decimals}f}"


def _fmt_tco2e(value: Optional[float]) -> str:
    """Format tCO2e with scale suffix."""
    if value is None:
        return "N/A"
    return f"{_fmt_num(value)} tCO2e"


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage with sign."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _pct_of(part: float, total: float) -> str:
    """Percentage of total, formatted."""
    if total == 0:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


class Scope3ExecutiveSummaryTemplate:
    """
    C-suite executive summary template for Scope 3 emissions.

    Renders a concise 2-4 page executive summary designed for board
    and senior leadership consumption. Focuses on headline numbers,
    top emission categories, full footprint context, trends, SBTi
    alignment, and priority actions. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = Scope3ExecutiveSummaryTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize Scope3ExecutiveSummaryTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def _scope3_total(self, data: Dict[str, Any]) -> float:
        """Calculate total Scope 3 emissions."""
        return data.get("scope3_total_tco2e", 0.0)

    def _full_footprint(self, data: Dict[str, Any]) -> float:
        """Calculate full footprint (Scope 1+2+3)."""
        s1 = data.get("scope1_total_tco2e", 0.0)
        s2 = data.get("scope2_total_tco2e", 0.0)
        s3 = self._scope3_total(data)
        return s1 + s2 + s3

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render executive summary as Markdown.

        Args:
            data: Validated executive summary data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_headline_metrics(data),
            self._md_top_categories(data),
            self._md_footprint_context(data),
            self._md_yoy_trend(data),
            self._md_sbti_alignment(data),
            self._md_priority_actions(data),
            self._md_data_quality(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render executive summary as HTML.

        Args:
            data: Validated executive summary data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_headline_metrics(data),
            self._html_top_categories(data),
            self._html_footprint_context(data),
            self._html_yoy_trend(data),
            self._html_sbti_alignment(data),
            self._html_priority_actions(data),
            self._html_data_quality(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render executive summary as JSON-serializable dict.

        Args:
            data: Validated executive summary data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        s3 = self._scope3_total(data)
        full = self._full_footprint(data)
        top_cats = sorted(
            data.get("top_categories", []),
            key=lambda c: c.get("emissions_tco2e", 0),
            reverse=True,
        )[:5]
        return {
            "template": "scope3_executive_summary",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "headline_metrics": {
                "scope3_total_tco2e": s3,
                "full_footprint_tco2e": full,
                "scope3_pct_of_total": (s3 / full * 100) if full > 0 else 0.0,
                "scope1_tco2e": data.get("scope1_total_tco2e", 0.0),
                "scope2_tco2e": data.get("scope2_total_tco2e", 0.0),
                "yoy_change_pct": data.get("yoy_change_pct"),
            },
            "top_5_categories": [
                {
                    "category_number": c.get("category_number"),
                    "category_name": c.get("category_name"),
                    "emissions_tco2e": c.get("emissions_tco2e"),
                    "pct_of_scope3": (c.get("emissions_tco2e", 0) / s3 * 100)
                    if s3 > 0 else 0.0,
                }
                for c in top_cats
            ],
            "waterfall_chart_data": self._json_waterfall(data, top_cats, s3),
            "sbti_alignment": data.get("sbti_alignment", {}),
            "priority_actions": data.get("priority_actions", []),
            "data_quality_summary": data.get("data_quality_summary", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Scope 3 Executive Summary - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_headline_metrics(self, data: Dict[str, Any]) -> str:
        """Render Markdown headline metrics."""
        s3 = self._scope3_total(data)
        full = self._full_footprint(data)
        s3_pct = (s3 / full * 100) if full > 0 else 0.0
        yoy = data.get("yoy_change_pct")
        categories_reported = data.get("categories_reported", 0)
        lines = [
            "## 1. Headline Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total Scope 3** | **{_fmt_tco2e(s3)}** |",
            f"| Full Footprint (S1+S2+S3) | {_fmt_tco2e(full)} |",
            f"| Scope 3 as % of Total | {s3_pct:.1f}% |",
            f"| Categories Reported | {categories_reported} of 15 |",
        ]
        if yoy is not None:
            lines.append(f"| Year-over-Year Change | {_fmt_pct(yoy)} |")
        return "\n".join(lines)

    def _md_top_categories(self, data: Dict[str, Any]) -> str:
        """Render Markdown top 5 categories."""
        s3 = self._scope3_total(data)
        top_cats = sorted(
            data.get("top_categories", []),
            key=lambda c: c.get("emissions_tco2e", 0),
            reverse=True,
        )[:5]
        if not top_cats:
            return "## 2. Top 5 Emission Categories\n\nNo category data available."
        lines = [
            "## 2. Top 5 Emission Categories",
            "",
            "| Rank | Category | tCO2e | % of Scope 3 |",
            "|------|----------|-------|-------------|",
        ]
        cumulative = 0.0
        for i, cat in enumerate(top_cats, 1):
            num = cat.get("category_number", "?")
            name = cat.get("category_name", "Unknown")
            em = cat.get("emissions_tco2e", 0.0)
            cumulative += em
            pct = _pct_of(em, s3)
            lines.append(f"| {i} | Cat {num} - {name} | {_fmt_tco2e(em)} | {pct} |")
        cum_pct = _pct_of(cumulative, s3)
        lines.append(f"\n*Top 5 categories account for {cum_pct} of total Scope 3.*")
        return "\n".join(lines)

    def _md_footprint_context(self, data: Dict[str, Any]) -> str:
        """Render Markdown full footprint context."""
        s1 = data.get("scope1_total_tco2e", 0.0)
        s2 = data.get("scope2_total_tco2e", 0.0)
        s3 = self._scope3_total(data)
        full = s1 + s2 + s3
        lines = [
            "## 3. Full Footprint Context",
            "",
            "| Scope | tCO2e | % of Total |",
            "|-------|-------|-----------|",
            f"| Scope 1 | {_fmt_tco2e(s1)} | {_pct_of(s1, full)} |",
            f"| Scope 2 | {_fmt_tco2e(s2)} | {_pct_of(s2, full)} |",
            f"| Scope 3 | {_fmt_tco2e(s3)} | {_pct_of(s3, full)} |",
            f"| **Total** | **{_fmt_tco2e(full)}** | **100.0%** |",
        ]
        return "\n".join(lines)

    def _md_yoy_trend(self, data: Dict[str, Any]) -> str:
        """Render Markdown year-over-year trend."""
        trend = data.get("yoy_trend", {})
        if not trend:
            yoy = data.get("yoy_change_pct")
            if yoy is None:
                return "## 4. Year-over-Year Trend\n\nNo prior year data available."
            return (
                "## 4. Year-over-Year Trend\n\n"
                f"**Year-over-Year Change:** {_fmt_pct(yoy)}"
            )
        lines = [
            "## 4. Year-over-Year Trend",
            "",
            "| Year | Scope 3 tCO2e | Change |",
            "|------|-------------|--------|",
        ]
        years = trend.get("years", [])
        for yr in years:
            year = yr.get("year", "")
            em = _fmt_tco2e(yr.get("scope3_tco2e"))
            change = yr.get("change_pct")
            change_str = _fmt_pct(change) if change is not None else "-"
            lines.append(f"| {year} | {em} | {change_str} |")
        driver_text = trend.get("key_drivers", "")
        if driver_text:
            lines.append(f"\n**Key Drivers:** {driver_text}")
        return "\n".join(lines)

    def _md_sbti_alignment(self, data: Dict[str, Any]) -> str:
        """Render Markdown SBTi alignment status."""
        sbti = data.get("sbti_alignment", {})
        if not sbti:
            return "## 5. SBTi Alignment Status\n\nNo SBTi alignment data available."
        status = sbti.get("status", "Not assessed")
        target_year = sbti.get("target_year", "-")
        target_reduction = sbti.get("target_reduction_pct")
        current_progress = sbti.get("current_progress_pct")
        pathway = sbti.get("pathway", "-")
        lines = [
            "## 5. SBTi Alignment Status",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Status | {status} |",
            f"| Pathway | {pathway} |",
            f"| Target Year | {target_year} |",
        ]
        if target_reduction is not None:
            lines.append(f"| Target Reduction | {target_reduction:.1f}% |")
        if current_progress is not None:
            lines.append(f"| Current Progress | {current_progress:.1f}% |")
        note = sbti.get("note", "")
        if note:
            lines.append(f"\n{note}")
        return "\n".join(lines)

    def _md_priority_actions(self, data: Dict[str, Any]) -> str:
        """Render Markdown priority actions."""
        actions = data.get("priority_actions", [])
        if not actions:
            return "## 6. Priority Actions\n\nNo priority actions defined."
        lines = [
            "## 6. Priority Actions",
            "",
        ]
        for i, action in enumerate(actions[:3], 1):
            title = action.get("title", "")
            description = action.get("description", "")
            impact = action.get("expected_impact", "")
            owner = action.get("owner", "-")
            deadline = action.get("deadline", "-")
            lines.append(f"### Action {i}: {title}")
            lines.append("")
            if description:
                lines.append(description)
            lines.append("")
            lines.append(f"- **Expected Impact:** {impact}")
            lines.append(f"- **Owner:** {owner}")
            lines.append(f"- **Deadline:** {deadline}")
            lines.append("")
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality summary."""
        quality = data.get("data_quality_summary", {})
        if not quality:
            return "## 7. Data Quality Summary\n\nNo data quality assessment available."
        overall = quality.get("overall_dqr_score")
        primary_data_pct = quality.get("primary_data_pct")
        coverage = quality.get("coverage_pct")
        lines = [
            "## 7. Data Quality Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if overall is not None:
            lines.append(f"| Overall DQR Score | {overall:.1f} / 5.0 |")
        if primary_data_pct is not None:
            lines.append(f"| Primary Data Coverage | {primary_data_pct:.0f}% |")
        if coverage is not None:
            lines.append(f"| Category Coverage | {coverage:.0f}% |")
        key_gap = quality.get("key_gap", "")
        if key_gap:
            lines.append(f"\n**Key Gap:** {key_gap}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Scope 3 Executive Summary - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1000px;line-height:1.6;}\n"
            "h1{color:#1B2A4A;border-bottom:3px solid #1B2A4A;padding-bottom:0.5rem;}\n"
            "h2{color:#1B2A4A;margin-top:2rem;border-bottom:1px solid #ccc;padding-bottom:0.3rem;}\n"
            "h3{color:#2C3E6B;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#eef1f6;font-weight:600;color:#1B2A4A;}\n"
            "tr:nth-child(even){background:#f8f9fc;}\n"
            ".total-row{font-weight:bold;background:#dce3ed;}\n"
            ".metric-card{display:inline-block;background:#eef1f6;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:180px;"
            "border-top:3px solid #1B2A4A;}\n"
            ".metric-value{font-size:1.8rem;font-weight:700;color:#1B2A4A;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".action-card{background:#f8f9fc;border-left:4px solid #1B2A4A;"
            "padding:1rem;margin:0.5rem 0;border-radius:0 4px 4px 0;}\n"
            ".action-title{font-weight:700;color:#1B2A4A;font-size:1.05rem;}\n"
            ".status-on-track{color:#27ae60;font-weight:700;}\n"
            ".status-at-risk{color:#e67e22;font-weight:700;}\n"
            ".status-off-track{color:#e74c3c;font-weight:700;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            '<div class="section">\n'
            f"<h1>Scope 3 Executive Summary &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_headline_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML headline metrics with cards."""
        s3 = self._scope3_total(data)
        full = self._full_footprint(data)
        s3_pct = (s3 / full * 100) if full > 0 else 0.0
        yoy = data.get("yoy_change_pct")
        cards = [
            ("Total Scope 3", _fmt_tco2e(s3)),
            ("Full Footprint", _fmt_tco2e(full)),
            ("Scope 3 Share", f"{s3_pct:.1f}%"),
        ]
        if yoy is not None:
            cards.append(("YoY Change", _fmt_pct(yoy)))
        card_html = ""
        for label, val in cards:
            card_html += (
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>1. Headline Metrics</h2>\n"
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_top_categories(self, data: Dict[str, Any]) -> str:
        """Render HTML top 5 categories."""
        s3 = self._scope3_total(data)
        top_cats = sorted(
            data.get("top_categories", []),
            key=lambda c: c.get("emissions_tco2e", 0),
            reverse=True,
        )[:5]
        if not top_cats:
            return ""
        rows = ""
        cumulative = 0.0
        for i, cat in enumerate(top_cats, 1):
            num = cat.get("category_number", "?")
            name = cat.get("category_name", "Unknown")
            em = cat.get("emissions_tco2e", 0.0)
            cumulative += em
            pct = _pct_of(em, s3)
            rows += f"<tr><td>{i}</td><td>Cat {num} - {name}</td><td>{_fmt_tco2e(em)}</td><td>{pct}</td></tr>\n"
        cum_pct = _pct_of(cumulative, s3)
        return (
            '<div class="section">\n'
            "<h2>2. Top 5 Emission Categories</h2>\n"
            "<table><thead><tr><th>Rank</th><th>Category</th><th>tCO2e</th>"
            f"<th>%</th></tr></thead>\n<tbody>{rows}</tbody></table>\n"
            f"<p><em>Top 5 categories account for {cum_pct} of total Scope 3.</em></p>\n</div>"
        )

    def _html_footprint_context(self, data: Dict[str, Any]) -> str:
        """Render HTML full footprint context."""
        s1 = data.get("scope1_total_tco2e", 0.0)
        s2 = data.get("scope2_total_tco2e", 0.0)
        s3 = self._scope3_total(data)
        full = s1 + s2 + s3
        return (
            '<div class="section">\n'
            "<h2>3. Full Footprint Context</h2>\n"
            "<table><thead><tr><th>Scope</th><th>tCO2e</th><th>%</th></tr></thead>\n<tbody>"
            f"<tr><td>Scope 1</td><td>{_fmt_tco2e(s1)}</td><td>{_pct_of(s1, full)}</td></tr>\n"
            f"<tr><td>Scope 2</td><td>{_fmt_tco2e(s2)}</td><td>{_pct_of(s2, full)}</td></tr>\n"
            f"<tr><td>Scope 3</td><td>{_fmt_tco2e(s3)}</td><td>{_pct_of(s3, full)}</td></tr>\n"
            f'<tr class="total-row"><td>Total</td><td>{_fmt_tco2e(full)}</td><td>100.0%</td></tr>\n'
            "</tbody></table>\n</div>"
        )

    def _html_yoy_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML year-over-year trend."""
        trend = data.get("yoy_trend", {})
        years = trend.get("years", [])
        if not years:
            return ""
        rows = ""
        for yr in years:
            year = yr.get("year", "")
            em = _fmt_tco2e(yr.get("scope3_tco2e"))
            change = yr.get("change_pct")
            change_str = _fmt_pct(change) if change is not None else "-"
            rows += f"<tr><td>{year}</td><td>{em}</td><td>{change_str}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>4. Year-over-Year Trend</h2>\n"
            "<table><thead><tr><th>Year</th><th>Scope 3 tCO2e</th>"
            f"<th>Change</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sbti_alignment(self, data: Dict[str, Any]) -> str:
        """Render HTML SBTi alignment status."""
        sbti = data.get("sbti_alignment", {})
        if not sbti:
            return ""
        status = sbti.get("status", "Not assessed")
        target_year = sbti.get("target_year", "-")
        pathway = sbti.get("pathway", "-")
        target_red = sbti.get("target_reduction_pct")
        progress = sbti.get("current_progress_pct")
        status_css = "status-on-track" if "track" in status.lower() or status == "Committed" else "status-at-risk"
        rows = (
            f'<tr><td>Status</td><td class="{status_css}">{status}</td></tr>\n'
            f"<tr><td>Pathway</td><td>{pathway}</td></tr>\n"
            f"<tr><td>Target Year</td><td>{target_year}</td></tr>\n"
        )
        if target_red is not None:
            rows += f"<tr><td>Target Reduction</td><td>{target_red:.1f}%</td></tr>\n"
        if progress is not None:
            rows += f"<tr><td>Current Progress</td><td>{progress:.1f}%</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>5. SBTi Alignment Status</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_priority_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML priority actions."""
        actions = data.get("priority_actions", [])
        if not actions:
            return ""
        action_html = ""
        for i, action in enumerate(actions[:3], 1):
            title = action.get("title", "")
            description = action.get("description", "")
            impact = action.get("expected_impact", "")
            owner = action.get("owner", "-")
            deadline = action.get("deadline", "-")
            action_html += (
                f'<div class="action-card">\n'
                f'<div class="action-title">Action {i}: {title}</div>\n'
                f"<p>{description}</p>\n"
                f"<p><strong>Impact:</strong> {impact} | "
                f"<strong>Owner:</strong> {owner} | "
                f"<strong>Deadline:</strong> {deadline}</p>\n</div>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>6. Priority Actions</h2>\n"
            f"{action_html}</div>"
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality summary."""
        quality = data.get("data_quality_summary", {})
        if not quality:
            return ""
        overall = quality.get("overall_dqr_score")
        primary = quality.get("primary_data_pct")
        coverage = quality.get("coverage_pct")
        rows = ""
        if overall is not None:
            rows += f"<tr><td>Overall DQR Score</td><td>{overall:.1f} / 5.0</td></tr>\n"
        if primary is not None:
            rows += f"<tr><td>Primary Data Coverage</td><td>{primary:.0f}%</td></tr>\n"
        if coverage is not None:
            rows += f"<tr><td>Category Coverage</td><td>{coverage:.0f}%</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>7. Data Quality Summary</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-042 Scope 3 Starter v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON HELPERS
    # ==================================================================

    def _json_waterfall(
        self,
        data: Dict[str, Any],
        top_cats: List[Dict[str, Any]],
        s3_total: float,
    ) -> List[Dict[str, Any]]:
        """Build waterfall chart data for top categories."""
        waterfall = []
        cumulative = 0.0
        for cat in top_cats:
            em = cat.get("emissions_tco2e", 0.0)
            waterfall.append({
                "label": f"Cat {cat.get('category_number', '?')}",
                "value": em,
                "cumulative": cumulative + em,
                "pct_of_total": (em / s3_total * 100) if s3_total > 0 else 0.0,
            })
            cumulative += em
        remaining = s3_total - cumulative
        if remaining > 0:
            waterfall.append({
                "label": "Other Categories",
                "value": remaining,
                "cumulative": s3_total,
                "pct_of_total": (remaining / s3_total * 100) if s3_total > 0 else 0.0,
            })
        return waterfall
