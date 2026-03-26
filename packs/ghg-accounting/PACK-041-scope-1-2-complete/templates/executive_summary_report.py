# -*- coding: utf-8 -*-
"""
ExecutiveSummaryReportTemplate - 2-4 Page Executive Summary for PACK-041.

Generates a concise executive summary report covering headline GHG metrics
(Scope 1 total, Scope 2 location and market), key changes from previous year,
top 5 emission sources, Scope 2 renewable energy procurement impact, SBTi
progress, compliance status across multiple frameworks, recommended actions,
and infographic-ready data points.

Sections:
    1. Headline Metrics
    2. Key Changes from Previous Year
    3. Top 5 Emission Sources
    4. Scope 2 RE Procurement Impact
    5. SBTi Progress
    6. Compliance Status
    7. Recommended Actions
    8. Infographic Data Points

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 41.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "41.0.0"


def _fmt_tco2e(value: Optional[float]) -> str:
    """Format tCO2e with scale suffix."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M tCO2e"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}K tCO2e"
    return f"{value:,.1f} tCO2e"


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage with sign."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


class ExecutiveSummaryReportTemplate:
    """
    Executive summary report template (2-4 pages).

    Renders concise executive summaries with headline metrics, key changes,
    top emission sources, RE procurement impact, SBTi progress, compliance
    status, and recommended actions. All outputs include SHA-256 provenance
    hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = ExecutiveSummaryReportTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ExecutiveSummaryReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render executive summary as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_headline_metrics(data),
            self._md_key_changes(data),
            self._md_top_sources(data),
            self._md_re_procurement(data),
            self._md_sbti_progress(data),
            self._md_compliance_status(data),
            self._md_recommended_actions(data),
            self._md_infographic_data(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render executive summary as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_headline_metrics(data),
            self._html_key_changes(data),
            self._html_top_sources(data),
            self._html_re_procurement(data),
            self._html_sbti_progress(data),
            self._html_compliance_status(data),
            self._html_recommended_actions(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render executive summary as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "executive_summary_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "headline_metrics": data.get("headline_metrics", {}),
            "key_changes": data.get("key_changes", []),
            "top_sources": data.get("top_sources", []),
            "re_procurement": data.get("re_procurement", {}),
            "sbti_progress": data.get("sbti_progress", {}),
            "compliance_status": data.get("compliance_status", []),
            "recommended_actions": data.get("recommended_actions", []),
            "infographic_data": data.get("infographic_data", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Executive Summary: GHG Emissions - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_headline_metrics(self, data: Dict[str, Any]) -> str:
        """Render Markdown headline metrics."""
        metrics = data.get("headline_metrics", {})
        s1 = metrics.get("scope1_total_tco2e", 0.0)
        s2_loc = metrics.get("scope2_location_tco2e", 0.0)
        s2_mkt = metrics.get("scope2_market_tco2e", 0.0)
        combined_loc = metrics.get("combined_location_tco2e", s1 + s2_loc)
        combined_mkt = metrics.get("combined_market_tco2e", s1 + s2_mkt)
        yoy = metrics.get("yoy_change_pct")
        lines = [
            "## 1. Headline Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total Scope 1** | **{_fmt_tco2e(s1)}** |",
            f"| Scope 2 (Location-Based) | {_fmt_tco2e(s2_loc)} |",
            f"| Scope 2 (Market-Based) | {_fmt_tco2e(s2_mkt)} |",
            f"| **Combined (Location)** | **{_fmt_tco2e(combined_loc)}** |",
            f"| Combined (Market) | {_fmt_tco2e(combined_mkt)} |",
        ]
        if yoy is not None:
            lines.append(f"| Year-over-Year Change | {_fmt_pct(yoy)} |")
        return "\n".join(lines)

    def _md_key_changes(self, data: Dict[str, Any]) -> str:
        """Render Markdown key changes from previous year."""
        changes = data.get("key_changes", [])
        if not changes:
            return "## 2. Key Changes from Previous Year\n\nNo significant changes identified."
        lines = [
            "## 2. Key Changes from Previous Year",
            "",
        ]
        for chg in changes:
            direction = chg.get("direction", "")
            description = chg.get("description", "")
            impact = _fmt_tco2e(chg.get("impact_tco2e"))
            lines.append(f"- **{direction}** {description} ({impact})")
        return "\n".join(lines)

    def _md_top_sources(self, data: Dict[str, Any]) -> str:
        """Render Markdown top 5 emission sources."""
        sources = data.get("top_sources", [])[:5]
        if not sources:
            return "## 3. Top 5 Emission Sources\n\nNo source ranking available."
        lines = [
            "## 3. Top 5 Emission Sources",
            "",
            "| Rank | Source | Scope | tCO2e | % of Total |",
            "|------|--------|-------|-------|-----------|",
        ]
        for i, src in enumerate(sources, 1):
            name = src.get("source_name", "")
            scope = src.get("scope", "-")
            em = _fmt_tco2e(src.get("emissions_tco2e"))
            pct = f"{src.get('pct_of_total', 0):.1f}%"
            lines.append(f"| {i} | {name} | {scope} | {em} | {pct} |")
        return "\n".join(lines)

    def _md_re_procurement(self, data: Dict[str, Any]) -> str:
        """Render Markdown RE procurement impact."""
        re_data = data.get("re_procurement", {})
        if not re_data:
            return "## 4. Scope 2 RE Procurement Impact\n\nNo RE procurement data."
        lines = [
            "## 4. Scope 2 RE Procurement Impact",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| RE Percentage | {re_data.get('re_percentage', 0):.1f}% |",
            f"| RE MWh Procured | {re_data.get('re_mwh', 0):,.0f} MWh |",
            f"| Avoided Emissions | {_fmt_tco2e(re_data.get('avoided_emissions_tco2e'))} |",
            f"| Market vs Location Reduction | {_fmt_tco2e(re_data.get('market_vs_location_reduction_tco2e'))} |",
        ]
        return "\n".join(lines)

    def _md_sbti_progress(self, data: Dict[str, Any]) -> str:
        """Render Markdown SBTi progress."""
        sbti = data.get("sbti_progress", {})
        if not sbti:
            return "## 5. SBTi Progress\n\nNo SBTi targets set."
        targets = sbti.get("targets", [])
        lines = [
            "## 5. SBTi Progress",
            "",
            "| Target | Scope | Base Year | Target Year | Required % | Actual % | On Track |",
            "|--------|-------|----------|------------|-----------|---------|---------|",
        ]
        for tgt in targets:
            name = tgt.get("target_name", "")
            scope = tgt.get("scope", "1+2")
            base = tgt.get("base_year", "-")
            target = tgt.get("target_year", "-")
            required = f"{tgt.get('required_reduction_pct', 0):.1f}%"
            actual = f"{tgt.get('actual_reduction_pct', 0):.1f}%"
            on_track = "Yes" if tgt.get("on_track") else "No"
            lines.append(f"| {name} | {scope} | {base} | {target} | {required} | {actual} | **{on_track}** |")
        return "\n".join(lines)

    def _md_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render Markdown compliance status across frameworks."""
        status = data.get("compliance_status", [])
        if not status:
            return "## 6. Compliance Status\n\nNo compliance assessment performed."
        lines = [
            "## 6. Compliance Status",
            "",
            "| Framework | Status | Score | Deadline | Notes |",
            "|-----------|--------|-------|---------|-------|",
        ]
        for fw in status:
            name = fw.get("framework", "")
            st = fw.get("status", "-")
            score = f"{fw.get('score', 0):.0f}%" if fw.get("score") is not None else "-"
            deadline = fw.get("deadline", "-")
            notes = fw.get("notes", "-")
            lines.append(f"| {name} | **{st}** | {score} | {deadline} | {notes} |")
        return "\n".join(lines)

    def _md_recommended_actions(self, data: Dict[str, Any]) -> str:
        """Render Markdown recommended actions."""
        actions = data.get("recommended_actions", [])
        if not actions:
            return "## 7. Recommended Actions\n\nNo actions recommended."
        lines = [
            "## 7. Recommended Actions",
            "",
            "| Priority | Action | Expected Impact | Timeline | Owner |",
            "|----------|--------|----------------|---------|-------|",
        ]
        for act in actions:
            priority = act.get("priority", "-")
            desc = act.get("description", "")
            impact = act.get("expected_impact", "-")
            timeline = act.get("timeline", "-")
            owner = act.get("owner", "-")
            lines.append(f"| {priority} | {desc} | {impact} | {timeline} | {owner} |")
        return "\n".join(lines)

    def _md_infographic_data(self, data: Dict[str, Any]) -> str:
        """Render Markdown infographic-ready data points."""
        infographic = data.get("infographic_data", {})
        if not infographic:
            return ""
        points = infographic.get("data_points", [])
        if not points:
            return ""
        lines = [
            "## 8. Infographic Data Points",
            "",
        ]
        for pt in points:
            label = pt.get("label", "")
            value = pt.get("value", "")
            unit = pt.get("unit", "")
            lines.append(f"- **{label}:** {value} {unit}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-041 Scope 1-2 Complete v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Executive Summary - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1000px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".metric-card{display:inline-block;background:linear-gradient(135deg,#f0f4f8,#e8eef4);"
            "border-radius:12px;padding:1.2rem 1.8rem;margin:0.5rem;text-align:center;"
            "min-width:200px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}\n"
            ".metric-value{font-size:1.8rem;font-weight:700;color:#1b263b;}\n"
            ".metric-label{font-size:0.85rem;color:#555;margin-top:0.3rem;}\n"
            ".on-track{color:#2a9d8f;font-weight:700;}\n"
            ".off-track{color:#e63946;font-weight:700;}\n"
            ".compliant{color:#2a9d8f;}\n"
            ".non-compliant{color:#e63946;}\n"
            ".partial{color:#f4a261;}\n"
            ".action-high{border-left:4px solid #e63946;}\n"
            ".action-medium{border-left:4px solid #f4a261;}\n"
            ".action-low{border-left:4px solid #2a9d8f;}\n"
            ".section{margin-bottom:2rem;}\n"
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
            f"<h1>Executive Summary &mdash; {company} ({year})</h1>\n"
            "<hr>\n</div>"
        )

    def _html_headline_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML headline metrics with cards."""
        metrics = data.get("headline_metrics", {})
        s1 = metrics.get("scope1_total_tco2e", 0.0)
        s2_loc = metrics.get("scope2_location_tco2e", 0.0)
        s2_mkt = metrics.get("scope2_market_tco2e", 0.0)
        combined = metrics.get("combined_location_tco2e", s1 + s2_loc)
        yoy = metrics.get("yoy_change_pct")
        cards = [
            ("Scope 1", _fmt_tco2e(s1)),
            ("Scope 2 (Location)", _fmt_tco2e(s2_loc)),
            ("Scope 2 (Market)", _fmt_tco2e(s2_mkt)),
            ("Combined Total", _fmt_tco2e(combined)),
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
        return f'<div class="section">\n<h2>1. Headline Metrics</h2>\n<div>{card_html}</div>\n</div>'

    def _html_key_changes(self, data: Dict[str, Any]) -> str:
        """Render HTML key changes."""
        changes = data.get("key_changes", [])
        if not changes:
            return ""
        items = ""
        for chg in changes:
            direction = chg.get("direction", "")
            desc = chg.get("description", "")
            impact = _fmt_tco2e(chg.get("impact_tco2e"))
            items += f"<li><strong>{direction}</strong> {desc} ({impact})</li>\n"
        return f'<div class="section">\n<h2>2. Key Changes</h2>\n<ul>{items}</ul>\n</div>'

    def _html_top_sources(self, data: Dict[str, Any]) -> str:
        """Render HTML top 5 sources."""
        sources = data.get("top_sources", [])[:5]
        if not sources:
            return ""
        rows = ""
        for i, src in enumerate(sources, 1):
            name = src.get("source_name", "")
            scope = src.get("scope", "-")
            em = _fmt_tco2e(src.get("emissions_tco2e"))
            pct = f"{src.get('pct_of_total', 0):.1f}%"
            bar_width = int(src.get("pct_of_total", 0) * 3)
            rows += (
                f"<tr><td>{i}</td><td>{name}</td><td>{scope}</td><td>{em}</td>"
                f'<td>{pct} <span style="display:inline-block;height:12px;width:{bar_width}px;'
                f'background:#457b9d;border-radius:2px"></span></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>3. Top 5 Sources</h2>\n'
            "<table><thead><tr><th>#</th><th>Source</th><th>Scope</th>"
            "<th>tCO2e</th><th>% of Total</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_re_procurement(self, data: Dict[str, Any]) -> str:
        """Render HTML RE procurement impact."""
        re_data = data.get("re_procurement", {})
        if not re_data:
            return ""
        return (
            '<div class="section">\n<h2>4. RE Procurement Impact</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n<tbody>"
            f"<tr><td>RE Percentage</td><td>{re_data.get('re_percentage', 0):.1f}%</td></tr>\n"
            f"<tr><td>RE MWh</td><td>{re_data.get('re_mwh', 0):,.0f}</td></tr>\n"
            f"<tr><td>Avoided Emissions</td><td>{_fmt_tco2e(re_data.get('avoided_emissions_tco2e'))}</td></tr>\n"
            "</tbody></table>\n</div>"
        )

    def _html_sbti_progress(self, data: Dict[str, Any]) -> str:
        """Render HTML SBTi progress."""
        sbti = data.get("sbti_progress", {})
        if not sbti:
            return ""
        targets = sbti.get("targets", [])
        rows = ""
        for tgt in targets:
            on_track = tgt.get("on_track")
            css = "on-track" if on_track else "off-track"
            label = "On Track" if on_track else "Off Track"
            rows += (
                f"<tr><td>{tgt.get('target_name', '')}</td>"
                f"<td>{tgt.get('scope', '1+2')}</td>"
                f"<td>{tgt.get('required_reduction_pct', 0):.1f}%</td>"
                f"<td>{tgt.get('actual_reduction_pct', 0):.1f}%</td>"
                f'<td class="{css}">{label}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>5. SBTi Progress</h2>\n'
            "<table><thead><tr><th>Target</th><th>Scope</th><th>Required</th>"
            "<th>Actual</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance status."""
        status = data.get("compliance_status", [])
        if not status:
            return ""
        rows = ""
        for fw in status:
            st = fw.get("status", "-")
            css = "compliant" if st in ("Compliant", "Ready") else ("non-compliant" if st in ("Non-Compliant", "Not Ready") else "partial")
            score = f"{fw.get('score', 0):.0f}%" if fw.get("score") is not None else "-"
            rows += (
                f"<tr><td>{fw.get('framework', '')}</td>"
                f'<td class="{css}"><strong>{st}</strong></td>'
                f"<td>{score}</td><td>{fw.get('deadline', '-')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>6. Compliance Status</h2>\n'
            "<table><thead><tr><th>Framework</th><th>Status</th>"
            "<th>Score</th><th>Deadline</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_recommended_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML recommended actions."""
        actions = data.get("recommended_actions", [])
        if not actions:
            return ""
        rows = ""
        for act in actions:
            priority = act.get("priority", "Medium")
            css = f"action-{priority.lower()}" if priority.lower() in ("high", "medium", "low") else ""
            rows += (
                f'<tr class="{css}"><td>{priority}</td><td>{act.get("description", "")}</td>'
                f'<td>{act.get("expected_impact", "-")}</td>'
                f'<td>{act.get("timeline", "-")}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>7. Recommended Actions</h2>\n'
            "<table><thead><tr><th>Priority</th><th>Action</th>"
            "<th>Impact</th><th>Timeline</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-041 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
