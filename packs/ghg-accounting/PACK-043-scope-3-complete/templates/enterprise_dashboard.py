# -*- coding: utf-8 -*-
"""
EnterpriseDashboardTemplate - Executive Scope 3 Dashboard for PACK-043.

Generates an executive-level dashboard presenting Scope 3 multi-year trends,
maturity progress gauge, SBTi trajectory vs target, climate risk summary,
supplier programme status, and top 5 category breakdown. Designed for
C-suite and board-level consumption with an executive dark-blue theme.

Sections:
    1. Executive Summary Metrics
    2. Multi-Year Trend (3-5 year timeline)
    3. Top 5 Category Breakdown
    4. Maturity Progress Gauge
    5. SBTi Trajectory vs Target
    6. Supplier Programme Status
    7. Climate Risk Summary
    8. Data Quality Snapshot

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, executive dark blue #0A1628 theme)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 43.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "43.0.0"

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    """Format numeric value with thousands separators and scale suffix."""
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


def _status_label(score: Optional[float]) -> str:
    """Map a 0-100 score to a status label."""
    if score is None:
        return "Not Assessed"
    if score >= 80:
        return "On Track"
    if score >= 50:
        return "In Progress"
    return "At Risk"


def _risk_label(level: Optional[str]) -> str:
    """Normalize risk level to standard labels."""
    if not level:
        return "Not Assessed"
    low = level.lower()
    if low in ("high", "critical"):
        return "High"
    if low in ("medium", "moderate"):
        return "Medium"
    return "Low"


class EnterpriseDashboardTemplate:
    """
    Executive Scope 3 dashboard template.

    Renders a comprehensive executive dashboard with multi-year trends,
    maturity progress, SBTi trajectory, climate risk, and supplier
    programme data. All outputs include SHA-256 provenance hashing.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = EnterpriseDashboardTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnterpriseDashboardTemplate.

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
        """Calculate total Scope 3 emissions from data."""
        return data.get("scope3_total_tco2e", 0.0)

    def _full_footprint(self, data: Dict[str, Any]) -> float:
        """Calculate full footprint (Scope 1+2+3)."""
        s1 = data.get("scope1_total_tco2e", 0.0)
        s2 = data.get("scope2_total_tco2e", 0.0)
        s3 = self._scope3_total(data)
        return s1 + s2 + s3

    def _current_year_emissions(self, data: Dict[str, Any]) -> float:
        """Get current year Scope 3 emissions from trends or total."""
        trends = data.get("multi_year_trends", [])
        if trends:
            return trends[-1].get("scope3_tco2e", self._scope3_total(data))
        return self._scope3_total(data)

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render enterprise dashboard as Markdown.

        Args:
            data: Validated dashboard data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_metrics(data),
            self._md_multi_year_trend(data),
            self._md_top_categories(data),
            self._md_maturity_progress(data),
            self._md_sbti_trajectory(data),
            self._md_supplier_programme(data),
            self._md_climate_risk(data),
            self._md_data_quality(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render enterprise dashboard as HTML.

        Args:
            data: Validated dashboard data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_executive_metrics(data),
            self._html_multi_year_trend(data),
            self._html_top_categories(data),
            self._html_maturity_progress(data),
            self._html_sbti_trajectory(data),
            self._html_supplier_programme(data),
            self._html_climate_risk(data),
            self._html_data_quality(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render enterprise dashboard as JSON-serializable dict.

        Args:
            data: Validated dashboard data dict.
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
            "template": "enterprise_dashboard",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "executive_metrics": {
                "scope3_total_tco2e": s3,
                "full_footprint_tco2e": full,
                "scope3_pct_of_total": (s3 / full * 100) if full > 0 else 0.0,
                "yoy_change_pct": data.get("yoy_change_pct"),
                "categories_reported": data.get("categories_reported", 0),
            },
            "multi_year_trends": self._json_multi_year(data),
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
            "maturity_progress": self._json_maturity(data),
            "sbti_trajectory": self._json_sbti(data),
            "supplier_programme": self._json_supplier_programme(data),
            "climate_risk_summary": self._json_climate_risk(data),
            "data_quality_snapshot": data.get("data_quality_summary", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        report_date = self._get_val(
            data, "report_date", datetime.utcnow().strftime("%Y-%m-%d")
        )
        return (
            f"# Scope 3 Enterprise Dashboard - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {report_date} | "
            f"**Pack:** PACK-043 Scope 3 Complete v{_MODULE_VERSION}\n\n"
            "---"
        )

    def _md_executive_metrics(self, data: Dict[str, Any]) -> str:
        """Render Markdown executive summary metrics."""
        s3 = self._scope3_total(data)
        full = self._full_footprint(data)
        s3_pct = (s3 / full * 100) if full > 0 else 0.0
        yoy = data.get("yoy_change_pct")
        cats = data.get("categories_reported", 0)
        lines = [
            "## 1. Executive Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total Scope 3** | **{_fmt_tco2e(s3)}** |",
            f"| Full Footprint (S1+S2+S3) | {_fmt_tco2e(full)} |",
            f"| Scope 3 Share of Total | {s3_pct:.1f}% |",
            f"| Categories Reported | {cats} of 15 |",
        ]
        if yoy is not None:
            lines.append(f"| Year-over-Year Change | {_fmt_pct(yoy)} |")
        return "\n".join(lines)

    def _md_multi_year_trend(self, data: Dict[str, Any]) -> str:
        """Render Markdown multi-year trend table."""
        trends = data.get("multi_year_trends", [])
        if not trends:
            return "## 2. Multi-Year Trend\n\nNo multi-year data available."
        lines = [
            "## 2. Multi-Year Trend",
            "",
            "| Year | Scope 3 tCO2e | Change | Intensity |",
            "|------|--------------|--------|-----------|",
        ]
        for entry in trends:
            year = entry.get("year", "")
            em = _fmt_tco2e(entry.get("scope3_tco2e"))
            change = entry.get("change_pct")
            change_str = _fmt_pct(change) if change is not None else "-"
            intensity = entry.get("intensity")
            int_str = f"{intensity:.2f}" if intensity is not None else "-"
            lines.append(f"| {year} | {em} | {change_str} | {int_str} |")
        return "\n".join(lines)

    def _md_top_categories(self, data: Dict[str, Any]) -> str:
        """Render Markdown top 5 category breakdown."""
        s3 = self._scope3_total(data)
        top_cats = sorted(
            data.get("top_categories", []),
            key=lambda c: c.get("emissions_tco2e", 0),
            reverse=True,
        )[:5]
        if not top_cats:
            return "## 3. Top 5 Emission Categories\n\nNo category data available."
        lines = [
            "## 3. Top 5 Emission Categories",
            "",
            "| Rank | Category | tCO2e | % of Scope 3 | YoY |",
            "|------|----------|-------|-------------|-----|",
        ]
        cumulative = 0.0
        for i, cat in enumerate(top_cats, 1):
            num = cat.get("category_number", "?")
            name = cat.get("category_name", "Unknown")
            em = cat.get("emissions_tco2e", 0.0)
            cumulative += em
            pct = _pct_of(em, s3)
            yoy = cat.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            lines.append(
                f"| {i} | Cat {num} - {name} | {_fmt_tco2e(em)} | {pct} | {yoy_str} |"
            )
        cum_pct = _pct_of(cumulative, s3)
        lines.append(f"\n*Top 5 categories account for {cum_pct} of total Scope 3.*")
        return "\n".join(lines)

    def _md_maturity_progress(self, data: Dict[str, Any]) -> str:
        """Render Markdown maturity progress gauge."""
        maturity = data.get("maturity_progress", {})
        if not maturity:
            return "## 4. Maturity Progress\n\nNo maturity data available."
        overall = maturity.get("overall_score")
        target = maturity.get("target_score")
        tier = maturity.get("current_tier", "-")
        lines = [
            "## 4. Maturity Progress",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Current Tier | {tier} |",
        ]
        if overall is not None:
            lines.append(f"| Overall Score | {overall:.0f}/100 |")
        if target is not None:
            lines.append(f"| Target Score | {target:.0f}/100 |")
        categories = maturity.get("category_tiers", [])
        if categories:
            lines.append("")
            lines.append("| Category | Current | Target |")
            lines.append("|----------|---------|--------|")
            for cat in categories:
                cname = cat.get("category_name", "")
                cur = cat.get("current_tier", "-")
                tgt = cat.get("target_tier", "-")
                lines.append(f"| {cname} | {cur} | {tgt} |")
        return "\n".join(lines)

    def _md_sbti_trajectory(self, data: Dict[str, Any]) -> str:
        """Render Markdown SBTi trajectory vs target."""
        sbti = data.get("sbti_trajectory", {})
        if not sbti:
            return "## 5. SBTi Trajectory\n\nNo SBTi trajectory data available."
        status = sbti.get("status", "Not assessed")
        pathway = sbti.get("pathway", "-")
        base_year = sbti.get("base_year", "-")
        target_year = sbti.get("target_year", "-")
        target_pct = sbti.get("target_reduction_pct")
        actual_pct = sbti.get("actual_reduction_pct")
        lines = [
            "## 5. SBTi Trajectory vs Target",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Status | {status} |",
            f"| Pathway | {pathway} |",
            f"| Base Year | {base_year} |",
            f"| Target Year | {target_year} |",
        ]
        if target_pct is not None:
            lines.append(f"| Target Reduction | {target_pct:.1f}% |")
        if actual_pct is not None:
            lines.append(f"| Actual Reduction | {actual_pct:.1f}% |")
        trajectory = sbti.get("trajectory_points", [])
        if trajectory:
            lines.append("")
            lines.append("| Year | Target tCO2e | Actual tCO2e | Variance |")
            lines.append("|------|-------------|-------------|----------|")
            for pt in trajectory:
                yr = pt.get("year", "")
                tgt = _fmt_tco2e(pt.get("target_tco2e"))
                act = _fmt_tco2e(pt.get("actual_tco2e"))
                var = pt.get("variance_pct")
                var_str = _fmt_pct(var) if var is not None else "-"
                lines.append(f"| {yr} | {tgt} | {act} | {var_str} |")
        return "\n".join(lines)

    def _md_supplier_programme(self, data: Dict[str, Any]) -> str:
        """Render Markdown supplier programme status."""
        supplier = data.get("supplier_programme", {})
        if not supplier:
            return "## 6. Supplier Programme Status\n\nNo supplier programme data."
        engaged = supplier.get("suppliers_engaged", 0)
        total = supplier.get("total_suppliers", 0)
        response_rate = supplier.get("response_rate_pct")
        coverage = supplier.get("emission_coverage_pct")
        lines = [
            "## 6. Supplier Programme Status",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Suppliers Engaged | {engaged} / {total} |",
        ]
        if response_rate is not None:
            lines.append(f"| Response Rate | {response_rate:.1f}% |")
        if coverage is not None:
            lines.append(f"| Emission Coverage | {coverage:.1f}% |")
        commitments = supplier.get("commitments", {})
        if commitments:
            lines.append(f"| SBTi Committed | {commitments.get('sbti', 0)} |")
            lines.append(f"| RE100 Committed | {commitments.get('re100', 0)} |")
            lines.append(f"| Net-Zero Pledged | {commitments.get('net_zero', 0)} |")
        return "\n".join(lines)

    def _md_climate_risk(self, data: Dict[str, Any]) -> str:
        """Render Markdown climate risk summary."""
        risk = data.get("climate_risk_summary", {})
        if not risk:
            return "## 7. Climate Risk Summary\n\nNo climate risk data available."
        transition = _risk_label(risk.get("transition_risk_level"))
        physical = _risk_label(risk.get("physical_risk_level"))
        financial = risk.get("financial_impact_mln")
        lines = [
            "## 7. Climate Risk Summary",
            "",
            "| Risk Type | Level | Key Driver |",
            "|-----------|-------|-----------|",
            f"| Transition | {transition} | {risk.get('transition_driver', '-')} |",
            f"| Physical | {physical} | {risk.get('physical_driver', '-')} |",
        ]
        if financial is not None:
            lines.append(f"\n**Estimated Financial Impact:** ${financial:.1f}M")
        opportunities = risk.get("opportunities", [])
        if opportunities:
            lines.append("\n**Opportunities:**")
            for opp in opportunities[:3]:
                lines.append(f"- {opp}")
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality snapshot."""
        quality = data.get("data_quality_summary", {})
        if not quality:
            return "## 8. Data Quality Snapshot\n\nNo data quality data available."
        overall = quality.get("overall_dqr_score")
        primary = quality.get("primary_data_pct")
        coverage = quality.get("coverage_pct")
        lines = [
            "## 8. Data Quality Snapshot",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if overall is not None:
            lines.append(f"| Overall DQR Score | {overall:.1f} / 5.0 |")
        if primary is not None:
            lines.append(f"| Primary Data Coverage | {primary:.0f}% |")
        if coverage is not None:
            lines.append(f"| Category Coverage | {coverage:.0f}% |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-043 Scope 3 Complete v{_MODULE_VERSION} | {ts}*\n"
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
            f"<title>Scope 3 Enterprise Dashboard - {company} ({year})</title>\n"
            "<style>\n"
            ":root{--primary:#0A1628;--primary-light:#1B2A4A;--accent:#3B82F6;"
            "--accent-light:#60A5FA;--bg:#F0F2F5;--card-bg:#FFFFFF;--text:#1A1A2E;"
            "--text-muted:#6B7280;--border:#D1D5DB;--success:#10B981;"
            "--warning:#F59E0B;--danger:#EF4444;}\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:2rem;"
            "background:var(--bg);color:var(--text);line-height:1.6;}\n"
            ".container{max-width:1100px;margin:0 auto;}\n"
            "h1{color:var(--primary);border-bottom:3px solid var(--accent);"
            "padding-bottom:0.5rem;margin-bottom:1.5rem;}\n"
            "h2{color:var(--primary-light);margin-top:2rem;"
            "border-bottom:1px solid var(--border);padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid var(--border);padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:var(--primary);color:#fff;font-weight:600;}\n"
            "tr:nth-child(even){background:#F8FAFC;}\n"
            ".total-row{font-weight:bold;background:#DBEAFE;}\n"
            ".metric-card{display:inline-block;background:var(--card-bg);border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:180px;"
            "border-top:3px solid var(--accent);box-shadow:0 1px 3px rgba(0,0,0,0.1);}\n"
            ".metric-value{font-size:1.8rem;font-weight:700;color:var(--primary);}\n"
            ".metric-label{font-size:0.85rem;color:var(--text-muted);}\n"
            ".section{margin-bottom:2rem;background:var(--card-bg);"
            "padding:1.5rem;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}\n"
            ".risk-high{color:var(--danger);font-weight:700;}\n"
            ".risk-medium{color:var(--warning);font-weight:700;}\n"
            ".risk-low{color:var(--success);font-weight:700;}\n"
            ".status-on-track{color:var(--success);font-weight:700;}\n"
            ".status-at-risk{color:var(--warning);font-weight:700;}\n"
            ".status-off-track{color:var(--danger);font-weight:700;}\n"
            ".gauge-bar{height:24px;border-radius:12px;background:#E5E7EB;overflow:hidden;}\n"
            ".gauge-fill{height:100%;border-radius:12px;transition:width 0.5s;}\n"
            ".provenance{font-size:0.8rem;color:var(--text-muted);font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n<div class=\"container\">\n"
            f"{body}\n"
            "</div>\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            '<div class="section">\n'
            f"<h1>Scope 3 Enterprise Dashboard &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Pack:</strong> PACK-043 v{_MODULE_VERSION}</p>\n"
            "<hr>\n</div>"
        )

    def _html_executive_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary metric cards."""
        s3 = self._scope3_total(data)
        full = self._full_footprint(data)
        s3_pct = (s3 / full * 100) if full > 0 else 0.0
        yoy = data.get("yoy_change_pct")
        cats = data.get("categories_reported", 0)
        cards = [
            ("Total Scope 3", _fmt_tco2e(s3)),
            ("Full Footprint", _fmt_tco2e(full)),
            ("Scope 3 Share", f"{s3_pct:.1f}%"),
            ("Categories", f"{cats} / 15"),
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
            "<h2>1. Executive Summary</h2>\n"
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_multi_year_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML multi-year trend table."""
        trends = data.get("multi_year_trends", [])
        if not trends:
            return ""
        rows = ""
        for entry in trends:
            year = entry.get("year", "")
            em = _fmt_tco2e(entry.get("scope3_tco2e"))
            change = entry.get("change_pct")
            change_str = _fmt_pct(change) if change is not None else "-"
            intensity = entry.get("intensity")
            int_str = f"{intensity:.2f}" if intensity is not None else "-"
            rows += (
                f"<tr><td>{year}</td><td>{em}</td>"
                f"<td>{change_str}</td><td>{int_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Multi-Year Trend</h2>\n"
            "<table><thead><tr><th>Year</th><th>Scope 3 tCO2e</th>"
            "<th>Change</th><th>Intensity</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
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
        for i, cat in enumerate(top_cats, 1):
            num = cat.get("category_number", "?")
            name = cat.get("category_name", "Unknown")
            em = cat.get("emissions_tco2e", 0.0)
            pct = _pct_of(em, s3)
            yoy = cat.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            rows += (
                f"<tr><td>{i}</td><td>Cat {num} - {name}</td>"
                f"<td>{_fmt_tco2e(em)}</td><td>{pct}</td><td>{yoy_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. Top 5 Emission Categories</h2>\n"
            "<table><thead><tr><th>Rank</th><th>Category</th><th>tCO2e</th>"
            "<th>%</th><th>YoY</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_maturity_progress(self, data: Dict[str, Any]) -> str:
        """Render HTML maturity progress with gauge bar."""
        maturity = data.get("maturity_progress", {})
        if not maturity:
            return ""
        overall = maturity.get("overall_score", 0)
        target = maturity.get("target_score", 100)
        tier = maturity.get("current_tier", "-")
        pct = min(100, (overall / target * 100) if target > 0 else 0)
        color = "#10B981" if pct >= 70 else "#F59E0B" if pct >= 40 else "#EF4444"
        gauge_html = (
            f'<div class="gauge-bar">'
            f'<div class="gauge-fill" style="width:{pct:.0f}%;background:{color};"></div>'
            f"</div>"
            f"<p><strong>{tier}</strong> &mdash; {overall:.0f} / {target:.0f} "
            f"({pct:.0f}% towards target)</p>"
        )
        cat_rows = ""
        for cat in maturity.get("category_tiers", []):
            cname = cat.get("category_name", "")
            cur = cat.get("current_tier", "-")
            tgt = cat.get("target_tier", "-")
            cat_rows += f"<tr><td>{cname}</td><td>{cur}</td><td>{tgt}</td></tr>\n"
        cat_table = ""
        if cat_rows:
            cat_table = (
                "<table><thead><tr><th>Category</th><th>Current</th>"
                f"<th>Target</th></tr></thead><tbody>{cat_rows}</tbody></table>"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Maturity Progress</h2>\n"
            f"{gauge_html}\n{cat_table}\n</div>"
        )

    def _html_sbti_trajectory(self, data: Dict[str, Any]) -> str:
        """Render HTML SBTi trajectory section."""
        sbti = data.get("sbti_trajectory", {})
        if not sbti:
            return ""
        status = sbti.get("status", "Not assessed")
        status_css = (
            "status-on-track" if "track" in status.lower()
            else "status-at-risk" if "risk" in status.lower()
            else "status-off-track"
        )
        rows = (
            f'<tr><td>Status</td><td class="{status_css}">{status}</td></tr>\n'
            f"<tr><td>Pathway</td><td>{sbti.get('pathway', '-')}</td></tr>\n"
            f"<tr><td>Base Year</td><td>{sbti.get('base_year', '-')}</td></tr>\n"
            f"<tr><td>Target Year</td><td>{sbti.get('target_year', '-')}</td></tr>\n"
        )
        if sbti.get("target_reduction_pct") is not None:
            rows += f"<tr><td>Target Reduction</td><td>{sbti['target_reduction_pct']:.1f}%</td></tr>\n"
        if sbti.get("actual_reduction_pct") is not None:
            rows += f"<tr><td>Actual Reduction</td><td>{sbti['actual_reduction_pct']:.1f}%</td></tr>\n"
        traj_table = ""
        trajectory = sbti.get("trajectory_points", [])
        if trajectory:
            traj_rows = ""
            for pt in trajectory:
                yr = pt.get("year", "")
                tgt = _fmt_tco2e(pt.get("target_tco2e"))
                act = _fmt_tco2e(pt.get("actual_tco2e"))
                var = pt.get("variance_pct")
                var_str = _fmt_pct(var) if var is not None else "-"
                traj_rows += (
                    f"<tr><td>{yr}</td><td>{tgt}</td><td>{act}</td><td>{var_str}</td></tr>\n"
                )
            traj_table = (
                "<table><thead><tr><th>Year</th><th>Target</th>"
                f"<th>Actual</th><th>Variance</th></tr></thead><tbody>{traj_rows}</tbody></table>"
            )
        return (
            '<div class="section">\n'
            "<h2>5. SBTi Trajectory vs Target</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n"
            f"{traj_table}\n</div>"
        )

    def _html_supplier_programme(self, data: Dict[str, Any]) -> str:
        """Render HTML supplier programme status."""
        supplier = data.get("supplier_programme", {})
        if not supplier:
            return ""
        engaged = supplier.get("suppliers_engaged", 0)
        total = supplier.get("total_suppliers", 0)
        rate = supplier.get("response_rate_pct")
        coverage = supplier.get("emission_coverage_pct")
        commitments = supplier.get("commitments", {})
        rows = f"<tr><td>Suppliers Engaged</td><td>{engaged} / {total}</td></tr>\n"
        if rate is not None:
            rows += f"<tr><td>Response Rate</td><td>{rate:.1f}%</td></tr>\n"
        if coverage is not None:
            rows += f"<tr><td>Emission Coverage</td><td>{coverage:.1f}%</td></tr>\n"
        if commitments:
            rows += f"<tr><td>SBTi Committed</td><td>{commitments.get('sbti', 0)}</td></tr>\n"
            rows += f"<tr><td>RE100 Committed</td><td>{commitments.get('re100', 0)}</td></tr>\n"
            rows += f"<tr><td>Net-Zero Pledged</td><td>{commitments.get('net_zero', 0)}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>6. Supplier Programme Status</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_climate_risk(self, data: Dict[str, Any]) -> str:
        """Render HTML climate risk summary."""
        risk = data.get("climate_risk_summary", {})
        if not risk:
            return ""
        trans = _risk_label(risk.get("transition_risk_level"))
        phys = _risk_label(risk.get("physical_risk_level"))
        trans_css = f"risk-{trans.lower()}"
        phys_css = f"risk-{phys.lower()}"
        rows = (
            f'<tr><td>Transition Risk</td><td class="{trans_css}">{trans}</td>'
            f"<td>{risk.get('transition_driver', '-')}</td></tr>\n"
            f'<tr><td>Physical Risk</td><td class="{phys_css}">{phys}</td>'
            f"<td>{risk.get('physical_driver', '-')}</td></tr>\n"
        )
        financial = risk.get("financial_impact_mln")
        fin_html = ""
        if financial is not None:
            fin_html = f"<p><strong>Estimated Financial Impact:</strong> ${financial:.1f}M</p>"
        return (
            '<div class="section">\n'
            "<h2>7. Climate Risk Summary</h2>\n"
            "<table><thead><tr><th>Risk Type</th><th>Level</th>"
            f"<th>Key Driver</th></tr></thead>\n<tbody>{rows}</tbody></table>\n"
            f"{fin_html}\n</div>"
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality snapshot."""
        quality = data.get("data_quality_summary", {})
        if not quality:
            return ""
        rows = ""
        overall = quality.get("overall_dqr_score")
        primary = quality.get("primary_data_pct")
        coverage = quality.get("coverage_pct")
        if overall is not None:
            rows += f"<tr><td>Overall DQR Score</td><td>{overall:.1f} / 5.0</td></tr>\n"
        if primary is not None:
            rows += f"<tr><td>Primary Data Coverage</td><td>{primary:.0f}%</td></tr>\n"
        if coverage is not None:
            rows += f"<tr><td>Category Coverage</td><td>{coverage:.0f}%</td></tr>\n"
        if not rows:
            return ""
        return (
            '<div class="section">\n'
            "<h2>8. Data Quality Snapshot</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-043 Scope 3 Complete v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON HELPERS
    # ==================================================================

    def _json_multi_year(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build multi-year trend chart data."""
        trends = data.get("multi_year_trends", [])
        return [
            {
                "year": entry.get("year"),
                "scope3_tco2e": entry.get("scope3_tco2e"),
                "change_pct": entry.get("change_pct"),
                "intensity": entry.get("intensity"),
            }
            for entry in trends
        ]

    def _json_maturity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build maturity progress data."""
        maturity = data.get("maturity_progress", {})
        return {
            "overall_score": maturity.get("overall_score"),
            "target_score": maturity.get("target_score"),
            "current_tier": maturity.get("current_tier"),
            "category_tiers": maturity.get("category_tiers", []),
        }

    def _json_sbti(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build SBTi trajectory data."""
        sbti = data.get("sbti_trajectory", {})
        return {
            "status": sbti.get("status"),
            "pathway": sbti.get("pathway"),
            "base_year": sbti.get("base_year"),
            "target_year": sbti.get("target_year"),
            "target_reduction_pct": sbti.get("target_reduction_pct"),
            "actual_reduction_pct": sbti.get("actual_reduction_pct"),
            "trajectory_points": sbti.get("trajectory_points", []),
        }

    def _json_supplier_programme(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build supplier programme data."""
        supplier = data.get("supplier_programme", {})
        return {
            "suppliers_engaged": supplier.get("suppliers_engaged"),
            "total_suppliers": supplier.get("total_suppliers"),
            "response_rate_pct": supplier.get("response_rate_pct"),
            "emission_coverage_pct": supplier.get("emission_coverage_pct"),
            "commitments": supplier.get("commitments", {}),
        }

    def _json_climate_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build climate risk summary data."""
        risk = data.get("climate_risk_summary", {})
        return {
            "transition_risk_level": risk.get("transition_risk_level"),
            "physical_risk_level": risk.get("physical_risk_level"),
            "transition_driver": risk.get("transition_driver"),
            "physical_driver": risk.get("physical_driver"),
            "financial_impact_mln": risk.get("financial_impact_mln"),
            "opportunities": risk.get("opportunities", []),
        }
