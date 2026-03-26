# -*- coding: utf-8 -*-
"""
SupplierEngagementReportTemplate - Supplier Engagement Dashboard for PACK-042.

Generates a supplier engagement status dashboard covering overall
engagement metrics, response rate trends, data quality distribution
(Level 1-5 pie chart data), top supplier profiles, engagement ROI,
upcoming deadlines, and quality improvement trajectory.

Sections:
    1. Overall Engagement Metrics
    2. Response Rate Trends
    3. Data Quality Distribution
    4. Top Supplier Profiles
    5. Engagement ROI
    6. Upcoming Deadlines
    7. Quality Improvement Trajectory

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, professional green theme)
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


class SupplierEngagementReportTemplate:
    """
    Supplier engagement status dashboard template.

    Renders supplier engagement dashboards with overall metrics,
    response rate trends, data quality distribution across 5 levels,
    top supplier profiles, engagement ROI calculations, upcoming
    deadlines, and quality improvement trajectories. All outputs
    include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = SupplierEngagementReportTemplate()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SupplierEngagementReportTemplate."""
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

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render supplier engagement report as Markdown.

        Args:
            data: Validated supplier engagement data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overall_metrics(data),
            self._md_response_trends(data),
            self._md_quality_distribution(data),
            self._md_top_suppliers(data),
            self._md_engagement_roi(data),
            self._md_upcoming_deadlines(data),
            self._md_quality_trajectory(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render supplier engagement report as HTML.

        Args:
            data: Validated supplier engagement data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overall_metrics(data),
            self._html_response_trends(data),
            self._html_quality_distribution(data),
            self._html_top_suppliers(data),
            self._html_engagement_roi(data),
            self._html_upcoming_deadlines(data),
            self._html_quality_trajectory(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render supplier engagement report as JSON-serializable dict.

        Args:
            data: Validated supplier engagement data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "supplier_engagement_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "overall_metrics": data.get("overall_metrics", {}),
            "response_trends": data.get("response_trends", []),
            "quality_distribution": data.get("quality_distribution", {}),
            "top_suppliers": data.get("top_suppliers", []),
            "engagement_roi": data.get("engagement_roi", {}),
            "upcoming_deadlines": data.get("upcoming_deadlines", []),
            "quality_trajectory": data.get("quality_trajectory", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Supplier Engagement Report - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_overall_metrics(self, data: Dict[str, Any]) -> str:
        """Render Markdown overall engagement metrics."""
        metrics = data.get("overall_metrics", {})
        if not metrics:
            return "## 1. Overall Engagement Metrics\n\nNo metrics available."
        lines = [
            "## 1. Overall Engagement Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        field_labels = [
            ("total_suppliers", "Total Suppliers"),
            ("suppliers_engaged", "Suppliers Engaged"),
            ("engagement_rate_pct", "Engagement Rate"),
            ("response_rate_pct", "Response Rate"),
            ("primary_data_suppliers", "Suppliers with Primary Data"),
            ("scope3_coverage_pct", "Scope 3 Coverage (by spend)"),
        ]
        for key, label in field_labels:
            val = metrics.get(key)
            if val is not None:
                if "pct" in key:
                    lines.append(f"| {label} | {val:.1f}% |")
                else:
                    lines.append(f"| {label} | {val} |")
        return "\n".join(lines)

    def _md_response_trends(self, data: Dict[str, Any]) -> str:
        """Render Markdown response rate trends."""
        trends = data.get("response_trends", [])
        if not trends:
            return "## 2. Response Rate Trends\n\nNo trend data available."
        lines = [
            "## 2. Response Rate Trends",
            "",
            "| Period | Sent | Responded | Response Rate | Data Quality Avg |",
            "|--------|------|-----------|---------------|-----------------|",
        ]
        for t in trends:
            period = t.get("period", "")
            sent = t.get("sent", 0)
            responded = t.get("responded", 0)
            rate = t.get("response_rate_pct")
            rate_str = f"{rate:.1f}%" if rate is not None else "-"
            dq_avg = t.get("avg_data_quality")
            dq_str = f"{dq_avg:.1f}" if dq_avg is not None else "-"
            lines.append(f"| {period} | {sent} | {responded} | {rate_str} | {dq_str} |")
        return "\n".join(lines)

    def _md_quality_distribution(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality distribution."""
        dist = data.get("quality_distribution", {})
        if not dist:
            return "## 3. Data Quality Distribution\n\nNo distribution data available."
        levels = dist.get("levels", [])
        lines = [
            "## 3. Data Quality Distribution",
            "",
            "| Level | Description | Supplier Count | % of Total | Emissions Coverage |",
            "|-------|------------|---------------|-----------|-------------------|",
        ]
        for level in levels:
            lvl = level.get("level", "")
            desc = level.get("description", "")
            count = level.get("supplier_count", 0)
            pct = level.get("pct_of_total")
            pct_str = f"{pct:.1f}%" if pct is not None else "-"
            coverage = level.get("emissions_coverage_pct")
            cov_str = f"{coverage:.1f}%" if coverage is not None else "-"
            lines.append(f"| {lvl} | {desc} | {count} | {pct_str} | {cov_str} |")
        return "\n".join(lines)

    def _md_top_suppliers(self, data: Dict[str, Any]) -> str:
        """Render Markdown top supplier profiles."""
        suppliers = data.get("top_suppliers", [])
        if not suppliers:
            return "## 4. Top Supplier Profiles\n\nNo supplier profiles available."
        lines = [
            "## 4. Top Supplier Profiles",
            "",
            "| Supplier | Spend | Emissions tCO2e | Data Level | Engagement | Last Response |",
            "|----------|-------|----------------|-----------|-----------|---------------|",
        ]
        for s in suppliers:
            name = s.get("supplier_name", "")
            spend = s.get("spend", "-")
            em = _fmt_tco2e(s.get("emissions_tco2e"))
            level = s.get("data_level", "-")
            engagement = s.get("engagement_status", "-")
            last_resp = s.get("last_response_date", "-")
            lines.append(
                f"| {name} | {spend} | {em} | {level} | {engagement} | {last_resp} |"
            )
        return "\n".join(lines)

    def _md_engagement_roi(self, data: Dict[str, Any]) -> str:
        """Render Markdown engagement ROI."""
        roi = data.get("engagement_roi", {})
        if not roi:
            return "## 5. Engagement ROI\n\nNo ROI data available."
        lines = [
            "## 5. Engagement ROI",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        field_labels = [
            ("total_investment", "Total Investment"),
            ("dq_improvement_pct", "DQ Improvement"),
            ("uncertainty_reduction_pct", "Uncertainty Reduction"),
            ("cost_per_supplier", "Cost per Supplier Engaged"),
            ("emissions_covered_tco2e", "Emissions Covered"),
            ("cost_per_tco2e_covered", "Cost per tCO2e Covered"),
        ]
        for key, label in field_labels:
            val = roi.get(key)
            if val is not None:
                if "pct" in key:
                    lines.append(f"| {label} | {val:.1f}% |")
                elif "tco2e" in key:
                    lines.append(f"| {label} | {_fmt_tco2e(val)} |")
                else:
                    lines.append(f"| {label} | {val} |")
        return "\n".join(lines)

    def _md_upcoming_deadlines(self, data: Dict[str, Any]) -> str:
        """Render Markdown upcoming deadlines."""
        deadlines = data.get("upcoming_deadlines", [])
        if not deadlines:
            return "## 6. Upcoming Deadlines\n\nNo upcoming deadlines."
        lines = [
            "## 6. Upcoming Deadlines",
            "",
            "| Deadline | Activity | Suppliers Affected | Status |",
            "|----------|----------|-------------------|--------|",
        ]
        for d in deadlines:
            date = d.get("deadline_date", "")
            activity = d.get("activity", "")
            affected = d.get("suppliers_affected", 0)
            status = d.get("status", "-")
            lines.append(f"| {date} | {activity} | {affected} | {status} |")
        return "\n".join(lines)

    def _md_quality_trajectory(self, data: Dict[str, Any]) -> str:
        """Render Markdown quality improvement trajectory."""
        trajectory = data.get("quality_trajectory", [])
        if not trajectory:
            return "## 7. Quality Improvement Trajectory\n\nNo trajectory data available."
        lines = [
            "## 7. Quality Improvement Trajectory",
            "",
            "| Period | Avg DQR Score | Primary Data % | Suppliers Engaged | Notes |",
            "|--------|-------------|---------------|-------------------|-------|",
        ]
        for t in trajectory:
            period = t.get("period", "")
            dqr = t.get("avg_dqr_score")
            dqr_str = f"{dqr:.1f}" if dqr is not None else "-"
            primary = t.get("primary_data_pct")
            primary_str = f"{primary:.0f}%" if primary is not None else "-"
            engaged = t.get("suppliers_engaged", "-")
            notes = t.get("notes", "-")
            lines.append(
                f"| {period} | {dqr_str} | {primary_str} | {engaged} | {notes} |"
            )
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
            f"<title>Supplier Engagement Report - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#27AE60;border-bottom:3px solid #27AE60;padding-bottom:0.5rem;}\n"
            "h2{color:#1E8449;margin-top:2rem;border-bottom:1px solid #ccc;padding-bottom:0.3rem;}\n"
            "h3{color:#27AE60;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#eafaf1;font-weight:600;color:#1E8449;}\n"
            "tr:nth-child(even){background:#f5fdf8;}\n"
            ".total-row{font-weight:bold;background:#d4efdf;}\n"
            ".metric-card{display:inline-block;background:#eafaf1;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:170px;"
            "border-top:3px solid #27AE60;}\n"
            ".metric-value{font-size:1.5rem;font-weight:700;color:#1E8449;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".level-5{color:#27AE60;font-weight:700;}\n"
            ".level-4{color:#2ECC71;font-weight:600;}\n"
            ".level-3{color:#F39C12;font-weight:600;}\n"
            ".level-2{color:#E67E22;font-weight:600;}\n"
            ".level-1{color:#E74C3C;font-weight:600;}\n"
            ".status-active{color:#27AE60;font-weight:700;}\n"
            ".status-pending{color:#F39C12;font-weight:700;}\n"
            ".status-overdue{color:#E74C3C;font-weight:700;}\n"
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
            f"<h1>Supplier Engagement Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overall_metrics(self, data: Dict[str, Any]) -> str:
        """Render HTML overall engagement metrics with cards."""
        metrics = data.get("overall_metrics", {})
        if not metrics:
            return ""
        cards = []
        if metrics.get("total_suppliers") is not None:
            cards.append(("Total Suppliers", str(metrics["total_suppliers"])))
        if metrics.get("engagement_rate_pct") is not None:
            cards.append(("Engagement Rate", f"{metrics['engagement_rate_pct']:.1f}%"))
        if metrics.get("response_rate_pct") is not None:
            cards.append(("Response Rate", f"{metrics['response_rate_pct']:.1f}%"))
        if metrics.get("scope3_coverage_pct") is not None:
            cards.append(("S3 Coverage", f"{metrics['scope3_coverage_pct']:.1f}%"))
        card_html = ""
        for label, val in cards:
            card_html += (
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>1. Overall Engagement Metrics</h2>\n"
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_response_trends(self, data: Dict[str, Any]) -> str:
        """Render HTML response rate trends."""
        trends = data.get("response_trends", [])
        if not trends:
            return ""
        rows = ""
        for t in trends:
            period = t.get("period", "")
            sent = t.get("sent", 0)
            responded = t.get("responded", 0)
            rate = t.get("response_rate_pct")
            rate_str = f"{rate:.1f}%" if rate is not None else "-"
            dq_avg = t.get("avg_data_quality")
            dq_str = f"{dq_avg:.1f}" if dq_avg is not None else "-"
            rows += f"<tr><td>{period}</td><td>{sent}</td><td>{responded}</td><td>{rate_str}</td><td>{dq_str}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>2. Response Rate Trends</h2>\n"
            "<table><thead><tr><th>Period</th><th>Sent</th><th>Responded</th>"
            f"<th>Rate</th><th>Avg DQ</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_quality_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality distribution."""
        dist = data.get("quality_distribution", {})
        levels = dist.get("levels", [])
        if not levels:
            return ""
        rows = ""
        for level in levels:
            lvl = level.get("level", "")
            desc = level.get("description", "")
            count = level.get("supplier_count", 0)
            pct = level.get("pct_of_total")
            pct_str = f"{pct:.1f}%" if pct is not None else "-"
            coverage = level.get("emissions_coverage_pct")
            cov_str = f"{coverage:.1f}%" if coverage is not None else "-"
            lvl_css = f"level-{lvl}" if str(lvl).isdigit() else ""
            rows += (
                f'<tr><td class="{lvl_css}">{lvl}</td><td>{desc}</td>'
                f"<td>{count}</td><td>{pct_str}</td><td>{cov_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. Data Quality Distribution</h2>\n"
            "<table><thead><tr><th>Level</th><th>Description</th><th>Count</th>"
            f"<th>% of Total</th><th>Emissions %</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_top_suppliers(self, data: Dict[str, Any]) -> str:
        """Render HTML top supplier profiles."""
        suppliers = data.get("top_suppliers", [])
        if not suppliers:
            return ""
        rows = ""
        for s in suppliers:
            name = s.get("supplier_name", "")
            spend = s.get("spend", "-")
            em = _fmt_tco2e(s.get("emissions_tco2e"))
            level = s.get("data_level", "-")
            engagement = s.get("engagement_status", "-")
            last_resp = s.get("last_response_date", "-")
            eng_css = "status-active" if engagement == "Active" else (
                "status-pending" if engagement == "Pending" else ""
            )
            rows += (
                f"<tr><td>{name}</td><td>{spend}</td><td>{em}</td>"
                f"<td>{level}</td>"
                f'<td class="{eng_css}">{engagement}</td><td>{last_resp}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>4. Top Supplier Profiles</h2>\n"
            "<table><thead><tr><th>Supplier</th><th>Spend</th><th>tCO2e</th>"
            f"<th>Level</th><th>Engagement</th><th>Last Response</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_engagement_roi(self, data: Dict[str, Any]) -> str:
        """Render HTML engagement ROI."""
        roi = data.get("engagement_roi", {})
        if not roi:
            return ""
        rows = ""
        field_labels = [
            ("total_investment", "Total Investment"),
            ("dq_improvement_pct", "DQ Improvement"),
            ("uncertainty_reduction_pct", "Uncertainty Reduction"),
            ("cost_per_supplier", "Cost per Supplier"),
            ("emissions_covered_tco2e", "Emissions Covered"),
        ]
        for key, label in field_labels:
            val = roi.get(key)
            if val is not None:
                if "pct" in key:
                    rows += f"<tr><td>{label}</td><td>{val:.1f}%</td></tr>\n"
                elif "tco2e" in key:
                    rows += f"<tr><td>{label}</td><td>{_fmt_tco2e(val)}</td></tr>\n"
                else:
                    rows += f"<tr><td>{label}</td><td>{val}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>5. Engagement ROI</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_upcoming_deadlines(self, data: Dict[str, Any]) -> str:
        """Render HTML upcoming deadlines."""
        deadlines = data.get("upcoming_deadlines", [])
        if not deadlines:
            return ""
        rows = ""
        for d in deadlines:
            date = d.get("deadline_date", "")
            activity = d.get("activity", "")
            affected = d.get("suppliers_affected", 0)
            status = d.get("status", "-")
            s_css = (
                "status-overdue" if status == "Overdue"
                else "status-pending" if status == "Pending"
                else "status-active" if status == "On Track"
                else ""
            )
            rows += (
                f"<tr><td>{date}</td><td>{activity}</td><td>{affected}</td>"
                f'<td class="{s_css}">{status}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>6. Upcoming Deadlines</h2>\n"
            "<table><thead><tr><th>Date</th><th>Activity</th>"
            f"<th>Suppliers</th><th>Status</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_quality_trajectory(self, data: Dict[str, Any]) -> str:
        """Render HTML quality improvement trajectory."""
        trajectory = data.get("quality_trajectory", [])
        if not trajectory:
            return ""
        rows = ""
        for t in trajectory:
            period = t.get("period", "")
            dqr = t.get("avg_dqr_score")
            dqr_str = f"{dqr:.1f}" if dqr is not None else "-"
            primary = t.get("primary_data_pct")
            primary_str = f"{primary:.0f}%" if primary is not None else "-"
            engaged = t.get("suppliers_engaged", "-")
            notes = t.get("notes", "-")
            rows += (
                f"<tr><td>{period}</td><td>{dqr_str}</td><td>{primary_str}</td>"
                f"<td>{engaged}</td><td>{notes}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>7. Quality Improvement Trajectory</h2>\n"
            "<table><thead><tr><th>Period</th><th>Avg DQR</th><th>Primary %</th>"
            f"<th>Engaged</th><th>Notes</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
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
