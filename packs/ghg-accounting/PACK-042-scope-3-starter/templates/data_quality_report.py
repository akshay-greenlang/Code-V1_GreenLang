# -*- coding: utf-8 -*-
"""
DataQualityReportTemplate - Data Quality Assessment and Roadmap for PACK-042.

Generates a data quality assessment and improvement roadmap covering
overall DQR score, per-category quality spider/radar chart data, 5 DQI
breakdown per category, quality trend over time, gap analysis, prioritized
improvement actions with effort/impact scoring, and framework minimum
thresholds comparison.

Sections:
    1. Overall DQR Score
    2. Per-Category Quality Assessment
    3. Five DQI Breakdown
    4. Quality Trend Over Time
    5. Gap Analysis
    6. Prioritized Improvement Actions
    7. Framework Minimum Thresholds

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, analytical blue theme)
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

# Five Data Quality Indicators per GHG Protocol
DQI_NAMES = [
    "Technological Representativeness",
    "Temporal Representativeness",
    "Geographical Representativeness",
    "Completeness",
    "Reliability",
]


def _fmt_score(value: Optional[float]) -> str:
    """Format DQR score."""
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def _quality_label(score: Optional[float]) -> str:
    """Convert DQR score to quality label."""
    if score is None:
        return "N/A"
    if score >= 4.0:
        return "HIGH"
    if score >= 2.5:
        return "MEDIUM"
    return "LOW"


def _quality_css(score: Optional[float]) -> str:
    """Return CSS class for quality score."""
    if score is None:
        return ""
    if score >= 4.0:
        return "quality-high"
    if score >= 2.5:
        return "quality-medium"
    return "quality-low"


class DataQualityReportTemplate:
    """
    Data quality assessment and improvement roadmap template.

    Renders data quality reports with overall DQR scores, per-category
    quality assessments, five DQI breakdowns, quality trends over time,
    gap analysis, prioritized improvement actions with effort/impact
    scoring, and framework minimum threshold comparisons. All outputs
    include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = DataQualityReportTemplate()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DataQualityReportTemplate."""
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
        """Render data quality report as Markdown.

        Args:
            data: Validated data quality data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overall_score(data),
            self._md_category_quality(data),
            self._md_dqi_breakdown(data),
            self._md_quality_trend(data),
            self._md_gap_analysis(data),
            self._md_improvement_actions(data),
            self._md_framework_thresholds(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render data quality report as HTML.

        Args:
            data: Validated data quality data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overall_score(data),
            self._html_category_quality(data),
            self._html_dqi_breakdown(data),
            self._html_quality_trend(data),
            self._html_gap_analysis(data),
            self._html_improvement_actions(data),
            self._html_framework_thresholds(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render data quality report as JSON-serializable dict.

        Args:
            data: Validated data quality data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "data_quality_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "overall_dqr": data.get("overall_dqr", {}),
            "category_quality": data.get("category_quality", []),
            "dqi_breakdown": data.get("dqi_breakdown", []),
            "quality_trend": data.get("quality_trend", []),
            "gap_analysis": data.get("gap_analysis", []),
            "improvement_actions": data.get("improvement_actions", []),
            "framework_thresholds": data.get("framework_thresholds", []),
            "radar_chart_data": self._json_radar_data(data),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Scope 3 Data Quality Report - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_overall_score(self, data: Dict[str, Any]) -> str:
        """Render Markdown overall DQR score."""
        dqr = data.get("overall_dqr", {})
        score = dqr.get("score")
        label = _quality_label(score)
        lines = [
            "## 1. Overall DQR Score",
            "",
        ]
        if score is not None:
            lines.append(f"**Overall Score:** {score:.1f} / 5.0 ({label})")
        else:
            lines.append("No overall DQR score available.")
        dqi_scores = dqr.get("dqi_scores", {})
        if dqi_scores:
            lines.append("")
            lines.append("| Data Quality Indicator | Score |")
            lines.append("|----------------------|-------|")
            for dqi_name in DQI_NAMES:
                key = dqi_name.lower().replace(" ", "_")
                s = dqi_scores.get(key)
                lines.append(f"| {dqi_name} | {_fmt_score(s)} |")
        return "\n".join(lines)

    def _md_category_quality(self, data: Dict[str, Any]) -> str:
        """Render Markdown per-category quality assessment."""
        categories = data.get("category_quality", [])
        if not categories:
            return "## 2. Per-Category Quality Assessment\n\nNo per-category quality data."
        lines = [
            "## 2. Per-Category Quality Assessment",
            "",
            "| Category | DQR Score | Level | Primary Data % | Coverage % | Key Issue |",
            "|----------|----------|-------|---------------|-----------|-----------|",
        ]
        for cat in categories:
            name = cat.get("category_name", "")
            score = cat.get("dqr_score")
            level = _quality_label(score)
            primary = cat.get("primary_data_pct")
            primary_str = f"{primary:.0f}%" if primary is not None else "-"
            coverage = cat.get("coverage_pct")
            cov_str = f"{coverage:.0f}%" if coverage is not None else "-"
            issue = cat.get("key_issue", "-")
            lines.append(
                f"| {name} | {_fmt_score(score)} | {level} | {primary_str} | {cov_str} | {issue} |"
            )
        return "\n".join(lines)

    def _md_dqi_breakdown(self, data: Dict[str, Any]) -> str:
        """Render Markdown 5 DQI breakdown per category."""
        breakdown = data.get("dqi_breakdown", [])
        if not breakdown:
            return "## 3. Five DQI Breakdown\n\nNo DQI breakdown data available."
        header = "| Category |"
        separator = "|----------|"
        for dqi in DQI_NAMES:
            short = dqi.split()[0][:4]
            header += f" {short} |"
            separator += "------|"
        lines = [
            "## 3. Five DQI Breakdown",
            "",
            header,
            separator,
        ]
        for cat in breakdown:
            name = cat.get("category_name", "")
            row = f"| {name} |"
            scores = cat.get("dqi_scores", {})
            for dqi in DQI_NAMES:
                key = dqi.lower().replace(" ", "_")
                s = scores.get(key)
                row += f" {_fmt_score(s)} |"
            lines.append(row)
        return "\n".join(lines)

    def _md_quality_trend(self, data: Dict[str, Any]) -> str:
        """Render Markdown quality trend over time."""
        trend = data.get("quality_trend", [])
        if not trend:
            return "## 4. Quality Trend Over Time\n\nNo trend data available."
        lines = [
            "## 4. Quality Trend Over Time",
            "",
            "| Period | Overall DQR | Primary Data % | Categories Assessed | Change |",
            "|--------|-----------|---------------|--------------------|---------| ",
        ]
        for t in trend:
            period = t.get("period", "")
            dqr = t.get("overall_dqr")
            dqr_str = _fmt_score(dqr)
            primary = t.get("primary_data_pct")
            primary_str = f"{primary:.0f}%" if primary is not None else "-"
            assessed = t.get("categories_assessed", "-")
            change = t.get("change_pct")
            change_str = f"{change:+.1f}%" if change is not None else "-"
            lines.append(
                f"| {period} | {dqr_str} | {primary_str} | {assessed} | {change_str} |"
            )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown gap analysis."""
        gaps = data.get("gap_analysis", [])
        if not gaps:
            return "## 5. Gap Analysis\n\nNo gaps identified."
        lines = [
            "## 5. Gap Analysis",
            "",
            "| Category | Current Score | Required Score | Gap | Priority | Action Required |",
            "|----------|-------------|---------------|-----|----------|----------------|",
        ]
        for g in gaps:
            cat = g.get("category_name", "")
            current = _fmt_score(g.get("current_score"))
            required = _fmt_score(g.get("required_score"))
            gap = g.get("gap")
            gap_str = f"{gap:.1f}" if gap is not None else "-"
            priority = g.get("priority", "-")
            action = g.get("action_required", "-")
            lines.append(
                f"| {cat} | {current} | {required} | {gap_str} | {priority} | {action} |"
            )
        return "\n".join(lines)

    def _md_improvement_actions(self, data: Dict[str, Any]) -> str:
        """Render Markdown prioritized improvement actions."""
        actions = data.get("improvement_actions", [])
        if not actions:
            return "## 6. Prioritized Improvement Actions\n\nNo improvement actions defined."
        lines = [
            "## 6. Prioritized Improvement Actions",
            "",
            "| Priority | Action | Category | Effort | Impact | Timeline | Expected DQR Gain |",
            "|----------|--------|----------|--------|--------|----------|------------------|",
        ]
        for i, a in enumerate(actions, 1):
            action = a.get("action", "")
            cat = a.get("category", "-")
            effort = a.get("effort", "-")
            impact = a.get("impact", "-")
            timeline = a.get("timeline", "-")
            gain = a.get("expected_dqr_gain")
            gain_str = f"+{gain:.1f}" if gain is not None else "-"
            lines.append(
                f"| {i} | {action} | {cat} | {effort} | {impact} | {timeline} | {gain_str} |"
            )
        return "\n".join(lines)

    def _md_framework_thresholds(self, data: Dict[str, Any]) -> str:
        """Render Markdown framework minimum thresholds comparison."""
        thresholds = data.get("framework_thresholds", [])
        if not thresholds:
            return "## 7. Framework Minimum Thresholds\n\nNo framework threshold data."
        lines = [
            "## 7. Framework Minimum Thresholds",
            "",
            "| Framework | Min DQR Required | Current DQR | Status | Gap |",
            "|-----------|-----------------|------------|--------|-----|",
        ]
        for fw in thresholds:
            name = fw.get("framework_name", "")
            min_dqr = fw.get("min_dqr_required")
            min_str = _fmt_score(min_dqr)
            current = fw.get("current_dqr")
            curr_str = _fmt_score(current)
            meets = fw.get("meets_threshold", False)
            status = "PASS" if meets else "FAIL"
            gap = fw.get("gap")
            gap_str = f"{gap:.1f}" if gap is not None else "-"
            lines.append(
                f"| {name} | {min_str} | {curr_str} | **{status}** | {gap_str} |"
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
            f"<title>Data Quality Report - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#2980B9;border-bottom:3px solid #2980B9;padding-bottom:0.5rem;}\n"
            "h2{color:#21618C;margin-top:2rem;border-bottom:1px solid #ccc;padding-bottom:0.3rem;}\n"
            "h3{color:#2980B9;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:#ebf5fb;font-weight:600;color:#21618C;}\n"
            "tr:nth-child(even){background:#f4f9fd;}\n"
            ".quality-high{color:#27AE60;font-weight:700;}\n"
            ".quality-medium{color:#F39C12;font-weight:700;}\n"
            ".quality-low{color:#E74C3C;font-weight:700;}\n"
            ".metric-card{display:inline-block;background:#ebf5fb;border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:170px;"
            "border-top:3px solid #2980B9;}\n"
            ".metric-value{font-size:1.5rem;font-weight:700;color:#21618C;}\n"
            ".metric-label{font-size:0.85rem;color:#555;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".pass{color:#27AE60;font-weight:700;}\n"
            ".fail{color:#E74C3C;font-weight:700;}\n"
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
            f"<h1>Scope 3 Data Quality Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overall_score(self, data: Dict[str, Any]) -> str:
        """Render HTML overall DQR score."""
        dqr = data.get("overall_dqr", {})
        score = dqr.get("score")
        if score is None:
            return ""
        label = _quality_label(score)
        css = _quality_css(score)
        card = (
            f'<div class="metric-card">'
            f'<div class="metric-value">{score:.1f} / 5.0</div>'
            f'<div class="metric-label {css}">{label}</div></div>\n'
        )
        dqi_scores = dqr.get("dqi_scores", {})
        rows = ""
        for dqi_name in DQI_NAMES:
            key = dqi_name.lower().replace(" ", "_")
            s = dqi_scores.get(key)
            s_css = _quality_css(s)
            rows += f'<tr><td>{dqi_name}</td><td class="{s_css}">{_fmt_score(s)}</td></tr>\n'
        table = ""
        if rows:
            table = (
                "<table><thead><tr><th>DQI</th><th>Score</th></tr></thead>\n"
                f"<tbody>{rows}</tbody></table>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>1. Overall DQR Score</h2>\n"
            f"<div>{card}</div>\n{table}</div>"
        )

    def _html_category_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML per-category quality assessment."""
        categories = data.get("category_quality", [])
        if not categories:
            return ""
        rows = ""
        for cat in categories:
            name = cat.get("category_name", "")
            score = cat.get("dqr_score")
            level = _quality_label(score)
            css = _quality_css(score)
            primary = cat.get("primary_data_pct")
            primary_str = f"{primary:.0f}%" if primary is not None else "-"
            coverage = cat.get("coverage_pct")
            cov_str = f"{coverage:.0f}%" if coverage is not None else "-"
            issue = cat.get("key_issue", "-")
            rows += (
                f'<tr><td>{name}</td><td>{_fmt_score(score)}</td>'
                f'<td class="{css}">{level}</td><td>{primary_str}</td>'
                f"<td>{cov_str}</td><td>{issue}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Per-Category Quality Assessment</h2>\n"
            "<table><thead><tr><th>Category</th><th>DQR</th><th>Level</th>"
            "<th>Primary %</th><th>Coverage</th>"
            f"<th>Issue</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_dqi_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML 5 DQI breakdown per category."""
        breakdown = data.get("dqi_breakdown", [])
        if not breakdown:
            return ""
        headers = "<th>Category</th>"
        for dqi in DQI_NAMES:
            short = dqi.split()[0][:4]
            headers += f"<th>{short}</th>"
        rows = ""
        for cat in breakdown:
            name = cat.get("category_name", "")
            row = f"<td>{name}</td>"
            scores = cat.get("dqi_scores", {})
            for dqi in DQI_NAMES:
                key = dqi.lower().replace(" ", "_")
                s = scores.get(key)
                css = _quality_css(s)
                row += f'<td class="{css}">{_fmt_score(s)}</td>'
            rows += f"<tr>{row}</tr>\n"
        return (
            '<div class="section">\n'
            "<h2>3. Five DQI Breakdown</h2>\n"
            f"<table><thead><tr>{headers}</tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_quality_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML quality trend over time."""
        trend = data.get("quality_trend", [])
        if not trend:
            return ""
        rows = ""
        for t in trend:
            period = t.get("period", "")
            dqr = t.get("overall_dqr")
            primary = t.get("primary_data_pct")
            primary_str = f"{primary:.0f}%" if primary is not None else "-"
            assessed = t.get("categories_assessed", "-")
            change = t.get("change_pct")
            change_str = f"{change:+.1f}%" if change is not None else "-"
            rows += (
                f"<tr><td>{period}</td><td>{_fmt_score(dqr)}</td>"
                f"<td>{primary_str}</td><td>{assessed}</td><td>{change_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Quality Trend Over Time</h2>\n"
            "<table><thead><tr><th>Period</th><th>DQR</th><th>Primary %</th>"
            f"<th>Assessed</th><th>Change</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap analysis."""
        gaps = data.get("gap_analysis", [])
        if not gaps:
            return ""
        rows = ""
        for g in gaps:
            cat = g.get("category_name", "")
            current = _fmt_score(g.get("current_score"))
            required = _fmt_score(g.get("required_score"))
            gap = g.get("gap")
            gap_str = f"{gap:.1f}" if gap is not None else "-"
            priority = g.get("priority", "-")
            action = g.get("action_required", "-")
            rows += (
                f"<tr><td>{cat}</td><td>{current}</td><td>{required}</td>"
                f"<td>{gap_str}</td><td>{priority}</td><td>{action}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>5. Gap Analysis</h2>\n"
            "<table><thead><tr><th>Category</th><th>Current</th><th>Required</th>"
            f"<th>Gap</th><th>Priority</th><th>Action</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_improvement_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML prioritized improvement actions."""
        actions = data.get("improvement_actions", [])
        if not actions:
            return ""
        rows = ""
        for i, a in enumerate(actions, 1):
            action = a.get("action", "")
            cat = a.get("category", "-")
            effort = a.get("effort", "-")
            impact = a.get("impact", "-")
            timeline = a.get("timeline", "-")
            gain = a.get("expected_dqr_gain")
            gain_str = f"+{gain:.1f}" if gain is not None else "-"
            rows += (
                f"<tr><td>{i}</td><td>{action}</td><td>{cat}</td>"
                f"<td>{effort}</td><td>{impact}</td><td>{timeline}</td>"
                f"<td>{gain_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>6. Prioritized Improvement Actions</h2>\n"
            "<table><thead><tr><th>#</th><th>Action</th><th>Category</th>"
            "<th>Effort</th><th>Impact</th><th>Timeline</th>"
            f"<th>DQR Gain</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_framework_thresholds(self, data: Dict[str, Any]) -> str:
        """Render HTML framework minimum thresholds."""
        thresholds = data.get("framework_thresholds", [])
        if not thresholds:
            return ""
        rows = ""
        for fw in thresholds:
            name = fw.get("framework_name", "")
            min_dqr = _fmt_score(fw.get("min_dqr_required"))
            current = _fmt_score(fw.get("current_dqr"))
            meets = fw.get("meets_threshold", False)
            status = "PASS" if meets else "FAIL"
            css = "pass" if meets else "fail"
            gap = fw.get("gap")
            gap_str = f"{gap:.1f}" if gap is not None else "-"
            rows += (
                f"<tr><td>{name}</td><td>{min_dqr}</td><td>{current}</td>"
                f'<td class="{css}">{status}</td><td>{gap_str}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>7. Framework Minimum Thresholds</h2>\n"
            "<table><thead><tr><th>Framework</th><th>Min DQR</th><th>Current</th>"
            f"<th>Status</th><th>Gap</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
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

    def _json_radar_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build radar/spider chart data for per-category DQI scores."""
        breakdown = data.get("dqi_breakdown", [])
        result = []
        for cat in breakdown:
            entry = {
                "category_name": cat.get("category_name", ""),
                "axes": [],
            }
            scores = cat.get("dqi_scores", {})
            for dqi in DQI_NAMES:
                key = dqi.lower().replace(" ", "_")
                entry["axes"].append({
                    "axis": dqi,
                    "value": scores.get(key),
                })
            result.append(entry)
        return result
