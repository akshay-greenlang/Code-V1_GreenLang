# -*- coding: utf-8 -*-
"""
SBTiProgressReportTemplate - SBTi Target vs Actual Trajectory for PACK-043.

Generates an SBTi progress tracking report with target vs actual trajectory
line chart data, category coverage assessment, near-term and long-term
milestone tracking, FLAG pathway (if applicable), variance analysis, and
submission package readiness status.

Sections:
    1. Target Summary
    2. Trajectory: Target vs Actual
    3. Category Coverage Assessment
    4. Near-Term Milestone Tracking
    5. Long-Term Milestone Tracking
    6. FLAG Pathway (if applicable)
    7. Variance Analysis
    8. Submission Readiness

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, SBTi red #C0392B theme)
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


def _readiness_label(score: Optional[float]) -> str:
    """Map readiness score to label."""
    if score is None:
        return "Not Assessed"
    if score >= 90:
        return "Ready"
    if score >= 70:
        return "Nearly Ready"
    if score >= 50:
        return "In Progress"
    return "Not Ready"


class SBTiProgressReportTemplate:
    """
    SBTi target vs actual trajectory tracking template.

    Renders SBTi progress reports with trajectory charts, category
    coverage, milestone tracking, FLAG pathways, and submission
    readiness assessments. All outputs include SHA-256 provenance.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = SBTiProgressReportTemplate()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SBTiProgressReportTemplate."""
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
        """Render SBTi progress report as Markdown.

        Args:
            data: Validated SBTi progress data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_target_summary(data),
            self._md_trajectory(data),
            self._md_category_coverage(data),
            self._md_near_term_milestones(data),
            self._md_long_term_milestones(data),
            self._md_flag_pathway(data),
            self._md_variance_analysis(data),
            self._md_submission_readiness(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render SBTi progress report as HTML.

        Args:
            data: Validated SBTi progress data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_target_summary(data),
            self._html_trajectory(data),
            self._html_category_coverage(data),
            self._html_near_term_milestones(data),
            self._html_long_term_milestones(data),
            self._html_flag_pathway(data),
            self._html_variance_analysis(data),
            self._html_submission_readiness(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render SBTi progress report as JSON-serializable dict.

        Args:
            data: Validated SBTi progress data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "sbti_progress_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "target_summary": data.get("target_summary", {}),
            "trajectory_data": data.get("trajectory_data", []),
            "category_coverage": data.get("category_coverage", []),
            "near_term_milestones": data.get("near_term_milestones", []),
            "long_term_milestones": data.get("long_term_milestones", []),
            "flag_pathway": data.get("flag_pathway", {}),
            "variance_analysis": data.get("variance_analysis", []),
            "submission_readiness": data.get("submission_readiness", {}),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# SBTi Progress Report - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_target_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown target summary."""
        ts = data.get("target_summary", {})
        if not ts:
            return "## 1. Target Summary\n\nNo target summary available."
        lines = [
            "## 1. Target Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Target Type | {ts.get('target_type', '-')} |",
            f"| Base Year | {ts.get('base_year', '-')} |",
            f"| Base Year Emissions | {_fmt_tco2e(ts.get('base_year_tco2e'))} |",
            f"| Target Year | {ts.get('target_year', '-')} |",
        ]
        target_red = ts.get("target_reduction_pct")
        if target_red is not None:
            lines.append(f"| Target Reduction | {target_red:.1f}% |")
        pathway = ts.get("pathway")
        if pathway:
            lines.append(f"| Pathway | {pathway} |")
        status = ts.get("status", "-")
        lines.append(f"| Current Status | {status} |")
        actual_red = ts.get("actual_reduction_pct")
        if actual_red is not None:
            lines.append(f"| Actual Reduction to Date | {actual_red:.1f}% |")
        return "\n".join(lines)

    def _md_trajectory(self, data: Dict[str, Any]) -> str:
        """Render Markdown trajectory table."""
        trajectory = data.get("trajectory_data", [])
        if not trajectory:
            return "## 2. Target vs Actual Trajectory\n\nNo trajectory data available."
        lines = [
            "## 2. Target vs Actual Trajectory",
            "",
            "| Year | Target tCO2e | Actual tCO2e | Variance | Status |",
            "|------|-------------|-------------|----------|--------|",
        ]
        for pt in trajectory:
            year = pt.get("year", "")
            target = _fmt_tco2e(pt.get("target_tco2e"))
            actual = _fmt_tco2e(pt.get("actual_tco2e"))
            var_pct = pt.get("variance_pct")
            var_str = _fmt_pct(var_pct) if var_pct is not None else "-"
            status = pt.get("status", "-")
            lines.append(f"| {year} | {target} | {actual} | {var_str} | {status} |")
        return "\n".join(lines)

    def _md_category_coverage(self, data: Dict[str, Any]) -> str:
        """Render Markdown category coverage assessment."""
        coverage = data.get("category_coverage", [])
        if not coverage:
            return "## 3. Category Coverage Assessment\n\nNo coverage data available."
        lines = [
            "## 3. Category Coverage Assessment",
            "",
            "| Category | Included | Emissions (tCO2e) | % of Scope 3 | Rationale |",
            "|----------|----------|------------------|-------------|-----------|",
        ]
        for cat in coverage:
            num = cat.get("category_number", "?")
            name = cat.get("category_name", "Unknown")
            included = "Yes" if cat.get("included", False) else "No"
            em = _fmt_tco2e(cat.get("emissions_tco2e"))
            pct = cat.get("pct_of_scope3")
            pct_str = f"{pct:.1f}%" if pct is not None else "-"
            rationale = cat.get("rationale", "-")
            lines.append(
                f"| Cat {num} - {name} | {included} | {em} | {pct_str} | {rationale} |"
            )
        total_cov = data.get("total_coverage_pct")
        if total_cov is not None:
            lines.append(f"\n**Total Coverage:** {total_cov:.1f}% of Scope 3 emissions")
        return "\n".join(lines)

    def _md_near_term_milestones(self, data: Dict[str, Any]) -> str:
        """Render Markdown near-term milestone tracking."""
        milestones = data.get("near_term_milestones", [])
        if not milestones:
            return "## 4. Near-Term Milestones\n\nNo near-term milestones defined."
        lines = [
            "## 4. Near-Term Milestones (5-10 years)",
            "",
            "| Milestone | Target Date | Status | Progress |",
            "|-----------|------------|--------|----------|",
        ]
        for ms in milestones:
            name = ms.get("milestone_name", "-")
            target_date = ms.get("target_date", "-")
            status = ms.get("status", "-")
            progress = ms.get("progress_pct")
            prog_str = f"{progress:.0f}%" if progress is not None else "-"
            lines.append(f"| {name} | {target_date} | {status} | {prog_str} |")
        return "\n".join(lines)

    def _md_long_term_milestones(self, data: Dict[str, Any]) -> str:
        """Render Markdown long-term milestone tracking."""
        milestones = data.get("long_term_milestones", [])
        if not milestones:
            return "## 5. Long-Term Milestones\n\nNo long-term milestones defined."
        lines = [
            "## 5. Long-Term Milestones (Net-Zero)",
            "",
            "| Milestone | Target Year | Required Reduction | Current Progress |",
            "|-----------|-----------|-------------------|-----------------|",
        ]
        for ms in milestones:
            name = ms.get("milestone_name", "-")
            target_year = ms.get("target_year", "-")
            req = ms.get("required_reduction_pct")
            req_str = f"{req:.1f}%" if req is not None else "-"
            progress = ms.get("current_progress_pct")
            prog_str = f"{progress:.1f}%" if progress is not None else "-"
            lines.append(f"| {name} | {target_year} | {req_str} | {prog_str} |")
        return "\n".join(lines)

    def _md_flag_pathway(self, data: Dict[str, Any]) -> str:
        """Render Markdown FLAG pathway section."""
        flag = data.get("flag_pathway", {})
        if not flag or not flag.get("applicable", False):
            return ""
        lines = [
            "## 6. FLAG Pathway",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| FLAG Applicable | Yes |",
            f"| FLAG Base Year Emissions | {_fmt_tco2e(flag.get('base_year_tco2e'))} |",
            f"| FLAG Target Reduction | {flag.get('target_reduction_pct', 'N/A')}% |",
            f"| FLAG Target Year | {flag.get('target_year', '-')} |",
        ]
        commodities = flag.get("commodities", [])
        if commodities:
            lines.append(f"| Key Commodities | {', '.join(commodities)} |")
        return "\n".join(lines)

    def _md_variance_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown variance analysis."""
        variances = data.get("variance_analysis", [])
        if not variances:
            return "## 7. Variance Analysis\n\nNo variance data available."
        lines = [
            "## 7. Variance Analysis",
            "",
            "| Factor | Impact (tCO2e) | Direction | Category |",
            "|--------|---------------|-----------|----------|",
        ]
        for v in variances:
            factor = v.get("factor", "-")
            impact = _fmt_tco2e(v.get("impact_tco2e"))
            direction = v.get("direction", "-")
            cat = v.get("category", "-")
            lines.append(f"| {factor} | {impact} | {direction} | {cat} |")
        return "\n".join(lines)

    def _md_submission_readiness(self, data: Dict[str, Any]) -> str:
        """Render Markdown submission readiness."""
        readiness = data.get("submission_readiness", {})
        if not readiness:
            return "## 8. Submission Readiness\n\nNo readiness data available."
        score = readiness.get("overall_score")
        lines = [
            "## 8. Submission Package Readiness",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if score is not None:
            lines.append(
                f"| Overall Readiness | {score:.0f}/100 ({_readiness_label(score)}) |"
            )
        checklist = readiness.get("checklist", [])
        if checklist:
            lines.append("")
            lines.append("| Requirement | Status |")
            lines.append("|-------------|--------|")
            for item in checklist:
                req = item.get("requirement", "-")
                status = "Pass" if item.get("met", False) else "Gap"
                lines.append(f"| {req} | {status} |")
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
        """Wrap body in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>SBTi Progress Report - {company} ({year})</title>\n"
            "<style>\n"
            ":root{--primary:#C0392B;--primary-light:#E74C3C;--accent:#F1948A;"
            "--bg:#FDF2F2;--card-bg:#FFFFFF;--text:#1A1A2E;--text-muted:#6B7280;"
            "--border:#D1D5DB;--success:#10B981;--warning:#F59E0B;--danger:#EF4444;}\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:2rem;"
            "background:var(--bg);color:var(--text);line-height:1.6;}\n"
            ".container{max-width:1100px;margin:0 auto;}\n"
            "h1{color:var(--primary);border-bottom:3px solid var(--primary);"
            "padding-bottom:0.5rem;}\n"
            "h2{color:var(--primary);margin-top:2rem;"
            "border-bottom:1px solid var(--border);padding-bottom:0.3rem;}\n"
            "h3{color:var(--primary-light);}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid var(--border);padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:var(--primary);color:#fff;font-weight:600;}\n"
            "tr:nth-child(even){background:#FDF2F2;}\n"
            ".section{margin-bottom:2rem;background:var(--card-bg);"
            "padding:1.5rem;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}\n"
            ".on-track{color:var(--success);font-weight:700;}\n"
            ".behind{color:var(--danger);font-weight:700;}\n"
            ".at-risk{color:var(--warning);font-weight:700;}\n"
            ".ready{color:var(--success);font-weight:700;}\n"
            ".not-ready{color:var(--danger);font-weight:700;}\n"
            ".gauge-bar{height:24px;border-radius:12px;background:#E5E7EB;overflow:hidden;}\n"
            ".gauge-fill{height:100%;border-radius:12px;}\n"
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
            f"<h1>SBTi Progress Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Pack:</strong> PACK-043 v{_MODULE_VERSION}</p>\n"
            "<hr>\n</div>"
        )

    def _html_target_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML target summary."""
        ts = data.get("target_summary", {})
        if not ts:
            return ""
        rows = (
            f"<tr><td>Target Type</td><td>{ts.get('target_type', '-')}</td></tr>\n"
            f"<tr><td>Base Year</td><td>{ts.get('base_year', '-')}</td></tr>\n"
            f"<tr><td>Base Year Emissions</td><td>{_fmt_tco2e(ts.get('base_year_tco2e'))}</td></tr>\n"
            f"<tr><td>Target Year</td><td>{ts.get('target_year', '-')}</td></tr>\n"
        )
        target_red = ts.get("target_reduction_pct")
        if target_red is not None:
            rows += f"<tr><td>Target Reduction</td><td>{target_red:.1f}%</td></tr>\n"
        status = ts.get("status", "-")
        status_css = "on-track" if "track" in status.lower() else "at-risk"
        rows += f'<tr><td>Status</td><td class="{status_css}">{status}</td></tr>\n'
        actual_red = ts.get("actual_reduction_pct")
        if actual_red is not None:
            rows += f"<tr><td>Actual Reduction</td><td>{actual_red:.1f}%</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>1. Target Summary</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_trajectory(self, data: Dict[str, Any]) -> str:
        """Render HTML trajectory table."""
        trajectory = data.get("trajectory_data", [])
        if not trajectory:
            return ""
        rows = ""
        for pt in trajectory:
            year = pt.get("year", "")
            target = _fmt_tco2e(pt.get("target_tco2e"))
            actual = _fmt_tco2e(pt.get("actual_tco2e"))
            var_pct = pt.get("variance_pct")
            var_str = _fmt_pct(var_pct) if var_pct is not None else "-"
            status = pt.get("status", "-")
            css = "on-track" if status.lower() == "on track" else "behind"
            rows += (
                f"<tr><td>{year}</td><td>{target}</td><td>{actual}</td>"
                f"<td>{var_str}</td><td class=\"{css}\">{status}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Target vs Actual Trajectory</h2>\n"
            "<table><thead><tr><th>Year</th><th>Target</th><th>Actual</th>"
            "<th>Variance</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_category_coverage(self, data: Dict[str, Any]) -> str:
        """Render HTML category coverage."""
        coverage = data.get("category_coverage", [])
        if not coverage:
            return ""
        rows = ""
        for cat in coverage:
            num = cat.get("category_number", "?")
            name = cat.get("category_name", "Unknown")
            included = cat.get("included", False)
            inc_str = "Yes" if included else "No"
            inc_css = "on-track" if included else "behind"
            em = _fmt_tco2e(cat.get("emissions_tco2e"))
            pct = cat.get("pct_of_scope3")
            pct_str = f"{pct:.1f}%" if pct is not None else "-"
            rows += (
                f"<tr><td>Cat {num} - {name}</td>"
                f'<td class="{inc_css}">{inc_str}</td>'
                f"<td>{em}</td><td>{pct_str}</td></tr>\n"
            )
        total_cov = data.get("total_coverage_pct")
        cov_html = ""
        if total_cov is not None:
            cov_html = f"<p><strong>Total Coverage:</strong> {total_cov:.1f}% of Scope 3</p>"
        return (
            '<div class="section">\n'
            "<h2>3. Category Coverage Assessment</h2>\n"
            "<table><thead><tr><th>Category</th><th>Included</th>"
            "<th>Emissions</th><th>% of Scope 3</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n{cov_html}\n</div>"
        )

    def _html_near_term_milestones(self, data: Dict[str, Any]) -> str:
        """Render HTML near-term milestones."""
        milestones = data.get("near_term_milestones", [])
        if not milestones:
            return ""
        rows = ""
        for ms in milestones:
            name = ms.get("milestone_name", "-")
            target_date = ms.get("target_date", "-")
            status = ms.get("status", "-")
            progress = ms.get("progress_pct")
            prog_str = f"{progress:.0f}%" if progress is not None else "-"
            css = "on-track" if "complete" in status.lower() or "track" in status.lower() else "at-risk"
            rows += (
                f"<tr><td>{name}</td><td>{target_date}</td>"
                f'<td class="{css}">{status}</td><td>{prog_str}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>4. Near-Term Milestones (5-10 years)</h2>\n"
            "<table><thead><tr><th>Milestone</th><th>Target Date</th>"
            "<th>Status</th><th>Progress</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_long_term_milestones(self, data: Dict[str, Any]) -> str:
        """Render HTML long-term milestones."""
        milestones = data.get("long_term_milestones", [])
        if not milestones:
            return ""
        rows = ""
        for ms in milestones:
            name = ms.get("milestone_name", "-")
            target_year = ms.get("target_year", "-")
            req = ms.get("required_reduction_pct")
            req_str = f"{req:.1f}%" if req is not None else "-"
            progress = ms.get("current_progress_pct")
            prog_str = f"{progress:.1f}%" if progress is not None else "-"
            rows += (
                f"<tr><td>{name}</td><td>{target_year}</td>"
                f"<td>{req_str}</td><td>{prog_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>5. Long-Term Milestones (Net-Zero)</h2>\n"
            "<table><thead><tr><th>Milestone</th><th>Target Year</th>"
            "<th>Required Reduction</th><th>Current Progress</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_flag_pathway(self, data: Dict[str, Any]) -> str:
        """Render HTML FLAG pathway section."""
        flag = data.get("flag_pathway", {})
        if not flag or not flag.get("applicable", False):
            return ""
        rows = (
            f"<tr><td>FLAG Base Year Emissions</td>"
            f"<td>{_fmt_tco2e(flag.get('base_year_tco2e'))}</td></tr>\n"
            f"<tr><td>FLAG Target Reduction</td>"
            f"<td>{flag.get('target_reduction_pct', 'N/A')}%</td></tr>\n"
            f"<tr><td>FLAG Target Year</td><td>{flag.get('target_year', '-')}</td></tr>\n"
        )
        commodities = flag.get("commodities", [])
        if commodities:
            rows += f"<tr><td>Key Commodities</td><td>{', '.join(commodities)}</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>6. FLAG Pathway</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_variance_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML variance analysis."""
        variances = data.get("variance_analysis", [])
        if not variances:
            return ""
        rows = ""
        for v in variances:
            factor = v.get("factor", "-")
            impact = _fmt_tco2e(v.get("impact_tco2e"))
            direction = v.get("direction", "-")
            dir_css = "on-track" if direction.lower() == "decrease" else "behind"
            cat = v.get("category", "-")
            rows += (
                f"<tr><td>{factor}</td><td>{impact}</td>"
                f'<td class="{dir_css}">{direction}</td><td>{cat}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>7. Variance Analysis</h2>\n"
            "<table><thead><tr><th>Factor</th><th>Impact</th>"
            "<th>Direction</th><th>Category</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_submission_readiness(self, data: Dict[str, Any]) -> str:
        """Render HTML submission readiness with gauge."""
        readiness = data.get("submission_readiness", {})
        if not readiness:
            return ""
        score = readiness.get("overall_score", 0)
        label = _readiness_label(score)
        color = "#10B981" if score >= 90 else "#F59E0B" if score >= 70 else "#EF4444"
        gauge_html = (
            f'<div class="gauge-bar">'
            f'<div class="gauge-fill" style="width:{score:.0f}%;background:{color};"></div>'
            f"</div>"
            f"<p><strong>{label}</strong> &mdash; {score:.0f}/100</p>"
        )
        checklist_html = ""
        checklist = readiness.get("checklist", [])
        if checklist:
            rows = ""
            for item in checklist:
                req = item.get("requirement", "-")
                met = item.get("met", False)
                css = "ready" if met else "not-ready"
                status = "Pass" if met else "Gap"
                rows += f'<tr><td>{req}</td><td class="{css}">{status}</td></tr>\n'
            checklist_html = (
                "<table><thead><tr><th>Requirement</th><th>Status</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>"
            )
        return (
            '<div class="section">\n'
            "<h2>8. Submission Package Readiness</h2>\n"
            f"{gauge_html}\n{checklist_html}\n</div>"
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
