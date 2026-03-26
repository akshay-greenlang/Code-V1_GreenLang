# -*- coding: utf-8 -*-
"""
BaseYearSelectionReport - Selection Criteria and Candidate Comparison for PACK-045.

Generates a base year selection report covering candidate year evaluation,
selection criteria scoring, comparison matrices, recommendation rationale,
and stakeholder alignment notes.

Sections:
    1. Selection Criteria Summary
    2. Candidate Year Comparison
    3. Scoring Matrix
    4. Recommendation Rationale
    5. Stakeholder Sign-off

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 45.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "45.0.0"


def _score_badge(score: float) -> str:
    """Return text badge for a score 0-100."""
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Fair"
    return "Poor"


def _score_css(score: float) -> str:
    """Return CSS class for a score value."""
    if score >= 80:
        return "score-excellent"
    if score >= 60:
        return "score-good"
    if score >= 40:
        return "score-fair"
    return "score-poor"


class BaseYearSelectionReport:
    """
    Base year selection report template.

    Renders selection criteria scores, candidate year comparison tables,
    scoring matrices, recommendation rationale, and stakeholder sign-off
    sections. All outputs include SHA-256 provenance hashing for audit
    trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = BaseYearSelectionReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BaseYearSelectionReport."""
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
        """Render base year selection report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_criteria_summary(data),
            self._md_candidate_comparison(data),
            self._md_scoring_matrix(data),
            self._md_recommendation(data),
            self._md_signoff(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render base year selection report as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_criteria_summary(data),
            self._html_candidate_comparison(data),
            self._html_scoring_matrix(data),
            self._html_recommendation(data),
            self._html_signoff(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render base year selection report as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "base_year_selection_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "selected_base_year": self._get_val(data, "selected_base_year", ""),
            "criteria": data.get("criteria", []),
            "candidates": data.get("candidates", []),
            "scoring_matrix": data.get("scoring_matrix", []),
            "recommendation": data.get("recommendation", {}),
            "stakeholder_signoff": data.get("stakeholder_signoff", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        selected = self._get_val(data, "selected_base_year", "TBD")
        return (
            f"# Base Year Selection Report - {company}\n\n"
            f"**Selected Base Year:** {selected} | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_criteria_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown selection criteria summary."""
        criteria = data.get("criteria", [])
        if not criteria:
            return "## 1. Selection Criteria Summary\n\nNo criteria defined."
        lines = [
            "## 1. Selection Criteria Summary",
            "",
            "| # | Criterion | Weight | Description |",
            "|---|----------|--------|-------------|",
        ]
        for i, c in enumerate(criteria, 1):
            name = c.get("name", "")
            weight = c.get("weight", 0)
            desc = c.get("description", "")
            lines.append(f"| {i} | {name} | {weight:.0f}% | {desc} |")
        return "\n".join(lines)

    def _md_candidate_comparison(self, data: Dict[str, Any]) -> str:
        """Render Markdown candidate year comparison."""
        candidates = data.get("candidates", [])
        if not candidates:
            return "## 2. Candidate Year Comparison\n\nNo candidates evaluated."
        lines = [
            "## 2. Candidate Year Comparison",
            "",
            "| Year | Total Emissions (tCO2e) | Data Quality | Coverage | Structural Changes | Score |",
            "|------|------------------------|-------------|----------|-------------------|-------|",
        ]
        for c in candidates:
            year = c.get("year", "")
            total = c.get("total_emissions_tco2e", 0)
            dq = c.get("data_quality_score", 0)
            cov = f"{c.get('coverage_pct', 0):.0f}%"
            structural = c.get("structural_changes", "None")
            score = c.get("overall_score", 0)
            lines.append(
                f"| {year} | {total:,.1f} | {dq:.0f}/100 | {cov} | "
                f"{structural} | **{score:.0f}** |"
            )
        return "\n".join(lines)

    def _md_scoring_matrix(self, data: Dict[str, Any]) -> str:
        """Render Markdown scoring matrix."""
        matrix = data.get("scoring_matrix", [])
        if not matrix:
            return "## 3. Scoring Matrix\n\nNo scoring data available."
        candidates = data.get("candidates", [])
        years = [str(c.get("year", "")) for c in candidates]
        header = "| Criterion | Weight | " + " | ".join(years) + " |"
        sep = "|----------|--------|" + "|------" * len(years) + "|"
        lines = ["## 3. Scoring Matrix", "", header, sep]
        for row in matrix:
            name = row.get("criterion", "")
            weight = row.get("weight", 0)
            scores = row.get("scores", {})
            cells = " | ".join(
                f"{scores.get(y, 0):.0f}" for y in years
            )
            lines.append(f"| {name} | {weight:.0f}% | {cells} |")
        return "\n".join(lines)

    def _md_recommendation(self, data: Dict[str, Any]) -> str:
        """Render Markdown recommendation rationale."""
        rec = data.get("recommendation", {})
        if not rec:
            return "## 4. Recommendation\n\nNo recommendation available."
        year = rec.get("recommended_year", "TBD")
        rationale = rec.get("rationale", "")
        strengths = rec.get("strengths", [])
        limitations = rec.get("limitations", [])
        lines = [
            "## 4. Recommendation",
            "",
            f"**Recommended Base Year:** {year}",
            "",
            f"**Rationale:** {rationale}",
            "",
            "**Strengths:**",
        ]
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("**Limitations:**")
        for lim in limitations:
            lines.append(f"- {lim}")
        return "\n".join(lines)

    def _md_signoff(self, data: Dict[str, Any]) -> str:
        """Render Markdown stakeholder sign-off."""
        signoffs = data.get("stakeholder_signoff", [])
        if not signoffs:
            return ""
        lines = [
            "## 5. Stakeholder Sign-off",
            "",
            "| Stakeholder | Role | Decision | Date |",
            "|------------|------|----------|------|",
        ]
        for s in signoffs:
            name = s.get("name", "")
            role = s.get("role", "")
            decision = s.get("decision", "Pending")
            date = s.get("date", "-")
            lines.append(f"| {name} | {role} | **{decision}** | {date} |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-045 Base Year Management v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body content in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Base Year Selection - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".score-excellent{color:#2a9d8f;font-weight:700;}\n"
            ".score-good{color:#52b788;font-weight:700;}\n"
            ".score-fair{color:#e9c46a;font-weight:700;}\n"
            ".score-poor{color:#e76f51;font-weight:700;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".rec-box{background:#f0f9f4;border:1px solid #2a9d8f;border-radius:8px;"
            "padding:1rem 1.5rem;margin:1rem 0;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        selected = self._get_val(data, "selected_base_year", "TBD")
        return (
            '<div class="section">\n'
            f"<h1>Base Year Selection Report &mdash; {company}</h1>\n"
            f"<p><strong>Selected Base Year:</strong> {selected}</p>\n<hr>\n</div>"
        )

    def _html_criteria_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML criteria summary table."""
        criteria = data.get("criteria", [])
        if not criteria:
            return ""
        rows = ""
        for i, c in enumerate(criteria, 1):
            name = c.get("name", "")
            weight = c.get("weight", 0)
            desc = c.get("description", "")
            rows += f"<tr><td>{i}</td><td>{name}</td><td>{weight:.0f}%</td><td>{desc}</td></tr>\n"
        return (
            '<div class="section">\n<h2>1. Selection Criteria Summary</h2>\n'
            "<table><thead><tr><th>#</th><th>Criterion</th><th>Weight</th>"
            "<th>Description</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_candidate_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML candidate comparison table."""
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        rows = ""
        for c in candidates:
            year = c.get("year", "")
            total = c.get("total_emissions_tco2e", 0)
            dq = c.get("data_quality_score", 0)
            cov = c.get("coverage_pct", 0)
            score = c.get("overall_score", 0)
            css = _score_css(score)
            rows += (
                f'<tr><td>{year}</td><td>{total:,.1f}</td><td>{dq:.0f}/100</td>'
                f'<td>{cov:.0f}%</td><td class="{css}"><strong>{score:.0f}</strong></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>2. Candidate Year Comparison</h2>\n'
            "<table><thead><tr><th>Year</th><th>Total tCO2e</th><th>Data Quality</th>"
            "<th>Coverage</th><th>Score</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_scoring_matrix(self, data: Dict[str, Any]) -> str:
        """Render HTML scoring matrix table."""
        matrix = data.get("scoring_matrix", [])
        candidates = data.get("candidates", [])
        if not matrix or not candidates:
            return ""
        years = [str(c.get("year", "")) for c in candidates]
        year_headers = "".join(f"<th>{y}</th>" for y in years)
        rows = ""
        for row in matrix:
            name = row.get("criterion", "")
            weight = row.get("weight", 0)
            scores = row.get("scores", {})
            cells = "".join(f"<td>{scores.get(y, 0):.0f}</td>" for y in years)
            rows += f"<tr><td>{name}</td><td>{weight:.0f}%</td>{cells}</tr>\n"
        return (
            '<div class="section">\n<h2>3. Scoring Matrix</h2>\n'
            f"<table><thead><tr><th>Criterion</th><th>Weight</th>{year_headers}</tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_recommendation(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendation section."""
        rec = data.get("recommendation", {})
        if not rec:
            return ""
        year = rec.get("recommended_year", "TBD")
        rationale = rec.get("rationale", "")
        strengths = rec.get("strengths", [])
        limitations = rec.get("limitations", [])
        s_items = "".join(f"<li>{s}</li>" for s in strengths)
        l_items = "".join(f"<li>{lim}</li>" for lim in limitations)
        return (
            '<div class="section">\n<h2>4. Recommendation</h2>\n'
            f'<div class="rec-box">\n'
            f"<p><strong>Recommended Base Year:</strong> {year}</p>\n"
            f"<p>{rationale}</p>\n"
            f"<p><strong>Strengths:</strong></p><ul>{s_items}</ul>\n"
            f"<p><strong>Limitations:</strong></p><ul>{l_items}</ul>\n"
            "</div>\n</div>"
        )

    def _html_signoff(self, data: Dict[str, Any]) -> str:
        """Render HTML stakeholder sign-off table."""
        signoffs = data.get("stakeholder_signoff", [])
        if not signoffs:
            return ""
        rows = ""
        for s in signoffs:
            name = s.get("name", "")
            role = s.get("role", "")
            decision = s.get("decision", "Pending")
            date = s.get("date", "-")
            rows += f"<tr><td>{name}</td><td>{role}</td><td><strong>{decision}</strong></td><td>{date}</td></tr>\n"
        return (
            '<div class="section">\n<h2>5. Stakeholder Sign-off</h2>\n'
            "<table><thead><tr><th>Stakeholder</th><th>Role</th>"
            "<th>Decision</th><th>Date</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-045 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
