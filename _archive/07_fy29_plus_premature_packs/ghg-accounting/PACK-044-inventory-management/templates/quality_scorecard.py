# -*- coding: utf-8 -*-
"""
QualityScorecard - QA/QC Quality Scorecard for PACK-044.

Generates a quality scorecard covering QA/QC pass/fail rates, quality
scores by scope and category, data quality indicator breakdowns,
improvement recommendations, and trend over time.

Sections:
    1. Quality Overview
    2. QA/QC Pass/Fail Summary
    3. Quality by Scope/Category
    4. Data Quality Indicators (DQI)
    5. Improvement Recommendations

Author: GreenLang Team
Version: 44.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "44.0.0"


class QualityScorecard:
    """
    QA/QC quality scorecard template for GHG inventory management.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = QualityScorecard()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize QualityScorecard."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render quality scorecard as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_pass_fail(data),
            self._md_quality_by_scope(data),
            self._md_dqi(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render quality scorecard as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overview(data),
            self._html_pass_fail(data),
            self._html_quality_by_scope(data),
            self._html_dqi(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render quality scorecard as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        return {
            "template": "quality_scorecard",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_provenance(data),
            "company_name": self._get_val(data, "company_name", ""),
            "overall_quality_score": data.get("overall_quality_score", 0.0),
            "qa_qc_summary": data.get("qa_qc_summary", {}),
            "quality_by_scope": data.get("quality_by_scope", []),
            "dqi_breakdown": data.get("dqi_breakdown", []),
            "recommendations": data.get("recommendations", []),
        }

    # ==================================================================
    # MARKDOWN
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        score = data.get("overall_quality_score", 0.0)
        return (
            f"# Quality Scorecard - {company}\n\n"
            f"**Overall Quality Score:** {score:.1f}/100 | "
            f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown quality overview."""
        summary = data.get("qa_qc_summary", {})
        if not summary:
            return "## 1. Quality Overview\n\nNo quality data available."
        total_checks = summary.get("total_checks", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        rate = (passed / total_checks * 100) if total_checks > 0 else 0.0
        return (
            "## 1. Quality Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total QA/QC Checks | {total_checks} |\n"
            f"| Passed | {passed} |\n"
            f"| Failed | {failed} |\n"
            f"| Pass Rate | {rate:.1f}% |"
        )

    def _md_pass_fail(self, data: Dict[str, Any]) -> str:
        """Render Markdown QA/QC pass/fail details."""
        checks = data.get("qa_qc_checks", [])
        if not checks:
            return ""
        lines = [
            "## 2. QA/QC Check Details",
            "",
            "| Check Name | Category | Result | Severity | Details |",
            "|-----------|----------|--------|---------|---------|",
        ]
        for check in checks:
            name = check.get("check_name", "")
            cat = check.get("category", "")
            result = check.get("result", "pass")
            sev = check.get("severity", "info")
            details = check.get("details", "-")
            lines.append(f"| {name} | {cat} | **{result}** | {sev} | {details} |")
        return "\n".join(lines)

    def _md_quality_by_scope(self, data: Dict[str, Any]) -> str:
        """Render Markdown quality by scope."""
        scopes = data.get("quality_by_scope", [])
        if not scopes:
            return ""
        lines = [
            "## 3. Quality by Scope/Category",
            "",
            "| Scope | Category | Score | Completeness | Accuracy | Consistency |",
            "|-------|---------|-------|-------------|----------|-------------|",
        ]
        for s in scopes:
            lines.append(
                f"| {s.get('scope', '')} | {s.get('category', '')} | "
                f"{s.get('score', 0):.0f} | {s.get('completeness_pct', 0):.0f}% | "
                f"{s.get('accuracy_pct', 0):.0f}% | {s.get('consistency_pct', 0):.0f}% |"
            )
        return "\n".join(lines)

    def _md_dqi(self, data: Dict[str, Any]) -> str:
        """Render Markdown DQI breakdown."""
        dqi = data.get("dqi_breakdown", [])
        if not dqi:
            return ""
        lines = [
            "## 4. Data Quality Indicators (DQI)",
            "",
            "| Indicator | Score | Weight | Weighted Score |",
            "|-----------|-------|--------|---------------|",
        ]
        for d in dqi:
            name = d.get("indicator", "")
            score = d.get("score", 0.0)
            weight = d.get("weight", 0.0)
            weighted = score * weight
            lines.append(f"| {name} | {score:.1f} | {weight:.2f} | {weighted:.2f} |")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render Markdown improvement recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        lines = ["## 5. Improvement Recommendations", ""]
        for i, rec in enumerate(recs, 1):
            priority = rec.get("priority", "medium")
            desc = rec.get("description", "")
            impact = rec.get("expected_impact", "")
            lines.append(f"{i}. **[{priority.upper()}]** {desc}")
            if impact:
                lines.append(f"   - Expected Impact: {impact}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f'<meta charset="UTF-8"><title>Quality Scorecard - {company}</title>\n'
            "<style>body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1200px;color:#1a1a2e;line-height:1.6;}"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;}h2{color:#1b263b;margin-top:2rem;}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}"
            "th{background:#f0f4f8;font-weight:600;}tr:nth-child(even){background:#fafbfc;}"
            ".fail{color:#e63946;font-weight:700;}.pass{color:#2a9d8f;font-weight:700;}"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}"
            "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        score = data.get("overall_quality_score", 0.0)
        return f'<div><h1>Quality Scorecard &mdash; {company}</h1><p><strong>Score:</strong> {score:.1f}/100</p><hr></div>'

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML quality overview."""
        summary = data.get("qa_qc_summary", {})
        if not summary:
            return ""
        total = summary.get("total_checks", 0)
        passed = summary.get("passed", 0)
        rate = (passed / total * 100) if total > 0 else 0.0
        return f'<div><h2>1. Overview</h2><p>Checks: {total} | Passed: {passed} | Rate: {rate:.1f}%</p></div>'

    def _html_pass_fail(self, data: Dict[str, Any]) -> str:
        """Render HTML pass/fail details."""
        checks = data.get("qa_qc_checks", [])
        if not checks:
            return ""
        rows = ""
        for c in checks:
            result = c.get("result", "pass")
            css = "pass" if result == "pass" else "fail"
            rows += (
                f'<tr><td>{c.get("check_name", "")}</td><td>{c.get("category", "")}</td>'
                f'<td class="{css}">{result}</td><td>{c.get("severity", "info")}</td></tr>\n'
            )
        return (
            '<div><h2>2. QA/QC Checks</h2>\n'
            "<table><thead><tr><th>Check</th><th>Category</th><th>Result</th><th>Severity</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_quality_by_scope(self, data: Dict[str, Any]) -> str:
        """Render HTML quality by scope."""
        scopes = data.get("quality_by_scope", [])
        if not scopes:
            return ""
        rows = ""
        for s in scopes:
            rows += (
                f"<tr><td>{s.get('scope', '')}</td><td>{s.get('category', '')}</td>"
                f"<td>{s.get('score', 0):.0f}</td><td>{s.get('completeness_pct', 0):.0f}%</td></tr>\n"
            )
        return (
            '<div><h2>3. Quality by Scope</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Category</th><th>Score</th><th>Completeness</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_dqi(self, data: Dict[str, Any]) -> str:
        """Render HTML DQI breakdown."""
        dqi = data.get("dqi_breakdown", [])
        if not dqi:
            return ""
        rows = ""
        for d in dqi:
            rows += f"<tr><td>{d.get('indicator', '')}</td><td>{d.get('score', 0):.1f}</td><td>{d.get('weight', 0):.2f}</td></tr>\n"
        return (
            '<div><h2>4. DQI Breakdown</h2>\n'
            "<table><thead><tr><th>Indicator</th><th>Score</th><th>Weight</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        li = ""
        for rec in recs:
            li += f"<li><strong>[{rec.get('priority', 'medium').upper()}]</strong> {rec.get('description', '')}</li>\n"
        return f'<div><h2>5. Recommendations</h2><ul>{li}</ul></div>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div style="font-size:0.85rem;color:#666;"><hr>'
            f"<p>Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}</p>"
            f'<p class="provenance">Provenance Hash: {provenance}</p></div>'
        )
