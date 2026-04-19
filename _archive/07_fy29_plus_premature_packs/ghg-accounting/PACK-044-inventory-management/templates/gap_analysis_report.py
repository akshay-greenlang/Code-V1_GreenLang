# -*- coding: utf-8 -*-
"""
GapAnalysisReport - Gap Identification and Recommendations for PACK-044.

Generates a gap analysis report identifying data gaps, methodology gaps,
coverage gaps, and quality gaps with prioritized recommendations.

Sections:
    1. Gap Summary
    2. Data Gaps
    3. Methodology Gaps
    4. Coverage Gaps
    5. Recommendations

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


class GapAnalysisReport:
    """
    Gap analysis report template for GHG inventory management.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GapAnalysisReport."""
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
        """Render gap analysis as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_summary(data),
            self._md_data_gaps(data),
            self._md_methodology_gaps(data),
            self._md_coverage_gaps(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render gap analysis as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_summary(data),
            self._html_data_gaps(data),
            self._html_methodology_gaps(data),
            self._html_coverage_gaps(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render gap analysis as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        return {
            "template": "gap_analysis_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": self._compute_provenance(data),
            "company_name": self._get_val(data, "company_name", ""),
            "gap_summary": data.get("gap_summary", {}),
            "data_gaps": data.get("data_gaps", []),
            "methodology_gaps": data.get("methodology_gaps", []),
            "coverage_gaps": data.get("coverage_gaps", []),
            "recommendations": data.get("recommendations", []),
        }

    def _md_header(self, data: Dict[str, Any]) -> str:
        company = self._get_val(data, "company_name", "Organization")
        return f"# Gap Analysis Report - {company}\n\n**Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n---"

    def _md_summary(self, data: Dict[str, Any]) -> str:
        summary = data.get("gap_summary", {})
        if not summary:
            return "## 1. Gap Summary\n\nNo gaps identified."
        return (
            "## 1. Gap Summary\n\n"
            f"| Gap Type | Count | Critical | High | Medium | Low |\n"
            f"|---------|-------|----------|------|--------|-----|\n"
            f"| Data | {summary.get('data_gap_count', 0)} | {summary.get('data_critical', 0)} | {summary.get('data_high', 0)} | {summary.get('data_medium', 0)} | {summary.get('data_low', 0)} |\n"
            f"| Methodology | {summary.get('methodology_gap_count', 0)} | {summary.get('meth_critical', 0)} | {summary.get('meth_high', 0)} | {summary.get('meth_medium', 0)} | {summary.get('meth_low', 0)} |\n"
            f"| Coverage | {summary.get('coverage_gap_count', 0)} | {summary.get('cov_critical', 0)} | {summary.get('cov_high', 0)} | {summary.get('cov_medium', 0)} | {summary.get('cov_low', 0)} |"
        )

    def _md_data_gaps(self, data: Dict[str, Any]) -> str:
        gaps = data.get("data_gaps", [])
        if not gaps:
            return ""
        lines = [
            "## 2. Data Gaps", "",
            "| Scope | Category | Facility | Gap Description | Severity | Est. Impact (tCO2e) |",
            "|-------|---------|----------|----------------|---------|-------------------|",
        ]
        for g in gaps:
            lines.append(
                f"| {g.get('scope', '')} | {g.get('category', '')} | {g.get('facility', '-')} | "
                f"{g.get('description', '')} | **{g.get('severity', '')}** | {g.get('estimated_impact_tco2e', 0):,.0f} |"
            )
        return "\n".join(lines)

    def _md_methodology_gaps(self, data: Dict[str, Any]) -> str:
        gaps = data.get("methodology_gaps", [])
        if not gaps:
            return ""
        lines = [
            "## 3. Methodology Gaps", "",
            "| Category | Current Approach | Recommended | Severity | Compliance Risk |",
            "|---------|----------------|-------------|---------|----------------|",
        ]
        for g in gaps:
            lines.append(
                f"| {g.get('category', '')} | {g.get('current_approach', '')} | "
                f"{g.get('recommended', '')} | **{g.get('severity', '')}** | {g.get('compliance_risk', '-')} |"
            )
        return "\n".join(lines)

    def _md_coverage_gaps(self, data: Dict[str, Any]) -> str:
        gaps = data.get("coverage_gaps", [])
        if not gaps:
            return ""
        lines = [
            "## 4. Coverage Gaps", "",
            "| Scope | Missing Category | Materiality | Action Required |",
            "|-------|----------------|------------|----------------|",
        ]
        for g in gaps:
            lines.append(
                f"| {g.get('scope', '')} | {g.get('missing_category', '')} | "
                f"{g.get('materiality', '')} | {g.get('action_required', '')} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        lines = ["## 5. Prioritized Recommendations", ""]
        for i, rec in enumerate(recs, 1):
            lines.append(
                f"{i}. **[{rec.get('priority', 'medium').upper()}]** {rec.get('recommendation', '')}\n"
                f"   - Gap: {rec.get('gap_reference', '-')} | Effort: {rec.get('effort', '-')} | Timeline: {rec.get('timeline', '-')}"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return f"---\n\n*Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}*\n*Provenance Hash: `{provenance}`*"

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f'<meta charset="UTF-8"><title>Gap Analysis - {company}</title>\n'
            "<style>body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;max-width:1200px;color:#1a1a2e;line-height:1.6;}"
            "h1{color:#0d1b2a;border-bottom:3px solid #e76f51;}h2{color:#1b263b;margin-top:2rem;}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}"
            "th{background:#f0f4f8;font-weight:600;}tr:nth-child(even){background:#fafbfc;}"
            ".critical{color:#e63946;font-weight:700;}.high{color:#e76f51;font-weight:600;}"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}"
            "</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        company = self._get_val(data, "company_name", "Organization")
        return f'<div><h1>Gap Analysis &mdash; {company}</h1><hr></div>'

    def _html_summary(self, data: Dict[str, Any]) -> str:
        summary = data.get("gap_summary", {})
        if not summary:
            return ""
        total = summary.get("data_gap_count", 0) + summary.get("methodology_gap_count", 0) + summary.get("coverage_gap_count", 0)
        return f'<div><h2>1. Summary</h2><p>Total gaps identified: {total}</p></div>'

    def _html_data_gaps(self, data: Dict[str, Any]) -> str:
        gaps = data.get("data_gaps", [])
        if not gaps:
            return ""
        rows = ""
        for g in gaps:
            sev = g.get("severity", "")
            css = "critical" if sev == "critical" else ("high" if sev == "high" else "")
            rows += f'<tr><td>{g.get("scope", "")}</td><td>{g.get("category", "")}</td><td>{g.get("description", "")}</td><td class="{css}">{sev}</td></tr>\n'
        return (
            '<div><h2>2. Data Gaps</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Category</th><th>Description</th><th>Severity</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_methodology_gaps(self, data: Dict[str, Any]) -> str:
        gaps = data.get("methodology_gaps", [])
        if not gaps:
            return ""
        rows = ""
        for g in gaps:
            rows += f'<tr><td>{g.get("category", "")}</td><td>{g.get("current_approach", "")}</td><td>{g.get("recommended", "")}</td></tr>\n'
        return (
            '<div><h2>3. Methodology Gaps</h2>\n'
            "<table><thead><tr><th>Category</th><th>Current</th><th>Recommended</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_coverage_gaps(self, data: Dict[str, Any]) -> str:
        gaps = data.get("coverage_gaps", [])
        if not gaps:
            return ""
        rows = ""
        for g in gaps:
            rows += f'<tr><td>{g.get("scope", "")}</td><td>{g.get("missing_category", "")}</td><td>{g.get("materiality", "")}</td></tr>\n'
        return (
            '<div><h2>4. Coverage Gaps</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Missing</th><th>Materiality</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        li = ""
        for rec in recs:
            li += f"<li><strong>[{rec.get('priority', 'medium').upper()}]</strong> {rec.get('recommendation', '')}</li>\n"
        return f'<div><h2>5. Recommendations</h2><ol>{li}</ol></div>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div style="font-size:0.85rem;color:#666;"><hr>'
            f"<p>Generated by GreenLang PACK-044 v{_MODULE_VERSION} | {ts}</p>"
            f'<p class="provenance">Provenance Hash: {provenance}</p></div>'
        )
