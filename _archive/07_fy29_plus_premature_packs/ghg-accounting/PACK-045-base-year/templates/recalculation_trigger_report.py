# -*- coding: utf-8 -*-
"""
RecalculationTriggerReport - Detected Triggers and Significance for PACK-045.

Generates a recalculation trigger report covering detected trigger events,
significance assessment results, threshold analysis, action recommendations,
and trigger history tracking.

Sections:
    1. Trigger Summary Dashboard
    2. Detected Triggers Detail
    3. Significance Assessment Results
    4. Threshold Analysis
    5. Action Recommendations

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


def _severity_badge(severity: str) -> str:
    """Return formatted severity text."""
    return severity.upper()


def _severity_css(severity: str) -> str:
    """Return CSS class for trigger severity."""
    mapping = {
        "critical": "severity-critical",
        "high": "severity-high",
        "medium": "severity-medium",
        "low": "severity-low",
    }
    return mapping.get(severity.lower(), "severity-low")


class RecalculationTriggerReport:
    """
    Recalculation trigger report template.

    Renders detected recalculation triggers, significance test results,
    threshold analysis against configured materiality, and recommended
    actions. All outputs include SHA-256 provenance hashing for audit
    trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = RecalculationTriggerReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RecalculationTriggerReport."""
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
        """Render recalculation trigger report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_summary_dashboard(data),
            self._md_trigger_details(data),
            self._md_significance_results(data),
            self._md_threshold_analysis(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render recalculation trigger report as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_summary_dashboard(data),
            self._html_trigger_details(data),
            self._html_significance_results(data),
            self._html_threshold_analysis(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render recalculation trigger report as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "recalculation_trigger_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "base_year": self._get_val(data, "base_year", ""),
            "assessment_date": self._get_val(data, "assessment_date", ""),
            "total_triggers": data.get("total_triggers", 0),
            "significant_triggers": data.get("significant_triggers", 0),
            "triggers": data.get("triggers", []),
            "significance_results": data.get("significance_results", []),
            "threshold_config": data.get("threshold_config", {}),
            "recommendations": data.get("recommendations", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        total = data.get("total_triggers", 0)
        significant = data.get("significant_triggers", 0)
        return (
            f"# Recalculation Trigger Report - {company}\n\n"
            f"**Base Year:** {base_year} | "
            f"**Triggers Detected:** {total} | "
            f"**Significant:** {significant} | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_summary_dashboard(self, data: Dict[str, Any]) -> str:
        """Render Markdown trigger summary dashboard."""
        triggers = data.get("triggers", [])
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for t in triggers:
            ttype = t.get("trigger_type", "other")
            sev = t.get("severity", "low")
            by_type[ttype] = by_type.get(ttype, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1
        lines = [
            "## 1. Trigger Summary",
            "",
            "**By Type:**",
        ]
        for k, v in sorted(by_type.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("**By Severity:**")
        for k, v in sorted(by_severity.items()):
            lines.append(f"- {_severity_badge(k)}: {v}")
        return "\n".join(lines)

    def _md_trigger_details(self, data: Dict[str, Any]) -> str:
        """Render Markdown trigger details."""
        triggers = data.get("triggers", [])
        if not triggers:
            return "## 2. Detected Triggers\n\nNo triggers detected."
        lines = [
            "## 2. Detected Triggers",
            "",
            "| # | Type | Description | Severity | Date Detected | Impact (tCO2e) |",
            "|---|------|------------|----------|--------------|---------------|",
        ]
        for i, t in enumerate(triggers, 1):
            ttype = t.get("trigger_type", "")
            desc = t.get("description", "")
            sev = _severity_badge(t.get("severity", "low"))
            date = t.get("date_detected", "-")
            impact = t.get("estimated_impact_tco2e", 0)
            lines.append(f"| {i} | {ttype} | {desc} | **{sev}** | {date} | {impact:,.1f} |")
        return "\n".join(lines)

    def _md_significance_results(self, data: Dict[str, Any]) -> str:
        """Render Markdown significance assessment results."""
        results = data.get("significance_results", [])
        if not results:
            return "## 3. Significance Assessment\n\nNo assessments completed."
        lines = [
            "## 3. Significance Assessment Results",
            "",
            "| Trigger | Test Applied | Threshold | Actual Impact | Significant? |",
            "|---------|------------|-----------|--------------|-------------|",
        ]
        for r in results:
            name = r.get("trigger_name", "")
            test = r.get("test_applied", "")
            threshold = r.get("threshold_pct", 0)
            actual = r.get("actual_impact_pct", 0)
            sig = "YES" if r.get("is_significant") else "No"
            lines.append(
                f"| {name} | {test} | {threshold:.1f}% | "
                f"{actual:.1f}% | **{sig}** |"
            )
        return "\n".join(lines)

    def _md_threshold_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown threshold analysis."""
        config = data.get("threshold_config", {})
        if not config:
            return ""
        lines = [
            "## 4. Threshold Configuration",
            "",
            f"- **Materiality Threshold:** {config.get('materiality_pct', 5)}%",
            f"- **Cumulative Threshold:** {config.get('cumulative_pct', 10)}%",
            f"- **Framework:** {config.get('framework', 'GHG Protocol')}",
            f"- **Auto-Recalculation:** {'Enabled' if config.get('auto_recalculate') else 'Disabled'}",
        ]
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render Markdown action recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        lines = ["## 5. Action Recommendations", ""]
        for r in recs:
            priority = r.get("priority", "medium").upper()
            action = r.get("action", "")
            rationale = r.get("rationale", "")
            lines.append(f"- **[{priority}]** {action}")
            if rationale:
                lines.append(f"  - Rationale: {rationale}")
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
            f"<title>Recalculation Triggers - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #e76f51;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".severity-critical{color:#d00;font-weight:700;}\n"
            ".severity-high{color:#e76f51;font-weight:700;}\n"
            ".severity-medium{color:#e9c46a;font-weight:700;}\n"
            ".severity-low{color:#2a9d8f;font-weight:700;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        total = data.get("total_triggers", 0)
        significant = data.get("significant_triggers", 0)
        return (
            '<div class="section">\n'
            f"<h1>Recalculation Trigger Report &mdash; {company}</h1>\n"
            f"<p><strong>Base Year:</strong> {base_year} | "
            f"<strong>Triggers:</strong> {total} | "
            f"<strong>Significant:</strong> {significant}</p>\n<hr>\n</div>"
        )

    def _html_summary_dashboard(self, data: Dict[str, Any]) -> str:
        """Render HTML trigger summary."""
        triggers = data.get("triggers", [])
        if not triggers:
            return ""
        by_severity: Dict[str, int] = {}
        for t in triggers:
            sev = t.get("severity", "low")
            by_severity[sev] = by_severity.get(sev, 0) + 1
        cards = ""
        for sev, count in sorted(by_severity.items()):
            css = _severity_css(sev)
            cards += f'<span class="{css}" style="margin-right:1.5rem;">{sev.upper()}: {count}</span>'
        return f'<div class="section">\n<h2>1. Trigger Summary</h2>\n<p>{cards}</p>\n</div>'

    def _html_trigger_details(self, data: Dict[str, Any]) -> str:
        """Render HTML trigger details table."""
        triggers = data.get("triggers", [])
        if not triggers:
            return ""
        rows = ""
        for i, t in enumerate(triggers, 1):
            ttype = t.get("trigger_type", "")
            desc = t.get("description", "")
            sev = t.get("severity", "low")
            css = _severity_css(sev)
            impact = t.get("estimated_impact_tco2e", 0)
            rows += (
                f'<tr><td>{i}</td><td>{ttype}</td><td>{desc}</td>'
                f'<td class="{css}">{sev.upper()}</td><td>{impact:,.1f}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>2. Detected Triggers</h2>\n'
            "<table><thead><tr><th>#</th><th>Type</th><th>Description</th>"
            "<th>Severity</th><th>Impact tCO2e</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_significance_results(self, data: Dict[str, Any]) -> str:
        """Render HTML significance assessment results."""
        results = data.get("significance_results", [])
        if not results:
            return ""
        rows = ""
        for r in results:
            name = r.get("trigger_name", "")
            test = r.get("test_applied", "")
            threshold = r.get("threshold_pct", 0)
            actual = r.get("actual_impact_pct", 0)
            sig = r.get("is_significant", False)
            css = "severity-critical" if sig else "severity-low"
            label = "YES" if sig else "No"
            rows += (
                f"<tr><td>{name}</td><td>{test}</td><td>{threshold:.1f}%</td>"
                f'<td>{actual:.1f}%</td><td class="{css}"><strong>{label}</strong></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>3. Significance Assessment</h2>\n'
            "<table><thead><tr><th>Trigger</th><th>Test</th><th>Threshold</th>"
            "<th>Actual</th><th>Significant?</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_threshold_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML threshold analysis."""
        config = data.get("threshold_config", {})
        if not config:
            return ""
        return (
            '<div class="section">\n<h2>4. Threshold Configuration</h2>\n'
            f"<p><strong>Materiality:</strong> {config.get('materiality_pct', 5)}%</p>\n"
            f"<p><strong>Cumulative:</strong> {config.get('cumulative_pct', 10)}%</p>\n"
            f"<p><strong>Framework:</strong> {config.get('framework', 'GHG Protocol')}</p>\n</div>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML action recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        items = ""
        for r in recs:
            priority = r.get("priority", "medium").upper()
            action = r.get("action", "")
            items += f"<li><strong>[{priority}]</strong> {action}</li>\n"
        return f'<div class="section">\n<h2>5. Recommendations</h2>\n<ul>{items}</ul>\n</div>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-045 v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )
