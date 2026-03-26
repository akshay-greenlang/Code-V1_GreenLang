# -*- coding: utf-8 -*-
"""
PolicyComplianceReport - Policy Settings and Framework Compliance for PACK-045.

Generates a policy compliance report covering base year policy configuration,
framework compliance mapping (GHG Protocol, ISO 14064, SBTi, CSRD),
gap analysis of policy vs requirements, and remediation recommendations.

Sections:
    1. Policy Configuration
    2. Framework Compliance Matrix
    3. Gap Analysis
    4. Remediation Recommendations
    5. Policy Change History

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


def _compliance_badge(status: str) -> str:
    """Return formatted compliance status."""
    mapping = {"compliant": "PASS", "partial": "PARTIAL", "non_compliant": "FAIL"}
    return mapping.get(status.lower(), status.upper())


def _compliance_css(status: str) -> str:
    """Return CSS class for compliance status."""
    mapping = {
        "compliant": "status-pass",
        "partial": "status-partial",
        "non_compliant": "status-fail",
    }
    return mapping.get(status.lower(), "status-partial")


class PolicyComplianceReport:
    """
    Policy compliance report template.

    Renders base year policy settings, multi-framework compliance matrices,
    gap analysis, and remediation recommendations. All outputs include
    SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = PolicyComplianceReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PolicyComplianceReport."""
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
        """Render policy compliance report as Markdown."""
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_policy_config(data),
            self._md_compliance_matrix(data),
            self._md_gap_analysis(data),
            self._md_remediation(data),
            self._md_change_history(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render policy compliance report as HTML."""
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_policy_config(data),
            self._html_compliance_matrix(data),
            self._html_gap_analysis(data),
            self._html_remediation(data),
            self._html_change_history(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render policy compliance report as JSON-serializable dict."""
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "policy_compliance_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "base_year": self._get_val(data, "base_year", ""),
            "policy_settings": data.get("policy_settings", {}),
            "framework_compliance": data.get("framework_compliance", []),
            "gaps": data.get("gaps", []),
            "remediation": data.get("remediation", []),
            "change_history": data.get("change_history", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        base_year = self._get_val(data, "base_year", "")
        return (
            f"# Base Year Policy Compliance Report - {company}\n\n"
            f"**Base Year:** {base_year} | "
            f"**Report Date:** {datetime.utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_policy_config(self, data: Dict[str, Any]) -> str:
        """Render Markdown policy configuration."""
        policy = data.get("policy_settings", {})
        if not policy:
            return "## 1. Policy Configuration\n\nNo policy settings defined."
        lines = [
            "## 1. Policy Configuration",
            "",
            f"- **Recalculation Policy:** {policy.get('recalculation_policy', 'Not set')}",
            f"- **Significance Threshold:** {policy.get('significance_threshold_pct', 5)}%",
            f"- **Consolidation Approach:** {policy.get('consolidation_approach', '')}",
            f"- **Base Year Lock:** {'Enabled' if policy.get('base_year_locked') else 'Disabled'}",
            f"- **Review Frequency:** {policy.get('review_frequency', 'Annual')}",
            f"- **Approval Levels:** {policy.get('approval_levels', 2)}",
        ]
        frameworks = policy.get("frameworks", [])
        if frameworks:
            lines.append(f"- **Target Frameworks:** {', '.join(frameworks)}")
        return "\n".join(lines)

    def _md_compliance_matrix(self, data: Dict[str, Any]) -> str:
        """Render Markdown framework compliance matrix."""
        compliance = data.get("framework_compliance", [])
        if not compliance:
            return "## 2. Framework Compliance\n\nNo compliance data available."
        lines = [
            "## 2. Framework Compliance Matrix",
            "",
            "| Framework | Requirement | Status | Details |",
            "|-----------|-----------|--------|---------|",
        ]
        for c in compliance:
            framework = c.get("framework", "")
            req = c.get("requirement", "")
            status = _compliance_badge(c.get("status", "non_compliant"))
            details = c.get("details", "")
            lines.append(f"| {framework} | {req} | **{status}** | {details} |")
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown gap analysis."""
        gaps = data.get("gaps", [])
        if not gaps:
            return "## 3. Gap Analysis\n\nNo gaps identified."
        lines = [
            "## 3. Gap Analysis",
            "",
            "| # | Framework | Gap Description | Severity | Impact |",
            "|---|----------|----------------|----------|--------|",
        ]
        for i, g in enumerate(gaps, 1):
            framework = g.get("framework", "")
            desc = g.get("description", "")
            severity = g.get("severity", "medium").upper()
            impact = g.get("impact", "")
            lines.append(f"| {i} | {framework} | {desc} | **{severity}** | {impact} |")
        return "\n".join(lines)

    def _md_remediation(self, data: Dict[str, Any]) -> str:
        """Render Markdown remediation recommendations."""
        recs = data.get("remediation", [])
        if not recs:
            return ""
        lines = ["## 4. Remediation Recommendations", ""]
        for r in recs:
            priority = r.get("priority", "medium").upper()
            action = r.get("action", "")
            effort = r.get("effort", "")
            timeline = r.get("timeline", "")
            lines.append(f"- **[{priority}]** {action}")
            if effort or timeline:
                lines.append(f"  - Effort: {effort} | Timeline: {timeline}")
        return "\n".join(lines)

    def _md_change_history(self, data: Dict[str, Any]) -> str:
        """Render Markdown policy change history."""
        history = data.get("change_history", [])
        if not history:
            return ""
        lines = [
            "## 5. Policy Change History",
            "",
            "| Date | Changed By | Setting | Old Value | New Value | Reason |",
            "|------|-----------|---------|----------|----------|--------|",
        ]
        for h in history:
            date = h.get("date", "")
            user = h.get("changed_by", "")
            setting = h.get("setting", "")
            old_val = h.get("old_value", "")
            new_val = h.get("new_value", "")
            reason = h.get("reason", "")
            lines.append(f"| {date} | {user} | {setting} | {old_val} | {new_val} | {reason} |")
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
            f"<title>Policy Compliance - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #264653;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".status-pass{color:#2a9d8f;font-weight:700;}\n"
            ".status-partial{color:#e9c46a;font-weight:700;}\n"
            ".status-fail{color:#e76f51;font-weight:700;}\n"
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
        return (
            '<div class="section">\n'
            f"<h1>Policy Compliance Report &mdash; {company}</h1>\n"
            f"<p><strong>Base Year:</strong> {base_year}</p>\n<hr>\n</div>"
        )

    def _html_policy_config(self, data: Dict[str, Any]) -> str:
        """Render HTML policy configuration."""
        policy = data.get("policy_settings", {})
        if not policy:
            return ""
        return (
            '<div class="section">\n<h2>1. Policy Configuration</h2>\n'
            f"<p><strong>Recalculation:</strong> {policy.get('recalculation_policy', 'N/A')}</p>\n"
            f"<p><strong>Threshold:</strong> {policy.get('significance_threshold_pct', 5)}%</p>\n"
            f"<p><strong>Consolidation:</strong> {policy.get('consolidation_approach', '')}</p>\n"
            f"<p><strong>Review:</strong> {policy.get('review_frequency', 'Annual')}</p>\n</div>"
        )

    def _html_compliance_matrix(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance matrix table."""
        compliance = data.get("framework_compliance", [])
        if not compliance:
            return ""
        rows = ""
        for c in compliance:
            framework = c.get("framework", "")
            req = c.get("requirement", "")
            status = c.get("status", "non_compliant")
            css = _compliance_css(status)
            label = _compliance_badge(status)
            rows += (
                f'<tr><td>{framework}</td><td>{req}</td>'
                f'<td class="{css}"><strong>{label}</strong></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>2. Compliance Matrix</h2>\n'
            "<table><thead><tr><th>Framework</th><th>Requirement</th>"
            "<th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap analysis table."""
        gaps = data.get("gaps", [])
        if not gaps:
            return ""
        rows = ""
        for i, g in enumerate(gaps, 1):
            framework = g.get("framework", "")
            desc = g.get("description", "")
            severity = g.get("severity", "medium").upper()
            rows += f"<tr><td>{i}</td><td>{framework}</td><td>{desc}</td><td><strong>{severity}</strong></td></tr>\n"
        return (
            '<div class="section">\n<h2>3. Gap Analysis</h2>\n'
            "<table><thead><tr><th>#</th><th>Framework</th>"
            "<th>Gap</th><th>Severity</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_remediation(self, data: Dict[str, Any]) -> str:
        """Render HTML remediation recommendations."""
        recs = data.get("remediation", [])
        if not recs:
            return ""
        items = ""
        for r in recs:
            priority = r.get("priority", "medium").upper()
            action = r.get("action", "")
            items += f"<li><strong>[{priority}]</strong> {action}</li>\n"
        return f'<div class="section">\n<h2>4. Remediation</h2>\n<ul>{items}</ul>\n</div>'

    def _html_change_history(self, data: Dict[str, Any]) -> str:
        """Render HTML policy change history table."""
        history = data.get("change_history", [])
        if not history:
            return ""
        rows = ""
        for h in history:
            date = h.get("date", "")
            user = h.get("changed_by", "")
            setting = h.get("setting", "")
            old_val = h.get("old_value", "")
            new_val = h.get("new_value", "")
            rows += f"<tr><td>{date}</td><td>{user}</td><td>{setting}</td><td>{old_val}</td><td>{new_val}</td></tr>\n"
        return (
            '<div class="section">\n<h2>5. Change History</h2>\n'
            "<table><thead><tr><th>Date</th><th>User</th><th>Setting</th>"
            "<th>Old</th><th>New</th></tr></thead>\n"
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
