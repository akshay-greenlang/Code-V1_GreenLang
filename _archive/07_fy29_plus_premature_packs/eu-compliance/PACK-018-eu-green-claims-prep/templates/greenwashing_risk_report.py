# -*- coding: utf-8 -*-
"""
GreenwashingRiskReportTemplate - EU Green Claims Greenwashing Risk Assessment

Assesses greenwashing risk across all environmental claims using the
TerraChoice Seven Sins of Greenwashing framework and the UCPD Blacklist
cross-reference. Produces a claim-level risk matrix, trend analysis, and
prioritised remediation actions to reduce regulatory exposure.

Sections:
    1. Executive Summary - Overall greenwashing risk posture
    2. Risk Overview - Aggregate risk distribution
    3. Seven Sins Analysis - TerraChoice framework evaluation
    4. UCPD Blacklist Check - Unfair Commercial Practices Directive review
    5. Claim Risk Matrix - Per-claim risk ratings and heatmap data
    6. Trend Analysis - Risk trend over reporting periods
    7. Remediation Actions - Prioritised corrective measures
    8. Provenance - Data lineage and hash chain

PACK Reference: PACK-018 EU Green Claims Prep Pack
Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

__all__ = ["GreenwashingRiskReportTemplate"]

_SECTIONS: List[Dict[str, Any]] = [
    {"id": "executive_summary", "title": "Executive Summary", "order": 1},
    {"id": "risk_overview", "title": "Risk Overview", "order": 2},
    {"id": "seven_sins_analysis", "title": "Seven Sins Analysis", "order": 3},
    {"id": "ucpd_blacklist_check", "title": "UCPD Blacklist Check", "order": 4},
    {"id": "claim_risk_matrix", "title": "Claim Risk Matrix", "order": 5},
    {"id": "trend_analysis", "title": "Trend Analysis", "order": 6},
    {"id": "remediation_actions", "title": "Remediation Actions", "order": 7},
    {"id": "provenance", "title": "Provenance", "order": 8},
]

_SEVEN_SINS = [
    "Hidden Trade-off",
    "No Proof",
    "Vagueness",
    "Irrelevance",
    "Lesser of Two Evils",
    "Fibbing",
    "Worshipping False Labels",
]

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class GreenwashingRiskReportTemplate:
    """
    EU Green Claims Directive - Greenwashing Risk Assessment Report.

    Evaluates all environmental claims against the TerraChoice Seven Sins
    of Greenwashing framework and cross-references with the UCPD Blacklist.
    Produces a claim-level risk matrix, tracks risk trends across periods,
    and delivers prioritised remediation actions to reduce greenwashing
    exposure under the EU Green Claims Directive.

    Example:
        >>> tpl = GreenwashingRiskReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GreenwashingRiskReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render greenwashing risk report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_risk_overview(data),
            self._md_seven_sins_analysis(data),
            self._md_ucpd_blacklist_check(data),
            self._md_claim_risk_matrix(data),
            self._md_trend_analysis(data),
            self._md_remediation_actions(data),
            self._md_provenance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render greenwashing risk report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_risk_overview(data),
            self._html_seven_sins(data),
            self._html_claim_risk_matrix(data),
            self._html_remediation_actions(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Greenwashing Risk Report - EU Green Claims</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render greenwashing risk report as structured JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "greenwashing_risk_report",
            "directive_reference": "EU Green Claims Directive 2023/0085",
            "frameworks": ["TerraChoice Seven Sins", "UCPD 2005/29/EC"],
            "version": "18.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_period": data.get("reporting_period", ""),
            "executive_summary": self._section_executive_summary(data),
            "risk_overview": self._section_risk_overview(data),
            "seven_sins_analysis": self._section_seven_sins(data),
            "ucpd_blacklist_check": self._section_ucpd_blacklist(data),
            "claim_risk_matrix": self._section_claim_risk_matrix(data),
            "trend_analysis": self._section_trend_analysis(data),
            "remediation_actions": self._section_remediation_actions(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def get_sections(self) -> List[Dict[str, Any]]:
        """Return list of available section definitions."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if not data.get("claims"):
            errors.append("claims list is required")
        if not data.get("reporting_period"):
            warnings.append("reporting_period missing; will default to empty")
        if not data.get("trend_data"):
            warnings.append("trend_data missing; trend analysis will be limited")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Section builders (dict)
    # ------------------------------------------------------------------

    def _section_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary section."""
        claims = data.get("claims", [])
        total = len(claims)
        high = sum(1 for c in claims if c.get("risk_level", "") == "high")
        medium = sum(1 for c in claims if c.get("risk_level", "") == "medium")
        low = total - high - medium
        avg_score = (
            round(sum(c.get("risk_score", 0.0) for c in claims) / total, 1)
            if total > 0 else 0.0
        )
        return {
            "title": "Executive Summary",
            "total_claims_assessed": total,
            "high_risk_claims": high,
            "medium_risk_claims": medium,
            "low_risk_claims": low,
            "average_risk_score": avg_score,
            "risk_posture": self._get_risk_posture(avg_score),
            "assessment_date": data.get("assessment_date", utcnow().isoformat()),
        }

    def _section_risk_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build risk overview section."""
        claims = data.get("claims", [])
        distribution: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for c in claims:
            level = c.get("risk_level", "low")
            distribution[level] = distribution.get(level, 0) + 1
        return {
            "title": "Risk Overview",
            "risk_distribution": distribution,
            "total_claims": len(claims),
            "highest_risk_claim": max(
                claims, key=lambda x: x.get("risk_score", 0.0)
            ).get("claim_id", "") if claims else "",
            "primary_risk_factors": data.get("primary_risk_factors", []),
        }

    def _section_seven_sins(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build TerraChoice Seven Sins analysis section."""
        claims = data.get("claims", [])
        sin_counts: Dict[str, int] = {sin: 0 for sin in _SEVEN_SINS}
        sin_claims: Dict[str, List[str]] = {sin: [] for sin in _SEVEN_SINS}
        for c in claims:
            for sin in c.get("sins_triggered", []):
                if sin in sin_counts:
                    sin_counts[sin] += 1
                    sin_claims[sin].append(c.get("claim_id", ""))
        return {
            "title": "Seven Sins of Greenwashing Analysis",
            "framework": "TerraChoice",
            "sin_distribution": sin_counts,
            "sin_details": [
                {"sin": sin, "count": sin_counts[sin],
                 "affected_claims": sin_claims[sin]}
                for sin in _SEVEN_SINS
            ],
            "total_sin_violations": sum(sin_counts.values()),
        }

    def _section_ucpd_blacklist(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build UCPD Blacklist check section."""
        claims = data.get("claims", [])
        flagged = [c for c in claims if c.get("ucpd_flagged", False)]
        return {
            "title": "UCPD Blacklist Check",
            "directive_reference": "Directive 2005/29/EC (UCPD)",
            "total_checked": len(claims),
            "flagged_count": len(flagged),
            "clean_count": len(claims) - len(flagged),
            "flagged_claims": [
                {
                    "claim_id": c.get("claim_id", ""),
                    "claim_text": c.get("claim_text", ""),
                    "ucpd_article": c.get("ucpd_article", ""),
                    "blacklist_reason": c.get("blacklist_reason", ""),
                    "penalty_exposure": c.get("penalty_exposure", ""),
                }
                for c in flagged
            ],
        }

    def _section_claim_risk_matrix(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build claim risk matrix section."""
        claims = data.get("claims", [])
        return {
            "title": "Claim Risk Matrix",
            "total_claims": len(claims),
            "matrix": [
                {
                    "claim_id": c.get("claim_id", ""),
                    "claim_text": c.get("claim_text", "")[:80],
                    "risk_level": c.get("risk_level", "low"),
                    "risk_score": round(c.get("risk_score", 0.0), 1),
                    "likelihood": c.get("likelihood", "low"),
                    "impact": c.get("impact", "low"),
                    "sins_triggered": c.get("sins_triggered", []),
                    "ucpd_flagged": c.get("ucpd_flagged", False),
                }
                for c in sorted(
                    claims,
                    key=lambda x: x.get("risk_score", 0),
                    reverse=True,
                )
            ],
        }

    def _section_trend_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build trend analysis section."""
        trend_data = data.get("trend_data", [])
        return {
            "title": "Trend Analysis",
            "periods": [
                {
                    "period": t.get("period", ""),
                    "average_risk_score": round(
                        t.get("average_risk_score", 0.0), 1
                    ),
                    "high_risk_count": t.get("high_risk_count", 0),
                    "total_claims": t.get("total_claims", 0),
                    "trend_direction": t.get("trend_direction", "stable"),
                }
                for t in trend_data
            ],
            "overall_trend": data.get("overall_trend", "stable"),
        }

    def _section_remediation_actions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build remediation actions section."""
        actions = data.get("remediation_actions", [])
        return {
            "title": "Remediation Actions",
            "total_actions": len(actions),
            "actions": [
                {
                    "priority": a.get("priority", 0),
                    "claim_id": a.get("claim_id", ""),
                    "sin_addressed": a.get("sin_addressed", ""),
                    "action": a.get("action", ""),
                    "effort": a.get("effort", ""),
                    "deadline": a.get("deadline", ""),
                    "risk_reduction": a.get("risk_reduction", ""),
                }
                for a in sorted(
                    actions, key=lambda x: x.get("priority", 999)
                )
            ],
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Greenwashing Risk Report - EU Green Claims Directive\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Frameworks:** TerraChoice Seven Sins + UCPD 2005/29/EC"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary as markdown."""
        sec = self._section_executive_summary(data)
        return (
            f"## {sec['title']}\n\n"
            f"### Risk Posture: {sec['risk_posture']}\n\n"
            f"**Average Risk Score:** {sec['average_risk_score']:.1f}  \n"
            f"**Assessment Date:** {sec['assessment_date']}\n\n"
            f"| Metric | Count |\n|--------|------:|\n"
            f"| Total Claims Assessed | {sec['total_claims_assessed']} |\n"
            f"| High Risk | {sec['high_risk_claims']} |\n"
            f"| Medium Risk | {sec['medium_risk_claims']} |\n"
            f"| Low Risk | {sec['low_risk_claims']} |"
        )

    def _md_risk_overview(self, data: Dict[str, Any]) -> str:
        """Render risk overview as markdown."""
        sec = self._section_risk_overview(data)
        dist = sec["risk_distribution"]
        lines = [
            f"## {sec['title']}\n",
            f"- **High:** {dist.get('high', 0)}",
            f"- **Medium:** {dist.get('medium', 0)}",
            f"- **Low:** {dist.get('low', 0)}\n",
            f"**Highest Risk Claim:** {sec['highest_risk_claim']}",
        ]
        if sec["primary_risk_factors"]:
            lines.append("\n**Primary Risk Factors:**\n")
            for rf in sec["primary_risk_factors"]:
                lines.append(f"- {rf}")
        return "\n".join(lines)

    def _md_seven_sins_analysis(self, data: Dict[str, Any]) -> str:
        """Render Seven Sins analysis as markdown."""
        sec = self._section_seven_sins(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Framework:** {sec['framework']}  \n"
            f"**Total Violations:** {sec['total_sin_violations']}\n",
            "| Sin | Count | Affected Claims |",
            "|-----|------:|-----------------|",
        ]
        for sd in sec["sin_details"]:
            affected = ", ".join(sd["affected_claims"][:5]) or "None"
            lines.append(f"| {sd['sin']} | {sd['count']} | {affected} |")
        return "\n".join(lines)

    def _md_ucpd_blacklist_check(self, data: Dict[str, Any]) -> str:
        """Render UCPD Blacklist check as markdown."""
        sec = self._section_ucpd_blacklist(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Directive:** {sec['directive_reference']}  \n"
            f"**Flagged:** {sec['flagged_count']} / {sec['total_checked']}\n",
        ]
        if sec["flagged_claims"]:
            lines.append("### Flagged Claims\n")
            for fc in sec["flagged_claims"]:
                lines.append(f"- **{fc['claim_id']}**: {fc['claim_text'][:60]}")
                lines.append(f"  - Article: {fc['ucpd_article']}")
                lines.append(f"  - Reason: {fc['blacklist_reason']}")
                lines.append(f"  - Penalty Exposure: {fc['penalty_exposure']}")
        else:
            lines.append("No claims flagged against UCPD Blacklist.")
        return "\n".join(lines)

    def _md_claim_risk_matrix(self, data: Dict[str, Any]) -> str:
        """Render claim risk matrix as markdown."""
        sec = self._section_claim_risk_matrix(data)
        lines = [
            f"## {sec['title']}\n",
            "| Claim ID | Risk Level | Score | Likelihood | Impact | UCPD |",
            "|----------|-----------|------:|-----------|--------|:----:|",
        ]
        for m in sec["matrix"]:
            ucpd = "Yes" if m["ucpd_flagged"] else "No"
            lines.append(
                f"| {m['claim_id']} | {m['risk_level']} | {m['risk_score']:.1f} "
                f"| {m['likelihood']} | {m['impact']} | {ucpd} |"
            )
        return "\n".join(lines)

    def _md_trend_analysis(self, data: Dict[str, Any]) -> str:
        """Render trend analysis as markdown."""
        sec = self._section_trend_analysis(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Overall Trend:** {sec['overall_trend']}\n",
            "| Period | Avg Risk Score | High Risk | Total Claims | Trend |",
            "|--------|-------------:|----------:|-------------:|-------|",
        ]
        for p in sec["periods"]:
            lines.append(
                f"| {p['period']} | {p['average_risk_score']:.1f} "
                f"| {p['high_risk_count']} | {p['total_claims']} "
                f"| {p['trend_direction']} |"
            )
        return "\n".join(lines)

    def _md_remediation_actions(self, data: Dict[str, Any]) -> str:
        """Render remediation actions as markdown."""
        sec = self._section_remediation_actions(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Actions:** {sec['total_actions']}\n",
            "| Priority | Claim | Sin Addressed | Action | Effort | Deadline |",
            "|---------:|-------|---------------|--------|--------|----------|",
        ]
        for a in sec["actions"]:
            lines.append(
                f"| {a['priority']} | {a['claim_id']} | {a['sin_addressed']} "
                f"| {a['action']} | {a['effort']} | {a['deadline']} |"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        """Render provenance section as markdown."""
        prov = _compute_hash(data)
        return (
            f"## Provenance\n\n"
            f"**Input Data Hash:** `{prov}`  \n"
            f"**Template Version:** 18.0.0  \n"
            f"**Generated At:** "
            f"{self.generated_at.isoformat() if self.generated_at else ''}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-018 EU Green Claims Prep Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1100px;margin:auto}"
            "h1{color:#1b5e20;border-bottom:2px solid #1b5e20;padding-bottom:.3em}"
            "h2{color:#2e7d32;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8f5e9}"
            ".high-risk{color:#c62828;font-weight:bold}"
            ".medium-risk{color:#e65100;font-weight:bold}"
            ".low-risk{color:#2e7d32;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Greenwashing Risk Report - EU Green Claims Directive</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> "
            f"| {data.get('reporting_period', '')}</p>"
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary HTML."""
        sec = self._section_executive_summary(data)
        css_class = self._risk_css_class(sec["average_risk_score"])
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{css_class}'>Risk Posture: {sec['risk_posture']}</p>\n"
            f"<p>Average Risk Score: {sec['average_risk_score']:.1f}</p>\n"
            f"<table><tr><th>Metric</th><th>Count</th></tr>"
            f"<tr><td>Total Claims</td><td>{sec['total_claims_assessed']}</td></tr>"
            f"<tr><td>High Risk</td><td>{sec['high_risk_claims']}</td></tr>"
            f"<tr><td>Medium Risk</td><td>{sec['medium_risk_claims']}</td></tr>"
            f"<tr><td>Low Risk</td><td>{sec['low_risk_claims']}</td></tr></table>"
        )

    def _html_risk_overview(self, data: Dict[str, Any]) -> str:
        """Render risk overview HTML."""
        sec = self._section_risk_overview(data)
        dist = sec["risk_distribution"]
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<ul><li class='high-risk'>High: {dist.get('high', 0)}</li>"
            f"<li class='medium-risk'>Medium: {dist.get('medium', 0)}</li>"
            f"<li class='low-risk'>Low: {dist.get('low', 0)}</li></ul>"
        )

    def _html_seven_sins(self, data: Dict[str, Any]) -> str:
        """Render Seven Sins analysis HTML."""
        sec = self._section_seven_sins(data)
        rows = "".join(
            f"<tr><td>{sd['sin']}</td><td>{sd['count']}</td></tr>"
            for sd in sec["sin_details"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total Violations: {sec['total_sin_violations']}</p>\n"
            f"<table><tr><th>Sin</th><th>Count</th></tr>{rows}</table>"
        )

    def _html_claim_risk_matrix(self, data: Dict[str, Any]) -> str:
        """Render claim risk matrix HTML."""
        sec = self._section_claim_risk_matrix(data)
        rows = "".join(
            f"<tr><td>{m['claim_id']}</td>"
            f"<td class='{m['risk_level']}-risk'>{m['risk_level']}</td>"
            f"<td>{m['risk_score']:.1f}</td>"
            f"<td>{m['likelihood']}</td><td>{m['impact']}</td></tr>"
            for m in sec["matrix"][:15]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Claim</th><th>Risk</th><th>Score</th>"
            f"<th>Likelihood</th><th>Impact</th></tr>{rows}</table>"
        )

    def _html_remediation_actions(self, data: Dict[str, Any]) -> str:
        """Render remediation actions HTML."""
        sec = self._section_remediation_actions(data)
        rows = "".join(
            f"<tr><td>{a['priority']}</td><td>{a['claim_id']}</td>"
            f"<td>{a['action']}</td><td>{a['effort']}</td></tr>"
            for a in sec["actions"][:10]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total Actions: {sec['total_actions']}</p>\n"
            f"<table><tr><th>Priority</th><th>Claim</th><th>Action</th>"
            f"<th>Effort</th></tr>{rows}</table>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_risk_posture(self, avg_score: float) -> str:
        """Determine risk posture from average risk score."""
        if avg_score <= 2.0:
            return "Low Risk"
        elif avg_score <= 4.0:
            return "Moderate Risk"
        elif avg_score <= 6.0:
            return "Elevated Risk"
        elif avg_score <= 8.0:
            return "High Risk"
        else:
            return "Critical Risk"

    def _risk_css_class(self, score: float) -> str:
        """Return CSS class for risk score."""
        if score <= 3.0:
            return "low-risk"
        elif score <= 6.0:
            return "medium-risk"
        else:
            return "high-risk"
