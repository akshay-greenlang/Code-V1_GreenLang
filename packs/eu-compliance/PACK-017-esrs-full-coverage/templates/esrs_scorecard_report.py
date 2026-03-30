# -*- coding: utf-8 -*-
"""
ESRSScorecard - Cross-Standard ESRS Compliance Scorecard

Aggregates compliance scores from all 12 ESRS standards (ESRS 2 General,
E1-E5 Environmental, S1-S4 Social, G1 Governance) to provide an overall
compliance readiness percentage, gap analysis, materiality matrix, and
priority improvement actions.

Sections:
    1. Executive Summary - Overall compliance score and status
    2. Standard Scores - Detailed scores per ESRS standard
    3. Gap Analysis - Missing disclosures and data quality issues
    4. Materiality Matrix - IRO mapping across standards
    5. Cross-Standard Consistency - Alignment checks
    6. Improvement Priorities - Ranked action items

Author: GreenLang Team
Version: 17.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_SECTIONS: List[str] = [
    "executive_summary",
    "standard_scores",
    "gap_analysis",
    "materiality_matrix",
    "cross_standard_consistency",
    "improvement_priorities",
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

class ESRSScorecard:
    """
    Cross-Standard ESRS Compliance Scorecard.

    Aggregates compliance data from all 12 ESRS standards to provide a
    comprehensive view of overall sustainability reporting readiness. Identifies
    gaps, inconsistencies, and priority actions needed to achieve full compliance.

    Standards Covered:
        - ESRS 2: General Disclosures
        - ESRS E1: Climate Change
        - ESRS E2: Pollution
        - ESRS E3: Water and Marine Resources
        - ESRS E4: Biodiversity and Ecosystems
        - ESRS E5: Resource Use and Circular Economy
        - ESRS S1: Own Workforce
        - ESRS S2: Workers in the Value Chain
        - ESRS S3: Affected Communities
        - ESRS S4: Consumers and End-Users
        - ESRS G1: Business Conduct

    Example:
        >>> scorecard = ESRSScorecard()
        >>> md = scorecard.render_markdown(data)
        >>> html = scorecard.render_html(data)
        >>> js = scorecard.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ESRSScorecard."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> str:
        """Render ESRS compliance scorecard as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_standard_scores(data),
            self._md_gap_analysis(data),
            self._md_materiality_matrix(data),
            self._md_consistency(data),
            self._md_priorities(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> str:
        """Render ESRS compliance scorecard as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_standard_scores(data),
            self._html_gap_analysis(data),
            self._html_priorities(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>ESRS Compliance Scorecard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Render ESRS compliance scorecard as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "esrs_scorecard_report",
            "esrs_reference": "ESRS 2, E1-E5, S1-S4, G1",
            "version": "17.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "executive_summary": self._section_executive_summary(data),
            "standard_scores": self._section_standard_scores(data),
            "gap_analysis": self._section_gap_analysis(data),
            "materiality_matrix": self._section_materiality_matrix(data),
            "cross_standard_consistency": self._section_consistency(data),
            "improvement_priorities": self._section_priorities(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def get_sections(self) -> List[str]:
        """Return list of available section names."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if "standard_scores" not in data:
            errors.append("standard_scores required (dict with keys: esrs2, e1-e5, s1-s4, g1)")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary section."""
        scores = data.get("standard_scores", {})
        all_scores = [scores.get(k, 0.0) for k in [
            "esrs2", "e1", "e2", "e3", "e4", "e5", "s1", "s2", "s3", "s4", "g1"
        ]]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        return {
            "title": "Executive Summary",
            "overall_compliance_pct": round(overall_score, 1),
            "total_standards": 12,
            "standards_fully_compliant": sum(1 for s in all_scores if s >= 95.0),
            "standards_partially_compliant": sum(1 for s in all_scores if 50.0 <= s < 95.0),
            "standards_non_compliant": sum(1 for s in all_scores if s < 50.0),
            "assessment_date": data.get("assessment_date", utcnow().isoformat()),
            "readiness_status": self._get_readiness_status(overall_score),
        }

    def _section_standard_scores(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build detailed standard scores section."""
        scores = data.get("standard_scores", {})
        return {
            "title": "Standard-by-Standard Compliance Scores",
            "scores": [
                {"standard": "ESRS 2", "name": "General Disclosures", "score": scores.get("esrs2", 0.0)},
                {"standard": "ESRS E1", "name": "Climate Change", "score": scores.get("e1", 0.0)},
                {"standard": "ESRS E2", "name": "Pollution", "score": scores.get("e2", 0.0)},
                {"standard": "ESRS E3", "name": "Water & Marine", "score": scores.get("e3", 0.0)},
                {"standard": "ESRS E4", "name": "Biodiversity & Ecosystems", "score": scores.get("e4", 0.0)},
                {"standard": "ESRS E5", "name": "Resource Use & Circular Economy", "score": scores.get("e5", 0.0)},
                {"standard": "ESRS S1", "name": "Own Workforce", "score": scores.get("s1", 0.0)},
                {"standard": "ESRS S2", "name": "Workers in Value Chain", "score": scores.get("s2", 0.0)},
                {"standard": "ESRS S3", "name": "Affected Communities", "score": scores.get("s3", 0.0)},
                {"standard": "ESRS S4", "name": "Consumers & End-Users", "score": scores.get("s4", 0.0)},
                {"standard": "ESRS G1", "name": "Business Conduct", "score": scores.get("g1", 0.0)},
            ],
        }

    def _section_gap_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build gap analysis section."""
        gaps = data.get("compliance_gaps", [])
        return {
            "title": "Gap Analysis",
            "total_gaps": len(gaps),
            "critical_gaps": [g for g in gaps if g.get("severity", "") == "critical"],
            "major_gaps": [g for g in gaps if g.get("severity", "") == "major"],
            "minor_gaps": [g for g in gaps if g.get("severity", "") == "minor"],
            "missing_disclosures": data.get("missing_disclosures", []),
            "data_quality_issues": data.get("data_quality_issues", []),
        }

    def _section_materiality_matrix(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build materiality matrix section."""
        iros = data.get("material_iros", [])
        return {
            "title": "Materiality Matrix (IRO Mapping)",
            "total_iros": len(iros),
            "material_topics": [
                {
                    "topic": iro.get("topic", ""),
                    "impact_score": iro.get("impact_score", 0),
                    "risk_score": iro.get("risk_score", 0),
                    "opportunity_score": iro.get("opportunity_score", 0),
                    "related_standards": iro.get("related_standards", []),
                }
                for iro in iros
            ],
            "materiality_threshold": data.get("materiality_threshold", 3.0),
        }

    def _section_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build cross-standard consistency section."""
        issues = data.get("consistency_issues", [])
        return {
            "title": "Cross-Standard Consistency Checks",
            "total_issues": len(issues),
            "inconsistencies": [
                {
                    "type": issue.get("type", ""),
                    "description": issue.get("description", ""),
                    "affected_standards": issue.get("affected_standards", []),
                }
                for issue in issues
            ],
            "alignment_score_pct": data.get("alignment_score_pct", 0.0),
        }

    def _section_priorities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build improvement priorities section."""
        actions = data.get("priority_actions", [])
        return {
            "title": "Improvement Priorities",
            "total_actions": len(actions),
            "priorities": [
                {
                    "rank": action.get("rank", 0),
                    "action": action.get("action", ""),
                    "affected_standards": action.get("affected_standards", []),
                    "effort": action.get("effort", ""),
                    "impact": action.get("impact", ""),
                    "deadline": action.get("deadline", ""),
                }
                for action in sorted(actions, key=lambda x: x.get("rank", 999))
            ],
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# ESRS Compliance Scorecard\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standards Assessed:** 12 (ESRS 2, E1-E5, S1-S4, G1)"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary as markdown."""
        sec = self._section_executive_summary(data)
        return (
            f"## {sec['title']}\n\n"
            f"### Overall Compliance: {sec['overall_compliance_pct']:.1f}%\n\n"
            f"**Readiness Status:** {sec['readiness_status']}  \n"
            f"**Assessment Date:** {sec['assessment_date']}\n\n"
            f"| Status | Count |\n|--------|------:|\n"
            f"| Fully Compliant (≥95%) | {sec['standards_fully_compliant']} |\n"
            f"| Partially Compliant (50-94%) | {sec['standards_partially_compliant']} |\n"
            f"| Non-Compliant (<50%) | {sec['standards_non_compliant']} |\n"
            f"| **Total Standards** | **{sec['total_standards']}** |"
        )

    def _md_standard_scores(self, data: Dict[str, Any]) -> str:
        """Render standard scores as markdown."""
        sec = self._section_standard_scores(data)
        lines = [f"## {sec['title']}\n", "| Standard | Name | Score |", "|----------|------|------:|"]
        for score in sec["scores"]:
            lines.append(
                f"| {score['standard']} | {score['name']} | {score['score']:.1f}% |"
            )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render gap analysis as markdown."""
        sec = self._section_gap_analysis(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Gaps Identified:** {sec['total_gaps']}\n",
            f"- Critical: {len(sec['critical_gaps'])}",
            f"- Major: {len(sec['major_gaps'])}",
            f"- Minor: {len(sec['minor_gaps'])}\n",
        ]
        if sec["critical_gaps"]:
            lines.append("### Critical Gaps")
            for gap in sec["critical_gaps"]:
                lines.append(f"- **{gap.get('standard', '')}**: {gap.get('description', '')}")
        return "\n".join(lines)

    def _md_materiality_matrix(self, data: Dict[str, Any]) -> str:
        """Render materiality matrix as markdown."""
        sec = self._section_materiality_matrix(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Material IROs:** {sec['total_iros']}  \n"
            f"**Materiality Threshold:** {sec['materiality_threshold']}\n",
            "| Topic | Impact | Risk | Opportunity | Standards |",
            "|-------|-------:|-----:|------------:|-----------|",
        ]
        for topic in sec["material_topics"]:
            stds = ", ".join(topic["related_standards"])
            lines.append(
                f"| {topic['topic']} | {topic['impact_score']} | {topic['risk_score']} | "
                f"{topic['opportunity_score']} | {stds} |"
            )
        return "\n".join(lines)

    def _md_consistency(self, data: Dict[str, Any]) -> str:
        """Render consistency checks as markdown."""
        sec = self._section_consistency(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Alignment Score:** {sec['alignment_score_pct']:.1f}%  \n"
            f"**Inconsistencies Found:** {sec['total_issues']}\n",
        ]
        if sec["inconsistencies"]:
            lines.append("### Key Inconsistencies")
            for issue in sec["inconsistencies"]:
                stds = ", ".join(issue["affected_standards"])
                lines.append(f"- **{issue['type']}** ({stds}): {issue['description']}")
        return "\n".join(lines)

    def _md_priorities(self, data: Dict[str, Any]) -> str:
        """Render improvement priorities as markdown."""
        sec = self._section_priorities(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Priority Actions:** {sec['total_actions']}\n",
            "| Rank | Action | Standards | Effort | Impact | Deadline |",
            "|-----:|--------|-----------|--------|--------|----------|",
        ]
        for p in sec["priorities"]:
            stds = ", ".join(p["affected_standards"])
            lines.append(
                f"| {p['rank']} | {p['action']} | {stds} | {p['effort']} | "
                f"{p['impact']} | {p['deadline']} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Scorecard generated by PACK-017 ESRS Full Coverage Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1200px;margin:auto}"
            "h1{color:#1b5e20;border-bottom:2px solid #1b5e20;padding-bottom:.3em}"
            "h2{color:#2e7d32;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8f5e9}"
            ".alert{color:#c62828;font-weight:bold}"
            ".success{color:#2e7d32;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>ESRS Compliance Scorecard</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> "
            f"| {data.get('reporting_year', '')}</p>"
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary HTML."""
        sec = self._section_executive_summary(data)
        css_class = "success" if sec["overall_compliance_pct"] >= 80.0 else "alert"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{css_class}'>Overall Compliance: "
            f"{sec['overall_compliance_pct']:.1f}%</p>\n"
            f"<p><strong>Status:</strong> {sec['readiness_status']}</p>\n"
            f"<table><tr><th>Status</th><th>Count</th></tr>"
            f"<tr><td>Fully Compliant (≥95%)</td><td>{sec['standards_fully_compliant']}</td></tr>"
            f"<tr><td>Partially Compliant (50-94%)</td><td>{sec['standards_partially_compliant']}</td></tr>"
            f"<tr><td>Non-Compliant (<50%)</td><td>{sec['standards_non_compliant']}</td></tr>"
            f"<tr><td><strong>Total</strong></td><td><strong>{sec['total_standards']}</strong></td></tr>"
            f"</table>"
        )

    def _html_standard_scores(self, data: Dict[str, Any]) -> str:
        """Render standard scores HTML."""
        sec = self._section_standard_scores(data)
        rows = "".join(
            f"<tr><td>{s['standard']}</td><td>{s['name']}</td><td>{s['score']:.1f}%</td></tr>"
            for s in sec["scores"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Standard</th><th>Name</th><th>Score</th></tr>"
            f"{rows}</table>"
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render gap analysis HTML."""
        sec = self._section_gap_analysis(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p><strong>Total Gaps:</strong> {sec['total_gaps']}</p>\n"
            f"<ul>"
            f"<li>Critical: {len(sec['critical_gaps'])}</li>"
            f"<li>Major: {len(sec['major_gaps'])}</li>"
            f"<li>Minor: {len(sec['minor_gaps'])}</li>"
            f"</ul>"
        )

    def _html_priorities(self, data: Dict[str, Any]) -> str:
        """Render improvement priorities HTML."""
        sec = self._section_priorities(data)
        rows = "".join(
            f"<tr><td>{p['rank']}</td><td>{p['action']}</td><td>{p['effort']}</td>"
            f"<td>{p['impact']}</td></tr>"
            for p in sec["priorities"][:10]  # Top 10 only
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p><strong>Total Actions:</strong> {sec['total_actions']}</p>\n"
            f"<table><tr><th>Rank</th><th>Action</th><th>Effort</th><th>Impact</th></tr>"
            f"{rows}</table>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_readiness_status(self, score: float) -> str:
        """Determine readiness status from overall score."""
        if score >= 95.0:
            return "Ready for Reporting"
        elif score >= 80.0:
            return "Near Ready (Minor Gaps)"
        elif score >= 60.0:
            return "Partially Ready (Significant Gaps)"
        elif score >= 40.0:
            return "Not Ready (Major Gaps)"
        else:
            return "Far from Ready (Critical Gaps)"
