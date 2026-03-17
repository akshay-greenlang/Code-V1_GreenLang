# -*- coding: utf-8 -*-
"""
ClaimAssessmentReportTemplate - EU Green Claims Directive Claim Assessment

Evaluates all environmental claims made by an entity against EU Green Claims
Directive requirements, performing substantiation checks, risk scoring, and
generating prioritised recommendations for each claim. The report provides
a complete inventory of claims with their current compliance posture.

Sections:
    1. Executive Summary - Overall assessment score and key findings
    2. Claim Inventory - Full catalogue of environmental claims
    3. Substantiation Results - Evidence quality per claim
    4. Risk Assessment - Risk ratings and exposure analysis
    5. Recommendations - Prioritised corrective actions
    6. Provenance - Data lineage and hash chain

PACK Reference: PACK-018 EU Green Claims Prep Pack
Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SECTIONS: List[Dict[str, Any]] = [
    {"id": "executive_summary", "title": "Executive Summary", "order": 1},
    {"id": "claim_inventory", "title": "Claim Inventory", "order": 2},
    {"id": "substantiation_results", "title": "Substantiation Results", "order": 3},
    {"id": "risk_assessment", "title": "Risk Assessment", "order": 4},
    {"id": "recommendations", "title": "Recommendations", "order": 5},
    {"id": "provenance", "title": "Provenance", "order": 6},
]


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class ClaimAssessmentReportTemplate:
    """
    EU Green Claims Directive - Claim Assessment Report.

    Renders a comprehensive assessment of all environmental claims made by
    an entity, evaluating each claim against EU Green Claims Directive
    substantiation requirements. Includes claim inventory, evidence quality
    scoring, risk ratings, and prioritised remediation recommendations.

    Example:
        >>> tpl = ClaimAssessmentReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ClaimAssessmentReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render claim assessment report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_claim_inventory(data),
            self._md_substantiation_results(data),
            self._md_risk_assessment(data),
            self._md_recommendations(data),
            self._md_provenance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render claim assessment report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_claim_inventory(data),
            self._html_risk_assessment(data),
            self._html_recommendations(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Claim Assessment Report - EU Green Claims</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render claim assessment report as structured JSON."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "claim_assessment_report",
            "directive_reference": "EU Green Claims Directive 2023/0085",
            "version": "18.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_period": data.get("reporting_period", ""),
            "executive_summary": self._section_executive_summary(data),
            "claim_inventory": self._section_claim_inventory(data),
            "substantiation_results": self._section_substantiation(data),
            "risk_assessment": self._section_risk_assessment(data),
            "recommendations": self._section_recommendations(data),
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
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Section builders (dict)
    # ------------------------------------------------------------------

    def _section_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary section."""
        claims = data.get("claims", [])
        total = len(claims)
        substantiated = sum(1 for c in claims if c.get("substantiated", False))
        high_risk = sum(1 for c in claims if c.get("risk_level", "") == "high")
        score = round(substantiated / total * 100, 1) if total > 0 else 0.0
        return {
            "title": "Executive Summary",
            "total_claims": total,
            "substantiated_claims": substantiated,
            "unsubstantiated_claims": total - substantiated,
            "high_risk_claims": high_risk,
            "overall_score_pct": score,
            "assessment_status": self._get_status(score),
            "assessment_date": data.get("assessment_date", _utcnow().isoformat()),
        }

    def _section_claim_inventory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build claim inventory section."""
        claims = data.get("claims", [])
        return {
            "title": "Claim Inventory",
            "total_claims": len(claims),
            "claims": [
                {
                    "claim_id": c.get("claim_id", ""),
                    "claim_text": c.get("claim_text", ""),
                    "claim_type": c.get("claim_type", ""),
                    "product_or_service": c.get("product_or_service", ""),
                    "channel": c.get("channel", ""),
                    "lifecycle_stage": c.get("lifecycle_stage", ""),
                }
                for c in claims
            ],
        }

    def _section_substantiation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build substantiation results section."""
        claims = data.get("claims", [])
        return {
            "title": "Substantiation Results",
            "results": [
                {
                    "claim_id": c.get("claim_id", ""),
                    "substantiated": c.get("substantiated", False),
                    "evidence_quality": c.get("evidence_quality", "unknown"),
                    "methodology": c.get("methodology", ""),
                    "data_sources": c.get("data_sources", []),
                    "gaps": c.get("substantiation_gaps", []),
                }
                for c in claims
            ],
        }

    def _section_risk_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build risk assessment section."""
        claims = data.get("claims", [])
        risk_counts = {"high": 0, "medium": 0, "low": 0}
        for c in claims:
            level = c.get("risk_level", "low")
            risk_counts[level] = risk_counts.get(level, 0) + 1
        return {
            "title": "Risk Assessment",
            "risk_distribution": risk_counts,
            "high_risk_claims": [
                {
                    "claim_id": c.get("claim_id", ""),
                    "claim_text": c.get("claim_text", ""),
                    "risk_level": c.get("risk_level", ""),
                    "risk_factors": c.get("risk_factors", []),
                    "potential_penalty": c.get("potential_penalty", ""),
                }
                for c in claims if c.get("risk_level") == "high"
            ],
        }

    def _section_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build recommendations section."""
        recs = data.get("recommendations", [])
        return {
            "title": "Recommendations",
            "total_recommendations": len(recs),
            "items": [
                {
                    "priority": r.get("priority", 0),
                    "claim_id": r.get("claim_id", ""),
                    "action": r.get("action", ""),
                    "effort": r.get("effort", ""),
                    "deadline": r.get("deadline", ""),
                    "expected_outcome": r.get("expected_outcome", ""),
                }
                for r in sorted(recs, key=lambda x: x.get("priority", 999))
            ],
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Claim Assessment Report - EU Green Claims Directive\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Directive:** EU Green Claims Directive 2023/0085"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary as markdown."""
        sec = self._section_executive_summary(data)
        return (
            f"## {sec['title']}\n\n"
            f"### Overall Score: {sec['overall_score_pct']:.1f}%\n\n"
            f"**Status:** {sec['assessment_status']}  \n"
            f"**Assessment Date:** {sec['assessment_date']}\n\n"
            f"| Metric | Count |\n|--------|------:|\n"
            f"| Total Claims | {sec['total_claims']} |\n"
            f"| Substantiated | {sec['substantiated_claims']} |\n"
            f"| Unsubstantiated | {sec['unsubstantiated_claims']} |\n"
            f"| High Risk | {sec['high_risk_claims']} |"
        )

    def _md_claim_inventory(self, data: Dict[str, Any]) -> str:
        """Render claim inventory as markdown."""
        sec = self._section_claim_inventory(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Claims:** {sec['total_claims']}\n",
            "| ID | Claim | Type | Product/Service | Channel |",
            "|----|-------|------|-----------------|---------|",
        ]
        for c in sec["claims"]:
            lines.append(
                f"| {c['claim_id']} | {c['claim_text'][:60]} | {c['claim_type']} "
                f"| {c['product_or_service']} | {c['channel']} |"
            )
        return "\n".join(lines)

    def _md_substantiation_results(self, data: Dict[str, Any]) -> str:
        """Render substantiation results as markdown."""
        sec = self._section_substantiation(data)
        lines = [
            "## Substantiation Results\n",
            "| Claim ID | Substantiated | Evidence Quality | Methodology |",
            "|----------|:------------:|-----------------|-------------|",
        ]
        for r in sec["results"]:
            check = "Yes" if r["substantiated"] else "No"
            lines.append(
                f"| {r['claim_id']} | {check} | {r['evidence_quality']} "
                f"| {r['methodology']} |"
            )
        return "\n".join(lines)

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render risk assessment as markdown."""
        sec = self._section_risk_assessment(data)
        dist = sec["risk_distribution"]
        lines = [
            f"## {sec['title']}\n",
            f"- **High:** {dist.get('high', 0)}",
            f"- **Medium:** {dist.get('medium', 0)}",
            f"- **Low:** {dist.get('low', 0)}\n",
        ]
        if sec["high_risk_claims"]:
            lines.append("### High-Risk Claims\n")
            for c in sec["high_risk_claims"]:
                factors = ", ".join(c["risk_factors"]) if c["risk_factors"] else "N/A"
                lines.append(f"- **{c['claim_id']}**: {c['claim_text']}")
                lines.append(f"  - Risk Factors: {factors}")
                lines.append(f"  - Potential Penalty: {c['potential_penalty']}")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations as markdown."""
        sec = self._section_recommendations(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Recommendations:** {sec['total_recommendations']}\n",
            "| Priority | Claim | Action | Effort | Deadline |",
            "|---------:|-------|--------|--------|----------|",
        ]
        for r in sec["items"]:
            lines.append(
                f"| {r['priority']} | {r['claim_id']} | {r['action']} "
                f"| {r['effort']} | {r['deadline']} |"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        """Render provenance section as markdown."""
        prov = _compute_hash(data)
        return (
            f"## Provenance\n\n"
            f"**Input Data Hash:** `{prov}`  \n"
            f"**Template Version:** 18.0.0  \n"
            f"**Generated At:** {self.generated_at.isoformat() if self.generated_at else ''}"
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
            ".success{color:#2e7d32;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Claim Assessment Report - EU Green Claims Directive</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> "
            f"| {data.get('reporting_period', '')}</p>"
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary HTML."""
        sec = self._section_executive_summary(data)
        css_class = "success" if sec["overall_score_pct"] >= 80.0 else "high-risk"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{css_class}'>Overall Score: {sec['overall_score_pct']:.1f}%</p>\n"
            f"<p><strong>Status:</strong> {sec['assessment_status']}</p>\n"
            f"<table><tr><th>Metric</th><th>Count</th></tr>"
            f"<tr><td>Total Claims</td><td>{sec['total_claims']}</td></tr>"
            f"<tr><td>Substantiated</td><td>{sec['substantiated_claims']}</td></tr>"
            f"<tr><td>Unsubstantiated</td><td>{sec['unsubstantiated_claims']}</td></tr>"
            f"<tr><td>High Risk</td><td>{sec['high_risk_claims']}</td></tr></table>"
        )

    def _html_claim_inventory(self, data: Dict[str, Any]) -> str:
        """Render claim inventory HTML."""
        sec = self._section_claim_inventory(data)
        rows = "".join(
            f"<tr><td>{c['claim_id']}</td><td>{c['claim_text'][:60]}</td>"
            f"<td>{c['claim_type']}</td><td>{c['product_or_service']}</td></tr>"
            for c in sec["claims"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total Claims: {sec['total_claims']}</p>\n"
            f"<table><tr><th>ID</th><th>Claim</th><th>Type</th><th>Product</th></tr>"
            f"{rows}</table>"
        )

    def _html_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render risk assessment HTML."""
        sec = self._section_risk_assessment(data)
        dist = sec["risk_distribution"]
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<ul><li>High: {dist.get('high', 0)}</li>"
            f"<li>Medium: {dist.get('medium', 0)}</li>"
            f"<li>Low: {dist.get('low', 0)}</li></ul>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations HTML."""
        sec = self._section_recommendations(data)
        rows = "".join(
            f"<tr><td>{r['priority']}</td><td>{r['claim_id']}</td>"
            f"<td>{r['action']}</td><td>{r['effort']}</td></tr>"
            for r in sec["items"][:10]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total Recommendations: {sec['total_recommendations']}</p>\n"
            f"<table><tr><th>Priority</th><th>Claim</th><th>Action</th>"
            f"<th>Effort</th></tr>{rows}</table>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_status(self, score: float) -> str:
        """Determine assessment status from overall score."""
        if score >= 95.0:
            return "Fully Compliant"
        elif score >= 80.0:
            return "Largely Compliant (Minor Gaps)"
        elif score >= 60.0:
            return "Partially Compliant (Action Required)"
        elif score >= 40.0:
            return "Non-Compliant (Significant Action Required)"
        else:
            return "Critical Non-Compliance"
