# -*- coding: utf-8 -*-
"""
ComplianceGapReportTemplate - EU Green Claims Compliance Gap Analysis

Performs a systematic gap analysis of environmental claims against EU Green
Claims Directive Articles 3-8. Maps each requirement to current practice,
identifies gaps with severity scoring, assesses cross-regulation consistency
(UCPD, ESRS, Taxonomy), and delivers a costed remediation roadmap.

Sections:
    1. Executive Summary - Overall gap assessment and readiness score
    2. Regulatory Mapping - Article-by-article requirement mapping
    3. Gap Findings - Identified compliance gaps with evidence
    4. Severity Analysis - Gap severity distribution and scoring
    5. Cross-Regulation Consistency - UCPD, ESRS, Taxonomy alignment
    6. Remediation Roadmap - Phased corrective action plan
    7. Cost-Benefit Analysis - Remediation cost vs risk exposure
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

logger = logging.getLogger(__name__)

__all__ = ["ComplianceGapReportTemplate"]

_SECTIONS: List[Dict[str, Any]] = [
    {"id": "executive_summary", "title": "Executive Summary", "order": 1},
    {"id": "regulatory_mapping", "title": "Regulatory Mapping", "order": 2},
    {"id": "gap_findings", "title": "Gap Findings", "order": 3},
    {"id": "severity_analysis", "title": "Severity Analysis", "order": 4},
    {"id": "cross_regulation_consistency", "title": "Cross-Regulation Consistency", "order": 5},
    {"id": "remediation_roadmap", "title": "Remediation Roadmap", "order": 6},
    {"id": "cost_benefit_analysis", "title": "Cost-Benefit Analysis", "order": 7},
    {"id": "provenance", "title": "Provenance", "order": 8},
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


class ComplianceGapReportTemplate:
    """
    EU Green Claims Directive - Compliance Gap Analysis Report.

    Performs an article-by-article gap analysis of environmental claims
    against EU Green Claims Directive Articles 3-8. Maps current practices
    to regulatory requirements, scores gap severity, checks cross-regulation
    consistency with UCPD, ESRS, and Taxonomy, and produces a costed
    remediation roadmap with phased milestones.

    Example:
        >>> tpl = ComplianceGapReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ComplianceGapReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render compliance gap report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_regulatory_mapping(data),
            self._md_gap_findings(data),
            self._md_severity_analysis(data),
            self._md_cross_regulation(data),
            self._md_remediation_roadmap(data),
            self._md_cost_benefit(data),
            self._md_provenance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render compliance gap report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_regulatory_mapping(data),
            self._html_gap_findings(data),
            self._html_severity_analysis(data),
            self._html_cost_benefit(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Compliance Gap Report - EU Green Claims</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render compliance gap report as structured JSON."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "compliance_gap_report",
            "directive_reference": "EU Green Claims Directive 2023/0085 Articles 3-8",
            "version": "18.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_period": data.get("reporting_period", ""),
            "executive_summary": self._section_executive_summary(data),
            "regulatory_mapping": self._section_regulatory_mapping(data),
            "gap_findings": self._section_gap_findings(data),
            "severity_analysis": self._section_severity_analysis(data),
            "cross_regulation_consistency": self._section_cross_regulation(data),
            "remediation_roadmap": self._section_remediation_roadmap(data),
            "cost_benefit_analysis": self._section_cost_benefit(data),
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
        if not data.get("gaps"):
            errors.append("gaps list is required")
        if not data.get("regulatory_requirements"):
            warnings.append("regulatory_requirements missing; mapping will be limited")
        if not data.get("reporting_period"):
            warnings.append("reporting_period missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Section builders (dict)
    # ------------------------------------------------------------------

    def _section_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary section."""
        gaps = data.get("gaps", [])
        reqs = data.get("regulatory_requirements", [])
        total_reqs = len(reqs)
        total_gaps = len(gaps)
        critical = sum(1 for g in gaps if g.get("severity", "") == "critical")
        high = sum(1 for g in gaps if g.get("severity", "") == "high")
        met = total_reqs - total_gaps if total_reqs >= total_gaps else 0
        readiness = round(met / total_reqs * 100, 1) if total_reqs > 0 else 0.0
        return {
            "title": "Executive Summary",
            "total_requirements": total_reqs,
            "requirements_met": met,
            "total_gaps": total_gaps,
            "critical_gaps": critical,
            "high_gaps": high,
            "readiness_score_pct": readiness,
            "readiness_status": self._get_readiness(readiness),
            "assessment_date": data.get("assessment_date", _utcnow().isoformat()),
        }

    def _section_regulatory_mapping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build regulatory mapping section."""
        reqs = data.get("regulatory_requirements", [])
        return {
            "title": "Regulatory Mapping (Articles 3-8)",
            "total_requirements": len(reqs),
            "requirements": [
                {
                    "requirement_id": r.get("requirement_id", ""),
                    "article": r.get("article", ""),
                    "description": r.get("description", ""),
                    "current_status": r.get("current_status", "not_assessed"),
                    "evidence_available": r.get("evidence_available", False),
                    "gap_identified": r.get("gap_identified", False),
                }
                for r in reqs
            ],
        }

    def _section_gap_findings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build gap findings section."""
        gaps = data.get("gaps", [])
        return {
            "title": "Gap Findings",
            "total_gaps": len(gaps),
            "findings": [
                {
                    "gap_id": g.get("gap_id", ""),
                    "article": g.get("article", ""),
                    "requirement": g.get("requirement", ""),
                    "current_state": g.get("current_state", ""),
                    "required_state": g.get("required_state", ""),
                    "severity": g.get("severity", "low"),
                    "evidence_gaps": g.get("evidence_gaps", []),
                    "affected_claims": g.get("affected_claims", []),
                }
                for g in gaps
            ],
        }

    def _section_severity_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build severity analysis section."""
        gaps = data.get("gaps", [])
        distribution: Dict[str, int] = {
            "critical": 0, "high": 0, "medium": 0, "low": 0,
        }
        for g in gaps:
            sev = g.get("severity", "low")
            distribution[sev] = distribution.get(sev, 0) + 1
        total = len(gaps) if gaps else 1
        return {
            "title": "Severity Analysis",
            "severity_distribution": distribution,
            "critical_pct": round(distribution["critical"] / total * 100, 1),
            "high_pct": round(distribution["high"] / total * 100, 1),
            "medium_pct": round(distribution["medium"] / total * 100, 1),
            "low_pct": round(distribution["low"] / total * 100, 1),
            "weighted_severity_score": self._weighted_severity(distribution),
        }

    def _section_cross_regulation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build cross-regulation consistency section."""
        checks = data.get("cross_regulation_checks", [])
        consistent = sum(1 for c in checks if c.get("consistent", False))
        return {
            "title": "Cross-Regulation Consistency",
            "regulations_checked": [
                "UCPD 2005/29/EC", "ESRS E1-E5", "EU Taxonomy",
            ],
            "total_checks": len(checks),
            "consistent": consistent,
            "inconsistent": len(checks) - consistent,
            "consistency_rate_pct": round(
                consistent / len(checks) * 100, 1
            ) if checks else 0.0,
            "findings": [
                {
                    "regulation": c.get("regulation", ""),
                    "requirement": c.get("requirement", ""),
                    "green_claims_position": c.get("green_claims_position", ""),
                    "other_regulation_position": c.get(
                        "other_regulation_position", ""
                    ),
                    "consistent": c.get("consistent", False),
                    "action_needed": c.get("action_needed", ""),
                }
                for c in checks
            ],
        }

    def _section_remediation_roadmap(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build remediation roadmap section."""
        phases = data.get("remediation_phases", [])
        return {
            "title": "Remediation Roadmap",
            "total_phases": len(phases),
            "phases": [
                {
                    "phase": p.get("phase", 0),
                    "name": p.get("name", ""),
                    "start_date": p.get("start_date", ""),
                    "end_date": p.get("end_date", ""),
                    "gaps_addressed": p.get("gaps_addressed", []),
                    "milestones": p.get("milestones", []),
                    "resources_required": p.get("resources_required", ""),
                }
                for p in sorted(
                    phases, key=lambda x: x.get("phase", 999)
                )
            ],
        }

    def _section_cost_benefit(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build cost-benefit analysis section."""
        items = data.get("cost_benefit_items", [])
        total_cost = sum(i.get("estimated_cost", 0.0) for i in items)
        total_risk = sum(i.get("risk_exposure", 0.0) for i in items)
        return {
            "title": "Cost-Benefit Analysis",
            "total_remediation_cost": round(total_cost, 2),
            "total_risk_exposure": round(total_risk, 2),
            "roi_ratio": round(
                total_risk / total_cost, 2
            ) if total_cost > 0 else 0.0,
            "currency": data.get("currency", "EUR"),
            "items": [
                {
                    "gap_id": i.get("gap_id", ""),
                    "estimated_cost": round(i.get("estimated_cost", 0.0), 2),
                    "risk_exposure": round(i.get("risk_exposure", 0.0), 2),
                    "priority": i.get("priority", "medium"),
                    "payback_period": i.get("payback_period", ""),
                }
                for i in sorted(
                    items,
                    key=lambda x: x.get("risk_exposure", 0),
                    reverse=True,
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
            f"# Compliance Gap Report - EU Green Claims Directive\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Scope:** Articles 3-8, EU Green Claims Directive 2023/0085"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary as markdown."""
        sec = self._section_executive_summary(data)
        return (
            f"## {sec['title']}\n\n"
            f"### Readiness: {sec['readiness_score_pct']:.1f}%\n\n"
            f"**Status:** {sec['readiness_status']}  \n"
            f"**Assessment Date:** {sec['assessment_date']}\n\n"
            f"| Metric | Count |\n|--------|------:|\n"
            f"| Total Requirements | {sec['total_requirements']} |\n"
            f"| Requirements Met | {sec['requirements_met']} |\n"
            f"| Total Gaps | {sec['total_gaps']} |\n"
            f"| Critical Gaps | {sec['critical_gaps']} |\n"
            f"| High Gaps | {sec['high_gaps']} |"
        )

    def _md_regulatory_mapping(self, data: Dict[str, Any]) -> str:
        """Render regulatory mapping as markdown."""
        sec = self._section_regulatory_mapping(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Requirements:** {sec['total_requirements']}\n",
            "| ID | Article | Description | Status | Evidence | Gap |",
            "|----|---------|-------------|--------|:--------:|:---:|",
        ]
        for r in sec["requirements"]:
            ev = "Yes" if r["evidence_available"] else "No"
            gap = "Yes" if r["gap_identified"] else "No"
            lines.append(
                f"| {r['requirement_id']} | {r['article']} "
                f"| {r['description'][:50]} | {r['current_status']} "
                f"| {ev} | {gap} |"
            )
        return "\n".join(lines)

    def _md_gap_findings(self, data: Dict[str, Any]) -> str:
        """Render gap findings as markdown."""
        sec = self._section_gap_findings(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Gaps:** {sec['total_gaps']}\n",
        ]
        for f in sec["findings"]:
            affected = ", ".join(f["affected_claims"][:3]) or "N/A"
            lines.append(f"### {f['gap_id']} - Article {f['article']}\n")
            lines.append(f"- **Requirement:** {f['requirement']}")
            lines.append(f"- **Current State:** {f['current_state']}")
            lines.append(f"- **Required State:** {f['required_state']}")
            lines.append(f"- **Severity:** {f['severity']}")
            lines.append(f"- **Affected Claims:** {affected}\n")
        return "\n".join(lines)

    def _md_severity_analysis(self, data: Dict[str, Any]) -> str:
        """Render severity analysis as markdown."""
        sec = self._section_severity_analysis(data)
        dist = sec["severity_distribution"]
        return (
            f"## {sec['title']}\n\n"
            f"**Weighted Severity Score:** {sec['weighted_severity_score']:.1f}\n\n"
            f"| Severity | Count | Percentage |\n"
            f"|----------|------:|-----------:|\n"
            f"| Critical | {dist['critical']} | {sec['critical_pct']:.1f}% |\n"
            f"| High | {dist['high']} | {sec['high_pct']:.1f}% |\n"
            f"| Medium | {dist['medium']} | {sec['medium_pct']:.1f}% |\n"
            f"| Low | {dist['low']} | {sec['low_pct']:.1f}% |"
        )

    def _md_cross_regulation(self, data: Dict[str, Any]) -> str:
        """Render cross-regulation consistency as markdown."""
        sec = self._section_cross_regulation(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Regulations:** {', '.join(sec['regulations_checked'])}  \n"
            f"**Consistency Rate:** {sec['consistency_rate_pct']:.1f}%\n",
            "| Regulation | Requirement | Consistent | Action |",
            "|------------|-------------|:----------:|--------|",
        ]
        for f in sec["findings"]:
            consistent = "Yes" if f["consistent"] else "No"
            lines.append(
                f"| {f['regulation']} | {f['requirement'][:40]} "
                f"| {consistent} | {f['action_needed']} |"
            )
        return "\n".join(lines)

    def _md_remediation_roadmap(self, data: Dict[str, Any]) -> str:
        """Render remediation roadmap as markdown."""
        sec = self._section_remediation_roadmap(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Phases:** {sec['total_phases']}\n",
        ]
        for p in sec["phases"]:
            gaps = ", ".join(p["gaps_addressed"][:5]) or "N/A"
            lines.append(f"### Phase {p['phase']}: {p['name']}\n")
            lines.append(f"- **Timeline:** {p['start_date']} to {p['end_date']}")
            lines.append(f"- **Gaps Addressed:** {gaps}")
            lines.append(f"- **Resources:** {p['resources_required']}")
            if p["milestones"]:
                lines.append("- **Milestones:**")
                for ms in p["milestones"]:
                    lines.append(f"  - {ms}")
            lines.append("")
        return "\n".join(lines)

    def _md_cost_benefit(self, data: Dict[str, Any]) -> str:
        """Render cost-benefit analysis as markdown."""
        sec = self._section_cost_benefit(data)
        ccy = sec["currency"]
        lines = [
            f"## {sec['title']}\n",
            f"**Total Remediation Cost:** {ccy} {sec['total_remediation_cost']:,.2f}  \n"
            f"**Total Risk Exposure:** {ccy} {sec['total_risk_exposure']:,.2f}  \n"
            f"**ROI Ratio:** {sec['roi_ratio']:.2f}x\n",
            "| Gap ID | Cost | Risk Exposure | Priority | Payback |",
            "|--------|-----:|-------------:|----------|---------|",
        ]
        for i in sec["items"]:
            lines.append(
                f"| {i['gap_id']} | {ccy} {i['estimated_cost']:,.2f} "
                f"| {ccy} {i['risk_exposure']:,.2f} "
                f"| {i['priority']} | {i['payback_period']} |"
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
            ".critical{color:#b71c1c;font-weight:bold}"
            ".high{color:#c62828;font-weight:bold}"
            ".medium{color:#e65100;font-weight:bold}"
            ".compliant{color:#2e7d32;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Compliance Gap Report - EU Green Claims Directive</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> "
            f"| {data.get('reporting_period', '')}</p>"
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary HTML."""
        sec = self._section_executive_summary(data)
        css_class = "compliant" if sec["readiness_score_pct"] >= 80.0 else "critical"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{css_class}'>"
            f"Readiness: {sec['readiness_score_pct']:.1f}%</p>\n"
            f"<p><strong>Status:</strong> {sec['readiness_status']}</p>\n"
            f"<table><tr><th>Metric</th><th>Count</th></tr>"
            f"<tr><td>Total Requirements</td>"
            f"<td>{sec['total_requirements']}</td></tr>"
            f"<tr><td>Requirements Met</td>"
            f"<td>{sec['requirements_met']}</td></tr>"
            f"<tr><td>Total Gaps</td><td>{sec['total_gaps']}</td></tr>"
            f"<tr><td>Critical</td><td>{sec['critical_gaps']}</td></tr></table>"
        )

    def _html_regulatory_mapping(self, data: Dict[str, Any]) -> str:
        """Render regulatory mapping HTML."""
        sec = self._section_regulatory_mapping(data)
        rows = "".join(
            f"<tr><td>{r['requirement_id']}</td><td>{r['article']}</td>"
            f"<td>{r['description'][:50]}</td>"
            f"<td>{r['current_status']}</td></tr>"
            for r in sec["requirements"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>ID</th><th>Article</th><th>Description</th>"
            f"<th>Status</th></tr>{rows}</table>"
        )

    def _html_gap_findings(self, data: Dict[str, Any]) -> str:
        """Render gap findings HTML."""
        sec = self._section_gap_findings(data)
        rows = "".join(
            f"<tr><td>{f['gap_id']}</td><td>Art. {f['article']}</td>"
            f"<td class='{f['severity']}'>{f['severity']}</td>"
            f"<td>{f['requirement'][:50]}</td></tr>"
            for f in sec["findings"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total Gaps: {sec['total_gaps']}</p>\n"
            f"<table><tr><th>Gap ID</th><th>Article</th><th>Severity</th>"
            f"<th>Requirement</th></tr>{rows}</table>"
        )

    def _html_severity_analysis(self, data: Dict[str, Any]) -> str:
        """Render severity analysis HTML."""
        sec = self._section_severity_analysis(data)
        dist = sec["severity_distribution"]
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Weighted Score: {sec['weighted_severity_score']:.1f}</p>\n"
            f"<ul><li class='critical'>Critical: {dist['critical']}</li>"
            f"<li class='high'>High: {dist['high']}</li>"
            f"<li class='medium'>Medium: {dist['medium']}</li>"
            f"<li>Low: {dist['low']}</li></ul>"
        )

    def _html_cost_benefit(self, data: Dict[str, Any]) -> str:
        """Render cost-benefit analysis HTML."""
        sec = self._section_cost_benefit(data)
        ccy = sec["currency"]
        rows = "".join(
            f"<tr><td>{i['gap_id']}</td>"
            f"<td>{ccy} {i['estimated_cost']:,.2f}</td>"
            f"<td>{ccy} {i['risk_exposure']:,.2f}</td>"
            f"<td>{i['priority']}</td></tr>"
            for i in sec["items"][:10]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>ROI Ratio: {sec['roi_ratio']:.2f}x</p>\n"
            f"<table><tr><th>Gap</th><th>Cost</th><th>Risk</th>"
            f"<th>Priority</th></tr>{rows}</table>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_readiness(self, score: float) -> str:
        """Determine readiness status from score."""
        if score >= 95.0:
            return "Directive Ready"
        elif score >= 80.0:
            return "Near Ready (Minor Gaps)"
        elif score >= 60.0:
            return "Partially Ready (Action Required)"
        elif score >= 40.0:
            return "Not Ready (Significant Gaps)"
        else:
            return "Critical Gaps (Major Overhaul Required)"

    def _weighted_severity(self, distribution: Dict[str, int]) -> float:
        """Calculate weighted severity score (0-10 scale)."""
        weights = {"critical": 10.0, "high": 7.0, "medium": 4.0, "low": 1.0}
        total_count = sum(distribution.values())
        if total_count == 0:
            return 0.0
        weighted = sum(
            weights.get(sev, 0.0) * count
            for sev, count in distribution.items()
        )
        return round(weighted / total_count, 1)
