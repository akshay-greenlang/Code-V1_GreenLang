# -*- coding: utf-8 -*-
"""
GrievanceMechanismReportTemplate - CSDDD Grievance Mechanism Assessment Report

Renders a complaints procedure assessment report including UNGP effectiveness
criteria evaluation, channel accessibility analysis, case statistics with
resolution tracking, and improvement recommendations per CSDDD Article 11.

Regulatory References:
    - Directive (EU) 2024/1760, Article 11 (Grievance Mechanism)
    - UN Guiding Principles on Business and Human Rights, Principle 31
      (Effectiveness Criteria for Non-Judicial Grievance Mechanisms)
    - OECD Due Diligence Guidance for Responsible Business Conduct (2018)

Sections:
    1. Mechanism Assessment - Overall grievance mechanism evaluation
    2. UNGP Criteria Status - Assessment against 8 UNGP effectiveness criteria
    3. Channel Accessibility - Communication channel coverage and accessibility
    4. Case Statistics - Grievance case volumes, types, and trends
    5. Resolution Analysis - Case resolution rates and timelines
    6. Recommendations - Improvement actions for the mechanism

Author: GreenLang Team
Version: 19.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

_SECTIONS: List[str] = [
    "mechanism_assessment",
    "ungp_criteria_status",
    "channel_accessibility",
    "case_statistics",
    "resolution_analysis",
    "recommendations",
]

# UNGP Principle 31 effectiveness criteria
_UNGP_CRITERIA: List[Dict[str, str]] = [
    {"id": "legitimate", "name": "Legitimate",
     "description": "Enabling trust from stakeholder groups and accountability for fair conduct"},
    {"id": "accessible", "name": "Accessible",
     "description": "Known to all stakeholder groups, providing adequate assistance"},
    {"id": "predictable", "name": "Predictable",
     "description": "Clear and known procedure with indicative timeframes"},
    {"id": "equitable", "name": "Equitable",
     "description": "Ensuring aggrieved parties have reasonable access to information and advice"},
    {"id": "transparent", "name": "Transparent",
     "description": "Keeping parties informed about progress and providing sufficient information"},
    {"id": "rights_compatible", "name": "Rights-Compatible",
     "description": "Outcomes and remedies accord with internationally recognised human rights"},
    {"id": "continuous_learning", "name": "Source of Continuous Learning",
     "description": "Drawing on measures to identify lessons for improving the mechanism"},
    {"id": "engagement_dialogue", "name": "Based on Engagement and Dialogue",
     "description": "Consulting stakeholder groups on design and performance"},
]


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


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


class GrievanceMechanismReportTemplate:
    """
    CSDDD Grievance Mechanism Assessment Report.

    Renders a comprehensive assessment of the entity's grievance mechanism
    against CSDDD Article 11 requirements and UNGP Principle 31 effectiveness
    criteria. Includes channel accessibility scoring, case volume statistics,
    resolution rate analysis, and prioritized recommendations.

    Example:
        >>> tpl = GrievanceMechanismReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GrievanceMechanismReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = _utcnow()
        report_id = _new_uuid()
        result: Dict[str, Any] = {"report_id": report_id}
        for section in _SECTIONS:
            result[section] = self.render_section(section, data)
        result["provenance_hash"] = _compute_hash(result)
        result["generated_at"] = self.generated_at.isoformat()
        return result

    def render_section(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render a single section by name."""
        handler = getattr(self, f"_section_{name}", None)
        if handler is None:
            raise ValueError(f"Unknown section: {name}")
        return handler(data)

    def get_sections(self) -> List[str]:
        """Return list of available section names."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if "mechanism" not in data:
            warnings.append("mechanism missing; assessment will use defaults")
        if "cases" not in data:
            warnings.append("cases missing; case statistics will be empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render grievance mechanism report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_mechanism_assessment(data),
            self._md_ungp_criteria(data),
            self._md_channel_accessibility(data),
            self._md_case_statistics(data),
            self._md_resolution_analysis(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render grievance mechanism report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_mechanism_assessment(data),
            self._html_ungp_criteria(data),
            self._html_channels(data),
            self._html_case_statistics(data),
            self._html_resolution(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Grievance Mechanism Report - CSDDD</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render grievance mechanism report as JSON."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "grievance_mechanism_report",
            "directive_reference": "Directive (EU) 2024/1760, Art 11",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "mechanism_assessment": self._section_mechanism_assessment(data),
            "ungp_criteria_status": self._section_ungp_criteria_status(data),
            "channel_accessibility": self._section_channel_accessibility(data),
            "case_statistics": self._section_case_statistics(data),
            "resolution_analysis": self._section_resolution_analysis(data),
            "recommendations": self._section_recommendations(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_mechanism_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build overall mechanism assessment section."""
        mechanism = data.get("mechanism", {})
        ungp_scores = self._calculate_ungp_scores(data)
        overall_score = (
            round(sum(ungp_scores.values()) / len(ungp_scores), 1)
            if ungp_scores else 0.0
        )
        return {
            "title": "Grievance Mechanism Assessment",
            "mechanism_exists": mechanism.get("exists", False),
            "mechanism_name": mechanism.get("name", ""),
            "established_date": mechanism.get("established_date", ""),
            "scope": mechanism.get("scope", ""),
            "coverage": mechanism.get("coverage", ""),
            "covers_own_operations": mechanism.get("covers_own_operations", False),
            "covers_value_chain": mechanism.get("covers_value_chain", False),
            "covers_external_stakeholders": mechanism.get("covers_external", False),
            "overall_effectiveness_pct": overall_score,
            "effectiveness_label": self._effectiveness_label(overall_score),
            "ungp_criteria_met": sum(1 for v in ungp_scores.values() if v >= 70.0),
            "total_ungp_criteria": len(_UNGP_CRITERIA),
            "independent_oversight": mechanism.get("independent_oversight", False),
            "annual_review_conducted": mechanism.get("annual_review", False),
        }

    def _section_ungp_criteria_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build UNGP Principle 31 criteria assessment section."""
        ungp_scores = self._calculate_ungp_scores(data)
        ungp_data = data.get("ungp_criteria", {})
        criteria_results: List[Dict[str, Any]] = []
        for criterion in _UNGP_CRITERIA:
            cid = criterion["id"]
            score = ungp_scores.get(cid, 0.0)
            detail = ungp_data.get(cid, {})
            criteria_results.append({
                "criterion_id": cid,
                "criterion_name": criterion["name"],
                "description": criterion["description"],
                "score_pct": round(score, 1),
                "status": self._criterion_status(score),
                "evidence": detail.get("evidence", []),
                "gaps": detail.get("gaps", []),
                "notes": detail.get("notes", ""),
            })
        met_count = sum(1 for c in criteria_results if c["score_pct"] >= 70.0)
        return {
            "title": "UNGP Principle 31 Effectiveness Criteria",
            "total_criteria": len(criteria_results),
            "criteria_met": met_count,
            "criteria_partially_met": sum(
                1 for c in criteria_results if 40.0 <= c["score_pct"] < 70.0
            ),
            "criteria_not_met": sum(
                1 for c in criteria_results if c["score_pct"] < 40.0
            ),
            "criteria": criteria_results,
        }

    def _section_channel_accessibility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build channel accessibility section."""
        channels = data.get("channels", [])
        entries: List[Dict[str, Any]] = []
        for ch in channels:
            entries.append({
                "channel_type": ch.get("type", ""),
                "name": ch.get("name", ""),
                "languages_supported": ch.get("languages", []),
                "available_24_7": ch.get("available_24_7", False),
                "anonymous_reporting": ch.get("anonymous", False),
                "accessible_to_disabled": ch.get("accessible_disabled", False),
                "digital": ch.get("digital", False),
                "cost_free": ch.get("cost_free", True),
                "average_response_time_days": ch.get("avg_response_days", 0),
                "usage_count": ch.get("usage_count", 0),
                "satisfaction_score": round(ch.get("satisfaction_score", 0.0), 1),
            })
        total_languages = len(set(
            lang for ch in entries for lang in ch["languages_supported"]
        ))
        return {
            "title": "Channel Accessibility Assessment",
            "total_channels": len(entries),
            "channels": entries,
            "total_languages": total_languages,
            "anonymous_channels": sum(1 for c in entries if c["anonymous_reporting"]),
            "digital_channels": sum(1 for c in entries if c["digital"]),
            "channels_24_7": sum(1 for c in entries if c["available_24_7"]),
            "cost_free_channels": sum(1 for c in entries if c["cost_free"]),
        }

    def _section_case_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build case statistics section."""
        cases = data.get("cases", [])
        total = len(cases)
        open_cases = sum(1 for c in cases if c.get("status") == "open")
        closed_cases = sum(1 for c in cases if c.get("status") == "closed")
        in_progress = sum(1 for c in cases if c.get("status") == "in_progress")
        type_counts: Dict[str, int] = {}
        for c in cases:
            ct = c.get("case_type", "unclassified")
            type_counts[ct] = type_counts.get(ct, 0) + 1
        domain_counts: Dict[str, int] = {}
        for c in cases:
            domain = c.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        source_counts: Dict[str, int] = {}
        for c in cases:
            src = c.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        prior_year = data.get("prior_year_cases", 0)
        yoy_change = (
            round((total - prior_year) / prior_year * 100, 1)
            if prior_year > 0 else 0.0
        )
        return {
            "title": "Case Statistics",
            "total_cases": total,
            "open_cases": open_cases,
            "closed_cases": closed_cases,
            "in_progress_cases": in_progress,
            "case_type_distribution": type_counts,
            "domain_distribution": domain_counts,
            "source_distribution": source_counts,
            "prior_year_total": prior_year,
            "year_over_year_change_pct": yoy_change,
        }

    def _section_resolution_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build resolution analysis section."""
        cases = data.get("cases", [])
        closed = [c for c in cases if c.get("status") == "closed"]
        resolution_days = [
            c.get("resolution_days", 0) for c in closed if c.get("resolution_days")
        ]
        avg_days = (
            round(sum(resolution_days) / len(resolution_days), 1)
            if resolution_days else 0.0
        )
        max_days = max(resolution_days) if resolution_days else 0
        min_days = min(resolution_days) if resolution_days else 0
        outcome_counts: Dict[str, int] = {}
        for c in closed:
            outcome = c.get("outcome", "unclassified")
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        satisfied = sum(1 for c in closed if c.get("complainant_satisfied", False))
        satisfaction_pct = (
            round(satisfied / len(closed) * 100, 1) if closed else 0.0
        )
        within_sla = sum(
            1 for c in closed
            if c.get("resolution_days", 0) <= c.get("sla_days", 30)
        )
        sla_compliance = (
            round(within_sla / len(closed) * 100, 1) if closed else 0.0
        )
        return {
            "title": "Resolution Analysis",
            "total_resolved": len(closed),
            "average_resolution_days": avg_days,
            "max_resolution_days": max_days,
            "min_resolution_days": min_days,
            "resolution_rate_pct": round(
                len(closed) / len(cases) * 100, 1
            ) if cases else 0.0,
            "outcome_distribution": outcome_counts,
            "complainant_satisfaction_pct": satisfaction_pct,
            "sla_compliance_pct": sla_compliance,
            "remediation_provided": sum(
                1 for c in closed if c.get("remediation_provided", False)
            ),
        }

    def _section_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build recommendations section."""
        recommendations = data.get("gm_recommendations", [])
        if not recommendations:
            recommendations = self._generate_default_recommendations(data)
        return {
            "title": "Improvement Recommendations",
            "total_recommendations": len(recommendations),
            "recommendations": [
                {
                    "priority": r.get("priority", "medium"),
                    "area": r.get("area", ""),
                    "action": r.get("action", ""),
                    "ungp_criterion": r.get("ungp_criterion", ""),
                    "timeline": r.get("timeline", ""),
                    "expected_impact": r.get("expected_impact", ""),
                }
                for r in recommendations
            ],
        }

    # ------------------------------------------------------------------
    # Calculation helpers
    # ------------------------------------------------------------------

    def _calculate_ungp_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate scores for each UNGP criterion."""
        ungp_data = data.get("ungp_criteria", {})
        scores: Dict[str, float] = {}
        for criterion in _UNGP_CRITERIA:
            cid = criterion["id"]
            detail = ungp_data.get(cid, {})
            scores[cid] = round(detail.get("score_pct", 0.0), 1)
        return scores

    def _effectiveness_label(self, score: float) -> str:
        """Map effectiveness score to label."""
        if score >= 85.0:
            return "Highly Effective"
        elif score >= 70.0:
            return "Effective"
        elif score >= 50.0:
            return "Partially Effective"
        elif score >= 25.0:
            return "Weak"
        else:
            return "Not Effective"

    def _criterion_status(self, score: float) -> str:
        """Map criterion score to met/partial/not met."""
        if score >= 70.0:
            return "met"
        elif score >= 40.0:
            return "partially_met"
        else:
            return "not_met"

    def _generate_default_recommendations(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate default recommendations from assessment gaps."""
        recs: List[Dict[str, Any]] = []
        mechanism = data.get("mechanism", {})
        if not mechanism.get("exists", False):
            recs.append({
                "priority": "critical",
                "area": "mechanism_establishment",
                "action": (
                    "Establish a formal grievance mechanism in compliance with "
                    "CSDDD Article 11 and UNGP Principle 31 effectiveness criteria"
                ),
                "ungp_criterion": "all",
                "timeline": "0-6 months",
                "expected_impact": "Regulatory compliance and stakeholder trust",
            })
            return recs
        ungp_scores = self._calculate_ungp_scores(data)
        for criterion in _UNGP_CRITERIA:
            cid = criterion["id"]
            score = ungp_scores.get(cid, 0.0)
            if score < 40.0:
                recs.append({
                    "priority": "high",
                    "area": cid,
                    "action": (
                        f"Strengthen the '{criterion['name']}' criterion: "
                        f"{criterion['description']}"
                    ),
                    "ungp_criterion": cid,
                    "timeline": "3-6 months",
                    "expected_impact": f"Improve {criterion['name']} score from {score:.0f}%",
                })
        channels = data.get("channels", [])
        if not any(ch.get("anonymous", False) for ch in channels):
            recs.append({
                "priority": "high",
                "area": "anonymous_reporting",
                "action": "Implement anonymous reporting channel to reduce fear of retaliation",
                "ungp_criterion": "accessible",
                "timeline": "3-6 months",
                "expected_impact": "Increased reporting from vulnerable stakeholders",
            })
        if len(channels) < 3:
            recs.append({
                "priority": "medium",
                "area": "channel_diversity",
                "action": "Add additional reporting channels (e.g., hotline, web portal, in-person)",
                "ungp_criterion": "accessible",
                "timeline": "3-6 months",
                "expected_impact": "Improved accessibility for diverse stakeholder groups",
            })
        return recs

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Grievance Mechanism Report - CSDDD\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Reference:** Directive (EU) 2024/1760, Art 11 | UNGP Principle 31"
        )

    def _md_mechanism_assessment(self, data: Dict[str, Any]) -> str:
        """Render mechanism assessment as markdown."""
        sec = self._section_mechanism_assessment(data)
        exists = "Yes" if sec["mechanism_exists"] else "No"
        vc = "Yes" if sec["covers_value_chain"] else "No"
        ext = "Yes" if sec["covers_external_stakeholders"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"**Mechanism Exists:** {exists}  \n"
            f"**Name:** {sec['mechanism_name']}  \n"
            f"**Established:** {sec['established_date']}\n\n"
            f"| Metric | Value |\n|--------|------:|\n"
            f"| Overall Effectiveness | {sec['overall_effectiveness_pct']:.1f}% |\n"
            f"| Effectiveness Label | {sec['effectiveness_label']} |\n"
            f"| UNGP Criteria Met | {sec['ungp_criteria_met']} / {sec['total_ungp_criteria']} |\n"
            f"| Covers Value Chain | {vc} |\n"
            f"| Covers External Stakeholders | {ext} |"
        )

    def _md_ungp_criteria(self, data: Dict[str, Any]) -> str:
        """Render UNGP criteria as markdown."""
        sec = self._section_ungp_criteria_status(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Met:** {sec['criteria_met']} | "
            f"**Partial:** {sec['criteria_partially_met']} | "
            f"**Not Met:** {sec['criteria_not_met']}\n",
            "| Criterion | Score | Status |",
            "|-----------|------:|--------|",
        ]
        for c in sec["criteria"]:
            status_display = c["status"].replace("_", " ").title()
            lines.append(
                f"| {c['criterion_name']} | {c['score_pct']:.1f}% | {status_display} |"
            )
        return "\n".join(lines)

    def _md_channel_accessibility(self, data: Dict[str, Any]) -> str:
        """Render channel accessibility as markdown."""
        sec = self._section_channel_accessibility(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Channels:** {sec['total_channels']}  \n"
            f"**Languages:** {sec['total_languages']}  \n"
            f"**Anonymous Channels:** {sec['anonymous_channels']}  \n"
            f"**24/7 Channels:** {sec['channels_24_7']}\n",
            "| Channel | Type | Anonymous | 24/7 | Languages | Avg Response |",
            "|---------|------|:---------:|:----:|----------:|-------------:|",
        ]
        for ch in sec["channels"]:
            anon = "Yes" if ch["anonymous_reporting"] else "No"
            avail = "Yes" if ch["available_24_7"] else "No"
            langs = len(ch["languages_supported"])
            lines.append(
                f"| {ch['name']} | {ch['channel_type']} | {anon} | {avail} | "
                f"{langs} | {ch['average_response_time_days']}d |"
            )
        return "\n".join(lines)

    def _md_case_statistics(self, data: Dict[str, Any]) -> str:
        """Render case statistics as markdown."""
        sec = self._section_case_statistics(data)
        lines = [
            f"## {sec['title']}\n",
            f"| Metric | Value |\n|--------|------:|\n"
            f"| Total Cases | {sec['total_cases']} |\n"
            f"| Open | {sec['open_cases']} |\n"
            f"| In Progress | {sec['in_progress_cases']} |\n"
            f"| Closed | {sec['closed_cases']} |\n"
            f"| Prior Year | {sec['prior_year_total']} |\n"
            f"| YoY Change | {sec['year_over_year_change_pct']:.1f}% |",
        ]
        if sec["case_type_distribution"]:
            lines.append("\n**By Type:**")
            for ct, count in sec["case_type_distribution"].items():
                lines.append(f"- {ct.replace('_', ' ').title()}: {count}")
        if sec["domain_distribution"]:
            lines.append("\n**By Domain:**")
            for domain, count in sec["domain_distribution"].items():
                lines.append(f"- {domain.replace('_', ' ').title()}: {count}")
        return "\n".join(lines)

    def _md_resolution_analysis(self, data: Dict[str, Any]) -> str:
        """Render resolution analysis as markdown."""
        sec = self._section_resolution_analysis(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Metric | Value |\n|--------|------:|\n"
            f"| Total Resolved | {sec['total_resolved']} |\n"
            f"| Resolution Rate | {sec['resolution_rate_pct']:.1f}% |\n"
            f"| Avg Resolution Time | {sec['average_resolution_days']:.1f} days |\n"
            f"| Min Resolution Time | {sec['min_resolution_days']} days |\n"
            f"| Max Resolution Time | {sec['max_resolution_days']} days |\n"
            f"| Satisfaction Rate | {sec['complainant_satisfaction_pct']:.1f}% |\n"
            f"| SLA Compliance | {sec['sla_compliance_pct']:.1f}% |\n"
            f"| Remediation Provided | {sec['remediation_provided']} |"
        )

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations as markdown."""
        sec = self._section_recommendations(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Recommendations:** {sec['total_recommendations']}\n",
            "| Priority | Area | Action | UNGP | Timeline |",
            "|----------|------|--------|------|----------|",
        ]
        for r in sec["recommendations"]:
            lines.append(
                f"| {r['priority'].upper()} | {r['area']} | "
                f"{r['action'][:50]} | {r['ungp_criterion']} | {r['timeline']} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-019 CSDDD Readiness Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1100px;margin:auto}"
            "h1{color:#1a237e;border-bottom:2px solid #1a237e;padding-bottom:.3em}"
            "h2{color:#283593;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8eaf6}"
            ".met{color:#2e7d32;font-weight:bold}"
            ".not-met{color:#c62828;font-weight:bold}"
            ".partial{color:#ef6c00}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Grievance Mechanism Report - CSDDD</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('reporting_year', '')}</p>"
        )

    def _html_mechanism_assessment(self, data: Dict[str, Any]) -> str:
        """Render mechanism assessment HTML."""
        sec = self._section_mechanism_assessment(data)
        eff_class = "met" if sec["overall_effectiveness_pct"] >= 70.0 else "not-met"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{eff_class}'>Effectiveness: "
            f"{sec['overall_effectiveness_pct']:.1f}% "
            f"({sec['effectiveness_label']})</p>\n"
            f"<p>UNGP Criteria Met: {sec['ungp_criteria_met']} / "
            f"{sec['total_ungp_criteria']}</p>"
        )

    def _html_ungp_criteria(self, data: Dict[str, Any]) -> str:
        """Render UNGP criteria HTML."""
        sec = self._section_ungp_criteria_status(data)
        rows = ""
        for c in sec["criteria"]:
            css = "met" if c["status"] == "met" else (
                "partial" if c["status"] == "partially_met" else "not-met"
            )
            rows += (
                f"<tr class='{css}'><td>{c['criterion_name']}</td>"
                f"<td>{c['score_pct']:.1f}%</td>"
                f"<td>{c['status'].replace('_', ' ').title()}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Criterion</th><th>Score</th><th>Status</th></tr>"
            f"{rows}</table>"
        )

    def _html_channels(self, data: Dict[str, Any]) -> str:
        """Render channel accessibility HTML."""
        sec = self._section_channel_accessibility(data)
        rows = "".join(
            f"<tr><td>{ch['name']}</td><td>{ch['channel_type']}</td>"
            f"<td>{'Yes' if ch['anonymous_reporting'] else 'No'}</td>"
            f"<td>{ch['average_response_time_days']}d</td></tr>"
            for ch in sec["channels"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Channels: {sec['total_channels']} | "
            f"Languages: {sec['total_languages']}</p>\n"
            f"<table><tr><th>Channel</th><th>Type</th><th>Anonymous</th>"
            f"<th>Avg Response</th></tr>{rows}</table>"
        )

    def _html_case_statistics(self, data: Dict[str, Any]) -> str:
        """Render case statistics HTML."""
        sec = self._section_case_statistics(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Status</th><th>Count</th></tr>"
            f"<tr><td>Open</td><td>{sec['open_cases']}</td></tr>"
            f"<tr><td>In Progress</td><td>{sec['in_progress_cases']}</td></tr>"
            f"<tr><td>Closed</td><td>{sec['closed_cases']}</td></tr>"
            f"<tr><td><strong>Total</strong></td>"
            f"<td><strong>{sec['total_cases']}</strong></td></tr></table>"
        )

    def _html_resolution(self, data: Dict[str, Any]) -> str:
        """Render resolution analysis HTML."""
        sec = self._section_resolution_analysis(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Resolution Rate</td><td>{sec['resolution_rate_pct']:.1f}%</td></tr>"
            f"<tr><td>Avg Resolution Days</td><td>{sec['average_resolution_days']:.1f}</td></tr>"
            f"<tr><td>Satisfaction</td><td>{sec['complainant_satisfaction_pct']:.1f}%</td></tr>"
            f"<tr><td>SLA Compliance</td><td>{sec['sla_compliance_pct']:.1f}%</td></tr>"
            f"</table>"
        )
