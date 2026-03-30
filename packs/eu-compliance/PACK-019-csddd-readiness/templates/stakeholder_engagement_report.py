# -*- coding: utf-8 -*-
"""
StakeholderEngagementReportTemplate - CSDDD Stakeholder Engagement Report

Renders stakeholder engagement activities, consultation outcomes, group
coverage analysis, and engagement quality assessment per CSDDD Article 10
requirements for meaningful engagement with affected stakeholders.

Regulatory References:
    - Directive (EU) 2024/1760, Article 10 (Meaningful Engagement with Stakeholders)
    - Directive (EU) 2024/1760, Article 6 (Identifying Adverse Impacts)
    - Directive (EU) 2024/1760, Article 8 (Prevention)
    - UN Guiding Principles on Business and Human Rights (2011)
    - AA1000 Stakeholder Engagement Standard (2015)

Sections:
    1. Engagement Overview - Summary of stakeholder engagement programme
    2. Group Coverage - Stakeholder groups identified and engaged
    3. Quality Assessment - Engagement quality scoring against AA1000
    4. Activity Log - Detailed engagement activity register
    5. Outcomes Summary - Key outcomes from consultation activities
    6. Recommendations - Improvement actions for stakeholder engagement

Author: GreenLang Team
Version: 19.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

_SECTIONS: List[str] = [
    "engagement_overview",
    "group_coverage",
    "quality_assessment",
    "activity_log",
    "outcomes_summary",
    "recommendations",
]

# Stakeholder group categories per CSDDD
_STAKEHOLDER_GROUPS: List[str] = [
    "own_workforce",
    "value_chain_workers",
    "affected_communities",
    "indigenous_peoples",
    "civil_society_organisations",
    "trade_unions",
    "human_rights_defenders",
    "consumers",
    "investors",
    "regulators",
]

# AA1000 quality principles
_AA1000_PRINCIPLES: List[Dict[str, str]] = [
    {"id": "inclusivity", "name": "Inclusivity",
     "description": "Including stakeholders in developing and achieving accountability"},
    {"id": "materiality", "name": "Materiality",
     "description": "Determining relevance and significance of issues to stakeholders"},
    {"id": "responsiveness", "name": "Responsiveness",
     "description": "Responding to stakeholder issues that affect performance"},
    {"id": "impact", "name": "Impact",
     "description": "Monitoring, measuring and being accountable for impact on stakeholders"},
]

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

class StakeholderEngagementReportTemplate:
    """
    CSDDD Stakeholder Engagement Report.

    Renders a comprehensive stakeholder engagement assessment covering
    group identification and coverage, engagement quality scoring against
    AA1000 principles, activity logging with participation tracking, and
    outcome-based analysis of consultation effectiveness per CSDDD Art 10.

    Example:
        >>> tpl = StakeholderEngagementReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize StakeholderEngagementReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = utcnow()
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
        if "stakeholder_groups" not in data:
            warnings.append("stakeholder_groups missing; coverage section will be empty")
        if "activities" not in data:
            warnings.append("activities missing; activity log will be empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render stakeholder engagement report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_engagement_overview(data),
            self._md_group_coverage(data),
            self._md_quality_assessment(data),
            self._md_activity_log(data),
            self._md_outcomes(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render stakeholder engagement report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_engagement_overview(data),
            self._html_group_coverage(data),
            self._html_quality(data),
            self._html_activities(data),
            self._html_outcomes(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Stakeholder Engagement Report - CSDDD</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render stakeholder engagement report as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "stakeholder_engagement_report",
            "directive_reference": "Directive (EU) 2024/1760, Art 10",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "engagement_overview": self._section_engagement_overview(data),
            "group_coverage": self._section_group_coverage(data),
            "quality_assessment": self._section_quality_assessment(data),
            "activity_log": self._section_activity_log(data),
            "outcomes_summary": self._section_outcomes_summary(data),
            "recommendations": self._section_recommendations(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_engagement_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build engagement overview section."""
        groups = data.get("stakeholder_groups", [])
        activities = data.get("activities", [])
        quality = data.get("quality_scores", {})
        principle_scores = [
            quality.get(p["id"], {}).get("score_pct", 0.0)
            for p in _AA1000_PRINCIPLES
        ]
        avg_quality = (
            round(sum(principle_scores) / len(principle_scores), 1)
            if principle_scores else 0.0
        )
        total_participants = sum(a.get("participants", 0) for a in activities)
        engaged_groups = set(
            g.get("group_type", "") for g in groups if g.get("engaged", False)
        )
        return {
            "title": "Stakeholder Engagement Overview",
            "total_stakeholder_groups": len(groups),
            "groups_engaged": len(engaged_groups),
            "engagement_coverage_pct": round(
                len(engaged_groups) / len(_STAKEHOLDER_GROUPS) * 100, 1
            ) if _STAKEHOLDER_GROUPS else 0.0,
            "total_activities": len(activities),
            "total_participants": total_participants,
            "average_quality_score_pct": avg_quality,
            "quality_label": self._quality_label(avg_quality),
            "engagement_policy_exists": data.get("engagement_policy_exists", False),
            "dedicated_engagement_team": data.get("dedicated_team", False),
            "budget_eur": round(data.get("engagement_budget_eur", 0.0), 2),
        }

    def _section_group_coverage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build stakeholder group coverage section."""
        groups = data.get("stakeholder_groups", [])
        group_entries: List[Dict[str, Any]] = []
        for g in groups:
            group_entries.append({
                "group_type": g.get("group_type", ""),
                "group_name": g.get("group_name", ""),
                "estimated_size": g.get("estimated_size", 0),
                "engaged": g.get("engaged", False),
                "engagement_frequency": g.get("frequency", ""),
                "engagement_methods": g.get("methods", []),
                "vulnerability_level": g.get("vulnerability_level", "medium"),
                "geographic_location": g.get("location", ""),
                "key_concerns": g.get("key_concerns", []),
                "last_engagement_date": g.get("last_engagement_date", ""),
            })
        covered = set(g["group_type"] for g in group_entries if g["engaged"])
        uncovered = [gt for gt in _STAKEHOLDER_GROUPS if gt not in covered]
        return {
            "title": "Stakeholder Group Coverage",
            "total_groups_identified": len(group_entries),
            "groups_engaged": sum(1 for g in group_entries if g["engaged"]),
            "groups_not_engaged": sum(1 for g in group_entries if not g["engaged"]),
            "coverage_pct": round(
                len(covered) / len(_STAKEHOLDER_GROUPS) * 100, 1
            ) if _STAKEHOLDER_GROUPS else 0.0,
            "groups": group_entries,
            "uncovered_standard_groups": uncovered,
            "vulnerable_groups_engaged": sum(
                1 for g in group_entries
                if g["engaged"] and g["vulnerability_level"] in ("high", "critical")
            ),
        }

    def _section_quality_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build engagement quality assessment section."""
        quality = data.get("quality_scores", {})
        principle_results: List[Dict[str, Any]] = []
        for principle in _AA1000_PRINCIPLES:
            pid = principle["id"]
            detail = quality.get(pid, {})
            score = round(detail.get("score_pct", 0.0), 1)
            principle_results.append({
                "principle_id": pid,
                "principle_name": principle["name"],
                "description": principle["description"],
                "score_pct": score,
                "status": self._principle_status(score),
                "evidence": detail.get("evidence", []),
                "improvement_areas": detail.get("improvement_areas", []),
            })
        scores = [p["score_pct"] for p in principle_results]
        overall = round(sum(scores) / len(scores), 1) if scores else 0.0
        return {
            "title": "Engagement Quality Assessment (AA1000)",
            "overall_quality_pct": overall,
            "quality_label": self._quality_label(overall),
            "principles": principle_results,
            "principles_met": sum(1 for p in principle_results if p["score_pct"] >= 70.0),
            "total_principles": len(principle_results),
        }

    def _section_activity_log(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build engagement activity log section."""
        activities = data.get("activities", [])
        entries: List[Dict[str, Any]] = []
        for a in activities:
            entries.append({
                "activity_id": a.get("activity_id", ""),
                "activity_type": a.get("type", ""),
                "title": a.get("title", ""),
                "date": a.get("date", ""),
                "stakeholder_groups": a.get("stakeholder_groups", []),
                "participants": a.get("participants", 0),
                "location": a.get("location", ""),
                "format": a.get("format", ""),
                "topics_discussed": a.get("topics", []),
                "outcomes": a.get("outcomes", []),
                "follow_up_actions": a.get("follow_up_actions", []),
                "satisfaction_score": round(a.get("satisfaction_score", 0.0), 1),
            })
        type_counts: Dict[str, int] = {}
        for e in entries:
            at = e["activity_type"]
            type_counts[at] = type_counts.get(at, 0) + 1
        return {
            "title": "Engagement Activity Log",
            "total_activities": len(entries),
            "total_participants": sum(e["participants"] for e in entries),
            "activity_type_distribution": type_counts,
            "activities": entries,
            "average_satisfaction": round(
                sum(e["satisfaction_score"] for e in entries) / len(entries), 1
            ) if entries else 0.0,
        }

    def _section_outcomes_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build engagement outcomes summary section."""
        outcomes = data.get("engagement_outcomes", [])
        activities = data.get("activities", [])
        all_outcomes: List[Dict[str, Any]] = []
        for o in outcomes:
            all_outcomes.append({
                "outcome_id": o.get("outcome_id", ""),
                "description": o.get("description", ""),
                "source_activity": o.get("source_activity", ""),
                "stakeholder_group": o.get("stakeholder_group", ""),
                "outcome_type": o.get("outcome_type", ""),
                "action_taken": o.get("action_taken", ""),
                "status": o.get("status", "pending"),
                "impact_on_dd": o.get("impact_on_dd", ""),
            })
        for a in activities:
            for outcome in a.get("outcomes", []):
                if isinstance(outcome, dict):
                    all_outcomes.append({
                        "outcome_id": outcome.get("outcome_id", ""),
                        "description": outcome.get("description", ""),
                        "source_activity": a.get("title", ""),
                        "stakeholder_group": ", ".join(a.get("stakeholder_groups", [])),
                        "outcome_type": outcome.get("type", "feedback"),
                        "action_taken": outcome.get("action_taken", ""),
                        "status": outcome.get("status", "pending"),
                        "impact_on_dd": outcome.get("impact_on_dd", ""),
                    })
        type_counts: Dict[str, int] = {}
        for o in all_outcomes:
            ot = o["outcome_type"]
            type_counts[ot] = type_counts.get(ot, 0) + 1
        actioned = sum(1 for o in all_outcomes if o["status"] in ("completed", "in_progress"))
        return {
            "title": "Engagement Outcomes Summary",
            "total_outcomes": len(all_outcomes),
            "outcomes_actioned": actioned,
            "action_rate_pct": round(
                actioned / len(all_outcomes) * 100, 1
            ) if all_outcomes else 0.0,
            "outcome_type_distribution": type_counts,
            "outcomes": all_outcomes,
            "outcomes_influencing_dd": sum(
                1 for o in all_outcomes if o["impact_on_dd"]
            ),
        }

    def _section_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build recommendations section."""
        recommendations = data.get("se_recommendations", [])
        if not recommendations:
            recommendations = self._generate_default_recommendations(data)
        return {
            "title": "Engagement Improvement Recommendations",
            "total_recommendations": len(recommendations),
            "recommendations": [
                {
                    "priority": r.get("priority", "medium"),
                    "area": r.get("area", ""),
                    "action": r.get("action", ""),
                    "target_groups": r.get("target_groups", []),
                    "timeline": r.get("timeline", ""),
                    "expected_improvement": r.get("expected_improvement", ""),
                }
                for r in recommendations
            ],
        }

    # ------------------------------------------------------------------
    # Calculation helpers
    # ------------------------------------------------------------------

    def _quality_label(self, score: float) -> str:
        """Map quality score to label."""
        if score >= 85.0:
            return "Excellent"
        elif score >= 70.0:
            return "Good"
        elif score >= 50.0:
            return "Adequate"
        elif score >= 30.0:
            return "Needs Improvement"
        else:
            return "Insufficient"

    def _principle_status(self, score: float) -> str:
        """Map principle score to status."""
        if score >= 70.0:
            return "met"
        elif score >= 40.0:
            return "partially_met"
        else:
            return "not_met"

    def _generate_default_recommendations(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate default recommendations from coverage gaps."""
        recs: List[Dict[str, Any]] = []
        coverage = self._section_group_coverage(data)
        if coverage["uncovered_standard_groups"]:
            recs.append({
                "priority": "high",
                "area": "coverage_gaps",
                "action": (
                    f"Establish engagement with uncovered groups: "
                    f"{', '.join(g.replace('_', ' ').title() for g in coverage['uncovered_standard_groups'][:5])}"
                ),
                "target_groups": coverage["uncovered_standard_groups"][:5],
                "timeline": "3-6 months",
                "expected_improvement": "Complete CSDDD Art 10 stakeholder coverage",
            })
        quality = self._section_quality_assessment(data)
        for p in quality["principles"]:
            if p["score_pct"] < 50.0:
                recs.append({
                    "priority": "high",
                    "area": p["principle_id"],
                    "action": (
                        f"Improve '{p['principle_name']}' engagement quality: "
                        f"{p['description']}"
                    ),
                    "target_groups": [],
                    "timeline": "3-6 months",
                    "expected_improvement": f"Raise {p['principle_name']} score above 70%",
                })
        vulnerable_not_engaged = [
            g for g in data.get("stakeholder_groups", [])
            if g.get("vulnerability_level") in ("high", "critical")
            and not g.get("engaged", False)
        ]
        if vulnerable_not_engaged:
            recs.append({
                "priority": "critical",
                "area": "vulnerable_groups",
                "action": "Prioritise engagement with vulnerable and at-risk stakeholder groups",
                "target_groups": [g.get("group_type", "") for g in vulnerable_not_engaged],
                "timeline": "0-3 months",
                "expected_improvement": "CSDDD compliance for most-affected stakeholders",
            })
        if not data.get("engagement_policy_exists", False):
            recs.append({
                "priority": "high",
                "area": "policy",
                "action": "Develop formal stakeholder engagement policy aligned with CSDDD Art 10",
                "target_groups": [],
                "timeline": "1-3 months",
                "expected_improvement": "Formalised engagement governance",
            })
        return recs

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Stakeholder Engagement Report - CSDDD\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Reference:** Directive (EU) 2024/1760, Art 10 | AA1000 SES"
        )

    def _md_engagement_overview(self, data: Dict[str, Any]) -> str:
        """Render engagement overview as markdown."""
        sec = self._section_engagement_overview(data)
        policy = "Yes" if sec["engagement_policy_exists"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"| Metric | Value |\n|--------|------:|\n"
            f"| Stakeholder Groups | {sec['total_stakeholder_groups']} |\n"
            f"| Groups Engaged | {sec['groups_engaged']} |\n"
            f"| Coverage | {sec['engagement_coverage_pct']:.1f}% |\n"
            f"| Total Activities | {sec['total_activities']} |\n"
            f"| Total Participants | {sec['total_participants']:,} |\n"
            f"| Quality Score | {sec['average_quality_score_pct']:.1f}% ({sec['quality_label']}) |\n"
            f"| Engagement Policy | {policy} |\n"
            f"| Budget (EUR) | {sec['budget_eur']:,.2f} |"
        )

    def _md_group_coverage(self, data: Dict[str, Any]) -> str:
        """Render group coverage as markdown."""
        sec = self._section_group_coverage(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Coverage:** {sec['coverage_pct']:.1f}% "
            f"({sec['groups_engaged']} engaged / "
            f"{sec['total_groups_identified']} identified)\n",
            "| Group | Type | Size | Engaged | Vulnerability | Frequency |",
            "|-------|------|-----:|:-------:|---------------|-----------|",
        ]
        for g in sec["groups"]:
            engaged = "Yes" if g["engaged"] else "No"
            lines.append(
                f"| {g['group_name']} | {g['group_type']} | "
                f"{g['estimated_size']:,} | {engaged} | "
                f"{g['vulnerability_level']} | {g['engagement_frequency']} |"
            )
        if sec["uncovered_standard_groups"]:
            lines.append("\n**Uncovered Standard Groups:**")
            for grp in sec["uncovered_standard_groups"]:
                lines.append(f"- {grp.replace('_', ' ').title()}")
        return "\n".join(lines)

    def _md_quality_assessment(self, data: Dict[str, Any]) -> str:
        """Render quality assessment as markdown."""
        sec = self._section_quality_assessment(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Overall Quality:** {sec['overall_quality_pct']:.1f}% "
            f"({sec['quality_label']})  \n"
            f"**Principles Met:** {sec['principles_met']} / {sec['total_principles']}\n",
            "| Principle | Score | Status |",
            "|-----------|------:|--------|",
        ]
        for p in sec["principles"]:
            lines.append(
                f"| {p['principle_name']} | {p['score_pct']:.1f}% | "
                f"{p['status'].replace('_', ' ').title()} |"
            )
        return "\n".join(lines)

    def _md_activity_log(self, data: Dict[str, Any]) -> str:
        """Render activity log as markdown."""
        sec = self._section_activity_log(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Activities:** {sec['total_activities']}  \n"
            f"**Total Participants:** {sec['total_participants']:,}  \n"
            f"**Average Satisfaction:** {sec['average_satisfaction']:.1f} / 5.0\n",
            "| Date | Activity | Type | Groups | Participants | Satisfaction |",
            "|------|----------|------|--------|------------:|-----------:|",
        ]
        for a in sec["activities"]:
            groups = ", ".join(a["stakeholder_groups"][:2])
            lines.append(
                f"| {a['date']} | {a['title'][:30]} | {a['activity_type']} | "
                f"{groups} | {a['participants']} | {a['satisfaction_score']:.1f} |"
            )
        return "\n".join(lines)

    def _md_outcomes(self, data: Dict[str, Any]) -> str:
        """Render outcomes summary as markdown."""
        sec = self._section_outcomes_summary(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Outcomes:** {sec['total_outcomes']}  \n"
            f"**Actioned:** {sec['outcomes_actioned']} "
            f"({sec['action_rate_pct']:.1f}%)  \n"
            f"**Influencing DD Processes:** {sec['outcomes_influencing_dd']}\n",
        ]
        if sec["outcomes"]:
            lines.append("| Outcome | Source | Group | Status | DD Impact |")
            lines.append("|---------|--------|-------|--------|-----------|")
            for o in sec["outcomes"][:15]:
                lines.append(
                    f"| {o['description'][:40]} | {o['source_activity'][:20]} | "
                    f"{o['stakeholder_group'][:15]} | {o['status']} | "
                    f"{'Yes' if o['impact_on_dd'] else 'No'} |"
                )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations as markdown."""
        sec = self._section_recommendations(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Recommendations:** {sec['total_recommendations']}\n",
            "| Priority | Area | Action | Timeline |",
            "|----------|------|--------|----------|",
        ]
        for r in sec["recommendations"]:
            lines.append(
                f"| {r['priority'].upper()} | {r['area']} | "
                f"{r['action'][:60]} | {r['timeline']} |"
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
            f"<h1>Stakeholder Engagement Report - CSDDD</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('reporting_year', '')}</p>"
        )

    def _html_engagement_overview(self, data: Dict[str, Any]) -> str:
        """Render engagement overview HTML."""
        sec = self._section_engagement_overview(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Groups Engaged</td><td>{sec['groups_engaged']}</td></tr>"
            f"<tr><td>Coverage</td><td>{sec['engagement_coverage_pct']:.1f}%</td></tr>"
            f"<tr><td>Activities</td><td>{sec['total_activities']}</td></tr>"
            f"<tr><td>Participants</td><td>{sec['total_participants']:,}</td></tr>"
            f"<tr><td>Quality</td><td>{sec['average_quality_score_pct']:.1f}%</td></tr>"
            f"</table>"
        )

    def _html_group_coverage(self, data: Dict[str, Any]) -> str:
        """Render group coverage HTML."""
        sec = self._section_group_coverage(data)
        rows = ""
        for g in sec["groups"]:
            css = "met" if g["engaged"] else "not-met"
            rows += (
                f"<tr><td>{g['group_name']}</td>"
                f"<td>{g['group_type']}</td>"
                f"<td class='{css}'>{'Yes' if g['engaged'] else 'No'}</td>"
                f"<td>{g['vulnerability_level']}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Coverage: {sec['coverage_pct']:.1f}%</p>\n"
            f"<table><tr><th>Group</th><th>Type</th><th>Engaged</th>"
            f"<th>Vulnerability</th></tr>{rows}</table>"
        )

    def _html_quality(self, data: Dict[str, Any]) -> str:
        """Render quality assessment HTML."""
        sec = self._section_quality_assessment(data)
        rows = ""
        for p in sec["principles"]:
            css = "met" if p["status"] == "met" else (
                "partial" if p["status"] == "partially_met" else "not-met"
            )
            rows += (
                f"<tr class='{css}'><td>{p['principle_name']}</td>"
                f"<td>{p['score_pct']:.1f}%</td>"
                f"<td>{p['status'].replace('_', ' ').title()}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Overall: {sec['overall_quality_pct']:.1f}% "
            f"({sec['quality_label']})</p>\n"
            f"<table><tr><th>Principle</th><th>Score</th><th>Status</th></tr>"
            f"{rows}</table>"
        )

    def _html_activities(self, data: Dict[str, Any]) -> str:
        """Render activity log HTML."""
        sec = self._section_activity_log(data)
        rows = "".join(
            f"<tr><td>{a['date']}</td><td>{a['title'][:40]}</td>"
            f"<td>{a['activity_type']}</td><td>{a['participants']}</td></tr>"
            for a in sec["activities"][:20]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Activities: {sec['total_activities']} | "
            f"Participants: {sec['total_participants']:,}</p>\n"
            f"<table><tr><th>Date</th><th>Activity</th><th>Type</th>"
            f"<th>Participants</th></tr>{rows}</table>"
        )

    def _html_outcomes(self, data: Dict[str, Any]) -> str:
        """Render outcomes HTML."""
        sec = self._section_outcomes_summary(data)
        rows = "".join(
            f"<tr><td>{o['description'][:50]}</td><td>{o['status']}</td>"
            f"<td>{'Yes' if o['impact_on_dd'] else 'No'}</td></tr>"
            for o in sec["outcomes"][:15]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Outcomes: {sec['total_outcomes']} | "
            f"Actioned: {sec['outcomes_actioned']}</p>\n"
            f"<table><tr><th>Outcome</th><th>Status</th><th>DD Impact</th></tr>"
            f"{rows}</table>"
        )
