# -*- coding: utf-8 -*-
"""
CSDDDScorecardTemplate - CSDDD Executive Scorecard Dashboard

Renders an executive-level dashboard with article-by-article compliance status,
key metrics, risk summary, trend analysis, action items, and strategic
recommendations for CSDDD (Directive (EU) 2024/1760) readiness.

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD), Articles 5-29
    - UN Guiding Principles on Business and Human Rights (2011)
    - OECD Due Diligence Guidance for Responsible Business Conduct (2018)

Sections:
    1. Overall Score - Aggregate compliance readiness score
    2. Article Status Grid - Compliance status per CSDDD article
    3. Key Metrics Dashboard - Core due diligence performance indicators
    4. Risk Summary - Top risks across value chain and operations
    5. Trend Analysis - Period-over-period compliance trend
    6. Action Items - Outstanding remediation and improvement actions
    7. Executive Recommendations - Strategic priorities for leadership

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
    "overall_score",
    "article_status_grid",
    "key_metrics",
    "risk_summary",
    "trend_analysis",
    "action_items",
    "executive_recommendations",
]

# CSDDD article groups for scorecard categorisation
_ARTICLE_GROUPS: Dict[str, List[Dict[str, Any]]] = {
    "due_diligence_core": [
        {"article": 5, "title": "Due Diligence Obligation"},
        {"article": 6, "title": "Identifying Adverse Impacts"},
        {"article": 7, "title": "Prioritisation of Impacts"},
        {"article": 8, "title": "Prevention of Potential Impacts"},
        {"article": 9, "title": "Remediation of Actual Impacts"},
    ],
    "stakeholder_engagement": [
        {"article": 10, "title": "Meaningful Engagement"},
        {"article": 11, "title": "Grievance Mechanism"},
    ],
    "monitoring_reporting": [
        {"article": 12, "title": "Monitoring Effectiveness"},
        {"article": 13, "title": "Communication"},
        {"article": 14, "title": "Contractual Cascading"},
        {"article": 17, "title": "Reporting Obligations"},
    ],
    "climate": [
        {"article": 22, "title": "Climate Transition Plan"},
    ],
    "governance_enforcement": [
        {"article": 25, "title": "Civil Liability"},
        {"article": 26, "title": "Penalties"},
    ],
}

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

class CSDDDScorecardTemplate:
    """
    CSDDD Executive Scorecard Dashboard.

    Renders an executive-level compliance scorecard with weighted article
    scoring, key performance indicators across due diligence dimensions,
    risk heatmap summary, period-over-period trend analysis, outstanding
    action tracking, and strategic recommendations for CSDDD readiness.

    Example:
        >>> tpl = CSDDDScorecardTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CSDDDScorecardTemplate."""
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
        if "article_scores" not in data:
            warnings.append("article_scores missing; will default to 0%")
        if "metrics" not in data:
            warnings.append("metrics missing; KPI dashboard will be empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render CSDDD scorecard as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_overall_score(data),
            self._md_article_status_grid(data),
            self._md_key_metrics(data),
            self._md_risk_summary(data),
            self._md_trend_analysis(data),
            self._md_action_items(data),
            self._md_executive_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render CSDDD scorecard as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overall_score(data),
            self._html_article_grid(data),
            self._html_key_metrics(data),
            self._html_risk_summary(data),
            self._html_actions(data),
            self._html_recommendations(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>CSDDD Scorecard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render CSDDD scorecard as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "csddd_scorecard",
            "directive_reference": "Directive (EU) 2024/1760",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "overall_score": self._section_overall_score(data),
            "article_status_grid": self._section_article_status_grid(data),
            "key_metrics": self._section_key_metrics(data),
            "risk_summary": self._section_risk_summary(data),
            "trend_analysis": self._section_trend_analysis(data),
            "action_items": self._section_action_items(data),
            "executive_recommendations": self._section_executive_recommendations(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_overall_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build overall compliance score section."""
        article_scores = data.get("article_scores", {})
        group_scores: Dict[str, float] = {}
        for group_name, articles in _ARTICLE_GROUPS.items():
            scores = [
                article_scores.get(f"art_{a['article']}", 0.0)
                for a in articles
            ]
            group_scores[group_name] = (
                round(sum(scores) / len(scores), 1) if scores else 0.0
            )
        all_scores = list(group_scores.values())
        overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0
        return {
            "title": "Overall CSDDD Compliance Score",
            "overall_score_pct": overall,
            "readiness_label": self._readiness_label(overall),
            "group_scores": group_scores,
            "compliant_articles": sum(
                1 for v in article_scores.values() if v >= 80.0
            ),
            "partial_articles": sum(
                1 for v in article_scores.values() if 40.0 <= v < 80.0
            ),
            "non_compliant_articles": sum(
                1 for v in article_scores.values() if v < 40.0
            ),
            "total_articles_scored": len(article_scores),
            "assessment_date": data.get("assessment_date", utcnow().isoformat()),
        }

    def _section_article_status_grid(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build article-by-article status grid section."""
        article_scores = data.get("article_scores", {})
        groups: List[Dict[str, Any]] = []
        for group_name, articles in _ARTICLE_GROUPS.items():
            group_articles: List[Dict[str, Any]] = []
            for a in articles:
                art_key = f"art_{a['article']}"
                score = article_scores.get(art_key, 0.0)
                group_articles.append({
                    "article": a["article"],
                    "title": a["title"],
                    "score_pct": round(score, 1),
                    "status": self._score_to_status(score),
                    "evidence_available": data.get(
                        "evidence", {}
                    ).get(art_key, False),
                })
            group_score = (
                round(
                    sum(ga["score_pct"] for ga in group_articles) / len(group_articles), 1
                ) if group_articles else 0.0
            )
            groups.append({
                "group_name": group_name.replace("_", " ").title(),
                "group_key": group_name,
                "group_score_pct": group_score,
                "articles": group_articles,
            })
        return {
            "title": "Article Status Grid",
            "groups": groups,
            "total_groups": len(groups),
        }

    def _section_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build key metrics dashboard section."""
        metrics = data.get("metrics", {})
        kpis: List[Dict[str, Any]] = [
            {
                "name": "Value Chain Coverage",
                "value": round(metrics.get("value_chain_coverage_pct", 0.0), 1),
                "unit": "%",
                "target": 100.0,
                "status": self._kpi_status(
                    metrics.get("value_chain_coverage_pct", 0.0), 100.0
                ),
            },
            {
                "name": "Supplier Assessments Completed",
                "value": metrics.get("supplier_assessments_completed", 0),
                "unit": "count",
                "target": metrics.get("supplier_assessments_target", 0),
                "status": self._kpi_status(
                    metrics.get("supplier_assessments_completed", 0),
                    metrics.get("supplier_assessments_target", 1),
                ),
            },
            {
                "name": "Adverse Impacts Identified",
                "value": metrics.get("adverse_impacts_identified", 0),
                "unit": "count",
                "target": None,
                "status": "informational",
            },
            {
                "name": "Prevention Measures Active",
                "value": metrics.get("prevention_measures_active", 0),
                "unit": "count",
                "target": metrics.get("prevention_measures_target", 0),
                "status": self._kpi_status(
                    metrics.get("prevention_measures_active", 0),
                    metrics.get("prevention_measures_target", 1),
                ),
            },
            {
                "name": "Grievances Received",
                "value": metrics.get("grievances_received", 0),
                "unit": "count",
                "target": None,
                "status": "informational",
            },
            {
                "name": "Grievance Resolution Rate",
                "value": round(metrics.get("grievance_resolution_rate_pct", 0.0), 1),
                "unit": "%",
                "target": 90.0,
                "status": self._kpi_status(
                    metrics.get("grievance_resolution_rate_pct", 0.0), 90.0
                ),
            },
            {
                "name": "Stakeholder Groups Engaged",
                "value": metrics.get("stakeholder_groups_engaged", 0),
                "unit": "count",
                "target": 10,
                "status": self._kpi_status(
                    metrics.get("stakeholder_groups_engaged", 0), 10
                ),
            },
            {
                "name": "DD Policy Score",
                "value": round(metrics.get("dd_policy_score_pct", 0.0), 1),
                "unit": "%",
                "target": 100.0,
                "status": self._kpi_status(
                    metrics.get("dd_policy_score_pct", 0.0), 100.0
                ),
            },
            {
                "name": "Climate Plan Progress",
                "value": round(metrics.get("climate_plan_progress_pct", 0.0), 1),
                "unit": "%",
                "target": 100.0,
                "status": self._kpi_status(
                    metrics.get("climate_plan_progress_pct", 0.0), 100.0
                ),
            },
            {
                "name": "DD Budget Utilisation",
                "value": round(metrics.get("dd_budget_utilisation_pct", 0.0), 1),
                "unit": "%",
                "target": 100.0,
                "status": self._kpi_status(
                    metrics.get("dd_budget_utilisation_pct", 0.0), 100.0
                ),
            },
        ]
        return {
            "title": "Key Metrics Dashboard",
            "total_kpis": len(kpis),
            "kpis_on_target": sum(1 for k in kpis if k["status"] == "on_target"),
            "kpis_at_risk": sum(1 for k in kpis if k["status"] == "at_risk"),
            "kpis_off_target": sum(1 for k in kpis if k["status"] == "off_target"),
            "kpis": kpis,
        }

    def _section_risk_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build risk summary section."""
        risks = data.get("top_risks", [])
        entries: List[Dict[str, Any]] = []
        for r in risks:
            entries.append({
                "risk_id": r.get("risk_id", ""),
                "description": r.get("description", ""),
                "category": r.get("category", ""),
                "severity": r.get("severity", "medium"),
                "likelihood": r.get("likelihood", "possible"),
                "risk_score": r.get("risk_score", 0.0),
                "affected_articles": r.get("affected_articles", []),
                "mitigation_status": r.get("mitigation_status", "not_started"),
                "trend": r.get("trend", "stable"),
            })
        entries.sort(key=lambda x: x["risk_score"], reverse=True)
        severity_dist: Dict[str, int] = {}
        for e in entries:
            sev = e["severity"]
            severity_dist[sev] = severity_dist.get(sev, 0) + 1
        return {
            "title": "Risk Summary",
            "total_risks": len(entries),
            "severity_distribution": severity_dist,
            "risks": entries,
            "risks_increasing": sum(1 for e in entries if e["trend"] == "increasing"),
            "risks_decreasing": sum(1 for e in entries if e["trend"] == "decreasing"),
            "risks_stable": sum(1 for e in entries if e["trend"] == "stable"),
        }

    def _section_trend_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build period-over-period trend analysis section."""
        trends = data.get("trend_data", [])
        entries: List[Dict[str, Any]] = []
        for t in trends:
            entries.append({
                "period": t.get("period", ""),
                "overall_score_pct": round(t.get("overall_score_pct", 0.0), 1),
                "dd_core_score_pct": round(t.get("dd_core_score_pct", 0.0), 1),
                "stakeholder_score_pct": round(t.get("stakeholder_score_pct", 0.0), 1),
                "monitoring_score_pct": round(t.get("monitoring_score_pct", 0.0), 1),
                "climate_score_pct": round(t.get("climate_score_pct", 0.0), 1),
                "adverse_impacts_count": t.get("adverse_impacts_count", 0),
                "grievances_count": t.get("grievances_count", 0),
            })
        current_score = entries[-1]["overall_score_pct"] if entries else 0.0
        prev_score = entries[-2]["overall_score_pct"] if len(entries) >= 2 else 0.0
        change = round(current_score - prev_score, 1)
        return {
            "title": "Trend Analysis",
            "total_periods": len(entries),
            "current_score_pct": current_score,
            "previous_score_pct": prev_score,
            "change_pct": change,
            "trend_direction": (
                "improving" if change > 0 else (
                    "declining" if change < 0 else "stable"
                )
            ),
            "periods": entries,
        }

    def _section_action_items(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build action items section."""
        actions = data.get("action_items", [])
        entries: List[Dict[str, Any]] = []
        for a in actions:
            entries.append({
                "action_id": a.get("action_id", ""),
                "description": a.get("description", ""),
                "priority": a.get("priority", "medium"),
                "status": a.get("status", "open"),
                "responsible": a.get("responsible", ""),
                "due_date": a.get("due_date", ""),
                "related_articles": a.get("related_articles", []),
                "completion_pct": round(a.get("completion_pct", 0.0), 1),
                "overdue": a.get("overdue", False),
            })
        open_count = sum(1 for e in entries if e["status"] == "open")
        overdue_count = sum(1 for e in entries if e["overdue"])
        return {
            "title": "Action Items",
            "total_actions": len(entries),
            "open_actions": open_count,
            "overdue_actions": overdue_count,
            "completed_actions": sum(1 for e in entries if e["status"] == "completed"),
            "in_progress_actions": sum(1 for e in entries if e["status"] == "in_progress"),
            "actions": entries,
            "completion_rate_pct": round(
                sum(1 for e in entries if e["status"] == "completed") / len(entries) * 100, 1
            ) if entries else 0.0,
        }

    def _section_executive_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive recommendations section."""
        recommendations = data.get("exec_recommendations", [])
        if not recommendations:
            recommendations = self._generate_default_recommendations(data)
        return {
            "title": "Executive Recommendations",
            "total_recommendations": len(recommendations),
            "recommendations": [
                {
                    "priority": r.get("priority", "medium"),
                    "area": r.get("area", ""),
                    "recommendation": r.get("recommendation", ""),
                    "expected_impact": r.get("expected_impact", ""),
                    "investment_required": r.get("investment_required", ""),
                    "timeline": r.get("timeline", ""),
                    "risk_if_not_addressed": r.get("risk_if_not_addressed", ""),
                }
                for r in recommendations
            ],
        }

    # ------------------------------------------------------------------
    # Calculation helpers
    # ------------------------------------------------------------------

    def _readiness_label(self, score: float) -> str:
        """Map overall score to readiness label."""
        if score >= 90.0:
            return "Compliance Ready"
        elif score >= 75.0:
            return "Near Ready"
        elif score >= 50.0:
            return "Partially Ready"
        elif score >= 25.0:
            return "Not Ready"
        else:
            return "Critical Gaps"

    def _score_to_status(self, score: float) -> str:
        """Map article score to compliance status."""
        if score >= 80.0:
            return "compliant"
        elif score >= 40.0:
            return "partial"
        else:
            return "non_compliant"

    def _kpi_status(self, value: float, target: float) -> str:
        """Determine KPI status against target."""
        if target <= 0:
            return "informational"
        ratio = value / target
        if ratio >= 0.9:
            return "on_target"
        elif ratio >= 0.6:
            return "at_risk"
        else:
            return "off_target"

    def _generate_default_recommendations(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate default executive recommendations from scorecard data."""
        recs: List[Dict[str, Any]] = []
        overall = self._section_overall_score(data)
        if overall["overall_score_pct"] < 50.0:
            recs.append({
                "priority": "critical",
                "area": "overall_readiness",
                "recommendation": (
                    "Establish a dedicated CSDDD compliance programme with "
                    "board-level oversight, adequate budget, and clear milestones"
                ),
                "expected_impact": "Foundation for systematic compliance",
                "investment_required": "Significant (dedicated team + budget)",
                "timeline": "0-3 months to establish",
                "risk_if_not_addressed": (
                    "Regulatory penalties up to 5% net worldwide turnover"
                ),
            })
        group_scores = overall["group_scores"]
        if group_scores.get("due_diligence_core", 0.0) < 50.0:
            recs.append({
                "priority": "high",
                "area": "due_diligence_core",
                "recommendation": (
                    "Implement core due diligence processes covering impact "
                    "identification (Art 6), prevention (Art 8), and "
                    "remediation (Art 9)"
                ),
                "expected_impact": "Address 5 core CSDDD obligations",
                "investment_required": "Moderate",
                "timeline": "3-6 months",
                "risk_if_not_addressed": "Civil liability exposure",
            })
        if group_scores.get("stakeholder_engagement", 0.0) < 50.0:
            recs.append({
                "priority": "high",
                "area": "stakeholder_engagement",
                "recommendation": (
                    "Establish meaningful stakeholder engagement programme "
                    "(Art 10) and operational grievance mechanism (Art 11)"
                ),
                "expected_impact": "Compliance with stakeholder requirements",
                "investment_required": "Moderate",
                "timeline": "3-6 months",
                "risk_if_not_addressed": "Non-compliance with Art 10-11",
            })
        if group_scores.get("climate", 0.0) < 50.0:
            recs.append({
                "priority": "high",
                "area": "climate_transition",
                "recommendation": (
                    "Develop Paris-aligned climate transition plan (Art 22) "
                    "with science-based targets and implementation milestones"
                ),
                "expected_impact": "Climate compliance and investor confidence",
                "investment_required": "Significant",
                "timeline": "6-12 months",
                "risk_if_not_addressed": "Non-compliance with Art 22",
            })
        action_sec = self._section_action_items(data)
        if action_sec["overdue_actions"] > 0:
            recs.append({
                "priority": "high",
                "area": "overdue_actions",
                "recommendation": (
                    f"Resolve {action_sec['overdue_actions']} overdue action items "
                    f"immediately and establish accountability mechanisms"
                ),
                "expected_impact": "Reduce compliance risk backlog",
                "investment_required": "Low to Moderate",
                "timeline": "Immediate (0-1 month)",
                "risk_if_not_addressed": "Accumulating compliance debt",
            })
        recs.append({
            "priority": "medium",
            "area": "continuous_improvement",
            "recommendation": (
                "Implement quarterly CSDDD compliance reviews with "
                "board reporting and stakeholder feedback integration"
            ),
            "expected_impact": "Sustained compliance improvement trajectory",
            "investment_required": "Low",
            "timeline": "Ongoing (quarterly)",
            "risk_if_not_addressed": "Compliance regression",
        })
        return recs

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# CSDDD Executive Scorecard\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Directive:** Directive (EU) 2024/1760 (CSDDD)"
        )

    def _md_overall_score(self, data: Dict[str, Any]) -> str:
        """Render overall score as markdown."""
        sec = self._section_overall_score(data)
        lines = [
            f"## {sec['title']}\n",
            f"### {sec['overall_score_pct']:.1f}% - {sec['readiness_label']}\n",
            "| Category | Score |",
            "|----------|------:|",
        ]
        for group, score in sec["group_scores"].items():
            lines.append(
                f"| {group.replace('_', ' ').title()} | {score:.1f}% |"
            )
        lines.append(
            f"\n**Compliant:** {sec['compliant_articles']} | "
            f"**Partial:** {sec['partial_articles']} | "
            f"**Non-Compliant:** {sec['non_compliant_articles']}"
        )
        return "\n".join(lines)

    def _md_article_status_grid(self, data: Dict[str, Any]) -> str:
        """Render article status grid as markdown."""
        sec = self._section_article_status_grid(data)
        lines = [f"## {sec['title']}\n"]
        for group in sec["groups"]:
            lines.append(
                f"### {group['group_name']} ({group['group_score_pct']:.1f}%)\n"
            )
            lines.append("| Article | Title | Score | Status |")
            lines.append("|--------:|-------|------:|--------|")
            for a in group["articles"]:
                status = a["status"].replace("_", " ").title()
                lines.append(
                    f"| Art {a['article']} | {a['title']} | "
                    f"{a['score_pct']:.1f}% | {status} |"
                )
            lines.append("")
        return "\n".join(lines)

    def _md_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render key metrics as markdown."""
        sec = self._section_key_metrics(data)
        lines = [
            f"## {sec['title']}\n",
            f"**On Target:** {sec['kpis_on_target']} | "
            f"**At Risk:** {sec['kpis_at_risk']} | "
            f"**Off Target:** {sec['kpis_off_target']}\n",
            "| KPI | Value | Target | Status |",
            "|-----|------:|-------:|--------|",
        ]
        for k in sec["kpis"]:
            target_str = (
                f"{k['target']}" if k["target"] is not None else "N/A"
            )
            lines.append(
                f"| {k['name']} | {k['value']} {k['unit']} | "
                f"{target_str} | {k['status'].replace('_', ' ').title()} |"
            )
        return "\n".join(lines)

    def _md_risk_summary(self, data: Dict[str, Any]) -> str:
        """Render risk summary as markdown."""
        sec = self._section_risk_summary(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Risks:** {sec['total_risks']}  \n"
            f"**Increasing:** {sec['risks_increasing']} | "
            f"**Stable:** {sec['risks_stable']} | "
            f"**Decreasing:** {sec['risks_decreasing']}\n",
            "| Risk | Category | Severity | Likelihood | Score | Trend |",
            "|------|----------|----------|------------|------:|-------|",
        ]
        for r in sec["risks"][:10]:
            lines.append(
                f"| {r['description'][:35]} | {r['category']} | "
                f"{r['severity']} | {r['likelihood']} | "
                f"{r['risk_score']:.1f} | {r['trend']} |"
            )
        return "\n".join(lines)

    def _md_trend_analysis(self, data: Dict[str, Any]) -> str:
        """Render trend analysis as markdown."""
        sec = self._section_trend_analysis(data)
        direction = sec["trend_direction"].title()
        lines = [
            f"## {sec['title']}\n",
            f"**Current:** {sec['current_score_pct']:.1f}% | "
            f"**Previous:** {sec['previous_score_pct']:.1f}% | "
            f"**Change:** {sec['change_pct']:+.1f}pp ({direction})\n",
        ]
        if sec["periods"]:
            lines.append("| Period | Overall | DD Core | Stakeholder | Climate |")
            lines.append("|--------|-------:|-------:|-----------:|-------:|")
            for p in sec["periods"]:
                lines.append(
                    f"| {p['period']} | {p['overall_score_pct']:.1f}% | "
                    f"{p['dd_core_score_pct']:.1f}% | "
                    f"{p['stakeholder_score_pct']:.1f}% | "
                    f"{p['climate_score_pct']:.1f}% |"
                )
        return "\n".join(lines)

    def _md_action_items(self, data: Dict[str, Any]) -> str:
        """Render action items as markdown."""
        sec = self._section_action_items(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total:** {sec['total_actions']} | "
            f"**Open:** {sec['open_actions']} | "
            f"**Overdue:** {sec['overdue_actions']} | "
            f"**Completed:** {sec['completed_actions']} | "
            f"**Completion Rate:** {sec['completion_rate_pct']:.1f}%\n",
            "| Action | Priority | Status | Responsible | Due Date | % |",
            "|--------|----------|--------|-------------|----------|--:|",
        ]
        for a in sec["actions"][:15]:
            overdue_marker = " [OVERDUE]" if a["overdue"] else ""
            lines.append(
                f"| {a['description'][:35]}{overdue_marker} | "
                f"{a['priority'].upper()} | {a['status']} | "
                f"{a['responsible']} | {a['due_date']} | "
                f"{a['completion_pct']:.0f}% |"
            )
        return "\n".join(lines)

    def _md_executive_recommendations(self, data: Dict[str, Any]) -> str:
        """Render executive recommendations as markdown."""
        sec = self._section_executive_recommendations(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Recommendations:** {sec['total_recommendations']}\n",
        ]
        for idx, r in enumerate(sec["recommendations"], 1):
            lines.append(
                f"### {idx}. [{r['priority'].upper()}] {r['area'].replace('_', ' ').title()}\n"
            )
            lines.append(f"**Recommendation:** {r['recommendation']}\n")
            lines.append(
                f"- **Expected Impact:** {r['expected_impact']}\n"
                f"- **Investment:** {r['investment_required']}\n"
                f"- **Timeline:** {r['timeline']}\n"
                f"- **Risk if Not Addressed:** {r['risk_if_not_addressed']}"
            )
            lines.append("")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Scorecard generated by PACK-019 CSDDD Readiness Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1200px;margin:auto}"
            "h1{color:#1a237e;border-bottom:2px solid #1a237e;padding-bottom:.3em}"
            "h2{color:#283593;margin-top:1.5em}"
            "h3{color:#3949ab;margin-top:1em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8eaf6}"
            ".compliant{color:#2e7d32;font-weight:bold}"
            ".partial{color:#ef6c00}"
            ".non-compliant{color:#c62828;font-weight:bold}"
            ".on-target{color:#2e7d32}"
            ".at-risk{color:#ef6c00}"
            ".off-target{color:#c62828}"
            ".overdue{background:#ffebee}"
            ".score-high{font-size:2em;color:#2e7d32;font-weight:bold}"
            ".score-medium{font-size:2em;color:#ef6c00;font-weight:bold}"
            ".score-low{font-size:2em;color:#c62828;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>CSDDD Executive Scorecard</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('reporting_year', '')}</p>"
        )

    def _html_overall_score(self, data: Dict[str, Any]) -> str:
        """Render overall score HTML."""
        sec = self._section_overall_score(data)
        score = sec["overall_score_pct"]
        css = "score-high" if score >= 75 else ("score-medium" if score >= 50 else "score-low")
        rows = "".join(
            f"<tr><td>{g.replace('_', ' ').title()}</td><td>{s:.1f}%</td></tr>"
            for g, s in sec["group_scores"].items()
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{css}'>{score:.1f}%</p>\n"
            f"<p>{sec['readiness_label']}</p>\n"
            f"<table><tr><th>Category</th><th>Score</th></tr>{rows}</table>"
        )

    def _html_article_grid(self, data: Dict[str, Any]) -> str:
        """Render article status grid HTML."""
        sec = self._section_article_status_grid(data)
        html_parts: List[str] = [f"<h2>{sec['title']}</h2>"]
        for group in sec["groups"]:
            html_parts.append(
                f"<h3>{group['group_name']} ({group['group_score_pct']:.1f}%)</h3>"
            )
            rows = ""
            for a in group["articles"]:
                css = a["status"].replace("_", "-")
                rows += (
                    f"<tr class='{css}'><td>Art {a['article']}</td>"
                    f"<td>{a['title']}</td><td>{a['score_pct']:.1f}%</td>"
                    f"<td>{a['status'].replace('_', ' ').title()}</td></tr>"
                )
            html_parts.append(
                f"<table><tr><th>Article</th><th>Title</th><th>Score</th>"
                f"<th>Status</th></tr>{rows}</table>"
            )
        return "\n".join(html_parts)

    def _html_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render key metrics HTML."""
        sec = self._section_key_metrics(data)
        rows = ""
        for k in sec["kpis"]:
            css = k["status"].replace("_", "-")
            target_str = str(k["target"]) if k["target"] is not None else "N/A"
            rows += (
                f"<tr class='{css}'><td>{k['name']}</td>"
                f"<td>{k['value']} {k['unit']}</td>"
                f"<td>{target_str}</td>"
                f"<td>{k['status'].replace('_', ' ').title()}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>KPI</th><th>Value</th><th>Target</th>"
            f"<th>Status</th></tr>{rows}</table>"
        )

    def _html_risk_summary(self, data: Dict[str, Any]) -> str:
        """Render risk summary HTML."""
        sec = self._section_risk_summary(data)
        rows = "".join(
            f"<tr><td>{r['description'][:50]}</td><td>{r['category']}</td>"
            f"<td>{r['severity']}</td><td>{r['risk_score']:.1f}</td></tr>"
            for r in sec["risks"][:10]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total Risks: {sec['total_risks']}</p>\n"
            f"<table><tr><th>Risk</th><th>Category</th><th>Severity</th>"
            f"<th>Score</th></tr>{rows}</table>"
        )

    def _html_actions(self, data: Dict[str, Any]) -> str:
        """Render action items HTML."""
        sec = self._section_action_items(data)
        rows = ""
        for a in sec["actions"][:15]:
            css = "overdue" if a["overdue"] else ""
            rows += (
                f"<tr class='{css}'><td>{a['description'][:40]}</td>"
                f"<td>{a['priority'].upper()}</td><td>{a['status']}</td>"
                f"<td>{a['due_date']}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Open: {sec['open_actions']} | Overdue: {sec['overdue_actions']}</p>\n"
            f"<table><tr><th>Action</th><th>Priority</th><th>Status</th>"
            f"<th>Due Date</th></tr>{rows}</table>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render executive recommendations HTML."""
        sec = self._section_executive_recommendations(data)
        items = ""
        for r in sec["recommendations"]:
            items += (
                f"<div style='margin:1em 0;padding:1em;border-left:4px solid #283593'>"
                f"<p><strong>[{r['priority'].upper()}] "
                f"{r['area'].replace('_', ' ').title()}</strong></p>"
                f"<p>{r['recommendation']}</p>"
                f"<p><em>Impact: {r['expected_impact']} | "
                f"Timeline: {r['timeline']}</em></p></div>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"{items}"
        )
