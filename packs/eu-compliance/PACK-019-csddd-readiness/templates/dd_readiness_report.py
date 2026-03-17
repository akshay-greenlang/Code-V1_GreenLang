# -*- coding: utf-8 -*-
"""
DDReadinessReportTemplate - CSDDD Due Diligence Readiness Assessment Report

Renders an overall CSDDD (Corporate Sustainability Due Diligence Directive,
Directive 2024/1760) readiness assessment with article-by-article compliance
status covering Articles 5-29, gap analysis, readiness scoring, and a
phased implementation timeline.

Regulatory References:
    - Directive (EU) 2024/1760 of the European Parliament and of the Council
    - Articles 5-16: Due Diligence Obligations
    - Articles 17-21: Reporting and Civil Liability
    - Article 22: Climate Transition Plan
    - Articles 23-29: Enforcement and Penalties

Sections:
    1. Executive Summary - Overall readiness score and status
    2. Scope Determination - Entity scope and applicability
    3. Article-by-Article Status - Compliance status per Art 5-29
    4. Gap Analysis - Missing obligations and effort estimates
    5. Readiness Score - Weighted compliance scoring
    6. Recommendations - Prioritized improvement actions
    7. Timeline - Phased implementation roadmap

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
    "executive_summary",
    "scope_determination",
    "article_status",
    "gap_analysis",
    "readiness_score",
    "recommendations",
    "timeline",
]

# CSDDD articles with descriptions and weights for scoring
_CSDDD_ARTICLES: List[Dict[str, Any]] = [
    {"article": 5, "title": "Due Diligence", "category": "obligations", "weight": 1.0},
    {"article": 6, "title": "Identifying Adverse Impacts", "category": "obligations", "weight": 1.0},
    {"article": 7, "title": "Prioritisation of Impacts", "category": "obligations", "weight": 0.8},
    {"article": 8, "title": "Prevention of Potential Adverse Impacts", "category": "obligations", "weight": 1.0},
    {"article": 9, "title": "Remediation of Actual Adverse Impacts", "category": "obligations", "weight": 1.0},
    {"article": 10, "title": "Meaningful Engagement with Stakeholders", "category": "obligations", "weight": 0.9},
    {"article": 11, "title": "Grievance Mechanism", "category": "obligations", "weight": 0.9},
    {"article": 12, "title": "Monitoring Effectiveness", "category": "obligations", "weight": 0.8},
    {"article": 13, "title": "Communication", "category": "obligations", "weight": 0.7},
    {"article": 14, "title": "Contractual Cascading", "category": "obligations", "weight": 0.8},
    {"article": 15, "title": "Additional Measures for Financial Undertakings", "category": "financial", "weight": 0.6},
    {"article": 16, "title": "Guidelines for Financial Undertakings", "category": "financial", "weight": 0.5},
    {"article": 17, "title": "Reporting Obligations", "category": "reporting", "weight": 0.7},
    {"article": 18, "title": "Model Contractual Clauses", "category": "reporting", "weight": 0.5},
    {"article": 19, "title": "Accompanying Measures for SMEs", "category": "reporting", "weight": 0.4},
    {"article": 20, "title": "Combating Climate Change", "category": "climate", "weight": 0.9},
    {"article": 21, "title": "Authorisation and Supervision", "category": "enforcement", "weight": 0.6},
    {"article": 22, "title": "Climate Transition Plan", "category": "climate", "weight": 1.0},
    {"article": 23, "title": "Supervisory Authority", "category": "enforcement", "weight": 0.5},
    {"article": 24, "title": "Powers of Supervisory Authorities", "category": "enforcement", "weight": 0.5},
    {"article": 25, "title": "Civil Liability", "category": "liability", "weight": 0.8},
    {"article": 26, "title": "Penalties", "category": "enforcement", "weight": 0.7},
    {"article": 27, "title": "European Network of Supervisory Authorities", "category": "enforcement", "weight": 0.3},
    {"article": 28, "title": "Transposition", "category": "enforcement", "weight": 0.2},
    {"article": 29, "title": "Review", "category": "enforcement", "weight": 0.2},
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


class DDReadinessReportTemplate:
    """
    CSDDD Due Diligence Readiness Assessment Report.

    Renders an article-by-article compliance assessment covering Arts 5-29
    of Directive (EU) 2024/1760, with weighted readiness scoring, gap
    analysis identifying missing obligations, prioritized recommendations,
    and a phased implementation timeline.

    Example:
        >>> tpl = DDReadinessReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DDReadinessReportTemplate."""
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
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if "article_statuses" not in data:
            warnings.append("article_statuses missing; will default to empty")
        if "scope" not in data:
            warnings.append("scope missing; will default to unknown")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render CSDDD readiness report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_scope_determination(data),
            self._md_article_status(data),
            self._md_gap_analysis(data),
            self._md_readiness_score(data),
            self._md_recommendations(data),
            self._md_timeline(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render CSDDD readiness report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_scope(data),
            self._html_article_status(data),
            self._html_gap_analysis(data),
            self._html_readiness(data),
            self._html_recommendations(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>CSDDD Readiness Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render CSDDD readiness report as JSON."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "dd_readiness_report",
            "directive_reference": "Directive (EU) 2024/1760",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "executive_summary": self._section_executive_summary(data),
            "scope_determination": self._section_scope_determination(data),
            "article_status": self._section_article_status(data),
            "gap_analysis": self._section_gap_analysis(data),
            "readiness_score": self._section_readiness_score(data),
            "recommendations": self._section_recommendations(data),
            "timeline": self._section_timeline(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary section."""
        score_data = self._calculate_readiness_score(data)
        overall = score_data["overall_score"]
        article_statuses = data.get("article_statuses", {})
        compliant_count = sum(
            1 for s in article_statuses.values()
            if s.get("status") == "compliant"
        )
        partial_count = sum(
            1 for s in article_statuses.values()
            if s.get("status") == "partial"
        )
        non_compliant_count = sum(
            1 for s in article_statuses.values()
            if s.get("status") in ("non_compliant", "not_started")
        )
        return {
            "title": "Executive Summary",
            "overall_readiness_pct": round(overall, 1),
            "readiness_status": self._get_readiness_label(overall),
            "total_articles_assessed": len(_CSDDD_ARTICLES),
            "articles_compliant": compliant_count,
            "articles_partial": partial_count,
            "articles_non_compliant": non_compliant_count,
            "assessment_date": data.get("assessment_date", _utcnow().isoformat()),
            "key_findings": data.get("key_findings", []),
        }

    def _section_scope_determination(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build scope determination section."""
        scope = data.get("scope", {})
        return {
            "title": "Scope Determination",
            "entity_type": scope.get("entity_type", "unknown"),
            "employee_count": scope.get("employee_count", 0),
            "net_turnover_eur": scope.get("net_turnover_eur", 0.0),
            "is_in_scope": scope.get("is_in_scope", False),
            "scope_tier": self._determine_scope_tier(scope),
            "application_date": scope.get("application_date", ""),
            "high_risk_sector": scope.get("high_risk_sector", False),
            "member_state_jurisdiction": scope.get("jurisdiction", ""),
            "group_structure": scope.get("group_structure", "standalone"),
            "third_country_entity": scope.get("third_country_entity", False),
            "third_country_turnover_eur": scope.get("third_country_turnover_eur", 0.0),
        }

    def _section_article_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build article-by-article compliance status section."""
        article_statuses = data.get("article_statuses", {})
        entries: List[Dict[str, Any]] = []
        for art_def in _CSDDD_ARTICLES:
            art_num = art_def["article"]
            art_key = f"art_{art_num}"
            status_info = article_statuses.get(art_key, {})
            entries.append({
                "article": art_num,
                "title": art_def["title"],
                "category": art_def["category"],
                "status": status_info.get("status", "not_assessed"),
                "compliance_pct": round(status_info.get("compliance_pct", 0.0), 1),
                "evidence_count": status_info.get("evidence_count", 0),
                "notes": status_info.get("notes", ""),
            })
        return {
            "title": "Article-by-Article Compliance Status (Art 5-29)",
            "articles": entries,
            "total_assessed": sum(
                1 for e in entries if e["status"] != "not_assessed"
            ),
        }

    def _section_gap_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build gap analysis section."""
        gaps = data.get("gaps", [])
        article_statuses = data.get("article_statuses", {})
        auto_gaps: List[Dict[str, Any]] = []
        for art_def in _CSDDD_ARTICLES:
            art_key = f"art_{art_def['article']}"
            status_info = article_statuses.get(art_key, {})
            if status_info.get("status") in ("non_compliant", "not_started", "not_assessed"):
                auto_gaps.append({
                    "article": art_def["article"],
                    "title": art_def["title"],
                    "gap_type": "missing_obligation",
                    "severity": "high" if art_def["weight"] >= 0.9 else "medium",
                    "effort_estimate_days": status_info.get("effort_estimate_days", 0),
                    "description": status_info.get(
                        "gap_description",
                        f"Article {art_def['article']} ({art_def['title']}) not addressed"
                    ),
                })
        all_gaps = auto_gaps + gaps
        return {
            "title": "Gap Analysis",
            "total_gaps": len(all_gaps),
            "high_severity": sum(1 for g in all_gaps if g.get("severity") == "high"),
            "medium_severity": sum(1 for g in all_gaps if g.get("severity") == "medium"),
            "low_severity": sum(1 for g in all_gaps if g.get("severity") == "low"),
            "total_effort_days": sum(g.get("effort_estimate_days", 0) for g in all_gaps),
            "gaps": all_gaps,
        }

    def _section_readiness_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build readiness score section."""
        score_data = self._calculate_readiness_score(data)
        return {
            "title": "Readiness Score",
            "overall_score_pct": round(score_data["overall_score"], 1),
            "readiness_label": self._get_readiness_label(score_data["overall_score"]),
            "category_scores": score_data["category_scores"],
            "scoring_methodology": (
                "Weighted average across CSDDD articles. Each article is "
                "weighted by regulatory importance (0.2-1.0). Category scores "
                "aggregate article scores within obligations, financial, "
                "reporting, climate, enforcement, and liability categories."
            ),
        }

    def _section_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build recommendations section."""
        recommendations = data.get("recommendations", [])
        if not recommendations:
            recommendations = self._generate_default_recommendations(data)
        return {
            "title": "Recommendations",
            "total_recommendations": len(recommendations),
            "recommendations": [
                {
                    "priority": r.get("priority", "medium"),
                    "area": r.get("area", ""),
                    "action": r.get("action", ""),
                    "responsible": r.get("responsible", ""),
                    "deadline": r.get("deadline", ""),
                    "estimated_cost_eur": r.get("estimated_cost_eur", 0),
                    "related_articles": r.get("related_articles", []),
                }
                for r in recommendations
            ],
        }

    def _section_timeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build phased implementation timeline section."""
        timeline = data.get("timeline", {})
        phases = timeline.get("phases", [])
        if not phases:
            phases = self._generate_default_phases(data)
        return {
            "title": "Implementation Timeline",
            "total_phases": len(phases),
            "earliest_deadline": timeline.get("earliest_deadline", "2027-07-26"),
            "phases": [
                {
                    "phase_number": p.get("phase_number", idx + 1),
                    "name": p.get("name", ""),
                    "start_date": p.get("start_date", ""),
                    "end_date": p.get("end_date", ""),
                    "key_activities": p.get("key_activities", []),
                    "deliverables": p.get("deliverables", []),
                    "status": p.get("status", "not_started"),
                }
                for idx, p in enumerate(phases)
            ],
        }

    # ------------------------------------------------------------------
    # Calculation helpers
    # ------------------------------------------------------------------

    def _calculate_readiness_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate weighted readiness score across all articles."""
        article_statuses = data.get("article_statuses", {})
        category_totals: Dict[str, Dict[str, float]] = {}
        for art_def in _CSDDD_ARTICLES:
            cat = art_def["category"]
            if cat not in category_totals:
                category_totals[cat] = {"weighted_sum": 0.0, "weight_sum": 0.0}
            art_key = f"art_{art_def['article']}"
            status_info = article_statuses.get(art_key, {})
            compliance_pct = status_info.get("compliance_pct", 0.0)
            weight = art_def["weight"]
            category_totals[cat]["weighted_sum"] += compliance_pct * weight
            category_totals[cat]["weight_sum"] += weight
        category_scores: Dict[str, float] = {}
        for cat, totals in category_totals.items():
            if totals["weight_sum"] > 0:
                category_scores[cat] = round(
                    totals["weighted_sum"] / totals["weight_sum"], 1
                )
            else:
                category_scores[cat] = 0.0
        all_weighted_sum = sum(t["weighted_sum"] for t in category_totals.values())
        all_weight_sum = sum(t["weight_sum"] for t in category_totals.values())
        overall = round(all_weighted_sum / all_weight_sum, 1) if all_weight_sum > 0 else 0.0
        return {
            "overall_score": overall,
            "category_scores": category_scores,
        }

    def _determine_scope_tier(self, scope: Dict[str, Any]) -> str:
        """Determine CSDDD scope tier based on size thresholds."""
        employees = scope.get("employee_count", 0)
        turnover = scope.get("net_turnover_eur", 0.0)
        if employees >= 1000 and turnover >= 450_000_000:
            return "tier_1"
        elif employees >= 500 and turnover >= 150_000_000:
            return "tier_2"
        elif scope.get("high_risk_sector", False) and employees >= 250:
            return "tier_2_high_risk"
        elif scope.get("third_country_entity", False):
            tc_turnover = scope.get("third_country_turnover_eur", 0.0)
            if tc_turnover >= 450_000_000:
                return "tier_1_third_country"
            elif tc_turnover >= 150_000_000:
                return "tier_2_third_country"
        return "out_of_scope"

    def _get_readiness_label(self, score: float) -> str:
        """Map readiness score to a human-readable label."""
        if score >= 90.0:
            return "Ready for Compliance"
        elif score >= 75.0:
            return "Near Ready (Minor Gaps)"
        elif score >= 50.0:
            return "Partially Ready (Significant Gaps)"
        elif score >= 25.0:
            return "Not Ready (Major Gaps)"
        else:
            return "Far from Ready (Critical Gaps)"

    def _generate_default_recommendations(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate default recommendations based on gap analysis."""
        gap_section = self._section_gap_analysis(data)
        recs: List[Dict[str, Any]] = []
        for gap in gap_section["gaps"][:10]:
            recs.append({
                "priority": "high" if gap.get("severity") == "high" else "medium",
                "area": gap.get("title", ""),
                "action": f"Address gap in Article {gap.get('article', '')}: "
                          f"{gap.get('description', '')}",
                "responsible": "Compliance Team",
                "deadline": "",
                "estimated_cost_eur": 0,
                "related_articles": [gap.get("article", 0)],
            })
        return recs

    def _generate_default_phases(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate default implementation phases."""
        return [
            {
                "phase_number": 1,
                "name": "Assessment and Gap Analysis",
                "start_date": "",
                "end_date": "",
                "key_activities": [
                    "Map value chain and identify business relationships",
                    "Conduct baseline human rights and environmental assessment",
                    "Identify existing policies and processes",
                ],
                "deliverables": ["Gap analysis report", "Value chain map"],
                "status": "not_started",
            },
            {
                "phase_number": 2,
                "name": "Policy Development and Governance",
                "start_date": "",
                "end_date": "",
                "key_activities": [
                    "Develop due diligence policy (Art 5)",
                    "Establish grievance mechanism (Art 11)",
                    "Design stakeholder engagement framework (Art 10)",
                ],
                "deliverables": [
                    "DD policy document",
                    "Grievance mechanism design",
                    "Stakeholder engagement plan",
                ],
                "status": "not_started",
            },
            {
                "phase_number": 3,
                "name": "Implementation and Integration",
                "start_date": "",
                "end_date": "",
                "key_activities": [
                    "Embed DD into business processes",
                    "Implement contractual cascading (Art 14)",
                    "Deploy monitoring systems (Art 12)",
                    "Develop climate transition plan (Art 22)",
                ],
                "deliverables": [
                    "Updated contracts",
                    "Monitoring dashboard",
                    "Climate transition plan",
                ],
                "status": "not_started",
            },
            {
                "phase_number": 4,
                "name": "Testing and Verification",
                "start_date": "",
                "end_date": "",
                "key_activities": [
                    "Test grievance mechanism",
                    "Verify stakeholder engagement effectiveness",
                    "Validate reporting processes",
                    "Conduct internal audit",
                ],
                "deliverables": [
                    "Internal audit report",
                    "Test results documentation",
                ],
                "status": "not_started",
            },
        ]

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# CSDDD Readiness Assessment Report\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Directive:** Directive (EU) 2024/1760 (CSDDD)"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary as markdown."""
        sec = self._section_executive_summary(data)
        lines = [
            f"## {sec['title']}\n",
            f"### Overall Readiness: {sec['overall_readiness_pct']:.1f}%\n",
            f"**Status:** {sec['readiness_status']}  \n"
            f"**Assessment Date:** {sec['assessment_date']}\n",
            "| Metric | Value |",
            "|--------|------:|",
            f"| Articles Compliant | {sec['articles_compliant']} |",
            f"| Articles Partially Compliant | {sec['articles_partial']} |",
            f"| Articles Non-Compliant | {sec['articles_non_compliant']} |",
            f"| **Total Assessed** | **{sec['total_articles_assessed']}** |",
        ]
        if sec["key_findings"]:
            lines.append("\n**Key Findings:**")
            for finding in sec["key_findings"]:
                lines.append(f"- {finding}")
        return "\n".join(lines)

    def _md_scope_determination(self, data: Dict[str, Any]) -> str:
        """Render scope determination as markdown."""
        sec = self._section_scope_determination(data)
        in_scope = "Yes" if sec["is_in_scope"] else "No"
        high_risk = "Yes" if sec["high_risk_sector"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"| Criterion | Value |\n|-----------|-------|\n"
            f"| Entity Type | {sec['entity_type']} |\n"
            f"| Employees | {sec['employee_count']:,} |\n"
            f"| Net Turnover (EUR) | {sec['net_turnover_eur']:,.0f} |\n"
            f"| In Scope | {in_scope} |\n"
            f"| Scope Tier | {sec['scope_tier']} |\n"
            f"| High-Risk Sector | {high_risk} |\n"
            f"| Application Date | {sec['application_date']} |\n"
            f"| Jurisdiction | {sec['member_state_jurisdiction']} |"
        )

    def _md_article_status(self, data: Dict[str, Any]) -> str:
        """Render article-by-article status as markdown."""
        sec = self._section_article_status(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Articles Assessed:** {sec['total_assessed']} / {len(sec['articles'])}\n",
            "| Article | Title | Category | Status | Compliance |",
            "|--------:|-------|----------|--------|----------:|",
        ]
        for art in sec["articles"]:
            status_display = art["status"].replace("_", " ").title()
            lines.append(
                f"| Art {art['article']} | {art['title']} | "
                f"{art['category']} | {status_display} | "
                f"{art['compliance_pct']:.1f}% |"
            )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render gap analysis as markdown."""
        sec = self._section_gap_analysis(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Gaps:** {sec['total_gaps']}  \n"
            f"**Total Effort Estimate:** {sec['total_effort_days']} days\n",
            f"- High Severity: {sec['high_severity']}",
            f"- Medium Severity: {sec['medium_severity']}",
            f"- Low Severity: {sec['low_severity']}\n",
        ]
        if sec["gaps"]:
            lines.append("| Article | Description | Severity | Effort (days) |")
            lines.append("|--------:|-------------|----------|-------------:|")
            for g in sec["gaps"]:
                lines.append(
                    f"| Art {g.get('article', '')} | {g.get('description', '')} | "
                    f"{g.get('severity', '')} | {g.get('effort_estimate_days', 0)} |"
                )
        return "\n".join(lines)

    def _md_readiness_score(self, data: Dict[str, Any]) -> str:
        """Render readiness score as markdown."""
        sec = self._section_readiness_score(data)
        lines = [
            f"## {sec['title']}\n",
            f"### Overall: {sec['overall_score_pct']:.1f}% - {sec['readiness_label']}\n",
            "| Category | Score |",
            "|----------|------:|",
        ]
        for cat, score in sec["category_scores"].items():
            lines.append(f"| {cat.replace('_', ' ').title()} | {score:.1f}% |")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations as markdown."""
        sec = self._section_recommendations(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Actions:** {sec['total_recommendations']}\n",
            "| Priority | Area | Action | Responsible |",
            "|----------|------|--------|-------------|",
        ]
        for r in sec["recommendations"]:
            lines.append(
                f"| {r['priority'].upper()} | {r['area']} | "
                f"{r['action'][:80]} | {r['responsible']} |"
            )
        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        """Render implementation timeline as markdown."""
        sec = self._section_timeline(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Earliest Compliance Deadline:** {sec['earliest_deadline']}  \n"
            f"**Phases:** {sec['total_phases']}\n",
        ]
        for phase in sec["phases"]:
            lines.append(f"### Phase {phase['phase_number']}: {phase['name']}")
            lines.append(
                f"**Period:** {phase['start_date']} to {phase['end_date']}  \n"
                f"**Status:** {phase['status'].replace('_', ' ').title()}\n"
            )
            if phase["key_activities"]:
                lines.append("**Activities:**")
                for act in phase["key_activities"]:
                    lines.append(f"- {act}")
            if phase["deliverables"]:
                lines.append("\n**Deliverables:**")
                for d in phase["deliverables"]:
                    lines.append(f"- {d}")
            lines.append("")
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
            "h3{color:#3949ab;margin-top:1em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8eaf6}"
            ".compliant{color:#2e7d32;font-weight:bold}"
            ".non-compliant{color:#c62828;font-weight:bold}"
            ".partial{color:#ef6c00;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>CSDDD Readiness Assessment Report</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('reporting_year', '')}</p>\n"
            f"<p>Directive (EU) 2024/1760</p>"
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary HTML."""
        sec = self._section_executive_summary(data)
        css_class = "compliant" if sec["overall_readiness_pct"] >= 75.0 else "non-compliant"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{css_class}'>Overall Readiness: "
            f"{sec['overall_readiness_pct']:.1f}%</p>\n"
            f"<p><strong>Status:</strong> {sec['readiness_status']}</p>\n"
            f"<table><tr><th>Metric</th><th>Count</th></tr>"
            f"<tr><td>Compliant</td><td>{sec['articles_compliant']}</td></tr>"
            f"<tr><td>Partial</td><td>{sec['articles_partial']}</td></tr>"
            f"<tr><td>Non-Compliant</td><td>{sec['articles_non_compliant']}</td></tr>"
            f"</table>"
        )

    def _html_scope(self, data: Dict[str, Any]) -> str:
        """Render scope determination HTML."""
        sec = self._section_scope_determination(data)
        in_scope = "Yes" if sec["is_in_scope"] else "No"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Criterion</th><th>Value</th></tr>"
            f"<tr><td>Entity Type</td><td>{sec['entity_type']}</td></tr>"
            f"<tr><td>Employees</td><td>{sec['employee_count']:,}</td></tr>"
            f"<tr><td>Net Turnover</td><td>EUR {sec['net_turnover_eur']:,.0f}</td></tr>"
            f"<tr><td>In Scope</td><td>{in_scope}</td></tr>"
            f"<tr><td>Tier</td><td>{sec['scope_tier']}</td></tr>"
            f"</table>"
        )

    def _html_article_status(self, data: Dict[str, Any]) -> str:
        """Render article status HTML."""
        sec = self._section_article_status(data)
        rows = ""
        for art in sec["articles"]:
            status = art["status"]
            css = "compliant" if status == "compliant" else (
                "partial" if status == "partial" else "non-compliant"
            )
            rows += (
                f"<tr><td>Art {art['article']}</td><td>{art['title']}</td>"
                f"<td class='{css}'>{status.replace('_', ' ').title()}</td>"
                f"<td>{art['compliance_pct']:.1f}%</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Article</th><th>Title</th><th>Status</th>"
            f"<th>Compliance</th></tr>{rows}</table>"
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render gap analysis HTML."""
        sec = self._section_gap_analysis(data)
        rows = "".join(
            f"<tr><td>Art {g.get('article', '')}</td>"
            f"<td>{g.get('description', '')}</td>"
            f"<td>{g.get('severity', '')}</td></tr>"
            for g in sec["gaps"][:15]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p><strong>Total Gaps:</strong> {sec['total_gaps']}</p>\n"
            f"<table><tr><th>Article</th><th>Description</th><th>Severity</th></tr>"
            f"{rows}</table>"
        )

    def _html_readiness(self, data: Dict[str, Any]) -> str:
        """Render readiness score HTML."""
        sec = self._section_readiness_score(data)
        rows = "".join(
            f"<tr><td>{cat.replace('_', ' ').title()}</td>"
            f"<td>{score:.1f}%</td></tr>"
            for cat, score in sec["category_scores"].items()
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p><strong>Overall Score:</strong> {sec['overall_score_pct']:.1f}% "
            f"({sec['readiness_label']})</p>\n"
            f"<table><tr><th>Category</th><th>Score</th></tr>{rows}</table>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations HTML."""
        sec = self._section_recommendations(data)
        rows = "".join(
            f"<tr><td>{r['priority'].upper()}</td><td>{r['area']}</td>"
            f"<td>{r['action'][:80]}</td></tr>"
            for r in sec["recommendations"][:10]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p><strong>Total:</strong> {sec['total_recommendations']}</p>\n"
            f"<table><tr><th>Priority</th><th>Area</th><th>Action</th></tr>"
            f"{rows}</table>"
        )
