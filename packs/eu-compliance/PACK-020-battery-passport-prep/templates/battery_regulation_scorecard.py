# -*- coding: utf-8 -*-
"""
BatteryRegulationScorecardTemplate - EU Battery Regulation Executive Dashboard Scorecard

Renders an executive-level compliance dashboard scorecard aggregating
compliance status across all key articles of Regulation (EU) 2023/1542.
Provides an overall readiness percentage, article-by-article status with
green/amber/red indicators, key performance metrics, regulatory milestone
timeline with deadlines, and prioritized recommendations for achieving
full compliance.

Sections:
    1. Overall Score - Weighted compliance score and readiness status
    2. Article-by-Article Status - Compliance per key article group
    3. Key Metrics - Top-line performance indicators
    4. Timeline - Regulatory milestones and upcoming deadlines
    5. Recommendations - Prioritized actions to close compliance gaps

Author: GreenLang Team
Version: 20.0.0
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
    "article_status",
    "key_metrics",
    "timeline",
    "recommendations",
]

# Key article groups for the scorecard
_ARTICLE_GROUPS: List[Dict[str, Any]] = [
    {"id": "ART7", "name": "Carbon Footprint Declaration",
     "articles": "Art 7", "weight": 0.15,
     "description": "Lifecycle carbon footprint declaration and performance class"},
    {"id": "ART8", "name": "Recycled Content",
     "articles": "Art 8", "weight": 0.10,
     "description": "Mandatory recycled content for Co, Li, Ni, Pb"},
    {"id": "ART10", "name": "Performance & Durability",
     "articles": "Art 10, Annex IV", "weight": 0.10,
     "description": "Electrochemical performance and durability parameters"},
    {"id": "ART13", "name": "Labelling & Marking",
     "articles": "Art 13-14", "weight": 0.10,
     "description": "Battery labelling, marking, and information requirements"},
    {"id": "ART48", "name": "Supply Chain Due Diligence",
     "articles": "Art 48-52", "weight": 0.15,
     "description": "OECD-aligned supply chain due diligence policies"},
    {"id": "ART56", "name": "End-of-Life Management",
     "articles": "Art 56-71", "weight": 0.10,
     "description": "Collection, recycling, and material recovery"},
    {"id": "ART77", "name": "Battery Passport",
     "articles": "Art 77-78, Annex XIII", "weight": 0.15,
     "description": "Digital battery passport with QR code access"},
    {"id": "SAFETY", "name": "Safety Requirements",
     "articles": "Art 6, Annex V", "weight": 0.10,
     "description": "Safety parameters, risk assessment, UN 38.3 testing"},
    {"id": "CONFORM", "name": "Conformity Assessment",
     "articles": "Art 17-20", "weight": 0.05,
     "description": "EU Declaration of Conformity and CE marking"},
]

# Key regulatory milestones
_MILESTONES: List[Dict[str, Any]] = [
    {"date": "2024-02-18", "milestone": "Battery Regulation enters into force",
     "article": "Art 96", "status_key": "regulation_in_force"},
    {"date": "2024-08-18", "milestone": "Carbon footprint calculation rules apply",
     "article": "Art 7(1)", "status_key": "cf_calc_rules"},
    {"date": "2025-02-18", "milestone": "Carbon footprint declaration mandatory",
     "article": "Art 7(1)", "status_key": "cf_declaration"},
    {"date": "2025-08-18", "milestone": "Due diligence policies required",
     "article": "Art 48", "status_key": "dd_policies"},
    {"date": "2025-08-18", "milestone": "Recycled content documentation required",
     "article": "Art 8(1)", "status_key": "rc_documentation"},
    {"date": "2026-08-18", "milestone": "Carbon footprint performance class label",
     "article": "Art 7(2)", "status_key": "cf_class_label"},
    {"date": "2027-02-18", "milestone": "Battery passport mandatory (industrial/EV)",
     "article": "Art 77", "status_key": "battery_passport"},
    {"date": "2027-08-18", "milestone": "Material recovery targets Phase 1",
     "article": "Annex XII", "status_key": "recovery_phase1"},
    {"date": "2028-08-18", "milestone": "Maximum lifecycle CF threshold",
     "article": "Art 7(3)", "status_key": "cf_threshold"},
    {"date": "2031-08-18", "milestone": "Recycled content Phase 1 targets",
     "article": "Art 8(4)(a)", "status_key": "rc_phase1"},
    {"date": "2031-08-18", "milestone": "Material recovery targets Phase 2",
     "article": "Annex XII", "status_key": "recovery_phase2"},
    {"date": "2036-08-18", "milestone": "Recycled content Phase 2 targets",
     "article": "Art 8(4)(b)", "status_key": "rc_phase2"},
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

class BatteryRegulationScorecardTemplate:
    """
    Battery Regulation Executive Dashboard Scorecard.

    Aggregates compliance status across all key articles of the EU Battery
    Regulation to provide an overall readiness score, article-level traffic
    light indicators, key performance metrics, regulatory milestone timeline,
    and prioritized recommendations.

    Regulatory References:
        - Regulation (EU) 2023/1542 (EU Battery Regulation)
        - All delegated and implementing acts thereunder

    Example:
        >>> tpl = BatteryRegulationScorecardTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BatteryRegulationScorecardTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "report_id": _new_uuid(),
            "generated_at": self.generated_at.isoformat(),
        }
        for section in _SECTIONS:
            result[section] = self.render_section(section, data)
        result["provenance_hash"] = _compute_hash(result)
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
        if "article_scores" not in data:
            errors.append(
                "article_scores dict is required (keys: ART7, ART8, ART10, "
                "ART13, ART48, ART56, ART77, SAFETY, CONFORM)"
            )
        if not data.get("reporting_year"):
            warnings.append("reporting_year not specified; using current year")
        if not data.get("battery_type"):
            warnings.append("battery_type not specified; defaulting to 'ev_battery'")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render scorecard as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_overall_score(data),
            self._md_article_status(data),
            self._md_key_metrics(data),
            self._md_timeline(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render scorecard as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overall_score(data),
            self._html_article_status(data),
            self._html_key_metrics(data),
            self._html_timeline(data),
            self._html_recommendations(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Battery Regulation Scorecard</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render scorecard as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "battery_regulation_scorecard",
            "regulation_reference": "EU Battery Regulation 2023/1542",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "overall_score": self._section_overall_score(data),
            "article_status": self._section_article_status(data),
            "key_metrics": self._section_key_metrics(data),
            "timeline": self._section_timeline(data),
            "recommendations": self._section_recommendations(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_overall_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build overall compliance score section."""
        scores = data.get("article_scores", {})
        weighted_total = 0.0
        total_weight = 0.0

        article_scores: List[Dict[str, Any]] = []
        for group in _ARTICLE_GROUPS:
            score = scores.get(group["id"], 0.0)
            weighted_total += score * group["weight"]
            total_weight += group["weight"]
            article_scores.append({
                "id": group["id"],
                "name": group["name"],
                "score": round(score, 1),
                "weight": group["weight"],
                "weighted_score": round(score * group["weight"], 2),
            })

        overall = round(weighted_total / total_weight, 1) if total_weight > 0 else 0.0
        green_count = sum(1 for a in article_scores if a["score"] >= 80.0)
        amber_count = sum(
            1 for a in article_scores if 50.0 <= a["score"] < 80.0
        )
        red_count = sum(1 for a in article_scores if a["score"] < 50.0)

        return {
            "title": "Overall Compliance Score",
            "overall_score_pct": overall,
            "readiness_status": self._readiness_status(overall),
            "total_article_groups": len(_ARTICLE_GROUPS),
            "green_count": green_count,
            "amber_count": amber_count,
            "red_count": red_count,
            "article_scores": article_scores,
            "assessment_date": data.get(
                "assessment_date", utcnow().strftime("%Y-%m-%d")
            ),
        }

    def _section_article_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build article-by-article status section."""
        scores = data.get("article_scores", {})
        details = data.get("article_details", {})

        statuses: List[Dict[str, Any]] = []
        for group in _ARTICLE_GROUPS:
            score = scores.get(group["id"], 0.0)
            detail = details.get(group["id"], {})
            statuses.append({
                "id": group["id"],
                "name": group["name"],
                "articles": group["articles"],
                "description": group["description"],
                "score": round(score, 1),
                "traffic_light": self._traffic_light(score),
                "key_findings": detail.get("findings", []),
                "gaps": detail.get("gaps", []),
                "gap_count": len(detail.get("gaps", [])),
                "evidence_documents": detail.get("evidence", []),
                "responsible_team": detail.get("responsible", ""),
                "last_assessed": detail.get("last_assessed", ""),
            })

        return {
            "title": "Article-by-Article Compliance Status",
            "statuses": statuses,
            "total_gaps": sum(s["gap_count"] for s in statuses),
        }

    def _section_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build key performance metrics section."""
        metrics = data.get("metrics", {})
        return {
            "title": "Key Performance Metrics",
            "metrics": [
                {
                    "name": "Carbon Footprint",
                    "value": metrics.get("carbon_footprint_kgco2e_per_kwh", 0.0),
                    "unit": "kgCO2e/kWh",
                    "target": metrics.get("cf_target_kgco2e_per_kwh", 0.0),
                    "status": self._metric_status(
                        metrics.get("carbon_footprint_kgco2e_per_kwh", 0.0),
                        metrics.get("cf_target_kgco2e_per_kwh", 0.0),
                        lower_is_better=True,
                    ),
                },
                {
                    "name": "Recycled Content (avg)",
                    "value": metrics.get("recycled_content_avg_pct", 0.0),
                    "unit": "%",
                    "target": metrics.get("rc_target_avg_pct", 0.0),
                    "status": self._metric_status(
                        metrics.get("recycled_content_avg_pct", 0.0),
                        metrics.get("rc_target_avg_pct", 0.0),
                        lower_is_better=False,
                    ),
                },
                {
                    "name": "DD Supplier Coverage",
                    "value": metrics.get("dd_coverage_pct", 0.0),
                    "unit": "%",
                    "target": 100.0,
                    "status": self._metric_status(
                        metrics.get("dd_coverage_pct", 0.0),
                        100.0,
                        lower_is_better=False,
                    ),
                },
                {
                    "name": "Collection Rate",
                    "value": metrics.get("collection_rate_pct", 0.0),
                    "unit": "%",
                    "target": metrics.get("collection_target_pct", 0.0),
                    "status": self._metric_status(
                        metrics.get("collection_rate_pct", 0.0),
                        metrics.get("collection_target_pct", 0.0),
                        lower_is_better=False,
                    ),
                },
                {
                    "name": "Recycling Efficiency",
                    "value": metrics.get("recycling_efficiency_pct", 0.0),
                    "unit": "%",
                    "target": metrics.get("recycling_target_pct", 0.0),
                    "status": self._metric_status(
                        metrics.get("recycling_efficiency_pct", 0.0),
                        metrics.get("recycling_target_pct", 0.0),
                        lower_is_better=False,
                    ),
                },
                {
                    "name": "Labelling Completeness",
                    "value": metrics.get("labelling_compliance_pct", 0.0),
                    "unit": "%",
                    "target": 100.0,
                    "status": self._metric_status(
                        metrics.get("labelling_compliance_pct", 0.0),
                        100.0,
                        lower_is_better=False,
                    ),
                },
                {
                    "name": "Battery Passport Readiness",
                    "value": metrics.get("passport_readiness_pct", 0.0),
                    "unit": "%",
                    "target": 100.0,
                    "status": self._metric_status(
                        metrics.get("passport_readiness_pct", 0.0),
                        100.0,
                        lower_is_better=False,
                    ),
                },
                {
                    "name": "State of Health (avg fleet)",
                    "value": metrics.get("avg_soh_pct", 0.0),
                    "unit": "%",
                    "target": 80.0,
                    "status": self._metric_status(
                        metrics.get("avg_soh_pct", 0.0),
                        80.0,
                        lower_is_better=False,
                    ),
                },
            ],
        }

    def _section_timeline(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build regulatory milestone timeline section."""
        milestone_status = data.get("milestone_status", {})
        reporting_year = data.get("reporting_year", utcnow().year)
        now_str = utcnow().strftime("%Y-%m-%d")

        timeline_items: List[Dict[str, Any]] = []
        upcoming: List[Dict[str, Any]] = []
        overdue: List[Dict[str, Any]] = []

        for ms in _MILESTONES:
            status = milestone_status.get(ms["status_key"], "not_started")
            is_past = ms["date"] <= now_str
            is_overdue = is_past and status not in ("completed", "compliant")

            item = {
                "date": ms["date"],
                "milestone": ms["milestone"],
                "article": ms["article"],
                "status": status,
                "is_past": is_past,
                "is_overdue": is_overdue,
            }
            timeline_items.append(item)

            if is_overdue:
                overdue.append(item)
            elif not is_past:
                upcoming.append(item)

        return {
            "title": "Regulatory Milestone Timeline",
            "total_milestones": len(timeline_items),
            "completed_milestones": sum(
                1 for t in timeline_items if t["status"] in ("completed", "compliant")
            ),
            "overdue_milestones": len(overdue),
            "upcoming_milestones": len(upcoming),
            "timeline": timeline_items,
            "overdue": overdue,
            "next_upcoming": upcoming[:3] if upcoming else [],
        }

    def _section_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build prioritized recommendations section."""
        scores = data.get("article_scores", {})
        article_details = data.get("article_details", {})
        recommendations: List[Dict[str, Any]] = []
        rank = 0

        # Generate recommendations from lowest-scoring articles
        sorted_groups = sorted(
            _ARTICLE_GROUPS,
            key=lambda g: scores.get(g["id"], 0.0),
        )

        for group in sorted_groups:
            score = scores.get(group["id"], 0.0)
            if score >= 95.0:
                continue  # No recommendation needed
            rank += 1
            detail = article_details.get(group["id"], {})
            gaps = detail.get("gaps", [])
            priority = "CRITICAL" if score < 50.0 else "HIGH" if score < 80.0 else "MEDIUM"

            recommendations.append({
                "rank": rank,
                "priority": priority,
                "article_group": group["id"],
                "name": group["name"],
                "articles": group["articles"],
                "current_score": round(score, 1),
                "target_score": 95.0,
                "score_gap": round(95.0 - score, 1),
                "gaps_identified": len(gaps),
                "top_gaps": gaps[:3],
                "recommended_actions": self._generate_actions(group["id"], score, gaps),
                "estimated_effort": self._estimate_group_effort(group["id"], score),
                "deadline": self._recommend_group_deadline(group["id"]),
            })

        critical_count = sum(1 for r in recommendations if r["priority"] == "CRITICAL")
        high_count = sum(1 for r in recommendations if r["priority"] == "HIGH")

        return {
            "title": "Prioritized Recommendations",
            "total_recommendations": len(recommendations),
            "critical_count": critical_count,
            "high_count": high_count,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Battery Regulation Compliance Scorecard\n"
            f"## EU Battery Regulation (EU) 2023/1542\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Battery Type:** {data.get('battery_type', 'ev_battery')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}"
        )

    def _md_overall_score(self, data: Dict[str, Any]) -> str:
        """Render overall score as markdown."""
        sec = self._section_overall_score(data)
        lines = [
            f"## {sec['title']}\n",
            f"### {sec['overall_score_pct']:.1f}% - {sec['readiness_status']}\n",
            f"**Assessment Date:** {sec['assessment_date']}\n",
            f"| Indicator | Count |",
            f"|-----------|------:|",
            f"| Green (>=80%) | {sec['green_count']} |",
            f"| Amber (50-79%) | {sec['amber_count']} |",
            f"| Red (<50%) | {sec['red_count']} |",
            f"| **Total Groups** | **{sec['total_article_groups']}** |",
        ]
        return "\n".join(lines)

    def _md_article_status(self, data: Dict[str, Any]) -> str:
        """Render article-by-article status as markdown."""
        sec = self._section_article_status(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Gaps:** {sec['total_gaps']}\n",
            "| ID | Area | Articles | Score | Status | Gaps |",
            "|----|------|----------|------:|:------:|-----:|",
        ]
        for s in sec["statuses"]:
            lines.append(
                f"| {s['id']} | {s['name']} | {s['articles']} | "
                f"{s['score']:.1f}% | {s['traffic_light']} | {s['gap_count']} |"
            )
        return "\n".join(lines)

    def _md_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render key metrics as markdown."""
        sec = self._section_key_metrics(data)
        lines = [
            f"## {sec['title']}\n",
            "| Metric | Value | Target | Status |",
            "|--------|------:|-------:|:------:|",
        ]
        for m in sec["metrics"]:
            val_str = f"{m['value']:.1f}" if isinstance(m["value"], float) else str(m["value"])
            tgt_str = f"{m['target']:.1f}" if isinstance(m["target"], float) else str(m["target"])
            lines.append(
                f"| {m['name']} | {val_str} {m['unit']} | "
                f"{tgt_str} {m['unit']} | {m['status']} |"
            )
        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        """Render timeline as markdown."""
        sec = self._section_timeline(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Completed:** {sec['completed_milestones']}/{sec['total_milestones']}  \n"
            f"**Overdue:** {sec['overdue_milestones']}  \n"
            f"**Upcoming:** {sec['upcoming_milestones']}\n",
        ]
        if sec["overdue"]:
            lines.append("### OVERDUE Milestones\n")
            for item in sec["overdue"]:
                lines.append(
                    f"- **{item['date']}** [{item['article']}]: "
                    f"{item['milestone']} - {item['status'].upper()}"
                )
            lines.append("")

        lines.append("### Full Timeline\n")
        lines.append("| Date | Milestone | Article | Status |")
        lines.append("|------|-----------|---------|:------:|")
        for item in sec["timeline"]:
            status_display = item["status"].upper()
            if item["is_overdue"]:
                status_display = f"**OVERDUE**"
            lines.append(
                f"| {item['date']} | {item['milestone']} | "
                f"{item['article']} | {status_display} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations as markdown."""
        sec = self._section_recommendations(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total:** {sec['total_recommendations']}  \n"
            f"**Critical:** {sec['critical_count']}  \n"
            f"**High:** {sec['high_count']}\n",
        ]
        for rec in sec["recommendations"]:
            lines.append(
                f"### {rec['rank']}. [{rec['priority']}] {rec['name']} "
                f"({rec['articles']})\n"
            )
            lines.append(
                f"**Current:** {rec['current_score']:.1f}% | "
                f"**Gap:** {rec['score_gap']:.1f}% | "
                f"**Effort:** {rec['estimated_effort']} | "
                f"**Deadline:** {rec['deadline']}\n"
            )
            if rec["recommended_actions"]:
                lines.append("**Actions:**")
                for action in rec["recommended_actions"]:
                    lines.append(f"  - {action}")
            lines.append("")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n*Scorecard generated by PACK-020 Battery Passport Prep Pack on {ts}*\n"
            f"*Regulation: EU Battery Regulation (EU) 2023/1542*"
        )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1200px;margin:auto}"
            "h1{color:#0d47a1;border-bottom:2px solid #0d47a1;padding-bottom:.3em}"
            "h2{color:#1565c0;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e3f2fd}"
            ".green{color:#2e7d32;font-weight:bold}"
            ".amber{color:#e65100;font-weight:bold}"
            ".red{color:#c62828;font-weight:bold}"
            ".score-display{font-size:2em;font-weight:bold;text-align:center;"
            "padding:16px;border-radius:8px;margin:16px 0}"
            ".score-high{background:#c8e6c9;color:#1b5e20}"
            ".score-mid{background:#fff9c4;color:#f57f17}"
            ".score-low{background:#ffcdd2;color:#b71c1c}"
            ".overdue{background:#ffcdd2}"
            ".metric-row td:last-child{font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Battery Regulation Compliance Scorecard</h1>\n"
            f"<p>EU Battery Regulation (EU) 2023/1542</p>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('battery_type', 'ev_battery')} | "
            f"Year: {data.get('reporting_year', '')}</p>"
        )

    def _html_overall_score(self, data: Dict[str, Any]) -> str:
        """Render overall score HTML."""
        sec = self._section_overall_score(data)
        score = sec["overall_score_pct"]
        score_cls = (
            "score-high" if score >= 80.0
            else "score-mid" if score >= 50.0
            else "score-low"
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<div class='score-display {score_cls}'>"
            f"{score:.1f}% - {sec['readiness_status']}</div>\n"
            f"<table><tr><th>Indicator</th><th>Count</th></tr>"
            f"<tr><td class='green'>Green (>=80%)</td><td>{sec['green_count']}</td></tr>"
            f"<tr><td class='amber'>Amber (50-79%)</td><td>{sec['amber_count']}</td></tr>"
            f"<tr><td class='red'>Red (<50%)</td><td>{sec['red_count']}</td></tr></table>"
        )

    def _html_article_status(self, data: Dict[str, Any]) -> str:
        """Render article status HTML."""
        sec = self._section_article_status(data)
        rows = ""
        for s in sec["statuses"]:
            light_cls = (
                "green" if s["traffic_light"] == "GREEN"
                else "amber" if s["traffic_light"] == "AMBER"
                else "red"
            )
            rows += (
                f"<tr><td>{s['id']}</td><td>{s['name']}</td>"
                f"<td>{s['articles']}</td><td>{s['score']:.1f}%</td>"
                f"<td class='{light_cls}'>{s['traffic_light']}</td>"
                f"<td>{s['gap_count']}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>ID</th><th>Area</th><th>Articles</th>"
            f"<th>Score</th><th>Status</th><th>Gaps</th></tr>{rows}</table>"
        )

    def _html_key_metrics(self, data: Dict[str, Any]) -> str:
        """Render key metrics HTML."""
        sec = self._section_key_metrics(data)
        rows = ""
        for m in sec["metrics"]:
            val_str = f"{m['value']:.1f}" if isinstance(m["value"], float) else str(m["value"])
            cls = (
                "green" if m["status"] == "ON_TRACK"
                else "amber" if m["status"] == "AT_RISK"
                else "red"
            )
            rows += (
                f"<tr class='metric-row'><td>{m['name']}</td>"
                f"<td>{val_str} {m['unit']}</td>"
                f"<td class='{cls}'>{m['status']}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Metric</th><th>Value</th><th>Status</th></tr>"
            f"{rows}</table>"
        )

    def _html_timeline(self, data: Dict[str, Any]) -> str:
        """Render timeline HTML."""
        sec = self._section_timeline(data)
        rows = ""
        for item in sec["timeline"]:
            cls = " class='overdue'" if item["is_overdue"] else ""
            status = "OVERDUE" if item["is_overdue"] else item["status"].upper()
            rows += (
                f"<tr{cls}><td>{item['date']}</td><td>{item['milestone']}</td>"
                f"<td>{item['article']}</td><td>{status}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Completed: {sec['completed_milestones']}/{sec['total_milestones']} | "
            f"Overdue: {sec['overdue_milestones']}</p>\n"
            f"<table><tr><th>Date</th><th>Milestone</th><th>Article</th>"
            f"<th>Status</th></tr>{rows}</table>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations HTML."""
        sec = self._section_recommendations(data)
        rows = ""
        for rec in sec["recommendations"]:
            cls = (
                "red" if rec["priority"] == "CRITICAL"
                else "amber" if rec["priority"] == "HIGH"
                else "green"
            )
            rows += (
                f"<tr><td>{rec['rank']}</td>"
                f"<td class='{cls}'>{rec['priority']}</td>"
                f"<td>{rec['name']}</td><td>{rec['articles']}</td>"
                f"<td>{rec['current_score']:.1f}%</td>"
                f"<td>{rec['estimated_effort']}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>#</th><th>Priority</th><th>Area</th>"
            f"<th>Articles</th><th>Score</th><th>Effort</th></tr>{rows}</table>"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _readiness_status(self, score: float) -> str:
        """Determine overall readiness status from score."""
        if score >= 95.0:
            return "Fully Compliant"
        elif score >= 80.0:
            return "Near Ready (Minor Gaps)"
        elif score >= 60.0:
            return "Partially Ready (Significant Gaps)"
        elif score >= 40.0:
            return "Not Ready (Major Gaps)"
        return "Far from Ready (Critical Gaps)"

    def _traffic_light(self, score: float) -> str:
        """Determine traffic light colour from score."""
        if score >= 80.0:
            return "GREEN"
        elif score >= 50.0:
            return "AMBER"
        return "RED"

    def _metric_status(
        self, value: float, target: float, lower_is_better: bool = False
    ) -> str:
        """Determine metric status (ON_TRACK, AT_RISK, OFF_TRACK)."""
        if target == 0.0:
            return "NO_TARGET"
        if lower_is_better:
            if value <= target:
                return "ON_TRACK"
            elif value <= target * 1.2:
                return "AT_RISK"
            return "OFF_TRACK"
        else:
            if value >= target:
                return "ON_TRACK"
            elif value >= target * 0.8:
                return "AT_RISK"
            return "OFF_TRACK"

    def _generate_actions(
        self, group_id: str, score: float, gaps: List[Any]
    ) -> List[str]:
        """Generate deterministic recommended actions for an article group."""
        actions_map: Dict[str, List[str]] = {
            "ART7": [
                "Complete lifecycle carbon footprint calculation per Delegated Act",
                "Obtain third-party verification of CF declaration",
                "Determine and label carbon footprint performance class",
            ],
            "ART8": [
                "Audit recycled content in supply chain for Co, Li, Ni, Pb",
                "Establish chain-of-custody documentation for recycled materials",
                "Secure additional recycled material supply contracts",
            ],
            "ART10": [
                "Conduct performance testing per IEC 62660-1 / IEC 62620",
                "Document rated capacity, cycle life, and efficiency data",
                "Implement SoH monitoring via battery management system",
            ],
            "ART13": [
                "Review battery labels against 20-element checklist",
                "Add missing mandatory markings (CE, waste bin, capacity)",
                "Implement QR code linking to battery passport",
            ],
            "ART48": [
                "Publish supply chain due diligence policy aligned with OECD",
                "Complete risk assessment of tier-1 and tier-2 suppliers",
                "Commission independent third-party audit of supply chain",
            ],
            "ART56": [
                "Register with battery collection scheme per Art 57",
                "Ensure recycling process meets efficiency targets",
                "Track per-material recovery rates for Co, Cu, Li, Ni, Pb",
            ],
            "ART77": [
                "Implement battery passport data infrastructure",
                "Populate all Annex XIII data fields for each battery model",
                "Deploy QR code on battery linking to passport endpoint",
            ],
            "SAFETY": [
                "Complete UN 38.3 transport safety testing",
                "Document safety risk assessment per Annex V",
                "Ensure safety instructions accompany each battery",
            ],
            "CONFORM": [
                "Prepare EU Declaration of Conformity",
                "Engage notified body for conformity assessment (if required)",
                "Affix CE marking to battery and packaging",
            ],
        }
        base_actions = actions_map.get(group_id, ["Review compliance gaps and create action plan"])
        if score < 50.0:
            return base_actions + ["Assign dedicated compliance lead for this area"]
        if score < 80.0:
            return base_actions[:2]
        return base_actions[:1]

    def _estimate_group_effort(self, group_id: str, score: float) -> str:
        """Estimate remediation effort for an article group."""
        if score >= 80.0:
            return "1-2 weeks"
        elif score >= 50.0:
            return "1-3 months"
        return "3-6 months"

    def _recommend_group_deadline(self, group_id: str) -> str:
        """Recommend deadline based on article group regulatory timeline."""
        deadline_map = {
            "ART7": "2025-02-18",
            "ART8": "2025-08-18",
            "ART10": "2027-02-18",
            "ART13": "2025-08-18",
            "ART48": "2025-08-18",
            "ART56": "2025-08-18",
            "ART77": "2027-02-18",
            "SAFETY": "2025-02-18",
            "CONFORM": "2025-02-18",
        }
        return deadline_map.get(group_id, "As soon as practicable")
