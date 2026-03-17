# -*- coding: utf-8 -*-
"""
S1WorkforceReportTemplate - ESRS S1 Own Workforce Report

Renders workforce policies, engagement processes, remediation channels,
actions, targets, demographics, non-employee workers, collective bargaining,
diversity, wages, social protection, disability, training, health and safety,
work-life balance, remuneration, and incidents per ESRS S1.

Sections:
    1. Workforce Policies
    2. Engagement Processes
    3. Remediation Channels
    4. Actions Summary
    5. Workforce Targets
    6. Employee Demographics
    7. Non-Employee Workers
    8. Collective Bargaining
    9. Diversity Metrics
    10. Adequate Wages
    11. Social Protection
    12. Disability Inclusion
    13. Training Development
    14. Health Safety
    15. Work-Life Balance
    16. Remuneration Fairness
    17. Incidents Complaints

Author: GreenLang Team
Version: 17.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SECTIONS: List[str] = [
    "workforce_policies",
    "engagement_processes",
    "remediation_channels",
    "actions_summary",
    "workforce_targets",
    "employee_demographics",
    "non_employee_workers",
    "collective_bargaining",
    "diversity_metrics",
    "adequate_wages",
    "social_protection",
    "disability_inclusion",
    "training_development",
    "health_safety",
    "work_life_balance",
    "remuneration_fairness",
    "incidents_complaints",
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


class S1WorkforceReportTemplate:
    """
    ESRS S1 Own Workforce report template.

    Renders comprehensive workforce disclosures including policies,
    engagement, remediation, demographics, collective bargaining,
    diversity, adequate wages, training, health and safety, work-life
    balance, remuneration fairness, and incident reporting per ESRS S1.

    Example:
        >>> tpl = S1WorkforceReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize S1WorkforceReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {}
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
        if "employees" not in data:
            warnings.append("employees missing; will default to empty")
        if "health_safety_data" not in data:
            warnings.append("health_safety_data missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render S1 Own Workforce report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_workforce_policies(data),
            self._md_engagement(data), self._md_remediation(data),
            self._md_actions(data), self._md_targets(data),
            self._md_demographics(data), self._md_non_employees(data),
            self._md_collective_bargaining(data), self._md_diversity(data),
            self._md_wages(data), self._md_social_protection(data),
            self._md_disability(data), self._md_training(data),
            self._md_health_safety(data), self._md_work_life(data),
            self._md_remuneration(data), self._md_incidents(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render S1 Own Workforce report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data), self._html_demographics(data),
            self._html_diversity(data), self._html_health_safety(data),
            self._html_incidents(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>ESRS S1 Own Workforce Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render S1 Own Workforce report as JSON string."""
        self.generated_at = _utcnow()
        result = {
            "template": "s1_workforce_report", "esrs_reference": "ESRS S1",
            "version": "17.0.0", "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "total_employees": data.get("total_employees", 0),
            "female_pct": data.get("female_pct", 0.0),
            "gender_pay_gap_pct": data.get("gender_pay_gap_pct", 0.0),
            "training_hours_per_employee": data.get("training_hours_per_employee", 0.0),
            "ltir": data.get("lost_time_injury_rate", 0.0),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_workforce_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build workforce policies section."""
        policies = data.get("workforce_policies", [])
        return {
            "title": "Workforce Policies", "policy_count": len(policies),
            "policies": [{"name": p.get("name", ""), "scope": p.get("scope", ""),
                          "ilo_aligned": p.get("ilo_aligned", False)} for p in policies],
        }

    def _section_engagement_processes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build engagement processes section."""
        return {
            "title": "Worker Engagement Processes",
            "has_works_council": data.get("has_works_council", False),
            "engagement_methods": data.get("engagement_methods", []),
            "survey_participation_pct": data.get("survey_participation_pct", 0.0),
            "engagement_score": data.get("engagement_score", 0.0),
        }

    def _section_remediation_channels(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build remediation channels section."""
        channels = data.get("remediation_channels", [])
        return {
            "title": "Remediation and Grievance Channels",
            "channel_count": len(channels),
            "channels": [{"name": c.get("name", ""), "type": c.get("type", ""),
                          "accessible_to": c.get("accessible_to", ""),
                          "anonymous": c.get("anonymous", False)} for c in channels],
        }

    def _section_actions_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build actions summary section."""
        actions = data.get("workforce_actions", [])
        return {
            "title": "Workforce Actions Summary", "action_count": len(actions),
            "actions": [{"description": a.get("description", ""), "status": a.get("status", ""),
                         "investment_eur": a.get("investment_eur", 0.0)} for a in actions],
        }

    def _section_workforce_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build workforce targets section."""
        targets = data.get("workforce_targets", [])
        return {
            "title": "Workforce Targets", "target_count": len(targets),
            "targets": [{"name": t.get("name", ""), "metric": t.get("metric", ""),
                         "target_year": t.get("target_year", ""),
                         "target_value": t.get("target_value", 0.0),
                         "current_value": t.get("current_value", 0.0)} for t in targets],
        }

    def _section_employee_demographics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build employee demographics section."""
        return {
            "title": "Employee Demographics",
            "total_headcount": data.get("total_employees", 0),
            "permanent": data.get("permanent_employees", 0),
            "temporary": data.get("temporary_employees", 0),
            "full_time": data.get("full_time_employees", 0),
            "part_time": data.get("part_time_employees", 0),
            "by_region": data.get("employees_by_region", {}),
            "turnover_rate_pct": round(data.get("turnover_rate_pct", 0.0), 1),
        }

    def _section_non_employee_workers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build non-employee workers section."""
        return {
            "title": "Non-Employee Workers",
            "total_count": data.get("non_employee_workers_count", 0),
            "contractors": data.get("contractors_count", 0),
            "agency_workers": data.get("agency_workers_count", 0),
            "types": data.get("non_employee_types", []),
        }

    def _section_collective_bargaining(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build collective bargaining section."""
        return {
            "title": "Collective Bargaining",
            "coverage_pct": round(data.get("collective_bargaining_pct", 0.0), 1),
            "social_dialogue_coverage_pct": round(data.get("social_dialogue_pct", 0.0), 1),
            "agreements_count": data.get("cba_count", 0),
        }

    def _section_diversity_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build diversity metrics section."""
        return {
            "title": "Diversity Metrics",
            "female_pct": round(data.get("female_pct", 0.0), 1),
            "female_management_pct": round(data.get("female_management_pct", 0.0), 1),
            "female_board_pct": round(data.get("female_board_pct", 0.0), 1),
            "age_distribution": data.get("age_distribution", {}),
            "nationality_count": data.get("nationality_count", 0),
        }

    def _section_adequate_wages(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build adequate wages section."""
        return {
            "title": "Adequate Wages",
            "living_wage_compliant": data.get("living_wage_compliant", False),
            "lowest_wage_ratio": data.get("lowest_wage_to_living_wage_ratio", 0.0),
            "countries_assessed": data.get("wage_countries_assessed", 0),
        }

    def _section_social_protection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build social protection section."""
        return {
            "title": "Social Protection",
            "covered_pct": round(data.get("social_protection_coverage_pct", 0.0), 1),
            "benefits_offered": data.get("benefits_offered", []),
            "parental_leave_eligible_pct": round(data.get("parental_leave_eligible_pct", 0.0), 1),
        }

    def _section_disability_inclusion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build disability inclusion section."""
        return {
            "title": "Persons with Disabilities",
            "disability_pct": round(data.get("disability_pct", 0.0), 1),
            "accessibility_measures": data.get("accessibility_measures", []),
            "accommodation_requests_met": data.get("accommodation_requests_met", 0),
        }

    def _section_training_development(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build training and development section."""
        return {
            "title": "Training and Skills Development",
            "avg_training_hours": round(data.get("training_hours_per_employee", 0.0), 1),
            "training_investment_eur": data.get("training_investment_eur", 0.0),
            "skills_programs": data.get("skills_programs", []),
            "performance_review_pct": round(data.get("performance_review_pct", 0.0), 1),
        }

    def _section_health_safety(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build health and safety section."""
        hs = data.get("health_safety_data", {})
        return {
            "title": "Health and Safety",
            "fatalities": hs.get("fatalities", 0),
            "recordable_incidents": hs.get("recordable_incidents", 0),
            "lost_time_injury_rate": round(hs.get("ltir", 0.0), 2),
            "total_recordable_rate": round(hs.get("trir", 0.0), 2),
            "lost_days": hs.get("lost_days", 0),
            "osh_management_system": hs.get("osh_management_system", ""),
            "workers_covered_pct": round(hs.get("workers_covered_pct", 0.0), 1),
        }

    def _section_work_life_balance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build work-life balance section."""
        return {
            "title": "Work-Life Balance",
            "family_leave_uptake_pct": round(data.get("family_leave_uptake_pct", 0.0), 1),
            "flexible_work_pct": round(data.get("flexible_work_pct", 0.0), 1),
            "avg_working_hours_per_week": round(data.get("avg_working_hours_per_week", 0.0), 1),
        }

    def _section_remuneration_fairness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build remuneration fairness section."""
        return {
            "title": "Remuneration Fairness",
            "gender_pay_gap_pct": round(data.get("gender_pay_gap_pct", 0.0), 1),
            "ceo_to_median_ratio": round(data.get("ceo_to_median_ratio", 0.0), 1),
            "total_remuneration_eur": data.get("total_remuneration_eur", 0.0),
        }

    def _section_incidents_complaints(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build incidents and complaints section."""
        return {
            "title": "Incidents and Complaints",
            "discrimination_incidents": data.get("discrimination_incidents", 0),
            "harassment_incidents": data.get("harassment_incidents", 0),
            "grievances_filed": data.get("grievances_filed", 0),
            "grievances_resolved": data.get("grievances_resolved", 0),
            "human_rights_complaints": data.get("human_rights_complaints", 0),
            "fines_eur": data.get("workforce_fines_eur", 0.0),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# ESRS S1 Own Workforce Report\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}  \n**Standard:** ESRS S1 Own Workforce"
        )

    def _md_workforce_policies(self, d: Dict[str, Any]) -> str:
        sec = self._section_workforce_policies(d)
        lines = [f"## {sec['title']}\n"]
        for p in sec["policies"]:
            ilo = "Yes" if p["ilo_aligned"] else "No"
            lines.append(f"- **{p['name']}** (Scope: {p['scope']}, ILO: {ilo})")
        return "\n".join(lines)

    def _md_engagement(self, d: Dict[str, Any]) -> str:
        sec = self._section_engagement_processes(d)
        wc = "Yes" if sec["has_works_council"] else "No"
        return (f"## {sec['title']}\n\n**Works Council:** {wc}  \n"
                f"**Survey Participation:** {sec['survey_participation_pct']:.1f}%  \n"
                f"**Engagement Score:** {sec['engagement_score']:.1f}")

    def _md_remediation(self, d: Dict[str, Any]) -> str:
        sec = self._section_remediation_channels(d)
        lines = [f"## {sec['title']}\n"]
        for c in sec["channels"]:
            anon = "Yes" if c["anonymous"] else "No"
            lines.append(f"- **{c['name']}** ({c['type']}) - Anonymous: {anon}")
        return "\n".join(lines)

    def _md_actions(self, d: Dict[str, Any]) -> str:
        sec = self._section_actions_summary(d)
        lines = [f"## {sec['title']}\n"]
        if sec["actions"]:
            lines.append("| Action | Status | Investment (EUR) |")
            lines.append("|--------|--------|----------------:|")
            for a in sec["actions"]:
                lines.append(f"| {a['description']} | {a['status']} | {a['investment_eur']:,.2f} |")
        return "\n".join(lines)

    def _md_targets(self, d: Dict[str, Any]) -> str:
        sec = self._section_workforce_targets(d)
        lines = [f"## {sec['title']}\n"]
        if sec["targets"]:
            lines.append("| Target | Metric | Current | Goal | Year |")
            lines.append("|--------|--------|--------:|-----:|-----:|")
            for t in sec["targets"]:
                lines.append(f"| {t['name']} | {t['metric']} | {t['current_value']:.1f} | {t['target_value']:.1f} | {t['target_year']} |")
        return "\n".join(lines)

    def _md_demographics(self, d: Dict[str, Any]) -> str:
        sec = self._section_employee_demographics(d)
        return (f"## {sec['title']}\n\n| Metric | Value |\n|--------|------:|\n"
                f"| Total Headcount | {sec['total_headcount']:,} |\n| Permanent | {sec['permanent']:,} |\n"
                f"| Temporary | {sec['temporary']:,} |\n| Full-Time | {sec['full_time']:,} |\n"
                f"| Part-Time | {sec['part_time']:,} |\n| Turnover Rate | {sec['turnover_rate_pct']:.1f}% |")

    def _md_non_employees(self, d: Dict[str, Any]) -> str:
        sec = self._section_non_employee_workers(d)
        return (f"## {sec['title']}\n\n**Total:** {sec['total_count']:,}  \n"
                f"**Contractors:** {sec['contractors']:,}  \n**Agency Workers:** {sec['agency_workers']:,}")

    def _md_collective_bargaining(self, d: Dict[str, Any]) -> str:
        sec = self._section_collective_bargaining(d)
        return (f"## {sec['title']}\n\n- **CBA Coverage:** {sec['coverage_pct']:.1f}%\n"
                f"- **Social Dialogue:** {sec['social_dialogue_coverage_pct']:.1f}%\n"
                f"- **Agreements:** {sec['agreements_count']}")

    def _md_diversity(self, d: Dict[str, Any]) -> str:
        sec = self._section_diversity_metrics(d)
        return (f"## {sec['title']}\n\n| Metric | Value |\n|--------|------:|\n"
                f"| Female % | {sec['female_pct']:.1f}% |\n| Female Management % | {sec['female_management_pct']:.1f}% |\n"
                f"| Female Board % | {sec['female_board_pct']:.1f}% |\n| Nationalities | {sec['nationality_count']} |")

    def _md_wages(self, d: Dict[str, Any]) -> str:
        sec = self._section_adequate_wages(d)
        compliant = "Yes" if sec["living_wage_compliant"] else "No"
        return (f"## {sec['title']}\n\n**Living Wage Compliant:** {compliant}  \n"
                f"**Lowest-to-Living Wage Ratio:** {sec['lowest_wage_ratio']:.2f}  \n"
                f"**Countries Assessed:** {sec['countries_assessed']}")

    def _md_social_protection(self, d: Dict[str, Any]) -> str:
        sec = self._section_social_protection(d)
        return (f"## {sec['title']}\n\n**Coverage:** {sec['covered_pct']:.1f}%  \n"
                f"**Parental Leave Eligible:** {sec['parental_leave_eligible_pct']:.1f}%")

    def _md_disability(self, d: Dict[str, Any]) -> str:
        sec = self._section_disability_inclusion(d)
        return (f"## {sec['title']}\n\n**Employees with Disabilities:** {sec['disability_pct']:.1f}%  \n"
                f"**Accommodation Requests Met:** {sec['accommodation_requests_met']}")

    def _md_training(self, d: Dict[str, Any]) -> str:
        sec = self._section_training_development(d)
        return (f"## {sec['title']}\n\n**Avg Training Hours:** {sec['avg_training_hours']:.1f}  \n"
                f"**Investment:** EUR {sec['training_investment_eur']:,.2f}  \n"
                f"**Performance Review Coverage:** {sec['performance_review_pct']:.1f}%")

    def _md_health_safety(self, d: Dict[str, Any]) -> str:
        sec = self._section_health_safety(d)
        return (f"## {sec['title']}\n\n| Metric | Value |\n|--------|------:|\n"
                f"| Fatalities | {sec['fatalities']} |\n| Recordable Incidents | {sec['recordable_incidents']} |\n"
                f"| LTIR | {sec['lost_time_injury_rate']:.2f} |\n| TRIR | {sec['total_recordable_rate']:.2f} |\n"
                f"| Lost Days | {sec['lost_days']:,} |\n| Workers Covered | {sec['workers_covered_pct']:.1f}% |")

    def _md_work_life(self, d: Dict[str, Any]) -> str:
        sec = self._section_work_life_balance(d)
        return (f"## {sec['title']}\n\n**Family Leave Uptake:** {sec['family_leave_uptake_pct']:.1f}%  \n"
                f"**Flexible Work:** {sec['flexible_work_pct']:.1f}%  \n"
                f"**Avg Hours/Week:** {sec['avg_working_hours_per_week']:.1f}")

    def _md_remuneration(self, d: Dict[str, Any]) -> str:
        sec = self._section_remuneration_fairness(d)
        return (f"## {sec['title']}\n\n**Gender Pay Gap:** {sec['gender_pay_gap_pct']:.1f}%  \n"
                f"**CEO-to-Median Ratio:** {sec['ceo_to_median_ratio']:.1f}x  \n"
                f"**Total Remuneration:** EUR {sec['total_remuneration_eur']:,.2f}")

    def _md_incidents(self, d: Dict[str, Any]) -> str:
        sec = self._section_incidents_complaints(d)
        return (f"## {sec['title']}\n\n| Type | Count |\n|------|------:|\n"
                f"| Discrimination | {sec['discrimination_incidents']} |\n"
                f"| Harassment | {sec['harassment_incidents']} |\n"
                f"| Grievances Filed | {sec['grievances_filed']} |\n"
                f"| Grievances Resolved | {sec['grievances_resolved']} |\n"
                f"| Human Rights | {sec['human_rights_complaints']} |\n"
                f"| Fines (EUR) | {sec['fines_eur']:,.2f} |")

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-017 ESRS Full Coverage Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return ("body{font-family:Arial,sans-serif;margin:2em;color:#333}"
                ".report{max-width:900px;margin:auto}"
                "h1{color:#1a5632;border-bottom:2px solid #1a5632;padding-bottom:.3em}"
                "h2{color:#2d7a4f;margin-top:1.5em}"
                "table{border-collapse:collapse;width:100%;margin:1em 0}"
                "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
                "th{background:#f0f7f3}.total{font-weight:bold;background:#e8f5e9}")

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (f"<h1>ESRS S1 Own Workforce Report</h1>\n"
                f"<p><strong>{data.get('entity_name', '')}</strong> | {data.get('reporting_year', '')}</p>")

    def _html_demographics(self, data: Dict[str, Any]) -> str:
        """Render demographics HTML."""
        sec = self._section_employee_demographics(data)
        return (f"<h2>{sec['title']}</h2>\n<table><tr><th>Metric</th><th>Value</th></tr>"
                f"<tr><td>Total</td><td>{sec['total_headcount']:,}</td></tr>"
                f"<tr><td>Permanent</td><td>{sec['permanent']:,}</td></tr>"
                f"<tr><td>Temporary</td><td>{sec['temporary']:,}</td></tr></table>")

    def _html_diversity(self, data: Dict[str, Any]) -> str:
        """Render diversity HTML."""
        sec = self._section_diversity_metrics(data)
        return (f"<h2>{sec['title']}</h2>\n<table><tr><th>Metric</th><th>%</th></tr>"
                f"<tr><td>Female</td><td>{sec['female_pct']:.1f}%</td></tr>"
                f"<tr><td>Female Mgmt</td><td>{sec['female_management_pct']:.1f}%</td></tr>"
                f"<tr><td>Female Board</td><td>{sec['female_board_pct']:.1f}%</td></tr></table>")

    def _html_health_safety(self, data: Dict[str, Any]) -> str:
        """Render health and safety HTML."""
        sec = self._section_health_safety(data)
        return (f"<h2>{sec['title']}</h2>\n<table><tr><th>Metric</th><th>Value</th></tr>"
                f"<tr><td>Fatalities</td><td>{sec['fatalities']}</td></tr>"
                f"<tr><td>LTIR</td><td>{sec['lost_time_injury_rate']:.2f}</td></tr>"
                f"<tr><td>TRIR</td><td>{sec['total_recordable_rate']:.2f}</td></tr></table>")

    def _html_incidents(self, data: Dict[str, Any]) -> str:
        """Render incidents HTML."""
        sec = self._section_incidents_complaints(data)
        return (f"<h2>{sec['title']}</h2>\n<table><tr><th>Type</th><th>Count</th></tr>"
                f"<tr><td>Discrimination</td><td>{sec['discrimination_incidents']}</td></tr>"
                f"<tr><td>Harassment</td><td>{sec['harassment_incidents']}</td></tr>"
                f"<tr><td>Grievances</td><td>{sec['grievances_filed']}</td></tr></table>")


# Alias for backward compatibility with templates/__init__.py
S1WorkforceReport = S1WorkforceReportTemplate
