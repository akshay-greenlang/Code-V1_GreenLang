# -*- coding: utf-8 -*-
"""
DueDiligenceReportTemplate - EU Battery Regulation Art 48 Supply Chain Due Diligence Report

Renders the supply chain due diligence findings report required by Article 48
of Regulation (EU) 2023/1542. Economic operators placing rechargeable
industrial and EV batteries on the EU market must establish supply chain due
diligence policies aligned with the OECD Due Diligence Guidance for Responsible
Supply Chains of Minerals from Conflict-Affected and High-Risk Areas. This
template covers supplier assessments, risk mapping, OECD compliance verification,
audit coverage, and mitigation measures.

Sections:
    1. Supplier Assessment - Tier-1/2/3 supplier evaluation and scoring
    2. Risk Map - Geographic and material risk heat map
    3. OECD Compliance - Alignment with OECD 5-step framework
    4. Audit Coverage - Third-party audit status and findings
    5. Mitigation Measures - Risk mitigation actions and effectiveness

Author: GreenLang Team
Version: 20.0.0
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
    "supplier_assessment",
    "risk_map",
    "oecd_compliance",
    "audit_coverage",
    "mitigation_measures",
]

# OECD 5-step framework for DD
_OECD_STEPS: List[Dict[str, str]] = [
    {"step": "1", "title": "Establish strong company management systems",
     "annex": "Annex II, Step 1"},
    {"step": "2", "title": "Identify and assess risk in the supply chain",
     "annex": "Annex II, Step 2"},
    {"step": "3", "title": "Design and implement a strategy to respond to identified risks",
     "annex": "Annex II, Step 3"},
    {"step": "4", "title": "Carry out independent third-party audit of supply chain DD",
     "annex": "Annex II, Step 4"},
    {"step": "5", "title": "Report on supply chain due diligence",
     "annex": "Annex II, Step 5"},
]

# Risk categories per Art 48(2)
_RISK_CATEGORIES: List[str] = [
    "child_labour",
    "forced_labour",
    "human_rights_abuses",
    "environmental_degradation",
    "conflict_financing",
    "corruption_bribery",
    "health_safety",
    "indigenous_rights",
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

class DueDiligenceReportTemplate:
    """
    Supply Chain Due Diligence report template per EU Battery Regulation Art 48.

    Generates a comprehensive due diligence findings report covering supply chain
    assessment, risk mapping, OECD framework alignment verification, audit coverage,
    and mitigation measures. Applicable to economic operators placing rechargeable
    industrial and EV batteries with a capacity above 2 kWh on the EU market.

    Regulatory References:
        - Regulation (EU) 2023/1542, Article 48
        - OECD Due Diligence Guidance for Responsible Supply Chains of Minerals
          from Conflict-Affected and High-Risk Areas (3rd Edition)

from greenlang.schemas import utcnow
        - Regulation (EU) 2017/821 (Conflict Minerals Regulation)
        - UN Guiding Principles on Business and Human Rights

    Example:
        >>> tpl = DueDiligenceReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DueDiligenceReportTemplate."""
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
        if "suppliers" not in data:
            errors.append("suppliers list is required for assessment")
        if "oecd_steps" not in data:
            warnings.append("oecd_steps data missing; OECD section will use defaults")
        if not data.get("audits"):
            warnings.append("audits data missing; audit coverage section limited")
        if not data.get("risk_areas"):
            warnings.append("risk_areas data missing; risk map will be empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render due diligence report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_supplier_assessment(data),
            self._md_risk_map(data),
            self._md_oecd_compliance(data),
            self._md_audit_coverage(data),
            self._md_mitigation_measures(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render due diligence report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_supplier_assessment(data),
            self._html_risk_map(data),
            self._html_oecd_compliance(data),
            self._html_audit_coverage(data),
            self._html_mitigation_measures(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Supply Chain Due Diligence Report - Art 48</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render due diligence report as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "due_diligence_report",
            "regulation_reference": "EU Battery Regulation 2023/1542, Art 48",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "supplier_assessment": self._section_supplier_assessment(data),
            "risk_map": self._section_risk_map(data),
            "oecd_compliance": self._section_oecd_compliance(data),
            "audit_coverage": self._section_audit_coverage(data),
            "mitigation_measures": self._section_mitigation_measures(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_supplier_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build supplier assessment section."""
        suppliers = data.get("suppliers", [])
        tier_counts = {"tier_1": 0, "tier_2": 0, "tier_3": 0, "unknown": 0}
        assessed_count = 0
        high_risk_count = 0
        supplier_details: List[Dict[str, Any]] = []

        for sup in suppliers:
            tier = sup.get("tier", "unknown")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            is_assessed = sup.get("assessed", False)
            if is_assessed:
                assessed_count += 1
            risk_level = sup.get("risk_level", "unknown")
            if risk_level == "high":
                high_risk_count += 1
            supplier_details.append({
                "name": sup.get("name", ""),
                "tier": tier,
                "country": sup.get("country", ""),
                "materials_supplied": sup.get("materials", []),
                "assessed": is_assessed,
                "risk_level": risk_level,
                "risk_score": sup.get("risk_score", 0),
                "last_assessment_date": sup.get("last_assessment_date", ""),
                "corrective_actions_open": sup.get("corrective_actions_open", 0),
            })

        total = len(suppliers)
        return {
            "title": "Supplier Assessment",
            "total_suppliers": total,
            "tier_breakdown": tier_counts,
            "assessed_count": assessed_count,
            "assessment_coverage_pct": (
                round(assessed_count / total * 100, 1) if total > 0 else 0.0
            ),
            "high_risk_suppliers": high_risk_count,
            "suppliers": supplier_details,
            "assessment_methodology": data.get(
                "assessment_methodology", "Risk-based supplier questionnaire + on-site audits"
            ),
        }

    def _section_risk_map(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build risk mapping section."""
        risk_areas = data.get("risk_areas", [])
        geographic_risks = data.get("geographic_risks", [])

        risk_by_category: Dict[str, List[Dict[str, Any]]] = {
            cat: [] for cat in _RISK_CATEGORIES
        }
        for risk in risk_areas:
            category = risk.get("category", "")
            if category in risk_by_category:
                risk_by_category[category].append({
                    "description": risk.get("description", ""),
                    "severity": risk.get("severity", "medium"),
                    "likelihood": risk.get("likelihood", "medium"),
                    "affected_materials": risk.get("affected_materials", []),
                    "affected_countries": risk.get("affected_countries", []),
                })

        total_risks = sum(len(v) for v in risk_by_category.values())
        high_severity = sum(
            1 for risks in risk_by_category.values()
            for r in risks if r["severity"] == "high"
        )

        return {
            "title": "Risk Map",
            "total_risks_identified": total_risks,
            "high_severity_count": high_severity,
            "risk_categories": {
                cat: {
                    "count": len(risks),
                    "risks": risks,
                }
                for cat, risks in risk_by_category.items()
            },
            "geographic_risk_areas": [
                {
                    "country": g.get("country", ""),
                    "region": g.get("region", ""),
                    "risk_level": g.get("risk_level", "medium"),
                    "risk_factors": g.get("risk_factors", []),
                    "cahra_listed": g.get("cahra_listed", False),
                }
                for g in geographic_risks
            ],
            "cahra_exposure": any(
                g.get("cahra_listed", False) for g in geographic_risks
            ),
        }

    def _section_oecd_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build OECD compliance verification section."""
        oecd_data = data.get("oecd_steps", {})
        step_results: List[Dict[str, Any]] = []
        compliant_count = 0

        for step_def in _OECD_STEPS:
            step_key = f"step_{step_def['step']}"
            step_info = oecd_data.get(step_key, {})
            status = step_info.get("status", "not_started")
            if status == "compliant":
                compliant_count += 1
            step_results.append({
                "step_number": step_def["step"],
                "title": step_def["title"],
                "annex_reference": step_def["annex"],
                "status": status,
                "evidence": step_info.get("evidence", []),
                "findings": step_info.get("findings", ""),
                "gaps": step_info.get("gaps", []),
            })

        overall_compliant = compliant_count == len(_OECD_STEPS)
        return {
            "title": "OECD Due Diligence Compliance",
            "framework": "OECD Due Diligence Guidance (3rd Edition)",
            "total_steps": len(_OECD_STEPS),
            "compliant_steps": compliant_count,
            "compliance_pct": round(compliant_count / len(_OECD_STEPS) * 100, 1),
            "overall_compliant": overall_compliant,
            "steps": step_results,
            "dd_policy_published": data.get("dd_policy_published", False),
            "dd_policy_url": data.get("dd_policy_url", ""),
            "grievance_mechanism": data.get("grievance_mechanism", False),
            "management_system_certified": data.get("management_system_certified", False),
        }

    def _section_audit_coverage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build audit coverage section."""
        audits = data.get("audits", [])
        total_suppliers = len(data.get("suppliers", []))
        audited_suppliers = set()

        audit_details: List[Dict[str, Any]] = []
        for audit in audits:
            audited_suppliers.update(audit.get("suppliers_covered", []))
            findings = audit.get("findings", [])
            critical = sum(1 for f in findings if f.get("severity") == "critical")
            major = sum(1 for f in findings if f.get("severity") == "major")
            minor = sum(1 for f in findings if f.get("severity") == "minor")
            audit_details.append({
                "audit_id": audit.get("audit_id", ""),
                "auditor": audit.get("auditor", ""),
                "audit_type": audit.get("type", "third_party"),
                "audit_date": audit.get("date", ""),
                "scope": audit.get("scope", ""),
                "suppliers_covered": audit.get("suppliers_covered", []),
                "suppliers_covered_count": len(audit.get("suppliers_covered", [])),
                "finding_counts": {
                    "critical": critical,
                    "major": major,
                    "minor": minor,
                    "total": len(findings),
                },
                "corrective_actions_issued": audit.get("corrective_actions", 0),
                "corrective_actions_closed": audit.get("corrective_actions_closed", 0),
                "audit_conclusion": audit.get("conclusion", ""),
            })

        audited_count = len(audited_suppliers)
        return {
            "title": "Audit Coverage",
            "total_audits_conducted": len(audits),
            "total_suppliers": total_suppliers,
            "audited_supplier_count": audited_count,
            "audit_coverage_pct": (
                round(audited_count / total_suppliers * 100, 1)
                if total_suppliers > 0 else 0.0
            ),
            "audits": audit_details,
            "total_findings": sum(a["finding_counts"]["total"] for a in audit_details),
            "critical_findings_open": sum(
                a["finding_counts"]["critical"] for a in audit_details
            ),
            "third_party_audits": sum(
                1 for a in audit_details if a["audit_type"] == "third_party"
            ),
        }

    def _section_mitigation_measures(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build mitigation measures section."""
        measures = data.get("mitigation_measures", [])
        measure_details: List[Dict[str, Any]] = []

        for idx, measure in enumerate(measures, 1):
            measure_details.append({
                "id": idx,
                "risk_category": measure.get("risk_category", ""),
                "description": measure.get("description", ""),
                "status": measure.get("status", "planned"),
                "responsible": measure.get("responsible", ""),
                "start_date": measure.get("start_date", ""),
                "target_date": measure.get("target_date", ""),
                "progress_pct": measure.get("progress_pct", 0),
                "effectiveness_rating": measure.get("effectiveness", ""),
                "cost_eur": measure.get("cost_eur", 0.0),
            })

        completed = sum(1 for m in measure_details if m["status"] == "completed")
        in_progress = sum(1 for m in measure_details if m["status"] == "in_progress")
        planned = sum(1 for m in measure_details if m["status"] == "planned")

        return {
            "title": "Mitigation Measures",
            "total_measures": len(measure_details),
            "completed": completed,
            "in_progress": in_progress,
            "planned": planned,
            "completion_pct": (
                round(completed / len(measure_details) * 100, 1)
                if measure_details else 0.0
            ),
            "measures": measure_details,
            "total_investment_eur": sum(m["cost_eur"] for m in measure_details),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Supply Chain Due Diligence Report\n"
            f"## EU Battery Regulation (EU) 2023/1542 - Article 48\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Period:** {data.get('reporting_period', '')}  \n"
            f"**Generated:** {ts}"
        )

    def _md_supplier_assessment(self, data: Dict[str, Any]) -> str:
        """Render supplier assessment as markdown."""
        sec = self._section_supplier_assessment(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Suppliers:** {sec['total_suppliers']}  \n"
            f"**Assessed:** {sec['assessed_count']} ({sec['assessment_coverage_pct']:.1f}%)  \n"
            f"**High Risk:** {sec['high_risk_suppliers']}\n",
            "### Tier Breakdown\n",
            f"- Tier 1 (Direct): {sec['tier_breakdown'].get('tier_1', 0)}",
            f"- Tier 2 (Sub-suppliers): {sec['tier_breakdown'].get('tier_2', 0)}",
            f"- Tier 3 (Raw material): {sec['tier_breakdown'].get('tier_3', 0)}\n",
            "### Supplier Details\n",
            "| Supplier | Tier | Country | Risk | Assessed | Open CAs |",
            "|----------|:----:|---------|:----:|:--------:|--------:|",
        ]
        for sup in sec["suppliers"]:
            assessed = "Yes" if sup["assessed"] else "No"
            lines.append(
                f"| {sup['name']} | {sup['tier']} | {sup['country']} | "
                f"{sup['risk_level']} | {assessed} | "
                f"{sup['corrective_actions_open']} |"
            )
        return "\n".join(lines)

    def _md_risk_map(self, data: Dict[str, Any]) -> str:
        """Render risk map as markdown."""
        sec = self._section_risk_map(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Risks:** {sec['total_risks_identified']}  \n"
            f"**High Severity:** {sec['high_severity_count']}  \n"
            f"**CAHRA Exposure:** {'Yes' if sec['cahra_exposure'] else 'No'}\n",
            "### Risk by Category\n",
            "| Category | Count | Highest Severity |",
            "|----------|------:|:----------------:|",
        ]
        for cat, info in sec["risk_categories"].items():
            if info["count"] > 0:
                max_sev = max(
                    (r["severity"] for r in info["risks"]),
                    key=lambda s: {"high": 3, "medium": 2, "low": 1}.get(s, 0),
                    default="none",
                )
            else:
                max_sev = "none"
            cat_label = cat.replace("_", " ").title()
            lines.append(f"| {cat_label} | {info['count']} | {max_sev} |")

        if sec["geographic_risk_areas"]:
            lines.append("\n### Geographic Risk Areas\n")
            lines.append("| Country | Region | Risk Level | CAHRA |")
            lines.append("|---------|--------|:----------:|:-----:|")
            for g in sec["geographic_risk_areas"]:
                cahra = "Yes" if g["cahra_listed"] else "No"
                lines.append(
                    f"| {g['country']} | {g['region']} | {g['risk_level']} | {cahra} |"
                )
        return "\n".join(lines)

    def _md_oecd_compliance(self, data: Dict[str, Any]) -> str:
        """Render OECD compliance as markdown."""
        sec = self._section_oecd_compliance(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Framework:** {sec['framework']}  \n"
            f"**Compliance:** {sec['compliant_steps']}/{sec['total_steps']} steps "
            f"({sec['compliance_pct']:.1f}%)  \n"
            f"**Overall:** {'COMPLIANT' if sec['overall_compliant'] else 'GAPS REMAIN'}\n",
            "| Step | Title | Status | Gaps |",
            "|:----:|-------|:------:|-----:|",
        ]
        for step in sec["steps"]:
            gap_count = len(step["gaps"])
            lines.append(
                f"| {step['step_number']} | {step['title']} | "
                f"{step['status'].upper()} | {gap_count} |"
            )
        lines.append(
            f"\n- DD Policy Published: {'Yes' if sec['dd_policy_published'] else 'No'}"
        )
        lines.append(
            f"- Grievance Mechanism: {'Yes' if sec['grievance_mechanism'] else 'No'}"
        )
        return "\n".join(lines)

    def _md_audit_coverage(self, data: Dict[str, Any]) -> str:
        """Render audit coverage as markdown."""
        sec = self._section_audit_coverage(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Audits Conducted:** {sec['total_audits_conducted']}  \n"
            f"**Supplier Coverage:** {sec['audited_supplier_count']}/"
            f"{sec['total_suppliers']} ({sec['audit_coverage_pct']:.1f}%)  \n"
            f"**Third-Party Audits:** {sec['third_party_audits']}  \n"
            f"**Total Findings:** {sec['total_findings']}  \n"
            f"**Critical Open:** {sec['critical_findings_open']}\n",
        ]
        if sec["audits"]:
            lines.append("### Audit Details\n")
            lines.append("| Auditor | Type | Date | Suppliers | Findings | CAs Open |")
            lines.append("|---------|------|------|----------:|--------:|---------:|")
            for audit in sec["audits"]:
                open_cas = audit["corrective_actions_issued"] - audit[
                    "corrective_actions_closed"
                ]
                lines.append(
                    f"| {audit['auditor']} | {audit['audit_type']} | "
                    f"{audit['audit_date']} | {audit['suppliers_covered_count']} | "
                    f"{audit['finding_counts']['total']} | {open_cas} |"
                )
        return "\n".join(lines)

    def _md_mitigation_measures(self, data: Dict[str, Any]) -> str:
        """Render mitigation measures as markdown."""
        sec = self._section_mitigation_measures(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Measures:** {sec['total_measures']}  \n"
            f"**Completed:** {sec['completed']} | **In Progress:** {sec['in_progress']} "
            f"| **Planned:** {sec['planned']}  \n"
            f"**Completion:** {sec['completion_pct']:.1f}%\n",
            "| # | Risk Category | Description | Status | Progress | Target |",
            "|--:|---------------|-------------|:------:|--------:|--------|",
        ]
        for m in sec["measures"]:
            cat_label = m["risk_category"].replace("_", " ").title()
            lines.append(
                f"| {m['id']} | {cat_label} | {m['description']} | "
                f"{m['status']} | {m['progress_pct']}% | {m['target_date']} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n*Report generated by PACK-020 Battery Passport Prep Pack on {ts}*\n"
            f"*Regulation: EU Battery Regulation (EU) 2023/1542, Article 48*"
        )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1100px;margin:auto}"
            "h1{color:#0d47a1;border-bottom:2px solid #0d47a1;padding-bottom:.3em}"
            "h2{color:#1565c0;margin-top:1.5em}"
            "h3{color:#1976d2}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e3f2fd}"
            ".pass{color:#2e7d32;font-weight:bold}"
            ".fail{color:#c62828;font-weight:bold}"
            ".risk-high{background:#ffcdd2;color:#b71c1c}"
            ".risk-medium{background:#fff9c4;color:#f57f17}"
            ".risk-low{background:#c8e6c9;color:#1b5e20}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Supply Chain Due Diligence Report</h1>\n"
            f"<p>EU Battery Regulation (EU) 2023/1542 - Article 48</p>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"Period: {data.get('reporting_period', '')}</p>"
        )

    def _html_supplier_assessment(self, data: Dict[str, Any]) -> str:
        """Render supplier assessment HTML."""
        sec = self._section_supplier_assessment(data)
        rows = "".join(
            f"<tr><td>{s['name']}</td><td>{s['tier']}</td><td>{s['country']}</td>"
            f"<td class='risk-{s['risk_level']}'>{s['risk_level']}</td>"
            f"<td>{'Yes' if s['assessed'] else 'No'}</td></tr>"
            for s in sec["suppliers"][:20]  # Top 20 in HTML view
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Assessed: {sec['assessed_count']}/{sec['total_suppliers']} "
            f"({sec['assessment_coverage_pct']:.1f}%) | "
            f"High risk: {sec['high_risk_suppliers']}</p>\n"
            f"<table><tr><th>Supplier</th><th>Tier</th><th>Country</th>"
            f"<th>Risk</th><th>Assessed</th></tr>{rows}</table>"
        )

    def _html_risk_map(self, data: Dict[str, Any]) -> str:
        """Render risk map HTML."""
        sec = self._section_risk_map(data)
        rows = ""
        for cat, info in sec["risk_categories"].items():
            if info["count"] > 0:
                cat_label = cat.replace("_", " ").title()
                rows += (
                    f"<tr><td>{cat_label}</td><td>{info['count']}</td></tr>"
                )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total: {sec['total_risks_identified']} | "
            f"High severity: {sec['high_severity_count']}</p>\n"
            f"<table><tr><th>Category</th><th>Count</th></tr>{rows}</table>"
        )

    def _html_oecd_compliance(self, data: Dict[str, Any]) -> str:
        """Render OECD compliance HTML."""
        sec = self._section_oecd_compliance(data)
        overall_cls = "pass" if sec["overall_compliant"] else "fail"
        rows = "".join(
            f"<tr><td>{s['step_number']}</td><td>{s['title']}</td>"
            f"<td class='{'pass' if s['status'] == 'compliant' else 'fail'}'>"
            f"{s['status'].upper()}</td></tr>"
            for s in sec["steps"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='{overall_cls}'>Overall: "
            f"{'COMPLIANT' if sec['overall_compliant'] else 'GAPS REMAIN'} "
            f"({sec['compliance_pct']:.1f}%)</p>\n"
            f"<table><tr><th>Step</th><th>Title</th><th>Status</th></tr>"
            f"{rows}</table>"
        )

    def _html_audit_coverage(self, data: Dict[str, Any]) -> str:
        """Render audit coverage HTML."""
        sec = self._section_audit_coverage(data)
        rows = "".join(
            f"<tr><td>{a['auditor']}</td><td>{a['audit_type']}</td>"
            f"<td>{a['audit_date']}</td>"
            f"<td>{a['finding_counts']['total']}</td></tr>"
            for a in sec["audits"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Coverage: {sec['audit_coverage_pct']:.1f}% | "
            f"Findings: {sec['total_findings']}</p>\n"
            f"<table><tr><th>Auditor</th><th>Type</th><th>Date</th>"
            f"<th>Findings</th></tr>{rows}</table>"
        )

    def _html_mitigation_measures(self, data: Dict[str, Any]) -> str:
        """Render mitigation measures HTML."""
        sec = self._section_mitigation_measures(data)
        rows = "".join(
            f"<tr><td>{m['id']}</td>"
            f"<td>{m['risk_category'].replace('_', ' ').title()}</td>"
            f"<td>{m['description']}</td><td>{m['status']}</td>"
            f"<td>{m['progress_pct']}%</td></tr>"
            for m in sec["measures"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Completion: {sec['completion_pct']:.1f}%</p>\n"
            f"<table><tr><th>#</th><th>Category</th><th>Description</th>"
            f"<th>Status</th><th>Progress</th></tr>{rows}</table>"
        )
