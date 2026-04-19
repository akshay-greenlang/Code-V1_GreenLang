# -*- coding: utf-8 -*-
"""
ComplianceReportTemplate - Standards Compliance Report for PACK-040.

Generates comprehensive standards compliance reports covering IPMVP
protocol checklist, ISO 50015 conformity assessment, FEMP M&V
requirements verification, ASHRAE Guideline 14 statistical criteria,
and EU EED Article 7 compliance mapping.

Sections:
    1. Compliance Summary
    2. IPMVP Checklist
    3. ISO 50015 Conformity
    4. FEMP Requirements
    5. ASHRAE 14 Criteria
    6. EU EED Article 7
    7. Data Quality Assessment
    8. Documentation Review
    9. Non-Conformities
    10. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IPMVP Core Concepts 2022
    - ISO 50015:2014 (M&V of energy performance)
    - FEMP M&V Guidelines 4.0
    - ASHRAE Guideline 14-2014
    - EU EED 2023/1791 Article 7

Author: GreenLang Team
Version: 40.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class ComplianceReportTemplate:
    """
    Standards compliance report template.

    Renders comprehensive standards compliance reports showing IPMVP
    protocol checklist with item-level conformity, ISO 50015 clause
    assessment, FEMP M&V requirements verification, ASHRAE Guideline 14
    statistical criteria pass/fail, EU EED Article 7 mapping, data
    quality assessment, and non-conformity register across markdown,
    HTML, and JSON formats. All outputs include SHA-256 provenance
    hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ComplianceReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render standards compliance report as Markdown.

        Args:
            data: Compliance engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_compliance_summary(data),
            self._md_ipmvp_checklist(data),
            self._md_iso50015(data),
            self._md_femp_requirements(data),
            self._md_ashrae14(data),
            self._md_eu_eed(data),
            self._md_data_quality(data),
            self._md_documentation_review(data),
            self._md_non_conformities(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render standards compliance report as self-contained HTML.

        Args:
            data: Compliance engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_compliance_summary(data),
            self._html_ipmvp_checklist(data),
            self._html_iso50015(data),
            self._html_femp_requirements(data),
            self._html_ashrae14(data),
            self._html_eu_eed(data),
            self._html_data_quality(data),
            self._html_documentation_review(data),
            self._html_non_conformities(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Standards Compliance Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render standards compliance report as structured JSON.

        Args:
            data: Compliance engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "compliance_report",
            "version": "40.0.0",
            "generated_at": self.generated_at.isoformat(),
            "compliance_summary": self._json_compliance_summary(data),
            "ipmvp_checklist": data.get("ipmvp_checklist", []),
            "iso50015_conformity": data.get("iso50015_conformity", []),
            "femp_requirements": data.get("femp_requirements", []),
            "ashrae14_criteria": data.get("ashrae14_criteria", {}),
            "eu_eed_article7": data.get("eu_eed_article7", []),
            "data_quality": data.get("data_quality", {}),
            "documentation_review": data.get("documentation_review", []),
            "non_conformities": data.get("non_conformities", []),
            "recommendations": data.get("recommendations", []),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with project metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Standards Compliance Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Project:** {data.get('project_name', '-')}  \n"
            f"**Audit Date:** {data.get('audit_date', '-')}  \n"
            f"**Auditor:** {data.get('auditor', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-040 ComplianceReportTemplate v40.0.0\n\n---"
        )

    def _md_compliance_summary(self, data: Dict[str, Any]) -> str:
        """Render compliance summary section."""
        s = data.get("compliance_summary", {})
        return (
            "## 1. Compliance Summary\n\n"
            "| Standard | Items | Pass | Fail | Score (%) | Status |\n"
            "|----------|------:|-----:|-----:|--------:|:------:|\n"
            f"| IPMVP | {s.get('ipmvp_total', 0)} | {s.get('ipmvp_pass', 0)} | {s.get('ipmvp_fail', 0)} | {self._fmt(s.get('ipmvp_score_pct', 0))}% | {s.get('ipmvp_status', '-')} |\n"
            f"| ISO 50015 | {s.get('iso_total', 0)} | {s.get('iso_pass', 0)} | {s.get('iso_fail', 0)} | {self._fmt(s.get('iso_score_pct', 0))}% | {s.get('iso_status', '-')} |\n"
            f"| FEMP | {s.get('femp_total', 0)} | {s.get('femp_pass', 0)} | {s.get('femp_fail', 0)} | {self._fmt(s.get('femp_score_pct', 0))}% | {s.get('femp_status', '-')} |\n"
            f"| ASHRAE 14 | {s.get('ashrae_total', 0)} | {s.get('ashrae_pass', 0)} | {s.get('ashrae_fail', 0)} | {self._fmt(s.get('ashrae_score_pct', 0))}% | {s.get('ashrae_status', '-')} |\n"
            f"| EU EED Art.7 | {s.get('eed_total', 0)} | {s.get('eed_pass', 0)} | {s.get('eed_fail', 0)} | {self._fmt(s.get('eed_score_pct', 0))}% | {s.get('eed_status', '-')} |\n"
            f"| **Overall** | {s.get('overall_total', 0)} | {s.get('overall_pass', 0)} | {s.get('overall_fail', 0)} | **{self._fmt(s.get('overall_score_pct', 0))}%** | **{s.get('overall_status', '-')}** |"
        )

    def _md_ipmvp_checklist(self, data: Dict[str, Any]) -> str:
        """Render IPMVP checklist section."""
        items = data.get("ipmvp_checklist", [])
        if not items:
            return "## 2. IPMVP Checklist\n\n_No IPMVP checklist data available._"
        lines = [
            "## 2. IPMVP Checklist\n",
            "| # | Requirement | Status | Evidence | Finding |",
            "|---|------------|:------:|----------|---------|",
        ]
        for i, item in enumerate(items, 1):
            lines.append(
                f"| {i} | {item.get('requirement', '-')} "
                f"| {item.get('status', '-')} "
                f"| {item.get('evidence', '-')} "
                f"| {item.get('finding', '-')} |"
            )
        return "\n".join(lines)

    def _md_iso50015(self, data: Dict[str, Any]) -> str:
        """Render ISO 50015 conformity section."""
        clauses = data.get("iso50015_conformity", [])
        if not clauses:
            return "## 3. ISO 50015 Conformity\n\n_No ISO 50015 conformity data available._"
        lines = [
            "## 3. ISO 50015 Conformity\n",
            "| Clause | Description | Status | Finding |",
            "|--------|------------|:------:|---------|",
        ]
        for clause in clauses:
            lines.append(
                f"| {clause.get('clause', '-')} "
                f"| {clause.get('description', '-')} "
                f"| {clause.get('status', '-')} "
                f"| {clause.get('finding', '-')} |"
            )
        return "\n".join(lines)

    def _md_femp_requirements(self, data: Dict[str, Any]) -> str:
        """Render FEMP requirements section."""
        reqs = data.get("femp_requirements", [])
        if not reqs:
            return "## 4. FEMP Requirements\n\n_No FEMP requirements data available._"
        lines = [
            "## 4. FEMP M&V Requirements\n",
            "| Requirement | Section | Status | Detail |",
            "|------------|---------|:------:|--------|",
        ]
        for req in reqs:
            lines.append(
                f"| {req.get('requirement', '-')} "
                f"| {req.get('section', '-')} "
                f"| {req.get('status', '-')} "
                f"| {req.get('detail', '-')} |"
            )
        return "\n".join(lines)

    def _md_ashrae14(self, data: Dict[str, Any]) -> str:
        """Render ASHRAE 14 criteria section."""
        ashrae = data.get("ashrae14_criteria", {})
        if not ashrae:
            return "## 5. ASHRAE Guideline 14 Criteria\n\n_No ASHRAE 14 criteria data available._"
        return (
            "## 5. ASHRAE Guideline 14 Criteria\n\n"
            "| Criterion | Value | Threshold | Result |\n|-----------|------:|----------:|:------:|\n"
            f"| CV(RMSE) Monthly | {self._fmt(ashrae.get('cvrmse_monthly_pct', 0), 1)}% | <= 15% | {ashrae.get('cvrmse_monthly_result', '-')} |\n"
            f"| CV(RMSE) Hourly | {self._fmt(ashrae.get('cvrmse_hourly_pct', 0), 1)}% | <= 30% | {ashrae.get('cvrmse_hourly_result', '-')} |\n"
            f"| NMBE Monthly | {self._fmt(ashrae.get('nmbe_monthly_pct', 0), 1)}% | <= +/- 5% | {ashrae.get('nmbe_monthly_result', '-')} |\n"
            f"| NMBE Hourly | {self._fmt(ashrae.get('nmbe_hourly_pct', 0), 1)}% | <= +/- 10% | {ashrae.get('nmbe_hourly_result', '-')} |\n"
            f"| R-squared | {self._fmt(ashrae.get('r_squared', 0), 4)} | >= 0.70 | {ashrae.get('r_squared_result', '-')} |\n"
            f"| Net Determination Bias | {self._fmt(ashrae.get('net_det_bias_pct', 0), 1)}% | <= 0.005% | {ashrae.get('net_det_bias_result', '-')} |\n"
            f"| Savings Uncertainty | {self._fmt(ashrae.get('savings_uncertainty_pct', 0))}% | <= 50% at 90% | {ashrae.get('savings_uncertainty_result', '-')} |\n"
            f"| Overall | - | - | {ashrae.get('overall_result', '-')} |"
        )

    def _md_eu_eed(self, data: Dict[str, Any]) -> str:
        """Render EU EED Article 7 section."""
        eed = data.get("eu_eed_article7", [])
        if not eed:
            return "## 6. EU EED Article 7\n\n_No EU EED Article 7 data available._"
        lines = [
            "## 6. EU EED Article 7 Compliance\n",
            "| Requirement | Status | Evidence |",
            "|------------|:------:|----------|",
        ]
        for item in eed:
            lines.append(
                f"| {item.get('requirement', '-')} "
                f"| {item.get('status', '-')} "
                f"| {item.get('evidence', '-')} |"
            )
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render data quality assessment section."""
        dq = data.get("data_quality", {})
        if not dq:
            return "## 7. Data Quality Assessment\n\n_No data quality data available._"
        return (
            "## 7. Data Quality Assessment\n\n"
            "| Metric | Value | Target | Status |\n|--------|------:|-------:|:------:|\n"
            f"| Completeness | {self._fmt(dq.get('completeness_pct', 0))}% | >= {self._fmt(dq.get('completeness_target_pct', 90))}% | {dq.get('completeness_status', '-')} |\n"
            f"| Accuracy | {self._fmt(dq.get('accuracy_pct', 0))}% | >= {self._fmt(dq.get('accuracy_target_pct', 95))}% | {dq.get('accuracy_status', '-')} |\n"
            f"| Timeliness | {dq.get('timeliness', '-')} | {dq.get('timeliness_target', '-')} | {dq.get('timeliness_status', '-')} |\n"
            f"| Consistency | {self._fmt(dq.get('consistency_pct', 0))}% | >= {self._fmt(dq.get('consistency_target_pct', 95))}% | {dq.get('consistency_status', '-')} |\n"
            f"| Gaps Detected | {dq.get('gaps_detected', 0)} | 0 | {dq.get('gaps_status', '-')} |\n"
            f"| Overall | - | - | {dq.get('overall_status', '-')} |"
        )

    def _md_documentation_review(self, data: Dict[str, Any]) -> str:
        """Render documentation review section."""
        docs = data.get("documentation_review", [])
        if not docs:
            return "## 8. Documentation Review\n\n_No documentation review data available._"
        lines = [
            "## 8. Documentation Review\n",
            "| Document | Required | Present | Current | Status |",
            "|----------|:--------:|:-------:|:-------:|:------:|",
        ]
        for doc in docs:
            lines.append(
                f"| {doc.get('document', '-')} "
                f"| {doc.get('required', '-')} "
                f"| {doc.get('present', '-')} "
                f"| {doc.get('current', '-')} "
                f"| {doc.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_non_conformities(self, data: Dict[str, Any]) -> str:
        """Render non-conformities section."""
        ncs = data.get("non_conformities", [])
        if not ncs:
            return "## 9. Non-Conformities\n\n_No non-conformities identified._"
        lines = [
            "## 9. Non-Conformities\n",
            "| # | Standard | Finding | Severity | Corrective Action | Due Date |",
            "|---|----------|---------|:--------:|-------------------|----------|",
        ]
        for i, nc in enumerate(ncs, 1):
            lines.append(
                f"| {i} | {nc.get('standard', '-')} "
                f"| {nc.get('finding', '-')} "
                f"| {nc.get('severity', '-')} "
                f"| {nc.get('corrective_action', '-')} "
                f"| {nc.get('due_date', '-')} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Address all non-conformities before next compliance audit",
                "Maintain ASHRAE 14 statistical criteria through model monitoring",
                "Update M&V documentation when significant changes occur",
                "Schedule annual compliance review with independent verifier",
            ]
        lines = ["## 10. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-040 M&V Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Standards Compliance Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Project: {data.get("project_name", "-")} | '
            f'Audit: {data.get("audit_date", "-")}</p>'
        )

    def _html_compliance_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance summary cards."""
        s = data.get("compliance_summary", {})
        overall_cls = "severity-low" if s.get("overall_status") == "PASS" else "severity-high"
        return (
            '<h2>1. Compliance Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">IPMVP</span>'
            f'<span class="value">{self._fmt(s.get("ipmvp_score_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">ISO 50015</span>'
            f'<span class="value">{self._fmt(s.get("iso_score_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">FEMP</span>'
            f'<span class="value">{self._fmt(s.get("femp_score_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">ASHRAE 14</span>'
            f'<span class="value">{self._fmt(s.get("ashrae_score_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Overall</span>'
            f'<span class="value {overall_cls}">{s.get("overall_status", "-")}</span></div>\n'
            '</div>'
        )

    def _html_ipmvp_checklist(self, data: Dict[str, Any]) -> str:
        """Render HTML IPMVP checklist table."""
        items = data.get("ipmvp_checklist", [])
        rows = ""
        for i, item in enumerate(items, 1):
            cls = "severity-low" if item.get("status") == "PASS" else "severity-high"
            rows += (
                f'<tr><td>{i}</td><td>{item.get("requirement", "-")}</td>'
                f'<td class="{cls}">{item.get("status", "-")}</td>'
                f'<td>{item.get("evidence", "-")}</td>'
                f'<td>{item.get("finding", "-")}</td></tr>\n'
            )
        return (
            '<h2>2. IPMVP Checklist</h2>\n'
            '<table>\n<tr><th>#</th><th>Requirement</th><th>Status</th>'
            f'<th>Evidence</th><th>Finding</th></tr>\n{rows}</table>'
        )

    def _html_iso50015(self, data: Dict[str, Any]) -> str:
        """Render HTML ISO 50015 conformity table."""
        clauses = data.get("iso50015_conformity", [])
        rows = ""
        for clause in clauses:
            cls = "severity-low" if clause.get("status") == "Conforms" else "severity-high"
            rows += (
                f'<tr><td>{clause.get("clause", "-")}</td>'
                f'<td>{clause.get("description", "-")}</td>'
                f'<td class="{cls}">{clause.get("status", "-")}</td>'
                f'<td>{clause.get("finding", "-")}</td></tr>\n'
            )
        return (
            '<h2>3. ISO 50015 Conformity</h2>\n'
            '<table>\n<tr><th>Clause</th><th>Description</th>'
            f'<th>Status</th><th>Finding</th></tr>\n{rows}</table>'
        )

    def _html_femp_requirements(self, data: Dict[str, Any]) -> str:
        """Render HTML FEMP requirements table."""
        reqs = data.get("femp_requirements", [])
        rows = ""
        for req in reqs:
            cls = "severity-low" if req.get("status") == "PASS" else "severity-high"
            rows += (
                f'<tr><td>{req.get("requirement", "-")}</td>'
                f'<td>{req.get("section", "-")}</td>'
                f'<td class="{cls}">{req.get("status", "-")}</td>'
                f'<td>{req.get("detail", "-")}</td></tr>\n'
            )
        return (
            '<h2>4. FEMP Requirements</h2>\n'
            '<table>\n<tr><th>Requirement</th><th>Section</th>'
            f'<th>Status</th><th>Detail</th></tr>\n{rows}</table>'
        )

    def _html_ashrae14(self, data: Dict[str, Any]) -> str:
        """Render HTML ASHRAE 14 criteria cards."""
        a = data.get("ashrae14_criteria", {})
        cv_cls = "severity-low" if a.get("cvrmse_monthly_result") == "PASS" else "severity-high"
        nmbe_cls = "severity-low" if a.get("nmbe_monthly_result") == "PASS" else "severity-high"
        rsq_cls = "severity-low" if a.get("r_squared_result") == "PASS" else "severity-high"
        return (
            '<h2>5. ASHRAE 14 Criteria</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">CV(RMSE)</span>'
            f'<span class="value {cv_cls}">{self._fmt(a.get("cvrmse_monthly_pct", 0), 1)}%</span></div>\n'
            f'  <div class="card"><span class="label">NMBE</span>'
            f'<span class="value {nmbe_cls}">{self._fmt(a.get("nmbe_monthly_pct", 0), 1)}%</span></div>\n'
            f'  <div class="card"><span class="label">R-squared</span>'
            f'<span class="value {rsq_cls}">{self._fmt(a.get("r_squared", 0), 4)}</span></div>\n'
            f'  <div class="card"><span class="label">Uncertainty</span>'
            f'<span class="value">{self._fmt(a.get("savings_uncertainty_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Overall</span>'
            f'<span class="value">{a.get("overall_result", "-")}</span></div>\n'
            '</div>'
        )

    def _html_eu_eed(self, data: Dict[str, Any]) -> str:
        """Render HTML EU EED Article 7 table."""
        eed = data.get("eu_eed_article7", [])
        rows = ""
        for item in eed:
            cls = "severity-low" if item.get("status") == "PASS" else "severity-high"
            rows += (
                f'<tr><td>{item.get("requirement", "-")}</td>'
                f'<td class="{cls}">{item.get("status", "-")}</td>'
                f'<td>{item.get("evidence", "-")}</td></tr>\n'
            )
        return (
            '<h2>6. EU EED Article 7</h2>\n'
            '<table>\n<tr><th>Requirement</th><th>Status</th>'
            f'<th>Evidence</th></tr>\n{rows}</table>'
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality assessment."""
        dq = data.get("data_quality", {})
        return (
            '<h2>7. Data Quality</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Completeness</span>'
            f'<span class="value">{self._fmt(dq.get("completeness_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Accuracy</span>'
            f'<span class="value">{self._fmt(dq.get("accuracy_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Consistency</span>'
            f'<span class="value">{self._fmt(dq.get("consistency_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Gaps</span>'
            f'<span class="value">{dq.get("gaps_detected", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Overall</span>'
            f'<span class="value">{dq.get("overall_status", "-")}</span></div>\n'
            '</div>'
        )

    def _html_documentation_review(self, data: Dict[str, Any]) -> str:
        """Render HTML documentation review table."""
        docs = data.get("documentation_review", [])
        rows = ""
        for doc in docs:
            cls = "severity-low" if doc.get("status") == "OK" else "severity-high"
            rows += (
                f'<tr><td>{doc.get("document", "-")}</td>'
                f'<td>{doc.get("required", "-")}</td>'
                f'<td>{doc.get("present", "-")}</td>'
                f'<td class="{cls}">{doc.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>8. Documentation Review</h2>\n'
            '<table>\n<tr><th>Document</th><th>Required</th>'
            f'<th>Present</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_non_conformities(self, data: Dict[str, Any]) -> str:
        """Render HTML non-conformities table."""
        ncs = data.get("non_conformities", [])
        rows = ""
        for i, nc in enumerate(ncs, 1):
            cls = "severity-high" if nc.get("severity") == "Major" else "severity-medium"
            rows += (
                f'<tr><td>{i}</td><td>{nc.get("standard", "-")}</td>'
                f'<td>{nc.get("finding", "-")}</td>'
                f'<td class="{cls}">{nc.get("severity", "-")}</td>'
                f'<td>{nc.get("corrective_action", "-")}</td>'
                f'<td>{nc.get("due_date", "-")}</td></tr>\n'
            )
        return (
            '<h2>9. Non-Conformities</h2>\n'
            '<table>\n<tr><th>#</th><th>Standard</th><th>Finding</th>'
            f'<th>Severity</th><th>Corrective Action</th><th>Due</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Address all non-conformities before next compliance audit",
            "Maintain ASHRAE 14 statistical criteria through model monitoring",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>10. Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_compliance_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON compliance summary."""
        s = data.get("compliance_summary", {})
        return {
            "ipmvp_score_pct": s.get("ipmvp_score_pct", 0),
            "iso_score_pct": s.get("iso_score_pct", 0),
            "femp_score_pct": s.get("femp_score_pct", 0),
            "ashrae_score_pct": s.get("ashrae_score_pct", 0),
            "eed_score_pct": s.get("eed_score_pct", 0),
            "overall_score_pct": s.get("overall_score_pct", 0),
            "overall_status": s.get("overall_status", ""),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        s = data.get("compliance_summary", {})
        ncs = data.get("non_conformities", [])
        severity_counts: Dict[str, int] = {}
        for nc in ncs:
            sev = nc.get("severity", "Unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        return {
            "compliance_radar": {
                "type": "radar",
                "labels": ["IPMVP", "ISO 50015", "FEMP", "ASHRAE 14", "EU EED"],
                "values": [
                    s.get("ipmvp_score_pct", 0),
                    s.get("iso_score_pct", 0),
                    s.get("femp_score_pct", 0),
                    s.get("ashrae_score_pct", 0),
                    s.get("eed_score_pct", 0),
                ],
            },
            "nc_severity": {
                "type": "donut",
                "labels": list(severity_counts.keys()),
                "values": list(severity_counts.values()),
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "h3{color:#495057;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".severity-high{color:#dc3545;font-weight:700;}"
            ".severity-medium{color:#fd7e14;font-weight:600;}"
            ".severity-low{color:#198754;font-weight:500;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string (e.g., 'EUR 1,234.00').
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _format_energy(self, val: Any) -> str:
        """Format an energy value with units.

        Args:
            val: Energy value in MWh.

        Returns:
            Formatted energy string (e.g., '1,234.00 MWh').
        """
        if isinstance(val, (int, float)):
            return f"{val:,.2f} MWh"
        return str(val)

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators.

        Args:
            val: Value to format.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage.

        Args:
            part: Numerator value.
            whole: Denominator value.

        Returns:
            Formatted percentage string.
        """
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
