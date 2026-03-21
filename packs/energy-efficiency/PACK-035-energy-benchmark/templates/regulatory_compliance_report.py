# -*- coding: utf-8 -*-
"""
RegulatoryComplianceReportTemplate - Regulatory compliance report for PACK-035.

Generates regulatory compliance reports for energy benchmarking requirements
including MEES (UK), BEPS (US), BER (IE), and EU EPBD obligations.
Covers building data summary, rating calculations, compliance status
determination, certificate data, submission requirements, and timelines.

Sections:
    1. Header
    2. Regulatory Requirements
    3. Building Data Summary
    4. Rating Calculation
    5. Compliance Status (PASS / FAIL / AT_RISK)
    6. Certificate Data
    7. Submission Requirements
    8. Timeline
    9. Provenance

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegulatoryComplianceReportTemplate:
    """
    Regulatory compliance report template for energy benchmarking.

    Renders compliance reports for building energy regulations with
    status determination, rating calculations, submission requirements,
    and timelines across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RegulatoryComplianceReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render regulatory compliance report as Markdown.

        Args:
            data: Compliance assessment data from engine processing.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_regulatory_requirements(data),
            self._md_building_data(data),
            self._md_rating_calculation(data),
            self._md_compliance_status(data),
            self._md_certificate_data(data),
            self._md_submission_requirements(data),
            self._md_timeline(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render regulatory compliance report as self-contained HTML.

        Args:
            data: Compliance assessment data from engine processing.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_regulatory_requirements(data),
            self._html_building_data(data),
            self._html_rating_calculation(data),
            self._html_compliance_status(data),
            self._html_certificate_data(data),
            self._html_submission_requirements(data),
            self._html_timeline(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Regulatory Compliance Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render regulatory compliance report as structured JSON.

        Args:
            data: Compliance assessment data from engine processing.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "regulatory_compliance_report",
            "version": "35.0.0",
            "generated_at": self.generated_at.isoformat(),
            "regulation": data.get("regulation", {}),
            "building": data.get("building", {}),
            "rating_calculation": data.get("rating_calculation", {}),
            "compliance_status": data.get("compliance_status", {}),
            "certificate": data.get("certificate", {}),
            "submission_requirements": data.get("submission_requirements", []),
            "timeline": data.get("timeline", []),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            "# Regulatory Compliance Report\n\n"
            f"**Facility:** {data.get('facility_name', '-')}  \n"
            f"**Regulation:** {data.get('regulation_name', '-')}  \n"
            f"**Jurisdiction:** {data.get('jurisdiction', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-035 RegulatoryComplianceReportTemplate v35.0.0\n\n---"
        )

    def _md_regulatory_requirements(self, data: Dict[str, Any]) -> str:
        """Render regulatory requirements section."""
        reg = data.get("regulation", {})
        requirements = reg.get("requirements", [])
        lines = [
            "## 1. Regulatory Requirements\n",
            f"**Regulation:** {reg.get('name', '-')}  ",
            f"**Version / Year:** {reg.get('version', '-')}  ",
            f"**Scope:** {reg.get('scope', '-')}  ",
            f"**Applicability:** {reg.get('applicability', '-')}  ",
            f"**Threshold:** {reg.get('threshold', '-')}  ",
            f"**Penalty for Non-Compliance:** {reg.get('penalty', '-')}",
        ]
        if requirements:
            lines.extend(["\n### Key Requirements\n"])
            for i, r in enumerate(requirements, 1):
                lines.append(f"{i}. {r}")
        return "\n".join(lines)

    def _md_building_data(self, data: Dict[str, Any]) -> str:
        """Render building data summary section."""
        b = data.get("building", {})
        return (
            "## 2. Building Data Summary\n\n"
            "| Property | Value |\n|----------|-------|\n"
            f"| Name | {b.get('name', '-')} |\n"
            f"| Address | {b.get('address', '-')} |\n"
            f"| Type | {b.get('type', '-')} |\n"
            f"| Floor Area | {self._fmt(b.get('floor_area_sqm', 0), 0)} m2 |\n"
            f"| Year Built | {b.get('year_built', '-')} |\n"
            f"| Annual Energy | {self._fmt(b.get('annual_energy_kwh', 0), 0)} kWh |\n"
            f"| Site EUI | {self._fmt(b.get('site_eui', 0))} kWh/m2/yr |\n"
            f"| CO2 Emissions | {self._fmt(b.get('co2_kg_yr', 0), 0)} kg/yr |"
        )

    def _md_rating_calculation(self, data: Dict[str, Any]) -> str:
        """Render rating calculation section."""
        rc = data.get("rating_calculation", {})
        steps = rc.get("calculation_steps", [])
        lines = [
            "## 3. Rating Calculation\n",
            f"**Method:** {rc.get('method', '-')}  ",
            f"**Input EUI:** {self._fmt(rc.get('input_eui', 0))} kWh/m2/yr  ",
            f"**Normalised EUI:** {self._fmt(rc.get('normalised_eui', 0))} kWh/m2/yr  ",
            f"**Rating Score:** {self._fmt(rc.get('score', 0), 0)}  ",
            f"**Rating Grade:** {rc.get('grade', '-')}",
        ]
        if steps:
            lines.extend(["\n### Calculation Steps\n"])
            for i, s in enumerate(steps, 1):
                lines.append(f"{i}. {s}")
        return "\n".join(lines)

    def _md_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render compliance status section."""
        cs = data.get("compliance_status", {})
        status = cs.get("status", "UNKNOWN")
        checks = cs.get("checks", [])
        lines = [
            "## 4. Compliance Status\n",
            f"**Overall Status:** **{status}**  ",
            f"**Minimum Requirement:** {cs.get('minimum_requirement', '-')}  ",
            f"**Current Rating:** {cs.get('current_rating', '-')}  ",
            f"**Margin:** {cs.get('margin', '-')}",
        ]
        if checks:
            lines.extend([
                "\n### Compliance Checks\n",
                "| Check | Requirement | Actual | Result |",
                "|-------|-----------|--------|--------|",
            ])
            for c in checks:
                lines.append(
                    f"| {c.get('check', '-')} "
                    f"| {c.get('requirement', '-')} "
                    f"| {c.get('actual', '-')} "
                    f"| {c.get('result', '-')} |"
                )
        return "\n".join(lines)

    def _md_certificate_data(self, data: Dict[str, Any]) -> str:
        """Render certificate data section."""
        cert = data.get("certificate", {})
        return (
            "## 5. Certificate Data\n\n"
            "| Field | Value |\n|-------|-------|\n"
            f"| Certificate Type | {cert.get('type', '-')} |\n"
            f"| Certificate Number | {cert.get('number', '-')} |\n"
            f"| Issue Date | {cert.get('issue_date', '-')} |\n"
            f"| Expiry Date | {cert.get('expiry_date', '-')} |\n"
            f"| Issuing Body | {cert.get('issuing_body', '-')} |\n"
            f"| Status | {cert.get('status', '-')} |"
        )

    def _md_submission_requirements(self, data: Dict[str, Any]) -> str:
        """Render submission requirements section."""
        reqs = data.get("submission_requirements", [])
        if not reqs:
            return "## 6. Submission Requirements\n\n_No submission requirements._"
        lines = [
            "## 6. Submission Requirements\n",
            "| # | Requirement | Format | Deadline | Status |",
            "|---|-----------|--------|----------|--------|",
        ]
        for i, r in enumerate(reqs, 1):
            lines.append(
                f"| {i} | {r.get('requirement', '-')} "
                f"| {r.get('format', '-')} "
                f"| {r.get('deadline', '-')} "
                f"| {r.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_timeline(self, data: Dict[str, Any]) -> str:
        """Render timeline section."""
        timeline = data.get("timeline", [])
        if not timeline:
            return "## 7. Timeline\n\n_No timeline events._"
        lines = [
            "## 7. Timeline\n",
            "| Date | Event | Action Required | Status |",
            "|------|-------|----------------|--------|",
        ]
        for t in timeline:
            lines.append(
                f"| {t.get('date', '-')} "
                f"| {t.get('event', '-')} "
                f"| {t.get('action', '-')} "
                f"| {t.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-035 Energy Benchmark Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Regulatory Compliance Report</h1>\n'
            f'<p class="subtitle">Facility: {data.get("facility_name", "-")} | '
            f'Regulation: {data.get("regulation_name", "-")} | '
            f'Generated: {ts}</p>'
        )

    def _html_regulatory_requirements(self, data: Dict[str, Any]) -> str:
        """Render HTML regulatory requirements."""
        reg = data.get("regulation", {})
        reqs = reg.get("requirements", [])
        items = "".join(f'<li>{r}</li>\n' for r in reqs)
        return (
            '<h2>Regulatory Requirements</h2>\n'
            f'<div class="info-box">'
            f'<p><strong>{reg.get("name", "-")}</strong> ({reg.get("version", "-")}) | '
            f'Scope: {reg.get("scope", "-")} | '
            f'Penalty: {reg.get("penalty", "-")}</p>'
            '</div>\n'
            f'<ol>\n{items}</ol>'
        )

    def _html_building_data(self, data: Dict[str, Any]) -> str:
        """Render HTML building data summary."""
        b = data.get("building", {})
        fields = [
            ("Name", b.get("name", "-")),
            ("Type", b.get("type", "-")),
            ("Floor Area", f"{self._fmt(b.get('floor_area_sqm', 0), 0)} m2"),
            ("Annual Energy", f"{self._fmt(b.get('annual_energy_kwh', 0), 0)} kWh"),
            ("Site EUI", f"{self._fmt(b.get('site_eui', 0))} kWh/m2/yr"),
        ]
        rows = "".join(
            f'<tr><td>{label}</td><td>{val}</td></tr>\n' for label, val in fields
        )
        return (
            '<h2>Building Data Summary</h2>\n'
            f'<table>\n<tr><th>Property</th><th>Value</th></tr>\n{rows}</table>'
        )

    def _html_rating_calculation(self, data: Dict[str, Any]) -> str:
        """Render HTML rating calculation."""
        rc = data.get("rating_calculation", {})
        return (
            '<h2>Rating Calculation</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Input EUI</span>'
            f'<span class="value">{self._fmt(rc.get("input_eui", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Normalised EUI</span>'
            f'<span class="value">{self._fmt(rc.get("normalised_eui", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Score</span>'
            f'<span class="value">{self._fmt(rc.get("score", 0), 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Grade</span>'
            f'<span class="value">{rc.get("grade", "-")}</span></div>\n'
            '</div>'
        )

    def _html_compliance_status(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance status with visual indicator."""
        cs = data.get("compliance_status", {})
        status = cs.get("status", "UNKNOWN")
        cls_map = {"PASS": "status-pass", "FAIL": "status-fail", "AT_RISK": "status-warn"}
        status_cls = cls_map.get(status, "")
        checks = cs.get("checks", [])
        rows = ""
        for c in checks:
            res = c.get("result", "")
            rcls = cls_map.get(res, "")
            rows += (
                f'<tr><td>{c.get("check", "-")}</td>'
                f'<td>{c.get("requirement", "-")}</td>'
                f'<td>{c.get("actual", "-")}</td>'
                f'<td class="{rcls}">{res}</td></tr>\n'
            )
        return (
            '<h2>Compliance Status</h2>\n'
            f'<p class="{status_cls}" style="font-size:1.5em;">Status: {status}</p>\n'
            '<table>\n<tr><th>Check</th><th>Requirement</th>'
            f'<th>Actual</th><th>Result</th></tr>\n{rows}</table>'
        )

    def _html_certificate_data(self, data: Dict[str, Any]) -> str:
        """Render HTML certificate data."""
        cert = data.get("certificate", {})
        return (
            '<h2>Certificate Data</h2>\n'
            f'<div class="info-box">'
            f'<p>Type: {cert.get("type", "-")} | '
            f'Number: {cert.get("number", "-")} | '
            f'Issued: {cert.get("issue_date", "-")} | '
            f'Expires: {cert.get("expiry_date", "-")} | '
            f'Status: {cert.get("status", "-")}</p></div>'
        )

    def _html_submission_requirements(self, data: Dict[str, Any]) -> str:
        """Render HTML submission requirements table."""
        reqs = data.get("submission_requirements", [])
        rows = "".join(
            f'<tr><td>{r.get("requirement", "-")}</td>'
            f'<td>{r.get("format", "-")}</td>'
            f'<td>{r.get("deadline", "-")}</td>'
            f'<td>{r.get("status", "-")}</td></tr>\n'
            for r in reqs
        )
        return (
            '<h2>Submission Requirements</h2>\n'
            '<table>\n<tr><th>Requirement</th><th>Format</th>'
            f'<th>Deadline</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_timeline(self, data: Dict[str, Any]) -> str:
        """Render HTML timeline."""
        timeline = data.get("timeline", [])
        items = "".join(
            f'<div class="timeline-item"><strong>{t.get("date", "-")}</strong> - '
            f'{t.get("event", "-")} | '
            f'Action: {t.get("action", "-")} | '
            f'Status: {t.get("status", "-")}</div>\n'
            for t in timeline
        )
        return f'<h2>Timeline</h2>\n{items}'

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
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".info-box{background:#e7f1ff;border-left:4px solid #0d6efd;padding:12px 16px;margin:15px 0;}"
            ".status-pass{color:#198754;font-weight:700;}"
            ".status-fail{color:#dc3545;font-weight:700;}"
            ".status-warn{color:#fd7e14;font-weight:700;}"
            ".timeline-item{border-left:3px solid #0d6efd;padding:8px 15px;margin:8px 0;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

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

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
