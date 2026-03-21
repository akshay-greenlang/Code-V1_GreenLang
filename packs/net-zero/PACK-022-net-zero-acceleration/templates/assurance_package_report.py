# -*- coding: utf-8 -*-
"""
AssurancePackageReportTemplate - Audit workpaper package for PACK-022.

Renders an assurance-ready workpaper package for external auditors covering
engagement summary, scope and boundary, standards applied, materiality,
methodology documentation, calculation traces, data lineage, control evidence,
exception register, completeness, change register, cross-checks, and provenance.

Sections:
    1. Engagement Summary
    2. Scope & Boundary
    3. Standards Applied
    4. Materiality Assessment
    5. Methodology Documentation (per scope)
    6. Calculation Traces (sample calculations)
    7. Data Lineage Map
    8. Control Evidence Summary
    9. Exception Register
   10. Completeness Matrix
   11. Change Register
   12. Cross-Check Results
   13. Provenance Chain
   14. Auditor Notes Section

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


def _dec_comma(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        negative = int_part.startswith("-")
        if negative:
            int_part = int_part[1:]
        formatted = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                formatted = "," + formatted
            formatted = ch + formatted
        if negative:
            formatted = "-" + formatted
        if len(parts) > 1:
            formatted += "." + parts[1]
        return formatted
    except Exception:
        return str(val)


class AssurancePackageReportTemplate:
    """
    Audit workpaper package report template for external assurance.

    Provides comprehensive audit evidence including methodology documentation,
    calculation traces, data lineage, control evidence, exception handling,
    cross-check results, and full provenance chain.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_engagement_summary(data),
            self._md_scope_boundary(data),
            self._md_standards(data),
            self._md_materiality(data),
            self._md_methodology(data),
            self._md_calculation_traces(data),
            self._md_data_lineage(data),
            self._md_control_evidence(data),
            self._md_exception_register(data),
            self._md_completeness_matrix(data),
            self._md_change_register(data),
            self._md_cross_checks(data),
            self._md_provenance_chain(data),
            self._md_auditor_notes(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_engagement_summary(data),
            self._html_scope_boundary(data),
            self._html_standards(data),
            self._html_materiality(data),
            self._html_methodology(data),
            self._html_calculation_traces(data),
            self._html_data_lineage(data),
            self._html_control_evidence(data),
            self._html_exception_register(data),
            self._html_completeness_matrix(data),
            self._html_change_register(data),
            self._html_cross_checks(data),
            self._html_provenance_chain(data),
            self._html_auditor_notes(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Assurance Package Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "assurance_package_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "engagement": data.get("engagement", {}),
            "scope_boundary": data.get("scope_boundary", {}),
            "standards": data.get("standards", []),
            "materiality": data.get("materiality", {}),
            "methodology": data.get("methodology", []),
            "calculation_traces": data.get("calculation_traces", []),
            "data_lineage": data.get("data_lineage", []),
            "control_evidence": data.get("control_evidence", []),
            "exceptions": data.get("exceptions", []),
            "completeness": data.get("completeness", []),
            "change_register": data.get("change_register", []),
            "cross_checks": data.get("cross_checks", []),
            "provenance_chain": data.get("provenance_chain", []),
            "auditor_notes": data.get("auditor_notes", ""),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Assurance Package Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Classification:** Confidential - Audit Workpaper\n\n---"
        )

    def _md_engagement_summary(self, data: Dict[str, Any]) -> str:
        eng = data.get("engagement", {})
        return (
            "## 1. Engagement Summary\n\n"
            f"- **Engagement Type:** {eng.get('type', 'Limited Assurance')}\n"
            f"- **Assurance Standard:** {eng.get('standard', 'ISAE 3410')}\n"
            f"- **Assurance Provider:** {eng.get('provider', 'N/A')}\n"
            f"- **Engagement Period:** {eng.get('period', 'N/A')}\n"
            f"- **Scope of Assurance:** {eng.get('scope', 'Scope 1 + Scope 2')}\n"
            f"- **Materiality Threshold:** {_dec(eng.get('materiality_pct', 5))}%\n"
            f"- **Prior Year Findings:** {eng.get('prior_findings', 0)} open items"
        )

    def _md_scope_boundary(self, data: Dict[str, Any]) -> str:
        sb = data.get("scope_boundary", {})
        entities = sb.get("entities", [])
        lines = [
            "## 2. Scope & Boundary\n",
            f"- **Organizational Boundary:** {sb.get('org_boundary', 'Operational Control')}\n"
            f"- **Reporting Period:** {sb.get('period', 'N/A')}\n"
            f"- **GHG Scopes:** {sb.get('ghg_scopes', 'Scope 1, 2, 3')}\n"
            f"- **Exclusions:** {sb.get('exclusions', 'None')}\n",
            "| Entity | Included | Justification |",
            "|--------|:--------:|---------------|",
        ]
        for e in entities:
            included = "Yes" if e.get("included", True) else "No"
            lines.append(f"| {e.get('name', '-')} | {included} | {e.get('justification', '-')} |")
        if not entities:
            lines.append("| _No entity boundary data_ | - | - |")
        return "\n".join(lines)

    def _md_standards(self, data: Dict[str, Any]) -> str:
        standards = data.get("standards", [])
        lines = [
            "## 3. Standards Applied\n",
            "| Standard | Version | Applicability | Notes |",
            "|----------|---------|---------------|-------|",
        ]
        for s in standards:
            lines.append(
                f"| {s.get('name', '-')} | {s.get('version', '-')} "
                f"| {s.get('applicability', '-')} | {s.get('notes', '-')} |"
            )
        if not standards:
            lines.append("| _No standards specified_ | - | - | - |")
        return "\n".join(lines)

    def _md_materiality(self, data: Dict[str, Any]) -> str:
        mat = data.get("materiality", {})
        items = mat.get("items", [])
        lines = [
            "## 4. Materiality Assessment\n",
            f"- **Overall Materiality:** {_dec(mat.get('threshold_pct', 5))}%\n"
            f"- **Quantitative Threshold:** {_dec_comma(mat.get('threshold_tco2e', 0))} tCO2e\n"
            f"- **Methodology:** {mat.get('methodology', 'Percentage of total emissions')}\n",
            "| Source | Emissions (tCO2e) | Material | Justification |",
            "|--------|------------------:|:--------:|---------------|",
        ]
        for item in items:
            material = "Yes" if item.get("material", True) else "No"
            lines.append(
                f"| {item.get('source', '-')} "
                f"| {_dec_comma(item.get('emissions_tco2e', 0))} "
                f"| {material} | {item.get('justification', '-')} |"
            )
        if not items:
            lines.append("| _No materiality items_ | - | - | - |")
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        methods = data.get("methodology", [])
        lines = ["## 5. Methodology Documentation\n"]
        for m in methods:
            lines.append(f"### {m.get('scope', 'Scope')}\n")
            lines.append(f"- **Calculation Method:** {m.get('method', 'N/A')}")
            lines.append(f"- **Emission Factors:** {m.get('emission_factors', 'N/A')}")
            lines.append(f"- **Data Sources:** {m.get('data_sources', 'N/A')}")
            lines.append(f"- **GWP Values:** {m.get('gwp_source', 'AR5')}")
            lines.append(f"- **Assumptions:** {m.get('assumptions', 'N/A')}")
            lines.append("")
        if not methods:
            lines.append("_No methodology documentation provided._")
        return "\n".join(lines)

    def _md_calculation_traces(self, data: Dict[str, Any]) -> str:
        traces = data.get("calculation_traces", [])
        lines = [
            "## 6. Calculation Traces\n",
            "Sample calculations for audit verification.\n",
        ]
        for i, trace in enumerate(traces, 1):
            lines.append(f"### Trace {i}: {trace.get('name', 'Calculation')}\n")
            lines.append(f"- **Source:** {trace.get('source', 'N/A')}")
            lines.append(f"- **Activity Data:** {trace.get('activity_data', 'N/A')}")
            lines.append(f"- **Emission Factor:** {trace.get('emission_factor', 'N/A')}")
            lines.append(f"- **Calculation:** {trace.get('formula', 'N/A')}")
            lines.append(f"- **Result:** {_dec_comma(trace.get('result_tco2e', 0))} tCO2e")
            lines.append(f"- **Verified:** {'Yes' if trace.get('verified', False) else 'No'}")
            lines.append("")
        if not traces:
            lines.append("_No calculation traces provided._")
        return "\n".join(lines)

    def _md_data_lineage(self, data: Dict[str, Any]) -> str:
        lineage = data.get("data_lineage", [])
        lines = [
            "## 7. Data Lineage Map\n",
            "| Data Point | Source System | Extraction | Transformation | Validation | Hash |",
            "|------------|:------------:|:----------:|:--------------:|:----------:|------|",
        ]
        for dl in lineage:
            lines.append(
                f"| {dl.get('data_point', '-')} "
                f"| {dl.get('source_system', '-')} "
                f"| {dl.get('extraction', '-')} "
                f"| {dl.get('transformation', '-')} "
                f"| {dl.get('validation', '-')} "
                f"| {dl.get('hash', '-')[:12]}... |"
            )
        if not lineage:
            lines.append("| _No lineage data_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_control_evidence(self, data: Dict[str, Any]) -> str:
        controls = data.get("control_evidence", [])
        lines = [
            "## 8. Control Evidence Summary\n",
            "| Control | Type | Operating Effectiveness | Evidence Ref | Last Tested |",
            "|---------|------|:----------------------:|:------------:|:-----------:|",
        ]
        for c in controls:
            effectiveness = c.get("effectiveness", "N/A")
            lines.append(
                f"| {c.get('control', '-')} | {c.get('type', '-')} "
                f"| {effectiveness} "
                f"| {c.get('evidence_ref', '-')} "
                f"| {c.get('last_tested', '-')} |"
            )
        if not controls:
            lines.append("| _No control evidence_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_exception_register(self, data: Dict[str, Any]) -> str:
        exceptions = data.get("exceptions", [])
        lines = [
            "## 9. Exception Register\n",
            "| # | Exception | Severity | Root Cause | Resolution | Status |",
            "|---|-----------|:--------:|------------|------------|:------:|",
        ]
        for i, ex in enumerate(exceptions, 1):
            lines.append(
                f"| {i} | {ex.get('description', '-')} "
                f"| {ex.get('severity', '-')} "
                f"| {ex.get('root_cause', '-')} "
                f"| {ex.get('resolution', '-')} "
                f"| {ex.get('status', '-')} |"
            )
        if not exceptions:
            lines.append("| - | _No exceptions recorded_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_completeness_matrix(self, data: Dict[str, Any]) -> str:
        completeness = data.get("completeness", [])
        lines = [
            "## 10. Completeness Matrix\n",
            "| Category | Expected Sources | Actual Sources | Coverage (%) | Gaps |",
            "|----------|:----------------:|:--------------:|:------------:|------|",
        ]
        for c in completeness:
            lines.append(
                f"| {c.get('category', '-')} "
                f"| {c.get('expected_sources', 0)} "
                f"| {c.get('actual_sources', 0)} "
                f"| {_dec(c.get('coverage_pct', 0))}% "
                f"| {c.get('gaps', '-')} |"
            )
        if not completeness:
            lines.append("| _No completeness data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_change_register(self, data: Dict[str, Any]) -> str:
        changes = data.get("change_register", [])
        lines = [
            "## 11. Change Register\n",
            "| Date | Change | Category | Impact | Approved By |",
            "|------|--------|----------|--------|-------------|",
        ]
        for ch in changes:
            lines.append(
                f"| {ch.get('date', '-')} | {ch.get('change', '-')} "
                f"| {ch.get('category', '-')} "
                f"| {ch.get('impact', '-')} "
                f"| {ch.get('approved_by', '-')} |"
            )
        if not changes:
            lines.append("| _No changes recorded_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_cross_checks(self, data: Dict[str, Any]) -> str:
        checks = data.get("cross_checks", [])
        lines = [
            "## 12. Cross-Check Results\n",
            "| Check | Source A | Source B | Variance (%) | Status | Notes |",
            "|-------|---------|---------|:------------:|:------:|-------|",
        ]
        for c in checks:
            status = "PASS" if c.get("pass", False) else "FAIL"
            lines.append(
                f"| {c.get('check', '-')} "
                f"| {c.get('source_a', '-')} "
                f"| {c.get('source_b', '-')} "
                f"| {_dec(c.get('variance_pct', 0))}% "
                f"| {status} "
                f"| {c.get('notes', '-')} |"
            )
        if not checks:
            lines.append("| _No cross-checks_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_provenance_chain(self, data: Dict[str, Any]) -> str:
        chain = data.get("provenance_chain", [])
        lines = [
            "## 13. Provenance Chain\n",
            "| Step | Actor | Action | Timestamp | Hash | Verified |",
            "|:----:|-------|--------|-----------|------|:--------:|",
        ]
        for i, step in enumerate(chain, 1):
            verified = "Yes" if step.get("verified", False) else "No"
            hash_val = step.get("hash", "")
            lines.append(
                f"| {i} | {step.get('actor', '-')} "
                f"| {step.get('action', '-')} "
                f"| {step.get('timestamp', '-')} "
                f"| {hash_val[:16]}... "
                f"| {verified} |"
            )
        if not chain:
            lines.append("| - | _No provenance data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_auditor_notes(self, data: Dict[str, Any]) -> str:
        notes = data.get("auditor_notes", "")
        return (
            "## 14. Auditor Notes\n\n"
            f"{notes if notes else '_Space reserved for auditor observations and recommendations._'}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-022 Net Zero Acceleration Pack on {ts}*  \n"
            f"*Assurance package prepared per ISAE 3000/3410 requirements.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;"
            "font-size:1.8em;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;"
            "padding-left:12px;font-size:1.3em;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".pass{color:#1b5e20;font-weight:700;}"
            ".fail{color:#c62828;font-weight:700;}"
            ".severity-high{color:#c62828;font-weight:600;}"
            ".severity-medium{color:#e65100;font-weight:600;}"
            ".severity-low{color:#1b5e20;font-weight:600;}"
            ".hash-code{font-family:monospace;font-size:0.85em;color:#616161;}"
            ".audit-notes{background:#f5f5f5;border:2px dashed #bdbdbd;border-radius:8px;"
            "padding:20px;min-height:100px;margin:15px 0;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Assurance Package Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts} | '
            f'<strong>Classification:</strong> Confidential</p>'
        )

    def _html_engagement_summary(self, data: Dict[str, Any]) -> str:
        eng = data.get("engagement", {})
        return (
            f'<h2>1. Engagement Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Type</div>'
            f'<div class="card-value">{eng.get("type", "Limited")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Standard</div>'
            f'<div class="card-value">{eng.get("standard", "ISAE 3410")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Provider</div>'
            f'<div class="card-value">{eng.get("provider", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Materiality</div>'
            f'<div class="card-value">{_dec(eng.get("materiality_pct", 5))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Prior Findings</div>'
            f'<div class="card-value">{eng.get("prior_findings", 0)}</div></div>\n'
            f'</div>'
        )

    def _html_scope_boundary(self, data: Dict[str, Any]) -> str:
        sb = data.get("scope_boundary", {})
        entities = sb.get("entities", [])
        rows = ""
        for e in entities:
            included = e.get("included", True)
            cls = "pass" if included else "fail"
            rows += (
                f'<tr><td>{e.get("name", "-")}</td>'
                f'<td class="{cls}">{"Yes" if included else "No"}</td>'
                f'<td>{e.get("justification", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. Scope & Boundary</h2>\n'
            f'<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Org Boundary</td><td>{sb.get("org_boundary", "Operational Control")}</td></tr>\n'
            f'<tr><td>Period</td><td>{sb.get("period", "N/A")}</td></tr>\n'
            f'<tr><td>GHG Scopes</td><td>{sb.get("ghg_scopes", "Scope 1, 2, 3")}</td></tr>\n'
            f'<tr><td>Exclusions</td><td>{sb.get("exclusions", "None")}</td></tr>\n'
            f'</table>\n'
            f'<h3>Entity Boundary</h3>\n'
            f'<table>\n'
            f'<tr><th>Entity</th><th>Included</th><th>Justification</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_standards(self, data: Dict[str, Any]) -> str:
        standards = data.get("standards", [])
        rows = ""
        for s in standards:
            rows += (
                f'<tr><td>{s.get("name", "-")}</td><td>{s.get("version", "-")}</td>'
                f'<td>{s.get("applicability", "-")}</td><td>{s.get("notes", "-")}</td></tr>\n'
            )
        return (
            f'<h2>3. Standards Applied</h2>\n'
            f'<table>\n'
            f'<tr><th>Standard</th><th>Version</th><th>Applicability</th><th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_materiality(self, data: Dict[str, Any]) -> str:
        mat = data.get("materiality", {})
        items = mat.get("items", [])
        rows = ""
        for item in items:
            material = item.get("material", True)
            cls = "pass" if material else "fail"
            rows += (
                f'<tr><td>{item.get("source", "-")}</td>'
                f'<td>{_dec_comma(item.get("emissions_tco2e", 0))}</td>'
                f'<td class="{cls}">{"Yes" if material else "No"}</td>'
                f'<td>{item.get("justification", "-")}</td></tr>\n'
            )
        return (
            f'<h2>4. Materiality Assessment</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Threshold</div>'
            f'<div class="card-value">{_dec(mat.get("threshold_pct", 5))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Quantitative</div>'
            f'<div class="card-value">{_dec_comma(mat.get("threshold_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Source</th><th>Emissions (tCO2e)</th><th>Material</th><th>Justification</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        methods = data.get("methodology", [])
        content = '<h2>5. Methodology Documentation</h2>\n'
        for m in methods:
            content += (
                f'<h3>{m.get("scope", "Scope")}</h3>\n'
                f'<table>\n'
                f'<tr><th>Parameter</th><th>Value</th></tr>\n'
                f'<tr><td>Calculation Method</td><td>{m.get("method", "N/A")}</td></tr>\n'
                f'<tr><td>Emission Factors</td><td>{m.get("emission_factors", "N/A")}</td></tr>\n'
                f'<tr><td>Data Sources</td><td>{m.get("data_sources", "N/A")}</td></tr>\n'
                f'<tr><td>GWP Values</td><td>{m.get("gwp_source", "AR5")}</td></tr>\n'
                f'<tr><td>Assumptions</td><td>{m.get("assumptions", "N/A")}</td></tr>\n'
                f'</table>\n'
            )
        return content

    def _html_calculation_traces(self, data: Dict[str, Any]) -> str:
        traces = data.get("calculation_traces", [])
        content = '<h2>6. Calculation Traces</h2>\n'
        for i, trace in enumerate(traces, 1):
            verified = trace.get("verified", False)
            v_cls = "pass" if verified else "fail"
            content += (
                f'<div style="margin:12px 0;padding:16px;border:1px solid #c8e6c9;border-radius:8px;">'
                f'<h3>Trace {i}: {trace.get("name", "Calculation")}</h3>\n'
                f'<table>\n'
                f'<tr><td><strong>Source:</strong></td><td>{trace.get("source", "N/A")}</td></tr>\n'
                f'<tr><td><strong>Activity Data:</strong></td><td>{trace.get("activity_data", "N/A")}</td></tr>\n'
                f'<tr><td><strong>Emission Factor:</strong></td><td>{trace.get("emission_factor", "N/A")}</td></tr>\n'
                f'<tr><td><strong>Formula:</strong></td><td><code>{trace.get("formula", "N/A")}</code></td></tr>\n'
                f'<tr><td><strong>Result:</strong></td><td>{_dec_comma(trace.get("result_tco2e", 0))} tCO2e</td></tr>\n'
                f'<tr><td><strong>Verified:</strong></td><td class="{v_cls}">{"Yes" if verified else "No"}</td></tr>\n'
                f'</table></div>\n'
            )
        return content

    def _html_data_lineage(self, data: Dict[str, Any]) -> str:
        lineage = data.get("data_lineage", [])
        rows = ""
        for dl in lineage:
            hash_val = dl.get("hash", "")
            rows += (
                f'<tr><td>{dl.get("data_point", "-")}</td>'
                f'<td>{dl.get("source_system", "-")}</td>'
                f'<td>{dl.get("extraction", "-")}</td>'
                f'<td>{dl.get("transformation", "-")}</td>'
                f'<td>{dl.get("validation", "-")}</td>'
                f'<td class="hash-code">{hash_val[:16]}...</td></tr>\n'
            )
        return (
            f'<h2>7. Data Lineage Map</h2>\n'
            f'<table>\n'
            f'<tr><th>Data Point</th><th>Source</th><th>Extraction</th>'
            f'<th>Transformation</th><th>Validation</th><th>Hash</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_control_evidence(self, data: Dict[str, Any]) -> str:
        controls = data.get("control_evidence", [])
        rows = ""
        for c in controls:
            eff = c.get("effectiveness", "N/A")
            cls = "pass" if eff.lower() in ("effective", "strong") else "fail" if eff.lower() in ("ineffective", "weak") else ""
            rows += (
                f'<tr><td>{c.get("control", "-")}</td><td>{c.get("type", "-")}</td>'
                f'<td class="{cls}">{eff}</td>'
                f'<td>{c.get("evidence_ref", "-")}</td>'
                f'<td>{c.get("last_tested", "-")}</td></tr>\n'
            )
        return (
            f'<h2>8. Control Evidence Summary</h2>\n'
            f'<table>\n'
            f'<tr><th>Control</th><th>Type</th><th>Effectiveness</th>'
            f'<th>Evidence Ref</th><th>Last Tested</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_exception_register(self, data: Dict[str, Any]) -> str:
        exceptions = data.get("exceptions", [])
        rows = ""
        for i, ex in enumerate(exceptions, 1):
            severity = ex.get("severity", "Medium")
            s_cls = (
                "severity-high" if severity.lower() == "high"
                else "severity-low" if severity.lower() == "low"
                else "severity-medium"
            )
            rows += (
                f'<tr><td>{i}</td><td>{ex.get("description", "-")}</td>'
                f'<td class="{s_cls}">{severity}</td>'
                f'<td>{ex.get("root_cause", "-")}</td>'
                f'<td>{ex.get("resolution", "-")}</td>'
                f'<td>{ex.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>9. Exception Register</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Exception</th><th>Severity</th>'
            f'<th>Root Cause</th><th>Resolution</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_completeness_matrix(self, data: Dict[str, Any]) -> str:
        completeness = data.get("completeness", [])
        rows = ""
        for c in completeness:
            pct = float(Decimal(str(c.get("coverage_pct", 0))))
            cls = "pass" if pct >= 95 else "fail" if pct < 80 else ""
            rows += (
                f'<tr><td>{c.get("category", "-")}</td>'
                f'<td>{c.get("expected_sources", 0)}</td>'
                f'<td>{c.get("actual_sources", 0)}</td>'
                f'<td class="{cls}">{_dec(pct)}%</td>'
                f'<td>{c.get("gaps", "-")}</td></tr>\n'
            )
        return (
            f'<h2>10. Completeness Matrix</h2>\n'
            f'<table>\n'
            f'<tr><th>Category</th><th>Expected</th><th>Actual</th>'
            f'<th>Coverage</th><th>Gaps</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_change_register(self, data: Dict[str, Any]) -> str:
        changes = data.get("change_register", [])
        rows = ""
        for ch in changes:
            rows += (
                f'<tr><td>{ch.get("date", "-")}</td><td>{ch.get("change", "-")}</td>'
                f'<td>{ch.get("category", "-")}</td><td>{ch.get("impact", "-")}</td>'
                f'<td>{ch.get("approved_by", "-")}</td></tr>\n'
            )
        return (
            f'<h2>11. Change Register</h2>\n'
            f'<table>\n'
            f'<tr><th>Date</th><th>Change</th><th>Category</th>'
            f'<th>Impact</th><th>Approved By</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_cross_checks(self, data: Dict[str, Any]) -> str:
        checks = data.get("cross_checks", [])
        rows = ""
        for c in checks:
            passed = c.get("pass", False)
            cls = "pass" if passed else "fail"
            icon = "&#10004;" if passed else "&#10008;"
            rows += (
                f'<tr><td>{c.get("check", "-")}</td>'
                f'<td>{c.get("source_a", "-")}</td>'
                f'<td>{c.get("source_b", "-")}</td>'
                f'<td>{_dec(c.get("variance_pct", 0))}%</td>'
                f'<td class="{cls}">{icon}</td>'
                f'<td>{c.get("notes", "-")}</td></tr>\n'
            )
        return (
            f'<h2>12. Cross-Check Results</h2>\n'
            f'<table>\n'
            f'<tr><th>Check</th><th>Source A</th><th>Source B</th>'
            f'<th>Variance</th><th>Status</th><th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_provenance_chain(self, data: Dict[str, Any]) -> str:
        chain = data.get("provenance_chain", [])
        rows = ""
        for i, step in enumerate(chain, 1):
            verified = step.get("verified", False)
            cls = "pass" if verified else "fail"
            hash_val = step.get("hash", "")
            rows += (
                f'<tr><td>{i}</td><td>{step.get("actor", "-")}</td>'
                f'<td>{step.get("action", "-")}</td>'
                f'<td>{step.get("timestamp", "-")}</td>'
                f'<td class="hash-code">{hash_val[:16]}...</td>'
                f'<td class="{cls}">{"Yes" if verified else "No"}</td></tr>\n'
            )
        return (
            f'<h2>13. Provenance Chain</h2>\n'
            f'<table>\n'
            f'<tr><th>Step</th><th>Actor</th><th>Action</th>'
            f'<th>Timestamp</th><th>Hash</th><th>Verified</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_auditor_notes(self, data: Dict[str, Any]) -> str:
        notes = data.get("auditor_notes", "")
        content = notes if notes else "<em>Space reserved for auditor observations and recommendations.</em>"
        return (
            f'<h2>14. Auditor Notes</h2>\n'
            f'<div class="audit-notes">{content}</div>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-022 Net Zero '
            f'Acceleration Pack on {ts}<br>'
            f'Assurance package prepared per ISAE 3000/3410.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
