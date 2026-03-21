# -*- coding: utf-8 -*-
"""
AssuranceEvidencePackageTemplate - ISO 14064-3 evidence package for PACK-029.

Renders an assurance evidence package with ISO 14064-3 workpaper structure,
evidence hierarchy, calculation trails, variance explanation documentation,
data quality assessment, and assurance provider checklist.
Multi-format output (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary
    2.  ISO 14064-3 Workpaper Structure
    3.  Evidence Hierarchy
    4.  Calculation Trails
    5.  Variance Explanation Documentation
    6.  Data Quality Assessment (Tier 1-5)
    7.  Assurance Provider Checklist
    8.  Materiality Assessment
    9.  Control Environment Review
    10. Document Register
    11. XBRL Tagging Summary
    12. Audit Trail & Provenance

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"
_TEMPLATE_ID = "assurance_evidence_package"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

EVIDENCE_HIERARCHY = [
    {"tier": 1, "name": "Primary Data (Measured)", "desc": "Continuous emissions monitoring (CEMS), metered data", "quality": "Highest"},
    {"tier": 2, "name": "Primary Data (Calculated)", "desc": "Activity data with site-specific emission factors", "quality": "High"},
    {"tier": 3, "name": "Secondary Data (Industry Average)", "desc": "Published emission factors (DEFRA, EPA, IPCC)", "quality": "Medium"},
    {"tier": 4, "name": "Modelled / Estimated", "desc": "Spend-based, proxy data, extrapolation", "quality": "Low"},
    {"tier": 5, "name": "Default / Assumed", "desc": "Default factors, assumptions, expert judgement", "quality": "Lowest"},
]

ASSURANCE_CHECKLIST = [
    {"id": "AC01", "item": "Organizational boundary documented", "standard": "ISO 14064-1, Clause 5.1"},
    {"id": "AC02", "item": "GHG sources and sinks identified", "standard": "ISO 14064-1, Clause 5.2"},
    {"id": "AC03", "item": "Quantification methodologies documented", "standard": "ISO 14064-1, Clause 5.3"},
    {"id": "AC04", "item": "Emission factors sources referenced", "standard": "ISO 14064-1, Clause 5.3"},
    {"id": "AC05", "item": "Activity data audit trail complete", "standard": "ISO 14064-3, Clause 6"},
    {"id": "AC06", "item": "Uncertainty assessment performed", "standard": "ISO 14064-1, Annex A"},
    {"id": "AC07", "item": "Base year recalculation policy documented", "standard": "GHG Protocol, Ch 5"},
    {"id": "AC08", "item": "Scope 3 screening completed (all 15 categories)", "standard": "GHG Protocol Scope 3"},
    {"id": "AC09", "item": "Data management system controls reviewed", "standard": "ISO 14064-3, Clause 6.3"},
    {"id": "AC10", "item": "Materiality threshold defined (5% or less)", "standard": "ISO 14064-3, Clause 6.4"},
    {"id": "AC11", "item": "Interim target methodology documented", "standard": "SBTi Criteria v5.1"},
    {"id": "AC12", "item": "Variance explanations for >5% deviations", "standard": "ISO 14064-3, Clause 6.5"},
    {"id": "AC13", "item": "Third-party verification statement obtained", "standard": "ISO 14064-3, Clause 7"},
    {"id": "AC14", "item": "Corrective actions documented (if applicable)", "standard": "ISO 14064-3, Clause 7.3"},
    {"id": "AC15", "item": "Board/management sign-off obtained", "standard": "Corporate Governance"},
]

XBRL_TAGS: Dict[str, str] = {
    "assurance_level": "gl:AssuranceLevel",
    "assurance_provider": "gl:AssuranceProvider",
    "assurance_standard": "gl:AssuranceStandard",
    "materiality_threshold": "gl:MaterialityThreshold",
    "data_quality_score": "gl:DataQualityScore",
    "evidence_tier_avg": "gl:EvidenceTierAverage",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
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


class AssuranceEvidencePackageTemplate:
    """
    Assurance evidence package template for PACK-029 Interim Targets Pack.

    Renders ISO 14064-3 aligned workpapers with evidence hierarchy,
    calculation trails, data quality assessment, and assurance checklist.
    Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = AssuranceEvidencePackageTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "assurance_level": "Limited",
        ...     "assurance_provider": "VerifyCo LLP",
        ...     "calculation_trails": [...],
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render assurance evidence package as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_workpaper(data), self._md_evidence_hierarchy(data),
            self._md_calculation_trails(data), self._md_variance_docs(data),
            self._md_data_quality(data), self._md_assurance_checklist(data),
            self._md_materiality(data), self._md_controls(data),
            self._md_document_register(data), self._md_xbrl(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render assurance evidence package as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_workpaper(data), self._html_evidence_hierarchy(data),
            self._html_calculation_trails(data), self._html_variance_docs(data),
            self._html_data_quality(data), self._html_assurance_checklist(data),
            self._html_materiality(data), self._html_controls(data),
            self._html_document_register(data), self._html_xbrl(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Assurance Evidence - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        self.generated_at = _utcnow()
        checklist_results = data.get("checklist_results", {})
        passed = sum(1 for v in checklist_results.values() if v.get("status") == "pass")
        total = len(ASSURANCE_CHECKLIST)

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "assurance": {
                "level": data.get("assurance_level", "Limited"),
                "provider": data.get("assurance_provider", ""),
                "standard": data.get("assurance_standard", "ISO 14064-3"),
                "checklist_passed": passed, "checklist_total": total,
                "readiness_score": str(round((passed / max(1, total)) * 100, 1)),
            },
            "calculation_trails": data.get("calculation_trails", []),
            "data_quality": data.get("data_quality_assessment", {}),
            "variance_explanations": data.get("variance_explanations", []),
            "documents": data.get("document_register", []),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"Assurance Evidence - {data.get('org_name','')}", "author": "GreenLang PACK-029"},
        }

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Assurance Evidence Package\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Assurance Level:** {data.get('assurance_level', 'Limited')}  \n"
            f"**Provider:** {data.get('assurance_provider', '')}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-029 v{_MODULE_VERSION}  \n"
            f"**Classification:** CONFIDENTIAL\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        checklist_results = data.get("checklist_results", {})
        passed = sum(1 for v in checklist_results.values() if v.get("status") == "pass")
        total = len(ASSURANCE_CHECKLIST)
        score = (passed / max(1, total)) * 100
        lines = [
            "## 1. Executive Summary\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Assurance Level | {data.get('assurance_level', 'Limited')} |",
            f"| Assurance Standard | {data.get('assurance_standard', 'ISO 14064-3')} |",
            f"| Provider | {data.get('assurance_provider', 'TBD')} |",
            f"| Checklist Score | {_dec(score, 1)}% ({passed}/{total}) |",
            f"| Readiness | {'Ready' if score >= 80 else 'Needs Work'} |",
            f"| Materiality Threshold | {data.get('materiality_threshold', '5%')} |",
        ]
        return "\n".join(lines)

    def _md_workpaper(self, data: Dict[str, Any]) -> str:
        workpapers = data.get("workpapers", [])
        lines = [
            "## 2. ISO 14064-3 Workpaper Structure\n",
            "| # | Workpaper | ISO Reference | Status | Reviewer |",
            "|---|-----------|-------------|--------|----------|",
        ]
        default_wps = [
            ("WP-01", "Organizational Boundary", "ISO 14064-1, 5.1"),
            ("WP-02", "GHG Sources & Sinks", "ISO 14064-1, 5.2"),
            ("WP-03", "Quantification Methodology", "ISO 14064-1, 5.3"),
            ("WP-04", "Activity Data Evidence", "ISO 14064-3, 6"),
            ("WP-05", "Emission Factor Validation", "ISO 14064-3, 6"),
            ("WP-06", "Calculation Verification", "ISO 14064-3, 6.3"),
            ("WP-07", "Uncertainty Assessment", "ISO 14064-1, Annex A"),
            ("WP-08", "Interim Target Validation", "SBTi Criteria v5.1"),
        ]
        for i, (ref, name, iso) in enumerate(default_wps, 1):
            wp = next((w for w in workpapers if w.get("reference") == ref), {})
            lines.append(
                f"| {i} | {ref}: {name} | {iso} | {wp.get('status', 'Pending')} | {wp.get('reviewer', 'TBD')} |"
            )
        return "\n".join(lines)

    def _md_evidence_hierarchy(self, data: Dict[str, Any]) -> str:
        scope_tiers = data.get("evidence_tiers", {})
        lines = [
            "## 3. Evidence Hierarchy\n",
            "| Tier | Category | Quality | Description |",
            "|:----:|----------|---------|-------------|",
        ]
        for eh in EVIDENCE_HIERARCHY:
            lines.append(f"| {eh['tier']} | {eh['name']} | {eh['quality']} | {eh['desc']} |")
        if scope_tiers:
            lines.extend([
                "\n### Evidence Tier by Scope\n",
                "| Scope | Tier 1 (%) | Tier 2 (%) | Tier 3 (%) | Tier 4 (%) | Tier 5 (%) | Weighted Avg |",
                "|-------|:---------:|:---------:|:---------:|:---------:|:---------:|:------------:|",
            ])
            for scope, tiers in scope_tiers.items():
                t1 = _dec(tiers.get("tier1_pct", 0))
                t2 = _dec(tiers.get("tier2_pct", 0))
                t3 = _dec(tiers.get("tier3_pct", 0))
                t4 = _dec(tiers.get("tier4_pct", 0))
                t5 = _dec(tiers.get("tier5_pct", 0))
                avg = _dec(tiers.get("weighted_avg", 0), 1)
                lines.append(f"| {scope} | {t1}% | {t2}% | {t3}% | {t4}% | {t5}% | {avg} |")
        return "\n".join(lines)

    def _md_calculation_trails(self, data: Dict[str, Any]) -> str:
        trails = data.get("calculation_trails", [])
        lines = [
            "## 4. Calculation Trails\n",
            "| # | Source | Activity Data | EF Source | EF Value | Emissions (tCO2e) | Verified |",
            "|---|--------|:------------:|---------:|:--------:|-----------------:|:--------:|",
        ]
        for i, t in enumerate(trails, 1):
            lines.append(
                f"| {i} | {t.get('source', '')} | {t.get('activity_data', '')} "
                f"| {t.get('ef_source', '')} | {t.get('ef_value', '')} "
                f"| {_dec_comma(t.get('emissions', 0), 0)} | {t.get('verified', 'No')} |"
            )
        if not trails:
            lines.append("| - | _No calculation trails provided_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_variance_docs(self, data: Dict[str, Any]) -> str:
        variances = data.get("variance_explanations", [])
        lines = [
            "## 5. Variance Explanation Documentation\n",
            "| # | Item | Prior (tCO2e) | Current (tCO2e) | Variance (%) | Explanation | Supported |",
            "|---|------|-------------:|----------------:|:------------:|------------|:---------:|",
        ]
        for i, v in enumerate(variances, 1):
            prior = float(v.get("prior", 0))
            current = float(v.get("current", 0))
            var_pct = ((current - prior) / prior * 100) if prior else 0
            lines.append(
                f"| {i} | {v.get('item', '')} | {_dec_comma(prior, 0)} | {_dec_comma(current, 0)} "
                f"| {'+' if var_pct > 0 else ''}{_dec(var_pct)}% "
                f"| {v.get('explanation', '')} | {v.get('supported', 'Yes')} |"
            )
        if not variances:
            lines.append("| - | _No variance explanations_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality_assessment", {})
        lines = [
            "## 6. Data Quality Assessment\n",
            "| Scope | Completeness | Consistency | Transparency | Accuracy | Overall |",
            "|-------|:-----------:|:----------:|:------------:|:--------:|:-------:|",
        ]
        for scope in ["scope1", "scope2", "scope3"]:
            sq = dq.get(scope, {})
            lines.append(
                f"| {scope.replace('scope', 'Scope ').title()} "
                f"| {sq.get('completeness', 'N/A')} | {sq.get('consistency', 'N/A')} "
                f"| {sq.get('transparency', 'N/A')} | {sq.get('accuracy', 'N/A')} "
                f"| {sq.get('overall', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_assurance_checklist(self, data: Dict[str, Any]) -> str:
        results = data.get("checklist_results", {})
        lines = [
            "## 7. Assurance Provider Checklist\n",
            "| ID | Item | Standard | Status | Notes |",
            "|----|------|----------|--------|-------|",
        ]
        passed = 0
        for ac in ASSURANCE_CHECKLIST:
            r = results.get(ac["id"], {})
            status = r.get("status", "pending")
            icon = "PASS" if status == "pass" else ("FAIL" if status == "fail" else "PENDING")
            if status == "pass":
                passed += 1
            lines.append(f"| {ac['id']} | {ac['item']} | {ac['standard']} | **{icon}** | {r.get('notes', '')} |")
        total = len(ASSURANCE_CHECKLIST)
        lines.append(f"\n**Score:** {_dec((passed / max(1, total)) * 100, 1)}% ({passed}/{total})")
        return "\n".join(lines)

    def _md_materiality(self, data: Dict[str, Any]) -> str:
        mat = data.get("materiality", {})
        lines = [
            "## 8. Materiality Assessment\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Materiality Threshold | {mat.get('threshold', '5%')} |",
            f"| Overall Materiality | {mat.get('overall', 'No material misstatements')} |",
            f"| Scope 1 Material Items | {mat.get('scope1_items', 0)} |",
            f"| Scope 2 Material Items | {mat.get('scope2_items', 0)} |",
            f"| Scope 3 Material Items | {mat.get('scope3_items', 0)} |",
        ]
        return "\n".join(lines)

    def _md_controls(self, data: Dict[str, Any]) -> str:
        controls = data.get("controls", {})
        lines = [
            "## 9. Control Environment Review\n",
            "| Control Area | Status | Effectiveness | Notes |",
            "|-------------|--------|:------------:|-------|",
        ]
        for area, details in controls.items():
            lines.append(
                f"| {area.replace('_', ' ').title()} | {details.get('status', 'N/A')} "
                f"| {details.get('effectiveness', 'N/A')} | {details.get('notes', '')} |"
            )
        if not controls:
            lines.append("| _No control areas reviewed_ | - | - | - |")
        return "\n".join(lines)

    def _md_document_register(self, data: Dict[str, Any]) -> str:
        docs = data.get("document_register", [])
        lines = [
            "## 10. Document Register\n",
            "| # | Document | Type | Date | Owner | Reference |",
            "|---|----------|------|------|-------|-----------|",
        ]
        for i, d in enumerate(docs, 1):
            lines.append(
                f"| {i} | {d.get('name', '')} | {d.get('type', '')} "
                f"| {d.get('date', '')} | {d.get('owner', '')} | {d.get('reference', '')} |"
            )
        if not docs:
            lines.append("| - | _No documents registered_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 11. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
        ]
        for key, tag in XBRL_TAGS.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {tag} | - |")
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 12. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-029 on {ts}*  \n*ISO 14064-3 assurance evidence package.*  \n*CONFIDENTIAL - For assurance provider use only.*"

    # -- HTML --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f9fbe7;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".pass{{color:{_SUCCESS};font-weight:700;}}.fail{{color:{_DANGER};font-weight:700;}}.pending{{color:{_WARN};}}"
            f".confidential{{background:#ffebee;color:{_DANGER};padding:8px 16px;border-radius:6px;font-weight:700;text-align:center;margin:10px 0;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="confidential">CONFIDENTIAL - Assurance Evidence Package</div>\n<h1>Assurance Evidence Package</h1>\n<p><strong>{data.get("org_name","")}</strong> | {data.get("assurance_level","Limited")} Assurance | {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        results = data.get("checklist_results", {})
        passed = sum(1 for v in results.values() if v.get("status") == "pass")
        total = len(ASSURANCE_CHECKLIST)
        score = (passed / max(1, total)) * 100
        return (
            f'<h2>1. Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Level</div><div class="card-value">{data.get("assurance_level","Limited")}</div></div>\n'
            f'<div class="card"><div class="card-label">Provider</div><div class="card-value">{data.get("assurance_provider","TBD")}</div></div>\n'
            f'<div class="card"><div class="card-label">Score</div><div class="card-value">{_dec(score,1)}%</div><div class="card-unit">{passed}/{total}</div></div>\n'
            f'</div>'
        )

    def _html_workpaper(self, data: Dict[str, Any]) -> str:
        return '<h2>2. Workpapers</h2>\n<p>See ISO 14064-3 workpaper structure in document register.</p>'

    def _html_evidence_hierarchy(self, data: Dict[str, Any]) -> str:
        rows = ""
        for eh in EVIDENCE_HIERARCHY:
            rows += f'<tr><td>{eh["tier"]}</td><td>{eh["name"]}</td><td>{eh["quality"]}</td><td>{eh["desc"]}</td></tr>\n'
        return f'<h2>3. Evidence Hierarchy</h2>\n<table>\n<tr><th>Tier</th><th>Category</th><th>Quality</th><th>Description</th></tr>\n{rows}</table>'

    def _html_calculation_trails(self, data: Dict[str, Any]) -> str:
        trails = data.get("calculation_trails", [])
        rows = ""
        for i, t in enumerate(trails, 1):
            rows += f'<tr><td>{i}</td><td>{t.get("source","")}</td><td>{t.get("activity_data","")}</td><td>{_dec_comma(t.get("emissions",0), 0)}</td><td>{t.get("verified","No")}</td></tr>\n'
        return f'<h2>4. Calculation Trails</h2>\n<table>\n<tr><th>#</th><th>Source</th><th>Activity</th><th>Emissions</th><th>Verified</th></tr>\n{rows}</table>'

    def _html_variance_docs(self, data: Dict[str, Any]) -> str:
        variances = data.get("variance_explanations", [])
        rows = ""
        for i, v in enumerate(variances, 1):
            rows += f'<tr><td>{i}</td><td>{v.get("item","")}</td><td>{_dec_comma(v.get("prior",0), 0)}</td><td>{_dec_comma(v.get("current",0), 0)}</td><td>{v.get("explanation","")}</td></tr>\n'
        return f'<h2>5. Variance Docs</h2>\n<table>\n<tr><th>#</th><th>Item</th><th>Prior</th><th>Current</th><th>Explanation</th></tr>\n{rows}</table>'

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality_assessment", {})
        rows = ""
        for scope in ["scope1", "scope2", "scope3"]:
            sq = dq.get(scope, {})
            rows += f'<tr><td>{scope.replace("scope","Scope ").title()}</td><td>{sq.get("completeness","N/A")}</td><td>{sq.get("accuracy","N/A")}</td><td>{sq.get("overall","N/A")}</td></tr>\n'
        return f'<h2>6. Data Quality</h2>\n<table>\n<tr><th>Scope</th><th>Completeness</th><th>Accuracy</th><th>Overall</th></tr>\n{rows}</table>'

    def _html_assurance_checklist(self, data: Dict[str, Any]) -> str:
        results = data.get("checklist_results", {})
        rows = ""
        for ac in ASSURANCE_CHECKLIST:
            r = results.get(ac["id"], {})
            s = r.get("status", "pending")
            cls = "pass" if s == "pass" else ("fail" if s == "fail" else "pending")
            rows += f'<tr><td>{ac["id"]}</td><td>{ac["item"]}</td><td class="{cls}">{"PASS" if s == "pass" else ("FAIL" if s == "fail" else "PENDING")}</td></tr>\n'
        return f'<h2>7. Checklist</h2>\n<table>\n<tr><th>ID</th><th>Item</th><th>Status</th></tr>\n{rows}</table>'

    def _html_materiality(self, data: Dict[str, Any]) -> str:
        mat = data.get("materiality", {})
        return f'<h2>8. Materiality</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Threshold</td><td>{mat.get("threshold","5%")}</td></tr>\n<tr><td>Overall</td><td>{mat.get("overall","No material misstatements")}</td></tr>\n</table>'

    def _html_controls(self, data: Dict[str, Any]) -> str:
        controls = data.get("controls", {})
        rows = ""
        for area, d in controls.items():
            rows += f'<tr><td>{area.replace("_"," ").title()}</td><td>{d.get("status","N/A")}</td><td>{d.get("effectiveness","N/A")}</td></tr>\n'
        return f'<h2>9. Controls</h2>\n<table>\n<tr><th>Area</th><th>Status</th><th>Effectiveness</th></tr>\n{rows}</table>'

    def _html_document_register(self, data: Dict[str, Any]) -> str:
        docs = data.get("document_register", [])
        rows = ""
        for i, d in enumerate(docs, 1):
            rows += f'<tr><td>{i}</td><td>{d.get("name","")}</td><td>{d.get("type","")}</td><td>{d.get("date","")}</td></tr>\n'
        return f'<h2>10. Documents</h2>\n<table>\n<tr><th>#</th><th>Document</th><th>Type</th><th>Date</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>11. XBRL Tags</h2>\n<table>\n<tr><th>Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>12. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-029 on {ts} - CONFIDENTIAL assurance evidence</div>'
