# -*- coding: utf-8 -*-
"""
CDPDisclosureTemplate - CDP Climate Change disclosure for PACK-029.

Renders CDP Climate Change questionnaire sections C4.1 (interim targets
description), C4.2 (interim targets table), cross-references to C5/C6,
public disclosure readiness check, and A-list optimization tips.
Multi-format output (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary
    2.  CDP C4.1 - Interim Targets Description
    3.  CDP C4.2 - Interim Targets Table
    4.  CDP C4.1a - Details of Interim Targets
    5.  Cross-Reference: C5 Emissions Performance
    6.  Cross-Reference: C6 Emissions Data & Methodology
    7.  Public Disclosure Readiness Check
    8.  A-List Optimization Tips
    9.  Data Quality Assessment
    10. XBRL Tagging Summary
    11. Audit Trail & Provenance

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"
_TEMPLATE_ID = "cdp_disclosure"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

CDP_READINESS_CHECKS = [
    {"id": "c4_1", "question": "C4.1: Have you set interim emissions targets?", "section": "C4.1"},
    {"id": "c4_2", "question": "C4.2: Interim target table completed with all required fields?", "section": "C4.2"},
    {"id": "c4_1a", "question": "C4.1a: Detailed target descriptions provided?", "section": "C4.1a"},
    {"id": "c5_ref", "question": "C5: Emissions performance data cross-referenced?", "section": "C5"},
    {"id": "c6_ref", "question": "C6: Methodology and emission factors documented?", "section": "C6"},
    {"id": "sbti", "question": "SBTi validation status disclosed?", "section": "C4.2"},
    {"id": "scope3", "question": "Scope 3 targets included (if >40% of total)?", "section": "C4.2"},
    {"id": "progress", "question": "Progress against targets reported (current year)?", "section": "C4.2"},
    {"id": "base_year", "question": "Base year and recalculation policy documented?", "section": "C4.1a"},
    {"id": "third_party", "question": "Third-party verification status disclosed?", "section": "C10"},
]

A_LIST_TIPS = [
    "Provide SBTi-validated targets (near-term + long-term + net-zero)",
    "Include all scopes with 95%+ boundary coverage",
    "Demonstrate year-over-year progress with quantified reductions",
    "Cross-reference targets to board-level governance (C1.1b)",
    "Link targets to transition plan and financial planning (C3.4)",
    "Provide detailed methodology and emission factor sources",
    "Ensure consistency between C4, C5, C6, and C7 responses",
    "Report on value chain engagement for Scope 3 (C12.1a)",
    "Disclose scenario analysis results aligned with TCFD (C3.2a)",
    "Include verification/assurance statement (C10.1a)",
]

XBRL_TAGS: Dict[str, str] = {
    "cdp_score": "gl:CDPClimateScore",
    "targets_disclosed": "gl:CDPInterimTargetsDisclosed",
    "readiness_score": "gl:CDPDisclosureReadinessScore",
    "sbti_status": "gl:CDPSBTiStatus",
}

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

class CDPDisclosureTemplate:
    """
    CDP Climate Change disclosure template for PACK-029.

    Renders CDP C4.1/C4.2 interim targets disclosure, cross-references
    to C5/C6, readiness checks, and A-list optimization guidance.
    Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = CDPDisclosureTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "targets": [
        ...         {"scope": "Scope 1+2", "base_year": 2022, "target_year": 2030,
        ...          "base_year_emissions": 50000, "target_reduction_pct": 46.2},
        ...     ],
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render CDP disclosure report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_c4_1(data), self._md_c4_2(data),
            self._md_c4_1a(data), self._md_c5_ref(data),
            self._md_c6_ref(data), self._md_readiness(data),
            self._md_a_list(data), self._md_data_quality(data),
            self._md_xbrl(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render CDP disclosure report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_c4_1(data), self._html_c4_2(data),
            self._html_c4_1a(data), self._html_c5_ref(data),
            self._html_c6_ref(data), self._html_readiness(data),
            self._html_a_list(data), self._html_data_quality(data),
            self._html_xbrl(data), self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>CDP Disclosure - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        self.generated_at = utcnow()
        targets = data.get("targets", [])
        readiness_results = data.get("readiness_results", {})
        passed = sum(1 for r in readiness_results.values() if r.get("status") == "pass")
        total = len(CDP_READINESS_CHECKS)
        score = (passed / max(1, total)) * 100

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "cdp_year": data.get("cdp_year", ""),
            "targets": targets,
            "c4_1_response": data.get("c4_1_response", "Yes"),
            "readiness": {
                "total_checks": total, "passed": passed,
                "score": str(round(score, 1)),
                "status": "Ready" if score >= 80 else "Needs Work",
            },
            "cross_references": {
                "c5": data.get("c5_reference", {}),
                "c6": data.get("c6_reference", {}),
            },
            "sbti_status": data.get("sbti_status", "Not committed"),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"CDP Disclosure - {data.get('org_name','')}", "author": "GreenLang PACK-029"},
        }

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# CDP Climate Change Disclosure Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**CDP Year:** {data.get('cdp_year', '')}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-029 v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        readiness_results = data.get("readiness_results", {})
        passed = sum(1 for r in readiness_results.values() if r.get("status") == "pass")
        total = len(CDP_READINESS_CHECKS)
        score = (passed / max(1, total)) * 100
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Targets Disclosed | {len(targets)} |",
            f"| SBTi Status | {data.get('sbti_status', 'Not committed')} |",
            f"| Readiness Score | {_dec(score, 1)}% ({passed}/{total} checks) |",
            f"| Disclosure Status | {'Ready for Submission' if score >= 80 else 'Needs Additional Work'} |",
        ]
        return "\n".join(lines)

    def _md_c4_1(self, data: Dict[str, Any]) -> str:
        response = data.get("c4_1_response", "Yes")
        description = data.get("c4_1_description", "")
        lines = [
            "## 2. CDP C4.1 - Interim Emission Reduction Targets\n",
            f"**C4.1: Did you have an emissions target that was active in the reporting year?**\n",
            f"**Response:** {response}\n",
        ]
        if description:
            lines.append(f"**Description:** {description}")
        targets = data.get("targets", [])
        if targets:
            lines.append(f"\n**Number of interim targets:** {len(targets)}")
        return "\n".join(lines)

    def _md_c4_2(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        lines = [
            "## 3. CDP C4.2 - Interim Targets Table\n",
            "| # | Target Ref | Scope | Base Year | Target Year | Base Emissions (tCO2e) | Target Reduction (%) | Target Emissions (tCO2e) | Progress (%) | SBTi Validated |",
            "|---|-----------|-------|:---------:|:-----------:|-----------------------:|---------------------:|-------------------------:|-------------:|:--------------:|",
        ]
        for i, t in enumerate(targets, 1):
            base_em = float(t.get("base_year_emissions", 0))
            red_pct = float(t.get("target_reduction_pct", 0))
            target_em = base_em * (1 - red_pct / 100)
            progress = float(t.get("progress_pct", 0))
            lines.append(
                f"| {i} | {t.get('reference', f'Target {i}')} | {t.get('scope', '')} "
                f"| {t.get('base_year', '')} | {t.get('target_year', '')} "
                f"| {_dec_comma(base_em, 0)} | {_dec(red_pct)}% "
                f"| {_dec_comma(target_em, 0)} | {_dec(progress)}% "
                f"| {t.get('sbti_validated', 'No')} |"
            )
        return "\n".join(lines)

    def _md_c4_1a(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        lines = ["## 4. CDP C4.1a - Interim Target Details\n"]
        for i, t in enumerate(targets, 1):
            lines.extend([
                f"### Target {i}: {t.get('reference', '')} ({t.get('scope', '')})\n",
                "| Field | Value |", "|-------|-------|",
                f"| Target Type | {t.get('target_type', 'Absolute')} |",
                f"| Scope | {t.get('scope', '')} |",
                f"| Covered Activities | {t.get('covered_activities', 'All operations')} |",
                f"| Base Year | {t.get('base_year', '')} |",
                f"| Start Year | {t.get('start_year', t.get('base_year', ''))} |",
                f"| Target Year | {t.get('target_year', '')} |",
                f"| Base Year Emissions | {_dec_comma(t.get('base_year_emissions', 0), 0)} tCO2e |",
                f"| Target Reduction | {_dec(t.get('target_reduction_pct', 0))}% |",
                f"| Methodology | {t.get('methodology', 'SBTi Absolute Contraction')} |",
                f"| SBTi Validated | {t.get('sbti_validated', 'No')} |",
                f"| Recalculation Policy | {t.get('recalculation_policy', 'Recalculate for structural changes >5%')} |",
                "",
            ])
        return "\n".join(lines)

    def _md_c5_ref(self, data: Dict[str, Any]) -> str:
        c5 = data.get("c5_reference", {})
        lines = [
            "## 5. Cross-Reference: C5 Emissions Performance\n",
            "| Field | Value |", "|-------|-------|",
            f"| Gross Global Scope 1 | {_dec_comma(c5.get('scope1', 0), 0)} tCO2e |",
            f"| Gross Global Scope 2 (Location) | {_dec_comma(c5.get('scope2_location', 0), 0)} tCO2e |",
            f"| Gross Global Scope 2 (Market) | {_dec_comma(c5.get('scope2_market', 0), 0)} tCO2e |",
            f"| Total Scope 3 | {_dec_comma(c5.get('scope3_total', 0), 0)} tCO2e |",
            f"| Reporting Year | {c5.get('reporting_year', '')} |",
            f"| Methodology | {c5.get('methodology', 'GHG Protocol')} |",
        ]
        return "\n".join(lines)

    def _md_c6_ref(self, data: Dict[str, Any]) -> str:
        c6 = data.get("c6_reference", {})
        lines = [
            "## 6. Cross-Reference: C6 Emissions Data & Methodology\n",
            "| Field | Value |", "|-------|-------|",
            f"| Consolidation Approach | {c6.get('consolidation', 'Operational control')} |",
            f"| Emission Factor Sources | {c6.get('ef_sources', 'IPCC AR6, IEA, DEFRA')} |",
            f"| GWP Values | {c6.get('gwp', 'IPCC AR6')} |",
            f"| Scope 3 Categories Reported | {c6.get('scope3_categories', '15')} |",
            f"| Scope 3 Methodology | {c6.get('scope3_methodology', 'Spend-based + activity-based hybrid')} |",
            f"| Verification | {c6.get('verification', 'Limited assurance')} |",
        ]
        return "\n".join(lines)

    def _md_readiness(self, data: Dict[str, Any]) -> str:
        readiness_results = data.get("readiness_results", {})
        lines = [
            "## 7. Public Disclosure Readiness Check\n",
            "| # | Check | Section | Status |",
            "|---|-------|---------|--------|",
        ]
        passed = 0
        for i, check in enumerate(CDP_READINESS_CHECKS, 1):
            result = readiness_results.get(check["id"], {})
            status = result.get("status", "pending")
            icon = "PASS" if status == "pass" else ("FAIL" if status == "fail" else "PENDING")
            if status == "pass":
                passed += 1
            lines.append(f"| {i} | {check['question']} | {check['section']} | **{icon}** |")
        total = len(CDP_READINESS_CHECKS)
        score = (passed / max(1, total)) * 100
        lines.append(f"\n**Readiness Score:** {_dec(score, 1)}% ({passed}/{total})")
        return "\n".join(lines)

    def _md_a_list(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 8. A-List Optimization Tips\n",
            "To maximize CDP Climate Change score and achieve A-list status:\n",
        ]
        for i, tip in enumerate(A_LIST_TIPS, 1):
            implemented = data.get("a_list_status", {}).get(f"tip_{i}", False)
            status = "Implemented" if implemented else "Recommended"
            lines.append(f"{i}. **{tip}** [{status}]")
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality", {})
        lines = [
            "## 9. Data Quality Assessment\n",
            "| Scope | Data Quality | Primary Data (%) | Verification |",
            "|-------|-------------|:----------------:|-------------|",
            f"| Scope 1 | {dq.get('scope1_quality', 'High')} | {_dec(dq.get('scope1_primary_pct', 95))}% | {dq.get('scope1_verification', 'Verified')} |",
            f"| Scope 2 | {dq.get('scope2_quality', 'High')} | {_dec(dq.get('scope2_primary_pct', 90))}% | {dq.get('scope2_verification', 'Verified')} |",
            f"| Scope 3 | {dq.get('scope3_quality', 'Medium')} | {_dec(dq.get('scope3_primary_pct', 40))}% | {dq.get('scope3_verification', 'Limited')} |",
        ]
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 10. XBRL Tagging Summary\n",
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
            "## 11. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-029 on {ts}*  \n*CDP Climate Change interim targets disclosure.*"

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
            f".score-bar{{height:20px;background:#e0e0e0;border-radius:10px;overflow:hidden;margin:8px 0;}}"
            f".score-fill{{height:20px;border-radius:10px;background:linear-gradient(90deg,{_ACCENT},{_SUCCESS});}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>CDP Climate Change Disclosure</h1>\n<p><strong>{data.get("org_name","")}</strong> | CDP {data.get("cdp_year","")} | {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        readiness_results = data.get("readiness_results", {})
        passed = sum(1 for r in readiness_results.values() if r.get("status") == "pass")
        score = (passed / max(1, len(CDP_READINESS_CHECKS))) * 100
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Targets</div><div class="card-value">{len(targets)}</div></div>\n'
            f'<div class="card"><div class="card-label">SBTi</div><div class="card-value">{data.get("sbti_status","N/A")}</div></div>\n'
            f'<div class="card"><div class="card-label">Readiness</div><div class="card-value">{_dec(score,1)}%</div></div>\n'
            f'</div>\n<div class="score-bar"><div class="score-fill" style="width:{score}%"></div></div>'
        )

    def _html_c4_1(self, data: Dict[str, Any]) -> str:
        return f'<h2>2. C4.1 - Interim Targets</h2>\n<p><strong>Response:</strong> {data.get("c4_1_response","Yes")}</p>\n<p>{data.get("c4_1_description","")}</p>'

    def _html_c4_2(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        rows = ""
        for i, t in enumerate(targets, 1):
            base_em = float(t.get("base_year_emissions", 0))
            red_pct = float(t.get("target_reduction_pct", 0))
            target_em = base_em * (1 - red_pct / 100)
            rows += (
                f'<tr><td>{i}</td><td>{t.get("scope","")}</td><td>{t.get("base_year","")}</td>'
                f'<td>{t.get("target_year","")}</td><td>{_dec_comma(base_em, 0)}</td>'
                f'<td>{_dec(red_pct)}%</td><td>{_dec_comma(target_em, 0)}</td>'
                f'<td>{_dec(t.get("progress_pct",0))}%</td><td>{t.get("sbti_validated","No")}</td></tr>\n'
            )
        return (
            f'<h2>3. C4.2 - Targets Table</h2>\n<table>\n'
            f'<tr><th>#</th><th>Scope</th><th>Base</th><th>Target</th><th>Base Em</th>'
            f'<th>Red %</th><th>Target Em</th><th>Progress</th><th>SBTi</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_c4_1a(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        html = '<h2>4. C4.1a - Target Details</h2>\n'
        for i, t in enumerate(targets, 1):
            html += (
                f'<h3>Target {i}: {t.get("reference","")}</h3>\n<table>\n<tr><th>Field</th><th>Value</th></tr>\n'
                f'<tr><td>Type</td><td>{t.get("target_type","Absolute")}</td></tr>\n'
                f'<tr><td>Scope</td><td>{t.get("scope","")}</td></tr>\n'
                f'<tr><td>Base Year</td><td>{t.get("base_year","")}</td></tr>\n'
                f'<tr><td>Target Year</td><td>{t.get("target_year","")}</td></tr>\n'
                f'<tr><td>Reduction</td><td>{_dec(t.get("target_reduction_pct",0))}%</td></tr>\n'
                f'</table>\n'
            )
        return html

    def _html_c5_ref(self, data: Dict[str, Any]) -> str:
        c5 = data.get("c5_reference", {})
        return (
            f'<h2>5. C5 Cross-Reference</h2>\n<table>\n<tr><th>Field</th><th>Value</th></tr>\n'
            f'<tr><td>Scope 1</td><td>{_dec_comma(c5.get("scope1",0), 0)} tCO2e</td></tr>\n'
            f'<tr><td>Scope 2 (Location)</td><td>{_dec_comma(c5.get("scope2_location",0), 0)} tCO2e</td></tr>\n'
            f'<tr><td>Scope 2 (Market)</td><td>{_dec_comma(c5.get("scope2_market",0), 0)} tCO2e</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{_dec_comma(c5.get("scope3_total",0), 0)} tCO2e</td></tr>\n'
            f'</table>'
        )

    def _html_c6_ref(self, data: Dict[str, Any]) -> str:
        c6 = data.get("c6_reference", {})
        return (
            f'<h2>6. C6 Cross-Reference</h2>\n<table>\n<tr><th>Field</th><th>Value</th></tr>\n'
            f'<tr><td>Consolidation</td><td>{c6.get("consolidation","Operational control")}</td></tr>\n'
            f'<tr><td>EF Sources</td><td>{c6.get("ef_sources","IPCC, IEA, DEFRA")}</td></tr>\n'
            f'<tr><td>GWP</td><td>{c6.get("gwp","IPCC AR6")}</td></tr>\n'
            f'</table>'
        )

    def _html_readiness(self, data: Dict[str, Any]) -> str:
        readiness_results = data.get("readiness_results", {})
        rows = ""
        for i, check in enumerate(CDP_READINESS_CHECKS, 1):
            r = readiness_results.get(check["id"], {})
            s = r.get("status", "pending")
            cls = "pass" if s == "pass" else ("fail" if s == "fail" else "pending")
            rows += f'<tr><td>{i}</td><td>{check["question"]}</td><td>{check["section"]}</td><td class="{cls}">{"PASS" if s == "pass" else ("FAIL" if s == "fail" else "PENDING")}</td></tr>\n'
        return f'<h2>7. Readiness Check</h2>\n<table>\n<tr><th>#</th><th>Check</th><th>Section</th><th>Status</th></tr>\n{rows}</table>'

    def _html_a_list(self, data: Dict[str, Any]) -> str:
        items = ""
        for i, tip in enumerate(A_LIST_TIPS, 1):
            items += f'<li>{tip}</li>\n'
        return f'<h2>8. A-List Tips</h2>\n<ol>\n{items}</ol>'

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality", {})
        return (
            f'<h2>9. Data Quality</h2>\n<table>\n<tr><th>Scope</th><th>Quality</th><th>Primary %</th></tr>\n'
            f'<tr><td>Scope 1</td><td>{dq.get("scope1_quality","High")}</td><td>{_dec(dq.get("scope1_primary_pct",95))}%</td></tr>\n'
            f'<tr><td>Scope 2</td><td>{dq.get("scope2_quality","High")}</td><td>{_dec(dq.get("scope2_primary_pct",90))}%</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{dq.get("scope3_quality","Medium")}</td><td>{_dec(dq.get("scope3_primary_pct",40))}%</td></tr>\n'
            f'</table>'
        )

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>10. XBRL Tags</h2>\n<table>\n<tr><th>Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>11. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-029 on {ts} - CDP disclosure</div>'
