# -*- coding: utf-8 -*-
"""
SBTiValidationReportTemplate - SBTi SDA pathway validation for PACK-028.

Renders a comprehensive SBTi target validation report with criteria checklist,
validation results, gap analysis, and improvement recommendations.
Multi-format (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary
    2.  Target Overview
    3.  SBTi Near-Term Criteria Checklist (C1-C10)
    4.  SDA-Specific Criteria
    5.  Coverage Analysis
    6.  Pathway Alignment Assessment
    7.  Long-Term / Net-Zero Criteria
    8.  FLAG Sector Assessment
    9.  Gap Analysis (Failed Criteria)
    10. Improvement Recommendations
    11. Submission Readiness Score
    12. XBRL Tagging Summary
    13. Audit Trail & Provenance

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "28.0.0"
_PACK_ID = "PACK-028"
_TEMPLATE_ID = "sbti_validation_report"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

NEAR_TERM_CRITERIA: List[Dict[str, str]] = [
    {"id": "C1", "criterion": "Boundary: 95% Scope 1+2 emissions coverage", "category": "Coverage"},
    {"id": "C2", "criterion": "Scope 3 screening completed (all 15 categories)", "category": "Coverage"},
    {"id": "C3", "criterion": "Scope 3 target if >40% of total (67% coverage)", "category": "Coverage"},
    {"id": "C4", "criterion": "Timeframe: 5-10 years from submission date", "category": "Timeframe"},
    {"id": "C5", "criterion": "Base year: most recent year with reliable data", "category": "Base Year"},
    {"id": "C6", "criterion": "Ambition: minimum 4.2% annual linear reduction (1.5C)", "category": "Ambition"},
    {"id": "C7", "criterion": "SDA method applied for eligible sectors", "category": "Methodology"},
    {"id": "C8", "criterion": "Intensity metric matches SBTi sector guidance", "category": "Methodology"},
    {"id": "C9", "criterion": "No exclusions without documented justification", "category": "Boundary"},
    {"id": "C10", "criterion": "Recalculation policy for structural changes >5%", "category": "Policy"},
]

SDA_CRITERIA: List[Dict[str, str]] = [
    {"id": "SDA-1", "criterion": "Sector classification per SBTi taxonomy (NACE/GICS/ISIC)", "category": "Classification"},
    {"id": "SDA-2", "criterion": "Sector-specific intensity metric used correctly", "category": "Intensity"},
    {"id": "SDA-3", "criterion": "Base year intensity validated against sector data", "category": "Baseline"},
    {"id": "SDA-4", "criterion": "Convergence pathway matches SBTi sector pathway", "category": "Pathway"},
    {"id": "SDA-5", "criterion": "Activity growth projections documented and justified", "category": "Growth"},
    {"id": "SDA-6", "criterion": "Absolute emissions check vs. intensity trajectory", "category": "Cross-Check"},
]

LONG_TERM_CRITERIA: List[Dict[str, str]] = [
    {"id": "LT-1", "criterion": "Net-zero target year by 2050 or sooner", "category": "Timeline"},
    {"id": "LT-2", "criterion": "Long-term target: 90%+ reduction from base year", "category": "Ambition"},
    {"id": "LT-3", "criterion": "Residual emissions <= 10% of base year", "category": "Residual"},
    {"id": "LT-4", "criterion": "Neutralization strategy (permanent removals only)", "category": "Neutralization"},
    {"id": "LT-5", "criterion": "Beyond value chain mitigation commitment", "category": "BVCM"},
]

XBRL_VALIDATION_TAGS: Dict[str, str] = {
    "submission_ready": "gl:SBTiSubmissionReadiness",
    "readiness_score": "gl:SBTiReadinessScore",
    "criteria_passed": "gl:SBTiCriteriaPassed",
    "criteria_total": "gl:SBTiCriteriaTotal",
    "gap_count": "gl:SBTiValidationGapCount",
    "ambition_level": "gl:SBTiAmbitionLevel",
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


class SBTiValidationReportTemplate:
    """
    SBTi SDA pathway validation and compliance report template.

    Validates sector pathway against SBTi criteria (near-term C1-C10,
    SDA-specific, long-term/net-zero), produces gap analysis, and
    generates improvement recommendations. Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = SBTiValidationReportTemplate()
        >>> data = {"org_name": "CementCo", "sector_id": "cement", "validation_results": {...}}
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_exec_summary(data),
            self._md_target_overview(data), self._md_near_term(data),
            self._md_sda_criteria(data), self._md_coverage(data),
            self._md_pathway_alignment(data), self._md_long_term(data),
            self._md_flag(data), self._md_gap_analysis(data),
            self._md_recommendations(data), self._md_readiness(data),
            self._md_xbrl(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_exec_summary(data),
            self._html_target_overview(data), self._html_near_term(data),
            self._html_sda_criteria(data), self._html_coverage(data),
            self._html_pathway_alignment(data), self._html_long_term(data),
            self._html_flag(data), self._html_gap_analysis(data),
            self._html_recommendations(data), self._html_readiness(data),
            self._html_xbrl(data), self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>SBTi Validation - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        vr = data.get("validation_results", {})
        nt = vr.get("near_term", {})
        sda = vr.get("sda", {})
        lt = vr.get("long_term", {})
        all_results = {**nt, **sda, **lt}
        total = len(all_results)
        passed = sum(1 for v in all_results.values() if v.get("status") == "pass")
        failed = sum(1 for v in all_results.values() if v.get("status") == "fail")
        score = (passed / max(1, total)) * 100

        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "sector_id": data.get("sector_id", ""),
            "summary": {
                "total_criteria": total, "passed": passed, "failed": failed,
                "pending": total - passed - failed,
                "readiness_score": str(round(score, 1)),
                "submission_ready": score >= 90,
            },
            "near_term_results": nt, "sda_results": sda, "long_term_results": lt,
            "coverage": data.get("coverage", {}),
            "pathway_alignment": data.get("pathway_alignment", {}),
            "flag_assessment": data.get("flag_assessment", {}),
            "gaps": [k for k, v in all_results.items() if v.get("status") == "fail"],
            "recommendations": data.get("recommendations", []),
            "xbrl_tags": {k: XBRL_VALIDATION_TAGS[k] for k in XBRL_VALIDATION_TAGS},
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"format": "pdf", "html_content": self.render_html(data),
                "structured_data": self.render_json(data),
                "metadata": {"title": f"SBTi Validation - {data.get('org_name','')}", "author": "GreenLang PACK-028"}}

    def _get_all_results(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        vr = data.get("validation_results", {})
        return {**vr.get("near_term", {}), **vr.get("sda", {}), **vr.get("long_term", {})}

    def _calc_score(self, data: Dict[str, Any]) -> float:
        results = self._get_all_results(data)
        total = len(results)
        passed = sum(1 for v in results.values() if v.get("status") == "pass")
        return (passed / max(1, total)) * 100

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# SBTi Validation Report\n\n**Organization:** {data.get('org_name','')}  \n**Sector:** {data.get('sector_id','').replace('_',' ').title()}  \n**Report Date:** {ts}  \n**Pack:** PACK-028 v{_MODULE_VERSION}\n\n---"

    def _md_exec_summary(self, data: Dict[str, Any]) -> str:
        results = self._get_all_results(data)
        total = len(results)
        passed = sum(1 for v in results.values() if v.get("status") == "pass")
        failed = sum(1 for v in results.values() if v.get("status") == "fail")
        pending = total - passed - failed
        score = self._calc_score(data)
        ready = "YES" if score >= 90 else "NO"
        lines = [
            "## 1. Executive Summary\n",
            f"| KPI | Value |", f"|-----|-------|",
            f"| Total Criteria | {total} |",
            f"| Passed | {passed} |",
            f"| Failed | {failed} |",
            f"| Pending | {pending} |",
            f"| Readiness Score | **{_dec(score, 1)}%** |",
            f"| Submission Ready | **{ready}** |",
        ]
        return "\n".join(lines)

    def _md_target_overview(self, data: Dict[str, Any]) -> str:
        target = data.get("target_overview", {})
        lines = [
            "## 2. Target Overview\n",
            f"| Parameter | Value |", f"|-----------|-------|",
            f"| Target Type | {target.get('type', 'SDA (Sector Decarbonization Approach)')} |",
            f"| Base Year | {target.get('base_year', '')} |",
            f"| Near-Term Target Year | {target.get('near_term_year', '')} |",
            f"| Long-Term Target Year | {target.get('long_term_year', 2050)} |",
            f"| Ambition Level | {target.get('ambition', '1.5C-aligned')} |",
            f"| Scope Coverage | {target.get('scope_coverage', 'Scope 1+2 (SDA) + Scope 3')} |",
            f"| Boundary | {target.get('boundary', 'Operational control')} |",
        ]
        return "\n".join(lines)

    def _md_near_term(self, data: Dict[str, Any]) -> str:
        nt = data.get("validation_results", {}).get("near_term", {})
        lines = [
            "## 3. Near-Term Criteria Checklist\n",
            "| ID | Criterion | Status | Notes |",
            "|----|-----------:|--------|-------|",
        ]
        for c in NEAR_TERM_CRITERIA:
            r = nt.get(c["id"], {})
            status = r.get("status", "pending")
            icon = "PASS" if status == "pass" else ("FAIL" if status == "fail" else "PENDING")
            lines.append(f"| {c['id']} | {c['criterion']} | **{icon}** | {r.get('notes', '')} |")
        return "\n".join(lines)

    def _md_sda_criteria(self, data: Dict[str, Any]) -> str:
        sda = data.get("validation_results", {}).get("sda", {})
        lines = [
            "## 4. SDA-Specific Criteria\n",
            "| ID | Criterion | Status | Notes |",
            "|----|-----------:|--------|-------|",
        ]
        for c in SDA_CRITERIA:
            r = sda.get(c["id"], {})
            status = r.get("status", "pending")
            icon = "PASS" if status == "pass" else ("FAIL" if status == "fail" else "PENDING")
            lines.append(f"| {c['id']} | {c['criterion']} | **{icon}** | {r.get('notes', '')} |")
        return "\n".join(lines)

    def _md_coverage(self, data: Dict[str, Any]) -> str:
        cov = data.get("coverage", {})
        lines = [
            "## 5. Coverage Analysis\n",
            f"| Scope | Coverage (%) | Requirement | Status |",
            f"|-------|------------:|-------------|--------|",
            f"| Scope 1 | {_dec(cov.get('scope1_pct', 100))}% | 95% | {'PASS' if float(cov.get('scope1_pct', 100)) >= 95 else 'FAIL'} |",
            f"| Scope 2 | {_dec(cov.get('scope2_pct', 100))}% | 95% | {'PASS' if float(cov.get('scope2_pct', 100)) >= 95 else 'FAIL'} |",
            f"| Scope 1+2 Combined | {_dec(cov.get('scope12_pct', 100))}% | 95% | {'PASS' if float(cov.get('scope12_pct', 100)) >= 95 else 'FAIL'} |",
            f"| Scope 3 | {_dec(cov.get('scope3_pct', 0))}% | 67% (if >40% total) | {'PASS' if float(cov.get('scope3_pct', 0)) >= 67 else 'NEEDS REVIEW'} |",
        ]
        return "\n".join(lines)

    def _md_pathway_alignment(self, data: Dict[str, Any]) -> str:
        pa = data.get("pathway_alignment", {})
        lines = [
            "## 6. Pathway Alignment Assessment\n",
            f"| Parameter | Value |", f"|-----------|-------|",
            f"| SDA Pathway | {pa.get('sda_pathway', 'N/A')} |",
            f"| Alignment to SBTi Benchmark | {_dec(pa.get('alignment_pct', 0))}% |",
            f"| Deviation from Pathway | {_dec(pa.get('deviation_pct', 0))}% |",
            f"| Annual Reduction Rate | {_dec(pa.get('annual_rate', 0))}% |",
            f"| Required Rate (1.5C) | 4.2% |",
            f"| Rate Sufficient | {'Yes' if float(pa.get('annual_rate', 0)) >= 4.2 else 'No'} |",
            f"| Convergence Year | {pa.get('convergence_year', 'N/A')} |",
        ]
        return "\n".join(lines)

    def _md_long_term(self, data: Dict[str, Any]) -> str:
        lt = data.get("validation_results", {}).get("long_term", {})
        lines = [
            "## 7. Long-Term / Net-Zero Criteria\n",
            "| ID | Criterion | Status | Notes |",
            "|----|-----------:|--------|-------|",
        ]
        for c in LONG_TERM_CRITERIA:
            r = lt.get(c["id"], {})
            status = r.get("status", "pending")
            icon = "PASS" if status == "pass" else ("FAIL" if status == "fail" else "PENDING")
            lines.append(f"| {c['id']} | {c['criterion']} | **{icon}** | {r.get('notes', '')} |")
        return "\n".join(lines)

    def _md_flag(self, data: Dict[str, Any]) -> str:
        flag = data.get("flag_assessment", {})
        applicable = flag.get("applicable", False)
        lines = [
            "## 8. FLAG Sector Assessment\n",
            f"**FLAG Applicable:** {'Yes' if applicable else 'No'}\n",
        ]
        if applicable:
            lines.extend([
                f"| Parameter | Value |", f"|-----------|-------|",
                f"| FLAG Emissions | {flag.get('flag_emissions_tco2e', 'N/A')} tCO2e |",
                f"| FLAG Share of Total | {_dec(flag.get('flag_share_pct', 0))}% |",
                f"| Separate FLAG Target | {flag.get('separate_target', 'Required')} |",
                f"| Land Use Change Accounting | {flag.get('luc_method', 'N/A')} |",
            ])
        else:
            lines.append("_FLAG pathway assessment not applicable for this sector._")
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        results = self._get_all_results(data)
        gaps = [(k, v) for k, v in results.items() if v.get("status") == "fail"]
        lines = [
            "## 9. Gap Analysis\n",
            f"**Failed Criteria:** {len(gaps)}\n",
        ]
        if gaps:
            lines.append("| Criterion ID | Issue | Severity | Remediation Effort |")
            lines.append("|-------------|-------|----------|-------------------|")
            for cid, v in gaps:
                lines.append(
                    f"| {cid} | {v.get('notes', 'Failed validation')} "
                    f"| {v.get('severity', 'High')} | {v.get('effort', 'Medium')} |"
                )
        else:
            lines.append("_No gaps identified - all criteria passed._")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        lines = ["## 10. Improvement Recommendations\n"]
        if recs:
            for i, r in enumerate(recs, 1):
                lines.append(
                    f"{i}. **{r.get('title', '')}** ({r.get('priority', 'Medium')} priority)  \n"
                    f"   {r.get('description', '')}  \n"
                    f"   _Effort: {r.get('effort', 'Medium')} | Timeline: {r.get('timeline', 'TBD')}_\n"
                )
        else:
            lines.append("_Recommendations generated based on gap analysis._")
        return "\n".join(lines)

    def _md_readiness(self, data: Dict[str, Any]) -> str:
        score = self._calc_score(data)
        results = self._get_all_results(data)
        total = len(results)
        passed = sum(1 for v in results.values() if v.get("status") == "pass")
        level = "Ready" if score >= 90 else ("Nearly Ready" if score >= 70 else ("Needs Work" if score >= 50 else "Significant Gaps"))
        lines = [
            "## 11. Submission Readiness Score\n",
            f"| Metric | Value |", f"|--------|-------|",
            f"| Readiness Score | **{_dec(score, 1)}%** |",
            f"| Readiness Level | **{level}** |",
            f"| Criteria Passed | {passed} / {total} |",
            f"| Submission Ready | **{'YES' if score >= 90 else 'NO'}** |",
        ]
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        score = self._calc_score(data)
        results = self._get_all_results(data)
        passed = sum(1 for v in results.values() if v.get("status") == "pass")
        gaps = sum(1 for v in results.values() if v.get("status") == "fail")
        lines = [
            "## 12. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
            f"| Submission Ready | {XBRL_VALIDATION_TAGS['submission_ready']} | {'Yes' if score >= 90 else 'No'} |",
            f"| Readiness Score | {XBRL_VALIDATION_TAGS['readiness_score']} | {_dec(score, 1)}% |",
            f"| Criteria Passed | {XBRL_VALIDATION_TAGS['criteria_passed']} | {passed} |",
            f"| Criteria Total | {XBRL_VALIDATION_TAGS['criteria_total']} | {len(results)} |",
            f"| Gap Count | {XBRL_VALIDATION_TAGS['gap_count']} | {gaps} |",
        ]
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f"## 13. Audit Trail\n\n| Parameter | Value |\n|-----------|-------|\n| Report ID | `{rid}` |\n| Generated | {ts} |\n| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n| Hash | `{dh[:16]}...` |"

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}*  \n*SBTi SDA pathway validation report.*"

    # -- HTML --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f9fbe7;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".pass{{color:{_SUCCESS};font-weight:700;}}.fail{{color:{_DANGER};font-weight:700;}}.pending{{color:{_WARN};font-style:italic;}}"
            f".score-bar{{height:24px;background:#e0e0e0;border-radius:12px;overflow:hidden;margin:8px 0;}}"
            f".score-fill{{height:24px;border-radius:12px;background:linear-gradient(90deg,{_ACCENT},{_SUCCESS});}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>SBTi Validation Report</h1>\n<p><strong>Organization:</strong> {data.get("org_name","")} | <strong>Sector:</strong> {data.get("sector_id","").replace("_"," ").title()} | <strong>Generated:</strong> {ts}</p>'

    def _html_exec_summary(self, data: Dict[str, Any]) -> str:
        results = self._get_all_results(data)
        total = len(results)
        passed = sum(1 for v in results.values() if v.get("status") == "pass")
        failed = sum(1 for v in results.values() if v.get("status") == "fail")
        score = self._calc_score(data)
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Score</div><div class="card-value">{_dec(score, 1)}%</div></div>\n'
            f'<div class="card"><div class="card-label">Passed</div><div class="card-value">{passed}/{total}</div></div>\n'
            f'<div class="card"><div class="card-label">Failed</div><div class="card-value">{failed}</div></div>\n'
            f'<div class="card"><div class="card-label">Ready</div><div class="card-value">{"YES" if score >= 90 else "NO"}</div></div>\n'
            f'</div>\n<div class="score-bar"><div class="score-fill" style="width:{score}%"></div></div>'
        )

    def _html_target_overview(self, data: Dict[str, Any]) -> str:
        t = data.get("target_overview", {})
        return f'<h2>2. Target Overview</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Type</td><td>{t.get("type","SDA")}</td></tr>\n<tr><td>Base Year</td><td>{t.get("base_year","")}</td></tr>\n<tr><td>Near-Term Year</td><td>{t.get("near_term_year","")}</td></tr>\n<tr><td>Long-Term Year</td><td>{t.get("long_term_year",2050)}</td></tr>\n<tr><td>Ambition</td><td>{t.get("ambition","1.5C")}</td></tr>\n</table>'

    def _html_near_term(self, data: Dict[str, Any]) -> str:
        nt = data.get("validation_results", {}).get("near_term", {})
        rows = ""
        for c in NEAR_TERM_CRITERIA:
            r = nt.get(c["id"], {})
            s = r.get("status", "pending")
            rows += f'<tr><td>{c["id"]}</td><td>{c["criterion"]}</td><td class="{s}">{"PASS" if s == "pass" else ("FAIL" if s == "fail" else "PENDING")}</td><td>{r.get("notes","")}</td></tr>\n'
        return f'<h2>3. Near-Term Criteria</h2>\n<table>\n<tr><th>ID</th><th>Criterion</th><th>Status</th><th>Notes</th></tr>\n{rows}</table>'

    def _html_sda_criteria(self, data: Dict[str, Any]) -> str:
        sda = data.get("validation_results", {}).get("sda", {})
        rows = ""
        for c in SDA_CRITERIA:
            r = sda.get(c["id"], {})
            s = r.get("status", "pending")
            rows += f'<tr><td>{c["id"]}</td><td>{c["criterion"]}</td><td class="{s}">{"PASS" if s == "pass" else ("FAIL" if s == "fail" else "PENDING")}</td><td>{r.get("notes","")}</td></tr>\n'
        return f'<h2>4. SDA Criteria</h2>\n<table>\n<tr><th>ID</th><th>Criterion</th><th>Status</th><th>Notes</th></tr>\n{rows}</table>'

    def _html_coverage(self, data: Dict[str, Any]) -> str:
        cov = data.get("coverage", {})
        s1 = float(cov.get("scope1_pct", 100))
        s2 = float(cov.get("scope2_pct", 100))
        s12 = float(cov.get("scope12_pct", 100))
        s3 = float(cov.get("scope3_pct", 0))
        return (
            f'<h2>5. Coverage</h2>\n<table>\n<tr><th>Scope</th><th>Coverage</th><th>Required</th><th>Status</th></tr>\n'
            f'<tr><td>Scope 1</td><td>{_dec(s1)}%</td><td>95%</td><td class="{"pass" if s1 >= 95 else "fail"}">{"PASS" if s1 >= 95 else "FAIL"}</td></tr>\n'
            f'<tr><td>Scope 2</td><td>{_dec(s2)}%</td><td>95%</td><td class="{"pass" if s2 >= 95 else "fail"}">{"PASS" if s2 >= 95 else "FAIL"}</td></tr>\n'
            f'<tr><td>Scope 1+2</td><td>{_dec(s12)}%</td><td>95%</td><td class="{"pass" if s12 >= 95 else "fail"}">{"PASS" if s12 >= 95 else "FAIL"}</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{_dec(s3)}%</td><td>67%</td><td class="{"pass" if s3 >= 67 else "pending"}">{"PASS" if s3 >= 67 else "REVIEW"}</td></tr>\n'
            f'</table>'
        )

    def _html_pathway_alignment(self, data: Dict[str, Any]) -> str:
        pa = data.get("pathway_alignment", {})
        return (
            f'<h2>6. Pathway Alignment</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Alignment</td><td>{_dec(pa.get("alignment_pct",0))}%</td></tr>\n'
            f'<tr><td>Annual Rate</td><td>{_dec(pa.get("annual_rate",0))}%</td></tr>\n'
            f'<tr><td>Required (1.5C)</td><td>4.2%</td></tr>\n'
            f'<tr><td>Sufficient</td><td class="{"pass" if float(pa.get("annual_rate",0)) >= 4.2 else "fail"}">{"Yes" if float(pa.get("annual_rate",0)) >= 4.2 else "No"}</td></tr>\n'
            f'</table>'
        )

    def _html_long_term(self, data: Dict[str, Any]) -> str:
        lt = data.get("validation_results", {}).get("long_term", {})
        rows = ""
        for c in LONG_TERM_CRITERIA:
            r = lt.get(c["id"], {})
            s = r.get("status", "pending")
            rows += f'<tr><td>{c["id"]}</td><td>{c["criterion"]}</td><td class="{s}">{"PASS" if s == "pass" else ("FAIL" if s == "fail" else "PENDING")}</td></tr>\n'
        return f'<h2>7. Long-Term Criteria</h2>\n<table>\n<tr><th>ID</th><th>Criterion</th><th>Status</th></tr>\n{rows}</table>'

    def _html_flag(self, data: Dict[str, Any]) -> str:
        flag = data.get("flag_assessment", {})
        if not flag.get("applicable", False):
            return f'<h2>8. FLAG Assessment</h2>\n<p>Not applicable for this sector.</p>'
        return f'<h2>8. FLAG Assessment</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>FLAG Emissions</td><td>{flag.get("flag_emissions_tco2e","N/A")} tCO2e</td></tr>\n<tr><td>FLAG Share</td><td>{_dec(flag.get("flag_share_pct",0))}%</td></tr>\n</table>'

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        results = self._get_all_results(data)
        gaps = [(k, v) for k, v in results.items() if v.get("status") == "fail"]
        rows = "".join(f'<tr><td>{cid}</td><td>{v.get("notes","")}</td><td>{v.get("severity","High")}</td></tr>\n' for cid, v in gaps)
        return f'<h2>9. Gap Analysis</h2>\n<p>{len(gaps)} gaps identified</p>\n<table>\n<tr><th>ID</th><th>Issue</th><th>Severity</th></tr>\n{rows}</table>'

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        rows = "".join(f'<tr><td>{i}</td><td>{r.get("title","")}</td><td>{r.get("priority","Medium")}</td><td>{r.get("description","")}</td></tr>\n' for i, r in enumerate(recs, 1))
        return f'<h2>10. Recommendations</h2>\n<table>\n<tr><th>#</th><th>Title</th><th>Priority</th><th>Description</th></tr>\n{rows}</table>'

    def _html_readiness(self, data: Dict[str, Any]) -> str:
        score = self._calc_score(data)
        return (
            f'<h2>11. Readiness Score</h2>\n'
            f'<div class="summary-cards"><div class="card"><div class="card-label">Score</div>'
            f'<div class="card-value">{_dec(score, 1)}%</div>'
            f'<div class="card-unit">{"READY" if score >= 90 else "NOT READY"}</div></div></div>\n'
            f'<div class="score-bar"><div class="score-fill" style="width:{score}%"></div></div>'
        )

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        score = self._calc_score(data)
        return f'<h2>12. XBRL Tags</h2>\n<table>\n<tr><th>Point</th><th>Tag</th><th>Value</th></tr>\n<tr><td>Ready</td><td><code>{XBRL_VALIDATION_TAGS["submission_ready"]}</code></td><td>{"Yes" if score >= 90 else "No"}</td></tr>\n<tr><td>Score</td><td><code>{XBRL_VALIDATION_TAGS["readiness_score"]}</code></td><td>{_dec(score, 1)}%</td></tr>\n</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>13. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-028 on {ts} - SBTi SDA validation</div>'
