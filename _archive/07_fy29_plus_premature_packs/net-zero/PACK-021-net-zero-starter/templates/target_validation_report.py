# -*- coding: utf-8 -*-
"""
TargetValidationReportTemplate - SBTi target validation and compliance for PACK-021.

Renders an SBTi-aligned target validation report with criteria checklists,
pathway analysis, coverage assessment, ambition scoring, milestone tracking,
sector benchmarking, and recommendations.

Sections:
    1. Target Summary
    2. SBTi Criteria Checklist (pass/fail)
    3. Pathway Details
    4. Coverage Analysis (by scope)
    5. Ambition Assessment (1.5C alignment)
    6. Milestones Timeline
    7. Comparison to Sector Benchmarks
    8. Recommendations

Author: GreenLang Team
Version: 21.0.0
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

_MODULE_VERSION = "21.0.0"

_SBTI_CRITERIA: List[Dict[str, str]] = [
    {"id": "C1", "name": "Scope 1+2 near-term target", "requirement": "Required for all companies"},
    {"id": "C2", "name": "Scope 3 screening completed", "requirement": "Required if Scope 3 >= 40% of total"},
    {"id": "C3", "name": "Scope 3 target (if material)", "requirement": "Cover >= 67% of Scope 3 emissions"},
    {"id": "C4", "name": "Near-term timeframe (5-10 years)", "requirement": "Target year within 5-10 years of submission"},
    {"id": "C5", "name": "Minimum ambition (1.5C or WB2C)", "requirement": "Cross-sector pathway or sector-specific"},
    {"id": "C6", "name": "Coverage >= 95% for S1+S2", "requirement": "Scope 1+2 boundary covers >= 95%"},
    {"id": "C7", "name": "Base year within last 5 years", "requirement": "Or recalculated to recent base"},
    {"id": "C8", "name": "No offsets in target boundary", "requirement": "Abatement only; offsets for beyond value chain"},
    {"id": "C9", "name": "Annual reduction rate sufficient", "requirement": ">= 4.2% linear for 1.5C cross-sector"},
    {"id": "C10", "name": "Long-term net-zero target", "requirement": "By 2050 or sooner; >= 90% abatement"},
]

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

class TargetValidationReportTemplate:
    """
    SBTi target validation and compliance report template.

    Validates net-zero targets against the SBTi Corporate Net-Zero Standard
    criteria, assesses pathway ambition, checks scope coverage, tracks
    milestones, and provides sector benchmarking context.

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
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_target_summary(data),
            self._md_criteria_checklist(data),
            self._md_pathway_details(data),
            self._md_coverage(data),
            self._md_ambition(data),
            self._md_milestones(data),
            self._md_benchmarks(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_target_summary(data),
            self._html_criteria_checklist(data),
            self._html_pathway_details(data),
            self._html_coverage(data),
            self._html_ambition(data),
            self._html_milestones(data),
            self._html_benchmarks(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Target Validation Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        criteria_results = data.get("criteria_results", {})
        pass_count = sum(1 for c in _SBTI_CRITERIA if criteria_results.get(c["id"], {}).get("pass", False))
        total_count = len(_SBTI_CRITERIA)

        result: Dict[str, Any] = {
            "template": "target_validation_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "target_summary": data.get("target_summary", {}),
            "criteria_checklist": {
                "total": total_count,
                "passed": pass_count,
                "failed": total_count - pass_count,
                "pass_rate_pct": str(
                    Decimal(str(pass_count)) / Decimal(str(total_count)) * 100
                    if total_count > 0 else Decimal("0")
                ),
                "criteria": [
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "requirement": c["requirement"],
                        "pass": criteria_results.get(c["id"], {}).get("pass", False),
                        "notes": criteria_results.get(c["id"], {}).get("notes", ""),
                    }
                    for c in _SBTI_CRITERIA
                ],
            },
            "pathway": data.get("pathway", {}),
            "coverage": data.get("coverage", {}),
            "ambition": data.get("ambition", {}),
            "milestones": data.get("milestones", []),
            "benchmarks": data.get("benchmarks", []),
            "recommendations": data.get("recommendations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# SBTi Target Validation Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Assessment Date:** {data.get('assessment_date', ts)}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_target_summary(self, data: Dict[str, Any]) -> str:
        ts_data = data.get("target_summary", {})
        near = ts_data.get("near_term", {})
        long = ts_data.get("long_term", {})
        return (
            "## 1. Target Summary\n\n"
            "### Near-Term Target\n\n"
            f"- **Scope:** {near.get('scope', 'S1+S2')}\n"
            f"- **Base Year:** {near.get('base_year', 'N/A')}\n"
            f"- **Target Year:** {near.get('target_year', 'N/A')}\n"
            f"- **Reduction:** {_dec(near.get('reduction_pct', 0))}%\n"
            f"- **Pathway:** {near.get('pathway', 'N/A')}\n"
            f"- **Annual Rate:** {_dec(near.get('annual_rate_pct', 0))}%/yr\n\n"
            "### Long-Term Target (Net-Zero)\n\n"
            f"- **Scope:** {long.get('scope', 'S1+S2+S3')}\n"
            f"- **Target Year:** {long.get('target_year', 'N/A')}\n"
            f"- **Reduction:** {_dec(long.get('reduction_pct', 0))}%\n"
            f"- **Residual Budget:** {_dec(long.get('residual_pct', 0))}% (max 10%)"
        )

    def _md_criteria_checklist(self, data: Dict[str, Any]) -> str:
        criteria_results = data.get("criteria_results", {})
        pass_count = sum(1 for c in _SBTI_CRITERIA if criteria_results.get(c["id"], {}).get("pass", False))
        lines = [
            "## 2. SBTi Criteria Checklist\n",
            f"**Result:** {pass_count}/{len(_SBTI_CRITERIA)} criteria passed\n",
            "| ID | Criterion | Requirement | Result | Notes |",
            "|----|-----------|------------|:------:|-------|",
        ]
        for c in _SBTI_CRITERIA:
            result = criteria_results.get(c["id"], {})
            passed = result.get("pass", False)
            icon = "PASS" if passed else "FAIL"
            notes = result.get("notes", "")
            lines.append(
                f"| {c['id']} | {c['name']} | {c['requirement']} "
                f"| {icon} | {notes} |"
            )
        return "\n".join(lines)

    def _md_pathway_details(self, data: Dict[str, Any]) -> str:
        pathway = data.get("pathway", {})
        scenario = pathway.get("scenario", "1.5C")
        method = pathway.get("method", "Absolute Contraction")
        base_emissions = pathway.get("base_emissions_tco2e", 0)
        target_emissions = pathway.get("target_emissions_tco2e", 0)
        annual_rate = pathway.get("annual_reduction_rate_pct", 0)
        years = pathway.get("yearly_projection", [])

        lines = [
            "## 3. Pathway Details\n",
            f"- **Scenario:** {scenario}\n"
            f"- **Method:** {method}\n"
            f"- **Base Emissions:** {_dec_comma(base_emissions)} tCO2e\n"
            f"- **Target Emissions:** {_dec_comma(target_emissions)} tCO2e\n"
            f"- **Annual Reduction Rate:** {_dec(annual_rate)}%/yr\n",
        ]
        if years:
            lines.append("| Year | Target Emissions (tCO2e) | Cumulative Reduction (%) |")
            lines.append("|------|-------------------------:|------------------------:|")
            for yr in years:
                lines.append(
                    f"| {yr.get('year', '-')} "
                    f"| {_dec_comma(yr.get('emissions_tco2e', 0))} "
                    f"| {_dec(yr.get('cumulative_reduction_pct', 0))}% |"
                )
        return "\n".join(lines)

    def _md_coverage(self, data: Dict[str, Any]) -> str:
        coverage = data.get("coverage", {})
        lines = [
            "## 4. Coverage Analysis\n",
            "| Scope | Total Emissions (tCO2e) | Covered (tCO2e) | Coverage (%) | Required (%) |",
            "|-------|------------------------:|----------------:|-------------:|-------------:|",
        ]
        for scope in ["scope1", "scope2", "scope3"]:
            sc = coverage.get(scope, {})
            total_e = sc.get("total_tco2e", 0)
            covered_e = sc.get("covered_tco2e", 0)
            coverage_pct = sc.get("coverage_pct", 0)
            required_pct = sc.get("required_pct", 0)
            label = scope.replace("scope", "Scope ")
            lines.append(
                f"| {label} | {_dec_comma(total_e)} | {_dec_comma(covered_e)} "
                f"| {_dec(coverage_pct)}% | {_dec(required_pct)}% |"
            )
        return "\n".join(lines)

    def _md_ambition(self, data: Dict[str, Any]) -> str:
        ambition = data.get("ambition", {})
        alignment = ambition.get("temperature_alignment", "N/A")
        probability = ambition.get("probability_pct", 0)
        classification = ambition.get("classification", "N/A")
        gap = ambition.get("ambition_gap_pct", 0)
        return (
            "## 5. Ambition Assessment\n\n"
            f"- **Temperature Alignment:** {alignment}\n"
            f"- **Probability:** {_dec(probability)}%\n"
            f"- **Classification:** {classification}\n"
            f"- **Ambition Gap:** {_dec(gap)}% (vs minimum 1.5C pathway)\n\n"
            f"**Assessment:** {'Target meets 1.5C ambition level.' if gap <= 0 else f'Target is {_dec(gap)}% below the minimum 1.5C pathway rate.'}"
        )

    def _md_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        lines = [
            "## 6. Milestones Timeline\n",
            "| Year | Milestone | Target (tCO2e) | Reduction from Base (%) | Status |",
            "|------|-----------|---------------:|------------------------:|--------|",
        ]
        for ms in milestones:
            lines.append(
                f"| {ms.get('year', '-')} | {ms.get('description', '-')} "
                f"| {_dec_comma(ms.get('target_tco2e', 0))} "
                f"| {_dec(ms.get('reduction_pct', 0))}% "
                f"| {ms.get('status', '-')} |"
            )
        if not milestones:
            lines.append("| - | _No milestones defined_ | - | - | - |")
        return "\n".join(lines)

    def _md_benchmarks(self, data: Dict[str, Any]) -> str:
        benchmarks = data.get("benchmarks", [])
        lines = [
            "## 7. Sector Benchmarks\n",
            "| Company / Benchmark | Annual Reduction (%) | Target Year | Pathway | Status |",
            "|---------------------|--------------------:|-------------|---------|--------|",
        ]
        for b in benchmarks:
            lines.append(
                f"| {b.get('name', '-')} | {_dec(b.get('annual_reduction_pct', 0))}% "
                f"| {b.get('target_year', '-')} | {b.get('pathway', '-')} "
                f"| {b.get('status', '-')} |"
            )
        if not benchmarks:
            lines.append("| _No sector benchmarks available_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        lines = ["## 8. Recommendations\n"]
        if recs:
            for i, rec in enumerate(recs, 1):
                priority = rec.get("priority", "MEDIUM")
                lines.append(
                    f"### {i}. [{priority}] {rec.get('title', 'Recommendation')}\n"
                )
                lines.append(f"{rec.get('description', '')}\n")
                if rec.get("actions"):
                    for action in rec["actions"]:
                        lines.append(f"  - {action}")
                lines.append("")
        else:
            lines.append("_All SBTi criteria are met. No further recommendations at this time._")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}*  \n"
            f"*SBTi Corporate Net-Zero Standard v1.0 criteria applied.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            "background:#f5f7f5;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;padding-left:12px;}"
            "h3{color:#388e3c;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".pass{color:#1b5e20;font-weight:700;font-size:1.2em;}"
            ".fail{color:#c62828;font-weight:700;font-size:1.2em;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-fail{border-left:4px solid #c62828;background:linear-gradient(135deg,#ffebee,#ffcdd2);}"
            ".card-fail .card-value{color:#c62828;}"
            ".progress-bar{background:#e0e0e0;border-radius:6px;height:18px;overflow:hidden;}"
            ".progress-fill{height:100%;border-radius:6px;}"
            ".fill-green{background:#43a047;}"
            ".fill-amber{background:#ff8f00;}"
            ".fill-red{background:#e53935;}"
            ".status-on_track{color:#1b5e20;font-weight:600;}"
            ".status-at_risk{color:#e65100;font-weight:600;}"
            ".status-behind{color:#c62828;font-weight:600;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>SBTi Target Validation Report</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_target_summary(self, data: Dict[str, Any]) -> str:
        ts_data = data.get("target_summary", {})
        near = ts_data.get("near_term", {})
        long = ts_data.get("long_term", {})
        return (
            f'<h2>1. Target Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Near-Term Scope</div>'
            f'<div class="card-value">{near.get("scope", "S1+S2")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Near-Term Year</div>'
            f'<div class="card-value">{near.get("target_year", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Near-Term Reduction</div>'
            f'<div class="card-value">{_dec(near.get("reduction_pct", 0))}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Net-Zero Year</div>'
            f'<div class="card-value">{long.get("target_year", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Net-Zero Reduction</div>'
            f'<div class="card-value">{_dec(long.get("reduction_pct", 0))}%</div></div>\n'
            f'</div>'
        )

    def _html_criteria_checklist(self, data: Dict[str, Any]) -> str:
        criteria_results = data.get("criteria_results", {})
        pass_count = sum(1 for c in _SBTI_CRITERIA if criteria_results.get(c["id"], {}).get("pass", False))
        total = len(_SBTI_CRITERIA)
        rows = ""
        for c in _SBTI_CRITERIA:
            result = criteria_results.get(c["id"], {})
            passed = result.get("pass", False)
            cls = "pass" if passed else "fail"
            icon = "&#10004;" if passed else "&#10008;"
            rows += (
                f'<tr><td>{c["id"]}</td><td>{c["name"]}</td>'
                f'<td>{c["requirement"]}</td>'
                f'<td class="{cls}">{icon}</td>'
                f'<td>{result.get("notes", "")}</td></tr>\n'
            )
        return (
            f'<h2>2. SBTi Criteria Checklist</h2>\n'
            f'<p><strong>Result:</strong> {pass_count}/{total} criteria passed</p>\n'
            f'<table>\n'
            f'<tr><th>ID</th><th>Criterion</th><th>Requirement</th>'
            f'<th>Result</th><th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_pathway_details(self, data: Dict[str, Any]) -> str:
        pathway = data.get("pathway", {})
        years = pathway.get("yearly_projection", [])
        rows = ""
        for yr in years:
            rows += (
                f'<tr><td>{yr.get("year", "-")}</td>'
                f'<td>{_dec_comma(yr.get("emissions_tco2e", 0))}</td>'
                f'<td>{_dec(yr.get("cumulative_reduction_pct", 0))}%</td></tr>\n'
            )
        return (
            f'<h2>3. Pathway Details</h2>\n'
            f'<p><strong>Scenario:</strong> {pathway.get("scenario", "1.5C")} | '
            f'<strong>Method:</strong> {pathway.get("method", "Absolute Contraction")} | '
            f'<strong>Annual Rate:</strong> {_dec(pathway.get("annual_reduction_rate_pct", 0))}%/yr</p>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Target Emissions (tCO2e)</th>'
            f'<th>Cumulative Reduction</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_coverage(self, data: Dict[str, Any]) -> str:
        coverage = data.get("coverage", {})
        rows = ""
        for scope in ["scope1", "scope2", "scope3"]:
            sc = coverage.get(scope, {})
            coverage_pct = float(Decimal(str(sc.get("coverage_pct", 0))))
            required = float(Decimal(str(sc.get("required_pct", 0))))
            met = coverage_pct >= required
            bar_color = "fill-green" if met else "fill-red"
            label = scope.replace("scope", "Scope ")
            rows += (
                f'<tr><td>{label}</td>'
                f'<td>{_dec_comma(sc.get("total_tco2e", 0))}</td>'
                f'<td>{_dec_comma(sc.get("covered_tco2e", 0))}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill {bar_color}" '
                f'style="width:{min(coverage_pct, 100)}%"></div></div> {_dec(coverage_pct)}%</td>'
                f'<td>{_dec(required)}%</td></tr>\n'
            )
        return (
            f'<h2>4. Coverage Analysis</h2>\n'
            f'<table>\n'
            f'<tr><th>Scope</th><th>Total (tCO2e)</th><th>Covered (tCO2e)</th>'
            f'<th>Coverage</th><th>Required</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_ambition(self, data: Dict[str, Any]) -> str:
        ambition = data.get("ambition", {})
        alignment = ambition.get("temperature_alignment", "N/A")
        classification = ambition.get("classification", "N/A")
        gap = float(Decimal(str(ambition.get("ambition_gap_pct", 0))))
        card_cls = "" if gap <= 0 else " card-fail"
        return (
            f'<h2>5. Ambition Assessment</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card{card_cls}"><div class="card-label">Temperature Alignment</div>'
            f'<div class="card-value">{alignment}</div></div>\n'
            f'  <div class="card{card_cls}"><div class="card-label">Classification</div>'
            f'<div class="card-value">{classification}</div></div>\n'
            f'  <div class="card{card_cls}"><div class="card-label">Ambition Gap</div>'
            f'<div class="card-value">{_dec(gap)}%</div></div>\n'
            f'</div>'
        )

    def _html_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        rows = ""
        for ms in milestones:
            status = ms.get("status", "pending")
            status_cls = (
                "status-on_track" if status.lower() in ("on_track", "on track", "completed")
                else "status-at_risk" if status.lower() in ("at_risk", "at risk")
                else "status-behind"
            )
            rows += (
                f'<tr><td>{ms.get("year", "-")}</td>'
                f'<td>{ms.get("description", "-")}</td>'
                f'<td>{_dec_comma(ms.get("target_tco2e", 0))}</td>'
                f'<td>{_dec(ms.get("reduction_pct", 0))}%</td>'
                f'<td class="{status_cls}">{status}</td></tr>\n'
            )
        return (
            f'<h2>6. Milestones Timeline</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Milestone</th><th>Target (tCO2e)</th>'
            f'<th>Reduction</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_benchmarks(self, data: Dict[str, Any]) -> str:
        benchmarks = data.get("benchmarks", [])
        rows = ""
        for b in benchmarks:
            rows += (
                f'<tr><td>{b.get("name", "-")}</td>'
                f'<td>{_dec(b.get("annual_reduction_pct", 0))}%</td>'
                f'<td>{b.get("target_year", "-")}</td>'
                f'<td>{b.get("pathway", "-")}</td>'
                f'<td>{b.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>7. Sector Benchmarks</h2>\n'
            f'<table>\n'
            f'<tr><th>Company / Benchmark</th><th>Annual Reduction</th>'
            f'<th>Target Year</th><th>Pathway</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        items = ""
        for i, rec in enumerate(recs, 1):
            priority = rec.get("priority", "MEDIUM")
            pri_cls = "fail" if priority == "HIGH" else "pass" if priority == "LOW" else ""
            actions_html = ""
            if rec.get("actions"):
                actions_html = "<ul>" + "".join(f"<li>{a}</li>" for a in rec["actions"]) + "</ul>"
            items += (
                f'<div style="margin:12px 0;padding:12px;border:1px solid #c8e6c9;border-radius:8px;">'
                f'<strong>{i}. <span class="{pri_cls}">[{priority}]</span> '
                f'{rec.get("title", "")}</strong>'
                f'<p>{rec.get("description", "")}</p>'
                f'{actions_html}</div>\n'
            )
        return (
            f'<h2>8. Recommendations</h2>\n'
            f'{items}'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-021 Net Zero Starter Pack on {ts}<br>'
            f'SBTi Corporate Net-Zero Standard v1.0</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
