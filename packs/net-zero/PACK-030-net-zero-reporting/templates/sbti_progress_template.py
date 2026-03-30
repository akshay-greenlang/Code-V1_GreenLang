# -*- coding: utf-8 -*-
"""
SBTiProgressTemplate - SBTi Annual Progress Report for PACK-030.

Renders an SBTi-compliant annual progress disclosure covering target
description, base year recalculation, progress tables by scope, variance
explanations, milestone tracking, forward-looking projections, and
next-steps action plan. Multi-format output (MD, HTML, JSON, PDF)
with SHA-256 provenance hashing.

Sections:
    1.  Executive Summary
    2.  Target Description & Commitment Status
    3.  Base Year & Recalculation Events
    4.  Progress Table by Scope
    5.  Variance Explanation
    6.  Milestone Tracking
    7.  Reduction Initiatives & Impact
    8.  Forward-Looking Projections
    9.  Methodology & Boundary
    10. Next Steps & Action Plan
    11. XBRL Tagging Summary
    12. Audit Trail & Provenance

Author: GreenLang Team
Version: 30.0.0
Pack: PACK-030 Net Zero Reporting Pack
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

_MODULE_VERSION = "30.0.0"
_PACK_ID = "PACK-030"
_TEMPLATE_ID = "sbti_progress"

_PRIMARY = "#0d3b66"
_SECONDARY = "#1a6b8a"
_ACCENT = "#28a745"
_LIGHT = "#e3f0f7"
_LIGHTER = "#f4f9fc"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

SBTI_COMMITMENT_STATUSES = [
    "Committed", "Target Set", "Validated - Near-term",
    "Validated - Net-Zero", "Target Expired", "Removed",
]

SBTI_DISCLOSURE_FIELDS = [
    "Organization name", "SBTi target status",
    "Base year and base year emissions", "Near-term target year and reduction",
    "Long-term target year and reduction", "Net-zero commitment year",
    "Current year emissions (Scope 1, 2, 3)",
    "Progress against near-term target (%)",
    "Progress against long-term target (%)",
    "Methodology used (ACA/SDA/combined)",
    "Sector classification", "Recalculation events",
    "Off-track disclosure and remediation plan",
]

XBRL_TAGS: Dict[str, str] = {
    "reporting_year": "gl:SBTiProgressReportingYear",
    "commitment_status": "gl:SBTiCommitmentStatus",
    "near_term_target": "gl:SBTiNearTermTarget",
    "long_term_target": "gl:SBTiLongTermTarget",
    "actual_emissions_total": "gl:SBTiActualEmissionsTotal",
    "target_emissions_total": "gl:SBTiTargetEmissionsTotal",
    "near_term_progress_pct": "gl:SBTiNearTermProgressPct",
    "long_term_progress_pct": "gl:SBTiLongTermProgressPct",
    "on_track": "gl:SBTiOnTrackStatus",
    "net_zero_year": "gl:SBTiNetZeroYear",
    "methodology": "gl:SBTiMethodology",
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

def _pct_change(current: Any, baseline: Any) -> Decimal:
    c = Decimal(str(current))
    b = Decimal(str(baseline))
    if b == 0:
        return Decimal("0.00")
    return ((c - b) / b * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def _variance(actual: float, target: float) -> Dict[str, Any]:
    diff = actual - target
    pct = (diff / target * 100) if target != 0 else 0
    if abs(pct) <= 5.0:
        rag = "GREEN"
    elif abs(pct) <= 15.0:
        rag = "AMBER"
    else:
        rag = "RED"
    return {
        "actual": actual, "target": target, "diff": round(diff, 2),
        "diff_pct": round(pct, 2), "rag": rag,
        "on_track": actual <= target,
    }

def _progress_pct(baseline: float, current: float, target: float) -> float:
    """Calculate progress percentage from baseline toward target."""
    total_reduction_needed = baseline - target
    if total_reduction_needed <= 0:
        return 100.0
    actual_reduction = baseline - current
    return round(min(actual_reduction / total_reduction_needed * 100, 100.0), 2)

class SBTiProgressTemplate:
    """
    SBTi annual progress report template for PACK-030 Net Zero Reporting Pack.

    Generates SBTi-compliant annual progress disclosure with target
    descriptions, base year info, progress tracking, variance analysis,
    milestone tracking, and forward projections. Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = SBTiProgressTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "reporting_year": 2025,
        ...     "commitment_status": "Validated - Near-term",
        ...     "baseline_year": 2020,
        ...     "baseline_emissions": {"scope1": 30000, "scope2": 20000, "scope3": 50000},
        ...     "actual_emissions": {"scope1": 25000, "scope2": 15000, "scope3": 42000},
        ...     "target_emissions": {"scope1": 24000, "scope2": 14000, "scope3": 40000},
        ...     "near_term_target": {"year": 2030, "reduction_pct": 46.2},
        ...     "long_term_target": {"year": 2050, "reduction_pct": 90.0},
        ...     "net_zero_year": 2050,
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    # Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render full SBTi progress report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_target_description(data),
            self._md_base_year(data),
            self._md_progress_table(data),
            self._md_variance(data),
            self._md_milestones(data),
            self._md_initiatives(data),
            self._md_projections(data),
            self._md_methodology(data),
            self._md_next_steps(data),
            self._md_xbrl(data),
            self._md_audit(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render full SBTi progress report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        parts = [
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_target_description(data),
            self._html_base_year(data),
            self._html_progress_table(data),
            self._html_variance(data),
            self._html_milestones(data),
            self._html_initiatives(data),
            self._html_projections(data),
            self._html_methodology(data),
            self._html_next_steps(data),
            self._html_xbrl(data),
            self._html_audit(data),
            self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>SBTi Progress Report - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        self.generated_at = utcnow()
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        baseline = data.get("baseline_emissions", {})
        total_actual = sum(float(v) for v in actual.values())
        total_target = sum(float(v) for v in target.values())
        total_baseline = sum(float(v) for v in baseline.values())
        var = _variance(total_actual, total_target)

        nt = data.get("near_term_target", {})
        lt = data.get("long_term_target", {})
        nt_target_em = total_baseline * (1 - float(nt.get("reduction_pct", 0)) / 100)
        lt_target_em = total_baseline * (1 - float(lt.get("reduction_pct", 0)) / 100)

        scope_progress = {}
        for scope in sorted(set(list(actual.keys()) + list(target.keys()))):
            a = float(actual.get(scope, 0))
            t = float(target.get(scope, 0))
            b = float(baseline.get(scope, 0))
            scope_progress[scope] = {
                "baseline": b, "actual": a, "target": t,
                "variance": _variance(a, t),
                "progress_pct": _progress_pct(b, a, t),
            }

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "commitment_status": data.get("commitment_status", ""),
            "baseline_year": data.get("baseline_year", ""),
            "net_zero_year": data.get("net_zero_year", ""),
            "near_term_target": {
                "year": nt.get("year", ""),
                "reduction_pct": str(nt.get("reduction_pct", 0)),
                "target_emissions": str(round(nt_target_em, 2)),
            },
            "long_term_target": {
                "year": lt.get("year", ""),
                "reduction_pct": str(lt.get("reduction_pct", 0)),
                "target_emissions": str(round(lt_target_em, 2)),
            },
            "total_baseline": str(total_baseline),
            "total_actual": str(total_actual),
            "total_target": str(total_target),
            "overall_variance": var,
            "near_term_progress_pct": str(_progress_pct(total_baseline, total_actual, nt_target_em)),
            "long_term_progress_pct": str(_progress_pct(total_baseline, total_actual, lt_target_em)),
            "scope_progress": scope_progress,
            "milestones": data.get("milestones", []),
            "initiatives": data.get("initiatives", []),
            "projections": data.get("projections", []),
            "recalculation_events": data.get("recalculation_events", []),
            "methodology": data.get("methodology", "Absolute Contraction Approach"),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready structured data."""
        return {
            "format": "pdf",
            "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {
                "title": f"SBTi Progress Report - {data.get('org_name', '')}",
                "author": "GreenLang PACK-030",
                "framework": "SBTi",
            },
        }

    # ------------------------------------------------------------------ #
    # Markdown sections
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# SBTi Annual Progress Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Commitment Status:** {data.get('commitment_status', '')}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-030 Net Zero Reporting Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        baseline = data.get("baseline_emissions", {})
        total_actual = sum(float(v) for v in actual.values())
        total_target = sum(float(v) for v in target.values())
        total_baseline = sum(float(v) for v in baseline.values())
        var = _variance(total_actual, total_target)
        reduction = float(_pct_change(total_actual, total_baseline)) if total_baseline else 0
        nt = data.get("near_term_target", {})
        nt_em = total_baseline * (1 - float(nt.get("reduction_pct", 0)) / 100)
        nt_progress = _progress_pct(total_baseline, total_actual, nt_em)
        lines = [
            "## 1. Executive Summary\n",
            "| KPI | Value |", "|-----|-------|",
            f"| Commitment Status | {data.get('commitment_status', '')} |",
            f"| Baseline Year | {data.get('baseline_year', '')} |",
            f"| Baseline Emissions | {_dec_comma(total_baseline, 0)} tCO2e |",
            f"| Current Year Emissions | {_dec_comma(total_actual, 0)} tCO2e |",
            f"| Annual Target Emissions | {_dec_comma(total_target, 0)} tCO2e |",
            f"| Variance | {'+' if var['diff'] > 0 else ''}{_dec_comma(var['diff'], 0)} tCO2e ({'+' if var['diff_pct'] > 0 else ''}{_dec(var['diff_pct'])}%) |",
            f"| RAG Status | **{var['rag']}** |",
            f"| Reduction from Baseline | {_dec(abs(reduction))}% |",
            f"| Near-Term Progress | {_dec(nt_progress)}% toward {nt.get('year', '')} target |",
            f"| Net-Zero Year | {data.get('net_zero_year', '')} |",
        ]
        return "\n".join(lines)

    def _md_target_description(self, data: Dict[str, Any]) -> str:
        nt = data.get("near_term_target", {})
        lt = data.get("long_term_target", {})
        lines = [
            "## 2. Target Description & Commitment Status\n",
            f"**Commitment Status:** {data.get('commitment_status', 'Not disclosed')}\n",
            "### Near-Term Target\n",
            "| Field | Value |", "|-------|-------|",
            f"| Target Year | {nt.get('year', '')} |",
            f"| Reduction (vs. baseline) | {_dec(nt.get('reduction_pct', 0))}% |",
            f"| Scope Coverage | {nt.get('scope_coverage', 'Scope 1 + 2 + 3')} |",
            f"| Methodology | {nt.get('methodology', 'Absolute Contraction')} |",
            f"| SBTi Validation Date | {nt.get('validation_date', 'N/A')} |",
            "",
            "### Long-Term Target\n",
            "| Field | Value |", "|-------|-------|",
            f"| Target Year | {lt.get('year', '')} |",
            f"| Reduction (vs. baseline) | {_dec(lt.get('reduction_pct', 0))}% |",
            f"| Net-Zero Commitment | {data.get('net_zero_year', '')} |",
            f"| Residual Emissions Plan | {lt.get('residual_plan', 'Neutralization via CDR')} |",
        ]
        return "\n".join(lines)

    def _md_base_year(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline_emissions", {})
        total_baseline = sum(float(v) for v in baseline.values())
        recalc = data.get("recalculation_events", [])
        lines = [
            "## 3. Base Year & Recalculation Events\n",
            f"**Base Year:** {data.get('baseline_year', '')}\n",
            "### Base Year Emissions\n",
            "| Scope | Emissions (tCO2e) | Share (%) |",
            "|-------|------------------:|----------:|",
        ]
        for scope in sorted(baseline.keys()):
            em = float(baseline[scope])
            share = (em / total_baseline * 100) if total_baseline else 0
            lines.append(f"| {scope.replace('_', ' ').title()} | {_dec_comma(em, 0)} | {_dec(share)}% |")
        lines.append(f"| **Total** | **{_dec_comma(total_baseline, 0)}** | **100.00%** |")
        if recalc:
            lines.extend(["", "### Recalculation Events\n",
                          "| # | Date | Reason | Impact (tCO2e) | Impact (%) |",
                          "|---|------|--------|---------------:|----------:|"])
            for i, r in enumerate(recalc, 1):
                lines.append(
                    f"| {i} | {r.get('date', '')} | {r.get('reason', '')} "
                    f"| {_dec_comma(r.get('impact_tco2e', 0), 0)} "
                    f"| {_dec(r.get('impact_pct', 0))}% |"
                )
        else:
            lines.append("\n*No recalculation events in this reporting period.*")
        return "\n".join(lines)

    def _md_progress_table(self, data: Dict[str, Any]) -> str:
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        baseline = data.get("baseline_emissions", {})
        scopes = sorted(set(list(actual.keys()) + list(target.keys()) + list(baseline.keys())))
        lines = [
            "## 4. Progress Table by Scope\n",
            "| Scope | Baseline (tCO2e) | Actual (tCO2e) | Target (tCO2e) | Reduction (%) | Progress (%) | Status |",
            "|-------|-----------------:|---------------:|---------------:|--------------:|-------------:|--------|",
        ]
        total_b, total_a, total_t = 0.0, 0.0, 0.0
        for scope in scopes:
            b = float(baseline.get(scope, 0))
            a = float(actual.get(scope, 0))
            t = float(target.get(scope, 0))
            total_b += b
            total_a += a
            total_t += t
            red = float(_pct_change(a, b)) if b else 0
            prog = _progress_pct(b, a, t)
            var = _variance(a, t)
            lines.append(
                f"| {scope.replace('_', ' ').title()} | {_dec_comma(b, 0)} | {_dec_comma(a, 0)} "
                f"| {_dec_comma(t, 0)} | {_dec(abs(red))}% | {_dec(prog)}% | **{var['rag']}** |"
            )
        total_red = float(_pct_change(total_a, total_b)) if total_b else 0
        total_prog = _progress_pct(total_b, total_a, total_t)
        total_var = _variance(total_a, total_t)
        lines.append(
            f"| **Total** | **{_dec_comma(total_b, 0)}** | **{_dec_comma(total_a, 0)}** "
            f"| **{_dec_comma(total_t, 0)}** | **{_dec(abs(total_red))}%** "
            f"| **{_dec(total_prog)}%** | **{total_var['rag']}** |"
        )
        return "\n".join(lines)

    def _md_variance(self, data: Dict[str, Any]) -> str:
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        total_actual = sum(float(v) for v in actual.values())
        total_target = sum(float(v) for v in target.values())
        var = _variance(total_actual, total_target)
        variance_factors = data.get("variance_factors", [])
        lines = [
            "## 5. Variance Explanation\n",
            f"**Overall Variance:** {'+' if var['diff'] > 0 else ''}{_dec_comma(var['diff'], 0)} tCO2e "
            f"({'+' if var['diff_pct'] > 0 else ''}{_dec(var['diff_pct'])}%)\n",
            f"**Status:** {'On Track' if var['on_track'] else 'Off Track'} ({var['rag']})\n",
        ]
        if variance_factors:
            lines.extend([
                "### Contributing Factors\n",
                "| # | Factor | Impact (tCO2e) | Impact (%) | Category |",
                "|---|--------|---------------:|----------:|----------|",
            ])
            for i, f in enumerate(variance_factors, 1):
                impact = float(f.get("impact_tco2e", 0))
                lines.append(
                    f"| {i} | {f.get('name', '')} | {'+' if impact > 0 else ''}{_dec_comma(impact, 0)} "
                    f"| {'+' if impact > 0 else ''}{_dec(f.get('impact_pct', 0))}% "
                    f"| {f.get('category', '')} |"
                )
        else:
            lines.append("*No variance factors provided.*")
        return "\n".join(lines)

    def _md_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        lines = [
            "## 6. Milestone Tracking\n",
            "| # | Milestone | Target Date | Status | Notes |",
            "|---|-----------|:-----------:|--------|-------|",
        ]
        for i, m in enumerate(milestones, 1):
            lines.append(
                f"| {i} | {m.get('name', '')} | {m.get('target_date', '')} "
                f"| {m.get('status', 'Planned')} | {m.get('notes', '')} |"
            )
        if not milestones:
            lines.append("| - | _No milestones defined_ | - | - | - |")
        return "\n".join(lines)

    def _md_initiatives(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        lines = [
            "## 7. Reduction Initiatives & Impact\n",
            "| # | Initiative | Scope | Status | Reduction (tCO2e) | Start | Completion |",
            "|---|-----------|-------|--------|------------------:|-------|------------|",
        ]
        total_reduction = 0.0
        for i, init in enumerate(initiatives, 1):
            red = float(init.get("reduction_tco2e", 0))
            total_reduction += red
            lines.append(
                f"| {i} | {init.get('name', '')} | {init.get('scope', '')} "
                f"| {init.get('status', 'Planned')} | {_dec_comma(red, 0)} "
                f"| {init.get('start_date', 'TBD')} | {init.get('completion_date', 'TBD')} |"
            )
        if not initiatives:
            lines.append("| - | _No initiatives reported_ | - | - | - | - | - |")
        lines.append(f"\n**Total Initiative Reduction:** {_dec_comma(total_reduction, 0)} tCO2e")
        return "\n".join(lines)

    def _md_projections(self, data: Dict[str, Any]) -> str:
        projections = data.get("projections", [])
        lines = [
            "## 8. Forward-Looking Projections\n",
            "| Year | Projected (tCO2e) | Target (tCO2e) | Gap (tCO2e) | Scenario | Confidence |",
            "|------|------------------:|---------------:|------------:|----------|------------|",
        ]
        for p in projections[:5]:
            gap = float(p.get("projected", 0)) - float(p.get("target", 0))
            lines.append(
                f"| {p.get('year', '')} | {_dec_comma(p.get('projected', 0), 0)} "
                f"| {_dec_comma(p.get('target', 0), 0)} "
                f"| {'+' if gap > 0 else ''}{_dec_comma(gap, 0)} "
                f"| {p.get('scenario', 'Central')} | {p.get('confidence', 'Medium')} |"
            )
        if not projections:
            lines.append("| - | _No projections provided_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        meth = data.get("methodology_detail", {})
        lines = [
            "## 9. Methodology & Boundary\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Target-Setting Methodology | {data.get('methodology', 'Absolute Contraction')} |",
            f"| Sector Classification | {meth.get('sector', 'Cross-sector')} |",
            f"| Consolidation Approach | {meth.get('consolidation', 'Operational control')} |",
            f"| Scope 1 Coverage | {meth.get('scope1_coverage', '100%')} |",
            f"| Scope 2 Coverage | {meth.get('scope2_coverage', '100%')} |",
            f"| Scope 3 Coverage | {meth.get('scope3_coverage', '67% (categories 1-8)')} |",
            f"| Emission Factor Source | {meth.get('ef_source', 'IPCC AR6, IEA 2024')} |",
            f"| GWP Values | {meth.get('gwp', 'IPCC AR6 100-year')} |",
            f"| Exclusions | {meth.get('exclusions', 'None')} |",
        ]
        return "\n".join(lines)

    def _md_next_steps(self, data: Dict[str, Any]) -> str:
        next_steps = data.get("next_steps", [])
        lines = [
            "## 10. Next Steps & Action Plan\n",
            "| # | Action | Owner | Deadline | Priority |",
            "|---|--------|-------|----------|----------|",
        ]
        for i, step in enumerate(next_steps, 1):
            lines.append(
                f"| {i} | {step.get('action', '')} | {step.get('owner', '')} "
                f"| {step.get('deadline', '')} | {step.get('priority', 'Medium')} |"
            )
        if not next_steps:
            lines.append("| - | _No action items defined_ | - | - | - |")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        total_actual = sum(float(v) for v in actual.values())
        total_target = sum(float(v) for v in target.values())
        var = _variance(total_actual, total_target)
        lines = [
            "## 11. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
            f"| Reporting Year | {XBRL_TAGS['reporting_year']} | {data.get('reporting_year', '')} |",
            f"| Commitment Status | {XBRL_TAGS['commitment_status']} | {data.get('commitment_status', '')} |",
            f"| Actual Emissions | {XBRL_TAGS['actual_emissions_total']} | {_dec_comma(total_actual, 0)} tCO2e |",
            f"| Target Emissions | {XBRL_TAGS['target_emissions_total']} | {_dec_comma(total_target, 0)} tCO2e |",
            f"| On Track | {XBRL_TAGS['on_track']} | {'Yes' if var['on_track'] else 'No'} |",
            f"| Net-Zero Year | {XBRL_TAGS['net_zero_year']} | {data.get('net_zero_year', '')} |",
            f"| Methodology | {XBRL_TAGS['methodology']} | {data.get('methodology', '')} |",
        ]
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
            f"| Pack | {_PACK_ID} |\n| Data Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n"
            f"*SBTi annual progress report - Science Based Targets initiative.*"
        )

    # ------------------------------------------------------------------ #
    # HTML sections
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #b3d4e6;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f0f7fb;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".rag-green{{background:#c8e6c9;color:#1b5e20;font-weight:700;padding:4px 12px;border-radius:4px;display:inline-block;}}"
            f".rag-amber{{background:#fff3e0;color:#e65100;font-weight:700;padding:4px 12px;border-radius:4px;display:inline-block;}}"
            f".rag-red{{background:#ffcdd2;color:#c62828;font-weight:700;padding:4px 12px;border-radius:4px;display:inline-block;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>SBTi Annual Progress Report</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Year:</strong> {data.get("reporting_year", "")} | '
            f'<strong>Status:</strong> {data.get("commitment_status", "")} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        baseline = data.get("baseline_emissions", {})
        total_actual = sum(float(v) for v in actual.values())
        total_target = sum(float(v) for v in target.values())
        total_baseline = sum(float(v) for v in baseline.values())
        var = _variance(total_actual, total_target)
        rag_cls = f"rag-{var['rag'].lower()}"
        nt = data.get("near_term_target", {})
        nt_em = total_baseline * (1 - float(nt.get("reduction_pct", 0)) / 100)
        nt_progress = _progress_pct(total_baseline, total_actual, nt_em)
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Baseline</div><div class="card-value">{_dec_comma(total_baseline, 0)}</div><div class="card-unit">tCO2e ({data.get("baseline_year", "")})</div></div>\n'
            f'<div class="card"><div class="card-label">Actual</div><div class="card-value">{_dec_comma(total_actual, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Target</div><div class="card-value">{_dec_comma(total_target, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Variance</div><div class="card-value">{_dec(var["diff_pct"])}%</div><div class="card-unit"><span class="{rag_cls}">{var["rag"]}</span></div></div>\n'
            f'<div class="card"><div class="card-label">NT Progress</div><div class="card-value">{_dec(nt_progress)}%</div><div class="card-unit">toward {nt.get("year", "")}</div></div>\n'
            f'<div class="card"><div class="card-label">Net-Zero</div><div class="card-value">{data.get("net_zero_year", "")}</div></div>\n'
            f'</div>'
        )

    def _html_target_description(self, data: Dict[str, Any]) -> str:
        nt = data.get("near_term_target", {})
        lt = data.get("long_term_target", {})
        return (
            f'<h2>2. Target Description</h2>\n'
            f'<h3>Near-Term Target</h3>\n<table>\n<tr><th>Field</th><th>Value</th></tr>\n'
            f'<tr><td>Year</td><td>{nt.get("year", "")}</td></tr>\n'
            f'<tr><td>Reduction</td><td>{_dec(nt.get("reduction_pct", 0))}%</td></tr>\n'
            f'<tr><td>Scope</td><td>{nt.get("scope_coverage", "Scope 1+2+3")}</td></tr>\n'
            f'<tr><td>Methodology</td><td>{nt.get("methodology", "ACA")}</td></tr>\n'
            f'</table>\n'
            f'<h3>Long-Term Target</h3>\n<table>\n<tr><th>Field</th><th>Value</th></tr>\n'
            f'<tr><td>Year</td><td>{lt.get("year", "")}</td></tr>\n'
            f'<tr><td>Reduction</td><td>{_dec(lt.get("reduction_pct", 0))}%</td></tr>\n'
            f'<tr><td>Net-Zero</td><td>{data.get("net_zero_year", "")}</td></tr>\n'
            f'</table>'
        )

    def _html_base_year(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline_emissions", {})
        total_baseline = sum(float(v) for v in baseline.values())
        rows = ""
        for scope in sorted(baseline.keys()):
            em = float(baseline[scope])
            share = (em / total_baseline * 100) if total_baseline else 0
            rows += f'<tr><td>{scope.replace("_", " ").title()}</td><td>{_dec_comma(em, 0)}</td><td>{_dec(share)}%</td></tr>\n'
        rows += f'<tr><td><strong>Total</strong></td><td><strong>{_dec_comma(total_baseline, 0)}</strong></td><td><strong>100%</strong></td></tr>\n'
        return (
            f'<h2>3. Base Year Emissions</h2>\n<table>\n'
            f'<tr><th>Scope</th><th>Emissions (tCO2e)</th><th>Share</th></tr>\n{rows}</table>'
        )

    def _html_progress_table(self, data: Dict[str, Any]) -> str:
        actual = data.get("actual_emissions", {})
        target = data.get("target_emissions", {})
        baseline = data.get("baseline_emissions", {})
        scopes = sorted(set(list(actual.keys()) + list(target.keys()) + list(baseline.keys())))
        rows = ""
        for scope in scopes:
            b = float(baseline.get(scope, 0))
            a = float(actual.get(scope, 0))
            t = float(target.get(scope, 0))
            prog = _progress_pct(b, a, t)
            var = _variance(a, t)
            rag_cls = f"rag-{var['rag'].lower()}"
            rows += (
                f'<tr><td>{scope.replace("_", " ").title()}</td><td>{_dec_comma(b, 0)}</td>'
                f'<td>{_dec_comma(a, 0)}</td><td>{_dec_comma(t, 0)}</td>'
                f'<td>{_dec(prog)}%</td><td><span class="{rag_cls}">{var["rag"]}</span></td></tr>\n'
            )
        return (
            f'<h2>4. Progress Table</h2>\n<table>\n'
            f'<tr><th>Scope</th><th>Baseline</th><th>Actual</th><th>Target</th><th>Progress</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_variance(self, data: Dict[str, Any]) -> str:
        variance_factors = data.get("variance_factors", [])
        rows = ""
        for f in variance_factors:
            impact = float(f.get("impact_tco2e", 0))
            rows += f'<tr><td>{f.get("name", "")}</td><td>{_dec_comma(impact, 0)}</td><td>{f.get("category", "")}</td></tr>\n'
        return (
            f'<h2>5. Variance Explanation</h2>\n<table>\n'
            f'<tr><th>Factor</th><th>Impact (tCO2e)</th><th>Category</th></tr>\n{rows}</table>'
        )

    def _html_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        rows = ""
        for i, m in enumerate(milestones, 1):
            rows += f'<tr><td>{i}</td><td>{m.get("name", "")}</td><td>{m.get("target_date", "")}</td><td>{m.get("status", "")}</td></tr>\n'
        return f'<h2>6. Milestones</h2>\n<table>\n<tr><th>#</th><th>Milestone</th><th>Date</th><th>Status</th></tr>\n{rows}</table>'

    def _html_initiatives(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        rows = ""
        for i, init in enumerate(initiatives, 1):
            rows += f'<tr><td>{i}</td><td>{init.get("name", "")}</td><td>{init.get("status", "")}</td><td>{_dec_comma(init.get("reduction_tco2e", 0), 0)}</td></tr>\n'
        return f'<h2>7. Initiatives</h2>\n<table>\n<tr><th>#</th><th>Initiative</th><th>Status</th><th>Reduction (tCO2e)</th></tr>\n{rows}</table>'

    def _html_projections(self, data: Dict[str, Any]) -> str:
        projections = data.get("projections", [])
        rows = ""
        for p in projections[:5]:
            rows += f'<tr><td>{p.get("year", "")}</td><td>{_dec_comma(p.get("projected", 0), 0)}</td><td>{_dec_comma(p.get("target", 0), 0)}</td><td>{p.get("confidence", "")}</td></tr>\n'
        return f'<h2>8. Projections</h2>\n<table>\n<tr><th>Year</th><th>Projected</th><th>Target</th><th>Confidence</th></tr>\n{rows}</table>'

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        meth = data.get("methodology_detail", {})
        return (
            f'<h2>9. Methodology</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Methodology</td><td>{data.get("methodology", "ACA")}</td></tr>\n'
            f'<tr><td>Sector</td><td>{meth.get("sector", "Cross-sector")}</td></tr>\n'
            f'<tr><td>Consolidation</td><td>{meth.get("consolidation", "Operational control")}</td></tr>\n'
            f'<tr><td>EF Source</td><td>{meth.get("ef_source", "IPCC AR6, IEA")}</td></tr>\n'
            f'</table>'
        )

    def _html_next_steps(self, data: Dict[str, Any]) -> str:
        next_steps = data.get("next_steps", [])
        rows = ""
        for i, step in enumerate(next_steps, 1):
            rows += f'<tr><td>{i}</td><td>{step.get("action", "")}</td><td>{step.get("owner", "")}</td><td>{step.get("deadline", "")}</td><td>{step.get("priority", "")}</td></tr>\n'
        return f'<h2>10. Next Steps</h2>\n<table>\n<tr><th>#</th><th>Action</th><th>Owner</th><th>Deadline</th><th>Priority</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_", " ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>11. XBRL Tags</h2>\n<table>\n<tr><th>Data Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            f'<h2>12. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n'
            f'<tr><td>Generated</td><td>{ts}</td></tr>\n'
            f'<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n'
            f'<tr><td>Version</td><td>{_MODULE_VERSION}</td></tr>\n'
            f'<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - SBTi Progress Report</div>'
