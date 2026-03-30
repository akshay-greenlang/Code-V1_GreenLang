# -*- coding: utf-8 -*-
"""
InterimTargetsSummaryTemplate - Interim targets summary for PACK-029.

Renders a comprehensive interim targets summary report with 5-year and
10-year interim targets by scope, baseline-to-net-zero pathway visualization,
annual emissions trajectory, cumulative carbon budget analysis, and SBTi
1.5C validation (42% near-term check). Multi-format output (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary
    2.  Baseline Profile
    3.  Near-Term Interim Targets (5-Year)
    4.  Medium-Term Interim Targets (10-Year)
    5.  Long-Term & Net-Zero Targets
    6.  Scope-Level Target Breakdown
    7.  Annual Pathway Chart Data
    8.  Cumulative Emissions & Carbon Budget
    9.  SBTi 1.5C Validation Status
    10. Target Summary Dashboard
    11. XBRL Tagging Summary
    12. Audit Trail & Provenance

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"
_TEMPLATE_ID = "interim_targets_summary"

# ---------------------------------------------------------------------------
# Colour Palette (Interim Targets Theme)
# ---------------------------------------------------------------------------
_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_WARN = "#ef6c00"
_DANGER = "#c62828"
_SUCCESS = "#2e7d32"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SBTI_NEAR_TERM_MIN_REDUCTION = 42.0  # 42% for 1.5C alignment (near-term)
SBTI_NEAR_TERM_YEARS = 10  # Near-term: ~5-10 years
SBTI_LONG_TERM_MIN_REDUCTION = 90.0  # 90% for net-zero

SCOPE_DEFINITIONS = [
    {"id": "scope1", "name": "Scope 1 (Direct)", "description": "Direct GHG emissions"},
    {"id": "scope2_location", "name": "Scope 2 (Location-Based)", "description": "Indirect - purchased electricity (location)"},
    {"id": "scope2_market", "name": "Scope 2 (Market-Based)", "description": "Indirect - purchased electricity (market)"},
    {"id": "scope3", "name": "Scope 3 (Value Chain)", "description": "All other indirect emissions"},
]

XBRL_TAGS: Dict[str, str] = {
    "baseline_year": "gl:InterimTargetBaselineYear",
    "baseline_emissions": "gl:InterimTargetBaselineEmissions",
    "near_term_target_year": "gl:NearTermTargetYear",
    "near_term_target_pct": "gl:NearTermTargetReductionPct",
    "medium_term_target_year": "gl:MediumTermTargetYear",
    "medium_term_target_pct": "gl:MediumTermTargetReductionPct",
    "net_zero_target_year": "gl:NetZeroTargetYear",
    "carbon_budget_remaining": "gl:CarbonBudgetRemaining",
    "sbti_validation": "gl:SBTi15CValidationStatus",
    "annual_reduction_rate": "gl:AnnualLinearReductionRate",
}

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

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

def _pct_reduction(current: Any, baseline: Any) -> Decimal:
    c = Decimal(str(current))
    b = Decimal(str(baseline))
    if b == 0:
        return Decimal("0.00")
    return ((b - c) / b * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def _linear_target(baseline: float, reduction_pct: float) -> float:
    """Calculate target emissions given baseline and reduction percentage."""
    return baseline * (1.0 - reduction_pct / 100.0)

def _annual_linear_rate(total_reduction_pct: float, years: int) -> float:
    """Calculate annualized linear reduction rate."""
    if years <= 0:
        return 0.0
    return total_reduction_pct / years

def _cumulative_emissions(
    baseline: float,
    target: float,
    base_year: int,
    target_year: int,
) -> float:
    """Calculate cumulative emissions assuming linear decline."""
    years = target_year - base_year
    if years <= 0:
        return 0.0
    return (baseline + target) / 2.0 * years

def _rag_status(value: float, threshold_green: float, threshold_amber: float) -> str:
    if value >= threshold_green:
        return "GREEN"
    elif value >= threshold_amber:
        return "AMBER"
    else:
        return "RED"

def _rag_color(status: str) -> str:
    return {"GREEN": _SUCCESS, "AMBER": _WARN, "RED": _DANGER}.get(status, "#999")

def _sbti_15c_check(reduction_pct: float, years: int) -> Dict[str, Any]:
    """Validate against SBTi 1.5C near-term criteria (42% in ~10 years)."""
    annual_rate = _annual_linear_rate(reduction_pct, years)
    min_annual = _annual_linear_rate(SBTI_NEAR_TERM_MIN_REDUCTION, SBTI_NEAR_TERM_YEARS)
    passed = annual_rate >= min_annual
    return {
        "passed": passed,
        "reduction_pct": round(reduction_pct, 2),
        "annual_rate": round(annual_rate, 2),
        "min_annual_rate": round(min_annual, 2),
        "min_reduction_pct": SBTI_NEAR_TERM_MIN_REDUCTION,
        "status": "ALIGNED" if passed else "NOT ALIGNED",
    }

# ---------------------------------------------------------------------------
# Template Class
# ---------------------------------------------------------------------------
class InterimTargetsSummaryTemplate:
    """
    Interim targets summary report template for PACK-029.

    Renders 5-year and 10-year interim targets by scope with baseline,
    long-term, and net-zero target visualization. Includes annual pathway
    charting data, cumulative carbon budget, and SBTi 1.5C validation.
    Supports Markdown, HTML, JSON, and PDF output.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.

    Example:
        >>> template = InterimTargetsSummaryTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "baseline_year": 2022,
        ...     "baseline_emissions": 100000,
        ...     "near_term_year": 2030,
        ...     "near_term_target_pct": 46.2,
        ...     "medium_term_year": 2035,
        ...     "medium_term_target_pct": 65.0,
        ...     "net_zero_year": 2050,
        ...     "net_zero_target_pct": 95.0,
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render full report as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_baseline(data),
            self._md_near_term(data),
            self._md_medium_term(data),
            self._md_long_term(data),
            self._md_scope_breakdown(data),
            self._md_annual_pathway(data),
            self._md_carbon_budget(data),
            self._md_sbti_validation(data),
            self._md_target_dashboard(data),
            self._md_xbrl(data),
            self._md_audit(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render full report as HTML with inline CSS."""
        self.generated_at = utcnow()
        css = self._css()
        body_parts = [
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_baseline(data),
            self._html_near_term(data),
            self._html_medium_term(data),
            self._html_long_term(data),
            self._html_scope_breakdown(data),
            self._html_annual_pathway(data),
            self._html_carbon_budget(data),
            self._html_sbti_validation(data),
            self._html_target_dashboard(data),
            self._html_xbrl(data),
            self._html_audit(data),
            self._html_footer(data),
        ]
        body = "\n".join(body_parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Interim Targets Summary - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured JSON."""
        self.generated_at = utcnow()
        baseline_yr = int(data.get("baseline_year", 2022))
        baseline_em = float(data.get("baseline_emissions", 0))
        nt_year = int(data.get("near_term_year", 2030))
        nt_pct = float(data.get("near_term_target_pct", 42))
        mt_year = int(data.get("medium_term_year", 2035))
        mt_pct = float(data.get("medium_term_target_pct", 65))
        nz_year = int(data.get("net_zero_year", 2050))
        nz_pct = float(data.get("net_zero_target_pct", 95))

        nt_target = _linear_target(baseline_em, nt_pct)
        mt_target = _linear_target(baseline_em, mt_pct)
        nz_target = _linear_target(baseline_em, nz_pct)

        sbti_check = _sbti_15c_check(nt_pct, nt_year - baseline_yr)

        annual_pathway = self._build_annual_pathway(
            baseline_em, baseline_yr, nt_target, nt_year,
            mt_target, mt_year, nz_target, nz_year,
        )

        cumulative = _cumulative_emissions(baseline_em, nz_target, baseline_yr, nz_year)
        budget = float(data.get("carbon_budget_total", 0))

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "baseline": {
                "year": baseline_yr,
                "total_emissions_tco2e": str(baseline_em),
                "scope_breakdown": data.get("baseline_scope_breakdown", {}),
            },
            "near_term_target": {
                "year": nt_year,
                "reduction_pct": str(nt_pct),
                "target_emissions_tco2e": str(round(nt_target, 2)),
                "annual_rate": str(round(_annual_linear_rate(nt_pct, nt_year - baseline_yr), 2)),
            },
            "medium_term_target": {
                "year": mt_year,
                "reduction_pct": str(mt_pct),
                "target_emissions_tco2e": str(round(mt_target, 2)),
            },
            "net_zero_target": {
                "year": nz_year,
                "reduction_pct": str(nz_pct),
                "target_emissions_tco2e": str(round(nz_target, 2)),
                "residual_emissions_tco2e": str(round(nz_target, 2)),
            },
            "sbti_validation": sbti_check,
            "annual_pathway": annual_pathway,
            "carbon_budget": {
                "cumulative_emissions_tco2e": str(round(cumulative, 2)),
                "budget_total_tco2e": str(budget),
                "budget_remaining_tco2e": str(round(max(0, budget - cumulative), 2)),
                "within_budget": cumulative <= budget if budget > 0 else None,
            },
            "scope_targets": data.get("scope_targets", {}),
            "xbrl_tags": {tag: XBRL_TAGS[tag] for tag in XBRL_TAGS},
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready structured data."""
        json_data = self.render_json(data)
        html_content = self.render_html(data)
        return {
            "format": "pdf",
            "html_content": html_content,
            "structured_data": json_data,
            "metadata": {
                "title": f"Interim Targets Summary - {data.get('org_name', '')}",
                "author": "GreenLang PACK-029",
                "subject": "Interim Decarbonization Targets",
                "creator": f"GreenLang v{_MODULE_VERSION}",
            },
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _build_annual_pathway(
        self,
        baseline: float,
        base_year: int,
        nt_target: float,
        nt_year: int,
        mt_target: float,
        mt_year: int,
        nz_target: float,
        nz_year: int,
    ) -> List[Dict[str, Any]]:
        """Build year-by-year pathway from baseline through milestones to net-zero."""
        pathway: List[Dict[str, Any]] = []
        # Phase 1: baseline -> near-term
        for yr in range(base_year, nt_year + 1):
            t = (yr - base_year) / max(1, nt_year - base_year)
            emissions = baseline + (nt_target - baseline) * t
            reduction = ((baseline - emissions) / baseline * 100) if baseline > 0 else 0
            pathway.append({
                "year": yr,
                "target_emissions": round(emissions, 2),
                "reduction_from_baseline_pct": round(reduction, 2),
                "phase": "near_term",
            })
        # Phase 2: near-term -> medium-term
        for yr in range(nt_year + 1, mt_year + 1):
            t = (yr - nt_year) / max(1, mt_year - nt_year)
            emissions = nt_target + (mt_target - nt_target) * t
            reduction = ((baseline - emissions) / baseline * 100) if baseline > 0 else 0
            pathway.append({
                "year": yr,
                "target_emissions": round(emissions, 2),
                "reduction_from_baseline_pct": round(reduction, 2),
                "phase": "medium_term",
            })
        # Phase 3: medium-term -> net-zero
        for yr in range(mt_year + 1, nz_year + 1):
            t = (yr - mt_year) / max(1, nz_year - mt_year)
            emissions = mt_target + (nz_target - mt_target) * t
            reduction = ((baseline - emissions) / baseline * 100) if baseline > 0 else 0
            pathway.append({
                "year": yr,
                "target_emissions": round(emissions, 2),
                "reduction_from_baseline_pct": round(reduction, 2),
                "phase": "long_term",
            })
        return pathway

    # ------------------------------------------------------------------
    # Markdown Sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Interim Targets Summary Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-029 Interim Targets Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        baseline_yr = int(data.get("baseline_year", 2022))
        nt_year = int(data.get("near_term_year", 2030))
        nt_pct = float(data.get("near_term_target_pct", 42))
        mt_year = int(data.get("medium_term_year", 2035))
        mt_pct = float(data.get("medium_term_target_pct", 65))
        nz_year = int(data.get("net_zero_year", 2050))
        nz_pct = float(data.get("net_zero_target_pct", 95))
        nt_target = _linear_target(baseline_em, nt_pct)
        sbti = _sbti_15c_check(nt_pct, nt_year - baseline_yr)
        lines = [
            "## 1. Executive Summary\n",
            f"This report presents the interim decarbonization targets for "
            f"**{data.get('org_name', '')}**, covering near-term ({nt_year}), "
            f"medium-term ({mt_year}), and net-zero ({nz_year}) milestones.\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Baseline Year | {baseline_yr} |",
            f"| Baseline Emissions | {_dec_comma(baseline_em, 0)} tCO2e |",
            f"| Near-Term Target ({nt_year}) | -{_dec(nt_pct)}% ({_dec_comma(nt_target, 0)} tCO2e) |",
            f"| Medium-Term Target ({mt_year}) | -{_dec(mt_pct)}% |",
            f"| Net-Zero Target ({nz_year}) | -{_dec(nz_pct)}% |",
            f"| SBTi 1.5C Alignment | **{sbti['status']}** |",
            f"| Annual Reduction Rate | {_dec(sbti['annual_rate'])}% per year |",
        ]
        return "\n".join(lines)

    def _md_baseline(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        baseline_yr = int(data.get("baseline_year", 2022))
        breakdown = data.get("baseline_scope_breakdown", {})
        lines = [
            "## 2. Baseline Profile\n",
            f"**Baseline Year:** {baseline_yr}  \n"
            f"**Total Emissions:** {_dec_comma(baseline_em, 0)} tCO2e\n",
            "| Scope | Emissions (tCO2e) | Share (%) |",
            "|-------|------------------:|----------:|",
        ]
        for scope_def in SCOPE_DEFINITIONS:
            val = float(breakdown.get(scope_def["id"], 0))
            share = (val / baseline_em * 100) if baseline_em > 0 else 0
            lines.append(f"| {scope_def['name']} | {_dec_comma(val, 0)} | {_dec(share)}% |")
        lines.append(f"| **Total** | **{_dec_comma(baseline_em, 0)}** | **100.00%** |")
        return "\n".join(lines)

    def _md_near_term(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        baseline_yr = int(data.get("baseline_year", 2022))
        nt_year = int(data.get("near_term_year", 2030))
        nt_pct = float(data.get("near_term_target_pct", 42))
        nt_target = _linear_target(baseline_em, nt_pct)
        years = nt_year - baseline_yr
        annual = _annual_linear_rate(nt_pct, years)
        # 5-year interim milestones
        mid_year = baseline_yr + years // 2
        mid_pct = nt_pct / 2
        mid_target = _linear_target(baseline_em, mid_pct)
        lines = [
            "## 3. Near-Term Interim Targets (5-Year Milestones)\n",
            "| Milestone | Year | Reduction (%) | Target Emissions (tCO2e) |",
            "|-----------|------|:-------------:|-------------------------:|",
            f"| Baseline | {baseline_yr} | 0.0% | {_dec_comma(baseline_em, 0)} |",
            f"| Mid-Point | {mid_year} | {_dec(mid_pct)}% | {_dec_comma(mid_target, 0)} |",
            f"| Near-Term | {nt_year} | {_dec(nt_pct)}% | {_dec_comma(nt_target, 0)} |",
            f"\n**Annual Linear Reduction Rate:** {_dec(annual)}% per year  \n"
            f"**Timeframe:** {years} years ({baseline_yr}-{nt_year})",
        ]
        return "\n".join(lines)

    def _md_medium_term(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        mt_year = int(data.get("medium_term_year", 2035))
        mt_pct = float(data.get("medium_term_target_pct", 65))
        mt_target = _linear_target(baseline_em, mt_pct)
        lines = [
            "## 4. Medium-Term Interim Targets (10-Year)\n",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Target Year | {mt_year} |",
            f"| Reduction from Baseline | {_dec(mt_pct)}% |",
            f"| Target Emissions | {_dec_comma(mt_target, 0)} tCO2e |",
            f"| Baseline Emissions | {_dec_comma(baseline_em, 0)} tCO2e |",
            f"| Absolute Reduction | {_dec_comma(baseline_em - mt_target, 0)} tCO2e |",
        ]
        return "\n".join(lines)

    def _md_long_term(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        nz_year = int(data.get("net_zero_year", 2050))
        nz_pct = float(data.get("net_zero_target_pct", 95))
        nz_target = _linear_target(baseline_em, nz_pct)
        residual = nz_target
        neutralization = data.get("neutralization_strategy", "Permanent carbon removal (CDR)")
        lines = [
            "## 5. Long-Term & Net-Zero Targets\n",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Net-Zero Target Year | {nz_year} |",
            f"| Long-Term Reduction | {_dec(nz_pct)}% |",
            f"| Residual Emissions | {_dec_comma(residual, 0)} tCO2e |",
            f"| Residual as % of Baseline | {_dec(100 - nz_pct)}% |",
            f"| SBTi Residual Limit | <=10% |",
            f"| Residual Compliant | {'Yes' if nz_pct >= 90 else 'No'} |",
            f"| Neutralization Strategy | {neutralization} |",
        ]
        return "\n".join(lines)

    def _md_scope_breakdown(self, data: Dict[str, Any]) -> str:
        scope_targets = data.get("scope_targets", {})
        baseline_em = float(data.get("baseline_emissions", 0))
        nt_year = int(data.get("near_term_year", 2030))
        nz_year = int(data.get("net_zero_year", 2050))
        lines = [
            "## 6. Scope-Level Target Breakdown\n",
            f"| Scope | Baseline (tCO2e) | {nt_year} Target | {nt_year} Emissions | {nz_year} Target |",
            f"|-------|------------------:|-----------------:|--------------------:|-----------------:|",
        ]
        for scope_def in SCOPE_DEFINITIONS:
            st = scope_targets.get(scope_def["id"], {})
            base = float(st.get("baseline", 0))
            nt_red = float(st.get("near_term_reduction_pct", 0))
            nz_red = float(st.get("net_zero_reduction_pct", 0))
            nt_em = _linear_target(base, nt_red)
            lines.append(
                f"| {scope_def['name']} | {_dec_comma(base, 0)} | -{_dec(nt_red)}% "
                f"| {_dec_comma(nt_em, 0)} | -{_dec(nz_red)}% |"
            )
        return "\n".join(lines)

    def _md_annual_pathway(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        baseline_yr = int(data.get("baseline_year", 2022))
        nt_year = int(data.get("near_term_year", 2030))
        nt_pct = float(data.get("near_term_target_pct", 42))
        mt_year = int(data.get("medium_term_year", 2035))
        mt_pct = float(data.get("medium_term_target_pct", 65))
        nz_year = int(data.get("net_zero_year", 2050))
        nz_pct = float(data.get("net_zero_target_pct", 95))
        pathway = self._build_annual_pathway(
            baseline_em, baseline_yr,
            _linear_target(baseline_em, nt_pct), nt_year,
            _linear_target(baseline_em, mt_pct), mt_year,
            _linear_target(baseline_em, nz_pct), nz_year,
        )
        lines = [
            "## 7. Annual Pathway Chart Data\n",
            "| Year | Target Emissions (tCO2e) | Reduction (%) | Phase |",
            "|------|-------------------------:|:-------------:|-------|",
        ]
        # Show every year for short pathways, every 2 for medium, every 5 for long
        step = 1 if len(pathway) <= 15 else (2 if len(pathway) <= 25 else 5)
        for i, p in enumerate(pathway):
            if i % step == 0 or p["year"] in (baseline_yr, nt_year, mt_year, nz_year):
                lines.append(
                    f"| {p['year']} | {_dec_comma(p['target_emissions'], 0)} "
                    f"| {_dec(p['reduction_from_baseline_pct'])}% | {p['phase'].replace('_', ' ').title()} |"
                )
        return "\n".join(lines)

    def _md_carbon_budget(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        baseline_yr = int(data.get("baseline_year", 2022))
        nz_year = int(data.get("net_zero_year", 2050))
        nz_pct = float(data.get("net_zero_target_pct", 95))
        nz_target = _linear_target(baseline_em, nz_pct)
        cumulative = _cumulative_emissions(baseline_em, nz_target, baseline_yr, nz_year)
        budget = float(data.get("carbon_budget_total", 0))
        remaining = max(0, budget - cumulative) if budget > 0 else 0
        within = cumulative <= budget if budget > 0 else None
        lines = [
            "## 8. Cumulative Emissions & Carbon Budget\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Pathway Period | {baseline_yr}-{nz_year} |",
            f"| Cumulative Emissions (Linear) | {_dec_comma(cumulative, 0)} tCO2e |",
            f"| Organizational Carbon Budget | {_dec_comma(budget, 0) if budget > 0 else 'Not Defined'} tCO2e |",
            f"| Budget Remaining | {_dec_comma(remaining, 0) if budget > 0 else 'N/A'} tCO2e |",
            f"| Within Budget | {'Yes' if within else ('No' if within is False else 'N/A')} |",
        ]
        return "\n".join(lines)

    def _md_sbti_validation(self, data: Dict[str, Any]) -> str:
        baseline_yr = int(data.get("baseline_year", 2022))
        nt_year = int(data.get("near_term_year", 2030))
        nt_pct = float(data.get("near_term_target_pct", 42))
        sbti = _sbti_15c_check(nt_pct, nt_year - baseline_yr)
        lines = [
            "## 9. SBTi 1.5C Validation Status\n",
            "| Criterion | Value | Requirement | Status |",
            "|-----------|-------|-------------|--------|",
            f"| Near-Term Reduction | {_dec(sbti['reduction_pct'])}% | >={_dec(sbti['min_reduction_pct'])}% | **{sbti['status']}** |",
            f"| Annual Linear Rate | {_dec(sbti['annual_rate'])}%/yr | >={_dec(sbti['min_annual_rate'])}%/yr | **{'PASS' if sbti['passed'] else 'FAIL'}** |",
            f"| Target Timeframe | {nt_year - baseline_yr} years | 5-10 years | **{'PASS' if 5 <= (nt_year - baseline_yr) <= 10 else 'REVIEW'}** |",
        ]
        if not sbti["passed"]:
            gap = sbti["min_annual_rate"] - sbti["annual_rate"]
            lines.append(
                f"\n**Gap to 1.5C Alignment:** {_dec(gap)}% per year additional reduction required"
            )
        return "\n".join(lines)

    def _md_target_dashboard(self, data: Dict[str, Any]) -> str:
        current_em = float(data.get("current_emissions", 0))
        baseline_em = float(data.get("baseline_emissions", 0))
        nt_pct = float(data.get("near_term_target_pct", 42))
        nt_target = _linear_target(baseline_em, nt_pct)
        if baseline_em > 0 and current_em > 0:
            current_reduction = float(_pct_reduction(current_em, baseline_em))
            progress_to_nt = (current_reduction / nt_pct * 100) if nt_pct > 0 else 0
            on_track = current_reduction >= (nt_pct * 0.9)  # within 10% of linear path
        else:
            current_reduction = 0
            progress_to_nt = 0
            on_track = None
        lines = [
            "## 10. Target Summary Dashboard\n",
            "| KPI | Value |",
            "|-----|-------|",
            f"| Current Emissions | {_dec_comma(current_em, 0)} tCO2e |",
            f"| Reduction Achieved | {_dec(current_reduction)}% |",
            f"| Progress to Near-Term Target | {_dec(progress_to_nt)}% |",
            f"| On Track | {'Yes' if on_track else ('No' if on_track is False else 'N/A')} |",
        ]
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 11. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |",
            "|------------|----------|-------|",
        ]
        tag_vals = {
            "baseline_year": data.get("baseline_year", ""),
            "baseline_emissions": _dec_comma(data.get("baseline_emissions", 0), 0),
            "near_term_target_year": data.get("near_term_year", ""),
            "near_term_target_pct": f"{_dec(data.get('near_term_target_pct', 0))}%",
            "medium_term_target_year": data.get("medium_term_year", ""),
            "medium_term_target_pct": f"{_dec(data.get('medium_term_target_pct', 0))}%",
            "net_zero_target_year": data.get("net_zero_year", ""),
        }
        for key, tag in XBRL_TAGS.items():
            val = tag_vals.get(key, "")
            if val:
                lines.append(f"| {key.replace('_', ' ').title()} | {tag} | {val} |")
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        report_id = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        data_hash = _compute_hash(data)
        return (
            "## 12. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n"
            f"|-----------|-------|\n"
            f"| Report ID | `{report_id}` |\n"
            f"| Generated At | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n"
            f"| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n"
            f"| Input Data Hash | `{data_hash[:16]}...` |\n"
            f"| Calculation Engine | Deterministic (zero-hallucination) |\n"
            f"| Reproducible | Yes (SHA-256 provenance chain) |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-029 Interim Targets Pack on {ts}*  \n"
            f"*Interim targets summary with SBTi 1.5C validation.*  \n"
            f"*All calculations deterministic - zero LLM in computation path.*"
        )

    # ------------------------------------------------------------------
    # HTML Sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            f"background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f9fbe7;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            f"gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;"
            f"padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".rag-green{{background:#c8e6c9;color:#1b5e20;font-weight:700;padding:4px 12px;"
            f"border-radius:4px;display:inline-block;}}"
            f".rag-amber{{background:#fff3e0;color:#e65100;font-weight:700;padding:4px 12px;"
            f"border-radius:4px;display:inline-block;}}"
            f".rag-red{{background:#ffcdd2;color:#c62828;font-weight:700;padding:4px 12px;"
            f"border-radius:4px;display:inline-block;}}"
            f".pass{{color:{_SUCCESS};font-weight:700;}}"
            f".fail{{color:{_DANGER};font-weight:700;}}"
            f".pending{{color:{_WARN};font-style:italic;}}"
            f".progress-bar{{height:12px;border-radius:6px;background:#e0e0e0;margin:4px 0;}}"
            f".progress-fill{{height:12px;border-radius:6px;background:linear-gradient(90deg,{_ACCENT},{_SUCCESS});}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};"
            f"color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Interim Targets Summary Report</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        nt_year = int(data.get("near_term_year", 2030))
        nt_pct = float(data.get("near_term_target_pct", 42))
        nz_year = int(data.get("net_zero_year", 2050))
        nz_pct = float(data.get("net_zero_target_pct", 95))
        nt_target = _linear_target(baseline_em, nt_pct)
        baseline_yr = int(data.get("baseline_year", 2022))
        sbti = _sbti_15c_check(nt_pct, nt_year - baseline_yr)
        rag_cls = "rag-green" if sbti["passed"] else "rag-red"
        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Baseline ({data.get("baseline_year", "")})</div>'
            f'<div class="card-value">{_dec_comma(baseline_em, 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Near-Term ({nt_year})</div>'
            f'<div class="card-value">-{_dec(nt_pct)}%</div>'
            f'<div class="card-unit">{_dec_comma(nt_target, 0)} tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Net-Zero ({nz_year})</div>'
            f'<div class="card-value">-{_dec(nz_pct)}%</div>'
            f'<div class="card-unit">residual only</div></div>\n'
            f'  <div class="card"><div class="card-label">SBTi 1.5C</div>'
            f'<div class="card-value"><span class="{rag_cls}">{sbti["status"]}</span></div>'
            f'<div class="card-unit">{_dec(sbti["annual_rate"])}%/yr</div></div>\n'
            f'</div>'
        )

    def _html_baseline(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        breakdown = data.get("baseline_scope_breakdown", {})
        rows = ""
        for scope_def in SCOPE_DEFINITIONS:
            val = float(breakdown.get(scope_def["id"], 0))
            share = (val / baseline_em * 100) if baseline_em > 0 else 0
            rows += f'<tr><td>{scope_def["name"]}</td><td>{_dec_comma(val, 0)}</td><td>{_dec(share)}%</td></tr>\n'
        return (
            f'<h2>2. Baseline Profile</h2>\n'
            f'<table>\n<tr><th>Scope</th><th>Emissions (tCO2e)</th><th>Share</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_near_term(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        nt_pct = float(data.get("near_term_target_pct", 42))
        nt_year = int(data.get("near_term_year", 2030))
        nt_target = _linear_target(baseline_em, nt_pct)
        return (
            f'<h2>3. Near-Term Targets</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Target Year</div>'
            f'<div class="card-value">{nt_year}</div></div>\n'
            f'  <div class="card"><div class="card-label">Reduction</div>'
            f'<div class="card-value">-{_dec(nt_pct)}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Target Emissions</div>'
            f'<div class="card-value">{_dec_comma(nt_target, 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_medium_term(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        mt_pct = float(data.get("medium_term_target_pct", 65))
        mt_year = int(data.get("medium_term_year", 2035))
        mt_target = _linear_target(baseline_em, mt_pct)
        return (
            f'<h2>4. Medium-Term Targets</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Target Year</div>'
            f'<div class="card-value">{mt_year}</div></div>\n'
            f'  <div class="card"><div class="card-label">Reduction</div>'
            f'<div class="card-value">-{_dec(mt_pct)}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Target Emissions</div>'
            f'<div class="card-value">{_dec_comma(mt_target, 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_long_term(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        nz_pct = float(data.get("net_zero_target_pct", 95))
        nz_year = int(data.get("net_zero_year", 2050))
        nz_target = _linear_target(baseline_em, nz_pct)
        compliant = nz_pct >= 90
        return (
            f'<h2>5. Long-Term & Net-Zero</h2>\n'
            f'<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Net-Zero Year</td><td>{nz_year}</td></tr>\n'
            f'<tr><td>Reduction</td><td>{_dec(nz_pct)}%</td></tr>\n'
            f'<tr><td>Residual</td><td>{_dec_comma(nz_target, 0)} tCO2e ({_dec(100 - nz_pct)}%)</td></tr>\n'
            f'<tr><td>SBTi Compliant (&lt;=10%)</td><td class="{"pass" if compliant else "fail"}">'
            f'{"Yes" if compliant else "No"}</td></tr>\n'
            f'</table>'
        )

    def _html_scope_breakdown(self, data: Dict[str, Any]) -> str:
        scope_targets = data.get("scope_targets", {})
        nt_year = int(data.get("near_term_year", 2030))
        rows = ""
        for scope_def in SCOPE_DEFINITIONS:
            st = scope_targets.get(scope_def["id"], {})
            base = float(st.get("baseline", 0))
            nt_red = float(st.get("near_term_reduction_pct", 0))
            nt_em = _linear_target(base, nt_red)
            rows += (
                f'<tr><td>{scope_def["name"]}</td><td>{_dec_comma(base, 0)}</td>'
                f'<td>-{_dec(nt_red)}%</td><td>{_dec_comma(nt_em, 0)}</td></tr>\n'
            )
        return (
            f'<h2>6. Scope-Level Targets</h2>\n'
            f'<table>\n<tr><th>Scope</th><th>Baseline (tCO2e)</th>'
            f'<th>{nt_year} Reduction</th><th>{nt_year} Emissions</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_annual_pathway(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        baseline_yr = int(data.get("baseline_year", 2022))
        nt_pct = float(data.get("near_term_target_pct", 42))
        nt_year = int(data.get("near_term_year", 2030))
        mt_pct = float(data.get("medium_term_target_pct", 65))
        mt_year = int(data.get("medium_term_year", 2035))
        nz_pct = float(data.get("net_zero_target_pct", 95))
        nz_year = int(data.get("net_zero_year", 2050))
        pathway = self._build_annual_pathway(
            baseline_em, baseline_yr,
            _linear_target(baseline_em, nt_pct), nt_year,
            _linear_target(baseline_em, mt_pct), mt_year,
            _linear_target(baseline_em, nz_pct), nz_year,
        )
        rows = ""
        step = 1 if len(pathway) <= 15 else (2 if len(pathway) <= 25 else 5)
        for i, p in enumerate(pathway):
            if i % step == 0 or p["year"] in (baseline_yr, nt_year, mt_year, nz_year):
                bar_w = min(100, max(0, p["reduction_from_baseline_pct"]))
                rows += (
                    f'<tr><td>{p["year"]}</td><td>{_dec_comma(p["target_emissions"], 0)}</td>'
                    f'<td>{_dec(p["reduction_from_baseline_pct"])}%</td>'
                    f'<td><div class="progress-bar"><div class="progress-fill" '
                    f'style="width:{bar_w}%"></div></div></td></tr>\n'
                )
        return (
            f'<h2>7. Annual Pathway</h2>\n'
            f'<table>\n<tr><th>Year</th><th>Target (tCO2e)</th>'
            f'<th>Reduction</th><th>Progress</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_carbon_budget(self, data: Dict[str, Any]) -> str:
        baseline_em = float(data.get("baseline_emissions", 0))
        baseline_yr = int(data.get("baseline_year", 2022))
        nz_year = int(data.get("net_zero_year", 2050))
        nz_pct = float(data.get("net_zero_target_pct", 95))
        nz_target = _linear_target(baseline_em, nz_pct)
        cumulative = _cumulative_emissions(baseline_em, nz_target, baseline_yr, nz_year)
        budget = float(data.get("carbon_budget_total", 0))
        return (
            f'<h2>8. Carbon Budget</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Cumulative Emissions</div>'
            f'<div class="card-value">{_dec_comma(cumulative, 0)}</div>'
            f'<div class="card-unit">tCO2e ({baseline_yr}-{nz_year})</div></div>\n'
            f'  <div class="card"><div class="card-label">Carbon Budget</div>'
            f'<div class="card-value">{_dec_comma(budget, 0) if budget > 0 else "N/A"}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_sbti_validation(self, data: Dict[str, Any]) -> str:
        baseline_yr = int(data.get("baseline_year", 2022))
        nt_year = int(data.get("near_term_year", 2030))
        nt_pct = float(data.get("near_term_target_pct", 42))
        sbti = _sbti_15c_check(nt_pct, nt_year - baseline_yr)
        rag_cls = "rag-green" if sbti["passed"] else "rag-red"
        return (
            f'<h2>9. SBTi 1.5C Validation</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Status</div>'
            f'<div class="card-value"><span class="{rag_cls}">{sbti["status"]}</span></div></div>\n'
            f'  <div class="card"><div class="card-label">Reduction</div>'
            f'<div class="card-value">{_dec(sbti["reduction_pct"])}%</div>'
            f'<div class="card-unit">vs {_dec(sbti["min_reduction_pct"])}% required</div></div>\n'
            f'  <div class="card"><div class="card-label">Annual Rate</div>'
            f'<div class="card-value">{_dec(sbti["annual_rate"])}%</div>'
            f'<div class="card-unit">vs {_dec(sbti["min_annual_rate"])}% required</div></div>\n'
            f'</div>'
        )

    def _html_target_dashboard(self, data: Dict[str, Any]) -> str:
        current_em = float(data.get("current_emissions", 0))
        baseline_em = float(data.get("baseline_emissions", 0))
        if baseline_em > 0 and current_em > 0:
            reduction = float(_pct_reduction(current_em, baseline_em))
        else:
            reduction = 0
        return (
            f'<h2>10. Target Dashboard</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Current Emissions</div>'
            f'<div class="card-value">{_dec_comma(current_em, 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Reduction Achieved</div>'
            f'<div class="card-value">{_dec(reduction)}%</div></div>\n'
            f'</div>'
        )

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_", " ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return (
            f'<h2>11. XBRL Tags</h2>\n'
            f'<table>\n<tr><th>Data Point</th><th>XBRL Tag</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            f'<h2>12. Audit Trail</h2>\n'
            f'<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n'
            f'<tr><td>Generated</td><td>{ts}</td></tr>\n'
            f'<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n'
            f'<tr><td>Version</td><td>{_MODULE_VERSION}</td></tr>\n'
            f'<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n'
            f'</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-029 Interim Targets Pack on {ts}<br>'
            f'Interim targets summary with SBTi 1.5C validation<br>'
            f'All calculations deterministic - zero LLM in computation path'
            f'</div>'
        )
