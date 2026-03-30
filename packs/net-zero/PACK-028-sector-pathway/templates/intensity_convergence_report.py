# -*- coding: utf-8 -*-
"""
IntensityConvergenceReportTemplate - Intensity convergence analysis for PACK-028.

Renders a detailed intensity metric tracking and convergence analysis report
with trend visualizations, gap-to-pathway analysis, convergence timeline
projection, and multi-format output (Markdown, HTML, JSON, PDF-ready).

Sections:
    1.  Executive Summary
    2.  Intensity Metric Definition
    3.  Historical Intensity Trend
    4.  Convergence Pathway Overlay
    5.  Gap-to-Pathway Tracking
    6.  Annual Reduction Rate Analysis
    7.  Convergence Timeline Projection
    8.  Peer Intensity Comparison
    9.  Intensity Decomposition (Activity/Efficiency)
    10. Data Quality Assessment
    11. XBRL Tagging Summary
    12. Audit Trail & Provenance

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
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

_MODULE_VERSION = "28.0.0"
_PACK_ID = "PACK-028"
_TEMPLATE_ID = "intensity_convergence_report"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_WARN = "#ef6c00"
_DANGER = "#c62828"
_SUCCESS = "#2e7d32"

INTENSITY_METRICS: Dict[str, Dict[str, str]] = {
    "power": {"metric": "gCO2/kWh", "description": "Grid average emission factor", "unit_num": "gCO2", "unit_den": "kWh"},
    "steel": {"metric": "tCO2e/tonne crude steel", "description": "Per tonne of crude steel produced", "unit_num": "tCO2e", "unit_den": "tonne"},
    "cement": {"metric": "tCO2e/tonne cement", "description": "Per tonne of cement (incl. clinker)", "unit_num": "tCO2e", "unit_den": "tonne"},
    "aluminum": {"metric": "tCO2e/tonne aluminum", "description": "Per tonne of primary aluminum", "unit_num": "tCO2e", "unit_den": "tonne"},
    "pulp_paper": {"metric": "tCO2e/tonne pulp", "description": "Per tonne of pulp produced", "unit_num": "tCO2e", "unit_den": "tonne"},
    "chemicals": {"metric": "tCO2e/tonne product", "description": "Per tonne of chemical product", "unit_num": "tCO2e", "unit_den": "tonne"},
    "aviation": {"metric": "gCO2/pkm", "description": "Per passenger-kilometer", "unit_num": "gCO2", "unit_den": "pkm"},
    "shipping": {"metric": "gCO2/tkm", "description": "Per tonne-kilometer", "unit_num": "gCO2", "unit_den": "tkm"},
    "road_transport": {"metric": "gCO2/vkm", "description": "Per vehicle-kilometer", "unit_num": "gCO2", "unit_den": "vkm"},
    "rail": {"metric": "gCO2/pkm", "description": "Per passenger-kilometer", "unit_num": "gCO2", "unit_den": "pkm"},
    "buildings_res": {"metric": "kgCO2/m2/yr", "description": "Per square metre per year (residential)", "unit_num": "kgCO2", "unit_den": "m2/yr"},
    "buildings_com": {"metric": "kgCO2/m2/yr", "description": "Per square metre per year (commercial)", "unit_num": "kgCO2", "unit_den": "m2/yr"},
    "agriculture": {"metric": "tCO2e/tonne food", "description": "Per tonne of food produced (FLAG)", "unit_num": "tCO2e", "unit_den": "tonne"},
    "food_bev": {"metric": "tCO2e/tonne product", "description": "Per tonne of F&B product", "unit_num": "tCO2e", "unit_den": "tonne"},
    "oil_gas": {"metric": "gCO2/MJ", "description": "Per megajoule of energy produced", "unit_num": "gCO2", "unit_den": "MJ"},
}

XBRL_CONVERGENCE_TAGS: Dict[str, str] = {
    "intensity_base": "gl:BaseYearIntensityValue",
    "intensity_current": "gl:CurrentYearIntensityValue",
    "intensity_target": "gl:TargetYearIntensityValue",
    "gap_absolute": "gl:IntensityGapAbsolute",
    "gap_pct": "gl:IntensityGapPercentage",
    "convergence_year": "gl:ProjectedConvergenceYear",
    "annual_reduction": "gl:AnnualReductionRate",
    "required_reduction": "gl:RequiredReductionRate",
    "peer_percentile": "gl:PeerIntensityPercentile",
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

def _rag_status(gap_pct: float) -> str:
    if gap_pct <= 5.0:
        return "GREEN"
    elif gap_pct <= 15.0:
        return "AMBER"
    return "RED"

def _linear_convergence(base: float, target: float, base_yr: int, target_yr: int, yr: int) -> float:
    if yr <= base_yr:
        return base
    if yr >= target_yr:
        return target
    t = (yr - base_yr) / (target_yr - base_yr)
    return base + (target - base) * t

def _projected_convergence_year(
    current_intensity: float,
    current_year: int,
    annual_reduction_rate: float,
    target_intensity: float,
    max_year: int = 2100,
) -> int:
    """Project when current trajectory will converge to target intensity."""
    if annual_reduction_rate <= 0 or current_intensity <= target_intensity:
        return current_year
    intensity = current_intensity
    for yr in range(current_year, max_year + 1):
        if intensity <= target_intensity:
            return yr
        intensity *= (1 - annual_reduction_rate / 100)
    return max_year

def _cagr(start_val: float, end_val: float, years: int) -> float:
    """Compound annual growth rate (negative = reduction)."""
    if start_val <= 0 or end_val <= 0 or years <= 0:
        return 0.0
    return (math.pow(end_val / start_val, 1.0 / years) - 1) * 100

# ---------------------------------------------------------------------------
# Template Class
# ---------------------------------------------------------------------------
class IntensityConvergenceReportTemplate:
    """
    Intensity convergence tracking and analysis report template.

    Renders detailed intensity metric trend analysis, convergence pathway
    overlay, gap tracking, reduction rate analysis, timeline projection,
    and peer comparison. Supports MD, HTML, JSON, and PDF output.

    Example:
        >>> template = IntensityConvergenceReportTemplate()
        >>> data = {
        ...     "org_name": "CementCo",
        ...     "sector_id": "cement",
        ...     "base_year": 2020,
        ...     "target_year": 2050,
        ...     "base_intensity": 0.62,
        ...     "current_intensity": 0.55,
        ...     "current_year": 2025,
        ...     "target_intensity_2050": 0.06,
        ...     "intensity_history": [...],
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
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_metric_definition(data),
            self._md_historical_trend(data),
            self._md_convergence_overlay(data),
            self._md_gap_tracking(data),
            self._md_reduction_rate(data),
            self._md_convergence_timeline(data),
            self._md_peer_comparison(data),
            self._md_decomposition(data),
            self._md_data_quality(data),
            self._md_xbrl_tags(data),
            self._md_audit_trail(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = self._css()
        body_parts = [
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_metric_definition(data),
            self._html_historical_trend(data),
            self._html_convergence_overlay(data),
            self._html_gap_tracking(data),
            self._html_reduction_rate(data),
            self._html_convergence_timeline(data),
            self._html_peer_comparison(data),
            self._html_decomposition(data),
            self._html_data_quality(data),
            self._html_xbrl_tags(data),
            self._html_audit_trail(data),
            self._html_footer(data),
        ]
        body = "\n".join(body_parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Intensity Convergence Report - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        sector_id = data.get("sector_id", "")
        metric_info = INTENSITY_METRICS.get(sector_id, {})

        base_yr = int(data.get("base_year", 2022))
        target_yr = int(data.get("target_year", 2050))
        current_yr = int(data.get("current_year", 2025))
        base_int = float(data.get("base_intensity", 0))
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))

        pathway_val = _linear_convergence(base_int, target_2050, base_yr, target_yr, current_yr)
        gap_abs = current_int - pathway_val
        gap_pct = (gap_abs / pathway_val * 100) if pathway_val > 0 else 0

        history = data.get("intensity_history", [])
        actual_cagr = 0.0
        if len(history) >= 2:
            first_val = float(history[0].get("intensity", 0))
            last_val = float(history[-1].get("intensity", 0))
            years_span = int(history[-1].get("year", 0)) - int(history[0].get("year", 0))
            if years_span > 0:
                actual_cagr = _cagr(first_val, last_val, years_span)

        required_annual = 0.0
        if current_int > target_2050 and (target_yr - current_yr) > 0:
            required_annual = abs(_cagr(current_int, target_2050, target_yr - current_yr))

        convergence_yr = _projected_convergence_year(
            current_int, current_yr, abs(actual_cagr) if actual_cagr < 0 else 1.0, target_2050
        )

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "sector_id": sector_id,
            "intensity_metric": metric_info.get("metric", ""),
            "baseline": {
                "base_year": base_yr,
                "base_intensity": str(base_int),
                "current_year": current_yr,
                "current_intensity": str(current_int),
            },
            "target": {
                "target_year": target_yr,
                "target_intensity": str(target_2050),
            },
            "convergence": {
                "pathway_value_current_year": str(round(pathway_val, 4)),
                "gap_absolute": str(round(gap_abs, 4)),
                "gap_percentage": str(round(gap_pct, 2)),
                "rag_status": _rag_status(abs(gap_pct)),
                "actual_annual_reduction_cagr": str(round(actual_cagr, 3)),
                "required_annual_reduction": str(round(required_annual, 3)),
                "projected_convergence_year": convergence_yr,
                "on_track": gap_pct <= 5.0,
            },
            "intensity_history": history,
            "peer_comparison": data.get("peer_comparison", {}),
            "decomposition": data.get("decomposition", {}),
            "data_quality": data.get("data_quality", {}),
            "xbrl_tags": {k: XBRL_CONVERGENCE_TAGS[k] for k in XBRL_CONVERGENCE_TAGS},
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        json_data = self.render_json(data)
        html_content = self.render_html(data)
        return {
            "format": "pdf",
            "html_content": html_content,
            "structured_data": json_data,
            "metadata": {
                "title": f"Intensity Convergence Report - {data.get('org_name', '')}",
                "author": "GreenLang PACK-028",
                "subject": "Intensity Convergence Analysis",
                "creator": f"GreenLang v{_MODULE_VERSION}",
            },
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_metric_info(self, data: Dict[str, Any]) -> Dict[str, str]:
        sector_id = data.get("sector_id", "")
        return INTENSITY_METRICS.get(sector_id, {
            "metric": "tCO2e/unit", "description": "Emissions per unit of output",
            "unit_num": "tCO2e", "unit_den": "unit",
        })

    def _calc_gap(self, data: Dict[str, Any]) -> Tuple[float, float, float, str]:
        base_int = float(data.get("base_intensity", 0))
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        base_yr = int(data.get("base_year", 2022))
        target_yr = int(data.get("target_year", 2050))
        current_yr = int(data.get("current_year", 2025))
        pathway_val = _linear_convergence(base_int, target_2050, base_yr, target_yr, current_yr)
        gap_abs = current_int - pathway_val
        gap_pct = (gap_abs / pathway_val * 100) if pathway_val > 0 else 0
        rag = _rag_status(abs(gap_pct))
        return pathway_val, gap_abs, gap_pct, rag

    # ------------------------------------------------------------------
    # Markdown Sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        metric_info = self._get_metric_info(data)
        return (
            f"# Intensity Convergence Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Sector:** {data.get('sector_id', '').replace('_', ' ').title()}  \n"
            f"**Intensity Metric:** {metric_info['metric']}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-028 Sector Pathway Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        metric_info = self._get_metric_info(data)
        base_int = float(data.get("base_intensity", 0))
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        change_pct = float(_pct_change(current_int, base_int))
        pathway_val, gap_abs, gap_pct, rag = self._calc_gap(data)

        history = data.get("intensity_history", [])
        actual_cagr = 0.0
        if len(history) >= 2:
            first_val = float(history[0].get("intensity", 0))
            last_val = float(history[-1].get("intensity", 0))
            years_span = int(history[-1].get("year", 0)) - int(history[0].get("year", 0))
            if years_span > 0:
                actual_cagr = _cagr(first_val, last_val, years_span)

        required_rate = 0.0
        current_yr = int(data.get("current_year", 2025))
        target_yr = int(data.get("target_year", 2050))
        if current_int > target_2050 and (target_yr - current_yr) > 0:
            required_rate = abs(_cagr(current_int, target_2050, target_yr - current_yr))

        lines = [
            "## 1. Executive Summary\n",
            f"| KPI | Value |",
            f"|-----|-------|",
            f"| Base Year Intensity | {_dec(base_int, 4)} {metric_info['metric']} |",
            f"| Current Intensity | {_dec(current_int, 4)} {metric_info['metric']} |",
            f"| Change from Base | {_dec(change_pct)}% |",
            f"| 2050 Target | {_dec(target_2050, 4)} {metric_info['metric']} |",
            f"| Pathway Target (current year) | {_dec(pathway_val, 4)} {metric_info['metric']} |",
            f"| Gap to Pathway | {_dec(gap_pct)}% ({_dec(gap_abs, 4)}) |",
            f"| RAG Status | **{rag}** |",
            f"| Actual Annual Reduction (CAGR) | {_dec(actual_cagr, 2)}% |",
            f"| Required Annual Reduction | {_dec(required_rate, 2)}% |",
            f"| Reduction Gap | {_dec(required_rate - abs(actual_cagr), 2)}% per year |",
        ]
        return "\n".join(lines)

    def _md_metric_definition(self, data: Dict[str, Any]) -> str:
        metric_info = self._get_metric_info(data)
        return (
            "## 2. Intensity Metric Definition\n\n"
            f"| Parameter | Value |\n"
            f"|-----------|-------|\n"
            f"| Metric | {metric_info['metric']} |\n"
            f"| Description | {metric_info['description']} |\n"
            f"| Numerator | {metric_info['unit_num']} (emissions) |\n"
            f"| Denominator | {metric_info['unit_den']} (activity) |\n"
            f"| Scope Coverage | Scope 1 + Scope 2 (location-based) |\n"
            f"| Boundary | Operational control |\n"
            f"| Gases | CO2, CH4, N2O, HFCs, PFCs, SF6, NF3 |"
        )

    def _md_historical_trend(self, data: Dict[str, Any]) -> str:
        history = data.get("intensity_history", [])
        metric_info = self._get_metric_info(data)
        lines = [
            "## 3. Historical Intensity Trend\n",
            f"| Year | Intensity ({metric_info['metric']}) | YoY Change (%) | Emissions (tCO2e) | Activity ({metric_info['unit_den']}) |",
            f"|------|----------|----------------|----------|----------|",
        ]
        prev_int = None
        for h in history:
            val = float(h.get("intensity", 0))
            yoy = float(_pct_change(val, prev_int)) if prev_int else 0
            yoy_str = f"{_dec(yoy)}%" if prev_int else "-"
            lines.append(
                f"| {h.get('year', '')} | {_dec(val, 4)} | {yoy_str} "
                f"| {_dec_comma(h.get('emissions', 0))} | {_dec_comma(h.get('activity', 0))} |"
            )
            prev_int = val

        if len(history) >= 2:
            first_val = float(history[0].get("intensity", 0))
            last_val = float(history[-1].get("intensity", 0))
            yrs = int(history[-1].get("year", 0)) - int(history[0].get("year", 0))
            cagr = _cagr(first_val, last_val, yrs) if yrs > 0 else 0
            lines.append(f"\n**Period CAGR:** {_dec(cagr, 2)}% per year")
        return "\n".join(lines)

    def _md_convergence_overlay(self, data: Dict[str, Any]) -> str:
        base_int = float(data.get("base_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        base_yr = int(data.get("base_year", 2022))
        target_yr = int(data.get("target_year", 2050))
        metric_info = self._get_metric_info(data)
        history = data.get("intensity_history", [])
        hist_map = {int(h.get("year", 0)): float(h.get("intensity", 0)) for h in history}
        sample_years = list(range(base_yr, target_yr + 1, 5))
        if target_yr not in sample_years:
            sample_years.append(target_yr)
        lines = [
            "## 4. Convergence Pathway Overlay\n",
            f"| Year | Pathway Target | Actual Intensity | Gap | Status |",
            f"|------|---------------:|------------------:|----:|--------|",
        ]
        for yr in sample_years:
            pv = _linear_convergence(base_int, target_2050, base_yr, target_yr, yr)
            actual = hist_map.get(yr)
            if actual is not None:
                gap = actual - pv
                gap_pct_val = (gap / pv * 100) if pv > 0 else 0
                rag = _rag_status(abs(gap_pct_val))
                lines.append(
                    f"| {yr} | {_dec(pv, 4)} | {_dec(actual, 4)} | {_dec(gap_pct_val)}% | {rag} |"
                )
            else:
                lines.append(f"| {yr} | {_dec(pv, 4)} | - | - | - |")
        return "\n".join(lines)

    def _md_gap_tracking(self, data: Dict[str, Any]) -> str:
        pathway_val, gap_abs, gap_pct, rag = self._calc_gap(data)
        metric_info = self._get_metric_info(data)
        current_int = float(data.get("current_intensity", 0))
        current_yr = int(data.get("current_year", 2025))
        target_yr = int(data.get("target_year", 2050))
        target_2050 = float(data.get("target_intensity_2050", 0))
        accel = abs(gap_abs) / max(1, target_yr - current_yr)
        lines = [
            "## 5. Gap-to-Pathway Tracking\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Pathway Target ({current_yr}) | {_dec(pathway_val, 4)} {metric_info['metric']} |",
            f"| Actual Intensity ({current_yr}) | {_dec(current_int, 4)} {metric_info['metric']} |",
            f"| Absolute Gap | {_dec(gap_abs, 4)} {metric_info['metric']} |",
            f"| Percentage Gap | {_dec(gap_pct)}% |",
            f"| RAG Status | **{rag}** |",
            f"| Additional Reduction Needed | {_dec(accel, 4)} {metric_info['metric']}/year |",
            f"| Years to Target | {target_yr - current_yr} years |",
        ]
        if gap_pct > 5:
            lines.append(
                f"\n> **Warning:** Current trajectory is {_dec(gap_pct)}% above the sector "
                f"convergence pathway. Acceleration of reduction efforts required."
            )
        elif gap_pct < -5:
            lines.append(
                f"\n> **Positive:** Organization is {_dec(abs(gap_pct))}% ahead of the "
                f"sector convergence pathway."
            )
        else:
            lines.append(
                f"\n> **On Track:** Organization intensity is within 5% of the "
                f"sector convergence pathway."
            )
        return "\n".join(lines)

    def _md_reduction_rate(self, data: Dict[str, Any]) -> str:
        history = data.get("intensity_history", [])
        metric_info = self._get_metric_info(data)
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        current_yr = int(data.get("current_year", 2025))
        target_yr = int(data.get("target_year", 2050))

        actual_cagr = 0.0
        if len(history) >= 2:
            first_val = float(history[0].get("intensity", 0))
            last_val = float(history[-1].get("intensity", 0))
            yrs = int(history[-1].get("year", 0)) - int(history[0].get("year", 0))
            if yrs > 0:
                actual_cagr = _cagr(first_val, last_val, yrs)

        required_rate = 0.0
        if current_int > target_2050 and (target_yr - current_yr) > 0:
            required_rate = abs(_cagr(current_int, target_2050, target_yr - current_yr))

        sbti_min_rate = 4.2  # SBTi minimum annual reduction for 1.5C

        lines = [
            "## 6. Annual Reduction Rate Analysis\n",
            f"| Metric | Rate (%/yr) |",
            f"|--------|------------:|",
            f"| Actual Historical CAGR | {_dec(actual_cagr, 2)}% |",
            f"| Required Rate to 2050 Target | {_dec(required_rate, 2)}% |",
            f"| SBTi Minimum (1.5C) | {_dec(sbti_min_rate, 1)}% |",
            f"| Rate Gap (Required - Actual) | {_dec(required_rate - abs(actual_cagr), 2)}% |",
        ]

        if len(history) >= 2:
            lines.append("\n### Year-over-Year Reduction Rates\n")
            lines.append("| Period | Reduction Rate (%) | On Track |")
            lines.append("|--------|-------------------:|----------|")
            for i in range(1, len(history)):
                yr_from = int(history[i - 1].get("year", 0))
                yr_to = int(history[i].get("year", 0))
                val_from = float(history[i - 1].get("intensity", 0))
                val_to = float(history[i].get("intensity", 0))
                if val_from > 0:
                    rate = (val_from - val_to) / val_from * 100
                    on_track = "Yes" if rate >= required_rate else "No"
                    lines.append(f"| {yr_from}-{yr_to} | {_dec(rate, 2)}% | {on_track} |")
        return "\n".join(lines)

    def _md_convergence_timeline(self, data: Dict[str, Any]) -> str:
        history = data.get("intensity_history", [])
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        current_yr = int(data.get("current_year", 2025))
        target_yr = int(data.get("target_year", 2050))

        actual_cagr = 0.0
        if len(history) >= 2:
            first_val = float(history[0].get("intensity", 0))
            last_val = float(history[-1].get("intensity", 0))
            yrs = int(history[-1].get("year", 0)) - int(history[0].get("year", 0))
            if yrs > 0:
                actual_cagr = _cagr(first_val, last_val, yrs)

        proj_yr = _projected_convergence_year(
            current_int, current_yr, abs(actual_cagr) if actual_cagr < 0 else 1.0, target_2050
        )
        delay = proj_yr - target_yr

        lines = [
            "## 7. Convergence Timeline Projection\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Target Convergence Year | {target_yr} |",
            f"| Projected Convergence Year (at current rate) | {proj_yr} |",
            f"| Delay | {delay} years {'behind' if delay > 0 else 'ahead' if delay < 0 else 'on track'} |",
            f"| Current Reduction Rate | {_dec(abs(actual_cagr), 2)}% per year |",
        ]

        lines.append("\n### Projected Intensity at Key Milestones\n")
        lines.append("| Year | Projected (current rate) | Pathway Target | Gap |")
        lines.append("|------|-------------------------:|---------------:|----:|")
        red_rate = abs(actual_cagr) / 100 if actual_cagr < 0 else 0.01
        intensity = current_int
        base_int = float(data.get("base_intensity", 0))
        base_yr = int(data.get("base_year", 2022))
        for yr in [2025, 2030, 2035, 2040, 2045, 2050]:
            if yr < current_yr:
                continue
            years_forward = yr - current_yr
            proj = current_int * math.pow(1 - red_rate, years_forward) if red_rate > 0 else current_int
            pv = _linear_convergence(base_int, target_2050, base_yr, target_yr, yr)
            gap = ((proj - pv) / pv * 100) if pv > 0 else 0
            lines.append(f"| {yr} | {_dec(proj, 4)} | {_dec(pv, 4)} | {_dec(gap)}% |")
        return "\n".join(lines)

    def _md_peer_comparison(self, data: Dict[str, Any]) -> str:
        peers = data.get("peer_comparison", {})
        peer_data = peers.get("peers", [])
        metric_info = self._get_metric_info(data)
        current_int = float(data.get("current_intensity", 0))

        lines = [
            "## 8. Peer Intensity Comparison\n",
            f"**Your Intensity:** {_dec(current_int, 4)} {metric_info['metric']}  \n"
            f"**Sector Average:** {_dec(peers.get('sector_average', 0), 4)} {metric_info['metric']}  \n"
            f"**Sector Leader:** {_dec(peers.get('sector_leader', 0), 4)} {metric_info['metric']}  \n"
            f"**Your Percentile:** {peers.get('your_percentile', 'N/A')}\n",
        ]
        if peer_data:
            lines.append("| Peer | Intensity | Percentile | SBTi Committed | Gap vs. You |")
            lines.append("|------|----------:|-----------|:--------------:|------------:|")
            for p in peer_data:
                p_int = float(p.get("intensity", 0))
                gap_vs = current_int - p_int
                lines.append(
                    f"| {p.get('name', '')} | {_dec(p_int, 4)} "
                    f"| {p.get('percentile', '')} "
                    f"| {p.get('sbti_committed', 'N/A')} "
                    f"| {'+' if gap_vs > 0 else ''}{_dec(gap_vs, 4)} |"
                )
        else:
            lines.append("_Peer comparison data not available._")
        return "\n".join(lines)

    def _md_decomposition(self, data: Dict[str, Any]) -> str:
        decomp = data.get("decomposition", {})
        factors = decomp.get("factors", [])
        lines = [
            "## 9. Intensity Decomposition\n",
            "Decomposition of intensity change into contributing factors:\n",
        ]
        if factors:
            lines.append("| Factor | Contribution | Impact (%) | Direction |")
            lines.append("|--------|-------------|-----------|-----------|")
            for f in factors:
                impact = float(f.get("impact_pct", 0))
                direction = "Reducing" if impact < 0 else "Increasing"
                lines.append(
                    f"| {f.get('factor', '')} | {f.get('description', '')} "
                    f"| {_dec(impact)}% | {direction} |"
                )
        else:
            lines.append(
                "| Emissions Change | Change in total emissions | - | - |\n"
                "| Activity Change | Change in production/activity volume | - | - |\n"
                "| Efficiency Change | Change in emission intensity per unit | - | - |\n"
                "| Structural Change | Change in product/process mix | - | - |\n\n"
                "_Detailed decomposition requires time-series emissions and activity data._"
            )
        return "\n".join(lines)

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality", {})
        lines = [
            "## 10. Data Quality Assessment\n",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Overall Score | {dq.get('overall_score', 'N/A')} |",
            f"| Emissions Data Quality | {dq.get('emissions_quality', 'Medium')} |",
            f"| Activity Data Quality | {dq.get('activity_quality', 'Medium')} |",
            f"| Coverage | {_dec(dq.get('coverage_pct', 100))}% |",
            f"| Estimation Share | {_dec(dq.get('estimation_share_pct', 0))}% |",
            f"| Verification Status | {dq.get('verification', 'Not verified')} |",
        ]
        return "\n".join(lines)

    def _md_xbrl_tags(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 11. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |",
            "|------------|----------|-------|",
        ]
        _, gap_abs, gap_pct, _ = self._calc_gap(data)
        tag_vals = {
            "intensity_base": _dec(data.get("base_intensity", 0), 4),
            "intensity_current": _dec(data.get("current_intensity", 0), 4),
            "intensity_target": _dec(data.get("target_intensity_2050", 0), 4),
            "gap_absolute": _dec(gap_abs, 4),
            "gap_pct": _dec(gap_pct),
        }
        for key, tag in XBRL_CONVERGENCE_TAGS.items():
            val = tag_vals.get(key, "")
            if val:
                lines.append(f"| {key.replace('_', ' ').title()} | {tag} | {val} |")
        return "\n".join(lines)

    def _md_audit_trail(self, data: Dict[str, Any]) -> str:
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
            f"| Calculation Engine | Deterministic (zero-hallucination) |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}*  \n"
            f"*Intensity convergence analysis aligned with SBTi SDA methodology.*"
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
            f".trend-up{{color:#c62828;}}.trend-down{{color:#1b5e20;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};"
            f"color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        metric_info = self._get_metric_info(data)
        return (
            f'<h1>Intensity Convergence Report</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Metric:</strong> {metric_info["metric"]} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        metric_info = self._get_metric_info(data)
        base_int = float(data.get("base_intensity", 0))
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        change_pct = float(_pct_change(current_int, base_int))
        pathway_val, _, gap_pct, rag = self._calc_gap(data)
        rag_cls = f"rag-{rag.lower()}"
        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Base Intensity</div>'
            f'<div class="card-value">{_dec(base_int, 4)}</div>'
            f'<div class="card-unit">{metric_info["metric"]}</div></div>\n'
            f'  <div class="card"><div class="card-label">Current</div>'
            f'<div class="card-value">{_dec(current_int, 4)}</div>'
            f'<div class="card-unit">{metric_info["metric"]}</div></div>\n'
            f'  <div class="card"><div class="card-label">Change</div>'
            f'<div class="card-value">{_dec(change_pct)}%</div></div>\n'
            f'  <div class="card"><div class="card-label">2050 Target</div>'
            f'<div class="card-value">{_dec(target_2050, 4)}</div>'
            f'<div class="card-unit">{metric_info["metric"]}</div></div>\n'
            f'  <div class="card"><div class="card-label">Gap</div>'
            f'<div class="card-value">{_dec(gap_pct)}%</div>'
            f'<div class="card-unit"><span class="{rag_cls}">{rag}</span></div></div>\n'
            f'</div>'
        )

    def _html_metric_definition(self, data: Dict[str, Any]) -> str:
        m = self._get_metric_info(data)
        return (
            f'<h2>2. Intensity Metric Definition</h2>\n'
            f'<table><tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Metric</td><td>{m["metric"]}</td></tr>\n'
            f'<tr><td>Description</td><td>{m["description"]}</td></tr>\n'
            f'<tr><td>Numerator</td><td>{m["unit_num"]}</td></tr>\n'
            f'<tr><td>Denominator</td><td>{m["unit_den"]}</td></tr>\n'
            f'</table>'
        )

    def _html_historical_trend(self, data: Dict[str, Any]) -> str:
        history = data.get("intensity_history", [])
        m = self._get_metric_info(data)
        rows = ""
        prev = None
        for h in history:
            val = float(h.get("intensity", 0))
            yoy = float(_pct_change(val, prev)) if prev else 0
            cls = "trend-down" if yoy < 0 else "trend-up" if yoy > 0 else ""
            rows += (
                f'<tr><td>{h.get("year", "")}</td><td>{_dec(val, 4)}</td>'
                f'<td class="{cls}">{_dec(yoy) + "%" if prev else "-"}</td>'
                f'<td>{_dec_comma(h.get("emissions", 0))}</td>'
                f'<td>{_dec_comma(h.get("activity", 0))}</td></tr>\n'
            )
            prev = val
        return (
            f'<h2>3. Historical Intensity Trend</h2>\n'
            f'<table>\n<tr><th>Year</th><th>Intensity</th><th>YoY</th>'
            f'<th>Emissions</th><th>Activity</th></tr>\n{rows}</table>'
        )

    def _html_convergence_overlay(self, data: Dict[str, Any]) -> str:
        base_int = float(data.get("base_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        base_yr = int(data.get("base_year", 2022))
        target_yr = int(data.get("target_year", 2050))
        history = data.get("intensity_history", [])
        hist_map = {int(h.get("year", 0)): float(h.get("intensity", 0)) for h in history}
        sample_years = list(range(base_yr, target_yr + 1, 5))
        if target_yr not in sample_years:
            sample_years.append(target_yr)
        rows = ""
        for yr in sample_years:
            pv = _linear_convergence(base_int, target_2050, base_yr, target_yr, yr)
            actual = hist_map.get(yr)
            if actual is not None:
                gap = ((actual - pv) / pv * 100) if pv > 0 else 0
                rag = _rag_status(abs(gap))
                rag_cls = f"rag-{rag.lower()}"
                rows += (
                    f'<tr><td>{yr}</td><td>{_dec(pv, 4)}</td><td>{_dec(actual, 4)}</td>'
                    f'<td>{_dec(gap)}%</td><td><span class="{rag_cls}">{rag}</span></td></tr>\n'
                )
            else:
                rows += f'<tr><td>{yr}</td><td>{_dec(pv, 4)}</td><td>-</td><td>-</td><td>-</td></tr>\n'
        return (
            f'<h2>4. Convergence Pathway Overlay</h2>\n'
            f'<table>\n<tr><th>Year</th><th>Pathway</th><th>Actual</th>'
            f'<th>Gap</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_gap_tracking(self, data: Dict[str, Any]) -> str:
        m = self._get_metric_info(data)
        pathway_val, gap_abs, gap_pct, rag = self._calc_gap(data)
        current_int = float(data.get("current_intensity", 0))
        rag_cls = f"rag-{rag.lower()}"
        return (
            f'<h2>5. Gap-to-Pathway Tracking</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Pathway Target</div>'
            f'<div class="card-value">{_dec(pathway_val, 4)}</div>'
            f'<div class="card-unit">{m["metric"]}</div></div>\n'
            f'  <div class="card"><div class="card-label">Actual</div>'
            f'<div class="card-value">{_dec(current_int, 4)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Gap</div>'
            f'<div class="card-value">{_dec(gap_pct)}%</div>'
            f'<div class="card-unit"><span class="{rag_cls}">{rag}</span></div></div>\n'
            f'</div>'
        )

    def _html_reduction_rate(self, data: Dict[str, Any]) -> str:
        history = data.get("intensity_history", [])
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        current_yr = int(data.get("current_year", 2025))
        target_yr = int(data.get("target_year", 2050))
        actual_cagr = 0.0
        if len(history) >= 2:
            fv = float(history[0].get("intensity", 0))
            lv = float(history[-1].get("intensity", 0))
            yrs = int(history[-1].get("year", 0)) - int(history[0].get("year", 0))
            if yrs > 0:
                actual_cagr = _cagr(fv, lv, yrs)
        req = abs(_cagr(current_int, target_2050, target_yr - current_yr)) if (target_yr - current_yr) > 0 else 0
        return (
            f'<h2>6. Annual Reduction Rate</h2>\n'
            f'<table>\n<tr><th>Metric</th><th>Rate (%/yr)</th></tr>\n'
            f'<tr><td>Actual CAGR</td><td>{_dec(actual_cagr, 2)}%</td></tr>\n'
            f'<tr><td>Required to 2050</td><td>{_dec(req, 2)}%</td></tr>\n'
            f'<tr><td>SBTi Minimum (1.5C)</td><td>4.2%</td></tr>\n'
            f'<tr><td>Gap</td><td>{_dec(req - abs(actual_cagr), 2)}%</td></tr>\n'
            f'</table>'
        )

    def _html_convergence_timeline(self, data: Dict[str, Any]) -> str:
        history = data.get("intensity_history", [])
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        current_yr = int(data.get("current_year", 2025))
        target_yr = int(data.get("target_year", 2050))
        base_int = float(data.get("base_intensity", 0))
        base_yr = int(data.get("base_year", 2022))
        actual_cagr = 0.0
        if len(history) >= 2:
            fv = float(history[0].get("intensity", 0))
            lv = float(history[-1].get("intensity", 0))
            yrs = int(history[-1].get("year", 0)) - int(history[0].get("year", 0))
            if yrs > 0:
                actual_cagr = _cagr(fv, lv, yrs)
        proj_yr = _projected_convergence_year(
            current_int, current_yr, abs(actual_cagr) if actual_cagr < 0 else 1.0, target_2050
        )
        red_rate = abs(actual_cagr) / 100 if actual_cagr < 0 else 0.01
        rows = ""
        for yr in [2025, 2030, 2035, 2040, 2045, 2050]:
            if yr < current_yr:
                continue
            yf = yr - current_yr
            proj = current_int * math.pow(1 - red_rate, yf) if red_rate > 0 else current_int
            pv = _linear_convergence(base_int, target_2050, base_yr, target_yr, yr)
            gap = ((proj - pv) / pv * 100) if pv > 0 else 0
            rag = _rag_status(abs(gap))
            rag_cls = f"rag-{rag.lower()}"
            rows += (
                f'<tr><td>{yr}</td><td>{_dec(proj, 4)}</td><td>{_dec(pv, 4)}</td>'
                f'<td><span class="{rag_cls}">{_dec(gap)}%</span></td></tr>\n'
            )
        return (
            f'<h2>7. Convergence Timeline</h2>\n'
            f'<p><strong>Projected convergence:</strong> {proj_yr} '
            f'({"on track" if proj_yr <= target_yr else f"{proj_yr - target_yr} years late"})</p>\n'
            f'<table>\n<tr><th>Year</th><th>Projected</th><th>Pathway</th><th>Gap</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_peer_comparison(self, data: Dict[str, Any]) -> str:
        peers = data.get("peer_comparison", {})
        peer_list = peers.get("peers", [])
        current_int = float(data.get("current_intensity", 0))
        rows = ""
        for p in peer_list:
            p_int = float(p.get("intensity", 0))
            gap = current_int - p_int
            rows += (
                f'<tr><td>{p.get("name", "")}</td><td>{_dec(p_int, 4)}</td>'
                f'<td>{p.get("percentile", "")}</td>'
                f'<td>{"+" if gap > 0 else ""}{_dec(gap, 4)}</td></tr>\n'
            )
        return (
            f'<h2>8. Peer Comparison</h2>\n'
            f'<table>\n<tr><th>Peer</th><th>Intensity</th><th>Percentile</th><th>Gap</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_decomposition(self, data: Dict[str, Any]) -> str:
        decomp = data.get("decomposition", {})
        factors = decomp.get("factors", [])
        rows = ""
        for f in factors:
            impact = float(f.get("impact_pct", 0))
            cls = "trend-down" if impact < 0 else "trend-up"
            rows += (
                f'<tr><td>{f.get("factor", "")}</td><td>{f.get("description", "")}</td>'
                f'<td class="{cls}">{_dec(impact)}%</td></tr>\n'
            )
        return (
            f'<h2>9. Intensity Decomposition</h2>\n'
            f'<table>\n<tr><th>Factor</th><th>Description</th><th>Impact</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        dq = data.get("data_quality", {})
        return (
            f'<h2>10. Data Quality</h2>\n'
            f'<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Overall Score</td><td>{dq.get("overall_score", "N/A")}</td></tr>\n'
            f'<tr><td>Emissions Quality</td><td>{dq.get("emissions_quality", "Medium")}</td></tr>\n'
            f'<tr><td>Activity Quality</td><td>{dq.get("activity_quality", "Medium")}</td></tr>\n'
            f'<tr><td>Coverage</td><td>{_dec(dq.get("coverage_pct", 100))}%</td></tr>\n'
            f'</table>'
        )

    def _html_xbrl_tags(self, data: Dict[str, Any]) -> str:
        _, gap_abs, gap_pct, _ = self._calc_gap(data)
        rows = ""
        vals = {
            "intensity_base": _dec(data.get("base_intensity", 0), 4),
            "intensity_current": _dec(data.get("current_intensity", 0), 4),
            "intensity_target": _dec(data.get("target_intensity_2050", 0), 4),
            "gap_absolute": _dec(gap_abs, 4),
            "gap_pct": _dec(gap_pct),
        }
        for k, tag in XBRL_CONVERGENCE_TAGS.items():
            v = vals.get(k, "")
            if v:
                rows += f'<tr><td>{k.replace("_", " ").title()}</td><td><code>{tag}</code></td><td>{v}</td></tr>\n'
        return (
            f'<h2>11. XBRL Tags</h2>\n'
            f'<table>\n<tr><th>Point</th><th>Tag</th><th>Value</th></tr>\n{rows}</table>'
        )

    def _html_audit_trail(self, data: Dict[str, Any]) -> str:
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
            f'Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}<br>'
            f'Intensity convergence analysis aligned with SBTi SDA methodology'
            f'</div>'
        )
