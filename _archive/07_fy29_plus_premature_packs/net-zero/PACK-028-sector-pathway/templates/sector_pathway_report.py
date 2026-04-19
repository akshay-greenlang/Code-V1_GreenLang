# -*- coding: utf-8 -*-
"""
SectorPathwayReportTemplate - Sector decarbonization pathway report for PACK-028.

Renders a comprehensive sector-specific decarbonization pathway report with
SDA/IEA alignment status, convergence curve charts, intensity trajectory
visualization, and multi-format output (Markdown, HTML, JSON, PDF-ready).

Sections:
    1.  Executive Summary
    2.  Sector Classification
    3.  Intensity Baseline Profile
    4.  SBTi SDA Pathway Alignment
    5.  IEA NZE Pathway Integration
    6.  Convergence Curve Analysis
    7.  Year-by-Year Pathway Targets
    8.  Activity Growth Projections
    9.  Absolute vs. Intensity Trajectory
    10. Gap-to-Pathway Analysis
    11. Regional Pathway Variants
    12. Pathway Confidence & Uncertainty
    13. XBRL Tagging Summary
    14. Audit Trail & Provenance

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
_TEMPLATE_ID = "sector_pathway_report"

# ---------------------------------------------------------------------------
# Colour Palette (Sector Pathway Theme)
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
# SDA Sector Definitions
# ---------------------------------------------------------------------------
SDA_SECTORS: List[Dict[str, str]] = [
    {"id": "power", "name": "Power Generation", "metric": "gCO2/kWh", "sda": "SDA-Power"},
    {"id": "steel", "name": "Steel", "metric": "tCO2e/tonne crude steel", "sda": "SDA-Steel"},
    {"id": "cement", "name": "Cement", "metric": "tCO2e/tonne cement", "sda": "SDA-Cement"},
    {"id": "aluminum", "name": "Aluminum", "metric": "tCO2e/tonne aluminum", "sda": "SDA-Aluminum"},
    {"id": "pulp_paper", "name": "Pulp & Paper", "metric": "tCO2e/tonne pulp", "sda": "SDA-Pulp"},
    {"id": "chemicals", "name": "Chemicals", "metric": "tCO2e/tonne product", "sda": "SDA-Chemicals"},
    {"id": "aviation", "name": "Aviation", "metric": "gCO2/pkm", "sda": "SDA-Aviation"},
    {"id": "shipping", "name": "Shipping", "metric": "gCO2/tkm", "sda": "SDA-Shipping"},
    {"id": "road_transport", "name": "Road Transport", "metric": "gCO2/vkm", "sda": "SDA-Transport"},
    {"id": "rail", "name": "Rail", "metric": "gCO2/pkm", "sda": "SDA-Rail"},
    {"id": "buildings_res", "name": "Buildings (Residential)", "metric": "kgCO2/m2/yr", "sda": "SDA-Buildings"},
    {"id": "buildings_com", "name": "Buildings (Commercial)", "metric": "kgCO2/m2/yr", "sda": "SDA-Buildings"},
    {"id": "agriculture", "name": "Agriculture", "metric": "tCO2e/tonne food", "sda": "FLAG"},
    {"id": "food_bev", "name": "Food & Beverage", "metric": "tCO2e/tonne product", "sda": "IEA-Only"},
    {"id": "oil_gas", "name": "Oil & Gas (Upstream)", "metric": "gCO2/MJ", "sda": "IEA-Only"},
]

IEA_SCENARIOS: List[Dict[str, str]] = [
    {"id": "nze", "name": "Net Zero Emissions (NZE)", "temp": "1.5C", "probability": "50%"},
    {"id": "wb2c", "name": "Well-Below 2C (WB2C)", "temp": "<2C", "probability": "66%"},
    {"id": "2c", "name": "2 Degrees (2DS)", "temp": "2C", "probability": "50%"},
    {"id": "aps", "name": "Announced Pledges (APS)", "temp": "1.7C", "probability": "N/A"},
    {"id": "steps", "name": "Stated Policies (STEPS)", "temp": "2.4C", "probability": "N/A"},
]

CONVERGENCE_TYPES = ["linear", "exponential", "s_curve", "stepped"]

XBRL_TAGS: Dict[str, str] = {
    "sector_id": "gl:SectorClassificationIdentifier",
    "intensity_base": "gl:BaseYearIntensityMetric",
    "intensity_current": "gl:CurrentYearIntensityMetric",
    "intensity_target_2030": "gl:NearTermIntensityTarget2030",
    "intensity_target_2050": "gl:LongTermIntensityTarget2050",
    "convergence_year": "gl:PathwayConvergenceYear",
    "gap_pct": "gl:GapToPathwayPercentage",
    "sda_alignment": "gl:SBTiSDAAlignmentStatus",
    "iea_alignment": "gl:IEANZEAlignmentStatus",
    "scenario": "gl:PathwayScenarioIdentifier",
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

def _pct_of(part: Any, total: Any) -> Decimal:
    p = Decimal(str(part))
    t = Decimal(str(total))
    if t == 0:
        return Decimal("0.00")
    return (p / t * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def _pct_change(current: Any, baseline: Any) -> Decimal:
    c = Decimal(str(current))
    b = Decimal(str(baseline))
    if b == 0:
        return Decimal("0.00")
    return ((c - b) / b * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def _rag_status(gap_pct: float) -> str:
    """Return RAG status based on gap-to-pathway percentage."""
    if gap_pct <= 5.0:
        return "GREEN"
    elif gap_pct <= 15.0:
        return "AMBER"
    else:
        return "RED"

def _rag_color(status: str) -> str:
    return {"GREEN": _SUCCESS, "AMBER": _WARN, "RED": _DANGER}.get(status, "#999")

def _convergence_value(
    base_intensity: float,
    target_intensity: float,
    base_year: int,
    target_year: int,
    current_year: int,
    method: str = "linear",
) -> float:
    """Calculate pathway intensity for a given year using convergence method."""
    if current_year <= base_year:
        return base_intensity
    if current_year >= target_year:
        return target_intensity
    t = (current_year - base_year) / (target_year - base_year)
    if method == "linear":
        return base_intensity + (target_intensity - base_intensity) * t
    elif method == "exponential":
        if base_intensity <= 0 or target_intensity <= 0:
            return base_intensity + (target_intensity - base_intensity) * t
        k = math.log(target_intensity / base_intensity) / (target_year - base_year)
        return base_intensity * math.exp(k * (current_year - base_year))
    elif method == "s_curve":
        t_mid = (target_year + base_year) / 2
        k = 0.3
        sigmoid = 1.0 / (1.0 + math.exp(-k * (current_year - t_mid)))
        sigmoid_base = 1.0 / (1.0 + math.exp(-k * (base_year - t_mid)))
        sigmoid_target = 1.0 / (1.0 + math.exp(-k * (target_year - t_mid)))
        norm = (sigmoid - sigmoid_base) / (sigmoid_target - sigmoid_base)
        return base_intensity + (target_intensity - base_intensity) * norm
    elif method == "stepped":
        steps = max(1, (target_year - base_year) // 5)
        step_idx = min(steps, (current_year - base_year) // 5)
        step_size = (target_intensity - base_intensity) / steps
        return base_intensity + step_size * step_idx
    return base_intensity + (target_intensity - base_intensity) * t

def _sector_lookup(sector_id: str) -> Optional[Dict[str, str]]:
    """Look up sector definition by ID."""
    for s in SDA_SECTORS:
        if s["id"] == sector_id:
            return s
    return None

def _scenario_lookup(scenario_id: str) -> Optional[Dict[str, str]]:
    """Look up IEA scenario by ID."""
    for s in IEA_SCENARIOS:
        if s["id"] == scenario_id:
            return s
    return None

# ---------------------------------------------------------------------------
# Template Class
# ---------------------------------------------------------------------------
class SectorPathwayReportTemplate:
    """
    Sector-specific decarbonization pathway report template.

    Renders a comprehensive sector pathway report with SBTi SDA alignment,
    IEA NZE pathway integration, convergence curve analysis, gap analysis,
    and year-by-year intensity targets. Supports Markdown, HTML, JSON,
    and PDF-ready output with XBRL tagging and SHA-256 provenance.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.

    Example:
        >>> template = SectorPathwayReportTemplate()
        >>> data = {
        ...     "org_name": "SteelCorp AG",
        ...     "sector_id": "steel",
        ...     "base_year": 2022,
        ...     "target_year": 2050,
        ...     "base_intensity": 1.85,
        ...     "current_intensity": 1.72,
        ...     "current_year": 2025,
        ...     "scenario": "nze",
        ...     "convergence_method": "linear",
        ... }
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> js = template.render_json(data)
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
            self._md_sector_classification(data),
            self._md_intensity_baseline(data),
            self._md_sda_alignment(data),
            self._md_iea_pathway(data),
            self._md_convergence_curve(data),
            self._md_year_by_year(data),
            self._md_activity_growth(data),
            self._md_abs_vs_intensity(data),
            self._md_gap_analysis(data),
            self._md_regional_variants(data),
            self._md_confidence(data),
            self._md_xbrl_tags(data),
            self._md_audit_trail(data),
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
            self._html_sector_classification(data),
            self._html_intensity_baseline(data),
            self._html_sda_alignment(data),
            self._html_iea_pathway(data),
            self._html_convergence_curve(data),
            self._html_year_by_year(data),
            self._html_activity_growth(data),
            self._html_abs_vs_intensity(data),
            self._html_gap_analysis(data),
            self._html_regional_variants(data),
            self._html_confidence(data),
            self._html_xbrl_tags(data),
            self._html_audit_trail(data),
            self._html_footer(data),
        ]
        body = "\n".join(body_parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Sector Pathway Report - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured JSON."""
        self.generated_at = utcnow()
        sector_id = data.get("sector_id", "")
        sector_def = _sector_lookup(sector_id) or {}
        scenario_id = data.get("scenario", "nze")
        scenario_def = _scenario_lookup(scenario_id) or {}

        base_year = int(data.get("base_year", 2022))
        target_year = int(data.get("target_year", 2050))
        current_year = int(data.get("current_year", 2025))
        base_intensity = float(data.get("base_intensity", 0))
        current_intensity = float(data.get("current_intensity", 0))
        target_intensity_2050 = float(data.get("target_intensity_2050", 0))
        convergence_method = data.get("convergence_method", "linear")

        pathway_targets = self._build_pathway_targets(
            base_intensity, target_intensity_2050, base_year, target_year, convergence_method
        )

        pathway_val = _convergence_value(
            base_intensity, target_intensity_2050, base_year, target_year,
            current_year, convergence_method
        )
        gap_abs = current_intensity - pathway_val if pathway_val > 0 else 0
        gap_pct = (gap_abs / pathway_val * 100) if pathway_val > 0 else 0

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "sector": {
                "id": sector_id,
                "name": sector_def.get("name", sector_id),
                "intensity_metric": sector_def.get("metric", ""),
                "sda_methodology": sector_def.get("sda", ""),
            },
            "scenario": {
                "id": scenario_id,
                "name": scenario_def.get("name", scenario_id),
                "temperature": scenario_def.get("temp", ""),
                "probability": scenario_def.get("probability", ""),
            },
            "baseline": {
                "base_year": base_year,
                "base_intensity": str(base_intensity),
                "current_year": current_year,
                "current_intensity": str(current_intensity),
                "intensity_change_pct": str(_pct_change(current_intensity, base_intensity)),
            },
            "pathway": {
                "target_year": target_year,
                "target_intensity_2030": str(data.get("target_intensity_2030", "")),
                "target_intensity_2050": str(target_intensity_2050),
                "convergence_method": convergence_method,
                "year_by_year": pathway_targets,
            },
            "gap_analysis": {
                "pathway_intensity_current_year": str(round(pathway_val, 4)),
                "actual_intensity": str(current_intensity),
                "gap_absolute": str(round(gap_abs, 4)),
                "gap_percentage": str(round(gap_pct, 2)),
                "rag_status": _rag_status(gap_pct),
                "required_acceleration_pct_per_year": str(
                    round(gap_abs / max(1, target_year - current_year), 4)
                ),
            },
            "sda_alignment": {
                "eligible": sector_def.get("sda", "").startswith("SDA"),
                "methodology": sector_def.get("sda", "N/A"),
                "coverage_scope1_2": str(data.get("coverage_scope1_2_pct", 95)),
                "alignment_status": data.get("sda_alignment_status", "pending"),
            },
            "iea_alignment": {
                "scenario": scenario_def.get("name", ""),
                "milestone_count": data.get("iea_milestone_count", 0),
                "milestones_on_track": data.get("iea_milestones_on_track", 0),
                "alignment_pct": str(data.get("iea_alignment_pct", 0)),
            },
            "regional_variants": data.get("regional_variants", []),
            "confidence": {
                "uncertainty_range_pct": str(data.get("uncertainty_range_pct", 10)),
                "data_quality_score": str(data.get("data_quality_score", "medium")),
                "model_confidence": str(data.get("model_confidence", "medium")),
            },
            "xbrl_tags": {
                tag: XBRL_TAGS[tag] for tag in XBRL_TAGS
            },
            "citations": data.get("citations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready structured data (render via external PDF engine)."""
        json_data = self.render_json(data)
        html_content = self.render_html(data)
        return {
            "format": "pdf",
            "html_content": html_content,
            "structured_data": json_data,
            "metadata": {
                "title": f"Sector Pathway Report - {data.get('org_name', '')}",
                "author": "GreenLang PACK-028",
                "subject": "Sector Decarbonization Pathway",
                "creator": f"GreenLang v{_MODULE_VERSION}",
            },
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _build_pathway_targets(
        self,
        base_intensity: float,
        target_intensity: float,
        base_year: int,
        target_year: int,
        method: str,
    ) -> List[Dict[str, Any]]:
        """Build year-by-year pathway target table."""
        targets = []
        for yr in range(base_year, target_year + 1):
            val = _convergence_value(
                base_intensity, target_intensity, base_year, target_year, yr, method
            )
            reduction_from_base = (
                ((base_intensity - val) / base_intensity * 100) if base_intensity > 0 else 0
            )
            targets.append({
                "year": yr,
                "intensity_target": round(val, 4),
                "reduction_from_base_pct": round(reduction_from_base, 2),
            })
        return targets

    def _get_sector_info(self, data: Dict[str, Any]) -> Dict[str, str]:
        sector_id = data.get("sector_id", "")
        return _sector_lookup(sector_id) or {
            "id": sector_id, "name": sector_id, "metric": "tCO2e/unit", "sda": "N/A"
        }

    def _get_scenario_info(self, data: Dict[str, Any]) -> Dict[str, str]:
        scenario_id = data.get("scenario", "nze")
        return _scenario_lookup(scenario_id) or {
            "id": scenario_id, "name": scenario_id, "temp": "N/A", "probability": "N/A"
        }

    # ------------------------------------------------------------------
    # Markdown Sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        sector = self._get_sector_info(data)
        scenario = self._get_scenario_info(data)
        return (
            f"# Sector Pathway Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Sector:** {sector['name']}  \n"
            f"**Scenario:** {scenario['name']} ({scenario['temp']})  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-028 Sector Pathway Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        scenario = self._get_scenario_info(data)
        base_yr = data.get("base_year", 2022)
        current_yr = data.get("current_year", 2025)
        base_int = float(data.get("base_intensity", 0))
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        change_pct = float(_pct_change(current_int, base_int))
        pathway_val = _convergence_value(
            base_int, target_2050,
            int(base_yr), int(data.get("target_year", 2050)),
            int(current_yr), data.get("convergence_method", "linear"),
        )
        gap_pct = ((current_int - pathway_val) / pathway_val * 100) if pathway_val > 0 else 0
        rag = _rag_status(gap_pct)
        lines = [
            "## 1. Executive Summary\n",
            f"This report presents the sector-specific decarbonization pathway for "
            f"**{data.get('org_name', '')}** in the **{sector['name']}** sector, "
            f"aligned with the **{scenario['name']}** scenario ({scenario['temp']}).\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Sector | {sector['name']} |",
            f"| SDA Methodology | {sector['sda']} |",
            f"| Intensity Metric | {sector['metric']} |",
            f"| Base Year ({base_yr}) Intensity | {_dec(base_int, 4)} {sector['metric']} |",
            f"| Current Year ({current_yr}) Intensity | {_dec(current_int, 4)} {sector['metric']} |",
            f"| Change from Base Year | {_dec(change_pct)}% |",
            f"| 2050 Target Intensity | {_dec(target_2050, 4)} {sector['metric']} |",
            f"| Gap to Pathway | {_dec(gap_pct)}% |",
            f"| RAG Status | **{rag}** |",
            f"| Scenario | {scenario['name']} ({scenario['temp']}) |",
        ]
        return "\n".join(lines)

    def _md_sector_classification(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        classification = data.get("classification", {})
        lines = [
            "## 2. Sector Classification\n",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Primary Sector | {sector['name']} |",
            f"| Sector ID | {sector['id']} |",
            f"| SDA Eligibility | {'Yes' if sector['sda'].startswith('SDA') else 'No (IEA pathway only)'} |",
            f"| NACE Rev.2 Code | {classification.get('nace_code', 'N/A')} |",
            f"| GICS Code | {classification.get('gics_code', 'N/A')} |",
            f"| ISIC Rev.4 Code | {classification.get('isic_code', 'N/A')} |",
            f"| Sub-Sectors | {', '.join(classification.get('sub_sectors', []))} |",
            f"| Revenue Share in Sector | {_dec(classification.get('revenue_share_pct', 100))}% |",
        ]
        return "\n".join(lines)

    def _md_intensity_baseline(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        base_int = data.get("base_intensity", 0)
        current_int = data.get("current_intensity", 0)
        historical = data.get("intensity_history", [])
        lines = [
            "## 3. Intensity Baseline Profile\n",
            f"**Intensity Metric:** {sector['metric']}  \n"
            f"**Base Year Intensity:** {_dec(base_int, 4)} {sector['metric']}  \n"
            f"**Current Intensity:** {_dec(current_int, 4)} {sector['metric']}\n",
        ]
        if historical:
            lines.append("### Historical Intensity Trend\n")
            lines.append("| Year | Intensity | YoY Change (%) |")
            lines.append("|------|----------:|---------------:|")
            prev = None
            for h in historical:
                yr = h.get("year", "")
                val = float(h.get("intensity", 0))
                if prev is not None:
                    yoy = float(_pct_change(val, prev))
                    lines.append(f"| {yr} | {_dec(val, 4)} | {_dec(yoy)}% |")
                else:
                    lines.append(f"| {yr} | {_dec(val, 4)} | - |")
                prev = val
        return "\n".join(lines)

    def _md_sda_alignment(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        sda = data.get("sda_alignment", {})
        lines = [
            "## 4. SBTi SDA Pathway Alignment\n",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| SDA Methodology | {sector['sda']} |",
            f"| SDA Eligibility | {'Eligible' if sector['sda'].startswith('SDA') else 'Not Eligible'} |",
            f"| Coverage (Scope 1+2) | {_dec(sda.get('coverage_pct', 95))}% |",
            f"| Coverage Requirement | 95% (SBTi minimum) |",
            f"| Coverage Status | {'PASS' if float(sda.get('coverage_pct', 95)) >= 95 else 'FAIL'} |",
            f"| Alignment Status | {sda.get('alignment_status', 'Pending')} |",
            f"| Ambition Level | {sda.get('ambition_level', '1.5C-aligned')} |",
            f"| Near-Term Target (2030) | {_dec(sda.get('near_term_target', 0), 4)} {sector['metric']} |",
            f"| Long-Term Target (2050) | {_dec(sda.get('long_term_target', 0), 4)} {sector['metric']} |",
            f"| Annual Reduction Rate | {_dec(sda.get('annual_reduction_rate', 4.2))}% |",
        ]
        criteria = sda.get("validation_criteria", [])
        if criteria:
            lines.append("\n### SDA Validation Checklist\n")
            lines.append("| # | Criterion | Status | Notes |")
            lines.append("|---|-----------|--------|-------|")
            for i, cr in enumerate(criteria, 1):
                status = cr.get("status", "pending")
                icon = "PASS" if status == "pass" else ("FAIL" if status == "fail" else "PENDING")
                lines.append(f"| {i} | {cr.get('criterion', '')} | {icon} | {cr.get('notes', '')} |")
        return "\n".join(lines)

    def _md_iea_pathway(self, data: Dict[str, Any]) -> str:
        scenario = self._get_scenario_info(data)
        iea = data.get("iea_pathway", {})
        milestones = iea.get("milestones", [])
        lines = [
            "## 5. IEA NZE Pathway Integration\n",
            f"**Scenario:** {scenario['name']}  \n"
            f"**Temperature Target:** {scenario['temp']}  \n"
            f"**Total Milestones Mapped:** {len(milestones)}  \n"
            f"**Milestones On Track:** {sum(1 for m in milestones if m.get('status') == 'on_track')}  \n"
            f"**Milestones Off Track:** {sum(1 for m in milestones if m.get('status') == 'off_track')}\n",
        ]
        if milestones:
            lines.append("### Key IEA Milestones\n")
            lines.append("| Year | Milestone | Status | Gap |")
            lines.append("|------|-----------|--------|-----|")
            for m in milestones[:20]:
                status = m.get("status", "pending")
                icon = "ON TRACK" if status == "on_track" else (
                    "OFF TRACK" if status == "off_track" else "PENDING"
                )
                lines.append(
                    f"| {m.get('year', '')} | {m.get('description', '')} | {icon} | {m.get('gap', '-')} |"
                )
        return "\n".join(lines)

    def _md_convergence_curve(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        method = data.get("convergence_method", "linear")
        base_int = float(data.get("base_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        base_yr = int(data.get("base_year", 2022))
        target_yr = int(data.get("target_year", 2050))
        lines = [
            "## 6. Convergence Curve Analysis\n",
            f"**Convergence Method:** {method.replace('_', ' ').title()}  \n"
            f"**Base Intensity ({base_yr}):** {_dec(base_int, 4)} {sector['metric']}  \n"
            f"**Target Intensity ({target_yr}):** {_dec(target_2050, 4)} {sector['metric']}\n",
            "### Convergence Comparison (All Methods)\n",
            "| Year | Linear | Exponential | S-Curve | Stepped |",
            "|------|-------:|------------:|--------:|--------:|",
        ]
        sample_years = list(range(base_yr, target_yr + 1, 5))
        if target_yr not in sample_years:
            sample_years.append(target_yr)
        for yr in sample_years:
            vals = []
            for m in CONVERGENCE_TYPES:
                v = _convergence_value(base_int, target_2050, base_yr, target_yr, yr, m)
                vals.append(_dec(v, 4))
            lines.append(f"| {yr} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
        lines.append(
            f"\n**Selected Method:** {method.replace('_', ' ').title()} "
            f"(highlighted in pathway targets)"
        )
        return "\n".join(lines)

    def _md_year_by_year(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        base_int = float(data.get("base_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        base_yr = int(data.get("base_year", 2022))
        target_yr = int(data.get("target_year", 2050))
        method = data.get("convergence_method", "linear")
        targets = self._build_pathway_targets(base_int, target_2050, base_yr, target_yr, method)
        lines = [
            "## 7. Year-by-Year Pathway Targets\n",
            f"| Year | Target Intensity ({sector['metric']}) | Reduction from Base (%) |",
            f"|------|--------------------------------------:|------------------------:|",
        ]
        for t in targets:
            lines.append(
                f"| {t['year']} | {_dec(t['intensity_target'], 4)} | {_dec(t['reduction_from_base_pct'])}% |"
            )
        return "\n".join(lines)

    def _md_activity_growth(self, data: Dict[str, Any]) -> str:
        growth = data.get("activity_growth", {})
        projections = growth.get("projections", [])
        lines = [
            "## 8. Activity Growth Projections\n",
            f"**Activity Metric:** {growth.get('metric', 'Production volume')}  \n"
            f"**Base Year Activity:** {_dec_comma(growth.get('base_activity', 0))}  \n"
            f"**CAGR Assumption:** {_dec(growth.get('cagr_pct', 0))}%\n",
        ]
        if projections:
            lines.append("| Year | Activity Volume | Growth (%) | Abs. Emissions (tCO2e) |")
            lines.append("|------|----------------:|-----------:|-----------------------:|")
            for p in projections:
                lines.append(
                    f"| {p.get('year', '')} | {_dec_comma(p.get('activity', 0))} "
                    f"| {_dec(p.get('growth_pct', 0))}% | {_dec_comma(p.get('absolute_emissions', 0))} |"
                )
        else:
            lines.append("_No activity growth projections provided._")
        return "\n".join(lines)

    def _md_abs_vs_intensity(self, data: Dict[str, Any]) -> str:
        abs_trajectory = data.get("absolute_trajectory", [])
        lines = [
            "## 9. Absolute vs. Intensity Trajectory\n",
            "This section compares absolute emission reductions against intensity "
            "pathway reductions, accounting for activity growth.\n",
        ]
        if abs_trajectory:
            lines.append("| Year | Absolute (tCO2e) | Intensity | Activity | ACA Target | SDA Target |")
            lines.append("|------|------------------:|----------:|---------:|-----------:|-----------:|")
            for row in abs_trajectory:
                lines.append(
                    f"| {row.get('year', '')} | {_dec_comma(row.get('absolute', 0))} "
                    f"| {_dec(row.get('intensity', 0), 4)} "
                    f"| {_dec_comma(row.get('activity', 0))} "
                    f"| {_dec_comma(row.get('aca_target', 0))} "
                    f"| {_dec(row.get('sda_target', 0), 4)} |"
                )
        else:
            lines.append("_Trajectory data will be calculated when activity growth data is provided._")
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        base_int = float(data.get("base_intensity", 0))
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        base_yr = int(data.get("base_year", 2022))
        target_yr = int(data.get("target_year", 2050))
        current_yr = int(data.get("current_year", 2025))
        method = data.get("convergence_method", "linear")
        pathway_val = _convergence_value(
            base_int, target_2050, base_yr, target_yr, current_yr, method
        )
        gap_abs = current_int - pathway_val
        gap_pct = (gap_abs / pathway_val * 100) if pathway_val > 0 else 0
        rag = _rag_status(abs(gap_pct))
        accel = gap_abs / max(1, target_yr - current_yr)
        years_delay = abs(gap_abs) / max(0.001, abs(
            (target_2050 - base_int) / max(1, target_yr - base_yr)
        )) if base_int != target_2050 else 0
        lines = [
            "## 10. Gap-to-Pathway Analysis\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Current Intensity ({current_yr}) | {_dec(current_int, 4)} {sector['metric']} |",
            f"| Pathway Target ({current_yr}) | {_dec(pathway_val, 4)} {sector['metric']} |",
            f"| Absolute Gap | {_dec(gap_abs, 4)} {sector['metric']} |",
            f"| Percentage Gap | {_dec(gap_pct)}% |",
            f"| RAG Status | **{rag}** |",
            f"| Required Acceleration | {_dec(accel, 4)} {sector['metric']}/year additional reduction |",
            f"| Equivalent Delay | ~{_dec(years_delay, 1)} years behind pathway |",
            f"| Convergence Method | {method.replace('_', ' ').title()} |",
        ]
        recommendations = data.get("gap_recommendations", [])
        if recommendations:
            lines.append("\n### Recommended Actions to Close Gap\n")
            for i, rec in enumerate(recommendations, 1):
                lines.append(
                    f"{i}. **{rec.get('action', '')}** - {rec.get('description', '')} "
                    f"(Impact: {_dec(rec.get('impact_pct', 0))}% gap closure)"
                )
        return "\n".join(lines)

    def _md_regional_variants(self, data: Dict[str, Any]) -> str:
        variants = data.get("regional_variants", [])
        lines = [
            "## 11. Regional Pathway Variants\n",
        ]
        if variants:
            lines.append("| Region | 2030 Target | 2050 Target | Pathway Type | Notes |")
            lines.append("|--------|------------:|------------:|-------------|-------|")
            for v in variants:
                lines.append(
                    f"| {v.get('region', '')} | {_dec(v.get('target_2030', 0), 4)} "
                    f"| {_dec(v.get('target_2050', 0), 4)} "
                    f"| {v.get('pathway_type', 'Global')} | {v.get('notes', '')} |"
                )
        else:
            lines.append(
                "Regional pathway variants not specified. Using global convergence pathway.\n\n"
                "Available regions: OECD, Emerging Markets, Global, EU, North America, Asia-Pacific"
            )
        return "\n".join(lines)

    def _md_confidence(self, data: Dict[str, Any]) -> str:
        confidence = data.get("confidence", {})
        lines = [
            "## 12. Pathway Confidence & Uncertainty\n",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Uncertainty Range | +/- {_dec(confidence.get('uncertainty_range_pct', 10))}% |",
            f"| Data Quality Score | {confidence.get('data_quality_score', 'Medium')} |",
            f"| Model Confidence | {confidence.get('model_confidence', 'Medium')} |",
            f"| Scenario Sensitivity | {confidence.get('scenario_sensitivity', 'Moderate')} |",
            f"| Key Assumptions | {confidence.get('key_assumptions', 'Standard IEA/SBTi parameters')} |",
        ]
        risk_factors = confidence.get("risk_factors", [])
        if risk_factors:
            lines.append("\n### Risk Factors\n")
            for rf in risk_factors:
                lines.append(
                    f"- **{rf.get('factor', '')}** ({rf.get('likelihood', 'medium')} likelihood, "
                    f"{rf.get('impact', 'medium')} impact): {rf.get('description', '')}"
                )
        return "\n".join(lines)

    def _md_xbrl_tags(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 13. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |",
            "|------------|----------|-------|",
        ]
        sector = self._get_sector_info(data)
        lines.append(f"| Sector ID | {XBRL_TAGS['sector_id']} | {sector['id']} |")
        lines.append(
            f"| Base Year Intensity | {XBRL_TAGS['intensity_base']} | {_dec(data.get('base_intensity', 0), 4)} |"
        )
        lines.append(
            f"| Current Intensity | {XBRL_TAGS['intensity_current']} | {_dec(data.get('current_intensity', 0), 4)} |"
        )
        lines.append(
            f"| 2030 Target | {XBRL_TAGS['intensity_target_2030']} | {_dec(data.get('target_intensity_2030', 0), 4)} |"
        )
        lines.append(
            f"| 2050 Target | {XBRL_TAGS['intensity_target_2050']} | {_dec(data.get('target_intensity_2050', 0), 4)} |"
        )
        lines.append(f"| Scenario | {XBRL_TAGS['scenario']} | {data.get('scenario', 'nze')} |")
        lines.append(
            f"| SDA Alignment | {XBRL_TAGS['sda_alignment']} | {data.get('sda_alignment_status', 'pending')} |"
        )
        return "\n".join(lines)

    def _md_audit_trail(self, data: Dict[str, Any]) -> str:
        report_id = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        data_hash = _compute_hash(data)
        return (
            "## 14. Audit Trail & Provenance\n\n"
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
            f"*Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}*  \n"
            f"*SBTi SDA + IEA NZE aligned sector pathway analysis.*  \n"
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
            f".status-pass{{color:#1b5e20;font-weight:600;}}"
            f".status-fail{{color:#c62828;font-weight:600;}}"
            f".status-pending{{color:#ef6c00;font-style:italic;}}"
            f".pathway-bar{{height:8px;border-radius:4px;background:#e0e0e0;margin:2px 0;}}"
            f".pathway-fill{{height:8px;border-radius:4px;background:{_ACCENT};}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};"
            f"color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        sector = self._get_sector_info(data)
        scenario = self._get_scenario_info(data)
        return (
            f'<h1>Sector Pathway Report</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Sector:</strong> {sector["name"]} | '
            f'<strong>Scenario:</strong> {scenario["name"]} ({scenario["temp"]}) | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        base_int = float(data.get("base_intensity", 0))
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        change_pct = float(_pct_change(current_int, base_int))
        pathway_val = _convergence_value(
            base_int, target_2050,
            int(data.get("base_year", 2022)), int(data.get("target_year", 2050)),
            int(data.get("current_year", 2025)), data.get("convergence_method", "linear"),
        )
        gap_pct = ((current_int - pathway_val) / pathway_val * 100) if pathway_val > 0 else 0
        rag = _rag_status(abs(gap_pct))
        rag_cls = f"rag-{rag.lower()}"
        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Base Year Intensity</div>'
            f'<div class="card-value">{_dec(base_int, 4)}</div>'
            f'<div class="card-unit">{sector["metric"]}</div></div>\n'
            f'  <div class="card"><div class="card-label">Current Intensity</div>'
            f'<div class="card-value">{_dec(current_int, 4)}</div>'
            f'<div class="card-unit">{sector["metric"]}</div></div>\n'
            f'  <div class="card"><div class="card-label">Change from Base</div>'
            f'<div class="card-value">{_dec(change_pct)}%</div>'
            f'<div class="card-unit">vs base year</div></div>\n'
            f'  <div class="card"><div class="card-label">2050 Target</div>'
            f'<div class="card-value">{_dec(target_2050, 4)}</div>'
            f'<div class="card-unit">{sector["metric"]}</div></div>\n'
            f'  <div class="card"><div class="card-label">Gap to Pathway</div>'
            f'<div class="card-value">{_dec(gap_pct)}%</div>'
            f'<div class="card-unit"><span class="{rag_cls}">{rag}</span></div></div>\n'
            f'</div>'
        )

    def _html_sector_classification(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        classification = data.get("classification", {})
        return (
            f'<h2>2. Sector Classification</h2>\n'
            f'<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Primary Sector</td><td>{sector["name"]}</td></tr>\n'
            f'<tr><td>SDA Eligibility</td><td class="status-{"pass" if sector["sda"].startswith("SDA") else "pending"}">'
            f'{"Eligible" if sector["sda"].startswith("SDA") else "Not Eligible"}</td></tr>\n'
            f'<tr><td>NACE Code</td><td>{classification.get("nace_code", "N/A")}</td></tr>\n'
            f'<tr><td>GICS Code</td><td>{classification.get("gics_code", "N/A")}</td></tr>\n'
            f'<tr><td>ISIC Code</td><td>{classification.get("isic_code", "N/A")}</td></tr>\n'
            f'</table>'
        )

    def _html_intensity_baseline(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        historical = data.get("intensity_history", [])
        rows = ""
        prev = None
        for h in historical:
            val = float(h.get("intensity", 0))
            yoy = float(_pct_change(val, prev)) if prev else 0
            rows += (
                f'<tr><td>{h.get("year", "")}</td>'
                f'<td>{_dec(val, 4)}</td>'
                f'<td>{_dec(yoy) + "%" if prev else "-"}</td></tr>\n'
            )
            prev = val
        return (
            f'<h2>3. Intensity Baseline Profile</h2>\n'
            f'<p><strong>Metric:</strong> {sector["metric"]}</p>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Intensity ({sector["metric"]})</th><th>YoY Change</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_sda_alignment(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        sda = data.get("sda_alignment", {})
        coverage = float(sda.get("coverage_pct", 95))
        coverage_cls = "status-pass" if coverage >= 95 else "status-fail"
        criteria = sda.get("validation_criteria", [])
        criteria_rows = ""
        for i, cr in enumerate(criteria, 1):
            status = cr.get("status", "pending")
            cls = f"status-{status}"
            criteria_rows += (
                f'<tr><td>{i}</td><td>{cr.get("criterion", "")}</td>'
                f'<td class="{cls}">{status.upper()}</td>'
                f'<td>{cr.get("notes", "")}</td></tr>\n'
            )
        return (
            f'<h2>4. SBTi SDA Pathway Alignment</h2>\n'
            f'<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>SDA Methodology</td><td>{sector["sda"]}</td></tr>\n'
            f'<tr><td>Coverage (Scope 1+2)</td><td class="{coverage_cls}">{_dec(coverage)}%</td></tr>\n'
            f'<tr><td>Alignment Status</td><td>{sda.get("alignment_status", "Pending")}</td></tr>\n'
            f'<tr><td>Ambition Level</td><td>{sda.get("ambition_level", "1.5C-aligned")}</td></tr>\n'
            f'</table>\n'
            + (f'<h3>Validation Checklist</h3>\n<table>\n'
               f'<tr><th>#</th><th>Criterion</th><th>Status</th><th>Notes</th></tr>\n'
               f'{criteria_rows}</table>' if criteria_rows else '')
        )

    def _html_iea_pathway(self, data: Dict[str, Any]) -> str:
        scenario = self._get_scenario_info(data)
        iea = data.get("iea_pathway", {})
        milestones = iea.get("milestones", [])
        on_track = sum(1 for m in milestones if m.get("status") == "on_track")
        off_track = sum(1 for m in milestones if m.get("status") == "off_track")
        rows = ""
        for m in milestones[:20]:
            status = m.get("status", "pending")
            cls = "status-pass" if status == "on_track" else (
                "status-fail" if status == "off_track" else "status-pending"
            )
            rows += (
                f'<tr><td>{m.get("year", "")}</td>'
                f'<td>{m.get("description", "")}</td>'
                f'<td class="{cls}">{status.replace("_", " ").upper()}</td>'
                f'<td>{m.get("gap", "-")}</td></tr>\n'
            )
        return (
            f'<h2>5. IEA NZE Pathway Integration</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Scenario</div>'
            f'<div class="card-value">{scenario["name"]}</div></div>\n'
            f'  <div class="card"><div class="card-label">On Track</div>'
            f'<div class="card-value">{on_track}</div></div>\n'
            f'  <div class="card"><div class="card-label">Off Track</div>'
            f'<div class="card-value">{off_track}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Milestone</th><th>Status</th><th>Gap</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_convergence_curve(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        base_int = float(data.get("base_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        base_yr = int(data.get("base_year", 2022))
        target_yr = int(data.get("target_year", 2050))
        method = data.get("convergence_method", "linear")
        sample_years = list(range(base_yr, target_yr + 1, 5))
        if target_yr not in sample_years:
            sample_years.append(target_yr)
        rows = ""
        for yr in sample_years:
            vals = [
                _convergence_value(base_int, target_2050, base_yr, target_yr, yr, m)
                for m in CONVERGENCE_TYPES
            ]
            rows += (
                f'<tr><td>{yr}</td>'
                + "".join(f'<td>{_dec(v, 4)}</td>' for v in vals)
                + '</tr>\n'
            )
        return (
            f'<h2>6. Convergence Curve Analysis</h2>\n'
            f'<p><strong>Selected Method:</strong> {method.replace("_", " ").title()}</p>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Linear</th><th>Exponential</th>'
            f'<th>S-Curve</th><th>Stepped</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_year_by_year(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        base_int = float(data.get("base_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        base_yr = int(data.get("base_year", 2022))
        target_yr = int(data.get("target_year", 2050))
        method = data.get("convergence_method", "linear")
        targets = self._build_pathway_targets(base_int, target_2050, base_yr, target_yr, method)
        rows = ""
        for t in targets:
            pct = t["reduction_from_base_pct"]
            bar_width = min(100, max(0, pct))
            rows += (
                f'<tr><td>{t["year"]}</td>'
                f'<td>{_dec(t["intensity_target"], 4)}</td>'
                f'<td>{_dec(pct)}%</td>'
                f'<td><div class="pathway-bar"><div class="pathway-fill" '
                f'style="width:{bar_width}%"></div></div></td></tr>\n'
            )
        return (
            f'<h2>7. Year-by-Year Pathway Targets</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Target ({sector["metric"]})</th>'
            f'<th>Reduction</th><th>Progress</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_activity_growth(self, data: Dict[str, Any]) -> str:
        growth = data.get("activity_growth", {})
        projections = growth.get("projections", [])
        rows = ""
        for p in projections:
            rows += (
                f'<tr><td>{p.get("year", "")}</td>'
                f'<td>{_dec_comma(p.get("activity", 0))}</td>'
                f'<td>{_dec(p.get("growth_pct", 0))}%</td>'
                f'<td>{_dec_comma(p.get("absolute_emissions", 0))}</td></tr>\n'
            )
        return (
            f'<h2>8. Activity Growth Projections</h2>\n'
            f'<p><strong>Metric:</strong> {growth.get("metric", "Production volume")} | '
            f'<strong>CAGR:</strong> {_dec(growth.get("cagr_pct", 0))}%</p>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Activity Volume</th><th>Growth</th>'
            f'<th>Absolute Emissions (tCO2e)</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_abs_vs_intensity(self, data: Dict[str, Any]) -> str:
        abs_trajectory = data.get("absolute_trajectory", [])
        rows = ""
        for row in abs_trajectory:
            rows += (
                f'<tr><td>{row.get("year", "")}</td>'
                f'<td>{_dec_comma(row.get("absolute", 0))}</td>'
                f'<td>{_dec(row.get("intensity", 0), 4)}</td>'
                f'<td>{_dec_comma(row.get("activity", 0))}</td>'
                f'<td>{_dec_comma(row.get("aca_target", 0))}</td>'
                f'<td>{_dec(row.get("sda_target", 0), 4)}</td></tr>\n'
            )
        return (
            f'<h2>9. Absolute vs. Intensity Trajectory</h2>\n'
            f'<table>\n'
            f'<tr><th>Year</th><th>Absolute (tCO2e)</th><th>Intensity</th>'
            f'<th>Activity</th><th>ACA Target</th><th>SDA Target</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        sector = self._get_sector_info(data)
        base_int = float(data.get("base_intensity", 0))
        current_int = float(data.get("current_intensity", 0))
        target_2050 = float(data.get("target_intensity_2050", 0))
        base_yr = int(data.get("base_year", 2022))
        target_yr = int(data.get("target_year", 2050))
        current_yr = int(data.get("current_year", 2025))
        method = data.get("convergence_method", "linear")
        pathway_val = _convergence_value(
            base_int, target_2050, base_yr, target_yr, current_yr, method
        )
        gap_pct = ((current_int - pathway_val) / pathway_val * 100) if pathway_val > 0 else 0
        rag = _rag_status(abs(gap_pct))
        rag_cls = f"rag-{rag.lower()}"
        return (
            f'<h2>10. Gap-to-Pathway Analysis</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Pathway Target</div>'
            f'<div class="card-value">{_dec(pathway_val, 4)}</div>'
            f'<div class="card-unit">{sector["metric"]}</div></div>\n'
            f'  <div class="card"><div class="card-label">Actual</div>'
            f'<div class="card-value">{_dec(current_int, 4)}</div>'
            f'<div class="card-unit">{sector["metric"]}</div></div>\n'
            f'  <div class="card"><div class="card-label">Gap</div>'
            f'<div class="card-value">{_dec(gap_pct)}%</div>'
            f'<div class="card-unit"><span class="{rag_cls}">{rag}</span></div></div>\n'
            f'</div>'
        )

    def _html_regional_variants(self, data: Dict[str, Any]) -> str:
        variants = data.get("regional_variants", [])
        rows = ""
        for v in variants:
            rows += (
                f'<tr><td>{v.get("region", "")}</td>'
                f'<td>{_dec(v.get("target_2030", 0), 4)}</td>'
                f'<td>{_dec(v.get("target_2050", 0), 4)}</td>'
                f'<td>{v.get("pathway_type", "Global")}</td>'
                f'<td>{v.get("notes", "")}</td></tr>\n'
            )
        return (
            f'<h2>11. Regional Pathway Variants</h2>\n'
            f'<table>\n'
            f'<tr><th>Region</th><th>2030 Target</th><th>2050 Target</th>'
            f'<th>Type</th><th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_confidence(self, data: Dict[str, Any]) -> str:
        confidence = data.get("confidence", {})
        return (
            f'<h2>12. Pathway Confidence & Uncertainty</h2>\n'
            f'<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Uncertainty Range</td><td>+/- {_dec(confidence.get("uncertainty_range_pct", 10))}%</td></tr>\n'
            f'<tr><td>Data Quality</td><td>{confidence.get("data_quality_score", "Medium")}</td></tr>\n'
            f'<tr><td>Model Confidence</td><td>{confidence.get("model_confidence", "Medium")}</td></tr>\n'
            f'<tr><td>Scenario Sensitivity</td><td>{confidence.get("scenario_sensitivity", "Moderate")}</td></tr>\n'
            f'</table>'
        )

    def _html_xbrl_tags(self, data: Dict[str, Any]) -> str:
        rows = ""
        tag_values = {
            "sector_id": data.get("sector_id", ""),
            "intensity_base": _dec(data.get("base_intensity", 0), 4),
            "intensity_current": _dec(data.get("current_intensity", 0), 4),
            "intensity_target_2030": _dec(data.get("target_intensity_2030", 0), 4),
            "intensity_target_2050": _dec(data.get("target_intensity_2050", 0), 4),
            "scenario": data.get("scenario", "nze"),
            "sda_alignment": data.get("sda_alignment_status", "pending"),
        }
        for key, tag in XBRL_TAGS.items():
            val = tag_values.get(key, "")
            if val:
                rows += f'<tr><td>{key.replace("_", " ").title()}</td><td><code>{tag}</code></td><td>{val}</td></tr>\n'
        return (
            f'<h2>13. XBRL Tagging Summary</h2>\n'
            f'<table>\n'
            f'<tr><th>Data Point</th><th>XBRL Tag</th><th>Value</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_audit_trail(self, data: Dict[str, Any]) -> str:
        report_id = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        data_hash = _compute_hash(data)
        return (
            f'<h2>14. Audit Trail & Provenance</h2>\n'
            f'<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Report ID</td><td><code>{report_id}</code></td></tr>\n'
            f'<tr><td>Generated At</td><td>{ts}</td></tr>\n'
            f'<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n'
            f'<tr><td>Version</td><td>{_MODULE_VERSION}</td></tr>\n'
            f'<tr><td>Input Hash</td><td><code>{data_hash[:16]}...</code></td></tr>\n'
            f'<tr><td>Engine</td><td>Deterministic (zero-hallucination)</td></tr>\n'
            f'</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}<br>'
            f'SBTi SDA + IEA NZE aligned sector pathway analysis<br>'
            f'All calculations deterministic - zero LLM in computation path'
            f'</div>'
        )
