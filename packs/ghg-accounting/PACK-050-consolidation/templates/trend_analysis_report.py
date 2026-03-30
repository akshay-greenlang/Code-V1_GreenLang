# -*- coding: utf-8 -*-
"""
TrendAnalysisReport - Year-over-year consolidated trend analysis for PACK-050.

Generates multi-year trend analysis including absolute emissions trends,
intensity metric trends, target tracking, base year comparison, like-for-like
analysis (excluding M&A), and decomposition analysis.

Sections:
    1. Trend Summary (latest year, CAGR, target progress)
    2. Absolute Emissions Trend (annual S1/S2/S3/total by year)
    3. Intensity Metric Trends (tCO2e per revenue, per FTE, per sqm)
    4. Target Tracking (reduction targets vs actual progress)
    5. Base Year Comparison (current vs base year)
    6. Like-for-Like Analysis (excluding M&A structural changes)
    7. Decomposition Analysis (activity, intensity, structure effects)
    8. Provenance Footer

Output Formats: Markdown, HTML, JSON, CSV

Author: GreenLang Team
Version: 50.0.0
"""

from __future__ import annotations

import hashlib, logging, uuid, json, time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class AnnualEmissions(BaseModel):
    """Annual emission data point."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    year: int = Field(...)
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_location_tco2e: Decimal = Field(Decimal("0"))
    total_market_tco2e: Decimal = Field(Decimal("0"))
    entity_count: int = Field(0)
    yoy_change_pct: Optional[Decimal] = Field(None)

class IntensityMetric(BaseModel):
    """Intensity metric data point."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    year: int = Field(...)
    metric_name: str = Field("")
    metric_unit: str = Field("")
    numerator_tco2e: Decimal = Field(Decimal("0"))
    denominator_value: Decimal = Field(Decimal("0"))
    denominator_unit: str = Field("")
    intensity_value: Decimal = Field(Decimal("0"))
    yoy_change_pct: Optional[Decimal] = Field(None)

class TargetProgress(BaseModel):
    """Target tracking data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_name: str = Field("")
    target_type: str = Field("", description="absolute, intensity, science_based")
    base_year: int = Field(0)
    target_year: int = Field(0)
    base_value: Decimal = Field(Decimal("0"))
    target_value: Decimal = Field(Decimal("0"))
    current_value: Decimal = Field(Decimal("0"))
    required_reduction_pct: Decimal = Field(Decimal("0"))
    actual_reduction_pct: Decimal = Field(Decimal("0"))
    on_track: bool = Field(False)
    gap_pct: Decimal = Field(Decimal("0"))

class LikeForLikeLine(BaseModel):
    """Like-for-like comparison excluding M&A."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    year: int = Field(...)
    reported_tco2e: Decimal = Field(Decimal("0"))
    mna_adjustment_tco2e: Decimal = Field(Decimal("0"))
    like_for_like_tco2e: Decimal = Field(Decimal("0"))
    like_for_like_yoy_pct: Optional[Decimal] = Field(None)

class DecompositionEffect(BaseModel):
    """Decomposition analysis effect."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    effect_name: str = Field("", description="activity, intensity, structure, residual")
    effect_tco2e: Decimal = Field(Decimal("0"))
    effect_pct: Decimal = Field(Decimal("0"))
    description: str = Field("")

class TrendAnalysisReportInput(BaseModel):
    """Complete input for the trend analysis report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    annual_emissions: List[Dict[str, Any]] = Field(default_factory=list)
    intensity_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    targets: List[Dict[str, Any]] = Field(default_factory=list)
    like_for_like: List[Dict[str, Any]] = Field(default_factory=list)
    decomposition: List[Dict[str, Any]] = Field(default_factory=list)
    base_year: int = Field(0)
    base_year_tco2e: Decimal = Field(Decimal("0"))

# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class TrendAnalysisReportOutput(BaseModel):
    """Rendered trend analysis report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    years_covered: int = Field(0)
    latest_total_tco2e: Decimal = Field(Decimal("0"))
    latest_yoy_pct: Optional[Decimal] = Field(None)
    cagr_pct: Optional[Decimal] = Field(None)
    base_year: int = Field(0)
    base_year_tco2e: Decimal = Field(Decimal("0"))
    reduction_from_base_pct: Optional[Decimal] = Field(None)
    annual_emissions: List[AnnualEmissions] = Field(default_factory=list)
    intensity_metrics: List[IntensityMetric] = Field(default_factory=list)
    targets: List[TargetProgress] = Field(default_factory=list)
    targets_on_track: int = Field(0)
    targets_off_track: int = Field(0)
    like_for_like: List[LikeForLikeLine] = Field(default_factory=list)
    decomposition: List[DecompositionEffect] = Field(default_factory=list)
    provenance_hash: str = Field("")

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class TrendAnalysisReport:
    """
    Year-over-year consolidated trend analysis report for PACK-050.

    Produces multi-year trend data with intensity metrics, target tracking,
    like-for-like analysis, and decomposition of emission drivers.

    Example:
        >>> tpl = TrendAnalysisReport()
        >>> report = tpl.render(data)
        >>> md = tpl.export_markdown(report)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    # ------------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> TrendAnalysisReportOutput:
        """Render trend analysis from input data."""
        start = time.monotonic()
        self.generated_at = utcnow()
        inp = TrendAnalysisReportInput(**data) if isinstance(data, dict) else data

        annual = [AnnualEmissions(**a) if isinstance(a, dict) else a for a in inp.annual_emissions]
        annual.sort(key=lambda a: a.year)

        # Compute YoY for each year
        for i in range(1, len(annual)):
            prev_total = annual[i - 1].total_location_tco2e
            curr_total = annual[i].total_location_tco2e
            if prev_total > Decimal("0"):
                annual[i].yoy_change_pct = (
                    (curr_total - prev_total) / prev_total * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        latest_total = annual[-1].total_location_tco2e if annual else Decimal("0")
        latest_yoy = annual[-1].yoy_change_pct if annual else None

        # CAGR
        cagr = None
        if len(annual) >= 2:
            first = annual[0].total_location_tco2e
            last = annual[-1].total_location_tco2e
            years = annual[-1].year - annual[0].year
            if first > Decimal("0") and years > 0:
                ratio = float(last / first)
                if ratio > 0:
                    cagr_val = (ratio ** (1.0 / years) - 1) * 100
                    cagr = Decimal(str(cagr_val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Base year reduction
        reduction_from_base = None
        if inp.base_year_tco2e > Decimal("0") and latest_total > Decimal("0"):
            reduction_from_base = (
                (inp.base_year_tco2e - latest_total) / inp.base_year_tco2e * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        intensity = [IntensityMetric(**m) if isinstance(m, dict) else m for m in inp.intensity_metrics]
        targets = [TargetProgress(**t) if isinstance(t, dict) else t for t in inp.targets]

        # Compute target gap
        for t in targets:
            if t.gap_pct == Decimal("0"):
                t.gap_pct = (t.required_reduction_pct - t.actual_reduction_pct).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                t.on_track = t.actual_reduction_pct >= t.required_reduction_pct

        on_track = sum(1 for t in targets if t.on_track)
        off_track = sum(1 for t in targets if not t.on_track)

        lfl = [LikeForLikeLine(**l) if isinstance(l, dict) else l for l in inp.like_for_like]
        lfl.sort(key=lambda l: l.year)

        decomp = [DecompositionEffect(**d) if isinstance(d, dict) else d for d in inp.decomposition]

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = TrendAnalysisReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            years_covered=len(annual),
            latest_total_tco2e=latest_total,
            latest_yoy_pct=latest_yoy,
            cagr_pct=cagr,
            base_year=inp.base_year,
            base_year_tco2e=inp.base_year_tco2e,
            reduction_from_base_pct=reduction_from_base,
            annual_emissions=annual,
            intensity_metrics=intensity,
            targets=targets,
            targets_on_track=on_track,
            targets_off_track=off_track,
            like_for_like=lfl,
            decomposition=decomp,
            provenance_hash=prov,
        )
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return output

    # ------------------------------------------------------------------
    # CONVENIENCE RENDER METHODS
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_markdown(r)

    def render_html(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_html(r)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_json(r)

    # ------------------------------------------------------------------
    # EXPORT METHODS
    # ------------------------------------------------------------------

    def export_markdown(self, r: TrendAnalysisReportOutput) -> str:
        """Export report as Markdown."""
        lines: List[str] = []
        lines.append(f"# Trend Analysis Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Years:** {r.years_covered} | **Latest Total:** {r.latest_total_tco2e:,.2f} tCO2e")
        if r.latest_yoy_pct is not None:
            lines.append(f"**YoY Change:** {r.latest_yoy_pct}%")
        if r.cagr_pct is not None:
            lines.append(f"**CAGR:** {r.cagr_pct}%")
        if r.reduction_from_base_pct is not None:
            lines.append(f"**Reduction from Base Year ({r.base_year}):** {r.reduction_from_base_pct}%")
        lines.append("")

        # Absolute emissions trend
        if r.annual_emissions:
            lines.append("## Absolute Emissions Trend")
            lines.append("| Year | S1 | S2-Loc | S2-Mkt | S3 | Total-Loc | Total-Mkt | Entities | YoY% |")
            lines.append("|------|----|--------|--------|----|-----------|-----------|----------|------|")
            for a in r.annual_emissions:
                yoy = f"{a.yoy_change_pct}%" if a.yoy_change_pct is not None else "N/A"
                lines.append(
                    f"| {a.year} | {a.scope_1_tco2e:,.0f} | {a.scope_2_location_tco2e:,.0f} | "
                    f"{a.scope_2_market_tco2e:,.0f} | {a.scope_3_tco2e:,.0f} | "
                    f"{a.total_location_tco2e:,.0f} | {a.total_market_tco2e:,.0f} | "
                    f"{a.entity_count} | {yoy} |"
                )
            lines.append("")

        # Intensity metrics
        if r.intensity_metrics:
            lines.append("## Intensity Metric Trends")
            lines.append("| Year | Metric | Intensity | Unit | YoY% |")
            lines.append("|------|--------|-----------|------|------|")
            for m in r.intensity_metrics:
                yoy = f"{m.yoy_change_pct}%" if m.yoy_change_pct is not None else "N/A"
                lines.append(
                    f"| {m.year} | {m.metric_name} | {m.intensity_value:,.4f} | "
                    f"{m.metric_unit} | {yoy} |"
                )
            lines.append("")

        # Target tracking
        if r.targets:
            lines.append(f"## Target Tracking (On Track: {r.targets_on_track} | Off Track: {r.targets_off_track})")
            lines.append("| Target | Type | Base | Target | Current | Required% | Actual% | Gap% | Status |")
            lines.append("|--------|------|------|--------|---------|-----------|---------|------|--------|")
            for t in r.targets:
                status = "ON TRACK" if t.on_track else "OFF TRACK"
                lines.append(
                    f"| {t.target_name} | {t.target_type} | {t.base_value:,.0f} | "
                    f"{t.target_value:,.0f} | {t.current_value:,.0f} | {t.required_reduction_pct}% | "
                    f"{t.actual_reduction_pct}% | {t.gap_pct}% | {status} |"
                )
            lines.append("")

        # Like-for-like
        if r.like_for_like:
            lines.append("## Like-for-Like Analysis (Excluding M&A)")
            lines.append("| Year | Reported tCO2e | M&A Adj | Like-for-Like | YoY% |")
            lines.append("|------|---------------|---------|---------------|------|")
            for l in r.like_for_like:
                yoy = f"{l.like_for_like_yoy_pct}%" if l.like_for_like_yoy_pct is not None else "N/A"
                lines.append(
                    f"| {l.year} | {l.reported_tco2e:,.0f} | {l.mna_adjustment_tco2e:,.0f} | "
                    f"{l.like_for_like_tco2e:,.0f} | {yoy} |"
                )
            lines.append("")

        # Decomposition
        if r.decomposition:
            lines.append("## Decomposition Analysis")
            lines.append("| Effect | tCO2e | Share | Description |")
            lines.append("|--------|------:|------:|-------------|")
            for d in r.decomposition:
                lines.append(f"| {d.effect_name} | {d.effect_tco2e:,.2f} | {d.effect_pct}% | {d.description} |")
            lines.append("")

        lines.append(f"---\n*Report ID: {r.report_id} | Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: TrendAnalysisReportOutput) -> str:
        """Export report as HTML."""
        md = self.export_markdown(r)
        html_lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>Trend Analysis - {r.company_name}</title>",
            "<style>body{font-family:sans-serif;margin:2em;}table{border-collapse:collapse;width:100%;margin:1em 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f5f5f5;}</style>",
            "</head><body>",
            f"<pre>{md}</pre>",
            "</body></html>",
        ]
        return "\n".join(html_lines)

    def export_json(self, r: TrendAnalysisReportOutput) -> Dict[str, Any]:
        """Export report as JSON-serializable dict."""
        return json.loads(r.model_dump_json())

    def export_csv(self, r: TrendAnalysisReportOutput) -> str:
        """Export annual emissions as CSV."""
        lines_out = [
            "year,scope_1,scope_2_location,scope_2_market,scope_3,"
            "total_location,total_market,entity_count,yoy_pct"
        ]
        for a in r.annual_emissions:
            yoy = str(a.yoy_change_pct) if a.yoy_change_pct is not None else ""
            lines_out.append(
                f"{a.year},{a.scope_1_tco2e},{a.scope_2_location_tco2e},"
                f"{a.scope_2_market_tco2e},{a.scope_3_tco2e},"
                f"{a.total_location_tco2e},{a.total_market_tco2e},"
                f"{a.entity_count},{yoy}"
            )
        return "\n".join(lines_out)

__all__ = [
    "TrendAnalysisReport",
    "TrendAnalysisReportInput",
    "TrendAnalysisReportOutput",
    "AnnualEmissions",
    "IntensityMetric",
    "TargetProgress",
    "LikeForLikeLine",
    "DecompositionEffect",
]
