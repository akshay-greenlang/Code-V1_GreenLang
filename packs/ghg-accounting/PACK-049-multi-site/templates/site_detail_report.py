# -*- coding: utf-8 -*-
"""
SiteDetailReport - Individual site drill-down for PACK-049.

Generates a detailed single-site report with site overview, emissions
breakdown by source, intensity KPIs, year-over-year trends, data quality
scores, and emission factor assignments.

Sections:
    1. Site Overview (name, type, region, area, headcount, status)
    2. Emissions Breakdown (by source category and scope)
    3. Intensity KPIs (tCO2e/sqm, tCO2e/FTE, tCO2e/unit)
    4. YoY Trend (current vs prior year with % change)
    5. Data Quality (dimension scores, tier, findings)
    6. Factor Assignments (emission factors used, sources, tiers)

Author: GreenLang Team
Version: 49.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
import json
import time
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class EmissionSourceEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    source_category: str = Field(...)
    scope: str = Field("scope_1")
    activity_data: Decimal = Field(Decimal("0"))
    activity_unit: str = Field("")
    emission_factor: Decimal = Field(Decimal("0"))
    ef_unit: str = Field("")
    ef_source: str = Field("")
    ef_tier: str = Field("tier_2")
    emissions_tco2e: Decimal = Field(Decimal("0"))
    share_pct: Decimal = Field(Decimal("0"))


class IntensityKPI(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    kpi_name: str = Field(...)
    value: Decimal = Field(Decimal("0"))
    unit: str = Field("")
    benchmark: Optional[Decimal] = Field(None)
    vs_benchmark_pct: Optional[Decimal] = Field(None)


class YoYDataPoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    year: int = Field(...)
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_tco2e: Decimal = Field(Decimal("0"))


class QualityDimensionScore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dimension: str = Field(...)
    score: Decimal = Field(Decimal("0"))
    tier: str = Field("")
    findings: List[str] = Field(default_factory=list)


class FactorAssignment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    source_category: str = Field(...)
    factor_name: str = Field("")
    factor_value: Decimal = Field(Decimal("0"))
    factor_unit: str = Field("")
    source_db: str = Field("")
    tier: str = Field("tier_2")
    year: int = Field(0)
    region: str = Field("")


class SiteDetailInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    site_id: str = Field(...)
    site_name: str = Field("")
    facility_type: str = Field("")
    region: str = Field("")
    country_code: str = Field("")
    business_unit: str = Field("")
    floor_area_sqm: Decimal = Field(Decimal("0"))
    headcount: int = Field(0)
    operating_hours_yr: int = Field(0)
    status: str = Field("active")
    reporting_period: str = Field("")
    emission_sources: List[Dict[str, Any]] = Field(default_factory=list)
    intensity_kpis: List[Dict[str, Any]] = Field(default_factory=list)
    yoy_data: List[Dict[str, Any]] = Field(default_factory=list)
    quality_scores: List[Dict[str, Any]] = Field(default_factory=list)
    factor_assignments: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class SiteDetailOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    site_id: str = Field("")
    site_name: str = Field("")
    generated_at: str = Field("")
    total_tco2e: Decimal = Field(Decimal("0"))
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    sources: List[EmissionSourceEntry] = Field(default_factory=list)
    kpis: List[IntensityKPI] = Field(default_factory=list)
    yoy_trend: List[YoYDataPoint] = Field(default_factory=list)
    yoy_change_pct: Optional[Decimal] = Field(None)
    quality_dimensions: List[QualityDimensionScore] = Field(default_factory=list)
    overall_quality: Decimal = Field(Decimal("0"))
    quality_tier: str = Field("")
    factors: List[FactorAssignment] = Field(default_factory=list)
    provenance_hash: str = Field("")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class SiteDetailReport:
    """
    Single-site detail report template for PACK-049.

    Example:
        >>> tpl = SiteDetailReport()
        >>> report = tpl.render(data)
        >>> md = tpl.export_markdown(report)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def render(self, data: Dict[str, Any]) -> SiteDetailOutput:
        """Render site detail report."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = SiteDetailInput(**data) if isinstance(data, dict) else data

        # Parse sources
        sources: List[EmissionSourceEntry] = []
        for raw in inp.emission_sources:
            sources.append(EmissionSourceEntry(
                source_category=raw.get("source_category", ""),
                scope=raw.get("scope", "scope_1"),
                activity_data=self._dec(raw.get("activity_data", "0")),
                activity_unit=raw.get("activity_unit", ""),
                emission_factor=self._dec(raw.get("emission_factor", "0")),
                ef_unit=raw.get("ef_unit", ""),
                ef_source=raw.get("ef_source", ""),
                ef_tier=raw.get("ef_tier", "tier_2"),
                emissions_tco2e=self._dec(raw.get("emissions_tco2e", "0")),
            ))

        total = sum(s.emissions_tco2e for s in sources) or Decimal("1")
        for s in sources:
            s.share_pct = (s.emissions_tco2e / total * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        s1 = sum(s.emissions_tco2e for s in sources if "1" in s.scope)
        s2 = sum(s.emissions_tco2e for s in sources if "2" in s.scope)
        s3 = sum(s.emissions_tco2e for s in sources if "3" in s.scope)

        # KPIs
        kpis: List[IntensityKPI] = []
        for raw in inp.intensity_kpis:
            bm = self._dec(raw.get("benchmark")) if raw.get("benchmark") else None
            val = self._dec(raw.get("value", "0"))
            vs = None
            if bm and bm > Decimal("0"):
                vs = ((val - bm) / bm * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            kpis.append(IntensityKPI(
                kpi_name=raw.get("kpi_name", ""),
                value=val, unit=raw.get("unit", ""),
                benchmark=bm, vs_benchmark_pct=vs,
            ))

        # YoY
        yoy: List[YoYDataPoint] = []
        for raw in inp.yoy_data:
            yoy.append(YoYDataPoint(
                year=raw.get("year", 0),
                scope_1_tco2e=self._dec(raw.get("scope_1_tco2e", "0")),
                scope_2_tco2e=self._dec(raw.get("scope_2_tco2e", "0")),
                scope_3_tco2e=self._dec(raw.get("scope_3_tco2e", "0")),
                total_tco2e=self._dec(raw.get("total_tco2e", "0")),
            ))
        yoy.sort(key=lambda y: y.year)

        yoy_change = None
        if len(yoy) >= 2:
            prev = yoy[-2].total_tco2e
            curr = yoy[-1].total_tco2e
            if prev > Decimal("0"):
                yoy_change = ((curr - prev) / prev * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

        # Quality
        quality_dims: List[QualityDimensionScore] = []
        for raw in inp.quality_scores:
            quality_dims.append(QualityDimensionScore(
                dimension=raw.get("dimension", ""),
                score=self._dec(raw.get("score", "0")),
                tier=raw.get("tier", ""),
                findings=raw.get("findings", []),
            ))
        overall_q = Decimal("0")
        if quality_dims:
            overall_q = (sum(d.score for d in quality_dims) / Decimal(str(len(quality_dims)))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        q_tier = "tier_3"
        if overall_q >= Decimal("90"): q_tier = "tier_1"
        elif overall_q >= Decimal("75"): q_tier = "tier_2"
        elif overall_q >= Decimal("60"): q_tier = "tier_3"
        elif overall_q >= Decimal("40"): q_tier = "tier_4"
        else: q_tier = "tier_5"

        # Factors
        factors: List[FactorAssignment] = []
        for raw in inp.factor_assignments:
            factors.append(FactorAssignment(
                source_category=raw.get("source_category", ""),
                factor_name=raw.get("factor_name", ""),
                factor_value=self._dec(raw.get("factor_value", "0")),
                factor_unit=raw.get("factor_unit", ""),
                source_db=raw.get("source_db", ""),
                tier=raw.get("tier", "tier_2"),
                year=raw.get("year", 0),
                region=raw.get("region", ""),
            ))

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = SiteDetailOutput(
            site_id=inp.site_id, site_name=inp.site_name,
            generated_at=self.generated_at.isoformat(),
            total_tco2e=sum(s.emissions_tco2e for s in sources),
            scope_1_tco2e=s1, scope_2_tco2e=s2, scope_3_tco2e=s3,
            sources=sources, kpis=kpis, yoy_trend=yoy,
            yoy_change_pct=yoy_change,
            quality_dimensions=quality_dims,
            overall_quality=overall_q, quality_tier=q_tier,
            factors=factors, provenance_hash=prov,
        )
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return output

    def render_markdown(self, data: Dict[str, Any]) -> str:
        report = self.render(data) if isinstance(data, dict) else data
        return self.export_markdown(report)

    def render_html(self, data: Dict[str, Any]) -> str:
        report = self.render(data) if isinstance(data, dict) else data
        return self.export_html(report)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        report = self.render(data) if isinstance(data, dict) else data
        return self.export_json(report)

    def export_markdown(self, report: SiteDetailOutput) -> str:
        lines: List[str] = []
        lines.append(f"# Site Detail Report: {report.site_name}")
        lines.append(f"**Site ID:** {report.site_id} | **Generated:** {report.generated_at}")
        lines.append("")
        lines.append("## Emissions Summary")
        lines.append(f"| Scope | tCO2e |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Scope 1 | {report.scope_1_tco2e:,.2f} |")
        lines.append(f"| Scope 2 | {report.scope_2_tco2e:,.2f} |")
        lines.append(f"| Scope 3 | {report.scope_3_tco2e:,.2f} |")
        lines.append(f"| **Total** | **{report.total_tco2e:,.2f}** |")
        lines.append("")
        lines.append("## Source Breakdown")
        lines.append("| Source | Scope | Activity | EF | tCO2e | Share |")
        lines.append("|--------|-------|----------|-----|-------|-------|")
        for s in report.sources:
            lines.append(
                f"| {s.source_category} | {s.scope} | {s.activity_data} {s.activity_unit} | "
                f"{s.emission_factor} | {s.emissions_tco2e:,.2f} | {s.share_pct}% |"
            )
        lines.append("")
        if report.kpis:
            lines.append("## Intensity KPIs")
            lines.append("| KPI | Value | Unit | Benchmark | vs Benchmark |")
            lines.append("|-----|-------|------|-----------|-------------|")
            for k in report.kpis:
                bm = f"{k.benchmark}" if k.benchmark else "N/A"
                vs = f"{k.vs_benchmark_pct}%" if k.vs_benchmark_pct is not None else "N/A"
                lines.append(f"| {k.kpi_name} | {k.value} | {k.unit} | {bm} | {vs} |")
            lines.append("")
        if report.yoy_trend:
            lines.append("## Year-over-Year Trend")
            lines.append("| Year | S1 | S2 | S3 | Total |")
            lines.append("|------|----|----|----|-------|")
            for y in report.yoy_trend:
                lines.append(f"| {y.year} | {y.scope_1_tco2e:,.0f} | {y.scope_2_tco2e:,.0f} | {y.scope_3_tco2e:,.0f} | {y.total_tco2e:,.0f} |")
            if report.yoy_change_pct is not None:
                lines.append(f"\nYoY Change: **{report.yoy_change_pct}%**")
            lines.append("")
        lines.append(f"## Data Quality: {report.overall_quality}/100 ({report.quality_tier})")
        for d in report.quality_dimensions:
            lines.append(f"- {d.dimension}: {d.score}/100")
        lines.append("")
        lines.append(f"---\n*Provenance: {report.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, report: SiteDetailOutput) -> str:
        md = self.export_markdown(report)
        return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{report.site_name}</title></head><body><pre>{md}</pre></body></html>"

    def export_json(self, report: SiteDetailOutput) -> Dict[str, Any]:
        return json.loads(report.model_dump_json())

    def export_csv(self, report: SiteDetailOutput) -> str:
        lines = ["source_category,scope,emissions_tco2e,share_pct"]
        for s in report.sources:
            lines.append(f"{s.source_category},{s.scope},{s.emissions_tco2e},{s.share_pct}")
        return "\n".join(lines)

    def _dec(self, value: Any) -> Decimal:
        if value is None: return Decimal("0")
        try: return Decimal(str(value))
        except Exception: return Decimal("0")


__all__ = ["SiteDetailReport", "SiteDetailInput", "SiteDetailOutput"]
