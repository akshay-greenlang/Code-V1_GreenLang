# -*- coding: utf-8 -*-
"""
MultiSiteTrendReport - Year-over-year trend report for PACK-049.

Sections: corporate_trend, scope_trend, top_sites, improvement_leaders,
          structural_changes, provenance.

Author: GreenLang Team
Version: 49.0.0
"""

from __future__ import annotations

import hashlib, logging, uuid, json, time
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


class CorporateTrendPoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    year: int = Field(...)
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_tco2e: Decimal = Field(Decimal("0"))
    site_count: int = Field(0)
    yoy_change_pct: Optional[Decimal] = Field(None)

class ScopeTrendPoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    year: int = Field(...)
    scope: str = Field("")
    tco2e: Decimal = Field(Decimal("0"))
    share_pct: Decimal = Field(Decimal("0"))

class TopSiteTrend(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_name: str = Field("")
    current_year_tco2e: Decimal = Field(Decimal("0"))
    prior_year_tco2e: Decimal = Field(Decimal("0"))
    change_tco2e: Decimal = Field(Decimal("0"))
    change_pct: Decimal = Field(Decimal("0"))
    is_increasing: bool = Field(False)

class ImprovementLeader(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rank: int = Field(0)
    site_name: str = Field("")
    reduction_tco2e: Decimal = Field(Decimal("0"))
    reduction_pct: Decimal = Field(Decimal("0"))
    primary_driver: str = Field("")

class StructuralChange(BaseModel):
    change_type: str = Field("")
    description: str = Field("")
    impact_tco2e: Decimal = Field(Decimal("0"))
    year: int = Field(0)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class TrendReportInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    corporate_trend: List[Dict[str, Any]] = Field(default_factory=list)
    scope_trend: List[Dict[str, Any]] = Field(default_factory=list)
    top_sites: List[Dict[str, Any]] = Field(default_factory=list)
    improvement_leaders: List[Dict[str, Any]] = Field(default_factory=list)
    structural_changes: List[Dict[str, Any]] = Field(default_factory=list)

class TrendReportOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    corporate_trend: List[CorporateTrendPoint] = Field(default_factory=list)
    latest_total_tco2e: Decimal = Field(Decimal("0"))
    latest_yoy_pct: Optional[Decimal] = Field(None)
    cagr_pct: Optional[Decimal] = Field(None)
    scope_trend: List[ScopeTrendPoint] = Field(default_factory=list)
    top_sites: List[TopSiteTrend] = Field(default_factory=list)
    improvement_leaders: List[ImprovementLeader] = Field(default_factory=list)
    total_portfolio_reduction_tco2e: Decimal = Field(Decimal("0"))
    structural_changes: List[StructuralChange] = Field(default_factory=list)
    provenance_hash: str = Field("")


class MultiSiteTrendReport:
    """Multi-site year-over-year trend report template."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def render(self, data: Dict[str, Any]) -> TrendReportOutput:
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = TrendReportInput(**data) if isinstance(data, dict) else data

        corp_trend = [CorporateTrendPoint(**c) if isinstance(c, dict) else c for c in inp.corporate_trend]
        corp_trend.sort(key=lambda x: x.year)

        # Compute YoY for each year
        for i in range(1, len(corp_trend)):
            prev = corp_trend[i - 1].total_tco2e
            curr = corp_trend[i].total_tco2e
            if prev > Decimal("0"):
                corp_trend[i].yoy_change_pct = (
                    (curr - prev) / prev * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        latest_total = corp_trend[-1].total_tco2e if corp_trend else Decimal("0")
        latest_yoy = corp_trend[-1].yoy_change_pct if corp_trend else None

        # CAGR
        cagr = None
        if len(corp_trend) >= 2:
            first_val = corp_trend[0].total_tco2e
            last_val = corp_trend[-1].total_tco2e
            years = corp_trend[-1].year - corp_trend[0].year
            if first_val > Decimal("0") and years > 0:
                ratio = float(last_val / first_val)
                if ratio > 0:
                    cagr_val = (ratio ** (1.0 / years) - 1) * 100
                    cagr = Decimal(str(cagr_val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        scope_trend = [ScopeTrendPoint(**s) if isinstance(s, dict) else s for s in inp.scope_trend]
        top_sites = [TopSiteTrend(**t) if isinstance(t, dict) else t for t in inp.top_sites]
        for ts in top_sites:
            ts.is_increasing = ts.change_tco2e > Decimal("0")

        leaders = [ImprovementLeader(**l) if isinstance(l, dict) else l for l in inp.improvement_leaders]
        total_reduction = sum(l.reduction_tco2e for l in leaders)

        changes = [StructuralChange(**c) if isinstance(c, dict) else c for c in inp.structural_changes]

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = TrendReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            corporate_trend=corp_trend,
            latest_total_tco2e=latest_total,
            latest_yoy_pct=latest_yoy,
            cagr_pct=cagr,
            scope_trend=scope_trend,
            top_sites=top_sites,
            improvement_leaders=leaders,
            total_portfolio_reduction_tco2e=total_reduction,
            structural_changes=changes,
            provenance_hash=prov,
        )
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return output

    def render_markdown(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_markdown(r)
    def render_html(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_html(r)
    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_json(r)

    def export_markdown(self, r: TrendReportOutput) -> str:
        lines: List[str] = []
        lines.append(f"# Multi-Site Trend Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Latest Total:** {r.latest_total_tco2e:,.2f} tCO2e")
        if r.latest_yoy_pct is not None:
            lines.append(f"**YoY Change:** {r.latest_yoy_pct}%")
        if r.cagr_pct is not None:
            lines.append(f"**CAGR:** {r.cagr_pct}%")
        lines.append("")
        lines.append("## Corporate Trend")
        lines.append("| Year | S1 | S2 | S3 | Total | Sites | YoY% |")
        lines.append("|------|----|----|----|-------|-------|------|")
        for c in r.corporate_trend:
            yoy = f"{c.yoy_change_pct}%" if c.yoy_change_pct is not None else "N/A"
            lines.append(f"| {c.year} | {c.scope_1_tco2e:,.0f} | {c.scope_2_tco2e:,.0f} | {c.scope_3_tco2e:,.0f} | {c.total_tco2e:,.0f} | {c.site_count} | {yoy} |")
        lines.append("")
        if r.scope_trend:
            lines.append("## Scope Trend")
            lines.append("| Year | Scope | tCO2e | Share |")
            lines.append("|------|-------|-------|-------|")
            for s in r.scope_trend:
                lines.append(f"| {s.year} | {s.scope} | {s.tco2e:,.0f} | {s.share_pct}% |")
            lines.append("")
        if r.top_sites:
            lines.append("## Top Sites Movement")
            lines.append("| Site | Current | Prior | Change | %Change | Direction |")
            lines.append("|------|---------|-------|--------|---------|-----------|")
            for t in r.top_sites:
                direction = "UP" if t.is_increasing else "DOWN"
                lines.append(f"| {t.site_name} | {t.current_year_tco2e:,.0f} | {t.prior_year_tco2e:,.0f} | {t.change_tco2e:,.0f} | {t.change_pct}% | {direction} |")
            lines.append("")
        if r.improvement_leaders:
            lines.append(f"## Improvement Leaders (Total Reduction: {r.total_portfolio_reduction_tco2e:,.2f} tCO2e)")
            for l in r.improvement_leaders:
                lines.append(f"{l.rank}. **{l.site_name}**: -{l.reduction_tco2e:,.0f} tCO2e ({l.reduction_pct}%) -- {l.primary_driver}")
            lines.append("")
        if r.structural_changes:
            lines.append("## Structural Changes")
            for c in r.structural_changes:
                lines.append(f"- [{c.change_type}] {c.description} ({c.impact_tco2e:,.0f} tCO2e, {c.year})")
            lines.append("")
        lines.append(f"---\n*Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: TrendReportOutput) -> str:
        md = self.export_markdown(r)
        return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Trend Report</title></head><body><pre>{md}</pre></body></html>"

    def export_json(self, r: TrendReportOutput) -> Dict[str, Any]:
        return json.loads(r.model_dump_json())

    def export_csv(self, r: TrendReportOutput) -> str:
        lines_out = ["year,scope_1,scope_2,scope_3,total,sites,yoy_pct"]
        for c in r.corporate_trend:
            yoy = str(c.yoy_change_pct) if c.yoy_change_pct is not None else ""
            lines_out.append(f"{c.year},{c.scope_1_tco2e},{c.scope_2_tco2e},{c.scope_3_tco2e},{c.total_tco2e},{c.site_count},{yoy}")
        return "\n".join(lines_out)


__all__ = ["MultiSiteTrendReport", "TrendReportInput", "TrendReportOutput"]
