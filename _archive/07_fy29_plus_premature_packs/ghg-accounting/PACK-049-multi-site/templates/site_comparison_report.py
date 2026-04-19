# -*- coding: utf-8 -*-
"""
SiteComparisonReport - Cross-site benchmarking report for PACK-049.

Sections: league_table, statistics, distribution, gap_analysis,
          best_practices, trend, provenance.

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())
def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

class LeagueTableRow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rank: int = Field(0)
    site_id: str = Field("")
    site_name: str = Field("")
    peer_group: str = Field("")
    kpi_name: str = Field("")
    kpi_value: Decimal = Field(Decimal("0"))
    kpi_unit: str = Field("")
    performance_band: str = Field("")
    percentile: Decimal = Field(Decimal("0"))

class GroupStatistics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    peer_group: str = Field("")
    kpi_name: str = Field("")
    min_value: Decimal = Field(Decimal("0"))
    max_value: Decimal = Field(Decimal("0"))
    mean_value: Decimal = Field(Decimal("0"))
    median_value: Decimal = Field(Decimal("0"))
    std_dev: Decimal = Field(Decimal("0"))
    site_count: int = Field(0)

class GapAnalysisRow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_name: str = Field("")
    peer_group: str = Field("")
    kpi_name: str = Field("")
    site_value: Decimal = Field(Decimal("0"))
    best_value: Decimal = Field(Decimal("0"))
    gap_to_best: Decimal = Field(Decimal("0"))
    gap_pct: Decimal = Field(Decimal("0"))
    reduction_potential_tco2e: Decimal = Field(Decimal("0"))

class BestPracticeItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    peer_group: str = Field("")
    kpi_name: str = Field("")
    best_site: str = Field("")
    best_value: Decimal = Field(Decimal("0"))
    median_value: Decimal = Field(Decimal("0"))
    spread: Decimal = Field(Decimal("0"))

class TrendPoint(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    year: int = Field(0)
    site_name: str = Field("")
    kpi_value: Decimal = Field(Decimal("0"))
    kpi_name: str = Field("")

class ComparisonReportInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    league_table: List[Dict[str, Any]] = Field(default_factory=list)
    statistics: List[Dict[str, Any]] = Field(default_factory=list)
    gap_analysis: List[Dict[str, Any]] = Field(default_factory=list)
    best_practices: List[Dict[str, Any]] = Field(default_factory=list)
    trend_data: List[Dict[str, Any]] = Field(default_factory=list)
    total_reduction_potential_tco2e: Decimal = Field(Decimal("0"))

class ComparisonReportOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    league_table: List[LeagueTableRow] = Field(default_factory=list)
    statistics: List[GroupStatistics] = Field(default_factory=list)
    distribution_quartiles: Dict[str, int] = Field(default_factory=dict)
    gap_analysis: List[GapAnalysisRow] = Field(default_factory=list)
    best_practices: List[BestPracticeItem] = Field(default_factory=list)
    trend: List[TrendPoint] = Field(default_factory=list)
    total_reduction_potential_tco2e: Decimal = Field(Decimal("0"))
    provenance_hash: str = Field("")

class SiteComparisonReport:
    """Cross-site benchmarking report template."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def render(self, data: Dict[str, Any]) -> ComparisonReportOutput:
        start = time.monotonic()
        self.generated_at = utcnow()
        inp = ComparisonReportInput(**data) if isinstance(data, dict) else data

        league = [LeagueTableRow(**r) if isinstance(r, dict) else r for r in inp.league_table]
        stats = [GroupStatistics(**s) if isinstance(s, dict) else s for s in inp.statistics]
        gaps = [GapAnalysisRow(**g) if isinstance(g, dict) else g for g in inp.gap_analysis]
        bps = [BestPracticeItem(**b) if isinstance(b, dict) else b for b in inp.best_practices]
        trend = [TrendPoint(**t) if isinstance(t, dict) else t for t in inp.trend_data]

        # Quartile distribution
        quartiles: Dict[str, int] = {"top_quartile": 0, "second_quartile": 0, "third_quartile": 0, "bottom_quartile": 0}
        for r in league:
            band = r.performance_band
            if band in quartiles:
                quartiles[band] += 1

        total_reduction = inp.total_reduction_potential_tco2e
        if total_reduction == Decimal("0") and gaps:
            total_reduction = sum(g.reduction_potential_tco2e for g in gaps)

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = ComparisonReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            league_table=league, statistics=stats,
            distribution_quartiles=quartiles,
            gap_analysis=gaps, best_practices=bps,
            trend=trend,
            total_reduction_potential_tco2e=total_reduction,
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

    def export_markdown(self, r: ComparisonReportOutput) -> str:
        lines: List[str] = []
        lines.append(f"# Site Comparison Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Reduction Potential:** {r.total_reduction_potential_tco2e:,.2f} tCO2e")
        lines.append("")
        lines.append("## League Table")
        lines.append("| Rank | Site | Group | KPI | Value | Unit | Band | Percentile |")
        lines.append("|------|------|-------|-----|-------|------|------|------------|")
        for row in r.league_table:
            lines.append(f"| {row.rank} | {row.site_name} | {row.peer_group} | {row.kpi_name} | {row.kpi_value} | {row.kpi_unit} | {row.performance_band} | {row.percentile}% |")
        lines.append("")
        if r.statistics:
            lines.append("## Group Statistics")
            lines.append("| Group | KPI | Min | Max | Mean | Median | Std Dev | Sites |")
            lines.append("|-------|-----|-----|-----|------|--------|---------|-------|")
            for s in r.statistics:
                lines.append(f"| {s.peer_group} | {s.kpi_name} | {s.min_value} | {s.max_value} | {s.mean_value} | {s.median_value} | {s.std_dev} | {s.site_count} |")
            lines.append("")
        lines.append("## Quartile Distribution")
        for q, count in r.distribution_quartiles.items():
            lines.append(f"- {q}: {count}")
        lines.append("")
        if r.gap_analysis:
            lines.append("## Gap Analysis")
            lines.append("| Site | Group | KPI | Value | Best | Gap | Gap% | Reduction tCO2e |")
            lines.append("|------|-------|-----|-------|------|-----|------|-----------------|")
            for g in r.gap_analysis:
                lines.append(f"| {g.site_name} | {g.peer_group} | {g.kpi_name} | {g.site_value} | {g.best_value} | {g.gap_to_best} | {g.gap_pct}% | {g.reduction_potential_tco2e:,.2f} |")
            lines.append("")
        if r.best_practices:
            lines.append("## Best Practices")
            for b in r.best_practices:
                lines.append(f"- **{b.peer_group}** ({b.kpi_name}): Best = {b.best_site} ({b.best_value}), Median = {b.median_value}")
            lines.append("")
        lines.append(f"---\n*Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: ComparisonReportOutput) -> str:
        md = self.export_markdown(r)
        return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Comparison Report</title></head><body><pre>{md}</pre></body></html>"

    def export_json(self, r: ComparisonReportOutput) -> Dict[str, Any]:
        return json.loads(r.model_dump_json())

    def export_csv(self, r: ComparisonReportOutput) -> str:
        lines_out = ["rank,site_name,peer_group,kpi_name,kpi_value,performance_band"]
        for row in r.league_table:
            lines_out.append(f"{row.rank},{row.site_name},{row.peer_group},{row.kpi_name},{row.kpi_value},{row.performance_band}")
        return "\n".join(lines_out)

__all__ = ["SiteComparisonReport", "ComparisonReportInput", "ComparisonReportOutput"]
