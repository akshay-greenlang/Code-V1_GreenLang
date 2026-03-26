# -*- coding: utf-8 -*-
"""
DataQualityReport - Quality heatmap report for PACK-049.

Sections: heatmap, corporate_score, dimensions, tier_distribution,
          priorities, remediation, provenance.

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


class HeatmapCell(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_name: str = Field("")
    dimension: str = Field("")
    score: Decimal = Field(Decimal("0"))
    tier: str = Field("")
    color: str = Field("")

class DimensionSummary(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dimension: str = Field("")
    avg_score: Decimal = Field(Decimal("0"))
    min_score: Decimal = Field(Decimal("0"))
    max_score: Decimal = Field(Decimal("0"))
    sites_below_threshold: int = Field(0)

class SiteQualitySummary(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field("")
    site_name: str = Field("")
    overall_score: Decimal = Field(Decimal("0"))
    tier: str = Field("")
    dimensions_below: int = Field(0)
    top_gap_dimension: str = Field("")
    top_gap_score: Decimal = Field(Decimal("0"))

class PriorityAction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rank: int = Field(0)
    site_name: str = Field("")
    dimension: str = Field("")
    current_score: Decimal = Field(Decimal("0"))
    target_score: Decimal = Field(Decimal("70"))
    gap_points: Decimal = Field(Decimal("0"))
    action: str = Field("")
    estimated_hours: int = Field(0)
    priority: str = Field("medium")

class QualityReportInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    site_quality: List[Dict[str, Any]] = Field(default_factory=list)
    dimension_summaries: List[Dict[str, Any]] = Field(default_factory=list)
    priority_actions: List[Dict[str, Any]] = Field(default_factory=list)
    corporate_score: Decimal = Field(Decimal("0"))
    corporate_tier: str = Field("")

class QualityReportOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    corporate_score: Decimal = Field(Decimal("0"))
    corporate_tier: str = Field("")
    heatmap: List[HeatmapCell] = Field(default_factory=list)
    site_summaries: List[SiteQualitySummary] = Field(default_factory=list)
    dimension_summaries: List[DimensionSummary] = Field(default_factory=list)
    tier_distribution: Dict[str, int] = Field(default_factory=dict)
    priorities: List[PriorityAction] = Field(default_factory=list)
    total_remediation_hours: int = Field(0)
    sites_below_threshold: int = Field(0)
    provenance_hash: str = Field("")


class DataQualityReport:
    """Data quality heatmap and remediation report template."""

    DIMENSIONS = ["completeness", "accuracy", "consistency", "transparency", "timeliness", "relevance"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def render(self, data: Dict[str, Any]) -> QualityReportOutput:
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = QualityReportInput(**data) if isinstance(data, dict) else data

        heatmap: List[HeatmapCell] = []
        site_summaries: List[SiteQualitySummary] = []
        tier_dist: Dict[str, int] = {}

        for raw in inp.site_quality:
            sid = raw.get("site_id", "")
            sname = raw.get("site_name", "")
            scores: Dict[str, Decimal] = {}
            for dim in self.DIMENSIONS:
                score = self._dec(raw.get(dim, "50"))
                scores[dim] = score
                color = self._score_color(score)
                tier = self._score_tier(score)
                heatmap.append(HeatmapCell(
                    site_name=sname, dimension=dim,
                    score=score, tier=tier, color=color,
                ))

            overall = Decimal("0")
            if scores:
                overall = (sum(scores.values()) / Decimal(str(len(scores)))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            tier = self._score_tier(overall)
            tier_dist[tier] = tier_dist.get(tier, 0) + 1

            dims_below = sum(1 for s in scores.values() if s < Decimal("70"))
            top_gap_dim = ""
            top_gap_score = Decimal("100")
            for dim, sc in scores.items():
                if sc < top_gap_score:
                    top_gap_score = sc
                    top_gap_dim = dim

            site_summaries.append(SiteQualitySummary(
                site_id=sid, site_name=sname, overall_score=overall,
                tier=tier, dimensions_below=dims_below,
                top_gap_dimension=top_gap_dim, top_gap_score=top_gap_score,
            ))

        dim_summaries = [DimensionSummary(**d) if isinstance(d, dict) else d for d in inp.dimension_summaries]
        if not dim_summaries and heatmap:
            for dim in self.DIMENSIONS:
                dim_cells = [c for c in heatmap if c.dimension == dim]
                if dim_cells:
                    scores = [c.score for c in dim_cells]
                    avg = (sum(scores) / Decimal(str(len(scores)))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    dim_summaries.append(DimensionSummary(
                        dimension=dim, avg_score=avg,
                        min_score=min(scores), max_score=max(scores),
                        sites_below_threshold=sum(1 for s in scores if s < Decimal("70")),
                    ))

        priorities = [PriorityAction(**p) if isinstance(p, dict) else p for p in inp.priority_actions]
        total_hours = sum(p.estimated_hours for p in priorities)
        below_threshold = sum(1 for s in site_summaries if s.overall_score < Decimal("75"))

        corp_score = inp.corporate_score
        if corp_score == Decimal("0") and site_summaries:
            corp_score = (sum(s.overall_score for s in site_summaries) / Decimal(str(len(site_summaries)))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        corp_tier = inp.corporate_tier or self._score_tier(corp_score)

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = QualityReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            corporate_score=corp_score, corporate_tier=corp_tier,
            heatmap=heatmap, site_summaries=site_summaries,
            dimension_summaries=dim_summaries,
            tier_distribution=tier_dist,
            priorities=priorities,
            total_remediation_hours=total_hours,
            sites_below_threshold=below_threshold,
            provenance_hash=prov,
        )
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return output

    def _score_color(self, score: Decimal) -> str:
        if score >= Decimal("80"): return "#2a9d8f"
        elif score >= Decimal("60"): return "#e9c46a"
        else: return "#e76f51"

    def _score_tier(self, score: Decimal) -> str:
        if score >= Decimal("90"): return "tier_1"
        elif score >= Decimal("75"): return "tier_2"
        elif score >= Decimal("60"): return "tier_3"
        elif score >= Decimal("40"): return "tier_4"
        else: return "tier_5"

    def _dec(self, value: Any) -> Decimal:
        if value is None: return Decimal("0")
        try: return Decimal(str(value))
        except Exception: return Decimal("0")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_markdown(r)
    def render_html(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_html(r)
    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_json(r)

    def export_markdown(self, r: QualityReportOutput) -> str:
        lines: List[str] = []
        lines.append(f"# Data Quality Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Corporate Score:** {r.corporate_score}/100 ({r.corporate_tier})")
        lines.append(f"**Sites Below Threshold:** {r.sites_below_threshold}")
        lines.append("")
        lines.append("## Quality Heatmap")
        dims = self.DIMENSIONS
        header = "| Site | " + " | ".join(dims) + " | Overall |"
        lines.append(header)
        lines.append("|" + "------|" * (len(dims) + 2))
        for ss in r.site_summaries:
            cells = []
            for dim in dims:
                cell = next((c for c in r.heatmap if c.site_name == ss.site_name and c.dimension == dim), None)
                cells.append(f"{cell.score}" if cell else "N/A")
            lines.append(f"| {ss.site_name} | " + " | ".join(cells) + f" | {ss.overall_score} |")
        lines.append("")
        if r.dimension_summaries:
            lines.append("## Dimension Summary")
            lines.append("| Dimension | Avg | Min | Max | Below Threshold |")
            lines.append("|-----------|-----|-----|-----|-----------------|")
            for d in r.dimension_summaries:
                lines.append(f"| {d.dimension} | {d.avg_score} | {d.min_score} | {d.max_score} | {d.sites_below_threshold} |")
            lines.append("")
        lines.append("## Tier Distribution")
        for tier, count in sorted(r.tier_distribution.items()):
            lines.append(f"- {tier}: {count} sites")
        lines.append("")
        if r.priorities:
            lines.append(f"## Remediation Priorities ({r.total_remediation_hours} hours)")
            for p in r.priorities:
                lines.append(f"{p.rank}. [{p.priority.upper()}] {p.site_name} / {p.dimension}: {p.current_score} -> {p.target_score} ({p.gap_points} pts) -- {p.action} ({p.estimated_hours}h)")
            lines.append("")
        lines.append(f"---\n*Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: QualityReportOutput) -> str:
        md = self.export_markdown(r)
        return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Quality Report</title></head><body><pre>{md}</pre></body></html>"

    def export_json(self, r: QualityReportOutput) -> Dict[str, Any]:
        return json.loads(r.model_dump_json())

    def export_csv(self, r: QualityReportOutput) -> str:
        lines_out = ["site_name,overall_score,tier,dimensions_below"]
        for s in r.site_summaries:
            lines_out.append(f"{s.site_name},{s.overall_score},{s.tier},{s.dimensions_below}")
        return "\n".join(lines_out)


__all__ = ["DataQualityReport", "QualityReportInput", "QualityReportOutput"]
