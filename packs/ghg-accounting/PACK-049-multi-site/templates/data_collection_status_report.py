# -*- coding: utf-8 -*-
"""
DataCollectionStatusReport - Submission tracker for PACK-049.

Sections: status_matrix, completeness_by_scope, overdue,
          gaps, estimation_coverage, provenance.

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


class SiteStatusRow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field("")
    site_name: str = Field("")
    status: str = Field("not_started")
    scope_1_complete: bool = Field(False)
    scope_2_complete: bool = Field(False)
    scope_3_complete: bool = Field(False)
    entries_count: int = Field(0)
    completeness_pct: Decimal = Field(Decimal("0"))
    errors: int = Field(0)
    warnings: int = Field(0)
    submitted_at: str = Field("")
    reviewer: str = Field("")

class ScopeCompleteness(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scope: str = Field("")
    sites_complete: int = Field(0)
    sites_partial: int = Field(0)
    sites_missing: int = Field(0)
    overall_pct: Decimal = Field(Decimal("0"))

class OverdueItem(BaseModel):
    site_name: str = Field("")
    days_overdue: int = Field(0)
    status: str = Field("")
    escalated: bool = Field(False)

class DataGap(BaseModel):
    site_name: str = Field("")
    scope: str = Field("")
    source_category: str = Field("")
    gap_type: str = Field("")
    description: str = Field("")

class EstimationEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_name: str = Field("")
    source_category: str = Field("")
    estimation_method: str = Field("")
    estimated_tco2e: Decimal = Field(Decimal("0"))
    confidence: str = Field("medium")

class CollectionStatusInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    deadline: str = Field("")
    site_statuses: List[Dict[str, Any]] = Field(default_factory=list)
    scope_completeness: List[Dict[str, Any]] = Field(default_factory=list)
    overdue_items: List[Dict[str, Any]] = Field(default_factory=list)
    data_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    estimations: List[Dict[str, Any]] = Field(default_factory=list)

class CollectionStatusOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    deadline: str = Field("")
    total_sites: int = Field(0)
    approved_count: int = Field(0)
    pending_count: int = Field(0)
    rejected_count: int = Field(0)
    overdue_count: int = Field(0)
    overall_completeness_pct: Decimal = Field(Decimal("0"))
    status_matrix: List[SiteStatusRow] = Field(default_factory=list)
    scope_completeness: List[ScopeCompleteness] = Field(default_factory=list)
    overdue: List[OverdueItem] = Field(default_factory=list)
    gaps: List[DataGap] = Field(default_factory=list)
    estimations: List[EstimationEntry] = Field(default_factory=list)
    estimation_coverage_pct: Decimal = Field(Decimal("0"))
    provenance_hash: str = Field("")


class DataCollectionStatusReport:
    """Data collection status and submission tracker template."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def render(self, data: Dict[str, Any]) -> CollectionStatusOutput:
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = CollectionStatusInput(**data) if isinstance(data, dict) else data

        statuses = [SiteStatusRow(**s) if isinstance(s, dict) else s for s in inp.site_statuses]
        scope_comp = [ScopeCompleteness(**s) if isinstance(s, dict) else s for s in inp.scope_completeness]
        overdue = [OverdueItem(**o) if isinstance(o, dict) else o for o in inp.overdue_items]
        gaps = [DataGap(**g) if isinstance(g, dict) else g for g in inp.data_gaps]
        ests = [EstimationEntry(**e) if isinstance(e, dict) else e for e in inp.estimations]

        approved = sum(1 for s in statuses if s.status == "approved")
        rejected = sum(1 for s in statuses if s.status == "rejected")
        pending = len(statuses) - approved - rejected

        overall_comp = Decimal("0")
        if statuses:
            overall_comp = (sum(s.completeness_pct for s in statuses) / Decimal(str(len(statuses)))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        est_coverage = Decimal("0")
        if gaps and ests:
            est_coverage = (Decimal(str(len(ests))) / Decimal(str(max(len(gaps), 1))) * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = CollectionStatusOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            deadline=inp.deadline,
            total_sites=len(statuses),
            approved_count=approved,
            pending_count=pending,
            rejected_count=rejected,
            overdue_count=len(overdue),
            overall_completeness_pct=overall_comp,
            status_matrix=statuses,
            scope_completeness=scope_comp,
            overdue=overdue, gaps=gaps,
            estimations=ests,
            estimation_coverage_pct=est_coverage,
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

    def export_markdown(self, r: CollectionStatusOutput) -> str:
        lines: List[str] = []
        lines.append(f"# Data Collection Status - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Deadline:** {r.deadline}")
        lines.append(f"**Sites:** {r.total_sites} | **Approved:** {r.approved_count} | **Pending:** {r.pending_count} | **Rejected:** {r.rejected_count} | **Overdue:** {r.overdue_count}")
        lines.append(f"**Overall Completeness:** {r.overall_completeness_pct}%")
        lines.append("")
        lines.append("## Status Matrix")
        lines.append("| Site | Status | S1 | S2 | S3 | Entries | Completeness | Errors | Warnings |")
        lines.append("|------|--------|-----|-----|-----|---------|-------------|--------|----------|")
        for s in r.status_matrix:
            s1 = "Y" if s.scope_1_complete else "N"
            s2 = "Y" if s.scope_2_complete else "N"
            s3 = "Y" if s.scope_3_complete else "N"
            lines.append(f"| {s.site_name} | {s.status} | {s1} | {s2} | {s3} | {s.entries_count} | {s.completeness_pct}% | {s.errors} | {s.warnings} |")
        lines.append("")
        if r.scope_completeness:
            lines.append("## Completeness by Scope")
            for sc in r.scope_completeness:
                lines.append(f"- {sc.scope}: {sc.overall_pct}% (Complete: {sc.sites_complete}, Partial: {sc.sites_partial}, Missing: {sc.sites_missing})")
            lines.append("")
        if r.overdue:
            lines.append("## Overdue")
            for o in r.overdue:
                esc = " [ESCALATED]" if o.escalated else ""
                lines.append(f"- {o.site_name}: {o.days_overdue} days overdue ({o.status}){esc}")
            lines.append("")
        if r.gaps:
            lines.append(f"## Data Gaps ({len(r.gaps)})")
            for g in r.gaps:
                lines.append(f"- {g.site_name}: {g.scope} / {g.source_category} -- {g.gap_type}: {g.description}")
            lines.append("")
        if r.estimations:
            lines.append(f"## Estimation Coverage ({r.estimation_coverage_pct}%)")
            for e in r.estimations:
                lines.append(f"- {e.site_name}: {e.source_category} -- {e.estimation_method} ({e.estimated_tco2e:,.2f} tCO2e, {e.confidence})")
            lines.append("")
        lines.append(f"---\n*Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: CollectionStatusOutput) -> str:
        md = self.export_markdown(r)
        return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Collection Status</title></head><body><pre>{md}</pre></body></html>"

    def export_json(self, r: CollectionStatusOutput) -> Dict[str, Any]:
        return json.loads(r.model_dump_json())

    def export_csv(self, r: CollectionStatusOutput) -> str:
        lines_out = ["site_name,status,completeness_pct,entries,errors,warnings"]
        for s in r.status_matrix:
            lines_out.append(f"{s.site_name},{s.status},{s.completeness_pct},{s.entries_count},{s.errors},{s.warnings}")
        return "\n".join(lines_out)


__all__ = ["DataCollectionStatusReport", "CollectionStatusInput", "CollectionStatusOutput"]
