# -*- coding: utf-8 -*-
"""
RegionalFactorReport - Factor assignment matrix for PACK-049.

Sections: assignment_matrix, tier_distribution, source_distribution,
          grid_regions, overrides, provenance.

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

class FactorAssignmentRow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field("")
    site_name: str = Field("")
    source_category: str = Field("")
    scope: str = Field("")
    factor_name: str = Field("")
    factor_value: Decimal = Field(Decimal("0"))
    factor_unit: str = Field("")
    source_db: str = Field("")
    tier: str = Field("tier_2")
    grid_region: str = Field("")
    country_code: str = Field("")
    year: int = Field(0)
    is_override: bool = Field(False)
    override_reason: str = Field("")

class GridRegionSummary(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    grid_region: str = Field("")
    country_code: str = Field("")
    site_count: int = Field(0)
    avg_grid_factor: Decimal = Field(Decimal("0"))
    factor_unit: str = Field("kgCO2e/kWh")
    source: str = Field("")

class FactorReportInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    assignments: List[Dict[str, Any]] = Field(default_factory=list)
    grid_regions: List[Dict[str, Any]] = Field(default_factory=list)

class FactorReportOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    assignment_matrix: List[FactorAssignmentRow] = Field(default_factory=list)
    tier_distribution: Dict[str, int] = Field(default_factory=dict)
    source_distribution: Dict[str, int] = Field(default_factory=dict)
    grid_regions: List[GridRegionSummary] = Field(default_factory=list)
    overrides_count: int = Field(0)
    total_assignments: int = Field(0)
    provenance_hash: str = Field("")

class RegionalFactorReport:
    """Factor assignment matrix report template."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def render(self, data: Dict[str, Any]) -> FactorReportOutput:
        start = time.monotonic()
        self.generated_at = utcnow()
        inp = FactorReportInput(**data) if isinstance(data, dict) else data

        assignments = [FactorAssignmentRow(**a) if isinstance(a, dict) else a for a in inp.assignments]
        grid_regions = [GridRegionSummary(**g) if isinstance(g, dict) else g for g in inp.grid_regions]

        tier_dist: Dict[str, int] = {}
        src_dist: Dict[str, int] = {}
        overrides = 0
        for a in assignments:
            tier_dist[a.tier] = tier_dist.get(a.tier, 0) + 1
            src_dist[a.source_db] = src_dist.get(a.source_db, 0) + 1
            if a.is_override:
                overrides += 1

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = FactorReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            assignment_matrix=assignments,
            tier_distribution=tier_dist,
            source_distribution=src_dist,
            grid_regions=grid_regions,
            overrides_count=overrides,
            total_assignments=len(assignments),
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

    def export_markdown(self, r: FactorReportOutput) -> str:
        lines: List[str] = []
        lines.append(f"# Regional Factor Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Assignments:** {r.total_assignments} | **Overrides:** {r.overrides_count}")
        lines.append("")
        lines.append("## Assignment Matrix")
        lines.append("| Site | Source | Scope | Factor | Value | Unit | DB | Tier | Grid | Override |")
        lines.append("|------|--------|-------|--------|-------|------|----|------|------|----------|")
        for a in r.assignment_matrix:
            ov = "Yes" if a.is_override else ""
            lines.append(f"| {a.site_name} | {a.source_category} | {a.scope} | {a.factor_name} | {a.factor_value} | {a.factor_unit} | {a.source_db} | {a.tier} | {a.grid_region} | {ov} |")
        lines.append("")
        lines.append("## Tier Distribution")
        for tier, count in sorted(r.tier_distribution.items()):
            lines.append(f"- {tier}: {count}")
        lines.append("")
        lines.append("## Source Distribution")
        for src, count in sorted(r.source_distribution.items()):
            lines.append(f"- {src}: {count}")
        lines.append("")
        if r.grid_regions:
            lines.append("## Grid Regions")
            lines.append("| Region | Country | Sites | Avg Factor | Unit | Source |")
            lines.append("|--------|---------|-------|------------|------|--------|")
            for g in r.grid_regions:
                lines.append(f"| {g.grid_region} | {g.country_code} | {g.site_count} | {g.avg_grid_factor} | {g.factor_unit} | {g.source} |")
            lines.append("")
        lines.append(f"---\n*Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: FactorReportOutput) -> str:
        md = self.export_markdown(r)
        return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Factor Report</title></head><body><pre>{md}</pre></body></html>"

    def export_json(self, r: FactorReportOutput) -> Dict[str, Any]:
        return json.loads(r.model_dump_json())

    def export_csv(self, r: FactorReportOutput) -> str:
        lines = ["site_name,source_category,scope,factor_name,factor_value,factor_unit,source_db,tier"]
        for a in r.assignment_matrix:
            lines.append(f"{a.site_name},{a.source_category},{a.scope},{a.factor_name},{a.factor_value},{a.factor_unit},{a.source_db},{a.tier}")
        return "\n".join(lines)

__all__ = ["RegionalFactorReport", "FactorReportInput", "FactorReportOutput"]
