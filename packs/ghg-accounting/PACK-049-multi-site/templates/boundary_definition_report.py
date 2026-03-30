# -*- coding: utf-8 -*-
"""
BoundaryDefinitionReport - Organisational boundary documentation for PACK-049.

Sections: approach_rationale, entity_hierarchy, inclusions, exclusions,
          materiality, yoy_changes, provenance.

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

class BoundaryEntity(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_id: str = Field("")
    entity_name: str = Field("")
    entity_type: str = Field("")
    country_code: str = Field("")
    ownership_pct: Decimal = Field(Decimal("100"))
    reporting_pct: Decimal = Field(Decimal("100"))
    facility_count: int = Field(0)
    is_included: bool = Field(True)
    materiality: str = Field("material")
    emissions_share_pct: Decimal = Field(Decimal("0"))

class ExclusionEntry(BaseModel):
    entity_name: str = Field("")
    reason: str = Field("")
    emissions_share_pct: Decimal = Field(Decimal("0"))
    model_config = ConfigDict(arbitrary_types_allowed=True)

class YoYChange(BaseModel):
    change_type: str = Field("")
    entity_name: str = Field("")
    description: str = Field("")

class BoundaryReportInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_year: int = Field(0)
    consolidation_approach: str = Field("operational_control")
    approach_rationale: str = Field("")
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    exclusions: List[Dict[str, Any]] = Field(default_factory=list)
    yoy_changes: List[Dict[str, Any]] = Field(default_factory=list)
    total_entities: int = Field(0)
    included_entities: int = Field(0)
    excluded_entities: int = Field(0)
    total_facilities: int = Field(0)
    lock_status: str = Field("locked")
    locked_by: str = Field("")
    locked_at: str = Field("")

class BoundaryReportOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_year: int = Field(0)
    generated_at: str = Field("")
    approach: str = Field("")
    approach_rationale: str = Field("")
    entity_hierarchy: List[BoundaryEntity] = Field(default_factory=list)
    inclusions: List[BoundaryEntity] = Field(default_factory=list)
    exclusions: List[ExclusionEntry] = Field(default_factory=list)
    yoy_changes: List[YoYChange] = Field(default_factory=list)
    total_entities: int = Field(0)
    included_count: int = Field(0)
    excluded_count: int = Field(0)
    total_facilities: int = Field(0)
    lock_status: str = Field("")
    provenance_hash: str = Field("")

class BoundaryDefinitionReport:
    """Organisational boundary documentation template."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def render(self, data: Dict[str, Any]) -> BoundaryReportOutput:
        start = time.monotonic()
        self.generated_at = utcnow()
        inp = BoundaryReportInput(**data) if isinstance(data, dict) else data

        entities = [BoundaryEntity(**e) if isinstance(e, dict) else e for e in inp.entities]
        inclusions = [e for e in entities if e.is_included]
        exclusions = [ExclusionEntry(**x) if isinstance(x, dict) else x for x in inp.exclusions]
        changes = [YoYChange(**c) if isinstance(c, dict) else c for c in inp.yoy_changes]

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = BoundaryReportOutput(
            company_name=inp.company_name, reporting_year=inp.reporting_year,
            generated_at=self.generated_at.isoformat(),
            approach=inp.consolidation_approach,
            approach_rationale=inp.approach_rationale or self._default_rationale(inp.consolidation_approach),
            entity_hierarchy=entities, inclusions=inclusions,
            exclusions=exclusions, yoy_changes=changes,
            total_entities=inp.total_entities or len(entities),
            included_count=inp.included_entities or len(inclusions),
            excluded_count=inp.excluded_entities or len(exclusions),
            total_facilities=inp.total_facilities,
            lock_status=inp.lock_status, provenance_hash=prov,
        )
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return output

    def _default_rationale(self, approach: str) -> str:
        rationales = {
            "equity_share": "The equity share approach reports emissions proportional to the organisation's equity interest in each entity, per GHG Protocol Chapter 3.",
            "financial_control": "The financial control approach reports 100% of emissions from entities over which the organisation has financial control, per GHG Protocol Chapter 3.",
            "operational_control": "The operational control approach reports 100% of emissions from entities over which the organisation has operational control, per GHG Protocol Chapter 3.",
        }
        return rationales.get(approach, "Consolidation approach applied per GHG Protocol Chapter 3.")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_markdown(r)

    def render_html(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_html(r)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_json(r)

    def export_markdown(self, r: BoundaryReportOutput) -> str:
        lines: List[str] = []
        lines.append(f"# Boundary Definition Report - {r.company_name} ({r.reporting_year})")
        lines.append(f"**Approach:** {r.approach} | **Lock Status:** {r.lock_status}")
        lines.append("")
        lines.append("## Approach Rationale")
        lines.append(r.approach_rationale)
        lines.append("")
        lines.append(f"## Entity Hierarchy ({r.total_entities} entities, {r.total_facilities} facilities)")
        lines.append("| Entity | Type | Country | Ownership | Reporting | Facilities | Materiality |")
        lines.append("|--------|------|---------|-----------|-----------|------------|-------------|")
        for e in r.entity_hierarchy:
            lines.append(f"| {e.entity_name} | {e.entity_type} | {e.country_code} | {e.ownership_pct}% | {e.reporting_pct}% | {e.facility_count} | {e.materiality} |")
        lines.append("")
        lines.append(f"## Inclusions ({r.included_count})")
        for e in r.inclusions:
            lines.append(f"- **{e.entity_name}** ({e.entity_type}) -- {e.ownership_pct}% ownership, {e.emissions_share_pct}% of emissions")
        lines.append("")
        if r.exclusions:
            lines.append(f"## Exclusions ({r.excluded_count})")
            for x in r.exclusions:
                lines.append(f"- **{x.entity_name}**: {x.reason} ({x.emissions_share_pct}% share)")
            lines.append("")
        if r.yoy_changes:
            lines.append("## Year-over-Year Changes")
            for c in r.yoy_changes:
                lines.append(f"- [{c.change_type}] {c.entity_name}: {c.description}")
            lines.append("")
        lines.append(f"---\n*Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: BoundaryReportOutput) -> str:
        md = self.export_markdown(r)
        return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Boundary Report</title></head><body><pre>{md}</pre></body></html>"

    def export_json(self, r: BoundaryReportOutput) -> Dict[str, Any]:
        return json.loads(r.model_dump_json())

    def export_csv(self, r: BoundaryReportOutput) -> str:
        lines = ["entity_name,entity_type,ownership_pct,reporting_pct,is_included,materiality"]
        for e in r.entity_hierarchy:
            lines.append(f"{e.entity_name},{e.entity_type},{e.ownership_pct},{e.reporting_pct},{e.is_included},{e.materiality}")
        return "\n".join(lines)

__all__ = ["BoundaryDefinitionReport", "BoundaryReportInput", "BoundaryReportOutput"]
