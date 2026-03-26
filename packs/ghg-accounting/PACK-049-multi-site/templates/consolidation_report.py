# -*- coding: utf-8 -*-
"""
ConsolidationReport - Corporate-level consolidated report for PACK-049.

Sections: consolidated_totals, scope_breakdown, entity_breakdown,
          eliminations, equity_adjustments, reconciliation, provenance.

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

class EntityEmissions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_id: str = Field("")
    entity_name: str = Field("")
    entity_type: str = Field("")
    ownership_pct: Decimal = Field(Decimal("100"))
    reporting_pct: Decimal = Field(Decimal("100"))
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_tco2e: Decimal = Field(Decimal("0"))
    site_count: int = Field(0)


class EliminationItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    elimination_type: str = Field("")
    from_entity: str = Field("")
    to_entity: str = Field("")
    eliminated_tco2e: Decimal = Field(Decimal("0"))
    description: str = Field("")


class AdjustmentItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_name: str = Field("")
    original_tco2e: Decimal = Field(Decimal("0"))
    ownership_pct: Decimal = Field(Decimal("100"))
    adjusted_tco2e: Decimal = Field(Decimal("0"))
    method: str = Field("")


class ReconciliationItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scope: str = Field("")
    bottom_up_tco2e: Decimal = Field(Decimal("0"))
    top_down_tco2e: Decimal = Field(Decimal("0"))
    variance_tco2e: Decimal = Field(Decimal("0"))
    variance_pct: Decimal = Field(Decimal("0"))
    status: str = Field("")


class ConsolidationReportInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    consolidation_approach: str = Field("operational_control")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_location_tco2e: Decimal = Field(Decimal("0"))
    total_market_tco2e: Decimal = Field(Decimal("0"))
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    eliminations: List[Dict[str, Any]] = Field(default_factory=list)
    adjustments: List[Dict[str, Any]] = Field(default_factory=list)
    reconciliation: List[Dict[str, Any]] = Field(default_factory=list)
    sites_count: int = Field(0)
    prior_year_total: Optional[Decimal] = Field(None)


# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class ConsolidationReportOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    approach: str = Field("")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_location_tco2e: Decimal = Field(Decimal("0"))
    total_market_tco2e: Decimal = Field(Decimal("0"))
    yoy_change_pct: Optional[Decimal] = Field(None)
    entities: List[EntityEmissions] = Field(default_factory=list)
    eliminations: List[EliminationItem] = Field(default_factory=list)
    total_eliminations_tco2e: Decimal = Field(Decimal("0"))
    adjustments: List[AdjustmentItem] = Field(default_factory=list)
    reconciliation: List[ReconciliationItem] = Field(default_factory=list)
    sites_count: int = Field(0)
    provenance_hash: str = Field("")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ConsolidationReport:
    """Corporate-level consolidated emissions report."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def render(self, data: Dict[str, Any]) -> ConsolidationReportOutput:
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = ConsolidationReportInput(**data) if isinstance(data, dict) else data

        entities = [EntityEmissions(**e) if isinstance(e, dict) else e for e in inp.entities]
        elims = [EliminationItem(**e) if isinstance(e, dict) else e for e in inp.eliminations]
        adjs = [AdjustmentItem(**a) if isinstance(a, dict) else a for a in inp.adjustments]
        recons = [ReconciliationItem(**r) if isinstance(r, dict) else r for r in inp.reconciliation]

        total_elim = sum(e.eliminated_tco2e for e in elims)

        yoy = None
        if inp.prior_year_total and inp.prior_year_total > Decimal("0"):
            yoy = ((inp.total_location_tco2e - inp.prior_year_total) / inp.prior_year_total * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = ConsolidationReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            approach=inp.consolidation_approach,
            scope_1_tco2e=inp.scope_1_tco2e,
            scope_2_location_tco2e=inp.scope_2_location_tco2e,
            scope_2_market_tco2e=inp.scope_2_market_tco2e,
            scope_3_tco2e=inp.scope_3_tco2e,
            total_location_tco2e=inp.total_location_tco2e,
            total_market_tco2e=inp.total_market_tco2e,
            yoy_change_pct=yoy,
            entities=entities,
            eliminations=elims,
            total_eliminations_tco2e=total_elim,
            adjustments=adjs,
            reconciliation=recons,
            sites_count=inp.sites_count,
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

    def export_markdown(self, r: ConsolidationReportOutput) -> str:
        lines: List[str] = []
        lines.append(f"# Consolidation Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Approach:** {r.approach} | **Sites:** {r.sites_count}")
        lines.append("")
        lines.append("## Consolidated Totals")
        lines.append("| Scope | tCO2e |")
        lines.append("|-------|-------|")
        lines.append(f"| Scope 1 | {r.scope_1_tco2e:,.2f} |")
        lines.append(f"| Scope 2 (Location) | {r.scope_2_location_tco2e:,.2f} |")
        lines.append(f"| Scope 2 (Market) | {r.scope_2_market_tco2e:,.2f} |")
        lines.append(f"| Scope 3 | {r.scope_3_tco2e:,.2f} |")
        lines.append(f"| **Total (Location)** | **{r.total_location_tco2e:,.2f}** |")
        lines.append(f"| **Total (Market)** | **{r.total_market_tco2e:,.2f}** |")
        if r.yoy_change_pct is not None:
            lines.append(f"\nYoY Change: **{r.yoy_change_pct}%**")
        lines.append("")
        if r.entities:
            lines.append("## Entity Breakdown")
            lines.append("| Entity | Type | Ownership | S1 | S2 | S3 | Total | Sites |")
            lines.append("|--------|------|-----------|-----|-----|-----|-------|-------|")
            for e in r.entities:
                lines.append(
                    f"| {e.entity_name} | {e.entity_type} | {e.ownership_pct}% | "
                    f"{e.scope_1_tco2e:,.0f} | {e.scope_2_tco2e:,.0f} | {e.scope_3_tco2e:,.0f} | "
                    f"{e.total_tco2e:,.0f} | {e.site_count} |"
                )
            lines.append("")
        if r.eliminations:
            lines.append("## Eliminations")
            lines.append("| Type | From | To | tCO2e | Description |")
            lines.append("|------|------|----|-------|-------------|")
            for e in r.eliminations:
                lines.append(f"| {e.elimination_type} | {e.from_entity} | {e.to_entity} | {e.eliminated_tco2e:,.2f} | {e.description} |")
            lines.append(f"\n**Total Eliminations:** {r.total_eliminations_tco2e:,.2f} tCO2e")
            lines.append("")
        if r.adjustments:
            lines.append("## Equity Adjustments")
            lines.append("| Entity | Original | Ownership | Adjusted | Method |")
            lines.append("|--------|----------|-----------|----------|--------|")
            for a in r.adjustments:
                lines.append(f"| {a.entity_name} | {a.original_tco2e:,.0f} | {a.ownership_pct}% | {a.adjusted_tco2e:,.0f} | {a.method} |")
            lines.append("")
        if r.reconciliation:
            lines.append("## Reconciliation")
            lines.append("| Scope | Bottom-Up | Top-Down | Variance | % | Status |")
            lines.append("|-------|-----------|---------|----------|---|--------|")
            for rc in r.reconciliation:
                lines.append(f"| {rc.scope} | {rc.bottom_up_tco2e:,.0f} | {rc.top_down_tco2e:,.0f} | {rc.variance_tco2e:,.0f} | {rc.variance_pct}% | {rc.status} |")
            lines.append("")
        lines.append(f"---\n*Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: ConsolidationReportOutput) -> str:
        md = self.export_markdown(r)
        return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Consolidation Report</title></head><body><pre>{md}</pre></body></html>"

    def export_json(self, r: ConsolidationReportOutput) -> Dict[str, Any]:
        return json.loads(r.model_dump_json())

    def export_csv(self, r: ConsolidationReportOutput) -> str:
        lines = ["entity_name,entity_type,ownership_pct,total_tco2e"]
        for e in r.entities:
            lines.append(f"{e.entity_name},{e.entity_type},{e.ownership_pct},{e.total_tco2e}")
        return "\n".join(lines)


__all__ = ["ConsolidationReport", "ConsolidationReportInput", "ConsolidationReportOutput"]
