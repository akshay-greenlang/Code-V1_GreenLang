# -*- coding: utf-8 -*-
"""
EntityBreakdownReport - Per-entity emission contributions for PACK-050.

Shows each entity's raw emissions, equity-adjusted emissions, percentage
contribution to group total, waterfall chart data, top-N entities, and
scope breakdown per entity.

Sections:
    1. Group Summary (total entities, total emissions, approach)
    2. Entity Contributions Table (raw, adjusted, pct contribution)
    3. Waterfall Chart Data (cumulative entity build-up to group total)
    4. Top-N Entities (largest contributors)
    5. Scope Breakdown Per Entity (S1/S2/S3 per entity)
    6. Provenance Footer

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

class EntityContribution(BaseModel):
    """Emission contribution for a single entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_id: str = Field("")
    entity_name: str = Field("")
    entity_type: str = Field("")
    country_code: str = Field("")
    ownership_pct: Decimal = Field(Decimal("100"))
    raw_scope_1_tco2e: Decimal = Field(Decimal("0"))
    raw_scope_2_tco2e: Decimal = Field(Decimal("0"))
    raw_scope_3_tco2e: Decimal = Field(Decimal("0"))
    raw_total_tco2e: Decimal = Field(Decimal("0"))
    adjusted_scope_1_tco2e: Decimal = Field(Decimal("0"))
    adjusted_scope_2_tco2e: Decimal = Field(Decimal("0"))
    adjusted_scope_3_tco2e: Decimal = Field(Decimal("0"))
    adjusted_total_tco2e: Decimal = Field(Decimal("0"))
    contribution_pct: Decimal = Field(Decimal("0"))


class WaterfallStep(BaseModel):
    """Single step in the waterfall chart."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_name: str = Field("")
    step_value_tco2e: Decimal = Field(Decimal("0"))
    cumulative_tco2e: Decimal = Field(Decimal("0"))
    contribution_pct: Decimal = Field(Decimal("0"))


class EntityBreakdownReportInput(BaseModel):
    """Complete input for the entity breakdown report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    consolidation_approach: str = Field("operational_control")
    group_total_tco2e: Decimal = Field(Decimal("0"))
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    top_n: int = Field(10, ge=1, le=100)


# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class EntityBreakdownReportOutput(BaseModel):
    """Rendered entity breakdown report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    consolidation_approach: str = Field("")
    group_total_tco2e: Decimal = Field(Decimal("0"))
    entity_count: int = Field(0)
    entities: List[EntityContribution] = Field(default_factory=list)
    waterfall: List[WaterfallStep] = Field(default_factory=list)
    top_entities: List[EntityContribution] = Field(default_factory=list)
    top_entities_pct: Decimal = Field(Decimal("0"))
    provenance_hash: str = Field("")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class EntityBreakdownReport:
    """
    Per-entity emission contribution report template for PACK-050.

    Produces entity-level breakdown with raw vs adjusted emissions,
    waterfall chart data, and top-N entity ranking.

    Example:
        >>> tpl = EntityBreakdownReport()
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

    def render(self, data: Dict[str, Any]) -> EntityBreakdownReportOutput:
        """Render entity breakdown from input data."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = EntityBreakdownReportInput(**data) if isinstance(data, dict) else data

        entities = [EntityContribution(**e) if isinstance(e, dict) else e for e in inp.entities]

        group_total = inp.group_total_tco2e or Decimal("1")

        # Compute contribution percentages
        for e in entities:
            if e.raw_total_tco2e == Decimal("0"):
                e.raw_total_tco2e = e.raw_scope_1_tco2e + e.raw_scope_2_tco2e + e.raw_scope_3_tco2e
            if e.adjusted_total_tco2e == Decimal("0"):
                e.adjusted_total_tco2e = e.adjusted_scope_1_tco2e + e.adjusted_scope_2_tco2e + e.adjusted_scope_3_tco2e
            e.contribution_pct = (
                e.adjusted_total_tco2e / group_total * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Sort by adjusted total descending
        entities.sort(key=lambda e: e.adjusted_total_tco2e, reverse=True)

        # Build waterfall
        waterfall: List[WaterfallStep] = []
        cumulative = Decimal("0")
        for e in entities:
            cumulative += e.adjusted_total_tco2e
            waterfall.append(WaterfallStep(
                entity_name=e.entity_name,
                step_value_tco2e=e.adjusted_total_tco2e,
                cumulative_tco2e=cumulative,
                contribution_pct=e.contribution_pct,
            ))

        # Top-N
        top_n = inp.top_n
        top_entities = entities[:top_n]
        top_pct = sum(e.contribution_pct for e in top_entities)

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = EntityBreakdownReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            consolidation_approach=inp.consolidation_approach,
            group_total_tco2e=inp.group_total_tco2e,
            entity_count=len(entities),
            entities=entities,
            waterfall=waterfall,
            top_entities=top_entities,
            top_entities_pct=top_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
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

    def export_markdown(self, r: EntityBreakdownReportOutput) -> str:
        """Export report as Markdown."""
        lines: List[str] = []
        lines.append(f"# Entity Breakdown Report - {r.company_name}")
        lines.append(
            f"**Period:** {r.reporting_period} | "
            f"**Group Total:** {r.group_total_tco2e:,.2f} tCO2e | "
            f"**Entities:** {r.entity_count}"
        )
        lines.append("")

        # Full entity table
        lines.append("## Entity Contributions")
        lines.append("| Entity | Type | Ownership | Raw tCO2e | Adjusted tCO2e | Contribution |")
        lines.append("|--------|------|-----------|-----------|----------------|--------------|")
        for e in r.entities:
            lines.append(
                f"| {e.entity_name} | {e.entity_type} | {e.ownership_pct}% | "
                f"{e.raw_total_tco2e:,.0f} | {e.adjusted_total_tco2e:,.0f} | {e.contribution_pct}% |"
            )
        lines.append("")

        # Top-N
        lines.append(f"## Top {len(r.top_entities)} Entities ({r.top_entities_pct}% of total)")
        lines.append("| Rank | Entity | Adjusted tCO2e | S1 | S2 | S3 | Contribution |")
        lines.append("|------|--------|----------------|----|----|-------|--------------|")
        for i, e in enumerate(r.top_entities, 1):
            lines.append(
                f"| {i} | {e.entity_name} | {e.adjusted_total_tco2e:,.0f} | "
                f"{e.adjusted_scope_1_tco2e:,.0f} | {e.adjusted_scope_2_tco2e:,.0f} | "
                f"{e.adjusted_scope_3_tco2e:,.0f} | {e.contribution_pct}% |"
            )
        lines.append("")

        # Waterfall
        lines.append("## Waterfall Chart Data")
        lines.append("| Entity | Step tCO2e | Cumulative tCO2e | Contribution |")
        lines.append("|--------|-----------|------------------|--------------|")
        for w in r.waterfall:
            lines.append(
                f"| {w.entity_name} | {w.step_value_tco2e:,.0f} | "
                f"{w.cumulative_tco2e:,.0f} | {w.contribution_pct}% |"
            )
        lines.append("")

        # Scope breakdown per entity
        lines.append("## Scope Breakdown Per Entity")
        lines.append("| Entity | S1 tCO2e | S2 tCO2e | S3 tCO2e | Total tCO2e |")
        lines.append("|--------|----------|----------|----------|-------------|")
        for e in r.entities:
            lines.append(
                f"| {e.entity_name} | {e.adjusted_scope_1_tco2e:,.0f} | "
                f"{e.adjusted_scope_2_tco2e:,.0f} | {e.adjusted_scope_3_tco2e:,.0f} | "
                f"{e.adjusted_total_tco2e:,.0f} |"
            )
        lines.append("")

        lines.append(f"---\n*Report ID: {r.report_id} | Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: EntityBreakdownReportOutput) -> str:
        """Export report as HTML."""
        md = self.export_markdown(r)
        html_lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>Entity Breakdown Report - {r.company_name}</title>",
            "<style>body{font-family:sans-serif;margin:2em;}table{border-collapse:collapse;width:100%;margin:1em 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f5f5f5;}</style>",
            "</head><body>",
            f"<pre>{md}</pre>",
            "</body></html>",
        ]
        return "\n".join(html_lines)

    def export_json(self, r: EntityBreakdownReportOutput) -> Dict[str, Any]:
        """Export report as JSON-serializable dict."""
        return json.loads(r.model_dump_json())

    def export_csv(self, r: EntityBreakdownReportOutput) -> str:
        """Export entity breakdown as CSV."""
        lines_out = [
            "entity_id,entity_name,entity_type,ownership_pct,"
            "raw_total_tco2e,adjusted_total_tco2e,contribution_pct"
        ]
        for e in r.entities:
            lines_out.append(
                f"{e.entity_id},{e.entity_name},{e.entity_type},{e.ownership_pct},"
                f"{e.raw_total_tco2e},{e.adjusted_total_tco2e},{e.contribution_pct}"
            )
        return "\n".join(lines_out)


__all__ = [
    "EntityBreakdownReport",
    "EntityBreakdownReportInput",
    "EntityBreakdownReportOutput",
    "EntityContribution",
    "WaterfallStep",
]
