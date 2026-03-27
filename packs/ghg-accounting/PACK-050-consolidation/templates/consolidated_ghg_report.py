# -*- coding: utf-8 -*-
"""
ConsolidatedGhgReport - Group-level consolidated GHG inventory for PACK-050.

Generates the complete group-level GHG inventory report showing consolidated
Scope 1, Scope 2 (location-based and market-based), and Scope 3 totals after
equity adjustments, intercompany eliminations, and manual adjustments.
Includes organisational boundary description, consolidation approach used,
and full entity list.

Sections:
    1. Report Header (company, period, consolidation approach, boundary)
    2. Consolidated Totals (S1, S2-location, S2-market, S3, grand totals)
    3. Adjustment Summary (raw -> equity-adjusted -> eliminations -> final)
    4. Entity List (all entities in scope with type and control status)
    5. Organisational Boundary (approach, rationale, exclusions)
    6. Notes and Caveats
    7. Provenance Footer

Output Formats: Markdown, HTML, JSON, CSV

Author: GreenLang Team
Version: 50.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
import json
import time
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
# Enums
# ---------------------------------------------------------------------------

class ConsolidationApproach(str, Enum):
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"


# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class EntityEntry(BaseModel):
    """Single entity within the consolidation boundary."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_id: str = Field("")
    entity_name: str = Field("")
    entity_type: str = Field("", description="subsidiary, jv, associate, parent")
    country_code: str = Field("")
    control_type: str = Field("", description="operational, financial, equity")
    ownership_pct: Decimal = Field(Decimal("100"))
    reporting_pct: Decimal = Field(Decimal("100"))
    included: bool = Field(True)
    exclusion_reason: str = Field("")


class AdjustmentSummaryLine(BaseModel):
    """Single line in the adjustment waterfall."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    label: str = Field("")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_location_tco2e: Decimal = Field(Decimal("0"))
    total_market_tco2e: Decimal = Field(Decimal("0"))


class BoundaryDescription(BaseModel):
    """Organisational boundary metadata."""
    approach: str = Field("operational_control")
    rationale: str = Field("")
    base_year: int = Field(0)
    boundary_last_reviewed: str = Field("")
    exclusions: List[str] = Field(default_factory=list)
    materiality_threshold_pct: Decimal = Field(Decimal("5"))
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ReportNote(BaseModel):
    """Disclosure note or caveat."""
    note_id: str = Field("")
    category: str = Field("")
    text: str = Field("")


class ConsolidatedGhgReportInput(BaseModel):
    """Complete input for the consolidated GHG report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    report_date: Optional[str] = Field(None)
    consolidation_approach: str = Field("operational_control")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_location_tco2e: Decimal = Field(Decimal("0"))
    total_market_tco2e: Decimal = Field(Decimal("0"))
    prior_year_location_tco2e: Optional[Decimal] = Field(None)
    prior_year_market_tco2e: Optional[Decimal] = Field(None)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    adjustment_waterfall: List[Dict[str, Any]] = Field(default_factory=list)
    boundary: Optional[Dict[str, Any]] = Field(None)
    notes: List[Dict[str, Any]] = Field(default_factory=list)
    total_entities: int = Field(0)
    included_entities: int = Field(0)
    excluded_entities: int = Field(0)
    currency_code: str = Field("USD")


# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class ConsolidatedGhgReportOutput(BaseModel):
    """Rendered consolidated GHG inventory report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    consolidation_approach: str = Field("")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_location_tco2e: Decimal = Field(Decimal("0"))
    total_market_tco2e: Decimal = Field(Decimal("0"))
    yoy_location_change_pct: Optional[Decimal] = Field(None)
    yoy_market_change_pct: Optional[Decimal] = Field(None)
    entities: List[EntityEntry] = Field(default_factory=list)
    total_entities: int = Field(0)
    included_entities: int = Field(0)
    excluded_entities: int = Field(0)
    adjustment_waterfall: List[AdjustmentSummaryLine] = Field(default_factory=list)
    boundary: Optional[BoundaryDescription] = Field(None)
    notes: List[ReportNote] = Field(default_factory=list)
    provenance_hash: str = Field("")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ConsolidatedGhgReport:
    """
    Group-level consolidated GHG inventory report template for PACK-050.

    Produces a complete consolidated emissions report with organisational
    boundary documentation, equity adjustments, eliminations, and provenance.

    Example:
        >>> tpl = ConsolidatedGhgReport()
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

    def render(self, data: Dict[str, Any]) -> ConsolidatedGhgReportOutput:
        """Render consolidated GHG report from input data."""
        start = time.monotonic()
        self.generated_at = _utcnow()

        inp = ConsolidatedGhgReportInput(**data) if isinstance(data, dict) else data

        entities = [EntityEntry(**e) if isinstance(e, dict) else e for e in inp.entities]
        waterfall = [
            AdjustmentSummaryLine(**w) if isinstance(w, dict) else w
            for w in inp.adjustment_waterfall
        ]
        boundary = BoundaryDescription(**(inp.boundary)) if isinstance(inp.boundary, dict) else inp.boundary
        notes = [ReportNote(**n) if isinstance(n, dict) else n for n in inp.notes]

        included = [e for e in entities if e.included]
        excluded = [e for e in entities if not e.included]

        # YoY calculations
        yoy_loc = self._compute_yoy(inp.total_location_tco2e, inp.prior_year_location_tco2e)
        yoy_mkt = self._compute_yoy(inp.total_market_tco2e, inp.prior_year_market_tco2e)

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = ConsolidatedGhgReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            consolidation_approach=inp.consolidation_approach,
            scope_1_tco2e=inp.scope_1_tco2e,
            scope_2_location_tco2e=inp.scope_2_location_tco2e,
            scope_2_market_tco2e=inp.scope_2_market_tco2e,
            scope_3_tco2e=inp.scope_3_tco2e,
            total_location_tco2e=inp.total_location_tco2e,
            total_market_tco2e=inp.total_market_tco2e,
            yoy_location_change_pct=yoy_loc,
            yoy_market_change_pct=yoy_mkt,
            entities=entities,
            total_entities=inp.total_entities or len(entities),
            included_entities=inp.included_entities or len(included),
            excluded_entities=inp.excluded_entities or len(excluded),
            adjustment_waterfall=waterfall,
            boundary=boundary,
            notes=notes,
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

    def export_markdown(self, r: ConsolidatedGhgReportOutput) -> str:
        """Export report as Markdown."""
        lines: List[str] = []
        approach_label = r.consolidation_approach.replace("_", " ").title()
        lines.append(f"# Consolidated GHG Inventory - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Approach:** {approach_label}")
        lines.append(f"**Entities:** {r.included_entities} included / {r.excluded_entities} excluded / {r.total_entities} total")
        lines.append("")

        # Consolidated totals
        lines.append("## Consolidated Totals")
        lines.append("| Scope | tCO2e |")
        lines.append("|-------|------:|")
        lines.append(f"| Scope 1 | {r.scope_1_tco2e:,.2f} |")
        lines.append(f"| Scope 2 (Location-based) | {r.scope_2_location_tco2e:,.2f} |")
        lines.append(f"| Scope 2 (Market-based) | {r.scope_2_market_tco2e:,.2f} |")
        lines.append(f"| Scope 3 | {r.scope_3_tco2e:,.2f} |")
        lines.append(f"| **Total (Location-based)** | **{r.total_location_tco2e:,.2f}** |")
        lines.append(f"| **Total (Market-based)** | **{r.total_market_tco2e:,.2f}** |")
        if r.yoy_location_change_pct is not None:
            lines.append(f"\nYoY Change (Location): **{r.yoy_location_change_pct}%**")
        if r.yoy_market_change_pct is not None:
            lines.append(f"YoY Change (Market): **{r.yoy_market_change_pct}%**")
        lines.append("")

        # Adjustment waterfall
        if r.adjustment_waterfall:
            lines.append("## Adjustment Waterfall")
            lines.append("| Step | S1 | S2-Loc | S2-Mkt | S3 | Total-Loc | Total-Mkt |")
            lines.append("|------|----|--------|--------|----|-----------|-----------|")
            for w in r.adjustment_waterfall:
                lines.append(
                    f"| {w.label} | {w.scope_1_tco2e:,.0f} | {w.scope_2_location_tco2e:,.0f} | "
                    f"{w.scope_2_market_tco2e:,.0f} | {w.scope_3_tco2e:,.0f} | "
                    f"{w.total_location_tco2e:,.0f} | {w.total_market_tco2e:,.0f} |"
                )
            lines.append("")

        # Entity list
        if r.entities:
            lines.append("## Entity List")
            lines.append("| Entity | Type | Country | Control | Ownership | Included |")
            lines.append("|--------|------|---------|---------|-----------|----------|")
            for e in r.entities:
                status = "Yes" if e.included else f"No ({e.exclusion_reason})"
                lines.append(
                    f"| {e.entity_name} | {e.entity_type} | {e.country_code} | "
                    f"{e.control_type} | {e.ownership_pct}% | {status} |"
                )
            lines.append("")

        # Boundary
        if r.boundary:
            b = r.boundary
            lines.append("## Organisational Boundary")
            lines.append(f"- **Approach:** {b.approach.replace('_', ' ').title()}")
            lines.append(f"- **Rationale:** {b.rationale}")
            lines.append(f"- **Base Year:** {b.base_year}")
            lines.append(f"- **Last Reviewed:** {b.boundary_last_reviewed}")
            lines.append(f"- **Materiality Threshold:** {b.materiality_threshold_pct}%")
            if b.exclusions:
                lines.append("- **Exclusions:**")
                for ex in b.exclusions:
                    lines.append(f"  - {ex}")
            lines.append("")

        # Notes
        if r.notes:
            lines.append("## Notes and Caveats")
            for n in r.notes:
                lines.append(f"- **[{n.category}]** {n.text}")
            lines.append("")

        lines.append(f"---\n*Report ID: {r.report_id} | Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: ConsolidatedGhgReportOutput) -> str:
        """Export report as HTML."""
        md = self.export_markdown(r)
        html_lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>Consolidated GHG Inventory - {r.company_name}</title>",
            "<style>body{font-family:sans-serif;margin:2em;}table{border-collapse:collapse;width:100%;margin:1em 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f5f5f5;}</style>",
            "</head><body>",
            f"<pre>{md}</pre>",
            "</body></html>",
        ]
        return "\n".join(html_lines)

    def export_json(self, r: ConsolidatedGhgReportOutput) -> Dict[str, Any]:
        """Export report as JSON-serializable dict."""
        return json.loads(r.model_dump_json())

    def export_csv(self, r: ConsolidatedGhgReportOutput) -> str:
        """Export entity-level data as CSV."""
        lines_out = [
            "entity_id,entity_name,entity_type,country_code,control_type,"
            "ownership_pct,reporting_pct,included"
        ]
        for e in r.entities:
            lines_out.append(
                f"{e.entity_id},{e.entity_name},{e.entity_type},{e.country_code},"
                f"{e.control_type},{e.ownership_pct},{e.reporting_pct},{e.included}"
            )
        return "\n".join(lines_out)

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_yoy(
        current: Decimal, prior: Optional[Decimal]
    ) -> Optional[Decimal]:
        """Compute year-over-year percentage change."""
        if prior is not None and prior > Decimal("0"):
            return ((current - prior) / prior * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        return None


__all__ = [
    "ConsolidatedGhgReport",
    "ConsolidatedGhgReportInput",
    "ConsolidatedGhgReportOutput",
    "EntityEntry",
    "AdjustmentSummaryLine",
    "BoundaryDescription",
    "ReportNote",
]
