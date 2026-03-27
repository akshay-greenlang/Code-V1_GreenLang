# -*- coding: utf-8 -*-
"""
MnaImpactReport - M&A adjustment impact report for PACK-050.

Tracks mergers and acquisitions impact on consolidated GHG inventory.
Includes event timeline, pro-rata calculations, before/after boundary
comparison, base year restatement details, organic vs structural growth
separation, and disclosure notes.

Sections:
    1. M&A Summary (total events, net impact, organic vs structural)
    2. Event Timeline (chronological list of M&A events)
    3. Pro-Rata Calculations (partial-year inclusion details)
    4. Boundary Before/After (entity lists pre and post event)
    5. Base Year Restatement (restated values and methodology)
    6. Organic vs Structural Decomposition
    7. Disclosure Notes
    8. Provenance Footer

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

class MnaEvent(BaseModel):
    """Single M&A event."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    event_id: str = Field("")
    event_type: str = Field("", description="acquisition, divestiture, merger, spin_off")
    entity_name: str = Field("")
    effective_date: str = Field("")
    closing_date: str = Field("")
    ownership_acquired_pct: Decimal = Field(Decimal("0"))
    annual_emissions_tco2e: Decimal = Field(Decimal("0"))
    pro_rata_days: int = Field(0)
    pro_rata_factor: Decimal = Field(Decimal("0"))
    pro_rata_emissions_tco2e: Decimal = Field(Decimal("0"))
    description: str = Field("")


class BoundaryComparison(BaseModel):
    """Before/after boundary comparison for a single event."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    event_id: str = Field("")
    entity_name: str = Field("")
    entities_before: int = Field(0)
    entities_after: int = Field(0)
    emissions_before_tco2e: Decimal = Field(Decimal("0"))
    emissions_after_tco2e: Decimal = Field(Decimal("0"))
    delta_tco2e: Decimal = Field(Decimal("0"))


class BaseYearRestatement(BaseModel):
    """Base year restatement entry."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    base_year: int = Field(0)
    original_tco2e: Decimal = Field(Decimal("0"))
    restated_tco2e: Decimal = Field(Decimal("0"))
    restatement_delta_tco2e: Decimal = Field(Decimal("0"))
    restatement_pct: Decimal = Field(Decimal("0"))
    methodology: str = Field("")
    trigger_event: str = Field("")


class GrowthDecomposition(BaseModel):
    """Organic vs structural growth decomposition."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    total_change_tco2e: Decimal = Field(Decimal("0"))
    organic_change_tco2e: Decimal = Field(Decimal("0"))
    structural_change_tco2e: Decimal = Field(Decimal("0"))
    organic_pct: Decimal = Field(Decimal("0"))
    structural_pct: Decimal = Field(Decimal("0"))


class DisclosureNote(BaseModel):
    """M&A-related disclosure note."""
    note_id: str = Field("")
    framework: str = Field("")
    text: str = Field("")


class MnaImpactReportInput(BaseModel):
    """Complete input for the M&A impact report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    events: List[Dict[str, Any]] = Field(default_factory=list)
    boundary_comparisons: List[Dict[str, Any]] = Field(default_factory=list)
    base_year_restatements: List[Dict[str, Any]] = Field(default_factory=list)
    growth_decomposition: Optional[Dict[str, Any]] = Field(None)
    disclosure_notes: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class MnaImpactReportOutput(BaseModel):
    """Rendered M&A impact report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    event_count: int = Field(0)
    acquisitions_count: int = Field(0)
    divestitures_count: int = Field(0)
    total_mna_impact_tco2e: Decimal = Field(Decimal("0"))
    events: List[MnaEvent] = Field(default_factory=list)
    boundary_comparisons: List[BoundaryComparison] = Field(default_factory=list)
    base_year_restatements: List[BaseYearRestatement] = Field(default_factory=list)
    growth_decomposition: Optional[GrowthDecomposition] = Field(None)
    disclosure_notes: List[DisclosureNote] = Field(default_factory=list)
    provenance_hash: str = Field("")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class MnaImpactReport:
    """
    M&A impact report template for PACK-050.

    Produces a comprehensive M&A adjustment analysis with pro-rata
    calculations, boundary changes, base year restatements, and
    organic vs structural growth decomposition.

    Example:
        >>> tpl = MnaImpactReport()
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

    def render(self, data: Dict[str, Any]) -> MnaImpactReportOutput:
        """Render M&A impact report from input data."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        inp = MnaImpactReportInput(**data) if isinstance(data, dict) else data

        events = [MnaEvent(**e) if isinstance(e, dict) else e for e in inp.events]

        # Compute pro-rata emissions when not provided
        for ev in events:
            if ev.pro_rata_factor == Decimal("0") and ev.pro_rata_days > 0:
                ev.pro_rata_factor = (
                    Decimal(str(ev.pro_rata_days)) / Decimal("365")
                ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            if ev.pro_rata_emissions_tco2e == Decimal("0") and ev.annual_emissions_tco2e > Decimal("0"):
                ev.pro_rata_emissions_tco2e = (
                    ev.annual_emissions_tco2e * ev.pro_rata_factor
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Sort events chronologically
        events.sort(key=lambda e: e.effective_date)

        acquisitions = [e for e in events if e.event_type in ("acquisition", "merger")]
        divestitures = [e for e in events if e.event_type in ("divestiture", "spin_off")]

        total_impact = sum(
            e.pro_rata_emissions_tco2e if e.event_type in ("acquisition", "merger")
            else -e.pro_rata_emissions_tco2e
            for e in events
        )

        boundary_comps = [
            BoundaryComparison(**b) if isinstance(b, dict) else b
            for b in inp.boundary_comparisons
        ]
        restatements = [
            BaseYearRestatement(**r) if isinstance(r, dict) else r
            for r in inp.base_year_restatements
        ]
        for rs in restatements:
            if rs.restatement_delta_tco2e == Decimal("0"):
                rs.restatement_delta_tco2e = rs.restated_tco2e - rs.original_tco2e
            if rs.restatement_pct == Decimal("0") and rs.original_tco2e > Decimal("0"):
                rs.restatement_pct = (
                    rs.restatement_delta_tco2e / rs.original_tco2e * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        decomp = None
        if inp.growth_decomposition and isinstance(inp.growth_decomposition, dict):
            decomp = GrowthDecomposition(**inp.growth_decomposition)

        notes = [DisclosureNote(**n) if isinstance(n, dict) else n for n in inp.disclosure_notes]

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = MnaImpactReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            event_count=len(events),
            acquisitions_count=len(acquisitions),
            divestitures_count=len(divestitures),
            total_mna_impact_tco2e=total_impact,
            events=events,
            boundary_comparisons=boundary_comps,
            base_year_restatements=restatements,
            growth_decomposition=decomp,
            disclosure_notes=notes,
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

    def export_markdown(self, r: MnaImpactReportOutput) -> str:
        """Export report as Markdown."""
        lines: List[str] = []
        lines.append(f"# M&A Impact Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Events:** {r.event_count}")
        lines.append(f"**Acquisitions:** {r.acquisitions_count} | **Divestitures:** {r.divestitures_count}")
        lines.append(f"**Net M&A Impact:** {r.total_mna_impact_tco2e:,.2f} tCO2e")
        lines.append("")

        # Event timeline
        if r.events:
            lines.append("## Event Timeline")
            lines.append("| Date | Type | Entity | Ownership | Annual tCO2e | Pro-Rata Days | Factor | Pro-Rata tCO2e |")
            lines.append("|------|------|--------|-----------|-------------|---------------|--------|----------------|")
            for ev in r.events:
                lines.append(
                    f"| {ev.effective_date} | {ev.event_type} | {ev.entity_name} | "
                    f"{ev.ownership_acquired_pct}% | {ev.annual_emissions_tco2e:,.0f} | "
                    f"{ev.pro_rata_days} | {ev.pro_rata_factor} | {ev.pro_rata_emissions_tco2e:,.0f} |"
                )
            lines.append("")

        # Boundary comparison
        if r.boundary_comparisons:
            lines.append("## Boundary Before/After")
            lines.append("| Event | Entity | Entities Before | After | Emissions Before | After | Delta |")
            lines.append("|-------|--------|-----------------|-------|-----------------|-------|-------|")
            for bc in r.boundary_comparisons:
                lines.append(
                    f"| {bc.event_id} | {bc.entity_name} | {bc.entities_before} | "
                    f"{bc.entities_after} | {bc.emissions_before_tco2e:,.0f} | "
                    f"{bc.emissions_after_tco2e:,.0f} | {bc.delta_tco2e:,.0f} |"
                )
            lines.append("")

        # Base year restatement
        if r.base_year_restatements:
            lines.append("## Base Year Restatement")
            lines.append("| Base Year | Original | Restated | Delta | % | Methodology | Trigger |")
            lines.append("|-----------|----------|----------|-------|---|-------------|---------|")
            for rs in r.base_year_restatements:
                lines.append(
                    f"| {rs.base_year} | {rs.original_tco2e:,.0f} | {rs.restated_tco2e:,.0f} | "
                    f"{rs.restatement_delta_tco2e:,.0f} | {rs.restatement_pct}% | "
                    f"{rs.methodology} | {rs.trigger_event} |"
                )
            lines.append("")

        # Growth decomposition
        if r.growth_decomposition:
            gd = r.growth_decomposition
            lines.append("## Organic vs Structural Growth")
            lines.append("| Component | tCO2e | Share |")
            lines.append("|-----------|------:|------:|")
            lines.append(f"| Total Change | {gd.total_change_tco2e:,.2f} | 100% |")
            lines.append(f"| Organic | {gd.organic_change_tco2e:,.2f} | {gd.organic_pct}% |")
            lines.append(f"| Structural (M&A) | {gd.structural_change_tco2e:,.2f} | {gd.structural_pct}% |")
            lines.append("")

        # Disclosure notes
        if r.disclosure_notes:
            lines.append("## Disclosure Notes")
            for dn in r.disclosure_notes:
                lines.append(f"- **[{dn.framework}]** {dn.text}")
            lines.append("")

        lines.append(f"---\n*Report ID: {r.report_id} | Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: MnaImpactReportOutput) -> str:
        """Export report as HTML."""
        md = self.export_markdown(r)
        html_lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>M&A Impact Report - {r.company_name}</title>",
            "<style>body{font-family:sans-serif;margin:2em;}table{border-collapse:collapse;width:100%;margin:1em 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f5f5f5;}</style>",
            "</head><body>",
            f"<pre>{md}</pre>",
            "</body></html>",
        ]
        return "\n".join(html_lines)

    def export_json(self, r: MnaImpactReportOutput) -> Dict[str, Any]:
        """Export report as JSON-serializable dict."""
        return json.loads(r.model_dump_json())

    def export_csv(self, r: MnaImpactReportOutput) -> str:
        """Export M&A events as CSV."""
        lines_out = [
            "event_id,event_type,entity_name,effective_date,ownership_pct,"
            "annual_tco2e,pro_rata_days,pro_rata_factor,pro_rata_tco2e"
        ]
        for ev in r.events:
            lines_out.append(
                f"{ev.event_id},{ev.event_type},{ev.entity_name},{ev.effective_date},"
                f"{ev.ownership_acquired_pct},{ev.annual_emissions_tco2e},"
                f"{ev.pro_rata_days},{ev.pro_rata_factor},{ev.pro_rata_emissions_tco2e}"
            )
        return "\n".join(lines_out)


__all__ = [
    "MnaImpactReport",
    "MnaImpactReportInput",
    "MnaImpactReportOutput",
    "MnaEvent",
    "BoundaryComparison",
    "BaseYearRestatement",
    "GrowthDecomposition",
    "DisclosureNote",
]
