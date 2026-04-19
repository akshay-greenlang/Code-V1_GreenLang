# -*- coding: utf-8 -*-
"""
ScopeBreakdownReport - Consolidated scope 1/2/3 breakdown for PACK-050.

Generates a detailed breakdown of consolidated emissions by scope including
Scope 1 by source category (stationary combustion, mobile, process, fugitive),
Scope 2 dual reporting (location-based and market-based), Scope 3 by category
(1-15), geographic breakdown, and sector breakdown.

Sections:
    1. Scope Overview (S1/S2/S3 totals and percentages)
    2. Scope 1 By Source Category (stationary, mobile, process, fugitive, etc.)
    3. Scope 2 Dual Reporting (location-based vs market-based comparison)
    4. Scope 3 By Category (15 categories with materiality flags)
    5. Geographic Breakdown (emissions by country/region)
    6. Sector Breakdown (emissions by business sector)
    7. Provenance Footer

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class ScopeOverviewLine(BaseModel):
    """High-level scope totals."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scope: str = Field("")
    tco2e: Decimal = Field(Decimal("0"))
    share_pct: Decimal = Field(Decimal("0"))
    yoy_change_pct: Optional[Decimal] = Field(None)

class Scope1SourceCategory(BaseModel):
    """Scope 1 emission source category."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    category: str = Field("", description="stationary_combustion, mobile, process, fugitive, other")
    tco2e: Decimal = Field(Decimal("0"))
    share_of_scope1_pct: Decimal = Field(Decimal("0"))
    entity_count: int = Field(0)
    source_count: int = Field(0)

class Scope2DualReporting(BaseModel):
    """Scope 2 dual reporting comparison."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    location_based_tco2e: Decimal = Field(Decimal("0"))
    market_based_tco2e: Decimal = Field(Decimal("0"))
    difference_tco2e: Decimal = Field(Decimal("0"))
    difference_pct: Decimal = Field(Decimal("0"))
    rec_retired_mwh: Decimal = Field(Decimal("0"))
    ppa_volume_mwh: Decimal = Field(Decimal("0"))
    residual_mix_tco2e: Decimal = Field(Decimal("0"))

class Scope3CategoryLine(BaseModel):
    """Scope 3 category-level breakdown."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    category_number: int = Field(0, ge=1, le=15)
    category_name: str = Field("")
    tco2e: Decimal = Field(Decimal("0"))
    share_of_scope3_pct: Decimal = Field(Decimal("0"))
    data_quality: str = Field("", description="primary, secondary, estimated, not_relevant")
    is_material: bool = Field(True)
    methodology: str = Field("")

class GeographicEmissions(BaseModel):
    """Emissions by geographic region."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    region: str = Field("")
    country_code: str = Field("")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_tco2e: Decimal = Field(Decimal("0"))
    share_pct: Decimal = Field(Decimal("0"))

class SectorEmissions(BaseModel):
    """Emissions by business sector."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sector: str = Field("")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_tco2e: Decimal = Field(Decimal("0"))
    share_pct: Decimal = Field(Decimal("0"))
    entity_count: int = Field(0)

class ScopeBreakdownReportInput(BaseModel):
    """Complete input for the scope breakdown report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    scope_overview: List[Dict[str, Any]] = Field(default_factory=list)
    scope_1_sources: List[Dict[str, Any]] = Field(default_factory=list)
    scope_2_dual: Optional[Dict[str, Any]] = Field(None)
    scope_3_categories: List[Dict[str, Any]] = Field(default_factory=list)
    geographic: List[Dict[str, Any]] = Field(default_factory=list)
    sectors: List[Dict[str, Any]] = Field(default_factory=list)
    total_scope_1_tco2e: Decimal = Field(Decimal("0"))
    total_scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    total_scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    total_scope_3_tco2e: Decimal = Field(Decimal("0"))

# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class ScopeBreakdownReportOutput(BaseModel):
    """Rendered scope breakdown report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    total_scope_1_tco2e: Decimal = Field(Decimal("0"))
    total_scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    total_scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    total_scope_3_tco2e: Decimal = Field(Decimal("0"))
    grand_total_location_tco2e: Decimal = Field(Decimal("0"))
    grand_total_market_tco2e: Decimal = Field(Decimal("0"))
    scope_overview: List[ScopeOverviewLine] = Field(default_factory=list)
    scope_1_sources: List[Scope1SourceCategory] = Field(default_factory=list)
    scope_2_dual: Optional[Scope2DualReporting] = Field(None)
    scope_3_categories: List[Scope3CategoryLine] = Field(default_factory=list)
    scope_3_material_count: int = Field(0)
    scope_3_not_relevant_count: int = Field(0)
    geographic: List[GeographicEmissions] = Field(default_factory=list)
    sectors: List[SectorEmissions] = Field(default_factory=list)
    provenance_hash: str = Field("")

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ScopeBreakdownReport:
    """
    Consolidated scope 1/2/3 breakdown report template for PACK-050.

    Produces detailed scope analysis with source categories, dual reporting,
    category-level Scope 3, geographic, and sector breakdowns.

    Example:
        >>> tpl = ScopeBreakdownReport()
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

    def render(self, data: Dict[str, Any]) -> ScopeBreakdownReportOutput:
        """Render scope breakdown from input data."""
        start = time.monotonic()
        self.generated_at = utcnow()
        inp = ScopeBreakdownReportInput(**data) if isinstance(data, dict) else data

        overview = [ScopeOverviewLine(**s) if isinstance(s, dict) else s for s in inp.scope_overview]
        s1_sources = [Scope1SourceCategory(**s) if isinstance(s, dict) else s for s in inp.scope_1_sources]
        s3_cats = [Scope3CategoryLine(**c) if isinstance(c, dict) else c for c in inp.scope_3_categories]
        s3_cats.sort(key=lambda c: c.category_number)

        s2_dual = None
        if inp.scope_2_dual and isinstance(inp.scope_2_dual, dict):
            s2_dual = Scope2DualReporting(**inp.scope_2_dual)
            if s2_dual.difference_tco2e == Decimal("0"):
                s2_dual.difference_tco2e = s2_dual.location_based_tco2e - s2_dual.market_based_tco2e
            if s2_dual.difference_pct == Decimal("0") and s2_dual.location_based_tco2e > Decimal("0"):
                s2_dual.difference_pct = (
                    s2_dual.difference_tco2e / s2_dual.location_based_tco2e * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        geo = [GeographicEmissions(**g) if isinstance(g, dict) else g for g in inp.geographic]
        sectors = [SectorEmissions(**s) if isinstance(s, dict) else s for s in inp.sectors]

        material_count = sum(1 for c in s3_cats if c.is_material)
        not_relevant = sum(1 for c in s3_cats if c.data_quality == "not_relevant")

        grand_loc = inp.total_scope_1_tco2e + inp.total_scope_2_location_tco2e + inp.total_scope_3_tco2e
        grand_mkt = inp.total_scope_1_tco2e + inp.total_scope_2_market_tco2e + inp.total_scope_3_tco2e

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = ScopeBreakdownReportOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            total_scope_1_tco2e=inp.total_scope_1_tco2e,
            total_scope_2_location_tco2e=inp.total_scope_2_location_tco2e,
            total_scope_2_market_tco2e=inp.total_scope_2_market_tco2e,
            total_scope_3_tco2e=inp.total_scope_3_tco2e,
            grand_total_location_tco2e=grand_loc,
            grand_total_market_tco2e=grand_mkt,
            scope_overview=overview,
            scope_1_sources=s1_sources,
            scope_2_dual=s2_dual,
            scope_3_categories=s3_cats,
            scope_3_material_count=material_count,
            scope_3_not_relevant_count=not_relevant,
            geographic=geo,
            sectors=sectors,
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

    def export_markdown(self, r: ScopeBreakdownReportOutput) -> str:
        """Export report as Markdown."""
        lines: List[str] = []
        lines.append(f"# Scope Breakdown Report - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period}")
        lines.append(f"**Grand Total (Location):** {r.grand_total_location_tco2e:,.2f} tCO2e | **Grand Total (Market):** {r.grand_total_market_tco2e:,.2f} tCO2e")
        lines.append("")

        # Scope overview
        lines.append("## Scope Overview")
        lines.append("| Scope | tCO2e | Share | YoY Change |")
        lines.append("|-------|------:|------:|----------:|")
        for so in r.scope_overview:
            yoy = f"{so.yoy_change_pct}%" if so.yoy_change_pct is not None else "N/A"
            lines.append(f"| {so.scope} | {so.tco2e:,.2f} | {so.share_pct}% | {yoy} |")
        lines.append("")

        # Scope 1 sources
        if r.scope_1_sources:
            lines.append(f"## Scope 1 by Source Category ({r.total_scope_1_tco2e:,.2f} tCO2e)")
            lines.append("| Category | tCO2e | Share of S1 | Entities | Sources |")
            lines.append("|----------|------:|------------|----------|---------|")
            for s in r.scope_1_sources:
                lines.append(
                    f"| {s.category} | {s.tco2e:,.2f} | {s.share_of_scope1_pct}% | "
                    f"{s.entity_count} | {s.source_count} |"
                )
            lines.append("")

        # Scope 2 dual reporting
        if r.scope_2_dual:
            d = r.scope_2_dual
            lines.append("## Scope 2 Dual Reporting")
            lines.append("| Metric | Value |")
            lines.append("|--------|------:|")
            lines.append(f"| Location-Based | {d.location_based_tco2e:,.2f} tCO2e |")
            lines.append(f"| Market-Based | {d.market_based_tco2e:,.2f} tCO2e |")
            lines.append(f"| Difference | {d.difference_tco2e:,.2f} tCO2e ({d.difference_pct}%) |")
            lines.append(f"| RECs Retired | {d.rec_retired_mwh:,.0f} MWh |")
            lines.append(f"| PPA Volume | {d.ppa_volume_mwh:,.0f} MWh |")
            lines.append(f"| Residual Mix | {d.residual_mix_tco2e:,.2f} tCO2e |")
            lines.append("")

        # Scope 3 categories
        if r.scope_3_categories:
            lines.append(f"## Scope 3 by Category ({r.total_scope_3_tco2e:,.2f} tCO2e)")
            lines.append(f"Material categories: {r.scope_3_material_count} | Not relevant: {r.scope_3_not_relevant_count}")
            lines.append("")
            lines.append("| Cat# | Category | tCO2e | Share | Quality | Material | Method |")
            lines.append("|------|----------|------:|------:|---------|----------|--------|")
            for c in r.scope_3_categories:
                mat = "Yes" if c.is_material else "No"
                lines.append(
                    f"| {c.category_number} | {c.category_name} | {c.tco2e:,.2f} | "
                    f"{c.share_of_scope3_pct}% | {c.data_quality} | {mat} | {c.methodology} |"
                )
            lines.append("")

        # Geographic
        if r.geographic:
            lines.append("## Geographic Breakdown")
            lines.append("| Region | Country | S1 | S2 | S3 | Total | Share |")
            lines.append("|--------|---------|----|----|-----|-------|-------|")
            for g in r.geographic:
                lines.append(
                    f"| {g.region} | {g.country_code} | {g.scope_1_tco2e:,.0f} | "
                    f"{g.scope_2_tco2e:,.0f} | {g.scope_3_tco2e:,.0f} | "
                    f"{g.total_tco2e:,.0f} | {g.share_pct}% |"
                )
            lines.append("")

        # Sector
        if r.sectors:
            lines.append("## Sector Breakdown")
            lines.append("| Sector | S1 | S2 | S3 | Total | Share | Entities |")
            lines.append("|--------|----|----|-----|-------|-------|----------|")
            for s in r.sectors:
                lines.append(
                    f"| {s.sector} | {s.scope_1_tco2e:,.0f} | {s.scope_2_tco2e:,.0f} | "
                    f"{s.scope_3_tco2e:,.0f} | {s.total_tco2e:,.0f} | "
                    f"{s.share_pct}% | {s.entity_count} |"
                )
            lines.append("")

        lines.append(f"---\n*Report ID: {r.report_id} | Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: ScopeBreakdownReportOutput) -> str:
        """Export report as HTML."""
        md = self.export_markdown(r)
        html_lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>Scope Breakdown - {r.company_name}</title>",
            "<style>body{font-family:sans-serif;margin:2em;}table{border-collapse:collapse;width:100%;margin:1em 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f5f5f5;}</style>",
            "</head><body>",
            f"<pre>{md}</pre>",
            "</body></html>",
        ]
        return "\n".join(html_lines)

    def export_json(self, r: ScopeBreakdownReportOutput) -> Dict[str, Any]:
        """Export report as JSON-serializable dict."""
        return json.loads(r.model_dump_json())

    def export_csv(self, r: ScopeBreakdownReportOutput) -> str:
        """Export scope 3 categories as CSV."""
        lines_out = [
            "category_number,category_name,tco2e,share_of_scope3_pct,"
            "data_quality,is_material,methodology"
        ]
        for c in r.scope_3_categories:
            lines_out.append(
                f"{c.category_number},{c.category_name},{c.tco2e},"
                f"{c.share_of_scope3_pct},{c.data_quality},{c.is_material},{c.methodology}"
            )
        return "\n".join(lines_out)

__all__ = [
    "ScopeBreakdownReport",
    "ScopeBreakdownReportInput",
    "ScopeBreakdownReportOutput",
    "ScopeOverviewLine",
    "Scope1SourceCategory",
    "Scope2DualReporting",
    "Scope3CategoryLine",
    "GeographicEmissions",
    "SectorEmissions",
]
