# -*- coding: utf-8 -*-
"""
SitePortfolioDashboard - All-sites overview for PACK-049.

Generates a multi-site portfolio dashboard with summary KPIs, geographic
distribution, facility type breakdown, scope breakdown, top emitters,
and status overview. All outputs include SHA-256 provenance hashing.

Sections:
    1. Summary KPIs (total sites, total tCO2e, avg intensity, quality score)
    2. Geographic Map Data (region counts, region emissions)
    3. Facility Type Breakdown (type counts, type emissions)
    4. Scope Breakdown (S1/S2/S3 totals and percentages)
    5. Top Emitters (top N sites by absolute emissions)
    6. Status Overview (active/inactive/pending counts)
    7. Provenance Footer

Output Formats: Markdown, HTML, JSON

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"

# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class SiteSummary(BaseModel):
    """Summary data for a single site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    site_name: str = Field("")
    country_code: str = Field("")
    region: str = Field("")
    facility_type: str = Field("")
    business_unit: str = Field("")
    status: str = Field("active")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_tco2e: Decimal = Field(Decimal("0"))
    intensity_tco2e_per_sqm: Decimal = Field(Decimal("0"))
    quality_score: Decimal = Field(Decimal("0"))
    data_completeness_pct: Decimal = Field(Decimal("0"))

class PortfolioDashboardInput(BaseModel):
    """Complete input for the portfolio dashboard."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    report_date: Optional[str] = Field(None)
    sites: List[SiteSummary] = Field(default_factory=list)
    top_n_emitters: int = Field(10, ge=1, le=50)
    consolidation_approach: str = Field("operational_control")
    currency_code: str = Field("USD")

# ---------------------------------------------------------------------------
# Output Models
# ---------------------------------------------------------------------------

class RegionData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    region: str = Field(...)
    site_count: int = Field(0)
    total_tco2e: Decimal = Field(Decimal("0"))
    share_pct: Decimal = Field(Decimal("0"))

class FacilityTypeData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    facility_type: str = Field(...)
    site_count: int = Field(0)
    total_tco2e: Decimal = Field(Decimal("0"))
    share_pct: Decimal = Field(Decimal("0"))

class ScopeBreakdown(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    scope_1_pct: Decimal = Field(Decimal("0"))
    scope_2_pct: Decimal = Field(Decimal("0"))
    scope_3_pct: Decimal = Field(Decimal("0"))

class TopEmitter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rank: int = Field(0)
    site_id: str = Field("")
    site_name: str = Field("")
    total_tco2e: Decimal = Field(Decimal("0"))
    share_pct: Decimal = Field(Decimal("0"))
    facility_type: str = Field("")

class StatusOverview(BaseModel):
    active: int = Field(0)
    inactive: int = Field(0)
    pending: int = Field(0)
    total: int = Field(0)

class PortfolioDashboardReport(BaseModel):
    """Rendered portfolio dashboard report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    total_sites: int = Field(0)
    total_tco2e: Decimal = Field(Decimal("0"))
    avg_intensity: Decimal = Field(Decimal("0"))
    avg_quality_score: Decimal = Field(Decimal("0"))
    avg_completeness_pct: Decimal = Field(Decimal("0"))
    geographic_data: List[RegionData] = Field(default_factory=list)
    facility_type_data: List[FacilityTypeData] = Field(default_factory=list)
    scope_breakdown: Optional[ScopeBreakdown] = Field(None)
    top_emitters: List[TopEmitter] = Field(default_factory=list)
    status_overview: Optional[StatusOverview] = Field(None)
    provenance_hash: str = Field("")

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class SitePortfolioDashboard:
    """
    Portfolio dashboard template for PACK-049 Multi-Site Management.

    Renders a comprehensive all-sites overview with KPIs, geographic
    distribution, facility type breakdown, scope split, top emitters,
    and status. All outputs include SHA-256 provenance.

    Example:
        >>> tpl = SitePortfolioDashboard()
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

    def render(self, data: Dict[str, Any]) -> PortfolioDashboardReport:
        """Render the portfolio dashboard from input data."""
        start = time.monotonic()
        self.generated_at = utcnow()

        inp = PortfolioDashboardInput(**data) if not isinstance(data, PortfolioDashboardInput) else data
        sites = inp.sites
        total = sum(s.total_tco2e for s in sites) or Decimal("1")

        # Geographic
        geo_buckets: Dict[str, List[SiteSummary]] = {}
        for s in sites:
            geo_buckets.setdefault(s.region or "Unknown", []).append(s)
        geo_data = []
        for region, group in sorted(geo_buckets.items()):
            rtotal = sum(s.total_tco2e for s in group)
            geo_data.append(RegionData(
                region=region, site_count=len(group), total_tco2e=rtotal,
                share_pct=(rtotal / total * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            ))

        # Facility type
        ft_buckets: Dict[str, List[SiteSummary]] = {}
        for s in sites:
            ft_buckets.setdefault(s.facility_type or "Other", []).append(s)
        ft_data = []
        for ft, group in sorted(ft_buckets.items()):
            ftotal = sum(s.total_tco2e for s in group)
            ft_data.append(FacilityTypeData(
                facility_type=ft, site_count=len(group), total_tco2e=ftotal,
                share_pct=(ftotal / total * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            ))

        # Scope breakdown
        s1 = sum(s.scope_1_tco2e for s in sites)
        s2 = sum(s.scope_2_tco2e for s in sites)
        s3 = sum(s.scope_3_tco2e for s in sites)
        stot = s1 + s2 + s3 or Decimal("1")
        scope = ScopeBreakdown(
            scope_1_tco2e=s1, scope_2_tco2e=s2, scope_3_tco2e=s3,
            scope_1_pct=(s1 / stot * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            scope_2_pct=(s2 / stot * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            scope_3_pct=(s3 / stot * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        )

        # Top emitters
        sorted_sites = sorted(sites, key=lambda s: s.total_tco2e, reverse=True)
        top_n = inp.top_n_emitters
        top_emitters = []
        for i, s in enumerate(sorted_sites[:top_n], 1):
            top_emitters.append(TopEmitter(
                rank=i, site_id=s.site_id, site_name=s.site_name,
                total_tco2e=s.total_tco2e,
                share_pct=(s.total_tco2e / total * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                facility_type=s.facility_type,
            ))

        # Status
        active = sum(1 for s in sites if s.status == "active")
        inactive = sum(1 for s in sites if s.status == "inactive")
        pending = sum(1 for s in sites if s.status not in ("active", "inactive"))
        status_ov = StatusOverview(active=active, inactive=inactive, pending=pending, total=len(sites))

        # Averages
        avg_intensity = Decimal("0")
        intensities = [s.intensity_tco2e_per_sqm for s in sites if s.intensity_tco2e_per_sqm > 0]
        if intensities:
            avg_intensity = (sum(intensities) / Decimal(str(len(intensities)))).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
        avg_quality = Decimal("0")
        qualities = [s.quality_score for s in sites if s.quality_score > 0]
        if qualities:
            avg_quality = (sum(qualities) / Decimal(str(len(qualities)))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        avg_compl = Decimal("0")
        if sites:
            avg_compl = (sum(s.data_completeness_pct for s in sites) / Decimal(str(len(sites)))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        report = PortfolioDashboardReport(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            total_sites=len(sites),
            total_tco2e=sum(s.total_tco2e for s in sites),
            avg_intensity=avg_intensity,
            avg_quality_score=avg_quality,
            avg_completeness_pct=avg_compl,
            geographic_data=geo_data,
            facility_type_data=ft_data,
            scope_breakdown=scope,
            top_emitters=top_emitters,
            status_overview=status_ov,
            provenance_hash=prov,
        )

        self.processing_time_ms = (time.monotonic() - start) * 1000
        return report

    # ------------------------------------------------------------------
    # EXPORT METHODS
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render as Markdown string."""
        report = self.render(data) if isinstance(data, dict) else data
        return self.export_markdown(report)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render as HTML string."""
        report = self.render(data) if isinstance(data, dict) else data
        return self.export_html(report)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as JSON dict."""
        report = self.render(data) if isinstance(data, dict) else data
        return self.export_json(report)

    def export_markdown(self, report: PortfolioDashboardReport) -> str:
        """Export report as Markdown."""
        lines: List[str] = []
        lines.append(f"# Site Portfolio Dashboard - {report.company_name}")
        lines.append(f"**Period:** {report.reporting_period} | **Generated:** {report.generated_at}")
        lines.append("")

        # Summary KPIs
        lines.append("## Summary KPIs")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Sites | {report.total_sites} |")
        lines.append(f"| Total Emissions | {report.total_tco2e:,.2f} tCO2e |")
        lines.append(f"| Avg Intensity | {report.avg_intensity:,.4f} tCO2e/sqm |")
        lines.append(f"| Avg Quality Score | {report.avg_quality_score:.1f}/100 |")
        lines.append(f"| Avg Completeness | {report.avg_completeness_pct:.1f}% |")
        lines.append("")

        # Geographic
        lines.append("## Geographic Distribution")
        lines.append("| Region | Sites | Emissions (tCO2e) | Share |")
        lines.append("|--------|-------|-------------------|-------|")
        for g in report.geographic_data:
            lines.append(f"| {g.region} | {g.site_count} | {g.total_tco2e:,.2f} | {g.share_pct}% |")
        lines.append("")

        # Facility type
        lines.append("## Facility Type Breakdown")
        lines.append("| Type | Sites | Emissions (tCO2e) | Share |")
        lines.append("|------|-------|-------------------|-------|")
        for f in report.facility_type_data:
            lines.append(f"| {f.facility_type} | {f.site_count} | {f.total_tco2e:,.2f} | {f.share_pct}% |")
        lines.append("")

        # Scope breakdown
        if report.scope_breakdown:
            sb = report.scope_breakdown
            lines.append("## Scope Breakdown")
            lines.append("| Scope | tCO2e | Share |")
            lines.append("|-------|-------|-------|")
            lines.append(f"| Scope 1 | {sb.scope_1_tco2e:,.2f} | {sb.scope_1_pct}% |")
            lines.append(f"| Scope 2 | {sb.scope_2_tco2e:,.2f} | {sb.scope_2_pct}% |")
            lines.append(f"| Scope 3 | {sb.scope_3_tco2e:,.2f} | {sb.scope_3_pct}% |")
            lines.append("")

        # Top emitters
        lines.append("## Top Emitters")
        lines.append("| Rank | Site | Type | tCO2e | Share |")
        lines.append("|------|------|------|-------|-------|")
        for t in report.top_emitters:
            lines.append(f"| {t.rank} | {t.site_name} | {t.facility_type} | {t.total_tco2e:,.2f} | {t.share_pct}% |")
        lines.append("")

        # Status
        if report.status_overview:
            so = report.status_overview
            lines.append("## Status Overview")
            lines.append(f"Active: {so.active} | Inactive: {so.inactive} | Pending: {so.pending} | Total: {so.total}")
            lines.append("")

        lines.append(f"---\n*Provenance: {report.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, report: PortfolioDashboardReport) -> str:
        """Export report as HTML."""
        md = self.export_markdown(report)
        html_lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>Site Portfolio Dashboard - {report.company_name}</title>",
            "<style>body{font-family:sans-serif;margin:2em;}table{border-collapse:collapse;width:100%;margin:1em 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f5f5f5;}</style>",
            "</head><body>",
            f"<pre>{md}</pre>",
            "</body></html>",
        ]
        return "\n".join(html_lines)

    def export_json(self, report: PortfolioDashboardReport) -> Dict[str, Any]:
        """Export report as JSON-serializable dict."""
        return json.loads(report.model_dump_json())

    def export_csv(self, report: PortfolioDashboardReport) -> str:
        """Export top emitters as CSV."""
        lines = ["rank,site_id,site_name,facility_type,total_tco2e,share_pct"]
        for t in report.top_emitters:
            lines.append(f"{t.rank},{t.site_id},{t.site_name},{t.facility_type},{t.total_tco2e},{t.share_pct}")
        return "\n".join(lines)

__all__ = [
    "SitePortfolioDashboard",
    "PortfolioDashboardInput",
    "PortfolioDashboardReport",
    "SiteSummary",
    "RegionData",
    "FacilityTypeData",
    "ScopeBreakdown",
    "TopEmitter",
    "StatusOverview",
]
