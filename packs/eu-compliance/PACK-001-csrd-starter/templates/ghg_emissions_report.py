# -*- coding: utf-8 -*-
"""
PACK-001 Phase 3: GHG Emissions Report Template
=================================================

Generates a comprehensive GHG emissions report aligned with GHG Protocol
Corporate Standard and ESRS E1. Covers Scope 1 (8 sub-categories),
Scope 2 (location/market-based, steam, cooling), Scope 3 (all 15
categories), intensity metrics, waterfall data, methodology references,
and science-based targets progress.

Output formats: Markdown, HTML, JSON.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class Scope1Source(str, Enum):
    """Scope 1 emission source sub-categories."""
    STATIONARY_COMBUSTION = "Stationary Combustion"
    MOBILE_COMBUSTION = "Mobile Combustion"
    PROCESS_EMISSIONS = "Process Emissions"
    FUGITIVE_EMISSIONS = "Fugitive Emissions"
    REFRIGERANTS = "Refrigerants & F-Gas"
    LAND_USE = "Land Use"
    WASTE_TREATMENT = "Waste Treatment"
    AGRICULTURAL = "Agricultural"


class Scope2Method(str, Enum):
    """Scope 2 accounting methods."""
    LOCATION_BASED = "Location-Based"
    MARKET_BASED = "Market-Based"
    STEAM_HEAT = "Steam/Heat Purchase"
    COOLING = "Cooling Purchase"


class IntensityDenominator(str, Enum):
    """Denominators for intensity metrics."""
    REVENUE = "revenue"
    EMPLOYEE = "employee"
    UNIT_PRODUCED = "unit_produced"
    SQUARE_METER = "square_meter"
    CUSTOM = "custom"


class TrendDirection(str, Enum):
    """Emission trend direction."""
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    STABLE = "STABLE"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class Scope1Breakdown(BaseModel):
    """Breakdown of Scope 1 emissions by source."""
    source: Scope1Source = Field(..., description="Emission source sub-category")
    emissions_tco2e: float = Field(0.0, ge=0.0, description="Emissions in tCO2e")
    co2_tco2e: Optional[float] = Field(None, ge=0.0, description="CO2 component")
    ch4_tco2e: Optional[float] = Field(None, ge=0.0, description="CH4 component in CO2e")
    n2o_tco2e: Optional[float] = Field(None, ge=0.0, description="N2O component in CO2e")
    hfc_tco2e: Optional[float] = Field(None, ge=0.0, description="HFC component in CO2e")
    pct_of_scope1: Optional[float] = Field(None, ge=0.0, le=100.0, description="% of total Scope 1")
    data_quality: str = Field("HIGH", description="Data quality level")
    notes: Optional[str] = Field(None, description="Additional notes")


class Scope2Breakdown(BaseModel):
    """Breakdown of Scope 2 emissions by method."""
    method: Scope2Method = Field(..., description="Accounting method")
    emissions_tco2e: float = Field(0.0, ge=0.0, description="Emissions in tCO2e")
    energy_consumed_mwh: Optional[float] = Field(None, ge=0.0, description="Energy consumed (MWh)")
    emission_factor: Optional[float] = Field(None, ge=0.0, description="Applied emission factor")
    emission_factor_source: Optional[str] = Field(None, description="Factor source reference")
    pct_of_scope2: Optional[float] = Field(None, ge=0.0, le=100.0, description="% of total Scope 2")
    recs_applied_mwh: Optional[float] = Field(None, ge=0.0, description="RECs applied (MWh)")
    notes: Optional[str] = Field(None, description="Additional notes")


class Scope3Category(BaseModel):
    """Single Scope 3 category emissions data."""
    category_number: int = Field(..., ge=1, le=15, description="Category 1-15")
    category_name: str = Field(..., description="Category name")
    emissions_tco2e: float = Field(0.0, ge=0.0, description="Emissions in tCO2e")
    calculation_approach: str = Field("", description="Spend-based, hybrid, supplier-specific, etc.")
    pct_of_scope3: Optional[float] = Field(None, ge=0.0, le=100.0, description="% of total Scope 3")
    data_quality: str = Field("MEDIUM", description="Data quality level")
    is_relevant: bool = Field(True, description="Whether category is relevant")
    exclusion_rationale: Optional[str] = Field(None, description="Rationale if not relevant")
    notes: Optional[str] = Field(None, description="Additional notes")


class IntensityMetric(BaseModel):
    """GHG intensity metric."""
    denominator_type: IntensityDenominator = Field(..., description="Denominator type")
    denominator_label: str = Field("", description="Human-readable denominator label")
    denominator_value: float = Field(..., gt=0.0, description="Denominator value")
    denominator_unit: str = Field("", description="Denominator unit")
    numerator_tco2e: float = Field(..., ge=0.0, description="Numerator in tCO2e")
    scope_coverage: str = Field("1+2", description="Which scopes are included")
    intensity_value: Optional[float] = Field(None, description="Computed intensity value")
    prior_year_intensity: Optional[float] = Field(None, description="Prior year intensity")

    def model_post_init(self, __context: Any) -> None:
        """Compute intensity value."""
        if self.intensity_value is None and self.denominator_value > 0:
            self.intensity_value = self.numerator_tco2e / self.denominator_value

    @property
    def yoy_change_pct(self) -> Optional[float]:
        """Year-over-year change in intensity."""
        if self.intensity_value is None or self.prior_year_intensity is None:
            return None
        if self.prior_year_intensity == 0:
            return None
        return ((self.intensity_value - self.prior_year_intensity) / abs(self.prior_year_intensity)) * 100


class EmissionTrend(BaseModel):
    """Historical emission data point for trend charting."""
    year: int = Field(..., ge=2015, le=2100, description="Year")
    scope1_tco2e: float = Field(0.0, ge=0.0, description="Scope 1 total")
    scope2_tco2e: float = Field(0.0, ge=0.0, description="Scope 2 total")
    scope3_tco2e: float = Field(0.0, ge=0.0, description="Scope 3 total")

    @property
    def total_tco2e(self) -> float:
        """Total emissions across all scopes."""
        return self.scope1_tco2e + self.scope2_tco2e + self.scope3_tco2e


class MethodologyReference(BaseModel):
    """Reference to a methodology, emission factor database, or guidance."""
    reference_id: str = Field(..., description="Short reference key")
    title: str = Field(..., description="Full reference title")
    source: str = Field("", description="Publisher or source organization")
    version_year: Optional[int] = Field(None, description="Version or publication year")
    url: Optional[str] = Field(None, description="URL for reference")
    notes: Optional[str] = Field(None, description="Usage notes")


class SBTiProgress(BaseModel):
    """Science-Based Targets initiative progress."""
    target_type: str = Field("", description="Near-term, Long-term, Net-Zero")
    base_year: int = Field(2020, description="Base year")
    target_year: int = Field(2030, description="Target year")
    target_reduction_pct: float = Field(0.0, description="Target reduction %")
    actual_reduction_pct: Optional[float] = Field(None, description="Actual reduction % achieved")
    scope_coverage: str = Field("1+2", description="Scopes covered")
    on_track: Optional[bool] = Field(None, description="Whether on track to meet target")
    notes: Optional[str] = Field(None, description="Additional notes")


class GHGEmissionsInput(BaseModel):
    """Full input for the GHG emissions report."""
    company_name: str = Field(..., description="Reporting entity")
    reporting_year: int = Field(..., ge=2020, le=2100, description="Fiscal year")
    report_date: date = Field(default_factory=date.today, description="Generation date")
    base_year: Optional[int] = Field(None, description="Emissions base year for comparison")
    scope1_breakdown: List[Scope1Breakdown] = Field(default_factory=list, description="Scope 1 detail")
    scope2_breakdown: List[Scope2Breakdown] = Field(default_factory=list, description="Scope 2 detail")
    scope3_categories: List[Scope3Category] = Field(default_factory=list, description="Scope 3 detail")
    intensity_metrics: List[IntensityMetric] = Field(default_factory=list, description="Intensity metrics")
    historical_trends: List[EmissionTrend] = Field(default_factory=list, description="Historical data")
    methodologies: List[MethodologyReference] = Field(default_factory=list, description="References")
    sbti_progress: Optional[SBTiProgress] = Field(None, description="SBTi targets progress")

    @property
    def scope1_total(self) -> float:
        """Total Scope 1 emissions."""
        return sum(s.emissions_tco2e for s in self.scope1_breakdown)

    @property
    def scope2_total(self) -> float:
        """Total Scope 2 emissions (sum of all methods)."""
        return sum(s.emissions_tco2e for s in self.scope2_breakdown)

    @property
    def scope3_total(self) -> float:
        """Total Scope 3 emissions."""
        return sum(c.emissions_tco2e for c in self.scope3_categories if c.is_relevant)

    @property
    def grand_total(self) -> float:
        """Grand total across all scopes."""
        return self.scope1_total + self.scope2_total + self.scope3_total


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _fmt_tco2e(value: float) -> str:
    """Format tCO2e with scale suffix."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M tCO2e"
    if value >= 1_000:
        return f"{value / 1_000:,.1f}K tCO2e"
    return f"{value:,.1f} tCO2e"


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage with sign."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _pct_of_total(part: float, total: float) -> str:
    """Calculate and format percentage of total."""
    if total == 0:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class GHGEmissionsReportTemplate:
    """Generate comprehensive GHG emissions report.

    Sections:
        1. Total GHG Emissions (tCO2e, YoY change)
        2. Scope 1 Breakdown (8 sub-categories)
        3. Scope 2 Breakdown (location, market, steam, cooling)
        4. Scope 3 Breakdown (all 15 categories)
        5. Emission Intensity Metrics
        6. Waterfall Chart Data
        7. Methodology & Emission Factors Used
        8. Science-Based Targets Progress

    Example:
        >>> template = GHGEmissionsReportTemplate()
        >>> data = GHGEmissionsInput(company_name="Acme", reporting_year=2025, ...)
        >>> md = template.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize the GHG emissions report template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: GHGEmissionsInput) -> str:
        """Render complete GHG report as Markdown.

        Args:
            data: Validated GHG input data.

        Returns:
            Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_totals(data),
            self._md_scope1(data),
            self._md_scope2(data),
            self._md_scope3(data),
            self._md_intensity(data),
            self._md_waterfall(data),
            self._md_methodology(data),
            self._md_sbti(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: GHGEmissionsInput) -> str:
        """Render complete GHG report as HTML.

        Args:
            data: Validated GHG input data.

        Returns:
            HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_totals(data),
            self._html_scope1(data),
            self._html_scope2(data),
            self._html_scope3(data),
            self._html_intensity(data),
            self._html_methodology(data),
            self._html_sbti(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data.company_name, data.reporting_year, body)

    def render_json(self, data: GHGEmissionsInput) -> Dict[str, Any]:
        """Render as JSON-serializable dict.

        Args:
            data: Validated GHG input data.

        Returns:
            Dictionary for serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance = self._compute_provenance(data)
        waterfall = self._build_waterfall(data)

        return {
            "template": "ghg_emissions_report",
            "version": "1.0.0",
            "generated_at": self._render_timestamp.isoformat(),
            "provenance_hash": provenance,
            "company_name": data.company_name,
            "reporting_year": data.reporting_year,
            "base_year": data.base_year,
            "totals": {
                "scope1_tco2e": data.scope1_total,
                "scope2_tco2e": data.scope2_total,
                "scope3_tco2e": data.scope3_total,
                "grand_total_tco2e": data.grand_total,
            },
            "scope1_breakdown": [s.model_dump(mode="json") for s in data.scope1_breakdown],
            "scope2_breakdown": [s.model_dump(mode="json") for s in data.scope2_breakdown],
            "scope3_categories": [c.model_dump(mode="json") for c in data.scope3_categories],
            "intensity_metrics": [m.model_dump(mode="json") for m in data.intensity_metrics],
            "waterfall_data": waterfall,
            "historical_trends": [t.model_dump(mode="json") for t in data.historical_trends],
            "methodologies": [m.model_dump(mode="json") for m in data.methodologies],
            "sbti_progress": data.sbti_progress.model_dump(mode="json") if data.sbti_progress else None,
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance(self, data: GHGEmissionsInput) -> str:
        """Compute SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # WATERFALL DATA
    # --------------------------------------------------------------------- #

    def _build_waterfall(self, data: GHGEmissionsInput) -> List[Dict[str, Any]]:
        """Build waterfall chart data showing contribution breakdown."""
        waterfall: List[Dict[str, Any]] = []
        running = 0.0
        for s1 in data.scope1_breakdown:
            running += s1.emissions_tco2e
            waterfall.append({
                "label": f"S1: {s1.source.value}",
                "value": s1.emissions_tco2e,
                "cumulative": running,
                "scope": "Scope 1",
            })
        for s2 in data.scope2_breakdown:
            running += s2.emissions_tco2e
            waterfall.append({
                "label": f"S2: {s2.method.value}",
                "value": s2.emissions_tco2e,
                "cumulative": running,
                "scope": "Scope 2",
            })
        for s3 in data.scope3_categories:
            if s3.is_relevant and s3.emissions_tco2e > 0:
                running += s3.emissions_tco2e
                waterfall.append({
                    "label": f"S3 Cat {s3.category_number}",
                    "value": s3.emissions_tco2e,
                    "cumulative": running,
                    "scope": "Scope 3",
                })
        waterfall.append({
            "label": "Total",
            "value": data.grand_total,
            "cumulative": data.grand_total,
            "scope": "Total",
        })
        return waterfall

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: GHGEmissionsInput) -> str:
        """Markdown header."""
        base = f" | **Base Year:** {data.base_year}" if data.base_year else ""
        return (
            f"# GHG Emissions Report - {data.company_name}\n"
            f"**Reporting Year:** {data.reporting_year} | "
            f"**Report Date:** {data.report_date.isoformat()}{base}\n\n---"
        )

    def _md_totals(self, data: GHGEmissionsInput) -> str:
        """Total emissions summary."""
        lines = [
            "## 1. Total GHG Emissions",
            "",
            "| Scope | Emissions | % of Total |",
            "|-------|----------|------------|",
            f"| Scope 1 | {_fmt_tco2e(data.scope1_total)} | {_pct_of_total(data.scope1_total, data.grand_total)} |",
            f"| Scope 2 | {_fmt_tco2e(data.scope2_total)} | {_pct_of_total(data.scope2_total, data.grand_total)} |",
            f"| Scope 3 | {_fmt_tco2e(data.scope3_total)} | {_pct_of_total(data.scope3_total, data.grand_total)} |",
            f"| **Grand Total** | **{_fmt_tco2e(data.grand_total)}** | **100.0%** |",
        ]
        if data.historical_trends:
            prev = next(
                (t for t in sorted(data.historical_trends, key=lambda x: x.year, reverse=True)
                 if t.year < data.reporting_year), None
            )
            if prev:
                change = ((data.grand_total - prev.total_tco2e) / prev.total_tco2e * 100) if prev.total_tco2e else 0
                lines.append(f"\n**Year-over-Year Change:** {_fmt_pct(change)} vs {prev.year}")
        return "\n".join(lines)

    def _md_scope1(self, data: GHGEmissionsInput) -> str:
        """Scope 1 breakdown."""
        lines = [
            "## 2. Scope 1 Breakdown",
            "",
            "| Source | Emissions | % of Scope 1 | CO2 | CH4 | N2O | Quality |",
            "|--------|----------|-------------|-----|-----|-----|---------|",
        ]
        s1_total = data.scope1_total
        for s in sorted(data.scope1_breakdown, key=lambda x: x.emissions_tco2e, reverse=True):
            pct = _pct_of_total(s.emissions_tco2e, s1_total)
            co2 = _fmt_tco2e(s.co2_tco2e) if s.co2_tco2e is not None else "-"
            ch4 = _fmt_tco2e(s.ch4_tco2e) if s.ch4_tco2e is not None else "-"
            n2o = _fmt_tco2e(s.n2o_tco2e) if s.n2o_tco2e is not None else "-"
            lines.append(
                f"| {s.source.value} | {_fmt_tco2e(s.emissions_tco2e)} | {pct} "
                f"| {co2} | {ch4} | {n2o} | {s.data_quality} |"
            )
        if not data.scope1_breakdown:
            lines.append("| - | No Scope 1 data | - | - | - | - | - |")
        lines.append(f"\n**Scope 1 Total:** {_fmt_tco2e(s1_total)}")
        return "\n".join(lines)

    def _md_scope2(self, data: GHGEmissionsInput) -> str:
        """Scope 2 breakdown."""
        lines = [
            "## 3. Scope 2 Breakdown",
            "",
            "| Method | Emissions | Energy (MWh) | EF | EF Source | RECs | Notes |",
            "|--------|----------|-------------|-----|----------|------|-------|",
        ]
        for s in data.scope2_breakdown:
            energy = f"{s.energy_consumed_mwh:,.0f}" if s.energy_consumed_mwh is not None else "-"
            ef = f"{s.emission_factor:.4f}" if s.emission_factor is not None else "-"
            ef_src = s.emission_factor_source or "-"
            recs = f"{s.recs_applied_mwh:,.0f}" if s.recs_applied_mwh is not None else "-"
            notes = s.notes or "-"
            lines.append(
                f"| {s.method.value} | {_fmt_tco2e(s.emissions_tco2e)} | {energy} "
                f"| {ef} | {ef_src} | {recs} | {notes} |"
            )
        if not data.scope2_breakdown:
            lines.append("| - | No Scope 2 data | - | - | - | - | - |")
        lines.append(f"\n**Scope 2 Total:** {_fmt_tco2e(data.scope2_total)}")
        return "\n".join(lines)

    def _md_scope3(self, data: GHGEmissionsInput) -> str:
        """Scope 3 breakdown."""
        lines = [
            "## 4. Scope 3 Breakdown",
            "",
            "| Cat | Name | Emissions | % of S3 | Approach | Quality | Relevant |",
            "|-----|------|----------|---------|----------|---------|----------|",
        ]
        s3_total = data.scope3_total
        for c in sorted(data.scope3_categories, key=lambda x: x.category_number):
            pct = _pct_of_total(c.emissions_tco2e, s3_total)
            relevant = "Yes" if c.is_relevant else "No"
            lines.append(
                f"| {c.category_number} | {c.category_name} | "
                f"{_fmt_tco2e(c.emissions_tco2e)} | {pct} | "
                f"{c.calculation_approach} | {c.data_quality} | {relevant} |"
            )
        if not data.scope3_categories:
            lines.append("| - | No Scope 3 data | - | - | - | - | - |")
        lines.append(f"\n**Scope 3 Total:** {_fmt_tco2e(s3_total)}")
        return "\n".join(lines)

    def _md_intensity(self, data: GHGEmissionsInput) -> str:
        """Intensity metrics."""
        if not data.intensity_metrics:
            return "## 5. Emission Intensity Metrics\n\nNo intensity metrics configured."
        lines = [
            "## 5. Emission Intensity Metrics",
            "",
            "| Denominator | Scopes | Numerator | Denominator Value | Intensity | Prior Year | YoY Change |",
            "|-------------|--------|-----------|-------------------|-----------|------------|------------|",
        ]
        for m in data.intensity_metrics:
            label = m.denominator_label or m.denominator_type.value
            intensity = f"{m.intensity_value:,.4f}" if m.intensity_value is not None else "N/A"
            prior = f"{m.prior_year_intensity:,.4f}" if m.prior_year_intensity is not None else "N/A"
            yoy = _fmt_pct(m.yoy_change_pct)
            lines.append(
                f"| {label} | {m.scope_coverage} | {_fmt_tco2e(m.numerator_tco2e)} "
                f"| {m.denominator_value:,.0f} {m.denominator_unit} | {intensity} "
                f"| {prior} | {yoy} |"
            )
        return "\n".join(lines)

    def _md_waterfall(self, data: GHGEmissionsInput) -> str:
        """Waterfall chart data in tabular form."""
        waterfall = self._build_waterfall(data)
        if len(waterfall) <= 1:
            return "## 6. Waterfall Chart Data\n\nInsufficient data for waterfall."
        lines = [
            "## 6. Waterfall Chart Data",
            "",
            "| Component | Value (tCO2e) | Cumulative | Scope |",
            "|-----------|--------------|-----------|-------|",
        ]
        for w in waterfall:
            lines.append(
                f"| {w['label']} | {_fmt_tco2e(w['value'])} | "
                f"{_fmt_tco2e(w['cumulative'])} | {w['scope']} |"
            )
        return "\n".join(lines)

    def _md_methodology(self, data: GHGEmissionsInput) -> str:
        """Methodology and references."""
        if not data.methodologies:
            return "## 7. Methodology & Emission Factors\n\nNo methodology references provided."
        lines = [
            "## 7. Methodology & Emission Factors",
            "",
            "| Ref | Title | Source | Year | Notes |",
            "|-----|-------|--------|------|-------|",
        ]
        for m in data.methodologies:
            yr = str(m.version_year) if m.version_year else "-"
            notes = m.notes or "-"
            lines.append(
                f"| {m.reference_id} | {m.title} | {m.source} | {yr} | {notes} |"
            )
        return "\n".join(lines)

    def _md_sbti(self, data: GHGEmissionsInput) -> str:
        """SBTi progress section."""
        if not data.sbti_progress:
            return ""
        s = data.sbti_progress
        actual = f"{s.actual_reduction_pct:.1f}%" if s.actual_reduction_pct is not None else "N/A"
        track = "Yes" if s.on_track else ("No" if s.on_track is False else "N/A")
        notes = s.notes or "-"
        return (
            "## 8. Science-Based Targets Progress\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Target Type | {s.target_type} |\n"
            f"| Base Year | {s.base_year} |\n"
            f"| Target Year | {s.target_year} |\n"
            f"| Target Reduction | {s.target_reduction_pct:.1f}% |\n"
            f"| Actual Reduction | {actual} |\n"
            f"| Scope Coverage | {s.scope_coverage} |\n"
            f"| On Track | {track} |\n"
            f"| Notes | {notes} |"
        )

    def _md_footer(self, data: GHGEmissionsInput) -> str:
        """Markdown footer."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n"
            f"*Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, company: str, year: int, body: str) -> str:
        """Wrap in full HTML document."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>GHG Emissions Report - {company} ({year})</title>\n"
            "<style>\n"
            "body{font-family:Arial,Helvetica,sans-serif;margin:2rem;color:#222;max-width:1100px;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f5f5f5;}\n"
            ".scope-1{border-left:4px solid #cf222e;}\n"
            ".scope-2{border-left:4px solid #b08800;}\n"
            ".scope-3{border-left:4px solid #1a7f37;}\n"
            ".total-row{font-weight:bold;background:#f0f0f0;}\n"
            ".section{margin-bottom:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: GHGEmissionsInput) -> str:
        """HTML header."""
        base = f" | <strong>Base Year:</strong> {data.base_year}" if data.base_year else ""
        return (
            '<div class="section">\n'
            f"<h1>GHG Emissions Report &mdash; {data.company_name}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {data.reporting_year} | "
            f"<strong>Report Date:</strong> {data.report_date.isoformat()}{base}</p>\n"
            "<hr>\n</div>"
        )

    def _html_totals(self, data: GHGEmissionsInput) -> str:
        """HTML totals table."""
        rows = [
            f'<tr class="scope-1"><td>Scope 1</td><td>{_fmt_tco2e(data.scope1_total)}</td>'
            f"<td>{_pct_of_total(data.scope1_total, data.grand_total)}</td></tr>",
            f'<tr class="scope-2"><td>Scope 2</td><td>{_fmt_tco2e(data.scope2_total)}</td>'
            f"<td>{_pct_of_total(data.scope2_total, data.grand_total)}</td></tr>",
            f'<tr class="scope-3"><td>Scope 3</td><td>{_fmt_tco2e(data.scope3_total)}</td>'
            f"<td>{_pct_of_total(data.scope3_total, data.grand_total)}</td></tr>",
            f'<tr class="total-row"><td>Grand Total</td><td>{_fmt_tco2e(data.grand_total)}</td>'
            f"<td>100.0%</td></tr>",
        ]
        return (
            '<div class="section">\n<h2>1. Total GHG Emissions</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Emissions</th>"
            f"<th>% of Total</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_scope1(self, data: GHGEmissionsInput) -> str:
        """HTML Scope 1 table."""
        s1_total = data.scope1_total
        rows = []
        for s in sorted(data.scope1_breakdown, key=lambda x: x.emissions_tco2e, reverse=True):
            pct = _pct_of_total(s.emissions_tco2e, s1_total)
            rows.append(
                f"<tr><td>{s.source.value}</td><td>{_fmt_tco2e(s.emissions_tco2e)}</td>"
                f"<td>{pct}</td><td>{s.data_quality}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="4">No Scope 1 data</td></tr>')
        return (
            '<div class="section">\n<h2>2. Scope 1 Breakdown</h2>\n'
            "<table><thead><tr><th>Source</th><th>Emissions</th>"
            "<th>% of Scope 1</th><th>Quality</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n"
            f"<p><strong>Scope 1 Total:</strong> {_fmt_tco2e(s1_total)}</p>\n</div>"
        )

    def _html_scope2(self, data: GHGEmissionsInput) -> str:
        """HTML Scope 2 table."""
        rows = []
        for s in data.scope2_breakdown:
            energy = f"{s.energy_consumed_mwh:,.0f} MWh" if s.energy_consumed_mwh is not None else "-"
            rows.append(
                f"<tr><td>{s.method.value}</td><td>{_fmt_tco2e(s.emissions_tco2e)}</td>"
                f"<td>{energy}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="3">No Scope 2 data</td></tr>')
        return (
            '<div class="section">\n<h2>3. Scope 2 Breakdown</h2>\n'
            "<table><thead><tr><th>Method</th><th>Emissions</th>"
            f"<th>Energy</th></tr></thead>\n<tbody>{''.join(rows)}</tbody></table>\n"
            f"<p><strong>Scope 2 Total:</strong> {_fmt_tco2e(data.scope2_total)}</p>\n</div>"
        )

    def _html_scope3(self, data: GHGEmissionsInput) -> str:
        """HTML Scope 3 table."""
        s3_total = data.scope3_total
        rows = []
        for c in sorted(data.scope3_categories, key=lambda x: x.category_number):
            pct = _pct_of_total(c.emissions_tco2e, s3_total)
            rows.append(
                f"<tr><td>{c.category_number}</td><td>{c.category_name}</td>"
                f"<td>{_fmt_tco2e(c.emissions_tco2e)}</td><td>{pct}</td>"
                f"<td>{c.calculation_approach}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="5">No Scope 3 data</td></tr>')
        return (
            '<div class="section">\n<h2>4. Scope 3 Breakdown</h2>\n'
            "<table><thead><tr><th>Cat</th><th>Name</th><th>Emissions</th>"
            "<th>% of S3</th><th>Approach</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n"
            f"<p><strong>Scope 3 Total:</strong> {_fmt_tco2e(s3_total)}</p>\n</div>"
        )

    def _html_intensity(self, data: GHGEmissionsInput) -> str:
        """HTML intensity metrics."""
        if not data.intensity_metrics:
            return ""
        rows = []
        for m in data.intensity_metrics:
            label = m.denominator_label or m.denominator_type.value
            intensity = f"{m.intensity_value:,.4f}" if m.intensity_value is not None else "N/A"
            yoy = _fmt_pct(m.yoy_change_pct)
            rows.append(
                f"<tr><td>{label}</td><td>{m.scope_coverage}</td>"
                f"<td>{intensity}</td><td>{yoy}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>5. Emission Intensity Metrics</h2>\n'
            "<table><thead><tr><th>Denominator</th><th>Scopes</th>"
            "<th>Intensity</th><th>YoY Change</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_methodology(self, data: GHGEmissionsInput) -> str:
        """HTML methodology references."""
        if not data.methodologies:
            return ""
        rows = []
        for m in data.methodologies:
            yr = str(m.version_year) if m.version_year else "-"
            rows.append(
                f"<tr><td>{m.reference_id}</td><td>{m.title}</td>"
                f"<td>{m.source}</td><td>{yr}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>7. Methodology & Emission Factors</h2>\n'
            "<table><thead><tr><th>Ref</th><th>Title</th>"
            "<th>Source</th><th>Year</th></tr></thead>\n"
            f"<tbody>{''.join(rows)}</tbody></table>\n</div>"
        )

    def _html_sbti(self, data: GHGEmissionsInput) -> str:
        """HTML SBTi progress."""
        if not data.sbti_progress:
            return ""
        s = data.sbti_progress
        actual = f"{s.actual_reduction_pct:.1f}%" if s.actual_reduction_pct is not None else "N/A"
        track = "Yes" if s.on_track else ("No" if s.on_track is False else "N/A")
        return (
            '<div class="section">\n<h2>8. Science-Based Targets Progress</h2>\n'
            f"<p><strong>Target Type:</strong> {s.target_type} | "
            f"<strong>Base Year:</strong> {s.base_year} | "
            f"<strong>Target Year:</strong> {s.target_year}</p>\n"
            f"<p><strong>Target Reduction:</strong> {s.target_reduction_pct:.1f}% | "
            f"<strong>Actual Reduction:</strong> {actual} | "
            f"<strong>On Track:</strong> {track}</p>\n</div>"
        )

    def _html_footer(self, data: GHGEmissionsInput) -> str:
        """HTML footer with provenance."""
        provenance = self._compute_provenance(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang CSRD Starter Pack v1.0.0 | {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
