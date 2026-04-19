# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack: Country Risk Matrix Template
=========================================================

Generates a country x commodity risk matrix visualization with
color-coded risk levels, detailed country profiles, commodity
hotspot analysis, supplier concentration data, and trend tracking.

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

PACK_ID = "PACK-006-eudr-starter"
TEMPLATE_NAME = "country_risk_matrix"
TEMPLATE_VERSION = "1.0.0"


# =============================================================================
# ENUMS
# =============================================================================

class CommodityType(str, Enum):
    """EUDR-regulated commodities."""
    CATTLE = "CATTLE"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    OIL_PALM = "OIL_PALM"
    RUBBER = "RUBBER"
    SOYA = "SOYA"
    WOOD = "WOOD"


class RiskLevel(str, Enum):
    """Risk level for matrix cells."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    NOT_APPLICABLE = "N/A"


class BenchmarkClassification(str, Enum):
    """Article 29 benchmark."""
    LOW = "LOW"
    STANDARD = "STANDARD"
    HIGH = "HIGH"


class TrendDirection(str, Enum):
    """Trend direction for risk changes."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    WORSENING = "WORSENING"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class MatrixCell(BaseModel):
    """Single cell in the country x commodity risk matrix."""
    country_iso: str = Field(..., description="ISO 3166-1 alpha-2")
    commodity: CommodityType = Field(..., description="Commodity type")
    risk_level: RiskLevel = Field(..., description="Risk level")
    risk_score: float = Field(0.0, ge=0.0, le=100.0, description="Risk score")
    supplier_count: int = Field(0, ge=0, description="Suppliers for this pair")
    volume_kg: Optional[float] = Field(None, ge=0, description="Volume in kg")


class CountryProfile(BaseModel):
    """Detailed country profile."""
    country_iso: str = Field(..., description="ISO 3166-1 alpha-2")
    country_name: str = Field(..., description="Full country name")
    benchmark: BenchmarkClassification = Field(..., description="Article 29 benchmark")
    deforestation_rate_pct: Optional[float] = Field(
        None, ge=0.0, description="Annual deforestation rate %"
    )
    forest_area_km2: Optional[float] = Field(
        None, ge=0, description="Forest area in km2"
    )
    governance_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Governance effectiveness"
    )
    corruption_index: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Corruption perception index"
    )
    rule_of_law_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Rule of law score"
    )
    total_suppliers: int = Field(0, ge=0, description="Total suppliers from country")
    total_volume_kg: Optional[float] = Field(None, ge=0, description="Total volume")
    overall_risk_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Overall country risk"
    )
    commodities_sourced: List[CommodityType] = Field(
        default_factory=list, description="Commodities sourced from this country"
    )


class CommodityHotspot(BaseModel):
    """Commodity hotspot analysis entry."""
    commodity: CommodityType = Field(..., description="Commodity type")
    high_risk_countries: List[str] = Field(
        default_factory=list, description="High-risk country ISOs"
    )
    total_supplier_count: int = Field(0, ge=0, description="Total suppliers")
    high_risk_supplier_count: int = Field(
        0, ge=0, description="Suppliers in high-risk countries"
    )
    high_risk_volume_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="% volume from high-risk countries"
    )
    deforestation_correlation: float = Field(
        0.0, ge=0.0, le=100.0, description="Deforestation correlation"
    )


class SupplierConcentration(BaseModel):
    """Supplier concentration for a country-commodity pair."""
    country_iso: str = Field(..., description="ISO 3166-1 alpha-2")
    country_name: str = Field(..., description="Country name")
    commodity: CommodityType = Field(..., description="Commodity type")
    supplier_count: int = Field(0, ge=0, description="Number of suppliers")
    volume_kg: Optional[float] = Field(None, ge=0, description="Volume in kg")
    concentration_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="% of total for this commodity"
    )


class TrendEntry(BaseModel):
    """Risk trend data point."""
    country_iso: str = Field(..., description="ISO 3166-1 alpha-2")
    commodity: CommodityType = Field(..., description="Commodity type")
    period: str = Field(..., description="Period label (e.g. Q1 2025)")
    risk_score: float = Field(0.0, ge=0.0, le=100.0, description="Risk score")
    direction: TrendDirection = Field(
        TrendDirection.STABLE, description="Trend direction"
    )


class CountryRiskMatrixInput(BaseModel):
    """Complete input for the Country Risk Matrix report."""
    company_name: str = Field(..., description="Reporting entity")
    report_date: date = Field(
        default_factory=date.today, description="Report date"
    )
    matrix_cells: List[MatrixCell] = Field(
        default_factory=list, description="Matrix data cells"
    )
    country_profiles: List[CountryProfile] = Field(
        default_factory=list, description="Country profiles"
    )
    commodity_hotspots: List[CommodityHotspot] = Field(
        default_factory=list, description="Commodity hotspot analysis"
    )
    supplier_concentrations: List[SupplierConcentration] = Field(
        default_factory=list, description="Supplier concentration data"
    )
    trends: List[TrendEntry] = Field(
        default_factory=list, description="Trend data"
    )


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _risk_badge(level: RiskLevel) -> str:
    """Text badge for risk level."""
    return f"[{level.value}]"


def _risk_css(level: RiskLevel) -> str:
    """Inline CSS for risk level."""
    mapping = {
        RiskLevel.LOW: "background:#d1fae5;color:#1a7f37;",
        RiskLevel.MEDIUM: "background:#fef3c7;color:#b08800;",
        RiskLevel.HIGH: "background:#fed7aa;color:#e36209;",
        RiskLevel.CRITICAL: "background:#fecaca;color:#cf222e;font-weight:bold;",
        RiskLevel.NOT_APPLICABLE: "background:#f3f4f6;color:#888;",
    }
    return mapping.get(level, "")


def _benchmark_badge(b: BenchmarkClassification) -> str:
    """Text badge for benchmark."""
    return f"[{b.value}]"


def _fmt_commodity(commodity: CommodityType) -> str:
    """Human-readable commodity name."""
    mapping = {
        CommodityType.CATTLE: "Cattle",
        CommodityType.COCOA: "Cocoa",
        CommodityType.COFFEE: "Coffee",
        CommodityType.OIL_PALM: "Oil Palm",
        CommodityType.RUBBER: "Rubber",
        CommodityType.SOYA: "Soya",
        CommodityType.WOOD: "Wood",
    }
    return mapping.get(commodity, commodity.value)


def _fmt_volume(kg: Optional[float]) -> str:
    """Format volume."""
    if kg is None:
        return "N/A"
    if kg >= 1_000_000:
        return f"{kg / 1_000:,.0f} t"
    if kg >= 1_000:
        return f"{kg / 1_000:,.1f} t"
    return f"{kg:,.0f} kg"


def _fmt_area(km2: Optional[float]) -> str:
    """Format area in km2."""
    if km2 is None:
        return "N/A"
    if km2 >= 1_000_000:
        return f"{km2 / 1_000_000:,.2f}M km2"
    return f"{km2:,.0f} km2"


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def _trend_badge(direction: TrendDirection) -> str:
    """Text indicator for trend."""
    mapping = {
        TrendDirection.IMPROVING: "[v IMPROVING]",
        TrendDirection.STABLE: "[= STABLE]",
        TrendDirection.WORSENING: "[^ WORSENING]",
    }
    return mapping.get(direction, "[?]")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class CountryRiskMatrix:
    """Generate country x commodity risk matrix report.

    Sections:
        1. Matrix Overview - Countries x commodities with risk colors
        2. High-Risk Countries - Detailed high-risk country profiles
        3. Country Profiles - Per-country summary
        4. Commodity Hotspots - Per-commodity high-risk analysis
        5. Supplier Concentration - Suppliers per country-commodity
        6. Trend Analysis - Risk trend over time

    Example:
        >>> matrix = CountryRiskMatrix()
        >>> data = CountryRiskMatrixInput(...)
        >>> md = matrix.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize the Country Risk Matrix template."""
        self._render_timestamp: Optional[datetime] = None

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def render_markdown(self, data: CountryRiskMatrixInput) -> str:
        """Render as Markdown.

        Args:
            data: Validated matrix input data.

        Returns:
            Complete Markdown string.
        """
        self._render_timestamp = datetime.utcnow()
        sections = [
            self._md_header(data),
            self._md_matrix_overview(data),
            self._md_high_risk_countries(data),
            self._md_country_profiles(data),
            self._md_commodity_hotspots(data),
            self._md_supplier_concentration(data),
            self._md_trend_analysis(data),
            self._md_provenance(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(self, data: CountryRiskMatrixInput) -> str:
        """Render as HTML.

        Args:
            data: Validated matrix input data.

        Returns:
            Complete HTML document string.
        """
        self._render_timestamp = datetime.utcnow()
        body_parts = [
            self._html_header(data),
            self._html_matrix_overview(data),
            self._html_high_risk_countries(data),
            self._html_country_profiles(data),
            self._html_commodity_hotspots(data),
            self._html_supplier_concentration(data),
            self._html_trend_analysis(data),
            self._html_provenance(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(self, data: CountryRiskMatrixInput) -> Dict[str, Any]:
        """Render as JSON-serializable dictionary.

        Args:
            data: Validated matrix input data.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        self._render_timestamp = datetime.utcnow()
        provenance_hash = self._compute_provenance_hash(data)

        return {
            "metadata": {
                "pack_id": PACK_ID,
                "template_name": TEMPLATE_NAME,
                "version": TEMPLATE_VERSION,
                "generated_at": self._render_timestamp.isoformat(),
                "provenance_hash": provenance_hash,
            },
            "company_name": data.company_name,
            "report_date": data.report_date.isoformat(),
            "matrix_cells": [c.model_dump(mode="json") for c in data.matrix_cells],
            "country_profiles": [
                p.model_dump(mode="json") for p in data.country_profiles
            ],
            "commodity_hotspots": [
                h.model_dump(mode="json") for h in data.commodity_hotspots
            ],
            "supplier_concentrations": [
                s.model_dump(mode="json") for s in data.supplier_concentrations
            ],
            "trends": [t.model_dump(mode="json") for t in data.trends],
        }

    # --------------------------------------------------------------------- #
    # PROVENANCE
    # --------------------------------------------------------------------- #

    def _compute_provenance_hash(self, data: CountryRiskMatrixInput) -> str:
        """Compute SHA-256 provenance hash."""
        raw = data.model_dump_json(exclude_none=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------------------------- #
    # INTERNAL HELPERS
    # --------------------------------------------------------------------- #

    def _build_matrix_lookup(
        self, cells: List[MatrixCell]
    ) -> Dict[tuple, MatrixCell]:
        """Build lookup dict from (country, commodity) to MatrixCell."""
        lookup: Dict[tuple, MatrixCell] = {}
        for cell in cells:
            lookup[(cell.country_iso, cell.commodity)] = cell
        return lookup

    def _get_countries(self, data: CountryRiskMatrixInput) -> List[str]:
        """Get sorted unique country ISOs from matrix cells."""
        return sorted(set(c.country_iso for c in data.matrix_cells))

    def _get_commodities(self, data: CountryRiskMatrixInput) -> List[CommodityType]:
        """Get sorted unique commodities from matrix cells."""
        seen = set()
        result = []
        for c in data.matrix_cells:
            if c.commodity not in seen:
                seen.add(c.commodity)
                result.append(c.commodity)
        return sorted(result, key=lambda x: x.value)

    # --------------------------------------------------------------------- #
    # MARKDOWN SECTIONS
    # --------------------------------------------------------------------- #

    def _md_header(self, data: CountryRiskMatrixInput) -> str:
        """Report header."""
        return (
            f"# Country Risk Matrix - {data.company_name}\n"
            f"**Report Date:** {data.report_date.isoformat()}\n\n---"
        )

    def _md_matrix_overview(self, data: CountryRiskMatrixInput) -> str:
        """Section 1: Matrix Overview."""
        countries = self._get_countries(data)
        commodities = self._get_commodities(data)
        lookup = self._build_matrix_lookup(data.matrix_cells)

        if not countries or not commodities:
            return "## 1. Matrix Overview\n\nNo matrix data available."

        # Build header row
        header = "| Country | " + " | ".join(
            _fmt_commodity(c) for c in commodities
        ) + " |"
        separator = "|---------|" + "|".join(
            "-" * (len(_fmt_commodity(c)) + 2) for c in commodities
        ) + "|"

        lines = [
            "## 1. Matrix Overview\n",
            "> Risk levels: [LOW] [MEDIUM] [HIGH] [CRITICAL]\n",
            header,
            separator,
        ]

        for country in countries:
            cells = []
            for commodity in commodities:
                cell = lookup.get((country, commodity))
                if cell:
                    cells.append(
                        f"{_risk_badge(cell.risk_level)} {cell.risk_score:.0f}"
                    )
                else:
                    cells.append("[N/A]")
            lines.append(f"| {country} | " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _md_high_risk_countries(self, data: CountryRiskMatrixInput) -> str:
        """Section 2: High-Risk Countries."""
        high_risk = [
            p for p in data.country_profiles
            if p.benchmark == BenchmarkClassification.HIGH
            or p.overall_risk_score >= 60
        ]
        if not high_risk:
            return "## 2. High-Risk Countries\n\nNo high-risk countries identified."

        lines = [
            "## 2. High-Risk Countries\n",
            "| Country | Benchmark | Risk Score | Deforestation | Governance | Suppliers |",
            "|---------|-----------|------------|---------------|------------|-----------|",
        ]
        for p in sorted(high_risk, key=lambda x: x.overall_risk_score, reverse=True):
            deforest = _fmt_pct(p.deforestation_rate_pct)
            governance = (
                f"{p.governance_score:.0f}/100" if p.governance_score is not None else "N/A"
            )
            lines.append(
                f"| {p.country_name} ({p.country_iso}) "
                f"| {_benchmark_badge(p.benchmark)} "
                f"| {p.overall_risk_score:.1f} | {deforest} "
                f"| {governance} | {p.total_suppliers} |"
            )
        return "\n".join(lines)

    def _md_country_profiles(self, data: CountryRiskMatrixInput) -> str:
        """Section 3: Country Profiles."""
        lines = [
            "## 3. Country Profiles\n",
            "| Country | Benchmark | Risk | Deforestation | Forest Area | "
            "Governance | Corruption | Suppliers | Volume |",
            "|---------|-----------|------|---------------|-------------|"
            "------------|------------|-----------|--------|",
        ]
        for p in sorted(
            data.country_profiles, key=lambda x: x.overall_risk_score, reverse=True
        ):
            deforest = _fmt_pct(p.deforestation_rate_pct)
            forest = _fmt_area(p.forest_area_km2)
            governance = (
                f"{p.governance_score:.0f}" if p.governance_score is not None else "N/A"
            )
            corruption = (
                f"{p.corruption_index:.0f}" if p.corruption_index is not None else "N/A"
            )
            volume = _fmt_volume(p.total_volume_kg)
            lines.append(
                f"| {p.country_name} ({p.country_iso}) "
                f"| {_benchmark_badge(p.benchmark)} "
                f"| {p.overall_risk_score:.1f} | {deforest} | {forest} "
                f"| {governance} | {corruption} | {p.total_suppliers} | {volume} |"
            )
        if not data.country_profiles:
            lines.append(
                "| - | No country profiles | - | - | - | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_commodity_hotspots(self, data: CountryRiskMatrixInput) -> str:
        """Section 4: Commodity Hotspots."""
        lines = [
            "## 4. Commodity Hotspots\n",
            "| Commodity | High-Risk Countries | Suppliers | High-Risk Suppliers "
            "| High-Risk Volume | Deforestation Corr. |",
            "|-----------|--------------------|-----------|--------------------|"
            "------------------|---------------------|",
        ]
        for h in data.commodity_hotspots:
            countries = ", ".join(h.high_risk_countries) if h.high_risk_countries else "None"
            lines.append(
                f"| {_fmt_commodity(h.commodity)} | {countries} "
                f"| {h.total_supplier_count} | {h.high_risk_supplier_count} "
                f"| {h.high_risk_volume_pct:.1f}% "
                f"| {h.deforestation_correlation:.1f} |"
            )
        if not data.commodity_hotspots:
            lines.append("| - | No hotspot data | - | - | - | - |")
        return "\n".join(lines)

    def _md_supplier_concentration(self, data: CountryRiskMatrixInput) -> str:
        """Section 5: Supplier Concentration."""
        lines = [
            "## 5. Supplier Concentration\n",
            "| Country | Commodity | Suppliers | Volume | Concentration |",
            "|---------|-----------|-----------|--------|---------------|",
        ]
        for s in sorted(
            data.supplier_concentrations,
            key=lambda x: x.concentration_pct,
            reverse=True,
        ):
            volume = _fmt_volume(s.volume_kg)
            lines.append(
                f"| {s.country_name} ({s.country_iso}) "
                f"| {_fmt_commodity(s.commodity)} "
                f"| {s.supplier_count} | {volume} "
                f"| {s.concentration_pct:.1f}% |"
            )
        if not data.supplier_concentrations:
            lines.append("| - | No concentration data | - | - | - |")
        return "\n".join(lines)

    def _md_trend_analysis(self, data: CountryRiskMatrixInput) -> str:
        """Section 6: Trend Analysis."""
        if not data.trends:
            return "## 6. Trend Analysis\n\nNo trend data available."

        lines = [
            "## 6. Trend Analysis\n",
            "| Country | Commodity | Period | Risk Score | Direction |",
            "|---------|-----------|--------|------------|-----------|",
        ]
        for t in sorted(data.trends, key=lambda x: (x.country_iso, x.commodity, x.period)):
            lines.append(
                f"| {t.country_iso} | {_fmt_commodity(t.commodity)} "
                f"| {t.period} | {t.risk_score:.1f} "
                f"| {_trend_badge(t.direction)} |"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: CountryRiskMatrixInput) -> str:
        """Provenance footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang EUDR Starter Pack v{TEMPLATE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # --------------------------------------------------------------------- #
    # HTML SECTIONS
    # --------------------------------------------------------------------- #

    def _wrap_html(self, data: CountryRiskMatrixInput, body: str) -> str:
        """Wrap body in HTML document."""
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Country Risk Matrix - {data.company_name}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;"
            "max-width:1200px;color:#222;line-height:1.5;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:center;}\n"
            "th{background:#f0f4f8;font-weight:600;text-align:left;}\n"
            "h1{color:#1a365d;border-bottom:3px solid #2b6cb0;padding-bottom:0.5rem;}\n"
            "h2{color:#2b6cb0;margin-top:2rem;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".cell-low{background:#d1fae5;color:#1a7f37;font-weight:bold;}\n"
            ".cell-medium{background:#fef3c7;color:#b08800;font-weight:bold;}\n"
            ".cell-high{background:#fed7aa;color:#e36209;font-weight:bold;}\n"
            ".cell-critical{background:#fecaca;color:#cf222e;font-weight:bold;}\n"
            ".cell-na{background:#f3f4f6;color:#888;}\n"
            ".provenance{font-size:0.85rem;color:#666;}\n"
            "code{background:#f5f5f5;padding:0.2rem 0.4rem;border-radius:3px;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: CountryRiskMatrixInput) -> str:
        """HTML header."""
        return (
            '<div class="section">\n'
            f"<h1>Country Risk Matrix &mdash; {data.company_name}</h1>\n"
            f"<p><strong>Report Date:</strong> "
            f"{data.report_date.isoformat()}</p>\n<hr>\n</div>"
        )

    def _html_matrix_overview(self, data: CountryRiskMatrixInput) -> str:
        """HTML Section 1: Matrix Overview."""
        countries = self._get_countries(data)
        commodities = self._get_commodities(data)
        lookup = self._build_matrix_lookup(data.matrix_cells)

        if not countries or not commodities:
            return (
                '<div class="section"><h2>1. Matrix Overview</h2>'
                "<p>No matrix data available.</p></div>"
            )

        header_cells = "".join(
            f"<th>{_fmt_commodity(c)}</th>" for c in commodities
        )
        rows = ""
        for country in countries:
            cells = ""
            for commodity in commodities:
                cell = lookup.get((country, commodity))
                if cell:
                    css_class = f"cell-{cell.risk_level.value.lower().replace('/', '')}"
                    cells += (
                        f'<td class="{css_class}">'
                        f"{cell.risk_level.value}<br>{cell.risk_score:.0f}</td>"
                    )
                else:
                    cells += '<td class="cell-na">N/A</td>'
            rows += f"<tr><th style='text-align:left;'>{country}</th>{cells}</tr>"

        return (
            '<div class="section">\n<h2>1. Matrix Overview</h2>\n'
            f"<table><thead><tr><th>Country</th>{header_cells}</tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_high_risk_countries(self, data: CountryRiskMatrixInput) -> str:
        """HTML Section 2: High-Risk Countries."""
        high_risk = [
            p for p in data.country_profiles
            if p.benchmark == BenchmarkClassification.HIGH
            or p.overall_risk_score >= 60
        ]
        if not high_risk:
            return (
                '<div class="section"><h2>2. High-Risk Countries</h2>'
                "<p>No high-risk countries identified.</p></div>"
            )

        rows = ""
        for p in sorted(high_risk, key=lambda x: x.overall_risk_score, reverse=True):
            deforest = _fmt_pct(p.deforestation_rate_pct)
            governance = (
                f"{p.governance_score:.0f}/100"
                if p.governance_score is not None
                else "N/A"
            )
            rows += (
                f"<tr><td>{p.country_name} ({p.country_iso})</td>"
                f"<td>{p.benchmark.value}</td>"
                f'<td class="cell-high">{p.overall_risk_score:.1f}</td>'
                f"<td>{deforest}</td><td>{governance}</td>"
                f"<td>{p.total_suppliers}</td></tr>"
            )
        return (
            '<div class="section">\n<h2>2. High-Risk Countries</h2>\n'
            "<table><thead><tr><th>Country</th><th>Benchmark</th>"
            "<th>Risk Score</th><th>Deforestation</th>"
            "<th>Governance</th><th>Suppliers</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_country_profiles(self, data: CountryRiskMatrixInput) -> str:
        """HTML Section 3: Country Profiles."""
        rows = ""
        for p in sorted(
            data.country_profiles, key=lambda x: x.overall_risk_score, reverse=True
        ):
            deforest = _fmt_pct(p.deforestation_rate_pct)
            forest = _fmt_area(p.forest_area_km2)
            governance = (
                f"{p.governance_score:.0f}"
                if p.governance_score is not None
                else "N/A"
            )
            corruption = (
                f"{p.corruption_index:.0f}"
                if p.corruption_index is not None
                else "N/A"
            )
            volume = _fmt_volume(p.total_volume_kg)
            rows += (
                f"<tr><td>{p.country_name} ({p.country_iso})</td>"
                f"<td>{p.benchmark.value}</td>"
                f"<td>{p.overall_risk_score:.1f}</td>"
                f"<td>{deforest}</td><td>{forest}</td>"
                f"<td>{governance}</td><td>{corruption}</td>"
                f"<td>{p.total_suppliers}</td><td>{volume}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="9">No country profiles</td></tr>'
        return (
            '<div class="section">\n<h2>3. Country Profiles</h2>\n'
            "<table><thead><tr><th>Country</th><th>Benchmark</th>"
            "<th>Risk</th><th>Deforestation</th><th>Forest Area</th>"
            "<th>Governance</th><th>Corruption</th><th>Suppliers</th>"
            f"<th>Volume</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_commodity_hotspots(self, data: CountryRiskMatrixInput) -> str:
        """HTML Section 4: Commodity Hotspots."""
        rows = ""
        for h in data.commodity_hotspots:
            countries = ", ".join(h.high_risk_countries) if h.high_risk_countries else "None"
            rows += (
                f"<tr><td>{_fmt_commodity(h.commodity)}</td>"
                f"<td>{countries}</td>"
                f"<td>{h.total_supplier_count}</td>"
                f"<td>{h.high_risk_supplier_count}</td>"
                f"<td>{h.high_risk_volume_pct:.1f}%</td>"
                f"<td>{h.deforestation_correlation:.1f}</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="6">No hotspot data</td></tr>'
        return (
            '<div class="section">\n<h2>4. Commodity Hotspots</h2>\n'
            "<table><thead><tr><th>Commodity</th><th>High-Risk Countries</th>"
            "<th>Total Suppliers</th><th>High-Risk Suppliers</th>"
            "<th>High-Risk Volume</th><th>Deforestation Corr.</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_supplier_concentration(self, data: CountryRiskMatrixInput) -> str:
        """HTML Section 5: Supplier Concentration."""
        rows = ""
        for s in sorted(
            data.supplier_concentrations,
            key=lambda x: x.concentration_pct,
            reverse=True,
        ):
            volume = _fmt_volume(s.volume_kg)
            rows += (
                f"<tr><td>{s.country_name} ({s.country_iso})</td>"
                f"<td>{_fmt_commodity(s.commodity)}</td>"
                f"<td>{s.supplier_count}</td>"
                f"<td>{volume}</td>"
                f"<td>{s.concentration_pct:.1f}%</td></tr>"
            )
        if not rows:
            rows = '<tr><td colspan="5">No concentration data</td></tr>'
        return (
            '<div class="section">\n<h2>5. Supplier Concentration</h2>\n'
            "<table><thead><tr><th>Country</th><th>Commodity</th>"
            "<th>Suppliers</th><th>Volume</th>"
            f"<th>Concentration</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_trend_analysis(self, data: CountryRiskMatrixInput) -> str:
        """HTML Section 6: Trend Analysis."""
        if not data.trends:
            return (
                '<div class="section"><h2>6. Trend Analysis</h2>'
                "<p>No trend data available.</p></div>"
            )

        rows = ""
        for t in sorted(
            data.trends, key=lambda x: (x.country_iso, x.commodity, x.period)
        ):
            trend_css = {
                TrendDirection.IMPROVING: "color:#1a7f37;",
                TrendDirection.STABLE: "color:#b08800;",
                TrendDirection.WORSENING: "color:#cf222e;",
            }.get(t.direction, "")
            rows += (
                f"<tr><td>{t.country_iso}</td>"
                f"<td>{_fmt_commodity(t.commodity)}</td>"
                f"<td>{t.period}</td>"
                f"<td>{t.risk_score:.1f}</td>"
                f'<td style="{trend_css}">{t.direction.value}</td></tr>'
            )
        return (
            '<div class="section">\n<h2>6. Trend Analysis</h2>\n'
            "<table><thead><tr><th>Country</th><th>Commodity</th>"
            "<th>Period</th><th>Risk Score</th>"
            f"<th>Direction</th></tr></thead>\n<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_provenance(self, data: CountryRiskMatrixInput) -> str:
        """HTML provenance footer."""
        provenance = self._compute_provenance_hash(data)
        ts = self._render_timestamp.isoformat() if self._render_timestamp else "N/A"
        return (
            '<div class="section provenance">\n<hr>\n'
            f"<p>Generated by GreenLang EUDR Starter Pack v{TEMPLATE_VERSION} "
            f"| {ts}</p>\n"
            f"<p>Provenance Hash: <code>{provenance}</code></p>\n</div>"
        )
