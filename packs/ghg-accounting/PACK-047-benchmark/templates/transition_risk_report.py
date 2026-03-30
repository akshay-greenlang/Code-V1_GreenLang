# -*- coding: utf-8 -*-
"""
TransitionRiskReport - Transition Risk Report for PACK-047.

Generates a transition risk report with composite transition risk score
(0-100) and dimension breakdown, ITR gauge chart data, stranding risk
timeline, regulatory exposure table (EU ETS, CBAM thresholds), competitive
position radar, and carbon price sensitivity table.

Regulatory References:
    - TCFD: Strategy and Risk Management pillars
    - EU ETS Directive 2003/87/EC (revised 2023)
    - EU CBAM Regulation 2023/956: Carbon border adjustment
    - SBTi: Implied Temperature Rise methodology
    - Network for Greening the Financial System (NGFS) scenarios

Sections:
    1. Composite Transition Risk Score (with dimension breakdown)
    2. ITR Gauge Data
    3. Stranding Risk Timeline
    4. Regulatory Exposure Table
    5. Competitive Position Radar
    6. Carbon Price Sensitivity
    7. Provenance Footer

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 47.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    """Supported output formats."""
    MD = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"

class TrafficLight(str, Enum):
    """Traffic light status indicators."""
    GREEN = "green"
    AMBER = "amber"
    RED = "red"

class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class RiskDimension(BaseModel):
    """Single dimension of composite transition risk score."""
    dimension_name: str = Field(..., description="Dimension name (e.g., Policy, Technology)")
    score: float = Field(0.0, ge=0, le=100, description="Score (0-100)")
    weight: float = Field(0.0, ge=0, le=1, description="Weight in composite (0-1)")
    weighted_score: float = Field(0.0, description="Weighted contribution")
    description: str = Field("", description="Dimension description")
    risk_level: RiskLevel = Field(RiskLevel.MEDIUM, description="Risk level")

class CompositeRiskScore(BaseModel):
    """Composite transition risk score."""
    total_score: float = Field(50.0, ge=0, le=100, description="Composite score (0-100)")
    risk_category: str = Field("Medium", description="Overall risk category")
    traffic_light: TrafficLight = Field(TrafficLight.AMBER, description="Traffic light")
    dimensions: List[RiskDimension] = Field(
        default_factory=list, description="Dimension breakdown"
    )
    methodology: str = Field("", description="Scoring methodology")

class ITRGaugeData(BaseModel):
    """Implied Temperature Rise gauge chart data."""
    itr_value: float = Field(2.0, description="ITR in degrees C")
    pathway_1_5c: float = Field(1.5, description="1.5C threshold")
    pathway_2c: float = Field(2.0, description="2C threshold")
    pathway_3c: float = Field(3.0, description="3C threshold")
    methodology: str = Field("", description="ITR methodology used")
    confidence_interval_low: Optional[float] = Field(None, description="Lower CI bound")
    confidence_interval_high: Optional[float] = Field(None, description="Upper CI bound")
    peer_median_itr: Optional[float] = Field(None, description="Peer median ITR")

class StrandingRiskEntry(BaseModel):
    """Stranding risk timeline entry."""
    year: int = Field(..., description="Year")
    asset_category: str = Field("", description="Asset category")
    at_risk_value: float = Field(0.0, description="Value at risk (EUR millions)")
    probability_pct: float = Field(0.0, ge=0, le=100, description="Probability of stranding (%)")
    scenario: str = Field("", description="Scenario (e.g., Net Zero 2050)")
    notes: str = Field("", description="Additional notes")

class RegulatoryExposureEntry(BaseModel):
    """Regulatory exposure table entry."""
    regulation: str = Field(..., description="Regulation name (e.g., EU ETS, CBAM)")
    scope: str = Field("", description="Emissions scope covered")
    current_exposure_eur: float = Field(0.0, description="Current annual cost (EUR)")
    projected_exposure_eur: float = Field(0.0, description="Projected cost (EUR)")
    projection_year: int = Field(0, description="Projection year")
    threshold_exceeded: bool = Field(False, description="Whether threshold is exceeded")
    free_allowance_pct: Optional[float] = Field(None, description="Free allowance remaining (%)")
    notes: str = Field("", description="Additional notes")

class CompetitiveRadarAxis(BaseModel):
    """Single axis on the competitive position radar chart."""
    axis_name: str = Field(..., description="Axis label")
    org_value: float = Field(0.0, ge=0, le=100, description="Organisation score (0-100)")
    peer_average: float = Field(0.0, ge=0, le=100, description="Peer average score")
    best_in_class: float = Field(0.0, ge=0, le=100, description="Best-in-class score")

class CarbonPriceSensitivity(BaseModel):
    """Carbon price sensitivity scenario."""
    carbon_price_eur: float = Field(0.0, description="Carbon price (EUR/tCO2e)")
    annual_cost_eur: float = Field(0.0, description="Annual cost at this price (EUR)")
    ebitda_impact_pct: float = Field(0.0, description="EBITDA impact (%)")
    margin_impact_pct: float = Field(0.0, description="Margin impact (%)")
    scenario_label: str = Field("", description="Scenario label")

class TransitionRiskInput(BaseModel):
    """Complete input model for TransitionRiskReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    composite_risk: Optional[CompositeRiskScore] = Field(
        None, description="Composite risk score"
    )
    itr_gauge: Optional[ITRGaugeData] = Field(None, description="ITR gauge data")
    stranding_risk: List[StrandingRiskEntry] = Field(
        default_factory=list, description="Stranding risk timeline"
    )
    regulatory_exposure: List[RegulatoryExposureEntry] = Field(
        default_factory=list, description="Regulatory exposure table"
    )
    competitive_radar: List[CompetitiveRadarAxis] = Field(
        default_factory=list, description="Competitive position radar axes"
    )
    carbon_price_sensitivity: List[CarbonPriceSensitivity] = Field(
        default_factory=list, description="Carbon price sensitivity scenarios"
    )

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _tl_label(status: TrafficLight) -> str:
    """Return uppercase label for traffic light."""
    return status.value.upper()

def _tl_color(status: TrafficLight) -> str:
    """Return hex colour for traffic light."""
    mapping = {
        TrafficLight.GREEN: "#2a9d8f",
        TrafficLight.AMBER: "#e9c46a",
        TrafficLight.RED: "#e76f51",
    }
    return mapping.get(status, "#e9c46a")

def _risk_color(level: RiskLevel) -> str:
    """Return hex colour for risk level."""
    mapping = {
        RiskLevel.LOW: "#2a9d8f",
        RiskLevel.MEDIUM: "#e9c46a",
        RiskLevel.HIGH: "#e76f51",
        RiskLevel.VERY_HIGH: "#c1121f",
    }
    return mapping.get(level, "#e9c46a")

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class TransitionRiskReport:
    """
    Transition risk report template for GHG emissions benchmarking.

    Renders composite risk scores, ITR gauge data, stranding risk timelines,
    regulatory exposure tables, competitive radar charts, and carbon price
    sensitivity analysis. All outputs include SHA-256 provenance hashing.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = TransitionRiskReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TransitionRiskReport."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render(self, data: Dict[str, Any], fmt: str = "markdown") -> Any:
        """Render in specified format."""
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        raise ValueError(f"Unsupported format: {fmt}")

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render transition risk as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render transition risk as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render transition risk as JSON dict."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def to_markdown(self, data: Dict[str, Any]) -> str:
        """Alias for render_markdown."""
        return self.render_markdown(data)

    def to_html(self, data: Dict[str, Any]) -> str:
        """Alias for render_html."""
        return self.render_html(data)

    def to_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for render_json."""
        return self.render_json(data)

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_composite_risk(data),
            self._md_itr_gauge(data),
            self._md_stranding_risk(data),
            self._md_regulatory_exposure(data),
            self._md_competitive_radar(data),
            self._md_carbon_sensitivity(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Transition Risk Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_composite_risk(self, data: Dict[str, Any]) -> str:
        """Render Markdown composite risk score."""
        risk = data.get("composite_risk")
        if not risk:
            return ""
        total = risk.get("total_score", 0)
        category = risk.get("risk_category", "")
        tl = TrafficLight(risk.get("traffic_light", "amber"))
        lines = [
            "## 1. Composite Transition Risk Score",
            "",
            f"**Overall Score:** {total:.0f} / 100 ({category}) - **{_tl_label(tl)}**",
            "",
        ]
        dims = risk.get("dimensions", [])
        if dims:
            lines.append("| Dimension | Score | Weight | Weighted | Risk Level |")
            lines.append("|-----------|-------|--------|----------|------------|")
            for d in dims:
                level = RiskLevel(d.get("risk_level", "medium"))
                lines.append(
                    f"| {d.get('dimension_name', '')} | {d.get('score', 0):.0f} | "
                    f"{d.get('weight', 0):.0%} | {d.get('weighted_score', 0):.1f} | "
                    f"{level.value.replace('_', ' ').title()} |"
                )
        methodology = risk.get("methodology", "")
        if methodology:
            lines.append("")
            lines.append(f"**Methodology:** {methodology}")
        return "\n".join(lines)

    def _md_itr_gauge(self, data: Dict[str, Any]) -> str:
        """Render Markdown ITR gauge summary."""
        itr = data.get("itr_gauge")
        if not itr:
            return ""
        lines = [
            "## 2. Implied Temperature Rise (ITR)",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| ITR | {itr.get('itr_value', 0):.1f} degC |",
            f"| 1.5C Threshold | {itr.get('pathway_1_5c', 1.5):.1f} degC |",
            f"| 2C Threshold | {itr.get('pathway_2c', 2.0):.1f} degC |",
        ]
        ci_low = itr.get("confidence_interval_low")
        ci_high = itr.get("confidence_interval_high")
        if ci_low is not None and ci_high is not None:
            lines.append(f"| Confidence Interval | {ci_low:.1f} - {ci_high:.1f} degC |")
        peer_itr = itr.get("peer_median_itr")
        if peer_itr is not None:
            lines.append(f"| Peer Median ITR | {peer_itr:.1f} degC |")
        methodology = itr.get("methodology", "")
        if methodology:
            lines.append("")
            lines.append(f"**Methodology:** {methodology}")
        return "\n".join(lines)

    def _md_stranding_risk(self, data: Dict[str, Any]) -> str:
        """Render Markdown stranding risk timeline."""
        entries = data.get("stranding_risk", [])
        if not entries:
            return ""
        lines = [
            "## 3. Stranding Risk Timeline",
            "",
            "| Year | Asset Category | Value at Risk (M EUR) | Probability | Scenario |",
            "|------|----------------|----------------------|-------------|----------|",
        ]
        for e in entries:
            lines.append(
                f"| {e.get('year', '')} | {e.get('asset_category', '')} | "
                f"{e.get('at_risk_value', 0):,.1f} | "
                f"{e.get('probability_pct', 0):.0f}% | {e.get('scenario', '')} |"
            )
        return "\n".join(lines)

    def _md_regulatory_exposure(self, data: Dict[str, Any]) -> str:
        """Render Markdown regulatory exposure table."""
        entries = data.get("regulatory_exposure", [])
        if not entries:
            return ""
        lines = [
            "## 4. Regulatory Exposure",
            "",
            "| Regulation | Scope | Current (EUR) | Projected (EUR) | Year | Threshold |",
            "|------------|-------|---------------|-----------------|------|-----------|",
        ]
        for e in entries:
            exceeded = "EXCEEDED" if e.get("threshold_exceeded", False) else "OK"
            lines.append(
                f"| {e.get('regulation', '')} | {e.get('scope', '')} | "
                f"{e.get('current_exposure_eur', 0):,.0f} | "
                f"{e.get('projected_exposure_eur', 0):,.0f} | "
                f"{e.get('projection_year', '')} | {exceeded} |"
            )
        return "\n".join(lines)

    def _md_competitive_radar(self, data: Dict[str, Any]) -> str:
        """Render Markdown competitive position radar data."""
        axes = data.get("competitive_radar", [])
        if not axes:
            return ""
        lines = [
            "## 5. Competitive Position",
            "",
            "| Axis | Org Score | Peer Average | Best-in-Class |",
            "|------|-----------|--------------|---------------|",
        ]
        for a in axes:
            lines.append(
                f"| {a.get('axis_name', '')} | {a.get('org_value', 0):.0f} | "
                f"{a.get('peer_average', 0):.0f} | {a.get('best_in_class', 0):.0f} |"
            )
        return "\n".join(lines)

    def _md_carbon_sensitivity(self, data: Dict[str, Any]) -> str:
        """Render Markdown carbon price sensitivity table."""
        scenarios = data.get("carbon_price_sensitivity", [])
        if not scenarios:
            return ""
        lines = [
            "## 6. Carbon Price Sensitivity",
            "",
            "| Scenario | Price (EUR/tCO2e) | Annual Cost (EUR) | EBITDA Impact | Margin Impact |",
            "|----------|-------------------|-------------------|---------------|---------------|",
        ]
        for s in scenarios:
            lines.append(
                f"| {s.get('scenario_label', '')} | {s.get('carbon_price_eur', 0):,.0f} | "
                f"{s.get('annual_cost_eur', 0):,.0f} | "
                f"{s.get('ebitda_impact_pct', 0):+.1f}% | "
                f"{s.get('margin_impact_pct', 0):+.1f}% |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-047 GHG Benchmark v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_composite_risk(data),
            self._html_itr_gauge(data),
            self._html_stranding_risk(data),
            self._html_regulatory_exposure(data),
            self._html_competitive_radar(data),
            self._html_carbon_sensitivity(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document."""
        company = self._get_val(data, "company_name", "Organization")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Transition Risk - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #c1121f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".tl-green{color:#2a9d8f;font-weight:700;}\n"
            ".tl-amber{color:#e9c46a;font-weight:700;}\n"
            ".tl-red{color:#e76f51;font-weight:700;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".risk-card{background:#f0f4f8;border-radius:8px;padding:1.5rem;"
            "text-align:center;margin:1rem 0;border-left:5px solid #e76f51;}\n"
            ".risk-value{font-size:2.2rem;font-weight:700;}\n"
            ".risk-label{font-size:0.9rem;color:#555;}\n"
            ".exceeded{color:#c1121f;font-weight:700;}\n"
            ".within{color:#2a9d8f;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            '<div class="section">\n'
            f"<h1>Transition Risk Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_composite_risk(self, data: Dict[str, Any]) -> str:
        """Render HTML composite risk score."""
        risk = data.get("composite_risk")
        if not risk:
            return ""
        total = risk.get("total_score", 0)
        tl = TrafficLight(risk.get("traffic_light", "amber"))
        color = _tl_color(tl)
        category = risk.get("risk_category", "")
        card = (
            f'<div class="risk-card" style="border-left-color:{color};">\n'
            f'<div class="risk-value" style="color:{color};">{total:.0f} / 100</div>\n'
            f'<div class="risk-label">{category}</div>\n</div>\n'
        )
        dims = risk.get("dimensions", [])
        dim_rows = ""
        for d in dims:
            level = RiskLevel(d.get("risk_level", "medium"))
            dim_color = _risk_color(level)
            dim_rows += (
                f"<tr><td>{d.get('dimension_name', '')}</td>"
                f"<td>{d.get('score', 0):.0f}</td>"
                f"<td>{d.get('weight', 0):.0%}</td>"
                f"<td>{d.get('weighted_score', 0):.1f}</td>"
                f'<td style="color:{dim_color};font-weight:600;">'
                f"{level.value.replace('_', ' ').title()}</td></tr>\n"
            )
        dim_table = ""
        if dim_rows:
            dim_table = (
                "<table><thead><tr><th>Dimension</th><th>Score</th><th>Weight</th>"
                "<th>Weighted</th><th>Risk</th></tr></thead>\n"
                f"<tbody>{dim_rows}</tbody></table>\n"
            )
        return (
            '<div class="section">\n<h2>1. Composite Transition Risk</h2>\n'
            f"{card}{dim_table}</div>"
        )

    def _html_itr_gauge(self, data: Dict[str, Any]) -> str:
        """Render HTML ITR gauge."""
        itr = data.get("itr_gauge")
        if not itr:
            return ""
        value = itr.get("itr_value", 0)
        color = "#2a9d8f" if value <= 1.5 else ("#e9c46a" if value <= 2.0 else "#e76f51")
        return (
            '<div class="section">\n<h2>2. Implied Temperature Rise</h2>\n'
            f'<div class="risk-card" style="border-left-color:{color};">\n'
            f'<div class="risk-value" style="color:{color};">{value:.1f}&deg;C</div>\n'
            f'<div class="risk-label">Implied Temperature Rise</div>\n</div>\n</div>'
        )

    def _html_stranding_risk(self, data: Dict[str, Any]) -> str:
        """Render HTML stranding risk timeline."""
        entries = data.get("stranding_risk", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            rows += (
                f"<tr><td>{e.get('year', '')}</td>"
                f"<td>{e.get('asset_category', '')}</td>"
                f"<td>{e.get('at_risk_value', 0):,.1f}</td>"
                f"<td>{e.get('probability_pct', 0):.0f}%</td>"
                f"<td>{e.get('scenario', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Stranding Risk</h2>\n'
            "<table><thead><tr><th>Year</th><th>Asset Category</th>"
            "<th>Value at Risk (M EUR)</th><th>Probability</th>"
            "<th>Scenario</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_regulatory_exposure(self, data: Dict[str, Any]) -> str:
        """Render HTML regulatory exposure table."""
        entries = data.get("regulatory_exposure", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            exceeded = e.get("threshold_exceeded", False)
            css = "exceeded" if exceeded else "within"
            label = "EXCEEDED" if exceeded else "OK"
            rows += (
                f"<tr><td>{e.get('regulation', '')}</td>"
                f"<td>{e.get('scope', '')}</td>"
                f"<td>{e.get('current_exposure_eur', 0):,.0f}</td>"
                f"<td>{e.get('projected_exposure_eur', 0):,.0f}</td>"
                f"<td>{e.get('projection_year', '')}</td>"
                f'<td class="{css}">{label}</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>4. Regulatory Exposure</h2>\n'
            "<table><thead><tr><th>Regulation</th><th>Scope</th>"
            "<th>Current (EUR)</th><th>Projected (EUR)</th>"
            "<th>Year</th><th>Threshold</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_competitive_radar(self, data: Dict[str, Any]) -> str:
        """Render HTML competitive position table."""
        axes = data.get("competitive_radar", [])
        if not axes:
            return ""
        rows = ""
        for a in axes:
            rows += (
                f"<tr><td>{a.get('axis_name', '')}</td>"
                f"<td>{a.get('org_value', 0):.0f}</td>"
                f"<td>{a.get('peer_average', 0):.0f}</td>"
                f"<td>{a.get('best_in_class', 0):.0f}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>5. Competitive Position</h2>\n'
            "<table><thead><tr><th>Dimension</th><th>Org</th>"
            "<th>Peer Avg</th><th>Best</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_carbon_sensitivity(self, data: Dict[str, Any]) -> str:
        """Render HTML carbon price sensitivity."""
        scenarios = data.get("carbon_price_sensitivity", [])
        if not scenarios:
            return ""
        rows = ""
        for s in scenarios:
            rows += (
                f"<tr><td>{s.get('scenario_label', '')}</td>"
                f"<td>{s.get('carbon_price_eur', 0):,.0f}</td>"
                f"<td>{s.get('annual_cost_eur', 0):,.0f}</td>"
                f"<td>{s.get('ebitda_impact_pct', 0):+.1f}%</td>"
                f"<td>{s.get('margin_impact_pct', 0):+.1f}%</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>6. Carbon Price Sensitivity</h2>\n'
            "<table><thead><tr><th>Scenario</th><th>Price (EUR/tCO2e)</th>"
            "<th>Annual Cost</th><th>EBITDA Impact</th>"
            "<th>Margin Impact</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-047 GHG Benchmark v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render transition risk as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "transition_risk_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "composite_risk": data.get("composite_risk"),
            "itr_gauge": data.get("itr_gauge"),
            "stranding_risk": data.get("stranding_risk", []),
            "regulatory_exposure": data.get("regulatory_exposure", []),
            "competitive_radar": data.get("competitive_radar", []),
            "carbon_price_sensitivity": data.get("carbon_price_sensitivity", []),
            "chart_data": {
                "risk_gauge": self._build_risk_gauge(data),
                "itr_gauge": self._build_itr_gauge(data),
                "radar": self._build_radar_chart(data),
                "sensitivity_line": self._build_sensitivity_chart(data),
            },
        }

    def _build_risk_gauge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build composite risk gauge chart data."""
        risk = data.get("composite_risk")
        if not risk:
            return {}
        return {
            "value": risk.get("total_score", 0),
            "min": 0,
            "max": 100,
            "thresholds": [
                {"label": "Low", "max": 30, "color": "#2a9d8f"},
                {"label": "Medium", "max": 60, "color": "#e9c46a"},
                {"label": "High", "max": 80, "color": "#e76f51"},
                {"label": "Very High", "max": 100, "color": "#c1121f"},
            ],
        }

    def _build_itr_gauge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build ITR gauge chart data."""
        itr = data.get("itr_gauge")
        if not itr:
            return {}
        return {
            "value": itr.get("itr_value", 0),
            "min": 1.0,
            "max": 4.0,
            "thresholds": [
                {"label": "1.5C aligned", "max": 1.5, "color": "#2a9d8f"},
                {"label": "2C aligned", "max": 2.0, "color": "#e9c46a"},
                {"label": "Above 2C", "max": 3.0, "color": "#e76f51"},
                {"label": "Above 3C", "max": 4.0, "color": "#c1121f"},
            ],
            "peer_median": itr.get("peer_median_itr"),
        }

    def _build_radar_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build competitive radar chart data."""
        axes = data.get("competitive_radar", [])
        if not axes:
            return {}
        return {
            "labels": [a.get("axis_name", "") for a in axes],
            "org_values": [a.get("org_value", 0) for a in axes],
            "peer_avg_values": [a.get("peer_average", 0) for a in axes],
            "best_values": [a.get("best_in_class", 0) for a in axes],
        }

    def _build_sensitivity_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build carbon price sensitivity chart data."""
        scenarios = data.get("carbon_price_sensitivity", [])
        if not scenarios:
            return {}
        return {
            "labels": [s.get("scenario_label", "") for s in scenarios],
            "prices": [s.get("carbon_price_eur", 0) for s in scenarios],
            "costs": [s.get("annual_cost_eur", 0) for s in scenarios],
            "ebitda_impact": [s.get("ebitda_impact_pct", 0) for s in scenarios],
        }
