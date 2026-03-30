# -*- coding: utf-8 -*-
"""
UncertaintyReport - Uncertainty Analysis for PACK-046.

Generates an uncertainty analysis report based on IPCC Tier 1/2
methodology, including data quality summary, uncertainty by metric,
combined uncertainty bands, confidence intervals, data improvement
recommendations, and quality trend analysis.

Sections:
    1. Methodology (IPCC Tier 1/2)
    2. Data Quality Summary Table
    3. Uncertainty by Metric
    4. Combined Uncertainty Bands
    5. Confidence Intervals
    6. Data Improvement Recommendations
    7. Quality Trend

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - PDF (via HTML rendering pipeline)
    - JSON (structured with error bar / heatmap data)

Author: GreenLang Team
Version: 46.0.0
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

class QualityRating(str, Enum):
    """Data quality rating levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class IPCCTier(str, Enum):
    """IPCC uncertainty tier."""
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class DataQualityEntry(BaseModel):
    """Data quality assessment for a single source/metric."""
    source_name: str = Field(..., description="Data source or metric name")
    quality_rating: QualityRating = Field(QualityRating.MEDIUM, description="Quality rating")
    quality_score: float = Field(0.0, ge=0, le=100, description="Quality score 0-100")
    completeness_pct: float = Field(0.0, description="Data completeness %")
    reliability: str = Field("", description="Reliability description")
    temporal_correlation: str = Field("", description="Temporal correlation rating")
    geographic_correlation: str = Field("", description="Geographic correlation rating")
    technology_correlation: str = Field("", description="Technology correlation rating")

class UncertaintyByMetric(BaseModel):
    """Uncertainty assessment for a single intensity metric."""
    metric_name: str = Field(..., description="Intensity metric name")
    central_value: float = Field(0.0, description="Central (best estimate) value")
    unit: str = Field("", description="Metric unit")
    tier: IPCCTier = Field(IPCCTier.TIER_1, description="IPCC tier applied")
    activity_data_uncertainty_pct: float = Field(0.0, description="Activity data uncertainty %")
    emission_factor_uncertainty_pct: float = Field(0.0, description="Emission factor uncertainty %")
    denominator_uncertainty_pct: float = Field(0.0, description="Denominator uncertainty %")
    combined_uncertainty_pct: float = Field(0.0, description="Combined uncertainty %")
    lower_bound: float = Field(0.0, description="Lower bound (95% CI)")
    upper_bound: float = Field(0.0, description="Upper bound (95% CI)")

class CombinedUncertaintyBand(BaseModel):
    """Combined uncertainty band for aggregate metrics."""
    scope_label: str = Field(..., description="Scope or aggregate label")
    central_intensity: float = Field(0.0, description="Central intensity value")
    combined_uncertainty_pct: float = Field(0.0, description="Combined uncertainty %")
    lower_bound: float = Field(0.0, description="Lower bound")
    upper_bound: float = Field(0.0, description="Upper bound")
    confidence_level: str = Field("95%", description="Confidence level")

class ConfidenceInterval(BaseModel):
    """Confidence interval for a metric at specified confidence level."""
    metric_name: str = Field(..., description="Metric name")
    confidence_level: str = Field("95%", description="Confidence level (e.g. 95%)")
    lower: float = Field(0.0, description="Lower bound")
    central: float = Field(0.0, description="Central value")
    upper: float = Field(0.0, description="Upper bound")
    half_width_pct: float = Field(0.0, description="Half-width as % of central")

class QualityTrendEntry(BaseModel):
    """Quality trend for a single year."""
    year: int = Field(..., description="Reporting year")
    overall_quality_score: float = Field(0.0, description="Overall quality score 0-100")
    combined_uncertainty_pct: float = Field(0.0, description="Combined uncertainty %")
    data_coverage_pct: float = Field(0.0, description="Data coverage %")

class ImprovementRecommendation(BaseModel):
    """Data improvement recommendation."""
    priority: int = Field(1, ge=1, le=5, description="Priority (1=highest)")
    area: str = Field("", description="Area of improvement")
    recommendation: str = Field(..., description="Recommendation text")
    current_uncertainty_pct: float = Field(0.0, description="Current uncertainty %")
    target_uncertainty_pct: float = Field(0.0, description="Target uncertainty %")
    expected_improvement: str = Field("", description="Expected improvement description")

class UncertaintyReportInput(BaseModel):
    """Complete input model for UncertaintyReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    methodology_description: str = Field("", description="Methodology narrative")
    ipcc_tier: IPCCTier = Field(IPCCTier.TIER_1, description="Primary IPCC tier")
    data_quality_summary: List[DataQualityEntry] = Field(
        default_factory=list, description="Data quality entries"
    )
    uncertainty_by_metric: List[UncertaintyByMetric] = Field(
        default_factory=list, description="Per-metric uncertainty"
    )
    combined_bands: List[CombinedUncertaintyBand] = Field(
        default_factory=list, description="Combined uncertainty bands"
    )
    confidence_intervals: List[ConfidenceInterval] = Field(
        default_factory=list, description="Confidence intervals"
    )
    improvement_recommendations: List[ImprovementRecommendation] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    quality_trend: List[QualityTrendEntry] = Field(
        default_factory=list, description="Quality trend over years"
    )

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _quality_label(rating: QualityRating) -> str:
    """Return human-readable label for quality rating."""
    mapping = {
        QualityRating.HIGH: "HIGH",
        QualityRating.MEDIUM: "MEDIUM",
        QualityRating.LOW: "LOW",
        QualityRating.VERY_LOW: "VERY LOW",
    }
    return mapping.get(rating, rating.value.upper())

def _quality_css(rating: QualityRating) -> str:
    """Return CSS class for quality rating."""
    mapping = {
        QualityRating.HIGH: "q-high",
        QualityRating.MEDIUM: "q-medium",
        QualityRating.LOW: "q-low",
        QualityRating.VERY_LOW: "q-very-low",
    }
    return mapping.get(rating, "q-medium")

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class UncertaintyReport:
    """
    Uncertainty analysis report template.

    Renders IPCC Tier 1/2 uncertainty analysis with data quality
    assessments, per-metric uncertainty bands, combined uncertainty,
    confidence intervals, improvement recommendations, and quality
    trends. All outputs include SHA-256 provenance hashing.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = UncertaintyReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize UncertaintyReport."""
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

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render uncertainty report as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render uncertainty report as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render uncertainty report as JSON dict."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_json(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    # ==================================================================
    # MARKDOWN RENDERING
    # ==================================================================

    def _render_md(self, data: Dict[str, Any]) -> str:
        """Render full Markdown document."""
        sections: List[str] = [
            self._md_header(data),
            self._md_methodology(data),
            self._md_data_quality(data),
            self._md_uncertainty_by_metric(data),
            self._md_combined_bands(data),
            self._md_confidence_intervals(data),
            self._md_recommendations(data),
            self._md_quality_trend(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        tier = self._get_val(data, "ipcc_tier", "tier_1")
        return (
            f"# Uncertainty Analysis Report - {company}\n\n"
            f"**Period:** {period} | **IPCC Tier:** {tier.replace('_', ' ').title()} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render Markdown methodology section."""
        desc = self._get_val(data, "methodology_description", "")
        tier = self._get_val(data, "ipcc_tier", "tier_1")
        if not desc:
            if tier == "tier_1":
                desc = (
                    "Uncertainty is assessed using IPCC Tier 1 methodology with "
                    "default uncertainty ranges from the 2006 IPCC Guidelines. "
                    "Combined uncertainty is calculated using error propagation "
                    "(root sum of squares) assuming independent, normally "
                    "distributed errors."
                )
            else:
                desc = (
                    "Uncertainty is assessed using IPCC Tier 2 methodology with "
                    "country-specific or facility-specific uncertainty ranges. "
                    "Monte Carlo simulation may be applied for asymmetric "
                    "distributions."
                )
        return f"## 1. Methodology\n\n{desc}"

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render Markdown data quality summary table."""
        entries = data.get("data_quality_summary", [])
        if not entries:
            return "## 2. Data Quality Summary\n\nNo data quality information available."
        lines = [
            "## 2. Data Quality Summary",
            "",
            "| Source | Rating | Score | Completeness | Reliability |",
            "|--------|--------|-------|-------------|-------------|",
        ]
        for e in entries:
            name = e.get("source_name", "")
            rating = QualityRating(e.get("quality_rating", "medium"))
            score = e.get("quality_score", 0)
            completeness = e.get("completeness_pct", 0)
            reliability = e.get("reliability", "-")
            lines.append(
                f"| {name} | **{_quality_label(rating)}** | "
                f"{score:.0f}/100 | {completeness:.0f}% | {reliability} |"
            )
        return "\n".join(lines)

    def _md_uncertainty_by_metric(self, data: Dict[str, Any]) -> str:
        """Render Markdown uncertainty by metric table."""
        metrics = data.get("uncertainty_by_metric", [])
        if not metrics:
            return "## 3. Uncertainty by Metric\n\nNo metric uncertainty data available."
        lines = [
            "## 3. Uncertainty by Metric",
            "",
            "| Metric | Central | AD Unc. | EF Unc. | Denom Unc. | Combined | 95% CI |",
            "|--------|---------|---------|---------|------------|----------|--------|",
        ]
        for m in metrics:
            name = m.get("metric_name", "")
            central = m.get("central_value", 0)
            unit = m.get("unit", "")
            ad_unc = m.get("activity_data_uncertainty_pct", 0)
            ef_unc = m.get("emission_factor_uncertainty_pct", 0)
            d_unc = m.get("denominator_uncertainty_pct", 0)
            combined = m.get("combined_uncertainty_pct", 0)
            lower = m.get("lower_bound", 0)
            upper = m.get("upper_bound", 0)
            lines.append(
                f"| {name} | {central:,.4f} {unit} | {ad_unc:.1f}% | "
                f"{ef_unc:.1f}% | {d_unc:.1f}% | {combined:.1f}% | "
                f"[{lower:,.4f}, {upper:,.4f}] |"
            )
        return "\n".join(lines)

    def _md_combined_bands(self, data: Dict[str, Any]) -> str:
        """Render Markdown combined uncertainty bands."""
        bands = data.get("combined_bands", [])
        if not bands:
            return ""
        lines = [
            "## 4. Combined Uncertainty Bands",
            "",
            "| Scope | Central | Uncertainty | Lower | Upper | Confidence |",
            "|-------|---------|-------------|-------|-------|------------|",
        ]
        for b in bands:
            scope = b.get("scope_label", "")
            central = b.get("central_intensity", 0)
            unc = b.get("combined_uncertainty_pct", 0)
            lower = b.get("lower_bound", 0)
            upper = b.get("upper_bound", 0)
            conf = b.get("confidence_level", "95%")
            lines.append(
                f"| {scope} | {central:,.4f} | +/-{unc:.1f}% | "
                f"{lower:,.4f} | {upper:,.4f} | {conf} |"
            )
        return "\n".join(lines)

    def _md_confidence_intervals(self, data: Dict[str, Any]) -> str:
        """Render Markdown confidence intervals."""
        cis = data.get("confidence_intervals", [])
        if not cis:
            return ""
        lines = [
            "## 5. Confidence Intervals",
            "",
            "| Metric | Confidence | Lower | Central | Upper | Half-width |",
            "|--------|-----------|-------|---------|-------|------------|",
        ]
        for ci in cis:
            name = ci.get("metric_name", "")
            conf = ci.get("confidence_level", "95%")
            lower = ci.get("lower", 0)
            central = ci.get("central", 0)
            upper = ci.get("upper", 0)
            hw = ci.get("half_width_pct", 0)
            lines.append(
                f"| {name} | {conf} | {lower:,.4f} | {central:,.4f} | "
                f"{upper:,.4f} | +/-{hw:.1f}% |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render Markdown data improvement recommendations."""
        recs = data.get("improvement_recommendations", [])
        if not recs:
            return ""
        lines = ["## 6. Data Improvement Recommendations", ""]
        for r in recs:
            priority = r.get("priority", 1)
            area = r.get("area", "")
            text = r.get("recommendation", "")
            current = r.get("current_uncertainty_pct", 0)
            target = r.get("target_uncertainty_pct", 0)
            lines.append(f"**P{priority} - {area}:** {text}")
            lines.append(
                f"   - Current: +/-{current:.1f}% -> Target: +/-{target:.1f}%"
            )
            lines.append("")
        return "\n".join(lines)

    def _md_quality_trend(self, data: Dict[str, Any]) -> str:
        """Render Markdown quality trend table."""
        trend = data.get("quality_trend", [])
        if not trend:
            return ""
        lines = [
            "## 7. Quality Trend",
            "",
            "| Year | Quality Score | Combined Unc. | Data Coverage |",
            "|------|-------------|---------------|---------------|",
        ]
        for t in trend:
            year = t.get("year", "")
            score = t.get("overall_quality_score", 0)
            unc = t.get("combined_uncertainty_pct", 0)
            coverage = t.get("data_coverage_pct", 0)
            lines.append(
                f"| {year} | {score:.0f}/100 | +/-{unc:.1f}% | {coverage:.0f}% |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-046 Intensity Metrics v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML RENDERING
    # ==================================================================

    def _render_html(self, data: Dict[str, Any]) -> str:
        """Render full HTML document."""
        body_parts: List[str] = [
            self._html_header(data),
            self._html_methodology(data),
            self._html_data_quality(data),
            self._html_uncertainty_by_metric(data),
            self._html_combined_bands(data),
            self._html_confidence_intervals(data),
            self._html_recommendations(data),
            self._html_quality_trend(data),
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
            f"<title>Uncertainty Report - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".q-high{color:#2a9d8f;font-weight:700;}\n"
            ".q-medium{color:#e9c46a;font-weight:700;}\n"
            ".q-low{color:#e76f51;font-weight:700;}\n"
            ".q-very-low{color:#d62828;font-weight:700;}\n"
            ".error-bar{display:inline-block;height:8px;background:#2a9d8f;"
            "border-radius:4px;vertical-align:middle;}\n"
            ".heatmap-cell{text-align:center;font-weight:600;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        tier = self._get_val(data, "ipcc_tier", "tier_1")
        return (
            '<div class="section">\n'
            f"<h1>Uncertainty Analysis Report &mdash; {company}</h1>\n"
            f"<p><strong>Period:</strong> {period} | "
            f"<strong>IPCC Tier:</strong> {tier.replace('_', ' ').title()}</p>\n"
            "<hr>\n</div>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology."""
        desc = self._get_val(data, "methodology_description", "")
        if not desc:
            desc = (
                "Uncertainty assessed using IPCC guidelines with error "
                "propagation (root sum of squares) for combined uncertainty."
            )
        return (
            '<div class="section">\n<h2>1. Methodology</h2>\n'
            f"<p>{desc}</p>\n</div>"
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Render HTML data quality summary table."""
        entries = data.get("data_quality_summary", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            name = e.get("source_name", "")
            rating = QualityRating(e.get("quality_rating", "medium"))
            score = e.get("quality_score", 0)
            completeness = e.get("completeness_pct", 0)
            css = _quality_css(rating)
            rows += (
                f'<tr><td>{name}</td><td class="{css}">'
                f"<strong>{_quality_label(rating)}</strong></td>"
                f"<td>{score:.0f}/100</td><td>{completeness:.0f}%</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. Data Quality Summary</h2>\n'
            "<table><thead><tr><th>Source</th><th>Rating</th>"
            "<th>Score</th><th>Completeness</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_uncertainty_by_metric(self, data: Dict[str, Any]) -> str:
        """Render HTML uncertainty by metric table."""
        metrics = data.get("uncertainty_by_metric", [])
        if not metrics:
            return ""
        rows = ""
        for m in metrics:
            name = m.get("metric_name", "")
            central = m.get("central_value", 0)
            combined = m.get("combined_uncertainty_pct", 0)
            lower = m.get("lower_bound", 0)
            upper = m.get("upper_bound", 0)
            bar_width = min(int(combined * 5), 200)
            rows += (
                f"<tr><td>{name}</td><td>{central:,.4f}</td>"
                f"<td>+/-{combined:.1f}%</td>"
                f"<td>[{lower:,.4f}, {upper:,.4f}]</td>"
                f'<td><span class="error-bar" style="width:{bar_width}px;"></span></td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>3. Uncertainty by Metric</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Central</th>"
            "<th>Uncertainty</th><th>95% CI</th><th>Visual</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_combined_bands(self, data: Dict[str, Any]) -> str:
        """Render HTML combined uncertainty bands."""
        bands = data.get("combined_bands", [])
        if not bands:
            return ""
        rows = ""
        for b in bands:
            scope = b.get("scope_label", "")
            central = b.get("central_intensity", 0)
            unc = b.get("combined_uncertainty_pct", 0)
            lower = b.get("lower_bound", 0)
            upper = b.get("upper_bound", 0)
            rows += (
                f"<tr><td>{scope}</td><td>{central:,.4f}</td>"
                f"<td>+/-{unc:.1f}%</td><td>{lower:,.4f}</td>"
                f"<td>{upper:,.4f}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Combined Uncertainty Bands</h2>\n'
            "<table><thead><tr><th>Scope</th><th>Central</th>"
            "<th>Uncertainty</th><th>Lower</th><th>Upper</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_confidence_intervals(self, data: Dict[str, Any]) -> str:
        """Render HTML confidence intervals."""
        cis = data.get("confidence_intervals", [])
        if not cis:
            return ""
        rows = ""
        for ci in cis:
            name = ci.get("metric_name", "")
            conf = ci.get("confidence_level", "95%")
            lower = ci.get("lower", 0)
            central = ci.get("central", 0)
            upper = ci.get("upper", 0)
            hw = ci.get("half_width_pct", 0)
            rows += (
                f"<tr><td>{name}</td><td>{conf}</td><td>{lower:,.4f}</td>"
                f"<td>{central:,.4f}</td><td>{upper:,.4f}</td>"
                f"<td>+/-{hw:.1f}%</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>5. Confidence Intervals</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Confidence</th>"
            "<th>Lower</th><th>Central</th><th>Upper</th>"
            "<th>Half-width</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML improvement recommendations."""
        recs = data.get("improvement_recommendations", [])
        if not recs:
            return ""
        items = ""
        for r in recs:
            priority = r.get("priority", 1)
            area = r.get("area", "")
            text = r.get("recommendation", "")
            current = r.get("current_uncertainty_pct", 0)
            target = r.get("target_uncertainty_pct", 0)
            items += (
                f"<li><strong>P{priority} - {area}:</strong> {text}"
                f" (Current: +/-{current:.1f}% &rarr; Target: +/-{target:.1f}%)</li>\n"
            )
        return (
            '<div class="section">\n<h2>6. Data Improvement Recommendations</h2>\n'
            f"<ol>{items}</ol>\n</div>"
        )

    def _html_quality_trend(self, data: Dict[str, Any]) -> str:
        """Render HTML quality trend table."""
        trend = data.get("quality_trend", [])
        if not trend:
            return ""
        rows = ""
        for t in trend:
            year = t.get("year", "")
            score = t.get("overall_quality_score", 0)
            unc = t.get("combined_uncertainty_pct", 0)
            coverage = t.get("data_coverage_pct", 0)
            rows += (
                f"<tr><td>{year}</td><td>{score:.0f}/100</td>"
                f"<td>+/-{unc:.1f}%</td><td>{coverage:.0f}%</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>7. Quality Trend</h2>\n'
            "<table><thead><tr><th>Year</th><th>Quality Score</th>"
            "<th>Uncertainty</th><th>Coverage</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-046 Intensity Metrics v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON RENDERING
    # ==================================================================

    def _render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render uncertainty report as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "uncertainty_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "ipcc_tier": self._get_val(data, "ipcc_tier", "tier_1"),
            "data_quality_summary": data.get("data_quality_summary", []),
            "uncertainty_by_metric": data.get("uncertainty_by_metric", []),
            "combined_bands": data.get("combined_bands", []),
            "confidence_intervals": data.get("confidence_intervals", []),
            "improvement_recommendations": data.get("improvement_recommendations", []),
            "quality_trend": data.get("quality_trend", []),
            "chart_data": {
                "error_bars": self._build_error_bar_data(data),
                "quality_heatmap": self._build_heatmap_data(data),
            },
        }

    def _build_error_bar_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build error bar chart data from uncertainty metrics."""
        metrics = data.get("uncertainty_by_metric", [])
        if not metrics:
            return {}
        return {
            "labels": [m.get("metric_name", "") for m in metrics],
            "central_values": [m.get("central_value", 0) for m in metrics],
            "lower_bounds": [m.get("lower_bound", 0) for m in metrics],
            "upper_bounds": [m.get("upper_bound", 0) for m in metrics],
        }

    def _build_heatmap_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build quality heatmap data from data quality summary."""
        entries = data.get("data_quality_summary", [])
        if not entries:
            return {}
        return {
            "labels": [e.get("source_name", "") for e in entries],
            "scores": [e.get("quality_score", 0) for e in entries],
            "completeness": [e.get("completeness_pct", 0) for e in entries],
        }
