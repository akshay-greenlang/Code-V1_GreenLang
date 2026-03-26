# -*- coding: utf-8 -*-
"""
CDPClimateBenchmarkSection - CDP Climate Benchmark Section for PACK-047.

Generates CDP-compliant benchmark context sections for C6 (emissions data)
and C7 (emissions breakdown) with sector comparison, performance band
positioning (A-list through D-), and sector-specific supplementary data.

CDP References:
    - C6: Emissions Data (C6.1-C6.10)
    - C7: Emissions Breakdown (C7.1-C7.9)
    - CDP Scoring Methodology: Performance bands A through D-
    - CDP Sector-Specific Supplementary Questionnaires

Sections:
    1. CDP C6 Emissions Benchmark Context
    2. CDP C7 Sector Comparison
    3. Performance Band Positioning
    4. Sector-Specific Supplementary Data
    5. Provenance Footer

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - JSON (structured CDP response data)

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

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


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


class CDPBand(str, Enum):
    """CDP performance band."""
    A = "A"
    A_MINUS = "A-"
    B = "B"
    B_MINUS = "B-"
    C = "C"
    C_MINUS = "C-"
    D = "D"
    D_MINUS = "D-"
    F = "F"


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class C6BenchmarkMetric(BaseModel):
    """CDP C6 emissions benchmark metric."""
    question_ref: str = Field("", description="CDP question reference (e.g., C6.1)")
    metric_name: str = Field(..., description="Metric name")
    org_value: float = Field(0.0, description="Organisation value")
    sector_average: Optional[float] = Field(None, description="Sector average")
    sector_median: Optional[float] = Field(None, description="Sector median")
    sector_best: Optional[float] = Field(None, description="Sector best performer")
    unit: str = Field("", description="Metric unit")
    yoy_change_pct: Optional[float] = Field(None, description="Year-over-year change (%)")
    benchmark_context: str = Field("", description="Contextual benchmark note")


class C7SectorComparison(BaseModel):
    """CDP C7 emissions breakdown sector comparison."""
    breakdown_category: str = Field(..., description="Breakdown category (e.g., by country)")
    metric_name: str = Field("", description="Metric name")
    org_value: float = Field(0.0, description="Organisation value")
    sector_average: Optional[float] = Field(None, description="Sector average")
    org_share_pct: float = Field(0.0, description="Organisation share (%)")
    sector_avg_share_pct: Optional[float] = Field(None, description="Sector avg share (%)")
    unit: str = Field("", description="Unit")


class PerformanceBandPosition(BaseModel):
    """CDP performance band positioning."""
    overall_band: CDPBand = Field(CDPBand.C, description="Overall CDP band")
    category_scores: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-category scores [{category, band, points, max_points}]",
    )
    sector_band_distribution: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sector band distribution [{band, count, pct}]",
    )
    org_percentile_in_sector: Optional[float] = Field(
        None, ge=0, le=100, description="Org percentile within sector"
    )
    year: int = Field(0, description="CDP scoring year")


class SectorSupplementaryData(BaseModel):
    """CDP sector-specific supplementary data."""
    sector_module: str = Field("", description="CDP sector module code (e.g., C-EU)")
    sector_name: str = Field("", description="Sector name (e.g., Electric Utilities)")
    question_ref: str = Field("", description="Question reference")
    metric_name: str = Field(..., description="Metric name")
    org_value: float = Field(0.0, description="Organisation value")
    sector_average: Optional[float] = Field(None, description="Sector average")
    unit: str = Field("", description="Unit")
    benchmark_note: str = Field("", description="Benchmark context note")


class CDPBenchmarkInput(BaseModel):
    """Complete input model for CDPClimateBenchmarkSection."""
    company_name: str = Field("Organization", description="Company name")
    reporting_year: int = Field(0, description="CDP reporting year")
    cdp_submission_year: Optional[int] = Field(None, description="Submission year")
    c6_benchmark_metrics: List[C6BenchmarkMetric] = Field(
        default_factory=list, description="C6 benchmark metrics"
    )
    c7_sector_comparisons: List[C7SectorComparison] = Field(
        default_factory=list, description="C7 sector comparisons"
    )
    performance_band: Optional[PerformanceBandPosition] = Field(
        None, description="Performance band positioning"
    )
    sector_supplementary: List[SectorSupplementaryData] = Field(
        default_factory=list, description="Sector supplementary data"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _band_color(band: CDPBand) -> str:
    """Return hex color for CDP band."""
    mapping = {
        CDPBand.A: "#1b4332",
        CDPBand.A_MINUS: "#2d6a4f",
        CDPBand.B: "#40916c",
        CDPBand.B_MINUS: "#52b788",
        CDPBand.C: "#e9c46a",
        CDPBand.C_MINUS: "#f4a261",
        CDPBand.D: "#e76f51",
        CDPBand.D_MINUS: "#c1121f",
        CDPBand.F: "#780000",
    }
    return mapping.get(band, "#888888")


# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class CDPClimateBenchmarkSection:
    """
    CDP climate benchmark section template.

    Renders CDP-compliant benchmark context for C6 emissions data and
    C7 emissions breakdown, including sector comparison, performance
    band positioning, and sector-specific supplementary data. All
    outputs include SHA-256 provenance hashing.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = CDPClimateBenchmarkSection()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CDPClimateBenchmarkSection."""
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
        """Render CDP benchmark section as Markdown."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render CDP benchmark section as HTML."""
        start = time.monotonic()
        self.generated_at = _utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render CDP benchmark section as JSON dict."""
        start = time.monotonic()
        self.generated_at = _utcnow()
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
            self._md_c6_benchmark(data),
            self._md_c7_comparison(data),
            self._md_performance_band(data),
            self._md_sector_supplementary(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# CDP Climate Benchmark Section - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {_utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_c6_benchmark(self, data: Dict[str, Any]) -> str:
        """Render Markdown C6 emissions benchmark context."""
        metrics = data.get("c6_benchmark_metrics", [])
        if not metrics:
            return ""
        lines = [
            "## 1. C6 Emissions Data - Benchmark Context",
            "",
            "| Question | Metric | Org Value | Sector Avg | Sector Best | Unit | YoY |",
            "|----------|--------|-----------|-----------|-------------|------|-----|",
        ]
        for m in metrics:
            avg = m.get("sector_average")
            best = m.get("sector_best")
            yoy = m.get("yoy_change_pct")
            avg_str = f"{avg:,.2f}" if avg is not None else "-"
            best_str = f"{best:,.2f}" if best is not None else "-"
            yoy_str = f"{yoy:+.1f}%" if yoy is not None else "-"
            lines.append(
                f"| {m.get('question_ref', '')} | {m.get('metric_name', '')} | "
                f"{m.get('org_value', 0):,.2f} | {avg_str} | {best_str} | "
                f"{m.get('unit', '')} | {yoy_str} |"
            )
        return "\n".join(lines)

    def _md_c7_comparison(self, data: Dict[str, Any]) -> str:
        """Render Markdown C7 sector comparison."""
        comparisons = data.get("c7_sector_comparisons", [])
        if not comparisons:
            return ""
        lines = [
            "## 2. C7 Emissions Breakdown - Sector Comparison",
            "",
            "| Category | Metric | Org Value | Org Share | Sector Avg Share | Unit |",
            "|----------|--------|-----------|----------|-----------------|------|",
        ]
        for c in comparisons:
            s_share = c.get("sector_avg_share_pct")
            s_share_str = f"{s_share:.1f}%" if s_share is not None else "-"
            lines.append(
                f"| {c.get('breakdown_category', '')} | {c.get('metric_name', '')} | "
                f"{c.get('org_value', 0):,.2f} | {c.get('org_share_pct', 0):.1f}% | "
                f"{s_share_str} | {c.get('unit', '')} |"
            )
        return "\n".join(lines)

    def _md_performance_band(self, data: Dict[str, Any]) -> str:
        """Render Markdown performance band positioning."""
        band_data = data.get("performance_band")
        if not band_data:
            return ""
        overall = CDPBand(band_data.get("overall_band", "C"))
        year = band_data.get("year", "")
        pctile = band_data.get("org_percentile_in_sector")
        lines = [
            "## 3. Performance Band Positioning",
            "",
            f"**Overall Band:** {overall.value} ({year})",
        ]
        if pctile is not None:
            lines.append(f"**Sector Percentile:** P{pctile:.0f}")
        lines.append("")
        # Category scores
        cat_scores = band_data.get("category_scores", [])
        if cat_scores:
            lines.append("### Category Scores")
            lines.append("")
            lines.append("| Category | Band | Points | Max Points |")
            lines.append("|----------|------|--------|------------|")
            for cs in cat_scores:
                lines.append(
                    f"| {cs.get('category', '')} | {cs.get('band', '')} | "
                    f"{cs.get('points', 0)} | {cs.get('max_points', 0)} |"
                )
            lines.append("")
        # Sector distribution
        dist = band_data.get("sector_band_distribution", [])
        if dist:
            lines.append("### Sector Band Distribution")
            lines.append("")
            lines.append("| Band | Count | % of Sector |")
            lines.append("|------|-------|------------|")
            for d in dist:
                lines.append(
                    f"| {d.get('band', '')} | {d.get('count', 0)} | "
                    f"{d.get('pct', 0):.1f}% |"
                )
        return "\n".join(lines)

    def _md_sector_supplementary(self, data: Dict[str, Any]) -> str:
        """Render Markdown sector-specific supplementary data."""
        entries = data.get("sector_supplementary", [])
        if not entries:
            return ""
        lines = [
            "## 4. Sector-Specific Supplementary Data",
            "",
            "| Module | Sector | Question | Metric | Org Value | Sector Avg | Unit |",
            "|--------|--------|----------|--------|-----------|-----------|------|",
        ]
        for e in entries:
            avg = e.get("sector_average")
            avg_str = f"{avg:,.2f}" if avg is not None else "-"
            lines.append(
                f"| {e.get('sector_module', '')} | {e.get('sector_name', '')} | "
                f"{e.get('question_ref', '')} | {e.get('metric_name', '')} | "
                f"{e.get('org_value', 0):,.2f} | {avg_str} | {e.get('unit', '')} |"
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
            self._html_c6_benchmark(data),
            self._html_c7_comparison(data),
            self._html_performance_band(data),
            self._html_sector_supplementary(data),
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
            f"<title>CDP Benchmark - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #006837;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#264653;margin-top:1.5rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#e8f5e9;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".band-card{display:inline-block;border-radius:8px;padding:1rem 2rem;"
            "text-align:center;margin:0.5rem;color:#fff;font-weight:700;font-size:2rem;}\n"
            "</style>\n</head>\n<body>\n"
            f"{body}\n"
            "</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            '<div class="section">\n'
            f"<h1>CDP Climate Benchmark &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year}</p>\n<hr>\n</div>"
        )

    def _html_c6_benchmark(self, data: Dict[str, Any]) -> str:
        """Render HTML C6 benchmark table."""
        metrics = data.get("c6_benchmark_metrics", [])
        if not metrics:
            return ""
        rows = ""
        for m in metrics:
            avg = m.get("sector_average")
            avg_str = f"{avg:,.2f}" if avg is not None else "-"
            rows += (
                f"<tr><td>{m.get('question_ref', '')}</td>"
                f"<td>{m.get('metric_name', '')}</td>"
                f"<td>{m.get('org_value', 0):,.2f}</td>"
                f"<td>{avg_str}</td>"
                f"<td>{m.get('unit', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>1. C6 Emissions Benchmark</h2>\n'
            "<table><thead><tr><th>Question</th><th>Metric</th>"
            "<th>Org Value</th><th>Sector Avg</th><th>Unit</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_c7_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML C7 sector comparison."""
        comparisons = data.get("c7_sector_comparisons", [])
        if not comparisons:
            return ""
        rows = ""
        for c in comparisons:
            rows += (
                f"<tr><td>{c.get('breakdown_category', '')}</td>"
                f"<td>{c.get('org_value', 0):,.2f}</td>"
                f"<td>{c.get('org_share_pct', 0):.1f}%</td>"
                f"<td>{c.get('unit', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. C7 Sector Comparison</h2>\n'
            "<table><thead><tr><th>Category</th><th>Org Value</th>"
            "<th>Org Share</th><th>Unit</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_performance_band(self, data: Dict[str, Any]) -> str:
        """Render HTML performance band positioning."""
        band_data = data.get("performance_band")
        if not band_data:
            return ""
        overall = CDPBand(band_data.get("overall_band", "C"))
        color = _band_color(overall)
        card = (
            f'<div class="band-card" style="background:{color};">'
            f"{overall.value}</div>\n"
        )
        return (
            '<div class="section">\n<h2>3. Performance Band</h2>\n'
            f"<div>{card}</div>\n</div>"
        )

    def _html_sector_supplementary(self, data: Dict[str, Any]) -> str:
        """Render HTML sector supplementary."""
        entries = data.get("sector_supplementary", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            avg = e.get("sector_average")
            avg_str = f"{avg:,.2f}" if avg is not None else "-"
            rows += (
                f"<tr><td>{e.get('sector_module', '')}</td>"
                f"<td>{e.get('metric_name', '')}</td>"
                f"<td>{e.get('org_value', 0):,.2f}</td>"
                f"<td>{avg_str}</td>"
                f"<td>{e.get('unit', '')}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Sector Supplementary</h2>\n'
            "<table><thead><tr><th>Module</th><th>Metric</th>"
            "<th>Org Value</th><th>Sector Avg</th><th>Unit</th></tr></thead>\n"
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
        """Render CDP benchmark as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "cdp_climate_benchmark_section",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year", ""),
            "cdp_submission_year": data.get("cdp_submission_year"),
            "c6_benchmark_metrics": data.get("c6_benchmark_metrics", []),
            "c7_sector_comparisons": data.get("c7_sector_comparisons", []),
            "performance_band": data.get("performance_band"),
            "sector_supplementary": data.get("sector_supplementary", []),
        }
