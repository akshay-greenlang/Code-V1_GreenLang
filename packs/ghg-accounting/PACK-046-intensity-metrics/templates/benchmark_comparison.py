# -*- coding: utf-8 -*-
"""
BenchmarkComparisonReport - Peer Benchmark Comparison for PACK-046.

Generates a benchmark comparison report with peer group definitions,
normalisation methodology, ranking tables, percentile chart data,
gap analysis (to average, best-in-class, and target), sector
distribution, and improvement recommendations.

Sections:
    1. Peer Group Definition
    2. Normalisation Methodology
    3. Ranking Table (org vs peers)
    4. Percentile Chart Data
    5. Gap Analysis
    6. Sector Distribution
    7. Improvement Recommendations

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS)
    - PDF (via HTML rendering pipeline)
    - JSON (structured with spider/radar chart data)

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

class GapType(str, Enum):
    """Gap analysis comparison type."""
    TO_AVERAGE = "to_average"
    TO_BEST = "to_best_in_class"
    TO_TARGET = "to_target"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class PeerGroupDefinition(BaseModel):
    """Definition of the peer group used for benchmarking."""
    peer_group_name: str = Field(..., description="Peer group name")
    sector: str = Field("", description="Industry sector")
    sub_sector: str = Field("", description="Sub-sector")
    region: str = Field("", description="Geographic region")
    size_range: str = Field("", description="Revenue or employee size range")
    peer_count: int = Field(0, description="Number of peers in group")
    data_source: str = Field("", description="Source of benchmark data")
    data_year: Optional[int] = Field(None, description="Year of benchmark data")

class PeerRankingEntry(BaseModel):
    """Single entry in the ranking table."""
    rank: int = Field(..., description="Rank position (1 = best)")
    entity_name: str = Field(..., description="Organisation or peer name")
    is_org: bool = Field(False, description="Whether this is the reporting organisation")
    intensity_value: float = Field(0.0, description="Intensity metric value")
    intensity_unit: str = Field("", description="Intensity unit")
    denominator_type: str = Field("", description="Denominator type used")
    percentile: Optional[float] = Field(None, description="Percentile position")

class GapAnalysisItem(BaseModel):
    """Gap analysis for a single metric."""
    metric_name: str = Field(..., description="Metric name")
    org_value: float = Field(0.0, description="Organisation value")
    peer_average: float = Field(0.0, description="Peer group average")
    best_in_class: float = Field(0.0, description="Best-in-class value")
    target_value: Optional[float] = Field(None, description="Target value if set")
    gap_to_average: float = Field(0.0, description="Gap to average (absolute)")
    gap_to_average_pct: float = Field(0.0, description="Gap to average (%)")
    gap_to_best: float = Field(0.0, description="Gap to best-in-class (absolute)")
    gap_to_best_pct: float = Field(0.0, description="Gap to best-in-class (%)")
    gap_to_target: Optional[float] = Field(None, description="Gap to target (absolute)")
    gap_to_target_pct: Optional[float] = Field(None, description="Gap to target (%)")

class SectorDistributionBucket(BaseModel):
    """Histogram bucket for sector distribution."""
    range_low: float = Field(0.0, description="Bucket lower bound")
    range_high: float = Field(0.0, description="Bucket upper bound")
    count: int = Field(0, description="Number of entities in bucket")
    includes_org: bool = Field(False, description="Whether org falls in this bucket")

class BenchmarkRecommendation(BaseModel):
    """Improvement recommendation from benchmark analysis."""
    priority: int = Field(1, ge=1, le=5, description="Priority (1 = highest)")
    area: str = Field("", description="Area of improvement")
    recommendation: str = Field(..., description="Recommendation text")
    potential_improvement: str = Field("", description="Estimated improvement potential")
    benchmark_reference: str = Field("", description="Reference peer or best practice")

class BenchmarkReportInput(BaseModel):
    """Complete input model for BenchmarkComparisonReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    peer_group: Optional[PeerGroupDefinition] = Field(
        None, description="Peer group definition"
    )
    normalisation_methodology: str = Field(
        "", description="Normalisation methodology description"
    )
    ranking_table: List[PeerRankingEntry] = Field(
        default_factory=list, description="Ranking table entries"
    )
    gap_analysis: List[GapAnalysisItem] = Field(
        default_factory=list, description="Gap analysis results"
    )
    sector_distribution: List[SectorDistributionBucket] = Field(
        default_factory=list, description="Sector distribution histogram"
    )
    recommendations: List[BenchmarkRecommendation] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    radar_metrics: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Spider/radar chart data points with label, org_value, peer_avg, best_value",
    )

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class BenchmarkComparisonReport:
    """
    Benchmark comparison report template.

    Renders peer group benchmarking results with ranking tables,
    gap analysis, sector distribution, spider chart data, and
    improvement recommendations. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = BenchmarkComparisonReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize BenchmarkComparisonReport."""
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
        """Render benchmark comparison as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render benchmark comparison as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render benchmark comparison as JSON dict."""
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
            self._md_peer_group(data),
            self._md_normalisation(data),
            self._md_ranking(data),
            self._md_gap_analysis(data),
            self._md_sector_distribution(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Benchmark Comparison Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_peer_group(self, data: Dict[str, Any]) -> str:
        """Render Markdown peer group definition."""
        pg = data.get("peer_group")
        if not pg:
            return "## 1. Peer Group Definition\n\nNo peer group defined."
        lines = [
            "## 1. Peer Group Definition",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Group Name | {pg.get('peer_group_name', '')} |",
            f"| Sector | {pg.get('sector', '')} |",
            f"| Sub-Sector | {pg.get('sub_sector', '-')} |",
            f"| Region | {pg.get('region', '')} |",
            f"| Size Range | {pg.get('size_range', '')} |",
            f"| Peer Count | {pg.get('peer_count', 0)} |",
            f"| Data Source | {pg.get('data_source', '')} |",
            f"| Data Year | {pg.get('data_year', '-')} |",
        ]
        return "\n".join(lines)

    def _md_normalisation(self, data: Dict[str, Any]) -> str:
        """Render Markdown normalisation methodology."""
        method = self._get_val(data, "normalisation_methodology", "")
        if not method:
            return "## 2. Normalisation Methodology\n\nStandard intensity normalisation applied."
        return f"## 2. Normalisation Methodology\n\n{method}"

    def _md_ranking(self, data: Dict[str, Any]) -> str:
        """Render Markdown ranking table."""
        entries = data.get("ranking_table", [])
        if not entries:
            return "## 3. Ranking Table\n\nNo ranking data available."
        lines = [
            "## 3. Ranking Table",
            "",
            "| Rank | Entity | Intensity | Unit | Percentile |",
            "|------|--------|-----------|------|------------|",
        ]
        for e in entries:
            rank = e.get("rank", 0)
            name = e.get("entity_name", "")
            is_org = e.get("is_org", False)
            intensity = e.get("intensity_value", 0)
            unit = e.get("intensity_unit", "")
            pctile = e.get("percentile")
            pctile_str = f"P{pctile:.0f}" if pctile is not None else "-"
            marker = " **[YOU]**" if is_org else ""
            lines.append(
                f"| {rank} | {name}{marker} | {intensity:,.4f} | "
                f"{unit} | {pctile_str} |"
            )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown gap analysis table."""
        gaps = data.get("gap_analysis", [])
        if not gaps:
            return "## 4. Gap Analysis\n\nNo gap analysis available."
        lines = [
            "## 4. Gap Analysis",
            "",
            "| Metric | Org Value | Peer Avg | Gap to Avg | Best-in-Class | Gap to Best |",
            "|--------|-----------|----------|------------|---------------|-------------|",
        ]
        for g in gaps:
            name = g.get("metric_name", "")
            org = g.get("org_value", 0)
            avg = g.get("peer_average", 0)
            gap_avg = g.get("gap_to_average_pct", 0)
            best = g.get("best_in_class", 0)
            gap_best = g.get("gap_to_best_pct", 0)
            lines.append(
                f"| {name} | {org:,.4f} | {avg:,.4f} | "
                f"{gap_avg:+.1f}% | {best:,.4f} | {gap_best:+.1f}% |"
            )
        return "\n".join(lines)

    def _md_sector_distribution(self, data: Dict[str, Any]) -> str:
        """Render Markdown sector distribution histogram."""
        buckets = data.get("sector_distribution", [])
        if not buckets:
            return ""
        lines = [
            "## 5. Sector Distribution",
            "",
            "| Range | Count | Your Position |",
            "|-------|-------|---------------|",
        ]
        for b in buckets:
            low = b.get("range_low", 0)
            high = b.get("range_high", 0)
            count = b.get("count", 0)
            includes = b.get("includes_org", False)
            marker = "<-- You are here" if includes else ""
            bar = "#" * min(count, 40)
            lines.append(
                f"| {low:,.2f} - {high:,.2f} | {count} {bar} | {marker} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render Markdown improvement recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        lines = ["## 6. Improvement Recommendations", ""]
        for r in recs:
            priority = r.get("priority", 1)
            area = r.get("area", "")
            text = r.get("recommendation", "")
            potential = r.get("potential_improvement", "")
            ref = r.get("benchmark_reference", "")
            lines.append(f"**P{priority} - {area}:** {text}")
            details = []
            if potential:
                details.append(f"Potential: {potential}")
            if ref:
                details.append(f"Ref: {ref}")
            if details:
                lines.append(f"   - {' | '.join(details)}")
            lines.append("")
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
            self._html_peer_group(data),
            self._html_normalisation(data),
            self._html_ranking(data),
            self._html_gap_analysis(data),
            self._html_sector_distribution(data),
            self._html_recommendations(data),
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
            f"<title>Benchmark Comparison - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #2a9d8f;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            "tr.org-row{background:#e8f5e9;font-weight:600;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".gap-positive{color:#e76f51;}\n"
            ".gap-negative{color:#2a9d8f;}\n"
            ".dist-bar{display:inline-block;background:#2a9d8f;height:16px;"
            "border-radius:3px;vertical-align:middle;}\n"
            ".dist-org{background:#e76f51;}\n"
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
            f"<h1>Benchmark Comparison Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period}</p>\n<hr>\n</div>"
        )

    def _html_peer_group(self, data: Dict[str, Any]) -> str:
        """Render HTML peer group definition."""
        pg = data.get("peer_group")
        if not pg:
            return ""
        rows = ""
        for key in ["peer_group_name", "sector", "sub_sector", "region",
                     "size_range", "peer_count", "data_source", "data_year"]:
            label = key.replace("_", " ").title()
            value = pg.get(key, "-")
            rows += f"<tr><td>{label}</td><td>{value}</td></tr>\n"
        return (
            '<div class="section">\n<h2>1. Peer Group Definition</h2>\n'
            "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_normalisation(self, data: Dict[str, Any]) -> str:
        """Render HTML normalisation methodology."""
        method = self._get_val(data, "normalisation_methodology", "")
        if not method:
            return ""
        return (
            '<div class="section">\n<h2>2. Normalisation Methodology</h2>\n'
            f"<p>{method}</p>\n</div>"
        )

    def _html_ranking(self, data: Dict[str, Any]) -> str:
        """Render HTML ranking table with org row highlighted."""
        entries = data.get("ranking_table", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            rank = e.get("rank", 0)
            name = e.get("entity_name", "")
            is_org = e.get("is_org", False)
            intensity = e.get("intensity_value", 0)
            unit = e.get("intensity_unit", "")
            pctile = e.get("percentile")
            pctile_str = f"P{pctile:.0f}" if pctile is not None else "-"
            row_class = ' class="org-row"' if is_org else ""
            marker = " [YOU]" if is_org else ""
            rows += (
                f"<tr{row_class}><td>{rank}</td><td>{name}{marker}</td>"
                f"<td>{intensity:,.4f}</td><td>{unit}</td>"
                f"<td>{pctile_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Ranking Table</h2>\n'
            "<table><thead><tr><th>Rank</th><th>Entity</th>"
            "<th>Intensity</th><th>Unit</th><th>Percentile</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML gap analysis table."""
        gaps = data.get("gap_analysis", [])
        if not gaps:
            return ""
        rows = ""
        for g in gaps:
            name = g.get("metric_name", "")
            org = g.get("org_value", 0)
            avg = g.get("peer_average", 0)
            gap_avg_pct = g.get("gap_to_average_pct", 0)
            best = g.get("best_in_class", 0)
            gap_best_pct = g.get("gap_to_best_pct", 0)
            avg_css = "gap-positive" if gap_avg_pct > 0 else "gap-negative"
            best_css = "gap-positive" if gap_best_pct > 0 else "gap-negative"
            rows += (
                f"<tr><td>{name}</td><td>{org:,.4f}</td><td>{avg:,.4f}</td>"
                f'<td class="{avg_css}">{gap_avg_pct:+.1f}%</td>'
                f"<td>{best:,.4f}</td>"
                f'<td class="{best_css}">{gap_best_pct:+.1f}%</td></tr>\n'
            )
        return (
            '<div class="section">\n<h2>4. Gap Analysis</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Org</th><th>Peer Avg</th>"
            "<th>Gap to Avg</th><th>Best</th><th>Gap to Best</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sector_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML sector distribution with bar chart."""
        buckets = data.get("sector_distribution", [])
        if not buckets:
            return ""
        max_count = max((b.get("count", 0) for b in buckets), default=1) or 1
        rows = ""
        for b in buckets:
            low = b.get("range_low", 0)
            high = b.get("range_high", 0)
            count = b.get("count", 0)
            includes = b.get("includes_org", False)
            bar_width = int((count / max_count) * 200)
            bar_class = "dist-bar dist-org" if includes else "dist-bar"
            marker = " (Your position)" if includes else ""
            rows += (
                f"<tr><td>{low:,.2f} - {high:,.2f}</td><td>{count}</td>"
                f'<td><span class="{bar_class}" style="width:{bar_width}px;">'
                f"</span>{marker}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>5. Sector Distribution</h2>\n'
            "<table><thead><tr><th>Range</th><th>Count</th>"
            "<th>Distribution</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML improvement recommendations."""
        recs = data.get("recommendations", [])
        if not recs:
            return ""
        items = ""
        for r in recs:
            priority = r.get("priority", 1)
            area = r.get("area", "")
            text = r.get("recommendation", "")
            potential = r.get("potential_improvement", "")
            items += (
                f"<li><strong>P{priority} - {area}:</strong> {text}"
            )
            if potential:
                items += f" <em>(Potential: {potential})</em>"
            items += "</li>\n"
        return (
            '<div class="section">\n<h2>6. Improvement Recommendations</h2>\n'
            f"<ol>{items}</ol>\n</div>"
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
        """Render benchmark comparison as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "benchmark_comparison",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "peer_group": data.get("peer_group"),
            "normalisation_methodology": self._get_val(data, "normalisation_methodology", ""),
            "ranking_table": data.get("ranking_table", []),
            "gap_analysis": data.get("gap_analysis", []),
            "sector_distribution": data.get("sector_distribution", []),
            "recommendations": data.get("recommendations", []),
            "chart_data": {
                "radar": self._build_radar_chart(data),
                "distribution_histogram": self._build_histogram_data(data),
            },
        }

    def _build_radar_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build spider/radar chart data for multi-dimensional comparison."""
        metrics = data.get("radar_metrics", [])
        if not metrics:
            gaps = data.get("gap_analysis", [])
            if gaps:
                return {
                    "labels": [g.get("metric_name", "") for g in gaps],
                    "org_values": [g.get("org_value", 0) for g in gaps],
                    "peer_avg_values": [g.get("peer_average", 0) for g in gaps],
                    "best_values": [g.get("best_in_class", 0) for g in gaps],
                }
            return {}
        return {
            "labels": [m.get("label", "") for m in metrics],
            "org_values": [m.get("org_value", 0) for m in metrics],
            "peer_avg_values": [m.get("peer_avg", 0) for m in metrics],
            "best_values": [m.get("best_value", 0) for m in metrics],
        }

    def _build_histogram_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build histogram data for sector distribution chart."""
        buckets = data.get("sector_distribution", [])
        if not buckets:
            return {}
        return {
            "bin_edges": [b.get("range_low", 0) for b in buckets] + [
                buckets[-1].get("range_high", 0) if buckets else 0
            ],
            "counts": [b.get("count", 0) for b in buckets],
            "org_bucket_index": next(
                (i for i, b in enumerate(buckets) if b.get("includes_org")),
                None,
            ),
        }
