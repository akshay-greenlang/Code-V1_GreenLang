# -*- coding: utf-8 -*-
"""
PeerComparisonReport - Peer Comparison Report for PACK-047.

Generates a detailed peer comparison report with league tables (sortable
by any metric), distribution charts (histogram and box plot data), gap
analysis tables (gap to average, median, best-in-class, and target),
peer group composition summary, normalisation methodology disclosure,
and multi-format export.

Regulatory References:
    - GHG Protocol Corporate Standard (Chapter 8: Quality Management)
    - TCFD Metrics and Targets: Peer comparison context
    - EU Benchmark Regulation (BMR) 2019/2089
    - SBTi Corporate Net-Zero Standard: Peer benchmarking context

Sections:
    1. Peer Group Composition
    2. League Table
    3. Distribution Analysis (histogram, box plot)
    4. Gap Analysis Table
    5. Normalisation Methodology
    6. Provenance Footer

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

class GapType(str, Enum):
    """Gap analysis comparison target."""
    TO_AVERAGE = "to_average"
    TO_MEDIAN = "to_median"
    TO_BEST = "to_best_in_class"
    TO_TARGET = "to_target"

class QuartilePosition(str, Enum):
    """Quartile position classification."""
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class PeerGroupComposition(BaseModel):
    """Peer group composition summary."""
    peer_group_name: str = Field(..., description="Peer group name")
    sector: str = Field("", description="Industry sector / NACE code")
    sub_sector: str = Field("", description="Sub-sector")
    region: str = Field("", description="Geographic region")
    size_range: str = Field("", description="Revenue or employee size range")
    peer_count: int = Field(0, ge=0, description="Number of peers in group")
    data_source: str = Field("", description="Source of benchmark data")
    data_year: Optional[int] = Field(None, description="Year of benchmark data")
    selection_criteria: str = Field("", description="Selection criteria description")

class LeagueTableEntry(BaseModel):
    """Single entry in the league table."""
    rank: int = Field(0, ge=0, description="Rank (1=best)")
    entity_name: str = Field(..., description="Company / entity name")
    is_org: bool = Field(False, description="Whether this is the reporting org")
    total_emissions_tco2e: Optional[float] = Field(None, description="Total emissions tCO2e")
    intensity_value: Optional[float] = Field(None, description="Intensity metric value")
    intensity_unit: str = Field("", description="Intensity unit")
    yoy_change_pct: Optional[float] = Field(None, description="YoY change %")
    percentile: Optional[float] = Field(None, ge=0, le=100, description="Percentile rank")
    quartile: QuartilePosition = Field(QuartilePosition.Q2, description="Quartile position")

class DistributionBin(BaseModel):
    """Single histogram bin."""
    bin_start: float = Field(0.0, description="Bin lower bound")
    bin_end: float = Field(0.0, description="Bin upper bound")
    count: int = Field(0, ge=0, description="Number of entities in bin")
    org_in_bin: bool = Field(False, description="Whether org falls in this bin")

class BoxPlotData(BaseModel):
    """Box plot statistics."""
    metric_name: str = Field("", description="Metric name")
    minimum: float = Field(0.0, description="Minimum value")
    q1: float = Field(0.0, description="First quartile")
    median: float = Field(0.0, description="Median value")
    q3: float = Field(0.0, description="Third quartile")
    maximum: float = Field(0.0, description="Maximum value")
    org_value: Optional[float] = Field(None, description="Organisation value")
    outliers: List[float] = Field(default_factory=list, description="Outlier values")

class GapAnalysisRow(BaseModel):
    """Gap analysis for a single metric."""
    metric_name: str = Field(..., description="Metric name")
    org_value: float = Field(0.0, description="Organisation current value")
    peer_average: Optional[float] = Field(None, description="Peer average")
    peer_median: Optional[float] = Field(None, description="Peer median")
    best_in_class: Optional[float] = Field(None, description="Best-in-class value")
    target_value: Optional[float] = Field(None, description="Target value")
    gap_to_average: Optional[float] = Field(None, description="Gap to average (%)")
    gap_to_median: Optional[float] = Field(None, description="Gap to median (%)")
    gap_to_best: Optional[float] = Field(None, description="Gap to best-in-class (%)")
    gap_to_target: Optional[float] = Field(None, description="Gap to target (%)")
    unit: str = Field("", description="Metric unit")

class NormalisationMethodology(BaseModel):
    """Normalisation methodology disclosure."""
    approach: str = Field("", description="Normalisation approach description")
    denominators_used: List[str] = Field(default_factory=list, description="Denominator metrics")
    adjustments: List[str] = Field(default_factory=list, description="Adjustments applied")
    limitations: List[str] = Field(default_factory=list, description="Known limitations")
    data_quality_notes: str = Field("", description="Data quality notes")

class PeerComparisonInput(BaseModel):
    """Complete input model for PeerComparisonReport."""
    company_name: str = Field("Organization", description="Company name")
    reporting_period: str = Field("", description="Reporting period")
    report_date: Optional[str] = Field(None, description="Report date (ISO)")
    peer_group: Optional[PeerGroupComposition] = Field(
        None, description="Peer group composition"
    )
    league_table: List[LeagueTableEntry] = Field(
        default_factory=list, description="League table entries"
    )
    histogram_data: List[DistributionBin] = Field(
        default_factory=list, description="Histogram bin data"
    )
    box_plot_data: List[BoxPlotData] = Field(
        default_factory=list, description="Box plot statistics"
    )
    gap_analysis: List[GapAnalysisRow] = Field(
        default_factory=list, description="Gap analysis rows"
    )
    normalisation: Optional[NormalisationMethodology] = Field(
        None, description="Normalisation methodology"
    )

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class PeerComparisonReport:
    """
    Peer comparison report template for GHG emissions benchmarking.

    Renders detailed peer comparison with league tables, distribution
    charts (histogram and box plot data), gap analysis tables, peer group
    composition, and normalisation methodology disclosure. All outputs
    include SHA-256 provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.
        processing_time_ms: Duration of last render in milliseconds.

    Example:
        >>> template = PeerComparisonReport()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize PeerComparisonReport."""
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
        """Render peer comparison as Markdown."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_md(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render peer comparison as HTML."""
        start = time.monotonic()
        self.generated_at = utcnow()
        result = self._render_html(data)
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return result

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render peer comparison as JSON dict."""
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
            self._md_peer_group(data),
            self._md_league_table(data),
            self._md_distribution(data),
            self._md_gap_analysis(data),
            self._md_normalisation(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        period = self._get_val(data, "reporting_period", "")
        return (
            f"# Peer Comparison Report - {company}\n\n"
            f"**Reporting Period:** {period} | "
            f"**Report Date:** {utcnow().strftime('%Y-%m-%d')}\n\n"
            "---"
        )

    def _md_peer_group(self, data: Dict[str, Any]) -> str:
        """Render Markdown peer group composition."""
        pg = data.get("peer_group")
        if not pg:
            return ""
        lines = [
            "## 1. Peer Group Composition",
            "",
            "| Attribute | Value |",
            "|-----------|-------|",
            f"| Peer Group | {pg.get('peer_group_name', '')} |",
            f"| Sector | {pg.get('sector', '')} |",
            f"| Sub-Sector | {pg.get('sub_sector', '')} |",
            f"| Region | {pg.get('region', '')} |",
            f"| Size Range | {pg.get('size_range', '')} |",
            f"| Peer Count | {pg.get('peer_count', 0)} |",
            f"| Data Source | {pg.get('data_source', '')} |",
        ]
        data_year = pg.get("data_year")
        if data_year:
            lines.append(f"| Data Year | {data_year} |")
        criteria = pg.get("selection_criteria", "")
        if criteria:
            lines.append("")
            lines.append(f"**Selection Criteria:** {criteria}")
        return "\n".join(lines)

    def _md_league_table(self, data: Dict[str, Any]) -> str:
        """Render Markdown league table."""
        entries = data.get("league_table", [])
        if not entries:
            return "## 2. League Table\n\nNo league table data available."
        lines = [
            "## 2. League Table",
            "",
            "| Rank | Entity | Emissions (tCO2e) | Intensity | Unit | YoY | Percentile | Quartile |",
            "|------|--------|-------------------|-----------|------|-----|------------|----------|",
        ]
        for e in entries:
            rank = e.get("rank", 0)
            name = e.get("entity_name", "")
            is_org = e.get("is_org", False)
            marker = " **[ORG]**" if is_org else ""
            emissions = e.get("total_emissions_tco2e")
            em_str = f"{emissions:,.0f}" if emissions is not None else "-"
            intensity = e.get("intensity_value")
            int_str = f"{intensity:,.4f}" if intensity is not None else "-"
            unit = e.get("intensity_unit", "")
            yoy = e.get("yoy_change_pct")
            yoy_str = f"{yoy:+.1f}%" if yoy is not None else "-"
            pctile = e.get("percentile")
            pctile_str = f"P{pctile:.0f}" if pctile is not None else "-"
            quartile = e.get("quartile", "Q2")
            lines.append(
                f"| {rank} | {name}{marker} | {em_str} | {int_str} | "
                f"{unit} | {yoy_str} | {pctile_str} | {quartile} |"
            )
        return "\n".join(lines)

    def _md_distribution(self, data: Dict[str, Any]) -> str:
        """Render Markdown distribution analysis (histogram + box plot)."""
        histogram = data.get("histogram_data", [])
        box_plots = data.get("box_plot_data", [])
        if not histogram and not box_plots:
            return ""
        lines = ["## 3. Distribution Analysis", ""]
        # Histogram
        if histogram:
            lines.append("### Histogram (Emissions Distribution)")
            lines.append("")
            lines.append("| Bin Range | Count | Org in Bin |")
            lines.append("|-----------|-------|------------|")
            for b in histogram:
                org_marker = "***" if b.get("org_in_bin", False) else ""
                lines.append(
                    f"| {b.get('bin_start', 0):,.0f} - {b.get('bin_end', 0):,.0f} | "
                    f"{b.get('count', 0)} | {org_marker} |"
                )
            lines.append("")
        # Box plots
        if box_plots:
            lines.append("### Box Plot Statistics")
            lines.append("")
            lines.append("| Metric | Min | Q1 | Median | Q3 | Max | Org Value |")
            lines.append("|--------|-----|----|---------|----|-----|-----------|")
            for bp in box_plots:
                org = bp.get("org_value")
                org_str = f"{org:,.2f}" if org is not None else "-"
                lines.append(
                    f"| {bp.get('metric_name', '')} | {bp.get('minimum', 0):,.2f} | "
                    f"{bp.get('q1', 0):,.2f} | {bp.get('median', 0):,.2f} | "
                    f"{bp.get('q3', 0):,.2f} | {bp.get('maximum', 0):,.2f} | {org_str} |"
                )
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown gap analysis table."""
        gaps = data.get("gap_analysis", [])
        if not gaps:
            return ""
        lines = [
            "## 4. Gap Analysis",
            "",
            "| Metric | Org Value | Average | Median | Best | Target | "
            "Gap to Avg | Gap to Med | Gap to Best | Gap to Target |",
            "|--------|-----------|---------|--------|------|--------|"
            "-----------|------------|-------------|---------------|",
        ]
        for g in gaps:
            name = g.get("metric_name", "")
            org = g.get("org_value", 0)
            avg = g.get("peer_average")
            med = g.get("peer_median")
            best = g.get("best_in_class")
            target = g.get("target_value")
            avg_str = f"{avg:,.2f}" if avg is not None else "-"
            med_str = f"{med:,.2f}" if med is not None else "-"
            best_str = f"{best:,.2f}" if best is not None else "-"
            target_str = f"{target:,.2f}" if target is not None else "-"
            g_avg = g.get("gap_to_average")
            g_med = g.get("gap_to_median")
            g_best = g.get("gap_to_best")
            g_tgt = g.get("gap_to_target")
            g_avg_str = f"{g_avg:+.1f}%" if g_avg is not None else "-"
            g_med_str = f"{g_med:+.1f}%" if g_med is not None else "-"
            g_best_str = f"{g_best:+.1f}%" if g_best is not None else "-"
            g_tgt_str = f"{g_tgt:+.1f}%" if g_tgt is not None else "-"
            lines.append(
                f"| {name} | {org:,.2f} | {avg_str} | {med_str} | {best_str} | "
                f"{target_str} | {g_avg_str} | {g_med_str} | {g_best_str} | {g_tgt_str} |"
            )
        return "\n".join(lines)

    def _md_normalisation(self, data: Dict[str, Any]) -> str:
        """Render Markdown normalisation methodology."""
        norm = data.get("normalisation")
        if not norm:
            return ""
        lines = ["## 5. Normalisation Methodology", ""]
        approach = norm.get("approach", "")
        if approach:
            lines.append(f"**Approach:** {approach}")
            lines.append("")
        denominators = norm.get("denominators_used", [])
        if denominators:
            lines.append("**Denominators Used:**")
            for d in denominators:
                lines.append(f"- {d}")
            lines.append("")
        adjustments = norm.get("adjustments", [])
        if adjustments:
            lines.append("**Adjustments Applied:**")
            for a in adjustments:
                lines.append(f"- {a}")
            lines.append("")
        limitations = norm.get("limitations", [])
        if limitations:
            lines.append("**Limitations:**")
            for lim in limitations:
                lines.append(f"- {lim}")
            lines.append("")
        dq = norm.get("data_quality_notes", "")
        if dq:
            lines.append(f"**Data Quality:** {dq}")
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
            self._html_peer_group(data),
            self._html_league_table(data),
            self._html_distribution(data),
            self._html_gap_analysis(data),
            self._html_normalisation(data),
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
            f"<title>Peer Comparison - {company}</title>\n"
            "<style>\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:2rem auto;color:#1a1a2e;"
            "max-width:1200px;line-height:1.6;}\n"
            "h1{color:#0d1b2a;border-bottom:3px solid #264653;padding-bottom:0.5rem;}\n"
            "h2{color:#1b263b;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:0.3rem;}\n"
            "h3{color:#264653;margin-top:1.5rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.85rem;}\n"
            "th,td{border:1px solid #ccc;padding:0.4rem 0.6rem;text-align:left;}\n"
            "th{background:#f0f4f8;font-weight:600;}\n"
            "tr:nth-child(even){background:#fafbfc;}\n"
            "tr.org-row{background:#e8f5e9;font-weight:600;}\n"
            ".section{margin-bottom:2rem;}\n"
            ".provenance{font-size:0.8rem;color:#888;font-family:monospace;}\n"
            ".gap-positive{color:#e76f51;}\n"
            ".gap-negative{color:#2a9d8f;}\n"
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
            f"<h1>Peer Comparison Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Period:</strong> {period} | "
            f"<strong>Report Date:</strong> {utcnow().strftime('%Y-%m-%d')}</p>\n"
            "<hr>\n</div>"
        )

    def _html_peer_group(self, data: Dict[str, Any]) -> str:
        """Render HTML peer group composition."""
        pg = data.get("peer_group")
        if not pg:
            return ""
        rows = (
            f"<tr><td>Peer Group</td><td>{pg.get('peer_group_name', '')}</td></tr>\n"
            f"<tr><td>Sector</td><td>{pg.get('sector', '')}</td></tr>\n"
            f"<tr><td>Sub-Sector</td><td>{pg.get('sub_sector', '')}</td></tr>\n"
            f"<tr><td>Region</td><td>{pg.get('region', '')}</td></tr>\n"
            f"<tr><td>Size Range</td><td>{pg.get('size_range', '')}</td></tr>\n"
            f"<tr><td>Peer Count</td><td>{pg.get('peer_count', 0)}</td></tr>\n"
            f"<tr><td>Data Source</td><td>{pg.get('data_source', '')}</td></tr>\n"
        )
        return (
            '<div class="section">\n<h2>1. Peer Group Composition</h2>\n'
            "<table><thead><tr><th>Attribute</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_league_table(self, data: Dict[str, Any]) -> str:
        """Render HTML league table."""
        entries = data.get("league_table", [])
        if not entries:
            return ""
        rows = ""
        for e in entries:
            rank = e.get("rank", 0)
            name = e.get("entity_name", "")
            is_org = e.get("is_org", False)
            row_cls = ' class="org-row"' if is_org else ""
            emissions = e.get("total_emissions_tco2e")
            em_str = f"{emissions:,.0f}" if emissions is not None else "-"
            intensity = e.get("intensity_value")
            int_str = f"{intensity:,.4f}" if intensity is not None else "-"
            unit = e.get("intensity_unit", "")
            pctile = e.get("percentile")
            pctile_str = f"P{pctile:.0f}" if pctile is not None else "-"
            quartile = e.get("quartile", "Q2")
            rows += (
                f"<tr{row_cls}><td>{rank}</td><td>{name}</td><td>{em_str}</td>"
                f"<td>{int_str}</td><td>{unit}</td>"
                f"<td>{pctile_str}</td><td>{quartile}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>2. League Table</h2>\n'
            "<table><thead><tr><th>Rank</th><th>Entity</th>"
            "<th>Emissions (tCO2e)</th><th>Intensity</th><th>Unit</th>"
            "<th>Percentile</th><th>Quartile</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_distribution(self, data: Dict[str, Any]) -> str:
        """Render HTML distribution analysis."""
        box_plots = data.get("box_plot_data", [])
        if not box_plots:
            return ""
        rows = ""
        for bp in box_plots:
            org = bp.get("org_value")
            org_str = f"{org:,.2f}" if org is not None else "-"
            rows += (
                f"<tr><td>{bp.get('metric_name', '')}</td>"
                f"<td>{bp.get('minimum', 0):,.2f}</td>"
                f"<td>{bp.get('q1', 0):,.2f}</td>"
                f"<td><strong>{bp.get('median', 0):,.2f}</strong></td>"
                f"<td>{bp.get('q3', 0):,.2f}</td>"
                f"<td>{bp.get('maximum', 0):,.2f}</td>"
                f"<td>{org_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>3. Distribution Analysis</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Min</th><th>Q1</th>"
            "<th>Median</th><th>Q3</th><th>Max</th><th>Org Value</th></tr></thead>\n"
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
            g_avg = g.get("gap_to_average")
            g_med = g.get("gap_to_median")
            g_best = g.get("gap_to_best")
            g_tgt = g.get("gap_to_target")

            def _gap_html(val: Optional[float]) -> str:
                if val is None:
                    return "-"
                css = "gap-positive" if val > 0 else "gap-negative"
                return f'<span class="{css}">{val:+.1f}%</span>'

            rows += (
                f"<tr><td>{name}</td><td>{org:,.2f}</td>"
                f"<td>{_gap_html(g_avg)}</td><td>{_gap_html(g_med)}</td>"
                f"<td>{_gap_html(g_best)}</td><td>{_gap_html(g_tgt)}</td></tr>\n"
            )
        return (
            '<div class="section">\n<h2>4. Gap Analysis</h2>\n'
            "<table><thead><tr><th>Metric</th><th>Org Value</th>"
            "<th>Gap to Avg</th><th>Gap to Median</th>"
            "<th>Gap to Best</th><th>Gap to Target</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_normalisation(self, data: Dict[str, Any]) -> str:
        """Render HTML normalisation methodology."""
        norm = data.get("normalisation")
        if not norm:
            return ""
        approach = norm.get("approach", "")
        content = ""
        if approach:
            content += f"<p><strong>Approach:</strong> {approach}</p>\n"
        denominators = norm.get("denominators_used", [])
        if denominators:
            content += "<p><strong>Denominators:</strong></p><ul>"
            for d in denominators:
                content += f"<li>{d}</li>"
            content += "</ul>\n"
        limitations = norm.get("limitations", [])
        if limitations:
            content += "<p><strong>Limitations:</strong></p><ul>"
            for lim in limitations:
                content += f"<li>{lim}</li>"
            content += "</ul>\n"
        return (
            '<div class="section">\n<h2>5. Normalisation Methodology</h2>\n'
            f"{content}</div>"
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
        """Render peer comparison as JSON-serializable dict."""
        provenance = self._compute_provenance(data)
        return {
            "template": "peer_comparison_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "provenance_hash": provenance,
            "processing_time_ms": self.processing_time_ms,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_period": self._get_val(data, "reporting_period", ""),
            "peer_group": data.get("peer_group"),
            "league_table": data.get("league_table", []),
            "histogram_data": data.get("histogram_data", []),
            "box_plot_data": data.get("box_plot_data", []),
            "gap_analysis": data.get("gap_analysis", []),
            "normalisation": data.get("normalisation"),
            "chart_data": {
                "histogram": self._build_histogram_chart(data),
                "box_plots": self._build_box_plot_chart(data),
                "league_bar": self._build_league_bar_chart(data),
            },
        }

    def _build_histogram_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build histogram chart data."""
        bins = data.get("histogram_data", [])
        if not bins:
            return {}
        return {
            "labels": [
                f"{b.get('bin_start', 0):,.0f}-{b.get('bin_end', 0):,.0f}"
                for b in bins
            ],
            "counts": [b.get("count", 0) for b in bins],
            "org_bin_index": next(
                (i for i, b in enumerate(bins) if b.get("org_in_bin")), None
            ),
        }

    def _build_box_plot_chart(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build box plot chart data."""
        return [
            {
                "metric": bp.get("metric_name", ""),
                "min": bp.get("minimum", 0),
                "q1": bp.get("q1", 0),
                "median": bp.get("median", 0),
                "q3": bp.get("q3", 0),
                "max": bp.get("maximum", 0),
                "org_value": bp.get("org_value"),
            }
            for bp in data.get("box_plot_data", [])
        ]

    def _build_league_bar_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build league table bar chart data."""
        entries = data.get("league_table", [])
        if not entries:
            return {}
        return {
            "entities": [e.get("entity_name", "") for e in entries],
            "values": [e.get("intensity_value", 0) for e in entries],
            "org_index": next(
                (i for i, e in enumerate(entries) if e.get("is_org")), None
            ),
        }
