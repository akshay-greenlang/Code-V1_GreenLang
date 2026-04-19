"""
Protected Area Report Template - PACK-007 EUDR Professional Pack

This module generates protected area analysis reports with WDPA/KBA overlay results,
buffer zone analysis, proximity scores, indigenous land flags, Ramsar/UNESCO proximity,
and risk amplification summaries for EUDR compliance.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from protected_area_report import ProtectedAreaReportTemplate, ReportData
    >>> data = ReportData(
    ...     operator_name="Tropical Commodities Ltd",
    ...     report_date="2026-03-15",
    ...     total_plots_analyzed=350
    ... )
    >>> template = ProtectedAreaReportTemplate()
    >>> report = template.render(data, format="markdown")
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ReportConfig(BaseModel):
    """Configuration for Protected Area Report generation."""

    include_buffer_analysis: bool = Field(
        default=True,
        description="Include buffer zone analysis"
    )
    include_indigenous_lands: bool = Field(
        default=True,
        description="Include indigenous territory analysis"
    )
    include_unesco_ramsar: bool = Field(
        default=True,
        description="Include UNESCO/Ramsar site proximity"
    )
    buffer_distances_km: List[float] = Field(
        default=[1.0, 5.0, 10.0],
        description="Buffer distances for analysis (km)"
    )


class ProtectedAreaOverlay(BaseModel):
    """Protected area overlay result for a plot."""

    plot_id: str = Field(..., description="Plot identifier")
    plot_name: str = Field(..., description="Plot name")
    overlaps_protected_area: bool = Field(..., description="True if plot overlaps PA")
    protected_area_name: Optional[str] = Field(None, description="PA name if overlapping")
    protected_area_type: Optional[str] = Field(
        None,
        description="PA type (e.g., National Park, Nature Reserve)"
    )
    iucn_category: Optional[str] = Field(None, description="IUCN category (I-VI)")
    overlap_area_hectares: float = Field(..., ge=0, description="Overlap area in hectares")
    overlap_percentage: float = Field(..., ge=0, le=100, description="Percentage of plot overlapping")
    designation_date: Optional[str] = Field(None, description="PA designation date")


class BufferZoneAnalysis(BaseModel):
    """Buffer zone proximity analysis."""

    plot_id: str = Field(..., description="Plot identifier")
    plot_name: str = Field(..., description="Plot name")
    nearest_pa_name: str = Field(..., description="Nearest protected area name")
    distance_km: float = Field(..., ge=0, description="Distance to nearest PA (km)")
    within_1km_buffer: bool = Field(..., description="Within 1km buffer")
    within_5km_buffer: bool = Field(..., description="Within 5km buffer")
    within_10km_buffer: bool = Field(..., description="Within 10km buffer")
    proximity_risk_score: float = Field(..., ge=0, le=100, description="Proximity risk score")


class IndigenousLandFlag(BaseModel):
    """Indigenous land proximity/overlap flag."""

    plot_id: str = Field(..., description="Plot identifier")
    plot_name: str = Field(..., description="Plot name")
    overlaps_indigenous_land: bool = Field(..., description="True if overlapping")
    indigenous_territory_name: Optional[str] = Field(None, description="Territory name")
    community_name: Optional[str] = Field(None, description="Indigenous community name")
    overlap_area_hectares: float = Field(..., ge=0, description="Overlap area")
    land_rights_status: str = Field(
        ...,
        description="FORMAL_TITLE, CUSTOMARY, CONTESTED, UNKNOWN"
    )
    fpic_required: bool = Field(..., description="Free Prior Informed Consent required")


class UNESCORamsarProximity(BaseModel):
    """UNESCO World Heritage or Ramsar site proximity."""

    plot_id: str = Field(..., description="Plot identifier")
    plot_name: str = Field(..., description="Plot name")
    site_type: str = Field(..., description="UNESCO or RAMSAR")
    site_name: str = Field(..., description="Site name")
    distance_km: float = Field(..., ge=0, description="Distance to site (km)")
    overlaps: bool = Field(..., description="True if plot overlaps site")
    designation_criteria: Optional[str] = Field(None, description="Designation criteria")


class RiskAmplification(BaseModel):
    """Risk amplification summary due to PA proximity."""

    category: str = Field(..., description="Risk category")
    baseline_risk_score: float = Field(..., ge=0, le=100, description="Baseline risk score")
    amplified_risk_score: float = Field(..., ge=0, le=100, description="Amplified risk score")
    amplification_factor: float = Field(..., ge=1, description="Amplification multiplier")
    reason: str = Field(..., description="Reason for amplification")


class ProximityStatistics(BaseModel):
    """Overall proximity statistics."""

    total_plots_analyzed: int = Field(..., ge=0, description="Total plots analyzed")
    plots_overlapping_pa: int = Field(..., ge=0, description="Plots overlapping PA")
    plots_within_1km: int = Field(..., ge=0, description="Plots within 1km buffer")
    plots_within_5km: int = Field(..., ge=0, description="Plots within 5km buffer")
    plots_within_10km: int = Field(..., ge=0, description="Plots within 10km buffer")
    plots_overlapping_indigenous: int = Field(..., ge=0, description="Plots on indigenous lands")
    plots_near_unesco_ramsar: int = Field(..., ge=0, description="Plots near UNESCO/Ramsar")
    avg_distance_to_pa_km: float = Field(..., ge=0, description="Average distance to nearest PA")


class ReportData(BaseModel):
    """Data model for Protected Area Report."""

    operator_name: str = Field(..., description="Operator name")
    report_date: str = Field(..., description="Report generation date (ISO format)")
    analysis_date: str = Field(..., description="Date of spatial analysis")
    total_plots_analyzed: int = Field(..., ge=0, description="Total plots analyzed")
    statistics: ProximityStatistics = Field(..., description="Overall statistics")
    overlays: List[ProtectedAreaOverlay] = Field(
        default_factory=list,
        description="Protected area overlays"
    )
    buffer_analysis: List[BufferZoneAnalysis] = Field(
        default_factory=list,
        description="Buffer zone analysis results"
    )
    indigenous_flags: List[IndigenousLandFlag] = Field(
        default_factory=list,
        description="Indigenous land flags"
    )
    unesco_ramsar_proximity: List[UNESCORamsarProximity] = Field(
        default_factory=list,
        description="UNESCO/Ramsar proximity"
    )
    risk_amplifications: List[RiskAmplification] = Field(
        default_factory=list,
        description="Risk amplification factors"
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources (e.g., WDPA, KBA, UNEP-WCMC)"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings from analysis"
    )

    @validator('report_date', 'analysis_date')
    def validate_dates(cls, v):
        """Validate dates are in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class ProtectedAreaReportTemplate:
    """
    Protected Area Report Template for EUDR Professional Pack.

    Generates comprehensive protected area analysis reports with WDPA/KBA overlays,
    buffer zone analysis, indigenous land flags, and UNESCO/Ramsar proximity.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = ProtectedAreaReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Protected Area Analysis" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Protected Area Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the protected area report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Protected Area Report for {data.operator_name} in {format} format"
        )

        if format == "markdown":
            content = self._render_markdown(data)
        elif format == "html":
            content = self._render_html(data)
        elif format == "json":
            content = self._render_json(data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Add provenance hash
        content_hash = self._calculate_hash(content)
        logger.info(f"Report generated with hash: {content_hash}")

        return content

    def _render_markdown(self, data: ReportData) -> str:
        """Render report in Markdown format."""
        sections = []

        # Header
        sections.append(f"# Protected Area Analysis Report")
        sections.append(f"")
        sections.append(f"**Operator:** {data.operator_name}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"**Analysis Date:** {data.analysis_date}")
        sections.append(f"**Plots Analyzed:** {data.total_plots_analyzed}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        stats = data.statistics
        sections.append(
            f"This protected area analysis report evaluates {stats.total_plots_analyzed} plots "
            f"against global protected area databases (WDPA, KBA) and indigenous land registries. "
            f"**{stats.plots_overlapping_pa}** plots ({stats.plots_overlapping_pa / stats.total_plots_analyzed * 100:.1f}%) "
            f"directly overlap with protected areas, and **{stats.plots_overlapping_indigenous}** plots "
            f"overlap with indigenous lands."
        )
        sections.append(f"")

        if data.data_sources:
            sections.append(f"**Data Sources:** {', '.join(data.data_sources)}")
            sections.append(f"")

        # Overall Statistics
        sections.append(f"## Overall Statistics")
        sections.append(f"")
        sections.append(f"| Metric | Value | Percentage |")
        sections.append(f"|--------|-------|------------|")
        sections.append(
            f"| Total Plots Analyzed | {stats.total_plots_analyzed:,} | 100% |"
        )
        sections.append(
            f"| Plots Overlapping PA | {stats.plots_overlapping_pa:,} | "
            f"{stats.plots_overlapping_pa / stats.total_plots_analyzed * 100:.1f}% |"
        )
        sections.append(
            f"| Plots Within 1km of PA | {stats.plots_within_1km:,} | "
            f"{stats.plots_within_1km / stats.total_plots_analyzed * 100:.1f}% |"
        )
        sections.append(
            f"| Plots Within 5km of PA | {stats.plots_within_5km:,} | "
            f"{stats.plots_within_5km / stats.total_plots_analyzed * 100:.1f}% |"
        )
        sections.append(
            f"| Plots Within 10km of PA | {stats.plots_within_10km:,} | "
            f"{stats.plots_within_10km / stats.total_plots_analyzed * 100:.1f}% |"
        )
        sections.append(
            f"| Plots on Indigenous Lands | {stats.plots_overlapping_indigenous:,} | "
            f"{stats.plots_overlapping_indigenous / stats.total_plots_analyzed * 100:.1f}% |"
        )
        sections.append(
            f"| Plots Near UNESCO/Ramsar | {stats.plots_near_unesco_ramsar:,} | "
            f"{stats.plots_near_unesco_ramsar / stats.total_plots_analyzed * 100:.1f}% |"
        )
        sections.append(f"")
        sections.append(f"**Average Distance to Nearest PA:** {stats.avg_distance_to_pa_km:.2f} km")
        sections.append(f"")

        # Protected Area Overlays
        if data.overlays:
            sections.append(f"## Protected Area Overlays")
            sections.append(f"")

            # Only plots with overlaps
            overlapping_plots = [o for o in data.overlays if o.overlaps_protected_area]

            if overlapping_plots:
                sections.append(
                    f"**{len(overlapping_plots)}** plots directly overlap with protected areas:"
                )
                sections.append(f"")

                sections.append(
                    f"| Plot ID | Plot Name | PA Name | PA Type | IUCN Cat | Overlap (ha) | Overlap % |"
                )
                sections.append(
                    f"|---------|-----------|---------|---------|----------|--------------|-----------|"
                )

                for overlay in sorted(overlapping_plots, key=lambda x: x.overlap_percentage, reverse=True):
                    sections.append(
                        f"| {overlay.plot_id} | {overlay.plot_name} | {overlay.protected_area_name} | "
                        f"{overlay.protected_area_type} | {overlay.iucn_category or 'N/A'} | "
                        f"{overlay.overlap_area_hectares:.2f} | {overlay.overlap_percentage:.1f}% |"
                    )
                sections.append(f"")

                # High overlap plots
                high_overlap = [o for o in overlapping_plots if o.overlap_percentage > 50]
                if high_overlap:
                    sections.append(f"### Critical Overlaps (>50%)")
                    sections.append(f"")
                    sections.append(
                        f"**{len(high_overlap)}** plots have >50% overlap with protected areas - "
                        f"these require immediate investigation:"
                    )
                    sections.append(f"")
                    for overlay in high_overlap[:10]:
                        sections.append(
                            f"- **{overlay.plot_name}** ({overlay.plot_id}): {overlay.overlap_percentage:.1f}% "
                            f"overlap with {overlay.protected_area_name}"
                        )
                    sections.append(f"")
            else:
                sections.append(f"No direct protected area overlaps detected.")
                sections.append(f"")

        # Buffer Zone Analysis
        if self.config.include_buffer_analysis and data.buffer_analysis:
            sections.append(f"## Buffer Zone Proximity Analysis")
            sections.append(f"")
            sections.append(
                f"Proximity analysis for plots near but not overlapping protected areas:"
            )
            sections.append(f"")

            sections.append(
                f"| Plot ID | Plot Name | Nearest PA | Distance (km) | 1km | 5km | 10km | Risk Score |"
            )
            sections.append(
                f"|---------|-----------|------------|---------------|-----|-----|------|------------|"
            )

            for buffer in sorted(data.buffer_analysis, key=lambda x: x.distance_km)[:20]:
                within_1km = "✓" if buffer.within_1km_buffer else "✗"
                within_5km = "✓" if buffer.within_5km_buffer else "✗"
                within_10km = "✓" if buffer.within_10km_buffer else "✗"
                sections.append(
                    f"| {buffer.plot_id} | {buffer.plot_name} | {buffer.nearest_pa_name} | "
                    f"{buffer.distance_km:.2f} | {within_1km} | {within_5km} | {within_10km} | "
                    f"{buffer.proximity_risk_score:.1f} |"
                )
            sections.append(f"")

        # Indigenous Land Flags
        if self.config.include_indigenous_lands and data.indigenous_flags:
            sections.append(f"## Indigenous Land Analysis")
            sections.append(f"")

            overlapping_indigenous = [f for f in data.indigenous_flags if f.overlaps_indigenous_land]

            if overlapping_indigenous:
                sections.append(
                    f"**{len(overlapping_indigenous)}** plots overlap with indigenous territories:"
                )
                sections.append(f"")

                sections.append(
                    f"| Plot ID | Plot Name | Territory | Community | Overlap (ha) | Rights Status | FPIC Required |"
                )
                sections.append(
                    f"|---------|-----------|-----------|-----------|--------------|---------------|---------------|"
                )

                for flag in overlapping_indigenous:
                    fpic = "YES" if flag.fpic_required else "NO"
                    sections.append(
                        f"| {flag.plot_id} | {flag.plot_name} | {flag.indigenous_territory_name or 'N/A'} | "
                        f"{flag.community_name or 'N/A'} | {flag.overlap_area_hectares:.2f} | "
                        f"{flag.land_rights_status} | {fpic} |"
                    )
                sections.append(f"")

                # FPIC required
                fpic_required = [f for f in overlapping_indigenous if f.fpic_required]
                if fpic_required:
                    sections.append(f"### Free Prior Informed Consent (FPIC) Required")
                    sections.append(f"")
                    sections.append(
                        f"**{len(fpic_required)}** plots require FPIC documentation:"
                    )
                    sections.append(f"")
                    for flag in fpic_required:
                        sections.append(
                            f"- **{flag.plot_name}** ({flag.plot_id}): {flag.community_name}"
                        )
                    sections.append(f"")
            else:
                sections.append(f"No indigenous land overlaps detected.")
                sections.append(f"")

        # UNESCO/Ramsar Proximity
        if self.config.include_unesco_ramsar and data.unesco_ramsar_proximity:
            sections.append(f"## UNESCO World Heritage & Ramsar Site Proximity")
            sections.append(f"")
            sections.append(
                f"Proximity to UNESCO World Heritage Sites and Ramsar Wetlands:"
            )
            sections.append(f"")

            sections.append(
                f"| Plot ID | Plot Name | Site Type | Site Name | Distance (km) | Overlaps |"
            )
            sections.append(
                f"|---------|-----------|-----------|-----------|---------------|----------|"
            )

            for prox in sorted(data.unesco_ramsar_proximity, key=lambda x: x.distance_km)[:20]:
                overlaps = "YES" if prox.overlaps else "NO"
                sections.append(
                    f"| {prox.plot_id} | {prox.plot_name} | {prox.site_type} | "
                    f"{prox.site_name} | {prox.distance_km:.2f} | {overlaps} |"
                )
            sections.append(f"")

        # Risk Amplification
        if data.risk_amplifications:
            sections.append(f"## Risk Amplification Analysis")
            sections.append(f"")
            sections.append(
                f"Protected area proximity amplifies risk scores as follows:"
            )
            sections.append(f"")

            sections.append(
                f"| Category | Baseline Risk | Amplified Risk | Factor | Reason |"
            )
            sections.append(
                f"|----------|---------------|----------------|--------|--------|"
            )

            for amp in data.risk_amplifications:
                sections.append(
                    f"| {amp.category} | {amp.baseline_risk_score:.1f} | "
                    f"{amp.amplified_risk_score:.1f} | {amp.amplification_factor:.2f}x | "
                    f"{amp.reason} |"
                )
            sections.append(f"")

        # Key Findings
        if data.key_findings:
            sections.append(f"## Key Findings")
            sections.append(f"")
            for idx, finding in enumerate(data.key_findings, 1):
                sections.append(f"{idx}. {finding}")
            sections.append(f"")

        # Recommendations
        sections.append(f"## Recommendations")
        sections.append(f"")

        if stats.plots_overlapping_pa > 0:
            sections.append(
                f"1. **Investigate Protected Area Overlaps:** {stats.plots_overlapping_pa} plots "
                f"overlap with protected areas - verify legal status and permissions"
            )

        if stats.plots_overlapping_indigenous > 0:
            sections.append(
                f"2. **Indigenous Rights Due Diligence:** {stats.plots_overlapping_indigenous} plots "
                f"on indigenous lands require FPIC documentation and community engagement"
            )

        if stats.plots_within_1km > 0:
            sections.append(
                f"3. **Buffer Zone Monitoring:** {stats.plots_within_1km} plots within 1km of PAs "
                f"require enhanced monitoring for encroachment"
            )

        sections.append(
            f"4. **Data Refresh:** Update protected area data quarterly to capture new designations"
        )
        sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Report generated on {data.report_date} using GreenLang EUDR Professional Pack*"
        )
        sections.append(f"")
        sections.append(
            f"**Data Sources:** Analysis based on WDPA (World Database on Protected Areas), "
            f"Key Biodiversity Areas (KBA), and indigenous land registries."
        )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render report in HTML format."""
        stats = data.statistics

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Protected Area Analysis Report - {data.operator_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #27ae60; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #27ae60; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .critical {{ background-color: #ffebee; color: #c0392b; font-weight: bold; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Protected Area Analysis Report</h1>

    <div class="summary">
        <p><strong>Operator:</strong> {data.operator_name}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
        <p><strong>Analysis Date:</strong> {data.analysis_date}</p>
        <p><strong>Plots Analyzed:</strong> {data.total_plots_analyzed}</p>
    </div>

    <h2>Overall Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Percentage</th></tr>
        <tr><td>Total Plots Analyzed</td><td>{stats.total_plots_analyzed:,}</td><td>100%</td></tr>
        <tr><td>Plots Overlapping PA</td><td>{stats.plots_overlapping_pa:,}</td><td>{stats.plots_overlapping_pa / stats.total_plots_analyzed * 100:.1f}%</td></tr>
        <tr><td>Plots Within 1km of PA</td><td>{stats.plots_within_1km:,}</td><td>{stats.plots_within_1km / stats.total_plots_analyzed * 100:.1f}%</td></tr>
        <tr><td>Plots on Indigenous Lands</td><td>{stats.plots_overlapping_indigenous:,}</td><td>{stats.plots_overlapping_indigenous / stats.total_plots_analyzed * 100:.1f}%</td></tr>
    </table>
"""

        if data.overlays:
            overlapping_plots = [o for o in data.overlays if o.overlaps_protected_area]
            if overlapping_plots:
                html += f"""
    <h2>Protected Area Overlays</h2>
    <table>
        <tr><th>Plot ID</th><th>Plot Name</th><th>PA Name</th><th>PA Type</th><th>IUCN</th><th>Overlap (ha)</th><th>Overlap %</th></tr>
"""
                for overlay in sorted(overlapping_plots, key=lambda x: x.overlap_percentage, reverse=True)[:30]:
                    row_class = ' class="critical"' if overlay.overlap_percentage > 50 else ''
                    html += f"""        <tr{row_class}>
            <td>{overlay.plot_id}</td>
            <td>{overlay.plot_name}</td>
            <td>{overlay.protected_area_name}</td>
            <td>{overlay.protected_area_type}</td>
            <td>{overlay.iucn_category or 'N/A'}</td>
            <td>{overlay.overlap_area_hectares:.2f}</td>
            <td>{overlay.overlap_percentage:.1f}%</td>
        </tr>
"""
                html += f"""    </table>
"""

        html += f"""
    <div class="footer">
        <p><em>Report generated on {data.report_date} using GreenLang EUDR Professional Pack</em></p>
        <p><strong>Data Sources:</strong> WDPA, Key Biodiversity Areas (KBA), Indigenous Land Registries</p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        report_dict = {
            "report_type": "protected_area_analysis",
            "operator_name": data.operator_name,
            "report_date": data.report_date,
            "analysis_date": data.analysis_date,
            "total_plots_analyzed": data.total_plots_analyzed,
            "statistics": data.statistics.dict(),
            "overlays": [overlay.dict() for overlay in data.overlays],
            "buffer_analysis": [buffer.dict() for buffer in data.buffer_analysis],
            "indigenous_flags": [flag.dict() for flag in data.indigenous_flags],
            "unesco_ramsar_proximity": [prox.dict() for prox in data.unesco_ramsar_proximity],
            "risk_amplifications": [amp.dict() for amp in data.risk_amplifications],
            "data_sources": data.data_sources,
            "key_findings": data.key_findings,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "config": self.config.dict(),
            }
        }

        return json.dumps(report_dict, indent=2)

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        return hashlib.sha256(content.encode()).hexdigest()
