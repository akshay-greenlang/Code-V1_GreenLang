"""
Satellite Monitoring Report Template - PACK-007 EUDR Professional Pack

This module generates satellite imagery analysis reports with monitoring summaries,
deforestation alerts, fire detection events, temporal comparisons, affected plots,
and alert statistics for EUDR compliance.

The template supports multiple output formats (markdown, HTML, JSON) and includes
SHA-256 provenance hashing for audit trails.

Example:
    >>> from satellite_monitoring_report import SatelliteMonitoringReportTemplate, ReportData
    >>> data = ReportData(
    ...     operator_name="Global Commodities Inc",
    ...     monitoring_period_start="2025-01-01",
    ...     monitoring_period_end="2026-03-15",
    ...     total_plots_monitored=450
    ... )
    >>> template = SatelliteMonitoringReportTemplate()
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
    """Configuration for Satellite Monitoring Report generation."""

    include_temporal_comparison: bool = Field(
        default=True,
        description="Include temporal change analysis"
    )
    include_fire_detection: bool = Field(
        default=True,
        description="Include fire detection events"
    )
    include_plot_details: bool = Field(
        default=True,
        description="Include detailed affected plot information"
    )
    alert_threshold_hectares: float = Field(
        default=0.1,
        description="Minimum deforestation area to trigger alert (hectares)"
    )


class DeforestationAlert(BaseModel):
    """Deforestation alert information."""

    alert_id: str = Field(..., description="Unique alert identifier")
    detection_date: str = Field(..., description="Date alert was detected")
    plot_id: str = Field(..., description="Affected plot identifier")
    area_hectares: float = Field(..., ge=0, description="Deforested area in hectares")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence (0-1)")
    alert_severity: str = Field(..., description="HIGH, MEDIUM, LOW")
    coordinates: str = Field(..., description="Geographic coordinates (lat, lon)")
    verification_status: str = Field(
        default="PENDING",
        description="PENDING, CONFIRMED, FALSE_POSITIVE"
    )


class FireDetectionEvent(BaseModel):
    """Fire detection event information."""

    event_id: str = Field(..., description="Unique event identifier")
    detection_date: str = Field(..., description="Date fire was detected")
    plot_id: str = Field(..., description="Affected plot identifier")
    fire_intensity: str = Field(..., description="HIGH, MEDIUM, LOW")
    affected_area_hectares: float = Field(..., ge=0, description="Fire-affected area")
    coordinates: str = Field(..., description="Geographic coordinates (lat, lon)")
    distance_to_plot_km: float = Field(..., ge=0, description="Distance to nearest plot")


class TemporalComparison(BaseModel):
    """Temporal change comparison."""

    period: str = Field(..., description="Time period (e.g., 2025-Q1)")
    total_alerts: int = Field(..., ge=0, description="Total alerts in period")
    total_area_hectares: float = Field(..., ge=0, description="Total deforested area")
    avg_confidence: float = Field(..., ge=0, le=1, description="Average alert confidence")
    confirmed_alerts: int = Field(..., ge=0, description="Number of confirmed alerts")


class AffectedPlot(BaseModel):
    """Affected plot summary."""

    plot_id: str = Field(..., description="Plot identifier")
    plot_name: str = Field(..., description="Plot name or description")
    total_area_hectares: float = Field(..., ge=0, description="Total plot area")
    deforested_area_hectares: float = Field(..., ge=0, description="Deforested area")
    deforestation_percentage: float = Field(..., ge=0, le=100, description="% deforested")
    alert_count: int = Field(..., ge=0, description="Number of alerts for this plot")
    last_alert_date: str = Field(..., description="Date of most recent alert")
    risk_level: str = Field(..., description="HIGH, MEDIUM, LOW")


class MonitoringStatistics(BaseModel):
    """Overall monitoring statistics."""

    total_plots_monitored: int = Field(..., ge=0, description="Total plots monitored")
    total_area_monitored_hectares: float = Field(..., ge=0, description="Total area monitored")
    total_alerts_issued: int = Field(..., ge=0, description="Total alerts issued")
    confirmed_alerts: int = Field(..., ge=0, description="Confirmed alerts")
    false_positives: int = Field(..., ge=0, description="False positive alerts")
    pending_verification: int = Field(..., ge=0, description="Alerts pending verification")
    total_deforestation_hectares: float = Field(..., ge=0, description="Total deforestation")
    plots_with_alerts: int = Field(..., ge=0, description="Number of plots with alerts")
    fire_events_detected: int = Field(..., ge=0, description="Fire events detected")


class ReportData(BaseModel):
    """Data model for Satellite Monitoring Report."""

    operator_name: str = Field(..., description="Operator name")
    monitoring_period_start: str = Field(..., description="Monitoring start date (ISO format)")
    monitoring_period_end: str = Field(..., description="Monitoring end date (ISO format)")
    report_date: str = Field(
        default_factory=lambda: datetime.now().isoformat()[:10],
        description="Report generation date"
    )
    statistics: MonitoringStatistics = Field(..., description="Overall statistics")
    deforestation_alerts: List[DeforestationAlert] = Field(
        default_factory=list,
        description="Deforestation alerts"
    )
    fire_events: List[FireDetectionEvent] = Field(
        default_factory=list,
        description="Fire detection events"
    )
    temporal_comparison: List[TemporalComparison] = Field(
        default_factory=list,
        description="Temporal change analysis"
    )
    affected_plots: List[AffectedPlot] = Field(
        default_factory=list,
        description="Plots affected by deforestation"
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Satellite data sources (e.g., Sentinel-2, Landsat)"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings from monitoring"
    )

    @validator('monitoring_period_start', 'monitoring_period_end', 'report_date')
    def validate_dates(cls, v):
        """Validate dates are in ISO format."""
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected ISO format (YYYY-MM-DD).")
        return v


class SatelliteMonitoringReportTemplate:
    """
    Satellite Monitoring Report Template for EUDR Professional Pack.

    Generates comprehensive satellite imagery analysis reports with monitoring summaries,
    deforestation alerts, fire detection events, temporal comparisons, and affected plot details.

    Attributes:
        config: Report configuration options

    Example:
        >>> template = SatelliteMonitoringReportTemplate()
        >>> report = template.render(data, format="markdown")
        >>> assert "Deforestation Alerts" in report
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize Satellite Monitoring Report Template."""
        self.config = config or ReportConfig()

    def render(
        self,
        data: ReportData,
        format: Literal["markdown", "html", "json"] = "markdown"
    ) -> str:
        """
        Render the satellite monitoring report.

        Args:
            data: Report data
            format: Output format (markdown, html, json)

        Returns:
            Rendered report content

        Raises:
            ValueError: If format is not supported
        """
        logger.info(
            f"Rendering Satellite Monitoring Report for {data.operator_name} in {format} format"
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
        sections.append(f"# Satellite Monitoring Report")
        sections.append(f"")
        sections.append(f"**Operator:** {data.operator_name}")
        sections.append(f"**Monitoring Period:** {data.monitoring_period_start} to {data.monitoring_period_end}")
        sections.append(f"**Report Date:** {data.report_date}")
        sections.append(f"")
        sections.append(f"---")
        sections.append(f"")

        # Executive Summary
        sections.append(f"## Executive Summary")
        sections.append(f"")
        stats = data.statistics
        sections.append(
            f"This satellite monitoring report covers {stats.total_plots_monitored} plots "
            f"({stats.total_area_monitored_hectares:,.1f} hectares) during the monitoring period. "
            f"A total of **{stats.total_alerts_issued}** deforestation alerts were issued, "
            f"of which {stats.confirmed_alerts} have been confirmed and {stats.false_positives} "
            f"were false positives. The total confirmed deforestation area is "
            f"**{stats.total_deforestation_hectares:.2f} hectares**."
        )
        sections.append(f"")

        if data.data_sources:
            sections.append(f"**Data Sources:** {', '.join(data.data_sources)}")
            sections.append(f"")

        # Monitoring Statistics
        sections.append(f"## Monitoring Statistics")
        sections.append(f"")
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|-------|")
        sections.append(f"| Total Plots Monitored | {stats.total_plots_monitored:,} |")
        sections.append(f"| Total Area Monitored | {stats.total_area_monitored_hectares:,.1f} ha |")
        sections.append(f"| Total Alerts Issued | {stats.total_alerts_issued:,} |")
        sections.append(f"| Confirmed Alerts | {stats.confirmed_alerts:,} |")
        sections.append(f"| False Positives | {stats.false_positives:,} |")
        sections.append(f"| Pending Verification | {stats.pending_verification:,} |")
        sections.append(f"| Total Deforestation | {stats.total_deforestation_hectares:.2f} ha |")
        sections.append(f"| Plots with Alerts | {stats.plots_with_alerts:,} |")
        sections.append(f"| Fire Events Detected | {stats.fire_events_detected:,} |")
        sections.append(f"")

        # Deforestation Alerts
        if data.deforestation_alerts:
            sections.append(f"## Deforestation Alerts")
            sections.append(f"")
            sections.append(
                f"The following deforestation alerts were detected during the monitoring period:"
            )
            sections.append(f"")

            sections.append(
                f"| Alert ID | Date | Plot ID | Area (ha) | Confidence | Severity | Status |"
            )
            sections.append(
                f"|----------|------|---------|-----------|------------|----------|--------|"
            )

            for alert in sorted(data.deforestation_alerts, key=lambda x: x.detection_date, reverse=True):
                sections.append(
                    f"| {alert.alert_id} | {alert.detection_date} | {alert.plot_id} | "
                    f"{alert.area_hectares:.2f} | {alert.confidence * 100:.0f}% | "
                    f"{alert.alert_severity} | {alert.verification_status} |"
                )
            sections.append(f"")

            # High severity alerts
            high_severity = [a for a in data.deforestation_alerts if a.alert_severity == "HIGH"]
            if high_severity:
                sections.append(f"### High Severity Alerts")
                sections.append(f"")
                sections.append(
                    f"**{len(high_severity)}** high severity alerts require immediate investigation:"
                )
                sections.append(f"")
                for alert in high_severity[:10]:
                    sections.append(
                        f"- **{alert.alert_id}**: {alert.area_hectares:.2f} ha at {alert.coordinates} "
                        f"({alert.verification_status})"
                    )
                sections.append(f"")

        # Fire Detection Events
        if self.config.include_fire_detection and data.fire_events:
            sections.append(f"## Fire Detection Events")
            sections.append(f"")
            sections.append(
                f"A total of **{len(data.fire_events)}** fire events were detected within or "
                f"near monitored plots:"
            )
            sections.append(f"")

            sections.append(
                f"| Event ID | Date | Plot ID | Intensity | Area (ha) | Distance (km) |"
            )
            sections.append(
                f"|----------|------|---------|-----------|-----------|---------------|"
            )

            for event in sorted(data.fire_events, key=lambda x: x.detection_date, reverse=True)[:20]:
                sections.append(
                    f"| {event.event_id} | {event.detection_date} | {event.plot_id} | "
                    f"{event.fire_intensity} | {event.affected_area_hectares:.2f} | "
                    f"{event.distance_to_plot_km:.2f} |"
                )
            sections.append(f"")

        # Temporal Comparison
        if self.config.include_temporal_comparison and data.temporal_comparison:
            sections.append(f"## Temporal Change Analysis")
            sections.append(f"")
            sections.append(
                f"Deforestation trends over time, aggregated by period:"
            )
            sections.append(f"")

            sections.append(
                f"| Period | Alerts | Area (ha) | Avg Confidence | Confirmed |"
            )
            sections.append(
                f"|--------|--------|-----------|----------------|-----------|"
            )

            for comp in data.temporal_comparison:
                sections.append(
                    f"| {comp.period} | {comp.total_alerts} | {comp.total_area_hectares:.2f} | "
                    f"{comp.avg_confidence * 100:.0f}% | {comp.confirmed_alerts} |"
                )
            sections.append(f"")

            # Trend analysis
            if len(data.temporal_comparison) >= 2:
                latest = data.temporal_comparison[-1]
                previous = data.temporal_comparison[-2]
                change = latest.total_area_hectares - previous.total_area_hectares
                pct_change = (change / previous.total_area_hectares * 100) if previous.total_area_hectares > 0 else 0

                trend = "increased" if change > 0 else "decreased"
                sections.append(
                    f"**Trend Analysis:** Deforestation area {trend} by {abs(pct_change):.1f}% "
                    f"({abs(change):.2f} ha) from {previous.period} to {latest.period}."
                )
                sections.append(f"")

        # Affected Plots
        if self.config.include_plot_details and data.affected_plots:
            sections.append(f"## Affected Plots")
            sections.append(f"")
            sections.append(
                f"The following plots have confirmed or suspected deforestation:"
            )
            sections.append(f"")

            sections.append(
                f"| Plot ID | Plot Name | Total Area (ha) | Deforested (ha) | % Loss | Alerts | Risk |"
            )
            sections.append(
                f"|---------|-----------|-----------------|-----------------|--------|--------|------|"
            )

            for plot in sorted(data.affected_plots, key=lambda x: x.deforestation_percentage, reverse=True):
                sections.append(
                    f"| {plot.plot_id} | {plot.plot_name} | {plot.total_area_hectares:.2f} | "
                    f"{plot.deforested_area_hectares:.2f} | {plot.deforestation_percentage:.1f}% | "
                    f"{plot.alert_count} | {plot.risk_level} |"
                )
            sections.append(f"")

            # High-risk plots
            high_risk = [p for p in data.affected_plots if p.risk_level == "HIGH"]
            if high_risk:
                sections.append(f"### High-Risk Plots")
                sections.append(f"")
                sections.append(
                    f"**{len(high_risk)}** plots classified as high-risk based on deforestation extent:"
                )
                sections.append(f"")
                for plot in high_risk[:5]:
                    sections.append(
                        f"- **{plot.plot_name}** ({plot.plot_id}): {plot.deforestation_percentage:.1f}% "
                        f"loss, last alert {plot.last_alert_date}"
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

        if stats.pending_verification > 0:
            sections.append(
                f"1. **Verify Pending Alerts:** {stats.pending_verification} alerts require "
                f"field verification or third-party validation"
            )

        if stats.plots_with_alerts > 0:
            sections.append(
                f"2. **Investigate Affected Plots:** Conduct due diligence on {stats.plots_with_alerts} "
                f"plots with deforestation alerts"
            )

        if data.fire_events:
            sections.append(
                f"3. **Fire Risk Assessment:** {len(data.fire_events)} fire events detected; "
                f"assess fire management practices"
            )

        sections.append(
            f"4. **Continuous Monitoring:** Maintain regular satellite monitoring frequency to "
            f"detect early-stage deforestation"
        )
        sections.append(f"")

        # Footer
        sections.append(f"---")
        sections.append(f"")
        sections.append(
            f"*Report generated on {data.report_date} using GreenLang EUDR Professional Pack*"
        )

        return "\n".join(sections)

    def _render_html(self, data: ReportData) -> str:
        """Render report in HTML format."""
        stats = data.statistics

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Satellite Monitoring Report - {data.operator_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #555; margin-top: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #27ae60; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #e74c3c; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .high-severity {{ color: #c0392b; font-weight: bold; }}
        .medium-severity {{ color: #e67e22; }}
        .low-severity {{ color: #27ae60; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Satellite Monitoring Report</h1>

    <div class="summary">
        <p><strong>Operator:</strong> {data.operator_name}</p>
        <p><strong>Monitoring Period:</strong> {data.monitoring_period_start} to {data.monitoring_period_end}</p>
        <p><strong>Report Date:</strong> {data.report_date}</p>
"""

        if data.data_sources:
            html += f"        <p><strong>Data Sources:</strong> {', '.join(data.data_sources)}</p>\n"

        html += f"""    </div>

    <h2>Executive Summary</h2>
    <p>
        This satellite monitoring report covers <strong>{stats.total_plots_monitored}</strong> plots
        ({stats.total_area_monitored_hectares:,.1f} hectares) during the monitoring period.
        A total of <span class="metric">{stats.total_alerts_issued}</span> deforestation alerts were issued,
        of which {stats.confirmed_alerts} have been confirmed. The total confirmed deforestation area is
        <span class="metric">{stats.total_deforestation_hectares:.2f} hectares</span>.
    </p>

    <h2>Monitoring Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Plots Monitored</td><td>{stats.total_plots_monitored:,}</td></tr>
        <tr><td>Total Area Monitored</td><td>{stats.total_area_monitored_hectares:,.1f} ha</td></tr>
        <tr><td>Total Alerts Issued</td><td>{stats.total_alerts_issued:,}</td></tr>
        <tr><td>Confirmed Alerts</td><td>{stats.confirmed_alerts:,}</td></tr>
        <tr><td>False Positives</td><td>{stats.false_positives:,}</td></tr>
        <tr><td>Pending Verification</td><td>{stats.pending_verification:,}</td></tr>
        <tr><td>Total Deforestation</td><td>{stats.total_deforestation_hectares:.2f} ha</td></tr>
        <tr><td>Plots with Alerts</td><td>{stats.plots_with_alerts:,}</td></tr>
        <tr><td>Fire Events Detected</td><td>{stats.fire_events_detected:,}</td></tr>
    </table>
"""

        if data.deforestation_alerts:
            html += f"""
    <h2>Deforestation Alerts</h2>
    <table>
        <tr>
            <th>Alert ID</th><th>Date</th><th>Plot ID</th><th>Area (ha)</th>
            <th>Confidence</th><th>Severity</th><th>Status</th>
        </tr>
"""
            for alert in sorted(data.deforestation_alerts, key=lambda x: x.detection_date, reverse=True)[:50]:
                severity_class = f"{alert.alert_severity.lower()}-severity"
                html += f"""        <tr>
            <td>{alert.alert_id}</td>
            <td>{alert.detection_date}</td>
            <td>{alert.plot_id}</td>
            <td>{alert.area_hectares:.2f}</td>
            <td>{alert.confidence * 100:.0f}%</td>
            <td class="{severity_class}">{alert.alert_severity}</td>
            <td>{alert.verification_status}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if self.config.include_fire_detection and data.fire_events:
            html += f"""
    <h2>Fire Detection Events</h2>
    <table>
        <tr><th>Event ID</th><th>Date</th><th>Plot ID</th><th>Intensity</th><th>Area (ha)</th><th>Distance (km)</th></tr>
"""
            for event in sorted(data.fire_events, key=lambda x: x.detection_date, reverse=True)[:30]:
                html += f"""        <tr>
            <td>{event.event_id}</td>
            <td>{event.detection_date}</td>
            <td>{event.plot_id}</td>
            <td>{event.fire_intensity}</td>
            <td>{event.affected_area_hectares:.2f}</td>
            <td>{event.distance_to_plot_km:.2f}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if self.config.include_temporal_comparison and data.temporal_comparison:
            html += f"""
    <h2>Temporal Change Analysis</h2>
    <table>
        <tr><th>Period</th><th>Alerts</th><th>Area (ha)</th><th>Avg Confidence</th><th>Confirmed</th></tr>
"""
            for comp in data.temporal_comparison:
                html += f"""        <tr>
            <td>{comp.period}</td>
            <td>{comp.total_alerts}</td>
            <td>{comp.total_area_hectares:.2f}</td>
            <td>{comp.avg_confidence * 100:.0f}%</td>
            <td>{comp.confirmed_alerts}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if self.config.include_plot_details and data.affected_plots:
            html += f"""
    <h2>Affected Plots</h2>
    <table>
        <tr><th>Plot ID</th><th>Plot Name</th><th>Total Area (ha)</th><th>Deforested (ha)</th><th>% Loss</th><th>Alerts</th><th>Risk</th></tr>
"""
            for plot in sorted(data.affected_plots, key=lambda x: x.deforestation_percentage, reverse=True):
                html += f"""        <tr>
            <td>{plot.plot_id}</td>
            <td>{plot.plot_name}</td>
            <td>{plot.total_area_hectares:.2f}</td>
            <td>{plot.deforested_area_hectares:.2f}</td>
            <td>{plot.deforestation_percentage:.1f}%</td>
            <td>{plot.alert_count}</td>
            <td>{plot.risk_level}</td>
        </tr>
"""
            html += f"""    </table>
"""

        if data.key_findings:
            html += f"""
    <h2>Key Findings</h2>
    <ol>
"""
            for finding in data.key_findings:
                html += f"        <li>{finding}</li>\n"
            html += f"""    </ol>
"""

        html += f"""
    <div class="footer">
        <p><em>Report generated on {data.report_date} using GreenLang EUDR Professional Pack</em></p>
    </div>
</body>
</html>"""

        return html

    def _render_json(self, data: ReportData) -> str:
        """Render report in JSON format."""
        report_dict = {
            "report_type": "satellite_monitoring",
            "operator_name": data.operator_name,
            "monitoring_period": {
                "start": data.monitoring_period_start,
                "end": data.monitoring_period_end,
            },
            "report_date": data.report_date,
            "statistics": data.statistics.dict(),
            "deforestation_alerts": [alert.dict() for alert in data.deforestation_alerts],
            "fire_events": [event.dict() for event in data.fire_events],
            "temporal_comparison": [comp.dict() for comp in data.temporal_comparison],
            "affected_plots": [plot.dict() for plot in data.affected_plots],
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
