# -*- coding: utf-8 -*-
"""
ConsolidationDashboard - Interactive consolidation dashboard data for PACK-050.

Generates dashboard data including KPI cards (total emissions, YoY change,
completeness %, entity count), entity contribution chart, scope pie chart,
geographic heat map data, timeline of M&A events, and alert summary.

Sections:
    1. KPI Cards (total emissions, YoY%, completeness, entities, scopes)
    2. Entity Contribution Chart (bar chart data)
    3. Scope Pie Chart (S1/S2/S3 distribution)
    4. Geographic Heat Map (country-level emissions for map rendering)
    5. M&A Event Timeline (chronological event list)
    6. Alert Summary (warnings, errors, info items)
    7. Data Quality Indicators
    8. Provenance Footer

Output Formats: Markdown, HTML, JSON, CSV

Author: GreenLang Team
Version: 50.0.0
"""

from __future__ import annotations

import hashlib, logging, uuid, json, time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class KpiCard(BaseModel):
    """Single KPI card for the dashboard."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    kpi_name: str = Field("")
    kpi_value: str = Field("")
    kpi_unit: str = Field("")
    trend_direction: str = Field("", description="up, down, flat")
    trend_value: str = Field("")
    status: str = Field("", description="good, warning, critical, neutral")

class EntityChartBar(BaseModel):
    """Entity contribution bar chart data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    entity_name: str = Field("")
    entity_type: str = Field("")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_tco2e: Decimal = Field(Decimal("0"))
    contribution_pct: Decimal = Field(Decimal("0"))

class ScopePieSlice(BaseModel):
    """Scope pie chart slice."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scope: str = Field("")
    tco2e: Decimal = Field(Decimal("0"))
    share_pct: Decimal = Field(Decimal("0"))
    color: str = Field("")

class GeoHeatMapPoint(BaseModel):
    """Geographic heat map data point."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    country_code: str = Field("")
    country_name: str = Field("")
    total_tco2e: Decimal = Field(Decimal("0"))
    entity_count: int = Field(0)
    intensity_level: str = Field("", description="low, medium, high, very_high")

class TimelineEvent(BaseModel):
    """M&A timeline event."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    event_date: str = Field("")
    event_type: str = Field("")
    entity_name: str = Field("")
    description: str = Field("")
    impact_tco2e: Decimal = Field(Decimal("0"))

class AlertItem(BaseModel):
    """Dashboard alert item."""
    alert_id: str = Field("")
    severity: str = Field("", description="info, warning, error, critical")
    category: str = Field("")
    message: str = Field("")
    entity_name: str = Field("")
    action_required: str = Field("")

class DataQualityIndicator(BaseModel):
    """Data quality indicator for the dashboard."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dimension: str = Field("")
    score: Decimal = Field(Decimal("0"))
    max_score: Decimal = Field(Decimal("100"))
    status: str = Field("")

class ConsolidationDashboardInput(BaseModel):
    """Complete input for the consolidation dashboard."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    company_name: str = Field("Organisation")
    reporting_period: str = Field("")
    kpi_cards: List[Dict[str, Any]] = Field(default_factory=list)
    entity_chart: List[Dict[str, Any]] = Field(default_factory=list)
    scope_pie: List[Dict[str, Any]] = Field(default_factory=list)
    geo_heat_map: List[Dict[str, Any]] = Field(default_factory=list)
    timeline_events: List[Dict[str, Any]] = Field(default_factory=list)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    data_quality: List[Dict[str, Any]] = Field(default_factory=list)
    total_emissions_tco2e: Decimal = Field(Decimal("0"))
    entity_count: int = Field(0)

# ---------------------------------------------------------------------------
# Output Model
# ---------------------------------------------------------------------------

class ConsolidationDashboardOutput(BaseModel):
    """Rendered consolidation dashboard data."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    report_id: str = Field(default_factory=_new_uuid)
    company_name: str = Field("")
    reporting_period: str = Field("")
    generated_at: str = Field("")
    total_emissions_tco2e: Decimal = Field(Decimal("0"))
    entity_count: int = Field(0)
    kpi_cards: List[KpiCard] = Field(default_factory=list)
    entity_chart: List[EntityChartBar] = Field(default_factory=list)
    scope_pie: List[ScopePieSlice] = Field(default_factory=list)
    geo_heat_map: List[GeoHeatMapPoint] = Field(default_factory=list)
    timeline_events: List[TimelineEvent] = Field(default_factory=list)
    alerts: List[AlertItem] = Field(default_factory=list)
    alerts_by_severity: Dict[str, int] = Field(default_factory=dict)
    data_quality: List[DataQualityIndicator] = Field(default_factory=list)
    overall_quality_score: Decimal = Field(Decimal("0"))
    provenance_hash: str = Field("")

# =============================================================================
# TEMPLATE CLASS
# =============================================================================

class ConsolidationDashboard:
    """
    Interactive consolidation dashboard template for PACK-050.

    Produces dashboard data with KPI cards, entity contribution charts,
    scope pie chart, geographic heat map, M&A timeline, and alerts.

    Example:
        >>> tpl = ConsolidationDashboard()
        >>> report = tpl.render(data)
        >>> md = tpl.export_markdown(report)
    """

    # Default scope colours
    SCOPE_COLORS = {
        "Scope 1": "#FF6B35",
        "Scope 2": "#1E88E5",
        "Scope 3": "#43A047",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None
        self.processing_time_ms: float = 0.0

    # ------------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> ConsolidationDashboardOutput:
        """Render consolidation dashboard from input data."""
        start = time.monotonic()
        self.generated_at = utcnow()
        inp = ConsolidationDashboardInput(**data) if isinstance(data, dict) else data

        kpi_cards = [KpiCard(**k) if isinstance(k, dict) else k for k in inp.kpi_cards]
        entity_chart = [EntityChartBar(**e) if isinstance(e, dict) else e for e in inp.entity_chart]
        entity_chart.sort(key=lambda e: e.total_tco2e, reverse=True)

        scope_pie = [ScopePieSlice(**s) if isinstance(s, dict) else s for s in inp.scope_pie]
        for sp in scope_pie:
            if not sp.color:
                sp.color = self.SCOPE_COLORS.get(sp.scope, "#757575")

        geo_map = [GeoHeatMapPoint(**g) if isinstance(g, dict) else g for g in inp.geo_heat_map]
        for gp in geo_map:
            if not gp.intensity_level:
                gp.intensity_level = self._classify_intensity(gp.total_tco2e)

        timeline = [TimelineEvent(**t) if isinstance(t, dict) else t for t in inp.timeline_events]
        timeline.sort(key=lambda t: t.event_date)

        alerts = [AlertItem(**a) if isinstance(a, dict) else a for a in inp.alerts]

        # Count alerts by severity
        alerts_by_severity: Dict[str, int] = {}
        for a in alerts:
            alerts_by_severity[a.severity] = alerts_by_severity.get(a.severity, 0) + 1

        dq = [DataQualityIndicator(**d) if isinstance(d, dict) else d for d in inp.data_quality]
        overall_quality = Decimal("0")
        if dq:
            total_score = sum(d.score for d in dq)
            total_max = sum(d.max_score for d in dq)
            if total_max > Decimal("0"):
                overall_quality = (
                    total_score / total_max * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        prov = _compute_hash(json.dumps(data, sort_keys=True, default=str))

        output = ConsolidationDashboardOutput(
            company_name=inp.company_name,
            reporting_period=inp.reporting_period,
            generated_at=self.generated_at.isoformat(),
            total_emissions_tco2e=inp.total_emissions_tco2e,
            entity_count=inp.entity_count,
            kpi_cards=kpi_cards,
            entity_chart=entity_chart,
            scope_pie=scope_pie,
            geo_heat_map=geo_map,
            timeline_events=timeline,
            alerts=alerts,
            alerts_by_severity=alerts_by_severity,
            data_quality=dq,
            overall_quality_score=overall_quality,
            provenance_hash=prov,
        )
        self.processing_time_ms = (time.monotonic() - start) * 1000
        return output

    # ------------------------------------------------------------------
    # CONVENIENCE RENDER METHODS
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_markdown(r)

    def render_html(self, data: Dict[str, Any]) -> str:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_html(r)

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r = self.render(data) if isinstance(data, dict) else data
        return self.export_json(r)

    # ------------------------------------------------------------------
    # EXPORT METHODS
    # ------------------------------------------------------------------

    def export_markdown(self, r: ConsolidationDashboardOutput) -> str:
        """Export dashboard as Markdown."""
        lines: List[str] = []
        lines.append(f"# Consolidation Dashboard - {r.company_name}")
        lines.append(f"**Period:** {r.reporting_period} | **Total:** {r.total_emissions_tco2e:,.2f} tCO2e | **Entities:** {r.entity_count}")
        lines.append("")

        # KPI cards
        if r.kpi_cards:
            lines.append("## KPI Cards")
            lines.append("| KPI | Value | Unit | Trend | Status |")
            lines.append("|-----|-------|------|-------|--------|")
            for k in r.kpi_cards:
                trend = f"{k.trend_direction} {k.trend_value}" if k.trend_value else k.trend_direction
                lines.append(f"| {k.kpi_name} | {k.kpi_value} | {k.kpi_unit} | {trend} | {k.status} |")
            lines.append("")

        # Entity chart
        if r.entity_chart:
            lines.append("## Entity Contribution Chart")
            lines.append("| Entity | Type | S1 | S2 | S3 | Total | Contribution |")
            lines.append("|--------|------|----|----|-----|-------|-------------|")
            for e in r.entity_chart:
                lines.append(
                    f"| {e.entity_name} | {e.entity_type} | {e.scope_1_tco2e:,.0f} | "
                    f"{e.scope_2_tco2e:,.0f} | {e.scope_3_tco2e:,.0f} | "
                    f"{e.total_tco2e:,.0f} | {e.contribution_pct}% |"
                )
            lines.append("")

        # Scope pie
        if r.scope_pie:
            lines.append("## Scope Distribution")
            lines.append("| Scope | tCO2e | Share | Color |")
            lines.append("|-------|------:|------:|-------|")
            for sp in r.scope_pie:
                lines.append(f"| {sp.scope} | {sp.tco2e:,.2f} | {sp.share_pct}% | {sp.color} |")
            lines.append("")

        # Geographic heat map
        if r.geo_heat_map:
            lines.append("## Geographic Heat Map")
            lines.append("| Country | Code | tCO2e | Entities | Intensity |")
            lines.append("|---------|------|------:|----------|-----------|")
            for g in r.geo_heat_map:
                lines.append(
                    f"| {g.country_name} | {g.country_code} | {g.total_tco2e:,.0f} | "
                    f"{g.entity_count} | {g.intensity_level} |"
                )
            lines.append("")

        # Timeline
        if r.timeline_events:
            lines.append("## M&A Event Timeline")
            lines.append("| Date | Type | Entity | Description | Impact tCO2e |")
            lines.append("|------|------|--------|-------------|-------------|")
            for t in r.timeline_events:
                lines.append(
                    f"| {t.event_date} | {t.event_type} | {t.entity_name} | "
                    f"{t.description} | {t.impact_tco2e:,.0f} |"
                )
            lines.append("")

        # Alerts
        if r.alerts:
            sev_summary = " | ".join(f"{k}: {v}" for k, v in sorted(r.alerts_by_severity.items()))
            lines.append(f"## Alerts ({sev_summary})")
            lines.append("| Severity | Category | Entity | Message | Action |")
            lines.append("|----------|----------|--------|---------|--------|")
            for a in r.alerts:
                lines.append(
                    f"| {a.severity} | {a.category} | {a.entity_name} | "
                    f"{a.message} | {a.action_required} |"
                )
            lines.append("")

        # Data quality
        if r.data_quality:
            lines.append(f"## Data Quality (Overall: {r.overall_quality_score}%)")
            lines.append("| Dimension | Score | Max | Status |")
            lines.append("|-----------|------:|----:|--------|")
            for d in r.data_quality:
                lines.append(f"| {d.dimension} | {d.score} | {d.max_score} | {d.status} |")
            lines.append("")

        lines.append(f"---\n*Report ID: {r.report_id} | Provenance: {r.provenance_hash}*")
        return "\n".join(lines)

    def export_html(self, r: ConsolidationDashboardOutput) -> str:
        """Export dashboard as HTML."""
        md = self.export_markdown(r)
        html_lines = [
            "<!DOCTYPE html><html><head>",
            "<meta charset='utf-8'>",
            f"<title>Consolidation Dashboard - {r.company_name}</title>",
            "<style>body{font-family:sans-serif;margin:2em;}table{border-collapse:collapse;width:100%;margin:1em 0;}",
            "th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f5f5f5;}</style>",
            "</head><body>",
            f"<pre>{md}</pre>",
            "</body></html>",
        ]
        return "\n".join(html_lines)

    def export_json(self, r: ConsolidationDashboardOutput) -> Dict[str, Any]:
        """Export dashboard as JSON-serializable dict."""
        return json.loads(r.model_dump_json())

    def export_csv(self, r: ConsolidationDashboardOutput) -> str:
        """Export entity chart data as CSV."""
        lines_out = [
            "entity_name,entity_type,scope_1_tco2e,scope_2_tco2e,"
            "scope_3_tco2e,total_tco2e,contribution_pct"
        ]
        for e in r.entity_chart:
            lines_out.append(
                f"{e.entity_name},{e.entity_type},{e.scope_1_tco2e},"
                f"{e.scope_2_tco2e},{e.scope_3_tco2e},{e.total_tco2e},"
                f"{e.contribution_pct}"
            )
        return "\n".join(lines_out)

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_intensity(tco2e: Decimal) -> str:
        """Classify emission intensity level for heat map colouring."""
        if tco2e > Decimal("100000"):
            return "very_high"
        elif tco2e > Decimal("50000"):
            return "high"
        elif tco2e > Decimal("10000"):
            return "medium"
        return "low"

__all__ = [
    "ConsolidationDashboard",
    "ConsolidationDashboardInput",
    "ConsolidationDashboardOutput",
    "KpiCard",
    "EntityChartBar",
    "ScopePieSlice",
    "GeoHeatMapPoint",
    "TimelineEvent",
    "AlertItem",
    "DataQualityIndicator",
]
