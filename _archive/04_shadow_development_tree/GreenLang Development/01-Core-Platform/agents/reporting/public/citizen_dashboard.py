# -*- coding: utf-8 -*-
"""
GL-REP-PUB-003: Citizen Climate Dashboard Agent
================================================

Creates public-facing climate dashboards for citizen engagement including
key metrics, visualizations, and progress indicators.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class MetricCategory(str, Enum):
    """Dashboard metric categories."""
    EMISSIONS = "emissions"
    ENERGY = "energy"
    TRANSPORTATION = "transportation"
    WASTE = "waste"
    ADAPTATION = "adaptation"
    ECONOMY = "economy"


class VisualizationType(str, Enum):
    """Types of visualizations."""
    GAUGE = "gauge"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    MAP = "map"
    TABLE = "table"
    COUNTER = "counter"


class TrendDirection(str, Enum):
    """Trend direction indicators."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    UNKNOWN = "unknown"


class DashboardMetric(BaseModel):
    """A metric for the citizen dashboard."""
    metric_id: str = Field(...)
    name: str = Field(...)
    category: MetricCategory = Field(...)
    current_value: float = Field(...)
    unit: str = Field(...)
    target_value: Optional[float] = Field(None)
    baseline_value: Optional[float] = Field(None)
    baseline_year: Optional[int] = Field(None)
    trend: TrendDirection = Field(default=TrendDirection.UNKNOWN)
    trend_pct_change: float = Field(default=0.0)
    visualization_type: VisualizationType = Field(default=VisualizationType.GAUGE)
    description: str = Field(default="")
    last_updated: datetime = Field(default_factory=DeterministicClock.now)
    source: Optional[str] = Field(None)


class TimeSeriesPoint(BaseModel):
    """A point in a time series."""
    date: datetime = Field(...)
    value: float = Field(...)
    label: Optional[str] = Field(None)


class TimeSeriesData(BaseModel):
    """Time series data for visualizations."""
    series_id: str = Field(...)
    metric_id: str = Field(...)
    name: str = Field(...)
    data_points: List[TimeSeriesPoint] = Field(default_factory=list)
    unit: str = Field(default="")


class ComparisonBenchmark(BaseModel):
    """Benchmark for comparing municipality performance."""
    benchmark_id: str = Field(...)
    name: str = Field(...)
    metric_id: str = Field(...)
    comparison_type: str = Field(...)  # national_average, peer_cities, best_in_class
    benchmark_value: float = Field(...)
    municipality_value: float = Field(...)
    performance_vs_benchmark_pct: float = Field(default=0.0)


class DashboardSection(BaseModel):
    """A section of the dashboard."""
    section_id: str = Field(...)
    title: str = Field(...)
    description: Optional[str] = Field(None)
    metrics: List[str] = Field(default_factory=list)  # metric_ids
    order: int = Field(default=0)
    visible: bool = Field(default=True)


class DashboardData(BaseModel):
    """Complete citizen dashboard data."""
    dashboard_id: str = Field(...)
    municipality_name: str = Field(...)
    title: str = Field(...)
    subtitle: Optional[str] = Field(None)
    metrics: List[DashboardMetric] = Field(default_factory=list)
    time_series: List[TimeSeriesData] = Field(default_factory=list)
    benchmarks: List[ComparisonBenchmark] = Field(default_factory=list)
    sections: List[DashboardSection] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0, le=100)
    overall_grade: str = Field(default="")
    last_updated: datetime = Field(default_factory=DeterministicClock.now)
    reporting_year: int = Field(default=2024)
    provenance_hash: Optional[str] = Field(None)


class CitizenDashboardInput(BaseModel):
    """Input for Citizen Dashboard Agent."""
    action: str = Field(...)
    dashboard_id: Optional[str] = Field(None)
    municipality_name: Optional[str] = Field(None)
    reporting_year: Optional[int] = Field(None)
    metric: Optional[DashboardMetric] = Field(None)
    time_series: Optional[TimeSeriesData] = Field(None)
    benchmark: Optional[ComparisonBenchmark] = Field(None)
    section: Optional[DashboardSection] = Field(None)
    user_id: Optional[str] = Field(None)

    @field_validator('action')
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid = {'create_dashboard', 'add_metric', 'add_time_series', 'add_benchmark',
                 'add_section', 'calculate_overall_score', 'update_trends',
                 'generate_public_view', 'get_dashboard', 'list_dashboards'}
        if v not in valid:
            raise ValueError(f"Invalid action: {v}")
        return v


class CitizenDashboardOutput(BaseModel):
    """Output from Citizen Dashboard Agent."""
    success: bool = Field(...)
    action: str = Field(...)
    dashboard: Optional[DashboardData] = Field(None)
    dashboards: Optional[List[DashboardData]] = Field(None)
    public_view: Optional[Dict[str, Any]] = Field(None)
    provenance_hash: Optional[str] = Field(None)
    calculation_trace: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(None)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    processing_time_ms: float = Field(default=0.0)


# Grade thresholds for overall climate performance
GRADE_THRESHOLDS = {
    "A+": 95, "A": 90, "A-": 85,
    "B+": 80, "B": 75, "B-": 70,
    "C+": 65, "C": 60, "C-": 55,
    "D+": 50, "D": 45, "D-": 40,
    "F": 0,
}


class CitizenClimateDashboardAgent(BaseAgent):
    """GL-REP-PUB-003: Citizen Climate Dashboard Agent"""

    AGENT_ID = "GL-REP-PUB-003"
    AGENT_NAME = "Citizen Climate Dashboard Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Public-facing climate dashboards for citizen engagement",
                version=self.VERSION,
            )
        super().__init__(config)
        self._dashboards: Dict[str, DashboardData] = {}

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        import time
        start_time = time.time()
        try:
            inp = CitizenDashboardInput(**input_data)
            handlers = {
                'create_dashboard': self._create_dashboard,
                'add_metric': self._add_metric,
                'add_time_series': self._add_time_series,
                'add_benchmark': self._add_benchmark,
                'add_section': self._add_section,
                'calculate_overall_score': self._calculate_score,
                'update_trends': self._update_trends,
                'generate_public_view': self._generate_public_view,
                'get_dashboard': self._get_dashboard,
                'list_dashboards': self._list_dashboards,
            }
            out = handlers[inp.action](inp)
            out.processing_time_ms = (time.time() - start_time) * 1000
            out.provenance_hash = self._hash_output(out)
            return AgentResult(success=out.success, data=out.model_dump(), error=out.error)
        except Exception as e:
            return AgentResult(success=False, error=str(e))

    def _create_dashboard(self, inp: CitizenDashboardInput) -> CitizenDashboardOutput:
        if not inp.municipality_name:
            return CitizenDashboardOutput(success=False, action='create_dashboard', error="Municipality name required")
        dashboard_id = f"CD-{inp.municipality_name.upper()[:3]}-{inp.reporting_year or 2024}"
        dashboard = DashboardData(
            dashboard_id=dashboard_id,
            municipality_name=inp.municipality_name,
            title=f"{inp.municipality_name} Climate Dashboard",
            subtitle="Track our community's progress on climate action",
            reporting_year=inp.reporting_year or 2024,
        )
        self._dashboards[dashboard_id] = dashboard
        return CitizenDashboardOutput(success=True, action='create_dashboard', dashboard=dashboard, calculation_trace=[f"Created {dashboard_id}"])

    def _add_metric(self, inp: CitizenDashboardInput) -> CitizenDashboardOutput:
        if not inp.dashboard_id or not inp.metric:
            return CitizenDashboardOutput(success=False, action='add_metric', error="Dashboard ID and metric required")
        dashboard = self._dashboards.get(inp.dashboard_id)
        if not dashboard:
            return CitizenDashboardOutput(success=False, action='add_metric', error="Dashboard not found")
        dashboard.metrics.append(inp.metric)
        dashboard.last_updated = DeterministicClock.now()
        return CitizenDashboardOutput(success=True, action='add_metric', dashboard=dashboard, calculation_trace=[f"Added metric {inp.metric.name}"])

    def _add_time_series(self, inp: CitizenDashboardInput) -> CitizenDashboardOutput:
        if not inp.dashboard_id or not inp.time_series:
            return CitizenDashboardOutput(success=False, action='add_time_series', error="Dashboard ID and time series required")
        dashboard = self._dashboards.get(inp.dashboard_id)
        if not dashboard:
            return CitizenDashboardOutput(success=False, action='add_time_series', error="Dashboard not found")
        dashboard.time_series.append(inp.time_series)
        dashboard.last_updated = DeterministicClock.now()
        return CitizenDashboardOutput(success=True, action='add_time_series', dashboard=dashboard, calculation_trace=[f"Added time series {inp.time_series.name}"])

    def _add_benchmark(self, inp: CitizenDashboardInput) -> CitizenDashboardOutput:
        if not inp.dashboard_id or not inp.benchmark:
            return CitizenDashboardOutput(success=False, action='add_benchmark', error="Dashboard ID and benchmark required")
        dashboard = self._dashboards.get(inp.dashboard_id)
        if not dashboard:
            return CitizenDashboardOutput(success=False, action='add_benchmark', error="Dashboard not found")
        bm = inp.benchmark
        if bm.benchmark_value > 0:
            bm.performance_vs_benchmark_pct = ((bm.municipality_value - bm.benchmark_value) / bm.benchmark_value) * 100
        dashboard.benchmarks.append(bm)
        dashboard.last_updated = DeterministicClock.now()
        return CitizenDashboardOutput(success=True, action='add_benchmark', dashboard=dashboard, calculation_trace=[f"Added benchmark {bm.name}"])

    def _add_section(self, inp: CitizenDashboardInput) -> CitizenDashboardOutput:
        if not inp.dashboard_id or not inp.section:
            return CitizenDashboardOutput(success=False, action='add_section', error="Dashboard ID and section required")
        dashboard = self._dashboards.get(inp.dashboard_id)
        if not dashboard:
            return CitizenDashboardOutput(success=False, action='add_section', error="Dashboard not found")
        dashboard.sections.append(inp.section)
        dashboard.last_updated = DeterministicClock.now()
        return CitizenDashboardOutput(success=True, action='add_section', dashboard=dashboard, calculation_trace=[f"Added section {inp.section.title}"])

    def _calculate_score(self, inp: CitizenDashboardInput) -> CitizenDashboardOutput:
        if not inp.dashboard_id:
            return CitizenDashboardOutput(success=False, action='calculate_overall_score', error="Dashboard ID required")
        dashboard = self._dashboards.get(inp.dashboard_id)
        if not dashboard:
            return CitizenDashboardOutput(success=False, action='calculate_overall_score', error="Dashboard not found")
        metric_scores = []
        for m in dashboard.metrics:
            if m.target_value and m.baseline_value and m.target_value != m.baseline_value:
                progress = (m.baseline_value - m.current_value) / (m.baseline_value - m.target_value) * 100
                metric_scores.append(min(100, max(0, progress)))
        dashboard.overall_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0
        for grade, threshold in sorted(GRADE_THRESHOLDS.items(), key=lambda x: -x[1]):
            if dashboard.overall_score >= threshold:
                dashboard.overall_grade = grade
                break
        dashboard.last_updated = DeterministicClock.now()
        return CitizenDashboardOutput(success=True, action='calculate_overall_score', dashboard=dashboard, calculation_trace=[f"Overall score: {dashboard.overall_score:.1f} ({dashboard.overall_grade})"])

    def _update_trends(self, inp: CitizenDashboardInput) -> CitizenDashboardOutput:
        if not inp.dashboard_id:
            return CitizenDashboardOutput(success=False, action='update_trends', error="Dashboard ID required")
        dashboard = self._dashboards.get(inp.dashboard_id)
        if not dashboard:
            return CitizenDashboardOutput(success=False, action='update_trends', error="Dashboard not found")
        for metric in dashboard.metrics:
            ts = next((t for t in dashboard.time_series if t.metric_id == metric.metric_id), None)
            if ts and len(ts.data_points) >= 2:
                sorted_points = sorted(ts.data_points, key=lambda x: x.date)
                old_val = sorted_points[0].value
                new_val = sorted_points[-1].value
                if old_val > 0:
                    metric.trend_pct_change = ((new_val - old_val) / old_val) * 100
                    # For emissions, decreasing is improving
                    if metric.category == MetricCategory.EMISSIONS:
                        if metric.trend_pct_change < -2:
                            metric.trend = TrendDirection.IMPROVING
                        elif metric.trend_pct_change > 2:
                            metric.trend = TrendDirection.DECLINING
                        else:
                            metric.trend = TrendDirection.STABLE
                    else:
                        if metric.trend_pct_change > 2:
                            metric.trend = TrendDirection.IMPROVING
                        elif metric.trend_pct_change < -2:
                            metric.trend = TrendDirection.DECLINING
                        else:
                            metric.trend = TrendDirection.STABLE
        dashboard.last_updated = DeterministicClock.now()
        return CitizenDashboardOutput(success=True, action='update_trends', dashboard=dashboard, calculation_trace=["Trends updated"])

    def _generate_public_view(self, inp: CitizenDashboardInput) -> CitizenDashboardOutput:
        if not inp.dashboard_id:
            return CitizenDashboardOutput(success=False, action='generate_public_view', error="Dashboard ID required")
        dashboard = self._dashboards.get(inp.dashboard_id)
        if not dashboard:
            return CitizenDashboardOutput(success=False, action='generate_public_view', error="Dashboard not found")
        public_view = {
            "title": dashboard.title,
            "subtitle": dashboard.subtitle,
            "municipality": dashboard.municipality_name,
            "year": dashboard.reporting_year,
            "overall_grade": dashboard.overall_grade,
            "overall_score": dashboard.overall_score,
            "last_updated": dashboard.last_updated.isoformat(),
            "metrics_summary": [
                {
                    "name": m.name,
                    "value": m.current_value,
                    "unit": m.unit,
                    "trend": m.trend.value,
                    "category": m.category.value,
                }
                for m in dashboard.metrics
            ],
            "sections": [
                {"title": s.title, "description": s.description}
                for s in dashboard.sections if s.visible
            ],
        }
        return CitizenDashboardOutput(success=True, action='generate_public_view', dashboard=dashboard, public_view=public_view, calculation_trace=["Public view generated"])

    def _get_dashboard(self, inp: CitizenDashboardInput) -> CitizenDashboardOutput:
        dashboard = self._dashboards.get(inp.dashboard_id) if inp.dashboard_id else None
        if not dashboard:
            return CitizenDashboardOutput(success=False, action='get_dashboard', error="Dashboard not found")
        return CitizenDashboardOutput(success=True, action='get_dashboard', dashboard=dashboard)

    def _list_dashboards(self, inp: CitizenDashboardInput) -> CitizenDashboardOutput:
        return CitizenDashboardOutput(success=True, action='list_dashboards', dashboards=list(self._dashboards.values()))

    def _hash_output(self, output: CitizenDashboardOutput) -> str:
        content = {"action": output.action, "success": output.success}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
