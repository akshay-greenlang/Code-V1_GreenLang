"""
Grafana Dashboard Definitions
=============================

Comprehensive dashboard configurations for:
- Executive dashboard
- Operations dashboard
- Agent performance dashboard
- Quality dashboard
- Financial dashboard

Author: GL-DevOpsEngineer
"""

import json
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


class DashboardType(Enum):
    """Dashboard types for different audiences"""
    EXECUTIVE = "executive"
    OPERATIONS = "operations"
    AGENT_PERFORMANCE = "agent_performance"
    QUALITY = "quality"
    FINANCIAL = "financial"


@dataclass
class Panel:
    """Grafana panel definition"""
    title: str
    type: str  # graph, singlestat, table, heatmap
    datasource: str
    targets: List[Dict[str, Any]]
    gridPos: Dict[str, int]
    options: Dict[str, Any] = field(default_factory=dict)
    fieldConfig: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Grafana dashboard definition"""
    uid: str
    title: str
    description: str
    tags: List[str]
    panels: List[Panel]
    refresh: str = "10s"
    time: Dict[str, str] = field(default_factory=lambda: {"from": "now-6h", "to": "now"})
    templating: Dict[str, Any] = field(default_factory=dict)


class DashboardGenerator:
    """Generate Grafana dashboard configurations"""

    def __init__(self, datasource: str = "Prometheus"):
        """
        Initialize dashboard generator

        Args:
            datasource: Default datasource name
        """
        self.datasource = datasource

    def generate_executive_dashboard(self) -> Dashboard:
        """Generate executive-level dashboard"""
        panels = [
            # KPI Summary
            Panel(
                title="Active Agents",
                type="stat",
                datasource=self.datasource,
                targets=[{
                    "expr": 'sum(greenlang_agents_agent_count)',
                    "refId": "A"
                }],
                gridPos={"h": 4, "w": 6, "x": 0, "y": 0},
                fieldConfig={
                    "defaults": {
                        "mappings": [],
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "red", "value": 0}
                            ]
                        },
                        "unit": "short"
                    }
                }
            ),

            # Throughput
            Panel(
                title="Request Throughput",
                type="stat",
                datasource=self.datasource,
                targets=[{
                    "expr": 'rate(greenlang_agents_messages_processed[5m])',
                    "refId": "A"
                }],
                gridPos={"h": 4, "w": 6, "x": 6, "y": 0},
                fieldConfig={
                    "defaults": {
                        "unit": "reqps"
                    }
                }
            ),

            # Availability
            Panel(
                title="System Availability",
                type="gauge",
                datasource=self.datasource,
                targets=[{
                    "expr": 'avg_over_time(up[24h]) * 100',
                    "refId": "A"
                }],
                gridPos={"h": 4, "w": 6, "x": 12, "y": 0},
                fieldConfig={
                    "defaults": {
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 95},
                                {"color": "green", "value": 99}
                            ]
                        }
                    }
                }
            ),

            # Cost Metrics
            Panel(
                title="Daily Cost Trend",
                type="graph",
                datasource=self.datasource,
                targets=[{
                    "expr": 'sum(increase(greenlang_agents_api_calls[24h])) * 0.002',
                    "legendFormat": "API Costs",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 12, "x": 0, "y": 4}
            ),

            # Business Impact
            Panel(
                title="Reports Generated",
                type="graph",
                datasource=self.datasource,
                targets=[{
                    "expr": 'sum(increase(greenlang_agents_reports_generated[1h]))',
                    "legendFormat": "{{report_type}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 4}
            )
        ]

        return Dashboard(
            uid="exec-dashboard",
            title="Executive Dashboard",
            description="High-level KPIs and business metrics",
            tags=["executive", "kpi"],
            panels=panels,
            refresh="1m"
        )

    def generate_operations_dashboard(self) -> Dashboard:
        """Generate operations team dashboard"""
        panels = [
            # Service Health Matrix
            Panel(
                title="Service Health",
                type="table",
                datasource=self.datasource,
                targets=[{
                    "expr": 'up',
                    "format": "table",
                    "instant": True,
                    "refId": "A"
                }],
                gridPos={"h": 6, "w": 12, "x": 0, "y": 0}
            ),

            # Latency Metrics
            Panel(
                title="Request Latency",
                type="graph",
                datasource=self.datasource,
                targets=[
                    {
                        "expr": 'histogram_quantile(0.5, greenlang_agents_task_completion_time_bucket)',
                        "legendFormat": "P50",
                        "refId": "A"
                    },
                    {
                        "expr": 'histogram_quantile(0.95, greenlang_agents_task_completion_time_bucket)',
                        "legendFormat": "P95",
                        "refId": "B"
                    },
                    {
                        "expr": 'histogram_quantile(0.99, greenlang_agents_task_completion_time_bucket)',
                        "legendFormat": "P99",
                        "refId": "C"
                    }
                ],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0},
                options={
                    "tooltip": {"mode": "multi"},
                    "legend": {"displayMode": "list"}
                }
            ),

            # Error Rate
            Panel(
                title="Error Rate",
                type="graph",
                datasource=self.datasource,
                targets=[{
                    "expr": 'sum(rate(greenlang_agents_error_rate[5m])) by (error_type)',
                    "legendFormat": "{{error_type}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 12, "x": 0, "y": 6},
                fieldConfig={
                    "defaults": {
                        "unit": "percent"
                    }
                }
            ),

            # Resource Utilization
            Panel(
                title="Resource Utilization",
                type="graph",
                datasource=self.datasource,
                targets=[
                    {
                        "expr": 'avg(greenlang_agents_cpu_utilization)',
                        "legendFormat": "CPU %",
                        "refId": "A"
                    },
                    {
                        "expr": 'avg(greenlang_agents_memory_usage_bytes) / 1024 / 1024',
                        "legendFormat": "Memory MB",
                        "refId": "B"
                    }
                ],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 8}
            ),

            # Alert Summary
            Panel(
                title="Active Alerts",
                type="table",
                datasource=self.datasource,
                targets=[{
                    "expr": 'ALERTS{alertstate="firing"}',
                    "format": "table",
                    "instant": True,
                    "refId": "A"
                }],
                gridPos={"h": 6, "w": 24, "x": 0, "y": 16}
            )
        ]

        return Dashboard(
            uid="ops-dashboard",
            title="Operations Dashboard",
            description="System health and performance metrics",
            tags=["operations", "monitoring"],
            panels=panels,
            refresh="10s"
        )

    def generate_agent_performance_dashboard(self) -> Dashboard:
        """Generate agent performance dashboard"""
        panels = [
            # Agent Lifecycle
            Panel(
                title="Agent State Distribution",
                type="piechart",
                datasource=self.datasource,
                targets=[{
                    "expr": 'sum(greenlang_agents_agent_count) by (state)',
                    "legendFormat": "{{state}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 8, "x": 0, "y": 0}
            ),

            # Message Flow
            Panel(
                title="Inter-Agent Communication",
                type="graph",
                datasource=self.datasource,
                targets=[{
                    "expr": 'rate(greenlang_agents_messages_processed[5m])',
                    "legendFormat": "{{agent_type}} - {{message_type}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 16, "x": 8, "y": 0}
            ),

            # Task Completion
            Panel(
                title="Task Success Rate",
                type="graph",
                datasource=self.datasource,
                targets=[{
                    "expr": '1 - (rate(greenlang_agents_error_rate[5m]) / rate(greenlang_agents_messages_processed[5m]))',
                    "legendFormat": "Success Rate",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 12, "x": 0, "y": 8},
                fieldConfig={
                    "defaults": {
                        "unit": "percentunit",
                        "min": 0,
                        "max": 1
                    }
                }
            ),

            # LLM Usage
            Panel(
                title="LLM API Usage",
                type="graph",
                datasource=self.datasource,
                targets=[
                    {
                        "expr": 'sum(rate(greenlang_agents_api_calls{api="llm"}[5m])) by (method)',
                        "legendFormat": "{{method}}",
                        "refId": "A"
                    }
                ],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 8}
            ),

            # Memory Utilization
            Panel(
                title="Agent Memory Usage",
                type="heatmap",
                datasource=self.datasource,
                targets=[{
                    "expr": 'greenlang_agents_memory_usage_bytes',
                    "format": "heatmap",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 16},
                options={
                    "calculate": False,
                    "cellGap": 1,
                    "cellValues": {"decimals": 0},
                    "color": {
                        "mode": "scheme",
                        "scheme": "Spectral",
                        "steps": 128
                    }
                }
            )
        ]

        return Dashboard(
            uid="agent-perf-dashboard",
            title="Agent Performance Dashboard",
            description="Detailed agent metrics and behavior",
            tags=["agents", "performance"],
            panels=panels,
            refresh="30s"
        )

    def generate_quality_dashboard(self) -> Dashboard:
        """Generate quality metrics dashboard"""
        panels = [
            # Quality Score
            Panel(
                title="Overall Quality Score",
                type="gauge",
                datasource=self.datasource,
                targets=[{
                    "expr": 'greenlang_agents_quality_score_overall',
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 8, "x": 0, "y": 0},
                fieldConfig={
                    "defaults": {
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "green", "value": 85}
                            ]
                        }
                    }
                }
            ),

            # Quality Dimensions
            Panel(
                title="Quality Dimensions",
                type="bargauge",
                datasource=self.datasource,
                targets=[{
                    "expr": 'greenlang_agents_quality_functional_quality',
                    "legendFormat": "Functional",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 16, "x": 8, "y": 0},
                options={
                    "orientation": "horizontal",
                    "displayMode": "gradient"
                }
            ),

            # Test Coverage
            Panel(
                title="Test Coverage",
                type="stat",
                datasource=self.datasource,
                targets=[{
                    "expr": 'greenlang_agents_test_coverage',
                    "refId": "A"
                }],
                gridPos={"h": 4, "w": 6, "x": 0, "y": 8},
                fieldConfig={
                    "defaults": {
                        "unit": "percent",
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": None},
                                {"color": "yellow", "value": 70},
                                {"color": "green", "value": 85}
                            ]
                        }
                    }
                }
            ),

            # Code Quality Metrics
            Panel(
                title="Code Quality Trends",
                type="graph",
                datasource=self.datasource,
                targets=[
                    {
                        "expr": 'greenlang_agents_quality_maintainability',
                        "legendFormat": "Maintainability",
                        "refId": "A"
                    },
                    {
                        "expr": 'greenlang_agents_quality_security',
                        "legendFormat": "Security",
                        "refId": "B"
                    }
                ],
                gridPos={"h": 8, "w": 18, "x": 6, "y": 8}
            )
        ]

        return Dashboard(
            uid="quality-dashboard",
            title="Quality Dashboard",
            description="Code quality and testing metrics",
            tags=["quality", "testing"],
            panels=panels,
            refresh="1h"
        )

    def generate_financial_dashboard(self) -> Dashboard:
        """Generate financial metrics dashboard"""
        panels = [
            # Infrastructure Costs
            Panel(
                title="Infrastructure Costs",
                type="graph",
                datasource=self.datasource,
                targets=[
                    {
                        "expr": 'sum(greenlang_agents_pod_count) * 0.05',
                        "legendFormat": "Compute",
                        "refId": "A"
                    },
                    {
                        "expr": 'sum(greenlang_agents_database_connections) * 0.02',
                        "legendFormat": "Database",
                        "refId": "B"
                    }
                ],
                gridPos={"h": 8, "w": 12, "x": 0, "y": 0}
            ),

            # API Costs
            Panel(
                title="API Costs",
                type="graph",
                datasource=self.datasource,
                targets=[{
                    "expr": 'sum(increase(greenlang_agents_api_calls[1h])) by (api) * 0.002',
                    "legendFormat": "{{api}}",
                    "refId": "A"
                }],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0}
            ),

            # Per-Agent Costs
            Panel(
                title="Cost per Agent",
                type="stat",
                datasource=self.datasource,
                targets=[{
                    "expr": '(sum(rate(greenlang_agents_api_calls[1h])) * 0.002) / sum(greenlang_agents_agent_count)',
                    "refId": "A"
                }],
                gridPos={"h": 4, "w": 6, "x": 0, "y": 8},
                fieldConfig={
                    "defaults": {
                        "unit": "currencyUSD"
                    }
                }
            ),

            # Cost Optimization
            Panel(
                title="Optimization Opportunities",
                type="table",
                datasource=self.datasource,
                targets=[{
                    "expr": 'topk(5, greenlang_agents_cache_hit_rate < 0.8)',
                    "format": "table",
                    "instant": True,
                    "refId": "A"
                }],
                gridPos={"h": 6, "w": 18, "x": 6, "y": 8}
            ),

            # Budget Tracking
            Panel(
                title="Budget vs Actual",
                type="bargauge",
                datasource=self.datasource,
                targets=[{
                    "expr": 'sum(increase(greenlang_agents_api_calls[30d])) * 0.002',
                    "refId": "A"
                }],
                gridPos={"h": 6, "w": 24, "x": 0, "y": 14},
                options={
                    "orientation": "horizontal",
                    "displayMode": "gradient"
                }
            )
        ]

        return Dashboard(
            uid="financial-dashboard",
            title="Financial Dashboard",
            description="Cost tracking and optimization",
            tags=["financial", "costs"],
            panels=panels,
            refresh="1h"
        )

    def generate_dashboard(self, dashboard_type: DashboardType) -> Dashboard:
        """Generate dashboard by type"""
        generators = {
            DashboardType.EXECUTIVE: self.generate_executive_dashboard,
            DashboardType.OPERATIONS: self.generate_operations_dashboard,
            DashboardType.AGENT_PERFORMANCE: self.generate_agent_performance_dashboard,
            DashboardType.QUALITY: self.generate_quality_dashboard,
            DashboardType.FINANCIAL: self.generate_financial_dashboard
        }

        generator = generators.get(dashboard_type)
        if not generator:
            raise ValueError(f"Unknown dashboard type: {dashboard_type}")

        return generator()

    def export_to_json(self, dashboard: Dashboard) -> str:
        """Export dashboard to Grafana JSON format"""
        return json.dumps({
            "dashboard": {
                "uid": dashboard.uid,
                "title": dashboard.title,
                "description": dashboard.description,
                "tags": dashboard.tags,
                "timezone": "browser",
                "panels": [
                    {
                        "id": i + 1,
                        "title": panel.title,
                        "type": panel.type,
                        "datasource": panel.datasource,
                        "targets": panel.targets,
                        "gridPos": panel.gridPos,
                        "options": panel.options,
                        "fieldConfig": panel.fieldConfig
                    }
                    for i, panel in enumerate(dashboard.panels)
                ],
                "time": dashboard.time,
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h"],
                    "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d"]
                },
                "templating": dashboard.templating,
                "annotations": {
                    "list": [
                        {
                            "builtIn": 1,
                            "datasource": "-- Grafana --",
                            "enable": True,
                            "hide": True,
                            "iconColor": "rgba(0, 211, 255, 1)",
                            "name": "Annotations & Alerts",
                            "type": "dashboard"
                        }
                    ]
                },
                "refresh": dashboard.refresh,
                "schemaVersion": 38,
                "version": 1
            },
            "overwrite": True
        }, indent=2)


def generate_dashboard(
    dashboard_type: DashboardType,
    datasource: str = "Prometheus"
) -> Dashboard:
    """Generate a dashboard by type"""
    generator = DashboardGenerator(datasource)
    return generator.generate_dashboard(dashboard_type)


def export_grafana_config(
    dashboard_type: DashboardType,
    datasource: str = "Prometheus"
) -> str:
    """Export dashboard configuration as JSON"""
    generator = DashboardGenerator(datasource)
    dashboard = generator.generate_dashboard(dashboard_type)
    return generator.export_to_json(dashboard)