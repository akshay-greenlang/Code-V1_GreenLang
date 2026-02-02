# -*- coding: utf-8 -*-
"""
GreenLang Operations Layer Agents
=================================

The Operations Layer provides real-time operational monitoring, optimization,
and continuous improvement agents for GreenLang Climate OS.

Agents:
    GL-OPS-X-001: Real-time Emissions Monitor - Monitors emissions in real-time
    GL-OPS-X-002: Alert & Anomaly Agent - Detects anomalies in emissions data
    GL-OPS-X-003: Optimization Scheduler - Schedules operations for efficiency
    GL-OPS-X-004: Demand Response Agent - Manages demand response
    GL-OPS-X-005: Continuous Improvement Agent - Identifies improvement opportunities
    GL-OPS-X-006: Operational Benchmarking Agent - Benchmarks operations
"""

from greenlang.agents.operations.realtime_emissions_monitor import (
    RealtimeEmissionsMonitor,
    EmissionsMonitorInput,
    EmissionsMonitorOutput,
    EmissionsReading,
    EmissionsSource,
    AggregationPeriod,
    MonitoringStatus,
    EmissionsTrend,
)

from greenlang.agents.operations.alert_anomaly_agent import (
    AlertAnomalyAgent,
    AlertAnomalyInput,
    AlertAnomalyOutput,
    AnomalyType,
    AnomalySeverity,
    AnomalyDetection,
    AlertConfiguration,
    AnomalyPattern,
    DetectionMethod,
)

from greenlang.agents.operations.optimization_scheduler import (
    OptimizationScheduler,
    SchedulerInput,
    SchedulerOutput,
    ScheduleEntry,
    OptimizationGoal,
    ScheduleConstraint,
    ResourceAllocation,
    SchedulePeriod,
)

from greenlang.agents.operations.demand_response_agent import (
    DemandResponseAgent,
    DemandResponseInput,
    DemandResponseOutput,
    DemandEvent,
    ResponseStrategy,
    LoadCurtailment,
    DemandForecast,
    GridSignal,
)

from greenlang.agents.operations.continuous_improvement_agent import (
    ContinuousImprovementAgent,
    ImprovementInput,
    ImprovementOutput,
    ImprovementOpportunity,
    ImprovementCategory,
    ImprovementPriority,
    ImplementationStatus,
    ImpactAssessment,
)

from greenlang.agents.operations.operational_benchmarking_agent import (
    OperationalBenchmarkingAgent,
    BenchmarkingInput,
    BenchmarkingOutput,
    BenchmarkMetric,
    BenchmarkComparison,
    PerformanceQuartile,
    BenchmarkSource,
    GapAnalysis,
)

__all__ = [
    # Real-time Emissions Monitor (GL-OPS-X-001)
    "RealtimeEmissionsMonitor",
    "EmissionsMonitorInput",
    "EmissionsMonitorOutput",
    "EmissionsReading",
    "EmissionsSource",
    "AggregationPeriod",
    "MonitoringStatus",
    "EmissionsTrend",
    # Alert & Anomaly Agent (GL-OPS-X-002)
    "AlertAnomalyAgent",
    "AlertAnomalyInput",
    "AlertAnomalyOutput",
    "AnomalyType",
    "AnomalySeverity",
    "AnomalyDetection",
    "AlertConfiguration",
    "AnomalyPattern",
    "DetectionMethod",
    # Optimization Scheduler (GL-OPS-X-003)
    "OptimizationScheduler",
    "SchedulerInput",
    "SchedulerOutput",
    "ScheduleEntry",
    "OptimizationGoal",
    "ScheduleConstraint",
    "ResourceAllocation",
    "SchedulePeriod",
    # Demand Response Agent (GL-OPS-X-004)
    "DemandResponseAgent",
    "DemandResponseInput",
    "DemandResponseOutput",
    "DemandEvent",
    "ResponseStrategy",
    "LoadCurtailment",
    "DemandForecast",
    "GridSignal",
    # Continuous Improvement Agent (GL-OPS-X-005)
    "ContinuousImprovementAgent",
    "ImprovementInput",
    "ImprovementOutput",
    "ImprovementOpportunity",
    "ImprovementCategory",
    "ImprovementPriority",
    "ImplementationStatus",
    "ImpactAssessment",
    # Operational Benchmarking Agent (GL-OPS-X-006)
    "OperationalBenchmarkingAgent",
    "BenchmarkingInput",
    "BenchmarkingOutput",
    "BenchmarkMetric",
    "BenchmarkComparison",
    "PerformanceQuartile",
    "BenchmarkSource",
    "GapAnalysis",
]
