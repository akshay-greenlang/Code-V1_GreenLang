# -*- coding: utf-8 -*-
"""
GreenLang Agent-Specific Metric Extensions
===========================================

Agent-specific metric extensions that build upon StandardAgentMetrics
to add domain-specific observability for specialized GreenLang agents.

This module provides extension classes for:
- GL-001 ProcessHeatOrchestrator (Multi-plant coordination)
- GL-002 BoilerOptimizer (Combustion optimization)
- GL-006 HeatRecoveryMaximizer (Waste heat recovery)

Each extension adds 2-20 additional metrics specific to the agent's domain,
while inheriting all 71 baseline metrics from StandardAgentMetrics.

Usage:
    >>> from greenlang.monitoring.agent_extensions import ProcessHeatMetrics
    >>>
    >>> metrics = ProcessHeatMetrics(
    ...     agent_id="GL-001",
    ...     agent_name="ProcessHeatOrchestrator",
    ...     codename="THERMOSYNC",
    ...     version="1.0.0",
    ...     domain="multi_plant_heat_coordination"
    ... )
    >>>
    >>> # Use standard metrics
    >>> with metrics.track_request("POST", "/api/v1/orchestrate"):
    ...     result = orchestrator.orchestrate()
    >>>
    >>> # Use agent-specific metrics
    >>> metrics.update_plant_metrics("PLANT-001", "MainPlant", {
    ...     "thermal_efficiency_percent": 87.5,
    ...     "heat_generation_mw": 150.0
    ... })

Author: GreenLang Team
License: Proprietary
"""

import logging
from typing import Dict, Any, Optional

from greenlang.monitoring.standard_metrics import StandardAgentMetrics, PROMETHEUS_AVAILABLE

if PROMETHEUS_AVAILABLE:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
else:
    # Use stub classes
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class CollectorRegistry:
        pass

logger = logging.getLogger(__name__)


class ProcessHeatMetrics(StandardAgentMetrics):
    """
    GL-001 ProcessHeatOrchestrator specific metrics.

    Extends StandardAgentMetrics with multi-plant coordination metrics:
    - Plant-level thermal efficiency
    - Cross-plant heat transfer
    - SCADA integration health
    - Heat distribution optimization
    - Emissions compliance

    Total metrics: 71 (baseline) + 15 (agent-specific) = 86 metrics
    """

    def __init__(
        self,
        agent_id: str = "GL-001",
        agent_name: str = "ProcessHeatOrchestrator",
        codename: str = "THERMOSYNC",
        version: str = "1.0.0",
        domain: str = "multi_plant_heat_coordination",
        registry: Optional[CollectorRegistry] = None
    ):
        """Initialize GL-001 specific metrics."""
        super().__init__(agent_id, agent_name, codename, version, domain, registry)

        # Initialize agent-specific metrics
        self._init_plant_metrics()
        self._init_scada_metrics()
        self._init_heat_distribution_metrics()

        logger.info(f"ProcessHeatMetrics initialized: 86 total metrics (71 baseline + 15 specific)")

    def _init_plant_metrics(self):
        """Initialize plant-level metrics (5 metrics)."""

        # Metric 72: Active plants count
        self.active_plants_count = Gauge(
            f"{self.metric_prefix}_active_plants_count",
            "Number of active plants being coordinated",
            registry=self.registry
        )

        # Metric 73: Plant thermal efficiency
        self.plant_thermal_efficiency_percent = Gauge(
            f"{self.metric_prefix}_plant_thermal_efficiency_percent",
            "Plant-level thermal efficiency",
            ["plant_id", "plant_name"],
            registry=self.registry
        )

        # Metric 74: Plant heat generation
        self.plant_heat_generation_mw = Gauge(
            f"{self.metric_prefix}_plant_heat_generation_mw",
            "Total heat generation per plant",
            ["plant_id", "plant_name"],
            registry=self.registry
        )

        # Metric 75: Plant heat demand
        self.plant_heat_demand_mw = Gauge(
            f"{self.metric_prefix}_plant_heat_demand_mw",
            "Total heat demand per plant",
            ["plant_id", "plant_name"],
            registry=self.registry
        )

        # Metric 76: Cross-plant heat transfer
        self.cross_plant_heat_transfer_mw = Gauge(
            f"{self.metric_prefix}_cross_plant_heat_transfer_mw",
            "Heat transferred between plants",
            ["source_plant", "destination_plant"],
            registry=self.registry
        )

    def _init_scada_metrics(self):
        """Initialize SCADA integration metrics (5 metrics)."""

        # Metric 77: SCADA connection status
        self.scada_connection_status = Gauge(
            f"{self.metric_prefix}_scada_connection_status",
            "SCADA connection status (1=connected, 0=disconnected)",
            ["plant_id", "scada_system"],
            registry=self.registry
        )

        # Metric 78: SCADA data points received
        self.scada_data_points_received_total = Counter(
            f"{self.metric_prefix}_scada_data_points_received_total",
            "Total SCADA data points received",
            ["plant_id", "data_category"],
            registry=self.registry
        )

        # Metric 79: SCADA data latency
        self.scada_sync_latency_ms = Histogram(
            f"{self.metric_prefix}_scada_sync_latency_ms",
            "SCADA data synchronization latency in milliseconds",
            ["plant_id"],
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=self.registry
        )

        # Metric 80: SCADA data quality
        self.scada_data_quality_percent = Gauge(
            f"{self.metric_prefix}_scada_data_quality_percent",
            "SCADA data quality score",
            ["plant_id", "data_category"],
            registry=self.registry
        )

        # Metric 81: SCADA alarms active
        self.scada_alarms_active = Gauge(
            f"{self.metric_prefix}_scada_alarms_active",
            "Number of active SCADA alarms",
            ["plant_id", "severity"],
            registry=self.registry
        )

    def _init_heat_distribution_metrics(self):
        """Initialize heat distribution metrics (5 metrics)."""

        # Metric 82: Heat distribution optimization score
        self.heat_distribution_optimization_score = Gauge(
            f"{self.metric_prefix}_heat_distribution_optimization_score",
            "Heat distribution optimization score (0-100)",
            ["plant_id"],
            registry=self.registry
        )

        # Metric 83: Heat distribution optimization count
        self.heat_distribution_optimization_count = Counter(
            f"{self.metric_prefix}_heat_distribution_optimization_count",
            "Number of heat distribution optimizations performed",
            ["plant_id", "optimization_type"],
            registry=self.registry
        )

        # Metric 84: Emissions CO2 intensity
        self.emissions_co2_intensity_kg_mwh = Gauge(
            f"{self.metric_prefix}_emissions_co2_intensity_kg_mwh",
            "CO2 emissions intensity",
            ["plant_id"],
            registry=self.registry
        )

        # Metric 85: Aggregate thermal efficiency
        self.aggregate_thermal_efficiency_percent = Gauge(
            f"{self.metric_prefix}_aggregate_thermal_efficiency_percent",
            "Aggregate thermal efficiency across all plants",
            registry=self.registry
        )

        # Metric 86: Sub-agent coordination latency
        self.subagent_coordination_latency_ms = Histogram(
            f"{self.metric_prefix}_subagent_coordination_latency_ms",
            "Sub-agent coordination latency in milliseconds",
            ["agent_id"],
            buckets=[5, 10, 25, 50, 100, 250, 500, 1000],
            registry=self.registry
        )

    def update_plant_metrics(self, plant_id: str, plant_name: str, metrics: Dict[str, Any]):
        """Update plant-level metrics."""
        if "thermal_efficiency_percent" in metrics:
            self.plant_thermal_efficiency_percent.labels(
                plant_id=plant_id,
                plant_name=plant_name
            ).set(metrics["thermal_efficiency_percent"])

        if "heat_generation_mw" in metrics:
            self.plant_heat_generation_mw.labels(
                plant_id=plant_id,
                plant_name=plant_name
            ).set(metrics["heat_generation_mw"])

        if "heat_demand_mw" in metrics:
            self.plant_heat_demand_mw.labels(
                plant_id=plant_id,
                plant_name=plant_name
            ).set(metrics["heat_demand_mw"])

    def update_scada_metrics(self, plant_id: str, scada_system: str, metrics: Dict[str, Any]):
        """Update SCADA integration metrics."""
        if "connection_status" in metrics:
            self.scada_connection_status.labels(
                plant_id=plant_id,
                scada_system=scada_system
            ).set(1 if metrics["connection_status"] == "connected" else 0)

        if "data_quality_percent" in metrics:
            for category, quality in metrics["data_quality_percent"].items():
                self.scada_data_quality_percent.labels(
                    plant_id=plant_id,
                    data_category=category
                ).set(quality)

    def get_metrics_count(self) -> int:
        """Get total metrics count including agent-specific."""
        return 86  # 71 baseline + 15 agent-specific


class BoilerOptimizerMetrics(StandardAgentMetrics):
    """
    GL-002 BoilerOptimizer specific metrics.

    Extends StandardAgentMetrics with boiler combustion metrics:
    - Boiler efficiency
    - Fuel optimization
    - Combustion parameters
    - Stack emissions
    - Blowdown optimization

    Total metrics: 71 (baseline) + 10 (agent-specific) = 81 metrics
    """

    def __init__(
        self,
        agent_id: str = "GL-002",
        agent_name: str = "BoilerOptimizer",
        codename: str = "BURNRIGHT",
        version: str = "1.0.0",
        domain: str = "combustion_optimization",
        registry: Optional[CollectorRegistry] = None
    ):
        """Initialize GL-002 specific metrics."""
        super().__init__(agent_id, agent_name, codename, version, domain, registry)

        # Initialize agent-specific metrics
        self._init_boiler_metrics()
        self._init_combustion_metrics()
        self._init_fuel_metrics()

        logger.info(f"BoilerOptimizerMetrics initialized: 81 total metrics (71 baseline + 10 specific)")

    def _init_boiler_metrics(self):
        """Initialize boiler performance metrics (4 metrics)."""

        # Metric 72: Boiler efficiency
        self.boiler_efficiency_percent = Gauge(
            f"{self.metric_prefix}_boiler_efficiency_percent",
            "Boiler thermal efficiency percentage",
            ["boiler_id"],
            registry=self.registry
        )

        # Metric 73: Boiler efficiency calculated
        self.boiler_efficiency_calculated_total = Counter(
            f"{self.metric_prefix}_boiler_efficiency_calculated_total",
            "Number of boiler efficiency calculations performed",
            ["boiler_id"],
            registry=self.registry
        )

        # Metric 74: Steam generation rate
        self.steam_generation_rate_kg_hr = Gauge(
            f"{self.metric_prefix}_steam_generation_rate_kg_hr",
            "Steam generation rate in kg/hr",
            ["boiler_id"],
            registry=self.registry
        )

        # Metric 75: Fuel consumption rate
        self.fuel_consumption_rate_kg_hr = Gauge(
            f"{self.metric_prefix}_fuel_consumption_rate_kg_hr",
            "Fuel consumption rate in kg/hr",
            ["boiler_id", "fuel_type"],
            registry=self.registry
        )

    def _init_combustion_metrics(self):
        """Initialize combustion optimization metrics (3 metrics)."""

        # Metric 76: Excess air percentage
        self.excess_air_percent = Gauge(
            f"{self.metric_prefix}_excess_air_percent",
            "Excess air percentage in flue gas",
            ["boiler_id"],
            registry=self.registry
        )

        # Metric 77: Flue gas temperature
        self.flue_gas_temperature_celsius = Gauge(
            f"{self.metric_prefix}_flue_gas_temperature_celsius",
            "Flue gas temperature in Celsius",
            ["boiler_id"],
            registry=self.registry
        )

        # Metric 78: Stack O2 percentage
        self.stack_o2_percent = Gauge(
            f"{self.metric_prefix}_stack_o2_percent",
            "Stack oxygen percentage",
            ["boiler_id"],
            registry=self.registry
        )

    def _init_fuel_metrics(self):
        """Initialize fuel optimization metrics (3 metrics)."""

        # Metric 79: Fuel optimization savings
        self.fuel_optimization_savings_usd_total = Counter(
            f"{self.metric_prefix}_fuel_optimization_savings_usd_total",
            "Total fuel cost savings from optimization in USD",
            ["boiler_id"],
            registry=self.registry
        )

        # Metric 80: Fuel blend ratio
        self.fuel_blend_ratio_percent = Gauge(
            f"{self.metric_prefix}_fuel_blend_ratio_percent",
            "Fuel blend ratio percentage",
            ["boiler_id", "fuel_type"],
            registry=self.registry
        )

        # Metric 81: Fuel heating value
        self.fuel_heating_value_mj_kg = Gauge(
            f"{self.metric_prefix}_fuel_heating_value_mj_kg",
            "Fuel heating value in MJ/kg",
            ["boiler_id", "fuel_type"],
            registry=self.registry
        )

    def update_boiler_metrics(self, boiler_id: str, metrics: Dict[str, Any]):
        """Update boiler performance metrics."""
        if "efficiency_percent" in metrics:
            self.boiler_efficiency_percent.labels(boiler_id=boiler_id).set(
                metrics["efficiency_percent"]
            )
            self.boiler_efficiency_calculated_total.labels(boiler_id=boiler_id).inc()

        if "steam_generation_kg_hr" in metrics:
            self.steam_generation_rate_kg_hr.labels(boiler_id=boiler_id).set(
                metrics["steam_generation_kg_hr"]
            )

        if "excess_air_percent" in metrics:
            self.excess_air_percent.labels(boiler_id=boiler_id).set(
                metrics["excess_air_percent"]
            )

    def get_metrics_count(self) -> int:
        """Get total metrics count including agent-specific."""
        return 81  # 71 baseline + 10 agent-specific


class HeatRecoveryMetrics(StandardAgentMetrics):
    """
    GL-006 HeatRecoveryMaximizer specific metrics.

    Extends StandardAgentMetrics with heat recovery metrics:
    - Waste heat identification
    - Heat exchanger performance
    - Pinch analysis results
    - Economic ROI calculations
    - Exergy analysis

    Total metrics: 71 (baseline) + 12 (agent-specific) = 83 metrics
    """

    def __init__(
        self,
        agent_id: str = "GL-006",
        agent_name: str = "HeatRecoveryMaximizer",
        codename: str = "HEATRECLAIM",
        version: str = "1.0.0",
        domain: str = "heat_recovery",
        registry: Optional[CollectorRegistry] = None
    ):
        """Initialize GL-006 specific metrics."""
        super().__init__(agent_id, agent_name, codename, version, domain, registry)

        # Initialize agent-specific metrics
        self._init_heat_recovery_metrics()
        self._init_pinch_analysis_metrics()
        self._init_economic_metrics()

        logger.info(f"HeatRecoveryMetrics initialized: 83 total metrics (71 baseline + 12 specific)")

    def _init_heat_recovery_metrics(self):
        """Initialize heat recovery metrics (5 metrics)."""

        # Metric 72: Recoverable heat
        self.total_recoverable_heat_kw = Gauge(
            f"{self.metric_prefix}_total_recoverable_heat_kw",
            "Total recoverable heat in kW",
            registry=self.registry
        )

        # Metric 73: Actually recovered heat
        self.actual_recovered_heat_kw = Gauge(
            f"{self.metric_prefix}_actual_recovered_heat_kw",
            "Actually recovered heat in kW",
            registry=self.registry
        )

        # Metric 74: Heat recovery efficiency
        self.heat_recovery_efficiency_percent = Gauge(
            f"{self.metric_prefix}_heat_recovery_efficiency_percent",
            "Heat recovery efficiency percentage",
            registry=self.registry
        )

        # Metric 75: Heat recovery opportunities identified
        self.heat_recovery_opportunities_identified_total = Counter(
            f"{self.metric_prefix}_heat_recovery_opportunities_identified_total",
            "Total heat recovery opportunities identified",
            ["priority"],
            registry=self.registry
        )

        # Metric 76: Streams analyzed
        self.streams_analyzed_total = Counter(
            f"{self.metric_prefix}_streams_analyzed_total",
            "Total number of streams analyzed",
            ["stream_type"],
            registry=self.registry
        )

    def _init_pinch_analysis_metrics(self):
        """Initialize pinch analysis metrics (4 metrics)."""

        # Metric 77: Pinch analyses performed
        self.pinch_analyses_total = Counter(
            f"{self.metric_prefix}_pinch_analyses_total",
            "Total pinch analyses performed",
            registry=self.registry
        )

        # Metric 78: Pinch temperature
        self.pinch_temperature_celsius = Gauge(
            f"{self.metric_prefix}_pinch_temperature_celsius",
            "Calculated pinch temperature in Celsius",
            registry=self.registry
        )

        # Metric 79: Minimum hot utility
        self.min_hot_utility_kw = Gauge(
            f"{self.metric_prefix}_min_hot_utility_kw",
            "Minimum hot utility requirement in kW",
            registry=self.registry
        )

        # Metric 80: Minimum cold utility
        self.min_cold_utility_kw = Gauge(
            f"{self.metric_prefix}_min_cold_utility_kw",
            "Minimum cold utility requirement in kW",
            registry=self.registry
        )

    def _init_economic_metrics(self):
        """Initialize economic analysis metrics (3 metrics)."""

        # Metric 81: ROI calculated
        self.roi_calculated_total = Counter(
            f"{self.metric_prefix}_roi_calculated_total",
            "Number of ROI calculations performed",
            ["project_id"],
            registry=self.registry
        )

        # Metric 82: ROI percentage
        self.roi_percent = Gauge(
            f"{self.metric_prefix}_roi_percent",
            "Return on investment percentage",
            ["project_id"],
            registry=self.registry
        )

        # Metric 83: Payback period
        self.payback_years = Gauge(
            f"{self.metric_prefix}_payback_years",
            "Simple payback period in years",
            ["project_id"],
            registry=self.registry
        )

    def record_heat_recovery_opportunity(self, priority: str):
        """Record a heat recovery opportunity."""
        self.heat_recovery_opportunities_identified_total.labels(priority=priority).inc()

    def record_pinch_analysis(
        self,
        pinch_temp: float,
        hot_utility: float,
        cold_utility: float
    ):
        """Record pinch analysis results."""
        self.pinch_analyses_total.inc()
        self.pinch_temperature_celsius.set(pinch_temp)
        self.min_hot_utility_kw.set(hot_utility)
        self.min_cold_utility_kw.set(cold_utility)

    def record_roi_calculation(
        self,
        project_id: str,
        roi_percent: float,
        payback_years: float
    ):
        """Record ROI calculation results."""
        self.roi_calculated_total.labels(project_id=project_id).inc()
        self.roi_percent.labels(project_id=project_id).set(roi_percent)
        self.payback_years.labels(project_id=project_id).set(payback_years)

    def get_metrics_count(self) -> int:
        """Get total metrics count including agent-specific."""
        return 83  # 71 baseline + 12 agent-specific


__all__ = [
    'ProcessHeatMetrics',
    'BoilerOptimizerMetrics',
    'HeatRecoveryMetrics',
]
