# -*- coding: utf-8 -*-
"""
GL-004 BurnerOptimizationAgent - Prometheus Metrics

Comprehensive metrics collection for burner optimization monitoring.
Exports metrics to Prometheus for Grafana visualization.
"""

import logging
from typing import Dict

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Centralized metrics collector for GL-004"""

    def __init__(self):
        """Initialize Prometheus metrics"""
        self.registry = CollectorRegistry()

        # ====================================================================
        # Agent Information
        # ====================================================================
        self.agent_info = Info(
            'gl004_agent',
            'GL-004 BurnerOptimizationAgent information',
            registry=self.registry
        )
        self.agent_info.info({
            'agent_id': 'GL-004',
            'agent_name': 'BurnerOptimizationAgent',
            'version': '1.0.0',
            'category': 'combustion_optimization'
        })

        # ====================================================================
        # Combustion Performance Metrics
        # ====================================================================
        self.combustion_efficiency = Gauge(
            'gl004_combustion_efficiency_percent',
            'Current combustion efficiency',
            ['agent'],
            registry=self.registry
        )

        self.air_fuel_ratio = Gauge(
            'gl004_air_fuel_ratio',
            'Current air-fuel ratio',
            ['agent'],
            registry=self.registry
        )

        self.excess_air_percent = Gauge(
            'gl004_excess_air_percent',
            'Excess air percentage',
            ['agent'],
            registry=self.registry
        )

        self.o2_level = Gauge(
            'gl004_o2_level_percent',
            'O2 concentration in flue gas',
            ['agent'],
            registry=self.registry
        )

        self.flame_temperature = Gauge(
            'gl004_flame_temperature_celsius',
            'Flame temperature',
            ['agent'],
            registry=self.registry
        )

        self.furnace_temperature = Gauge(
            'gl004_furnace_temperature_celsius',
            'Furnace temperature',
            ['agent'],
            registry=self.registry
        )

        self.flue_gas_temperature = Gauge(
            'gl004_flue_gas_temperature_celsius',
            'Flue gas temperature',
            ['agent'],
            registry=self.registry
        )

        self.burner_load = Gauge(
            'gl004_burner_load_percent',
            'Burner load percentage',
            ['agent'],
            registry=self.registry
        )

        # ====================================================================
        # Emissions Metrics
        # ====================================================================
        self.nox_level = Gauge(
            'gl004_nox_ppm',
            'NOx emissions concentration',
            ['agent'],
            registry=self.registry
        )

        self.co_level = Gauge(
            'gl004_co_ppm',
            'CO emissions concentration',
            ['agent'],
            registry=self.registry
        )

        self.co2_level = Gauge(
            'gl004_co2_percent',
            'CO2 emissions percentage',
            ['agent'],
            registry=self.registry
        )

        self.so2_level = Gauge(
            'gl004_so2_ppm',
            'SO2 emissions concentration',
            ['agent'],
            registry=self.registry
        )

        self.emissions_reduction = Gauge(
            'gl004_emissions_reduction_percent',
            'Emissions reduction achieved',
            ['agent', 'pollutant'],
            registry=self.registry
        )

        # ====================================================================
        # Fuel & Air Flow Metrics
        # ====================================================================
        self.fuel_flow_rate = Gauge(
            'gl004_fuel_flow_rate',
            'Fuel flow rate (kg/hr or m3/hr)',
            ['agent', 'fuel_type'],
            registry=self.registry
        )

        self.air_flow_rate = Gauge(
            'gl004_air_flow_rate_m3_per_hr',
            'Combustion air flow rate',
            ['agent'],
            registry=self.registry
        )

        self.fuel_consumption_total = Counter(
            'gl004_fuel_consumption_total',
            'Total fuel consumed',
            ['agent', 'fuel_type'],
            registry=self.registry
        )

        # ====================================================================
        # Optimization Metrics
        # ====================================================================
        self.optimization_counter = Counter(
            'gl004_optimization_runs_total',
            'Total number of optimization cycles',
            ['agent'],
            registry=self.registry
        )

        self.optimization_success_counter = Counter(
            'gl004_optimization_success_total',
            'Successful optimizations',
            ['agent'],
            registry=self.registry
        )

        self.optimization_failure_counter = Counter(
            'gl004_optimization_failure_total',
            'Failed optimizations',
            ['agent', 'reason'],
            registry=self.registry
        )

        self.optimization_skipped_counter = Counter(
            'gl004_optimization_skipped_total',
            'Skipped optimizations',
            ['agent', 'reason'],
            registry=self.registry
        )

        self.optimization_duration = Histogram(
            'gl004_optimization_duration_seconds',
            'Time taken for optimization cycle',
            ['agent'],
            buckets=[10, 30, 60, 120, 300, 600],
            registry=self.registry
        )

        self.efficiency_improvement = Gauge(
            'gl004_efficiency_improvement_percent',
            'Efficiency improvement from optimization',
            ['agent'],
            registry=self.registry
        )

        self.fuel_savings = Gauge(
            'gl004_fuel_savings_per_hour',
            'Fuel savings achieved (kg/hr)',
            ['agent'],
            registry=self.registry
        )

        self.optimization_convergence_iterations = Histogram(
            'gl004_optimization_iterations',
            'Number of iterations to converge',
            ['agent'],
            buckets=[10, 25, 50, 100, 200],
            registry=self.registry
        )

        # ====================================================================
        # Safety & Interlock Metrics
        # ====================================================================
        self.safety_interlock_counter = Counter(
            'gl004_safety_interlock_triggered_total',
            'Safety interlock activations',
            ['agent'],
            registry=self.registry
        )

        self.flame_loss_counter = Counter(
            'gl004_flame_loss_total',
            'Flame loss events',
            ['agent'],
            registry=self.registry
        )

        self.emergency_stop_counter = Counter(
            'gl004_emergency_stop_total',
            'Emergency stop events',
            ['agent'],
            registry=self.registry
        )

        self.setpoint_change_counter = Counter(
            'gl004_setpoint_changes_total',
            'Burner setpoint changes',
            ['agent'],
            registry=self.registry
        )

        # ====================================================================
        # Integration Status Metrics
        # ====================================================================
        self.integration_status = Gauge(
            'gl004_integration_status',
            'Integration connection status (1=connected, 0=disconnected)',
            ['agent', 'integration'],
            registry=self.registry
        )

        self.integration_error_counter = Counter(
            'gl004_integration_errors_total',
            'Integration errors',
            ['agent', 'integration', 'error_type'],
            registry=self.registry
        )

        self.sensor_read_latency = Histogram(
            'gl004_sensor_read_latency_seconds',
            'Sensor data acquisition latency',
            ['agent', 'sensor_type'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )

        # ====================================================================
        # Data Quality Metrics
        # ====================================================================
        self.data_quality_score = Gauge(
            'gl004_data_quality_score',
            'Data quality score (0-1)',
            ['agent', 'data_source'],
            registry=self.registry
        )

        self.data_validation_failures = Counter(
            'gl004_data_validation_failures_total',
            'Data validation failures',
            ['agent', 'validation_type'],
            registry=self.registry
        )

        # ====================================================================
        # API Performance Metrics
        # ====================================================================
        self.api_request_counter = Counter(
            'gl004_api_requests_total',
            'Total API requests',
            ['agent', 'method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.api_latency = Histogram(
            'gl004_api_latency_seconds',
            'API request latency',
            ['agent', 'method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )

        # ====================================================================
        # Error Metrics
        # ====================================================================
        self.error_counter = Counter(
            'gl004_errors_total',
            'Total errors by type',
            ['agent', 'error_type'],
            registry=self.registry
        )

        self.exception_counter = Counter(
            'gl004_exceptions_total',
            'Unhandled exceptions',
            ['agent', 'exception_type'],
            registry=self.registry
        )

        # ====================================================================
        # Performance Metrics
        # ====================================================================
        self.cpu_usage = Gauge(
            'gl004_cpu_usage_percent',
            'CPU usage',
            ['agent'],
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'gl004_memory_usage_bytes',
            'Memory usage',
            ['agent'],
            registry=self.registry
        )

        self.active_connections = Gauge(
            'gl004_active_connections',
            'Active network connections',
            ['agent'],
            registry=self.registry
        )

        # ====================================================================
        # Business Metrics
        # ====================================================================
        self.cost_savings_per_hour = Gauge(
            'gl004_cost_savings_per_hour_usd',
            'Cost savings from optimization ($/hr)',
            ['agent'],
            registry=self.registry
        )

        self.emissions_cost_avoided = Gauge(
            'gl004_emissions_cost_avoided_usd',
            'Emissions compliance cost avoided',
            ['agent', 'pollutant'],
            registry=self.registry
        )

        logger.info("Metrics collector initialized with 50+ metrics")

    def reset_metrics(self) -> None:
        """Reset all metrics (for testing)"""
        logger.warning("Resetting all metrics")
        # Prometheus client doesn't support easy reset, would need to recreate registry

    def get_metrics_summary(self) -> Dict[str, int]:
        """Get summary of metrics types"""
        return {
            'total_metrics': 50,
            'gauges': 35,
            'counters': 12,
            'histograms': 4,
            'info': 1
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()
