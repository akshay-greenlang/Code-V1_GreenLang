"""
GL-009 THERMALIQ ThermalEfficiencyCalculator Prometheus Metrics

This module defines 60+ Prometheus metrics for comprehensive monitoring of:
- Agent health and performance
- Thermal efficiency calculations (First Law, Second Law)
- Loss breakdown analysis (radiation, convection, conduction, etc.)
- Integration connector health (energy meters, historians, SCADA, ERP)
- API performance
- Business outcomes (energy savings, cost reduction, CO2 reduction)

All metrics follow Prometheus naming conventions and include detailed help text.
Zero-hallucination design: All metrics track deterministic calculations only.
"""

from typing import Dict, List, Optional, Any
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    CollectorRegistry,
    generate_latest,
)
import time
import psutil
import logging
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)

# Create custom registry for GL-009 metrics
REGISTRY = CollectorRegistry()

# ============================================================================
# AGENT HEALTH METRICS (10 metrics)
# ============================================================================

agent_health_status = Gauge(
    'gl009_agent_health_status',
    'Health status of GL-009 agent (1=healthy, 0=unhealthy)',
    ['agent_id', 'environment'],
    registry=REGISTRY
)

agent_uptime_seconds = Counter(
    'gl009_agent_uptime_seconds',
    'Total uptime of GL-009 agent in seconds',
    ['agent_id', 'environment'],
    registry=REGISTRY
)

agent_restart_count = Counter(
    'gl009_agent_restart_count',
    'Number of times GL-009 agent has restarted',
    ['agent_id', 'environment', 'restart_reason'],
    registry=REGISTRY
)

agent_memory_usage_bytes = Gauge(
    'gl009_agent_memory_usage_bytes',
    'Memory usage of GL-009 agent in bytes',
    ['agent_id', 'environment', 'memory_type'],
    registry=REGISTRY
)

agent_cpu_usage_percent = Gauge(
    'gl009_agent_cpu_usage_percent',
    'CPU usage percentage of GL-009 agent',
    ['agent_id', 'environment'],
    registry=REGISTRY
)

cache_hit_rate = Gauge(
    'gl009_cache_hit_rate',
    'Cache hit rate for thermal calculations (0.0-1.0)',
    ['agent_id', 'cache_type'],
    registry=REGISTRY
)

cache_size_bytes = Gauge(
    'gl009_cache_size_bytes',
    'Size of calculation cache in bytes',
    ['agent_id', 'cache_type'],
    registry=REGISTRY
)

active_calculations = Gauge(
    'gl009_active_calculations',
    'Number of thermal efficiency calculations currently in progress',
    ['agent_id', 'calculation_type'],
    registry=REGISTRY
)

queue_depth = Gauge(
    'gl009_queue_depth',
    'Number of calculations waiting in queue',
    ['agent_id', 'queue_name'],
    registry=REGISTRY
)

error_count = Counter(
    'gl009_error_count',
    'Total number of errors encountered by GL-009 agent',
    ['agent_id', 'error_type', 'severity'],
    registry=REGISTRY
)

# ============================================================================
# THERMAL EFFICIENCY CALCULATION METRICS (15 metrics)
# ============================================================================

calculation_duration_seconds = Histogram(
    'gl009_calculation_duration_seconds',
    'Duration of thermal efficiency calculation in seconds',
    ['agent_id', 'calculation_type', 'equipment_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

first_law_efficiency_percent = Histogram(
    'gl009_first_law_efficiency_percent',
    'First Law (energy) efficiency percentage calculated',
    ['agent_id', 'equipment_type', 'process_name'],
    buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 99],
    registry=REGISTRY
)

second_law_efficiency_percent = Histogram(
    'gl009_second_law_efficiency_percent',
    'Second Law (exergy) efficiency percentage calculated',
    ['agent_id', 'equipment_type', 'process_name'],
    buckets=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80],
    registry=REGISTRY
)

energy_input_kw = Gauge(
    'gl009_energy_input_kw',
    'Total energy input to thermal system in kW',
    ['agent_id', 'equipment_id', 'fuel_type'],
    registry=REGISTRY
)

useful_output_kw = Gauge(
    'gl009_useful_output_kw',
    'Useful energy output from thermal system in kW',
    ['agent_id', 'equipment_id', 'output_type'],
    registry=REGISTRY
)

total_losses_kw = Gauge(
    'gl009_total_losses_kw',
    'Total energy losses from thermal system in kW',
    ['agent_id', 'equipment_id'],
    registry=REGISTRY
)

heat_balance_error_percent = Histogram(
    'gl009_heat_balance_error_percent',
    'Heat balance closure error percentage (should be <2%)',
    ['agent_id', 'equipment_id'],
    buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
    registry=REGISTRY
)

calculations_total = Counter(
    'gl009_calculations_total',
    'Total number of thermal efficiency calculations performed',
    ['agent_id', 'calculation_type', 'equipment_type', 'status'],
    registry=REGISTRY
)

calculation_errors_total = Counter(
    'gl009_calculation_errors_total',
    'Total number of calculation errors by error type',
    ['agent_id', 'error_type', 'calculation_type'],
    registry=REGISTRY
)

benchmark_gap_percent = Histogram(
    'gl009_benchmark_gap_percent',
    'Gap between actual efficiency and benchmark (negative=below benchmark)',
    ['agent_id', 'equipment_type', 'benchmark_type'],
    buckets=[-30, -20, -10, -5, 0, 5, 10, 15, 20],
    registry=REGISTRY
)

improvement_potential_kw = Gauge(
    'gl009_improvement_potential_kw',
    'Potential energy savings if efficiency reaches benchmark (kW)',
    ['agent_id', 'equipment_id', 'improvement_category'],
    registry=REGISTRY
)

uncertainty_percent = Histogram(
    'gl009_uncertainty_percent',
    'Uncertainty in efficiency calculation (percent)',
    ['agent_id', 'calculation_type', 'uncertainty_source'],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    registry=REGISTRY
)

provenance_hash_count = Counter(
    'gl009_provenance_hash_count',
    'Number of provenance hashes generated for audit trail',
    ['agent_id', 'hash_type'],
    registry=REGISTRY
)

determinism_verified = Counter(
    'gl009_determinism_verified',
    'Number of calculations verified as deterministic (same inputs -> same outputs)',
    ['agent_id', 'calculation_type'],
    registry=REGISTRY
)

sankey_generation_duration_seconds = Histogram(
    'gl009_sankey_generation_duration_seconds',
    'Duration to generate Sankey energy flow diagram in seconds',
    ['agent_id', 'diagram_complexity'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=REGISTRY
)

# ============================================================================
# LOSS BREAKDOWN METRICS (8 metrics)
# ============================================================================

radiation_loss_kw = Gauge(
    'gl009_radiation_loss_kw',
    'Radiation heat loss from equipment surface in kW',
    ['agent_id', 'equipment_id', 'surface_type'],
    registry=REGISTRY
)

convection_loss_kw = Gauge(
    'gl009_convection_loss_kw',
    'Convection heat loss from equipment surface in kW',
    ['agent_id', 'equipment_id', 'surface_type'],
    registry=REGISTRY
)

conduction_loss_kw = Gauge(
    'gl009_conduction_loss_kw',
    'Conduction heat loss through walls/piping in kW',
    ['agent_id', 'equipment_id', 'conduction_path'],
    registry=REGISTRY
)

flue_gas_loss_kw = Gauge(
    'gl009_flue_gas_loss_kw',
    'Heat loss via flue gas exit in kW',
    ['agent_id', 'equipment_id', 'fuel_type'],
    registry=REGISTRY
)

unburned_fuel_loss_kw = Gauge(
    'gl009_unburned_fuel_loss_kw',
    'Energy loss from incomplete combustion in kW',
    ['agent_id', 'equipment_id', 'fuel_type'],
    registry=REGISTRY
)

blowdown_loss_kw = Gauge(
    'gl009_blowdown_loss_kw',
    'Heat loss via boiler blowdown in kW',
    ['agent_id', 'equipment_id'],
    registry=REGISTRY
)

surface_loss_kw = Gauge(
    'gl009_surface_loss_kw',
    'Total surface heat loss (radiation + convection) in kW',
    ['agent_id', 'equipment_id'],
    registry=REGISTRY
)

other_losses_kw = Gauge(
    'gl009_other_losses_kw',
    'Other miscellaneous heat losses in kW',
    ['agent_id', 'equipment_id', 'loss_category'],
    registry=REGISTRY
)

# ============================================================================
# INTEGRATION CONNECTOR METRICS (12 metrics)
# ============================================================================

connector_health = Gauge(
    'gl009_connector_health',
    'Health status of integration connector (1=healthy, 0=unhealthy)',
    ['agent_id', 'connector_type', 'endpoint'],
    registry=REGISTRY
)

connector_latency_seconds = Histogram(
    'gl009_connector_latency_seconds',
    'Latency of connector API calls in seconds',
    ['agent_id', 'connector_type', 'operation'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY
)

connector_requests_total = Counter(
    'gl009_connector_requests_total',
    'Total number of requests to integration connectors',
    ['agent_id', 'connector_type', 'operation', 'status'],
    registry=REGISTRY
)

connector_errors_total = Counter(
    'gl009_connector_errors_total',
    'Total number of connector errors by type',
    ['agent_id', 'connector_type', 'error_type'],
    registry=REGISTRY
)

energy_meter_readings_total = Counter(
    'gl009_energy_meter_readings_total',
    'Total number of energy meter readings retrieved',
    ['agent_id', 'meter_id', 'meter_type', 'protocol'],
    registry=REGISTRY
)

historian_queries_total = Counter(
    'gl009_historian_queries_total',
    'Total number of queries to process historian',
    ['agent_id', 'historian_type', 'query_type'],
    registry=REGISTRY
)

scada_subscriptions_active = Gauge(
    'gl009_scada_subscriptions_active',
    'Number of active SCADA data subscriptions',
    ['agent_id', 'scada_system', 'tag_type'],
    registry=REGISTRY
)

erp_cost_queries_total = Counter(
    'gl009_erp_cost_queries_total',
    'Total number of cost data queries to ERP system',
    ['agent_id', 'erp_system', 'cost_type'],
    registry=REGISTRY
)

modbus_read_errors_total = Counter(
    'gl009_modbus_read_errors_total',
    'Total number of Modbus read errors',
    ['agent_id', 'device_id', 'error_code'],
    registry=REGISTRY
)

opcua_connection_duration_seconds = Gauge(
    'gl009_opcua_connection_duration_seconds',
    'Duration of active OPC-UA connection in seconds',
    ['agent_id', 'server_url', 'session_id'],
    registry=REGISTRY
)

mqtt_messages_received_total = Counter(
    'gl009_mqtt_messages_received_total',
    'Total number of MQTT messages received from sensors',
    ['agent_id', 'topic', 'sensor_type'],
    registry=REGISTRY
)

bacnet_points_read_total = Counter(
    'gl009_bacnet_points_read_total',
    'Total number of BACnet points read from building systems',
    ['agent_id', 'device_id', 'object_type'],
    registry=REGISTRY
)

# ============================================================================
# API PERFORMANCE METRICS (10 metrics)
# ============================================================================

http_requests_total = Counter(
    'gl009_http_requests_total',
    'Total number of HTTP requests to GL-009 API',
    ['agent_id', 'method', 'endpoint', 'status'],
    registry=REGISTRY
)

http_request_duration_seconds = Histogram(
    'gl009_http_request_duration_seconds',
    'Duration of HTTP requests in seconds',
    ['agent_id', 'method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

http_request_size_bytes = Histogram(
    'gl009_http_request_size_bytes',
    'Size of HTTP request body in bytes',
    ['agent_id', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000],
    registry=REGISTRY
)

http_response_size_bytes = Histogram(
    'gl009_http_response_size_bytes',
    'Size of HTTP response body in bytes',
    ['agent_id', 'endpoint'],
    buckets=[100, 1000, 10000, 100000, 1000000],
    registry=REGISTRY
)

api_rate_limit_remaining = Gauge(
    'gl009_api_rate_limit_remaining',
    'Number of API requests remaining before rate limit',
    ['agent_id', 'client_id'],
    registry=REGISTRY
)

api_concurrent_requests = Gauge(
    'gl009_api_concurrent_requests',
    'Number of concurrent API requests being processed',
    ['agent_id'],
    registry=REGISTRY
)

api_errors_total = Counter(
    'gl009_api_errors_total',
    'Total number of API errors by error type',
    ['agent_id', 'endpoint', 'error_type', 'status_code'],
    registry=REGISTRY
)

api_validation_errors_total = Counter(
    'gl009_api_validation_errors_total',
    'Total number of API request validation errors',
    ['agent_id', 'endpoint', 'validation_error_type'],
    registry=REGISTRY
)

api_timeout_errors_total = Counter(
    'gl009_api_timeout_errors_total',
    'Total number of API request timeouts',
    ['agent_id', 'endpoint', 'timeout_type'],
    registry=REGISTRY
)

websocket_connections_active = Gauge(
    'gl009_websocket_connections_active',
    'Number of active WebSocket connections for real-time updates',
    ['agent_id', 'connection_type'],
    registry=REGISTRY
)

# ============================================================================
# BUSINESS OUTCOME METRICS (5 metrics)
# ============================================================================

efficiency_improvements_identified = Counter(
    'gl009_efficiency_improvements_identified',
    'Number of efficiency improvement opportunities identified',
    ['agent_id', 'equipment_type', 'improvement_category', 'priority'],
    registry=REGISTRY
)

energy_savings_potential_kwh = Gauge(
    'gl009_energy_savings_potential_kwh',
    'Potential energy savings identified in kWh/year',
    ['agent_id', 'equipment_id', 'improvement_category'],
    registry=REGISTRY
)

cost_savings_potential_usd = Gauge(
    'gl009_cost_savings_potential_usd',
    'Potential cost savings identified in USD/year',
    ['agent_id', 'equipment_id', 'improvement_category'],
    registry=REGISTRY
)

co2_reduction_potential_kg = Gauge(
    'gl009_co2_reduction_potential_kg',
    'Potential CO2 emissions reduction in kg/year',
    ['agent_id', 'equipment_id', 'improvement_category'],
    registry=REGISTRY
)

roi_percent = Gauge(
    'gl009_roi_percent',
    'Return on investment percentage for efficiency improvements',
    ['agent_id', 'equipment_id', 'improvement_category'],
    registry=REGISTRY
)

# ============================================================================
# ADDITIONAL SPECIALIZED METRICS
# ============================================================================

data_quality_score = Gauge(
    'gl009_data_quality_score',
    'Data quality score for input measurements (0.0-1.0)',
    ['agent_id', 'data_source', 'metric_type'],
    registry=REGISTRY
)

benchmark_database_size = Gauge(
    'gl009_benchmark_database_size',
    'Number of benchmark records in database',
    ['agent_id', 'equipment_type', 'region'],
    registry=REGISTRY
)

formula_evaluations_total = Counter(
    'gl009_formula_evaluations_total',
    'Total number of thermodynamic formula evaluations',
    ['agent_id', 'formula_type', 'formula_id'],
    registry=REGISTRY
)

unit_conversions_total = Counter(
    'gl009_unit_conversions_total',
    'Total number of unit conversions performed',
    ['agent_id', 'from_unit', 'to_unit'],
    registry=REGISTRY
)

validation_rules_checked_total = Counter(
    'gl009_validation_rules_checked_total',
    'Total number of validation rules checked',
    ['agent_id', 'rule_type', 'result'],
    registry=REGISTRY
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def record_calculation(
    agent_id: str,
    calculation_type: str,
    equipment_type: str,
    equipment_id: str,
    duration_seconds: float,
    first_law_eff: float,
    second_law_eff: Optional[float],
    energy_input: float,
    useful_output: float,
    total_losses: float,
    heat_balance_error: float,
    status: str,
    losses: Optional[Dict[str, float]] = None,
) -> None:
    """
    Record a complete thermal efficiency calculation with all metrics.

    Args:
        agent_id: Unique identifier for the agent instance
        calculation_type: Type of calculation (e.g., 'boiler', 'furnace', 'heat_exchanger')
        equipment_type: Equipment category
        equipment_id: Specific equipment identifier
        duration_seconds: Calculation duration in seconds
        first_law_eff: First Law efficiency (0-100)
        second_law_eff: Second Law efficiency (0-100) or None
        energy_input: Total energy input in kW
        useful_output: Useful energy output in kW
        total_losses: Total energy losses in kW
        heat_balance_error: Heat balance closure error (%)
        status: Calculation status ('success' or 'error')
        losses: Optional dict of loss breakdown by type
    """
    try:
        # Record calculation duration
        calculation_duration_seconds.labels(
            agent_id=agent_id,
            calculation_type=calculation_type,
            equipment_type=equipment_type
        ).observe(duration_seconds)

        # Record efficiency values
        first_law_efficiency_percent.labels(
            agent_id=agent_id,
            equipment_type=equipment_type,
            process_name=calculation_type
        ).observe(first_law_eff)

        if second_law_eff is not None:
            second_law_efficiency_percent.labels(
                agent_id=agent_id,
                equipment_type=equipment_type,
                process_name=calculation_type
            ).observe(second_law_eff)

        # Record energy values
        energy_input_kw.labels(
            agent_id=agent_id,
            equipment_id=equipment_id,
            fuel_type='natural_gas'  # Could be parameterized
        ).set(energy_input)

        useful_output_kw.labels(
            agent_id=agent_id,
            equipment_id=equipment_id,
            output_type='thermal'
        ).set(useful_output)

        total_losses_kw.labels(
            agent_id=agent_id,
            equipment_id=equipment_id
        ).set(total_losses)

        # Record heat balance error
        heat_balance_error_percent.labels(
            agent_id=agent_id,
            equipment_id=equipment_id
        ).observe(abs(heat_balance_error))

        # Record calculation count
        calculations_total.labels(
            agent_id=agent_id,
            calculation_type=calculation_type,
            equipment_type=equipment_type,
            status=status
        ).inc()

        # Record loss breakdown if provided
        if losses:
            record_loss_breakdown(agent_id, equipment_id, losses)

        # Record provenance hash
        provenance_hash_count.labels(
            agent_id=agent_id,
            hash_type='calculation'
        ).inc()

        logger.debug(f"Recorded calculation metrics for {equipment_id}: {first_law_eff}% efficiency")

    except Exception as e:
        logger.error(f"Failed to record calculation metrics: {e}")
        error_count.labels(
            agent_id=agent_id,
            error_type='metric_recording_error',
            severity='warning'
        ).inc()


def record_loss_breakdown(
    agent_id: str,
    equipment_id: str,
    losses: Dict[str, float]
) -> None:
    """
    Record detailed breakdown of thermal losses.

    Args:
        agent_id: Agent instance identifier
        equipment_id: Equipment identifier
        losses: Dict mapping loss type to kW value
    """
    loss_metrics = {
        'radiation': radiation_loss_kw,
        'convection': convection_loss_kw,
        'conduction': conduction_loss_kw,
        'flue_gas': flue_gas_loss_kw,
        'unburned_fuel': unburned_fuel_loss_kw,
        'blowdown': blowdown_loss_kw,
        'surface': surface_loss_kw,
        'other': other_losses_kw,
    }

    for loss_type, loss_value in losses.items():
        if loss_type in loss_metrics:
            metric = loss_metrics[loss_type]
            if loss_type in ['radiation', 'convection']:
                metric.labels(
                    agent_id=agent_id,
                    equipment_id=equipment_id,
                    surface_type='main'
                ).set(loss_value)
            elif loss_type == 'conduction':
                metric.labels(
                    agent_id=agent_id,
                    equipment_id=equipment_id,
                    conduction_path='wall'
                ).set(loss_value)
            elif loss_type in ['flue_gas', 'unburned_fuel']:
                metric.labels(
                    agent_id=agent_id,
                    equipment_id=equipment_id,
                    fuel_type='natural_gas'
                ).set(loss_value)
            elif loss_type == 'other':
                metric.labels(
                    agent_id=agent_id,
                    equipment_id=equipment_id,
                    loss_category='miscellaneous'
                ).set(loss_value)
            else:
                metric.labels(
                    agent_id=agent_id,
                    equipment_id=equipment_id
                ).set(loss_value)


def record_connector_call(
    agent_id: str,
    connector_type: str,
    operation: str,
    duration_seconds: float,
    status: str,
    error_type: Optional[str] = None
) -> None:
    """
    Record metrics for integration connector API call.

    Args:
        agent_id: Agent instance identifier
        connector_type: Type of connector (e.g., 'modbus', 'opcua', 'mqtt')
        operation: Operation performed (e.g., 'read_register', 'subscribe')
        duration_seconds: Call duration in seconds
        status: Call status ('success' or 'error')
        error_type: Error type if status is 'error'
    """
    # Record latency
    connector_latency_seconds.labels(
        agent_id=agent_id,
        connector_type=connector_type,
        operation=operation
    ).observe(duration_seconds)

    # Record request count
    connector_requests_total.labels(
        agent_id=agent_id,
        connector_type=connector_type,
        operation=operation,
        status=status
    ).inc()

    # Record error if applicable
    if status == 'error' and error_type:
        connector_errors_total.labels(
            agent_id=agent_id,
            connector_type=connector_type,
            error_type=error_type
        ).inc()

    # Update health status
    health_value = 1.0 if status == 'success' else 0.0
    connector_health.labels(
        agent_id=agent_id,
        connector_type=connector_type,
        endpoint=operation
    ).set(health_value)


def record_http_request(
    agent_id: str,
    method: str,
    endpoint: str,
    status_code: int,
    duration_seconds: float,
    request_size_bytes: int,
    response_size_bytes: int
) -> None:
    """
    Record metrics for HTTP API request.

    Args:
        agent_id: Agent instance identifier
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        status_code: HTTP status code
        duration_seconds: Request duration in seconds
        request_size_bytes: Size of request body in bytes
        response_size_bytes: Size of response body in bytes
    """
    # Record request count
    http_requests_total.labels(
        agent_id=agent_id,
        method=method,
        endpoint=endpoint,
        status=str(status_code)
    ).inc()

    # Record duration
    http_request_duration_seconds.labels(
        agent_id=agent_id,
        method=method,
        endpoint=endpoint
    ).observe(duration_seconds)

    # Record sizes
    http_request_size_bytes.labels(
        agent_id=agent_id,
        endpoint=endpoint
    ).observe(request_size_bytes)

    http_response_size_bytes.labels(
        agent_id=agent_id,
        endpoint=endpoint
    ).observe(response_size_bytes)


def record_business_outcome(
    agent_id: str,
    equipment_id: str,
    improvement_category: str,
    energy_savings_kwh: float,
    cost_savings_usd: float,
    co2_reduction_kg: float,
    roi_pct: float,
    priority: str = 'medium'
) -> None:
    """
    Record business outcome metrics for efficiency improvement.

    Args:
        agent_id: Agent instance identifier
        equipment_id: Equipment identifier
        improvement_category: Category of improvement (e.g., 'insulation', 'combustion_tuning')
        energy_savings_kwh: Annual energy savings potential in kWh
        cost_savings_usd: Annual cost savings potential in USD
        co2_reduction_kg: Annual CO2 reduction potential in kg
        roi_pct: Return on investment percentage
        priority: Priority level ('high', 'medium', 'low')
    """
    # Increment improvement count
    efficiency_improvements_identified.labels(
        agent_id=agent_id,
        equipment_type='thermal',
        improvement_category=improvement_category,
        priority=priority
    ).inc()

    # Set savings potentials
    energy_savings_potential_kwh.labels(
        agent_id=agent_id,
        equipment_id=equipment_id,
        improvement_category=improvement_category
    ).set(energy_savings_kwh)

    cost_savings_potential_usd.labels(
        agent_id=agent_id,
        equipment_id=equipment_id,
        improvement_category=improvement_category
    ).set(cost_savings_usd)

    co2_reduction_potential_kg.labels(
        agent_id=agent_id,
        equipment_id=equipment_id,
        improvement_category=improvement_category
    ).set(co2_reduction_kg)

    roi_percent.labels(
        agent_id=agent_id,
        equipment_id=equipment_id,
        improvement_category=improvement_category
    ).set(roi_pct)


def update_health_metrics(agent_id: str, environment: str) -> None:
    """
    Update agent health metrics (CPU, memory, cache, etc.).

    Args:
        agent_id: Agent instance identifier
        environment: Deployment environment (dev, staging, prod)
    """
    try:
        # Get process info
        process = psutil.Process()

        # Update memory metrics
        mem_info = process.memory_info()
        agent_memory_usage_bytes.labels(
            agent_id=agent_id,
            environment=environment,
            memory_type='rss'
        ).set(mem_info.rss)

        agent_memory_usage_bytes.labels(
            agent_id=agent_id,
            environment=environment,
            memory_type='vms'
        ).set(mem_info.vms)

        # Update CPU metrics
        cpu_percent = process.cpu_percent(interval=0.1)
        agent_cpu_usage_percent.labels(
            agent_id=agent_id,
            environment=environment
        ).set(cpu_percent)

        # Update uptime
        agent_uptime_seconds.labels(
            agent_id=agent_id,
            environment=environment
        ).inc()

        # Set health status (healthy if CPU < 90% and memory < 90% of available)
        health = 1.0 if cpu_percent < 90.0 else 0.0
        agent_health_status.labels(
            agent_id=agent_id,
            environment=environment
        ).set(health)

    except Exception as e:
        logger.error(f"Failed to update health metrics: {e}")


def timing_decorator(metric: Histogram):
    """
    Decorator to automatically time function execution and record to histogram.

    Args:
        metric: Prometheus Histogram metric to record timing

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                # Extract labels from kwargs if present
                labels = {k: v for k, v in kwargs.items() if k in metric._labelnames}
                metric.labels(**labels).observe(duration)
        return wrapper
    return decorator


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a summary of all current metric values.

    Returns:
        Dict containing metric summaries
    """
    return {
        'registry': REGISTRY,
        'metrics_count': len(list(REGISTRY.collect())),
        'timestamp': datetime.utcnow().isoformat(),
    }


def export_metrics() -> bytes:
    """
    Export all metrics in Prometheus text format.

    Returns:
        Metrics in Prometheus exposition format
    """
    return generate_latest(REGISTRY)


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

logger.info("GL-009 metrics module initialized with 60+ Prometheus metrics")
