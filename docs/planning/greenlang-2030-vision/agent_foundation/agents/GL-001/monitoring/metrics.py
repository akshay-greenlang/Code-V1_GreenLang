# -*- coding: utf-8 -*-
"""
Prometheus metrics for GL-001 ProcessHeatOrchestrator.

Provides comprehensive metrics tracking for:
- Master orchestrator operations
- Multi-plant coordination metrics
- Sub-agent coordination (GL-002 through GL-100)
- SCADA integration health (per plant)
- ERP integration metrics
- Thermal efficiency metrics (per plant, aggregate)
- Heat distribution metrics
- Energy balance metrics
- Emissions compliance metrics
- Task delegation metrics
- Performance metrics (latency, throughput)
- HTTP request latency and throughput
- System resource utilization
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, Info
from typing import Optional, Callable, Dict, Any, List
import functools
import time
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# HTTP REQUEST METRICS
# ==============================================================================

http_requests_total = Counter(
    'gl_001_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'gl_001_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

http_request_size_bytes = Histogram(
    'gl_001_http_request_size_bytes',
    'HTTP request body size',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000)
)

http_response_size_bytes = Histogram(
    'gl_001_http_response_size_bytes',
    'HTTP response body size',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000)
)

# ==============================================================================
# MASTER ORCHESTRATOR METRICS
# ==============================================================================

orchestrator_health_status = Gauge(
    'gl_001_orchestrator_health_status',
    'Master orchestrator health status (1=healthy, 0=unhealthy)'
)

orchestrator_uptime_seconds = Gauge(
    'gl_001_orchestrator_uptime_seconds',
    'Master orchestrator uptime in seconds'
)

orchestration_requests_total = Counter(
    'gl_001_orchestration_requests_total',
    'Total orchestration requests processed',
    ['orchestration_type', 'status']  # status: success, failure, partial
)

orchestration_duration_seconds = Histogram(
    'gl_001_orchestration_duration_seconds',
    'Orchestration execution time',
    ['orchestration_type'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)

orchestration_complexity_score = Histogram(
    'gl_001_orchestration_complexity_score',
    'Complexity score of orchestration tasks (0-100)',
    ['orchestration_type'],
    buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
)

orchestrator_state = Gauge(
    'gl_001_orchestrator_state',
    'Current orchestrator state (0=init, 1=ready, 2=executing, 3=error, 4=recovering, 5=terminated)',
    ['state_name']
)

# ==============================================================================
# MULTI-PLANT COORDINATION METRICS
# ==============================================================================

active_plants_count = Gauge(
    'gl_001_active_plants_count',
    'Number of active plants being coordinated'
)

plant_health_status = Gauge(
    'gl_001_plant_health_status',
    'Plant health status (1=healthy, 0=unhealthy)',
    ['plant_id', 'plant_name', 'location']
)

plant_thermal_efficiency_percent = Gauge(
    'gl_001_plant_thermal_efficiency_percent',
    'Plant-level thermal efficiency',
    ['plant_id', 'plant_name']
)

plant_heat_generation_mw = Gauge(
    'gl_001_plant_heat_generation_mw',
    'Total heat generation per plant',
    ['plant_id', 'plant_name']
)

plant_heat_demand_mw = Gauge(
    'gl_001_plant_heat_demand_mw',
    'Total heat demand per plant',
    ['plant_id', 'plant_name']
)

plant_heat_losses_mw = Gauge(
    'gl_001_plant_heat_losses_mw',
    'Total heat losses per plant',
    ['plant_id', 'plant_name', 'loss_type']  # loss_type: distribution, radiation, flue_gas
)

plant_capacity_utilization_percent = Gauge(
    'gl_001_plant_capacity_utilization_percent',
    'Plant capacity utilization',
    ['plant_id', 'plant_name']
)

cross_plant_heat_transfer_mw = Gauge(
    'gl_001_cross_plant_heat_transfer_mw',
    'Heat transferred between plants',
    ['source_plant', 'destination_plant']
)

# ==============================================================================
# SUB-AGENT COORDINATION METRICS
# ==============================================================================

active_subagents_count = Gauge(
    'gl_001_active_subagents_count',
    'Number of active sub-agents',
    ['agent_category']  # category: boiler, steam, furnace, heat_recovery, etc.
)

subagent_health_status = Gauge(
    'gl_001_subagent_health_status',
    'Sub-agent health status (1=healthy, 0=unhealthy)',
    ['agent_id', 'agent_type', 'plant_id']
)

subagent_response_time_seconds = Histogram(
    'gl_001_subagent_response_time_seconds',
    'Sub-agent response time',
    ['agent_id', 'agent_type'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0)
)

subagent_task_assignments_total = Counter(
    'gl_001_subagent_task_assignments_total',
    'Total tasks assigned to sub-agents',
    ['agent_id', 'task_type', 'priority']  # priority: low, medium, high, critical
)

subagent_task_completion_total = Counter(
    'gl_001_subagent_task_completion_total',
    'Total tasks completed by sub-agents',
    ['agent_id', 'task_type', 'status']  # status: success, failure, timeout
)

subagent_coordination_failures = Counter(
    'gl_001_subagent_coordination_failures_total',
    'Total sub-agent coordination failures',
    ['agent_id', 'failure_type']  # failure_type: timeout, error, unavailable
)

subagent_message_queue_depth = Gauge(
    'gl_001_subagent_message_queue_depth',
    'Message queue depth for sub-agents',
    ['agent_id']
)

subagent_task_latency_seconds = Histogram(
    'gl_001_subagent_task_latency_seconds',
    'Time from task assignment to completion',
    ['agent_id', 'task_type'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600)
)

# ==============================================================================
# SCADA INTEGRATION METRICS (Per Plant)
# ==============================================================================

scada_connection_status = Gauge(
    'gl_001_scada_connection_status',
    'SCADA connection status (1=connected, 0=disconnected)',
    ['plant_id', 'scada_system']
)

scada_data_points_received_total = Counter(
    'gl_001_scada_data_points_received_total',
    'Total SCADA data points received',
    ['plant_id', 'data_category']  # category: temperature, pressure, flow, level
)

scada_data_latency_seconds = Histogram(
    'gl_001_scada_data_latency_seconds',
    'SCADA data latency from source to processing',
    ['plant_id', 'scada_system'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

scada_data_quality_percent = Gauge(
    'gl_001_scada_data_quality_percent',
    'SCADA data quality score',
    ['plant_id', 'data_category']
)

scada_integration_errors_total = Counter(
    'gl_001_scada_integration_errors_total',
    'Total SCADA integration errors',
    ['plant_id', 'error_type']  # error_type: timeout, invalid_data, connection_lost
)

scada_tags_monitored = Gauge(
    'gl_001_scada_tags_monitored',
    'Number of SCADA tags being monitored',
    ['plant_id']
)

scada_alarms_active = Gauge(
    'gl_001_scada_alarms_active',
    'Number of active SCADA alarms',
    ['plant_id', 'severity']  # severity: low, medium, high, critical
)

# ==============================================================================
# ERP INTEGRATION METRICS
# ==============================================================================

erp_connection_status = Gauge(
    'gl_001_erp_connection_status',
    'ERP connection status (1=connected, 0=disconnected)',
    ['erp_system']
)

erp_data_sync_total = Counter(
    'gl_001_erp_data_sync_total',
    'Total ERP data synchronizations',
    ['data_type', 'status']  # data_type: production_schedule, costs, inventory
)

erp_data_sync_duration_seconds = Histogram(
    'gl_001_erp_data_sync_duration_seconds',
    'ERP data synchronization duration',
    ['data_type'],
    buckets=(1, 5, 10, 30, 60, 120)
)

erp_production_schedule_updates = Counter(
    'gl_001_erp_production_schedule_updates_total',
    'Production schedule updates from ERP',
    ['plant_id']
)

erp_cost_data_updates = Counter(
    'gl_001_erp_cost_data_updates_total',
    'Cost data updates from ERP',
    ['cost_category']  # category: fuel, electricity, maintenance, labor
)

erp_integration_errors_total = Counter(
    'gl_001_erp_integration_errors_total',
    'Total ERP integration errors',
    ['error_type']
)

# ==============================================================================
# THERMAL EFFICIENCY METRICS
# ==============================================================================

# Aggregate across all plants
aggregate_thermal_efficiency_percent = Gauge(
    'gl_001_aggregate_thermal_efficiency_percent',
    'Aggregate thermal efficiency across all plants'
)

aggregate_heat_generation_mw = Gauge(
    'gl_001_aggregate_heat_generation_mw',
    'Total heat generation across all plants'
)

aggregate_heat_demand_mw = Gauge(
    'gl_001_aggregate_heat_demand_mw',
    'Total heat demand across all plants'
)

aggregate_heat_losses_mw = Gauge(
    'gl_001_aggregate_heat_losses_mw',
    'Total heat losses across all plants'
)

# Per plant efficiency metrics
plant_boiler_efficiency_percent = Gauge(
    'gl_001_plant_boiler_efficiency_percent',
    'Average boiler efficiency per plant',
    ['plant_id', 'plant_name']
)

plant_heat_recovery_efficiency_percent = Gauge(
    'gl_001_plant_heat_recovery_efficiency_percent',
    'Heat recovery efficiency per plant',
    ['plant_id', 'plant_name']
)

plant_distribution_efficiency_percent = Gauge(
    'gl_001_plant_distribution_efficiency_percent',
    'Heat distribution efficiency per plant',
    ['plant_id', 'plant_name']
)

# Efficiency improvement tracking
efficiency_optimization_events = Counter(
    'gl_001_efficiency_optimization_events_total',
    'Efficiency optimization events',
    ['plant_id', 'optimization_type', 'status']
)

efficiency_improvement_percent = Histogram(
    'gl_001_efficiency_improvement_percent',
    'Efficiency improvement from optimizations',
    ['plant_id', 'optimization_type'],
    buckets=(0, 1, 2, 3, 5, 10, 15, 20)
)

# ==============================================================================
# HEAT DISTRIBUTION METRICS
# ==============================================================================

heat_distribution_optimization_score = Gauge(
    'gl_001_heat_distribution_optimization_score',
    'Heat distribution optimization score (0-100)',
    ['plant_id']
)

heat_distribution_imbalance_mw = Gauge(
    'gl_001_heat_distribution_imbalance_mw',
    'Heat distribution imbalance',
    ['plant_id']
)

heat_distribution_strategy_changes = Counter(
    'gl_001_heat_distribution_strategy_changes_total',
    'Heat distribution strategy changes',
    ['plant_id', 'strategy_type']
)

heat_network_pressure_bar = Gauge(
    'gl_001_heat_network_pressure_bar',
    'Heat network pressure',
    ['plant_id', 'network_segment']
)

heat_network_temperature_c = Gauge(
    'gl_001_heat_network_temperature_c',
    'Heat network temperature',
    ['plant_id', 'network_segment', 'measurement_point']
)

heat_network_flow_rate_kg_hr = Gauge(
    'gl_001_heat_network_flow_rate_kg_hr',
    'Heat network flow rate',
    ['plant_id', 'network_segment']
)

# ==============================================================================
# ENERGY BALANCE METRICS
# ==============================================================================

energy_balance_closure_percent = Gauge(
    'gl_001_energy_balance_closure_percent',
    'Energy balance closure percentage',
    ['plant_id']
)

energy_balance_error_mw = Gauge(
    'gl_001_energy_balance_error_mw',
    'Energy balance error',
    ['plant_id']
)

energy_balance_calculations_total = Counter(
    'gl_001_energy_balance_calculations_total',
    'Total energy balance calculations',
    ['plant_id', 'status']  # status: balanced, imbalanced, error
)

energy_input_mw = Gauge(
    'gl_001_energy_input_mw',
    'Total energy input',
    ['plant_id', 'energy_type']  # energy_type: fuel, electricity, steam_import
)

energy_output_mw = Gauge(
    'gl_001_energy_output_mw',
    'Total energy output',
    ['plant_id', 'output_type']  # output_type: process_heat, power, steam_export
)

energy_storage_mwh = Gauge(
    'gl_001_energy_storage_mwh',
    'Energy storage capacity and state',
    ['plant_id', 'storage_type', 'metric']  # metric: capacity, current_state, charge_rate
)

# ==============================================================================
# EMISSIONS COMPLIANCE METRICS
# ==============================================================================

emissions_compliance_status = Gauge(
    'gl_001_emissions_compliance_status',
    'Emissions compliance status (1=compliant, 0=violation)',
    ['plant_id', 'regulation']
)

emissions_co2_tons_hr = Gauge(
    'gl_001_emissions_co2_tons_hr',
    'CO2 emissions rate',
    ['plant_id']
)

emissions_co2_intensity_kg_mwh = Gauge(
    'gl_001_emissions_co2_intensity_kg_mwh',
    'CO2 emissions intensity',
    ['plant_id']
)

emissions_nox_kg_hr = Gauge(
    'gl_001_emissions_nox_kg_hr',
    'NOx emissions rate',
    ['plant_id']
)

emissions_sox_kg_hr = Gauge(
    'gl_001_emissions_sox_kg_hr',
    'SOx emissions rate',
    ['plant_id']
)

emissions_compliance_violations = Counter(
    'gl_001_emissions_compliance_violations_total',
    'Total emissions compliance violations',
    ['plant_id', 'pollutant', 'regulation']
)

emissions_reduction_initiatives = Counter(
    'gl_001_emissions_reduction_initiatives_total',
    'Emissions reduction initiatives executed',
    ['plant_id', 'initiative_type']
)

carbon_credit_balance_tons = Gauge(
    'gl_001_carbon_credit_balance_tons',
    'Carbon credit balance',
    ['plant_id', 'credit_type']
)

# ==============================================================================
# TASK DELEGATION METRICS
# ==============================================================================

tasks_delegated_total = Counter(
    'gl_001_tasks_delegated_total',
    'Total tasks delegated to sub-agents',
    ['task_category', 'priority']
)

tasks_completed_total = Counter(
    'gl_001_tasks_completed_total',
    'Total tasks completed',
    ['task_category', 'status']
)

task_queue_depth = Gauge(
    'gl_001_task_queue_depth',
    'Current task queue depth',
    ['priority']
)

task_execution_duration_seconds = Histogram(
    'gl_001_task_execution_duration_seconds',
    'Task execution duration',
    ['task_category'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800)
)

task_retry_attempts = Counter(
    'gl_001_task_retry_attempts_total',
    'Task retry attempts',
    ['task_category', 'retry_reason']
)

task_failure_rate_percent = Gauge(
    'gl_001_task_failure_rate_percent',
    'Task failure rate (rolling 1 hour)',
    ['task_category']
)

# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================

calculation_cache_hits_total = Counter(
    'gl_001_calculation_cache_hits_total',
    'Total calculation cache hits',
    ['calculation_type']
)

calculation_cache_misses_total = Counter(
    'gl_001_calculation_cache_misses_total',
    'Total calculation cache misses',
    ['calculation_type']
)

calculation_cache_hit_rate_percent = Gauge(
    'gl_001_calculation_cache_hit_rate_percent',
    'Calculation cache hit rate',
    ['calculation_type']
)

calculation_cache_size = Gauge(
    'gl_001_calculation_cache_size',
    'Number of entries in calculation cache'
)

calculation_duration_seconds = Histogram(
    'gl_001_calculation_duration_seconds',
    'Calculation execution time',
    ['calculation_type'],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0)
)

memory_short_term_size = Gauge(
    'gl_001_memory_short_term_size',
    'Short-term memory entries count'
)

memory_long_term_size = Gauge(
    'gl_001_memory_long_term_size',
    'Long-term memory entries count'
)

message_bus_messages_total = Counter(
    'gl_001_message_bus_messages_total',
    'Total messages sent via message bus',
    ['message_type', 'recipient_category']
)

message_bus_latency_seconds = Histogram(
    'gl_001_message_bus_latency_seconds',
    'Message bus delivery latency',
    ['message_type'],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0)
)

# ==============================================================================
# DATABASE METRICS
# ==============================================================================

db_connection_pool_size = Gauge(
    'gl_001_db_connection_pool_size',
    'Active database connections'
)

db_query_duration_seconds = Histogram(
    'gl_001_db_query_duration_seconds',
    'Database query latency',
    ['query_type'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
)

db_query_errors_total = Counter(
    'gl_001_db_query_errors_total',
    'Total database query errors',
    ['query_type', 'error_type']
)

# ==============================================================================
# SYSTEM METRICS
# ==============================================================================

system_uptime_seconds = Gauge(
    'gl_001_system_uptime_seconds',
    'Application uptime'
)

system_memory_usage_bytes = Gauge(
    'gl_001_system_memory_usage_bytes',
    'Memory usage',
    ['type']  # type: rss, vms, heap
)

system_cpu_usage_percent = Gauge(
    'gl_001_system_cpu_usage_percent',
    'CPU usage percentage'
)

system_disk_usage_bytes = Gauge(
    'gl_001_system_disk_usage_bytes',
    'Disk usage',
    ['mount_point']
)

system_goroutines_count = Gauge(
    'gl_001_system_goroutines_count',
    'Number of active goroutines/threads'
)

# ==============================================================================
# BUSINESS METRICS
# ==============================================================================

optimization_annual_savings_usd = Gauge(
    'gl_001_optimization_annual_savings_usd',
    'Estimated annual cost savings',
    ['plant_id']
)

optimization_annual_energy_savings_mwh = Gauge(
    'gl_001_optimization_annual_energy_savings_mwh',
    'Estimated annual energy savings',
    ['plant_id']
)

optimization_annual_emissions_reduction_tons = Gauge(
    'gl_001_optimization_annual_emissions_reduction_tons',
    'Estimated annual CO2 reduction',
    ['plant_id']
)

total_cost_savings_usd_hr = Gauge(
    'gl_001_total_cost_savings_usd_hr',
    'Current cost savings rate',
    ['plant_id', 'savings_category']  # category: fuel, electricity, maintenance
)

roi_payback_period_months = Gauge(
    'gl_001_roi_payback_period_months',
    'ROI payback period for optimizations',
    ['plant_id', 'optimization_type']
)

# ==============================================================================
# DETERMINISM METRICS
# ==============================================================================

determinism_verification_failures = Counter(
    'gl_001_determinism_verification_failures_total',
    'Total determinism verification failures',
    ['violation_type']
)

determinism_score = Gauge(
    'gl_001_determinism_score_percent',
    'Determinism score (0-100%, target: 100%)',
    ['component']
)

determinism_verification_duration_seconds = Histogram(
    'gl_001_determinism_verification_duration_seconds',
    'Time spent verifying determinism',
    ['verification_type'],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0)
)

provenance_hash_verifications = Counter(
    'gl_001_provenance_hash_verifications_total',
    'Total provenance hash verifications',
    ['status']
)

# ==============================================================================
# DECORATOR FOR AUTOMATIC METRICS TRACKING
# ==============================================================================

def track_request_metrics(method: str, endpoint: str):
    """
    Decorator to automatically track HTTP request metrics.

    Usage:
        @track_request_metrics('GET', '/api/v1/orchestrate')
        async def orchestrate_endpoint():
            return {"status": "success"}
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status='success'
                ).inc()
                return result
            except Exception as e:
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status='error'
                ).inc()
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status='success'
                ).inc()
                return result
            except Exception as e:
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status='error'
                ).inc()
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_orchestration_metrics(orchestration_type: str):
    """
    Decorator to track orchestration request metrics.

    Usage:
        @track_orchestration_metrics('multi_plant_optimization')
        async def orchestrate():
            return {"status": "optimized"}
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                orchestration_duration_seconds.labels(
                    orchestration_type=orchestration_type
                ).observe(duration)
                orchestration_requests_total.labels(
                    orchestration_type=orchestration_type,
                    status='success'
                ).inc()

                return result
            except Exception as e:
                duration = time.time() - start_time
                orchestration_duration_seconds.labels(
                    orchestration_type=orchestration_type
                ).observe(duration)
                orchestration_requests_total.labels(
                    orchestration_type=orchestration_type,
                    status='failure'
                ).inc()
                raise

        return async_wrapper

    return decorator


# ==============================================================================
# COLLECTOR FOR CUSTOM METRICS
# ==============================================================================

class MetricsCollector:
    """Collects and updates system metrics for GL-001."""

    @staticmethod
    def update_orchestrator_state(state: str, is_healthy: bool):
        """Update orchestrator state and health."""
        orchestrator_health_status.set(1 if is_healthy else 0)
        state_values = {
            'INIT': 0, 'READY': 1, 'EXECUTING': 2,
            'ERROR': 3, 'RECOVERING': 4, 'TERMINATED': 5
        }
        orchestrator_state.labels(state_name=state).set(
            state_values.get(state, 0)
        )

    @staticmethod
    def update_plant_metrics(plant_id: str, plant_name: str, metrics: Dict[str, Any]):
        """Update plant-level metrics."""
        if 'health_status' in metrics:
            plant_health_status.labels(
                plant_id=plant_id,
                plant_name=plant_name,
                location=metrics.get('location', 'unknown')
            ).set(1 if metrics['health_status'] == 'healthy' else 0)

        if 'thermal_efficiency_percent' in metrics:
            plant_thermal_efficiency_percent.labels(
                plant_id=plant_id,
                plant_name=plant_name
            ).set(metrics['thermal_efficiency_percent'])

        if 'heat_generation_mw' in metrics:
            plant_heat_generation_mw.labels(
                plant_id=plant_id,
                plant_name=plant_name
            ).set(metrics['heat_generation_mw'])

        if 'heat_demand_mw' in metrics:
            plant_heat_demand_mw.labels(
                plant_id=plant_id,
                plant_name=plant_name
            ).set(metrics['heat_demand_mw'])

        if 'capacity_utilization_percent' in metrics:
            plant_capacity_utilization_percent.labels(
                plant_id=plant_id,
                plant_name=plant_name
            ).set(metrics['capacity_utilization_percent'])

    @staticmethod
    def update_subagent_metrics(agent_id: str, agent_type: str, plant_id: str, metrics: Dict[str, Any]):
        """Update sub-agent metrics."""
        if 'health_status' in metrics:
            subagent_health_status.labels(
                agent_id=agent_id,
                agent_type=agent_type,
                plant_id=plant_id
            ).set(1 if metrics['health_status'] == 'healthy' else 0)

        if 'message_queue_depth' in metrics:
            subagent_message_queue_depth.labels(
                agent_id=agent_id
            ).set(metrics['message_queue_depth'])

    @staticmethod
    def record_task_delegation(task_category: str, priority: str, agent_id: str, task_type: str):
        """Record task delegation to sub-agent."""
        tasks_delegated_total.labels(
            task_category=task_category,
            priority=priority
        ).inc()

        subagent_task_assignments_total.labels(
            agent_id=agent_id,
            task_type=task_type,
            priority=priority
        ).inc()

    @staticmethod
    def record_task_completion(task_category: str, status: str, agent_id: str, task_type: str):
        """Record task completion."""
        tasks_completed_total.labels(
            task_category=task_category,
            status=status
        ).inc()

        subagent_task_completion_total.labels(
            agent_id=agent_id,
            task_type=task_type,
            status=status
        ).inc()

    @staticmethod
    def update_scada_metrics(plant_id: str, scada_system: str, metrics: Dict[str, Any]):
        """Update SCADA integration metrics."""
        if 'connection_status' in metrics:
            scada_connection_status.labels(
                plant_id=plant_id,
                scada_system=scada_system
            ).set(1 if metrics['connection_status'] == 'connected' else 0)

        if 'data_quality_percent' in metrics:
            for category, quality in metrics['data_quality_percent'].items():
                scada_data_quality_percent.labels(
                    plant_id=plant_id,
                    data_category=category
                ).set(quality)

        if 'tags_monitored' in metrics:
            scada_tags_monitored.labels(plant_id=plant_id).set(
                metrics['tags_monitored']
            )

        if 'active_alarms' in metrics:
            for severity, count in metrics['active_alarms'].items():
                scada_alarms_active.labels(
                    plant_id=plant_id,
                    severity=severity
                ).set(count)

    @staticmethod
    def update_erp_metrics(erp_system: str, metrics: Dict[str, Any]):
        """Update ERP integration metrics."""
        if 'connection_status' in metrics:
            erp_connection_status.labels(
                erp_system=erp_system
            ).set(1 if metrics['connection_status'] == 'connected' else 0)

    @staticmethod
    def update_aggregate_metrics(metrics: Dict[str, Any]):
        """Update aggregate (enterprise-wide) metrics."""
        if 'thermal_efficiency_percent' in metrics:
            aggregate_thermal_efficiency_percent.set(
                metrics['thermal_efficiency_percent']
            )

        if 'heat_generation_mw' in metrics:
            aggregate_heat_generation_mw.set(metrics['heat_generation_mw'])

        if 'heat_demand_mw' in metrics:
            aggregate_heat_demand_mw.set(metrics['heat_demand_mw'])

        if 'heat_losses_mw' in metrics:
            aggregate_heat_losses_mw.set(metrics['heat_losses_mw'])

        if 'active_plants_count' in metrics:
            active_plants_count.set(metrics['active_plants_count'])

    @staticmethod
    def update_emissions_metrics(plant_id: str, emissions: Dict[str, Any]):
        """Update emissions metrics."""
        if 'co2_tons_hr' in emissions:
            emissions_co2_tons_hr.labels(plant_id=plant_id).set(
                emissions['co2_tons_hr']
            )

        if 'co2_intensity_kg_mwh' in emissions:
            emissions_co2_intensity_kg_mwh.labels(plant_id=plant_id).set(
                emissions['co2_intensity_kg_mwh']
            )

        if 'nox_kg_hr' in emissions:
            emissions_nox_kg_hr.labels(plant_id=plant_id).set(
                emissions['nox_kg_hr']
            )

        if 'compliance_status' in emissions:
            for regulation, status in emissions['compliance_status'].items():
                emissions_compliance_status.labels(
                    plant_id=plant_id,
                    regulation=regulation
                ).set(1 if status == 'compliant' else 0)

    @staticmethod
    def record_cache_operation(calculation_type: str, hit: bool):
        """Record cache hit/miss."""
        if hit:
            calculation_cache_hits_total.labels(
                calculation_type=calculation_type
            ).inc()
        else:
            calculation_cache_misses_total.labels(
                calculation_type=calculation_type
            ).inc()

    @staticmethod
    def update_system_metrics(metrics: Dict[str, Any]):
        """Update system resource metrics."""
        if 'uptime_seconds' in metrics:
            system_uptime_seconds.set(metrics['uptime_seconds'])
            orchestrator_uptime_seconds.set(metrics['uptime_seconds'])

        if 'memory_rss_bytes' in metrics:
            system_memory_usage_bytes.labels(type='rss').set(
                metrics['memory_rss_bytes']
            )

        if 'memory_vms_bytes' in metrics:
            system_memory_usage_bytes.labels(type='vms').set(
                metrics['memory_vms_bytes']
            )

        if 'cpu_percent' in metrics:
            system_cpu_usage_percent.set(metrics['cpu_percent'])

        if 'disk_usage' in metrics:
            for mount_point, usage in metrics['disk_usage'].items():
                system_disk_usage_bytes.labels(
                    mount_point=mount_point
                ).set(usage)

    @staticmethod
    def update_business_metrics(plant_id: str, metrics: Dict[str, Any]):
        """Update business metrics."""
        if 'annual_savings_usd' in metrics:
            optimization_annual_savings_usd.labels(
                plant_id=plant_id
            ).set(metrics['annual_savings_usd'])

        if 'annual_energy_savings_mwh' in metrics:
            optimization_annual_energy_savings_mwh.labels(
                plant_id=plant_id
            ).set(metrics['annual_energy_savings_mwh'])

        if 'annual_emissions_reduction_tons' in metrics:
            optimization_annual_emissions_reduction_tons.labels(
                plant_id=plant_id
            ).set(metrics['annual_emissions_reduction_tons'])
