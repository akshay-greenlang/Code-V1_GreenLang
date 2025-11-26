"""
GL-009 THERMALIQ ThermalEfficiencyCalculator Monitoring Package

This package provides comprehensive monitoring, metrics, and observability
for the GL-009 agent, tracking thermal efficiency calculations, loss analysis,
integration health, and business outcomes.

Exports:
    - All Prometheus metrics collectors
    - Monitoring utilities
    - Health check functions
    - SLO tracking utilities
"""

from monitoring.metrics import (
    # Agent Health Metrics
    agent_health_status,
    agent_uptime_seconds,
    agent_restart_count,
    agent_memory_usage_bytes,
    agent_cpu_usage_percent,
    cache_hit_rate,
    cache_size_bytes,
    active_calculations,
    queue_depth,
    error_count,

    # Thermal Efficiency Calculation Metrics
    calculation_duration_seconds,
    first_law_efficiency_percent,
    second_law_efficiency_percent,
    energy_input_kw,
    useful_output_kw,
    total_losses_kw,
    heat_balance_error_percent,
    calculations_total,
    calculation_errors_total,
    benchmark_gap_percent,
    improvement_potential_kw,
    uncertainty_percent,
    provenance_hash_count,
    determinism_verified,
    sankey_generation_duration_seconds,

    # Loss Breakdown Metrics
    radiation_loss_kw,
    convection_loss_kw,
    conduction_loss_kw,
    flue_gas_loss_kw,
    unburned_fuel_loss_kw,
    blowdown_loss_kw,
    surface_loss_kw,
    other_losses_kw,

    # Integration Connector Metrics
    connector_health,
    connector_latency_seconds,
    connector_requests_total,
    connector_errors_total,
    energy_meter_readings_total,
    historian_queries_total,
    scada_subscriptions_active,
    erp_cost_queries_total,

    # API Performance Metrics
    http_requests_total,
    http_request_duration_seconds,
    http_request_size_bytes,
    http_response_size_bytes,

    # Business Outcome Metrics
    efficiency_improvements_identified,
    energy_savings_potential_kwh,
    cost_savings_potential_usd,
    co2_reduction_potential_kg,
    roi_percent,

    # Metric Registry
    REGISTRY,

    # Utility Functions
    record_calculation,
    record_loss_breakdown,
    record_connector_call,
    record_http_request,
    record_business_outcome,
    update_health_metrics,
)

__all__ = [
    # Agent Health
    'agent_health_status',
    'agent_uptime_seconds',
    'agent_restart_count',
    'agent_memory_usage_bytes',
    'agent_cpu_usage_percent',
    'cache_hit_rate',
    'cache_size_bytes',
    'active_calculations',
    'queue_depth',
    'error_count',

    # Thermal Efficiency
    'calculation_duration_seconds',
    'first_law_efficiency_percent',
    'second_law_efficiency_percent',
    'energy_input_kw',
    'useful_output_kw',
    'total_losses_kw',
    'heat_balance_error_percent',
    'calculations_total',
    'calculation_errors_total',
    'benchmark_gap_percent',
    'improvement_potential_kw',
    'uncertainty_percent',
    'provenance_hash_count',
    'determinism_verified',
    'sankey_generation_duration_seconds',

    # Loss Breakdown
    'radiation_loss_kw',
    'convection_loss_kw',
    'conduction_loss_kw',
    'flue_gas_loss_kw',
    'unburned_fuel_loss_kw',
    'blowdown_loss_kw',
    'surface_loss_kw',
    'other_losses_kw',

    # Integration
    'connector_health',
    'connector_latency_seconds',
    'connector_requests_total',
    'connector_errors_total',
    'energy_meter_readings_total',
    'historian_queries_total',
    'scada_subscriptions_active',
    'erp_cost_queries_total',

    # API Performance
    'http_requests_total',
    'http_request_duration_seconds',
    'http_request_size_bytes',
    'http_response_size_bytes',

    # Business Outcomes
    'efficiency_improvements_identified',
    'energy_savings_potential_kwh',
    'cost_savings_potential_usd',
    'co2_reduction_potential_kg',
    'roi_percent',

    # Registry and Utilities
    'REGISTRY',
    'record_calculation',
    'record_loss_breakdown',
    'record_connector_call',
    'record_http_request',
    'record_business_outcome',
    'update_health_metrics',
]

__version__ = '1.0.0'
