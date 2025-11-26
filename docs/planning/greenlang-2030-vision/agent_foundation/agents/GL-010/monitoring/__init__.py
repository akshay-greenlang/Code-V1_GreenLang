"""
GL-010 EMISSIONWATCH Monitoring Module

This module provides comprehensive monitoring infrastructure for the
EmissionsComplianceAgent, including Prometheus metrics, Grafana dashboards,
alert definitions, and SLO tracking.

Components:
-----------
- metrics.py: 130+ Prometheus metrics for emissions, compliance, CEMS, and system health
- grafana/: Grafana dashboard JSON definitions
  - gl010_emissions_dashboard.json: Real-time emissions monitoring
  - gl010_compliance_dashboard.json: Compliance status and violations
  - gl010_cems_operations.json: CEMS operations and maintenance
- alerts/: Prometheus alerting rules
  - prometheus_alerts.yaml: 55+ alert rules for all monitoring categories
- SLO_DEFINITIONS.md: 9 Service Level Objectives with specifications

Usage:
------
>>> from gl010_monitoring.metrics import (
...     nox_emissions_ppm,
...     compliance_status,
...     cems_data_quality,
...     record_emissions_reading,
...     record_compliance_status,
... )
>>>
>>> # Record an emissions reading
>>> record_emissions_reading(
...     pollutant='NOx',
...     value=45.2,
...     unit='ppm',
...     source='boiler_1',
...     stack='stack_a'
... )
>>>
>>> # Record compliance status
>>> record_compliance_status(
...     jurisdiction='EPA',
...     pollutant='NOx',
...     is_compliant=True,
...     margin_percent=15.3,
...     permit_id='PSD-2024-001',
...     source='boiler_1'
... )

Metrics Categories:
-------------------
1. Emissions Metrics (50+)
   - NOx, SOx, CO2, CO, PM, VOC concentrations and rates
   - Daily, monthly, annual totals
   - Rolling averages (24hr, 30-day)
   - Stack parameters (flow, temperature, oxygen)

2. Compliance Metrics (15+)
   - Compliance status by jurisdiction/pollutant
   - Violation counts and durations
   - Permit expiration tracking
   - Regulatory deadlines

3. CEMS Metrics (25+)
   - Data quality scores
   - Availability percentages
   - Calibration status
   - Drift measurements
   - RATA/linearity status

4. Reporting Metrics (15+)
   - Report generation counts
   - Submission status
   - Deadline tracking
   - Error counts

5. Alert Metrics (10+)
   - Alert counts by severity
   - Response times
   - Acknowledgment tracking

6. Performance Metrics (20+)
   - Request latencies
   - Cache hit rates
   - Database performance
   - Queue depths

7. System Health Metrics (15+)
   - Memory/CPU usage
   - Disk space
   - Thread counts
   - Health check status

Alert Categories:
-----------------
1. Emissions Critical: Permit exceedances, violations
2. Emissions Warning: Approaching limits (80%, 90%, 95%)
3. CEMS Critical: System down, calibration invalid
4. CEMS Warning: Drift, maintenance due
5. Reporting: Deadlines, submission failures
6. System Health: Resource usage, connectivity

SLO Definitions:
----------------
1. CEMS Data Availability: 99.9%
2. Emissions Calculation Accuracy: 99.99%
3. Compliance Check Latency: <100ms p99
4. Alert Notification Latency: <30s
5. Report Submission Success: 99.9%
6. Zero Missed Violations: 100%
7. CEMS Calibration Compliance: 100%
8. System Uptime: 99.95%
9. API Response Time: <500ms p99

Version History:
----------------
1.0.0 - Initial release with comprehensive monitoring infrastructure
"""

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent__ = "GL-010 EMISSIONWATCH"

# Import all metrics for convenient access
from .metrics import (
    # Agent Information
    AGENT_INFO,
    AGENT_VERSION,
    AGENT_CONFIG,

    # NOx Emissions
    nox_emissions_ppm,
    nox_emissions_rate,
    nox_emissions_kg_hr,
    nox_limit_percent,
    nox_daily_total_tons,
    nox_monthly_total_tons,
    nox_annual_total_tons,
    nox_rolling_avg_24hr,
    nox_rolling_avg_30day,

    # SOx Emissions
    sox_emissions_ppm,
    sox_emissions_rate,
    sox_emissions_kg_hr,
    sox_limit_percent,
    sox_daily_total_tons,
    sox_monthly_total_tons,
    sox_annual_total_tons,
    sox_rolling_avg_24hr,

    # CO2 Emissions
    co2_emissions_tons_hr,
    co2_emissions_mtco2e,
    co2_emissions_kg_hr,
    co2_concentration_percent,
    co2_daily_total_tons,
    co2_monthly_total_tons,
    co2_annual_total_mtco2e,
    co2_intensity_kg_mwh,
    co2_cap_utilization_percent,

    # CO Emissions
    co_emissions_ppm,
    co_emissions_rate,
    co_limit_percent,
    co_daily_total_tons,

    # PM Emissions
    pm_emissions_mg_m3,
    pm_emissions_rate,
    pm10_emissions_ug_m3,
    pm25_emissions_ug_m3,
    pm_limit_percent,
    pm_daily_total_tons,
    opacity_percent,
    opacity_limit_percent,

    # VOC Emissions
    voc_emissions_ppm,
    voc_emissions_rate,
    voc_limit_percent,
    voc_daily_total_tons,

    # Other Pollutants
    hcl_emissions_ppm,
    hf_emissions_ppm,
    hg_emissions_ug_m3,
    ammonia_emissions_ppm,
    ammonia_slip_percent,

    # Stack Parameters
    stack_flow_rate_acfm,
    stack_flow_rate_scfm,
    stack_temperature_f,
    stack_temperature_c,
    stack_pressure_inwc,
    stack_oxygen_percent,
    stack_moisture_percent,
    diluent_co2_percent,
    diluent_o2_percent,

    # Compliance
    compliance_status,
    compliance_margin_percent,
    compliance_score,
    violations_total,
    violations_active,
    violations_resolved_total,
    exceedance_duration_seconds,
    exceedance_magnitude_percent,
    time_to_compliance_seconds,
    permit_expiration_days,
    regulatory_deadline_days,
    enforcement_actions_total,

    # CEMS
    cems_data_points_total,
    cems_data_quality,
    cems_availability_percent,
    cems_uptime_seconds,
    cems_downtime_seconds,
    cems_calibration_status,
    cems_calibration_days_remaining,
    cems_calibration_error_percent,
    cems_drift_percent,
    cems_drift_checks_total,
    cems_cylinder_gas_days_remaining,
    cems_response_time_seconds,
    cems_maintenance_status,
    cems_maintenance_days_until,
    cems_rata_status,
    cems_rata_days_remaining,
    cems_linearity_check_status,
    cems_substitute_data_percent,

    # Reporting
    reports_generated_total,
    reports_submitted_total,
    reports_accepted_total,
    reports_rejected_total,
    report_generation_duration_seconds,
    report_submission_duration_seconds,
    report_errors_total,
    report_validation_errors_total,
    report_queue_size,
    report_deadline_days,
    quarterly_report_status,
    annual_report_status,
    ecmps_submission_status,

    # Alerts
    alerts_triggered_total,
    alerts_active,
    alerts_acknowledged_total,
    alerts_auto_resolved_total,
    alerts_escalated_total,
    alert_response_time_seconds,
    alert_resolution_time_seconds,
    alert_notification_duration_seconds,
    alert_notification_failures_total,
    alert_suppressed_total,

    # Calculations
    calculations_total,
    calculation_duration_seconds,
    calculation_errors_total,
    calculation_retries_total,
    emission_factor_lookups_total,
    emission_factor_cache_hits,
    emission_factor_cache_misses,
    mass_balance_calculations_total,
    f_factor_calculations_total,
    provenance_hashes_generated,

    # Integrations
    dcs_connection_status,
    dcs_data_points_received,
    historian_connection_status,
    historian_queries_total,
    historian_query_duration_seconds,
    regulatory_portal_connection_status,
    regulatory_api_calls_total,
    regulatory_api_errors_total,

    # Performance
    request_duration_seconds,
    request_size_bytes,
    response_size_bytes,
    requests_in_flight,
    cache_hits_total,
    cache_misses_total,
    cache_size_bytes,
    cache_items_count,
    cache_evictions_total,

    # System Health
    memory_usage_bytes,
    memory_limit_bytes,
    cpu_usage_percent,
    disk_usage_bytes,
    disk_usage_percent,
    open_file_descriptors,
    thread_count,
    goroutine_count,
    gc_pause_duration_seconds,
    gc_collections_total,
    uptime_seconds,
    start_time_seconds,
    last_successful_calculation_timestamp,
    health_check_status,
    health_check_duration_seconds,

    # Database
    db_connection_pool_size,
    db_connections_active,
    db_connections_idle,
    db_query_duration_seconds,
    db_queries_total,
    db_errors_total,
    db_transactions_total,

    # Queue
    queue_size,
    queue_messages_total,
    queue_message_age_seconds,
    queue_processing_duration_seconds,
    queue_dead_letter_total,

    # Audit
    audit_events_total,
    audit_trail_size_bytes,
    data_lineage_records_total,
    provenance_verification_total,
    provenance_verification_failures,

    # Batch
    batch_jobs_total,
    batch_job_duration_seconds,
    batch_records_processed_total,
    batch_job_errors_total,
    batch_job_queue_size,

    # Helper Functions
    initialize_agent_info,
    record_emissions_reading,
    record_compliance_status,
    record_cems_data_point,
    record_calculation,
    record_alert,
    record_report_generation,

    # Metrics List
    ALL_METRICS,
    METRICS_COUNT,
)

# Module-level constants
GRAFANA_DASHBOARDS = [
    "gl010_emissions_dashboard.json",
    "gl010_compliance_dashboard.json",
    "gl010_cems_operations.json",
]

ALERT_RULES_FILE = "prometheus_alerts.yaml"

SLO_DOCUMENT = "SLO_DEFINITIONS.md"

# Export all public symbols
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__agent__",

    # Agent Information
    "AGENT_INFO",
    "AGENT_VERSION",
    "AGENT_CONFIG",

    # All emissions metrics
    "nox_emissions_ppm",
    "nox_emissions_rate",
    "nox_emissions_kg_hr",
    "nox_limit_percent",
    "nox_daily_total_tons",
    "nox_monthly_total_tons",
    "nox_annual_total_tons",
    "nox_rolling_avg_24hr",
    "nox_rolling_avg_30day",
    "sox_emissions_ppm",
    "sox_emissions_rate",
    "sox_emissions_kg_hr",
    "sox_limit_percent",
    "sox_daily_total_tons",
    "sox_monthly_total_tons",
    "sox_annual_total_tons",
    "sox_rolling_avg_24hr",
    "co2_emissions_tons_hr",
    "co2_emissions_mtco2e",
    "co2_emissions_kg_hr",
    "co2_concentration_percent",
    "co2_daily_total_tons",
    "co2_monthly_total_tons",
    "co2_annual_total_mtco2e",
    "co2_intensity_kg_mwh",
    "co2_cap_utilization_percent",
    "co_emissions_ppm",
    "co_emissions_rate",
    "co_limit_percent",
    "co_daily_total_tons",
    "pm_emissions_mg_m3",
    "pm_emissions_rate",
    "pm10_emissions_ug_m3",
    "pm25_emissions_ug_m3",
    "pm_limit_percent",
    "pm_daily_total_tons",
    "opacity_percent",
    "opacity_limit_percent",
    "voc_emissions_ppm",
    "voc_emissions_rate",
    "voc_limit_percent",
    "voc_daily_total_tons",
    "hcl_emissions_ppm",
    "hf_emissions_ppm",
    "hg_emissions_ug_m3",
    "ammonia_emissions_ppm",
    "ammonia_slip_percent",
    "stack_flow_rate_acfm",
    "stack_flow_rate_scfm",
    "stack_temperature_f",
    "stack_temperature_c",
    "stack_pressure_inwc",
    "stack_oxygen_percent",
    "stack_moisture_percent",
    "diluent_co2_percent",
    "diluent_o2_percent",

    # Compliance metrics
    "compliance_status",
    "compliance_margin_percent",
    "compliance_score",
    "violations_total",
    "violations_active",
    "violations_resolved_total",
    "exceedance_duration_seconds",
    "exceedance_magnitude_percent",
    "time_to_compliance_seconds",
    "permit_expiration_days",
    "regulatory_deadline_days",
    "enforcement_actions_total",

    # CEMS metrics
    "cems_data_points_total",
    "cems_data_quality",
    "cems_availability_percent",
    "cems_uptime_seconds",
    "cems_downtime_seconds",
    "cems_calibration_status",
    "cems_calibration_days_remaining",
    "cems_calibration_error_percent",
    "cems_drift_percent",
    "cems_drift_checks_total",
    "cems_cylinder_gas_days_remaining",
    "cems_response_time_seconds",
    "cems_maintenance_status",
    "cems_maintenance_days_until",
    "cems_rata_status",
    "cems_rata_days_remaining",
    "cems_linearity_check_status",
    "cems_substitute_data_percent",

    # Reporting metrics
    "reports_generated_total",
    "reports_submitted_total",
    "reports_accepted_total",
    "reports_rejected_total",
    "report_generation_duration_seconds",
    "report_submission_duration_seconds",
    "report_errors_total",
    "report_validation_errors_total",
    "report_queue_size",
    "report_deadline_days",
    "quarterly_report_status",
    "annual_report_status",
    "ecmps_submission_status",

    # Alert metrics
    "alerts_triggered_total",
    "alerts_active",
    "alerts_acknowledged_total",
    "alerts_auto_resolved_total",
    "alerts_escalated_total",
    "alert_response_time_seconds",
    "alert_resolution_time_seconds",
    "alert_notification_duration_seconds",
    "alert_notification_failures_total",
    "alert_suppressed_total",

    # Calculation metrics
    "calculations_total",
    "calculation_duration_seconds",
    "calculation_errors_total",
    "calculation_retries_total",
    "emission_factor_lookups_total",
    "emission_factor_cache_hits",
    "emission_factor_cache_misses",
    "mass_balance_calculations_total",
    "f_factor_calculations_total",
    "provenance_hashes_generated",

    # Integration metrics
    "dcs_connection_status",
    "dcs_data_points_received",
    "historian_connection_status",
    "historian_queries_total",
    "historian_query_duration_seconds",
    "regulatory_portal_connection_status",
    "regulatory_api_calls_total",
    "regulatory_api_errors_total",

    # Performance metrics
    "request_duration_seconds",
    "request_size_bytes",
    "response_size_bytes",
    "requests_in_flight",
    "cache_hits_total",
    "cache_misses_total",
    "cache_size_bytes",
    "cache_items_count",
    "cache_evictions_total",

    # System health metrics
    "memory_usage_bytes",
    "memory_limit_bytes",
    "cpu_usage_percent",
    "disk_usage_bytes",
    "disk_usage_percent",
    "open_file_descriptors",
    "thread_count",
    "goroutine_count",
    "gc_pause_duration_seconds",
    "gc_collections_total",
    "uptime_seconds",
    "start_time_seconds",
    "last_successful_calculation_timestamp",
    "health_check_status",
    "health_check_duration_seconds",

    # Database metrics
    "db_connection_pool_size",
    "db_connections_active",
    "db_connections_idle",
    "db_query_duration_seconds",
    "db_queries_total",
    "db_errors_total",
    "db_transactions_total",

    # Queue metrics
    "queue_size",
    "queue_messages_total",
    "queue_message_age_seconds",
    "queue_processing_duration_seconds",
    "queue_dead_letter_total",

    # Audit metrics
    "audit_events_total",
    "audit_trail_size_bytes",
    "data_lineage_records_total",
    "provenance_verification_total",
    "provenance_verification_failures",

    # Batch metrics
    "batch_jobs_total",
    "batch_job_duration_seconds",
    "batch_records_processed_total",
    "batch_job_errors_total",
    "batch_job_queue_size",

    # Helper functions
    "initialize_agent_info",
    "record_emissions_reading",
    "record_compliance_status",
    "record_cems_data_point",
    "record_calculation",
    "record_alert",
    "record_report_generation",

    # Module constants
    "ALL_METRICS",
    "METRICS_COUNT",
    "GRAFANA_DASHBOARDS",
    "ALERT_RULES_FILE",
    "SLO_DOCUMENT",
]
