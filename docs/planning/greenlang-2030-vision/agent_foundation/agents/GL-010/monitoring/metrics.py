"""
GL-010 EMISSIONWATCH Prometheus Metrics

This module defines comprehensive Prometheus metrics for the EmissionsComplianceAgent,
covering emissions monitoring, compliance tracking, CEMS operations, reporting,
alerting, and system performance.

The metrics follow Prometheus naming conventions:
- Counter: Monotonically increasing values (totals)
- Gauge: Values that can go up and down (current state)
- Histogram: Observations bucketed by value (latencies, durations)
- Summary: Quantile observations (similar to histogram but client-side)
- Info: Static metadata about the service

Usage:
    >>> from gl010_monitoring.metrics import (
    ...     nox_emissions_ppm,
    ...     compliance_status,
    ...     cems_data_quality
    ... )
    >>> nox_emissions_ppm.labels(source='boiler_1', stack='stack_a').set(45.2)
    >>> compliance_status.labels(jurisdiction='EPA', pollutant='NOx').set(1)
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, Info
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# AGENT INFORMATION
# =============================================================================

AGENT_INFO = Info(
    'gl010_emissionwatch',
    'GL-010 EMISSIONWATCH EmissionsComplianceAgent information'
)

AGENT_VERSION = Info(
    'gl010_version',
    'GL-010 agent version and build information'
)

AGENT_CONFIG = Info(
    'gl010_config',
    'GL-010 agent configuration metadata'
)

# =============================================================================
# EMISSIONS METRICS - NOx (Nitrogen Oxides)
# =============================================================================

nox_emissions_ppm = Gauge(
    'gl010_nox_emissions_ppm',
    'Current NOx emissions concentration in parts per million (ppm)',
    ['source', 'stack', 'unit_id']
)

nox_emissions_rate = Gauge(
    'gl010_nox_emissions_lb_hr',
    'NOx emission rate in pounds per hour (lb/hr)',
    ['source', 'unit_id']
)

nox_emissions_kg_hr = Gauge(
    'gl010_nox_emissions_kg_hr',
    'NOx emission rate in kilograms per hour (kg/hr)',
    ['source', 'unit_id']
)

nox_limit_percent = Gauge(
    'gl010_nox_limit_percent',
    'NOx emissions as percentage of permit limit (100 = at limit)',
    ['jurisdiction', 'permit_id', 'source']
)

nox_daily_total_tons = Gauge(
    'gl010_nox_daily_total_tons',
    'Daily cumulative NOx emissions in tons',
    ['source', 'date']
)

nox_monthly_total_tons = Gauge(
    'gl010_nox_monthly_total_tons',
    'Monthly cumulative NOx emissions in tons',
    ['source', 'year_month']
)

nox_annual_total_tons = Gauge(
    'gl010_nox_annual_total_tons',
    'Annual cumulative NOx emissions in tons',
    ['source', 'year']
)

nox_rolling_avg_24hr = Gauge(
    'gl010_nox_rolling_avg_24hr_ppm',
    '24-hour rolling average NOx concentration in ppm',
    ['source', 'stack']
)

nox_rolling_avg_30day = Gauge(
    'gl010_nox_rolling_avg_30day_lb_hr',
    '30-day rolling average NOx emission rate in lb/hr',
    ['source']
)

# =============================================================================
# EMISSIONS METRICS - SOx (Sulfur Oxides)
# =============================================================================

sox_emissions_ppm = Gauge(
    'gl010_sox_emissions_ppm',
    'Current SOx emissions concentration in parts per million (ppm)',
    ['source', 'stack', 'unit_id']
)

sox_emissions_rate = Gauge(
    'gl010_sox_emissions_lb_hr',
    'SOx emission rate in pounds per hour (lb/hr)',
    ['source', 'unit_id']
)

sox_emissions_kg_hr = Gauge(
    'gl010_sox_emissions_kg_hr',
    'SOx emission rate in kilograms per hour (kg/hr)',
    ['source', 'unit_id']
)

sox_limit_percent = Gauge(
    'gl010_sox_limit_percent',
    'SOx emissions as percentage of permit limit',
    ['jurisdiction', 'permit_id', 'source']
)

sox_daily_total_tons = Gauge(
    'gl010_sox_daily_total_tons',
    'Daily cumulative SOx emissions in tons',
    ['source', 'date']
)

sox_monthly_total_tons = Gauge(
    'gl010_sox_monthly_total_tons',
    'Monthly cumulative SOx emissions in tons',
    ['source', 'year_month']
)

sox_annual_total_tons = Gauge(
    'gl010_sox_annual_total_tons',
    'Annual cumulative SOx emissions in tons',
    ['source', 'year']
)

sox_rolling_avg_24hr = Gauge(
    'gl010_sox_rolling_avg_24hr_ppm',
    '24-hour rolling average SOx concentration in ppm',
    ['source', 'stack']
)

# =============================================================================
# EMISSIONS METRICS - CO2 (Carbon Dioxide)
# =============================================================================

co2_emissions_tons_hr = Gauge(
    'gl010_co2_emissions_tons_hr',
    'CO2 emission rate in short tons per hour',
    ['source', 'unit_id']
)

co2_emissions_mtco2e = Gauge(
    'gl010_co2_emissions_mtco2e',
    'CO2 equivalent emissions in metric tons (MT CO2e)',
    ['scope', 'source', 'category']
)

co2_emissions_kg_hr = Gauge(
    'gl010_co2_emissions_kg_hr',
    'CO2 emission rate in kilograms per hour',
    ['source', 'unit_id']
)

co2_concentration_percent = Gauge(
    'gl010_co2_concentration_percent',
    'CO2 concentration in stack gas as percentage',
    ['source', 'stack']
)

co2_daily_total_tons = Gauge(
    'gl010_co2_daily_total_tons',
    'Daily cumulative CO2 emissions in tons',
    ['source', 'date']
)

co2_monthly_total_tons = Gauge(
    'gl010_co2_monthly_total_tons',
    'Monthly cumulative CO2 emissions in tons',
    ['source', 'year_month']
)

co2_annual_total_mtco2e = Gauge(
    'gl010_co2_annual_total_mtco2e',
    'Annual cumulative CO2e emissions in metric tons',
    ['source', 'year', 'scope']
)

co2_intensity_kg_mwh = Gauge(
    'gl010_co2_intensity_kg_mwh',
    'CO2 emission intensity in kg CO2 per MWh generated',
    ['source', 'fuel_type']
)

co2_cap_utilization_percent = Gauge(
    'gl010_co2_cap_utilization_percent',
    'Utilization of CO2 emission cap as percentage',
    ['program', 'vintage_year']
)

# =============================================================================
# EMISSIONS METRICS - CO (Carbon Monoxide)
# =============================================================================

co_emissions_ppm = Gauge(
    'gl010_co_emissions_ppm',
    'Current CO emissions concentration in parts per million (ppm)',
    ['source', 'stack', 'unit_id']
)

co_emissions_rate = Gauge(
    'gl010_co_emissions_lb_hr',
    'CO emission rate in pounds per hour (lb/hr)',
    ['source', 'unit_id']
)

co_limit_percent = Gauge(
    'gl010_co_limit_percent',
    'CO emissions as percentage of permit limit',
    ['jurisdiction', 'source']
)

co_daily_total_tons = Gauge(
    'gl010_co_daily_total_tons',
    'Daily cumulative CO emissions in tons',
    ['source', 'date']
)

# =============================================================================
# EMISSIONS METRICS - Particulate Matter (PM)
# =============================================================================

pm_emissions_mg_m3 = Gauge(
    'gl010_pm_emissions_mg_m3',
    'Particulate matter concentration in milligrams per cubic meter',
    ['source', 'stack', 'size_class']
)

pm_emissions_rate = Gauge(
    'gl010_pm_emissions_lb_hr',
    'Particulate matter emission rate in pounds per hour',
    ['source', 'unit_id', 'size_class']
)

pm10_emissions_ug_m3 = Gauge(
    'gl010_pm10_emissions_ug_m3',
    'PM10 concentration in micrograms per cubic meter',
    ['source', 'stack']
)

pm25_emissions_ug_m3 = Gauge(
    'gl010_pm25_emissions_ug_m3',
    'PM2.5 concentration in micrograms per cubic meter',
    ['source', 'stack']
)

pm_limit_percent = Gauge(
    'gl010_pm_limit_percent',
    'PM emissions as percentage of permit limit',
    ['jurisdiction', 'source', 'size_class']
)

pm_daily_total_tons = Gauge(
    'gl010_pm_daily_total_tons',
    'Daily cumulative PM emissions in tons',
    ['source', 'date', 'size_class']
)

opacity_percent = Gauge(
    'gl010_opacity_percent',
    'Stack opacity as percentage (visibility reduction)',
    ['source', 'stack']
)

opacity_limit_percent = Gauge(
    'gl010_opacity_limit_percent',
    'Opacity as percentage of permit limit',
    ['jurisdiction', 'source']
)

# =============================================================================
# EMISSIONS METRICS - VOCs (Volatile Organic Compounds)
# =============================================================================

voc_emissions_ppm = Gauge(
    'gl010_voc_emissions_ppm',
    'VOC emissions concentration in parts per million',
    ['source', 'stack']
)

voc_emissions_rate = Gauge(
    'gl010_voc_emissions_lb_hr',
    'VOC emission rate in pounds per hour',
    ['source', 'unit_id']
)

voc_limit_percent = Gauge(
    'gl010_voc_limit_percent',
    'VOC emissions as percentage of permit limit',
    ['jurisdiction', 'source']
)

voc_daily_total_tons = Gauge(
    'gl010_voc_daily_total_tons',
    'Daily cumulative VOC emissions in tons',
    ['source', 'date']
)

# =============================================================================
# EMISSIONS METRICS - Other Pollutants
# =============================================================================

hcl_emissions_ppm = Gauge(
    'gl010_hcl_emissions_ppm',
    'Hydrogen chloride (HCl) concentration in ppm',
    ['source', 'stack']
)

hf_emissions_ppm = Gauge(
    'gl010_hf_emissions_ppm',
    'Hydrogen fluoride (HF) concentration in ppm',
    ['source', 'stack']
)

hg_emissions_ug_m3 = Gauge(
    'gl010_hg_emissions_ug_m3',
    'Mercury (Hg) concentration in micrograms per cubic meter',
    ['source', 'stack']
)

ammonia_emissions_ppm = Gauge(
    'gl010_ammonia_emissions_ppm',
    'Ammonia (NH3) slip concentration in ppm',
    ['source', 'stack']
)

ammonia_slip_percent = Gauge(
    'gl010_ammonia_slip_percent',
    'Ammonia slip as percentage of injection rate',
    ['source', 'scr_unit']
)

# =============================================================================
# EMISSIONS METRICS - Stack Parameters
# =============================================================================

stack_flow_rate_acfm = Gauge(
    'gl010_stack_flow_rate_acfm',
    'Stack gas flow rate in actual cubic feet per minute',
    ['source', 'stack']
)

stack_flow_rate_scfm = Gauge(
    'gl010_stack_flow_rate_scfm',
    'Stack gas flow rate in standard cubic feet per minute',
    ['source', 'stack']
)

stack_temperature_f = Gauge(
    'gl010_stack_temperature_f',
    'Stack gas temperature in degrees Fahrenheit',
    ['source', 'stack']
)

stack_temperature_c = Gauge(
    'gl010_stack_temperature_c',
    'Stack gas temperature in degrees Celsius',
    ['source', 'stack']
)

stack_pressure_inwc = Gauge(
    'gl010_stack_pressure_inwc',
    'Stack pressure in inches of water column',
    ['source', 'stack']
)

stack_oxygen_percent = Gauge(
    'gl010_stack_oxygen_percent',
    'Stack gas oxygen concentration as percentage',
    ['source', 'stack']
)

stack_moisture_percent = Gauge(
    'gl010_stack_moisture_percent',
    'Stack gas moisture content as percentage',
    ['source', 'stack']
)

diluent_co2_percent = Gauge(
    'gl010_diluent_co2_percent',
    'Diluent CO2 concentration for emission calculations',
    ['source', 'stack']
)

diluent_o2_percent = Gauge(
    'gl010_diluent_o2_percent',
    'Diluent O2 concentration for emission calculations',
    ['source', 'stack']
)

# =============================================================================
# COMPLIANCE METRICS
# =============================================================================

compliance_status = Gauge(
    'gl010_compliance_status',
    'Compliance status (1=compliant, 0=violation)',
    ['jurisdiction', 'pollutant', 'permit_id', 'source']
)

compliance_margin_percent = Gauge(
    'gl010_compliance_margin_percent',
    'Margin to permit limit as percentage (positive = under limit)',
    ['pollutant', 'jurisdiction', 'source']
)

compliance_score = Gauge(
    'gl010_compliance_score',
    'Overall compliance score from 0-100',
    ['facility_id', 'jurisdiction']
)

violations_total = Counter(
    'gl010_violations_total',
    'Total number of violations detected since startup',
    ['pollutant', 'severity', 'jurisdiction', 'source']
)

violations_active = Gauge(
    'gl010_violations_active',
    'Number of currently active (unresolved) violations',
    ['jurisdiction', 'severity']
)

violations_resolved_total = Counter(
    'gl010_violations_resolved_total',
    'Total violations resolved',
    ['pollutant', 'resolution_type']
)

exceedance_duration_seconds = Histogram(
    'gl010_exceedance_duration_seconds',
    'Duration of emission exceedances in seconds',
    ['pollutant', 'severity'],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400)
)

exceedance_magnitude_percent = Histogram(
    'gl010_exceedance_magnitude_percent',
    'Magnitude of exceedance as percent over limit',
    ['pollutant'],
    buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000)
)

time_to_compliance_seconds = Histogram(
    'gl010_time_to_compliance_seconds',
    'Time taken to return to compliance after exceedance',
    ['pollutant', 'source'],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800)
)

permit_expiration_days = Gauge(
    'gl010_permit_expiration_days',
    'Days until permit expiration (negative if expired)',
    ['permit_id', 'permit_type', 'jurisdiction']
)

regulatory_deadline_days = Gauge(
    'gl010_regulatory_deadline_days',
    'Days until regulatory reporting deadline',
    ['deadline_type', 'jurisdiction']
)

enforcement_actions_total = Counter(
    'gl010_enforcement_actions_total',
    'Total enforcement actions received',
    ['action_type', 'jurisdiction', 'severity']
)

# =============================================================================
# CEMS (Continuous Emissions Monitoring System) METRICS
# =============================================================================

cems_data_points_total = Counter(
    'gl010_cems_data_points_total',
    'Total CEMS data points received',
    ['analyzer', 'stack', 'pollutant']
)

cems_data_quality = Gauge(
    'gl010_cems_data_quality',
    'CEMS data quality score from 0-100',
    ['analyzer', 'stack', 'quality_aspect']
)

cems_availability_percent = Gauge(
    'gl010_cems_availability_percent',
    'CEMS data availability as percentage (regulatory requirement typically 90%+)',
    ['stack', 'pollutant', 'period']
)

cems_uptime_seconds = Counter(
    'gl010_cems_uptime_seconds_total',
    'Total CEMS uptime in seconds',
    ['analyzer', 'stack']
)

cems_downtime_seconds = Counter(
    'gl010_cems_downtime_seconds_total',
    'Total CEMS downtime in seconds',
    ['analyzer', 'stack', 'reason']
)

cems_calibration_status = Gauge(
    'gl010_cems_calibration_status',
    'CEMS calibration status (1=valid, 0=invalid/expired)',
    ['analyzer', 'stack', 'gas_type']
)

cems_calibration_days_remaining = Gauge(
    'gl010_cems_calibration_days_remaining',
    'Days until next calibration required',
    ['analyzer', 'stack']
)

cems_calibration_error_percent = Gauge(
    'gl010_cems_calibration_error_percent',
    'Calibration error as percentage (must be within tolerance)',
    ['analyzer', 'stack', 'gas_level']
)

cems_drift_percent = Gauge(
    'gl010_cems_drift_percent',
    'CEMS drift from calibration as percentage',
    ['analyzer', 'stack', 'direction', 'pollutant']
)

cems_drift_checks_total = Counter(
    'gl010_cems_drift_checks_total',
    'Total drift checks performed',
    ['analyzer', 'result']
)

cems_cylinder_gas_days_remaining = Gauge(
    'gl010_cems_cylinder_gas_days_remaining',
    'Days of calibration gas remaining',
    ['analyzer', 'gas_type', 'level']
)

cems_response_time_seconds = Gauge(
    'gl010_cems_response_time_seconds',
    'CEMS analyzer response time in seconds',
    ['analyzer', 'pollutant']
)

cems_maintenance_status = Gauge(
    'gl010_cems_maintenance_status',
    'CEMS maintenance status (1=ok, 0=maintenance required)',
    ['analyzer', 'stack']
)

cems_maintenance_days_until = Gauge(
    'gl010_cems_maintenance_days_until',
    'Days until scheduled maintenance',
    ['analyzer', 'maintenance_type']
)

cems_rata_status = Gauge(
    'gl010_cems_rata_status',
    'Relative Accuracy Test Audit (RATA) status (1=valid, 0=invalid)',
    ['stack', 'pollutant']
)

cems_rata_days_remaining = Gauge(
    'gl010_cems_rata_days_remaining',
    'Days until RATA expires',
    ['stack', 'pollutant']
)

cems_linearity_check_status = Gauge(
    'gl010_cems_linearity_check_status',
    'Linearity check status (1=pass, 0=fail)',
    ['analyzer', 'pollutant']
)

cems_substitute_data_percent = Gauge(
    'gl010_cems_substitute_data_percent',
    'Percentage of data using substitute data procedures',
    ['stack', 'pollutant', 'period']
)

# =============================================================================
# REPORTING METRICS
# =============================================================================

reports_generated_total = Counter(
    'gl010_reports_generated_total',
    'Total reports generated',
    ['type', 'jurisdiction', 'format']
)

reports_submitted_total = Counter(
    'gl010_reports_submitted_total',
    'Total reports submitted to regulatory portals',
    ['portal', 'report_type', 'jurisdiction']
)

reports_accepted_total = Counter(
    'gl010_reports_accepted_total',
    'Total reports accepted by regulatory agencies',
    ['portal', 'report_type']
)

reports_rejected_total = Counter(
    'gl010_reports_rejected_total',
    'Total reports rejected by regulatory agencies',
    ['portal', 'report_type', 'rejection_reason']
)

report_generation_duration_seconds = Histogram(
    'gl010_report_generation_duration_seconds',
    'Time to generate reports in seconds',
    ['report_type', 'complexity'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)
)

report_submission_duration_seconds = Histogram(
    'gl010_report_submission_duration_seconds',
    'Time to submit reports in seconds',
    ['portal', 'report_type'],
    buckets=(1, 5, 10, 30, 60, 120, 300)
)

report_errors_total = Counter(
    'gl010_report_errors_total',
    'Total report generation errors',
    ['type', 'error_category']
)

report_validation_errors_total = Counter(
    'gl010_report_validation_errors_total',
    'Total report validation errors',
    ['report_type', 'validation_rule']
)

report_queue_size = Gauge(
    'gl010_report_queue_size',
    'Number of reports pending generation',
    ['report_type', 'priority']
)

report_deadline_days = Gauge(
    'gl010_report_deadline_days',
    'Days until report submission deadline',
    ['report_type', 'jurisdiction', 'period']
)

quarterly_report_status = Gauge(
    'gl010_quarterly_report_status',
    'Quarterly report status (0=not started, 1=in progress, 2=completed, 3=submitted)',
    ['quarter', 'year', 'report_type']
)

annual_report_status = Gauge(
    'gl010_annual_report_status',
    'Annual report status (0=not started, 1=in progress, 2=completed, 3=submitted)',
    ['year', 'report_type', 'jurisdiction']
)

ecmps_submission_status = Gauge(
    'gl010_ecmps_submission_status',
    'EPA ECMPS submission status',
    ['quarter', 'year', 'file_type']
)

# =============================================================================
# ALERT METRICS
# =============================================================================

alerts_triggered_total = Counter(
    'gl010_alerts_triggered_total',
    'Total alerts triggered',
    ['severity', 'pollutant', 'alert_type', 'source']
)

alerts_active = Gauge(
    'gl010_alerts_active',
    'Number of currently active alerts',
    ['severity', 'category']
)

alerts_acknowledged_total = Counter(
    'gl010_alerts_acknowledged_total',
    'Total alerts acknowledged by operators',
    ['severity', 'acknowledged_by']
)

alerts_auto_resolved_total = Counter(
    'gl010_alerts_auto_resolved_total',
    'Total alerts automatically resolved',
    ['alert_type', 'resolution_reason']
)

alerts_escalated_total = Counter(
    'gl010_alerts_escalated_total',
    'Total alerts escalated to higher severity',
    ['original_severity', 'escalated_severity', 'alert_type']
)

alert_response_time_seconds = Histogram(
    'gl010_alert_response_time_seconds',
    'Time to acknowledge alerts in seconds',
    ['severity', 'alert_type'],
    buckets=(10, 30, 60, 120, 300, 600, 1800, 3600)
)

alert_resolution_time_seconds = Histogram(
    'gl010_alert_resolution_time_seconds',
    'Time to resolve alerts in seconds',
    ['severity', 'alert_type'],
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400)
)

alert_notification_duration_seconds = Histogram(
    'gl010_alert_notification_duration_seconds',
    'Time to send alert notifications in seconds',
    ['channel', 'severity'],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30)
)

alert_notification_failures_total = Counter(
    'gl010_alert_notification_failures_total',
    'Total alert notification failures',
    ['channel', 'failure_reason']
)

alert_suppressed_total = Counter(
    'gl010_alert_suppressed_total',
    'Total alerts suppressed due to rules',
    ['alert_type', 'suppression_reason']
)

# =============================================================================
# CALCULATION METRICS
# =============================================================================

calculations_total = Counter(
    'gl010_calculations_total',
    'Total calculations performed',
    ['type', 'method', 'pollutant']
)

calculation_duration_seconds = Histogram(
    'gl010_calculation_duration_seconds',
    'Time to perform calculations in seconds',
    ['type', 'complexity'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10)
)

calculation_errors_total = Counter(
    'gl010_calculation_errors_total',
    'Total calculation errors',
    ['type', 'error_category']
)

calculation_retries_total = Counter(
    'gl010_calculation_retries_total',
    'Total calculation retries',
    ['type', 'retry_reason']
)

emission_factor_lookups_total = Counter(
    'gl010_emission_factor_lookups_total',
    'Total emission factor database lookups',
    ['source_type', 'pollutant']
)

emission_factor_cache_hits = Counter(
    'gl010_emission_factor_cache_hits_total',
    'Emission factor cache hits',
    ['pollutant']
)

emission_factor_cache_misses = Counter(
    'gl010_emission_factor_cache_misses_total',
    'Emission factor cache misses',
    ['pollutant']
)

mass_balance_calculations_total = Counter(
    'gl010_mass_balance_calculations_total',
    'Total mass balance calculations',
    ['material_type']
)

f_factor_calculations_total = Counter(
    'gl010_f_factor_calculations_total',
    'Total F-factor method calculations',
    ['fuel_type', 'f_factor_type']
)

provenance_hashes_generated = Counter(
    'gl010_provenance_hashes_generated_total',
    'Total SHA-256 provenance hashes generated',
    ['data_type']
)

# =============================================================================
# INTEGRATION METRICS
# =============================================================================

dcs_connection_status = Gauge(
    'gl010_dcs_connection_status',
    'Distributed Control System connection status (1=connected, 0=disconnected)',
    ['dcs_system', 'facility']
)

dcs_data_points_received = Counter(
    'gl010_dcs_data_points_received_total',
    'Total data points received from DCS',
    ['dcs_system', 'data_type']
)

historian_connection_status = Gauge(
    'gl010_historian_connection_status',
    'Process historian connection status',
    ['historian_system']
)

historian_queries_total = Counter(
    'gl010_historian_queries_total',
    'Total queries to process historian',
    ['historian_system', 'query_type']
)

historian_query_duration_seconds = Histogram(
    'gl010_historian_query_duration_seconds',
    'Process historian query duration',
    ['historian_system', 'query_type'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30)
)

regulatory_portal_connection_status = Gauge(
    'gl010_regulatory_portal_connection_status',
    'Regulatory portal connection status',
    ['portal', 'jurisdiction']
)

regulatory_api_calls_total = Counter(
    'gl010_regulatory_api_calls_total',
    'Total API calls to regulatory systems',
    ['portal', 'endpoint', 'method']
)

regulatory_api_errors_total = Counter(
    'gl010_regulatory_api_errors_total',
    'Total regulatory API errors',
    ['portal', 'error_type']
)

# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

request_duration_seconds = Histogram(
    'gl010_request_duration_seconds',
    'HTTP request duration in seconds',
    ['endpoint', 'method', 'status_code'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)

request_size_bytes = Histogram(
    'gl010_request_size_bytes',
    'HTTP request size in bytes',
    ['endpoint', 'method'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000)
)

response_size_bytes = Histogram(
    'gl010_response_size_bytes',
    'HTTP response size in bytes',
    ['endpoint', 'method'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000)
)

requests_in_flight = Gauge(
    'gl010_requests_in_flight',
    'Number of requests currently being processed',
    ['endpoint']
)

cache_hits_total = Counter(
    'gl010_cache_hits_total',
    'Total cache hits',
    ['cache_type', 'cache_name']
)

cache_misses_total = Counter(
    'gl010_cache_misses_total',
    'Total cache misses',
    ['cache_type', 'cache_name']
)

cache_size_bytes = Gauge(
    'gl010_cache_size_bytes',
    'Current cache size in bytes',
    ['cache_type', 'cache_name']
)

cache_items_count = Gauge(
    'gl010_cache_items_count',
    'Number of items in cache',
    ['cache_type', 'cache_name']
)

cache_evictions_total = Counter(
    'gl010_cache_evictions_total',
    'Total cache evictions',
    ['cache_type', 'cache_name', 'eviction_reason']
)

# =============================================================================
# SYSTEM HEALTH METRICS
# =============================================================================

memory_usage_bytes = Gauge(
    'gl010_memory_usage_bytes',
    'Current memory usage in bytes',
    ['memory_type']
)

memory_limit_bytes = Gauge(
    'gl010_memory_limit_bytes',
    'Memory limit in bytes',
    ['memory_type']
)

cpu_usage_percent = Gauge(
    'gl010_cpu_usage_percent',
    'Current CPU usage as percentage',
    ['cpu_type']
)

disk_usage_bytes = Gauge(
    'gl010_disk_usage_bytes',
    'Current disk usage in bytes',
    ['mount_point', 'usage_type']
)

disk_usage_percent = Gauge(
    'gl010_disk_usage_percent',
    'Current disk usage as percentage',
    ['mount_point']
)

open_file_descriptors = Gauge(
    'gl010_open_file_descriptors',
    'Number of open file descriptors'
)

thread_count = Gauge(
    'gl010_thread_count',
    'Number of active threads',
    ['thread_type']
)

goroutine_count = Gauge(
    'gl010_goroutine_count',
    'Number of active goroutines (if applicable)'
)

gc_pause_duration_seconds = Summary(
    'gl010_gc_pause_duration_seconds',
    'Garbage collection pause duration'
)

gc_collections_total = Counter(
    'gl010_gc_collections_total',
    'Total garbage collection cycles',
    ['generation']
)

uptime_seconds = Gauge(
    'gl010_uptime_seconds',
    'Agent uptime in seconds'
)

start_time_seconds = Gauge(
    'gl010_start_time_seconds',
    'Agent start time as Unix timestamp'
)

last_successful_calculation_timestamp = Gauge(
    'gl010_last_successful_calculation_timestamp',
    'Timestamp of last successful calculation',
    ['calculation_type']
)

health_check_status = Gauge(
    'gl010_health_check_status',
    'Health check status (1=healthy, 0=unhealthy)',
    ['check_type']
)

health_check_duration_seconds = Histogram(
    'gl010_health_check_duration_seconds',
    'Health check duration in seconds',
    ['check_type'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1)
)

# =============================================================================
# DATABASE METRICS
# =============================================================================

db_connection_pool_size = Gauge(
    'gl010_db_connection_pool_size',
    'Database connection pool size',
    ['database', 'pool_type']
)

db_connections_active = Gauge(
    'gl010_db_connections_active',
    'Number of active database connections',
    ['database']
)

db_connections_idle = Gauge(
    'gl010_db_connections_idle',
    'Number of idle database connections',
    ['database']
)

db_query_duration_seconds = Histogram(
    'gl010_db_query_duration_seconds',
    'Database query duration in seconds',
    ['database', 'query_type'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10)
)

db_queries_total = Counter(
    'gl010_db_queries_total',
    'Total database queries',
    ['database', 'query_type', 'status']
)

db_errors_total = Counter(
    'gl010_db_errors_total',
    'Total database errors',
    ['database', 'error_type']
)

db_transactions_total = Counter(
    'gl010_db_transactions_total',
    'Total database transactions',
    ['database', 'transaction_type', 'status']
)

# =============================================================================
# QUEUE METRICS
# =============================================================================

queue_size = Gauge(
    'gl010_queue_size',
    'Current queue size',
    ['queue_name', 'priority']
)

queue_messages_total = Counter(
    'gl010_queue_messages_total',
    'Total messages processed from queue',
    ['queue_name', 'status']
)

queue_message_age_seconds = Histogram(
    'gl010_queue_message_age_seconds',
    'Age of messages when processed in seconds',
    ['queue_name'],
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600)
)

queue_processing_duration_seconds = Histogram(
    'gl010_queue_processing_duration_seconds',
    'Time to process queue messages in seconds',
    ['queue_name', 'message_type'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30)
)

queue_dead_letter_total = Counter(
    'gl010_queue_dead_letter_total',
    'Total messages sent to dead letter queue',
    ['queue_name', 'reason']
)

# =============================================================================
# AUDIT METRICS
# =============================================================================

audit_events_total = Counter(
    'gl010_audit_events_total',
    'Total audit events recorded',
    ['event_type', 'entity_type', 'user']
)

audit_trail_size_bytes = Gauge(
    'gl010_audit_trail_size_bytes',
    'Current audit trail storage size in bytes'
)

data_lineage_records_total = Counter(
    'gl010_data_lineage_records_total',
    'Total data lineage records created',
    ['data_type', 'source_system']
)

provenance_verification_total = Counter(
    'gl010_provenance_verification_total',
    'Total provenance hash verifications',
    ['verification_result']
)

provenance_verification_failures = Counter(
    'gl010_provenance_verification_failures_total',
    'Total provenance verification failures',
    ['failure_reason']
)

# =============================================================================
# BATCH PROCESSING METRICS
# =============================================================================

batch_jobs_total = Counter(
    'gl010_batch_jobs_total',
    'Total batch jobs executed',
    ['job_type', 'status']
)

batch_job_duration_seconds = Histogram(
    'gl010_batch_job_duration_seconds',
    'Batch job duration in seconds',
    ['job_type'],
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200)
)

batch_records_processed_total = Counter(
    'gl010_batch_records_processed_total',
    'Total records processed in batch jobs',
    ['job_type', 'status']
)

batch_job_errors_total = Counter(
    'gl010_batch_job_errors_total',
    'Total batch job errors',
    ['job_type', 'error_type']
)

batch_job_queue_size = Gauge(
    'gl010_batch_job_queue_size',
    'Number of batch jobs waiting to run',
    ['job_type']
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def initialize_agent_info(
    version: str,
    build_date: str,
    facility_id: str,
    facility_name: str,
    jurisdictions: str
) -> None:
    """
    Initialize agent information metrics.

    Args:
        version: Agent version string
        build_date: Build date in ISO format
        facility_id: Facility identifier
        facility_name: Human-readable facility name
        jurisdictions: Comma-separated list of jurisdictions
    """
    AGENT_INFO.info({
        'version': version,
        'build_date': build_date,
        'facility_id': facility_id,
        'facility_name': facility_name,
        'jurisdictions': jurisdictions
    })
    logger.info(f"Initialized GL-010 agent info: version={version}, facility={facility_id}")


def record_emissions_reading(
    pollutant: str,
    value: float,
    unit: str,
    source: str,
    stack: str,
    unit_id: str = 'default'
) -> None:
    """
    Record an emissions reading to appropriate metrics.

    Args:
        pollutant: Pollutant type (NOx, SOx, CO2, CO, PM, etc.)
        value: Measured value
        unit: Unit of measurement (ppm, lb/hr, tons/hr, etc.)
        source: Emission source identifier
        stack: Stack identifier
        unit_id: Unit identifier
    """
    pollutant_lower = pollutant.lower()

    metric_map = {
        ('nox', 'ppm'): lambda: nox_emissions_ppm.labels(source=source, stack=stack, unit_id=unit_id).set(value),
        ('nox', 'lb/hr'): lambda: nox_emissions_rate.labels(source=source, unit_id=unit_id).set(value),
        ('nox', 'kg/hr'): lambda: nox_emissions_kg_hr.labels(source=source, unit_id=unit_id).set(value),
        ('sox', 'ppm'): lambda: sox_emissions_ppm.labels(source=source, stack=stack, unit_id=unit_id).set(value),
        ('sox', 'lb/hr'): lambda: sox_emissions_rate.labels(source=source, unit_id=unit_id).set(value),
        ('sox', 'kg/hr'): lambda: sox_emissions_kg_hr.labels(source=source, unit_id=unit_id).set(value),
        ('co2', 'tons/hr'): lambda: co2_emissions_tons_hr.labels(source=source, unit_id=unit_id).set(value),
        ('co2', 'kg/hr'): lambda: co2_emissions_kg_hr.labels(source=source, unit_id=unit_id).set(value),
        ('co', 'ppm'): lambda: co_emissions_ppm.labels(source=source, stack=stack, unit_id=unit_id).set(value),
        ('co', 'lb/hr'): lambda: co_emissions_rate.labels(source=source, unit_id=unit_id).set(value),
        ('pm', 'mg/m3'): lambda: pm_emissions_mg_m3.labels(source=source, stack=stack, size_class='total').set(value),
        ('pm', 'lb/hr'): lambda: pm_emissions_rate.labels(source=source, unit_id=unit_id, size_class='total').set(value),
        ('voc', 'ppm'): lambda: voc_emissions_ppm.labels(source=source, stack=stack).set(value),
        ('voc', 'lb/hr'): lambda: voc_emissions_rate.labels(source=source, unit_id=unit_id).set(value),
    }

    key = (pollutant_lower, unit.lower())
    if key in metric_map:
        metric_map[key]()
        logger.debug(f"Recorded {pollutant} emission: {value} {unit} from {source}/{stack}")
    else:
        logger.warning(f"Unknown pollutant/unit combination: {pollutant}/{unit}")


def record_compliance_status(
    jurisdiction: str,
    pollutant: str,
    is_compliant: bool,
    margin_percent: float,
    permit_id: str,
    source: str
) -> None:
    """
    Record compliance status for a pollutant/jurisdiction combination.

    Args:
        jurisdiction: Regulatory jurisdiction (EPA, CARB, etc.)
        pollutant: Pollutant type
        is_compliant: True if in compliance, False if violation
        margin_percent: Margin to limit as percentage (positive = under limit)
        permit_id: Permit identifier
        source: Emission source
    """
    compliance_status.labels(
        jurisdiction=jurisdiction,
        pollutant=pollutant,
        permit_id=permit_id,
        source=source
    ).set(1 if is_compliant else 0)

    compliance_margin_percent.labels(
        pollutant=pollutant,
        jurisdiction=jurisdiction,
        source=source
    ).set(margin_percent)

    if not is_compliant:
        severity = 'critical' if margin_percent < -10 else 'warning'
        violations_total.labels(
            pollutant=pollutant,
            severity=severity,
            jurisdiction=jurisdiction,
            source=source
        ).inc()
        logger.warning(
            f"Compliance violation: {pollutant} in {jurisdiction}, "
            f"margin={margin_percent:.1f}%, source={source}"
        )


def record_cems_data_point(
    analyzer: str,
    stack: str,
    pollutant: str,
    quality_score: float
) -> None:
    """
    Record a CEMS data point and quality score.

    Args:
        analyzer: Analyzer identifier
        stack: Stack identifier
        pollutant: Pollutant being measured
        quality_score: Data quality score 0-100
    """
    cems_data_points_total.labels(
        analyzer=analyzer,
        stack=stack,
        pollutant=pollutant
    ).inc()

    cems_data_quality.labels(
        analyzer=analyzer,
        stack=stack,
        quality_aspect='overall'
    ).set(quality_score)


def record_calculation(
    calculation_type: str,
    duration_seconds: float,
    success: bool,
    method: str = 'default',
    pollutant: str = 'general'
) -> None:
    """
    Record a calculation execution.

    Args:
        calculation_type: Type of calculation performed
        duration_seconds: Time taken in seconds
        success: Whether calculation succeeded
        method: Calculation method used
        pollutant: Pollutant calculated (if applicable)
    """
    calculations_total.labels(
        type=calculation_type,
        method=method,
        pollutant=pollutant
    ).inc()

    calculation_duration_seconds.labels(
        type=calculation_type,
        complexity='standard'
    ).observe(duration_seconds)

    if not success:
        calculation_errors_total.labels(
            type=calculation_type,
            error_category='calculation_failed'
        ).inc()


def record_alert(
    severity: str,
    pollutant: str,
    alert_type: str,
    source: str
) -> None:
    """
    Record an alert being triggered.

    Args:
        severity: Alert severity (critical, warning, info)
        pollutant: Related pollutant
        alert_type: Type of alert
        source: Source that triggered the alert
    """
    alerts_triggered_total.labels(
        severity=severity,
        pollutant=pollutant,
        alert_type=alert_type,
        source=source
    ).inc()

    logger.info(f"Alert triggered: {alert_type} ({severity}) for {pollutant} at {source}")


def record_report_generation(
    report_type: str,
    jurisdiction: str,
    format_type: str,
    duration_seconds: float,
    success: bool
) -> None:
    """
    Record report generation metrics.

    Args:
        report_type: Type of report (quarterly, annual, etc.)
        jurisdiction: Regulatory jurisdiction
        format_type: Report format (XML, PDF, etc.)
        duration_seconds: Generation time in seconds
        success: Whether generation succeeded
    """
    if success:
        reports_generated_total.labels(
            type=report_type,
            jurisdiction=jurisdiction,
            format=format_type
        ).inc()
    else:
        report_errors_total.labels(
            type=report_type,
            error_category='generation_failed'
        ).inc()

    report_generation_duration_seconds.labels(
        report_type=report_type,
        complexity='standard'
    ).observe(duration_seconds)


# =============================================================================
# EXPORTED METRICS REGISTRY
# =============================================================================

# List of all metric objects for registration
ALL_METRICS = [
    # Agent Info
    AGENT_INFO,
    AGENT_VERSION,
    AGENT_CONFIG,
    # NOx
    nox_emissions_ppm,
    nox_emissions_rate,
    nox_emissions_kg_hr,
    nox_limit_percent,
    nox_daily_total_tons,
    nox_monthly_total_tons,
    nox_annual_total_tons,
    nox_rolling_avg_24hr,
    nox_rolling_avg_30day,
    # SOx
    sox_emissions_ppm,
    sox_emissions_rate,
    sox_emissions_kg_hr,
    sox_limit_percent,
    sox_daily_total_tons,
    sox_monthly_total_tons,
    sox_annual_total_tons,
    sox_rolling_avg_24hr,
    # CO2
    co2_emissions_tons_hr,
    co2_emissions_mtco2e,
    co2_emissions_kg_hr,
    co2_concentration_percent,
    co2_daily_total_tons,
    co2_monthly_total_tons,
    co2_annual_total_mtco2e,
    co2_intensity_kg_mwh,
    co2_cap_utilization_percent,
    # CO
    co_emissions_ppm,
    co_emissions_rate,
    co_limit_percent,
    co_daily_total_tons,
    # PM
    pm_emissions_mg_m3,
    pm_emissions_rate,
    pm10_emissions_ug_m3,
    pm25_emissions_ug_m3,
    pm_limit_percent,
    pm_daily_total_tons,
    opacity_percent,
    opacity_limit_percent,
    # VOC
    voc_emissions_ppm,
    voc_emissions_rate,
    voc_limit_percent,
    voc_daily_total_tons,
    # Other pollutants
    hcl_emissions_ppm,
    hf_emissions_ppm,
    hg_emissions_ug_m3,
    ammonia_emissions_ppm,
    ammonia_slip_percent,
    # Stack parameters
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
    # System health
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
]

# Count of metrics for verification
METRICS_COUNT = len(ALL_METRICS)

logger.info(f"GL-010 EMISSIONWATCH metrics module loaded: {METRICS_COUNT} metrics defined")
