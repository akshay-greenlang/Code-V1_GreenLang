# -*- coding: utf-8 -*-
"""
Prometheus metrics for GL-007 FurnacePerformanceMonitor.

Provides comprehensive metrics tracking for:
- HTTP request latency and throughput
- Furnace performance monitoring
- Thermal efficiency tracking
- Maintenance prediction accuracy
- System resource utilization
- SCADA integration health
- ML model performance
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
from typing import Optional, Callable
import functools
import time
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# HTTP REQUEST METRICS
# ==============================================================================

http_requests_total = Counter(
    'gl_007_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'gl_007_http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

http_request_size_bytes = Histogram(
    'gl_007_http_request_size_bytes',
    'HTTP request body size',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000)
)

http_response_size_bytes = Histogram(
    'gl_007_http_response_size_bytes',
    'HTTP response body size',
    ['method', 'endpoint'],
    buckets=(100, 1000, 10000, 100000, 1000000)
)

# ==============================================================================
# AGENT HEALTH METRICS
# ==============================================================================

agent_health_status = Gauge(
    'gl_007_agent_health_status',
    'Agent health (1=healthy, 0=unhealthy)'
)

furnace_monitoring_active = Gauge(
    'gl_007_furnace_monitoring_active',
    'Number of furnaces actively monitored'
)

calculation_duration_seconds = Histogram(
    'gl_007_calculation_duration_seconds',
    'Calculation execution time',
    ['calculation_type'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0)
)

calculation_requests_total = Counter(
    'gl_007_calculation_requests_total',
    'Total calculation requests',
    ['calculation_type', 'status']  # status: success, failure, timeout
)

# ==============================================================================
# FURNACE OPERATING METRICS
# ==============================================================================

furnace_thermal_efficiency = Gauge(
    'gl_007_furnace_thermal_efficiency_percent',
    'Furnace thermal efficiency (%)',
    ['furnace_id', 'zone']
)

furnace_fuel_consumption = Gauge(
    'gl_007_furnace_fuel_consumption_kg_hr',
    'Fuel consumption rate (kg/hr)',
    ['furnace_id', 'fuel_type']
)

furnace_temperature = Gauge(
    'gl_007_furnace_temperature_celsius',
    'Furnace temperature (°C)',
    ['furnace_id', 'zone', 'sensor_id']
)

furnace_pressure = Gauge(
    'gl_007_furnace_pressure_bar',
    'Furnace pressure (bar)',
    ['furnace_id', 'zone']
)

furnace_draft = Gauge(
    'gl_007_furnace_draft_pa',
    'Furnace draft (Pa)',
    ['furnace_id', 'zone']
)

furnace_oxygen_level = Gauge(
    'gl_007_furnace_oxygen_level_percent',
    'Oxygen level in flue gas (%)',
    ['furnace_id', 'zone']
)

furnace_production_rate = Gauge(
    'gl_007_furnace_production_rate_tons_hr',
    'Production rate (tons/hr)',
    ['furnace_id', 'product_type']
)

furnace_specific_energy_consumption = Gauge(
    'gl_007_furnace_specific_energy_consumption_kwh_ton',
    'Specific energy consumption (kWh/ton)',
    ['furnace_id', 'product_type']
)

# ==============================================================================
# THERMAL PERFORMANCE METRICS
# ==============================================================================

heat_recovery_efficiency = Gauge(
    'gl_007_heat_recovery_efficiency_percent',
    'Heat recovery system efficiency (%)',
    ['furnace_id', 'recovery_system']
)

heat_loss_rate = Gauge(
    'gl_007_heat_loss_rate_kw',
    'Heat loss rate (kW)',
    ['furnace_id', 'loss_type']  # radiation, convection, conduction, flue_gas
)

flame_temperature = Gauge(
    'gl_007_flame_temperature_celsius',
    'Flame temperature (°C)',
    ['furnace_id', 'burner_id']
)

wall_temperature = Gauge(
    'gl_007_wall_temperature_celsius',
    'Furnace wall temperature (°C)',
    ['furnace_id', 'zone', 'position']
)

refractory_temperature = Gauge(
    'gl_007_refractory_temperature_celsius',
    'Refractory lining temperature (°C)',
    ['furnace_id', 'zone', 'depth']
)

# ==============================================================================
# COMBUSTION METRICS
# ==============================================================================

air_fuel_ratio = Gauge(
    'gl_007_air_fuel_ratio',
    'Air-fuel ratio',
    ['furnace_id', 'burner_id']
)

excess_air_percent = Gauge(
    'gl_007_excess_air_percent',
    'Excess air percentage',
    ['furnace_id', 'zone']
)

combustion_efficiency = Gauge(
    'gl_007_combustion_efficiency_percent',
    'Combustion efficiency (%)',
    ['furnace_id', 'fuel_type']
)

flue_gas_temperature = Gauge(
    'gl_007_flue_gas_temperature_celsius',
    'Flue gas temperature (°C)',
    ['furnace_id', 'measurement_point']
)

# ==============================================================================
# MAINTENANCE METRICS
# ==============================================================================

maintenance_alerts_total = Counter(
    'gl_007_maintenance_alerts_total',
    'Total maintenance alerts generated',
    ['furnace_id', 'severity', 'alert_type']
)

maintenance_prediction_confidence = Histogram(
    'gl_007_maintenance_prediction_confidence',
    'Maintenance prediction confidence score',
    ['furnace_id', 'component'],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
)

component_remaining_life_hours = Gauge(
    'gl_007_component_remaining_life_hours',
    'Predicted remaining life (hours)',
    ['furnace_id', 'component']
)

refractory_degradation_rate = Gauge(
    'gl_007_refractory_degradation_rate_mm_day',
    'Refractory degradation rate (mm/day)',
    ['furnace_id', 'zone']
)

burner_performance_index = Gauge(
    'gl_007_burner_performance_index',
    'Burner performance index (0-100)',
    ['furnace_id', 'burner_id']
)

# ==============================================================================
# PREDICTION & ML METRICS
# ==============================================================================

prediction_accuracy = Histogram(
    'gl_007_prediction_accuracy',
    'ML model prediction accuracy',
    ['model_type'],
    buckets=(0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0)
)

model_inference_duration_seconds = Histogram(
    'gl_007_model_inference_duration_seconds',
    'ML model inference time',
    ['model_type'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
)

model_training_duration_seconds = Histogram(
    'gl_007_model_training_duration_seconds',
    'ML model training time',
    ['model_type'],
    buckets=(1, 10, 60, 300, 600, 1800, 3600)
)

model_prediction_error = Histogram(
    'gl_007_model_prediction_error',
    'ML model prediction error (RMSE)',
    ['model_type'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)
)

anomaly_detection_alerts = Counter(
    'gl_007_anomaly_detection_alerts_total',
    'Total anomaly detection alerts',
    ['furnace_id', 'anomaly_type', 'severity']
)

# ==============================================================================
# SCADA INTEGRATION METRICS
# ==============================================================================

scada_connection_status = Gauge(
    'gl_007_scada_connection_status',
    'SCADA connection status (1=connected, 0=disconnected)',
    ['scada_system']
)

scada_data_points_received = Counter(
    'gl_007_scada_data_points_received_total',
    'Total SCADA data points received',
    ['scada_system', 'tag_type']
)

scada_data_latency_seconds = Histogram(
    'gl_007_scada_data_latency_seconds',
    'SCADA data latency',
    ['scada_system'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
)

scada_polling_errors = Counter(
    'gl_007_scada_polling_errors_total',
    'Total SCADA polling errors',
    ['scada_system', 'error_type']
)

scada_tag_quality = Gauge(
    'gl_007_scada_tag_quality',
    'SCADA tag quality (1=good, 0=bad)',
    ['scada_system', 'tag_name']
)

# ==============================================================================
# PERFORMANCE CORRELATION METRICS
# ==============================================================================

energy_production_correlation = Gauge(
    'gl_007_energy_production_correlation',
    'Correlation between energy use and production',
    ['furnace_id']
)

efficiency_temperature_correlation = Gauge(
    'gl_007_efficiency_temperature_correlation',
    'Correlation between efficiency and temperature',
    ['furnace_id', 'zone']
)

production_quality_correlation = Gauge(
    'gl_007_production_quality_correlation',
    'Correlation between production rate and quality',
    ['furnace_id', 'product_type']
)

# ==============================================================================
# DATABASE METRICS
# ==============================================================================

db_connection_pool_size = Gauge(
    'gl_007_db_connection_pool_size',
    'Active database connections'
)

db_query_duration_seconds = Histogram(
    'gl_007_db_query_duration_seconds',
    'Database query latency',
    ['query_type'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
)

db_query_errors_total = Counter(
    'gl_007_db_query_errors_total',
    'Total database query errors',
    ['query_type', 'error_type']
)

time_series_data_points_stored = Counter(
    'gl_007_time_series_data_points_stored_total',
    'Total time-series data points stored',
    ['furnace_id', 'metric_type']
)

# ==============================================================================
# CACHE METRICS
# ==============================================================================

cache_hits_total = Counter(
    'gl_007_cache_hits_total',
    'Total cache hits',
    ['cache_key_pattern']
)

cache_misses_total = Counter(
    'gl_007_cache_misses_total',
    'Total cache misses',
    ['cache_key_pattern']
)

cache_evictions_total = Counter(
    'gl_007_cache_evictions_total',
    'Total cache evictions',
    ['cache_key_pattern']
)

cache_operation_duration_seconds = Histogram(
    'gl_007_cache_operation_duration_seconds',
    'Cache operation latency',
    ['operation'],  # get, set, delete
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1)
)

# ==============================================================================
# EXTERNAL API METRICS
# ==============================================================================

external_api_requests_total = Counter(
    'gl_007_external_api_requests_total',
    'Total external API requests',
    ['api_name', 'status_code']
)

external_api_duration_seconds = Histogram(
    'gl_007_external_api_duration_seconds',
    'External API latency',
    ['api_name'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

external_api_errors_total = Counter(
    'gl_007_external_api_errors_total',
    'Total external API errors',
    ['api_name', 'error_type']
)

# ==============================================================================
# SYSTEM METRICS
# ==============================================================================

system_uptime_seconds = Gauge(
    'gl_007_system_uptime_seconds',
    'Application uptime'
)

system_memory_usage_bytes = Gauge(
    'gl_007_system_memory_usage_bytes',
    'Memory usage',
    ['type']  # type: rss, vms, heap
)

system_cpu_usage_percent = Gauge(
    'gl_007_system_cpu_usage_percent',
    'CPU usage percentage'
)

system_disk_usage_bytes = Gauge(
    'gl_007_system_disk_usage_bytes',
    'Disk usage',
    ['mount_point']
)

goroutines_count = Gauge(
    'gl_007_goroutines_count',
    'Number of goroutines (if applicable)'
)

# ==============================================================================
# BUSINESS METRICS
# ==============================================================================

energy_cost_savings_usd_hr = Gauge(
    'gl_007_energy_cost_savings_usd_hr',
    'Energy cost savings ($/hr)',
    ['furnace_id']
)

annual_energy_savings_usd = Gauge(
    'gl_007_annual_energy_savings_usd',
    'Projected annual energy savings ($)',
    ['furnace_id']
)

carbon_emissions_reduction_kg_hr = Gauge(
    'gl_007_carbon_emissions_reduction_kg_hr',
    'Carbon emissions reduction (kg CO2/hr)',
    ['furnace_id']
)

annual_carbon_reduction_tons = Gauge(
    'gl_007_annual_carbon_reduction_tons',
    'Projected annual carbon reduction (tons CO2)',
    ['furnace_id']
)

maintenance_cost_avoidance_usd = Counter(
    'gl_007_maintenance_cost_avoidance_usd_total',
    'Maintenance cost avoidance through prediction ($)',
    ['furnace_id', 'component']
)

production_uptime_percent = Gauge(
    'gl_007_production_uptime_percent',
    'Production uptime percentage',
    ['furnace_id']
)

# ==============================================================================
# ALERT & NOTIFICATION METRICS
# ==============================================================================

alerts_triggered_total = Counter(
    'gl_007_alerts_triggered_total',
    'Total alerts triggered',
    ['alert_type', 'severity', 'furnace_id']
)

notifications_sent_total = Counter(
    'gl_007_notifications_sent_total',
    'Total notifications sent',
    ['channel', 'status']  # channel: email, sms, slack; status: success, failure
)

alert_response_time_seconds = Histogram(
    'gl_007_alert_response_time_seconds',
    'Time from alert to acknowledgment',
    ['alert_type'],
    buckets=(60, 300, 900, 1800, 3600, 7200)
)

# ==============================================================================
# DETERMINISM METRICS
# ==============================================================================

determinism_verification_failures = Counter(
    'gl_007_determinism_verification_failures_total',
    'Total determinism verification failures',
    ['violation_type']  # ai_config, calculation, provenance_hash, cache_key, seed
)

determinism_score = Gauge(
    'gl_007_determinism_score_percent',
    'Determinism score (0-100%, target: 100%)',
    ['component']  # orchestrator, tools, calculators, validators
)

provenance_hash_verifications = Counter(
    'gl_007_provenance_hash_verifications_total',
    'Total provenance hash verifications',
    ['status']  # success, failure
)

calculation_reproducibility_score = Gauge(
    'gl_007_calculation_reproducibility_score',
    'Calculation reproducibility score (0-1)',
    ['calculation_type']
)


# ==============================================================================
# DECORATOR FOR AUTOMATIC METRICS TRACKING
# ==============================================================================

def track_request_metrics(method: str, endpoint: str):
    """
    Decorator to automatically track HTTP request metrics.

    Usage:
        @track_request_metrics('GET', '/api/v1/health')
        async def health_endpoint():
            return {"status": "healthy"}
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


def track_calculation_metrics(calculation_type: str):
    """
    Decorator to track calculation request metrics.

    Usage:
        @track_calculation_metrics('thermal_efficiency')
        async def calculate_efficiency():
            return {"efficiency": 85.2}
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                calculation_duration_seconds.labels(
                    calculation_type=calculation_type
                ).observe(duration)
                calculation_requests_total.labels(
                    calculation_type=calculation_type,
                    status='success'
                ).inc()

                return result
            except Exception as e:
                duration = time.time() - start_time
                calculation_duration_seconds.labels(
                    calculation_type=calculation_type
                ).observe(duration)
                calculation_requests_total.labels(
                    calculation_type=calculation_type,
                    status='failure'
                ).inc()
                raise

        return async_wrapper

    return decorator


# ==============================================================================
# COLLECTOR FOR CUSTOM METRICS
# ==============================================================================

class MetricsCollector:
    """Collects and updates furnace performance metrics."""

    @staticmethod
    def update_furnace_metrics(furnace_id: str, metrics: dict):
        """Update furnace operating metrics."""
        if 'thermal_efficiency' in metrics:
            furnace_thermal_efficiency.labels(
                furnace_id=furnace_id,
                zone=metrics.get('zone', 'overall')
            ).set(metrics['thermal_efficiency'])

        if 'fuel_consumption' in metrics:
            furnace_fuel_consumption.labels(
                furnace_id=furnace_id,
                fuel_type=metrics.get('fuel_type', 'unknown')
            ).set(metrics['fuel_consumption'])

        if 'temperature' in metrics:
            furnace_temperature.labels(
                furnace_id=furnace_id,
                zone=metrics.get('zone', 'main'),
                sensor_id=metrics.get('sensor_id', 'default')
            ).set(metrics['temperature'])

        if 'pressure' in metrics:
            furnace_pressure.labels(
                furnace_id=furnace_id,
                zone=metrics.get('zone', 'main')
            ).set(metrics['pressure'])

        if 'production_rate' in metrics:
            furnace_production_rate.labels(
                furnace_id=furnace_id,
                product_type=metrics.get('product_type', 'default')
            ).set(metrics['production_rate'])

        if 'specific_energy_consumption' in metrics:
            furnace_specific_energy_consumption.labels(
                furnace_id=furnace_id,
                product_type=metrics.get('product_type', 'default')
            ).set(metrics['specific_energy_consumption'])

    @staticmethod
    def update_thermal_metrics(furnace_id: str, thermal_data: dict):
        """Update thermal performance metrics."""
        if 'heat_recovery_efficiency' in thermal_data:
            heat_recovery_efficiency.labels(
                furnace_id=furnace_id,
                recovery_system=thermal_data.get('recovery_system', 'main')
            ).set(thermal_data['heat_recovery_efficiency'])

        if 'heat_loss_rate' in thermal_data:
            for loss_type, rate in thermal_data['heat_loss_rate'].items():
                heat_loss_rate.labels(
                    furnace_id=furnace_id,
                    loss_type=loss_type
                ).set(rate)

        if 'wall_temperatures' in thermal_data:
            for zone, temp_data in thermal_data['wall_temperatures'].items():
                wall_temperature.labels(
                    furnace_id=furnace_id,
                    zone=zone,
                    position=temp_data.get('position', 'center')
                ).set(temp_data['temperature'])

    @staticmethod
    def record_maintenance_alert(furnace_id: str, severity: str, alert_type: str):
        """Record maintenance alert."""
        maintenance_alerts_total.labels(
            furnace_id=furnace_id,
            severity=severity,
            alert_type=alert_type
        ).inc()

    @staticmethod
    def update_prediction_metrics(model_type: str, accuracy: float, duration: float):
        """Update ML model prediction metrics."""
        prediction_accuracy.labels(model_type=model_type).observe(accuracy)
        model_inference_duration_seconds.labels(model_type=model_type).observe(duration)

    @staticmethod
    def update_scada_metrics(scada_system: str, connected: bool, latency: float = None):
        """Update SCADA integration metrics."""
        scada_connection_status.labels(scada_system=scada_system).set(1 if connected else 0)
        if latency is not None:
            scada_data_latency_seconds.labels(scada_system=scada_system).observe(latency)

    @staticmethod
    def record_cache_operation(operation: str, hit: bool, duration: float = None):
        """Record cache hit/miss."""
        cache_key_pattern = operation.split('_')[0] if '_' in operation else operation
        if hit:
            cache_hits_total.labels(cache_key_pattern=cache_key_pattern).inc()
        else:
            cache_misses_total.labels(cache_key_pattern=cache_key_pattern).inc()

        if duration is not None:
            cache_operation_duration_seconds.labels(operation='get').observe(duration)

    @staticmethod
    def update_system_metrics(metrics: dict):
        """Update system resource metrics."""
        if 'uptime_seconds' in metrics:
            system_uptime_seconds.set(metrics['uptime_seconds'])

        if 'memory_rss_bytes' in metrics:
            system_memory_usage_bytes.labels(type='rss').set(metrics['memory_rss_bytes'])

        if 'memory_vms_bytes' in metrics:
            system_memory_usage_bytes.labels(type='vms').set(metrics['memory_vms_bytes'])

        if 'cpu_percent' in metrics:
            system_cpu_usage_percent.set(metrics['cpu_percent'])

        if 'disk_usage' in metrics:
            for mount_point, usage in metrics['disk_usage'].items():
                system_disk_usage_bytes.labels(mount_point=mount_point).set(usage)

    @staticmethod
    def update_business_metrics(furnace_id: str, business_data: dict):
        """Update business impact metrics."""
        if 'energy_cost_savings_usd_hr' in business_data:
            energy_cost_savings_usd_hr.labels(furnace_id=furnace_id).set(
                business_data['energy_cost_savings_usd_hr']
            )

        if 'annual_energy_savings_usd' in business_data:
            annual_energy_savings_usd.labels(furnace_id=furnace_id).set(
                business_data['annual_energy_savings_usd']
            )

        if 'carbon_emissions_reduction_kg_hr' in business_data:
            carbon_emissions_reduction_kg_hr.labels(furnace_id=furnace_id).set(
                business_data['carbon_emissions_reduction_kg_hr']
            )

        if 'annual_carbon_reduction_tons' in business_data:
            annual_carbon_reduction_tons.labels(furnace_id=furnace_id).set(
                business_data['annual_carbon_reduction_tons']
            )

    @staticmethod
    def update_determinism_metrics(component: str, score: float):
        """Update determinism score metrics."""
        determinism_score.labels(component=component).set(score)

    @staticmethod
    def record_determinism_violation(violation_type: str):
        """Record determinism verification failure."""
        determinism_verification_failures.labels(violation_type=violation_type).inc()
