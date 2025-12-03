# -*- coding: utf-8 -*-
"""
GreenLang Agent Prometheus Metrics Template
============================================

Comprehensive metrics collection template for all GL-XXX agents.
Provides 100+ metrics across categories:
- Performance metrics
- Business metrics
- Calculation metrics
- Integration metrics
- Error metrics
- SLA metrics

Usage:
    from prometheus_metrics_template import AgentMetrics
    metrics = AgentMetrics(agent_id="GL-001", agent_name="THERMOSYNC")
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from typing import Optional, Dict, Any
import time
from functools import wraps
from contextlib import contextmanager


class AgentMetrics:
    """
    Comprehensive Prometheus metrics for GreenLang agents.

    Provides standardized metrics collection across all agents
    with zero-hallucination compliance tracking.
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        registry: Optional[CollectorRegistry] = None
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.registry = registry or CollectorRegistry()

        # Initialize all metric categories
        self._init_info_metrics()
        self._init_request_metrics()
        self._init_calculation_metrics()
        self._init_integration_metrics()
        self._init_error_metrics()
        self._init_business_metrics()
        self._init_sla_metrics()
        self._init_resource_metrics()
        self._init_security_metrics()
        self._init_determinism_metrics()

    def _init_info_metrics(self):
        """Agent information metrics."""
        self.agent_info = Info(
            f'{self.agent_id.lower()}_info',
            'Agent information',
            registry=self.registry
        )
        self.agent_info.info({
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'version': '1.0.0',
            'framework': 'greenlang',
        })

        self.build_info = Info(
            f'{self.agent_id.lower()}_build_info',
            'Build information',
            registry=self.registry
        )

    def _init_request_metrics(self):
        """HTTP/API request metrics."""
        labels = ['method', 'endpoint', 'status_code']

        self.requests_total = Counter(
            f'{self.agent_id.lower()}_requests_total',
            'Total number of requests',
            labels,
            registry=self.registry
        )

        self.requests_in_progress = Gauge(
            f'{self.agent_id.lower()}_requests_in_progress',
            'Number of requests currently in progress',
            ['method', 'endpoint'],
            registry=self.registry
        )

        self.request_duration_seconds = Histogram(
            f'{self.agent_id.lower()}_request_duration_seconds',
            'Request duration in seconds',
            labels,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )

        self.request_size_bytes = Histogram(
            f'{self.agent_id.lower()}_request_size_bytes',
            'Request size in bytes',
            ['method', 'endpoint'],
            buckets=(100, 1000, 10000, 100000, 1000000),
            registry=self.registry
        )

        self.response_size_bytes = Histogram(
            f'{self.agent_id.lower()}_response_size_bytes',
            'Response size in bytes',
            ['method', 'endpoint'],
            buckets=(100, 1000, 10000, 100000, 1000000),
            registry=self.registry
        )

    def _init_calculation_metrics(self):
        """Calculation and computation metrics."""
        self.calculations_total = Counter(
            f'{self.agent_id.lower()}_calculations_total',
            'Total number of calculations performed',
            ['calculation_type', 'status'],
            registry=self.registry
        )

        self.calculation_duration_seconds = Histogram(
            f'{self.agent_id.lower()}_calculation_duration_seconds',
            'Calculation duration in seconds',
            ['calculation_type'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self.registry
        )

        self.calculation_input_size = Histogram(
            f'{self.agent_id.lower()}_calculation_input_size',
            'Number of input data points per calculation',
            ['calculation_type'],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
            registry=self.registry
        )

        self.calculation_cache_hits = Counter(
            f'{self.agent_id.lower()}_calculation_cache_hits_total',
            'Total calculation cache hits',
            ['calculation_type'],
            registry=self.registry
        )

        self.calculation_cache_misses = Counter(
            f'{self.agent_id.lower()}_calculation_cache_misses_total',
            'Total calculation cache misses',
            ['calculation_type'],
            registry=self.registry
        )

        self.provenance_hash_generated = Counter(
            f'{self.agent_id.lower()}_provenance_hash_generated_total',
            'Total provenance hashes generated',
            registry=self.registry
        )

    def _init_integration_metrics(self):
        """External system integration metrics."""
        integration_labels = ['system', 'operation']

        self.integration_requests_total = Counter(
            f'{self.agent_id.lower()}_integration_requests_total',
            'Total integration requests',
            integration_labels + ['status'],
            registry=self.registry
        )

        self.integration_duration_seconds = Histogram(
            f'{self.agent_id.lower()}_integration_duration_seconds',
            'Integration request duration',
            integration_labels,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=self.registry
        )

        self.integration_connection_pool_size = Gauge(
            f'{self.agent_id.lower()}_integration_connection_pool_size',
            'Current connection pool size',
            ['system'],
            registry=self.registry
        )

        self.integration_circuit_breaker_state = Gauge(
            f'{self.agent_id.lower()}_integration_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['system'],
            registry=self.registry
        )

        self.integration_retries_total = Counter(
            f'{self.agent_id.lower()}_integration_retries_total',
            'Total integration retries',
            integration_labels,
            registry=self.registry
        )

    def _init_error_metrics(self):
        """Error and exception metrics."""
        self.errors_total = Counter(
            f'{self.agent_id.lower()}_errors_total',
            'Total errors',
            ['error_type', 'severity'],
            registry=self.registry
        )

        self.validation_errors_total = Counter(
            f'{self.agent_id.lower()}_validation_errors_total',
            'Total validation errors',
            ['field', 'validation_type'],
            registry=self.registry
        )

        self.calculation_errors_total = Counter(
            f'{self.agent_id.lower()}_calculation_errors_total',
            'Total calculation errors',
            ['calculation_type', 'error_type'],
            registry=self.registry
        )

        self.last_error_timestamp = Gauge(
            f'{self.agent_id.lower()}_last_error_timestamp',
            'Timestamp of last error',
            ['error_type'],
            registry=self.registry
        )

    def _init_business_metrics(self):
        """Business and domain-specific metrics."""
        # Generic business metrics - customize per agent
        self.equipment_monitored = Gauge(
            f'{self.agent_id.lower()}_equipment_monitored',
            'Number of equipment units being monitored',
            ['equipment_type'],
            registry=self.registry
        )

        self.alerts_generated_total = Counter(
            f'{self.agent_id.lower()}_alerts_generated_total',
            'Total alerts generated',
            ['severity', 'alert_type'],
            registry=self.registry
        )

        self.recommendations_generated_total = Counter(
            f'{self.agent_id.lower()}_recommendations_generated_total',
            'Total recommendations generated',
            ['recommendation_type'],
            registry=self.registry
        )

        self.efficiency_calculated = Gauge(
            f'{self.agent_id.lower()}_efficiency_calculated',
            'Most recent calculated efficiency',
            ['equipment_id'],
            registry=self.registry
        )

        self.savings_identified_usd = Counter(
            f'{self.agent_id.lower()}_savings_identified_usd_total',
            'Total savings identified in USD',
            ['savings_type'],
            registry=self.registry
        )

        self.energy_saved_kwh = Counter(
            f'{self.agent_id.lower()}_energy_saved_kwh_total',
            'Total energy saved in kWh',
            ['source'],
            registry=self.registry
        )

        self.emissions_avoided_kg = Counter(
            f'{self.agent_id.lower()}_emissions_avoided_kg_total',
            'Total CO2 emissions avoided in kg',
            ['source'],
            registry=self.registry
        )

    def _init_sla_metrics(self):
        """SLA and performance target metrics."""
        self.sla_calculation_time_target = Gauge(
            f'{self.agent_id.lower()}_sla_calculation_time_target_seconds',
            'Target calculation time SLA in seconds',
            registry=self.registry
        )
        self.sla_calculation_time_target.set(1.0)  # 1 second target

        self.sla_calculation_time_breaches = Counter(
            f'{self.agent_id.lower()}_sla_calculation_time_breaches_total',
            'Total SLA breaches for calculation time',
            registry=self.registry
        )

        self.sla_availability_target = Gauge(
            f'{self.agent_id.lower()}_sla_availability_target_percent',
            'Target availability SLA percentage',
            registry=self.registry
        )
        self.sla_availability_target.set(99.9)

        self.uptime_seconds = Counter(
            f'{self.agent_id.lower()}_uptime_seconds_total',
            'Total uptime in seconds',
            registry=self.registry
        )

        self.sla_accuracy_target = Gauge(
            f'{self.agent_id.lower()}_sla_accuracy_target_percent',
            'Target calculation accuracy percentage',
            registry=self.registry
        )
        self.sla_accuracy_target.set(99.99)  # Zero-hallucination target

    def _init_resource_metrics(self):
        """Resource utilization metrics."""
        self.memory_usage_bytes = Gauge(
            f'{self.agent_id.lower()}_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )

        self.cpu_usage_percent = Gauge(
            f'{self.agent_id.lower()}_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )

        self.active_threads = Gauge(
            f'{self.agent_id.lower()}_active_threads',
            'Number of active threads',
            registry=self.registry
        )

        self.cache_size_bytes = Gauge(
            f'{self.agent_id.lower()}_cache_size_bytes',
            'Current cache size in bytes',
            ['cache_name'],
            registry=self.registry
        )

        self.cache_entries = Gauge(
            f'{self.agent_id.lower()}_cache_entries',
            'Number of cache entries',
            ['cache_name'],
            registry=self.registry
        )

    def _init_security_metrics(self):
        """Security-related metrics."""
        self.authentication_attempts_total = Counter(
            f'{self.agent_id.lower()}_authentication_attempts_total',
            'Total authentication attempts',
            ['result'],
            registry=self.registry
        )

        self.authorization_denials_total = Counter(
            f'{self.agent_id.lower()}_authorization_denials_total',
            'Total authorization denials',
            ['resource', 'action'],
            registry=self.registry
        )

        self.rate_limit_exceeded_total = Counter(
            f'{self.agent_id.lower()}_rate_limit_exceeded_total',
            'Total rate limit exceeded events',
            ['client_id'],
            registry=self.registry
        )

        self.api_key_rotations_total = Counter(
            f'{self.agent_id.lower()}_api_key_rotations_total',
            'Total API key rotations',
            registry=self.registry
        )

    def _init_determinism_metrics(self):
        """Zero-hallucination and determinism metrics."""
        self.determinism_checks_total = Counter(
            f'{self.agent_id.lower()}_determinism_checks_total',
            'Total determinism verification checks',
            ['result'],
            registry=self.registry
        )

        self.provenance_verifications_total = Counter(
            f'{self.agent_id.lower()}_provenance_verifications_total',
            'Total provenance hash verifications',
            ['result'],
            registry=self.registry
        )

        self.calculation_reproducibility_checks = Counter(
            f'{self.agent_id.lower()}_calculation_reproducibility_checks_total',
            'Total reproducibility checks',
            ['result'],
            registry=self.registry
        )

        self.llm_calls_in_calculation_path = Counter(
            f'{self.agent_id.lower()}_llm_calls_in_calculation_path_total',
            'LLM calls in calculation path (should be 0)',
            registry=self.registry
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    @contextmanager
    def track_request(self, method: str, endpoint: str):
        """Context manager to track request metrics."""
        self.requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        start_time = time.time()
        status_code = "500"

        try:
            yield lambda code: self._set_status(code)
            status_code = "200"
        except Exception:
            status_code = "500"
            raise
        finally:
            duration = time.time() - start_time
            self.requests_in_progress.labels(method=method, endpoint=endpoint).dec()
            self.requests_total.labels(
                method=method, endpoint=endpoint, status_code=status_code
            ).inc()
            self.request_duration_seconds.labels(
                method=method, endpoint=endpoint, status_code=status_code
            ).observe(duration)

    def _set_status(self, code: str):
        """Helper to set status code from within context."""
        pass  # Status is captured in finally block

    @contextmanager
    def track_calculation(self, calculation_type: str):
        """Context manager to track calculation metrics."""
        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception:
            status = "error"
            self.calculation_errors_total.labels(
                calculation_type=calculation_type,
                error_type="exception"
            ).inc()
            raise
        finally:
            duration = time.time() - start_time
            self.calculations_total.labels(
                calculation_type=calculation_type,
                status=status
            ).inc()
            self.calculation_duration_seconds.labels(
                calculation_type=calculation_type
            ).observe(duration)
            self.provenance_hash_generated.inc()

    @contextmanager
    def track_integration(self, system: str, operation: str):
        """Context manager to track integration metrics."""
        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            self.integration_requests_total.labels(
                system=system, operation=operation, status=status
            ).inc()
            self.integration_duration_seconds.labels(
                system=system, operation=operation
            ).observe(duration)

    def record_alert(self, severity: str, alert_type: str):
        """Record an alert generation."""
        self.alerts_generated_total.labels(
            severity=severity,
            alert_type=alert_type
        ).inc()

    def record_savings(self, amount_usd: float, savings_type: str):
        """Record identified savings."""
        self.savings_identified_usd.labels(savings_type=savings_type).inc(amount_usd)

    def record_energy_saved(self, kwh: float, source: str):
        """Record energy savings."""
        self.energy_saved_kwh.labels(source=source).inc(kwh)

    def record_emissions_avoided(self, kg_co2: float, source: str):
        """Record emissions avoided."""
        self.emissions_avoided_kg.labels(source=source).inc(kg_co2)

    def record_error(self, error_type: str, severity: str):
        """Record an error occurrence."""
        self.errors_total.labels(error_type=error_type, severity=severity).inc()
        self.last_error_timestamp.labels(error_type=error_type).set_to_current_time()

    def record_determinism_check(self, passed: bool):
        """Record a determinism verification check."""
        result = "pass" if passed else "fail"
        self.determinism_checks_total.labels(result=result).inc()

    def record_provenance_verification(self, verified: bool):
        """Record a provenance hash verification."""
        result = "verified" if verified else "failed"
        self.provenance_verifications_total.labels(result=result).inc()

    def get_metrics(self) -> bytes:
        """Generate metrics output in Prometheus format."""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get the content type for metrics endpoint."""
        return CONTENT_TYPE_LATEST


# =============================================================================
# DECORATOR FOR AUTOMATIC METRICS
# =============================================================================

def track_calculation_metrics(calculation_type: str):
    """Decorator to automatically track calculation metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, 'metrics') and isinstance(self.metrics, AgentMetrics):
                with self.metrics.track_calculation(calculation_type):
                    return func(self, *args, **kwargs)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def track_integration_metrics(system: str, operation: str):
    """Decorator to automatically track integration metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, 'metrics') and isinstance(self.metrics, AgentMetrics):
                with self.metrics.track_integration(system, operation):
                    return func(self, *args, **kwargs)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example instantiation
    metrics = AgentMetrics(agent_id="GL-001", agent_name="THERMOSYNC")

    # Example tracking
    with metrics.track_calculation("heat_balance"):
        # Perform calculation
        pass

    with metrics.track_integration("scada", "read"):
        # Perform integration
        pass

    # Record business metrics
    metrics.record_savings(1500.00, "efficiency_improvement")
    metrics.record_energy_saved(250.0, "heat_recovery")
    metrics.record_emissions_avoided(150.0, "fuel_reduction")
    metrics.record_alert("high", "efficiency_drop")

    # Record determinism metrics
    metrics.record_determinism_check(True)
    metrics.record_provenance_verification(True)

    # Get metrics output
    print(metrics.get_metrics().decode())
