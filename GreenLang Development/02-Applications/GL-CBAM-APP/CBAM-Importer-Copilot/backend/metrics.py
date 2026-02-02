# -*- coding: utf-8 -*-
"""
CBAM Importer Copilot - Prometheus Metrics

Production-grade metrics collection for monitoring application performance,
business metrics, and system health.

Metrics Categories:
1. Request Metrics - Request rate, latency, errors
2. Business Metrics - Shipments processed, emissions calculated
3. System Metrics - Memory usage, processing time
4. Agent Metrics - Per-agent performance tracking

Prometheus Integration:
- Counter: Monotonically increasing values (requests, errors)
- Gauge: Values that can go up/down (memory, active requests)
- Histogram: Distribution of values (latency, processing time)
- Summary: Similar to histogram with quantiles

Version: 1.0.0
Author: GreenLang CBAM Team (Team A3: Monitoring & Observability)
"""

import time
import psutil
import platform
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, Any, Optional
from pathlib import Path

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        push_to_gateway, delete_from_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not installed. Metrics will not be collected.")


# ============================================================================
# METRICS REGISTRY
# ============================================================================

class CBAMMetrics:
    """
    Centralized metrics collection for CBAM Importer Copilot.

    Provides Prometheus-compatible metrics for:
    - Pipeline execution
    - Agent performance
    - Data validation
    - Emissions calculations
    - System resources
    """

    def __init__(self, registry: Optional['CollectorRegistry'] = None, namespace: str = "cbam"):
        """
        Initialize metrics collector.

        Args:
            registry: Prometheus registry (None for default)
            namespace: Metric namespace prefix
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client is required for metrics collection. "
                "Install it with: pip install prometheus-client"
            )

        self.registry = registry or CollectorRegistry()
        self.namespace = namespace

        # Initialize all metrics
        self._init_pipeline_metrics()
        self._init_agent_metrics()
        self._init_validation_metrics()
        self._init_emissions_metrics()
        self._init_system_metrics()
        self._init_error_metrics()
        self._init_info_metrics()

    # ========================================================================
    # PIPELINE METRICS
    # ========================================================================

    def _init_pipeline_metrics(self):
        """Initialize pipeline-level metrics."""

        # Pipeline executions counter
        self.pipeline_executions_total = Counter(
            f'{self.namespace}_pipeline_executions_total',
            'Total number of pipeline executions',
            ['status'],  # success, failed
            registry=self.registry
        )

        # Pipeline duration histogram
        self.pipeline_duration_seconds = Histogram(
            f'{self.namespace}_pipeline_duration_seconds',
            'Pipeline execution duration in seconds',
            ['stage'],  # intake, calculate, package, total
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],  # 1s to 10min
            registry=self.registry
        )

        # Active pipeline executions
        self.pipeline_active = Gauge(
            f'{self.namespace}_pipeline_active',
            'Number of currently active pipeline executions',
            registry=self.registry
        )

        # Records processed
        self.records_processed_total = Counter(
            f'{self.namespace}_records_processed_total',
            'Total number of records processed',
            ['stage', 'status'],  # stage: intake/calculate/package, status: success/failed
            registry=self.registry
        )

        # Processing rate
        self.records_per_second = Gauge(
            f'{self.namespace}_records_per_second',
            'Current record processing rate',
            ['stage'],
            registry=self.registry
        )

    # ========================================================================
    # AGENT METRICS
    # ========================================================================

    def _init_agent_metrics(self):
        """Initialize agent-specific metrics."""

        # Agent execution counter
        self.agent_executions_total = Counter(
            f'{self.namespace}_agent_executions_total',
            'Total number of agent executions',
            ['agent', 'status'],  # agent: intake/calculator/packager, status: success/failed
            registry=self.registry
        )

        # Agent execution duration
        self.agent_duration_seconds = Histogram(
            f'{self.namespace}_agent_duration_seconds',
            'Agent execution duration in seconds',
            ['agent'],
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60],
            registry=self.registry
        )

        # Agent performance (ms per record)
        self.agent_ms_per_record = Summary(
            f'{self.namespace}_agent_ms_per_record',
            'Milliseconds per record processed by agent',
            ['agent'],
            registry=self.registry
        )

    # ========================================================================
    # VALIDATION METRICS
    # ========================================================================

    def _init_validation_metrics(self):
        """Initialize validation metrics."""

        # Validation results
        self.validation_results_total = Counter(
            f'{self.namespace}_validation_results_total',
            'Total validation results',
            ['result'],  # valid, invalid, warning
            registry=self.registry
        )

        # Validation errors by type
        self.validation_errors_by_type = Counter(
            f'{self.namespace}_validation_errors_by_type',
            'Validation errors by type',
            ['error_type'],
            registry=self.registry
        )

        # Validation duration
        self.validation_duration_seconds = Histogram(
            f'{self.namespace}_validation_duration_seconds',
            'Validation duration in seconds',
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 5],
            registry=self.registry
        )

    # ========================================================================
    # EMISSIONS METRICS
    # ========================================================================

    def _init_emissions_metrics(self):
        """Initialize emissions calculation metrics."""

        # Total emissions calculated
        self.emissions_calculated_tco2 = Counter(
            f'{self.namespace}_emissions_calculated_tco2',
            'Total emissions calculated in tCO2',
            registry=self.registry
        )

        # Current emissions calculation rate
        self.emissions_calculation_rate = Gauge(
            f'{self.namespace}_emissions_calculation_rate',
            'Current emissions calculation rate (tCO2/second)',
            registry=self.registry
        )

        # Calculation method distribution
        self.calculation_method_total = Counter(
            f'{self.namespace}_calculation_method_total',
            'Count of calculations by method',
            ['method'],  # default, actual, hybrid
            registry=self.registry
        )

        # Emissions by CN code
        self.emissions_by_cn_code = Counter(
            f'{self.namespace}_emissions_by_cn_code',
            'Total emissions by CN code',
            ['cn_code'],
            registry=self.registry
        )

    # ========================================================================
    # SYSTEM METRICS
    # ========================================================================

    def _init_system_metrics(self):
        """Initialize system resource metrics."""

        # Memory usage
        self.memory_usage_bytes = Gauge(
            f'{self.namespace}_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )

        # CPU usage
        self.cpu_usage_percent = Gauge(
            f'{self.namespace}_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )

        # Disk usage
        self.disk_usage_bytes = Gauge(
            f'{self.namespace}_disk_usage_bytes',
            'Current disk usage in bytes',
            ['path'],
            registry=self.registry
        )

        # File descriptor usage (Unix/Linux only)
        if platform.system() != 'Windows':
            self.open_file_descriptors = Gauge(
                f'{self.namespace}_open_file_descriptors',
                'Number of open file descriptors',
                registry=self.registry
            )

    # ========================================================================
    # ERROR METRICS
    # ========================================================================

    def _init_error_metrics(self):
        """Initialize error tracking metrics."""

        # Errors by type
        self.errors_total = Counter(
            f'{self.namespace}_errors_total',
            'Total number of errors',
            ['error_type', 'severity'],  # severity: warning, error, critical
            registry=self.registry
        )

        # Exceptions by type
        self.exceptions_total = Counter(
            f'{self.namespace}_exceptions_total',
            'Total number of exceptions',
            ['exception_type'],
            registry=self.registry
        )

    # ========================================================================
    # INFO METRICS
    # ========================================================================

    def _init_info_metrics(self):
        """Initialize informational metrics."""

        # Application version info
        self.application_info = Info(
            f'{self.namespace}_application',
            'Application version information',
            registry=self.registry
        )

        # Set application info
        self.application_info.info({
            'version': '1.0.0',
            'service': 'cbam-importer-copilot',
            'python_version': platform.python_version(),
            'platform': platform.system()
        })

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def update_system_metrics(self):
        """Update system resource metrics."""
        # Memory
        process = psutil.Process()
        self.memory_usage_bytes.set(process.memory_info().rss)

        # CPU
        self.cpu_usage_percent.set(process.cpu_percent())

        # Open file descriptors (Unix/Linux only)
        if platform.system() != 'Windows':
            try:
                self.open_file_descriptors.set(process.num_fds())
            except:
                pass

    def record_pipeline_execution(self, status: str, duration_seconds: float, stage: str = "total"):
        """
        Record a pipeline execution.

        Args:
            status: Execution status (success, failed)
            duration_seconds: Execution duration
            stage: Pipeline stage
        """
        self.pipeline_executions_total.labels(status=status).inc()
        self.pipeline_duration_seconds.labels(stage=stage).observe(duration_seconds)

    def record_agent_execution(
        self,
        agent: str,
        status: str,
        duration_seconds: float,
        records_processed: int = 0
    ):
        """
        Record an agent execution.

        Args:
            agent: Agent name (intake, calculator, packager)
            status: Execution status
            duration_seconds: Execution duration
            records_processed: Number of records processed
        """
        self.agent_executions_total.labels(agent=agent, status=status).inc()
        self.agent_duration_seconds.labels(agent=agent).observe(duration_seconds)

        if records_processed > 0:
            ms_per_record = (duration_seconds * 1000) / records_processed
            self.agent_ms_per_record.labels(agent=agent).observe(ms_per_record)

    def record_validation_result(self, result: str, error_type: Optional[str] = None):
        """
        Record a validation result.

        Args:
            result: Validation result (valid, invalid, warning)
            error_type: Type of validation error (if applicable)
        """
        self.validation_results_total.labels(result=result).inc()

        if error_type:
            self.validation_errors_by_type.labels(error_type=error_type).inc()

    def record_emissions_calculation(
        self,
        tco2: float,
        method: str,
        cn_code: Optional[str] = None
    ):
        """
        Record emissions calculation.

        Args:
            tco2: Emissions in tCO2
            method: Calculation method (default, actual, hybrid)
            cn_code: CN code (optional)
        """
        self.emissions_calculated_tco2.inc(tco2)
        self.calculation_method_total.labels(method=method).inc()

        if cn_code:
            self.emissions_by_cn_code.labels(cn_code=cn_code).inc(tco2)

    def record_error(self, error_type: str, severity: str = "error"):
        """
        Record an error.

        Args:
            error_type: Type of error
            severity: Error severity (warning, error, critical)
        """
        self.errors_total.labels(error_type=error_type, severity=severity).inc()

    def record_exception(self, exception: Exception):
        """
        Record an exception.

        Args:
            exception: Exception instance
        """
        exception_type = type(exception).__name__
        self.exceptions_total.labels(exception_type=exception_type).inc()


# ============================================================================
# DECORATORS FOR AUTOMATIC METRICS
# ============================================================================

def track_execution_time(metrics: CBAMMetrics, operation: str):
    """
    Decorator to automatically track execution time.

    Usage:
        @track_execution_time(metrics, "pipeline")
        def run_pipeline():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Record success
                metrics.record_pipeline_execution("success", duration, operation)

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Record failure
                metrics.record_pipeline_execution("failed", duration, operation)
                metrics.record_exception(e)

                raise

        return wrapper
    return decorator


def track_agent_execution(metrics: CBAMMetrics, agent_name: str):
    """
    Decorator to automatically track agent execution.

    Usage:
        @track_agent_execution(metrics, "intake")
        def process_shipments():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Try to extract records count from result
                records = 0
                if isinstance(result, dict):
                    if 'metadata' in result:
                        records = result['metadata'].get('total_records', 0)
                    elif 'shipments' in result:
                        records = len(result['shipments'])

                metrics.record_agent_execution(agent_name, "success", duration, records)

                return result

            except Exception as e:
                duration = time.time() - start_time
                metrics.record_agent_execution(agent_name, "failed", duration)
                metrics.record_exception(e)

                raise

        return wrapper
    return decorator


# ============================================================================
# METRICS EXPORTER
# ============================================================================

class MetricsExporter:
    """Export metrics for Prometheus scraping or push gateway."""

    def __init__(self, metrics: CBAMMetrics):
        self.metrics = metrics

    def export_text(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Metrics in Prometheus exposition format
        """
        return generate_latest(self.metrics.registry).decode('utf-8')

    def get_content_type(self) -> str:
        """Get content type for Prometheus metrics."""
        return CONTENT_TYPE_LATEST

    def push_to_gateway(
        self,
        gateway_url: str,
        job: str = "cbam-importer-copilot"
    ):
        """
        Push metrics to Prometheus push gateway.

        Args:
            gateway_url: URL of push gateway
            job: Job name for grouping metrics
        """
        push_to_gateway(gateway_url, job=job, registry=self.metrics.registry)

    def delete_from_gateway(
        self,
        gateway_url: str,
        job: str = "cbam-importer-copilot"
    ):
        """
        Delete metrics from push gateway.

        Args:
            gateway_url: URL of push gateway
            job: Job name
        """
        delete_from_gateway(gateway_url, job=job)


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def create_metrics_endpoint(metrics: CBAMMetrics):
    """
    Create FastAPI endpoint for Prometheus metrics.

    Example:
        from fastapi import FastAPI
        from backend.metrics import CBAMMetrics, create_metrics_endpoint

        app = FastAPI()
        metrics = CBAMMetrics()
        metrics_route = create_metrics_endpoint(metrics)
        app.add_api_route(**metrics_route)
    """
    try:
        from fastapi import Response
    except ImportError:
        print("Warning: FastAPI not installed. Metrics endpoint will not be available.")
        return None

    exporter = MetricsExporter(metrics)

    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        # Update system metrics before export
        metrics.update_system_metrics()

        return Response(
            content=exporter.export_text(),
            media_type=exporter.get_content_type()
        )

    return {
        "path": "/metrics",
        "endpoint": metrics_endpoint,
        "methods": ["GET"],
        "tags": ["metrics"]
    }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Global metrics instance
_global_metrics: Optional[CBAMMetrics] = None


def get_metrics() -> CBAMMetrics:
    """Get or create global metrics instance."""
    global _global_metrics

    if _global_metrics is None:
        if PROMETHEUS_AVAILABLE:
            _global_metrics = CBAMMetrics()
        else:
            raise ImportError("Prometheus client not available")

    return _global_metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create metrics
    metrics = CBAMMetrics()

    # Simulate pipeline execution
    print("Simulating metrics collection...")

    # Record pipeline execution
    metrics.record_pipeline_execution("success", 45.5, "total")
    metrics.record_pipeline_execution("success", 10.2, "intake")
    metrics.record_pipeline_execution("success", 25.1, "calculate")
    metrics.record_pipeline_execution("success", 10.2, "package")

    # Record agent executions
    metrics.record_agent_execution("intake", "success", 10.2, 1000)
    metrics.record_agent_execution("calculator", "success", 25.1, 1000)
    metrics.record_agent_execution("packager", "success", 10.2, 1000)

    # Record validation results
    metrics.record_validation_result("valid")
    metrics.record_validation_result("invalid", "missing_cn_code")
    metrics.record_validation_result("warning")

    # Record emissions calculations
    metrics.record_emissions_calculation(123.45, "default", "7208.10")
    metrics.record_emissions_calculation(67.89, "actual", "7208.10")

    # Update system metrics
    metrics.update_system_metrics()

    # Export metrics
    exporter = MetricsExporter(metrics)
    print("\n" + "="*80)
    print("PROMETHEUS METRICS EXPORT")
    print("="*80)
    print(exporter.export_text())
