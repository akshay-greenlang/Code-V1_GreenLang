# -*- coding: utf-8 -*-
"""
CSRD Reporting Platform - Prometheus Metrics
============================================

Production-grade Prometheus metrics with ESRS/CSRD-specific monitoring.

Features:
- HTTP request metrics
- Agent performance metrics
- ESRS data coverage metrics
- Validation metrics
- Report generation metrics
- LLM API cost tracking
- Security metrics

Author: GreenLang Operations Team (Team B3)
Date: 2025-11-08
"""

from typing import Dict, Any, Optional
import time
from datetime import datetime
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, REGISTRY, CollectorRegistry
)
from fastapi import APIRouter, Response
from contextvars import ContextVar


# ============================================================================
# CONTEXT VARIABLES
# ============================================================================

current_request_ctx: ContextVar[Optional[Dict[str, Any]]] = ContextVar('current_request', default=None)


# ============================================================================
# PROMETHEUS METRICS - HTTP/API
# ============================================================================

# HTTP request counters
http_requests_total = Counter(
    'csrd_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'csrd_http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

http_request_size_bytes = Summary(
    'csrd_http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

http_response_size_bytes = Summary(
    'csrd_http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)


# ============================================================================
# PROMETHEUS METRICS - ESRS DATA
# ============================================================================

# ESRS data coverage metrics
esrs_data_point_coverage = Gauge(
    'csrd_esrs_data_point_coverage_ratio',
    'ESRS data point coverage (0-1)',
    ['esrs_standard', 'company_id']
)

esrs_required_data_points = Gauge(
    'csrd_esrs_required_data_points_total',
    'Total required data points for ESRS standard',
    ['esrs_standard']
)

esrs_available_data_points = Gauge(
    'csrd_esrs_available_data_points_total',
    'Available data points for ESRS standard',
    ['esrs_standard', 'company_id']
)

esrs_missing_data_points = Gauge(
    'csrd_esrs_missing_data_points_total',
    'Missing data points for ESRS standard',
    ['esrs_standard', 'company_id']
)

# ESRS data quality metrics
esrs_data_quality_score = Gauge(
    'csrd_esrs_data_quality_score',
    'Overall data quality score (0-100)',
    ['esrs_standard', 'company_id']
)

esrs_data_completeness_ratio = Gauge(
    'csrd_esrs_data_completeness_ratio',
    'Data completeness ratio (0-1)',
    ['esrs_standard', 'company_id']
)

esrs_data_accuracy_ratio = Gauge(
    'csrd_esrs_data_accuracy_ratio',
    'Data accuracy ratio (0-1)',
    ['esrs_standard', 'company_id']
)

esrs_data_timeliness_ratio = Gauge(
    'csrd_esrs_data_timeliness_ratio',
    'Data timeliness ratio (0-1)',
    ['esrs_standard', 'company_id']
)


# ============================================================================
# PROMETHEUS METRICS - VALIDATION
# ============================================================================

# Validation metrics
validation_checks_total = Counter(
    'csrd_validation_checks_total',
    'Total validation checks performed',
    ['validation_type', 'esrs_standard', 'status']
)

validation_errors_total = Counter(
    'csrd_validation_errors_total',
    'Total validation errors',
    ['error_type', 'esrs_standard', 'severity']
)

validation_duration_seconds = Histogram(
    'csrd_validation_duration_seconds',
    'Validation duration in seconds',
    ['validation_type', 'esrs_standard'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
)

validation_rules_applied = Gauge(
    'csrd_validation_rules_applied_total',
    'Total validation rules applied',
    ['esrs_standard']
)


# ============================================================================
# PROMETHEUS METRICS - AGENTS
# ============================================================================

# Agent execution metrics
agent_execution_total = Counter(
    'csrd_agent_execution_total',
    'Total agent executions',
    ['agent_name', 'operation', 'status']
)

agent_execution_duration_seconds = Histogram(
    'csrd_agent_execution_duration_seconds',
    'Agent execution duration in seconds',
    ['agent_name', 'operation'],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0)
)

agent_execution_failures_total = Counter(
    'csrd_agent_execution_failures_total',
    'Total agent execution failures',
    ['agent_name', 'error_type']
)

agent_memory_usage_bytes = Gauge(
    'csrd_agent_memory_usage_bytes',
    'Agent memory usage in bytes',
    ['agent_name']
)

agent_active_tasks = Gauge(
    'csrd_agent_active_tasks',
    'Number of active agent tasks',
    ['agent_name']
)


# ============================================================================
# PROMETHEUS METRICS - REPORTS
# ============================================================================

# Report generation metrics
reports_generated_total = Counter(
    'csrd_reports_generated_total',
    'Total reports generated',
    ['report_type', 'format', 'status']
)

report_generation_duration_seconds = Histogram(
    'csrd_report_generation_duration_seconds',
    'Report generation duration in seconds',
    ['report_type', 'format'],
    buckets=(5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

report_generation_failures_total = Counter(
    'csrd_report_generation_failures_total',
    'Total report generation failures',
    ['report_type', 'error_type']
)

report_size_bytes = Histogram(
    'csrd_report_size_bytes',
    'Report file size in bytes',
    ['report_type', 'format'],
    buckets=(1024, 10240, 102400, 1048576, 10485760, 104857600)
)

# XBRL-specific metrics
xbrl_generation_duration_seconds = Histogram(
    'csrd_xbrl_generation_duration_seconds',
    'XBRL generation duration in seconds',
    ['format'],
    buckets=(5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
)

xbrl_validation_errors_total = Counter(
    'csrd_xbrl_validation_errors_total',
    'Total XBRL validation errors',
    ['error_type']
)

xbrl_taxonomies_loaded = Gauge(
    'csrd_xbrl_taxonomies_loaded',
    'Number of XBRL taxonomies loaded'
)


# ============================================================================
# PROMETHEUS METRICS - LLM API
# ============================================================================

# LLM API usage metrics
llm_api_calls_total = Counter(
    'csrd_llm_api_calls_total',
    'Total LLM API calls',
    ['provider', 'model', 'status']
)

llm_api_duration_seconds = Histogram(
    'csrd_llm_api_duration_seconds',
    'LLM API call duration in seconds',
    ['provider', 'model'],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

llm_api_tokens_used_total = Counter(
    'csrd_llm_api_tokens_used_total',
    'Total LLM API tokens used',
    ['provider', 'model', 'type']  # type: input or output
)

llm_api_cost_usd_total = Counter(
    'csrd_llm_api_cost_usd_total',
    'Total LLM API cost in USD',
    ['provider', 'model']
)

llm_api_errors_total = Counter(
    'csrd_llm_api_errors_total',
    'Total LLM API errors',
    ['provider', 'model', 'error_type']
)


# ============================================================================
# PROMETHEUS METRICS - DATA PROCESSING
# ============================================================================

# Data processing metrics
data_records_processed_total = Counter(
    'csrd_data_records_processed_total',
    'Total data records processed',
    ['source', 'status']
)

data_processing_duration_seconds = Histogram(
    'csrd_data_processing_duration_seconds',
    'Data processing duration in seconds',
    ['source', 'operation'],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0)
)

data_import_batch_size = Histogram(
    'csrd_data_import_batch_size',
    'Number of records in import batch',
    ['source'],
    buckets=(10, 50, 100, 500, 1000, 5000, 10000)
)


# ============================================================================
# PROMETHEUS METRICS - SECURITY
# ============================================================================

# Security metrics
authentication_attempts_total = Counter(
    'csrd_authentication_attempts_total',
    'Total authentication attempts',
    ['method', 'status']
)

authentication_failures_total = Counter(
    'csrd_authentication_failures_total',
    'Total authentication failures',
    ['method', 'reason']
)

encryption_operations_total = Counter(
    'csrd_encryption_operations_total',
    'Total encryption operations',
    ['operation', 'status']  # operation: encrypt or decrypt
)

encryption_failures_total = Counter(
    'csrd_encryption_failures_total',
    'Total encryption failures',
    ['operation', 'error_type']
)

api_key_usage_total = Counter(
    'csrd_api_key_usage_total',
    'Total API key usage',
    ['key_id', 'endpoint']
)


# ============================================================================
# PROMETHEUS METRICS - COMPLIANCE
# ============================================================================

# Compliance deadline metrics
compliance_deadline_days_remaining = Gauge(
    'csrd_compliance_deadline_days_remaining',
    'Days remaining until compliance deadline',
    ['deadline_type', 'company_id']
)

compliance_tasks_total = Gauge(
    'csrd_compliance_tasks_total',
    'Total compliance tasks',
    ['status', 'company_id']
)

compliance_score = Gauge(
    'csrd_compliance_score',
    'Overall compliance score (0-100)',
    ['company_id']
)


# ============================================================================
# PROMETHEUS METRICS - SYSTEM
# ============================================================================

# System health metrics
health_check_status = Gauge(
    'csrd_health_check_status',
    'Health check status (1=healthy, 0=unhealthy)',
    ['service', 'check_type']
)

application_info = Info(
    'csrd_application',
    'Application information'
)

# Set application info
application_info.info({
    'version': '1.0.0',
    'service': 'csrd-reporting-platform',
    'environment': 'production'
})


# ============================================================================
# METRICS CONTEXT MANAGERS
# ============================================================================

class MetricsTimer:
    """
    Context manager for timing operations and recording to Prometheus histogram.

    Usage:
        with MetricsTimer(agent_execution_duration_seconds, labels={'agent_name': 'intake', 'operation': 'process'}):
            # ... do work ...
    """

    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]] = None):
        """
        Initialize metrics timer.

        Args:
            histogram: Prometheus Histogram to record to
            labels: Labels for the metric
        """
        self.histogram = histogram
        self.labels = labels or {}
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record duration."""
        duration = time.time() - self.start_time
        self.histogram.labels(**self.labels).observe(duration)


class MetricsCounter:
    """
    Context manager for counting operations and handling errors.

    Usage:
        with MetricsCounter(reports_generated_total, labels={'report_type': 'annual', 'format': 'pdf'}):
            # ... generate report ...
    """

    def __init__(
        self,
        counter: Counter,
        labels: Optional[Dict[str, str]] = None,
        error_counter: Optional[Counter] = None
    ):
        """
        Initialize metrics counter.

        Args:
            counter: Prometheus Counter to increment
            labels: Labels for the metric
            error_counter: Optional error counter for failures
        """
        self.counter = counter
        self.labels = labels or {}
        self.error_counter = error_counter
        self.success = True

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and record metric."""
        if exc_type is not None:
            self.success = False
            # Record error if error counter provided
            if self.error_counter:
                error_labels = {**self.labels, 'error_type': exc_type.__name__}
                self.error_counter.labels(**error_labels).inc()

        # Record to main counter
        status_labels = {**self.labels, 'status': 'success' if self.success else 'error'}
        self.counter.labels(**status_labels).inc()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def record_http_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float,
    request_size: Optional[int] = None,
    response_size: Optional[int] = None
):
    """
    Record HTTP request metrics.

    Args:
        method: HTTP method
        endpoint: API endpoint
        status_code: Response status code
        duration: Request duration in seconds
        request_size: Request size in bytes
        response_size: Response size in bytes
    """
    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code)
    ).inc()

    http_request_duration_seconds.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)

    if request_size is not None:
        http_request_size_bytes.labels(
            method=method,
            endpoint=endpoint
        ).observe(request_size)

    if response_size is not None:
        http_response_size_bytes.labels(
            method=method,
            endpoint=endpoint
        ).observe(response_size)


def record_esrs_coverage(
    esrs_standard: str,
    company_id: str,
    required_points: int,
    available_points: int
):
    """
    Record ESRS data coverage metrics.

    Args:
        esrs_standard: ESRS standard (e.g., "E1", "S1")
        company_id: Company ID
        required_points: Required data points
        available_points: Available data points
    """
    coverage_ratio = available_points / required_points if required_points > 0 else 0
    missing_points = max(0, required_points - available_points)

    esrs_data_point_coverage.labels(
        esrs_standard=esrs_standard,
        company_id=company_id
    ).set(coverage_ratio)

    esrs_required_data_points.labels(
        esrs_standard=esrs_standard
    ).set(required_points)

    esrs_available_data_points.labels(
        esrs_standard=esrs_standard,
        company_id=company_id
    ).set(available_points)

    esrs_missing_data_points.labels(
        esrs_standard=esrs_standard,
        company_id=company_id
    ).set(missing_points)


def record_llm_usage(
    provider: str,
    model: str,
    duration: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    success: bool = True,
    error_type: Optional[str] = None
):
    """
    Record LLM API usage metrics.

    Args:
        provider: LLM provider (e.g., "openai", "anthropic")
        model: Model name
        duration: API call duration in seconds
        input_tokens: Input tokens used
        output_tokens: Output tokens used
        cost_usd: Cost in USD
        success: Whether call succeeded
        error_type: Error type if failed
    """
    status = 'success' if success else 'error'

    llm_api_calls_total.labels(
        provider=provider,
        model=model,
        status=status
    ).inc()

    if success:
        llm_api_duration_seconds.labels(
            provider=provider,
            model=model
        ).observe(duration)

        llm_api_tokens_used_total.labels(
            provider=provider,
            model=model,
            type='input'
        ).inc(input_tokens)

        llm_api_tokens_used_total.labels(
            provider=provider,
            model=model,
            type='output'
        ).inc(output_tokens)

        llm_api_cost_usd_total.labels(
            provider=provider,
            model=model
        ).inc(cost_usd)
    else:
        llm_api_errors_total.labels(
            provider=provider,
            model=model,
            error_type=error_type or 'unknown'
        ).inc()


# ============================================================================
# METRICS ENDPOINT
# ============================================================================

metrics_router = APIRouter(prefix="/metrics", tags=["metrics"])


@metrics_router.get("")
@metrics_router.get("/")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus exposition format for scraping.
    """
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain; version=0.0.4"
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example: Record HTTP request
    record_http_request(
        method="GET",
        endpoint="/api/v1/companies",
        status_code=200,
        duration=0.125,
        request_size=256,
        response_size=4096
    )

    # Example: Record ESRS coverage
    record_esrs_coverage(
        esrs_standard="E1",
        company_id="comp-123",
        required_points=150,
        available_points=142
    )

    # Example: Using timer context manager
    with MetricsTimer(
        agent_execution_duration_seconds,
        labels={'agent_name': 'intake', 'operation': 'process'}
    ):
        time.sleep(0.1)  # Simulate work

    # Example: Using counter context manager
    with MetricsCounter(
        reports_generated_total,
        labels={'report_type': 'annual', 'format': 'pdf'},
        error_counter=report_generation_failures_total
    ):
        # Generate report
        pass

    # Example: Record LLM usage
    record_llm_usage(
        provider="openai",
        model="gpt-4",
        duration=2.5,
        input_tokens=500,
        output_tokens=200,
        cost_usd=0.015,
        success=True
    )

    print("Metrics recorded successfully!")
    print("\nSample metrics output:")
    print(generate_latest(REGISTRY).decode('utf-8'))
