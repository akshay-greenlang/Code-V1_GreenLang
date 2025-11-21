# -*- coding: utf-8 -*-
"""
GL-VCCI Scope 3 Platform - Prometheus Metrics

Production-grade metrics collection for monitoring:
- Carbon calculation performance (15 categories, 3 tiers)
- Entity resolution success rates
- Supplier engagement metrics
- LLM API usage and costs
- Data quality scores
- System health

Prometheus Integration:
- Counter: Monotonically increasing values (calculations, emissions)
- Gauge: Values that can go up/down (active suppliers, queue depth)
- Histogram: Distribution of values (latency, calculation duration)
- Summary: Similar to histogram with quantiles

Version: 1.0.0
Author: GreenLang VCCI Team (Monitoring & Observability)
Date: 2025-11-08
"""

import logging
import time
import psutil
import platform
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, Any, Optional, List
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        push_to_gateway, delete_from_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics will not be collected.")


# ============================================================================
# ENUMS FOR METRIC LABELS
# ============================================================================

class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 Categories (15 total)"""
    CATEGORY_1 = "1"   # Purchased goods and services
    CATEGORY_2 = "2"   # Capital goods
    CATEGORY_3 = "3"   # Fuel and energy related
    CATEGORY_4 = "4"   # Upstream transportation
    CATEGORY_5 = "5"   # Waste generated in operations
    CATEGORY_6 = "6"   # Business travel
    CATEGORY_7 = "7"   # Employee commuting
    CATEGORY_8 = "8"   # Upstream leased assets
    CATEGORY_9 = "9"   # Downstream transportation
    CATEGORY_10 = "10" # Processing of sold products
    CATEGORY_11 = "11" # Use of sold products
    CATEGORY_12 = "12" # End-of-life treatment
    CATEGORY_13 = "13" # Downstream leased assets
    CATEGORY_14 = "14" # Franchises
    CATEGORY_15 = "15" # Investments


class DataTier(str, Enum):
    """Data quality tiers for carbon calculations"""
    TIER_1 = "tier1"  # Supplier-provided primary data
    TIER_2 = "tier2"  # Industry average data
    TIER_3 = "tier3"  # Proxy/default data


# ============================================================================
# METRICS REGISTRY
# ============================================================================

class VCCIMetrics:
    """
    Centralized metrics collection for VCCI Scope 3 Platform.

    Provides Prometheus-compatible metrics for:
    - Carbon calculations (15 categories, 3 tiers)
    - Entity resolution
    - Supplier engagement
    - LLM API usage
    - Data quality
    - System resources
    """

    def __init__(self, registry: Optional['CollectorRegistry'] = None, namespace: str = "vcci"):
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
        self._init_carbon_metrics()
        self._init_entity_metrics()
        self._init_supplier_metrics()
        self._init_llm_metrics()
        self._init_data_quality_metrics()
        self._init_system_metrics()
        self._init_agent_metrics()
        self._init_info_metrics()

    # ========================================================================
    # CARBON CALCULATION METRICS
    # ========================================================================

    def _init_carbon_metrics(self):
        """Initialize carbon-specific metrics."""

        # Total emissions calculated (tCO2e)
        self.emissions_calculated_total = Counter(
            f'{self.namespace}_emissions_calculated_total',
            'Total emissions calculated in tCO2e',
            ['category', 'tier'],  # category: 1-15, tier: tier1/tier2/tier3
            registry=self.registry
        )

        # Calculation count
        self.calculations_total = Counter(
            f'{self.namespace}_calculations_total',
            'Total number of carbon calculations performed',
            ['category', 'tier', 'status'],  # status: success/failed
            registry=self.registry
        )

        # Calculation duration
        self.calculation_duration_seconds = Histogram(
            f'{self.namespace}_calculation_duration_seconds',
            'Carbon calculation duration in seconds',
            ['category'],
            buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120],  # 100ms to 2min
            registry=self.registry
        )

        # Emissions by category (gauge for current period)
        self.emissions_by_category = Gauge(
            f'{self.namespace}_emissions_by_category_tco2',
            'Current period emissions by category in tCO2e',
            ['category', 'organization_id'],
            registry=self.registry
        )

        # Tier distribution (gauge)
        self.tier_distribution_percent = Gauge(
            f'{self.namespace}_tier_distribution_percent',
            'Percentage of calculations by data tier',
            ['tier', 'category'],
            registry=self.registry
        )

        # Active calculations
        self.active_calculations = Gauge(
            f'{self.namespace}_active_calculations',
            'Number of currently active carbon calculations',
            registry=self.registry
        )

        # Category coverage (gauge)
        self.category_coverage_percent = Gauge(
            f'{self.namespace}_category_coverage_percent',
            'Percentage of Scope 3 categories covered',
            ['organization_id'],
            registry=self.registry
        )

    # ========================================================================
    # ENTITY RESOLUTION METRICS
    # ========================================================================

    def _init_entity_metrics(self):
        """Initialize entity resolution metrics."""

        # Entity resolution attempts
        self.entity_resolution_total = Counter(
            f'{self.namespace}_entity_resolution_total',
            'Total entity resolution attempts',
            ['status', 'source'],  # status: success/failed, source: lei/duns/opencorporates/llm
            registry=self.registry
        )

        # Entity resolution duration
        self.entity_resolution_duration_seconds = Histogram(
            f'{self.namespace}_entity_resolution_duration_seconds',
            'Entity resolution duration in seconds',
            ['source'],
            buckets=[0.1, 0.5, 1, 2, 5, 10],
            registry=self.registry
        )

        # Entity match confidence
        self.entity_match_confidence = Histogram(
            f'{self.namespace}_entity_match_confidence',
            'Entity match confidence scores',
            ['source'],
            buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
            registry=self.registry
        )

        # Manual review queue depth
        self.entity_review_queue_depth = Gauge(
            f'{self.namespace}_entity_review_queue_depth',
            'Number of entities in manual review queue',
            registry=self.registry
        )

        # Resolved entities total
        self.entities_resolved_total = Counter(
            f'{self.namespace}_entities_resolved_total',
            'Total unique entities successfully resolved',
            ['source'],
            registry=self.registry
        )

    # ========================================================================
    # SUPPLIER ENGAGEMENT METRICS
    # ========================================================================

    def _init_supplier_metrics(self):
        """Initialize supplier engagement metrics."""

        # Supplier engagement rate
        self.supplier_engagement_rate = Gauge(
            f'{self.namespace}_supplier_engagement_rate',
            'Percentage of suppliers actively engaged',
            ['organization_id'],
            registry=self.registry
        )

        # Supplier portal logins
        self.portal_logins_total = Counter(
            f'{self.namespace}_portal_logins_total',
            'Total supplier portal logins',
            ['organization_id'],
            registry=self.registry
        )

        # Data submissions from suppliers
        self.supplier_data_submissions_total = Counter(
            f'{self.namespace}_supplier_data_submissions_total',
            'Total data submissions from suppliers',
            ['category', 'organization_id'],
            registry=self.registry
        )

        # Email campaign metrics
        self.emails_sent_total = Counter(
            f'{self.namespace}_emails_sent_total',
            'Total engagement emails sent',
            ['campaign_type', 'status'],  # status: sent/bounced/opened/clicked
            registry=self.registry
        )

        # Active suppliers
        self.active_suppliers = Gauge(
            f'{self.namespace}_active_suppliers',
            'Number of active suppliers providing data',
            ['tier', 'organization_id'],
            registry=self.registry
        )

        # Suppliers invited
        self.suppliers_invited_total = Counter(
            f'{self.namespace}_suppliers_invited_total',
            'Total suppliers invited to platform',
            ['organization_id'],
            registry=self.registry
        )

    # ========================================================================
    # LLM API METRICS
    # ========================================================================

    def _init_llm_metrics(self):
        """Initialize LLM API usage metrics."""

        # LLM API calls
        self.llm_api_calls_total = Counter(
            f'{self.namespace}_llm_api_calls_total',
            'Total LLM API calls',
            ['provider', 'model', 'purpose'],  # provider: anthropic/openai, purpose: entity/classification
            registry=self.registry
        )

        # LLM API errors
        self.llm_api_errors_total = Counter(
            f'{self.namespace}_llm_api_errors_total',
            'Total LLM API errors',
            ['provider', 'error_type'],
            registry=self.registry
        )

        # LLM API latency
        self.llm_api_duration_seconds = Histogram(
            f'{self.namespace}_llm_api_duration_seconds',
            'LLM API call duration in seconds',
            ['provider', 'model'],
            buckets=[0.5, 1, 2, 5, 10, 20, 30],
            registry=self.registry
        )

        # LLM tokens used
        self.llm_tokens_used_total = Counter(
            f'{self.namespace}_llm_tokens_used_total',
            'Total LLM tokens consumed',
            ['provider', 'model', 'type'],  # type: input/output
            registry=self.registry
        )

        # LLM cost estimate
        self.llm_cost_usd = Counter(
            f'{self.namespace}_llm_cost_usd',
            'Estimated LLM API cost in USD',
            ['provider', 'model'],
            registry=self.registry
        )

    # ========================================================================
    # DATA QUALITY METRICS
    # ========================================================================

    def _init_data_quality_metrics(self):
        """Initialize data quality metrics."""

        # Overall data quality score
        self.data_quality_score = Gauge(
            f'{self.namespace}_data_quality_score',
            'Overall data quality score (0-100)',
            ['organization_id'],
            registry=self.registry
        )

        # Data completeness
        self.data_completeness_percent = Gauge(
            f'{self.namespace}_data_completeness_percent',
            'Data completeness percentage',
            ['category', 'organization_id'],
            registry=self.registry
        )

        # Missing data points
        self.missing_data_points_total = Gauge(
            f'{self.namespace}_missing_data_points_total',
            'Total number of missing data points',
            ['category', 'organization_id'],
            registry=self.registry
        )

        # Data validation errors
        self.validation_errors_total = Counter(
            f'{self.namespace}_validation_errors_total',
            'Total data validation errors',
            ['error_type', 'category'],
            registry=self.registry
        )

        # DQI (Data Quality Indicator) scores
        self.dqi_score = Gauge(
            f'{self.namespace}_dqi_score',
            'Data Quality Indicator score per GHG Protocol',
            ['category', 'organization_id'],
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

        # Active users
        self.active_users = Gauge(
            f'{self.namespace}_active_users',
            'Number of currently active users',
            registry=self.registry
        )

        # Active tenants/organizations
        self.active_tenants = Gauge(
            f'{self.namespace}_active_tenants',
            'Number of active tenant organizations',
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
            ['agent', 'status'],  # agent: intake/calculator/hotspot/engagement/reporting
            registry=self.registry
        )

        # Agent execution duration
        self.agent_duration_seconds = Histogram(
            f'{self.namespace}_agent_duration_seconds',
            'Agent execution duration in seconds',
            ['agent'],
            buckets=[1, 5, 10, 30, 60, 120, 300],
            registry=self.registry
        )

        # Queue depth per agent
        self.agent_queue_depth = Gauge(
            f'{self.namespace}_agent_queue_depth',
            'Queue depth for each agent',
            ['agent'],
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
            'version': '2.0.0',
            'service': 'vcci-scope3-platform',
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

    def record_carbon_calculation(
        self,
        category: str,
        tier: str,
        emissions_tco2: float,
        duration_seconds: float,
        status: str = "success",
        organization_id: Optional[str] = None
    ):
        """
        Record a carbon calculation.

        Args:
            category: Scope 3 category (1-15)
            tier: Data tier (tier1/tier2/tier3)
            emissions_tco2: Emissions calculated in tCO2e
            duration_seconds: Calculation duration
            status: Calculation status (success/failed)
            organization_id: Organization identifier
        """
        self.calculations_total.labels(category=category, tier=tier, status=status).inc()
        self.calculation_duration_seconds.labels(category=category).observe(duration_seconds)

        if status == "success" and emissions_tco2 > 0:
            self.emissions_calculated_total.labels(category=category, tier=tier).inc(emissions_tco2)

            if organization_id:
                # Update current period emissions
                current = self.emissions_by_category.labels(
                    category=category,
                    organization_id=organization_id
                )._value.get()
                self.emissions_by_category.labels(
                    category=category,
                    organization_id=organization_id
                ).set((current or 0) + emissions_tco2)

    def record_entity_resolution(
        self,
        source: str,
        status: str,
        duration_seconds: float,
        confidence: Optional[float] = None
    ):
        """
        Record entity resolution attempt.

        Args:
            source: Resolution source (lei/duns/opencorporates/llm)
            status: Resolution status (success/failed)
            duration_seconds: Resolution duration
            confidence: Match confidence score (0-1)
        """
        self.entity_resolution_total.labels(status=status, source=source).inc()
        self.entity_resolution_duration_seconds.labels(source=source).observe(duration_seconds)

        if confidence is not None:
            self.entity_match_confidence.labels(source=source).observe(confidence)

        if status == "success":
            self.entities_resolved_total.labels(source=source).inc()

    def record_llm_api_call(
        self,
        provider: str,
        model: str,
        purpose: str,
        duration_seconds: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        status: str = "success",
        error_type: Optional[str] = None
    ):
        """
        Record LLM API call.

        Args:
            provider: LLM provider (anthropic/openai)
            model: Model name (claude-3-5-sonnet/gpt-4)
            purpose: Call purpose (entity/classification/etc)
            duration_seconds: API call duration
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            cost_usd: Estimated cost in USD
            status: Call status (success/failed)
            error_type: Error type if failed
        """
        if status == "success":
            self.llm_api_calls_total.labels(
                provider=provider,
                model=model,
                purpose=purpose
            ).inc()
            self.llm_api_duration_seconds.labels(provider=provider, model=model).observe(duration_seconds)
            self.llm_tokens_used_total.labels(provider=provider, model=model, type="input").inc(input_tokens)
            self.llm_tokens_used_total.labels(provider=provider, model=model, type="output").inc(output_tokens)
            self.llm_cost_usd.labels(provider=provider, model=model).inc(cost_usd)
        else:
            self.llm_api_errors_total.labels(provider=provider, error_type=error_type or "unknown").inc()

    def record_supplier_engagement(
        self,
        organization_id: str,
        event_type: str,
        **kwargs
    ):
        """
        Record supplier engagement event.

        Args:
            organization_id: Organization identifier
            event_type: Event type (login/submission/invite)
            **kwargs: Additional event-specific parameters
        """
        if event_type == "login":
            self.portal_logins_total.labels(organization_id=organization_id).inc()
        elif event_type == "submission":
            category = kwargs.get('category', 'unknown')
            self.supplier_data_submissions_total.labels(
                category=category,
                organization_id=organization_id
            ).inc()
        elif event_type == "invite":
            self.suppliers_invited_total.labels(organization_id=organization_id).inc()


# ============================================================================
# DECORATORS FOR AUTOMATIC METRICS
# ============================================================================

def track_carbon_calculation(metrics: VCCIMetrics):
    """
    Decorator to automatically track carbon calculations.

    Usage:
        @track_carbon_calculation(metrics)
        def calculate_category_1(data):
            return {"emissions_tco2": 123.45, "category": "1", "tier": "tier1"}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            metrics.active_calculations.inc()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Extract calculation details from result
                if isinstance(result, dict):
                    metrics.record_carbon_calculation(
                        category=result.get('category', 'unknown'),
                        tier=result.get('tier', 'tier3'),
                        emissions_tco2=result.get('emissions_tco2', 0),
                        duration_seconds=duration,
                        status="success",
                        organization_id=result.get('organization_id')
                    )

                metrics.active_calculations.dec()
                return result

            except Exception as e:
                duration = time.time() - start_time
                metrics.active_calculations.dec()
                raise

        return wrapper
    return decorator


# ============================================================================
# METRICS EXPORTER
# ============================================================================

class MetricsExporter:
    """Export metrics for Prometheus scraping or push gateway."""

    def __init__(self, metrics: VCCIMetrics):
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
        job: str = "vcci-scope3-platform"
    ):
        """
        Push metrics to Prometheus push gateway.

        Args:
            gateway_url: URL of push gateway
            job: Job name for grouping metrics
        """
        push_to_gateway(gateway_url, job=job, registry=self.metrics.registry)


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def create_metrics_endpoint(metrics: VCCIMetrics):
    """
    Create FastAPI endpoint for Prometheus metrics.

    Example:
        from fastapi import FastAPI
        from services.metrics import VCCIMetrics, create_metrics_endpoint

        app = FastAPI()
        metrics = VCCIMetrics()
        metrics_route = create_metrics_endpoint(metrics)
        app.add_api_route(**metrics_route)
    """
    try:
        from fastapi import Response
    except ImportError:
        logger.warning("FastAPI not installed. Metrics endpoint will not be available.")
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
_global_metrics: Optional[VCCIMetrics] = None


def get_metrics() -> VCCIMetrics:
    """Get or create global metrics instance."""
    global _global_metrics

    if _global_metrics is None:
        if PROMETHEUS_AVAILABLE:
            _global_metrics = VCCIMetrics()
        else:
            raise ImportError("Prometheus client not available")

    return _global_metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Create metrics
    metrics = VCCIMetrics()

    # Simulate carbon calculations
    logger.info("Simulating VCCI metrics collection...")

    # Category 1: Purchased goods and services
    metrics.record_carbon_calculation(
        category="1",
        tier="tier1",
        emissions_tco2=1234.56,
        duration_seconds=5.2,
        status="success",
        organization_id="org-123"
    )

    # Category 4: Upstream transportation
    metrics.record_carbon_calculation(
        category="4",
        tier="tier2",
        emissions_tco2=567.89,
        duration_seconds=3.1,
        status="success",
        organization_id="org-123"
    )

    # Entity resolution
    metrics.record_entity_resolution(
        source="lei",
        status="success",
        duration_seconds=0.8,
        confidence=0.95
    )

    # LLM API call
    metrics.record_llm_api_call(
        provider="anthropic",
        model="claude-3-5-sonnet",
        purpose="entity_extraction",
        duration_seconds=1.5,
        input_tokens=1500,
        output_tokens=300,
        cost_usd=0.015,
        status="success"
    )

    # Supplier engagement
    metrics.record_supplier_engagement(
        organization_id="org-123",
        event_type="login"
    )

    # Update system metrics
    metrics.update_system_metrics()

    # Export metrics
    exporter = MetricsExporter(metrics)
    print("\n" + "="*80)
    print("PROMETHEUS METRICS EXPORT (VCCI CARBON-SPECIFIC)")
    print("="*80)
    print(exporter.export_text())
