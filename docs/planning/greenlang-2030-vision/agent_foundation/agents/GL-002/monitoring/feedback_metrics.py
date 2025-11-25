# -*- coding: utf-8 -*-
"""
Prometheus Metrics for Feedback and Experiments

This module defines and exposes Prometheus metrics for monitoring
feedback trends and experiment performance.

Example:
    >>> from prometheus_client import start_http_server
    >>> metrics = FeedbackMetrics(db_url="postgresql://...")
    >>> await metrics.initialize()
    >>> await metrics.collect_metrics()  # Run periodically
    >>> start_http_server(8000)  # Expose metrics on :8000/metrics
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import asyncpg
import asyncio

from greenlang.determinism import DeterministicClock
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    Info,
    CollectorRegistry,
    generate_latest
)

logger = logging.getLogger(__name__)


class FeedbackMetrics:
    """
    Prometheus metrics collector for feedback and experiments.

    Exposes metrics for Grafana dashboards and alerting.

    Attributes:
        db_pool: PostgreSQL connection pool
        registry: Prometheus metrics registry
    """

    def __init__(self, db_url: str, registry: Optional[CollectorRegistry] = None):
        """
        Initialize FeedbackMetrics.

        Args:
            db_url: PostgreSQL connection URL
            registry: Optional Prometheus registry (defaults to default registry)
        """
        self.db_url = db_url
        self.db_pool: Optional[asyncpg.Pool] = None
        self.registry = registry

        # Define metrics
        self._define_metrics()

        logger.info("FeedbackMetrics initialized")

    def _define_metrics(self) -> None:
        """Define all Prometheus metrics."""

        # Feedback metrics
        self.feedback_total = Counter(
            'gl002_feedback_total',
            'Total number of feedback submissions',
            ['rating', 'category'],
            registry=self.registry
        )

        self.feedback_rating_avg = Gauge(
            'gl002_feedback_rating_average',
            'Average feedback rating',
            ['period'],
            registry=self.registry
        )

        self.feedback_nps = Gauge(
            'gl002_feedback_nps_score',
            'Net Promoter Score',
            ['period'],
            registry=self.registry
        )

        self.feedback_accuracy_avg = Gauge(
            'gl002_feedback_accuracy_average',
            'Average prediction accuracy percentage',
            ['period'],
            registry=self.registry
        )

        self.feedback_processing_time = Histogram(
            'gl002_feedback_processing_seconds',
            'Feedback processing time in seconds',
            registry=self.registry
        )

        # Experiment metrics
        self.experiments_active = Gauge(
            'gl002_experiments_active',
            'Number of active experiments',
            registry=self.registry
        )

        self.experiment_users = Gauge(
            'gl002_experiment_users_total',
            'Total users in experiment',
            ['experiment_id', 'variant'],
            registry=self.registry
        )

        self.experiment_metric_value = Gauge(
            'gl002_experiment_metric_value',
            'Experiment metric value',
            ['experiment_id', 'variant', 'metric_name'],
            registry=self.registry
        )

        self.experiment_conversion_rate = Gauge(
            'gl002_experiment_conversion_rate',
            'Experiment conversion rate',
            ['experiment_id', 'variant'],
            registry=self.registry
        )

        # Alert metrics
        self.alerts_active = Gauge(
            'gl002_alerts_active',
            'Number of active unacknowledged alerts',
            ['severity'],
            registry=self.registry
        )

        # System metrics
        self.system_info = Info(
            'gl002_system',
            'System information',
            registry=self.registry
        )

    async def initialize(self) -> None:
        """Initialize database connection."""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )

            # Set system info
            self.system_info.info({
                'version': '1.0.0',
                'agent': 'GL-002',
                'component': 'BoilerEfficiencyOptimizer'
            })

            logger.info("FeedbackMetrics database connection initialized")

        except Exception as e:
            logger.error(f"Failed to initialize FeedbackMetrics: {e}", exc_info=True)
            raise

    async def close(self) -> None:
        """Close database connection."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("FeedbackMetrics closed")

    async def collect_metrics(self) -> None:
        """
        Collect all metrics from database.

        Should be called periodically (e.g., every 60 seconds).
        """
        if not self.db_pool:
            raise RuntimeError("Metrics collector not initialized")

        try:
            async with self.db_pool.acquire() as conn:
                # Collect feedback metrics
                await self._collect_feedback_metrics(conn)

                # Collect experiment metrics
                await self._collect_experiment_metrics(conn)

                # Collect alert metrics
                await self._collect_alert_metrics(conn)

            logger.debug("Metrics collection complete")

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}", exc_info=True)

    async def _collect_feedback_metrics(self, conn: asyncpg.Connection) -> None:
        """Collect feedback-related metrics."""

        # Average rating by period
        for period_days, label in [(1, 'day'), (7, 'week'), (30, 'month')]:
            period_start = DeterministicClock.utcnow() - timedelta(days=period_days)

            query = """
                SELECT
                    AVG(rating) as avg_rating,
                    AVG(savings_accuracy) as avg_accuracy,
                    COUNT(*) as total_count,
                    COUNT(CASE WHEN rating >= 4 THEN 1 END) as promoters,
                    COUNT(CASE WHEN rating <= 3 THEN 1 END) as detractors
                FROM optimization_feedback
                WHERE timestamp >= $1
            """

            row = await conn.fetchrow(query, period_start)

            if row and row['total_count'] > 0:
                # Average rating
                self.feedback_rating_avg.labels(period=label).set(
                    row['avg_rating'] or 0
                )

                # Average accuracy
                if row['avg_accuracy']:
                    self.feedback_accuracy_avg.labels(period=label).set(
                        row['avg_accuracy']
                    )

                # NPS score
                nps = ((row['promoters'] - row['detractors']) / row['total_count']) * 100
                self.feedback_nps.labels(period=label).set(nps)

        # Feedback count by rating and category
        query = """
            SELECT rating, category, COUNT(*) as count
            FROM optimization_feedback
            WHERE timestamp >= $1
            GROUP BY rating, category
        """

        period_start = DeterministicClock.utcnow() - timedelta(days=1)
        rows = await conn.fetch(query, period_start)

        # Reset counter (since we're setting absolute values)
        for row in rows:
            # Note: Counter increment, not set
            # In production, use a Gauge or track incremental changes
            pass  # Counters should only increment

    async def _collect_experiment_metrics(self, conn: asyncpg.Connection) -> None:
        """Collect experiment-related metrics."""

        # Active experiments count
        query = """
            SELECT COUNT(*) as count
            FROM experiments
            WHERE status = 'running'
        """

        row = await conn.fetchrow(query)
        self.experiments_active.set(row['count'] if row else 0)

        # Per-experiment metrics
        query = """
            SELECT
                e.experiment_id,
                e.name,
                a.variant_name,
                COUNT(DISTINCT a.user_id) as user_count
            FROM experiments e
            JOIN experiment_assignments a ON e.experiment_id = a.experiment_id
            WHERE e.status = 'running'
            GROUP BY e.experiment_id, e.name, a.variant_name
        """

        rows = await conn.fetch(query)

        for row in rows:
            self.experiment_users.labels(
                experiment_id=row['experiment_id'],
                variant=row['variant_name']
            ).set(row['user_count'])

        # Experiment metric values
        query = """
            SELECT
                experiment_id,
                variant_name,
                metric_name,
                AVG(metric_value) as avg_value
            FROM experiment_metrics
            WHERE recorded_at >= $1
            GROUP BY experiment_id, variant_name, metric_name
        """

        period_start = DeterministicClock.utcnow() - timedelta(hours=24)
        rows = await conn.fetch(query, period_start)

        for row in rows:
            self.experiment_metric_value.labels(
                experiment_id=row['experiment_id'],
                variant=row['variant_name'],
                metric_name=row['metric_name']
            ).set(row['avg_value'])

    async def _collect_alert_metrics(self, conn: asyncpg.Connection) -> None:
        """Collect alert-related metrics."""

        query = """
            SELECT severity, COUNT(*) as count
            FROM feedback_alerts
            WHERE NOT acknowledged
            GROUP BY severity
        """

        rows = await conn.fetch(query)

        # Reset all severities to 0 first
        for severity in ['critical', 'warning', 'info']:
            self.alerts_active.labels(severity=severity).set(0)

        # Set actual counts
        for row in rows:
            self.alerts_active.labels(severity=row['severity']).set(row['count'])

    async def record_feedback_submission(
        self,
        rating: int,
        category: str,
        processing_time: float
    ) -> None:
        """
        Record a feedback submission event.

        Args:
            rating: User rating (1-5)
            category: Feedback category
            processing_time: Processing time in seconds
        """
        # Increment counter
        self.feedback_total.labels(
            rating=str(rating),
            category=category
        ).inc()

        # Record processing time
        self.feedback_processing_time.observe(processing_time)

        logger.debug(f"Recorded feedback: rating={rating}, category={category}")

    def get_metrics(self) -> bytes:
        """
        Get current metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)


async def start_metrics_collector(
    db_url: str,
    collection_interval: int = 60
) -> None:
    """
    Start metrics collection background task.

    Args:
        db_url: PostgreSQL connection URL
        collection_interval: Seconds between collections
    """
    metrics = FeedbackMetrics(db_url)
    await metrics.initialize()

    try:
        while True:
            await metrics.collect_metrics()
            await asyncio.sleep(collection_interval)

    except asyncio.CancelledError:
        logger.info("Metrics collection cancelled")
        await metrics.close()

    except Exception as e:
        logger.error(f"Metrics collection error: {e}", exc_info=True)
        await metrics.close()
        raise
